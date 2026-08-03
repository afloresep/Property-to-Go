"""C24 -- the external-validity substrate: a text LM and a computable text attribute.

**This module is not part of the molecular pipeline.**  Nothing under `scripts/0*.py`,
`scripts/1[0-8]_*.py` or `scripts/20_*.py` imports it, it never loads GP-MoLFormer, and
it cannot change any number in `reports/pilot_report.md`.  It exists so that the two
most portable findings of the project -- that post-hoc calibration of the probe is
algebraically a rescale of lambda, and that the best-predicting probe layer is
mid-network and is not the best-steering layer -- can be tested on a **second,
non-molecular** instance of the same pipeline.

Scope note, because the brief forbids a second generator: the "no second generator" rule
in the specification is about the *molecular* experiment -- it forbids comparing or
ensembling molecular generators so that the negative result cannot be blamed on a
generator choice.  A different-*domain* generator used only for an external-validity
check is outside that rule.  See `reports/section_c24_generality.md` C24.0.

What is deliberately shared with the molecular library, and why:

    guidance.combine_scores      the decoding rule itself.  The single load-bearing
                                 line is literally the molecular one, so a difference
                                 between the two domains cannot be an implementation
                                 difference in the rule.
    guidance.TargetScorer,       head -> P(y_final in I), and the calibrated variant.
    calibration.*
    headroom.candidate_weights,  the per-position weightings.
    headroom.guided_weights
    binning.*, metrics.*         binning, AUROC / ECE / Brier.
    bestofn.selection_key,       the interval semantics, including the docs/HANDOFF.md
    bestofn.target_error         section 4 boundary bug and its fix.
    compute.ComputeMeter,        token accounting.
    compute.solve_best_of_n
    splits.split_by_group        grouped splitting, so no sequence's prefixes straddle
                                 train and test.

What is re-implemented here, and why:

    the generator loader          GPT-2, not GP-MoLFormer.
    the sampler / guided decoder  GPT-2's KV cache is the standard
                                  `(key, value)` per layer; `generation.repeat_cache`
                                  is written for Molformer's linear-attention cache and
                                  is not applicable.  The decoding *rule* is imported.
    the head trainer              `heads.train_head` is CPU-only; the device-aware copy
                                  here is asserted bit-identical to it on CPU by
                                  `tests/test_generality.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .compute import ComputeMeter
from .guidance import TargetScorer, combine_scores  # the rule under test, shared verbatim

# --------------------------------------------------------------------------- pins

#: The frozen text generator.  Pinned by commit, exactly as `configs/model.yaml` pins
#: GP-MoLFormer, so "the model" is a checkable object and not a name.
TEXT_MODEL_REPO = "gpt2"
TEXT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TEXT_TOKENIZER_REPO = "gpt2"
TEXT_TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

#: `<|endoftext|>`, used as the unconditional start token.
TEXT_BOS_ID = 50256


# ---------------------------------------------------------------------- attributes


def _words(text: str) -> list[str]:
    return text.split()


def digit_count(text: str) -> float:
    """Number of digit characters in the completed text."""
    return float(sum(c.isdigit() for c in text))


def upper_count(text: str) -> float:
    """Number of upper-case characters in the completed text."""
    return float(sum(c.isupper() for c in text))


def mean_word_length(text: str) -> float:
    """Mean length in characters of the whitespace-delimited words."""
    w = _words(text)
    return float(np.mean([len(x) for x in w])) if w else 0.0


#: The battery.  Every one is an *exact* function of the completed text, computable in
#: microseconds, which is what makes compute-matched best-of-N available (the baseline
#: gets ground truth, exactly as RDKit gives it ground truth in the molecular pipeline).
ATTRIBUTES: dict[str, Callable[[str], float]] = {
    "digit_count": digit_count,
    "upper_count": upper_count,
    "mean_word_length": mean_word_length,
}

#: Count-valued attributes.  Declared for the same reason `bestofn.INTEGER_PROPERTIES`
#: is declared: `target_error` must measure to the nearest *attainable* value inside
#: `[lo, hi)`, and forgetting this reintroduces the docs/HANDOFF.md section 4 bug.
INTEGER_ATTRIBUTES = frozenset({"digit_count", "upper_count"})

ATTRIBUTE_ORDER = ("digit_count", "upper_count", "mean_word_length")


def compute_attributes(text: str) -> dict[str, float]:
    return {name: fn(text) for name, fn in ATTRIBUTES.items()}


#: Cheap prefix statistics -- the analogue of `tokens.FEATURE_NAMES`.  The `trivial`
#: head reads these and nothing else, so "does the hidden state beat surface counting?"
#: is answerable in this domain too.
TRIVIAL_FEATURE_NAMES = (
    "prefix_content_tokens",
    "prefix_chars",
    "prefix_digits",
    "prefix_uppercase",
    "prefix_spaces",
    "prefix_words",
    "prefix_mean_word_length",
    "prefix_punctuation",
)


def trivial_features(text: str, n_content_tokens: int) -> np.ndarray:
    w = _words(text)
    return np.array(
        [
            float(n_content_tokens),
            float(len(text)),
            float(sum(c.isdigit() for c in text)),
            float(sum(c.isupper() for c in text)),
            float(sum(c.isspace() for c in text)),
            float(len(w)),
            float(np.mean([len(x) for x in w])) if w else 0.0,
            float(sum(c in ",.;:!?'\"()-" for c in text)),
        ],
        dtype=np.float32,
    )


# ------------------------------------------------------------------ the generator


@dataclass
class TextGenerator:
    """A frozen causal text LM, with the same surface `FrozenGenerator` exposes."""

    model: Any
    tokenizer: Any
    device: torch.device
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def bos_id(self) -> int:
        return TEXT_BOS_ID

    @property
    def eos_id(self) -> int:
        return int(self.model.config.eos_token_id)

    @property
    def pad_id(self) -> int:
        return int(self.model.config.eos_token_id)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.n_embd)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.n_layer)

    @property
    def n_probe_points(self) -> int:
        """`hidden_states` has n_layer + 1 entries: 0 is the embedding output."""
        return self.n_layers + 1

    def decode(self, ids) -> list[str]:
        return self.tokenizer.batch_decode(ids)

    def fingerprint(self) -> dict[str, Any]:
        total = 0.0
        n = 0
        for p in self.model.parameters():
            total += float(p.detach().double().sum())
            n += p.numel()
        return {
            "n_parameters": n,
            "parameter_sum": total,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "repo": TEXT_MODEL_REPO,
            "revision": TEXT_MODEL_REVISION,
        }


def load_text_generator(device: str = "cuda", dtype: str = "float32") -> TextGenerator:
    """Load the pinned GPT-2 revision, frozen, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_TOKENIZER_REPO, revision=TEXT_TOKENIZER_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_REPO,
        revision=TEXT_MODEL_REVISION,
        torch_dtype=getattr(torch, dtype),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = torch.device(device)
    model.to(dev)
    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=dev,
        config={
            "model_repo": TEXT_MODEL_REPO,
            "model_revision": TEXT_MODEL_REVISION,
            "tokenizer_repo": TEXT_TOKENIZER_REPO,
            "tokenizer_revision": TEXT_TOKENIZER_REVISION,
            "device": str(dev),
            "dtype": dtype,
        },
    )


def repeat_cache_gpt2(past_key_values, repeats: int):
    """Repeat a standard `(key, value)`-per-layer KV cache along the batch dimension.

    GPT-2's cache is `((k, v), ...)` with `k`, `v` of shape `(B, H, T, D)`.  The
    Molformer helper in `generation.repeat_cache` is written for a linear-attention
    running-sum cache and cannot be reused; this is the standard-attention equivalent.
    """
    out = []
    for layer_cache in past_key_values:
        k, v = layer_cache[0], layer_cache[1]
        out.append(
            (k.repeat_interleave(repeats, dim=0), v.repeat_interleave(repeats, dim=0))
            + tuple(layer_cache[2:])
        )
    return tuple(out)


# ----------------------------------------------------------------------- sampling


@torch.no_grad()
def sample_base(
    gen: TextGenerator,
    n: int,
    seed: int,
    n_content: int,
    temperature: float = 1.0,
    batch_size: int = 256,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """Draw `n` unconditional fixed-length completions under the base policy.

    Every sequence is `[bos] + n_content` tokens.  Fixed length rather than
    EOS-terminated: GPT-2 does not emit a natural terminator often enough for an
    end-of-sequence rule to give a clean "completed sequence", and a truncation
    boundary would give the decoder a degenerate lever (steer towards being cut off).
    The cost is that the length confound the molecular pipeline had to standardise for
    does not exist here; that is stated as a design difference, not hidden.
    """
    return guided_sample_text(
        gen,
        scorer=None,
        n=n,
        seed=seed,
        n_content=n_content,
        temperature=temperature,
        batch_size=batch_size,
        meter=meter,
        window_fn=lambda t: False,
    )


@torch.no_grad()
def guided_sample_text(
    gen: TextGenerator,
    scorer: TargetScorer | None,
    n: int,
    seed: int,
    n_content: int,
    window_fn: Callable[[int], bool] | None = None,
    temperature: float = 1.0,
    top_k: int = 8,
    lam: float = 1.0,
    eps: float = 1e-6,
    layer: int = -1,
    batch_size: int = 64,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """The decoding rule under test, on text.

        score(a) = log p_base(a | prefix) + lam * log(P(y_final in I | prefix + a) + eps)

    sampled from a softmax over the base model's `top_k` candidates.  The combination is
    `guidance.combine_scores` -- the molecular function, imported, not re-derived.  Only
    the cache plumbing and the fixed-length stopping rule differ.
    """
    if window_fn is None:
        window_fn = lambda t: True  # noqa: E731
    torch.manual_seed(seed)
    results: list[list[int]] = []

    while len(results) < n:
        b = min(batch_size, n - len(results))
        seqs = [[gen.bos_id] for _ in range(b)]
        cur = torch.full((b, 1), gen.bos_id, dtype=torch.long, device=gen.device)
        pkv = None

        for t in range(n_content):
            out = gen.model(
                input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True
            )
            pkv = out.past_key_values
            logits = out.logits[:, -1, :].float() / temperature
            logprobs = torch.log_softmax(logits, dim=-1)
            if meter is not None:
                meter.add_forward(b, b)

            guide = window_fn(t) and (scorer is not None or lam == 0.0)
            if guide and lam == 0.0:
                cand_lp, cand_ids = torch.topk(logprobs, top_k, dim=-1)
                pick = torch.multinomial(torch.softmax(cand_lp, dim=-1), 1)
                nxt = cand_ids.gather(1, pick).squeeze(1)
            elif guide:
                cand_lp, cand_ids = torch.topk(logprobs, top_k, dim=-1)
                cand_cache = repeat_cache_gpt2(pkv, top_k)
                flat = cand_ids.reshape(b * top_k, 1)
                cout = gen.model.transformer(
                    input_ids=flat,
                    past_key_values=cand_cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = cout.hidden_states[layer][:, 0, :].reshape(b, top_k, -1)
                if meter is not None:
                    # actual: one token per candidate from a shared cache.
                    # full recompute: every candidate re-reads the whole prefix.
                    meter.add_forward(b * top_k, b * top_k * (t + 2))
                q = scorer(hidden.reshape(-1, hidden.shape[-1])).reshape(cand_lp.shape)
                score = combine_scores(cand_lp, q, lam, eps)
                pick = torch.multinomial(torch.softmax(score, dim=-1), 1)
                nxt = cand_ids.gather(1, pick).squeeze(1)
            else:
                nxt = torch.multinomial(torch.softmax(logprobs, dim=-1), 1).squeeze(1)

            for i in range(b):
                seqs[i].append(int(nxt[i]))
            cur = nxt.unsqueeze(1)

        for s in seqs:
            results.append(s)
            if meter is not None:
                meter.molecules_returned += 1

    return results[:n]


@torch.no_grad()
def continue_prefixes(
    gen: TextGenerator,
    prefixes: list[list[int]],
    n_content: int,
    seed: int,
    temperature: float = 1.0,
    batch_size: int = 512,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """Base-policy continuations of `prefixes` up to `[bos] + n_content` tokens.

    Prefixes are grouped by exact length so no padding is ever needed.
    """
    torch.manual_seed(seed)
    out: list[list[int] | None] = [None] * len(prefixes)
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prefixes):
        by_len.setdefault(len(p), []).append(i)

    for length, idxs in sorted(by_len.items()):
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            ids = torch.tensor(
                [prefixes[i] for i in chunk], dtype=torch.long, device=gen.device
            )
            seqs = [list(prefixes[i]) for i in chunk]
            # Prefill through the transformer body only.  `GPT2LMHeadModel` applies the
            # 50257-way LM head at *every* position, which for a batch of prefixes is
            # gigabytes of logits that are immediately discarded.
            res = gen.model.transformer(input_ids=ids, use_cache=True, return_dict=True)
            pkv = res.past_key_values
            logits = gen.model.lm_head(res.last_hidden_state[:, -1, :]).float() / temperature
            if meter is not None:
                meter.add_forward(len(chunk) * length, len(chunk) * length)
            for t in range(length, n_content + 1):
                lp = torch.log_softmax(logits, dim=-1)
                nxt = torch.multinomial(torch.softmax(lp, dim=-1), 1).squeeze(1)
                for r in range(len(chunk)):
                    seqs[r].append(int(nxt[r]))
                if t == n_content:
                    break
                res = gen.model.transformer(
                    input_ids=nxt.unsqueeze(1),
                    past_key_values=pkv,
                    use_cache=True,
                    return_dict=True,
                )
                pkv = res.past_key_values
                logits = gen.model.lm_head(res.last_hidden_state[:, -1, :]).float() / temperature
                if meter is not None:
                    meter.add_forward(len(chunk), len(chunk))
            for r, i in enumerate(chunk):
                out[i] = seqs[r]
                if meter is not None:
                    meter.molecules_returned += 1
    return [o for o in out]  # type: ignore[return-value]


@torch.no_grad()
def all_layer_states(
    gen: TextGenerator,
    sequences: list[list[int]],
    positions: np.ndarray,
    batch_size: int = 128,
    meter: ComputeMeter | None = None,
) -> np.ndarray:
    """States at every probe point, for one position set per sequence.

    `sequences` are all the same length (fixed-length completions), so no padding and
    no attention mask are needed.  Returns `(n_sequences, n_positions, n_probe_points,
    hidden)` as float32.  One forward pass returns all probe points -- the same fact
    section 21.1.1 records for the molecular sweep, so 13 probe points cost the tokens
    of one.
    """
    n_seq = len(sequences)
    n_pos = positions.shape[1]
    n_probe = gen.n_probe_points
    out = np.zeros((n_seq, n_pos, n_probe, gen.hidden_size), dtype=np.float32)
    for start in range(0, n_seq, batch_size):
        chunk = list(range(start, min(start + batch_size, n_seq)))
        ids = torch.tensor([sequences[i] for i in chunk], dtype=torch.long, device=gen.device)
        res = gen.model.transformer(
            input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True
        )
        hs = torch.stack(res.hidden_states, dim=2)  # (B, T, n_probe, H)
        hs = hs.float().cpu().numpy()
        for r, i in enumerate(chunk):
            out[i] = hs[r, positions[i], :, :]
        if meter is not None:
            meter.add_forward(int(ids.numel()), int(ids.numel()))
    return out


@torch.no_grad()
def top_k_candidates(
    gen: TextGenerator,
    prefixes: list[list[int]],
    k: int,
    temperature: float = 1.0,
    batch_size: int = 128,
    meter: ComputeMeter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-`k` next tokens, their full-vocabulary log-probabilities, and the extended
    prefixes' states at every probe point.

    Returns `(ids (n, k), logprobs (n, k), states (n, k, n_probe, hidden))`.
    """
    n = len(prefixes)
    ids_out = np.zeros((n, k), dtype=np.int64)
    lp_out = np.zeros((n, k), dtype=np.float64)
    st_out = np.zeros((n, k, gen.n_probe_points, gen.hidden_size), dtype=np.float32)

    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prefixes):
        by_len.setdefault(len(p), []).append(i)

    for length, idxs in sorted(by_len.items()):
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start : start + batch_size]
            b = len(chunk)
            ids = torch.tensor(
                [prefixes[i] for i in chunk], dtype=torch.long, device=gen.device
            )
            res = gen.model.transformer(input_ids=ids, use_cache=True, return_dict=True)
            pkv = res.past_key_values
            logits = gen.model.lm_head(res.last_hidden_state[:, -1, :]).float() / temperature
            lp = torch.log_softmax(logits, dim=-1)
            cand_lp, cand_ids = torch.topk(lp, k, dim=-1)
            if meter is not None:
                meter.add_forward(int(ids.numel()), int(ids.numel()))
            cout = gen.model.transformer(
                input_ids=cand_ids.reshape(b * k, 1),
                past_key_values=repeat_cache_gpt2(pkv, k),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = torch.stack(cout.hidden_states, dim=2)[:, 0, :, :]  # (b*k, n_probe, H)
            hs = hs.float().cpu().numpy().reshape(b, k, gen.n_probe_points, gen.hidden_size)
            if meter is not None:
                meter.add_forward(b * k, b * k * (length + 1))
            for r, i in enumerate(chunk):
                ids_out[i] = cand_ids[r].cpu().numpy()
                lp_out[i] = cand_lp[r].cpu().numpy()
                st_out[i] = hs[r]
    return ids_out, lp_out, st_out


# ------------------------------------------------------------------ head training


def train_head_on_device(head: nn.Module, x_train, y_train, x_val, y_val, cfg, device):
    """`heads.train_head`, with the tensors on `device`.

    A literal copy with `.to(device)` added, kept here rather than in `heads.py` so the
    molecular library is untouched.  `tests/test_generality.py::
    test_the_device_trainer_reproduces_the_molecular_trainer_on_cpu` asserts the two
    give bit-identical parameters on CPU, which is what licenses calling this "the same
    recipe".
    """
    from .heads import TrainResult

    torch.manual_seed(int(cfg["seed"]))
    head.set_standardiser(x_train)
    head = head.to(device)

    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    opt = torch.optim.AdamW(
        head.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    loss_fn = nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    n = len(xt)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    best_epoch = -1
    history: list[dict[str, float]] = []
    patience = int(cfg["patience"])
    since_best = 0
    generator = torch.Generator().manual_seed(int(cfg["seed"]))

    for epoch in range(int(cfg["max_epochs"])):
        head.train()
        perm = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        train_nll = total / n

        head.eval()
        with torch.no_grad():
            val_nll = float(loss_fn(head(xv), yv))
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})

        if val_nll < best_val - 1e-5:
            best_val = val_nll
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    head.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_nll=best_val, history=history)


@torch.no_grad()
def predict_proba_on_device(head: nn.Module, x: np.ndarray, device, batch_size=8192):
    head = head.to(device)
    head.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.softmax(head(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, head.n_bins))


# ------------------------------------------------------------- target intervals


#: Fixed quantile grid the target-interval rule searches over.  Committed in C24.0.
TARGET_QUANTILE_GRID = tuple(round(0.05 * i, 2) for i in range(21))


def resolve_target_band(
    values: np.ndarray,
    integer_valued: bool,
    target_rate: float = 0.10,
    grid: tuple[float, ...] = TARGET_QUANTILE_GRID,
) -> dict[str, Any]:
    """The pre-registered target-interval rule (C24.0.3).

    Over all `[q_a, q_b)` with `a < b` drawn from `grid`, take the band whose realised
    base rate is closest to `target_rate`; ties broken by smallest `a`, then smallest
    `b`.  Integer-valued attributes have both edges rounded to the nearest integer
    first, so the band is a union of `CategoricalBinner` bins by construction (the
    docs/HANDOFF.md section 3.6 invariant).

    Deterministic given the frozen base sample, and stated before that sample existed.
    """
    v = np.asarray(values, dtype=np.float64)
    best = None
    for ia, a in enumerate(grid):
        for b in grid[ia + 1 :]:
            lo = float(np.quantile(v, a))
            hi = float(np.quantile(v, b))
            if integer_valued:
                lo = float(round(lo))
                hi = float(round(hi))
            if not (hi > lo):
                continue
            rate = float(((v >= lo) & (v < hi)).mean())
            key = (abs(rate - target_rate), a, b)
            if best is None or key < best[0]:
                best = (key, {"lo": lo, "hi": hi, "q_lo": a, "q_hi": b, "base_rate": rate})
    if best is None:
        raise ValueError("no admissible target band")
    return best[1]


# ------------------------------------------------------------------- best-of-N


def best_of_n_text(
    gen: TextGenerator,
    n_sequences: int,
    n_candidates: int,
    seed: int,
    attribute: str,
    lo: float,
    hi: float,
    n_content: int,
    temperature: float = 1.0,
    batch_size: int = 256,
    meter: ComputeMeter | None = None,
) -> list[dict[str, Any]]:
    """Compute-matched best-of-N: draw `N` base samples per slot, keep the best.

    Selection uses `bestofn.selection_key`, the molecular ranking key, so a genuine hit
    always outranks a boundary miss (docs/HANDOFF.md section 4).
    """
    from .bestofn import selection_key

    local = meter if meter is not None else ComputeMeter()
    total = n_sequences * n_candidates
    seqs = sample_base(
        gen, total, seed=seed, n_content=n_content, temperature=temperature,
        batch_size=batch_size, meter=local,
    )
    texts = gen.decode([s[1:] for s in seqs])
    fn = ATTRIBUTES[attribute]
    values = [fn(t) for t in texts]

    selected = []
    for i in range(n_sequences):
        base = i * n_candidates
        best = None
        best_key = None
        for j in range(base, base + n_candidates):
            key = selection_key(values[j], lo, hi)
            if best_key is None or key < best_key:
                best_key = key
                best = {"text": texts[j], "token_ids": seqs[j], attribute: values[j]}
        selected.append(best)
    if meter is not None:
        meter.molecules_returned = n_sequences
    return selected


def summarise_texts(
    texts: list[str], attribute: str, lo: float, hi: float
) -> dict[str, Any]:
    """Hit rate and the descriptive statistics, on completed texts."""
    from .bestofn import target_error

    fn = ATTRIBUTES[attribute]
    vals = np.array([fn(t) for t in texts], dtype=np.float64)
    hit = (vals >= lo) & (vals < hi)
    integer_valued = attribute in INTEGER_ATTRIBUTES
    errs = np.array([target_error(v, lo, hi, integer_valued) for v in vals])
    return {
        "n": int(len(texts)),
        "hit_rate": float(hit.mean()),
        "uniqueness": float(len(set(texts)) / len(texts)),
        "abs_target_error_mean": float(errs.mean()),
        "abs_target_error_median": float(np.median(errs)),
        f"{attribute}_mean": float(vals.mean()),
        f"{attribute}_std": float(vals.std()),
        "mean_characters": float(np.mean([len(t) for t in texts])),
        "mean_words": float(np.mean([len(t.split()) for t in texts])),
    }
