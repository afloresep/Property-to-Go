"""C25 -- a readout over a *window* of positions, instead of over one position.

`reports/pilot_report.md` §8.3 names three suspects for why the frozen state
underperforms as a guidance signal: an earlier layer, a larger head, a different
pooling.  §20.4 closed the second at the final layer and §21/C23 closed the first.
**Pooling is the one nobody has touched**: every head in this project, baseline and
variant, reads the single hidden state `h_t` at one position.

This module is additive.  It changes no existing behaviour and no existing artefact,
and in particular it does **not** touch `guidance.py`, which is load-bearing for every
existing number.  `pooled_guided_sample` below is a deliberate transcription of
`guidance.guided_sample` with one addition (a rolling buffer of committed hidden
states); `tests/test_pooled_readout.py` asserts that at window size 1 it returns
molecules identical to `guidance.guided_sample`, so the copy cannot silently diverge.

Three things live here.

1.  **The window itself.**  For a prefix at position `p` (the state after consuming
    tokens `0..p` inclusive -- `generation.hidden_states_for_positions`' contract), the
    window of size `w` is the positions `max(0, p-w+1) .. p`.  It contains
    `c = min(w, p+1)` *distinct* positions.  Stacks are stored left-padded by repeating
    the earliest available state (index clamping), so the tensor shape is fixed and the
    `c` distinct states are always the **last** `c` slots.  `counts` is stored beside
    the stack so a pooling operator can tell padding from data.

2.  **The pooling operators.**  Fixed operators (`last`, `mean`, `concat`) are pure
    functions of the stack and are therefore precomputable into a 2-D feature array,
    which is fed to the *unmodified* `heads.MLPHead` through the *unmodified*
    `probe_layers.train_one_probe`.  That is what makes the window-size-1 validity gate
    meaningful: `last` at `w=1` is not merely equivalent to the deployed readout, it is
    the identical array through the identical trainer.  The learned operator (`attn`)
    cannot be precomputed and has its own head class and trainer below, built so that
    at `w=1` it consumes the ambient RNG in the same order as `MLPHead` and reduces to
    it exactly.

3.  **Decode-time computability, which is a compute-accounting question and was checked
    before the variants were chosen.**  Every operator here needs only the last `w-1`
    *committed* hidden states plus the candidate state.  The committed states are
    produced by the KV-cached forward pass the decoder already runs -- setting
    `output_hidden_states=True` on that call returns them for free, because the pass
    processes exactly the same tokens either way.  So a pooled readout costs the same
    `processed_tokens_actual` and the same `processed_tokens_full_recompute` as the
    deployed single-position readout, and no variant here needs full-recompute
    accounting to be affordable.  A pooling operator that needed to *re-read* the prefix
    (for example, attention over the whole prefix with keys recomputed under the
    candidate) would not have that property and is deliberately not in the family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .compute import ComputeMeter
from .generation import repeat_cache
from .heads import MLPHead, TrainResult
from .model_io import FrozenGenerator

# ---------------------------------------------------------------------------------
# The variant family, fixed in reports/section_c25_pooling.md §C25.0.2 before any
# measurement.  Adding a name here after results exist is exactly the thing the
# pre-registration forbids, so the family lives in code and the tests count it.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolSpec:
    name: str
    window: int
    mode: str          # last | mean | concat | attn
    hidden_dim: int | None = None   # None -> the config's head.hidden_dim

    @property
    def precomputable(self) -> bool:
        return self.mode in ("last", "mean", "concat")

    def in_dim(self, hidden_size: int) -> int:
        return hidden_size * self.window if self.mode == "concat" else hidden_size


#: The pre-registered family.  `last1` is the deployed readout and the validity gate.
POOL_VARIANTS: tuple[PoolSpec, ...] = (
    PoolSpec("last1", 1, "last"),
    PoolSpec("mean4", 4, "mean"),
    PoolSpec("mean16", 16, "mean"),
    PoolSpec("concat4", 4, "concat"),
    PoolSpec("attn4", 4, "attn"),
    # Capacity x depth: §8.3's "larger head" suspect, which §20.4 excluded at probe
    # point 12 only.  Window 1, so it is a capacity control and not a pooling variant.
    PoolSpec("wide1", 1, "last", hidden_dim=1024),
)

VARIANTS_BY_NAME = {v.name: v for v in POOL_VARIANTS}

#: The stack that is materialised on disk.  Larger windows are stored as their
#: precomputed mean instead, because only `concat`/`attn` need the individual states.
STACK_WINDOW = 4


def window_positions(p: int, w: int) -> list[int]:
    """The `w` positions of the window ending at `p`, left-padded by index clamping."""
    return [max(0, p - w + 1 + j) for j in range(w)]


def window_count(p: int, w: int) -> int:
    """How many of those positions are distinct."""
    return min(w, p + 1)


def masked_mean(stack: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean over the `counts` *distinct* trailing slots of a clamped window stack.

    Equivalent to the mean over positions `max(0, p-w+1)..p`, i.e. padding is not
    counted.  Written with a cumulative sum from the right so it is one pass over the
    array rather than a Python loop over rows.
    """
    n, w, d = stack.shape
    counts = np.asarray(counts).astype(np.int64)
    # suffix_sum[:, k] = sum of slots k..w-1
    suffix = np.cumsum(stack[:, ::-1, :], axis=1)[:, ::-1, :]
    take = (w - counts)                      # first valid slot index
    out = suffix[np.arange(n), take, :] / counts[:, None].astype(np.float32)
    return out.astype(np.float32, copy=False)


def pooled_features(spec: PoolSpec, stack: np.ndarray, counts: np.ndarray,
                    mean_wide: np.ndarray | None = None) -> np.ndarray:
    """2-D feature array for a *fixed* pooling operator.

    `stack` is the (n, STACK_WINDOW, d) clamped window stack; `mean_wide` is the
    precomputed mean over a window larger than `STACK_WINDOW` (only `mean16` uses it).
    """
    if spec.mode == "attn":
        raise ValueError("attn is learned and has no precomputable feature array")
    if spec.window > stack.shape[1]:
        if mean_wide is None or spec.mode != "mean":
            raise ValueError(
                f"{spec.name}: window {spec.window} exceeds the stored stack "
                f"({stack.shape[1]}) and is not a precomputed mean"
            )
        return mean_wide
    sub = stack[:, stack.shape[1] - spec.window:, :]
    sub_counts = np.minimum(counts, spec.window)
    if spec.mode == "last":
        return np.ascontiguousarray(sub[:, -1, :])
    if spec.mode == "mean":
        return masked_mean(sub, sub_counts)
    if spec.mode == "concat":
        return np.ascontiguousarray(sub.reshape(len(sub), -1))
    raise ValueError(f"unknown pooling mode {spec.mode!r}")


# ---------------------------------------------------------------------------------
# The learned pool
# ---------------------------------------------------------------------------------


class AttnPoolHead(nn.Module):
    """Single-query attention pool over a window, then the deployed MLP head.

        e_j   = v . tanh(W h_j)                       (scores, padding masked out)
        alpha = softmax(e)
        pooled= sum_j alpha_j h_j
        logits= MLPHead(pooled)

    `MLPHead` is constructed **first** so this module draws from the ambient RNG in
    exactly the order `MLPHead(...)` alone would.  At `w = 1` the softmax is
    identically 1, the pooled vector is `h_t`, and the extra parameters receive zero
    gradient -- so training this at `w = 1` reproduces the deployed head bit for bit.
    `tests/test_pooled_readout.py` asserts that rather than asserting the reasoning.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_bins: int, dropout: float = 0.1,
                 attn_dim: int = 128):
        super().__init__()
        self.mlp = MLPHead(in_dim, hidden_dim, n_bins, dropout)
        self.attn_proj = nn.Linear(in_dim, attn_dim)
        self.attn_query = nn.Linear(attn_dim, 1, bias=False)
        self.in_dim = in_dim
        self.n_bins = n_bins
        self.attn_dim = attn_dim

    def set_standardiser(self, x: np.ndarray) -> None:
        """Standardise on the *pooled* distribution's proxy: the individual states.

        The pooled vector is a convex combination of the window's states, so the
        per-feature mean of the states is also the mean of any convex combination of
        them; the scale is an upper bound.  Using the state distribution keeps this
        identical to `MLPHead.set_standardiser` at w = 1.
        """
        flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        self.mlp.set_standardiser(flat)

    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.attn_query(torch.tanh(self.attn_proj(x))).squeeze(-1)   # (B, w)
        e = e.masked_fill(~mask, float("-inf"))
        return torch.softmax(e, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = self.weights(x, mask)
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)
        return self.mlp(pooled)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, mask: np.ndarray,
                      batch_size: int = 4096) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(np.asarray(x[i:i + batch_size]), dtype=torch.float32)
            mb = torch.as_tensor(np.asarray(mask[i:i + batch_size]), dtype=torch.bool)
            out.append(torch.softmax(self(xb, mb), dim=-1).numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_bins))


def counts_to_mask(counts: np.ndarray, w: int) -> np.ndarray:
    """(n, w) bool: True on the `counts` trailing slots, which hold distinct states."""
    counts = np.asarray(counts).astype(np.int64)
    idx = np.arange(w)[None, :]
    return idx >= (w - np.minimum(counts, w))[:, None]


def train_attn_head(head: AttnPoolHead, x_train, m_train, y_train,
                    x_val, m_val, y_val, cfg: dict[str, Any]) -> TrainResult:
    """`heads.train_head`, transcribed for a (n, w, d) input plus a padding mask.

    Deliberately a transcription and not a refactor of `train_head`: the point of C25
    is a comparison between readouts, and any difference in the optimiser, the epoch
    selection or the shuffling order between "the deployed head as script 03 trained
    it" and "a pooled head as C25 trained it" would be indistinguishable from a
    pooling effect.  The window-size-1 gate is what checks the transcription.
    """
    torch.manual_seed(int(cfg["seed"]))
    head.set_standardiser(x_train)

    xt = torch.as_tensor(np.asarray(x_train), dtype=torch.float32)
    mt = torch.as_tensor(np.asarray(m_train), dtype=torch.bool)
    yt = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    xv = torch.as_tensor(np.asarray(x_val), dtype=torch.float32)
    mv = torch.as_tensor(np.asarray(m_val), dtype=torch.bool)
    yv = torch.as_tensor(np.asarray(y_val), dtype=torch.long)

    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
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
        perm = torch.randperm(n, generator=generator)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(head(xt[idx], mt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        train_nll = total / n

        head.eval()
        with torch.no_grad():
            val_nll = 0.0
            for i in range(0, len(xv), 8192):
                lg = head(xv[i:i + 8192], mv[i:i + 8192])
                val_nll += float(loss_fn(lg, yv[i:i + 8192])) * len(lg)
            val_nll /= len(xv)
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


# ---------------------------------------------------------------------------------
# Extraction: window stacks at chosen probe points, one forward pass for all of them
# ---------------------------------------------------------------------------------


@torch.no_grad()
def extract_window_states(
    gen: FrozenGenerator,
    sequences: Sequence[Sequence[int]],
    positions: Sequence[Sequence[int]],
    layers: Sequence[int],
    stacks: dict[int, np.ndarray],
    means: dict[int, np.ndarray] | None,
    counts_out: np.ndarray,
    counts_wide_out: np.ndarray | None,
    row_offsets: Sequence[int],
    stack_window: int = STACK_WINDOW,
    wide_window: int = 16,
    batch_size: int = 96,
    meter: ComputeMeter | None = None,
) -> None:
    """Window stacks (and one wide mean) at every requested probe point.

    Batching is `probe_layers.hidden_states_all_layers`' batching, unchanged: length
    sorted, right padded, explicit attention mask, `use_cache=False`.  That is what
    makes `stacks[L][:, -1, :]` bit-identical to the C17 layer arrays and hence to the
    dataset's own `hidden.npy`, which is the extraction half of the validity gate.

    One forward pass serves every probe point *and* every window size, so the token
    cost is charged once.
    """
    layers = list(layers)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))

    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        lens = [len(sequences[i]) for i in chunk]
        maxlen = max(lens)
        ids = torch.full((len(chunk), maxlen), gen.pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, :lens[r]] = torch.tensor(sequences[i], dtype=torch.long)
            mask[r, :lens[r]] = 1

        res = gen.model(
            input_ids=ids.to(gen.device),
            attention_mask=mask.to(gen.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        for L in layers:
            hs = res.hidden_states[L].float().cpu().numpy()
            for r, i in enumerate(chunk):
                o = int(row_offsets[i])
                ps = list(positions[i])
                idx = np.array([window_positions(int(p), stack_window) for p in ps])
                stacks[L][o:o + len(ps)] = hs[r][idx]
                if means is not None:
                    widx = np.array([window_positions(int(p), wide_window) for p in ps])
                    wc = np.array([window_count(int(p), wide_window) for p in ps])
                    blk = hs[r][widx]                       # (k, wide_window, d)
                    means[L][o:o + len(ps)] = masked_mean(blk, wc)
        for r, i in enumerate(chunk):
            o = int(row_offsets[i])
            ps = list(positions[i])
            counts_out[o:o + len(ps)] = [window_count(int(p), stack_window) for p in ps]
            if counts_wide_out is not None:
                counts_wide_out[o:o + len(ps)] = [
                    window_count(int(p), wide_window) for p in ps
                ]
        if meter is not None:
            meter.add_forward(int(sum(lens)))
        del res


# ---------------------------------------------------------------------------------
# Decode time
# ---------------------------------------------------------------------------------


class PooledTargetScorer:
    """`guidance.TargetScorer` for a pooled readout.

    Takes the window stack `(B, K, w, d)` and its padding mask and returns
    `P(y_final in I)` per candidate.  For a fixed operator the head is a plain
    `MLPHead` and the pooling happens here; for `attn` the head does its own pooling.
    """

    def __init__(self, head, binner, lo: float, hi: float, spec: PoolSpec):
        self.head = head
        self.binner = binner
        self.lo = lo
        self.hi = hi
        self.spec = spec
        self.mask = torch.as_tensor(binner.interval_mask(lo, hi))
        self._device: torch.device | None = None

    def to(self, device) -> "PooledTargetScorer":
        device = torch.device(device)
        if self._device != device:
            self.head = self.head.to(device)
            self.mask = self.mask.to(device)
            self._device = device
        return self

    @torch.no_grad()
    def __call__(self, window: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """window: (N, w, d) float; valid: (N, w) bool. Returns (N,)."""
        self.to(window.device)
        self.head.eval()
        w = window.shape[1]
        spec = self.spec
        if spec.mode == "attn":
            logits = self.head(window.float(), valid.to(window.device))
        else:
            sub = window[:, w - spec.window:, :].float()
            vsub = valid[:, w - spec.window:].to(window.device)
            if spec.mode == "last":
                feat = sub[:, -1, :]
            elif spec.mode == "mean":
                c = vsub.sum(dim=1, keepdim=True).clamp_min(1).to(sub.dtype)
                feat = (sub * vsub.unsqueeze(-1).to(sub.dtype)).sum(dim=1) / c
            elif spec.mode == "concat":
                feat = sub.reshape(len(sub), -1)
            else:
                raise ValueError(f"unknown pooling mode {spec.mode!r}")
            logits = self.head(feat)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self.mask].sum(dim=-1)


@torch.no_grad()
def _candidate_states_cached(gen: FrozenGenerator, past_key_values,
                             candidate_tokens: torch.Tensor, layer: int) -> torch.Tensor:
    b, k = candidate_tokens.shape
    cand_cache = repeat_cache(past_key_values, k)
    flat = candidate_tokens.reshape(b * k, 1)
    out = gen.model(input_ids=flat, past_key_values=cand_cache, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    return out.hidden_states[layer][:, 0, :].reshape(b, k, -1)


@torch.no_grad()
def _candidate_states_full(gen: FrozenGenerator, prefix_ids: torch.Tensor,
                           candidate_tokens: torch.Tensor, layer: int) -> torch.Tensor:
    b, k = candidate_tokens.shape
    t = prefix_ids.shape[1]
    ext = torch.cat(
        [prefix_ids.unsqueeze(1).expand(b, k, t).reshape(b * k, t),
         candidate_tokens.reshape(b * k, 1)], dim=1)
    out = gen.model(input_ids=ext, use_cache=False, output_hidden_states=True,
                    return_dict=True)
    return out.hidden_states[layer][:, -1, :].reshape(b, k, -1)


@torch.no_grad()
def pooled_guided_sample(
    gen: FrozenGenerator,
    scorer: PooledTargetScorer | None,
    window_fn,
    policy: dict,
    n_molecules: int,
    seed: int,
    top_k: int = 8,
    lam: float = 1.0,
    eps: float = 1e-6,
    backend: str = "cached",
    batch_size: int = 64,
    layer: int = -1,
    window: int = 1,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """`guidance.guided_sample` with a pooled readout over the last `window` positions.

    **The only structural difference** from `guided_sample` is the rolling buffer
    `hist` of committed hidden states at `layer`, obtained by asking the *same*
    KV-cached forward pass for its hidden states.  No extra token is processed, and the
    compute accounting is therefore unchanged; `tests/test_pooled_readout.py` asserts
    both the identical molecules and the identical token counts at `window = 1`.

    The buffer holds the states at positions `t-window+2 .. t`; the candidate supplies
    position `t+1`.  Padding, when the prefix is shorter than the window, is the same
    index-clamping used at training time and is reported through the `valid` mask so a
    mean is taken over the distinct positions only.
    """
    torch.manual_seed(seed)
    temperature = float(policy["temperature"])
    max_len = int(policy["max_length"])
    results: list[list[int]] = []
    keep = max(0, int(window) - 1)

    while len(results) < n_molecules:
        b = min(batch_size, n_molecules - len(results))
        seqs = [[gen.bos_id] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool)
        cur = torch.full((b, 1), gen.bos_id, dtype=torch.long, device=gen.device)
        pkv = None
        hist: list[torch.Tensor] = []   # committed states, oldest first, len <= keep

        for t in range(max_len - 1):
            need_states = keep > 0
            out = gen.model(input_ids=cur, past_key_values=pkv, use_cache=True,
                            output_hidden_states=need_states, return_dict=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :].float() / temperature
            logprobs = torch.log_softmax(logits, dim=-1)
            if meter is not None:
                active = int((~done).sum())
                meter.add_forward(active, active)
            if need_states:
                hist.append(out.hidden_states[layer][:, -1, :].float())
                if len(hist) > keep:
                    hist.pop(0)

            guide = window_fn(t) and bool((~done).any()) and (scorer is not None or lam == 0.0)
            if guide and lam == 0.0:
                cand_lp, cand_ids = torch.topk(logprobs, top_k, dim=-1)
                pick = torch.multinomial(torch.softmax(cand_lp, dim=-1), 1)
                nxt = cand_ids.gather(1, pick).squeeze(1)
            elif guide:
                cand_lp, cand_ids = torch.topk(logprobs, top_k, dim=-1)
                if backend == "cached":
                    hidden = _candidate_states_cached(gen, pkv, cand_ids, layer)
                    actual = int((~done).sum()) * top_k
                elif backend == "full":
                    prefix_ids = torch.tensor(
                        [s + [gen.pad_id] * (t + 1 - len(s)) for s in seqs],
                        dtype=torch.long, device=gen.device)
                    hidden = _candidate_states_full(gen, prefix_ids, cand_ids, layer)
                    actual = int((~done).sum()) * top_k * (t + 2)
                else:
                    raise ValueError(f"unknown backend {backend!r}")
                if meter is not None:
                    meter.add_forward(actual, int((~done).sum()) * top_k * (t + 2))

                # (B, K, w, d): the last `keep` committed states, then the candidate.
                d = hidden.shape[-1]
                if keep:
                    ctx = torch.stack(hist, dim=1)                      # (B, m, d)
                    m = ctx.shape[1]
                    if m < keep:                                        # index clamping
                        ctx = torch.cat([ctx[:, :1].expand(b, keep - m, d), ctx], dim=1)
                    ctx = ctx.unsqueeze(1).expand(b, top_k, keep, d)
                    win = torch.cat([ctx, hidden.unsqueeze(2)], dim=2)
                    # positions available: t+2 states exist at candidate position t+1
                    n_avail = min(int(window), t + 2)
                    valid = torch.zeros((b, top_k, int(window)), dtype=torch.bool,
                                        device=hidden.device)
                    valid[:, :, int(window) - n_avail:] = True
                else:
                    win = hidden.unsqueeze(2)
                    valid = torch.ones((b, top_k, 1), dtype=torch.bool,
                                       device=hidden.device)

                q = scorer(win.reshape(-1, win.shape[-2], d),
                           valid.reshape(-1, valid.shape[-1])).reshape(cand_lp.shape)
                score = cand_lp + lam * torch.log(q.clamp_min(0.0) + eps)
                pick = torch.multinomial(torch.softmax(score, dim=-1), 1)
                nxt = cand_ids.gather(1, pick).squeeze(1)
            else:
                nxt = torch.multinomial(torch.softmax(logprobs, dim=-1), 1).squeeze(1)

            nxt = torch.where(done.to(nxt.device), torch.full_like(nxt, gen.pad_id), nxt)
            for i in range(b):
                if not done[i]:
                    seqs[i].append(int(nxt[i]))
            done = done | (nxt.cpu() == gen.eos_id)
            cur = nxt.unsqueeze(1)
            if bool(done.all()):
                break

        for s in seqs:
            results.append(s)
            if meter is not None:
                meter.molecules_returned += 1

    return results[:n_molecules]
