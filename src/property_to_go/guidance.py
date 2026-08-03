"""Property-to-Go guided decoding.

At a guided step, for each of the eight most likely next tokens a:

    score(a) = log p_base(a | prefix) + lambda * log(P(y_final in I | prefix + a) + eps)

and the next token is sampled from softmax(score) over those eight candidates.  The
generator is frozen; only the sampling distribution changes.

Two candidate-evaluation backends produce the extended-prefix hidden states:

  "full"   re-runs each of the eight extended prefixes from <bos> in one padded
           batch.  This is the reference implementation in the README.
  "cached" prefills the shared prefix once through the model's own public
           past_key_values API and takes one token step per candidate.

They are mathematically the same computation; tests/test_candidate_backends.py
checks that they agree numerically on real prefixes.  Token accounting records the
cost of both so results can be quoted under either rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .compute import ComputeMeter
from .generation import repeat_cache
from .model_io import FrozenGenerator

WindowFn = Callable[[int], bool]


def combine_scores(
    base_logprobs: torch.Tensor, target_probs: torch.Tensor, lam: float, eps: float
) -> torch.Tensor:
    """score(a) = log p_base(a | prefix) + lambda * log(P(final in I | prefix + a) + eps).

    Kept separate from the decoding loop so the combination rule itself is testable.
    lam=0 reproduces the base log-probabilities exactly (up to an additive constant).
    """
    return base_logprobs + lam * torch.log(target_probs.clamp_min(0.0) + eps)


@dataclass
class Windows:
    """Frozen early/middle/late token windows, defined from base sequence lengths."""

    t33: int
    t67: int
    source: str = "base content-length distribution"

    @classmethod
    def from_lengths(cls, lengths: np.ndarray, quantiles: tuple[float, float]) -> "Windows":
        """Split generation positions into three equal-mass thirds.

        The quantiles are taken over the pooled distribution of *token positions*
        the base model actually generates (position t = 1..n for a trajectory of
        length n), not over final lengths.  Quantiles of the length distribution
        would put the 33rd percentile near the median molecule's end, so `early`
        would swallow almost the whole trajectory.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        positions = np.concatenate([np.arange(1, n + 1) for n in lengths])
        q = np.quantile(positions.astype(np.float64), list(quantiles))
        return cls(
            t33=int(round(q[0])),
            t67=int(round(q[1])),
            source="pooled generated-position distribution of the base model",
        )

    def fn(self, condition: str) -> WindowFn:
        """Map a condition name to a predicate over the 0-based step index."""
        if condition in ("throughout", "truncation_control"):
            return lambda t: True
        if condition == "early":
            return lambda t: t < self.t33
        if condition == "middle":
            return lambda t: self.t33 <= t < self.t67
        if condition == "late":
            return lambda t: t >= self.t67
        if condition == "unguided":
            return lambda t: False
        raise ValueError(f"unknown condition {condition!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "t33": self.t33,
            "t67": self.t67,
            "source": self.source,
            "early": [1, self.t33],
            "middle": [self.t33, self.t67],
            "late": [self.t67, None],
        }


class TargetScorer:
    """Wraps a trained head as h_t -> P(y_final in I).

    Heads are checkpointed and loaded with `map_location="cpu"` while the frozen
    generator may live on a GPU, so the head is migrated to whatever device the
    candidate hidden states arrive on.  Done lazily, and once: on CPU this is a
    no-op, and the alternative -- every caller remembering to `.to(device)` -- is
    the kind of thing that works until one script forgets.
    """

    def __init__(self, head, binner, lo: float, hi: float):
        self.head = head
        self.binner = binner
        self.lo = lo
        self.hi = hi
        self.mask = torch.as_tensor(binner.interval_mask(lo, hi))
        self._device: torch.device | None = None

    def to(self, device) -> "TargetScorer":
        device = torch.device(device)
        if self._device != device:
            self.head = self.head.to(device)
            self.mask = self.mask.to(device)
            self._device = device
        return self

    @torch.no_grad()
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.to(hidden.device)
        self.head.eval()
        probs = torch.softmax(self.head(hidden.float()), dim=-1)
        return probs[:, self.mask].sum(dim=-1)


@torch.no_grad()
def _candidate_states_cached(
    gen: FrozenGenerator,
    past_key_values,
    candidate_tokens: torch.Tensor,  # (B, K)
    layer: int,
) -> torch.Tensor:
    """Hidden states after appending each candidate, via the model's cache API.

    The cache *layout* is the one thing here that is architecture-specific.
    `generation.repeat_cache` is written for GP-MoLFormer's linear-attention
    running-sum cache; a standard-attention generator carries `(key, value)` per layer
    and needs `generality.repeat_cache_gpt2`.  A generator may therefore supply its own
    repeat via `repeat_cache_fn` (C31's `second_generator.ZincGPT2Generator` does).
    `FrozenGenerator` has no such attribute, so every molecular call site takes the
    `repeat_cache` default and no molecular number can move -- which is why this is a
    lookup with a default rather than a branch on model type.
    """
    b, k = candidate_tokens.shape
    repeat = getattr(gen, "repeat_cache_fn", None) or repeat_cache
    cand_cache = repeat(past_key_values, k)
    flat = candidate_tokens.reshape(b * k, 1)
    out = gen.model(
        input_ids=flat,
        past_key_values=cand_cache,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[layer][:, 0, :].reshape(b, k, -1)


@torch.no_grad()
def _candidate_states_full(
    gen: FrozenGenerator,
    prefix_ids: torch.Tensor,  # (B, T) committed tokens including <bos>
    candidate_tokens: torch.Tensor,  # (B, K)
    layer: int,
) -> torch.Tensor:
    """Hidden states after appending each candidate, by full-prefix recomputation."""
    b, k = candidate_tokens.shape
    t = prefix_ids.shape[1]
    ext = torch.cat(
        [
            prefix_ids.unsqueeze(1).expand(b, k, t).reshape(b * k, t),
            candidate_tokens.reshape(b * k, 1),
        ],
        dim=1,
    )
    out = gen.model(input_ids=ext, use_cache=False, output_hidden_states=True, return_dict=True)
    return out.hidden_states[layer][:, -1, :].reshape(b, k, -1)


@torch.no_grad()
def guided_sample(
    gen: FrozenGenerator,
    scorer: TargetScorer | None,
    window_fn: WindowFn,
    policy: dict,
    n_molecules: int,
    seed: int,
    top_k: int = 8,
    lam: float = 1.0,
    eps: float = 1e-6,
    backend: str = "cached",
    batch_size: int = 64,
    layer: int = -1,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """Sample molecules under Property-to-Go guidance restricted to a window.

    Steps outside the window are sampled from the unmodified base policy over the
    full vocabulary, so a windowed run differs from the unguided baseline only
    inside its window.
    """
    torch.manual_seed(seed)
    temperature = float(policy["temperature"])
    max_len = int(policy["max_length"])
    results: list[list[int]] = []

    while len(results) < n_molecules:
        b = min(batch_size, n_molecules - len(results))
        seqs = [[gen.bos_id] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool)
        cur = torch.full((b, 1), gen.bos_id, dtype=torch.long, device=gen.device)
        pkv = None

        for t in range(max_len - 1):
            out = gen.model(
                input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True
            )
            pkv = out.past_key_values
            logits = out.logits[:, -1, :].float() / temperature
            logprobs = torch.log_softmax(logits, dim=-1)
            if meter is not None:
                active = int((~done).sum())
                meter.add_forward(active, active)

            guide = window_fn(t) and bool((~done).any()) and (scorer is not None or lam == 0.0)
            if guide and lam == 0.0:
                # truncation control: top-8 restriction with no property term at all,
                # so the head is never evaluated and costs nothing.
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
                        dtype=torch.long,
                        device=gen.device,
                    )
                    hidden = _candidate_states_full(gen, prefix_ids, cand_ids, layer)
                    actual = int((~done).sum()) * top_k * (t + 2)
                else:
                    raise ValueError(f"unknown backend {backend!r}")
                if meter is not None:
                    # full-recompute accounting: every candidate re-reads the whole prefix
                    meter.add_forward(actual, int((~done).sum()) * top_k * (t + 2))

                q = scorer(hidden.reshape(-1, hidden.shape[-1])).reshape(cand_lp.shape)
                score = combine_scores(cand_lp, q, lam, eps)
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
