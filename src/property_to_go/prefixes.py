"""Prefix selection.

For each completed trajectory with n content tokens, one prefix is drawn uniformly
from each quartile of [1, n].  Quartile q covers content lengths

    k in ( floor((q-1)n/4), floor(qn/4) ]

so the four prefixes span the trajectory from very early to essentially complete.
A prefix of content length k corresponds to index k in the full token sequence
(index 0 is <bos>), which is the position whose hidden state we read.
"""

from __future__ import annotations

import numpy as np


def quartile_bounds(n_content: int) -> list[tuple[int, int]]:
    """Inclusive (lo, hi) content-length bounds for the four quartiles."""
    bounds = []
    prev = 0
    for q in range(1, 5):
        hi = (q * n_content) // 4
        lo = prev + 1
        bounds.append((lo, hi))
        prev = hi
    return bounds


def select_quartile_prefixes(n_content: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return [(quartile, prefix_content_length), ...], one per quartile.

    Raises ValueError when the trajectory is too short for four non-empty quartiles.
    """
    if n_content < 4:
        raise ValueError(f"need >= 4 content tokens for four quartiles, got {n_content}")
    out = []
    for q, (lo, hi) in enumerate(quartile_bounds(n_content), start=1):
        if hi < lo:
            raise ValueError(f"empty quartile {q} for n_content={n_content}")
        out.append((q, int(rng.integers(lo, hi + 1))))
    return out


def relative_position(prefix_len: int, n_content: int) -> float:
    return prefix_len / n_content if n_content else 0.0


def balanced_position_sample(
    prefix_lens: np.ndarray,
    quartiles: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices of `n_total` prefixes spread evenly across the four quartiles."""
    per = n_total // 4
    chosen: list[int] = []
    for q in (1, 2, 3, 4):
        idx = np.flatnonzero(quartiles == q)
        take = min(per, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.array(sorted(chosen))
