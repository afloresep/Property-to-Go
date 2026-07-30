"""Grouped train/val/test splitting.

Splits are grouped by *canonical completed molecule*.  GP-MoLFormer-Uniq re-emits
popular molecules often, so an ungrouped split would put the same molecule in train
and test and inflate every held-out number.  Assignment is by a stable hash of the
canonical SMILES, so the split is reproducible and does not depend on the order in
which trajectories were generated or on the dataset size.
"""

from __future__ import annotations

import hashlib

import numpy as np

SPLITS = ("train", "val", "test")


def _stable_unit(key: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_split(group_key: str, fractions: dict[str, float], seed: int) -> str:
    """Map one group key to a split, deterministically."""
    total = sum(fractions[s] for s in SPLITS)
    u = _stable_unit(group_key, seed) * total
    acc = 0.0
    for s in SPLITS:
        acc += fractions[s]
        if u < acc:
            return s
    return SPLITS[-1]


def split_by_group(
    group_keys: list[str], fractions: dict[str, float], seed: int
) -> np.ndarray:
    """Per-row split labels for rows carrying the given group keys."""
    cache: dict[str, str] = {}
    out = []
    for k in group_keys:
        if k not in cache:
            cache[k] = assign_split(k, fractions, seed)
        out.append(cache[k])
    return np.array(out)


def check_no_group_leakage(group_keys: np.ndarray, splits: np.ndarray) -> dict[str, int]:
    """Assert every group lives in exactly one split; return group counts per split."""
    seen: dict[str, str] = {}
    for g, s in zip(group_keys, splits):
        if g in seen and seen[g] != s:
            raise AssertionError(f"group {g!r} appears in both {seen[g]} and {s}")
        seen[g] = s
    counts = {s: 0 for s in SPLITS}
    for s in seen.values():
        counts[s] += 1
    return counts
