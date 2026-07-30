"""Compute-matched best-of-N.

The comparison Property-to-Go has to survive: spend the same number of processed
generator tokens on plain sampling, keep the candidate closest to the target, and
see whether guidance still wins.
"""

from __future__ import annotations

import numpy as np

from .compute import ComputeMeter, solve_best_of_n
from .generation import sample_unconditional
from .model_io import FrozenGenerator
from .properties import compute_properties


def in_target(value: float, lo: float, hi: float) -> bool:
    """The interval is half-open [lo, hi), matching how hit rate is scored."""
    return bool(lo <= value < hi)


def target_distance(value: float, lo: float, hi: float) -> float:
    """Set distance to [lo, hi): 0 inside, otherwise distance to the nearer edge.

    Note this is 0 for `value == hi` as well, because hi is the interval's
    infimum from above.  That makes it correct as a *distance* but useless as a
    sole ranking key for integer-valued properties: with the aromatic-ring target
    [3, 4), a 4-ring molecule is distance 0 yet is not a hit.  Rank with
    `selection_key`, which puts membership ahead of distance.
    """
    if value < lo:
        return lo - value
    if value >= hi:
        return value - hi
    return 0.0


#: Properties whose values are counts.  For these the target interval [v, v+1)
#: contains exactly one attainable value, so "distance to the interval" must be
#: measured to that value and not to the open upper edge.
INTEGER_PROPERTIES = frozenset({"aromatic_rings"})


def target_error(value: float, lo: float, hi: float, integer_valued: bool = False) -> float:
    """How far a molecule actually missed the target by.

    `target_distance` is the mathematical distance to the half-open interval and
    is therefore 0 at `value == hi`.  Reporting 0 error for a molecule that did
    not hit the target is simply wrong, so this function measures to the nearest
    *attainable* target value: for a count property with target [3, 4), a 4-ring
    molecule is one ring out, not zero.
    """
    if in_target(value, lo, hi):
        return 0.0
    if value < lo:
        return lo - value
    upper = (hi - 1.0) if integer_valued else hi
    return value - upper


def selection_key(value: float, lo: float, hi: float) -> tuple[int, float]:
    """Ranking key for best-of-N: genuine hits beat every non-hit, then distance.

    Ordering by `target_distance` alone silently ties a 4-ring molecule with a
    3-ring hit when the target is [3, 4), so best-of-N returns the first of the
    two and its measured hit rate collapses.  Membership must dominate.
    """
    return (0 if in_target(value, lo, hi) else 1, target_distance(value, lo, hi))


def best_of_n(
    gen: FrozenGenerator,
    policy: dict,
    prop: str,
    lo: float,
    hi: float,
    n_molecules: int,
    n_candidates: int,
    seed: int,
    meter: ComputeMeter | None = None,
) -> list[dict]:
    """Return one selected molecule per output slot, each chosen from N candidates.

    Invalid candidates are ranked worst, so a slot can still return an invalid
    molecule if every candidate was invalid.  That keeps validity comparable with
    the guided runs instead of being silently repaired by the selection rule.
    """
    selected: list[dict] = []
    local = meter if meter is not None else ComputeMeter()

    # Draw the whole candidate pool with the generator's own batching, then split it
    # into per-slot groups.  Drawing N at a time would batch at N, which is far
    # slower for small N and gives the baseline an unfair wall-clock handicap.
    total = n_molecules * n_candidates
    seqs = sample_unconditional(gen, policy, total, seed=seed, meter=local)
    smiles = gen.decode(seqs)

    for i in range(n_molecules):
        lo_i = i * n_candidates
        best = None
        best_key = None
        for s, ids in zip(smiles[lo_i : lo_i + n_candidates], seqs[lo_i : lo_i + n_candidates]):
            props = compute_properties(s)
            if props is None:
                key = (1, 1, float("inf"))
                cand = {"smiles": s, "token_ids": ids, "valid": False}
            else:
                key = (0, *selection_key(props[prop], lo, hi))
                cand = {"smiles": s, "token_ids": ids, "valid": True, **props}
            if best_key is None or key < best_key:
                best_key, best = key, cand
        selected.append(best)  # type: ignore[arg-type]

    if meter is not None:
        # molecules_returned must count *returned* molecules, not candidates drawn,
        # otherwise tokens-per-molecule would understate the true cost by N.
        meter.molecules_returned = n_molecules
    return selected


def match_n_to_guided(
    guided_tokens_per_molecule: float, base_tokens_per_molecule: float
) -> int:
    return solve_best_of_n(guided_tokens_per_molecule, base_tokens_per_molecule)


def summarise(records: list[dict], prop: str, lo: float, hi: float) -> dict:
    vals = np.array([r[prop] for r in records if r.get("valid")], dtype=np.float64)
    n = len(records)
    if len(vals) == 0:
        return {"n": n, "validity": 0.0}
    hit = (vals >= lo) & (vals < hi)
    integer_valued = prop in INTEGER_PROPERTIES
    dists = np.array([target_error(v, lo, hi, integer_valued) for v in vals])
    canon = [r["canonical_smiles"] for r in records if r.get("valid")]
    return {
        "n": n,
        "n_valid": int(len(vals)),
        "validity": float(len(vals) / n),
        "uniqueness": float(len(set(canon)) / len(canon)),
        "hit_rate": float(hit.mean()),
        "hit_rate_over_all_returned": float(hit.sum() / n),
        "abs_target_error_mean": float(dists.mean()),
        "abs_target_error_median": float(np.median(dists)),
        f"{prop}_mean": float(vals.mean()),
        f"{prop}_std": float(vals.std()),
        f"{prop}_quantiles": {
            str(q): float(np.quantile(vals, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }
