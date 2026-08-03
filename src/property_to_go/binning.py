"""Discretising the terminal property distribution.

The head predicts a categorical distribution over bins rather than a point value, so
one trained head serves any target interval: P(y in I) is the sum of the bin
probabilities inside I.  Target intervals are therefore always defined on bin
boundaries, which makes that sum exact rather than an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class QuantileBinner:
    """Quantile bins fitted on training terminal values (used for cLogP, MW)."""

    edges: np.ndarray  # length n_bins + 1, outer edges are -inf / +inf
    centers: np.ndarray  # representative value per bin (train median)
    quantile_levels: np.ndarray

    kind: str = "quantile"

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        n_bins: int,
        extra_edges: tuple[float, ...] | list[float] = (),
    ) -> "QuantileBinner":
        """Fit equal-mass bins, optionally forcing extra values to be bin boundaries.

        `extra_edges` exists because of a real defect it prevents.  `interval_mask`
        selects only bins lying *wholly* inside [lo, hi), so if a target-interval edge
        falls in the middle of a bin, that bin is dropped and `P(y in I)` silently
        becomes the probability of a *subset* of the target.  Nothing raises; the head
        simply learns to predict the wrong event.

        This is not hypothetical.  The pilot's cLogP interval came from the full
        49,823-molecule sample while its binner was fitted on the train split's 158,588
        prefix rows, so the two disagreed by ~1.3e-3 -- enough to drop one of the two
        bins and leave the head predicting a 0.050-mass event for a 0.100-base-rate
        target.  See `pilot_report.md` §11.5.

        Passing the target interval's edges here makes the interval exactly a union of
        bins, which is the invariant this module's docstring claims and did not enforce.
        An extra edge closer than 1e-9 to an existing one is dropped, since
        `interval_mask`'s own tolerance already covers that case.
        """
        values = np.asarray(values, dtype=np.float64)
        levels = np.linspace(0.0, 1.0, n_bins + 1)
        inner = np.quantile(values, levels[1:-1])
        inner = np.maximum.accumulate(inner)  # guard against ties
        for e in extra_edges:
            e = float(e)
            if np.isfinite(e) and (len(inner) == 0 or np.abs(inner - e).min() > 1e-9):
                inner = np.append(inner, e)
        inner = np.unique(inner)  # sorted and deduplicated
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        # `len(edges) - 1`, not `n_bins`: an extra edge adds a bin, and using the
        # requested count here would silently truncate the last one.
        idx = cls._digitize(edges, values)
        centers = np.array(
            [
                np.median(values[idx == b]) if np.any(idx == b) else 0.5 * (edges[b] + edges[b + 1])
                for b in range(len(edges) - 1)
            ]
        )
        return cls(edges=edges, centers=centers, quantile_levels=levels)

    @staticmethod
    def _digitize(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)

    @property
    def n_bins(self) -> int:
        return len(self.centers)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self._digitize(self.edges, np.asarray(values, dtype=np.float64))

    def expected_value(self, probs: np.ndarray) -> np.ndarray:
        return probs @ self.centers

    def interval_mask(self, lo: float, hi: float) -> np.ndarray:
        """Bins whose whole range lies inside [lo, hi).

        Because target intervals are defined at quantile edges this is exact for
        the intervals this project actually uses; a tolerance handles float noise.
        """
        tol = 1e-9
        return np.array(
            [
                (self.edges[b] >= lo - tol) and (self.edges[b + 1] <= hi + tol)
                for b in range(self.n_bins)
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "edges": [float(e) for e in self.edges],
            "centers": [float(c) for c in self.centers],
            "quantile_levels": [float(q) for q in self.quantile_levels],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantileBinner":
        return cls(
            edges=np.array(d["edges"], dtype=np.float64),
            centers=np.array(d["centers"], dtype=np.float64),
            quantile_levels=np.array(d["quantile_levels"], dtype=np.float64),
        )


@dataclass
class CategoricalBinner:
    """Integer categories 0..max_value, with max_value acting as 'max_value or more'."""

    max_value: int
    kind: str = "categorical"

    @property
    def n_bins(self) -> int:
        return self.max_value + 1

    @property
    def centers(self) -> np.ndarray:
        return np.arange(self.n_bins, dtype=np.float64)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values).astype(int), 0, self.max_value)

    def expected_value(self, probs: np.ndarray) -> np.ndarray:
        return probs @ self.centers

    def interval_mask(self, lo: float, hi: float) -> np.ndarray:
        """Categories c with lo <= c < hi (an exact value v is [v, v+1))."""
        c = self.centers
        return (c >= lo) & (c < hi)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "max_value": int(self.max_value)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CategoricalBinner":
        return cls(max_value=int(d["max_value"]))


def binner_from_dict(d: dict[str, Any]):
    if d["kind"] == "quantile":
        return QuantileBinner.from_dict(d)
    if d["kind"] == "categorical":
        return CategoricalBinner.from_dict(d)
    raise ValueError(f"unknown binner kind {d['kind']!r}")


def interval_probability(probs: np.ndarray, binner, lo: float, hi: float) -> np.ndarray:
    """P(y in [lo, hi)) from a categorical distribution over bins."""
    mask = binner.interval_mask(lo, hi)
    probs = np.atleast_2d(probs)
    return probs[:, mask].sum(axis=1)


def in_interval(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return (v >= lo) & (v < hi)


def interval_mask_coverage(binner, lo: float, hi: float, values: np.ndarray) -> dict[str, Any]:
    """How faithfully `binner`'s interval mask represents [lo, hi) on real values.

    The head is trained to predict a distribution over bins and `P(y in I)` is read off
    as a sum of bin probabilities.  That is exact only when [lo, hi) is a union of whole
    bins.  When it is not, `interval_mask` drops the partially covered bins and the head
    ends up predicting a strict subset of the target -- quietly, with no error and no
    obviously wrong number, presenting instead as a miscalibrated head.

    So the identity is checked numerically rather than reasoned about: bin the values,
    sum the mask, and compare against the empirical rate.

    Returns `masked_rate`, `true_rate`, their absolute difference, and `is_exact`.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mask = np.asarray(binner.interval_mask(lo, hi), dtype=bool)
    bins = binner.transform(values)
    masked_rate = float(mask[bins].mean()) if len(values) else float("nan")
    true_rate = float(in_interval(values, lo, hi).mean()) if len(values) else float("nan")
    return {
        "n_bins": int(binner.n_bins),
        "n_bins_selected": int(mask.sum()),
        "masked_rate": masked_rate,
        "true_rate": true_rate,
        "abs_difference": abs(masked_rate - true_rate),
        "is_exact": bool(abs(masked_rate - true_rate) < 1e-6),
    }


def resolve_target_interval(rule: dict[str, Any], values: np.ndarray) -> dict[str, Any]:
    """Turn a `target_interval_rule` entry into a concrete half-open [lo, hi).

    Lives here rather than inline in `scripts/02_generate_trajectories.py` because it
    is the pre-registration: the rule is committed in `configs/guidance.yaml` before
    the base distribution of a new property has been looked at, and the interval falls
    out of the rule.  Being a function, it is testable and it is the *same* code path
    for the pilot's three frozen intervals as for the four phase-2 ones -- which is
    what makes "re-deriving them cannot move them" a checkable statement.

    Three kinds:

    `quantile_band {lo: a, hi: b}`
        [quantile(a), quantile(b)).  Base rate is b - a by construction, so every
        property using this rule is base-rate matched to every other one.

    `exact_value {value: v}`
        [v, v+1).  Retained because it is what the pilot froze for aromatic rings.

    `quantile_value {q: a}`
        [v, v+1) for v = round(quantile(a)).  The count-property analogue of
        `quantile_band`: one count unit wide, positioned by the base distribution
        rather than by hand.  Base rate is whatever the distribution gives and is
        reported, not tuned -- a count distribution can be lumpy, and a target with an
        awkward base rate is a fact to report rather than a reason to pick again.
    """
    values = np.asarray(values, dtype=np.float64)
    kind = rule["kind"]
    if kind == "quantile_band":
        lo = float(np.quantile(values, float(rule["lo"])))
        hi = float(np.quantile(values, float(rule["hi"])))
    elif kind == "exact_value":
        lo = float(int(rule["value"]))
        hi = lo + 1.0
    elif kind == "quantile_value":
        lo = float(int(round(float(np.quantile(values, float(rule["q"]))))))
        hi = lo + 1.0
    else:
        raise ValueError(f"unknown target_interval_rule kind {kind!r}")
    # Deliberately the same four keys the pilot's frozen `target_intervals.json`
    # already has, and in the same order.  Interval width is `hi - lo` and is
    # recomputed where it is needed; adding it here would gratuitously change a
    # tracked artefact that three phases of results are bound to.
    return {
        "lo": lo,
        "hi": hi,
        "rule": rule,
        "base_rate": float(in_interval(values, lo, hi).mean()),
    }
