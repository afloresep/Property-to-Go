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
    def fit(cls, values: np.ndarray, n_bins: int) -> "QuantileBinner":
        values = np.asarray(values, dtype=np.float64)
        levels = np.linspace(0.0, 1.0, n_bins + 1)
        inner = np.quantile(values, levels[1:-1])
        inner = np.maximum.accumulate(inner)  # guard against ties
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        idx = cls._digitize(edges, values)
        centers = np.array(
            [
                np.median(values[idx == b]) if np.any(idx == b) else 0.5 * (edges[b] + edges[b + 1])
                for b in range(n_bins)
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
