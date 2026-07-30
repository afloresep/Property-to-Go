"""Held-out evaluation metrics for the future-property heads."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .binning import interval_probability


def categorical_nll(probs: np.ndarray, y_bin: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y_bin)), y_bin], eps, 1.0)
    return float(-np.mean(np.log(p)))


def expected_value_mae(probs: np.ndarray, y_true: np.ndarray, binner) -> float:
    return float(np.mean(np.abs(binner.expected_value(probs) - y_true)))


def expected_value_rmse(probs: np.ndarray, y_true: np.ndarray, binner) -> float:
    return float(np.sqrt(np.mean((binner.expected_value(probs) - y_true) ** 2)))


def rank_correlation(probs: np.ndarray, y_true: np.ndarray, binner) -> float:
    pred = binner.expected_value(probs)
    if len(np.unique(pred)) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(spearmanr(pred, y_true).statistic)


def brier(probs_in_interval: np.ndarray, hit: np.ndarray) -> float:
    return float(np.mean((probs_in_interval - hit.astype(np.float64)) ** 2))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC; nan when only one class is present."""
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reliability(
    probs_in_interval: np.ndarray, hit: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Coarse reliability curve: mean predicted vs observed frequency per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], probs_in_interval, side="right"), 0, n_bins - 1)
    pred, obs, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        count.append(int(m.sum()))
        pred.append(float(probs_in_interval[m].mean()) if m.any() else float("nan"))
        obs.append(float(hit[m].mean()) if m.any() else float("nan"))
    return {"bin_edges": edges.tolist(), "mean_predicted": pred, "observed": obs, "count": count}


def expected_calibration_error(probs_in_interval: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> float:
    r = reliability(probs_in_interval, hit, n_bins)
    total = sum(r["count"])
    if total == 0:
        return float("nan")
    ece = 0.0
    for p, o, c in zip(r["mean_predicted"], r["observed"], r["count"]):
        if c:
            ece += c / total * abs(p - o)
    return float(ece)


def evaluate(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_bin: np.ndarray,
    binner,
    intervals: dict[str, tuple[float, float]],
) -> dict:
    """Full metric bundle for one head on one evaluation set."""
    out: dict = {
        "n": int(len(y_true)),
        "nll": categorical_nll(probs, y_bin),
        "expected_value_mae": expected_value_mae(probs, y_true, binner),
        "expected_value_rmse": expected_value_rmse(probs, y_true, binner),
        "spearman": rank_correlation(probs, y_true, binner),
        "intervals": {},
    }
    for name, (lo, hi) in intervals.items():
        q = interval_probability(probs, binner, lo, hi)
        hit = (y_true >= lo) & (y_true < hi)
        out["intervals"][name] = {
            "lo": float(lo),
            "hi": float(hi),
            "base_rate": float(hit.mean()),
            "brier": brier(q, hit),
            "auroc": auroc(q, hit),
            "ece": expected_calibration_error(q, hit),
            "mean_predicted": float(q.mean()),
            "reliability": reliability(q, hit),
        }
    return out


def evaluate_by_group(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_bin: np.ndarray,
    groups: np.ndarray,
    binner,
    intervals: dict[str, tuple[float, float]],
) -> dict:
    """Same bundle, computed separately per prefix-position quartile."""
    res = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < 20:
            continue
        res[str(g)] = evaluate(probs[m], y_true[m], y_bin[m], binner, intervals)
    return res


def bootstrap_ci(
    fn, *arrays, n_boot: int = 500, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for a scalar metric over paired arrays."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
