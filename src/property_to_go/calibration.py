"""Post-hoc calibration of the head's target-interval probability (C18, route a).

The head is trained on base-policy prefixes and consumed on the prefixes guidance
itself produces.  `pilot_report.md` §9.2.1 measured a large off-policy gap; §11.6 then
showed that roughly half of it was an interval-mask defect in our own code rather than
distribution shift.  This module is the machinery for re-measuring that gap and for
the one permitted cheap fix -- **post-hoc** recalibration, which is not DAgger.

## The algebra that has to be stated before any of this is used

The decoder samples the next token from

    softmax_a( log p_base(a) + lam * log( q(a) + eps ) )

over k = 8 candidates.  Apply a calibration map `g` to `q` and the score becomes
`log p_base(a) + lam * log(g(q(a)) + eps)`.  Then:

**A power map is exactly a lambda rescale.**  For `g(q) = c * q**alpha`,

    lam * log g(q) = (lam * alpha) * log q + lam * log c

and the second term does not depend on the candidate, so the softmax cancels it --
the same shift-invariance §16.3 used.  The calibrated head at `lam` induces *exactly*
the sampling distribution the uncalibrated head induces at `lam * alpha`.  Anything in
this family is therefore a point on the lambda sweep already reported in §19 and buys
no new information.  `PowerCalibrator.equivalent_lambda` states the mapping and
`equivalent_lambda_is_exact` checks it numerically.

**Platt scaling is that family to first order.**  `g(q) = sigmoid(a*logit(q) + b)`
tends to `exp(b) * q**a` as `q -> 0`, and our candidate probabilities live near base
rates of 0.08-0.17, so Platt scaling is a lambda rescale by its slope plus a residual
that has to be measured rather than assumed.

**Isotonic regression is not, but it is still monotone in q.**  It makes the effective
lambda `d log g(q) / d log q` depend on q, which no scalar lambda can do.  What it
cannot do is change *which* candidate the head prefers, because it is monotone -- so it
is bounded by the same argmax the `lam -> infinity` limit converges on.

**Temperature scaling of the head's bin logits is not even a function of q alone.**
`q(T) = sum_{i in M} exp(z_i/T) / sum_j exp(z_j/T)` depends on the whole logit vector,
so two candidates with equal q can move differently and the candidate ranking *can*
change.  That is the only thing in this module that a lambda rescale cannot imitate.

Everything here is pure numpy plus a thin torch wrapper, deliberately: the project's
`metrics.py` is dependency-light and a calibrator that needs scikit-learn at decode
time would be a new pin for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .binning import in_interval
from .guidance import TargetScorer

__all__ = [
    "PowerCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "IdentityCalibrator",
    "calibrator_from_dict",
    "fit_platt",
    "fit_isotonic",
    "fit_power_approximation",
    "calibration_report",
    "CalibratedTargetScorer",
    "equivalent_lambda_is_exact",
]

_CLIP = 1e-6


def _logit(q: np.ndarray, clip: float = _CLIP) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), clip, 1.0 - clip)
    return np.log(q) - np.log1p(-q)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# --------------------------------------------------------------------------- maps


@dataclass
class IdentityCalibrator:
    """No calibration.  Exists so the baseline is a first-class object, not a None."""

    kind: str = "identity"

    def apply(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass
class PowerCalibrator:
    """`g(q) = exp(log_c) * q**alpha` -- exactly a rescale of lambda.

    Kept as an explicit class rather than left implicit because the *point* of C18 is
    that this family is not a new experiment.  Having it as an object means the claim
    can be asserted by a test instead of argued in prose.
    """

    alpha: float
    log_c: float = 0.0
    kind: str = "power"

    def apply(self, q: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
        with np.errstate(divide="ignore"):
            return np.exp(self.log_c + self.alpha * np.log(np.clip(q, _CLIP, None)))

    def equivalent_lambda(self, lam: float) -> float:
        """The lambda an *uncalibrated* head would need to induce the same sampling."""
        return float(lam) * float(self.alpha)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "alpha": float(self.alpha), "log_c": float(self.log_c)}


@dataclass
class PlattCalibrator:
    """`g(q) = sigmoid(a * logit(q) + b)`, the standard scalar recalibration.

    `power_limit()` returns the `PowerCalibrator` it converges to as `q -> 0`, which is
    what makes the "this is a lambda rescale" claim checkable rather than rhetorical.
    """

    a: float
    b: float
    kind: str = "platt"

    def apply(self, q: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * _logit(q) + self.b)

    def power_limit(self) -> PowerCalibrator:
        return PowerCalibrator(alpha=float(self.a), log_c=float(self.b))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": float(self.a), "b": float(self.b)}


@dataclass
class IsotonicCalibrator:
    """Monotone step calibration: linear interpolation of a PAVA solution.

    Monotone in `q`, therefore rank-preserving over the eight candidates at a position.
    It can change how sharply the decoder discriminates but never *which* candidate it
    prefers, which is the ceiling on what any q-only calibration can buy.
    """

    x: np.ndarray
    y: np.ndarray
    kind: str = "isotonic"

    def apply(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.interp(q, self.x, self.y, left=float(self.y[0]), right=float(self.y[-1]))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "x": [float(v) for v in self.x],
                "y": [float(v) for v in self.y]}


def calibrator_from_dict(d: dict[str, Any]):
    kind = d["kind"]
    if kind == "identity":
        return IdentityCalibrator()
    if kind == "power":
        return PowerCalibrator(alpha=float(d["alpha"]), log_c=float(d["log_c"]))
    if kind == "platt":
        return PlattCalibrator(a=float(d["a"]), b=float(d["b"]))
    if kind == "isotonic":
        return IsotonicCalibrator(
            x=np.asarray(d["x"], dtype=np.float64), y=np.asarray(d["y"], dtype=np.float64)
        )
    raise ValueError(f"unknown calibrator kind {kind!r}")


# ------------------------------------------------------------------------ fitting


def fit_platt(q: np.ndarray, hit: np.ndarray, max_iter: int = 200, tol: float = 1e-10):
    """Logistic regression of `hit` on `logit(q)`, by Newton-Raphson on two parameters.

    Two parameters and a strictly concave log-likelihood, so Newton converges in a
    handful of steps and there is nothing to tune.  A ridge of 1e-8 keeps the Hessian
    invertible if every prediction happens to be identical.
    """
    z = _logit(q)
    y = np.asarray(hit, dtype=np.float64)
    x = np.stack([z, np.ones_like(z)], axis=1)
    w = np.zeros(2, dtype=np.float64)
    w[0] = 1.0
    for _ in range(max_iter):
        p = _sigmoid(x @ w)
        grad = x.T @ (y - p)
        s = np.clip(p * (1.0 - p), 1e-12, None)
        hess = (x * s[:, None]).T @ x + 1e-8 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w = w + step
        if float(np.abs(step).max()) < tol:
            break
    return PlattCalibrator(a=float(w[0]), b=float(w[1]))


def _pava(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators, the standard isotonic solution."""
    y = np.asarray(y, dtype=np.float64).copy()
    w = np.asarray(weight, dtype=np.float64).copy()
    n = len(y)
    level = y.copy()
    lw = w.copy()
    size = np.ones(n, dtype=np.int64)
    j = 0
    idx = [0]
    for i in range(1, n):
        level[j + 1] = y[i]
        lw[j + 1] = w[i]
        size[j + 1] = 1
        j += 1
        while j > 0 and level[j - 1] > level[j]:
            tot = lw[j - 1] + lw[j]
            level[j - 1] = (lw[j - 1] * level[j - 1] + lw[j] * level[j]) / tot
            lw[j - 1] = tot
            size[j - 1] += size[j]
            j -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for b in range(j + 1):
        out[pos : pos + size[b]] = level[b]
        pos += size[b]
    del idx
    return out


def fit_isotonic(q: np.ndarray, hit: np.ndarray) -> IsotonicCalibrator:
    """Isotonic regression of `hit` on `q`, returned as an interpolation table."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(hit, dtype=np.float64)
    order = np.argsort(q, kind="mergesort")
    qs, ys = q[order], y[order]
    fitted = _pava(ys, np.ones_like(ys))
    # collapse to the distinct x values, keeping the last fitted level per x
    keep = np.ones(len(qs), dtype=bool)
    keep[:-1] = qs[1:] != qs[:-1]
    x = qs[keep]
    yv = np.maximum.accumulate(fitted[keep])
    if len(x) == 1:  # degenerate: one distinct prediction
        x = np.array([x[0], x[0] + 1e-9])
        yv = np.array([yv[0], yv[0]])
    return IsotonicCalibrator(x=x, y=yv)


def fit_power_approximation(
    calibrator, q_grid: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    """Least-squares power law `c * q**alpha` fitted to an arbitrary calibrator.

    The residual is reported in **log space**, because that is the space the decoder
    scores in: `lam * log g(q)` is what enters the softmax, so a deviation of d nats in
    `log g` is a deviation of `lam * d` nats in the score.
    """
    q = np.asarray(q_grid, dtype=np.float64)
    ok = q > 0
    q = q[ok]
    g = np.clip(np.asarray(calibrator.apply(q), dtype=np.float64), _CLIP, None)
    lx, ly = np.log(q), np.log(g)
    w = np.ones_like(lx) if weights is None else np.asarray(weights, dtype=np.float64)[ok]
    W = w.sum()
    mx, my = (w * lx).sum() / W, (w * ly).sum() / W
    var = (w * (lx - mx) ** 2).sum() / W
    alpha = float(((w * (lx - mx) * (ly - my)).sum() / W) / var) if var > 0 else 0.0
    log_c = float(my - alpha * mx)
    resid = ly - (log_c + alpha * lx)
    # only differences across candidates matter, so centre the residual per fit
    return {
        "alpha": alpha,
        "log_c": log_c,
        "residual_rms_nats": float(np.sqrt((w * resid**2).sum() / W)),
        "residual_max_abs_nats": float(np.abs(resid).max()),
        "n_points": int(len(q)),
    }


# ---------------------------------------------------------------------- reporting


def calibration_report(q: np.ndarray, hit: np.ndarray) -> dict[str, Any]:
    """Everything §9.2.1 reported, plus the ratio, on one (prediction, outcome) pair."""
    from . import metrics as M

    q = np.asarray(q, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    mean_pred = float(q.mean())
    observed = float(hit.mean())
    return {
        "n": int(len(q)),
        "mean_predicted": mean_pred,
        "observed": observed,
        "under_confidence_factor": (observed / mean_pred) if mean_pred > 0 else None,
        "ece": M.expected_calibration_error(q, hit),
        "auroc": M.auroc(q, hit),
        "brier": M.brier(q, hit),
        "reliability": M.reliability(q, hit),
    }


def hits_for(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return in_interval(np.asarray(values, dtype=np.float64), lo, hi)


# ------------------------------------------------------------------- decode time


class CalibratedTargetScorer(TargetScorer):
    """`TargetScorer` with an optional bin-logit temperature and an optional map on q.

    Subclassed rather than patched so `guidance.TargetScorer` -- which every executed
    result in the report used -- is left byte-for-byte alone.
    """

    def __init__(self, head, binner, lo, hi, calibrator=None, bin_temperature=None):
        super().__init__(head, binner, lo, hi)
        self.calibrator = calibrator or IdentityCalibrator()
        self.bin_temperature = None if bin_temperature is None else float(bin_temperature)

    def __call__(self, hidden):
        import torch

        self.to(hidden.device)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(hidden.float())
            if self.bin_temperature is not None:
                logits = logits / self.bin_temperature
            probs = torch.softmax(logits, dim=-1)
            q = probs[:, self.mask].sum(dim=-1)
            if isinstance(self.calibrator, IdentityCalibrator):
                return q
            out = self.calibrator.apply(q.detach().cpu().numpy().astype(np.float64))
            return torch.as_tensor(out, dtype=q.dtype, device=q.device)


# ----------------------------------------------------------------- the identity


def equivalent_lambda_is_exact(
    base_logprobs: np.ndarray,
    q: np.ndarray,
    alpha: float,
    log_c: float,
    lam: float,
    eps: float,
) -> dict[str, Any]:
    """Numerical check of "a power calibration is exactly a lambda rescale".

    Compares the candidate sampling distribution under
    `(calibrated head, lam)` against `(raw head, lam*alpha)`.  They are equal in exact
    arithmetic; the only source of disagreement is the `eps` floor inside
    `log(q + eps)`, which is why `eps` is an argument and the maximum absolute
    difference is returned rather than asserted here.
    """
    from . import headroom as H

    cal = PowerCalibrator(alpha=alpha, log_c=log_c)
    w_cal = H.guided_weights(base_logprobs, cal.apply(q), lam, eps)
    w_lam = H.guided_weights(base_logprobs, q, lam * alpha, eps)
    return {
        "lambda": float(lam),
        "alpha": float(alpha),
        "equivalent_lambda": float(lam * alpha),
        "eps": float(eps),
        "max_abs_weight_difference": float(np.abs(w_cal - w_lam).max()),
        "mean_abs_weight_difference": float(np.abs(w_cal - w_lam).mean()),
        "argmax_agreement": float((w_cal.argmax(axis=-1) == w_lam.argmax(axis=-1)).mean()),
    }
