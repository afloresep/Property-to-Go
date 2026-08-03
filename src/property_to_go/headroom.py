"""Steering headroom: the ceiling on what any decoding rule could do at a position.

This is the phase-2 measurement the pilot could not make.  The pilot's negative result
("guided decoding loses to compute-matched best-of-N") is ambiguous between two
explanations it had no way to separate:

  **no lever** -- at this position no available token choice moves the final property
      much, so guidance cannot work here and no amount of lambda tuning or head
      recalibration will change that;
  **bad head** -- the lever exists and our head failed to pull it, in which case the
      negative result is about our head rather than about the method.

Headroom decides between them, because it is defined without reference to any head and
without reference to lambda.  At a prefix x_{<=t} with the base model's top-k candidate
tokens a_1..a_k, estimate by base-policy rollouts

    mu(a_i)   = E[y_final | x_{<=t}, a_i]
    p(a_i)    = P(y_final in I | x_{<=t}, a_i)

and take the spread across candidates.  Since any decoding rule at this position can do
no more than choose among those candidates, the spread is an **upper bound** on the
one-step effect of every such rule, this project's included.  See
`docs/LEXICAL_LOCALITY.md` §3.

Two units, both reported, because they answer different questions:

*property units*  `headroom / (hi - lo)` -- "does one token choice move the property by
    a target-interval width?"  This is the locality score the hypothesis is stated in.
*probability units*  spread of `p(a_i)` -- directly comparable to a hit rate, which is
    what makes "what fraction of the available headroom did guidance capture?"
    answerable at all.

## The estimator correction that matters

`mu(a_i)` is a mean over K rollouts, so it carries sampling noise, and `max - min` over
k noisy means is **biased upward**: even k identical candidates show a positive spread.
Reporting raw headroom would therefore overstate every property's lever, and overstate
it most where rollout variance is highest -- which is a confound aligned with exactly
the diffuse/local axis under test.

So a permutation null is computed alongside: pool a prefix's k*K rollouts, repartition
them at random into k groups of K, and recompute the spread.  That is the spread
expected if every candidate had the same true mean.  `excess = raw - null` is the
noise-corrected quantity.

Both are reported.  This correction is an estimator detail fixed before any headroom
number was computed; it does not touch predictions P1-P6.
"""

from __future__ import annotations

import numpy as np


def candidate_weights(base_logprobs: np.ndarray) -> np.ndarray:
    """Base-policy probabilities over the top-k candidates, renormalised to sum to 1.

    This is the weighting a `truncation_control` step uses: the base policy restricted
    to the same candidate set guidance sees.  It is the right reference point for
    "what did guidance add", because comparing against the *unrestricted* base policy
    would fold in the effect of the top-k truncation itself -- the confound the pilot
    added `truncation_control` to remove (`pilot_report.md` §7.9).
    """
    lp = np.asarray(base_logprobs, dtype=np.float64)
    lp = lp - lp.max(axis=-1, keepdims=True)
    w = np.exp(lp)
    return w / w.sum(axis=-1, keepdims=True)


def guided_weights(
    base_logprobs: np.ndarray, target_probs: np.ndarray, lam: float, eps: float
) -> np.ndarray:
    """The sampling distribution `guided_sample` induces over the top-k candidates.

    Mirrors `guidance.combine_scores` followed by a softmax over the candidates, which
    is exactly what the decoder does, so `guided_weights @ p` is the per-step effect the
    deployed rule actually achieves and is comparable to the ceiling.
    """
    score = np.asarray(base_logprobs, dtype=np.float64) + lam * np.log(
        np.clip(np.asarray(target_probs, dtype=np.float64), 0.0, None) + eps
    )
    score = score - score.max(axis=-1, keepdims=True)
    w = np.exp(score)
    return w / w.sum(axis=-1, keepdims=True)


def _group_spread(values: np.ndarray, group_sizes: np.ndarray) -> float:
    ends = np.cumsum(group_sizes)
    starts = ends - group_sizes
    means = np.array([values[s:e].mean() for s, e in zip(starts, ends)])
    return float(means.max() - means.min())


def permutation_null_spread(
    per_candidate: list[np.ndarray], n_perm: int, rng: np.random.Generator
) -> float:
    """Expected `max - min` of k group means if all k candidates were identical.

    `per_candidate[i]` holds candidate i's usable rollout values, so the group sizes
    reflect the real per-candidate sample sizes (they differ: a candidate whose
    continuations are more often unparseable contributes fewer).  Pooling and
    repartitioning at the *observed* sizes keeps the null matched to the statistic it
    is correcting.

    Returns 0.0 when the null is undefined (fewer than two non-empty candidates).
    """
    sizes = np.array([len(v) for v in per_candidate], dtype=np.int64)
    if (sizes > 0).sum() < 2:
        return 0.0
    pooled = np.concatenate([v for v in per_candidate if len(v)])
    sizes = sizes[sizes > 0]
    spreads = np.empty(n_perm, dtype=np.float64)
    for r in range(n_perm):
        spreads[r] = _group_spread(rng.permutation(pooled), sizes)
    return float(spreads.mean())


def summarise_headroom(
    mu: np.ndarray,
    null: np.ndarray,
    interval_width: float,
    quartile: np.ndarray,
    min_prefixes: int = 5,
) -> dict:
    """Aggregate per-prefix headroom, overall and by prefix-position quartile.

    `mu` and `null` are per-prefix raw and null spreads in property units.  Prefixes
    where headroom is undefined (fewer than two candidates produced usable rollouts)
    arrive as NaN and are excluded, with the count reported.

    A cell holding fewer than `min_prefixes` prefixes reports None rather than a mean
    nobody should quote.
    """
    mu = np.asarray(mu, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    keep = np.isfinite(mu) & np.isfinite(null)
    excess = mu - null

    def block(mask) -> dict | None:
        if mask.sum() < min_prefixes:
            return None
        return {
            "n_prefixes": int(mask.sum()),
            "headroom_raw_mean": float(mu[mask].mean()),
            "headroom_raw_median": float(np.median(mu[mask])),
            "headroom_null_mean": float(null[mask].mean()),
            "headroom_excess_mean": float(excess[mask].mean()),
            "headroom_excess_median": float(np.median(excess[mask])),
            "relative_headroom_raw_mean": float(mu[mask].mean() / interval_width),
            "relative_headroom_excess_mean": float(excess[mask].mean() / interval_width),
            "relative_headroom_excess_median": float(
                np.median(excess[mask]) / interval_width
            ),
            "frac_prefixes_excess_above_one_interval": float(
                (excess[mask] >= interval_width).mean()
            ),
        }

    return {
        "interval_width": float(interval_width),
        "n_prefixes_undefined": int((~keep).sum()),
        "overall": block(keep),
        "by_quartile": {
            str(q): block(keep & (np.asarray(quartile) == q)) for q in (1, 2, 3, 4)
        },
    }


def permutation_null_ceiling(
    per_candidate: list[np.ndarray],
    base_w: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Expected `max_i p_i - sum_i w_i p_i` if all k candidates were identical.

    The companion to `permutation_null_spread`, for the capture analysis.  `available`
    is `ceiling - base` and `ceiling` is a **max over noisy estimates**, so it is biased
    upward exactly as the spread is.  `base` is a fixed-weight average and is unbiased.
    Leaving the bias in would inflate the denominator of `achieved / available` and
    understate how much of the headroom guidance captured -- i.e. it would flatter the
    "our head is bad" reading over the "there is no lever" one, which is precisely the
    distinction this measurement exists to make.

    `per_candidate[i]` holds candidate i's usable rollout indicator values (1 for a hit).
    """
    sizes = np.array([len(v) for v in per_candidate], dtype=np.int64)
    if (sizes > 0).sum() < 2:
        return 0.0
    keep = sizes > 0
    pooled = np.concatenate([v for v in per_candidate if len(v)])
    sizes = sizes[keep]
    w = np.asarray(base_w, dtype=np.float64)[keep]
    w = w / w.sum()
    ends = np.cumsum(sizes)
    starts = ends - sizes
    out = np.empty(n_perm, dtype=np.float64)
    for r in range(n_perm):
        v = rng.permutation(pooled)
        means = np.array([v[s:e].mean() for s, e in zip(starts, ends)])
        out[r] = means.max() - float(w @ means)
    return float(out.mean())


def permutation_null_oracle_gain(
    hits: np.ndarray,
    sizes: np.ndarray,
    base_w: np.ndarray,
    lam: float,
    eps: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Gain an *in-sample* oracle head would show if all candidates were identical.

    The third explanation for low capture -- that lambda = 1 itself caps how far the rule
    can deviate from `log p_base` -- is tested by substituting an oracle head, the
    realised rollout hit rate, into the same lambda = 1 softmax the decoder uses.  But
    that oracle is scored on the very rollouts that define it, so it upweights candidates
    whose estimate is high *by chance* and then collects the chance.  With ~13 usable
    rollouts per candidate that bias is not small.

    This is the matching null: pool the prefix's hits, redistribute them at random into
    groups of the observed per-candidate sizes, and recompute the oracle's gain.  Whatever
    it "achieves" there is pure self-exploitation, and subtracting it leaves the part
    attributable to real between-candidate differences.

    Without this correction the oracle looks far more capable than it is, which would
    understate how much lambda = 1 costs and overstate how much of the remainder our own
    head fails to get -- i.e. it would bias the conclusion against our head twice over.

    `hits[i]` and `sizes[i]` are candidate i's hit count and usable-rollout count.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    total = int(sizes.sum())
    if total == 0 or (sizes > 0).sum() < 2:
        return 0.0
    keep = sizes > 0
    sizes = sizes[keep]
    w = np.asarray(base_w, dtype=np.float64)[keep]
    lp = np.log(np.clip(w / w.sum(), 1e-300, None))[None, :]
    pool = np.zeros(int(sizes.sum()))
    n_hit = min(int(np.asarray(hits)[keep].sum()), len(pool))
    pool[:n_hit] = 1.0
    ends = np.cumsum(sizes)
    starts = ends - sizes
    acc = 0.0
    for _ in range(n_perm):
        v = rng.permutation(pool)
        q = np.array([v[s:e].mean() for s, e in zip(starts, ends)])
        base = float(np.asarray(w / w.sum()) @ q)
        acc += float(guided_weights(lp, q[None, :], lam, eps)[0] @ q) - base
    return acc / n_perm


def summarise_capture(
    p_hit: np.ndarray,
    base_w: np.ndarray,
    guided_w: np.ndarray,
    quartile: np.ndarray,
    usable: np.ndarray,
    min_prefixes: int = 5,
    available_null: np.ndarray | None = None,
) -> dict:
    """What share of the available one-step headroom the deployed rule captured.

    All three quantities are in hit-rate units at a single position:

        base      = sum_i base_w_i   * p_i    the top-k-restricted base policy
        guided    = sum_i guided_w_i * p_i    the rule the pilot actually ran
        ceiling   = max_i p_i                 the best any rule could do here
        available = ceiling - base
        achieved  = guided  - base

    `captured = achieved / available` is the number that separates "no lever" from "bad
    head": near 1 means the head is extracting essentially all of the position's
    available signal and the loss to best-of-N is structural; near 0 with a large
    `available` means the lever was there and the head missed it.

    Aggregated as `sum(achieved) / sum(available)` rather than as a mean of per-prefix
    ratios, because `available` is near zero at many prefixes and per-prefix ratios
    there are numerically meaningless while contributing equal weight.
    """
    p_hit = np.asarray(p_hit, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    base = (np.asarray(base_w) * p_hit).sum(axis=1)
    guided = (np.asarray(guided_w) * p_hit).sum(axis=1)
    ceiling = p_hit.max(axis=1)
    floor = p_hit.min(axis=1)

    null = (np.zeros(len(p_hit)) if available_null is None
            else np.nan_to_num(np.asarray(available_null, dtype=np.float64)))

    def block(mask) -> dict | None:
        if mask.sum() < min_prefixes:
            return None
        available = ceiling[mask] - base[mask]
        achieved = guided[mask] - base[mask]
        excess = available - null[mask]
        tot_avail = float(available.sum())
        tot_excess = float(excess.sum())
        return {
            "n_prefixes": int(mask.sum()),
            "base_policy_target_prob": float(base[mask].mean()),
            "guided_target_prob": float(guided[mask].mean()),
            "best_candidate_target_prob": float(ceiling[mask].mean()),
            "worst_candidate_target_prob": float(floor[mask].mean()),
            "headroom_prob_spread": float((ceiling[mask] - floor[mask]).mean()),
            "available_mean": float(available.mean()),
            "available_null_mean": float(null[mask].mean()),
            # `available` is a max over noisy per-candidate estimates and is therefore
            # biased upward; `achieved` uses head-derived weights and is not. The
            # noise-corrected denominator is the one to quote.
            "available_excess_mean": float(excess.mean()),
            "achieved_mean": float(achieved.mean()),
            "captured_fraction_raw": (
                float(achieved.sum() / tot_avail) if tot_avail > 0 else None
            ),
            "captured_fraction": (
                float(achieved.sum() / tot_excess) if tot_excess > 0 else None
            ),
        }

    return {
        "overall": block(usable),
        "by_quartile": {
            str(q): block(usable & (np.asarray(quartile) == q)) for q in (1, 2, 3, 4)
        },
    }
