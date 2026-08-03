"""Steering headroom: the phase-2 ceiling measurement (docs/LEXICAL_LOCALITY.md §3).

The load-bearing test here is `test_the_null_removes_the_finite_sample_bias`. Headroom
is `max - min` over k means estimated from K rollouts each, which is biased upward by
sampling noise -- and biased *most* where rollout variance is largest, which is aligned
with the very diffuse/local axis the hypothesis is about. Without the correction the
measurement would manufacture the predicted ordering out of noise.
"""

import numpy as np
import pytest

from property_to_go import headroom as H


def test_candidate_weights_are_the_renormalised_base_policy():
    lp = np.log(np.array([[0.4, 0.3, 0.2, 0.1]]))  # already normalised over the top-4
    w = H.candidate_weights(lp)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_candidate_weights_renormalise_a_truncated_head():
    """Top-k logprobs do not sum to 1; the weights must still be a distribution."""
    lp = np.log(np.array([[0.2, 0.1, 0.05]]))  # the other 0.65 of the vocab is cut off
    w = H.candidate_weights(lp)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx([0.2 / 0.35, 0.1 / 0.35, 0.05 / 0.35])


def test_candidate_weights_survive_very_small_probabilities():
    """The max-subtraction is what stops exp() underflowing to an all-zero row."""
    lp = np.array([[-900.0, -901.0, -910.0]])
    w = H.candidate_weights(lp)
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
    assert w[0, 0] > w[0, 1] > w[0, 2]


def test_lambda_zero_guidance_is_exactly_the_base_policy():
    """lam=0 must reproduce the truncated base policy, or the reference point is wrong."""
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(8), size=5))
    q = rng.uniform(0, 1, (5, 8))
    assert H.guided_weights(lp, q, lam=0.0, eps=1e-6) == pytest.approx(H.candidate_weights(lp))


def test_large_lambda_concentrates_on_the_head_s_favourite():
    lp = np.log(np.array([[0.5, 0.3, 0.2]]))
    q = np.array([[0.01, 0.02, 0.9]])
    w = H.guided_weights(lp, q, lam=8.0, eps=1e-6)
    assert w[0, 2] > 0.99, "at high lambda the property term should dominate log p_base"


def test_guided_weights_match_the_decoder_s_own_combination_rule():
    """Must agree with `guidance.combine_scores` + softmax, or `achieved` is not what
    the decoder achieved."""
    import torch

    from property_to_go.guidance import combine_scores

    rng = np.random.default_rng(1)
    lp = np.log(rng.dirichlet(np.ones(8), size=4))
    q = rng.uniform(0, 1, (4, 8))
    lam, eps = 1.0, 1e-6

    ref = torch.softmax(
        combine_scores(torch.tensor(lp), torch.tensor(q), lam, eps), dim=-1
    ).numpy()
    assert H.guided_weights(lp, q, lam, eps) == pytest.approx(ref, abs=1e-12)


class TestPermutationNull:
    def test_identical_candidates_give_a_null_close_to_the_raw_spread(self):
        """Averaged over draws, because `raw` for one prefix is itself a noisy draw
        while `null` is already an average over permutations."""
        rng = np.random.default_rng(2)
        raws, nulls = [], []
        for _ in range(40):
            per_cand = [rng.normal(3.0, 1.5, 16) for _ in range(8)]
            raws.append(max(v.mean() for v in per_cand) - min(v.mean() for v in per_cand))
            nulls.append(H.permutation_null_spread(per_cand, 150, rng))
        raw, null = float(np.mean(raws)), float(np.mean(nulls))
        assert raw > 0.3, "finite-sample noise alone produces a sizeable raw spread"
        assert abs(raw - null) < 0.15 * raw, "the null must account for nearly all of it"

    def test_genuinely_separated_candidates_exceed_the_null(self):
        rng = np.random.default_rng(3)
        per_cand = [rng.normal(m, 0.4, 16) for m in (0.0, 0.0, 0.0, 5.0)]
        raw = max(v.mean() for v in per_cand) - min(v.mean() for v in per_cand)
        null = H.permutation_null_spread(per_cand, 400, rng)
        assert raw - null > 3.0, "a real 5-unit lever must survive the correction"

    def test_the_null_removes_the_finite_sample_bias(self):
        """The reason the correction exists.

        Two properties with the same (zero) true headroom but very different rollout
        variance. Raw headroom ranks the high-variance one far above the low-variance
        one -- a spurious ordering driven by nothing but noise. Excess headroom must
        rank them together, near zero.
        """
        rng = np.random.default_rng(4)
        raws, excesses = {}, {}
        for name, sd in (("low_variance", 0.2), ("high_variance", 4.0)):
            r, e = [], []
            for _ in range(60):
                per_cand = [rng.normal(0.0, sd, 16) for _ in range(8)]
                raw = max(v.mean() for v in per_cand) - min(v.mean() for v in per_cand)
                r.append(raw)
                e.append(raw - H.permutation_null_spread(per_cand, 120, rng))
            raws[name], excesses[name] = float(np.mean(r)), float(np.mean(e))

        assert raws["high_variance"] > 10 * raws["low_variance"], (
            "raw headroom is dominated by rollout variance, which is the whole problem"
        )
        assert abs(excesses["high_variance"]) < 0.35 * raws["high_variance"]
        assert abs(excesses["low_variance"]) < 0.35 * raws["low_variance"]

    def test_null_is_undefined_with_fewer_than_two_scored_candidates(self):
        rng = np.random.default_rng(5)
        assert H.permutation_null_spread([np.array([1.0, 2.0])], 10, rng) == 0.0
        assert H.permutation_null_spread([np.array([]), np.array([1.0])], 10, rng) == 0.0

    def test_unequal_candidate_sample_sizes_are_respected(self):
        """Group sizes must match the observed per-candidate counts: a candidate scored
        from 4 rollouts is noisier than one scored from 16, and a null that pretended
        they were equal would under-correct."""
        rng = np.random.default_rng(6)
        per_cand = [rng.normal(0.0, 1.0, n) for n in (4, 16, 16, 16)]
        null = H.permutation_null_spread(per_cand, 400, rng)
        equal = H.permutation_null_spread([rng.normal(0.0, 1.0, 16) for _ in range(4)],
                                         400, rng)
        assert null > equal, "the small group should widen the expected spread"


class TestPermutationNullCeiling:
    """The companion correction for the capture denominator.

    `available = max_i p_i - base` has a biased first term and an unbiased second, so
    leaving the bias in inflates the denominator and *understates* how much of the
    headroom guidance captured -- which would systematically favour the "our head is bad"
    reading over "there is no lever", the exact distinction this measurement exists to
    draw. So the bias has to be removed from the same side it enters.
    """

    def test_identical_candidates_give_a_positive_null(self):
        rng = np.random.default_rng(10)
        w = np.full(8, 1 / 8)
        nulls = []
        for _ in range(20):
            per_cand = [rng.binomial(1, 0.2, 16).astype(float) for _ in range(8)]
            nulls.append(H.permutation_null_ceiling(per_cand, w, 150, rng))
        assert np.mean(nulls) > 0.05, (
            "with 16 Bernoulli draws per candidate, `max - base` is substantially "
            "positive even when every candidate is identical"
        )

    def test_the_null_is_smaller_than_a_real_lever(self):
        rng = np.random.default_rng(11)
        w = np.full(4, 0.25)
        per_cand = [rng.binomial(1, p, 16).astype(float) for p in (0.05, 0.05, 0.05, 0.95)]
        p_hat = np.array([v.mean() for v in per_cand])
        available = p_hat.max() - float(w @ p_hat)
        null = H.permutation_null_ceiling(per_cand, w, 300, rng)
        assert available - null > 0.4, "a real 0.9-wide lever must survive the correction"

    def test_the_correction_removes_the_bias_on_a_null_property(self):
        """End to end through `summarise_capture`: a property with no lever at all must
        not appear to have one."""
        rng = np.random.default_rng(12)
        n = 60
        p_hit, base_w, null = [], [], []
        for _ in range(n):
            per_cand = [rng.binomial(1, 0.25, 16).astype(float) for _ in range(6)]
            w = np.full(6, 1 / 6)
            p_hit.append([v.mean() for v in per_cand])
            base_w.append(w)
            null.append(H.permutation_null_ceiling(per_cand, w, 120, rng))
        p_hit, base_w = np.array(p_hit), np.array(base_w)
        out = H.summarise_capture(p_hit, base_w, base_w, np.ones(n, dtype=int),
                                 np.ones(n, dtype=bool), available_null=np.array(null))
        o = out["overall"]
        assert o["available_mean"] > 0.05, "the raw denominator is inflated by noise"
        assert abs(o["available_excess_mean"]) < 0.4 * o["available_mean"], (
            "the corrected denominator must be much closer to its true value of zero"
        )

    def test_it_uses_the_supplied_base_weights(self):
        """A skewed base policy has a different noise floor from a uniform one, because
        `base` tracks the dominant candidate more closely and so is closer to the max."""
        rng = np.random.default_rng(13)
        per_cand = [rng.binomial(1, 0.3, 16).astype(float) for _ in range(4)]
        uniform = H.permutation_null_ceiling(per_cand, np.full(4, 0.25), 400, rng)
        skewed = H.permutation_null_ceiling(per_cand, np.array([0.97, 0.01, 0.01, 0.01]),
                                            400, rng)
        assert skewed != uniform

    def test_undefined_with_fewer_than_two_scored_candidates(self):
        rng = np.random.default_rng(14)
        assert H.permutation_null_ceiling([np.array([1.0, 0.0])], np.array([1.0]),
                                          10, rng) == 0.0


class TestPermutationNullOracleGain:
    """The correction that keeps the "is it lambda or the head?" split honest.

    Substituting an oracle head -- the realised rollout hit rate -- into the lambda = 1
    softmax tests whether lambda itself caps capture. But that oracle is scored on the
    rollouts that define it, so it upweights candidates that got lucky and then banks the
    luck. Uncorrected, it looks far more capable than any real head could be, which biases
    the conclusion against our head twice: it overstates what lambda permits *and*
    overstates the share our head misses.
    """

    def _oracle_gain(self, hits, sizes, w, lam, eps):
        q = np.array([h / s for h, s in zip(hits, sizes)])
        lp = np.log(w / w.sum())[None, :]
        base = float((w / w.sum()) @ q)
        return float(H.guided_weights(lp, q[None, :], lam, eps)[0] @ q) - base

    def test_a_null_property_shows_a_spurious_oracle_gain(self):
        """The bias itself. Every candidate has the same true rate, so the true oracle
        gain is zero -- yet the in-sample oracle 'achieves' a positive one."""
        rng = np.random.default_rng(20)
        w = np.full(8, 1 / 8)
        sizes = np.full(8, 13)
        gains, nulls = [], []
        for _ in range(30):
            hits = rng.binomial(13, 0.2, 8)
            gains.append(self._oracle_gain(hits, sizes, w, 1.0, 1e-6))
            nulls.append(H.permutation_null_oracle_gain(hits, sizes, w, 1.0, 1e-6, 150, rng))
        assert np.mean(gains) > 0.01, "the in-sample oracle banks noise"
        assert abs(np.mean(gains) - np.mean(nulls)) < 0.4 * np.mean(gains), (
            "and the null must account for nearly all of it when there is no real signal"
        )

    def test_a_real_signal_survives_the_correction(self):
        rng = np.random.default_rng(21)
        w = np.full(4, 0.25)
        sizes = np.full(4, 13)
        hits = np.array([1, 1, 1, 12])
        gain = self._oracle_gain(hits, sizes, w, 1.0, 1e-6)
        null = H.permutation_null_oracle_gain(hits, sizes, w, 1.0, 1e-6, 300, rng)
        assert gain - null > 0.1, "a candidate that genuinely hits 12/13 must survive"

    def test_the_null_respects_the_base_policy_concentration(self):
        """A concentrated base policy leaves the oracle less room to move at lambda = 1,
        so its noise floor should be lower."""
        rng = np.random.default_rng(22)
        sizes = np.full(6, 13)
        hits = rng.binomial(13, 0.25, 6)
        flat = H.permutation_null_oracle_gain(hits, sizes, np.full(6, 1 / 6),
                                             1.0, 1e-6, 400, rng)
        peaked = H.permutation_null_oracle_gain(
            hits, sizes, np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01]),
            1.0, 1e-6, 400, rng)
        assert peaked < flat

    def test_higher_lambda_raises_the_noise_floor(self):
        """Because a stronger property term chases the noisy estimate harder."""
        rng = np.random.default_rng(23)
        sizes = np.full(6, 13)
        hits = rng.binomial(13, 0.25, 6)
        w = np.full(6, 1 / 6)
        low = H.permutation_null_oracle_gain(hits, sizes, w, 0.5, 1e-6, 400, rng)
        high = H.permutation_null_oracle_gain(hits, sizes, w, 4.0, 1e-6, 400, rng)
        assert high > low

    def test_degenerate_inputs_return_zero(self):
        rng = np.random.default_rng(24)
        assert H.permutation_null_oracle_gain([0], [0], np.array([1.0]),
                                              1.0, 1e-6, 10, rng) == 0.0
        assert H.permutation_null_oracle_gain([1, 0], [13, 0], np.array([0.5, 0.5]),
                                              1.0, 1e-6, 10, rng) == 0.0

    def test_hit_counts_exceeding_the_pool_are_clamped(self):
        """Hit counts are reconstructed as round(p_hit * n_valid) and could in principle
        exceed the pool by a rounding unit; that must not raise."""
        rng = np.random.default_rng(25)
        out = H.permutation_null_oracle_gain([14, 14], [13, 13], np.full(2, 0.5),
                                             1.0, 1e-6, 20, rng)
        assert np.isfinite(out)


class TestSummariseCapture:
    def _fixture(self, reps=3):
        p_hit = np.tile(np.array([[0.1, 0.2, 0.6, 0.1], [0.0, 0.5, 0.5, 0.0]]), (reps, 1))
        base_w = np.full_like(p_hit, 0.25)
        quartile = np.ones(len(p_hit), dtype=int)
        usable = np.ones(len(p_hit), dtype=bool)
        return p_hit, base_w, quartile, usable

    def test_no_guidance_captures_nothing(self):
        p_hit, base_w, quartile, usable = self._fixture()
        out = H.summarise_capture(p_hit, base_w, base_w, quartile, usable)
        assert out["overall"]["achieved_mean"] == pytest.approx(0.0)
        assert out["overall"]["captured_fraction"] == pytest.approx(0.0)

    def test_a_perfect_oracle_captures_all_of_it(self):
        p_hit, base_w, quartile, usable = self._fixture()
        oracle = np.zeros_like(p_hit)
        oracle[np.arange(len(p_hit)), p_hit.argmax(axis=1)] = 1.0
        out = H.summarise_capture(p_hit, base_w, oracle, quartile, usable)
        assert out["overall"]["captured_fraction"] == pytest.approx(1.0)
        assert out["overall"]["best_candidate_target_prob"] == pytest.approx(0.55)

    def test_an_anti_oracle_captures_a_negative_fraction(self):
        """Guidance can be actively worse than the base policy; that must be reportable
        rather than clipped to zero."""
        p_hit, base_w, quartile, usable = self._fixture()
        worst = np.zeros_like(p_hit)
        worst[np.arange(len(p_hit)), p_hit.argmin(axis=1)] = 1.0
        out = H.summarise_capture(p_hit, base_w, worst, quartile, usable)
        assert out["overall"]["captured_fraction"] < 0

    def test_capture_aggregates_totals_not_per_prefix_ratios(self):
        """A prefix with no available headroom must not contribute a garbage ratio.

        Half the prefixes have a real lever; the other half have none (all candidates
        equal), so their per-prefix ratio is 0/0. Averaging ratios would be undefined
        or would dilute the answer; a total-over-total ratio is well defined and equals
        the levered prefixes' own.
        """
        p_hit = np.tile(np.array([[0.0, 0.8], [0.3, 0.3]]), (3, 1))
        base_w = np.full_like(p_hit, 0.5)
        oracle = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (3, 1))
        out = H.summarise_capture(p_hit, base_w, oracle, np.ones(6, dtype=int),
                                  np.ones(6, dtype=bool))
        assert out["overall"]["captured_fraction"] == pytest.approx(1.0)

    def test_a_cell_with_too_few_prefixes_reports_none(self):
        p_hit, base_w, quartile, usable = self._fixture(reps=1)
        out = H.summarise_capture(p_hit, base_w, base_w, quartile, usable)
        assert out["overall"] is None, "must not report a 2-prefix capture fraction"

    def test_zero_available_headroom_reports_none_rather_than_dividing(self):
        p_hit = np.full((6, 3), 0.25)
        base_w = np.full((6, 3), 1 / 3)
        out = H.summarise_capture(p_hit, base_w, base_w, np.ones(6, dtype=int),
                                  np.ones(6, dtype=bool))
        assert out["overall"]["captured_fraction"] is None


class TestSummariseHeadroom:
    def test_relative_headroom_is_normalised_by_interval_width(self):
        mu = np.full(40, 2.0)
        null = np.full(40, 0.5)
        q = np.tile([1, 2, 3, 4], 10)
        out = H.summarise_headroom(mu, null, interval_width=1.5, quartile=q)
        assert out["overall"]["relative_headroom_raw_mean"] == pytest.approx(2.0 / 1.5)
        assert out["overall"]["relative_headroom_excess_mean"] == pytest.approx(1.5 / 1.5)
        assert out["overall"]["frac_prefixes_excess_above_one_interval"] == pytest.approx(1.0)

    def test_undefined_prefixes_are_excluded_and_counted(self):
        mu = np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0])
        null = np.zeros(6)
        out = H.summarise_headroom(mu, null, 1.0, np.ones(6, dtype=int))
        assert out["n_prefixes_undefined"] == 1
        assert out["overall"]["n_prefixes"] == 5

    def test_a_quartile_with_too_few_prefixes_reports_none(self):
        mu = np.ones(8)
        out = H.summarise_headroom(mu, np.zeros(8), 1.0, np.array([1] * 6 + [4] * 2))
        assert out["by_quartile"]["1"] is not None
        assert out["by_quartile"]["4"] is None, "must not report a 2-prefix mean"
