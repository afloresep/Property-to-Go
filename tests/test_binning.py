import numpy as np
import pytest

from property_to_go.binning import (
    CategoricalBinner,
    QuantileBinner,
    binner_from_dict,
    in_interval,
    interval_mask_coverage,
    interval_probability,
    resolve_target_interval,
)


@pytest.fixture
def values():
    return np.random.default_rng(0).normal(2.7, 1.4, 20000)


def test_quantile_bins_are_equal_mass(values):
    b = QuantileBinner.fit(values, 20)
    counts = np.bincount(b.transform(values), minlength=20)
    assert b.n_bins == 20
    # each bin should hold ~5% of the mass
    assert counts.min() > 0.04 * len(values)
    assert counts.max() < 0.06 * len(values)


def test_outer_edges_are_infinite_so_nothing_falls_outside(values):
    b = QuantileBinner.fit(values, 10)
    assert b.edges[0] == -np.inf and b.edges[-1] == np.inf
    extreme = np.array([-1e9, 1e9])
    assert list(b.transform(extreme)) == [0, 9]


def test_interval_mask_is_exact_on_bin_edges(values):
    b = QuantileBinner.fit(values, 20)
    lo, hi = float(np.quantile(values, 0.85)), float(np.quantile(values, 0.95))
    mask = b.interval_mask(lo, hi)
    assert mask.sum() == 2, "a [0.85, 0.95] band spans exactly two 5% bins"
    # the summed bin probability must equal the empirical rate the bins encode
    onehot = np.eye(20)[b.transform(values)]
    predicted = interval_probability(onehot, b, lo, hi)
    truth = in_interval(values, lo, hi)
    assert predicted.mean() == pytest.approx(truth.mean(), abs=1e-9)


def test_interval_probability_sums_bins():
    b = CategoricalBinner(max_value=5)
    probs = np.array([[0.1, 0.2, 0.3, 0.25, 0.1, 0.05]])
    assert interval_probability(probs, b, 3, 4)[0] == pytest.approx(0.25)
    assert interval_probability(probs, b, 0, 6)[0] == pytest.approx(1.0)
    assert interval_probability(probs, b, 2, 5)[0] == pytest.approx(0.65)


def test_categorical_clips_the_tail():
    b = CategoricalBinner(max_value=5)
    assert list(b.transform(np.array([0, 3, 5, 7, 12]))) == [0, 3, 5, 5, 5]


def test_expected_value_matches_hand_computation():
    b = CategoricalBinner(max_value=3)
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    assert b.expected_value(probs)[0] == pytest.approx(1.5)


def test_expected_value_of_a_confident_head_is_near_truth(values):
    b = QuantileBinner.fit(values, 40)
    onehot = np.eye(40)[b.transform(values)]
    assert np.abs(b.expected_value(onehot) - values).mean() < 0.06


def test_roundtrip_serialisation(values):
    for b in (QuantileBinner.fit(values, 12), CategoricalBinner(max_value=5)):
        r = binner_from_dict(b.to_dict())
        assert np.allclose(r.transform(values[:100]), b.transform(values[:100]))
        assert r.n_bins == b.n_bins


class TestTargetIntervalMustBeAUnionOfBins:
    """Regression tests for the defect in `pilot_report.md` §11.5.

    `interval_mask` keeps only bins lying *wholly* inside [lo, hi). So if a target edge
    falls mid-bin, that bin is dropped and `P(y in I)` silently becomes the probability
    of a strict subset of the target. Nothing raises. It presents as a miscalibrated
    head, which is how the pilot recorded it.

    The trigger is structural rather than exotic: the interval is a quantile of the full
    sample and the binner is fitted on the train split, so the two never quite agree.
    """

    # The pilot's actual offsets: its train-split quantiles sat slightly ABOVE the
    # full-sample quantiles the interval was cut from, so both interval edges fell just
    # below a bin edge. `both_below` reproduces that exactly. `straddling` is the case
    # where the two edges fall on opposite sides, which is worse still.
    OFFSETS = {
        "both_below (the pilot's case)": (-1.324e-3, -1.980e-3, 1),
        "straddling": (+1.3e-3, -2.0e-3, 0),
    }

    @pytest.mark.parametrize("name", list(OFFSETS))
    def test_a_slightly_misaligned_interval_silently_shrinks_the_target(self, values, name):
        """The defect itself, reproduced. This test documents the failure mode."""
        d_lo, d_hi, expect_bins = self.OFFSETS[name]
        b = QuantileBinner.fit(values, 20)
        lo = float(np.quantile(values, 0.85)) + d_lo
        hi = float(np.quantile(values, 0.95)) + d_hi
        cov = interval_mask_coverage(b, lo, hi, values)
        assert cov["n_bins_selected"] == expect_bins, name
        assert cov["true_rate"] == pytest.approx(0.10, abs=0.01)
        assert cov["masked_rate"] < cov["true_rate"] - 0.04, (
            "the head would predict a strict subset of the target"
        )
        assert not cov["is_exact"], "and nothing about it raises"

    @pytest.mark.parametrize("name", list(OFFSETS))
    def test_extra_edges_make_the_interval_exact(self, values, name):
        """The fix. Same misaligned intervals, now exactly representable."""
        d_lo, d_hi, _ = self.OFFSETS[name]
        lo = float(np.quantile(values, 0.85)) + d_lo
        hi = float(np.quantile(values, 0.95)) + d_hi
        b = QuantileBinner.fit(values, 20, extra_edges=(lo, hi))
        cov = interval_mask_coverage(b, lo, hi, values)
        assert cov["is_exact"], (name, cov)
        assert cov["masked_rate"] == pytest.approx(cov["true_rate"], abs=1e-9)

    def test_extra_edges_add_bins_rather_than_replacing_them(self, values):
        b0 = QuantileBinner.fit(values, 20)
        lo = float(np.quantile(values, 0.85)) - 1.324e-3
        hi = float(np.quantile(values, 0.95)) - 1.980e-3
        b1 = QuantileBinner.fit(values, 20, extra_edges=(lo, hi))
        assert b1.n_bins == b0.n_bins + 2
        assert len(b1.centers) == b1.n_bins, "centers must cover every bin"
        # the original quantile edges must all survive
        for e in b0.edges[1:-1]:
            assert np.abs(b1.edges - e).min() < 1e-12

    def test_an_edge_already_present_is_not_duplicated(self, values):
        b0 = QuantileBinner.fit(values, 20)
        exact = float(b0.edges[17])
        b1 = QuantileBinner.fit(values, 20, extra_edges=(exact,))
        assert b1.n_bins == b0.n_bins, "no sliver bin of zero width"

    def test_exactness_holds_across_many_random_offsets(self, values):
        """The fix must not depend on the particular offset that exposed the bug."""
        rng = np.random.default_rng(7)
        for _ in range(25):
            a, c = sorted(rng.uniform(0.05, 0.95, 2))
            if c - a < 0.05:
                continue
            lo = float(np.quantile(values, a)) + float(rng.normal(0, 0.01))
            hi = float(np.quantile(values, c)) + float(rng.normal(0, 0.01))
            if hi <= lo:
                continue
            b = QuantileBinner.fit(values, 20, extra_edges=(lo, hi))
            assert interval_mask_coverage(b, lo, hi, values)["is_exact"]

    def test_a_categorical_binner_is_exact_for_integer_targets_without_help(self):
        """Why aromatic rings escaped the defect and cLogP did not."""
        rng = np.random.default_rng(8)
        counts = rng.poisson(1.8, 20000)
        b = CategoricalBinner(max_value=5)
        assert interval_mask_coverage(b, 3.0, 4.0, counts)["is_exact"]

    def test_expected_value_is_essentially_unchanged_by_the_extra_edges(self, values):
        """The fix must not move the head's other output. A sliver bin barely shifts the
        bin centres, so E[y] should be unaffected to well within its own error."""
        lo = float(np.quantile(values, 0.85)) - 1.324e-3
        hi = float(np.quantile(values, 0.95)) - 1.980e-3
        b0 = QuantileBinner.fit(values, 20)
        b1 = QuantileBinner.fit(values, 20, extra_edges=(lo, hi))
        e0 = b0.expected_value(np.eye(b0.n_bins)[b0.transform(values)])
        e1 = b1.expected_value(np.eye(b1.n_bins)[b1.transform(values)])
        assert np.abs(e0 - e1).mean() < 0.02


class TestResolveTargetInterval:
    """`resolve_target_interval` is the pre-registration mechanism.

    The point of expressing the target as a *rule* is that the rule can be committed
    before anyone has seen a new property's base distribution, so the interval is a
    consequence rather than a choice. See configs/guidance.yaml.
    """

    def test_quantile_band_has_the_base_rate_it_claims(self, values):
        out = resolve_target_interval({"kind": "quantile_band", "lo": 0.85, "hi": 0.95}, values)
        assert out["base_rate"] == pytest.approx(0.10, abs=0.005)
        assert out["lo"] == pytest.approx(float(np.quantile(values, 0.85)))
        assert out["hi"] == pytest.approx(float(np.quantile(values, 0.95)))

    def test_every_quantile_band_property_is_base_rate_matched(self):
        """Why the continuous properties are comparable to each other at all."""
        rng = np.random.default_rng(1)
        rule = {"kind": "quantile_band", "lo": 0.85, "hi": 0.95}
        for dist in (rng.normal(0, 1, 50000), rng.exponential(3, 50000),
                     rng.beta(2, 5, 50000), rng.uniform(0, 140, 50000)):
            assert resolve_target_interval(rule, dist)["base_rate"] == pytest.approx(0.10, abs=0.01)

    def test_exact_value_reproduces_the_frozen_aromatic_ring_target(self):
        rings = np.repeat([0, 1, 2, 3, 4], [4001, 15812, 19065, 8539, 2406])
        out = resolve_target_interval({"kind": "exact_value", "value": 3}, rings)
        assert (out["lo"], out["hi"]) == (3.0, 4.0)
        assert out["base_rate"] == pytest.approx(8539 / len(rings))

    def test_quantile_value_agrees_with_the_pilots_hand_picked_ring_target(self):
        """The uniform count rule must not contradict what the pilot froze.

        Counts from `outputs/pilot_50k/dataset_summary.json`: the base cumulative
        distribution reaches 0.780 at 2 rings and 0.952 at 3, so q=0.90 lands on 3 --
        the value the pilot chose by hand.
        """
        rings = np.repeat([0, 1, 2, 3, 4], [4001, 15812, 19065, 8539, 2406])
        out = resolve_target_interval({"kind": "quantile_value", "q": 0.90}, rings)
        frozen = resolve_target_interval({"kind": "exact_value", "value": 3}, rings)
        assert (out["lo"], out["hi"]) == (3.0, 4.0)
        # The `rule` field records which rule produced it and so differs by design;
        # the interval and its base rate are what must agree.
        for key in ("lo", "hi", "base_rate"):
            assert out[key] == frozen[key]

    def test_quantile_value_is_always_one_count_unit_wide(self):
        rng = np.random.default_rng(2)
        for lam in (0.5, 2.0, 5.0, 9.0):
            out = resolve_target_interval(
                {"kind": "quantile_value", "q": 0.90}, rng.poisson(lam, 50000)
            )
            assert out["hi"] - out["lo"] == 1.0
            assert float(out["lo"]).is_integer()

    def test_keys_match_the_frozen_pilot_artefact_exactly(self, values):
        """Three phases of results are bound to target_intervals.json; adding keys to
        it gratuitously would change a tracked artefact for no reason."""
        out = resolve_target_interval({"kind": "quantile_band", "lo": 0.85, "hi": 0.95}, values)
        assert list(out) == ["lo", "hi", "rule", "base_rate"]

    def test_unknown_rule_kind_is_refused_rather_than_guessed(self, values):
        with pytest.raises(ValueError, match="unknown target_interval_rule"):
            resolve_target_interval({"kind": "vibes", "lo": 0.1}, values)
