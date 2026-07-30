import numpy as np
import pytest

from property_to_go.binning import (
    CategoricalBinner,
    QuantileBinner,
    binner_from_dict,
    in_interval,
    interval_probability,
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
