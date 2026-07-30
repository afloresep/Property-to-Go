import numpy as np
import pytest

from property_to_go.prefixes import (
    balanced_position_sample,
    quartile_bounds,
    relative_position,
    select_quartile_prefixes,
)


@pytest.mark.parametrize("n", list(range(4, 60)))
def test_quartiles_tile_the_sequence_exactly(n):
    bounds = quartile_bounds(n)
    assert len(bounds) == 4
    assert bounds[0][0] == 1
    assert bounds[-1][1] == n
    for (lo, hi), (nlo, _) in zip(bounds, bounds[1:]):
        assert hi >= lo, "no quartile may be empty"
        assert nlo == hi + 1, "quartiles must be contiguous and non-overlapping"
    assert sum(hi - lo + 1 for lo, hi in bounds) == n


@pytest.mark.parametrize("n", [8, 13, 40, 41, 202])
def test_selected_prefixes_lie_in_their_quartile(n):
    rng = np.random.default_rng(0)
    for _ in range(50):
        picks = select_quartile_prefixes(n, rng)
        assert [q for q, _ in picks] == [1, 2, 3, 4]
        for (q, k), (lo, hi) in zip(picks, quartile_bounds(n)):
            assert lo <= k <= hi
            assert 1 <= k <= n


def test_selection_is_seed_reproducible():
    a = select_quartile_prefixes(37, np.random.default_rng(5))
    b = select_quartile_prefixes(37, np.random.default_rng(5))
    assert a == b


def test_selection_actually_varies():
    rng = np.random.default_rng(1)
    seen = {tuple(k for _, k in select_quartile_prefixes(40, rng)) for _ in range(50)}
    assert len(seen) > 1, "prefix positions must be sampled, not fixed"


def test_too_short_raises():
    with pytest.raises(ValueError):
        select_quartile_prefixes(3, np.random.default_rng(0))


def test_relative_position():
    assert relative_position(10, 40) == 0.25
    assert relative_position(40, 40) == 1.0
    assert relative_position(0, 0) == 0.0


def test_balanced_position_sample_is_quartile_balanced():
    quartiles = np.array([1] * 100 + [2] * 100 + [3] * 100 + [4] * 100)
    lens = np.arange(400)
    idx = balanced_position_sample(lens, quartiles, 40, np.random.default_rng(0))
    counts = np.bincount(quartiles[idx], minlength=5)[1:]
    assert list(counts) == [10, 10, 10, 10]
    assert len(set(idx.tolist())) == len(idx), "no prefix sampled twice"
