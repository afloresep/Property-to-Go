import numpy as np
import pytest
import torch

from property_to_go.bestofn import target_distance
from property_to_go.compute import ComputeMeter, solve_best_of_n
from property_to_go.guidance import Windows, combine_scores


# ---------------------------------------------------------------- score combination
def test_lambda_zero_leaves_the_base_policy_untouched():
    base = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.5, -1.0]]), dim=-1)
    q = torch.tensor([[0.9, 0.01, 0.5, 0.2]])
    scored = combine_scores(base, q, lam=0.0, eps=1e-6)
    assert torch.allclose(torch.softmax(scored, -1), torch.softmax(base, -1), atol=1e-7)


def test_guidance_reweights_towards_high_target_probability():
    base = torch.log_softmax(torch.tensor([[1.0, 1.0]]), dim=-1)  # tied base policy
    q = torch.tensor([[0.9, 0.1]])
    p = torch.softmax(combine_scores(base, q, lam=1.0, eps=1e-6), -1)
    assert p[0, 0] > p[0, 1]
    # with tied base logprobs the reweighting is exactly proportional to q
    assert float(p[0, 0] / p[0, 1]) == pytest.approx(9.0, rel=1e-4)


def test_larger_lambda_sharpens_towards_the_target():
    base = torch.log_softmax(torch.tensor([[1.0, 1.0]]), dim=-1)
    q = torch.tensor([[0.8, 0.2]])
    ratios = [
        float(torch.softmax(combine_scores(base, q, lam=l, eps=1e-6), -1)[0, 0])
        for l in (0.0, 0.5, 1.0, 2.0)
    ]
    assert ratios == sorted(ratios)


def test_epsilon_keeps_zero_probability_candidates_finite():
    base = torch.log_softmax(torch.tensor([[1.0, 1.0]]), dim=-1)
    q = torch.tensor([[0.0, 1.0]])
    s = combine_scores(base, q, lam=1.0, eps=1e-6)
    assert torch.isfinite(s).all()
    p = torch.softmax(s, -1)
    assert p[0, 0] < 1e-5 and p[0, 1] > 0.999


def test_a_dominant_base_token_can_still_win_against_a_weak_target():
    base = torch.log_softmax(torch.tensor([[10.0, 0.0]]), dim=-1)
    q = torch.tensor([[0.3, 0.6]])  # only a 2x target advantage
    p = torch.softmax(combine_scores(base, q, lam=1.0, eps=1e-6), -1)
    assert p[0, 0] > p[0, 1], "lambda=1 must not override a 10-nat base preference"


# ------------------------------------------------------------------- compute meter
def test_meter_separates_actual_from_full_recompute_cost():
    m = ComputeMeter()
    m.add_forward(8, 8 * 21)  # eight candidates, cached vs 21-token full prefixes
    m.molecules_returned = 1
    d = m.as_dict()
    assert d["processed_tokens_actual"] == 8
    assert d["processed_tokens_full_recompute"] == 168
    assert d["tokens_per_molecule_actual"] == 8
    assert d["forward_calls"] == 1


def test_default_full_recompute_equals_actual():
    m = ComputeMeter()
    m.add_forward(42)
    assert m.processed_tokens_full_recompute == 42


def test_meters_merge_additively():
    a, b = ComputeMeter(), ComputeMeter()
    a.add_forward(10, 100)
    a.molecules_returned = 2
    b.add_forward(5, 50)
    b.molecules_returned = 3
    a.merge(b)
    assert (a.processed_tokens_actual, a.processed_tokens_full_recompute) == (15, 150)
    assert a.molecules_returned == 5 and a.forward_calls == 2


def test_wall_clock_accumulates_across_start_stop():
    m = ComputeMeter()
    with m:
        pass
    first = m.wall_seconds
    with m:
        pass
    assert m.wall_seconds >= first >= 0.0


# ------------------------------------------------------------- compute matching
def test_best_of_n_never_exceeds_the_guided_budget():
    n = solve_best_of_n(target_tokens_per_molecule=400.0, base_tokens_per_molecule=42.0)
    assert n == 9
    assert n * 42.0 <= 400.0
    assert (n + 1) * 42.0 > 400.0


def test_best_of_n_is_at_least_one():
    assert solve_best_of_n(10.0, 42.0) == 1


def test_best_of_n_rejects_zero_base_cost():
    with pytest.raises(ValueError):
        solve_best_of_n(100.0, 0.0)


# ------------------------------------------------------------------ target distance
def test_target_distance_is_zero_inside_and_grows_outside():
    assert target_distance(3.0, 2.0, 4.0) == 0.0
    assert target_distance(2.0, 2.0, 4.0) == 0.0, "lower edge is inclusive"
    assert target_distance(4.0, 2.0, 4.0) == 0.0 + 0.0 or target_distance(4.0, 2.0, 4.0) == 0.0
    assert target_distance(1.5, 2.0, 4.0) == pytest.approx(0.5)
    assert target_distance(5.0, 2.0, 4.0) == pytest.approx(1.0)


# ------------------------------------------------------------------------ windows
def test_windows_partition_every_step_exactly_once():
    w = Windows(t33=14, t67=28)
    fns = {c: w.fn(c) for c in ("early", "middle", "late")}
    for t in range(0, 60):
        assert sum(fn(t) for fn in fns.values()) == 1, f"step {t} covered {[c for c,f in fns.items() if f(t)]}"


def test_throughout_covers_everything_and_unguided_nothing():
    w = Windows(t33=14, t67=28)
    assert all(w.fn("throughout")(t) for t in range(60))
    assert not any(w.fn("unguided")(t) for t in range(60))
    assert all(w.fn("truncation_control")(t) for t in range(60))


def test_windows_split_generated_positions_into_equal_thirds():
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 60, 5000)
    w = Windows.from_lengths(lengths, (1 / 3, 2 / 3))
    positions = np.concatenate([np.arange(1, n + 1) for n in lengths])
    for share in (
        (positions < w.t33).mean(),
        ((positions >= w.t33) & (positions < w.t67)).mean(),
        (positions >= w.t67).mean(),
    ):
        assert abs(share - 1 / 3) < 0.03, "each window must carry about a third of all positions"


def test_windows_are_not_quantiles_of_final_length():
    """Regression: length quantiles would put t33 near the median molecule's end."""
    lengths = np.full(1000, 41)
    w = Windows.from_lengths(lengths, (1 / 3, 2 / 3))
    assert w.t33 < 20 and w.t67 < 32, (w.t33, w.t67)


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        Windows(t33=1, t67=2).fn("sometimes")
