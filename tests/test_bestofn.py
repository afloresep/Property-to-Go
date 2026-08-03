"""Best-of-N selection semantics, exercised without the checkpoint."""

import numpy as np
import pytest

from property_to_go import bestofn
from property_to_go.compute import ComputeMeter


class FakeGen:
    """Stands in for FrozenGenerator: returns a fixed pool of SMILES."""

    def __init__(self, pool):
        self.pool = pool
        self.calls = 0

    def decode(self, seqs):
        return [self.pool[i[0]] for i in seqs]


def patched(monkeypatch, pool):
    gen = FakeGen(pool)

    def fake_sample(g, policy, n, seed, meter=None):
        g.calls += 1
        seqs = [[i % len(pool)] for i in range(n)]
        if meter is not None:
            for _ in seqs:
                meter.add_forward(40)
                meter.molecules_returned += 1
        return seqs

    monkeypatch.setattr(bestofn, "sample_unconditional", fake_sample)
    return gen


def test_a_count_one_above_the_target_is_not_a_zero_distance_tie():
    """Regression: with target [3, 4), a 4-ring molecule scored distance 0.

    `target_distance` is the true set distance to a half-open interval, so it is
    0 at value == hi.  Used as the sole ranking key that ties a miss with a hit,
    and best-of-N then returns whichever came first -- which collapsed the
    measured aromatic-ring hit rate from ~0.81 to ~0.70.
    """
    assert bestofn.target_distance(4, 3.0, 4.0) == 0.0, "the underlying subtlety"
    assert not bestofn.in_target(4, 3.0, 4.0)

    hit = bestofn.selection_key(3, 3.0, 4.0)
    miss = bestofn.selection_key(4, 3.0, 4.0)
    assert hit < miss, "a genuine hit must outrank a value sitting exactly on hi"
    assert bestofn.selection_key(3, 3.0, 4.0) < bestofn.selection_key(2, 3.0, 4.0)


def test_every_discrete_property_is_declared_integer_valued():
    """docs/HANDOFF.md §4: an integer property missing from INTEGER_PROPERTIES silently
    reintroduces the boundary-selection bug -- best-of-N ranks a value sitting on `hi`
    as a tie with a real hit, and the only symptom is a hit rate below its own binomial
    prediction. Derived from DISCRETE_PROPERTIES rather than restated, so adding a
    count property and forgetting the declaration fails here instead of in a result.
    """
    from property_to_go import properties as P

    missing = set(P.DISCRETE_PROPERTIES) - set(bestofn.INTEGER_PROPERTIES)
    assert not missing, f"integer-valued but not declared in INTEGER_PROPERTIES: {missing}"
    spurious = set(bestofn.INTEGER_PROPERTIES) - set(P.DISCRETE_PROPERTIES)
    assert not spurious, f"declared integer-valued but not a count property: {spurious}"


def test_the_boundary_bug_is_guarded_for_each_count_property():
    """The §4 regression, exercised at each count property's own target width."""
    from property_to_go import properties as P

    for prop in sorted(P.DISCRETE_PROPERTIES):
        lo, hi = 2.0, 3.0  # any one-unit count target
        assert bestofn.target_distance(hi, lo, hi) == 0.0, "the underlying subtlety"
        assert not bestofn.in_target(hi, lo, hi)
        assert bestofn.selection_key(lo, lo, hi) < bestofn.selection_key(hi, lo, hi)
        assert bestofn.target_error(
            hi, lo, hi, integer_valued=prop in bestofn.INTEGER_PROPERTIES
        ) == pytest.approx(1.0), f"{prop}: one step out must not score as zero error"


def test_target_error_measures_to_the_attainable_value_for_counts():
    """Reporting 0 error for a molecule that missed the target is wrong."""
    assert bestofn.target_error(4, 3.0, 4.0, integer_valued=True) == pytest.approx(1.0)
    assert bestofn.target_error(5, 3.0, 4.0, integer_valued=True) == pytest.approx(2.0)
    assert bestofn.target_error(3, 3.0, 4.0, integer_valued=True) == 0.0
    assert bestofn.target_error(2, 3.0, 4.0, integer_valued=True) == pytest.approx(1.0)
    # continuous properties keep the plain interval distance
    assert bestofn.target_error(5.5, 4.0, 5.0) == pytest.approx(0.5)
    assert bestofn.target_error(4.5, 4.0, 5.0) == 0.0


def test_best_of_n_prefers_a_real_hit_over_a_boundary_miss(monkeypatch):
    """End-to-end: a 3-ring hit must beat a 4-ring miss even when the miss is first.

    The 4-ring quaterphenyl is placed *first* in the pool deliberately: under the
    old distance-only key both scored 0.0 and the first one won, so this test
    fails against the buggy version and passes against the fixed one.
    """
    quaterphenyl = "c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1"  # 4 aromatic rings
    terphenyl = "c1ccc(-c2cccc(-c3ccccc3)c2)cc1"  # 3 aromatic rings
    gen = patched(monkeypatch, [quaterphenyl, terphenyl])
    picked = bestofn.best_of_n(gen, {}, "aromatic_rings", 3.0, 4.0,
                               n_molecules=1, n_candidates=2, seed=0)
    assert picked[0]["aromatic_rings"] == 3, "must not return the boundary miss"


def test_picks_the_candidate_inside_the_target(monkeypatch):
    # cLogP: ethanol -0.00, benzene 1.69, octane 3.37, dodecane 4.93
    pool = ["CCO", "c1ccccc1", "CCCCCCCC", "CCCCCCCCCCCC"]
    gen = patched(monkeypatch, pool)
    picked = bestofn.best_of_n(gen, {}, "clogp", 4.5, 5.5, n_molecules=1, n_candidates=4, seed=0)
    assert len(picked) == 1
    assert picked[0]["smiles"] == "CCCCCCCCCCCC"
    assert 4.5 <= picked[0]["clogp"] < 5.5


def test_picks_the_nearest_candidate_when_none_hit(monkeypatch):
    pool = ["CCO", "c1ccccc1", "CCCCCCCC", "C"]
    gen = patched(monkeypatch, pool)
    picked = bestofn.best_of_n(gen, {}, "clogp", 4.5, 5.5, n_molecules=1, n_candidates=4, seed=0)
    assert picked[0]["smiles"] == "CCCCCCCC", "octane at 3.37 is nearest to [4.5, 5.5)"


def test_invalid_candidates_rank_worst_but_are_still_returnable(monkeypatch):
    gen = patched(monkeypatch, ["not_a_molecule", "CCO"])
    picked = bestofn.best_of_n(gen, {}, "clogp", 0.0, 1.0, n_molecules=1, n_candidates=2, seed=0)
    assert picked[0]["valid"] is True and picked[0]["smiles"] == "CCO"

    gen = patched(monkeypatch, ["not_a_molecule", "also_bad"])
    picked = bestofn.best_of_n(gen, {}, "clogp", 0.0, 1.0, n_molecules=1, n_candidates=2, seed=0)
    assert picked[0]["valid"] is False, "an all-invalid pool must return an invalid molecule"


def test_one_generation_call_covers_the_whole_pool(monkeypatch):
    gen = patched(monkeypatch, ["CCO", "CCCCCCCC"])
    m = ComputeMeter()
    bestofn.best_of_n(gen, {}, "clogp", 3.5, 5.0, n_molecules=8, n_candidates=5, seed=0, meter=m)
    assert gen.calls == 1, "candidates must be drawn in one batched call, not one call per slot"


def test_cost_is_reported_per_returned_molecule(monkeypatch):
    gen = patched(monkeypatch, ["CCO", "CCCCCCCC"])
    m = ComputeMeter()
    bestofn.best_of_n(gen, {}, "clogp", 3.5, 5.0, n_molecules=10, n_candidates=7, seed=0, meter=m)
    d = m.as_dict()
    assert d["molecules_returned"] == 10, "returned molecules, not candidates drawn"
    assert d["processed_tokens_actual"] == 10 * 7 * 40
    assert d["tokens_per_molecule_actual"] == pytest.approx(7 * 40)


def test_summarise_reports_hit_rate_and_spread(monkeypatch):
    recs = [
        {"valid": True, "clogp": 4.0, "canonical_smiles": "A"},
        {"valid": True, "clogp": 1.0, "canonical_smiles": "B"},
        {"valid": True, "clogp": 4.5, "canonical_smiles": "A"},
        {"valid": False},
    ]
    s = bestofn.summarise(recs, "clogp", 3.5, 5.0)
    assert s["n"] == 4 and s["n_valid"] == 3
    assert s["validity"] == pytest.approx(0.75)
    assert s["uniqueness"] == pytest.approx(2 / 3)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["hit_rate_over_all_returned"] == pytest.approx(0.5)
    assert s["abs_target_error_mean"] == pytest.approx(2.5 / 3)


def test_summarise_handles_a_fully_invalid_batch():
    s = bestofn.summarise([{"valid": False}] * 3, "clogp", 0.0, 1.0)
    assert s["validity"] == 0.0 and s["n"] == 3


def test_length_matched_hit_rate_removes_a_pure_length_effect():
    """A condition whose only effect is on length must show no length-matched advantage."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "guided", __file__.replace("tests/test_bestofn.py", "scripts/05_guided_generation.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)

    def make(lengths):
        # property is a deterministic function of length: nothing but length matters
        return [
            {"valid": True, "n_content_tokens": int(L), "clogp": float(L) / 10.0,
             "canonical_smiles": f"m{i}"}
            for i, L in enumerate(lengths)
        ]

    # same rule and same length support, but the mass is concentrated on the
    # lengths that happen to land in the target window
    ref = make(rng.integers(20, 80, 8000))
    reshaped = make(20 + (60 * rng.beta(2, 2, 8000)).astype(int).clip(0, 59))

    lo, hi = 4.0, 5.0
    raw_ref = np.mean([lo <= r["clogp"] < hi for r in ref])
    raw_long = np.mean([lo <= r["clogp"] < hi for r in reshaped])
    assert raw_long > raw_ref, "the reshaped condition looks better before matching"

    matched = mod.length_matched_hit_rate(reshaped, ref, "clogp", lo, hi)
    assert matched["coverage"] > 0.99, "the two conditions share their length support"
    assert abs(matched["length_matched_hit_rate"] - raw_ref) < 0.03, (
        "after matching the reference length distribution the advantage must vanish"
    )


def test_length_matching_reports_low_coverage_when_supports_barely_overlap():
    """The estimate is only defined on the common length support; say so."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "guided", __file__.replace("tests/test_bestofn.py", "scripts/05_guided_generation.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def make(lengths):
        return [
            {"valid": True, "n_content_tokens": int(L), "clogp": float(L) / 10.0,
             "canonical_smiles": f"m{i}"}
            for i, L in enumerate(lengths)
        ]

    ref = make(np.arange(20, 60))
    disjoint = make(np.arange(60, 100))
    out = mod.length_matched_hit_rate(disjoint, ref, "clogp", 4.0, 5.0)
    assert out["coverage"] == 0.0
