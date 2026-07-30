"""The length/size confound test must kill fake advantages and keep real ones."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "confound", ROOT / "scripts" / "09_confound_analysis.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def make(lengths, atoms, props):
    return [
        {"valid": True, "n_content_tokens": int(L), "n_heavy_atoms": int(A), "clogp": float(p)}
        for L, A, p in zip(lengths, atoms, props)
    ]


def test_a_pure_size_effect_is_removed_by_size_matching():
    """Property is a function of heavy atoms only: matching on size must erase the gain."""
    rng = np.random.default_rng(0)
    n = 20000
    support = np.arange(15, 45)
    ref_atoms = rng.choice(support, n)
    # identical support, but mass tilted towards the heavy end that lands in the
    # target.  The tilt is linear rather than beta-shaped so that every stratum
    # keeps non-trivial mass and the estimator is defined everywhere.
    w = np.arange(1, len(support) + 1, dtype=float)
    cond_atoms = rng.choice(support, n, p=w / w.sum())

    # clogp determined by heavy atoms; token count carries no extra information
    ref = make(ref_atoms * 2, ref_atoms, ref_atoms / 10.0)
    cond = make(cond_atoms * 2, cond_atoms, cond_atoms / 10.0)

    lo, hi = 3.5, 4.5
    raw_ref = mod.standardised_hit_rate(ref, ref, "clogp", lo, hi, "raw")
    raw_cond = mod.standardised_hit_rate(cond, ref, "clogp", lo, hi, "raw")
    assert raw_cond["hit_rate"] > raw_ref["hit_rate"], "condition looks better before matching"

    sized = mod.standardised_hit_rate(cond, ref, "clogp", lo, hi, "size")
    assert sized["coverage"] > 0.95
    assert sized["hit_rate"] == pytest.approx(raw_ref["hit_rate"], abs=0.03), (
        "a pure heavy-atom effect must vanish once heavy-atom count is matched"
    )


def test_a_real_within_stratum_effect_survives_every_matching():
    """Property shifted *inside* each stratum: no matching should erase it."""
    rng = np.random.default_rng(1)
    n = 6000
    atoms = rng.integers(15, 45, n)
    lengths = atoms * 2

    # identical length and size distributions; only the property differs
    ref = make(lengths, atoms, rng.normal(2.0, 0.5, n))
    cond = make(lengths, atoms, rng.normal(4.0, 0.5, n))

    lo, hi = 3.5, 4.5
    base = mod.standardised_hit_rate(ref, ref, "clogp", lo, hi, "raw")["hit_rate"]
    for kind in ("raw", "length", "size", "joint"):
        s = mod.standardised_hit_rate(cond, ref, "clogp", lo, hi, kind)
        assert s["coverage"] > 0.95, kind
        assert s["hit_rate"] - base > 0.3, f"real effect must survive {kind} matching"


def test_length_matching_alone_can_miss_a_size_effect():
    """Why matching on tokens is not enough: same token count, different molecule size."""
    rng = np.random.default_rng(2)
    n = 6000
    lengths = rng.integers(30, 60, n)
    # reference: heavy atoms track length.  condition: same lengths, but denser
    # molecules (fewer branch/ring tokens), so more heavy atoms per token.
    ref = make(lengths, (lengths * 0.5).astype(int), (lengths * 0.5) / 10.0)
    cond = make(lengths, (lengths * 0.8).astype(int), (lengths * 0.8) / 10.0)

    lo, hi = 3.5, 4.5
    by_len = mod.standardised_hit_rate(cond, ref, "clogp", lo, hi, "length")
    by_size = mod.standardised_hit_rate(cond, ref, "clogp", lo, hi, "size")
    base = mod.standardised_hit_rate(ref, ref, "clogp", lo, hi, "raw")["hit_rate"]

    assert by_len["hit_rate"] - base > 0.2, "token matching leaves the size effect intact"
    assert by_size["coverage"] < 0.6, (
        "the size supports barely overlap, which the coverage figure must expose"
    )


def test_disjoint_strata_report_zero_coverage():
    ref = make(np.arange(20, 60), np.arange(10, 50), np.full(40, 4.0))
    cond = make(np.arange(80, 120), np.arange(60, 100), np.full(40, 4.0))
    out = mod.standardised_hit_rate(cond, ref, "clogp", 3.5, 4.5, "joint")
    assert out["coverage"] == 0.0
    assert np.isnan(out["hit_rate"])


def test_empty_reference_is_reported_not_crashed():
    out = mod.standardised_hit_rate(make([30], [15], [4.0]), [], "clogp", 3.5, 4.5, "length")
    assert out["coverage"] == 0.0 and np.isnan(out["hit_rate"])
