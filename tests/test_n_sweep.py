"""C26 -- artefact-binding tests for the N sweep and the compute-accuracy frontier.

Every number `reports/section_c26_n_sweep.md` prints is re-derived here from the JSON and
required to appear in the section text **in the exact format the section prints it**, in the
style of `tests/test_report_matches_artifacts.py`.

Several are deliberately tripwires on the *data* rather than on the prose: if a re-run moves
a verdict -- if the frontier gains a second violation, if the head-seed sign flip goes away,
if a validity gate stops being an identity -- a test fails rather than leaving the section
standing unchallenged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "section_c26_n_sweep.md"
PREREG = OUT / "c26_prereg" / "C26.0_preregistration.md"
SUMMARY = OUT / "c26_summary" / "c26_metrics.json"

ANCHORS = ("aromatic_rings", "hbd_count", "qed")
SEEDS = ("101", "202", "303")

# C26.2.2: the compute-matched N is 9 for two anchors and 8 for QED, because QED's guided
# run is cheaper.  Hard-coded here on purpose -- the summariser reads it from the artefact,
# so a test that also read it from the artefact would not catch the two agreeing wrongly.
PUBLISHED_N = {"aromatic_rings": 9, "hbd_count": 9, "qed": 8}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _have(*paths) -> bool:
    return all(p.exists() for p in paths)


requires_sweep = pytest.mark.skipif(
    not _have(*(OUT / f"c26_nsweep_{p}" / "n_sweep_metrics.json" for p in ANCHORS)),
    reason="C26 sweep artefacts not present",
)
requires_summary = pytest.mark.skipif(
    not SUMMARY.exists(), reason="C26 summary not present")
requires_section = pytest.mark.skipif(
    not SECTION.exists(), reason="C26 report section not present")


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


@pytest.fixture(scope="module")
def section():
    return SECTION.read_text()


@pytest.fixture(scope="module")
def sweeps():
    return {p: _load(OUT / f"c26_nsweep_{p}" / "n_sweep_metrics.json") for p in ANCHORS}


def assert_in(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C26 section"


# ------------------------------------------------------------------ pre-registration


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
@requires_sweep
def test_the_prereg_was_written_before_every_measurement():
    """Ordering by file mtime rather than by trust."""
    t = PREREG.stat().st_mtime
    dirs = [OUT / f"c26_nsweep_{p}" for p in ANCHORS]
    dirs += [OUT / "c26_gate_exact_N9_aromatic_rings", OUT / "c26_summary"]
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                assert t < p.stat().st_mtime, f"{p} is not newer than the pre-registration"


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
@requires_section
def test_the_report_copies_the_prereg_verbatim(section):
    """C26.0 must be the pre-registered text, byte for byte, from C26.0.1 onward."""
    prereg = PREREG.read_text()
    body = prereg[prereg.index("## C26.0.1 Why"):].rstrip()
    assert body in section, "the C26.0 copy is not byte-identical to the pre-registration"


# ------------------------------------------------------------------- validity gates


@requires_summary
@requires_section
def test_gate_1_is_an_exact_reproduction_and_the_section_says_so(summary, section):
    """Gate 1: the published call signature, hit rate AND token cost, residual 0.0."""
    g = summary["validity_gates"]["exact_N9_aromatic_rings"]
    assert g["max_abs_hit_residual"] == 0.0
    for seed, v in g["per_seed"].items():
        assert v["hit_rate_residual"] == 0.0, seed
        assert v["token_residual"] == 0.0, seed
        assert_in(section, repr(v["gate_hit_rate"]), f"gate1 hit seed {seed}")
        assert_in(section, repr(v["gate_tokens_per_molecule"]), f"gate1 tokens seed {seed}")


@requires_summary
def test_gate_2_is_an_identity_not_an_approximation(summary):
    """The sweep's first 512 groups must BE scripts/06_best_of_n.py's, exactly."""
    g = summary["validity_gates"]["first_512_groups_identity"]
    assert g["passes"] is True
    assert g["max_abs_residual"] == 0.0
    for prop, v in g["per_property"].items():
        assert v["published_n"] == PUBLISHED_N[prop], (
            f"{prop}: published N is {v['published_n']}, expected {PUBLISHED_N[prop]}")
        for seed, cell in v["per_seed"].items():
            assert cell["residual"] == 0.0, f"{prop} seed {seed}"


@requires_summary
@requires_section
def test_the_section_prints_every_gate_2_cell(summary, section):
    g = summary["validity_gates"]["first_512_groups_identity"]
    for prop, v in g["per_property"].items():
        for seed, cell in v["per_seed"].items():
            assert_in(section, repr(cell["first_512_groups_hit_rate"]),
                      f"gate2 {prop} seed {seed}")


@requires_sweep
def test_the_sweep_used_all_disjoint_groups_not_the_superseded_nesting(sweeps):
    """The v1 estimator is retired; at N=9 the pool must yield 1820 groups, not 512."""
    for prop, r in sweeps.items():
        row = r["per_seed"]["101"]["rows"]["9"]
        assert row["n_groups"] == 16384 // 9, f"{prop}: {row['n_groups']} groups at N=9"
        assert "first_512_groups_hit_rate" in row


# ------------------------------------------------------------------------ the curves


@requires_sweep
@requires_section
def test_the_section_prints_every_curve_point(sweeps, section):
    """Each (anchor, N) row: hit rate, sd, tokens/mol as the section formats them."""
    missing = []
    for prop, r in sweeps.items():
        for n in r["grid"]:
            c = r["curve"][str(n)]
            for val, fmt in ((c["hit_rate_mean"], f"{c['hit_rate_mean']:.4f}"),
                             (c["hit_rate_sd"], f"{c['hit_rate_sd']:.4f}"),
                             (c["tokens_per_molecule_actual"],
                              f"{c['tokens_per_molecule_actual']:.1f}")):
                if fmt not in section:
                    missing.append((prop, n, fmt))
    assert not missing, f"curve values absent from the section: {missing}"


@requires_summary
def test_every_curve_is_concave_on_the_unequal_grid(summary):
    """Prediction 1.  Divided differences, not the invalid uniform-grid second difference."""
    for prop in ANCHORS:
        p = summary["properties"][prop]
        assert p["curve_is_concave_in_n"] is True, prop
        s = p["curve_secant_slopes_in_n"]
        assert all(s[i + 1] <= s[i] + 1e-12 for i in range(len(s) - 1)), prop


@requires_summary
def test_the_invalid_second_difference_is_kept_and_labelled(summary):
    """The discarded statistic stays visible under a name that says it is invalid."""
    for prop in ANCHORS:
        p = summary["properties"][prop]
        assert "curve_second_differences_uniform_grid_invalid" in p
        # and it really would have given the wrong answer, which is why it is labelled
        second = p["curve_second_differences_uniform_grid_invalid"]
        assert not all(x <= 1e-9 for x in second), (
            f"{prop}: the naive statistic no longer disagrees, so the label is now "
            "misleading and the section's C26.3.1 needs rewriting")


@requires_summary
def test_n_equals_one_reproduces_the_base_policy(summary):
    """Gate 3, as a tripwire: N=1 must stay near the frozen dataset base rate."""
    for prop in ANCHORS:
        n1 = summary["properties"][prop]["curve"]["1"]["hit_rate_mean"]
        assert 0.05 < n1 < 0.25, f"{prop}: N=1 hit rate {n1} is not a base-policy rate"


# --------------------------------------------------------------------- the frontier


@requires_summary
@requires_section
def test_the_section_prints_every_frontier_advantage(summary, section):
    """The headline tables: each arm's advantage as printed, for all 46 arms."""
    missing = []
    total = 0
    for prop in ANCHORS:
        for row in summary["properties"][prop]["guidance_points"]:
            total += 1
            printed = f"{row['advantage']:+.4f}"
            if printed not in section:
                missing.append((prop, row["run"], printed))
    assert total == 46, f"expected 46 priced guidance arms, found {total}"
    assert not missing, f"advantages absent from the section: {missing}"


@requires_summary
@requires_section
def test_the_section_prints_every_frontier_arm_name_and_budget(summary, section):
    missing = []
    for prop in ANCHORS:
        for row in summary["properties"][prop]["guidance_points"]:
            for s in (f"`{row['run']}`",
                      f"{row['tokens_per_molecule_actual']:.1f}",
                      f"{row['guided_hit_rate']:.4f}",
                      f"{row['best_of_n_interpolated_hit_rate']:.4f}"):
                if s not in section:
                    missing.append((prop, row["run"], s))
    assert not missing, f"frontier fields absent from the section: {missing}"


@requires_summary
def test_exactly_one_arm_sits_above_the_best_of_n_curve(summary):
    """Tripwire on D1.  If a re-run adds or removes a violation, this fails."""
    d1 = summary["decision_rules"]["D1_best_of_n_dominates_everywhere"]
    assert d1["upheld"] is False
    assert len(d1["violations"]) == 1, [v["run"] for v in d1["violations"]]
    v = d1["violations"][0]
    assert v["property"] == "hbd_count"
    assert v["run"] == "c23_guided_L4_lam2_hbd_count"
    assert v["advantage"] > 0


@requires_summary
def test_the_one_violation_is_not_significant_on_the_t_interval(summary):
    v = summary["decision_rules"]["D1_best_of_n_dominates_everywhere"]["violations"][0]
    assert v["advantage_seed_t_interval"]["excludes_zero"] is False
    # one of the three generation seeds is on the wrong side of zero
    assert min(v["advantage_per_seed"]) < 0


@requires_summary
def test_the_seed_bootstrap_is_recorded_as_vacuous_at_n3(summary):
    """At n=3 the percentile bootstrap of a mean is exactly [min, max].

    P(all three resampled indices hit the minimum) = 1/27 = 0.037 > 0.025, so the 2.5th
    percentile IS the minimum for any three numbers.  The interval therefore conveys only
    a three-way sign test at null probability 0.25 and must never be read as a CI.  This
    test fails if a future edit quietly restores it to a load-bearing role.
    """
    v = summary["decision_rules"]["D1_best_of_n_dominates_everywhere"]["violations"][0]
    ci = v["advantage_seed_sign_test"]
    ps = v["advantage_per_seed"]
    assert ci["degenerate_equals_min_max"] is True
    assert ci["degenerate_bootstrap_lo"] == min(ps)
    assert ci["degenerate_bootstrap_hi"] == max(ps)
    assert ci["sign_test_p_two_sided"] == 0.25
    assert "advantage_seed_bootstrap_ci" not in v, (
        "the degenerate interval has been reinstated under its old CI name")


@requires_summary
@requires_section
def test_the_section_prints_d2s_own_per_seed_advantages(summary, section):
    """Guards the defect this section had: it printed D2b's per-seed vector next to D2's
    intervals.  The two decompositions differ (+0.0602/-0.0030/+0.0229 against
    +0.0607/-0.0006/+0.0198) and mixing them makes the published interval unverifiable."""
    d2 = summary["decision_rules"]["D2_c23_arm_priced_on_the_curve"]
    for x in d2["advantage_per_seed"]:
        assert_in(section, f"{x:+.4f}", "D2 per-seed advantage")
    t = d2["advantage_seed_t_interval"]
    for x in (t["mean"], t["lo"], t["hi"]):
        assert_in(section, f"{x:+.4f}", "D2 t interval")


@requires_summary
@requires_section
def test_guidance_has_no_compute_knob(summary, section):
    """C26.4.4.  The structural finding, bound to the data that produces it."""
    for prop in ANCHORS:
        toks = [r["tokens_per_molecule_actual"]
                for r in summary["properties"][prop]["guidance_points"]]
        spread = (max(toks) - min(toks)) / min(toks)
        assert spread < 0.20, f"{prop}: guidance token spread is now {spread:.3f}"
        assert_in(section, f"{max(toks) - min(toks):.1f}", f"{prop} token spread")
        curve = summary["properties"][prop]["curve"]
        grid = summary["properties"][prop]["grid"]
        ct = [curve[str(n)]["tokens_per_molecule_actual"] for n in grid]
        assert max(ct) / min(ct) > 20, f"{prop}: best-of-N span collapsed"


# ------------------------------------------- D2b: the head-seed replication, the verdict


@requires_summary
def test_the_c23_rule_b_arm_flips_sign_across_head_seeds(summary):
    """The decisive tripwire.  If this stops failing to be positive, Rule B is back."""
    d2b = summary["decision_rules"]["D2b_c23_arm_across_c25_head_seeds"]
    agg = d2b["_across_head_seeds"]
    assert agg["sign_flips"] is True
    assert agg["mean"] < 0, agg["mean"]
    assert agg["min"] < 0 < agg["max"]
    assert len(agg["advantages"]) == 3


@requires_summary
def test_head_seed_variance_dominates_generation_seed_variance(summary):
    """The reason C23's three generation seeds were replicates of the wrong thing."""
    d2b = summary["decision_rules"]["D2b_c23_arm_across_c25_head_seeds"]
    head_span = d2b["_across_head_seeds"]["guided_hit_rate_span"]
    gen_spans = []
    for hs, v in d2b.items():
        if hs.startswith("_"):
            continue
        run = OUT / v["run"] / "guidance_metrics.json"
        vals = _load(run)["conditions"]["throughout"]["aggregate"]["hit_rate"]["values"]
        gen_spans.append(max(vals) - min(vals))
    assert head_span > 5 * min(gen_spans), (
        f"head-seed span {head_span:.4f} vs smallest generation-seed span "
        f"{min(gen_spans):.4f}")


@requires_summary
@requires_section
def test_the_section_prints_every_head_seed_cell(summary, section):
    d2b = summary["decision_rules"]["D2b_c23_arm_across_c25_head_seeds"]
    for hs, v in d2b.items():
        if hs.startswith("_"):
            continue
        for s in (f"{v['guided_hit_rate']:.4f}",
                  f"{v['tokens_per_molecule_actual']:.2f}",
                  f"{v['best_of_n_interpolated_hit_rate']:.4f}",
                  f"{v['advantage']:+.4f}"):
            assert_in(section, s, f"head seed {hs}")
    agg = d2b["_across_head_seeds"]
    assert_in(section, f"{agg['mean']:+.4f}", "head-seed mean advantage")
    assert_in(section, f"{agg['sd']:.4f}", "head-seed sd")
    assert_in(section, f"{agg['guided_hit_rate_span']:.4f}", "head-seed hit-rate span")


# ------------------------------------------------------------------------ D3, saturation


@requires_summary
@requires_section
def test_the_section_prints_the_saturation_numbers(summary, section):
    d3 = summary["decision_rules"]["D3_best_of_n_saturates"]
    for prop, v in d3.items():
        assert_in(section, f"{v['hit_rate_at_max_n']:.4f}", f"{prop} hit at N=32")
        assert_in(section, f"{v['last_doubling_gain']:+.4f}", f"{prop} last doubling")


@requires_summary
def test_only_aromatic_rings_has_saturated(summary):
    """D3's verdict, as a tripwire on the data."""
    d3 = summary["decision_rules"]["D3_best_of_n_saturates"]
    assert d3["aromatic_rings"]["last_doubling_gain"] < 0.10
    assert d3["hbd_count"]["last_doubling_gain"] > 0.10
    assert d3["qed"]["last_doubling_gain"] > 0.10


# ----------------------------------------------------------- the section's own honesty


@requires_section
def test_the_section_does_not_claim_rule_b_survives(section):
    """A prose tripwire: the verdict must be stated, not hedged away."""
    assert "C23's Rule B is dead" in section
    assert "does not survive head-seed replication" in section


@requires_section
def test_the_section_records_the_superseded_estimator(section):
    """C26.0.2's pre-registered design failed its own gate; that must stay visible."""
    for s in ("c26_nsweep_v1_nested", "−0.0195", "+0.0209", "+0.0384"):
        assert_in(section, s, "superseded estimator record")


@requires_section
def test_the_section_records_that_the_published_comparator_moved(section):
    """The more precise estimator lowers best-of-9; that favours guidance and is adopted."""
    for s in ("0.8150", "0.8294", "0.0144"):
        assert_in(section, s, "comparator shift")
