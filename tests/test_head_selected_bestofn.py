"""C27 -- artefact-binding tests for the head-selected best-of-N control.

Every number `reports/section_c27_head_selected_bestofn.md` prints is re-derived here from
JSON and required to appear in the section text **in the exact format the section prints
it**, in the style of `tests/test_report_matches_artifacts.py` and `tests/test_n_sweep.py`.

Several are deliberately tripwires on the *data* rather than on the prose: if a re-run moves
a verdict -- if the head arm stops reproducing C26's oracle arm exactly, if the head's pool
AUROC collapses towards chance, if the count of guidance arms above the head-selected curve
changes, if E4 flips sign on any anchor -- a test fails rather than leaving the section
standing unchallenged.

The section writes machine-derived numbers with ASCII hyphens.  A Unicode minus (U+2212)
would silently break every `in` assertion below, so one test forbids it outright.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "section_c27_head_selected_bestofn.md"
PREREG = OUT / "c27_prereg" / "C27.0_preregistration.md"
LOCK = OUT / "c27_prereg" / "prereg_lock.json"
SUMMARY = OUT / "c27_summary" / "c27_metrics.json"

ANCHORS = ("aromatic_rings", "hbd_count", "qed")
SEEDS = ("101", "202", "303")
ARMS = ("oracle_selected", "head_selected", "head_selected_at_75pct")
GRID = (1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32)

# The deployed head, hard-coded rather than read from the artefact: a test that read the
# checkpoint name from the same file the summariser read it from would not catch the two
# agreeing wrongly.  C27.0.3 resolves it from
# outputs/pilot_50k_p2_guided_<prop>/guidance_metrics.json.
DEPLOYED_HEAD = {p: f"head_{p}_frozen_state.pt" for p in ANCHORS}
DEPLOYED_HEAD_SEED = 1234
DEPLOYED_LAYER = -1


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _have(*paths) -> bool:
    return all(p.exists() for p in paths)


requires_sweeps = pytest.mark.skipif(
    not _have(*(OUT / f"c27_headsel_{p}" / "head_selected_metrics.json" for p in ANCHORS)),
    reason="C27 sweep artefacts not present")
requires_summary = pytest.mark.skipif(
    not SUMMARY.exists(), reason="C27 summary not present")
requires_section = pytest.mark.skipif(
    not SECTION.exists(), reason="C27 report section not present")


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


@pytest.fixture(scope="module")
def full_section():
    return SECTION.read_text()


@pytest.fixture(scope="module")
def section(full_section):
    """The section WITHOUT the verbatim pre-registration copy.

    Every `assert_in` below binds a number this section derived from JSON.  Checking
    against the whole file would let a number pass merely because the pre-registration
    quoted it -- 0.5603, 0.7904, 0.7781 and 0.7355 all appear in C27.0.9 -- so the quoted
    block is removed before any binding assertion runs.
    """
    return _outside_prereg(full_section)


@pytest.fixture(scope="module")
def sweeps():
    return {p: _load(OUT / f"c27_headsel_{p}" / "head_selected_metrics.json") for p in ANCHORS}


def f4(x: float) -> str:
    return f"{x:.4f}"


def f1(x: float) -> str:
    return f"{x:.1f}"


def assert_in(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C27 section"


# ------------------------------------------------------------------ pre-registration


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
@requires_sweeps
def test_the_prereg_was_written_before_every_measurement():
    """Ordering by file mtime rather than by trust."""
    t = PREREG.stat().st_mtime
    for d in sorted(OUT.glob("c27_*")):
        if d.name == "c27_prereg":
            continue
        target = [d] if d.is_file() else list(d.rglob("*"))
        for p in target:
            if p.is_file():
                assert t < p.stat().st_mtime, f"{p} is not newer than the pre-registration"


@pytest.mark.skipif(not (PREREG.exists() and LOCK.exists()), reason="prereg not present")
def test_the_prereg_lock_records_the_prereg_hash():
    lock = _load(LOCK)
    assert lock["file_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    body = PREREG.read_text()
    block = body[body.index("## C27.0.1 Why"):].rstrip()
    assert lock["prereg_block_sha256"] == hashlib.sha256(block.encode()).hexdigest()
    assert lock["prereg_block_chars"] == len(block)


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
@requires_section
def test_the_report_copies_the_prereg_verbatim(full_section):
    """C27.0 must be the pre-registered text, byte for byte, from C27.0.1 onward."""
    prereg = PREREG.read_text()
    body = prereg[prereg.index("## C27.0.1 Why"):].rstrip()
    assert body in full_section, "the C27.0 copy is not byte-identical to the pre-registration"


def _outside_prereg(section: str) -> str:
    """The section with the verbatim pre-registration copy removed.

    The pre-registration is a frozen historical document that must be reproduced byte for
    byte, and it happens to contain two Unicode minus signs.  The ASCII-hyphen rule binds
    the numbers *this section* derives from JSON, not the quoted prereg.
    """
    a = section.index("<!-- BEGIN VERBATIM PREREG COPY -->")
    b = section.index("<!-- END VERBATIM PREREG COPY -->")
    return section[:a] + section[b:]


@requires_section
def test_machine_derived_numbers_use_ascii_hyphens(section):
    """A Unicode minus would break every binding assertion in this file silently."""
    assert "\u2212" not in section, "the section contains a Unicode minus (U+2212)"


# ------------------------------------------------------------------- validity gates


@requires_summary
@requires_section
def test_gate_1_the_oracle_arm_reproduces_c26_exactly(summary, section):
    """Gate 1 is an identity, not an approximation: residual must be exactly 0.0."""
    g = summary["validity_gates"]["gate_1_oracle_arm_reproduces_c26"]
    assert g["max_abs_hit_rate_residual"] == 0.0
    assert g["max_abs_token_residual"] == 0.0
    assert g["passes"] is True
    for prop, v in g["per_property"].items():
        assert v["max_abs_hit_rate_residual"] == 0.0, prop
        assert v["max_abs_token_residual"] == 0.0, prop
        # 11 grid points x 3 seeds per anchor
        assert v["n_cells"] == len(GRID) * len(SEEDS), prop
    assert_in(section, "0.0", "gate 1 residual")
    assert_in(section, str(len(GRID) * len(SEEDS) * len(ANCHORS)), "gate 1 cell count")


@requires_summary
@requires_section
def test_gate_2_head_pool_auroc_is_reported_per_anchor_and_seed(summary, section):
    """Gate 2: the number that says whether the head arm measures anything at all."""
    g = summary["validity_gates"]["gate_2_head_pool_auroc"]
    for prop in ANCHORS:
        v = g["per_property"][prop]
        for seed in SEEDS:
            assert_in(section, f4(v["per_seed"][seed]["terminal"]),
                      f"gate2 terminal AUROC {prop}/{seed}")
        assert_in(section, f4(v["at_75pct_mean"]), f"gate2 75% AUROC mean {prop}")
        # tripwire: if the head ever becomes uninformative the section's whole
        # equal-information claim is void, and this test must fail rather than the
        # section quietly meaning something else.
        assert v["terminal_min"] > 0.55, f"{prop}: head AUROC near chance"
    assert g["any_near_chance"] is False


@requires_summary
def test_gate_3_n1_is_identical_across_all_three_arms(summary):
    """One candidate, nothing to select: a difference at N=1 is a selection bug."""
    g = summary["validity_gates"]["gate_3_n1_identical_across_arms"]
    assert g["max_abs_residual"] == 0.0
    assert g["passes"] is True
    for prop, v in g["per_property"].items():
        for cell, r in v["cells"].items():
            assert r["residual"] == 0.0, f"{prop} {cell}"


@requires_summary
@requires_section
def test_gate_4_the_head_is_the_deployed_one(summary, section):
    """The checkpoint must be the deployed lambda=1 head, by CONTENT.

    C27.0.5's file-SHA-256 criterion is expected to FAIL -- `torch.save` names the zip
    archive after the output file, so the unsuffixed checkpoint and its seed-1234 twin
    differ in bytes while holding identical tensors.  The failure is asserted here, so
    that a future serialisation change which silently made the criterion pass would break
    this test and force section C27.2.4 to be rewritten rather than left stale.
    """
    g = summary["validity_gates"]["gate_4_head_provenance"]
    assert g["preregistered_file_sha256_criterion_passes"] is False
    assert g["all_tensors_identical_to_seed_twin"] is True
    assert g["all_metadata_identical_to_seed_twin"] is True
    assert g["all_binners_identical_to_seed_twin"] is True
    assert g["all_layers_minus_one"] is True
    for prop in ANCHORS:
        h = g["per_property"][prop]
        assert h["head_checkpoint_name"] == DEPLOYED_HEAD[prop]
        assert h["head_seed"] == DEPLOYED_HEAD_SEED
        assert h["heads_json_head_seed"] == DEPLOYED_HEAD_SEED
        assert h["layer"] == DEPLOYED_LAYER
        assert h["head_input"] == "frozen_state"
        assert h["head_input_in_checkpoint"] == "frozen_state"
        assert h["prereg_criterion_file_sha256_equal"] is False
        assert h["file_bytes"] != h["seed_twin_file_bytes"]
        assert_in(section, DEPLOYED_HEAD[prop], f"gate4 checkpoint name {prop}")
        assert_in(section, h["parameter_sha256"][:16], f"gate4 parameter sha256 {prop}")
        assert_in(section, str(h["file_bytes"]), f"gate4 file bytes {prop}")
        assert_in(section, str(h["seed_twin_file_bytes"]), f"gate4 twin bytes {prop}")
        assert_in(section, str(h["n_bins"]), f"gate4 n_bins {prop}")


@requires_summary
@requires_section
def test_gate_5_both_arms_cost_the_same_tokens_and_match_c26(summary, section):
    """Selection differs; generation does not.  44.4 at N=1 up to 1422.0 at N=32."""
    g = summary["validity_gates"]["gate_5_token_identity_across_arms"]
    assert g["passes"] is True
    for prop in ANCHORS:
        v = g["per_property"][prop]
        assert v["max_abs_token_residual"] == 0.0, prop
        assert f1(v["tokens_per_molecule_at_n1"]) == "44.4", prop
        assert f1(v["tokens_per_molecule_at_n32"]) == "1422.0", prop
    assert_in(section, "44.4", "N=1 tokens/molecule")
    assert_in(section, "1422.0", "N=32 tokens/molecule")


# ------------------------------------------------------------------------- the curves


@requires_summary
@requires_section
def test_every_printed_curve_value_is_in_the_section(summary, section):
    """Both curves, every anchor, every grid point, in the section's own format."""
    for prop in ANCHORS:
        pr = summary["properties"][prop]
        for arm in ("oracle_selected", "head_selected", "head_selected_at_75pct"):
            for n in GRID:
                c = pr["curves"][arm][str(n)]
                assert_in(section, f4(c["hit_rate_mean"]), f"{prop} {arm} N={n} hit")
        for n in GRID:
            assert_in(section, f1(pr["curves"]["oracle_selected"][str(n)]
                                  ["tokens_per_molecule_actual"]),
                      f"{prop} N={n} tokens")


@requires_summary
def test_head_selection_never_beats_the_oracle_it_is_evaluated_against(summary):
    """A head arm above the oracle arm at the same N would be an evaluation bug."""
    for prop in ANCHORS:
        pr = summary["properties"][prop]
        for n in GRID:
            o = pr["curves"]["oracle_selected"][str(n)]["hit_rate_mean"]
            for arm in ("head_selected", "head_selected_at_75pct"):
                h = pr["curves"][arm][str(n)]["hit_rate_mean"]
                assert h <= o + 1e-12, f"{prop} {arm} N={n}: {h} > oracle {o}"


# ------------------------------------------------------------------- decision rules


@requires_summary
@requires_section
def test_E1_arm_count_and_verdict_match_the_section(summary, section):
    """Tripwire: if the number of arms above the head-selected curve moves, this fails."""
    e1 = summary["decision_rules"]["E1_head_selection_still_beats_steering"]
    assert e1["n_arms_total"] == 46, "the frontier should carry 46 existing guidance arms"
    assert_in(section, str(e1["n_arms_total"]), "E1 total arms")
    assert_in(section, str(e1["n_arms_above"]), "E1 arms above")
    for prop in ANCHORS:
        pr = summary["properties"][prop]
        assert_in(section, str(pr["n_guidance_arms"]), f"E1 arm count {prop}")
    for v in e1["violations"]:
        assert_in(section, v["run"], "E1 violation run name")
        assert_in(section, f4(v["advantage_vs_head_selected"]), "E1 violation margin")


@requires_summary
@requires_section
def test_E2_price_of_ground_truth_is_printed(summary, section):
    e2 = summary["decision_rules"]["E2_price_of_ground_truth"]
    for prop in ANCHORS:
        v = e2[prop]
        assert_in(section, f4(v["gap_at_n9"]), f"E2 gap at N=9 {prop}")
        assert_in(section, f4(v["gap_at_n32"]), f"E2 gap at N=32 {prop}")
        assert_in(section, f4(v["max_gap"]), f"E2 max gap {prop}")


@requires_summary
def test_E3_degeneracy_check_is_recorded_for_every_anchor(summary):
    e3 = summary["decision_rules"]["E3_degeneracy_check"]
    for prop in ANCHORS:
        v = e3[prop]
        assert v["threshold"] == 0.02
        # tripwire: a degenerate head arm voids E1 on that anchor, so the section would
        # have to say something different.
        assert v["degenerate"] is False, f"{prop}: head arm gained < 0.02 from N=1 to N=32"


@requires_summary
@requires_section
def test_E4_deployed_arm_per_seed_values_and_t_interval(summary, section):
    """The headline comparison: deployed lambda=1 guidance vs the head-selected curve."""
    e4 = summary["decision_rules"]["E4_deployed_lambda1_arm_vs_head_selected_curve"]
    for prop in ANCHORS:
        v = e4[prop]
        assert v["run"] == f"pilot_50k_p2_guided_{prop}"
        assert_in(section, f4(v["guided_hit_rate"]), f"E4 guided hit {prop}")
        assert_in(section, f4(v["head_selected_interpolated_hit_rate"]),
                  f"E4 head-selected @ budget {prop}")
        assert_in(section, f4(v["advantage_vs_head_selected"]), f"E4 advantage {prop}")
        for a in v["advantage_vs_head_selected_per_seed"]:
            assert_in(section, f4(a), f"E4 per-seed advantage {prop}")
        ti = v["advantage_seed_t_interval"]
        assert ti["n_seeds"] == 3
        assert_in(section, f4(ti["lo"]), f"E4 t interval lo {prop}")
        assert_in(section, f4(ti["hi"]), f"E4 t interval hi {prop}")
        # tripwire on the verdict itself
        assert v["advantage_vs_head_selected"] < 0, (
            f"{prop}: the deployed guidance arm is now ABOVE the head-selected curve; "
            "C27's headline has flipped and the section must be rewritten")


@requires_summary
@requires_section
def test_the_headline_gap_reduction_is_re_derived(summary, section):
    """How much of C26's measured gap was the oracle: 1 - adv_head / adv_oracle."""
    e4 = summary["decision_rules"]["E4_deployed_lambda1_arm_vs_head_selected_curve"]
    for prop in ANCHORS:
        v = e4[prop]
        frac = 1.0 - v["advantage_vs_head_selected"] / v["advantage_vs_oracle_selected"]
        assert_in(section, f"{frac:.3f}", f"gap reduction {prop}")
        # tripwire: the whole section turns on this being most of the gap
        assert frac > 0.5, f"{prop}: equalising information no longer removes most of the gap"
        assert_in(section, f4(v["advantage_vs_oracle_selected"]),
                  f"E4 advantage vs oracle {prop}")


@requires_summary
@requires_section
def test_E1_violations_are_high_lambda_or_mid_layer(summary, section):
    """The section's explanation of *which* arms win must hold in the data."""
    e1 = summary["decision_rules"]["E1_head_selection_still_beats_steering"]
    hi_or_mid = [v for v in e1["violations"]
                 if v["lam"] >= 2.0 or v["layer"] not in (-1, 12)]
    others = [v for v in e1["violations"] if v not in hi_or_mid]
    assert len(hi_or_mid) == 14, "C27.5.1 counts 14 high-lambda-or-mid-layer violations"
    assert [v["run"] for v in others] == ["c18_guided_binT0p4_aromatic_rings"], (
        "C27.5.1 names exactly one exception to the high-lambda-or-mid-layer rule")
    assert_in(section, str(len(hi_or_mid)), "count of high-lambda-or-mid-layer violations")
    for fam, n in (("c23_mid_layer", 10), ("section19_lambda_sweep", 4),
                   ("c18_calibration_or_readout", 1)):
        assert sum(1 for v in e1["violations"] if v["family"] == fam) == n, fam
        assert_in(section, str(n), f"count of {fam} violations")
    # no deployed-configuration arm may be among them, or the section's headline
    # ("the deployed configuration still loses") contradicts its own table
    assert not any(v["family"] == "deployed_lambda1" for v in e1["violations"])


@requires_section
def test_no_bootstrap_is_reported_anywhere_in_c27(section, summary):
    """C27.0.7 forbids a three-seed bootstrap; nothing may reintroduce one."""
    blob = json.dumps(summary)
    assert "bootstrap_ci" not in blob
    assert "seed_bootstrap" not in blob
    lowered = section.lower()
    for bad in ("bootstrap ci", "bootstrapped ci", "percentile bootstrap of the advantage"):
        assert bad not in lowered, f"the section reports a bootstrap: {bad!r}"


# -------------------------------------------------------------------- sensitivity S1


@requires_summary
@requires_section
def test_S1_pessimistic_accounting_is_reported(summary, section):
    """If head scoring were charged in full, does any verdict change?"""
    s1 = summary["sensitivity_S1_pessimistic_accounting"]
    for prop in ANCHORS:
        v = s1[prop]
        assert v["recompute_tokens_per_pool_molecule"] > 0
        assert_in(section, f1(v["recompute_tokens_per_pool_molecule"]),
                  f"S1 recompute cost {prop}")
        assert_in(section, str(v["n_arms_above_head_selected_curve"]),
                  f"S1 arms above {prop}")


# --------------------------------------------------------------------- predictions


@requires_summary
@requires_section
def test_every_prediction_is_scored_in_the_section(summary, section):
    preds = summary["predictions"]
    for prop in ANCHORS:
        assert prop in preds
    for tag in ("prediction 1", "prediction 2", "prediction 3",
                "prediction 4", "prediction 5", "prediction 6"):
        assert tag in section.lower(), f"{tag} is not scored in the section"


@requires_summary
@requires_section
def test_prediction_5_pool_auroc_vs_pooled_heldout(summary, section):
    """Both numbers must be printed, whichever way the prediction went."""
    for prop in ANCHORS:
        p = summary["predictions"][prop]
        assert_in(section, f4(p["heads_json_pooled_test_auroc"]),
                  f"pooled held-out AUROC {prop}")


# ------------------------------------------------------------------------- REPRODUCE


@requires_section
def test_the_section_carries_a_reproduce_block(section):
    assert "## REPRODUCE" in section
    assert "scripts/22_head_selected_bestofn.py" in section
    assert "scripts/22_summarise_c27.py" in section
    assert "tests/test_head_selected_bestofn.py" in section
