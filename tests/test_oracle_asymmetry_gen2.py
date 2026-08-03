"""C33 -- artefact-binding tests for the oracle-asymmetry replication on generator 2.

Every quantitative claim `reports/section_c33_oracle_asymmetry_gen2.md` makes is re-derived
here from the JSON it came from and required to appear in the section **in the exact format
the section prints it**, in the style of `tests/test_head_selected_bestofn.py` and
`tests/test_report_matches_artifacts.py`.

Several assertions are deliberately tripwires on the *data* rather than on the prose: if a
re-run stops reproducing C31's oracle curve exactly, if the head's pool AUROC collapses
towards chance, if the arm counts move, or if the verdict flips, a test fails rather than
leaving the section standing unchallenged.

Two structural invariants are also asserted, because the pre-registration's validity rests
on them:

  * the section carries the pre-registration block from `## C33.0.1` onward **byte-
    identically**, as `prereg_lock.json` promises; and
  * the pre-registration's mtime **strictly precedes** every C33 measurement artefact.

The section writes machine-derived numbers with ASCII hyphens.  A Unicode minus (U+2212)
would silently break every `in` assertion below, so one test forbids it outright in the
part of the section C33 wrote.
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
SECTION = ROOT / "reports" / "section_c33_oracle_asymmetry_gen2.md"
PREREG = OUT / "c33_prereg" / "C33.0_preregistration.md"
LOCK = OUT / "c33_prereg" / "prereg_lock.json"
SUMMARY = OUT / "c33_summary" / "c33_metrics.json"

ANCHORS = ("aromatic_rings", "hbd_count", "qed")
SEEDS = ("101", "202", "303")
ARMS = ("oracle_selected", "head_selected", "head_selected_at_75pct")
GRID = (1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32)
BLOCK_START = "## C33.0.1 Why this experiment exists"

# The deployed head, hard-coded rather than read from the artefact: a test that read the
# checkpoint name from the same file the summariser read it from would not catch the two
# agreeing wrongly.  C33.0.3 resolves it from all five C31 deployed k cells.
DEPLOYED_HEAD = {p: f"head_{p}_L12_seed1234.pt" for p in ANCHORS}
DEPLOYED_HEAD_SEED = 1234
DEPLOYED_PROBE_POINT = 12
#: C33.0.8's thresholds, restated here rather than imported.
F1_SHARE_FLOOR = 0.50
F2_SHARE_BAND = (0.75, 1.00)
F4_THRESHOLD = 0.02
G5_NEAR_CHANCE = 0.55
#: C27's generator-1 deployed-arm shares, which C33 must NOT silently drift towards.
C27_GEN1_SHARES = {"aromatic_rings": 0.8756, "hbd_count": 0.8819, "qed": 0.8594}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _have(*paths) -> bool:
    return all(p.exists() for p in paths)


requires_sweeps = pytest.mark.skipif(
    not _have(*(OUT / f"c33_headsel_{p}" / "head_selected_metrics.json" for p in ANCHORS)),
    reason="C33 sweep artefacts not present")
requires_summary = pytest.mark.skipif(not SUMMARY.exists(), reason="C33 summary not present")
requires_section = pytest.mark.skipif(not SECTION.exists(), reason="C33 section not present")
requires_prereg = pytest.mark.skipif(not _have(PREREG, LOCK), reason="C33 prereg not present")


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


@pytest.fixture(scope="module")
def sweeps():
    return {p: _load(OUT / f"c33_headsel_{p}" / "head_selected_metrics.json")
            for p in ANCHORS}


@pytest.fixture(scope="module")
def full_section():
    return SECTION.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prereg_block():
    body = PREREG.read_text(encoding="utf-8")
    return body[body.index(BLOCK_START):]


@pytest.fixture(scope="module")
def section(full_section, prereg_block):
    """The section WITHOUT the verbatim pre-registration copy.

    Every `assert_in` below binds a number C33 derived from JSON.  Checking against the
    whole file would let a number pass merely because the pre-registration quoted it --
    0.3532, 0.2472, 0.3715, 0.876, 0.882, 0.859, 0.0317, 0.1756 and 0.0809 all appear in
    C33.0.1 and C33.0.6 -- so the quoted block is removed before any binding assertion runs.
    """
    assert full_section.endswith(prereg_block)
    return full_section[:len(full_section) - len(prereg_block)]


def f1(x: float) -> str:
    return f"{x:.1f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def s4(x: float) -> str:
    return f"{x:+.4f}"


def assert_in(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C33 section"


# ============================================================ the pre-registration itself


@requires_prereg
def test_the_prereg_lock_records_the_prereg_hash_and_block():
    lock = _load(LOCK)
    assert lock["file_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert lock["file_bytes"] == PREREG.stat().st_size
    body = PREREG.read_text(encoding="utf-8")
    block = body[body.index(BLOCK_START):]
    assert lock["prereg_block_sha256"] == hashlib.sha256(block.encode("utf-8")).hexdigest()
    assert lock["prereg_block_chars"] == len(block)


@requires_prereg
@requires_sweeps
def test_the_prereg_strictly_precedes_every_c33_measurement_artefact():
    """Ordering by file mtime rather than by trust.  C33's whole claim to being
    pre-registered rests on this, so it is asserted rather than asserted-about."""
    t = PREREG.stat().st_mtime
    checked = 0
    for d in sorted(OUT.glob("c33_*")):
        if d.name == "c33_prereg":
            continue
        for p in ([d] if d.is_file() else sorted(d.rglob("*"))):
            if p.is_file():
                assert t < p.stat().st_mtime, f"{p} is not newer than the pre-registration"
                checked += 1
    assert checked > 0, "no C33 artefact was checked"


@requires_prereg
@requires_section
def test_the_section_copies_the_prereg_byte_identically(full_section, prereg_block):
    """C33.14 must be the pre-registered text, byte for byte, from C33.0.1 to end of file."""
    assert full_section.endswith(prereg_block), (
        "the verbatim pre-registration copy at the end of the C33 section is not "
        "byte-identical to outputs/c33_prereg/C33.0_preregistration.md")
    lock = _load(LOCK)
    i = full_section.rindex(BLOCK_START)
    assert hashlib.sha256(full_section[i:].encode("utf-8")).hexdigest() == \
        lock["prereg_block_sha256"]
    assert len(full_section[i:]) == lock["prereg_block_chars"]


@requires_section
def test_machine_derived_numbers_use_ascii_hyphens(section):
    """A Unicode minus would break every binding assertion in this file silently."""
    assert "−" not in section, "the section contains a Unicode minus (U+2212)"
    assert "–" not in section and "—" not in section


@requires_summary
def test_the_summary_records_the_prereg_lock(summary):
    assert summary["prereg"] == "outputs/c33_prereg/C33.0_preregistration.md"
    assert summary["prereg_file_sha256_matches_lock"] is True
    assert summary["prereg_lock"]["file_sha256"] == summary["prereg_file_sha256_now"]
    assert summary["artefact_mtime_ordering"][
        "prereg_strictly_precedes_every_artefact"] is True


# ========================================================================== the gates


@requires_summary
@requires_section
def test_g1_the_oracle_arm_reproduces_c31_exactly(summary, section):
    """G1 is blocking and is an identity, not an approximation."""
    g = summary["gates"]["G1_pool_identity"]
    assert g["max_abs_hit_rate_residual"] == 0.0
    assert g["max_abs_token_residual"] == 0.0
    assert g["passes"] is True
    for prop in ANCHORS:
        v = g["per_property"][prop]
        assert v["max_abs_hit_rate_residual"] == 0.0, prop
        assert v["max_abs_token_residual"] == 0.0, prop
        assert v["n_cells"] == len(GRID) * len(SEEDS), prop
    assert_in(section, "`0.0`", "G1 residual")
    assert_in(section, str(len(GRID) * len(SEEDS)), "G1 per-anchor cell count")
    assert_in(section, str(len(GRID) * len(SEEDS) * len(ANCHORS)), "G1 total cell count")


@requires_summary
@requires_sweeps
@requires_section
def test_g2_head_provenance_is_the_c31_deployed_head(summary, sweeps, section):
    g = summary["gates"]["G2_head_provenance"]
    assert g["passes"] is True
    for prop in ANCHORS:
        h = g["per_property"][prop]
        assert h["head_checkpoint_name"] == DEPLOYED_HEAD[prop]
        assert h["head_seed"] == DEPLOYED_HEAD_SEED
        assert h["probe_point"] == DEPLOYED_PROBE_POINT
        assert h["layer"] == h["probe_point"]
        assert h["head_input"] == "frozen_state"
        assert h["n_cells_agreeing"] == 5, "all five C31 deployed k cells must agree"
        assert h["metadata_consistent"] is True
        assert h["target_interval_matches_c31_cell"] is True
        assert h["in_dim"] == 768 and h["hidden_dim"] == 256
        assert h["head_file_sha256_stable"] is True
        assert h["head_parameter_sha256"] and len(h["head_parameter_sha256"]) == 64
        # the head is read off disk, never trained by C33
        assert sweeps[prop]["head"]["head_file"].endswith(DEPLOYED_HEAD[prop])
        assert_in(section, f"`{DEPLOYED_HEAD[prop]}`", f"{prop} checkpoint name")
        assert_in(section, h["head_file_sha256"][:16], f"{prop} file sha prefix")
        assert_in(section, h["head_parameter_sha256"][:16], f"{prop} parameter sha prefix")
        assert_in(section, str(h["head_file_bytes"]), f"{prop} head file bytes")
        assert_in(section, str(h["n_bins"]), f"{prop} n_bins")
        assert_in(section, h["binner_kind"], f"{prop} binner kind")


@requires_summary
@requires_section
def test_g2_scores_the_file_hash_criterion_as_evidence_not_as_a_gate(summary, section):
    """C27's FILE-hash gate fails by construction; C33 demonstrates it rather than
    asserting it, and does not use it as a pass/fail criterion."""
    g = summary["gates"]["G2_head_provenance"]
    assert g["file_hash_criterion_is_evidence_only"] is True
    demo = g["c27_file_hash_criterion_failed_by_construction"]
    assert demo["file_sha256_equal"] is False, (
        "the demonstration must show two saves of one dict differing in file bytes")
    assert demo["tensors_identical"] is True
    assert demo["parameter_sha256_a"] == demo["parameter_sha256_b"]
    assert demo["file_sha256_a"] != demo["file_sha256_b"]
    assert_in(section, demo["file_sha256_a"], "file hash demo, save A")
    assert_in(section, demo["file_sha256_b"], "file hash demo, save B")
    assert_in(section, demo["parameter_sha256_a"], "file hash demo, parameter hash")


@requires_summary
@requires_section
def test_g3_cost_identity(summary, section):
    g = summary["gates"]["G3_cost_identity"]
    assert g["a_max_abs_residual"] == 0.0
    assert g["a_passes"] is True
    assert g["b_max_abs_residual"] == 0.0
    assert g["b_max_abs_residual_vs_recomputed"] == 0.0
    assert g["b_max_cost_identity_residual"] == 0
    assert g["b_passes"] is True
    assert len(g["b_against_c31"]) == 30, "all 30 C31 k cells are re-priced"
    assert_in(section, "`0`", "G3(b) cost identity residual")
    assert_in(section, "30", "the 30 re-priced cells")


@requires_summary
def test_g4_n1_is_an_identity_across_arms_and_with_c31(summary):
    g = summary["gates"]["G4_n1_identity"]
    assert g["max_abs_residual"] == 0.0
    assert g["passes"] is True


@requires_summary
@requires_section
def test_g5_the_head_discriminates_on_every_anchor(summary, section):
    g = summary["gates"]["G5_head_discriminates"]
    assert g["any_near_chance"] is False
    for prop in ANCHORS:
        v = g["per_property"][prop]
        assert v["terminal_min"] >= G5_NEAR_CHANCE, prop
        assert v["near_chance"] is False, prop
        assert v["terminal_exceeds_75pct_on_every_seed"] is True, prop
        assert_in(section, f4(v["terminal_mean"]), f"{prop} terminal AUROC mean")
        assert_in(section, f4(v["at_75pct_mean"]), f"{prop} 75% AUROC mean")
        for s in SEEDS:
            assert_in(section, f4(v["per_seed"][s]["terminal"]),
                      f"{prop} seed {s} terminal AUROC")


@requires_summary
@requires_section
def test_g6_the_interval_is_c31s_frozen_one(summary, section):
    g = summary["gates"]["G6_frozen_interval"]
    assert g["passes"] is True
    assert g["target_intervals_sha256_stable"] is True
    assert g["windows_sha256_stable"] is True
    for prop in ANCHORS:
        assert g["per_property"][prop]["matches"] is True, prop
    assert_in(section, "G6", "G6 named in the section")


@requires_summary
@requires_section
def test_the_blocking_gates_pass_so_the_headline_may_be_stated(summary, section):
    b = summary["blocking_gates"]
    assert set(b["status"]) == {"G1", "G3a", "G6"}
    assert b["all_pass"] is True
    assert_in(section, "All three blocking gates pass", "blocking gate statement")


# ========================================================================= the curves


@requires_summary
@requires_section
def test_every_curve_row_is_printed_with_every_per_seed_value(summary, section):
    """C33.0.12 requires the two curves with EVERY per-seed value in full.  The whole
    markdown row is rebuilt from JSON, so a single moved digit fails."""
    for prop in ANCHORS:
        v = summary["properties"][prop]
        ps = v["per_seed_hit_rates"]
        assert tuple(v["grid"]) == GRID, prop
        for i, n in enumerate(v["grid"]):
            cells = [str(n), f1(v["tokens_per_molecule_actual"][i])]
            for arm in ARMS:
                cells.append(f4(v["curves"][arm][str(n)]["hit_rate_mean"]))
                cells.append(" / ".join(f4(ps[arm][s][i]) for s in SEEDS))
            cells.append(f4(v["budget_matched_gap_per_n"][str(n)]))
            row = "| " + " | ".join(cells) + " |"
            assert row in section, f"{prop} N={n} curve row not found:\n{row}"


@requires_sweeps
def test_the_arms_share_one_pool_so_n1_agrees_and_tokens_are_identical(sweeps):
    for prop in ANCHORS:
        sw = sweeps[prop]
        for s in SEEDS:
            base = sw["per_seed"][s]["arms"]["oracle_selected"]["1"]["hit_rate"]
            for arm in ARMS:
                assert sw["per_seed"][s]["arms"][arm]["1"]["hit_rate"] == base
        for n in GRID:
            tok = sw["curves"]["oracle_selected"][str(n)]["tokens_per_molecule_actual"]
            for arm in ARMS:
                assert sw["curves"][arm][str(n)]["tokens_per_molecule_actual"] == tok


@requires_sweeps
def test_the_head_arm_is_never_above_the_oracle_arm(sweeps):
    """The oracle arm selects with ground truth on the same groups, so it cannot be beaten
    on the true hit rate.  A violation would mean the arms are not selecting from the same
    pool -- a bug alarm, not a finding."""
    for prop in ANCHORS:
        sw = sweeps[prop]
        for n in GRID:
            o = sw["curves"]["oracle_selected"][str(n)]["hit_rate_mean"]
            for arm in ARMS[1:]:
                h = sw["curves"][arm][str(n)]["hit_rate_mean"]
                assert h <= o + 1e-12, f"{prop} {arm} N={n}: {h} > oracle {o}"


@requires_sweeps
def test_no_head_is_trained_and_the_generator_is_frozen(sweeps):
    for prop in ANCHORS:
        fp = sweeps[prop]["generator"]["fingerprint"]
        assert fp["all_parameters_frozen"] is True
        assert fp["training_mode"] is False
        assert fp["n_parameters"] == 87331584
        assert sweeps[prop]["generator"]["revision"] == \
            "f42a5a10e24c0350aeadb50865bd90a714d0b2bf"
        assert sweeps[prop]["head_scoring_token_charge"] == 0


# ============================================================= gaps, share, arm counts


@requires_summary
@requires_section
def test_the_budget_matched_gap_at_the_headline_budget(summary, section):
    for prop in ANCHORS:
        h = summary["headline"]["per_property"][prop]
        gap = h["budget_matched_gap_at_cell_budget"]
        assert gap > 0.0, f"{prop}: the oracle curve must sit above the head curve"
        assert_in(section, f4(gap), f"{prop} gap at the deployed k=2 budget")
        assert_in(section, f2(h["tokens_per_molecule_actual"]), f"{prop} k=2 budget")
        # Q9: smaller than C27's generator-1 gap
        assert gap < summary["c27_generator_1_reference"][prop]["gap"], prop


@requires_summary
@requires_section
def test_the_headline_oracle_share_table(summary, section):
    """The pre-registered headline: the deployed arm at k = 2, one cell per anchor."""
    hl = summary["headline"]["per_property"]
    assert set(hl) == set(ANCHORS)
    for prop in ANCHORS:
        h = hl[prop]
        assert h["cell"].endswith("_deployed_L12_lam1_k2")
        assert_in(section, s4(h["advantage_vs_oracle_selected"]), f"{prop} adv vs oracle")
        assert_in(section, s4(h["advantage_vs_head_selected"]), f"{prop} adv vs head")
        for key in ("advantage_vs_oracle_selected_per_seed",
                    "advantage_vs_head_selected_per_seed"):
            assert len(h[key]) == len(SEEDS)
            for x in h[key]:
                assert_in(section, s4(x), f"{prop} per-seed {key}")
        for key in ("advantage_vs_oracle_selected_seed_t_interval",
                    "advantage_vs_head_selected_seed_t_interval"):
            ti = h[key]
            assert ti["n_seeds"] == 3 and ti["note"] == "seed-level t interval, 2 df"
            assert_in(section, f"[{s4(ti['lo'])}, {s4(ti['hi'])}]", f"{prop} {key}")
        if h["oracle_share"] is None:
            assert h["advantage_vs_oracle_selected"] >= 0.0
            assert h["oracle_share_status"] == (
                "not defined (advantage vs oracle curve is not negative)")
            assert_in(section, "not defined (advantage vs oracle curve is not negative)",
                      f"{prop} undefined share")
        else:
            assert h["advantage_vs_oracle_selected"] < 0.0
            assert_in(section, f4(h["oracle_share"]), f"{prop} share")


@requires_summary
def test_the_share_is_never_computed_on_a_non_negative_advantage(summary):
    """C33.0.6 rule 1, asserted over all 30 cells and not only the headline."""
    for name, c in summary["cells"].items():
        if not c.get("priced"):
            continue
        if c["advantage_vs_oracle_selected"] < 0.0:
            assert c["oracle_share"] is not None, name
            assert c["oracle_share_status"] == "computed", name
        else:
            assert c["oracle_share"] is None, name
            assert c["oracle_share_status"] == (
                "not defined (advantage vs oracle curve is not negative)"), name


@requires_summary
def test_the_share_is_the_preregistered_formula_and_is_not_clipped(summary):
    for name, c in summary["cells"].items():
        if not c.get("priced") or c["oracle_share"] is None:
            continue
        expected = 1.0 - c["advantage_vs_head_selected"] / c["advantage_vs_oracle_selected"]
        assert abs(c["oracle_share"] - expected) < 1e-12, name
        if c["oracle_share"] > 1.0:
            assert c["advantage_vs_head_selected"] > 0.0, name


@requires_summary
def test_near_zero_negative_denominators_carry_the_unresolved_flag(summary):
    """C33.0.6 rule 3 is what stops a share of 20.18 being read as a result."""
    flagged = {c["dir"]: c for c in summary["cells"].values()
               if c.get("oracle_share_flag")}
    for name, c in flagged.items():
        assert c["oracle_share_flag"] == "not resolved at three generation seeds"
        assert c["advantage_vs_oracle_selected_seed_t_interval"]["excludes_zero"] is False
    # both extreme shares must be flagged rather than quoted bare
    for name, c in summary["cells"].items():
        if c.get("oracle_share") is not None and c["oracle_share"] > 2.5:
            assert c.get("oracle_share_flag") or c["advantage_vs_head_selected"] > 0.0, name


@requires_summary
@requires_section
def test_every_one_of_the_30_cells_is_printed_with_both_advantages(summary, section):
    cells = summary["cells"]
    assert len(cells) == 30
    for name, c in sorted(cells.items()):
        assert c["priced"] is True, name
        share = c["oracle_share"]
        if share is None:
            cell_txt = "--"
        elif c.get("oracle_share_flag"):
            cell_txt = f"{f4(share)} !"
        else:
            cell_txt = f4(share)
        row = ("| `" + c["dir"].replace("c31_ksweep_", "") + "` | "
               + f4(c["hit_rate_mean"]) + " | "
               + f2(c["tokens_per_molecule_actual"]) + " | "
               + s4(c["advantage_vs_oracle_selected"]) + " | "
               + s4(c["advantage_vs_head_selected"]) + " | "
               + f4(c["budget_matched_gap_at_cell_budget"]) + " | "
               + cell_txt + " |")
        assert row in section, f"{name}: row not found in the section:\n{row}"


@requires_summary
@requires_section
def test_the_extrapolated_cells_are_named(summary, section):
    extrap = sorted(c["dir"] for c in summary["cells"].values()
                    if c.get("priced") and c["extrapolated_beyond_grid"])
    assert len(extrap) == 4
    assert all(d.endswith("_k32") for d in extrap)
    assert_in(section, "Four cells are flagged `extrapolated_beyond_grid`", "extrap count")
    for d in extrap:
        assert_in(section, "`" + d.replace("c31_ksweep_", "") + "`", f"{d} named")


@requires_summary
@requires_section
def test_the_crossing_cells_are_tabulated_separately(summary, section):
    crossing = summary["headline"]["crossing_cells_reported_separately"]
    expected = {c["dir"] for c in summary["cells"].values()
                if c.get("priced") and not c["advantage_vs_oracle_selected"] < 0}
    assert set(crossing) == expected
    assert len(crossing) == 7
    for name, c in crossing.items():
        assert c["oracle_share_status"] == (
            "not defined (advantage vs oracle curve is not negative)")
        assert_in(section, "`" + name.replace("c31_ksweep_", "") + "`", f"{name} in section")
    assert_in(section, "Seven of 30 cells", "crossing cell count in words")


@requires_summary
@requires_section
def test_the_arm_counts_against_both_curves(summary, section):
    a = summary["arm_counts"]
    assert a["n_cells"] == 30
    assert a["n_arms_above_head_selected_curve"] == 18
    assert a["n_arms_above_oracle_selected_curve"] == 7
    assert a["n_arms_above_head_selected_curve"] > a["n_arms_above_oracle_selected_curve"]
    assert a["c27_generator_1"] == {"n_cells": 46,
                                    "n_arms_above_head_selected_curve": 15,
                                    "n_arms_above_oracle_selected_curve": 1}
    per = a["per_property"]
    assert sum(v["above_head_selected"] for v in per.values()) == 18
    assert sum(v["above_oracle_selected"] for v in per.values()) == 7
    for prop in ANCHORS:
        row = (f"| {prop} | {per[prop]['n_cells']} | {per[prop]['above_oracle_selected']} "
               f"| {per[prop]['above_head_selected']} |")
        assert row in section, f"{prop} arm-count row not found:\n{row}"
    assert_in(section, "18", "arm count above the head curve")
    assert_in(section, "15/46", "C27's arm count")


# ================================================================== the diagnostic arm


@requires_summary
@requires_section
def test_the_75pct_diagnostic_arm(summary, section):
    fired = []
    for prop in ANCHORS:
        f5 = summary["properties"][prop]["F5_diagnostic_arm_ordering"]
        if f5["fires"]:
            fired.append(prop)
        else:
            bad = [n for n, v in f5["per_n"].items() if not v["strictly_between"]]
            assert bad, prop
            for n in bad:
                assert_in(section, f"N={n}", f"{prop} F5 violation at N={n}")
    assert fired == ["aromatic_rings", "hbd_count"], (
        "F5 must fire on exactly aromatic_rings and hbd_count on this data")
    assert summary["decision_rules"]["F5"]["n_anchors_firing"] == 2


# ==================================================================== decision rules


@requires_summary
@requires_section
def test_decision_rules_scored_as_written(summary, section):
    dr = summary["decision_rules"]
    assert dr["F1"]["fires"] is False
    assert dr["F2"]["fires"] is False
    assert dr["F3"]["fires"] is True
    assert dr["F4"]["fires"] is True
    assert dr["F5"]["fires"] is False
    assert dr["F6"]["fires"] is True
    # F1 fails because of aromatic_rings, and only because of it
    shares = dr["F1"]["per_property"]
    assert set(shares) == {"aromatic_rings", "qed"}
    assert shares["aromatic_rings"] < F1_SHARE_FLOOR
    assert shares["qed"] >= F1_SHARE_FLOOR
    assert dr["F1"]["anchors_without_computable_share"] == ["hbd_count"]
    # F2 fails from both sides
    assert not (F2_SHARE_BAND[0] <= shares["aromatic_rings"] <= F2_SHARE_BAND[1])
    assert not (F2_SHARE_BAND[0] <= shares["qed"] <= F2_SHARE_BAND[1])
    for prop in ANCHORS:
        g = dr["F4"]["per_property"][prop]
        assert g["gain"] >= F4_THRESHOLD, prop
        assert_in(section, f4(g["gain"]), f"{prop} F4 gain")
    assert dr["F6"]["n_cells_not_resolved"] == 2
    assert dr["F6"]["deployed_k2_anchors_with_t_interval_spanning_zero"] == ["hbd_count"]
    for name in dr["F6"]["cells_not_resolved"]:
        assert_in(section, "`" + name.replace("c31_ksweep_", "") + "`", f"F6 cell {name}")


@requires_summary
@requires_section
def test_the_verdict_is_does_not_replicate_and_the_section_says_so(summary, section):
    v = summary["verdict"]
    assert v["verdict"] == "DOES NOT REPLICATE"
    assert v["n_anchors_with_computable_share"] == 2
    assert_in(section, "DOES NOT REPLICATE", "the verdict")
    assert_in(section, "Verdict: DOES NOT REPLICATE", "the verdict in the opening line")
    # C33 must not be quoting C27's shares as its own
    for prop, share in C27_GEN1_SHARES.items():
        own = summary["headline"]["per_property"][prop]["oracle_share"]
        if own is not None:
            assert abs(own - share) > 0.05, (
                f"{prop}: the C33 share has drifted onto C27's number")


# ======================================================================= predictions


@requires_summary
@requires_section
def test_predictions_scored_including_the_falsified_ones(summary, section):
    p = summary["predictions"]
    assert set(p) == {f"Q{i}" for i in range(1, 11)}
    s = summary["prediction_summary"]
    assert s["falsified"] == ["Q4", "Q5"]
    assert len(s["confirmed"]) == 8
    for name, v in p.items():
        assert v["outcome"] in ("CONFIRMED", "FALSIFIED")
        assert_in(section, f"| {name} |", f"prediction {name} row")
    assert p["Q1"]["outcome"] == "CONFIRMED"
    assert p["Q7"]["outcome"] == "CONFIRMED"     # hbd_count share not computable
    assert p["Q9"]["outcome"] == "CONFIRMED"     # the gap is smaller on generator 2
    assert p["Q10"]["outcome"] == "CONFIRMED"    # no C31 number moved
    assert p["Q10"]["detail"]["c31_untouched_since_prereg_freeze"] is True
    assert_in(section, "8 confirmed, 2 falsified", "prediction summary")


# ======================================================================== sensitivity


@requires_summary
@requires_section
def test_sensitivity_s1_is_reported_and_the_verdict_is_labelled_accounting_dependent(
        summary, section):
    s1 = summary["sensitivity_S1_pessimistic_accounting"]
    assert s1["total_n_arms_above_head_selected"] == 23
    for prop in ANCHORS:
        rec = s1[prop]["recompute_tokens_per_pool_molecule"]
        assert rec > 0
        assert_in(section, f4(rec), f"{prop} recompute tokens")
        share = s1[prop]["deployed_k2_oracle_share"]
        free = summary["headline"]["per_property"][prop]["oracle_share"]
        if free is None:
            assert share is None, prop
        else:
            # charging the recompute always helps the guided arm
            assert share > free, prop
            assert_in(section, f4(share), f"{prop} S1 share")
    # under S1 F1 would fire and F2 would still not
    s1_shares = [s1[p]["deployed_k2_oracle_share"] for p in ANCHORS
                 if s1[p]["deployed_k2_oracle_share"] is not None]
    assert all(x >= F1_SHARE_FLOOR for x in s1_shares)
    assert not all(F2_SHARE_BAND[0] <= x <= F2_SHARE_BAND[1] for x in s1_shares)
    assert_in(section, "23", "S1 arm count")
    assert_in(section, "survives only under the free accounting", "S1 labelling")


# ============================================================ the pre-registration defects


@requires_summary
@requires_section
def test_the_section_discloses_the_prereg_miscount_without_amending_it(
        summary, section, prereg_block):
    """C33.0.6 says '5 of its 30 cells already sit above the oracle-selected curve (all
    mid-network)'.  Seven do on the point-estimate criterion, and one is a DEPLOYED arm.
    The section must disclose that, and the pre-registration must be unamended."""
    assert "5 of its 30 cells" in prereg_block, "the pre-registration text has been altered"
    assert "already sit **above** the oracle-selected curve (all mid-network" in \
        prereg_block, "the pre-registration text has been altered"
    n = summary["arm_counts"]["n_arms_above_oracle_selected_curve"]
    assert n == 7
    above = summary["arm_counts"]["cells_above_oracle_selected_curve"]
    assert any("deployed" in c for c in above), (
        "at least one crossing cell is a deployed arm, contradicting the prereg's "
        "'all mid-network' parenthetical")
    assert "miscounts" in section or "conflates" in section


@requires_section
def test_the_section_discloses_the_runner_defects_it_fixed(section):
    assert "defects in the runner" in section
    assert "by_probe_point" in section


@requires_prereg
def test_the_preregistration_file_was_not_amended():
    """The single most important invariant in C33: the frozen document is unchanged."""
    lock = _load(LOCK)
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == lock["file_sha256"]
    assert PREREG.stat().st_size == lock["file_bytes"]
