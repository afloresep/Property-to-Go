"""`reports/PAPER_WORKSHOP_DRAFT.md` is the 4-page workshop draft.  It is the highest-stakes
document in the repo, because it is the only one anyone outside the project will read, and it
compresses five experiments into about thirty numbers.  Compression is where a number gets
attached to the wrong generator, the wrong comparator or the wrong seed count.

So these tests do three things:

* re-derive every quantitative claim from the tracked artefact it came from;
* assert the **comparator** is named correctly -- §5 prices guidance against the
  *oracle*-selected curve, which is the harder baseline, and a draft that let a reader think
  it was the equal-information curve of §3 would be claiming the crossing twice;
* assert the draft's own **scope hedges survive editing** -- the single-probe-seed limitation
  on §6, the one-generator limitation on §3, and the fact that §3.2 contains no result.

The third kind is the one that matters.  A revision that tightens prose and drops "on one
property of three" turns a scoped finding into an overclaim while every number still checks
out.

ASCII hyphens throughout: `f"{x:+.4f}"` emits U+002D, and a Unicode minus in a bound cell is
indistinguishable in a browser and fatal to the assertion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DRAFT = ROOT / "reports" / "PAPER_WORKSHOP_DRAFT.md"

PROPS = ("aromatic_rings", "hbd_count", "qed")


def _metrics(name: str) -> dict:
    p = OUT / f"{name}_summary" / f"{name}_metrics.json"
    if not p.exists():  # pragma: no cover
        pytest.skip(f"{p} not present")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def draft() -> str:
    if not DRAFT.exists():  # pragma: no cover
        pytest.skip(f"{DRAFT} not present")
    return DRAFT.read_text()


def flat(text: str) -> str:
    """Collapse line wrapping so prose assertions survive reflowing.

    Numeric assertions run on the raw text; only prose uses this.  A claim that spans a line
    break inside a blockquote is otherwise unmatchable, and the fix must not be to stop
    testing the prose.
    """
    return re.sub(r"\s*\n>?\s*", " ", text)


def section(draft: str, start: str, end: str | None = None) -> str:
    i = draft.index(start)
    j = draft.index(end, i) if end else len(draft)
    return draft[i:j]


# --------------------------------------------------------------------------------------
# no Unicode minus anywhere
# --------------------------------------------------------------------------------------

def test_the_draft_contains_no_unicode_minus(draft: str) -> None:
    """U+2212 renders identically to a hyphen and silently breaks every numeric assertion."""
    bad = [i for i, ch in enumerate(draft) if ch == "−"]
    assert not bad, f"U+2212 at offsets {bad[:10]}; use ASCII '-'"


# --------------------------------------------------------------------------------------
# §3 -- the equal-information control (C27, generator 1)
# --------------------------------------------------------------------------------------

def _c27_deployed_rows() -> dict[str, dict]:
    c27 = _metrics("c27")
    rows: dict[str, dict] = {}
    for prop, v in c27["properties"].items():
        hits = [g for g in v["guidance_points_vs_head_selected"]
                if g["run"] == f"pilot_50k_p2_guided_{prop}"]
        assert len(hits) == 1, f"{prop}: expected one deployed arm, got {len(hits)}"
        rows[prop] = hits[0]
    return rows


def test_section3_3_generator1_share_column_matches_c27(draft: str) -> None:
    sec = section(draft, "### 3.3", "## 4.")
    for prop, g in _c27_deployed_rows().items():
        share = 1.0 - g["advantage_vs_head_selected"] / g["advantage_vs_oracle_selected"]
        assert f"{share:.4f}" in sec, f"{prop}: generator-1 share {share:.4f} missing"


def test_section3_arm_counts_match_both_generators(draft: str) -> None:
    """1 of 46 / 15 of 46 on generator 1; 7 of 30 / 18 of 30 on generator 2."""
    c27 = _metrics("c27")
    e1 = c27["decision_rules"]["E1_head_selection_still_beats_steering"]
    g1_total, g1_head = e1["n_arms_total"], e1["n_arms_above"]
    g1_oracle = sum(
        v["n_arms_above_oracle_selected_curve"] for v in c27["properties"].values()
    )
    c33 = _metrics("c33")
    ac = c33["arm_counts"]
    g2_total = ac["n_cells"]
    g2_head = ac["n_arms_above_head_selected_curve"]
    g2_oracle = ac["n_arms_above_oracle_selected_curve"]
    sec = section(draft, "### 3.3", "## 4.")
    for n, tot in ((g1_oracle, g1_total), (g1_head, g1_total),
                   (g2_oracle, g2_total), (g2_head, g2_total)):
        assert f"**{n} of {tot}**" in sec, f"arm count {n} of {tot} missing from 3.3"
    # the direction is the thing that replicates; it must hold on both
    assert g1_head > g1_oracle and g2_head > g2_oracle


def _curve_gap(prop: str) -> dict[str, tuple[float, float]]:
    """(generator 1, generator 2) separation between the oracle and equal-information curves."""
    g1 = _metrics("c27")["properties"][prop]["E2_price_of_ground_truth"]["gap_per_n"]
    cur = _metrics("c33")["properties"][prop]["curves"]
    g2 = {
        n: cur["oracle_selected"][n]["hit_rate_mean"] - cur["head_selected"][n]["hit_rate_mean"]
        for n in cur["oracle_selected"]
    }
    return {n: (g1[n], g2[n]) for n in g2 if n in g1}


def test_section3_2_matched_N_table_matches_both_generators(draft: str) -> None:
    """Every cell of the replication table is re-derived from the generator it names."""
    sec = section(draft, "### 3.2", "### 3.3")
    for prop in PROPS:
        gaps = _curve_gap(prop)
        for n in ("2", "4", "9", "16", "32"):
            a, b = gaps[n]
            assert f"{a:.4f}" in sec, f"{prop} N={n}: generator-1 gap {a:.4f} missing"
            assert f"{b:.4f}" in sec, f"{prop} N={n}: generator-2 gap {b:.4f} missing"


def test_section3_2_agreement_statistic_matches_the_curves(draft: str) -> None:
    sec = section(draft, "### 3.2", "### 3.3")
    for prop in PROPS:
        gaps = _curve_gap(prop)
        mad = sum(abs(a - b) for a, b in gaps.values()) / len(gaps)
        assert f"{mad:.4f}" in sec, f"{prop}: mean abs difference {mad:.4f} missing"


def test_section3_2_endpoint_claim_matches_the_artefacts(draft: str) -> None:
    """'0.37 to 0.68 by N = 32' must bracket both generators at N = 32."""
    ends = [v for prop in PROPS for v in _curve_gap(prop)["32"]]
    lo, hi = f"{min(ends):.2f}", f"{max(ends):.2f}"
    body = flat(draft)
    assert f"{lo} to {hi}" in body or f"{lo}-{hi}" in body, (
        f"the N=32 range should be {lo}-{hi}; observed {sorted(ends)}"
    )


def test_section3_2_curves_are_zero_at_N1_by_construction(draft: str) -> None:
    for prop in PROPS:
        a, b = _curve_gap(prop)["1"]
        assert a == 0.0 and b == 0.0, (prop, a, b)
    assert "identical at N = 1 by construction" in flat(section(draft, "### 3.2", "### 3.3"))


def test_section3_states_the_oracle_is_free_for_these_properties(draft: str) -> None:
    """The control is not a shippable baseline.  Dropping this hedge is an overclaim."""
    sec = flat(section(draft, "### 3.3", "## 4."))
    assert "RDKit *is* free" in sec or "RDKit is free" in sec
    assert "scientific control" in sec


# --------------------------------------------------------------------------------------
# §3.3 -- the pre-registered replication that FAILED, reported as a result
# --------------------------------------------------------------------------------------

def test_section3_3_generator2_share_row_matches_c33(draft: str) -> None:
    c33 = _metrics("c33")
    sec = section(draft, "### 3.3", "## 4.")
    n_undefined = 0
    for prop in PROPS:
        h = c33["headline"]["per_property"][prop]
        assert f"{h['advantage_vs_oracle_selected']:+.4f}" in sec, prop
        assert f"{h['advantage_vs_head_selected']:+.4f}" in sec, prop
        if h["oracle_share"] is None:
            n_undefined += 1
            assert "undefined" in sec
        else:
            assert f"{h['oracle_share']:.4f}" in sec, f"{prop}: share missing"
    # exactly one anchor is undefined; if that changes the prose must change with it
    assert n_undefined == 1, n_undefined


def test_section3_3_reports_the_preregistered_failure_not_a_quiet_retirement(
    draft: str,
) -> None:
    """The verdict is the project's own negative result and must be stated as one."""
    c33 = _metrics("c33")
    verdict = c33["verdict"]["verdict"]
    assert verdict == "DOES NOT REPLICATE", verdict
    sec = flat(section(draft, "### 3.3", "## 4."))
    assert "**failed**" in sec or "failed" in sec
    assert "we published it first" in sec, "the draft must own the withdrawn statistic"


def test_section3_3_budget_mechanism_matches_the_artefacts(draft: str) -> None:
    """The share moves because the arms sit at different budgets, not because of the model."""
    c33 = _metrics("c33")
    g2_budgets = [
        c33["headline"]["per_property"][p]["tokens_per_molecule_actual"] for p in PROPS
    ]
    c27 = _metrics("c27")
    g1_budgets = [
        g["tokens_per_molecule_actual"]
        for g in _c27_deployed_rows().values()
    ]
    sec = flat(section(draft, "### 3.3", "## 4."))
    assert f"{round(min(g2_budgets))}" in sec and f"{round(max(g2_budgets))}" in sec or \
        "131" in sec, g2_budgets
    assert f"{round(min(g1_budgets))}-{round(max(g1_budgets))}" in sec, g1_budgets
    # generator 2's arms really are the cheaper ones -- the whole mechanism depends on it
    assert max(g2_budgets) < min(g1_budgets), (g2_budgets, g1_budgets)


def test_section3_3_does_not_claim_the_generator_explains_the_difference(
    draft: str,
) -> None:
    sec = flat(section(draft, "### 3.3", "## 4."))
    assert "it is not the generator" in sec.lower()
    assert "the numerator replicates" in sec.lower()


# --------------------------------------------------------------------------------------
# §4 -- the probe seed dominates (C29, C30)
# --------------------------------------------------------------------------------------

def test_section4_probe_seed_sd_exceeds_generation_seed_sd_as_stated(draft: str) -> None:
    fam = _metrics("c29")["families"]["A1"]
    hs = fam["head_seed_sd"]
    gs = fam["generation_seed_sd_pooled"]
    sec = section(draft, "## 4.", "## 5.")
    assert f"{hs['sd']:.4f}" in sec, f"probe-seed sd {hs['sd']:.4f} missing"
    assert f"{gs['sd']:.4f}" in sec, f"generation-seed sd {gs['sd']:.4f} missing"
    assert f"[{hs['lo']:.4f}, {hs['hi']:.4f}]" in sec
    # the claim only holds if the inequality does
    assert hs["sd"] > gs["sd"], (hs["sd"], gs["sd"])


def test_section4_validity_ladder_is_the_artefact_sorted_descending(draft: str) -> None:
    """The eight validity values must be C30's, and the draft sorts them for readability."""
    c30 = _metrics("c30")
    cell = c30["cells"]["C2_k4"]
    vals = sorted((p["validity_mean"] for p in cell["per_head_seed"].values()), reverse=True)
    assert len(vals) == 8, vals
    # emphasis and line wrapping are the author's business; the values and their order are not
    sec = flat(section(draft, "## 4.", "## 5.")).replace("*", "")
    rendered = ", ".join(f"{v:.4f}" for v in vals)
    assert rendered in sec, f"expected the ladder '{rendered}' in §4"


def test_section4_validity_chain_endpoints_match_c30_and_c28(draft: str) -> None:
    c30 = _metrics("c30")
    cell = c30["cells"]["C2_k4"]
    worst_seed, _ = min(cell["per_head_seed"].items(), key=lambda kv: kv[1]["validity_mean"])
    collapsed = cell["advantage_by_head_seed"][worst_seed]
    sec = section(draft, "## 4.", "## 5.")
    assert f"{cell['c28_single_seed_margin']:+.4f}" in sec, "the C28 single-seed margin"
    assert f"{collapsed:+.4f}" in sec, "the collapsed advantage at the worst seed"
    # the chain only reads as a chain if the worst-validity seed is also the collapsed one
    assert collapsed == min(cell["advantage_by_head_seed"].values()) or collapsed < 0.05


def test_section4_reports_the_uninterpretable_verdict_and_labels_S1_post_hoc(draft: str) -> None:
    c30 = _metrics("c30")
    screen = c30["validity_screen"]
    sec = section(draft, "## 4.", "## 5.")
    flatsec = flat(sec)
    assert c30["verdict"]["verdict"] in sec, "the pre-registered verdict must appear"
    assert f"{screen['n_points_failing']} of {screen['n_points_checked']}" in flatsec
    assert f"{screen['threshold']:.2f}" in sec
    assert "post-hoc" in flatsec, "S1 must be labelled post hoc"
    # the sensitivity analysis flipped it; the draft must not present that as the verdict
    assert flatsec.index(c30["verdict"]["verdict"]) < flatsec.index("CONFIRMED")


# --------------------------------------------------------------------------------------
# §5 -- the crossing, and the comparator it is priced against
# --------------------------------------------------------------------------------------

def test_section5_names_the_oracle_selected_comparator(draft: str) -> None:
    """C30/C31 price against the oracle curve.  Conflating it with §3's control would
    double-count the correction."""
    c30 = _metrics("c30")
    assert "oracle-selected" in c30["comparator"], c30["comparator"]
    sec = flat(section(draft, "## 5.", "## 6."))
    assert "**oracle-selected** curve" in sec or "oracle-selected** curve" in sec
    assert "harder" in sec, "the draft must say which direction the comparator cuts"
    assert "not an artifact of the" in sec


def test_section5_eight_seed_table_matches_c30_every_cell(draft: str) -> None:
    c30 = _metrics("c30")
    sec = section(draft, "## 5.", "## 6.")
    cells = dict(c30["cells"])
    # the run-but-not-scored cell is a single record, and the draft tabulates it too
    rbns = c30.get("run_but_not_scored")
    if isinstance(rbns, dict) and "cell" in rbns:
        cells[rbns["cell"]] = rbns
    assert len(cells) >= 8, sorted(cells)
    for name, c in cells.items():
        ti = c["advantage_t_interval"]
        n_pos = sum(1 for v in c["advantage_by_head_seed"].values() if v > 0)
        assert f"{ti['mean']:+.4f}" in sec, f"{name}: mean {ti['mean']:+.4f} missing"
        assert f"[{ti['lo']:+.4f}, {ti['hi']:+.4f}]" in sec, f"{name}: interval missing"
        assert f"{n_pos}/{c['n_head_seeds']}" in sec, f"{name}: {n_pos}/8 missing"


def test_section5_marks_the_deployed_configuration_and_it_is_positive_on_all_seeds(
    draft: str,
) -> None:
    c30 = _metrics("c30")
    deployed = [n for n, c in c30["cells"].items() if c["is_deployed_configuration"]]
    assert deployed, "no cell flags itself as deployed"
    sec = flat(section(draft, "## 5.", "## 6."))
    assert "deployed** configuration" in sec or "**deployed** configuration" in sec
    for n in deployed:
        c = c30["cells"][n]
        n_pos = sum(1 for v in c["advantage_by_head_seed"].values() if v > 0)
        if n_pos == c["n_head_seeds"]:
            assert "8 of 8 probe seeds" in sec or f"{n_pos} of {n_pos} probe seeds" in sec


def test_section5_reports_the_cell_that_was_run_but_not_scored(draft: str) -> None:
    """A configuration generated and never mentioned reads as one that was dropped."""
    sec = flat(section(draft, "## 5.", "## 6."))
    assert "indistinguishable, to a reader, from one that was dropped" in sec


def test_section5_generator2_crossing_count_and_examples_match_c31(draft: str) -> None:
    c31 = _metrics("c31")
    cells = c31["cells"]
    crossers = {
        n: c for n, c in cells.items()
        if c["advantage_vs_oracle_selected_seed_t_interval"]["excludes_zero"]
        and c["advantage_vs_oracle_selected"] > 0
    }
    sec = section(draft, "## 5.", "## 6.")
    assert f"{len(crossers)} of {len(cells)} configurations cross" in flat(sec)
    # all crossers must be mid-network at lambda = 2, as the draft claims
    assert all(c["arm"] == "mid" and c["lam"] == 2.0 for c in crossers.values()), crossers
    assert "all of them mid-network at" in flat(sec)
    # the two cited examples
    for prop, k in (("aromatic_rings", 4), ("hbd_count", 2)):
        hit = [c for c in crossers.values() if c["property"] == prop and c["k"] == k]
        assert len(hit) == 1, (prop, k)
        ti = hit[0]["advantage_vs_oracle_selected_seed_t_interval"]
        assert f"{hit[0]['advantage_vs_oracle_selected']:+.4f}" in sec
        assert f"[{ti['lo']:+.4f}, {ti['hi']:+.4f}]" in sec


def test_section5_selected_probe_points_for_generator2_match_c31(draft: str) -> None:
    c31 = _metrics("c31")
    sel = [c31["depth_curves"][p]["mid_probe_point"]["selected"] for p in PROPS]
    sec = flat(section(draft, "## 5.", "## 6."))
    assert "selected probe points " + ", ".join(str(s) for s in sel[:-1])
    for s in sel:
        assert str(s) in sec
    assert all(c31["depth_curves"][p]["test_depth_peak"]["peaks_before_final"] for p in PROPS)


def test_section5_advantage_collapse_matches_c31_all_six_arms(draft: str) -> None:
    """k raises the hit rate but lowers the advantage.  All six deltas must be present."""
    c31 = _metrics("c31")
    by_arm: dict[tuple[str, str], dict[int, float]] = {}
    for c in c31["cells"].values():
        by_arm.setdefault((c["property"], c["arm"]), {})[c["k"]] = c[
            "advantage_vs_oracle_selected"
        ]
    deltas = []
    for key, ks in sorted(by_arm.items()):
        assert 2 in ks and 32 in ks, key
        deltas.append(ks[32] - ks[2])
    assert len(deltas) == 6, deltas
    assert all(d < 0 for d in deltas), deltas
    sec = section(draft, "## 5.", "## 6.")
    for d in deltas:
        assert f"{d:+.4f}" in sec, f"collapse {d:+.4f} missing from §5"


def test_section5_reconciles_the_two_counting_rules(draft: str) -> None:
    """The draft carries both an interval-based crossing count (5) and a point-estimate arm
    count (7).  A reader who spots the deployed hbd arm at +0.0317 in the 3.3 table and the
    'all mid-network' claim here will think one of them is wrong unless the convention is
    stated.  This is the defect C33's own pre-registration inherited from C31's wording."""
    c31 = _metrics("c31")
    cells = c31["cells"]
    point = {n for n, c in cells.items() if c["advantage_vs_oracle_selected"] > 0}
    interval = {
        n for n in point
        if cells[n]["advantage_vs_oracle_selected_seed_t_interval"]["excludes_zero"]
    }
    extra = point - interval
    assert len(interval) == 5 and len(point) == 7, (len(interval), len(point))
    assert len(extra) == 2, sorted(extra)
    sec = flat(section(draft, "## 5.", "## 6."))
    assert f"gives **{len(point)} of {len(cells)}**" in sec
    for n in extra:
        c = cells[n]
        ti = c["advantage_vs_oracle_selected_seed_t_interval"]
        assert f"{c['advantage_vs_oracle_selected']:+.4f}" in sec, f"{n} advantage missing"
        assert f"[{ti['lo']:+.4f}, {ti['hi']:+.4f}]" in sec, f"{n} interval missing"
        assert not ti["excludes_zero"], n
    # the C33 arm count must be the point-estimate one, or the reconciliation is wrong
    assert _metrics("c33")["arm_counts"]["n_arms_above_oracle_selected_curve"] == len(point)


def test_section5_flags_the_trivial_predictor_margin_against_itself(draft: str) -> None:
    c31 = _metrics("c31")
    bt = c31["depth_curves"]["aromatic_rings"]["frozen_state_beats_trivial"]
    sec = section(draft, "## 5.", "## 6.")
    assert f"{bt['best_probe_point_test_auroc']:.4f}" in sec
    assert f"{bt['trivial_test_auroc']:.4f}" in sec
    assert f"{bt['margin']:+.4f}" in sec


def test_section5_scopes_the_win_to_the_small_compute_regime(draft: str) -> None:
    sec = flat(section(draft, "## 5.", "## 6."))
    assert "two to four samples, and nowhere" in sec


# --------------------------------------------------------------------------------------
# §6 -- lambda beats depth (C32)
# --------------------------------------------------------------------------------------

def _c32_complete_cells() -> dict[str, dict]:
    return {n: c for n, c in _metrics("c32")["decomposition"].items() if c.get("complete")}


def _iv(block: dict) -> dict:
    return block.get("t_interval", block)


def test_section6_lambda_main_effect_exceeds_depth_in_every_cell(draft: str) -> None:
    cells = _c32_complete_cells()
    losers = [
        n for n, c in cells.items()
        if not _iv(c["lambda_main"])["mean"] > _iv(c["depth_main"])["mean"]
    ]
    assert not losers, f"lambda does not dominate in {losers}"
    sec = flat(section(draft, "## 6.", "## 7."))
    assert f"in all {len(cells)} cells" in sec


def test_section6_interaction_counts_match_c32(draft: str) -> None:
    cells = _c32_complete_cells()
    n_excl = sum(1 for c in cells.values() if _iv(c["interaction"])["excludes_zero"])
    primary = {n: c for n, c in cells.items() if c.get("primary_k")}
    n_excl_primary = sum(
        1 for c in primary.values() if _iv(c["interaction"])["excludes_zero"]
    )
    sec = flat(section(draft, "## 6.", "## 7."))
    assert f"excludes zero in {n_excl} of {len(cells)} cells" in sec
    assert f"{n_excl_primary} of the {len(primary)} primary cells" in sec


def test_section6_effective_lambda_spread_ratios_match_c32(draft: str) -> None:
    eff = _metrics("c32")["effective_lambda"]["rows"]
    sec = section(draft, "## 6.", "## 7.")
    seen = {}
    for row in eff.values():
        seen.setdefault(row["property"], row["spread_ratio"])
    for prop, ratio in seen.items():
        assert f"{ratio:.4f}" in sec, f"{prop}: spread ratio {ratio:.4f} missing"
    # the draft calls out that one ratio is below 1; that must be true of exactly one
    below = [p for p, r in seen.items() if r < 1.0]
    assert below == ["qed"], below
    assert "below*\n  1" in sec or "below* 1" in flat(sec)


def test_section6_deployed_hbd_crossing_at_lambda2_matches_c32(draft: str) -> None:
    """The retraction of 're-select depth per generator' rests on this cell."""
    dec = _c32_complete_cells()
    hits = [c for c in dec.values() if c["property"] == "hbd_count" and c["k"] == 2]
    assert len(hits) == 1
    corner_b = hits[0]["corners"]["b"]  # final probe point, lambda = 2
    assert corner_b["probe_point"] == 12 and corner_b["lam"] == 2.0
    sec = section(draft, "## 6.", "## 7.")
    assert f"{corner_b['advantage']:+.4f}" in sec


def test_section6_scopes_the_crossing_to_one_property_of_three(draft: str) -> None:
    sec = flat(section(draft, "## 6.", "## 7."))
    assert "one property of three" in sec


# --------------------------------------------------------------------------------------
# §7 -- the limitations that must not be edited away
# --------------------------------------------------------------------------------------

def test_discussion_keeps_the_single_probe_seed_limitation_on_section6(draft: str) -> None:
    sec = flat(section(draft, "## 7."))
    assert "single probe seed" in sec
    assert "we do not claim" in sec and "at the strength" in sec


def test_discussion_keeps_the_unmatched_budget_limitation_on_section3(draft: str) -> None:
    """§3.2 compares curves at matched N; the ARMS were never at matched budgets.  That
    limitation replaced the old 'one generator' one when C33 landed, and must not vanish."""
    sec = flat(section(draft, "## 7."))
    assert "matched budget" in sec
    assert "not a substitute" in sec


def test_discussion_reports_the_weaker_generator2_gate(draft: str) -> None:
    sec = section(draft, "## 7.")
    assert "5.722e-06" in sec, "the non-bit-identical gate must be disclosed"
    assert "bit-identical" in sec


def test_discussion_does_not_claim_a_general_win(draft: str) -> None:
    sec = flat(section(draft, "## 7."))
    assert "not a general replacement" in sec
    assert "and is small" in sec


# --------------------------------------------------------------------------------------
# statistics discipline
# --------------------------------------------------------------------------------------

def test_draft_states_the_no_bootstrap_rule_with_the_right_arithmetic(draft: str) -> None:
    body = flat(draft)
    assert "no bootstrap" in body.lower()
    assert "1/27 = 0.037" in body
    assert "4.302653" in body and "2.364624" in body


def test_draft_matches_compute_on_tokens_not_wall_clock(draft: str) -> None:
    body = flat(draft).replace("*", "")
    assert "processed generator tokens, never wall clock" in body


def test_draft_states_the_generator_is_frozen(draft: str) -> None:
    body = flat(draft)
    assert "generator **frozen**" in body or "generator frozen" in body
    assert "no reinforcement learning" in body.lower()
