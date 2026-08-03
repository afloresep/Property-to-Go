"""§22 of `reports/pilot_report.md` is the merge of C23-C29 into the main report.

Every other section of that report is bound to its own experiment's artefacts by its own
test module.  §22 quotes numbers from *seven* experiments at once, which is exactly the
situation in which a transcription error is easiest to make and hardest to notice -- the
0.3395/0.2988 mix-up corrected in `section_c24_generality.md` on 2026-08-03 was one, and it
survived a full review because nothing re-derived it.

So each number §22 prints is re-read here from the `outputs/*_summary/*.json` that produced
it and asserted to appear in the section, formatted exactly as the artefact formats it.
A re-run that moves a number fails a test rather than leaving §22 standing unchallenged.

Two deliberate choices:

* **ASCII only.**  `f"{x:+.4f}"` emits an ASCII hyphen.  §22 must too, in every cell these
  tests bind, or the assertion fails on a Unicode minus that looks identical in a browser.
* **Skip, do not fail, when an artefact is absent.**  The bulk `outputs/` directories are
  git-ignored as of 2026-08-03; a fresh clone has the summary JSONs but not every arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = ROOT / "reports" / "pilot_report.md"


def _load(name: str) -> dict:
    path = OUT / f"{name}_summary" / f"{name}_metrics.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def section() -> str:
    if not REPORT.exists():  # pragma: no cover
        pytest.skip("pilot_report.md not present")
    text = REPORT.read_text()
    marker = "## 22. What C23"
    assert marker in text, "§22, the C23-C29 merge, is missing from the report"
    return text[text.index(marker):]


def assert_in(section: str, value: str, what: str) -> None:
    assert value in section, f"§22 does not print {value} ({what})"


# --------------------------------------------------------------- the merge exists at all

def test_the_merge_section_is_reachable_from_the_top_of_the_report():
    """A merge nobody is pointed to is a merge nobody reads."""
    text = REPORT.read_text()
    head = text[:3000]
    assert "§22" in head, "the report's opening summary does not point at the merge"


def test_the_merge_names_every_section_it_merges(section):
    for name in ("C23", "C24", "C25", "C26", "C27", "C28", "C29"):
        assert f"section_c{name[1:]}" in section, f"{name}'s file is not cited in §22"


def test_the_merge_does_not_claim_the_earlier_sections_were_rewritten(section):
    """The project's rule is that superseded text stays and is contradicted, not deleted."""
    assert "Nothing in §§1–21 has been deleted" in section


# ------------------------------------------------------- 22.1.1  the oracle re-scoping

def test_the_oracle_and_head_selected_gaps_are_the_ones_c27_measured(section):
    c27 = _load("c27")
    e4 = c27["decision_rules"]["E4_deployed_lambda1_arm_vs_head_selected_curve"]
    for prop in ("aromatic_rings", "hbd_count", "qed"):
        cell = e4[prop]
        head = cell["advantage_vs_head_selected"]
        oracle = cell["advantage_vs_oracle_selected"]
        assert_in(section, f"{head:+.4f}", f"{prop} deployed gap vs head-selected best-of-N")
        assert_in(section, f"{oracle:+.4f}", f"{prop} deployed gap vs oracle best-of-N")
        # the fraction of the gap that was the oracle, to 3 dp as §22 prints it
        fraction = (oracle - head) / oracle
        assert_in(section, f"{fraction:.3f}", f"{prop} oracle share of the gap")


def test_the_arm_counts_above_each_curve_are_the_pre_registered_ones(section):
    c26, c27 = _load("c26"), _load("c27")
    d1 = c26["decision_rules"]["D1_best_of_n_dominates_everywhere"]
    e1 = c27["decision_rules"]["E1_head_selection_still_beats_steering"]
    n_oracle = len(d1["violations"])
    n_head = e1["n_arms_above"]
    total = e1["n_arms_total"]
    assert_in(section, f"**{n_oracle} of {total}**", "arms above the oracle curve")
    assert_in(section, f"**{n_head} of {total}**", "arms above the head-selected curve")
    assert n_oracle < n_head, "the oracle comparator must be the harder one"


def test_none_of_the_arms_above_the_head_selected_curve_is_a_deployed_one(section):
    """§22.1.1 says so; if a deployed arm ever crosses, the sentence must change."""
    e1 = _load("c27")["decision_rules"]["E1_head_selection_still_beats_steering"]
    families = {v["family"] for v in e1["violations"]}
    assert "deployed_lambda1" not in families, (
        "a deployed arm now sits above the head-selected curve; §22.1.1's "
        "'None of the 15 is a deployed configuration' is stale"
    )
    assert "None of the 15" in section or "none of the" in section.lower()


# ------------------------------------------------------- 22.1.2  the withdrawn no-knob claim

def test_the_k_sweep_cost_span_is_quoted_and_refutes_the_no_knob_claim(section):
    d1 = _load("c28")["decision_rules"]["D1_cost_band"]["per_strand"]
    ratios = [s["token_ratio_max_over_min"] for s in d1.values()]
    assert_in(section, f"{min(ratios):.2f}", "smallest within-strand cost ratio")
    assert_in(section, f"{max(ratios):.2f}", "largest within-strand cost ratio")
    assert all(s["exceeds_threshold"] for s in d1.values()), (
        "no strand exceeds C28's own refutation threshold; §22.1.2 must be rewritten"
    )
    assert "withdrawn" in section.lower(), "§22 must say the no-knob claim is withdrawn"


def test_the_knob_buys_nothing_and_the_number_spans_the_whole_k_grid(section):
    """§22.1.2's -0.0218 is the move across the FULL k span, k = 2 to k = 32.

    Not k = 8 to k = 32, which is C28's D2 verdict statistic (+0.0078, NULL) and a
    different claim.  The cost figures §22 quotes beside it -- 140.83 to 1447.42 tokens per
    molecule -- are the k = 2 and k = 32 endpoints, so the accuracy figure must be too.
    """
    c28 = _load("c28")
    by_k = c28["decision_rules"]["D5_the_defence"]["per_strand"]["A1"]["hit_rate_by_k"]
    grid = sorted(by_k, key=int)
    move = by_k[grid[-1]] - by_k[grid[0]]
    assert_in(section, f"{move:+.4f}", "deployed hbd arm across the whole k span")
    assert move < 0, "the k knob now helps; §22.1.2's 'buys nothing' is stale"

    band = c28["decision_rules"]["D1_cost_band"]["per_strand"]["A1"]
    assert_in(section, f"{band['tokens_min']:.2f}", "cheapest k budget")
    assert_in(section, f"{band['tokens_max']:.2f}", "dearest k budget")

    best_k = c28["decision_rules"]["D2_does_the_knob_buy_accuracy"]["per_strand"]["A1"]
    assert best_k["best_k_by_hit_rate"] == int(grid[0]), (
        "the cheapest k is no longer the best; §22.1.2's parenthetical is stale"
    )


# ------------------------------------------------------- 22.2  the cheap-end crossing

def test_the_count_of_cells_above_the_oracle_curve_is_re_derived_not_recalled(section):
    """The positive result in the whole merge.  It is re-counted from the artefact."""
    strands = _load("c28")["strands"]
    above, excluding_zero, total = [], 0, 0
    for name, strand in strands.items():
        for k, cell in (strand.get("cells") or {}).items():
            total += 1
            adv = cell.get("advantage_vs_oracle_selected")
            if adv is not None and adv > 0:
                above.append((name, k, adv, cell))
                ti = cell.get("advantage_vs_oracle_selected_seed_t_interval") or {}
                if ti.get("excludes_zero"):
                    excluding_zero += 1
    assert_in(section, f"**{len(above)} of {total}", "cells above the oracle curve")
    assert_in(section, str(excluding_zero), "cells whose t interval excludes zero")
    for _, _, adv, _ in above:
        assert_in(section, f"{adv:+.4f}", "a cell advantage over the oracle curve")


def test_no_cell_above_the_oracle_curve_is_extrapolated_or_degenerate(section):
    """§22.2 claims both.  Neither is an editorial judgement; both are in the artefact."""
    strands = _load("c28")["strands"]
    validities = []
    for strand in strands.values():
        for cell in (strand.get("cells") or {}).values():
            adv = cell.get("advantage_vs_oracle_selected")
            if adv is not None and adv > 0:
                assert not cell.get("extrapolated_beyond_grid"), (
                    "a winning cell is extrapolated beyond the measured grid; "
                    "§22.2's 'none extrapolated' is false"
                )
                validities.append(cell["validity_mean"])
    assert validities, "no winning cells found -- §22.2 has nothing to bind"
    assert_in(section, f"{min(validities):.3f}", "lowest validity among winning cells")
    assert min(validities) > 0.9, "a winning cell is degenerate; §22.2 must say so"


# ------------------------------------------------------- 22.3  the retracted multiplier

def test_the_25x_multiplier_is_retracted_and_not_merely_softened(section):
    assert "retracted" in section.lower(), "§22.3 must use the word"
    c29 = _load("c29")
    r2 = c29["decision_rules"]["R2"]
    assert_in(section, f"{r2['c25_reported_span_ratio']:.2f}", "C25's withdrawn figure")
    assert not r2["c25_span_ratio_inside_interval"], (
        "C25's 24.76 is now inside C29's interval; the retraction would need revisiting"
    )
    assert not r2["fires"], "R2 fires; §22.3's 'every interval contains 1' is stale"


def test_every_head_seed_ratio_interval_contains_one(section):
    """The claim §22.3 rests on.  Seven families, all of them."""
    families = _load("c29")["families"]
    checked = 0
    for name, fam in families.items():
        vr = fam["variance_ratio"]
        lo, hi = vr["lo"], vr["hi"]
        checked += 1
        assert lo <= 1.0 <= hi, f"{name}'s ratio interval [{lo}, {hi}] excludes 1"
        # the three named arms in §22.3's table print their ratio to 2 dp
        if name in ("A1", "A2", "A3"):
            assert_in(section, f"{vr['ratio']:.2f}", f"{name}'s variance ratio")
            assert_in(section, f"{lo:.2f}", f"{name}'s interval lower end")
            assert_in(section, f"{hi:.2f}", f"{name}'s interval upper end")
    assert checked == 7, f"expected seven families, found {checked}"
    assert "all contain 1" in section or "contains 1" in section


def test_the_depth_test_that_makes_this_about_probes_in_general_fires(section):
    r3 = _load("c29")["decision_rules"]["R3"]
    assert r3["fires"], "R3 no longer fires; §22.3's 'learned probes generally' is stale"
    for anchor in r3["per_anchor"].values():
        assert_in(section, f"{anchor['q']:.2f}", "a per-anchor depth ratio q")
    assert r3["conclusion"] == "probes in general"


# ------------------------------------------------------- 22.4  Rule A, halved

def test_rule_a_survives_at_eight_head_seeds_with_the_intervals_printed(section):
    r4 = _load("c29")["decision_rules"]["R4"]
    assert r4["fires"], "R4 no longer fires; §22.4 claims Rule A survives"
    for arm in r4["per_arm"].values():
        assert_in(section, f"{arm['mean']:+.4f}", "a paired mid-minus-deployed mean")
        lo, hi = arm["ci"]
        assert_in(section, f"{lo:+.4f}", "its interval's lower end")
        assert_in(section, f"{hi:+.4f}", "its interval's upper end")
        assert lo > 0, "an interval no longer excludes zero; §22.4 overstates Rule A"


def test_the_effective_lambda_correction_is_stated_as_a_halving_not_a_refutation(section):
    """§22.4's load-bearing pair: 15/15 becomes 10/15, and Rule A still survives."""
    assert "15 of 15" in section and "10 of 15" in section
    r5 = _load("c29")["decision_rules"]["R5"]
    surviving = [a for a in r5["per_arm"].values() if a.get("t", {}).get("excludes_zero")]
    assert len(surviving) >= 2, (
        "fewer than two anchors survive the effective-lambda correction; §22.4's "
        "'Rule A nevertheless survives' would be false"
    )


# ------------------------------------------------------- 22.5  Rule B, unresolved

def test_rule_b_is_reported_as_comparator_dependent_and_not_as_a_win(section):
    r6 = _load("c29")["decision_rules"]["R6"]
    assert_in(section, f"{r6['mean']:+.4f}", "Rule B advantage against C23's comparator")
    for harder in ("also_token_conservative", "also_c26_corrected_curve"):
        cell = r6[harder]
        assert_in(section, f"{cell['mean']:+.4f}", f"Rule B under {harder}")
        assert not cell["fires"], f"{harder} now fires; §22.5 must be rewritten"
    assert not r6["survives_every_comparator"]
    assert "unresolved" in section.lower(), (
        "§22.5 must not present a comparator-dependent result as settled"
    )


# ------------------------------------------------------- 22.9  the withdrawn bootstrap

def test_the_section_states_why_a_three_seed_percentile_bootstrap_is_vacuous(section):
    """Not a number -- a method claim the project now enforces everywhere."""
    assert "1/27" in section or "0.0370" in section
    assert "4.302653" in section, "the t critical value on 2 df must be quoted"
    assert "[min, max]" in section


def test_no_three_seed_percentile_bootstrap_survives_in_any_summary_artefact():
    """The withdrawal is only real if the artefacts stopped emitting it."""
    for name in ("c23", "c24", "c26"):
        path = OUT / f"{name}_summary"
        if not path.exists():  # pragma: no cover
            continue
        for f in path.glob("*.json"):
            blob = f.read_text()
            assert "advantage_seed_bootstrap_ci" not in blob, (
                f"{f} still emits the withdrawn three-seed percentile bootstrap"
            )


# ------------------------------------------------------- 22.11  the ledger

def test_the_ledger_marks_the_two_withdrawn_claims(section):
    ledger = section[section.index("22.11"):] if "22.11" in section else section
    assert "withdrawn" in ledger.lower()
    assert "retracted" in ledger.lower()


def test_the_three_sentence_summary_does_not_force_a_positive_conclusion(section):
    """A standing instruction on this project.  The closing paragraph must carry the loss."""
    closing = section[section.index("22.12"):] if "22.12" in section else ""
    assert closing, "§22.12 is missing"
    assert "cannot follow it upward" in closing or "cannot follow" in closing, (
        "the closing summary states the win without the matching limitation"
    )
