"""§25 of `reports/pilot_report.md` merges C33 -- the oracle asymmetry on generator 2.

C33 is the experiment that **falsified this report's own most-promoted methodological
claim**, so the tests here weight one direction over the other: re-deriving §25's numbers
matters, but what matters more is that the sentences C33 contradicts have actually been
retracted where they were written, rather than contradicted only in the new section.

That is the failure mode this file exists to catch. A merge that adds §25 and leaves
§22.1.1's "0.03-0.05 against an equal-information comparator" standing as a general claim,
or leaves §24.5 saying the asymmetry was "not re-measured", reads as current to anyone who
stops before §25 -- and §22 is what the paper's appendix pointer sends them to.

The C-number namespace split is also asserted here, because it is what makes the §22.11
ledger legible: claim-inventory IDs are `CL<n>` and experiment IDs are `C<n>`, and §22.11
is the one table in the report that carries both.

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
REPORT = ROOT / "reports" / "pilot_report.md"

PROPS = ("aromatic_rings", "hbd_count", "qed")
GRID = (1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32)


def flat(text: str) -> str:
    """Collapse the report's hard line wraps so a bound phrase can span a newline."""
    return re.sub(r"\s+", " ", text)


def _metrics(name: str) -> dict:
    p = OUT / f"{name}_summary" / f"{name}_metrics.json"
    if not p.exists():  # pragma: no cover
        pytest.skip(f"{p} not present")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def report() -> str:
    if not REPORT.exists():  # pragma: no cover
        pytest.skip("pilot_report.md not present")
    return REPORT.read_text()


@pytest.fixture(scope="module")
def section(report: str) -> str:
    marker = "## 25. Result 13"
    assert marker in report, "§25, the C33 merge, is missing from the report"
    return report[report.index(marker):]


@pytest.fixture(scope="module")
def c33() -> dict:
    return _metrics("c33")


@pytest.fixture(scope="module")
def c27() -> dict:
    return _metrics("c27")


# ------------------------------------------------------------------ the verdict itself


def test_section25_states_the_preregistered_verdict(section: str, c33: dict) -> None:
    assert c33["verdict"]["verdict"] == "DOES NOT REPLICATE"
    assert "DOES NOT REPLICATE" in section
    assert "F1" in section and "0.4162" in section


def test_section25_reports_both_falsified_predictions(section: str, c33: dict) -> None:
    """8 confirmed / 2 falsified.  The two falsified carry the replication claim."""
    assert sorted(c33["prediction_summary"]["falsified"]) == ["Q4", "Q5"]
    assert "8 confirmed, 2 falsified" in section
    assert "Q4" in section and "Q5" in section


def test_section25_does_not_hide_the_accounting_dependence(section: str, c33: dict) -> None:
    """Under S1 the verdict softens; §25 must say so rather than quoting the harsher one."""
    assert "S1" in section
    assert "REPLICATES IN DIRECTION, NOT IN MAGNITUDE" in flat(section)


def test_section25_blocking_gates_all_pass(section: str, c33: dict) -> None:
    assert c33["blocking_gates"]["all_pass"] is True
    for gate in ("G1", "G3a", "G6"):
        assert c33["blocking_gates"]["status"][gate] is True


# ------------------------------------------------------------------ the headline numbers


def test_section25_headline_advantages_match_the_artefact(section: str, c33: dict) -> None:
    for prop in PROPS:
        cell = c33["headline"]["per_property"][prop]
        for key in ("advantage_vs_oracle_selected", "advantage_vs_head_selected"):
            assert f"{cell[key]:+.4f}" in section, (prop, key, cell[key])


def test_section25_shares_match_the_artefact(section: str, c33: dict) -> None:
    computed = {}
    for prop in PROPS:
        cell = c33["headline"]["per_property"][prop]
        if cell["oracle_share_status"] == "computed":
            computed[prop] = cell["oracle_share"]
            assert f"{cell['oracle_share']:.4f}" in section, prop
    assert set(computed) == {"aromatic_rings", "qed"}, "exactly two anchors are computable"
    assert "undefined" in section, "hbd_count's share must be reported as undefined"


def test_section25_generator1_shares_are_labelled_as_generator_1(section: str) -> None:
    """C27's 0.8756/0.8819/0.8594 may appear only beside a generator-1 label."""
    for value in ("0.8756", "0.8819", "0.8594"):
        assert value in section, value
    assert "share g1" in section or "generator 1" in section


def test_section25_matched_n_curve_matches_both_artefacts(
    section: str, c27: dict, c33: dict
) -> None:
    for prop in PROPS:
        for tag, metrics in (("g1", c27), ("g2", c33)):
            curves = metrics["properties"][prop]["curves"]
            for n in GRID:
                gap = (curves["oracle_selected"][str(n)]["hit_rate_mean"]
                       - curves["head_selected"][str(n)]["hit_rate_mean"])
                assert f"{gap:.4f}" in section, (prop, tag, n, gap)


def test_section25_agreement_statistic_matches_the_curves(
    section: str, c27: dict, c33: dict
) -> None:
    for prop in PROPS:
        diffs = []
        for n in GRID:
            a = (c27["properties"][prop]["curves"]["oracle_selected"][str(n)]["hit_rate_mean"]
                 - c27["properties"][prop]["curves"]["head_selected"][str(n)]["hit_rate_mean"])
            b = (c33["properties"][prop]["curves"]["oracle_selected"][str(n)]["hit_rate_mean"]
                 - c33["properties"][prop]["curves"]["head_selected"][str(n)]["hit_rate_mean"])
            diffs.append(abs(a - b))
        mad = sum(diffs) / len(diffs)
        assert f"{mad:.4f}" in section, (prop, mad)


def test_section25_arm_counts_match_the_artefact(section: str, c33: dict) -> None:
    counts = c33["arm_counts"]
    assert counts["n_arms_above_head_selected_curve"] == 18
    assert counts["n_arms_above_oracle_selected_curve"] == 7
    assert "18" in section and "7" in section


def test_section25_budget_table_is_the_diagnosis(section: str, c27: dict, c33: dict) -> None:
    """§25.4's claim is that the two generators' arms sit at different budgets."""
    deployed_g1 = c27["decision_rules"]["E4_deployed_lambda1_arm_vs_head_selected_curve"]
    deployed_g2 = c33["headline"]["per_property"]
    for prop in PROPS:
        assert f"{deployed_g1[prop]['tokens_per_molecule_actual']:.2f}" in section, prop
        assert f"{deployed_g2[prop]['tokens_per_molecule_actual']:.2f}" in section, prop
        t1_g1 = c27["properties"][prop]["tokens_per_molecule_actual"][0]
        t1_g2 = c33["properties"][prop]["tokens_per_molecule_actual"][0]
        n_g1 = deployed_g1[prop]["tokens_per_molecule_actual"] / t1_g1
        n_g2 = deployed_g2[prop]["tokens_per_molecule_actual"] / t1_g2
        assert f"{n_g1:.2f}" in section and f"{n_g2:.2f}" in section, prop
        assert n_g1 > 2 * n_g2, prop


# ------------------------------------- the retractions, which are the point of the merge


def test_section_22_1_1_is_marked_withdrawn_not_merely_contradicted(report: str) -> None:
    """The 0.03-0.05 sentence must carry its retraction where it is written."""
    body = report[report.index("#### 22.1.1"):report.index("#### 22.1.2")]
    assert "0.03–0.05" in body or "0.03-0.05" in body, "the original wording should remain"
    assert "Withdrawn as a general statement" in body
    assert "§25" in body, "the retraction must point at where the replacement lives"
    for value in ("-0.1025", "+0.0980", "+0.0355"):
        assert value in body, f"the generator-2 counterexample {value} is missing"


def test_the_claim_ledger_row_is_superseded(report: str) -> None:
    ledger = report[report.index("### 22.11"):report.index("### 22.12")]
    row = next(line for line in ledger.splitlines() if line.startswith("| CL4 / CL22"))
    assert "Superseded by §25.6" in row
    assert "single-budget" in row


def test_section_24_5_no_longer_says_the_asymmetry_was_not_re_measured(report: str) -> None:
    ledger = report[report.index("### 24.5"):report.index("### 24.6")]
    row = next(line for line in ledger.splitlines() if "oracle asymmetry" in line)
    assert "not re-measured on the second generator" not in row
    assert "does not replicate as a ratio" in row


def test_section_24_7_item_3_is_closed(report: str) -> None:
    body = report[report.index("### 24.7"):report.index("### 24.8")]
    assert "Run as C33" in body


def test_the_eight_fold_claim_is_not_left_standing_unqualified(report: str) -> None:
    """§24.4's 'worth roughly 8x' is the sentence C33 falsified.  It must be scoped."""
    body = report[report.index("### 24.4"):report.index("### 24.5")]
    assert "8×" in body, "the original wording should remain, per the report's own policy"
    assert "Superseded" in body and "§25.7" in body


def test_section_25_7_replaces_the_ratio_with_the_curve(section: str) -> None:
    body = section[section.index("### 25.7"):section.index("### 25.8")]
    assert "0.37 to 0.68 hit-rate points" in flat(body)
    assert "a curve, not the single ~8× ratio" in flat(body)


# ------------------------------------------------------ the C-number namespace, split out


def test_claim_ids_and_experiment_ids_are_distinguishable(report: str) -> None:
    """§22.11 carries rows from both namespaces; they must not both be bare `C<n>`."""
    ledger = report[report.index("### 22.11"):report.index("### 22.12")]
    rows = [line for line in ledger.splitlines() if line.startswith("| C")]
    claim_rows = [r for r in rows if re.match(r"^\| CL\d+", r)]
    experiment_rows = [r for r in rows if re.match(r"^\| C\d+", r)]
    assert claim_rows, "the claim-inventory rows should be CL-prefixed"
    assert experiment_rows, "the experiment rows should stay bare C<n>"
    for claim in (4, 13, 30, 31, 32, 33, 34, 35, 36, 37):
        assert any(r.startswith(f"| CL{claim} ") or f"| CL{claim} /" in r
                   for r in claim_rows), claim


def test_the_namespace_split_is_documented_where_it_bites(report: str) -> None:
    ledger_note = report[report.index("### 22.11"):report.index("| claim | status")]
    assert "CL<n>` is a claim" in ledger_note
    assert "C<n>` is an experiment" in ledger_note


def test_no_bare_claim_id_survives_in_the_ledger(report: str) -> None:
    """C30-C33 were doubly booked; after the split a bare C3x row must be an experiment."""
    ledger = report[report.index("| claim | status after C23–C29"):report.index("### 22.12")]
    for line in ledger.splitlines():
        m = re.match(r"^\| C(\d+)", line)
        if m:
            assert m.group(1) in {"23", "25", "26"}, (
                f"bare C{m.group(1)} in the ledger is ambiguous; claims are CL-prefixed"
            )
