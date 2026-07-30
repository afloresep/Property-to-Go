"""The report's numbers must match the JSON artefacts they claim to come from.

Every table in `reports/pilot_report.md` was transcribed by hand from files under
`outputs/`.  Hand-transcription is the one error mode that no amount of reasoning
about the pipeline can rule out, so it is checked mechanically here: each value is
re-read from its artefact, formatted the way the report formats it, and required to
appear somewhere in the report text.

These tests skip when the artefacts are absent, so a fresh clone still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = ROOT / "reports" / "pilot_report.md"


@pytest.fixture(scope="module")
def report_text() -> str:
    if not REPORT.exists():
        pytest.skip("report not written yet")
    return REPORT.read_text()


def _load(rel: str):
    p = OUT / rel
    if not p.exists():
        pytest.skip(f"{rel} not produced yet")
    return json.loads(p.read_text())


def _assert_present(report: str, value: float, fmt: str, label: str) -> None:
    rendered = fmt % value
    assert rendered in report, f"{label}: report is missing {rendered}"


@pytest.mark.parametrize("prop", ["clogp", "aromatic_rings", "mol_weight"])
def test_head_metrics_match(report_text, prop):
    heads = _load("pilot_50k_heads/head_metrics.json")["properties"][prop]["heads"]
    for name in ("frozen_state", "trivial"):
        test = heads[name]["test"]
        _assert_present(report_text, test["nll"], "%.3f", f"{prop}/{name} nll")
        _assert_present(
            report_text, test["intervals"]["target"]["auroc"], "%.3f", f"{prop}/{name} auroc"
        )


@pytest.mark.parametrize("prop", ["clogp", "aromatic_rings"])
def test_guided_hit_rates_match(report_text, prop):
    conds = _load(f"pilot_50k_guided_{prop}/guidance_metrics.json")["conditions"]
    for cond, v in conds.items():
        _assert_present(
            report_text, v["aggregate"]["hit_rate"]["mean"], "%.4f", f"{prop}/{cond} hit rate"
        )


@pytest.mark.parametrize("prop", ["clogp", "aromatic_rings"])
def test_best_of_n_and_advantage_match(report_text, prop):
    matches = _load(f"pilot_50k_bestofn_{prop}/bestofn_metrics.json")["matches"]
    for accounting, m in matches.items():
        _assert_present(
            report_text, m["aggregate"]["hit_rate"]["mean"], "%.4f", f"{prop}/{accounting} best-of-N"
        )
        _assert_present(
            report_text,
            m["comparison_vs_guided_throughout"]["guidance_advantage"],
            "%.4f",
            f"{prop}/{accounting} advantage",
        )


def test_guidance_always_loses_to_compute_matched_best_of_n():
    """The pilot's headline negative result, asserted rather than only narrated."""
    for prop in ("clogp", "aromatic_rings"):
        matches = _load(f"pilot_50k_bestofn_{prop}/bestofn_metrics.json")["matches"]
        for accounting, m in matches.items():
            adv = m["comparison_vs_guided_throughout"]["guidance_advantage"]
            assert adv < 0, f"{prop}/{accounting}: report says guidance loses, but advantage={adv}"


def test_confound_deltas_match(report_text):
    for prop in ("clogp", "aromatic_rings"):
        d = _load(f"pilot_50k_confound_{prop}/confound_metrics.json")
        ref = d["conditions"]["unguided"]["raw"]["hit_rate"]
        for cond, e in d["conditions"].items():
            if cond == "unguided":
                continue
            for kind in ("raw", "length", "size", "joint"):
                _assert_present(
                    report_text, e[kind]["hit_rate"] - ref, "%+.4f", f"{prop}/{cond}/{kind}"
                )


def test_effect_survives_joint_length_and_size_matching():
    """Rejection criterion R2: the report claims it does not fire."""
    for prop, floor in (("clogp", 0.90), ("aromatic_rings", 0.85)):
        d = _load(f"pilot_50k_confound_{prop}/confound_metrics.json")["conditions"]
        ref = d["unguided"]["raw"]["hit_rate"]
        raw = d["throughout"]["raw"]["hit_rate"] - ref
        joint = d["throughout"]["joint"]["hit_rate"] - ref
        assert joint / raw > floor, f"{prop}: only {joint / raw:.0%} of the effect survives matching"
        assert d["throughout"]["joint"]["coverage"] > 0.95


def test_rollout_quartile_correlations_match(report_text):
    props = _load("pilot_50k_rollouts/rollout_metrics.json")["properties"]
    for prop, v in props.items():
        for name in ("frozen_state", "trivial"):
            for q, stats in v["heads"][name]["by_quartile"].items():
                _assert_present(
                    report_text,
                    stats["spearman_vs_empirical_mean"],
                    "%.3f",
                    f"rollout {prop}/{name}/Q{q}",
                )


def test_the_aromatic_ring_crossover_is_real():
    """The pilot's most-defended claim, asserted directly against the rollout bank."""
    v = _load("pilot_50k_rollouts/rollout_metrics.json")["properties"]["aromatic_rings"]["heads"]
    frozen = v["frozen_state"]["by_quartile"]
    trivial = v["trivial"]["by_quartile"]
    key = "spearman_vs_empirical_mean"
    assert frozen["2"][key] > trivial["2"][key], "frozen state should lead early"
    assert trivial["4"][key] > frozen["4"][key], "token counting should lead late"
    assert frozen["4"][key] < frozen["2"][key], "the frozen-state curve should decline late"


@pytest.mark.parametrize("prop", ["clogp", "aromatic_rings"])
def test_quality_panel_numbers_match(report_text, prop):
    """Section 10's tables, re-read from the quality artefacts."""
    panels = _load(f"pilot_50k_quality_{prop}/quality_metrics.json")["panels"]
    for cond, p in panels.items():
        hits = p["hits"]
        if not hits.get("n"):
            continue
        for key in ("sa_score", "qed", "longest_chain", "max_ring_size"):
            _assert_present(
                report_text, hits["descriptors"][key]["mean"], "%.3f", f"{prop}/{cond}/{key}"
            )
        _assert_present(
            report_text, hits["degeneracy_rate"]["any"], "%.3f", f"{prop}/{cond}/degeneracy"
        )


def test_late_ring_guidance_is_the_only_condition_that_degrades_quality():
    """Section 10.2's claim, asserted against the artefact rather than narrated.

    Two independent descriptors must both exclude zero for `late`, and no other
    condition may show a significant SA increase -- that conjunction is the claim.
    """
    v = _load("pilot_50k_quality_aromatic_rings/quality_metrics.json")["vs_unguided_hits"]
    for key in ("sa_score", "longest_chain"):
        d = v["late"][key]
        assert d["difference"] > 0 and d["excludes_zero"], (
            f"report says late ring guidance degrades {key}, artefact says {d}"
        )
    for cond in ("throughout", "early", "middle"):
        d = v[cond]["sa_score"]
        assert not (d["difference"] > 0 and d["excludes_zero"]), (
            f"report says only `late` degrades SA, but {cond} does too: {d}"
        )


def test_guidance_does_not_reduce_drug_likeness():
    """Section 10.3: the expected degeneracy does not appear at lambda = 1."""
    for prop in ("clogp", "aromatic_rings"):
        v = _load(f"pilot_50k_quality_{prop}/quality_metrics.json")["vs_unguided_hits"]
        d = v["throughout"]["qed"]
        assert not (d["difference"] < 0 and d["excludes_zero"]), (
            f"{prop}: report claims QED is not harmed, artefact says {d}"
        )
