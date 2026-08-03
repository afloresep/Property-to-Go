"""C28 -- artefact-binding tests for the top-k sweep and the guided-drafts composition.

Every number `reports/section_c28_k_sweep.md` prints is re-derived here from JSON and
required to appear in the section text **in the exact format the section prints it**, in the
style of `tests/test_head_selected_bestofn.py` and `tests/test_n_sweep.py`.

Several are deliberately tripwires on the *data* rather than on the prose: if a re-run breaks
the k = 8 identity against the published deployed artefact, if the cost identity
`processed_tokens_actual mod (k + 1) == 0` stops holding, if the guided-draft pool stops being
the deployed sampler's own pool, if the k profile stops being flat, or if strand A1 crosses
either best-of-N curve -- a test fails rather than leaving the section standing unchallenged.

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
SECTION = ROOT / "reports" / "section_c28_k_sweep.md"
PREREG = OUT / "c28_prereg" / "C28.0_preregistration.md"
LOCK = OUT / "c28_prereg" / "prereg_lock.json"
SUMMARY = OUT / "c28_summary" / "c28_metrics.json"
DRAFTS = OUT / "c28_guided_drafts_hbd_count" / "guided_drafts_metrics.json"

SEEDS = ("101", "202", "303")
K_GRID = (2, 4, 8, 16, 32)
STRANDS = ("A1", "A2", "A3", "C1", "C2", "C3")
GATE_STRANDS = ("A1", "A2", "A3")
DRAFT_GRID = (1, 2, 4, 8)

#: The published deployed hbd_count run, hard-coded rather than read from the artefact the
#: summariser read it from: two files agreeing wrongly must not pass.
DEPLOYED_HBD_HIT_RATE = 0.29875114714835704
DEPLOYED_HBD_TOKENS = 401.619140625
DEPLOYED_BASE_TOKENS = 43.448567708333336
#: C26 section C26.4.4's claim that C28 tests.
C26_BAND_RATIO = 1.17


def _load(path):
    with open(path) as fh:
        return json.load(fh)


requires_summary = pytest.mark.skipif(not SUMMARY.exists(), reason="C28 summary not present")
requires_section = pytest.mark.skipif(not SECTION.exists(), reason="C28 section not present")
requires_drafts = pytest.mark.skipif(not DRAFTS.exists(), reason="C28 strand B not present")


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


@pytest.fixture(scope="module")
def drafts():
    return _load(DRAFTS)


@pytest.fixture(scope="module")
def full_section():
    return SECTION.read_text()


def _outside_prereg(section: str) -> str:
    """The section with the verbatim pre-registration copy removed.

    The pre-registration quotes several numbers it predicted (0.9271, 0.4855, 0.5603, ...).
    Binding an assertion against the whole file would let a measured number pass merely
    because the frozen prediction happened to contain it.
    """
    a = section.index("<!-- BEGIN VERBATIM PREREG COPY -->")
    b = section.index("<!-- END VERBATIM PREREG COPY -->")
    return section[:a] + section[b:]


@pytest.fixture(scope="module")
def section(full_section):
    return _outside_prereg(full_section)


def f4(x: float) -> str:
    return f"{x:.4f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def assert_in(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C28 section"


# ------------------------------------------------------------------ pre-registration


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
def test_the_prereg_was_written_before_every_measurement():
    """Ordering by file mtime rather than by trust."""
    t = PREREG.stat().st_mtime
    for d in sorted(OUT.glob("c28_*")):
        if d.name in ("c28_prereg", "c28_logs"):
            continue
        targets = [d] if d.is_file() else list(d.rglob("*"))
        for p in targets:
            if p.is_file():
                assert t < p.stat().st_mtime, f"{p} is not newer than the pre-registration"


@pytest.mark.skipif(not (PREREG.exists() and LOCK.exists()), reason="prereg not present")
def test_the_prereg_lock_records_the_prereg_hash():
    lock = _load(LOCK)
    assert lock["file_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    body = PREREG.read_text()
    block = body[body.index("## C28.0.1 Why"):]
    assert lock["prereg_block_sha256"] == hashlib.sha256(block.encode()).hexdigest()
    assert lock["prereg_block_chars"] == len(block)


@pytest.mark.skipif(not PREREG.exists(), reason="prereg not present")
@requires_section
def test_the_report_copies_the_prereg_verbatim(full_section):
    prereg = PREREG.read_text()
    body = prereg[prereg.index("## C28.0.1 Why"):].rstrip()
    assert body in full_section, "the C28.0 copy is not byte-identical to the pre-registration"


@requires_section
def test_machine_derived_numbers_use_ascii_hyphens(section):
    assert "−" not in section, "the section contains a Unicode minus (U+2212)"


# ------------------------------------------------------------------- validity gates


@requires_summary
@requires_section
def test_gates_G1_G3_are_identities_not_approximations(summary, section):
    """The k = 8 cell must reproduce its frozen artefact with residual exactly 0.0."""
    g = summary["validity_gates"]["G1_G3_k8_reproduces_frozen_artefact"]
    assert g["max_abs_hit_rate_residual"] == 0.0
    assert g["max_abs_token_residual"] == 0.0
    assert g["passes"] is True
    for strand in GATE_STRANDS:
        v = g["per_strand"][strand]
        assert v["checked"] is True, strand
        assert v["hit_rate_residual"] == 0.0, strand
        assert v["tokens_residual"] == 0.0, strand
        assert v["max_abs_seed_hit_rate_residual"] == 0.0, strand
        assert v["max_abs_seed_token_residual"] == 0.0, strand
        assert set(v["per_seed"]) == set(SEEDS), strand
    # the gate the task named explicitly, against hard-coded published values
    a1 = g["per_strand"]["A1"]
    assert a1["hit_rate_reference"] == DEPLOYED_HBD_HIT_RATE
    assert a1["tokens_reference"] == DEPLOYED_HBD_TOKENS
    assert a1["hit_rate_measured"] == DEPLOYED_HBD_HIT_RATE
    assert a1["tokens_measured"] == DEPLOYED_HBD_TOKENS
    assert_in(section, "0.0", "G1 residual")
    assert_in(section, str(DEPLOYED_HBD_HIT_RATE), "published deployed hit rate")
    assert_in(section, str(DEPLOYED_HBD_TOKENS), "published deployed tokens per molecule")


@requires_summary
def test_gate_G4_cost_identity_holds_exactly_for_every_cell_and_seed(summary):
    """(k+1) x base steps: a non-zero remainder means the accounting is not what we claim."""
    g = summary["validity_gates"]["G4_cost_identity"]
    assert g["max_residual"] == 0
    assert g["passes"] is True
    for strand, s in summary["strands"].items():
        if strand == "B_guided_drafts":
            continue
        for k, cell in s["cells"].items():
            assert cell["cost_identity_max_residual"] == 0, f"{strand}/k={k}"
            for seed in SEEDS:
                assert cell["per_seed"][seed]["cost_identity_tokens_mod_k_plus_1"] == 0


@requires_summary
@requires_drafts
def test_gates_G5_G6_G7_for_the_composition(summary, drafts):
    assert summary["validity_gates"]["G5_n1_arms_identical"]["max_abs_residual"] == 0.0
    assert summary["validity_gates"]["G5_n1_arms_identical"]["passes"] is True
    g6 = summary["validity_gates"]["G6_pool_provenance"]["mismatches_per_seed"]
    assert set(g6) == set(SEEDS)
    assert all(v == 0 for v in g6.values()), g6
    g7 = summary["validity_gates"]["G7_per_draft_token_attribution"]["residual_per_seed"]
    assert all(v == 0 for v in g7.values()), g7
    # the same statements, straight from the generating artefact
    assert all(drafts["gates"]["G6_first_512_smiles_mismatches"][s] == 0 for s in SEEDS)
    assert all(drafts["gates"]["G7_per_draft_token_residual"][s] == 0 for s in SEEDS)


# ------------------------------------------------------------------------ the k sweep


@requires_summary
def test_every_strand_has_the_full_preregistered_k_grid(summary):
    for strand in STRANDS:
        s = summary["strands"].get(strand)
        if s is None:
            continue
        assert s["k_grid"] == list(K_GRID), strand
        assert len(s["cells"]) == len(K_GRID), strand


@requires_summary
@requires_section
def test_the_k_table_numbers_are_bound_to_json(summary, section):
    """Every hit rate and token cost the k table prints must come from the artefacts."""
    for strand in ("A1", "A2", "A3"):
        s = summary["strands"].get(strand)
        if s is None:
            continue
        for k in K_GRID:
            cell = s["cells"][str(k)]
            assert_in(section, f4(cell["hit_rate_mean"]), f"{strand} k={k} hit rate")
            assert_in(section, f2(cell["tokens_per_molecule_actual"]), f"{strand} k={k} tokens")


@requires_summary
def test_the_cost_is_k_plus_one_times_base_within_ten_percent(summary):
    """Tripwire on P1: if this stops holding, k is no longer the compute knob we described."""
    p1 = summary["predictions"]["P1_cost_is_k_plus_1_times_base"]
    assert p1["holds"] is True
    assert p1["max_abs_relative_error"] <= 0.10
    for strand, s in summary["strands"].items():
        if strand == "B_guided_drafts":
            continue
        for k in K_GRID:
            got = s["cells"][str(k)]["tokens_per_molecule_actual"]
            # base cost is measured, not assumed: the deployed unguided run's own number
            expect = DEPLOYED_BASE_TOKENS * (k + 1)
            assert abs(got - expect) / expect < 0.20, f"{strand}/k={k}: {got} vs {expect}"


@requires_summary
@requires_section
def test_D1_the_cost_band_is_refuted_as_stated(summary, section):
    """The measurement C26 never made: k spans an order of magnitude in cost."""
    d1 = summary["decision_rules"]["D1_cost_band"]
    assert d1["verdict"] == "REFUTED AS STATED"
    for strand, v in d1["per_strand"].items():
        assert v["c26_band_ratio"] == C26_BAND_RATIO
        # every strand spans far more than C26's 1.170
        assert v["token_ratio_max_over_min"] > 5.0, strand
        assert_in(section, f2(v["token_ratio_max_over_min"]), f"{strand} token ratio")
    assert_in(section, "REFUTED AS STATED", "D1 verdict")


@requires_summary
@requires_section
def test_D2_the_knob_is_null_and_the_verdict_is_printed(summary, section):
    """Tripwire: the whole section turns on the k profile being flat."""
    d2 = summary["decision_rules"]["D2_does_the_knob_buy_accuracy"]["per_strand"]
    for strand, v in d2.items():
        assert v["verdict"] in ("NULL", "PRODUCTIVE", "HARMFUL", "INCONSISTENT"), strand
        assert_in(section, f4(v["difference"]), f"{strand} D2 difference")
        assert_in(section, v["verdict"], f"{strand} D2 verdict")
    # the anchor strand: hit rate must not have moved materially across a 10x cost span
    assert d2["A1"]["verdict"] == "NULL"
    assert abs(d2["A1"]["difference"]) <= 0.02


@requires_summary
@requires_section
def test_D3_guidance_sits_below_both_curves_at_k32(summary, section):
    d3 = summary["decision_rules"]["D3_frontier_verdict_at_k32"]
    assert d3["D3a_all_below"] is True
    for strand, v in d3["per_strand"].items():
        assert v["advantage_vs_oracle_selected"] < 0, strand
        assert_in(section, f4(v["advantage_vs_oracle_selected"]),
                  f"{strand} advantage vs oracle-selected")
        assert_in(section, f4(v["advantage_vs_head_selected"]),
                  f"{strand} advantage vs head-selected")
    # the deployed configuration, named in the pre-registration as the one that would be news
    assert d3["per_strand"]["A1"]["D3b_above_head_selected"] is False


# ------------------------------------------------------------------- the composition


@requires_summary
@requires_section
def test_D4_the_composition_numbers_are_bound(summary, section):
    d4 = summary["decision_rules"]["D4_composition"]["per_arm"]
    assert set(d4) == {"oracle_reranked", "head_reranked"}
    for arm, v in d4.items():
        for n in DRAFT_GRID:
            r = v["per_n"][str(n)]
            assert_in(section, f4(r["hit_rate_mean"]), f"{arm} N={n} hit rate")
            assert_in(section, f2(r["tokens_per_molecule_actual"]), f"{arm} N={n} tokens")


@requires_drafts
def test_the_composition_is_the_deployed_sampler_at_depth(drafts):
    """The pool is guided decoding at the deployed setting, not a new sampler."""
    assert drafts["top_k"] == 8
    assert drafts["lambda"] == 1.0
    assert drafts["layer"] == -1
    assert drafts["head_checkpoint"] == "head_hbd_count_frozen_state.pt"
    assert drafts["deployed_reference"]["hit_rate"] == DEPLOYED_HBD_HIT_RATE
    assert drafts["deployed_reference"]["tokens_per_molecule_actual"] == DEPLOYED_HBD_TOKENS
    assert drafts["head_scoring_token_charge"] == 0
    for s in SEEDS:
        # tripwire: the head must still discriminate on its OWN steered distribution
        assert drafts["per_seed"][s]["head_auroc_terminal_position_on_guided_pool"] > 0.55, s


@requires_summary
def test_the_composition_costs_N_times_the_guided_run(summary):
    """Every draft is a full guided molecule, so cost is exactly linear in N."""
    b = summary["strands"]["B_guided_drafts"]
    one = b["arms"]["oracle_reranked"]["1"]["tokens_per_molecule_actual"]
    for n in DRAFT_GRID:
        got = b["arms"]["oracle_reranked"][str(n)]["tokens_per_molecule_actual"]
        assert abs(got - n * one) / (n * one) < 0.01, n
        # both arms select from one pool, so the budget is identical
        assert b["arms"]["head_reranked"][str(n)]["tokens_per_molecule_actual"] == got


# ----------------------------------------------------------------------- predictions


@requires_summary
@requires_section
def test_every_preregistered_prediction_is_scored_in_the_section(summary, section):
    """A prediction that is not scored, or scored only when it passed, is not a prediction."""
    preds = summary["predictions"]
    expected = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}
    seen = {k.split("_")[0] for k in preds}
    assert expected <= seen, sorted(expected - seen)
    for key, v in preds.items():
        tag = key.split("_")[0]
        assert "holds" in v, key
        assert_in(section, tag, f"prediction {tag} is scored")
    # the failures are the point: at least one must be recorded as failing
    assert any(v["holds"] is False for v in preds.values()), \
        "no pre-registered prediction failed, which is itself suspicious at n = 10"


@requires_summary
def test_P2_and_P8_failed_and_are_recorded_as_failures(summary):
    """Tripwire on honesty: these two failed as pre-registered and must stay recorded."""
    preds = summary["predictions"]
    assert preds["P2_hit_rate_non_decreasing_in_k"]["holds"] is False
    assert preds["P8_oracle_reranked_above_0p90_but_below_curve"]["holds"] is False


# ------------------------------------------------------------ the post-hoc extension


@requires_summary
def test_the_post_hoc_extension_is_labelled_as_post_hoc(summary):
    ph = summary.get("post_hoc_extended_best_of_n_curve")
    if ph is None:
        pytest.skip("extended best-of-N curve not present")
    assert ph["status"] == "POST HOC -- NOT PRE-REGISTERED"
    for prop, v in ph["consistency_with_c26_c27_at_n32"].items():
        # the extended pool re-estimates N = 32 over 2.5x more disjoint groups, so this is a
        # consistency check and not an identity; a disagreement above 0.03 would mean the two
        # pools are not the same base policy.
        assert abs(v["oracle_difference"]) < 0.03, prop
        assert abs(v["head_difference"]) < 0.03, prop


@requires_summary
@requires_section
def test_the_extended_curve_removes_the_extrapolation_for_the_composition(summary, section):
    ph = summary.get("post_hoc_extended_best_of_n_curve")
    if ph is None:
        pytest.skip("extended best-of-N curve not present")
    comp = ph["composition"]["oracle_reranked"]["8"]
    assert comp["extrapolated_beyond_grid_oracle_selected"] is False
    assert_in(section, f4(comp["advantage_vs_oracle_selected"]),
              "composition N=8 advantage against the measured extended oracle curve")
