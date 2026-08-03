"""C29 -- artefact-binding tests for the head-seed replication and the effective-lambda control.

Every number `reports/section_c29_head_seeds.md` prints is re-derived here from
`outputs/c29_summary/c29_metrics.json` and required to appear in the section text **in the
exact format the section prints it**, in the style of `tests/test_head_selected_bestofn.py`
and `tests/test_pooled_readout.py`.

Several assertions are deliberately tripwires on the *data* rather than on the prose.  If a
re-run moves a verdict -- if a checkpoint identity stops holding, if the unguided condition
starts depending on the head seed, if the head-seed sd collapses, if the effective-lambda
correction stops removing most of Rule A -- a test fails rather than leaving the section
standing unchallenged.

The section writes machine-derived numbers with ASCII hyphens.  A Unicode minus (U+2212)
would silently break every `in` assertion below, so one test forbids it outright.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "section_c29_head_seeds.md"
PREREG = OUT / "c29_prereg" / "C29.0_preregistration.md"
LOCK = OUT / "c29_prereg" / "prereg_lock.json"
SUMMARY = OUT / "c29_summary" / "c29_metrics.json"

ANCHORS = ("aromatic_rings", "hbd_count", "qed")
GEN_SEEDS = ("101", "202", "303")
MID_KEYS = ("A1", "A2", "A3")
HEAD_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901)
MIN_HEAD_SEEDS = 6

#: The arms, hard-coded rather than read from the artefact: a test that read the design
#: from the same file the summariser wrote would not catch the two agreeing wrongly.
MID_ARM_DESIGN = {"A1": ("hbd_count", 4, 2.0),
                  "A2": ("aromatic_rings", 3, 1.0),
                  "A3": ("qed", 4, 1.0)}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


requires_summary = pytest.mark.skipif(
    not SUMMARY.exists(), reason="C29 summary not present")
requires_section = pytest.mark.skipif(
    not SECTION.exists(), reason="C29 report section not present")
requires_prereg = pytest.mark.skipif(
    not PREREG.exists(), reason="C29 pre-registration not present")


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


@pytest.fixture(scope="module")
def full_section():
    return SECTION.read_text()


def _outside_prereg(text: str) -> str:
    a = text.index("<!-- BEGIN VERBATIM PREREG COPY -->")
    b = text.index("<!-- END VERBATIM PREREG COPY -->")
    return text[:a] + text[b:]


@pytest.fixture(scope="module")
def section(full_section):
    """The section WITHOUT the verbatim pre-registration copy.

    The pre-registration quotes C25's figures (0.0916, 0.5603, ...) and C17's spreads, so
    binding against the whole file would let a number pass merely because the prereg
    mentioned it.
    """
    return _outside_prereg(full_section)


def f2(x: float) -> str:
    return f"{x:.2f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def assert_in(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C29 section"


# ------------------------------------------------------------------ pre-registration


@requires_prereg
@requires_summary
def test_the_prereg_was_written_before_every_measurement():
    """Ordering by file mtime rather than by trust."""
    t = PREREG.stat().st_mtime
    for d in sorted(OUT.glob("c29_*")):
        if d.name == "c29_prereg":
            continue
        targets = [d] if d.is_file() else list(d.rglob("*"))
        for p in targets:
            if p.is_file():
                assert t < p.stat().st_mtime, f"{p} is not newer than the pre-registration"


@requires_prereg
@pytest.mark.skipif(not LOCK.exists(), reason="prereg lock not present")
def test_the_prereg_lock_records_the_prereg_hash():
    lock = _load(LOCK)
    assert lock["file_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    body = PREREG.read_text()
    block = body[body.index("## C29.0.1 Why"):]
    assert lock["prereg_block_sha256"] == hashlib.sha256(block.encode()).hexdigest()
    assert lock["prereg_block_chars"] == len(block)


@requires_prereg
@requires_section
def test_the_report_copies_the_prereg_verbatim(full_section):
    prereg = PREREG.read_text()
    body = prereg[prereg.index("## C29.0.1 Why"):].rstrip()
    assert body in full_section, \
        "the C29.0 copy is not byte-identical to the pre-registration"


@requires_section
def test_machine_derived_numbers_use_ascii_hyphens(section):
    assert "−" not in section, "the section contains a Unicode minus (U+2212)"


# ------------------------------------------------------------------- validity gates


@requires_summary
@requires_section
def test_gate_g1_g3_checkpoints_are_identical_tensor_by_tensor(summary, section):
    """G1/G3 are identities, not approximations: the residual must be exactly 0.0.

    C27's gate 4 failed because it compared file bytes; `torch.save` names the zip archive
    after the output path, so identical tensors give different file hashes.  C29 compares
    tensors, and this test asserts that the comparison is the tensor one.
    """
    g = summary["validity_gates"]["G1_G3_checkpoint_identity"]
    assert g["n_comparable"] > 0
    assert g["max_abs_parameter_residual"] == 0.0
    assert g["passes"] is True
    for row in g["rows"]:
        if row.get("comparable"):
            assert row["max_abs_parameter_residual"] == 0.0, row
            assert row["binner_identical"] is True, row
    assert_in(section, str(g["n_comparable"]), "G1/G3 comparison count")


@requires_summary
@requires_section
def test_gate_g2_end_to_end_identity(summary, section):
    """The C29 pipeline at head seed 1234 must reproduce the published run exactly."""
    g = summary["validity_gates"]["G2_end_to_end_identity"]
    assert g["n_run"] >= 1
    assert g["max_abs_hit_rate_residual"] == 0.0
    assert g["all_molecules_identical"] is True
    assert g["passes"] is True
    total = 0
    for row in g["rows"]:
        if row.get("run"):
            assert row["max_abs_hit_rate_residual"] == 0.0, row["replay"]
            assert row["molecules_identical"] is True, row["replay"]
            total += row["n_molecules_compared"]
            assert_in(section, row["reference"], "G2 reference run name")
    assert_in(section, str(total), "G2 molecules compared")


@requires_summary
def test_gate_g4_training_does_not_depend_on_the_seed_list(summary):
    g = summary["validity_gates"]["G4_seed_list_independence"]
    assert g["n_comparable"] > 0
    assert g["max_residual"] == 0.0
    assert g["passes"] is True


@requires_summary
def test_gate_g5_envelope_runs_are_the_deployed_code_path(summary):
    g = summary["validity_gates"]["G5_envelope_provenance"]
    if g["n_run"] == 0:
        pytest.skip("the lambda envelope was not run")
    assert g["max_lambda_residual"] == 0.0
    for row in g["rows"]:
        if row.get("run"):
            assert row["layer"] == -1, row["dir"]
            assert row["layer_source"] == "default (-1)", row["dir"]
            assert row["head_file_source"] == "default", row["dir"]
            assert row["lambda_source"] == "cli --lam", row["dir"]
            assert row["on_deployed_code_path"] is True, row["dir"]
    assert g["passes"] is True


@requires_summary
@requires_section
def test_gate_g6_unguided_cannot_depend_on_the_head_seed(summary, section):
    """A bug alarm, not a finding: the base policy is not steered by anything."""
    g = summary["validity_gates"]["G6_unguided_invariance"]
    assert g["max_span"] == 0.0
    assert g["passes"] is True
    for prop, row in g["per_property"].items():
        assert row["span"] == 0.0, prop
        assert row["sd"] == 0.0, prop
        assert row["n_runs"] >= 1
        assert_in(section, f4(row["mean"]), f"G6 unguided mean {prop}")


# --------------------------------------------------------------------- the families


@requires_summary
def test_the_design_is_the_preregistered_one(summary):
    assert tuple(summary["head_seeds_preregistered"]) == HEAD_SEEDS
    assert summary["preregistered_minimum_head_seeds"] == MIN_HEAD_SEEDS
    assert tuple(summary["generation_seeds"]) == GEN_SEEDS
    for key, (prop, layer, lam) in MID_ARM_DESIGN.items():
        fam = summary["families"][key]
        assert fam["property"] == prop
        assert fam["probe_point"] == layer
        assert fam["lam"] == lam
    for prop in ANCHORS:
        fam = summary["families"][f"D_{prop}"]
        assert fam["property"] == prop
        assert fam["probe_point"] == 12
        assert fam["lam"] == 1.0


@requires_summary
def test_no_head_seed_cell_is_reused_by_two_arms(summary):
    """A directory serving two arms would silently correlate them."""
    seen = {}
    for key, fam in summary["families"].items():
        for hs, d in fam["dirs"].items():
            assert d not in seen, f"{d} serves both {seen.get(d)} and {key}:{hs}"
            seen[d] = f"{key}:{hs}"


@requires_summary
def test_every_cell_has_three_generation_seeds(summary):
    for key, fam in summary["families"].items():
        for hs, vals in fam["hit_rate_by_generation_seed"].items():
            assert len(vals) == len(GEN_SEEDS), f"{key}:{hs}"
            assert abs(sum(vals) / len(vals)
                       - fam["hit_rate_by_head_seed"][hs]) < 1e-12, f"{key}:{hs}"


@requires_summary
@requires_section
def test_every_head_seed_hit_rate_is_printed(summary, section):
    for key in MID_KEYS + tuple(f"D_{p}" for p in ANCHORS):
        fam = summary["families"][key]
        for hs, v in fam["hit_rate_by_head_seed"].items():
            assert_in(section, f4(v), f"{key} head seed {hs} hit rate")


@requires_summary
@requires_section
def test_the_head_seed_sd_and_its_chi_square_interval_are_printed(summary, section):
    for key in MID_KEYS + tuple(f"D_{p}" for p in ANCHORS):
        fam = summary["families"][key]
        sd = fam["head_seed_sd"]
        if sd["n"] < 2:
            continue
        assert_in(section, f4(sd["sd"]), f"{key} head-seed sd")
        assert_in(section, f4(sd["lo"]), f"{key} head-seed sd CI lo")
        assert_in(section, f4(sd["hi"]), f"{key} head-seed sd CI hi")
        assert sd["lo"] <= sd["sd"] <= sd["hi"]


@requires_summary
@requires_section
def test_the_pooled_generation_seed_sd_is_printed(summary, section):
    for key in MID_KEYS + tuple(f"D_{p}" for p in ANCHORS):
        fam = summary["families"][key]
        g = fam.get("generation_seed_sd_pooled")
        if not g:
            continue
        assert_in(section, f4(g["sd"]), f"{key} pooled generation-seed sd")
        assert g["df"] == g["n_cells"] * (len(GEN_SEEDS) - 1)


@requires_summary
@requires_section
def test_the_variance_ratio_and_its_f_interval_are_printed(summary, section):
    """The headline uncertainty C25 did not have."""
    for key in MID_KEYS:
        vr = summary["families"][key].get("variance_ratio")
        if not vr:
            continue
        assert_in(section, f2(vr["ratio"]), f"{key} variance ratio")
        assert_in(section, f2(vr["lo"]), f"{key} variance ratio CI lo")
        assert_in(section, f2(vr["hi"]), f"{key} variance ratio CI hi")
        assert vr["lo"] <= vr["ratio"] <= vr["hi"]


@requires_summary
def test_the_span_is_never_compared_across_different_n_without_rescaling(summary):
    """§C29.0.4.4: a raw span at n=8 is not comparable to one at n=3."""
    for key, fam in summary["families"].items():
        sb = fam["head_seed_span"]
        if sb["n"] < 2:
            continue
        assert sb["d2_n"] is not None, key
        assert sb["sigma_from_span"] == pytest.approx(sb["span"] / sb["d2_n"])


@requires_summary
def test_c25s_generation_seed_span_was_the_minimum_of_three_not_a_pooled_estimate(summary):
    """The correction C29 exists to make.

    C25 quoted a generation-seed span of 0.0040 -- head seed 3456's -- against a head-seed
    span of 0.0916.  The other head seeds on the same arm span far more.  If a re-run ever
    made the three per-cell generation-seed spans comparable, the section's central
    correction would be wrong and this test must fail.
    """
    per_cell = summary["families"]["A1"]["generation_seed_sd_pooled"]["per_cell_span"]
    if "3456" not in per_cell or len(per_cell) < 3:
        pytest.skip("A1 does not carry C25's three head seeds")
    spans = sorted(per_cell.values())
    assert spans[0] == pytest.approx(0.0040, abs=5e-4), \
        "head seed 3456's generation-seed span is no longer ~0.0040"
    assert max(per_cell.values()) > 4 * min(per_cell.values()), (
        "the per-cell generation-seed spans are no longer wildly unequal, so quoting the "
        "minimum would no longer be the error C29 reports")


# ------------------------------------------------------------------- effective lambda


@requires_summary
def test_the_spread_table_is_c17s_verbatim(summary):
    """The rescale factor must come from C17's artefact, not from anything C29 measured."""
    ref = _load(OUT / "c17_layer_steering" / "layer_steering_metrics.json")
    for prop in ANCHORS:
        layers = ref["properties"][prop]["layers"]
        mine = summary["spread_table"][prop]
        for k, v in layers.items():
            assert mine["spread"][k] == v["mean_head_q_spread_across_candidates"], (prop, k)
            assert mine["ratio"][k] == pytest.approx(
                v["mean_head_q_spread_across_candidates"]
                / layers["12"]["mean_head_q_spread_across_candidates"], rel=1e-12)
        assert mine["ratio"]["12"] == 1.0


@requires_summary
@requires_section
def test_the_effective_lambda_of_every_c23_arm_is_printed(summary, section):
    rows = summary["effective_lambda"]["seed_1234_table"]
    assert rows, "no C23 arm was priced"
    for r in rows:
        assert r["lambda_effective"] == pytest.approx(r["lam"] * r["spread_ratio"])
        assert_in(section, f4(r["lambda_effective"]), "effective lambda")
        assert_in(section, f4(r["mid_hit_rate"]), "mid arm hit rate")
        if r.get("advantage_raw") is not None:
            assert_in(section, f4(r["advantage_raw"]), "raw advantage")
        for tag in ("coarse", "fine"):
            if r.get(f"advantage_effective_{tag}") is not None:
                assert_in(section, f4(r[f"advantage_effective_{tag}"]),
                          f"effective-lambda advantage ({tag})")


@requires_summary
def test_the_effective_lambda_correction_never_raises_a_positive_advantage(summary):
    """The deployed envelope rises with lambda over the range every arm sits in, so
    pricing at a *higher* effective lambda can only cost an arm.  A correction that made
    an arm look better would mean the envelope is falling there, and the section's
    interpretation would not hold."""
    for r in summary["effective_lambda"]["seed_1234_table"]:
        raw = r.get("advantage_raw")
        eff = r.get("advantage_effective_fine", r.get("advantage_effective_coarse"))
        if raw is None or eff is None:
            continue
        env = r.get("envelope_fine") if r.get("advantage_effective_fine") is not None \
            else r.get("envelope_coarse")
        if env and env.get("bracket_hit_rates") and \
                env["bracket_hit_rates"][1] < env["bracket_hit_rates"][0]:
            continue          # the envelope is falling in this bracket; exempt
        assert eff <= raw + 1e-12, r


@requires_summary
@requires_section
def test_the_confound_share_is_printed_for_the_three_reviewer_arms(summary, section):
    named = {("aromatic_rings", 3, 1.0), ("aromatic_rings", 6, 1.0),
             ("hbd_count", 4, 1.0)}
    hits = 0
    for r in summary["effective_lambda"]["seed_1234_table"]:
        if (r["property"], r["probe_point"], r["lam"]) not in named:
            continue
        share = r.get("confound_share_fine", r.get("confound_share_coarse"))
        if share is None:
            continue
        hits += 1
        assert_in(section, f2(share), "confound share")
    assert hits == len(named), "not every named reviewer arm was priced"


# -------------------------------------------------------------------- decision rules


@requires_summary
@requires_section
def test_every_rule_is_scored_in_the_section(summary, section):
    rules = summary["decision_rules"]
    for tag in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert tag in rules, f"{tag} was not scored"
        assert tag in section, f"{tag} is not reported in the section"


@requires_summary
@requires_section
def test_R1_and_R2_print_both_sides_of_the_comparison(summary, section):
    r1 = summary["decision_rules"]["R1"]
    assert_in(section, f4(r1["head_seed_sd"]), "R1 head-seed sd")
    assert_in(section, f4(r1["generation_seed_sd_pooled"]), "R1 generation-seed sd")
    r2 = summary["decision_rules"]["R2"]
    assert_in(section, f2(r2["ratio"]), "R2 ratio")
    assert_in(section, f2(r2["ci"][0]), "R2 ratio CI lo")
    assert_in(section, f2(r2["ci"][1]), "R2 ratio CI hi")
    # tripwire: the whole of priority 1 is the claim that C25's 25x is not pinned down
    assert r2["ci"][0] < 25.0, \
        "the ratio's lower bound now exceeds 25; C29's central correction is void"


@requires_summary
@requires_section
def test_R3_reports_the_deployed_comparison_whichever_way_it_goes(summary, section):
    r3 = summary["decision_rules"]["R3"]
    assert r3["band"] == [0.5, 2.0]
    assert r3["conclusion"] in ("probes in general", "specific to mid-network probes",
                               "undetermined")
    assert r3["conclusion"] in section
    for prop, row in r3["per_anchor"].items():
        assert_in(section, f2(row["q"]), f"R3 q {prop}")
        assert_in(section, f4(row["deployed_head_seed_sd"]), f"R3 deployed sd {prop}")


@requires_summary
@requires_section
def test_R4_reports_the_preregistration_defect(summary, section):
    """A1 sits at lambda=2 and the deployed family was fixed at lambda=1.  The defect is
    reported, in the style of C27's gate 4, not amended away."""
    r4 = summary["decision_rules"]["R4"]
    assert "preregistration_defect" in r4
    assert "defect" in section.lower()
    for key, row in r4["per_arm"].items():
        assert_in(section, f4(row["mean"]), f"R4 mean {key}")
        assert_in(section, f4(row["ci"][0]), f"R4 CI lo {key}")
        assert_in(section, f4(row["ci"][1]), f"R4 CI hi {key}")


@requires_summary
@requires_section
def test_R5_prints_raw_and_corrected_for_every_mid_arm(summary, section):
    r5 = summary["decision_rules"]["R5"]
    for key, row in r5["per_arm"].items():
        if not row.get("ok"):
            continue
        assert_in(section, f4(row["deployed_at_lambda_effective"]),
                  f"R5 deployed at lambda_eff {key}")
        assert_in(section, f4(row["t"]["mean"]), f"R5 corrected mean {key}")
        assert_in(section, f4(row["raw_t"]["mean"]), f"R5 raw mean {key}")


@requires_summary
@requires_section
def test_R6_reports_rule_b_at_n_at_least_six(summary, section):
    r6 = summary["decision_rules"].get("R6")
    if r6 is None:
        pytest.skip("A1's best-of-N comparators were not run")
    assert_in(section, f4(r6["mean"]), "R6 mean advantage")
    assert_in(section, f4(r6["ci"][0]), "R6 CI lo")
    assert_in(section, f4(r6["ci"][1]), "R6 CI hi")
    assert_in(section, str(r6["n_positive"]), "R6 positive head-seed count")


@requires_summary
def test_no_three_seed_bootstrap_is_reported_anywhere(summary):
    """§C29.0.4.6.  A bootstrap may only appear at n >= 6."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("computed") is True and "n_boot" in node:
                found.append(node["n"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(summary)
    for n in found:
        assert n >= MIN_HEAD_SEEDS, f"a bootstrap was computed at n = {n}"


# ---------------------------------------------------------------------- predictions


@requires_summary
@requires_section
def test_every_prediction_is_scored_in_the_section(summary, section):
    preds = summary["predictions"]
    assert preds, "no prediction was scored"
    for key in preds:
        tag = key.split("_")[0]
        assert tag in section, f"{tag} is not scored in the section"
        assert isinstance(preds[key]["holds"], bool)


@requires_summary
@requires_section
def test_predictions_report_their_measured_value_not_just_a_verdict(summary, section):
    p = summary["predictions"]
    if "P1_sd_head_A1_in_0.030_0.080" in p:
        assert_in(section, f4(p["P1_sd_head_A1_in_0.030_0.080"]["value"]), "P1 value")
    if "P2_sd_gen_A1_below_0.010" in p:
        assert_in(section, f4(p["P2_sd_gen_A1_below_0.010"]["value"]), "P2 value")
    if "P3_ratio_above_5_and_25_not_pinned" in p:
        assert_in(section, f2(p["P3_ratio_above_5_and_25_not_pinned"]["ratio"]),
                  "P3 ratio")


# ------------------------------------------------------------------------ tripwires


@requires_summary
def test_head_seed_variance_is_not_negligible_on_the_decisive_arm(summary):
    """If this ever fails, the project's most transferable finding has evaporated and the
    section must be rewritten rather than left standing."""
    a1 = summary["families"]["A1"]
    assert a1["head_seed_sd"]["n"] >= 3
    assert a1["head_seed_sd"]["sd"] > 0.01, \
        "the head-seed sd on the Rule B arm has collapsed below 0.01"


@requires_summary
def test_the_arms_meet_the_preregistered_minimum_or_say_so(summary):
    """n < 6 is allowed by C29.0.2 but must be recorded, never silently averaged."""
    for key, fam in summary["families"].items():
        assert isinstance(fam["meets_preregistered_minimum"], bool)
        assert fam["n_head_seeds"] == len(fam["head_seeds_present"])
        assert (fam["n_head_seeds"] >= MIN_HEAD_SEEDS) == fam["meets_preregistered_minimum"]


@requires_summary
def test_tokens_per_molecule_are_comparable_across_head_seeds(summary):
    """Head seed changes the head's weights, not the decoding cost; a large token spread
    would mean the arms are not compute-matched to each other."""
    for key, fam in summary["families"].items():
        t = list(fam["tokens_by_head_seed"].values())
        if len(t) < 2:
            continue
        assert max(t) / min(t) < 1.25, f"{key}: tokens per molecule span {min(t)}-{max(t)}"


@requires_summary
def test_no_reported_quantity_is_nan(summary):
    bad = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, float) and (math.isnan(node) or math.isinf(node)):
            bad.append(path)

    walk(summary)
    assert not bad, f"non-finite values in the summary: {bad[:10]}"


# ---------------------------------------------------- priority 4: C27 across head seeds


@requires_summary
@requires_section
def test_priority_4_is_either_reported_with_numbers_or_declared_not_run(summary, section):
    p4 = summary.get("priority_4_c27_across_head_seeds")
    if not p4 or not p4["per_property"]:
        assert "not run" in section.lower() or "did not" in section.lower()
        pytest.skip("priority 4 was not run")
    for prop, v in p4["per_property"].items():
        assert v["published_e4"] == pytest.approx(
            {"aromatic_rings": -0.0439, "hbd_count": -0.0292, "qed": -0.0522}[prop])
        assert_in(section, f4(v["mean"]), f"P4 mean {prop}")
        assert_in(section, f4(v["span"]), f"P4 span {prop}")
        for hs, r in v["per_head_seed"].items():
            assert r["head_seed_in_checkpoint"] == int(hs), (prop, hs)
            assert_in(section, f4(r["advantage_vs_head_selected"]), f"P4 adv {prop}/{hs}")
            assert_in(section, f4(r["advantage_vs_oracle_selected"]),
                      f"P4 oracle adv {prop}/{hs}")
            # tripwire: the oracle comparison is not what C29 is questioning
            assert r["advantage_vs_oracle_selected"] < 0, (prop, hs)


@requires_summary
def test_priority_4_head_seed_instability_is_recorded(summary):
    """C29's finding is that one of C27's three anchors does not replicate.  If a re-run
    made all three stable, §C29.10 would be wrong and must be rewritten."""
    p4 = summary.get("priority_4_c27_across_head_seeds")
    if not p4 or not p4["per_property"]:
        pytest.skip("priority 4 was not run")
    assert p4["all_signs_stable"] is False, \
        "every C27 anchor now replicates across head seeds; §C29.10 is stale"
    unstable = [p for p, v in p4["per_property"].items() if not v["sign_stable"]]
    assert unstable == ["hbd_count"], unstable


# ------------------------------------------------------------------------- REPRODUCE


@requires_section
def test_the_section_carries_a_reproduce_block(section):
    assert "## REPRODUCE" in section
    assert "scripts/23_head_seed_variance.py" in section
    assert "scripts/23_summarise_c29.py" in section
    assert "tests/test_head_seeds.py" in section


@requires_section
def test_the_section_lists_conflicts_rather_than_editing_other_sections(section):
    assert "for the owner to merge" in section
