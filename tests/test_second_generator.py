"""C31 -- artefact-binding and implementation tests for the second-generator replication.

C28 found the project's only positive result on GP-MoLFormer-Uniq; C30 replicated it across
probe seeds.  Both are facts about **one generator**.  C31 re-runs the pipeline on
`entropy/gpt2_zinc_87m` -- GPT-2, full softmax attention, 87M parameters, byte-level BPE
over SMILES -- and asks whether the crossing is a property of the method or of that model.

These tests hold five things in place:

1. **The pre-registration was frozen before any measurement**, byte-for-byte, and the
   section copies it verbatim.  Enforced by mtime and SHA-256, the machinery C23-C30 use.
   C31 froze it before the *feasibility* stage, not merely before the decision stage, so
   the mtime sweep covers every C31 directory including `c31_feasibility`.
2. **The gates are real.**  G1 (cached states equal full recomputation at all 13 probe
   points) is what makes the token accounting mean anything, and G2 (the cached decode
   makes the same *decision*) is what makes G1 relevant to a sampled token.  A test that
   checked only the verdict would pass on a run whose gate silently failed.
3. **The generator is frozen and is the one that was pinned.**  Revision SHA, parameter
   count, parameter sum, `requires_grad` all False, `training` False.
4. **The comparison is like for like.**  The decoding rule, the summariser, the property
   calculators and the interpolator are the molecular ones, imported -- asserted here on
   the import graph, so a future copy-paste fork fails a test.
5. **Every number the section prints is re-derived from the artefacts.**  Machine-derived
   numbers are formatted with ASCII hyphens (`f"{x:+.4f}"`), never a Unicode minus, or the
   binding assertions would fail on characters that look identical.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREREG = OUT / "c31_prereg" / "C31.0_preregistration.md"
LOCK = OUT / "c31_prereg" / "prereg_lock.json"
METRICS = OUT / "c31_summary" / "c31_metrics.json"
SECTION = ROOT / "reports" / "section_c31_second_generator.md"

GENERATION_SEEDS = ("101", "202", "303")
HEAD_SEEDS = (1234, 2345, 3456)
K_GRID = (2, 4, 8, 16, 32)
REQUIRED_PROPERTIES = ("hbd_count", "aromatic_rings")
PINNED_REVISION = "f42a5a10e24c0350aeadb50865bd90a714d0b2bf"
PINNED_REPO = "entropy/gpt2_zinc_87m"


def _json(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def metrics():
    return _json(METRICS)


@pytest.fixture(scope="module")
def section():
    if not SECTION.exists():
        pytest.skip(f"{SECTION} not present")
    return SECTION.read_text()


# ======================================================== the pre-registration is a freeze


def test_the_prereg_hash_matches_the_recorded_one():
    lock = _json(LOCK)
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock["file_sha256"], (
        "the C31 pre-registration was edited after it was frozen"
    )
    assert len(raw) == lock["file_bytes"]


def test_the_prereg_was_written_before_every_measurement():
    """mtime ordering, the same enforcement C23-C30 use.

    C31 froze before the feasibility stage, so `c31_feasibility` is in the sweep: the
    validity floor that decides whether the experiment is interpretable at all was
    committed before the validity was known.
    """
    lock = _json(LOCK)
    t0 = PREREG.stat().st_mtime
    checked = 0
    for pattern in ("c31_feasibility/*.json", "c31_zinc50k/*.json", "c31_gates/*.json",
                    "c31_heads/*.json", "c31_bestofn_*/*.json",
                    "c31_ksweep_*/*.json", "c31_summary/*.json"):
        for f in OUT.glob(pattern):
            checked += 1
            assert f.stat().st_mtime > t0, (
                f"{f} is older than the pre-registration -- the ordering that makes C31's "
                f"decision rules pre-registered rather than post-hoc is broken"
            )
    if checked == 0:
        pytest.skip("no C31 artefacts yet")
    assert lock["file_mtime_utc"] <= lock["frozen_at_utc"]


def test_the_section_copies_the_prereg_verbatim(section):
    text = PREREG.read_text()
    block = text[text.index("## C31.0.1 Why this experiment exists"):]
    lock = _json(LOCK)
    assert hashlib.sha256(block.encode()).hexdigest() == lock["prereg_block_sha256"]
    assert block.strip() in section, (
        "the C31 section does not reproduce the pre-registration byte for byte"
    )


# ============================================================ the generator is what it says


def test_the_generator_is_the_pinned_revision_and_is_frozen(metrics):
    fp = metrics["feasibility"]["generator_fingerprint"]
    assert fp["repo"] == PINNED_REPO
    assert fp["revision"] == PINNED_REVISION
    assert fp["all_parameters_frozen"] is True, "the generator was not frozen"
    assert fp["training_mode"] is False, "the generator was left in training mode"
    assert fp["n_layers"] == 12
    assert fp["n_probe_points"] == 13
    assert metrics["generator"]["revision"] == PINNED_REVISION


def test_the_same_frozen_generator_produced_every_stage(metrics):
    """One parameter-sum checksum across feasibility, dataset and best-of-N.

    A silently reloaded or differently-cast model would move this number, and every
    comparison in the section assumes one frozen generator throughout.
    """
    ref = metrics["feasibility"]["generator_fingerprint"]["parameter_sum"]
    seen = 0
    for f in [OUT / "c31_zinc50k" / "dataset_metrics.json"] + \
            sorted(OUT.glob("c31_bestofn_*/n_sweep_metrics.json")):
        if not f.exists():
            continue
        fp = json.loads(f.read_text())["generator"]["fingerprint"]
        assert fp["parameter_sum"] == ref, f
        assert fp["revision"] == PINNED_REVISION, f
        seen += 1
    assert seen >= 1


def test_no_alternative_serialization_is_used_anywhere():
    """SMILES only.  SAFE and SELFIES models are forbidden without owner permission.

    Naming them in a prose exclusion is required, not forbidden -- so the check is that
    the *loaded* repository is the pinned SMILES one, in the config and in every artefact
    that records what it ran.
    """
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "c31_second_generator.yaml").read_text())
    assert cfg["model_repo"] == PINNED_REPO
    assert cfg["tokenizer_repo"] == PINNED_REPO
    assert cfg["model_revision"] == PINNED_REVISION

    banned = ("safe-gpt", "NV-GenMol", "selfies")
    seen = 0
    for f in list(OUT.glob("c31_*/configs_used.json")) + \
            list(OUT.glob("c31_*/provenance.json")):
        blob = f.read_text().lower()
        for b in banned:
            assert b not in blob, f"{f} records a forbidden serialization: {b}"
        seen += 1
    assert seen >= 1


# ================================================================ the method is not forked


def test_c31_imports_the_molecular_decoding_rule_rather_than_reimplementing_it():
    """`guided_sample` and `combine_scores` must be the molecular functions."""
    src = (ROOT / "scripts" / "25_c31_second_generator.py").read_text()
    tree = ast.parse(src)
    imported_from = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported_from[a.name] = node.module
    assert imported_from.get("guided_sample") == "property_to_go.guidance"
    assert imported_from.get("combine_scores") == "property_to_go.guidance"
    assert imported_from.get("TargetScorer") == "property_to_go.guidance"
    # and the decoder is never redefined locally
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "guided_sample" not in defined
    assert "combine_scores" not in defined


def test_the_per_cell_summary_is_the_molecular_summariser():
    """`summarise` comes from scripts/05_guided_generation.py, not from a copy."""
    src = (ROOT / "scripts" / "25_c31_second_generator.py").read_text()
    assert '_load_module(ROOT / "scripts" / "05_guided_generation.py"' in src
    assert "summarise = _s05.summarise" in src
    assert 'score_pool = _s21.score_pool' in src


def test_the_frontier_interpolator_is_c26s_unmodified():
    src = (ROOT / "scripts" / "25_summarise_c31.py").read_text()
    assert '_load_module(ROOT / "scripts" / "21_summarise_c26.py"' in src
    assert "interp = _c26.interp" in src
    assert "t_interval = _c26.t_interval" in src
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "interp" not in defined and "t_interval" not in defined


def test_the_molecular_library_hook_defaults_to_the_molecular_behaviour():
    """The one line C31 adds to `guidance.py` must be inert for GP-MoLFormer."""
    from property_to_go import generation, guidance, model_io

    src = (ROOT / "src" / "property_to_go" / "guidance.py").read_text()
    assert 'getattr(gen, "repeat_cache_fn", None) or repeat_cache' in src
    assert not hasattr(model_io.FrozenGenerator, "repeat_cache_fn"), (
        "FrozenGenerator gained a repeat_cache_fn, so molecular runs would no longer take "
        "the linear-attention path they were measured on"
    )
    assert guidance.repeat_cache is generation.repeat_cache


def test_the_probe_trainer_hook_defaults_to_the_molecular_trainer():
    import inspect

    from property_to_go import heads, probe_layers

    sig = inspect.signature(probe_layers.train_one_probe)
    assert sig.parameters["trainer"].default is None
    src = inspect.getsource(probe_layers.train_one_probe)
    assert "train = trainer if trainer is not None else train_head" in src
    assert probe_layers.train_head is heads.train_head


# ============================================================================== the gates


def test_gate_g0_the_validity_floor_was_measured_and_passed(metrics):
    g0 = metrics["validity_gates"]["G0"]
    assert g0["threshold"] == 0.80
    assert g0["passes"] is True, f"G0 failed: {g0}"
    assert g0["measured"] == metrics["feasibility"]["validity"]


def test_gate_g1_cached_states_equal_full_recomputation_at_every_probe_point(metrics):
    g1 = metrics["validity_gates"]["G1"]
    assert g1["passes"] is True, f"G1 failed: {g1}"
    assert g1["tolerance"] == 2e-3, "C24's tolerance was re-chosen rather than reused"
    assert len(g1["max_abs_difference_by_probe_point"]) == 13, (
        "G1 did not check all 13 probe points"
    )
    assert g1["max_abs_difference"] <= g1["tolerance"]
    # the claim is a measured residual, not bit identity -- the section must not overstate
    assert g1["bit_identical"] is False


def test_gate_g2_the_cached_decode_makes_the_same_decision(metrics):
    g2 = metrics["validity_gates"]["G2"]
    assert g2["passes"] is True, f"G2 failed: {g2}"
    assert g2["max_abs_probability_difference"] <= g2["prob_tolerance"]
    assert g2["min_argmax_agreement"] >= g2["argmax_agreement_threshold"]
    # every arm x every k must have been checked, or G1 is not connected to a token
    assert g2["n_cells_checked"] >= 2 * len(K_GRID) * len(REQUIRED_PROPERTIES)


def test_gate_g4_cost_identity_holds_in_every_cell(metrics):
    g4 = metrics["validity_gates"]["G4"]
    assert g4["passes"] is True, f"G4 failed: {g4}"
    assert g4["max_residual"] == 0
    assert g4["n_cells_checked"] == len(metrics["cells"])
    # re-derive from the cells themselves, not only from the gate's own summary
    for name, c in metrics["cells"].items():
        assert c["cost_identity_max_residual"] == 0, name
        for s, ps in c["per_seed"].items():
            assert ps["cost_identity_tokens_mod_k_plus_1"] == 0, (name, s)


def test_gate_g5_every_target_interval_is_an_exact_union_of_bins(metrics):
    """pilot_report.md section 11.5's bug: it presents as a miscalibrated head, not an error."""
    g5 = metrics["validity_gates"]["G5"]
    assert g5["passes"] is True, f"G5 failed: {g5}"
    for prop, row in g5["properties"].items():
        assert row["is_exact"] is True, prop
        assert row["n_bins_selected"] >= 1, prop
        assert row["target_below_top_category"] is True, prop
        assert abs(row["masked_rate"] - row["true_rate"]) < 1e-6, prop


def test_gate_g6_no_group_leakage_and_the_freeze_is_hashed(metrics):
    g6 = metrics["validity_gates"]["G6"]
    assert g6["passes"] is True
    assert g6["no_group_leakage"] is True
    assert len(g6["target_intervals_sha256"]) == 64
    assert len(g6["windows_sha256"]) == 64


def test_the_frozen_target_intervals_still_hash_to_what_was_recorded(metrics):
    """A freeze that is not re-checked is a claim, not a freeze."""
    g6 = metrics["validity_gates"]["G6"]
    for name, key in (("target_intervals.json", "target_intervals_sha256"),
                      ("windows.json", "windows_sha256")):
        f = OUT / "c31_zinc50k" / name
        if not f.exists():
            pytest.skip(f"{f} not present")
        assert hashlib.sha256(f.read_bytes()).hexdigest() == g6[key], (
            f"{name} changed after it was frozen"
        )


def test_gate_g3_is_reported_as_a_residual_and_does_not_claim_exact_equality(metrics):
    g3 = metrics["validity_gates"].get("G3")
    if g3 is None:
        pytest.skip("G3 not run")
    assert "does not block" in g3["rule"]
    assert abs(g3["hit_rate_residual"]) <= g3["hit_rate_tolerance"]
    assert g3["token_full_recompute_relative_residual"] <= g3["token_relative_tolerance"]
    # the `actual` counts differ by design; the section must not present them as equal
    assert (g3["cached_compute"]["processed_tokens_actual"]
            != g3["full_compute"]["processed_tokens_actual"])


def test_no_decision_rule_is_scored_past_a_failed_gate(metrics):
    u = metrics["uninterpretability"]
    if metrics["verdict"]["verdict"] != "UNINTERPRETABLE":
        assert u["all_required_gates_pass"] is True, (
            "a verdict was issued while a required gate failed"
        )
        assert u["experiment_degenerate"] is False


# ================================================================= design integrity


def test_every_cell_has_all_three_generation_seeds(metrics):
    for name, c in metrics["cells"].items():
        assert sorted(c["per_seed"]) == sorted(GENERATION_SEEDS), name
        assert len(c["hit_rate_values"]) == 3, name


def test_the_uncertainty_is_a_t_interval_on_two_df_and_not_a_bootstrap(metrics):
    """The rule this project adopted after finding the n = 3 bootstrap vacuous."""
    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    assert not [k for k in keys(metrics) if "bootstrap" in k.lower()], (
        "a bootstrap statistic appears in a C31 artefact"
    )
    n = 0
    for name, c in metrics["cells"].items():
        ti = c.get("advantage_vs_oracle_selected_seed_t_interval")
        if ti is None:
            continue
        n += 1
        assert ti["n_seeds"] == 3, name
        assert "2 df" in ti["note"], name
        # t(0.975, 2) = 4.302653 -- re-derive the half-width from the raw values
        vals = c["advantage_vs_oracle_selected_per_seed"]
        mean = sum(vals) / 3
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2)
        half = 4.302653 * sd / math.sqrt(3)
        assert abs(ti["lo"] - (mean - half)) < 1e-9, name
        assert abs(ti["hi"] - (mean + half)) < 1e-9, name
    assert n == len(metrics["cells"])


def test_the_mid_probe_point_was_selected_on_prediction_not_on_steering(metrics):
    """C17 section 21.5.2: selecting a probe point on its steering outcome manufactures a
    positive.  The rule must be validation AUROC, and the selection must be the argmax of
    the validation column -- re-derived here, not trusted."""
    for prop, d in metrics["depth_curves"].items():
        mid = d["mid_probe_point"]
        assert "VALIDATION" in mid["rule"]
        assert "never by steering outcome" in mid["rule"]
        cands = mid["candidates"]
        assert 0 not in cands and 12 not in cands
        best = max(cands, key=lambda L: (d["by_probe_point"][str(L)]["val_target_auroc_mean"], -L))
        assert mid["selected"] == best, (
            f"{prop}: the recorded mid probe point is not the argmax of validation AUROC"
        )


def test_the_advantage_is_recomputed_from_the_curve_and_the_cell(metrics):
    """Re-derive every headline advantage from the two artefacts it comes from."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "c26_summarise", ROOT / "scripts" / "21_summarise_c26.py")
    c26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c26)

    for name, c in metrics["cells"].items():
        cv = metrics["curves"][c["property"]]
        h, _, _, extrap = c26.interp(cv["tokens_per_molecule_actual"],
                                     cv["oracle_selected"],
                                     c["tokens_per_molecule_actual"])
        assert abs(h - c["oracle_selected_interpolated_hit_rate"]) < 1e-12, name
        assert abs((c["hit_rate_mean"] - h)
                   - c["advantage_vs_oracle_selected"]) < 1e-12, name
        assert extrap == c["extrapolated_beyond_grid_oracle_selected"], name


def test_a_crossing_requires_the_interval_the_sign_and_the_validity_floor(metrics):
    """C31.0.6's definition, re-derived rather than trusted."""
    for name, c in metrics["cells"].items():
        ti = c["advantage_vs_oracle_selected_seed_t_interval"]
        expected = bool(c["advantage_vs_oracle_selected"] > 0 and ti["lo"] > 0
                        and c["validity_mean"] >= 0.80)
        assert c["crosses"] == expected, name
        if c["validity_mean"] < 0.80:
            assert c["excluded_from_crossing_on_validity"] is True, name
            assert c["crosses"] is False, name


def test_the_comparator_is_the_oracle_selected_curve_and_it_never_sees_the_probe(metrics):
    for prop in metrics["curves"]:
        f = OUT / f"c31_bestofn_{prop}" / "n_sweep_metrics.json"
        d = json.loads(f.read_text())
        assert d["arm"] == "oracle_selected"
        assert "TRUE RDKit property" in d["selection"]
        assert d["grid"] == [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]


def test_the_verdict_follows_the_pre_registered_rule_not_the_prose(metrics):
    dr = metrics["decision_rules"]
    u = metrics["uninterpretability"]
    v = metrics["verdict"]["verdict"]
    if u["uninterpretable"]:
        assert v == "UNINTERPRETABLE"
    elif dr["D1"]["fires"]:
        assert v == "REPLICATES"
    elif dr["D5"]["fires"]:
        assert v == "DOES NOT REPLICATE"
    else:
        assert v == "PARTIAL"
    # D1 and D5 cannot both fire
    assert not (dr["D1"]["fires"] and dr["D5"]["fires"])


def test_a_partial_result_is_never_written_up_as_a_replication(metrics, section):
    if metrics["verdict"]["verdict"] == "REPLICATES":
        pytest.skip("verdict is REPLICATES")
    low = section.lower()
    assert "the crossing replicates" not in low or "does not" in low, (
        "a non-replication is being reported with replication wording"
    )


def test_d7_the_honesty_rule_is_scored_whatever_the_others_did(metrics, section):
    d7 = metrics["decision_rules"]["D7"]
    assert "D7" in section
    for name in d7["cells"]:
        assert name in section, f"D7 fired on {name} but the section does not name it"


# ==================================================== every printed number is re-derivable


def f4(x: float) -> str:
    """ASCII, always.  A Unicode minus looks identical and fails every assertion."""
    return f"{x:.4f}"


def s4(x: float) -> str:
    return f"{x:+.4f}"


def test_the_section_prints_the_stage_zero_numbers(metrics, section):
    fe = metrics["feasibility"]
    assert f4(fe["validity"]) in section
    assert f4(fe["uniqueness"]) in section
    assert str(fe["n_molecules"]) in section
    assert str(fe["n_probe_points"]) in section


def test_the_section_prints_the_gate_residuals(metrics, section):
    g1 = metrics["validity_gates"]["G1"]
    g2 = metrics["validity_gates"]["G2"]
    assert f"{g1['max_abs_difference']:.3e}" in section
    assert f"{g2['max_abs_probability_difference']:.3e}" in section
    assert f"{g2['min_argmax_agreement']:.6f}" in section


def test_the_section_prints_every_depth_curve_point(metrics, section):
    for prop, d in metrics["depth_curves"].items():
        for L, r in d["by_probe_point"].items():
            assert f4(r["test_target_auroc_mean"]) in section, (prop, L)
        assert f4(d["trivial_test_target_auroc_mean"]) in section, prop
        assert str(d["mid_probe_point"]["selected"]) in section


def test_the_section_prints_every_cell_hit_rate_advantage_and_interval(metrics, section):
    for name, c in metrics["cells"].items():
        assert f4(c["hit_rate_mean"]) in section, f"{name} hit rate"
        assert s4(c["advantage_vs_oracle_selected"]) in section, f"{name} advantage"
        ti = c["advantage_vs_oracle_selected_seed_t_interval"]
        assert s4(ti["lo"]) in section, f"{name} interval lo"
        assert s4(ti["hi"]) in section, f"{name} interval hi"
        assert f"{c['tokens_per_molecule_actual']:.2f}" in section, f"{name} budget"


def test_the_section_prints_the_per_seed_values_in_full(metrics, section):
    """C31.0.5 requires the raw per-seed values beside every interval."""
    for name, c in metrics["cells"].items():
        for v in c["hit_rate_values"]:
            assert f4(v) in section, f"{name} per-seed hit rate {v}"


def test_the_section_prints_the_best_of_n_curve(metrics, section):
    for prop, cv in metrics["curves"].items():
        for h in cv["oracle_selected"]:
            assert f4(h) in section, (prop, h)


def test_every_prediction_is_scored_including_the_falsified_ones(metrics, section):
    P = metrics["predictions"]
    assert P, "no predictions scored"
    for name, p in P.items():
        assert name in section, f"prediction {name} is not scored in the section"
        assert p["outcome"] in ("CONFIRMED", "FALSIFIED")
    falsified = metrics["prediction_summary"]["falsified"]
    for name in falsified:
        assert "FALSIFIED" in section
        assert name in section
    # the summary in the section must not quietly drop the failures
    assert str(len(P)) in section


def test_every_decision_rule_is_scored_in_the_section(metrics, section):
    for name, rule in metrics["decision_rules"].items():
        assert name in section, f"decision rule {name} is not scored in the section"
    assert metrics["verdict"]["verdict"] in section


def test_the_section_has_the_required_structure(section):
    for heading in ("What C31 changes elsewhere", "Limitations", "REPRODUCE"):
        assert heading in section, f"the section is missing its {heading!r} part"


def test_the_owner_instruction_lifting_the_no_second_generator_rule_is_stated(section):
    low = section.lower()
    assert "no second generator" in low
    assert "owner" in low
