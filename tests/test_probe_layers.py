"""C17 -- the probe-layer sweep, bound to its artefacts.

Two kinds of test live here, kept apart on purpose.

**Contracts.**  The multi-layer extractor has to be the single-layer extractor at the
final probe point, and the C17.0 decision rules have to fire where the pre-registration
says they fire.  If either drifts, every cross-layer comparison in
`reports/section_c17_probe_layers.md` is measuring the tooling rather than the model.

**Artefact binding.**  Every number asserted in that section file is re-read from the
JSON it came from, formatted the way the section formats it, and required to appear in
the text -- the pattern `tests/test_report_matches_artifacts.py` uses, for the same
reason: hand-transcription is the one error mode no amount of reasoning about the
pipeline rules out.  These skip when the artefacts are absent, so a fresh clone passes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from property_to_go import generation
from property_to_go import probe_layers as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "pilot_report.md"  # C17 merged as section 21

STATES_DIR = OUT / "c17_layer_states_pilot_50k_p2"
SWEEP_DIR = OUT / "c17_probe_layers"
STEER_DIR = OUT / "c17_layer_steering"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def section_text() -> str:
    if not SECTION.exists():
        pytest.skip("section not written yet")
    return SECTION.read_text()


@pytest.fixture(scope="module")
def sweep():
    return _load(SWEEP_DIR / "probe_layer_metrics.json")


@pytest.fixture(scope="module")
def steering():
    return _load(STEER_DIR / "layer_steering_metrics.json")


def _present(text: str, value: float, fmt: str, label: str) -> None:
    rendered = fmt % value
    assert rendered in text, f"{label}: section is missing {rendered}"


# =================================================================================
# Contracts
# =================================================================================

PREFIXES = [
    "CCOc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc2[nH]ccc2c1",
    "CN1CCN(CC1)c1ncnc2[nH]ccc12",
]


@pytest.mark.model
def test_the_multilayer_extractor_is_the_single_layer_extractor_at_the_final_point(generator):
    """C17's whole premise: 13 probe points cost one pass and change nothing at 12.

    If this drifts, layer 12 of the sweep stops being the number pilot_report.md §13
    reports and every "layer L versus the final layer" comparison silently changes its
    reference point.
    """
    seqs = [generator.tokenizer(s, return_tensors="pt").input_ids[0, :-1].tolist()
            for s in PREFIXES]
    positions = [[1, len(s) - 1] for s in seqs]
    layers = P.probe_points(generator)
    assert len(layers) == generator.model.config.num_hidden_layers + 1

    multi = P.hidden_states_all_layers(generator, seqs, positions, layers, batch_size=4)
    single = generation.hidden_states_for_positions(
        generator, seqs, positions, layer=-1, batch_size=4
    )
    for got, want in zip(multi[layers[-1]], single):
        assert np.array_equal(got, want), "probe point 12 is not `layer=-1`"

    # And the intermediate points are genuinely different representations, not aliases.
    assert not np.allclose(multi[0][0], multi[layers[-1]][0])
    assert not np.allclose(multi[6][0], multi[layers[-1]][0])


@pytest.mark.model
def test_the_multilayer_extractor_is_batch_size_invariant(generator):
    """Right padding is exact, so the probe-point sweep must not depend on batching."""
    seqs = [generator.tokenizer(s, return_tensors="pt").input_ids[0, :-1].tolist()
            for s in PREFIXES]
    positions = [[len(s) - 1] for s in seqs]
    a = P.hidden_states_all_layers(generator, seqs, positions, [0, 6, 12], batch_size=4)
    b = P.hidden_states_all_layers(generator, seqs, positions, [0, 6, 12], batch_size=1)
    for L in (0, 6, 12):
        for x, y in zip(a[L], b[L]):
            assert np.abs(x - y).max() < 1e-5


def test_the_bootstrap_is_script_03s_bootstrap():
    """C17 reuses script 03's paired bootstrap; a divergent copy would be invisible."""
    spec = importlib.util.spec_from_file_location(
        "_script03", ROOT / "scripts" / "03_train_heads.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    fn = lambda x: float(np.mean(x))  # noqa: E731
    theirs = mod.paired_bootstrap_diff(fn, (a,), (b,), n_boot=200, seed=3)
    mine = P.paired_bootstrap_diff(fn, (a,), (b,), n_boot=200, seed=3)
    for k in ("mean", "lo", "hi"):
        assert theirs[k] == pytest.approx(mine[k], abs=1e-12)


def test_the_bonferroni_level_widens_the_interval():
    """C17.0.6 rule 1 is only a correction if the corrected interval is wider."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 1.0, size=500)
    b = rng.normal(0.0, 1.0, size=500)
    fn = lambda x: float(np.mean(x))  # noqa: E731
    nominal = P.paired_bootstrap_diff(fn, (a,), (b,), n_boot=400, seed=5)
    bonf = P.paired_bootstrap_diff(fn, (a,), (b,), n_boot=400, seed=5, alpha=0.05 / 13)
    assert bonf["lo"] < nominal["lo"]
    assert bonf["hi"] > nominal["hi"]


def test_the_crossover_rule_fires_exactly_where_the_preregistration_says():
    """The three branches of C17.0.4, including both boundaries.

    Written against the pre-registration's literal thresholds rather than against the
    observed numbers, so editing the rule to match data fails a test instead of passing
    quietly -- the same guard `test_the_pre_registration_is_pinned_literally` gives
    PREDICTED_LOCALITY_ORDER.
    """
    triv = 0.8269
    # Clearly better than trivial: ARTEFACT.
    v = P.crossover_verdict({12: 0.7878, 6: triv + 0.02}, triv)
    assert v["verdict_auroc_arm"] == "ARTEFACT" and v["best_layer"] == 6
    # Exactly on the material margin: still ARTEFACT (the rule is >=).
    assert P.crossover_verdict({6: triv + 0.010}, triv)["verdict_auroc_arm"] == "ARTEFACT"
    # Just inside it: TIE.
    assert (P.crossover_verdict({6: triv + 0.0099}, triv)["verdict_auroc_arm"]
            == "TIE_AT_THE_BEST_LAYER")
    # One seed sd below trivial: still TIE.
    assert (P.crossover_verdict({6: triv - 0.004}, triv)["verdict_auroc_arm"]
            == "TIE_AT_THE_BEST_LAYER")
    # More than one seed sd below: REPRESENTATION, the claim-strengthening branch.
    assert (P.crossover_verdict({12: 0.7878, 6: triv - 0.0041}, triv)["verdict_auroc_arm"]
            == "REPRESENTATION")
    assert P.SEED_SD == 0.004 and P.MATERIAL_MARGIN == 0.010


def test_an_isolated_maximum_is_rejected_as_noise():
    """C17.0.6 rule 2: one layer up, neighbours flat, is not a representational effect."""
    ref = 12
    spike = {12: 0.70, 5: 0.70, 6: 0.72, 7: 0.70}
    r = P.isolated_spike(spike, 6, ref)
    assert r["clears_material_margin"] and not r["neighbours_support_it"]
    assert r["genuinely_better"] is False

    broad = {12: 0.70, 5: 0.708, 6: 0.72, 7: 0.707}
    r2 = P.isolated_spike(broad, 6, ref)
    assert r2["neighbours_support_it"] and r2["genuinely_better"] is True

    # A large but unsupported gain is still rejected; the margin alone is not enough.
    assert P.isolated_spike({12: 0.70, 5: 0.69, 6: 0.90, 7: 0.69}, 6, ref)[
        "genuinely_better"] is False


def test_script_03_still_defaults_to_the_datasets_own_hidden_states():
    """The layer flag must be additive: absent, script 03 reads `hidden.npy`."""
    src = (ROOT / "scripts" / "03_train_heads.py").read_text()
    assert '"--hidden-file", default=None' in src
    assert 'hidden_path = data_dir / "hidden.npy"' in src
    assert '"hidden_file_is_dataset_default"' in src


def test_script_02_still_defaults_to_writing_one_hidden_array():
    """Same for script 02: `--layers` omitted must leave the existing path untouched."""
    src = (ROOT / "scripts" / "02_generate_trajectories.py").read_text()
    assert '"--layers", type=int, nargs="*", default=None' in src
    assert "if args.layers:" in src
    assert "extra_layers: list[int] = []" in src


# =================================================================================
# Artefact binding -- step 1, extraction
# =================================================================================

def test_the_replay_reproduces_the_datasets_own_hidden_states_bit_for_bit():
    """C17.0.2 gate 1. Without this the sweep measures the replay, not the layer."""
    s = _load(STATES_DIR / "layer_states_summary.json")
    g = s["validity_gate"]
    assert g["checked"] is True
    assert g["bit_identical"] is True
    assert g["max_abs_difference"] == 0.0
    assert g["probe_point"] == 12


def test_thirteen_probe_points_cost_the_tokens_of_one(section_text):
    s = _load(STATES_DIR / "layer_states_summary.json")
    assert s["n_probe_points"] == 13
    assert s["probe_points"] == list(range(13))
    assert (s["processed_tokens_if_one_pass_per_layer"]
            == s["processed_tokens_actual"] * 13)
    _present(section_text, s["processed_tokens_actual"], "%d", "extraction tokens")
    _present(section_text, s["processed_tokens_if_one_pass_per_layer"], "%d",
             "13-pass token cost")
    _present(section_text, s["n_prefix_rows"], "%d", "prefix rows")
    _present(section_text, s["wall_seconds"], "%.1f", "extraction wall seconds")


def test_the_run_cost_and_seed_spread_quoted_in_the_section_are_the_artefacts(
    sweep, steering, section_text
):
    """The remaining asserted scalars: training cost, seed spread, steering token cost."""
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    _present(section_text, sweep["wall_seconds_total"], "%.1f", "sweep wall seconds")
    _present(section_text, steering["compute"]["processed_tokens_actual"], "%d",
             "steering tokens")

    worst = max(
        (sweep["properties"][p]["layers"][str(L)]["across_seeds"]["auroc"]["std"], p, L)
        for p in PREDICTED_LOCALITY_ORDER for L in range(13)
    )
    assert worst[0] < 0.0058, worst
    assert (worst[1], worst[2]) == ("hbd_count", 8), worst
    _present(section_text, worst[0], "%.4f", "largest seed sd")


# =================================================================================
# Artefact binding -- step 2, the sweep
# =================================================================================

def test_probe_point_12_reproduces_the_phase_2_head_metrics(sweep):
    """C17.0.2 gate 2: the per-layer trainer IS script 03's trainer."""
    g = sweep["validity_gate"]
    assert g["checked"] is True
    assert g["reproduces_to_4dp"] is True
    assert len(g["per_property"]) == 6
    for row in g["per_property"]:
        assert row["abs_auroc_difference"] < 5e-5, row
        assert row["abs_nll_difference"] < 5e-5, row


def test_the_sweep_covers_every_probe_point_and_the_whole_battery(sweep):
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    assert sweep["probe_points"] == list(range(13))
    assert sweep["head_seeds"] == [1234, 2345, 3456]
    assert set(sweep["properties"]) == set(PREDICTED_LOCALITY_ORDER)
    for prop in PREDICTED_LOCALITY_ORDER:
        assert set(sweep["properties"][prop]["layers"]) == {str(L) for L in range(13)}
        for L in range(13):
            assert (sweep["properties"][prop]["layers"][str(L)]["across_seeds"]["n_seeds"]
                    == 3)


def test_the_preregistered_thresholds_were_not_edited_after_the_fact(sweep):
    pre = sweep["preregistration"]
    assert pre["seed_sd"] == 0.004
    assert pre["material_margin"] == 0.010
    assert pre["neighbour_margin"] == 0.005
    assert pre["bonferroni_family_size"] == 13
    assert pre["primary_metric"].startswith("held-out target-interval AUROC")


def test_the_aromatic_ring_verdict_matches_what_the_section_claims(sweep, section_text):
    """The headline. Bound to the artefact, in both directions.

    `verdict_auroc_arm` is recomputed here from the stored per-layer means rather than
    trusted, so a hand-edited verdict field would fail.
    """
    v = sweep["verdicts"]["aromatic_rings"]
    auroc_by_layer = {int(k): x for k, x in v["auroc_by_layer"].items()}
    triv = sweep["properties"]["aromatic_rings"]["trivial"]["across_seeds"]["auroc"]["mean"]
    recomputed = P.crossover_verdict(auroc_by_layer, triv)
    assert recomputed["verdict_auroc_arm"] == v["verdict_auroc_arm"]
    assert recomputed["best_layer"] == v["best_layer"]
    assert v["verdict_auroc_arm"] in section_text
    _present(section_text, v["best_auroc"], "%.4f", "best aromatic-ring AUROC")
    _present(section_text, v["trivial_auroc"], "%.4f", "trivial aromatic-ring AUROC")
    _present(section_text, v["margin_over_trivial"], "%+.4f", "margin over trivial")
    _present(section_text, v["best_layer"], "%d", "best layer")


def test_every_layers_aromatic_ring_auroc_appears_in_the_section(sweep, section_text):
    """The full depth curve is in the section, so no probe point can be quietly dropped."""
    layers = sweep["properties"]["aromatic_rings"]["layers"]
    for L in range(13):
        _present(section_text, layers[str(L)]["across_seeds"]["auroc"]["mean"], "%.4f",
                 f"aromatic_rings L{L} AUROC")


def test_the_per_property_best_layers_appear_in_the_section(sweep, section_text):
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    for prop in PREDICTED_LOCALITY_ORDER:
        v = sweep["verdicts"][prop]
        _present(section_text, v["best_auroc"], "%.4f", f"{prop} best AUROC")
        _present(section_text, v["trivial_auroc"], "%.4f", f"{prop} trivial AUROC")
        _present(section_text, v["auroc_by_layer"]["12"], "%.4f", f"{prop} L12 AUROC")


def test_the_no_isolated_spike_rule_as_the_section_reports_it(sweep, section_text):
    """C17.0.6 rule 2 applied to every property, bound to §C17.4.2's table.

    Five of six best layers are neighbour-supported improvements over the final layer;
    cLogP is the one exception and fails on the material margin rather than on
    smoothness. Asserted in **both** directions, so a future re-run that flips any cell
    fails here instead of leaving the section's table standing unchallenged.
    """
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    supported = {
        prop: sweep["verdicts"][prop]["spike_check_vs_reference_layer"]["genuinely_better"]
        for prop in PREDICTED_LOCALITY_ORDER
    }
    assert supported == {
        "aromatic_rings": True, "hbd_count": True, "rotatable_bonds": True,
        "tpsa": True, "clogp": False, "qed": True,
    }, supported
    clogp = sweep["verdicts"]["clogp"]["spike_check_vs_reference_layer"]
    assert clogp["clears_material_margin"] is False
    assert clogp["neighbours_support_it"] is True, "cLogP fails the margin, not smoothness"
    for prop in PREDICTED_LOCALITY_ORDER:
        _present(section_text,
                 sweep["verdicts"][prop]["spike_check_vs_reference_layer"]["gain_over_reference"],
                 "%+.4f", f"{prop} gain over probe point 12")


def test_the_depth_curve_has_the_shape_the_section_claims(sweep):
    """§C17.3.1's three facts, for all six properties, checked against the artefact.

    The shape is what makes C17 credible -- a single lucky layer would not reproduce
    across six properties -- so it is asserted rather than described.
    """
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    for prop in PREDICTED_LOCALITY_ORDER:
        a = {int(L): x for L, x in sweep["verdicts"][prop]["auroc_by_layer"].items()}
        # 1. the peak is mid-network
        assert 3 <= sweep["verdicts"][prop]["argmax_layer_by_auroc"] <= 5, prop
        # 2. monotone decline from the peak to the final layer, up to seed noise
        peak = sweep["verdicts"][prop]["argmax_layer_by_auroc"]
        for L in range(peak, 12):
            assert a[L + 1] <= a[L] + P.SEED_SD, (prop, L, a[L], a[L + 1])
        # 3. the final layer is the worst probe point on the descending stretch
        assert a[12] == min(a[L] for L in range(peak, 13)), prop
        # the embedding control is far below every contextual layer
        assert a[0] < min(a[L] for L in range(1, 13)) - 0.05, prop

    # Two stronger claims the section explicitly refuses to make, pinned so that a
    # future draft cannot quietly reinstate them. Earlier drafts asserted both and these
    # assertions are what caught them.
    ar = {int(L): x for L, x in sweep["verdicts"]["aromatic_rings"]["auroc_by_layer"].items()}
    assert ar[1] > ar[12] and ar[12] == min(ar[L] for L in range(1, 13))
    for prop in PREDICTED_LOCALITY_ORDER:
        if prop == "aromatic_rings":
            continue
        a = {int(L): x for L, x in sweep["verdicts"][prop]["auroc_by_layer"].items()}
        assert a[1] < a[12], f"{prop}: probe point 1 is not the worst contextual layer"
    cl = {int(L): x for L, x in sweep["verdicts"]["clogp"]["auroc_by_layer"].items()}
    assert cl[3] < cl[12], "cLogP probe point 3 is no longer below probe point 12"


def test_the_aromatic_ring_artefact_claim_clears_all_three_side_conditions(sweep, section_text):
    """§C17.4.1. The headline overturn, and every condition C17.0.4 attached to it."""
    v = sweep["verdicts"]["aromatic_rings"]
    assert v["artefact_claim_fires"] is True
    assert v["metric_dependent"] is False
    assert v["margin_over_trivial"] >= P.MATERIAL_MARGIN
    bonf = v["bonferroni_ci_at_best_layer"]
    assert bonf["lo"] > 0
    assert bonf["alpha"] == pytest.approx(0.05 / 13)
    assert v["nll_not_worse_than_trivial"] is True
    assert v["spike_check_vs_reference_layer"]["genuinely_better"] is True
    _present(section_text, bonf["lo"], "%+.4f", "aromatic-ring Bonferroni CI lo")
    _present(section_text, bonf["hi"], "%+.4f", "aromatic-ring Bonferroni CI hi")
    _present(section_text, v["nll_at_best_layer"], "%.4f", "aromatic-ring best-layer NLL")


def test_the_probe_point_12_column_still_equals_the_reported_phase_2_numbers(sweep, section_text):
    """§C17.2's gate table. Not 4 dp -- identical, and the section says so."""
    for row in sweep["validity_gate"]["per_property"]:
        assert row["abs_auroc_difference"] == 0.0, row
        assert row["abs_nll_difference"] == 0.0, row
        _present(section_text, row["auroc"], "%.4f", f"{row['property']} gate AUROC")
        _present(section_text, row["nll"], "%.4f", f"{row['property']} gate NLL")


# =================================================================================
# Artefact binding -- step 3, steering value
# =================================================================================

def test_the_steering_recomputation_reproduces_phase_2_at_the_final_layer(steering):
    """Consistency gates: same head_q pipeline, same `our_head_gain`, at probe point 12."""
    g = steering["consistency_gates"]
    assert len(g["head_q_matches_headroom_npz"]) == 6
    for prop, row in g["head_q_matches_headroom_npz"].items():
        assert row["allclose_1e-5"] is True, (prop, row)
    for prop, row in g["our_head_gain_matches_locality"].items():
        assert row["abs_difference"] < 1e-9, (prop, row)


def test_the_layer_is_selected_by_prediction_not_by_steering(steering, sweep):
    """C17.0.5 step 1. Choosing L* on the steering number would be a free parameter."""
    for prop, row in steering["c17_0_5_criterion"]["per_property"].items():
        assert row["L_star_selected_by_auroc"] == sweep["verdicts"][prop]["argmax_layer_by_auroc"]


def test_the_c17_0_5_criterion_is_scored_as_the_section_reports_it(steering, section_text):
    c = steering["c17_0_5_criterion"]
    n_better = sum(1 for v in c["per_property"].values() if v["absolute_improvement"] > 0)
    assert c["n_properties_improved"] == n_better
    assert c["material"] == bool(
        n_better >= 4 and c["median_relative_improvement"] >= 0.25
    )
    verdict = "MATERIAL" if c["material"] else "NOT MATERIAL"
    assert verdict in section_text
    _present(section_text, c["median_relative_improvement"], "%+.3f",
             "median relative improvement")
    _present(section_text, c["n_properties_improved"], "%d", "properties improved")


def test_question_two_failed_and_the_section_says_so(steering, section_text):
    """§C17.5. The negative, asserted as a negative.

    C17.0.7 binds the write-up not to soften a question-2 failure by quoting the AUROC
    table instead, so the failure is pinned to the artefact here: if a re-run ever makes
    it pass, this test fails and forces the section to be rewritten rather than letting
    "NOT MATERIAL" stand next to passing data.
    """
    c = steering["c17_0_5_criterion"]
    assert c["material"] is False
    assert c["n_properties_improved"] == 2
    assert c["median_relative_improvement"] < 0
    assert "NOT MATERIAL" in section_text
    # And the section must not quietly present the post-hoc argmax as the result: the
    # per-property L* it reports has to be the AUROC-selected one.
    for prop, row in c["per_property"].items():
        _present(section_text, row["relative_improvement"], "%+.3f", f"{prop} relative")


def test_the_post_hoc_best_layer_is_labelled_as_post_hoc(steering, section_text):
    """§C17.5.2 quotes the selection-on-outcome reading; it must match the artefact."""
    for prop, best_expected in (("aromatic_rings", 6), ("hbd_count", 6),
                                ("rotatable_bonds", 3)):
        rows = steering["properties"][prop]["layers"]
        best = max(rows, key=lambda k: rows[k]["our_head_gain"])
        assert int(best) == best_expected, (prop, best)
        _present(section_text,
                 rows[best]["our_head_share_of_the_lambda1_optimum"] * 100, "%.1f",
                 f"{prop} post-hoc best share")


def test_the_head_gap_numbers_in_the_section_come_from_the_artefact(steering, section_text):
    from property_to_go.properties import PREDICTED_LOCALITY_ORDER

    for prop in PREDICTED_LOCALITY_ORDER:
        row = steering["c17_0_5_criterion"]["per_property"][prop]
        _present(section_text, row["share_at_reference"] * 100, "%.1f",
                 f"{prop} share at L12")
        _present(section_text, row["share_at_L_star"] * 100, "%.1f",
                 f"{prop} share at L*")


def test_no_generation_happened_in_the_steering_step(steering):
    """The claim that this step needs no new molecules, made checkable.

    `molecules_returned` counts what `ComputeMeter` saw; the only forward passes here
    are the 3,200 extended-prefix state extractions, which return no molecules.
    """
    assert steering["compute"]["molecules_returned"] == 0
    assert steering["n_prefixes_sampled"] == 400
    assert steering["top_k"] == 8
