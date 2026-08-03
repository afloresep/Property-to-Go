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


# --------------------------------------------------------------------------------
# The frozen pre-registration: windows and target intervals were fixed BEFORE any
# guided molecule was inspected, and every phase-1 and phase-2 result is comparable
# only because they have not moved since.
#
# Pinned here as literals, transcribed from the values the pilot committed, so that
# re-deriving them on different hardware or through generalised code is checked rather
# than assumed. This is setup gate (d) of the phase-2 kickoff made permanent: the
# original check was "regenerate and diff against the tracked files", which works once;
# this works every time anyone touches scripts/02 or the interval rules.
# --------------------------------------------------------------------------------
FROZEN_PILOT_INTERVALS = {
    "clogp": (4.172676000000004, 5.038300000000004, 0.09999397868454328),
    "aromatic_rings": (3.0, 4.0, 0.1713867089496819),
    "mol_weight": (435.4494000000001, 486.9970000000001, 0.09999397868454328),
}
FROZEN_PILOT_WINDOWS = {"t33": 15, "t67": 29}


def test_the_frozen_target_intervals_did_not_move():
    intervals = _load("pilot_50k/target_intervals.json")
    for prop, (lo, hi, base_rate) in FROZEN_PILOT_INTERVALS.items():
        got = intervals[prop]
        assert got["lo"] == pytest.approx(lo, abs=1e-9), f"{prop} lo moved"
        assert got["hi"] == pytest.approx(hi, abs=1e-9), f"{prop} hi moved"
        assert got["base_rate"] == pytest.approx(base_rate, abs=1e-9), f"{prop} base rate moved"


def test_the_frozen_windows_did_not_move():
    windows = _load("pilot_50k/windows.json")
    for key, value in FROZEN_PILOT_WINDOWS.items():
        assert windows[key] == value, f"window {key} moved from {value} to {windows[key]}"


def test_the_phase_two_battery_has_frozen_intervals_too():
    """Every property that will be steered needs an interval fixed before it is.

    A missing entry here means a guided run would have had to derive its own target,
    which is the one thing the pre-registration forbids. Checked on the phase-2 dataset,
    which is where the battery lives -- `pilot_50k/` is the pilot's and deliberately
    still carries only its three.
    """
    from property_to_go.properties import LOCALITY_BATTERY

    intervals = _load("pilot_50k_p2/target_intervals.json")
    for prop in LOCALITY_BATTERY:
        assert prop in intervals, f"{prop} has no frozen target interval"
        iv = intervals[prop]
        assert iv["hi"] > iv["lo"], f"{prop} interval is empty"
        assert 0.0 < iv["base_rate"] < 1.0, f"{prop} base rate {iv['base_rate']} is degenerate"
        assert "rule" in iv, f"{prop} does not record the rule that produced it"


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


# ================================================================================
# Phase 2 -- the lexical-locality test. Same contract as above: every number in the
# prose is re-read from its artefact and required to appear, so the report cannot
# drift from the data it claims to describe.
# ================================================================================


class TestPhase2DeviceFindings:
    """Section 11: what moving the frozen generator to a GPU did and did not change."""

    def test_device_equivalence_numbers_match(self, report_text):
        d = _load("device_equivalence/device_equivalence.json")
        _assert_present(
            report_text, d["forward_pass"]["max_abs_logit_difference"], "%.3e",
            "CPU-vs-CUDA logit difference",
        )
        assert str(d["n_parameters"]) in report_text.replace(",", "") or (
            f"{d['n_parameters']:,}" in report_text
        ), "parameter count is not in the report"
        assert str(d["sampling"]["n_molecules"]) in report_text
        assert str(d["sampling"]["n_identical_token_sequences"]) in report_text

    def test_the_numerics_agree_and_only_the_rng_stream_differs(self):
        """Section 11.2's causal claim, asserted rather than narrated.

        The report says the divergence is an RNG-stream effect and not a numerical one.
        That is only defensible if the forward pass agrees AND the candidate set guided
        decoding consumes is identical -- otherwise the method under test would itself
        be device-dependent.
        """
        d = _load("device_equivalence/device_equivalence.json")
        assert d["weights_identical"], "the two devices must hold the same frozen model"
        assert d["forward_pass"]["max_abs_logit_difference"] < 1e-4, (
            "report claims float32 agreement; artefact disagrees"
        )
        assert d["forward_pass"]["all_top_k_candidate_sets_identical"], (
            "report claims the top-k candidate set is device-independent"
        )
        assert d["sampling"]["fraction_identical"] == 0.0, (
            "report claims the same seed draws entirely different molecules"
        )
        assert d["sampling"]["same_device_same_seed_identical"], (
            "report claims within-device determinism still holds"
        )

    def test_wall_time_reproducibility_is_measured_not_assumed(self, report_text):
        """The pilot refused timing claims on its laptop; section 11.4 says the band was
        measured here rather than inherited. So the measurement must exist, and token
        counts must still be identical across identical repeats."""
        d = _load("device_equivalence/device_equivalence.json")["timing_reproducibility"]
        assert d["repeats"] >= 2
        assert d["processed_tokens_identical"], (
            "identical workloads must process identical token counts"
        )
        _assert_present(report_text, d["wall_seconds_relative_spread"] * 100, "%.1f",
                        "wall-time relative spread")

    def test_the_frozen_intervals_were_inherited_not_rederived(self):
        """Section 11.2's remedy. The phase-2 dataset must carry the pilot's intervals
        byte-for-byte, or phase-1 and phase-2 results are not comparable."""
        prov = _load("pilot_50k_p2/target_intervals_provenance.json")
        p2 = _load("pilot_50k_p2/target_intervals.json")
        pilot = _load("pilot_50k/target_intervals.json")
        for prop in ("clogp", "aromatic_rings", "mol_weight"):
            assert prop in prov["inherited_verbatim"], f"{prop} was re-derived"
            assert p2[prop] == pilot[prop], f"{prop} interval is not byte-identical"

    def test_the_reported_interval_divergence_matches_the_artifact(self, report_text):
        """The table of re-derived intervals in section 11.2 must be the real numbers."""
        prov = _load("pilot_50k_p2/target_intervals_provenance.json")
        for prop, fmt in (("clogp", "%.6f"), ("mol_weight", "%.4f")):
            d = prov["derived_values_for_comparison"][prop]
            _assert_present(report_text, d["lo"], fmt, f"{prop} re-derived lo")
            _assert_present(report_text, d["hi"], fmt, f"{prop} re-derived hi")
        _assert_present(
            report_text,
            prov["derived_values_for_comparison"]["aromatic_rings"]["base_rate"],
            "%.6f", "aromatic rings re-derived base rate",
        )

    def test_the_gpu_reproduces_the_pilots_intervention_effect(self, report_text):
        """Setup gate (e). The effect size, not the individual hit rates, is the claim."""
        cpu = _load("pilot_50k_guided_aromatic_rings/guidance_metrics.json")["conditions"]
        gpu = _load(
            "pilot_50k_gpucheck_guided_aromatic_rings/guidance_metrics.json"
        )["conditions"]
        effects = {}
        for name, c in (("cpu", cpu), ("gpu", gpu)):
            effects[name] = (
                c["throughout"]["aggregate"]["hit_rate"]["mean"]
                - c["unguided"]["aggregate"]["hit_rate"]["mean"]
            )
            _assert_present(report_text, effects[name], "%+.4f", f"{name} guidance effect")
        assert abs(effects["gpu"] - effects["cpu"]) < 0.05, (
            f"report claims the effect reproduces, but {effects} disagree"
        )
        for c in (cpu, gpu):
            for cond in ("unguided", "throughout"):
                _assert_present(
                    report_text, c[cond]["aggregate"]["hit_rate"]["mean"], "%.4f",
                    f"{cond} hit rate",
                )


class TestPhase2IntervalMaskDefect:
    """Section 11.5: the target interval was not a union of bins."""

    def test_every_phase_two_head_represents_its_target_exactly(self):
        """The invariant the pilot violated, now enforced. This is the important one:
        if it fails, some head is being trained to predict the wrong event."""
        heads = _load("pilot_50k_heads_p2/head_metrics.json")
        assert not heads["legacy_interval_mask"]
        for prop, pr in heads["properties"].items():
            cov = pr["interval_mask_coverage"]
            assert cov["is_exact"], f"{prop}: {cov}"
            assert cov["masked_rate"] == pytest.approx(cov["true_rate"], abs=1e-6), prop
            assert cov["n_bins_selected"] >= 1, prop

    def test_the_defect_is_reproduced_and_its_cost_measured(self, report_text):
        """Section 11.7. The legacy run must actually exhibit the defect, or the
        comparison measures nothing."""
        d = _load("interval_mask_impact/interval_mask_impact.json")
        assert d["affected_properties"], (
            "the legacy run shows no misalignment, so it is not reproducing the defect"
        )
        assert "clogp" in d["affected_properties"], (
            "cLogP is the property the pilot's defect landed on"
        )
        for prop in ("aromatic_rings",):
            assert prop in d["unaffected_properties"], (
                f"report claims {prop} was immune (CategoricalBinner is exact)"
            )
        for prop in d["affected_properties"]:
            e = d["properties"][prop]
            _assert_present(report_text, e["target_auroc"]["fixed"], "%.4f",
                            f"{prop} fixed-mask AUROC")
            _assert_present(report_text, e["target_auroc"]["legacy"], "%.4f",
                            f"{prop} legacy-mask AUROC")

    def test_the_control_metric_confirms_only_the_mask_moved(self):
        """The comparison is only interpretable if nothing but the mask differs.

        NLL looks like the obvious control and is not one: aligning the binner adds two
        bins (22 against 20), and a finer partition has higher entropy, so NLL is expected
        to move. Expected-value MAE is in property units and is nearly binning-invariant,
        so it is the control that actually controls. Section 11.7 finding 4.
        """
        d = _load("interval_mask_impact/interval_mask_impact.json")
        for prop, e in d["properties"].items():
            mae = e["control_expected_value_mae"]
            scale = max(abs(mae["fixed"]), 1e-9)
            assert abs(mae["difference"]) / scale < 5e-3, (
                f"{prop}: expected-value MAE moved by {mae['difference']}, so the two runs "
                "differ by more than the interval mask"
            )
        # And the affected properties must gain bins, which is the mechanism.
        for prop in d["affected_properties"]:
            n = d["properties"][prop]["n_bins"]
            assert n["fixed"] > n["legacy"], prop

    def test_the_defect_is_structural_for_continuous_targets_not_bad_luck(self):
        """Section 11.7 finding 1: every quantile-band target loses exactly one of its two
        bins, and every count target is immune. If this ever fails, the account of *why*
        the pilot hit this is wrong even if the numbers still match."""
        from property_to_go.properties import DISCRETE_PROPERTIES

        d = _load("interval_mask_impact/interval_mask_impact.json")
        for prop, e in d["properties"].items():
            if prop in DISCRETE_PROPERTIES:
                assert not e["affected"], f"{prop} is a count property and must be immune"
            else:
                assert e["affected"], f"{prop} is a quantile band and should have been hit"
                assert e["n_bins_selected"]["legacy"] == 1
                # One of the band's two bins is lost, so roughly half the target mass.
                # Not exactly half for every property: the two bins are equal-mass on the
                # train split the binner was fitted to, and this is the test split.
                assert e["masked_rate_ratio_legacy_over_true"] == pytest.approx(0.5, abs=0.05)

    def test_discrimination_barely_moved_so_the_pilots_aurocs_stand(self):
        """Section 11.7 finding 3, which corrects section 11.5's first guess.

        The report now says the pilot's cLogP AUROCs are sound as reported. That claim is
        only honest if the measured shift really is negligible.
        """
        d = _load("interval_mask_impact/interval_mask_impact.json")
        for prop in d["affected_properties"]:
            shift = d["properties"][prop]["target_auroc"]["difference"]
            assert abs(shift) < 0.01, (
                f"{prop}: AUROC moved by {shift:+.4f}; the report claims the pilot's "
                "AUROCs stand, which would no longer be safe"
            )

    def test_the_legacy_mask_understates_the_target_probability(self, report_text):
        """Section 11.5's mechanical account of the pilot's 'under-confident head'."""
        d = _load("interval_mask_impact/interval_mask_impact.json")
        e = d["properties"]["clogp"]
        ratio = e["masked_rate_ratio_legacy_over_true"]
        assert ratio < 0.75, (
            f"report claims the legacy mask covered roughly half the target; ratio={ratio}"
        )
        mp = e["mean_predicted_target_prob"]
        assert mp["legacy"] < mp["fixed"], (
            "the legacy head must be the more under-confident of the two"
        )
        _assert_present(report_text, ratio, "%.3f", "clogp legacy mask coverage ratio")


class TestPhase2HeadReplication:
    """Section 12: the phase-2 battery, trained on an independent 50k sample."""

    def test_head_seed_spread_is_reported(self, report_text):
        heads = _load("pilot_50k_heads_p2/head_metrics.json")
        assert heads["seeds_initialisation"], (
            "phase-2 replicates must seed initialisation, not only shuffling"
        )
        assert len(heads["head_seeds"]) >= 3
        for prop, pr in heads["properties"].items():
            for name in ("frozen_state", "trivial"):
                a = pr["heads"][name].get("across_seeds")
                assert a is not None, f"{prop}/{name} has no seed replicates"
                assert a["n_seeds"] == len(heads["head_seeds"])

    def test_the_aromatic_ring_crossover_replicates_on_an_independent_sample(self):
        """The pilot's most-defended claim, re-measured on a different 50k draw with a
        different head-initialisation seed. Immune to the section 11.5 defect by
        construction, so this is a clean replication."""
        pr = _load("pilot_50k_heads_p2/head_metrics.json")["properties"]["aromatic_rings"]
        frozen = pr["heads"]["frozen_state"]["across_seeds"]
        trivial = pr["heads"]["trivial"]["across_seeds"]
        assert trivial["nll"]["mean"] < frozen["nll"]["mean"], (
            "token counting must still beat the frozen state on aromatic rings"
        )
        assert trivial["auroc"]["mean"] > frozen["auroc"]["mean"]
        assert not pr["frozen_vs_trivial"]["passes_gate"]

    @pytest.mark.parametrize("prop", ["clogp", "aromatic_rings", "hbd_count",
                                      "rotatable_bonds", "tpsa", "qed"])
    def test_phase2_head_numbers_match(self, report_text, prop):
        pr = _load("pilot_50k_heads_p2/head_metrics.json")["properties"][prop]
        for name in ("frozen_state", "trivial"):
            a = pr["heads"][name]["across_seeds"]
            _assert_present(report_text, a["nll"]["mean"], "%.4f", f"{prop}/{name} NLL")
            _assert_present(report_text, a["auroc"]["mean"], "%.4f", f"{prop}/{name} AUROC")


class TestPhase2Headroom:
    """Section 15: the head-free, lambda-free ceiling on one-step steering."""

    @pytest.fixture(scope="class")
    def hr(self):
        return _load("pilot_50k_p2_headroom/headroom_metrics.json")

    @pytest.fixture(scope="class")
    def battery(self):
        from property_to_go.properties import LOCALITY_BATTERY

        return list(LOCALITY_BATTERY)

    def test_headroom_numbers_match(self, report_text, hr, battery):
        for prop in battery:
            c = hr["properties"][prop]["capture"]["overall"]
            for key, fmt in (("base_policy_target_prob", "%.4f"),
                             ("guided_target_prob", "%.4f"),
                             ("best_candidate_target_prob", "%.4f"),
                             ("captured_fraction", "%.4f")):
                _assert_present(report_text, c[key], fmt, f"{prop}/{key}")
            _assert_present(report_text, c["available_excess_mean"], "%+.4f",
                            f"{prop}/available")

    def test_relative_headroom_numbers_match(self, report_text, hr, battery):
        for prop in battery:
            o = hr["properties"][prop]["headroom"]["overall"]
            _assert_present(report_text, o["relative_headroom_raw_mean"], "%.3f",
                            f"{prop}/relative headroom raw")
            _assert_present(report_text, o["relative_headroom_excess_mean"], "%.3f",
                            f"{prop}/relative headroom excess")

    def test_there_is_a_lever_for_every_property(self, hr, battery):
        """Section 15.1's central claim, asserted rather than narrated.

        This is the one that settles the pilot's unanswerable question. If it ever fails,
        the "no lever to pull" explanation is back on the table and section 15.1 is wrong.
        """
        for prop in battery:
            c = hr["properties"][prop]["capture"]["overall"]
            assert c["available_excess_mean"] > 0.02, (
                f"{prop}: no noise-corrected headroom, so 'there is a lever' is false"
            )
            ratio = c["best_candidate_target_prob"] / c["base_policy_target_prob"]
            assert ratio > 1.9, (
                f"{prop}: report claims the best candidate roughly doubles to triples the "
                f"target probability; measured ratio is {ratio:.2f}"
            )
            assert ratio < 3.6, f"{prop}: ratio {ratio:.2f} exceeds the claimed range"

    def test_capture_is_uniformly_low(self, hr, battery):
        """Section 15.1: between 4.8% and 10.9%, a factor of 2.3 across six properties."""
        caps = {p: hr["properties"][p]["capture"]["overall"]["captured_fraction"]
                for p in battery}
        assert all(v is not None for v in caps.values())
        assert max(caps.values()) < 0.15, f"report claims capture stays low: {caps}"
        assert min(caps.values()) > 0.02, caps
        assert max(caps.values()) / min(caps.values()) < 3.0, (
            f"report claims the range is narrow (a factor of ~2.3): {caps}"
        )

    def test_the_ceiling_is_not_survivorship(self, battery):
        """Section 15.3. Recomputed from the arrays, because the claim is about a
        quantity the metrics JSON does not carry."""
        import numpy as np

        p = OUT / "pilot_50k_p2_headroom/headroom_arrays.npz"
        if not p.exists():
            pytest.skip("headroom arrays not produced yet")
        a = np.load(p)
        n_valid, k = a["n_valid"], 16
        for prop in battery:
            ph = a[f"p_hit_{prop}"]
            scored = np.isfinite(ph)
            filled = np.where(scored, ph, 0.0)
            usable = (scored.sum(axis=1) >= 2) & scored.all(axis=1)
            argmax = filled.argmax(axis=1)
            v = (n_valid[np.arange(len(ph)), argmax] / k)[usable].mean()
            assert v > 0.95, (
                f"{prop}: the ceiling-setting candidate has validity {v:.4f}, so the "
                "ceiling may be a survivorship artefact after all"
            )

    def test_p1_is_falsified_in_the_pre_registered_units(self, hr, battery):
        """Section 15.4. The report says the width-normalised ordering is very nearly the
        exact reverse of the prediction. That is a strong claim and must be checked."""
        from scipy.stats import spearmanr

        from property_to_go.properties import PREDICTED_LOCALITY_ORDER

        loc = [hr["properties"][p]["headroom"]["overall"]["relative_headroom_excess_mean"]
               for p in battery]
        predicted = [-(PREDICTED_LOCALITY_ORDER.index(p) + 1) for p in battery]
        rho = spearmanr(predicted, loc).statistic
        assert rho < -0.8, f"report claims rho is about -0.886; measured {rho:+.3f}"

    def test_the_width_normalisation_largely_measures_width(self, hr, battery):
        """Section 15.4's mechanism, and section 12.3's pre-data flag."""
        from scipy.stats import spearmanr

        loc = [hr["properties"][p]["headroom"]["overall"]["relative_headroom_excess_mean"]
               for p in battery]
        inv_width = [1.0 / hr["properties"][p]["interval_width"] for p in battery]
        rho = spearmanr(inv_width, loc).statistic
        assert rho > 0.5, f"report claims rho is about +0.698; measured {rho:+.3f}"

    def test_probability_units_rank_aromatic_rings_first(self, hr, battery):
        """The pre-registered top of the ordering does hold in band-width-free units."""
        locp = {p: hr["properties"][p]["headroom_probability_units"]["overall"][
            "headroom_excess_mean"] for p in battery}
        assert max(locp, key=locp.get) == "aromatic_rings", locp

    def test_p2_fails_because_diffuse_headroom_rises_rather_than_declines(self, hr):
        """Section 15.5. P2 predicted a decline for the diffuse properties; they rise.

        The direction is the whole point, so it is asserted, not just tabulated.
        """
        from property_to_go.properties import (
            P2_DIFFUSE_PROPERTIES, P2_LOCAL_COUNT_PROPERTIES,
        )

        def q4_minus_q1(prop, block="headroom", key="relative_headroom_excess_mean"):
            byq = hr["properties"][prop][block]["by_quartile"]
            return byq["4"][key] - byq["1"][key]

        for prop in P2_DIFFUSE_PROPERTIES:
            assert q4_minus_q1(prop) > -0.01, (
                f"{prop}: P2 predicted a decline and the report says it rises instead"
            )
        for prop in P2_LOCAL_COUNT_PROPERTIES:
            assert abs(q4_minus_q1(prop)) < 0.15, (
                f"{prop}: the report calls the local counts essentially flat"
            )
        # And in probability units every property rises.
        for prop in list(P2_DIFFUSE_PROPERTIES) + list(P2_LOCAL_COUNT_PROPERTIES):
            assert q4_minus_q1(prop, "headroom_probability_units",
                               "headroom_excess_mean") > 0, prop

    def test_clogp_late_headroom_exceeds_a_full_interval_width(self, hr):
        """Section 15.5's refutation of the phase-1 explanation for cLogP's null late
        result. Phase 1 argued no remaining choice could move cLogP by an interval width;
        at Q4 a single token choice moves it by more than one."""
        byq = hr["properties"]["clogp"]["headroom"]["by_quartile"]
        v = {q: byq[str(q)]["relative_headroom_excess_mean"] for q in (1, 2, 3, 4)}
        for q in (2, 3, 4):
            assert v[q] > 1.0, (
                f"report claims cLogP's relative headroom exceeds one interval width from "
                f"Q2 onward; Q{q} measured {v[q]:.3f}"
            )
        assert v[4] > v[1], (
            f"report claims it is higher late than early: Q1={v[1]:.3f} Q4={v[4]:.3f}"
        )

    def test_the_base_policy_is_concentrated_which_is_why_lambda_matters(self, battery):
        """Section 15.2: median 0.90 of the top-8 mass on one token. This is the fact that
        makes 'lambda = 1 caps capture' a live third explanation."""
        import numpy as np

        p = OUT / "pilot_50k_p2_headroom/headroom_arrays.npz"
        if not p.exists():
            pytest.skip("headroom arrays not produced yet")
        lp = np.load(p)["candidate_base_logprobs"]
        w = np.exp(lp - lp.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        assert w.max(axis=1).mean() > 0.7
        assert np.median(w.max(axis=1)) > 0.85


class TestPhase2LambdaVersusHead:
    """Section 15.6: splitting the capture loss between lambda = 1 and the head."""

    @pytest.fixture(scope="class")
    def loc(self):
        return _load("pilot_50k_p2_locality/locality_metrics.json")

    def test_the_split_numbers_match(self, report_text, loc):
        for prop, r in loc["properties"].items():
            e = r["lambda1_ceiling_analysis"]
            for key, fmt in (("our_head_gain", "%+.4f"),
                             ("oracle_head_gain_raw", "%+.4f"),
                             ("oracle_head_gain_null", "%+.4f"),
                             ("oracle_head_gain", "%+.4f"),
                             ("noise_corrected_ceiling_gain", "%+.4f")):
                _assert_present(report_text, e[key], fmt, f"{prop}/{key}")
            for key in ("fraction_of_ceiling_lambda1_permits",
                        "our_head_share_of_the_lambda1_optimum"):
                _assert_present(report_text, e[key] * 100, "%.1f", f"{prop}/{key}")

    def test_both_constraints_bind_and_the_head_binds_harder(self, loc):
        """Section 15.6's central claim. If this flips, the recommendation to deprioritise
        the lambda sweep flips with it, so it is asserted rather than narrated."""
        for prop, r in loc["properties"].items():
            e = r["lambda1_ceiling_analysis"]
            lam_permits = e["fraction_of_ceiling_lambda1_permits"]
            head_gets = e["our_head_share_of_the_lambda1_optimum"]
            assert 0.25 < lam_permits < 0.60, (
                f"{prop}: report claims lambda=1 permits 32-53% of the ceiling; "
                f"measured {lam_permits:.3f}"
            )
            assert 0.08 < head_gets < 0.25, (
                f"{prop}: report claims the head gets 12-22% of that; measured {head_gets:.3f}"
            )
            assert head_gets < lam_permits, (
                f"{prop}: report claims the head is the larger loss of the two"
            )

    def test_the_oracle_null_reconstructs_hit_counts_exactly(self):
        """`permutation_null_oracle_gain` is handed `n_valid` as the per-candidate sample
        size and `rint(p_hit * n_valid)` as the hit count. That is only correct while
        every parseable molecule also has a value for every property -- QED is the one
        that can raise on its own. It holds here (zero QED failures across the bank), but
        silently, so it is asserted: if a future property fails on some molecules the
        rounding would distort the null rather than error."""
        import numpy as np

        p = OUT / "pilot_50k_p2_headroom/headroom_arrays.npz"
        if not p.exists():
            pytest.skip("headroom arrays not produced yet")
        a = np.load(p)
        nv = a["n_valid"]
        for prop in ("aromatic_rings", "hbd_count", "rotatable_bonds",
                     "tpsa", "clogp", "qed"):
            ph = a[f"p_hit_{prop}"]
            sc = np.isfinite(ph)
            prod = ph[sc] * nv[sc]
            assert np.abs(prod - np.rint(prod)).max() == 0.0, (
                f"{prop}: p_hit was computed over fewer rollouts than n_valid, so the "
                "oracle null's group sizes are wrong"
            )

    def test_the_oracle_noise_correction_is_material(self, loc):
        """Section 15.6 point 2: skipping the correction would change the conclusion, which
        is the justification for it living in the library with tests."""
        for prop, r in loc["properties"].items():
            e = r["lambda1_ceiling_analysis"]
            ratio = e["oracle_head_gain_null"] / e["oracle_head_gain_raw"]
            assert 0.20 < ratio < 0.65, (
                f"{prop}: the report says the noise floor is 26-57% of the raw oracle gain; "
                f"measured {ratio:.3f}"
            )
            assert e["oracle_head_gain"] < e["oracle_head_gain_raw"]
        # And the report's claim about what skipping the correction would have concluded.
        uncorrected = {
            p: r["lambda1_ceiling_analysis"]["oracle_head_gain_raw"]
            / r["lambda1_ceiling_analysis"]["noise_corrected_ceiling_gain"]
            for p, r in loc["properties"].items()
        }
        assert max(uncorrected.values()) > 1.0, (
            "report says the uncorrected split yields a nonsensical value above 100% for at "
            f"least one property: {uncorrected}"
        )


class TestPhase2PerStepIsNotEndToEnd:
    """Section 15.6 consequence 3, added by the audit of the first phase-2 write-up.

    The first draft converted the per-position split ("lambda = 1 permits 32-53% of the
    ceiling, our head gets 12-22% of that") into end-to-end guidance for which experiment
    to run next, and section 17.3 went further and asserted that a lambda sweep "cannot
    close a 0.22-0.36 gap". Neither inference was available from the measurement, and the
    arithmetic below is why. These tests exist so the conflation cannot be reintroduced
    by an edit that looks harmless.
    """

    @pytest.fixture(scope="class")
    def loc(self):
        return _load("pilot_50k_p2_locality/locality_metrics.json")

    def test_end_to_end_lift_is_an_order_of_magnitude_above_the_per_step_gain(self, loc):
        for prop, r in loc["properties"].items():
            s = r["per_step_versus_end_to_end"]
            assert s["amplification"] > 10.0, (
                f"{prop}: the report claims end-to-end lift is 20-48x the per-step gain; "
                f"measured {s['amplification']:.1f}x. If this ever approaches 1 the "
                "per-step decomposition could be read end-to-end after all and section "
                "15.6 consequence 3 needs rewriting, not patching."
            )
            assert s["amplification"] < 60.0, f"{prop}: {s['amplification']:.1f}x"

    def test_linear_transfer_of_the_per_step_ratios_is_impossible(self, loc):
        """The decisive check: if the per-position share transferred linearly, most
        properties would need an end-to-end lift above the arithmetic maximum."""
        impossible = [p for p, r in loc["properties"].items()
                      if r["per_step_versus_end_to_end"]["linear_transfer_is_impossible"]]
        assert len(impossible) >= 4, (
            "the report claims linear transfer of the per-step share is impossible for "
            f"four of six properties at the lambda=1 optimum; it fires for {impossible}"
        )
        for prop, r in loc["properties"].items():
            s = r["per_step_versus_end_to_end"]
            assert (s["implied_lift_if_per_step_ceiling_transferred_linearly"]
                    > s["largest_arithmetically_possible_lift"]), (
                f"{prop}: the report claims ALL six exceed the bound at the ceiling"
            )

    def test_the_report_does_not_reassert_the_withdrawn_inference(self, report_text):
        """Section 17.3's withdrawn sentence, as a string. Crude, and it is meant to be:
        the failure mode is someone restoring a fluent sentence, not a number moving."""
        for banned in ("cannot close a 0.22", "has at most a factor of two to give"):
            occurrences = report_text.count(banned)
            assert occurrences <= 1, (
                f"'{banned}' appears {occurrences} times; it is permitted only once, "
                "inside the paragraph that withdraws it"
            )


class TestPhase2CaptureIsDisclosedHonestly:
    """Section 15.1's caveats, also from the audit.

    Three facts about the headline capture number were in the artefacts but not the
    report: the denominator deviates from the pre-registered formula, the analysis uses a
    subset of the sampled prefixes, and the survivorship table in section 15.3 was
    computed on a *different* subset from the table in section 15.1.
    """

    @pytest.fixture(scope="class")
    def loc(self):
        return _load("pilot_50k_p2_locality/locality_metrics.json")

    def test_the_preregistered_capture_formula_is_reported_too(self, report_text, loc):
        raw = {p: r["captured_fraction_of_headroom_preregistered_formula"]
               for p, r in loc["properties"].items()}
        assert min(raw.values()) > 0.015 and max(raw.values()) < 0.07, raw
        # The report quotes the range as 2.1%-5.8%.
        _assert_present(report_text, min(raw.values()) * 100, "%.1f", "capture raw min")
        _assert_present(report_text, max(raw.values()) * 100, "%.1f", "capture raw max")
        for p, r in loc["properties"].items():
            assert r["captured_fraction_of_headroom"] > raw[p], (
                f"{p}: the report says the noise correction moves capture UP, i.e. makes "
                "our head look better, which is what makes it conservative for 15.6"
            )

    def test_the_two_prefix_counts_are_in_the_report(self, report_text, loc):
        counts = {(r["n_prefixes_capture"], r["n_prefixes_headroom"],
                   r["n_prefixes_sampled"]) for r in loc["properties"].values()}
        assert len(counts) == 1, f"the six properties disagree on prefix counts: {counts}"
        cap, head, total = counts.pop()
        assert cap < head < total
        for v in (cap, head, total):
            assert str(v) in report_text, (
                f"prefix count {v} is not disclosed in the report; section 15.1 must say "
                "which subset each table is computed on"
            )

    def test_the_survivorship_table_uses_the_capture_prefix_set(self, loc):
        """Section 15.3 must reproduce section 15.1's ceiling exactly, because they are
        the same quantity. The first draft differed by up to 0.024 because the two tables
        were computed on different prefix subsets."""
        hr = _load("pilot_50k_p2_headroom/headroom_metrics.json")
        for prop, r in loc["properties"].items():
            s = r["survivorship_check"]
            best = hr["properties"][prop]["capture"]["overall"]["best_candidate_target_prob"]
            assert abs(s["ceiling_valid_only"] - best) < 1e-9, (
                f"{prop}: section 15.3's ceiling {s['ceiling_valid_only']:.4f} does not "
                f"match section 15.1's {best:.4f}; they are the same quantity"
            )
            assert s["n_prefixes"] == r["n_prefixes_capture"]

    def test_the_base_policy_concentration_is_quoted_on_the_right_prefix_set(
        self, report_text, loc
    ):
        """Sections 15.2 and 15.6 argue that lambda = 1 structurally cannot reach the
        ceiling because the base policy is concentrated. The first draft quoted that
        concentration over all 400 sampled prefixes (mean 0.800, median 0.905) while every
        other figure in those sections uses the 267-prefix capture set, where it is
        appreciably lower (0.738 / 0.758) -- i.e. the wider set overstated the argument."""
        import numpy as np

        vals = {r["lambda1_ceiling_analysis"]
                 ["base_policy_weight_on_its_own_top_candidate"]
                for r in loc["properties"].values()}
        assert len(vals) == 1, f"properties disagree on a property-free quantity: {vals}"
        conc = vals.pop()
        _assert_present(report_text, conc, "%.3f", "base-policy concentration")
        assert "median 0.90 of its top-8 mass" not in report_text, (
            "the all-400 median is back in an argument computed on the 267-prefix set"
        )

        p = OUT / "pilot_50k_p2_headroom/headroom_arrays.npz"
        if not p.exists():
            pytest.skip("headroom arrays not produced yet")
        a = np.load(p)
        lp = a["candidate_base_logprobs"]
        w = np.exp(lp - lp.max(1, keepdims=True))
        w = w / w.sum(1, keepdims=True)
        assert w.max(1).mean() > conc, (
            "the all-prefix concentration should exceed the capture-set one; if that "
            "reverses, the correction in section 15.2 needs revisiting"
        )

    def test_counting_invalid_completions_as_misses_barely_moves_the_ceiling(self, loc):
        for prop, r in loc["properties"].items():
            s = r["survivorship_check"]
            drop = s["ceiling_valid_only"] - s["ceiling_invalid_counted_as_miss"]
            assert 0.0 < drop < 0.01, f"{prop}: ceiling drops by {drop:.4f}"
            assert drop / s["ceiling_valid_only"] < 0.015, (
                f"{prop}: the report claims under 1.5% of ceiling"
            )
            assert s["validity_of_the_ceiling_setting_candidate"] > 0.98, prop


class TestPhase2PreRegistrationWasNotFullyBlind:
    """Section 17.3, from the audit. The first draft called PREDICTED_LOCALITY_ORDER the
    quantity with 'no post-hoc degrees of freedom whatsoever', which contradicts section
    12.2's own concession that the hypothesis exists to explain two results already in
    hand. The leave-phase-1-out correlation is the check that repairs it."""

    @pytest.fixture(scope="class")
    def loc(self):
        return _load("pilot_50k_p2_locality/locality_metrics.json")

    def test_the_two_phase_one_properties_sit_where_phase_one_measured_them(self):
        from property_to_go.properties import PREDICTED_LOCALITY_ORDER

        order = list(PREDICTED_LOCALITY_ORDER)
        assert order.index("aromatic_rings") < order.index("clogp"), (
            "the concession in section 17.3 is that the predicted ordering reproduces "
            "phase 1's aromatic-rings-over-cLogP result; if that stopped being true the "
            "paragraph would need rewriting"
        )

    def test_the_ordering_survives_dropping_them(self, loc):
        t = loc["tests"]["pre_registered_order_vs_steerability_excluding_phase1_properties"]
        assert t["n"] == 4, t
        assert set(t["excluded"]) == {"aromatic_rings", "clogp"}
        assert t["rho"] > 0.6, (
            f"the report claims the a-priori ordering survives out of sample at rho=+0.800; "
            f"measured {t['rho']:.3f}"
        )
        assert t["p_value"] > 0.05, (
            "four points cannot reach significance and the report must not imply they do"
        )

    def test_the_report_no_longer_claims_zero_degrees_of_freedom(self, report_text):
        assert "no post-hoc degrees of freedom whatsoever" not in report_text, (
            "section 17.3's overstatement was withdrawn by the audit; two of the six "
            "predicted ranks restate phase-1 results"
        )


class TestPhase2CentralTest:
    """Section 16: guidance and compute-matched best-of-N across all six properties."""

    @pytest.fixture(scope="class")
    def battery(self):
        from property_to_go.properties import LOCALITY_BATTERY

        return list(LOCALITY_BATTERY)

    def test_phase2_guided_hit_rates_match(self, report_text, battery):
        for prop in battery:
            conds = _load(
                f"pilot_50k_p2_guided_{prop}/guidance_metrics.json"
            )["conditions"]
            for cond in ("unguided", "throughout", "early", "middle", "late",
                         "truncation_control"):
                if cond in conds:
                    _assert_present(
                        report_text, conds[cond]["aggregate"]["hit_rate"]["mean"],
                        "%.4f", f"{prop}/{cond} hit rate",
                    )

    def test_phase2_bestofn_and_advantage_match(self, report_text, battery):
        for prop in battery:
            matches = _load(f"pilot_50k_p2_bestofn_{prop}/bestofn_metrics.json")["matches"]
            for accounting, m in matches.items():
                _assert_present(report_text, m["aggregate"]["hit_rate"]["mean"], "%.4f",
                                f"{prop}/{accounting} best-of-N")
                _assert_present(
                    report_text,
                    m["comparison_vs_guided_throughout"]["guidance_advantage"],
                    "%+.4f", f"{prop}/{accounting} advantage",
                )

    def test_guidance_loses_to_compute_matched_best_of_n_on_every_property(self, battery):
        """The pilot's headline negative result, extended from two properties to six and
        asserted rather than narrated -- the phase-2 analogue of
        `test_guidance_always_loses_to_compute_matched_best_of_n`."""
        for prop in battery:
            matches = _load(f"pilot_50k_p2_bestofn_{prop}/bestofn_metrics.json")["matches"]
            for accounting, m in matches.items():
                adv = m["comparison_vs_guided_throughout"]["guidance_advantage"]
                assert adv < 0, f"{prop}/{accounting}: advantage={adv}"

    def test_the_pilots_two_properties_replicate_on_the_new_sample(self):
        """cLogP and aromatic rings were measured twice, on independent 50k samples, with
        the section 11.5 defect fixed the second time. The effect sizes must agree."""
        for prop, tol in (("aromatic_rings", 0.06), ("clogp", 0.10)):
            p1 = _load(f"pilot_50k_guided_{prop}/guidance_metrics.json")["conditions"]
            p2 = _load(f"pilot_50k_p2_guided_{prop}/guidance_metrics.json")["conditions"]
            e1 = (p1["throughout"]["aggregate"]["hit_rate"]["mean"]
                  - p1["unguided"]["aggregate"]["hit_rate"]["mean"])
            e2 = (p2["throughout"]["aggregate"]["hit_rate"]["mean"]
                  - p2["unguided"]["aggregate"]["hit_rate"]["mean"])
            assert abs(e1 - e2) < tol, (
                f"{prop}: phase-1 effect {e1:+.4f} against phase-2 {e2:+.4f}"
            )

    def test_hbd_count_is_steerable_which_is_the_discriminating_case(self, battery):
        """docs/LEXICAL_LOCALITY.md §6 and docs/TODO.md C9. HBD count is SLIM's hardest
        property under additive latent steering, and lexical locality predicts it should be
        easy under token choice. If it is hard for us too, P1's discriminating case fails
        and the report must say so -- so the verdict is pinned to the artefact."""
        c = _load("pilot_50k_p2_guided_hbd_count/guidance_metrics.json")["conditions"]
        lift = (c["throughout"]["aggregate"]["hit_rate"]["mean"]
                - c["unguided"]["aggregate"]["hit_rate"]["mean"])
        assert lift > 0.10, (
            f"report claims HBD count is clearly steerable by token choice; lift={lift:+.4f}"
        )
        # and it must be near the top of the six, not merely non-zero
        lifts = {}
        for prop in battery:
            cc = _load(f"pilot_50k_p2_guided_{prop}/guidance_metrics.json")["conditions"]
            lifts[prop] = (cc["throughout"]["aggregate"]["hit_rate"]["mean"]
                           - cc["unguided"]["aggregate"]["hit_rate"]["mean"])
        rank = sorted(lifts, key=lambda p: -lifts[p]).index("hbd_count") + 1
        assert rank <= 2, f"report claims HBD count ranks near the top; rank {rank} of {lifts}"

    def test_phase2_effects_survive_joint_length_and_size_matching(self, battery):
        """Rejection criterion R2, extended to six properties."""
        for prop in battery:
            d = _load(f"pilot_50k_p2_confound_{prop}/confound_metrics.json")["conditions"]
            ref = d["unguided"]["raw"]["hit_rate"]
            raw = d["throughout"]["raw"]["hit_rate"] - ref
            joint = d["throughout"]["joint"]["hit_rate"] - ref
            assert raw > 0, prop
            assert joint / raw > 0.70, (
                f"{prop}: only {joint / raw:.0%} of the effect survives joint matching"
            )
            assert d["throughout"]["joint"]["coverage"] > 0.90, prop

    def test_the_interval_mask_defect_barely_changes_guided_decoding(self, report_text):
        """Section 16.3, and the correction to section 8.2's reasoning.

        A softmax over candidates is invariant to an additive constant, so scaling every
        candidate's target probability by the same factor cancels. The prediction is that
        the legacy mask barely moves the guided hit rate; measured on the same dataset and
        seeds with only the mask differing.
        """
        fixed = _load("pilot_50k_p2_guided_clogp/guidance_metrics.json")["conditions"]
        legacy = _load(
            "pilot_50k_p2_guided_clogp_legacymask/guidance_metrics.json"
        )["conditions"]
        for cond in ("unguided", "throughout"):
            _assert_present(report_text, legacy[cond]["aggregate"]["hit_rate"]["mean"],
                            "%.4f", f"legacy-mask {cond} hit rate")
        e_fixed = (fixed["throughout"]["aggregate"]["hit_rate"]["mean"]
                   - fixed["unguided"]["aggregate"]["hit_rate"]["mean"])
        e_legacy = (legacy["throughout"]["aggregate"]["hit_rate"]["mean"]
                    - legacy["unguided"]["aggregate"]["hit_rate"]["mean"])
        assert abs(e_fixed - e_legacy) < 0.06, (
            f"report claims the mask barely matters to decoding: fixed {e_fixed:+.4f} "
            f"against legacy {e_legacy:+.4f}"
        )


class TestPhase2LambdaSweep:
    """Section 19: the lambda sweep (C10) and quality at every lambda (C12)."""

    ANCHORS = ("aromatic_rings", "hbd_count", "qed")

    @pytest.fixture(scope="class")
    def sw(self):
        p = OUT / "pilot_50k_p2_lambda_sweep/lambda_sweep_metrics.json"
        if not p.exists():
            pytest.skip("lambda sweep not run yet")
        return _load("pilot_50k_p2_lambda_sweep/lambda_sweep_metrics.json")

    def test_every_anchor_and_lambda_is_present(self, sw):
        assert not sw["missing"], sw["missing"]
        for prop in self.ANCHORS:
            assert prop in sw["properties"], prop
            assert sorted(sw["properties"][prop]["by_lambda"], key=float) == [
                "0.25", "0.5", "1", "2", "4", "8"
            ]

    def test_each_run_recorded_the_lambda_it_was_asked_for(self, sw):
        """`--lam` folds the override into the config dict. If that ever regresses, every
        directory would silently carry configs/guidance.yaml's 1.0 instead."""
        for prop, r in sw["properties"].items():
            for k, row in r["by_lambda"].items():
                assert float(row["lambda_recorded_in_the_run"]) == float(k), (prop, k)
                if float(k) != 1.0:
                    assert row["lambda_source"] == "cli --lam", (prop, k)

    def test_the_unguided_condition_is_identical_at_every_lambda(self, sw):
        """It cannot depend on lambda. It is regenerated anyway, as a bug alarm."""
        for prop, r in sw["properties"].items():
            vals = {round(row["hit_rate_unguided"], 10)
                    for row in r["by_lambda"].values()}
            assert len(vals) == 1, (
                f"{prop}: the unguided hit rate moved across lambda ({vals}), which is "
                "impossible and means the runs are not comparable"
            )

    def test_the_response_is_an_inverted_u(self, report_text, sw):
        """Section 19.1's central claim, and the branch HANDOFF's pre-committed
        interpretation did NOT allow for: not monotone, interior optimum, worse at the
        largest lambda than at the optimum."""
        for prop, r in sw["properties"].items():
            assert r["lift_is_non_monotonic_in_lambda"], (
                f"{prop}: the report says every anchor has an interior optimum; the best "
                f"lambda measured is {r['best_lambda']}"
            )
            assert r["best_lambda"] in (2.0, 4.0), (prop, r["best_lambda"])
            assert r["by_lambda"]["8"]["lift"] < r["best_lift"], prop
            _assert_present(report_text, r["best_lift"], "%+.4f", f"{prop}/best_lift")

    def test_tuning_lambda_is_worth_between_1_and_2x(self, report_text, sw):
        """Section 19.1: 1.29x / 1.61x / 1.69x. This is the end-to-end number section
        15.6 could not produce, and the reason its per-position bound was withdrawn."""
        gains = {p: r["gain_from_tuning_lambda"] for p, r in sw["properties"].items()}
        assert min(gains.values()) > 1.2, gains
        assert max(gains.values()) < 2.0, (
            f"the report says tuning lambda is worth 1.29-1.69x end to end: {gains}"
        )
        for p, g in gains.items():
            _assert_present(report_text, g, "%.2f", f"{p}/gain_from_tuning_lambda")

    def test_p5_is_not_falsified_at_any_lambda(self, sw):
        """No lambda beats compute-matched best-of-N. If this ever flips, C4 is
        overturned and the paper changes completely (docs/HANDOFF.md E1)."""
        for prop, r in sw["properties"].items():
            assert not r["any_lambda_beats_best_of_n"], (
                f"{prop}: guidance beat compute-matched best-of-N at some lambda. That "
                "overturns C4 and section 19.2 must be rewritten, not adjusted."
            )
            assert r["best_guidance_advantage"] < 0.0
        best = {p: r["best_guidance_advantage"] for p, r in sw["properties"].items()}
        assert max(best.values()) > -0.15, (
            f"the report says HBD count at lambda=2 narrows the gap to -0.0931: {best}"
        )

    def test_pushing_lambda_past_the_optimum_destroys_the_base_policy(self, sw):
        """Section 19.1's mechanism: validity collapses, so the high-lambda hit rate is
        computed over molecules a tenth of which no longer parse."""
        for prop, r in sw["properties"].items():
            at1 = r["by_lambda"]["1"]["validity"]
            at8 = r["validity_at_the_largest_lambda"]
            assert at1 > 0.99, (prop, at1)
            assert at8 < 0.91, (
                f"{prop}: the report says validity falls to 0.807-0.902 at lambda=8; "
                f"measured {at8:.4f}"
            )

    def test_c12_the_degenerate_molecules_appear_above_the_optimum(self, report_text, sw):
        """Section 19.3. The pilot's quality null result is confirmed lambda-specific:
        R3 does not fire at lambda <= 2 and does fire at lambda >= 4."""
        for prop, r in sw["properties"].items():
            d = r["degeneracy_rate_by_lambda"]
            assert d["8"] > 3 * d["1"], (
                f"{prop}: the report says degeneracy among hits rises sharply with lambda; "
                f"lambda=1 {d['1']:.4f} vs lambda=8 {d['8']:.4f}"
            )
            for k in ("0.25", "0.5", "1", "2"):
                q = r["by_lambda"][k]["quality"]
                assert not q["degeneracy_excludes_zero"] or q["degeneracy_difference"] < 0, (
                    f"{prop} lambda={k}: R3 fires at or below lambda=2, which contradicts "
                    "section 19.3"
                )
        # QED is the property the report singles out: its optimum is already degenerate.
        qed = sw["properties"]["qed"]
        assert qed["best_lambda"] == 4.0
        q4 = qed["by_lambda"]["4"]["quality"]
        assert q4["degeneracy_excludes_zero"] and q4["degeneracy_difference"] > 0, (
            "section 19.3 says QED has no lambda that is both best for hit rate and clean"
        )

    def test_the_failure_mode_is_fragmentation_not_long_tails(self, sw):
        """Section 19.3: the literature's reward-hacked molecule is a long greasy tail;
        into a bounded interval we get shorter, more fragmented molecules instead. That
        is a claim about a *direction* and it is the interesting part of C12."""
        for prop, r in sw["properties"].items():
            d8 = r["by_lambda"]["8"]["quality"]["descriptors"]
            assert d8["longest_chain"]["difference_vs_base_hits"] < 0, (
                f"{prop}: longest chain rose at lambda=8, which is the failure mode "
                "section 19.3 says we do NOT see"
            )
            assert d8["longest_chain"]["excludes_zero"], prop
            assert d8["n_fragments"]["difference_vs_base_hits"] > 0, prop


def test_guidance_does_not_reduce_drug_likeness():
    """Section 10.3: the expected degeneracy does not appear at lambda = 1."""
    for prop in ("clogp", "aromatic_rings"):
        v = _load(f"pilot_50k_quality_{prop}/quality_metrics.json")["vs_unguided_hits"]
        d = v["throughout"]["qed"]
        assert not (d["difference"] < 0 and d["excludes_zero"]), (
            f"{prop}: report claims QED is not harmed, artefact says {d}"
        )
