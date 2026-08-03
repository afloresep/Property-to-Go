"""C18 -- binding tests for `reports/section_c18_head_fix.md`.

Two kinds of test live here and they are deliberately separated.

**Algebra.**  The central claim of C18 is that a whole family of "fixes" to the head is
*algebraically identical* to a rescale of lambda, which `pilot_report.md` §19 has
already swept.  That is a mathematical statement, so it is checked with constructed
inputs and exact tolerances rather than with a measurement.  These tests do not skip.

**Artefacts.**  Every number the section file asserts is re-read from the JSON that
produced it, formatted the way the section formats it, and required to appear in the
text -- the discipline `tests/test_report_matches_artifacts.py` established, for the
same reason: hand-transcription is the one error mode no reasoning can rule out.
These skip when the artefacts are absent, so a fresh clone still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from property_to_go import calibration as C, headroom as H
from property_to_go.binning import QuantileBinner, interval_mask_coverage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "pilot_report.md"  # C18 merged as section 20


def _load(rel: str):
    p = OUT / rel
    if not p.exists():
        pytest.skip(f"{rel} not produced yet")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def section_text() -> str:
    if not SECTION.exists():
        pytest.skip("section not written yet")
    return SECTION.read_text()


def _normalise(text: str) -> str:
    """Fold the report's typographic minus onto ASCII before matching.

    `pilot_report.md` writes negative numbers with U+2212 MINUS SIGN, which is right for
    prose and wrong for `"%+.4f" % value`.  Normalising here keeps the binding exact
    while leaving the section readable; the alternative -- ASCII hyphens in the tables --
    would make this section the only one in the report that looks different.
    """
    return text.replace("−", "-").replace("–", "-")


def _assert_present(text: str, value, fmt: str, label: str) -> None:
    rendered = fmt % value
    assert rendered in _normalise(text), f"{label}: section is missing {rendered}"


def _candidates(seed: int = 0, n: int = 500, k: int = 8):
    rng = np.random.default_rng(seed)
    lp = np.log(rng.dirichlet(np.ones(k) * 0.3, size=n))
    q = rng.uniform(1e-4, 0.45, size=(n, k))
    return lp, q


# =============================================================================
# 1. The algebra.  These are the tests that make the C18 conclusion a theorem
#    about the decoder rather than an observation about one run.
# =============================================================================


def test_a_power_calibration_is_exactly_a_lambda_rescale():
    """g(q) = c*q**alpha at lambda is the raw head at lambda*alpha, exactly.

    With eps = 0 the two scores differ by `lam*log c`, which is constant across the
    eight candidates, and a softmax is invariant to an additive constant.  This is the
    whole reason `docs/TODO.md` C18's "cheap version" is not a new experiment.
    """
    lp, q = _candidates()
    for alpha in (0.3, 0.5, 0.916, 1.0, 2.0, 3.7):
        for log_c in (-4.0, 0.0, 2.5):
            r = C.equivalent_lambda_is_exact(lp, q, alpha, log_c, lam=1.0, eps=0.0)
            assert r["equivalent_lambda"] == pytest.approx(alpha)
            assert r["max_abs_weight_difference"] < 1e-12
            assert r["argmax_agreement"] == 1.0


def test_the_identity_survives_the_deployed_eps_floor():
    """At the deployed eps = 1e-6 the identity is approximate, and negligibly so."""
    lp, q = _candidates()
    r = C.equivalent_lambda_is_exact(lp, q, 0.6, -1.0, lam=1.0, eps=1e-6)
    assert r["max_abs_weight_difference"] < 1e-3
    assert r["argmax_agreement"] == 1.0


def test_a_pure_level_shift_of_the_head_changes_nothing():
    """Scaling every candidate's q by a constant cancels -- the section 16.3 mechanism.

    This is why an off-policy gap that is a *level* error (predicted 0.16 against an
    observed 0.30) cannot be the reason guidance underperforms: the decoder never sees
    the level, only the differences.
    """
    lp, q = _candidates(seed=1)
    w0 = H.guided_weights(lp, q, 1.0, 0.0)
    for c in (0.25, 2.0, 10.0):
        w1 = H.guided_weights(lp, c * q, 1.0, 0.0)
        assert np.abs(w0 - w1).max() < 1e-12


def test_the_platt_intercept_acts_only_through_saturation_and_lowers_effective_lambda():
    """The intercept is *not* a clean no-op, and the way it fails matters.

    In the small-q limit `sigmoid(a*logit q + b) -> exp(b) q**a`, so the intercept is a
    multiplicative constant and cancels in the softmax.  Where it pushes the calibrated
    probability toward 1 the sigmoid saturates, the map flattens, and the *effective*
    exponent falls.  Both halves are asserted, because the second is what makes
    "correct the head's under-confidence" a lambda **decrease** rather than a no-op --
    and section 19 measures the lift falling steeply below lambda = 1.
    """
    lp, q = _candidates(seed=2)
    small = np.clip(q, 1e-4, 0.05)
    w_ref = H.guided_weights(lp, C.PlattCalibrator(a=0.9, b=0.0).apply(small), 1.0, 0.0)
    for b in (-2.0, 0.5, 1.5):
        w = H.guided_weights(lp, C.PlattCalibrator(a=0.9, b=b).apply(small), 1.0, 0.0)
        assert np.abs(w - w_ref).max() < 5e-2

    grid = np.linspace(0.002, 0.35, 400)
    alphas = [C.fit_power_approximation(C.PlattCalibrator(a=0.9, b=b), grid)["alpha"]
              for b in (0.0, 1.2, 2.4)]
    assert alphas[0] > alphas[1] > alphas[2], (
        "a larger intercept must flatten the map, i.e. lower the effective lambda"
    )
    assert alphas[0] < 0.9  # even at b = 0, saturation shaves the exponent


def test_platt_tends_to_a_power_law_where_our_probabilities_live():
    """`sigmoid(a*logit q + b)` -> `exp(b) q**a` as q -> 0, so Platt is a lambda rescale.

    Two thresholds, both stated in the units that matter.  Asymptotically the exponent
    *is* the slope.  Over the range our candidate probabilities actually occupy the
    log-space residual must be small compared with one step of the section 19 sweep,
    which is a factor of two, i.e. `ln 2 = 0.693` nats.
    """
    fit_small = C.fit_power_approximation(
        C.PlattCalibrator(a=0.9, b=1.2), np.linspace(1e-5, 0.02, 400)
    )
    assert fit_small["alpha"] == pytest.approx(0.9, abs=0.06)

    fit_op = C.fit_power_approximation(
        C.PlattCalibrator(a=0.9, b=1.2), np.linspace(0.002, 0.35, 400)
    )
    assert fit_op["residual_rms_nats"] < 0.1 * np.log(2.0)


def test_isotonic_calibration_cannot_strictly_reorder_the_candidates():
    """Monotone in q, therefore rank-preserving -- the ceiling on any q-only fix.

    A calibrator that cannot change *which* candidate the head prefers cannot move the
    decoder toward the head-free ceiling of §15.1, which is attained by picking a
    particular candidate.  It can only change how hard the decoder pushes, which is
    what lambda already does.

    Isotonic regression is *weakly* monotone -- it has flat steps -- so the correct
    statement is that the head's preferred candidate still attains the maximum
    calibrated value, not that `argmax` returns the same index.  Ties are the only
    thing it can create.
    """
    rng = np.random.default_rng(3)
    q_fit = rng.uniform(0.0, 0.5, 4000)
    hit = rng.uniform(size=4000) < np.clip(2.2 * q_fit, 0, 1)
    iso = C.fit_isotonic(q_fit, hit)
    assert np.all(np.diff(iso.y) >= -1e-12)

    _, q = _candidates(seed=4)
    g = iso.apply(q)
    rows = np.arange(len(q))
    assert np.all(g[rows, q.argmax(axis=1)] >= g.max(axis=1) - 1e-12)
    # and no strict inversion anywhere
    for i in range(len(q)):
        order = np.argsort(q[i])
        assert np.all(np.diff(g[i][order]) >= -1e-12)


def test_bin_logit_temperature_is_not_a_function_of_q_alone():
    """The one post-hoc family that CAN reorder candidates, demonstrated by example.

    Two candidates with nearly the same interval probability but different bin
    distributions move differently under a temperature, and here their order flips.
    That is exactly what no rescale of lambda and no monotone map on q can do, which is
    why this arm is measured rather than dismissed.
    """
    mask = np.array([False, True, False])  # bins 0..2, target is bin 1
    z = np.array([[-5.0, 2.0, 2.1], [0.5, 1.0, 0.4]])

    def q_of(T):
        zz = z / T
        zz = zz - zz.max(axis=1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, mask].sum(axis=1)

    q1, q2 = q_of(1.0), q_of(0.25)
    assert q1[0] > q1[1], "constructed example no longer orders A above B at T = 1"
    assert q2[0] < q2[1], "constructed example no longer flips the order at T = 0.25"


def test_fit_platt_recovers_a_map_it_was_generated_from():
    rng = np.random.default_rng(5)
    q = rng.uniform(0.001, 0.6, 60000)
    p = C._sigmoid(1.4 * C._logit(q) - 0.7)
    hit = rng.uniform(size=len(q)) < p
    fit = C.fit_platt(q, hit)
    assert fit.a == pytest.approx(1.4, abs=0.08)
    assert fit.b == pytest.approx(-0.7, abs=0.08)


def test_pava_produces_a_non_decreasing_fit():
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    out = C._pava(y, np.ones_like(y))
    assert np.all(np.diff(out) >= -1e-12)
    assert out.sum() == pytest.approx(y.sum())


def test_calibrated_scorer_agrees_with_the_numpy_path():
    """The decode-time scorer and the offline analysis must compute the same q."""
    torch = pytest.importorskip("torch")
    from property_to_go.binning import CategoricalBinner
    from property_to_go.heads import MLPHead

    torch.manual_seed(0)
    head = MLPHead(in_dim=16, hidden_dim=8, n_bins=6, dropout=0.0)
    binner = CategoricalBinner(max_value=5)
    cal = C.PlattCalibrator(a=0.8, b=1.1)
    scorer = C.CalibratedTargetScorer(head, binner, 3.0, 4.0, calibrator=cal)
    x = torch.randn(32, 16)
    got = scorer(x).numpy()
    probs = head.predict_proba(x.numpy())
    want = cal.apply(probs[:, np.asarray(binner.interval_mask(3.0, 4.0))].sum(axis=1))
    assert np.abs(got - want).max() < 1e-5


def test_the_focused_readout_represents_the_target_interval_exactly():
    """A three-bin [-inf, lo), [lo, hi), [hi, inf) readout is a union of bins by build.

    This is the §11.5 invariant, and the point of the `focused` variant is that it
    satisfies it by construction for continuous and count properties alike.
    """
    rng = np.random.default_rng(7)
    y = rng.normal(3.0, 1.5, 20000)
    lo, hi = 4.1726760, 5.0383000
    binner = QuantileBinner(
        edges=np.array([-np.inf, lo, hi, np.inf]),
        centers=np.array([2.0, 4.5, 6.0]),
        quantile_levels=np.array([0.0, 0.5, 0.9, 1.0]),
    )
    mask = np.asarray(binner.interval_mask(lo, hi))
    assert mask.tolist() == [False, True, False]
    cov = interval_mask_coverage(binner, lo, hi, y)
    assert cov["is_exact"]
    assert cov["n_bins_selected"] == 1


# =============================================================================
# 2. The artefacts.  Numbers asserted in reports/section_c18_head_fix.md.
# =============================================================================

OFFPOLICY = "c18_offpolicy_calibration/offpolicy_calibration.json"
PERPOS = "c18_per_position/per_position_capture.json"
VARIANTS = "c18_head_variants/head_variants_summary.json"
PREDICTION = "c18_prediction/prediction.json"


def test_the_prediction_was_written_before_the_measurements():
    """The C18 brief requires the Trap-2 prediction to be committed in writing first.

    Enforced by file mtime rather than by trust: the prediction artefact must predate
    every measurement artefact it is scored against.
    """
    pred = OUT / "c18_prediction" / "prediction.json"
    if not pred.exists():
        pytest.skip("prediction not written")
    assert json.loads(pred.read_text())["written_before_any_measurement"] is True
    t = pred.stat().st_mtime
    for rel in (OFFPOLICY, PERPOS, VARIANTS):
        p = OUT / rel
        if p.exists():
            assert p.stat().st_mtime >= t, f"{rel} predates the prediction"


def test_off_policy_miscalibration_is_smaller_than_the_pilot_reported(section_text):
    """Trap 1: the 3.5x factor must be re-measured, and the section must quote it."""
    rep = _load(OFFPOLICY)["properties"]
    for prop, entry in rep.items():
        on = entry["on_policy_base_prefixes"]
        off = entry["off_policy_guided_prefixes"]
        _assert_present(section_text, on["mean_predicted"], "%.4f", f"{prop} on-policy pred")
        _assert_present(section_text, off["mean_predicted"], "%.4f", f"{prop} off-policy pred")
        _assert_present(section_text, off["observed"], "%.4f", f"{prop} off-policy observed")
        _assert_present(section_text, off["under_confidence_factor"], "%.2f",
                        f"{prop} off-policy factor")
        _assert_present(section_text, off["ece"], "%.4f", f"{prop} off-policy ECE")
        _assert_present(section_text, off["auroc"], "%.4f", f"{prop} off-policy AUROC")


def test_the_clogp_off_policy_factor_is_well_below_the_reported_3_5():
    """The pilot's headline calibration number, re-measured on the fixed head."""
    off = _load(OFFPOLICY)["properties"]["clogp"]["off_policy_guided_prefixes"]
    assert 1.0 < off["under_confidence_factor"] < 3.0, (
        "if this fails the section's central Trap-1 claim is wrong and must be rewritten"
    )


def test_the_head_still_ranks_off_policy():
    """AUROC is what the softmax over eight candidates consumes; calibration is not."""
    rep = _load(OFFPOLICY)["properties"]
    for prop, entry in rep.items():
        assert entry["off_policy_guided_prefixes"]["auroc"] > 0.55, prop


def test_post_hoc_calibration_reduces_ece_on_held_out_guided_prefixes():
    """The calibrators do what calibrators do -- so any failure downstream is not this."""
    rep = _load(OFFPOLICY)["properties"]
    for prop, entry in rep.items():
        c = entry["post_hoc_calibrated_on_held_out_guided_prefixes"]
        assert c["isotonic"]["ece"] <= c["uncalibrated"]["ece"] + 1e-9, prop
        assert c["platt"]["ece"] <= c["uncalibrated"]["ece"] + 1e-9, prop


def test_the_platt_fits_are_reported_and_are_near_unit_slope(section_text):
    """The decisive quantity: only the slope acts on decoding, and it is the small part."""
    rep = _load(OFFPOLICY)["properties"]
    for prop, entry in rep.items():
        a = entry["fitted_calibrators"]["platt"]["a"]
        _assert_present(section_text, a, "%.3f", f"{prop} platt slope")


def test_per_position_reproduces_the_published_lambda1_decomposition():
    """The reconstruction gate: baseline head must reproduce section 15.6's own value.

    If this fails, nothing else in the per-position table means anything, because the
    rollouts on disk would belong to different prefixes than the ones being scored.
    """
    rep = _load(PERPOS)
    assert rep["prefix_reconstruction"]["candidate_ids_identical"] is True
    assert rep["prefix_reconstruction"]["base_logprob_max_abs_difference"] < 1e-5
    for prop, entry in rep["properties"].items():
        pub = entry["published_reference"]["our_head_gain"]
        got = entry["arms"]["baseline"]["our_head_gain"]
        assert got == pytest.approx(pub, abs=2e-4), (
            f"{prop}: recomputed baseline gain {got} != published {pub}"
        )


def test_per_position_numbers_in_the_section_match_the_artefact(section_text):
    rep = _load(PERPOS)["properties"]
    for prop, entry in rep.items():
        for arm, vals in entry["arms"].items():
            _assert_present(section_text, vals["our_head_gain"], "%+.4f",
                            f"{prop}/{arm} per-position gain")


def test_the_section_carries_the_per_step_is_not_end_to_end_warning(section_text):
    """docs/TODO.md C22.1 is the single most important methodological requirement here."""
    assert "per-position" in section_text
    assert "end to end" in section_text or "end-to-end" in section_text
    rep = _load(PERPOS)
    assert "MUST_BE_MEASURED_END_TO_END" in rep


def test_a_monotone_calibrator_never_changes_which_candidate_is_picked():
    """Measured on the real 400x8 candidate array, not on synthetic data.

    Platt is strictly monotone, so the rate is unchanged to the last bit.  Isotonic is
    only weakly monotone, so it can create ties and the rate can only go **up** -- and
    an increase there is a tie, not a reordering.  Both are asserted, because the
    difference is the difference between "this calibrator found a better candidate" and
    "this calibrator stopped distinguishing two candidates".
    """
    rep = _load(PERPOS)["properties"]
    for prop, entry in rep.items():
        arms = entry["arms"]
        if "baseline_isotonic" not in arms:
            continue
        base = arms["baseline"]["picks_the_best_candidate_rate"]
        for arm in ("baseline_platt", "baseline_isotonic"):
            # Platt is strictly monotone except that `_logit` clips at 1e-6, so
            # candidates whose probability falls below the clip become tied; isotonic
            # has genuine flat steps. Both mechanisms can only ADD ties, so the rate
            # can only rise, and a rise is a tie rather than a reordering.
            assert arms[arm]["picks_the_best_candidate_rate"] >= base - 1e-12, f"{prop}/{arm}"


def test_platt_calibration_lands_on_its_own_equivalent_lambda():
    """The identity, measured end of the per-position pipeline rather than argued.

    Platt is a power map only to first order, so the two are close rather than equal;
    the residual is what the section reports.
    """
    rep = _load(PERPOS)["properties"]
    for prop, entry in rep.items():
        eq = entry.get("lambda_rescale_at_the_platt_equivalent")
        if eq is None:
            continue
        assert abs(eq["difference"]) < 0.6 * abs(eq["our_head_gain"]), prop


def test_post_hoc_calibration_does_not_improve_per_position_capture():
    """The C18 result, bound to the artefact.

    Platt and isotonic are monotone functions of q and both come out BELOW the
    uncalibrated head at every property.  If this ever fails, the section's conclusion
    is wrong and has to be rewritten rather than patched.
    """
    rep = _load(PERPOS)["properties"]
    for prop, entry in rep.items():
        arms = entry["arms"]
        base = arms["baseline"]["our_head_gain"]
        for arm in ("baseline_platt", "baseline_isotonic"):
            if arm in arms:
                assert arms[arm]["our_head_gain"] < base, f"{prop}/{arm}"


def test_the_power_identity_holds_on_the_real_candidate_array():
    """Checked at the deployed eps = 1e-6, which is the only place the identity leaks.

    The leak is the `log(q + eps)` floor: a candidate whose probability is far below
    eps has its calibrated value pulled up differently under the two parametrisations.
    The MEAN weight difference is at the 1e-4 level everywhere; the max reaches 0.08 at
    a handful of prefixes with a near-zero candidate.  The end-to-end consequence is
    measured separately in `c18_identity/` and is nil.
    """
    rep = _load(PERPOS)["properties"]
    for prop, entry in rep.items():
        ident = entry.get("power_calibration_is_a_lambda_rescale")
        if ident is None:
            continue
        assert ident["argmax_agreement"] == 1.0, prop
        assert ident["mean_abs_weight_difference"] < 1e-3, prop


def test_the_identity_is_exact_end_to_end_at_eps_zero():
    """Two full guided runs that must return the same molecules, and do.

    This is the claim that makes `docs/HANDOFF.md` E2's cheap version redundant rather
    than untried: a power-calibrated head at lambda = 1 IS the raw head at
    lambda = alpha.
    """
    rep = _load("c18_identity/identity_check.json")
    a = rep["arms"]["eps0"]
    assert a["identical_molecule_fraction"] == 1.0
    assert a["hit_rate_difference"] == 0.0
    b = rep["arms"]["epsdeployed"]
    assert b["identical_molecule_fraction"] > 0.99
    assert abs(b["hit_rate_difference"]) < 1e-9


def test_head_variant_target_metrics_match_the_section(section_text):
    rep = _load(VARIANTS)["variants"]
    for variant, v in rep.items():
        for prop, entry in v["properties"].items():
            _assert_present(section_text, entry["test"]["target_auroc"], "%.4f",
                            f"{variant}/{prop} AUROC")


def test_every_head_variant_has_an_exact_interval_mask():
    """The §11.5 defect must not be reintroduced by a new readout."""
    rep = _load(VARIANTS)["variants"]
    for variant, v in rep.items():
        for prop, entry in v["properties"].items():
            assert entry["interval_mask_coverage"]["is_exact"], f"{variant}/{prop}"


# ----------------------------------------------------------------- end to end

SUMMARY = "c18_summary/c18_summary.json"
MATCHED = "c18_matched_best_of_n/matched_best_of_n.json"


def test_the_uncalibrated_control_reproduces_the_central_test():
    """If the control does not reproduce §16.1, nothing else end to end counts."""
    rep = _load(SUMMARY)["properties"]
    for prop, e in rep.items():
        pub, ctl = e["published_lambda1"], e["arms"]["uncalibrated"]
        assert ctl["hit_rate_unguided"] == pytest.approx(pub["hit_rate_unguided"], abs=1e-12), prop
        assert ctl["hit_rate_throughout"] == pytest.approx(
            pub["hit_rate_throughout"], abs=1e-12
        ), prop


def test_no_c18_arm_beats_compute_matched_best_of_n():
    """The headline. R4 continues to fire across every arm at every anchor."""
    rep = _load(SUMMARY)
    assert rep["any_arm_anywhere_beats_compute_matched_best_of_n"] is False
    for prop, e in rep["properties"].items():
        for arm, a in e["arms"].items():
            assert a["guidance_advantage"] < 0, f"{prop}/{arm}"


def test_every_post_hoc_calibration_arm_is_worse_end_to_end():
    """The pre-committed directional prediction, at every calibration cell.

    `binT0p4` is excluded because it is not a calibration arm: it is the
    decoder-optimal sharpening of §20.5.1, run deliberately as a lambda result.
    """
    rep = _load(SUMMARY)["properties"]
    for prop, e in rep.items():
        base = e["arms"]["uncalibrated"]["lift"]
        for arm in ("platt", "isotonic"):
            if arm in e["arms"]:
                assert e["arms"][arm]["lift"] < base, f"{prop}/{arm}"


def test_the_end_to_end_numbers_in_the_section_match_the_artefact(section_text):
    rep = _load(SUMMARY)["properties"]
    for prop, e in rep.items():
        for arm, a in e["arms"].items():
            _assert_present(section_text, a["hit_rate_throughout"], "%.4f",
                            f"{prop}/{arm} throughout")
            _assert_present(section_text, a["lift"], "%+.4f", f"{prop}/{arm} lift")
            _assert_present(section_text, a["guidance_advantage"], "%+.4f",
                            f"{prop}/{arm} advantage")


def test_best_of_n_is_shared_because_every_arm_solves_to_the_same_n():
    """The cost-saving shortcut, checked rather than assumed.

    If any arm ever solved to a different N, `17_matched_best_of_n.py` would give it its
    own run; this asserts that it did not have to, and that the realised token match is
    inside the band §16.2 reports (0.95-1.02 across twelve runs).
    """
    rep = _load(MATCHED)["properties"]
    for prop, e in rep.items():
        ns = {a["n_candidates_solved"] for a in e["arms"].values()}
        assert len(ns) == 1, f"{prop}: arms solved to different N: {ns}"
        for arm, a in e["arms"].items():
            assert 0.90 <= a["realised_token_ratio"] <= 1.10, f"{prop}/{arm}"


def test_a_per_position_gain_did_not_transfer_end_to_end():
    """The C22.1 demonstration, bound to the two artefacts it is computed from.

    HBD count's `wide` readout gains 1.7x or more per position and essentially nothing
    end to end.  If that ever stops being true the section's headline methodological
    claim has to be rewritten.
    """
    pp = _load(PERPOS)["properties"]["hbd_count"]["arms"]
    e2e = _load(SUMMARY)["properties"]["hbd_count"]["arms"]
    per_step_ratio = pp["c18_heads_wide"]["our_head_gain"] / pp["baseline"]["our_head_gain"]
    end_ratio = e2e["head_wide"]["lift"] / e2e["uncalibrated"]["lift"]
    assert per_step_ratio > 1.7
    assert end_ratio < 1.02
    # and the sign reversal for the other readout
    per_step_wf = (pp["c18_heads_wide_focused"]["our_head_gain"]
                   / pp["baseline"]["our_head_gain"])
    end_wf = e2e["head_wide_focused"]["lift"] / e2e["uncalibrated"]["lift"]
    assert per_step_wf > 1.0 and end_wf < 1.0


def test_the_decoder_optimal_temperature_does_not_beat_plain_lambda():
    """§20.5.1: sharpening the head is a slightly worse version of raising lambda."""
    e2e = _load(SUMMARY)["properties"]["aromatic_rings"]["arms"]
    if "binT0p4" not in e2e:
        pytest.skip("decoder-optimal temperature arm not run")
    lam2 = _load("pilot_50k_p2_lam2_guided_aromatic_rings/guidance_metrics.json")
    lam2_thr = lam2["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"]
    lam2_val = lam2["conditions"]["throughout"]["aggregate"]["validity"]["mean"]
    assert e2e["binT0p4"]["hit_rate_throughout"] < lam2_thr
    assert e2e["binT0p4"]["validity_throughout"] < lam2_val
    assert e2e["binT0p4"]["guidance_advantage"] < 0
