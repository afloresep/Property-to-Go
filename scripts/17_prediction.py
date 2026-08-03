"""C18 -- the written prediction, committed to disk BEFORE any measurement.

`docs/TODO.md` C18 sets two traps and this script exists so that the answer to the
second one is on disk with a timestamp rather than reconstructed afterwards.

Trap 1: the reported off-policy miscalibration (predicted 0.076 against observed
0.267, a factor of 3.5) is partly an artefact of our own interval-mask defect
(`pilot_report.md` §11.5, §11.6), so it must be RE-MEASURED on the phase-2 heads
before anything is fixed.

Trap 2: a post-hoc calibration of the head's interval probability may be
*algebraically identical* to a rescale of lambda, which `pilot_report.md` §19 has
already swept.  Which calibration families are and are not equivalent is a question
with an exact answer, and it is written down here before a single number is measured.

    .venv/bin/python scripts/17_prediction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import OUTPUT_DIR, write_json, write_run_context  # noqa: E402

PREDICTION = {
    "experiment": "C18 -- fixing the head",
    "written_before_any_measurement": True,
    "scored_against": "outputs/c18_*/ and reports/section_c18_head_fix.md",
    # ------------------------------------------------------------------ trap 2
    "trap_2_which_calibration_families_are_a_lambda_rescale": {
        "decoder": "p(a) = softmax_a( log p_base(a) + lam * log(q(a) + eps) ), k = 8",
        "family_1_power_map": {
            "map": "g(q) = c * q**alpha, c > 0, alpha > 0",
            "claim": (
                "EXACTLY a lambda rescale. lam*log(c*q**alpha) = "
                "(lam*alpha)*log q + lam*log c, and the second term is constant "
                "across candidates, so the softmax is invariant to it. The induced "
                "sampling distribution equals the UNCALIBRATED head's at "
                "lam' = lam*alpha, up to the eps floor."
            ),
            "verdict": "not equivalent only through eps; contributes nothing new",
            "already_measured_by": "pilot_report.md section 19 (lam sweep)",
        },
        "family_2_platt_on_the_log_odds": {
            "map": "g(q) = sigmoid(a * logit(q) + b)",
            "claim": (
                "ASYMPTOTICALLY family 1 in the regime we are in. For q << 1, "
                "logit(q) -> log q and sigmoid(x) -> exp(x), so g(q) -> exp(b) * "
                "q**a. Our candidate q values sit around the base rates 0.08-0.17, "
                "so the power approximation should be good and Platt scaling is a "
                "lambda rescale by the fitted slope a."
            ),
            "verdict": "a lambda rescale to first order; quantify the residual",
        },
        "family_3_temperature_on_the_bin_logits": {
            "map": "q(T) = sum_{i in M} exp(z_i/T) / sum_j exp(z_j/T)",
            "claim": (
                "NOT a lambda rescale and NOT even a function of q alone -- it "
                "depends on the whole bin-logit vector, so two candidates with the "
                "same q can move differently and the candidate RANKING can change. "
                "That is the one thing a lambda rescale can never do."
            ),
            "verdict": "genuinely distinct, but predicted to reorder candidates rarely",
        },
        "family_4_isotonic_on_q": {
            "map": "g = any monotone non-decreasing step function of q",
            "claim": (
                "NOT a lambda rescale -- it makes the effective lambda "
                "d log g(q) / d log q depend on q. But it is MONOTONE IN q, so it "
                "cannot change which candidate the head prefers; it can only "
                "reweight. As it sharpens it converges on argmax-q selection, which "
                "is exactly the lam -> infinity limit. It can beat a single global "
                "lambda only by being sharp where the head is trustworthy and flat "
                "where it is not."
            ),
            "verdict": "distinct from a lambda rescale, bounded by the same argmax",
        },
        "what_is_genuinely_not_a_lambda_rescale": [
            "retraining the readout: a different function q'(.) can change the "
            "candidate RANKING, which is where section 15.6 locates the loss "
            "(oracle at lam=1 gains +0.019..+0.071, our head +0.003..+0.015)",
            "a calibrator whose input is more than q -- e.g. conditioned on "
            "position t or on the candidate token. That is a lambda SCHEDULE, "
            "which section 19 did not sweep (it swept a scalar lambda).",
        ],
    },
    # ------------------------------------------------- the directional prediction
    "prediction_the_obvious_fix_should_HURT": {
        "argument": (
            "The stated defect is UNDER-confidence: predicted << observed. Any map "
            "that corrects that at small q must satisfy g(q) > q there. A power map "
            "with q < 1 does that only when alpha < 1. But alpha < 1 means "
            "lam_eff = lam * alpha < 1, and section 19 measures the lift falling "
            "steeply below lam = 1 (aromatic rings +0.2949 at lam=1 -> +0.1225 at "
            "lam=0.5). So 'fix the calibration by raising the head's probabilities' "
            "is, to the extent it is a power map, a lambda DECREASE."
        ),
        "falsifiable_claims": [
            "the Platt slope fitted on guided prefixes will be alpha < 1",
            "a Platt/power-calibrated head at lam=1 will produce the SAME molecules "
            "as the raw head at lam=alpha under the same seed, to within the eps floor",
            "its end-to-end lift will therefore be LOWER than the uncalibrated "
            "lam=1 lift, not higher",
        ],
    },
    # ------------------------------------------------------------------ trap 1
    "prediction_re_measured_off_policy_miscalibration": {
        "reported_by_the_pilot": {
            "property": "clogp", "mean_predicted": 0.076, "observed": 0.267,
            "factor": 3.5, "ece": 0.190, "auroc": 0.651,
            "source": "pilot_report.md section 9.2.1",
        },
        "claims": [
            "on-policy (base-policy prefixes) the phase-2 clogp head will be close "
            "to calibrated, ratio observed/predicted about 1.0, per section 11.6's "
            "measured 1.014",
            "off-policy (guided prefixes) the ratio will be materially smaller than "
            "3.5 -- predicted 1.5 to 2.2 for clogp -- because section 11.6 measured "
            "about a factor of 2 of the 3.5 to be the interval-mask defect",
            "the count properties (aromatic_rings, hbd_count, rotatable_bonds) never "
            "had the defect, so their off-policy ratio should be smaller still",
            "AUROC off-policy will stay well above 0.5, i.e. the head still RANKS "
            "even where it does not CALIBRATE -- and ranking is the only thing the "
            "softmax over 8 candidates consumes",
        ],
    },
    # -------------------------------------------------------------- the head route
    "prediction_retrained_readout": {
        "variants_to_try": [
            "wide: the same two-layer MLP with hidden_dim 1024 instead of 256",
            "focused: a 3-bin readout whose middle bin IS the target interval, so "
            "the head optimises the quantity guidance consumes instead of a "
            "20-way distribution of which the target is a 2-bin marginal",
        ],
        "claims": [
            "per-position `our_head_gain` may rise, because a focused readout is "
            "trained on the discrimination the decoder needs",
            "end-to-end lift will rise by MUCH less than the per-position ratio "
            "implies -- docs/TODO.md C22.1, per-step is not end-to-end",
            "NO variant will beat compute-matched best-of-N under `actual` "
            "accounting; the lam=1 gaps are -0.22 to -0.36 and the largest "
            "end-to-end factor anything has bought so far is 1.3-1.7x (section 19)",
        ],
    },
    "what_would_falsify_the_whole_prediction": (
        "A post-hoc calibrator that raises end-to-end lift above the section 19 "
        "lambda-sweep envelope for the same property, or a retrained head that "
        "closes the gap to compute-matched best-of-N."
    ),
}


def main() -> int:
    out_dir = OUTPUT_DIR / "c18_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "prediction.json", PREDICTION)
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
