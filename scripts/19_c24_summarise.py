"""C24 stage 6 -- score the pre-registration in `outputs/c24_prereg/prereg.md`, verbatim.

Every decision rule in C24.0.6 and C24.0.7 is evaluated here from the artefacts, so the
verdicts in `reports/section_c24_generality.md` are read off a JSON rather than argued in
prose.  Failures are recorded with the same weight as successes.

    .venv/bin/python scripts/19_c24_summarise.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402

N_BOOT = 10000


T_CRIT_2DF = 4.302653  # t_{0.975, 2}


def paired_seed_diff(a: list[float], b: list[float]):
    """Paired seed-level summary of mean(a) - mean(b) at n = 3.

    **This replaces a three-seed percentile bootstrap, which was vacuous.**  At n = 3 the
    percentile bootstrap of a mean is *identically* [min, max] of the three paired
    differences: the smallest attainable bootstrap mean is the minimum, attained when all
    three resampled indices land on it, with probability 1/27 = 0.0370 > 0.025, so the
    2.5th percentile IS the minimum and the 97.5th IS the maximum, for any three numbers
    whatsoever.  "The CI excludes zero" therefore carried exactly the information of "all
    three seeds share a sign" -- a three-way sign test with two-sided null probability
    2 * (1/2)**3 = 0.25, which cannot reject at any conventional level.  Reporting it as a
    confidence interval overstated the evidence in every table it appeared in.

    What is reported instead: the per-seed differences themselves, a Student t interval on
    2 df (which is honest about how little three points buy -- it needs |mean|/sd > 2.48 to
    exclude zero), and the sign test stated as a sign test.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = T_CRIT_2DF * sd / np.sqrt(n) if n > 1 else 0.0
    lo, hi = mean - half, mean + half
    return {
        "mean_difference": mean,
        "sd": sd,
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "interval": f"Student t, {n - 1} df, t_crit={T_CRIT_2DF}",
        "n_seeds": int(n),
        "all_seeds_share_sign": bool(all(x > 0 for x in d) or all(x < 0 for x in d)),
        "sign_test_p_two_sided": 2.0 * 0.5 ** n,
        "per_seed_difference": d.tolist(),
        "note": ("the three-seed percentile bootstrap this replaced returned exactly "
                 "[min, max] of per_seed_difference and conveyed only a sign test at "
                 "p_null = 0.25; see the docstring"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c24_summary")
    args = ap.parse_args()
    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"n_boot": N_BOOT})

    depth = read_json(OUTPUT_DIR / "c24_probe_layers" / "probe_layer_metrics.json")
    cal = read_json(OUTPUT_DIR / "c24_calibration" / "calibration_metrics.json")
    steer = read_json(OUTPUT_DIR / "c24_layer_steering" / "layer_steering_metrics.json")
    e2e = read_json(OUTPUT_DIR / "c24_endtoend" / "endtoend_metrics.json")
    prereg = read_json(OUTPUT_DIR / "c24_prereg" / "prereg.json")

    n_probe = int(depth["n_probe_points"])
    final = n_probe - 1
    attrs = list(depth["attributes"].keys())

    per_attr: dict[str, dict] = {}
    for a in attrs:
        d = depth["attributes"][a]["depth"]
        c = cal["attributes"][a]
        s = steer["attributes"][a]
        E = e2e["attributes"][a]["arms"]
        best_L = int(d["best_probe_point"])

        unguided = E["unguided"]["hit_rate_mean"]
        # The reference for "lift" is the TRUNCATION CONTROL, not `unguided`: restricting
        # GPT-2 to its top 8 tokens is itself a large intervention on every attribute
        # here (unlike the molecular case, §7.9), so the gap to `unguided` mixes the
        # property term with the truncation.  Both references are reported; the arm
        # *ordering* is identical under either, because they differ by a constant.
        trunc = E["truncation_control"]["hit_rate_mean"]
        base_arm = E[f"throughout_L{final}"]
        lift_base = base_arm["hit_rate_mean"] - trunc
        lift_base_vs_unguided = base_arm["hit_rate_mean"] - unguided

        def lift(name):
            return E[name]["hit_rate_mean"] - trunc

        def ratio(name):
            return (lift(name) / lift_base) if lift_base != 0 else None

        gain_final = s["per_probe_point"][str(final)]["our_head_gain"]
        gain_best = s["per_probe_point"][str(best_L)]["our_head_gain"]

        e2e_depth = None
        if best_L != final:
            e2e_depth = paired_seed_diff(
                E[f"throughout_L{best_L}"]["per_seed_hit_rate"],
                base_arm["per_seed_hit_rate"],
            )

        per_attr[a] = {
            "target": depth["attributes"][a]["target"],
            # --- Claim 1 ------------------------------------------------------
            "platt_slope": c["platt_slope"],
            "slope_below_one": bool(c["platt_slope"] < 1.0),
            "auroc_delta_platt": c["auroc_delta_platt"],
            "auroc_unchanged_platt": bool(abs(c["auroc_delta_platt"]) <= 1e-4),
            "auroc_delta_isotonic": c["auroc_delta_isotonic"],
            "ece_uncalibrated": c["calibrated"]["uncalibrated"]["ece"],
            "ece_platt": c["calibrated"]["platt"]["ece"],
            "ece_isotonic": c["calibrated"]["isotonic"]["ece"],
            "ece_factor_platt": c["ece_factor_platt"],
            "ece_factor_isotonic": c["ece_factor_isotonic"],
            "off_policy_factor": c["off_policy_factor"],
            "on_policy_factor": c["on_policy"]["under_confidence_factor"],
            "identity": e2e["attributes"][a]["identity"],
            "identity_numeric": c["identity_numeric"],
            "lift_uncalibrated": lift_base,
            "lift_platt": lift(f"platt_L{final}"),
            "lift_isotonic": lift(f"isotonic_L{final}"),
            "ratio_platt": ratio(f"platt_L{final}"),
            "ratio_isotonic": ratio(f"isotonic_L{final}"),
            # --- Claim 2 ------------------------------------------------------
            "best_probe_point": best_L,
            "auroc_best": d["auroc_best"],
            "auroc_final": d["auroc_final"],
            "auroc_trivial": d["auroc_trivial"],
            "gain_over_final": d["gain_over_final"],
            "peak_in_first_half": d["peak_in_first_half"],
            "final_is_min_over_peak_to_end": d["final_is_min_over_peak_to_end"],
            "bonferroni_ci_excludes_zero": d["bonferroni_ci_excludes_zero"],
            "no_isolated_spike": d["no_isolated_spike"],
            "per_position_gain_final": gain_final,
            "per_position_gain_best": gain_best,
            "per_position_relative": ((gain_best - gain_final) / abs(gain_final))
            if gain_final != 0 else None,
            "per_position_improves": bool(gain_best > gain_final),
            "endtoend_hit_final": base_arm["hit_rate_mean"],
            "endtoend_hit_best": E.get(f"throughout_L{best_L}", {}).get("hit_rate_mean"),
            "endtoend_lift_final": lift_base,
            "endtoend_lift_best": lift(f"throughout_L{best_L}") if best_L != final else None,
            "endtoend_depth_ratio": ratio(f"throughout_L{best_L}") if best_L != final else None,
            "endtoend_depth_seed_interval": e2e_depth,
            "endtoend_depth_improves": bool(
                e2e_depth is not None and e2e_depth["mean_difference"] > 0
                and e2e_depth["excludes_zero"]
            ),
            # --- context -------------------------------------------------------
            "unguided_hit": unguided,
            "truncation_control_hit": trunc,
            "truncation_effect": trunc - unguided,
            "lift_uncalibrated_vs_unguided": lift_base_vs_unguided,
            # Named for the reference it is actually computed against.  An earlier draft
            # called this `guided_beats_unguided` while computing it against the
            # truncation control; the two references disagree in *sign* for
            # `mean_word_length`, so the misnomer was load-bearing and is corrected here.
            "guided_beats_truncation_control": bool(lift_base > 0),
            "guided_beats_unguided": bool(lift_base_vs_unguided > 0),
            "best_of_n": {n: r.get("best_of_n") for n, r in E.items() if r.get("best_of_n")},
            "per_seed_hit_rates": {n: r["per_seed_hit_rate"] for n, r in E.items()},
        }

    n = len(attrs)
    # ---------------------------------------------------------------- Claim 1
    c1a = all(per_attr[a]["slope_below_one"] for a in attrs)
    c1b = all(per_attr[a]["auroc_unchanged_platt"] for a in attrs)
    c1c = all(per_attr[a]["ece_factor_platt"] >= 2.0
              and per_attr[a]["ece_factor_isotonic"] >= 2.0 for a in attrs)
    c1d = all(
        per_attr[a]["identity"]["eps0"]["identical_fraction"] == 1.0
        and per_attr[a]["identity"]["eps0"]["hit_rate_difference"] == 0.0
        for a in attrs
    )
    n_platt_below = sum(per_attr[a]["ratio_platt"] < 1.0 for a in attrs)
    n_iso_below = sum(per_attr[a]["ratio_isotonic"] < 1.0 for a in attrs)
    c1e = (n_platt_below >= 2) and (n_iso_below >= 2)

    if not (c1a and c1b and c1d):
        claim1 = "FAILS TO REPLICATE"
    elif c1e:
        claim1 = "REPLICATES"
    else:
        claim1 = "PARTIALLY REPLICATES"

    # ---------------------------------------------------------------- Claim 2
    n_peak_first_half = sum(per_attr[a]["peak_in_first_half"] for a in attrs)
    none_peak_final = all(per_attr[a]["best_probe_point"] != final for a in attrs)
    c2a = (n_peak_first_half >= 2) and none_peak_final
    c2b = all(per_attr[a]["final_is_min_over_peak_to_end"] for a in attrs)

    n_pp_improve = sum(per_attr[a]["per_position_improves"] for a in attrs)
    rels = [per_attr[a]["per_position_relative"] for a in attrs
            if per_attr[a]["per_position_relative"] is not None]
    median_rel = float(np.median(rels)) if rels else None
    per_position_material = (n_pp_improve >= 2) and (median_rel is not None
                                                     and median_rel >= 0.25)
    n_e2e_improve = sum(per_attr[a]["endtoend_depth_improves"] for a in attrs)
    endtoend_positive = n_e2e_improve >= 2

    cell = ("MATERIAL" if per_position_material else "NOT MATERIAL",
            "POSITIVE" if endtoend_positive else "NOT POSITIVE")
    verdict_2e = {
        ("NOT MATERIAL", "POSITIVE"): "DIVERGENCE REPLICATES",
        ("MATERIAL", "POSITIVE"): "NO DIVERGENCE - the proxy was informative; Claim 2's "
                                  "methodological half FAILS TO REPLICATE",
        ("NOT MATERIAL", "NOT POSITIVE"): "NO DIVERGENCE - both say the layer does not "
                                          "help; Claim 2's methodological half FAILS TO "
                                          "REPLICATE",
        ("MATERIAL", "NOT POSITIVE"): "DIVERGENCE WITH THE OPPOSITE SIGN",
    }[cell]

    summary = {
        "prereg": prereg,
        "n_attributes": n,
        "attributes": per_attr,
        "claim_1_calibration": {
            "1a_all_platt_slopes_below_one": c1a,
            "1a_slopes": {a: per_attr[a]["platt_slope"] for a in attrs},
            "1b_auroc_unchanged_platt": c1b,
            "1b_deltas": {a: per_attr[a]["auroc_delta_platt"] for a in attrs},
            "1c_ece_halved_by_both": c1c,
            "1c_factors": {a: [per_attr[a]["ece_factor_platt"],
                               per_attr[a]["ece_factor_isotonic"]] for a in attrs},
            "1d_identity_exact_at_eps0": c1d,
            "1d_identical_fractions": {
                a: per_attr[a]["identity"]["eps0"]["identical_fraction"] for a in attrs},
            "1d_hit_rate_differences": {
                a: per_attr[a]["identity"]["eps0"]["hit_rate_difference"] for a in attrs},
            "1e_platt_below_one_count": int(n_platt_below),
            "1e_isotonic_below_one_count": int(n_iso_below),
            "1e_ratios_platt": {a: per_attr[a]["ratio_platt"] for a in attrs},
            "1e_ratios_isotonic": {a: per_attr[a]["ratio_isotonic"] for a in attrs},
            "1e_holds": c1e,
            "VERDICT": claim1,
        },
        "claim_2_depth": {
            "2a_peak_in_first_half_count": int(n_peak_first_half),
            "2a_no_attribute_peaks_at_final": none_peak_final,
            "2a_holds": c2a,
            "2a_best_probe_points": {a: per_attr[a]["best_probe_point"] for a in attrs},
            "2b_final_is_min_after_peak_all": c2b,
            "2c_per_position_improves_count": int(n_pp_improve),
            "2c_median_relative": median_rel,
            "2c_verdict": cell[0],
            "2c_matches_prediction_not_material": not per_position_material,
            "2d_endtoend_improves_count": int(n_e2e_improve),
            "2d_verdict": cell[1],
            "2d_ratios": {a: per_attr[a]["endtoend_depth_ratio"] for a in attrs},
            "2e_cell": list(cell),
            "2e_VERDICT": verdict_2e,
        },
        "sanity_C24_0_10": {
            "attributes_surviving_base_rate_gate": n,
            "max_auroc_any_layer": {
                a: max(depth["attributes"][a]["probe_points"][str(L)]["target_auroc"]["mean"]
                       for L in range(n_probe)) for a in attrs},
            # C24.0.10's criterion is "not positive at *any* attribute".  Both references
            # are published because they disagree: against the top-8 truncation control
            # every attribute steers, against `unguided` only one does, and the
            # difference is the truncation penalty, not the property term.
            "guided_beats_truncation_control_all": all(
                per_attr[a]["guided_beats_truncation_control"] for a in attrs),
            "guided_beats_unguided_any": any(per_attr[a]["guided_beats_unguided"]
                                             for a in attrs),
            "guided_beats_unguided_all": all(per_attr[a]["guided_beats_unguided"]
                                             for a in attrs),
            "truncation_effect": {a: per_attr[a]["truncation_effect"] for a in attrs},
        },
        "best_of_n": {
            a: {name: {"n_candidates": b["n_candidates"], "hit_rate": b["hit_rate"],
                       "advantage": b["advantage_guided_minus_bestofn"],
                       "realised_token_ratio": b["realised_token_ratio"]}
                for name, b in per_attr[a]["best_of_n"].items()}
            for a in attrs
        },
        "any_arm_anywhere_beats_compute_matched_best_of_n": bool(any(
            b["advantage_guided_minus_bestofn"] > 0
            for a in attrs for b in per_attr[a]["best_of_n"].values()
        )),
        "processed_tokens": {
            "dataset": read_json(OUTPUT_DIR / "c24_dataset" / "dataset_metrics.json")
            ["compute"]["processed_tokens_actual"],
            "steering": steer["compute"]["processed_tokens_actual"],
            "calibration": cal["compute"]["processed_tokens_actual"],
            "endtoend": sum(
                read_json(p)["compute"]["processed_tokens_actual"]
                for p in (OUTPUT_DIR / "c24_endtoend").glob("arm_*.json")
            ),
        },
    }
    summary["processed_tokens"]["total"] = sum(summary["processed_tokens"].values())
    write_json(out / "c24_summary.json", summary)

    print("CLAIM 1:", claim1)
    print("  1a slopes below 1:", c1a, summary["claim_1_calibration"]["1a_slopes"])
    print("  1b AUROC unchanged:", c1b)
    print("  1c ECE halved:", c1c)
    print("  1d identity exact:", c1d,
          summary["claim_1_calibration"]["1d_identical_fractions"])
    print("  1e ratios platt:", summary["claim_1_calibration"]["1e_ratios_platt"])
    print("  1e ratios isotonic:", summary["claim_1_calibration"]["1e_ratios_isotonic"])
    print("CLAIM 2:", verdict_2e)
    print("  2a:", c2a, summary["claim_2_depth"]["2a_best_probe_points"])
    print("  2c per position:", cell[0], "improves", n_pp_improve, "median rel", median_rel)
    print("  2d end to end:", cell[1], "improves", n_e2e_improve,
          summary["claim_2_depth"]["2d_ratios"])
    print("best-of-N ever beaten:",
          summary["any_arm_anywhere_beats_compute_matched_best_of_n"])


if __name__ == "__main__":
    main()
