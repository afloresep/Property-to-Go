"""Phase 2 -- what the target-interval/binner misalignment actually cost.

`pilot_report.md` §11.5 documents a defect in the executed pilot: `interval_mask` keeps
only bins lying wholly inside [lo, hi), the target interval was a quantile of the full
sample while the binner was fitted on the train split, and the two never exactly agreed.
For cLogP that dropped one of the two bins the target spans, so the head was trained and
used to predict a 0.050-mass event for a 0.100-base-rate target.

Reasoning about which direction that biases the reported AUROC is not enough -- ranking
the target event by a *subset* of it is a mismatched but positively correlated score, and
the sign is not obvious. So it is measured: two head-training runs on the same phase-2
sample with the same initialisation seed, differing *only* in whether the target
interval's edges are forced to be bin boundaries.

    python scripts/14_interval_mask_impact.py \
        --fixed pilot_50k_heads_p2 --legacy pilot_50k_heads_p2_legacymask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)
from property_to_go.properties import ALL_PROPERTIES  # noqa: E402

#: The head guidance can actually use at decode time (see docs/HANDOFF.md §3.3).
HEAD = "frozen_state"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="pilot_50k_heads_p2")
    ap.add_argument("--legacy", default="pilot_50k_heads_p2_legacymask")
    ap.add_argument("--head", default=HEAD)
    ap.add_argument("--out", default="interval_mask_impact")
    args = ap.parse_args()

    fixed = read_json(OUTPUT_DIR / args.fixed / "head_metrics.json")
    legacy = read_json(OUTPUT_DIR / args.legacy / "head_metrics.json")
    out_dir = OUTPUT_DIR / args.out

    assert not fixed["legacy_interval_mask"], "--fixed run must have the aligned binner"
    assert legacy["legacy_interval_mask"], "--legacy run must have the un-aligned binner"

    report: dict = {
        "fixed_run": args.fixed,
        "legacy_run": args.legacy,
        "head_input": args.head,
        "note": (
            "Identical dataset, identical head-initialisation seed, identical training "
            "recipe. The only difference is whether the target interval's edges are "
            "forced to be bin boundaries. `legacy` reproduces the pilot's behaviour."
        ),
        "properties": {},
    }

    print(f"{'property':<17}{'bins':>9}{'masked/true rate':>26}{'target AUROC':>18}"
          f"{'mean predicted':>20}{'ECE':>16}")
    print(f"{'':<17}{'fix leg':>9}{'fixed':>13}{'legacy':>13}{'fixed':>9}{'legacy':>9}"
          f"{'fixed':>8}{'legacy':>7}{'base':>6}{'fixed':>9}{'legacy':>7}")
    for prop in ALL_PROPERTIES:
        if prop not in fixed["properties"] or prop not in legacy["properties"]:
            continue
        f, g = fixed["properties"][prop], legacy["properties"][prop]
        fc, gc = f["interval_mask_coverage"], g["interval_mask_coverage"]
        ft = f["heads"][args.head]["test"]
        gt = g["heads"][args.head]["test"]
        fi, gi = ft["intervals"]["target"], gt["intervals"]["target"]

        entry = {
            "affected": not gc["is_exact"],
            "n_bins": {"fixed": fc["n_bins"], "legacy": gc["n_bins"]},
            "n_bins_selected": {"fixed": fc["n_bins_selected"], "legacy": gc["n_bins_selected"]},
            "masked_rate": {"fixed": fc["masked_rate"], "legacy": gc["masked_rate"]},
            "true_rate": {"fixed": fc["true_rate"], "legacy": gc["true_rate"]},
            # How badly the legacy head's target probability understates the real one.
            "masked_rate_ratio_legacy_over_true": (
                gc["masked_rate"] / gc["true_rate"] if gc["true_rate"] else None
            ),
            "target_auroc": {"fixed": fi["auroc"], "legacy": gi["auroc"],
                             "difference": fi["auroc"] - gi["auroc"]},
            "target_brier": {"fixed": fi["brier"], "legacy": gi["brier"],
                             "difference": fi["brier"] - gi["brier"]},
            # The pilot recorded this defect as the head being "under-confident": mean
            # predicted target probability far below the base rate. These two columns are
            # where that shows up, and where the fix should remove most of it.
            "mean_predicted_target_prob": {
                "fixed": fi["mean_predicted"], "legacy": gi["mean_predicted"],
                "base_rate": fi["base_rate"],
                "legacy_ratio_to_base_rate": (
                    gi["mean_predicted"] / fi["base_rate"] if fi["base_rate"] else None
                ),
                "fixed_ratio_to_base_rate": (
                    fi["mean_predicted"] / fi["base_rate"] if fi["base_rate"] else None
                ),
            },
            "target_ece": {"fixed": fi["ece"], "legacy": gi["ece"],
                           "difference": fi["ece"] - gi["ece"]},
            # NLL and expected-value MAE never touch the mask, so they are the control:
            # they should be essentially unchanged, and if they are not, something other
            # than the mask moved and this whole comparison is confounded.
            "control_nll": {"fixed": ft["nll"], "legacy": gt["nll"],
                            "difference": ft["nll"] - gt["nll"]},
            "control_expected_value_mae": {
                "fixed": ft["expected_value_mae"], "legacy": gt["expected_value_mae"],
                "difference": ft["expected_value_mae"] - gt["expected_value_mae"],
            },
        }
        by_q = {}
        for q in ("1", "2", "3", "4"):
            fq = f["heads"][args.head]["test_by_quartile"].get(q)
            gq = g["heads"][args.head]["test_by_quartile"].get(q)
            if fq and gq:
                by_q[q] = {
                    "fixed": fq["intervals"]["target"]["auroc"],
                    "legacy": gq["intervals"]["target"]["auroc"],
                    "difference": (fq["intervals"]["target"]["auroc"]
                                   - gq["intervals"]["target"]["auroc"]),
                }
        entry["target_auroc_by_quartile"] = by_q
        report["properties"][prop] = entry

        print(f"{prop:<17}{fc['n_bins_selected']:>5}{gc['n_bins_selected']:>4}"
              f"{fc['masked_rate']:>7.4f}/{fc['true_rate']:.4f}"
              f"{gc['masked_rate']:>8.4f}/{gc['true_rate']:.4f}"
              f"{fi['auroc']:>9.4f}{gi['auroc']:>9.4f}"
              f"{fi['mean_predicted']:>8.4f}{gi['mean_predicted']:>7.4f}{fi['base_rate']:>6.3f}"
              f"{fi['ece']:>9.4f}{gi['ece']:>7.4f}"
              + ("   <-- AFFECTED" if not gc["is_exact"] else ""))

    affected = [p for p, e in report["properties"].items() if e["affected"]]
    report["affected_properties"] = affected
    report["unaffected_properties"] = [
        p for p in report["properties"] if p not in affected
    ]
    report["max_control_nll_shift"] = max(
        abs(e["control_nll"]["difference"]) for e in report["properties"].values()
    )
    write_json(out_dir / "interval_mask_impact.json", report)
    write_run_context(out_dir)

    print(f"\naffected by the legacy mask: {affected}")
    print(f"unaffected: {report['unaffected_properties']}")
    print(f"largest NLL shift on the control metric: {report['max_control_nll_shift']:.4f} "
          f"nats (should be ~0: NLL never uses the mask)")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
