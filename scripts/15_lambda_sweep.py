"""Phase 2 -- the lambda sweep (docs/TODO.md C10), and quality at every lambda (C12).

Reads only; generates nothing. It assembles the runs produced by
`bash scripts/run_phase2.sh lambda` plus the lambda = 1 runs already produced by the
central test, and answers three questions the per-position headroom decomposition
(`pilot_report.md` §15.6) could **not** answer, because that decomposition is a
single-position quantity and these are end-to-end ones:

  1. How much does tuning lambda actually buy, end to end?  Section 15.6 bounded the
     lambda term at a factor of ~2 *per position* and an earlier draft of the report
     read that as an end-to-end bound. It is not one. This measures it.
  2. **P5** (`docs/LEXICAL_LOCALITY.md` §4): does *any* lambda beat compute-matched
     best-of-N?  P5 is falsified if one does.
  3. **C12**: the pilot's null result on chemical quality is specific to lambda = 1 into a
     bounded interval. High lambda is where the literature's degenerate molecules should
     appear, so this is a genuine prediction rather than a box to tick.

Anchors only -- three properties, chosen by a rule fixed before the sweep ran: the most
and least steerable at lambda = 1, plus the pre-registered discriminating case. Depth on
anchors, breadth at lambda = 1 elsewhere (`docs/TODO.md` §E rules out the full grid).

    python scripts/15_lambda_sweep.py --dataset pilot_50k_p2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

#: Anchors, and the rule that picked them. Stated here so the choice is inspectable.
ANCHORS = ("aromatic_rings", "hbd_count", "qed")
ANCHOR_RULE = (
    "most steerable at lambda=1 (aromatic_rings, +0.2949); least steerable "
    "(qed, +0.1012); the pre-registered discriminating case (hbd_count)."
)

#: lambda = 1 comes from the central test rather than being re-run.
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Quality descriptors carried per lambda. `sa_score` and `longest_chain` are the two the
#: molecular-optimisation literature uses to catch reward hacking; `qed` is reported
#: because bounded-interval guidance was observed to improve it at lambda = 1.
QUALITY_KEYS = ("sa_score", "longest_chain", "carbon_fraction", "qed", "n_fragments")


def tag(lam: float) -> str:
    """`0.25 -> lam0p25`, matching the directory names `run_phase2.sh lambda` writes."""
    s = f"{lam:g}".replace(".", "p")
    return f"lam{s}"


def dirs_for(ds: str, prop: str, lam: float) -> dict[str, Path]:
    if lam == 1.0:
        return {"guided": OUTPUT_DIR / f"{ds}_guided_{prop}",
                "bestofn": OUTPUT_DIR / f"{ds}_bestofn_{prop}",
                "quality": OUTPUT_DIR / f"{ds}_quality_{prop}"}
    t = tag(lam)
    return {"guided": OUTPUT_DIR / f"{ds}_{t}_guided_{prop}",
            "bestofn": OUTPUT_DIR / f"{ds}_{t}_bestofn_{prop}",
            "quality": OUTPUT_DIR / f"{ds}_{t}_quality_{prop}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--properties", nargs="*", default=list(ANCHORS))
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    out_dir = OUTPUT_DIR / (args.out or f"{ds}_lambda_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "dataset": ds,
        "anchor_properties": list(args.properties),
        "anchor_selection_rule": ANCHOR_RULE,
        "lambdas": list(args.lambdas),
        "note": (
            "lambda = 1 is the central test's own run, not a re-run: same script, seeds, "
            "n, head and backend. Only `unguided` and `throughout` are generated at the "
            "other lambdas, and `unguided` is regenerated at each one as a bug alarm -- it "
            "cannot depend on lambda and must reproduce exactly."
        ),
        "properties": {},
        "missing": [],
    }

    for prop in args.properties:
        rows: dict[str, dict] = {}
        for lam in args.lambdas:
            d = dirs_for(ds, prop, lam)
            gpath = d["guided"] / "guidance_metrics.json"
            if not gpath.exists():
                report["missing"].append(str(gpath))
                continue
            g = read_json(gpath)
            agg = g["conditions"]["throughout"]["aggregate"]
            ung = g["conditions"]["unguided"]["aggregate"]
            row: dict = {
                "lambda_recorded_in_the_run": g["lambda"],
                "lambda_source": g.get("lambda_source"),
                "guided_dir": d["guided"].name,
                "hit_rate_unguided": ung["hit_rate"]["mean"],
                "hit_rate_throughout": agg["hit_rate"]["mean"],
                "lift": agg["hit_rate"]["mean"] - ung["hit_rate"]["mean"],
                "lift_sd_over_seeds": float(np.std(
                    np.array(agg["hit_rate"]["values"])
                    - np.array(ung["hit_rate"]["values"])
                )),
                "validity": agg["validity"]["mean"],
                "uniqueness": agg["uniqueness"]["mean"],
                "abs_target_error_mean": agg["abs_target_error_mean"]["mean"],
                "content_length_mean": agg["content_length_mean"]["mean"],
                "tokens_per_molecule_actual":
                    agg["compute_total"]["tokens_per_molecule_actual"],
                "length_matched_lift":
                    g["length_confound"]["throughout"]["delta_vs_unguided_length_matched"],
            }
            if float(g["lambda"]) != float(lam):
                raise SystemExit(
                    f"{d['guided'].name} records lambda={g['lambda']}, expected {lam}"
                )

            bpath = d["bestofn"] / "bestofn_metrics.json"
            if bpath.exists():
                b = read_json(bpath)
                m = b["matches"].get("actual")
                if m is not None:
                    c = m["comparison_vs_guided_throughout"]
                    row["best_of_n"] = {
                        "n_candidates": m["n_candidates"],
                        "hit_rate": c["best_of_n_hit_rate"],
                        "guidance_advantage": c["guidance_advantage"],
                        "guidance_beats_best_of_n": bool(c["guidance_advantage"] > 0),
                    }
                # Saturation of the other accounting, checkable rather than assumed.
                ns = b.get("n_candidates_solved", {})
                if ns and b.get("base_rate") is not None:
                    n_full = int(ns["full_recompute"])
                    row["best_of_n_full_recompute_saturation"] = {
                        "n_candidates": n_full,
                        "base_rate": b["base_rate"],
                        "probability_all_n_draws_miss":
                            float((1.0 - b["base_rate"]) ** n_full),
                    }

            qpath = d["quality"] / "quality_metrics.json"
            if qpath.exists():
                q = read_json(qpath)
                hits = q["panels"]["throughout"]["hits"]
                base = q["panels"]["unguided"]["hits"]
                vs = q["vs_unguided_hits"].get("throughout", {})
                row["quality"] = {
                    "n_guided_hits": hits["n"],
                    "n_base_hits": base["n"],
                    "degeneracy_rate_guided_hits": hits["degeneracy_rate"]["any"],
                    "degeneracy_rate_base_hits": base["degeneracy_rate"]["any"],
                    "degeneracy_difference": vs.get("any_degeneracy", {}).get("difference"),
                    "degeneracy_excludes_zero":
                        vs.get("any_degeneracy", {}).get("excludes_zero"),
                    "descriptors": {
                        k: {
                            "guided_hits_mean": hits["descriptors"][k]["mean"],
                            "difference_vs_base_hits": vs.get(k, {}).get("difference"),
                            "excludes_zero": vs.get(k, {}).get("excludes_zero"),
                        }
                        for k in QUALITY_KEYS if k in hits["descriptors"]
                    },
                }
            rows[f"{lam:g}"] = row

        if not rows:
            continue
        lams = sorted(rows, key=float)
        lifts = {k: rows[k]["lift"] for k in lams}
        best = max(lifts, key=lambda k: lifts[k])
        at1 = lifts.get("1")
        adv = {k: rows[k].get("best_of_n", {}).get("guidance_advantage") for k in lams}
        report["properties"][prop] = {
            "by_lambda": rows,
            "best_lambda": float(best),
            "best_lift": lifts[best],
            "lift_at_lambda_1": at1,
            # The end-to-end answer to "what is tuning lambda worth", which is the
            # quantity section 15.6 could only bound per position.
            "gain_from_tuning_lambda": (
                None if not at1 else float(lifts[best] / at1)
            ),
            "lift_is_non_monotonic_in_lambda": bool(
                float(best) < float(lams[-1])
            ),
            # P5: falsified if guidance beats compute-matched best-of-N at ANY lambda.
            "any_lambda_beats_best_of_n": bool(
                any(v is not None and v > 0 for v in adv.values())
            ),
            "best_guidance_advantage": (
                max([v for v in adv.values() if v is not None], default=None)
            ),
            "validity_at_the_largest_lambda": rows[lams[-1]]["validity"],
            "degeneracy_rate_by_lambda": {
                k: rows[k].get("quality", {}).get("degeneracy_rate_guided_hits")
                for k in lams
            },
        }

    write_json(out_dir / "lambda_sweep_metrics.json", report)
    write_run_context(out_dir)

    for prop, r in report["properties"].items():
        print(f"\n{prop}  (best lambda={r['best_lambda']:g}, "
              f"tuning is worth {r['gain_from_tuning_lambda']:.2f}x end to end)")
        print(f"  {'lam':>5s} {'lift':>8s} {'hit':>7s} {'valid':>6s} {'uniq':>6s} "
              f"{'len':>6s} {'bo-N':>7s} {'adv':>8s} {'degen':>7s}")
        for k in sorted(r["by_lambda"], key=float):
            row = r["by_lambda"][k]
            b = row.get("best_of_n", {})
            q = row.get("quality", {})
            print(f"  {k:>5s} {row['lift']:+8.4f} {row['hit_rate_throughout']:7.4f} "
                  f"{row['validity']:6.3f} {row['uniqueness']:6.3f} "
                  f"{row['content_length_mean']:6.1f} "
                  f"{b.get('hit_rate', float('nan')):7.4f} "
                  f"{b.get('guidance_advantage', float('nan')):+8.4f} "
                  f"{q.get('degeneracy_rate_guided_hits', float('nan')):7.4f}")
        print(f"  P5: any lambda beats compute-matched best-of-N? "
              f"{r['any_lambda_beats_best_of_n']} "
              f"(best advantage {r['best_guidance_advantage']:+.4f})")
    if report["missing"]:
        print(f"\nmissing {len(report['missing'])} run(s):")
        for m in report["missing"]:
            print(f"  {m}")
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
