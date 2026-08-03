"""Phase 5 -- compute-matched best-of-N.

N is solved so that best-of-N spends the same number of processed generator tokens
per *returned* molecule as the guided run it is matched against.  Two accountings
are reported:

  actual          tokens the guided implementation really processed
  full_recompute  tokens the README's full-prefix-recomputation reference would
                  have processed (much larger, so N is much larger)

Wall time is reported separately, because batching changes hardware efficiency and
a token match is not a time match.

    python scripts/06_best_of_n.py --dataset pilot_50k --property clogp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import bestofn  # noqa: E402
from property_to_go.compute import ComputeMeter, solve_best_of_n  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.model_io import load_generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--property", default="clogp")
    ap.add_argument("--guided", default=None, help="guided run directory to match")
    ap.add_argument("--n", type=int, default=None, help="molecules returned per seed")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--accounting", default="both", choices=["actual", "full_recompute", "both"])
    ap.add_argument("--full-recompute-n", type=int, default=256,
                    help="molecules per seed for the (much costlier) full-recompute match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    guided_dir = OUTPUT_DIR / (args.guided or f"{args.dataset}_guided_{args.property}")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_bestofn_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    intervals = read_json(data_dir / "target_intervals.json")
    guided = read_json(guided_dir / "guidance_metrics.json")

    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    seeds = args.seeds or list(gcfg["seeds"])

    tp = guided["conditions"]["throughout"]["aggregate"]["compute_total"]
    base = guided["conditions"]["unguided"]["aggregate"]["compute_total"]
    guided_actual = tp["tokens_per_molecule_actual"]
    guided_full = tp["tokens_per_molecule_full_recompute"]
    base_per_mol = base["tokens_per_molecule_actual"]

    budgets = {}
    if args.accounting in ("actual", "both"):
        budgets["actual"] = (solve_best_of_n(guided_actual, base_per_mol),
                             args.n or int(gcfg["n_molecules_per_condition"]))
    if args.accounting in ("full_recompute", "both"):
        budgets["full_recompute"] = (solve_best_of_n(guided_full, base_per_mol),
                                     args.full_recompute_n)

    print(f"guided tokens/molecule: actual={guided_actual:.0f} full_recompute={guided_full:.0f}")
    print(f"base tokens/molecule:   {base_per_mol:.1f}")
    for k, (n, m) in budgets.items():
        print(f"  match on {k:15s} -> N={n}, {m} molecules/seed")

    gen = load_generator(model_cfg)
    report: dict = {
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": iv,
        "guided_run": str(guided_dir.name),
        "guided_tokens_per_molecule_actual": guided_actual,
        "guided_tokens_per_molecule_full_recompute": guided_full,
        "base_tokens_per_molecule": base_per_mol,
        # Recorded whether or not that accounting is actually run, so a run restricted to
        # `actual` still lets a reader check the saturation argument: at N draws against
        # base rate p, best-of-N misses with probability (1-p)^N.
        "n_candidates_solved": {
            "actual": solve_best_of_n(guided_actual, base_per_mol),
            "full_recompute": solve_best_of_n(guided_full, base_per_mol),
        },
        "base_rate": float(iv["base_rate"]),
        "accounting_run": args.accounting,
        "matches": {},
    }
    t_start = time.perf_counter()

    for accounting, (n_cand, n_mol) in budgets.items():
        entry: dict = {"n_candidates": n_cand, "n_molecules_per_seed": n_mol, "seeds": {}}
        for seed in seeds:
            meter = ComputeMeter().start()
            picked = bestofn.best_of_n(
                gen, policy, args.property, lo, hi, n_mol, n_cand, seed=seed * 1000, meter=meter
            )
            meter.stop()
            s = bestofn.summarise(picked, args.property, lo, hi)
            s["compute"] = meter.as_dict()
            lens = [p.get("n_heavy_atoms") for p in picked if p.get("valid")]
            s["n_heavy_atoms_mean"] = float(np.mean(lens)) if lens else float("nan")
            entry["seeds"][str(seed)] = s
            print(f"  [{accounting}] N={n_cand} seed={seed} hit={s['hit_rate']:.4f} "
                  f"err={s['abs_target_error_mean']:.4f} val={s['validity']:.3f} "
                  f"tok/mol={s['compute']['tokens_per_molecule_actual']:.0f} "
                  f"({meter.wall_seconds:.0f}s)")
        agg = {}
        for k in ("hit_rate", "abs_target_error_mean", "validity", "uniqueness"):
            v = [entry["seeds"][str(s)][k] for s in seeds]
            agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
        agg["tokens_per_molecule_actual"] = float(
            np.mean([entry["seeds"][str(s)]["compute"]["tokens_per_molecule_actual"] for s in seeds])
        )
        agg["wall_seconds_mean"] = float(
            np.mean([entry["seeds"][str(s)]["compute"]["wall_seconds"] for s in seeds])
        )
        entry["aggregate"] = agg
        report["matches"][accounting] = entry

        g = guided["conditions"]["throughout"]["aggregate"]
        report["matches"][accounting]["comparison_vs_guided_throughout"] = {
            "guided_hit_rate": g["hit_rate"]["mean"],
            "best_of_n_hit_rate": agg["hit_rate"]["mean"],
            "guidance_advantage": g["hit_rate"]["mean"] - agg["hit_rate"]["mean"],
            "guided_wall_seconds": g["compute_total"]["wall_seconds"] / len(seeds),
            "best_of_n_wall_seconds": agg["wall_seconds_mean"],
            "guided_validity": g["validity"]["mean"],
            "best_of_n_validity": agg["validity"]["mean"],
        }
        c = report["matches"][accounting]["comparison_vs_guided_throughout"]
        print(f"  [{accounting}] guided={c['guided_hit_rate']:.4f} vs "
              f"best-of-{n_cand}={c['best_of_n_hit_rate']:.4f} "
              f"advantage={c['guidance_advantage']:+.4f}")

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "bestofn_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
