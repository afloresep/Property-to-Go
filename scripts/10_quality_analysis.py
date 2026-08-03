"""Are the steered molecules chemically plausible, or did guidance cheat?

Hit rate is blind to how the target was reached. This scores every molecule already
saved by script 05 on the descriptors the molecular-optimisation literature uses to
catch degenerate solutions (synthetic accessibility, drug-likeness, longest acyclic
alkyl chain, carbon fraction, ring sizes) and asks one question:

    among molecules that HIT the target, are the guided ones worse than the
    base-policy ones?

The restriction to hits is the whole point. Comparing all guided molecules against
all unguided molecules confounds quality with the property shift itself -- oilier
molecules are legitimately greasier and more flexible. Comparing hits against hits
holds the achieved property roughly fixed and asks only how the molecule got there.

The `unguided (hit target)` row is also the baseline's actual output: compute-matched
best-of-N returns base-policy samples selected for target proximity, so its molecules
are drawn from exactly that set. No extra generation is needed to characterise it.

    python scripts/10_quality_analysis.py --dataset pilot_50k --property clogp

Reads  outputs/<dataset>_guided_<property>/molecules.json
Writes outputs/<dataset>_quality_<property>/quality_metrics.json
       outputs/<dataset>_quality_<property>/examples.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.properties import ALL_PROPERTIES  # noqa: E402
from property_to_go.quality import (  # noqa: E402
    HIGHER_IS_BETTER, HIGHER_IS_WORSE, bootstrap_difference, degeneracy_flags,
    molecule_quality, quality_panel,
)

#: Descriptors printed in the terminal table; the JSON keeps all of them.
HEADLINE = ("sa_score", "qed", "longest_chain", "carbon_fraction", "max_ring_size")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--property", default="clogp", choices=list(ALL_PROPERTIES))
    ap.add_argument("--guided", default=None, help="override guided run directory name")
    ap.add_argument("--n-examples", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    guided_dir = OUTPUT_DIR / (args.guided or f"{args.dataset}_guided_{args.property}")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_quality_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = read_json(data_dir / "target_intervals.json")
    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    mols = read_json(guided_dir / "molecules.json")
    # The guidance strength can be overridden per run (scripts/05 --lam), so the value
    # that actually produced these molecules lives in the guided run's own metrics, not
    # in configs/guidance.yaml. Carry it through or a lambda sweep's quality artefacts
    # all claim lambda=1.
    guided_metrics = read_json(guided_dir / "guidance_metrics.json")
    lam = float(guided_metrics["lambda"])

    t_start = time.perf_counter()
    print(f"property={args.property} target=[{lo:.4f},{hi:.4f})")
    print(f"scoring molecules from {guided_dir.name}")

    # ---- score every molecule once ------------------------------------------
    scored: dict[str, list[dict]] = {}
    for cond, by_seed in mols.items():
        rows = []
        for seed, records in by_seed.items():
            for r in records:
                if not r.get("valid") or r.get(args.property) is None:
                    continue
                q = molecule_quality(r["smiles"])
                if q is None:
                    continue
                rows.append({
                    "seed": int(seed),
                    "smiles": r["smiles"],
                    "value": float(r[args.property]),
                    "hit": bool(lo <= r[args.property] < hi),
                    "n_content_tokens": int(r["n_content_tokens"]),
                    "n_heavy_atoms": int(r["n_heavy_atoms"]),
                    **q,
                })
        scored[cond] = rows
        print(f"  {cond:20s} {len(rows)} valid molecules, {sum(r['hit'] for r in rows)} hits")

    # ---- panels, for every molecule and for hits only ------------------------
    report: dict = {
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": iv,
        "guided_run": guided_dir.name,
        "lambda": lam,
        "note": (
            "The 'hits' panels are the comparison that matters: they hold the achieved "
            "property roughly fixed and ask only how the molecule got there. The "
            "unguided hits panel also describes compute-matched best-of-N's output, "
            "since best-of-N returns base-policy samples selected for target proximity."
        ),
        "panels": {},
        "vs_unguided_hits": {},
    }
    for cond, rows in scored.items():
        report["panels"][cond] = {
            "all": quality_panel([{k: r[k] for k in r} for r in rows]),
            "hits": quality_panel([{k: r[k] for k in r} for r in rows if r["hit"]]),
        }

    # ---- guided hits vs base-policy hits ------------------------------------
    base_hits = [r for r in scored.get("unguided", []) if r["hit"]]
    for cond, rows in scored.items():
        if cond == "unguided":
            continue
        cond_hits = [r for r in rows if r["hit"]]
        entry: dict = {"n_cond_hits": len(cond_hits), "n_base_hits": len(base_hits)}
        for key in HIGHER_IS_WORSE + HIGHER_IS_BETTER:
            entry[key] = bootstrap_difference(
                np.array([r[key] for r in cond_hits], dtype=float),
                np.array([r[key] for r in base_hits], dtype=float),
                n_boot=args.bootstrap,
                seed=17,
            )
            entry[key]["direction"] = (
                "higher is worse" if key in HIGHER_IS_WORSE else "higher is better"
            )
        base_flags = [any(degeneracy_flags(r).values()) for r in base_hits]
        cond_flags = [any(degeneracy_flags(r).values()) for r in cond_hits]
        entry["any_degeneracy"] = bootstrap_difference(
            np.array(cond_flags, dtype=float), np.array(base_flags, dtype=float),
            n_boot=args.bootstrap, seed=17,
        )
        report["vs_unguided_hits"][cond] = entry

    # ---- examples a human can look at ---------------------------------------
    examples: dict = {}
    for cond, rows in scored.items():
        hits = [r for r in rows if r["hit"]]
        worst = sorted(hits, key=lambda r: -r["sa_score"])[: args.n_examples]
        longest = sorted(hits, key=lambda r: -r["longest_chain"])[: args.n_examples]
        rng = np.random.default_rng(0)
        pick = rng.choice(len(hits), min(args.n_examples, len(hits)), replace=False) if hits else []
        keep = ("smiles", "value", "sa_score", "qed", "longest_chain", "carbon_fraction",
                "max_ring_size", "n_heavy_atoms")
        examples[cond] = {
            "random_hits": [{k: hits[i][k] for k in keep} for i in pick],
            "worst_sa_hits": [{k: r[k] for k in keep} for r in worst],
            "longest_chain_hits": [{k: r[k] for k in keep} for r in longest],
        }

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "quality_metrics.json", report)
    write_json(out_dir / "examples.json", examples)
    write_run_context(out_dir, {"guidance": {**load_config("guidance"), "lam": lam}})

    # ---- terminal table ------------------------------------------------------
    print(f"\nquality of molecules that HIT [{lo:.3f},{hi:.3f}) "
          f"(unguided hits == compute-matched best-of-N output)")
    head = "condition".ljust(20) + "n".rjust(6) + "".join(k.rjust(16) for k in HEADLINE) \
        + "degenerate".rjust(12)
    print(head)
    for cond in scored:
        p = report["panels"][cond]["hits"]
        if not p.get("n"):
            continue
        line = cond.ljust(20) + str(p["n"]).rjust(6)
        for k in HEADLINE:
            line += f"{p['descriptors'][k]['mean']:.3f}".rjust(16)
        line += f"{p['degeneracy_rate']['any']:.3f}".rjust(12)
        print(line)

    print("\nguided hits minus base-policy hits (95% bootstrap CI; * excludes zero)")
    for cond, e in report["vs_unguided_hits"].items():
        parts = []
        for k in HEADLINE:
            if k not in e:
                continue
            d = e[k]
            parts.append(f"{k}={d['difference']:+.3f}[{d['lo']:+.3f},{d['hi']:+.3f}]"
                         f"{'*' if d['excludes_zero'] else ''}")
        print(f"  {cond:20s} " + "  ".join(parts))
        d = e["any_degeneracy"]
        print(f"  {'':20s} degenerate={d['difference']:+.4f}"
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]{'*' if d['excludes_zero'] else ''}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
