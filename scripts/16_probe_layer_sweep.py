"""C17 step 2 -- train a frozen-state head at every probe point, for six properties.

Consumes `scripts/16_extract_layer_states.py`'s output.  Everything except the probe
point is held at phase 2's values: same dataset, same splits, same target intervals, same
binners, same head recipe, same three head seeds (1234 / 2345 / 3456).  The `trivial`
head does not depend on the layer, so it is trained once per property and reused as the
comparison line for all 13 points.

The scoring rule is pre-registered in `reports/section_c17_probe_layers.md` §C17.0 and is
applied by `property_to_go.probe_layers.crossover_verdict` / `isolated_spike`, so the
verdict in `probe_layer_metrics.json` is a lookup rather than a judgement formed after
seeing the table.

Validity gate (C17.0.2): the probe-point-12 heads must reproduce
`outputs/pilot_50k_heads_p2/head_metrics.json` -- mean AUROC and mean NLL per property to
four decimals.  If the recipe here is not script 03's recipe, every cross-layer difference
below is measuring the refactor instead of the layer, so the gate is checked and reported
before anything else is read.

    .venv/bin/python scripts/16_probe_layer_sweep.py --dataset pilot_50k_p2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import metrics as M  # noqa: E402
from property_to_go import probe_layers as P  # noqa: E402
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, interval_mask_coverage,
)
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.properties import (  # noqa: E402
    DISCRETE_PROPERTIES, PREDICTED_LOCALITY_ORDER,
)

#: C17.0.6 rule 1. 13 probe points in the aromatic-ring family.
N_PROBE_POINTS_FOR_BONFERRONI = 13


def build_binner(prop: str, y_train: np.ndarray, iv: dict, cfg: dict):
    """Script 03's binner construction, including the §3.6 `extra_edges` fix."""
    if prop in DISCRETE_PROPERTIES:
        return CategoricalBinner(max_value=int(cfg["binning"][f"{prop}_max"]))
    return QuantileBinner.fit(
        y_train, int(cfg["binning"][f"{prop}_n_bins"]), extra_edges=(iv["lo"], iv["hi"])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--config", default="pilot_50k")
    ap.add_argument("--states", default=None,
                    help="dir written by scripts/16_extract_layer_states.py")
    ap.add_argument("--reference-heads", default="pilot_50k_heads_p2",
                    help="artefact the probe-point-12 validity gate is checked against")
    ap.add_argument("--head-seeds", type=int, nargs="*", default=[1234, 2345, 3456])
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="probe points; default every layer present in --states")
    ap.add_argument("--properties", nargs="*", default=list(PREDICTED_LOCALITY_ORDER))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore any partial_*.json in --out and retrain everything")
    ap.add_argument("--out", default="c17_probe_layers")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    states_dir = OUTPUT_DIR / (args.states or f"c17_layer_states_{args.dataset}")
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    layers = args.layers
    if layers is None:
        layers = sorted(int(p.name[5:]) for p in states_dir.glob("layer*") if p.is_dir())
    reference_layer = max(layers)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    features = np.load(data_dir / "features.npy")
    intervals_cfg = read_json(data_dir / "target_intervals.json")
    split = meta["split"].to_numpy()
    quartile = meta["quartile"].to_numpy()
    head_seeds = list(args.head_seeds)

    report: dict = {
        "dataset": args.dataset,
        "states_dir": states_dir.name,
        "n_rows": int(len(meta)),
        "probe_points": list(layers),
        "reference_probe_point": int(reference_layer),
        "head_seeds": head_seeds,
        "properties": {},
        "preregistration": {
            "document": "reports/section_c17_probe_layers.md §C17.0",
            "primary_metric": "held-out target-interval AUROC, mean over head seeds",
            "secondary_metric": "held-out NLL",
            "seed_sd": P.SEED_SD,
            "material_margin": P.MATERIAL_MARGIN,
            "neighbour_margin": P.NEIGHBOUR_MARGIN,
            "bonferroni_family_size": N_PROBE_POINTS_FOR_BONFERRONI,
        },
    }
    t_start = time.perf_counter()

    # Layer arrays are ~612 MB each, so the loop is layer-major: load once, train every
    # property against it, drop it. Property-major would re-read 6x.
    cache: dict[str, dict] = {}

    for prop in args.properties:
        y = meta[prop].to_numpy().astype(np.float64)
        scored = np.isfinite(y)
        masks = {s: (split == s) & scored for s in ("train", "val", "test")}
        iv = intervals_cfg[prop]
        intervals = {"target": (iv["lo"], iv["hi"])}
        binner = build_binner(prop, y[masks["train"]], iv, cfg)
        coverage = interval_mask_coverage(binner, iv["lo"], iv["hi"], y[masks["test"]])
        if not coverage["is_exact"]:
            raise SystemExit(f"{prop}: target interval is not a union of bins: {coverage}")
        cache[prop] = {
            "y": y, "masks": masks, "binner": binner, "intervals": intervals, "iv": iv,
            "y_bin": binner.transform(y),
        }
        report["properties"][prop] = {
            "n_bins": binner.n_bins,
            "target_interval": iv,
            "interval_mask_coverage": coverage,
            "split_sizes": {s: int(m.sum()) for s, m in masks.items()},
            "layers": {},
        }

    # ---- the layer-independent comparison line -----------------------------------
    # Checkpointed to disk, like every layer below. 13 layers x 6 properties x 3 seeds is
    # a multi-hour CPU job and losing it to an interrupted shell is a pure waste; the
    # partials also make a half-finished run inspectable instead of opaque. Resume is
    # exact rather than approximate: every head is a deterministic function of (seed,
    # array, recipe), so a resumed layer is the layer a single uninterrupted run would
    # have produced, and the probe-point-12 validity gate would catch it if it were not.
    trivial_json = out_dir / "partial_trivial.json"
    trivial_npz = out_dir / "partial_trivial_probs.npz"
    if args.resume and trivial_json.exists() and trivial_npz.exists():
        stored = read_json(trivial_json)
        probs_store = np.load(trivial_npz)
        for prop in args.properties:
            cache[prop]["trivial_probs"] = probs_store[prop]
            cache[prop]["trivial_across"] = stored[prop]["across_seeds"]
            report["properties"][prop]["trivial"] = stored[prop]
        print(f"resumed the trivial baseline from {trivial_json.name}")
    else:
        probs_out = {}
        for prop in args.properties:
            c = cache[prop]
            per_seed = []
            for hseed in head_seeds:
                entry, probs, _ = P.train_one_probe(
                    features, c["y"], c["y_bin"], c["binner"], c["masks"], quartile,
                    c["intervals"], cfg["head"], hseed,
                )
                per_seed.append(entry)
                if hseed == head_seeds[0]:
                    c["trivial_probs"] = probs
                    probs_out[prop] = probs
            c["trivial_across"] = P.across_seeds(per_seed)
            report["properties"][prop]["trivial"] = {
                "per_seed": [{k: v for k, v in e.items() if k != "test_by_quartile"}
                             for e in per_seed],
                "across_seeds": c["trivial_across"],
            }
            print(f"{prop:16s} trivial      auroc={c['trivial_across']['auroc']['mean']:.4f}"
                  f"±{c['trivial_across']['auroc']['std']:.4f} "
                  f"nll={c['trivial_across']['nll']['mean']:.4f}", flush=True)
        write_json(trivial_json, {p: report["properties"][p]["trivial"]
                                  for p in args.properties})
        np.savez_compressed(trivial_npz, **probs_out)

    # ---- the sweep ------------------------------------------------------------------
    for L in layers:
        partial = out_dir / f"partial_L{L}.json"
        if args.resume and partial.exists():
            stored = read_json(partial)
            for prop in args.properties:
                report["properties"][prop]["layers"][str(L)] = stored[prop]
            print(f"resumed probe point {L} from {partial.name}", flush=True)
            continue
        x_all = np.load(states_dir / f"layer{L}" / "hidden.npy")
        for prop in args.properties:
            c = cache[prop]
            per_seed = []
            for hseed in head_seeds:
                entry, probs, head = P.train_one_probe(
                    x_all, c["y"], c["y_bin"], c["binner"], c["masks"], quartile,
                    c["intervals"], cfg["head"], hseed,
                )
                per_seed.append(entry)
                if hseed == head_seeds[0]:
                    frozen_probs = probs
                    torch.save(
                        {"state_dict": head.state_dict(), "in_dim": int(x_all.shape[1]),
                         "hidden_dim": int(cfg["head"]["hidden_dim"]),
                         "n_bins": c["binner"].n_bins,
                         "dropout": float(cfg["head"]["dropout"]),
                         "binner": c["binner"].to_dict(), "property": prop,
                         "input": "frozen_state", "head_seed": int(hseed),
                         "probe_point": int(L)},
                        out_dir / f"head_{prop}_frozen_state_L{L}.pt",
                    )
            across = P.across_seeds(per_seed)

            # Paired bootstrap against the trivial head on the identical held-out rows,
            # at the nominal level AND at C17.0.6's Bonferroni-corrected level.
            yt = c["y"][c["masks"]["test"]]
            ybt = c["y_bin"][c["masks"]["test"]]
            hit = (yt >= c["iv"]["lo"]) & (yt < c["iv"]["hi"])
            qf = P.target_interval_scores(frozen_probs, c["binner"], c["iv"]["lo"], c["iv"]["hi"])
            qt = P.target_interval_scores(c["trivial_probs"], c["binner"], c["iv"]["lo"], c["iv"]["hi"])
            auroc_diff = P.paired_bootstrap_diff(
                lambda s, h: M.auroc(s, h), (qf, hit), (qt, hit), n_boot=args.n_boot
            )
            auroc_diff_bonf = P.paired_bootstrap_diff(
                lambda s, h: M.auroc(s, h), (qf, hit), (qt, hit), n_boot=args.n_boot,
                alpha=0.05 / N_PROBE_POINTS_FOR_BONFERRONI,
            )
            # Script 03's orientation: trivial minus frozen, so positive = frozen better.
            nll_diff = P.paired_bootstrap_diff(
                lambda p, b: M.categorical_nll(p, b), (c["trivial_probs"], ybt),
                (frozen_probs, ybt), n_boot=args.n_boot,
            )
            report["properties"][prop]["layers"][str(L)] = {
                "probe_point": int(L),
                "per_seed": [{k: v for k, v in e.items() if k != "test_by_quartile"}
                             for e in per_seed],
                "test_by_quartile_seed0": per_seed[0]["test_by_quartile"],
                "across_seeds": across,
                "frozen_minus_trivial": {
                    "auroc_gain": auroc_diff,
                    "auroc_gain_bonferroni": auroc_diff_bonf,
                    "nll_gain_nats": nll_diff,
                },
            }
            print(f"L{L:<3d} {prop:16s} auroc={across['auroc']['mean']:.4f}"
                  f"±{across['auroc']['std']:.4f} nll={across['nll']['mean']:.4f}"
                  f"±{across['nll']['std']:.4f}  vs trivial "
                  f"{across['auroc']['mean'] - c['trivial_across']['auroc']['mean']:+.4f}",
                  flush=True)
        del x_all
        write_json(partial, {p: report["properties"][p]["layers"][str(L)]
                             for p in args.properties})

    # ---- validity gate C17.0.2 -----------------------------------------------------
    gate = {"checked": False}
    ref_path = OUTPUT_DIR / args.reference_heads / "head_metrics.json"
    if ref_path.exists():
        ref = read_json(ref_path)
        rows = []
        ok = True
        for prop in args.properties:
            if prop not in ref["properties"]:
                continue
            r = ref["properties"][prop]["heads"]["frozen_state"].get("across_seeds")
            if r is None:
                continue
            mine = report["properties"][prop]["layers"][str(reference_layer)]["across_seeds"]
            d_auroc = abs(r["auroc"]["mean"] - mine["auroc"]["mean"])
            d_nll = abs(r["nll"]["mean"] - mine["nll"]["mean"])
            ok &= (d_auroc < 5e-5) and (d_nll < 5e-5)
            rows.append({"property": prop,
                         "reference_auroc": r["auroc"]["mean"], "auroc": mine["auroc"]["mean"],
                         "abs_auroc_difference": d_auroc,
                         "reference_nll": r["nll"]["mean"], "nll": mine["nll"]["mean"],
                         "abs_nll_difference": d_nll})
        gate = {"checked": True, "reference": args.reference_heads,
                "probe_point": int(reference_layer),
                "reproduces_to_4dp": bool(ok), "per_property": rows}
        print(f"\nvalidity gate (probe point {reference_layer} vs {args.reference_heads}): "
              f"{'PASS' if ok else 'FAIL'}")
        for r in rows:
            print(f"  {r['property']:16s} auroc {r['reference_auroc']:.4f} -> {r['auroc']:.4f} "
                  f"(Δ{r['abs_auroc_difference']:.2e})  nll {r['reference_nll']:.4f} -> "
                  f"{r['nll']:.4f} (Δ{r['abs_nll_difference']:.2e})")
    report["validity_gate"] = gate

    # ---- the pre-registered verdicts ------------------------------------------------
    verdicts: dict = {}
    for prop in args.properties:
        auroc_by_layer = {
            int(L): report["properties"][prop]["layers"][str(L)]["across_seeds"]["auroc"]["mean"]
            for L in layers
        }
        nll_by_layer = {
            int(L): report["properties"][prop]["layers"][str(L)]["across_seeds"]["nll"]["mean"]
            for L in layers
        }
        triv = report["properties"][prop]["trivial"]["across_seeds"]
        v = P.crossover_verdict(auroc_by_layer, triv["auroc"]["mean"])
        best = v["best_layer"]
        bonf = (report["properties"][prop]["layers"][str(best)]
                ["frozen_minus_trivial"]["auroc_gain_bonferroni"])
        v["bonferroni_ci_excludes_zero_at_best_layer"] = bool(bonf["lo"] > 0)
        v["bonferroni_ci_at_best_layer"] = bonf
        v["nll_at_best_layer"] = nll_by_layer[best]
        v["trivial_nll"] = triv["nll"]["mean"]
        v["nll_not_worse_than_trivial"] = bool(nll_by_layer[best] <= triv["nll"]["mean"])
        v["artefact_claim_fires"] = bool(
            v["verdict_auroc_arm"] == "ARTEFACT"
            and v["bonferroni_ci_excludes_zero_at_best_layer"]
            and v["nll_not_worse_than_trivial"]
        )
        v["metric_dependent"] = bool(
            v["verdict_auroc_arm"] == "ARTEFACT" and not v["nll_not_worse_than_trivial"]
        )
        v["spike_check_vs_reference_layer"] = P.isolated_spike(
            auroc_by_layer, best, reference_layer
        )
        v["auroc_by_layer"] = auroc_by_layer
        v["nll_by_layer"] = nll_by_layer
        v["argmax_layer_by_auroc"] = int(best)
        v["argmin_layer_by_nll"] = int(min(nll_by_layer, key=lambda L: nll_by_layer[L]))
        verdicts[prop] = v
    report["verdicts"] = verdicts

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "probe_layer_metrics.json", report)
    write_json(out_dir / "config_used.json", cfg)
    write_run_context(out_dir, {"pilot": cfg})

    print("\nverdicts (AUROC arm of the C17.0.4 rule; see the JSON for the side conditions)")
    for prop, v in verdicts.items():
        print(f"  {prop:16s} best L{v['best_layer']:<3d} {v['best_auroc']:.4f} "
              f"vs trivial {v['trivial_auroc']:.4f} ({v['margin_over_trivial']:+.4f})  "
              f"-> {v['verdict_auroc_arm']}"
              f"{'  [ARTEFACT CLAIM FIRES]' if v['artefact_claim_fires'] else ''}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
