"""C25 step 2 -- train a pooled-readout head for every (property, depth, variant).

Everything except the readout is held at phase 2's values: same dataset, same splits,
same target intervals, same binners, same head recipe, same three head seeds
(1234 / 2345 / 3456), same trainer.  Fixed pooling operators produce a 2-D feature array
and go through `probe_layers.train_one_probe` -- the *same function* C17 used and a
transcription of script 03's `frozen_state` branch -- so `last1` is not merely equivalent
to the deployed readout, it is the identical array through the identical trainer.  The
learned operator (`attn4`) has its own head and trainer in `property_to_go.pooling`, and
its own window-size-1 identity test.

Two depths per property, both fixed before any C25 measurement:

  * probe point 12, the final layer, where §8.3's claim lives;
  * that property's AUROC-best mid-network probe point from §21
    (aromatic_rings 3, hbd_count 4, rotatable_bonds 4, tpsa 5, clogp 5, qed 4).

Validity gate (§C25.0.1): the `last1` variant must reproduce
`outputs/pilot_50k_heads_p2/head_metrics.json` at probe point 12, and
`outputs/c17_probe_layers/probe_layer_metrics.json` at every probe point, to a residual
of **exactly 0.0** in mean AUROC and mean NLL.  The residual is written to the artefact
as a number.

    .venv/bin/python scripts/20_pooled_sweep.py --dataset pilot_50k_p2
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
from property_to_go import pooling as PL  # noqa: E402
from property_to_go import probe_layers as P  # noqa: E402
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, interval_mask_coverage,
)
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.properties import DISCRETE_PROPERTIES  # noqa: E402

#: §21's AUROC-best probe point per property, read off C17's frozen table.  Fixed in
#: §C25.0.3 before any C25 number existed and never re-derived from a C25 outcome.
BEST_MID_PROBE_POINT = {
    "aromatic_rings": 3,
    "hbd_count": 4,
    "rotatable_bonds": 4,
    "tpsa": 5,
    "clogp": 5,
    "qed": 4,
}
FINAL_PROBE_POINT = 12
PROPERTIES = ("aromatic_rings", "hbd_count", "rotatable_bonds", "tpsa", "clogp", "qed")

#: §C25.0.4's multiplicity family: 5 non-baseline variants x 6 properties x 2 depths,
#: minus the 6 cells `wide1` does not have at the final layer (§20.4 already ran it).
N_COMPARISONS_FOR_BONFERRONI = 4 * 6 * 2 + 6

#: The pre-registered work order.  `mid/last1` is second rather than third so that the
#: mid-layer checkpoints priority 4 (head-seed replication, a GPU job) depends on exist
#: early and the GPU and CPU stages can run at the same time.  This changes which
#: results arrive first and changes no measured quantity.
WORK_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("final", ("last1",)),
    ("mid", ("last1",)),
    ("final", ("mean4", "mean16", "concat4", "attn4")),
    ("mid", ("mean4", "mean16", "concat4", "attn4")),
    ("mid", ("wide1",)),
)


def build_binner(prop: str, y_train: np.ndarray, iv: dict, cfg: dict):
    if prop in DISCRETE_PROPERTIES:
        return CategoricalBinner(max_value=int(cfg["binning"][f"{prop}_max"]))
    return QuantileBinner.fit(
        y_train, int(cfg["binning"][f"{prop}_n_bins"]), extra_edges=(iv["lo"], iv["hi"]))


def train_attn_cell(spec, x_stack, counts, y, y_bin, binner, masks, quartile,
                    intervals, head_cfg, head_seed):
    """`probe_layers.train_one_probe` for the learned pool, same structure and metrics."""
    w = spec.window
    sub = np.ascontiguousarray(x_stack[:, x_stack.shape[1] - w:, :])
    mask_w = PL.counts_to_mask(np.minimum(counts, w), w)
    torch.manual_seed(int(head_seed))
    head = PL.AttnPoolHead(
        in_dim=sub.shape[-1],
        hidden_dim=int(spec.hidden_dim or head_cfg["hidden_dim"]),
        n_bins=binner.n_bins,
        dropout=float(head_cfg["dropout"]),
    )
    tr = PL.train_attn_head(
        head, sub[masks["train"]], mask_w[masks["train"]], y_bin[masks["train"]],
        sub[masks["val"]], mask_w[masks["val"]], y_bin[masks["val"]],
        {**head_cfg, "seed": int(head_seed)})
    probs_test = head.predict_proba(sub[masks["test"]], mask_w[masks["test"]])
    entry = {
        "head_seed": int(head_seed),
        "input_dim": int(sub.shape[-1]),
        "best_epoch": tr.best_epoch,
        "epochs_run": len(tr.history),
        "test": M.evaluate(probs_test, y[masks["test"]], y_bin[masks["test"]], binner,
                           intervals),
        "test_by_quartile": M.evaluate_by_group(
            probs_test, y[masks["test"]], y_bin[masks["test"]], quartile[masks["test"]],
            binner, intervals),
    }
    return entry, probs_test, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--config", default="pilot_50k")
    ap.add_argument("--states", default=None)
    ap.add_argument("--reference-heads", default="pilot_50k_heads_p2")
    ap.add_argument("--reference-c17", default="c17_probe_layers")
    ap.add_argument("--head-seeds", type=int, nargs="*", default=[1234, 2345, 3456])
    ap.add_argument("--properties", nargs="*", default=list(PROPERTIES))
    ap.add_argument("--variants", nargs="*", default=None,
                    help="restrict to these variant names; default the whole family")
    ap.add_argument("--depths", nargs="*", default=None, choices=["final", "mid"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads; the box is shared, so this is explicit")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--out", default="c25_pooled_heads")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(int(args.threads))
    data_dir = OUTPUT_DIR / args.dataset
    states_dir = OUTPUT_DIR / (args.states or f"c25_window_states_{args.dataset}")
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    intervals_cfg = read_json(data_dir / "target_intervals.json")
    split = meta["split"].to_numpy()
    quartile = meta["quartile"].to_numpy()
    counts_all = np.load(states_dir / "counts.npy")

    cache: dict[str, dict] = {}
    for prop in args.properties:
        y = meta[prop].to_numpy().astype(np.float64)
        scored = np.isfinite(y)
        masks = {s: (split == s) & scored for s in ("train", "val", "test")}
        iv = intervals_cfg[prop]
        binner = build_binner(prop, y[masks["train"]], iv, cfg)
        cov = interval_mask_coverage(binner, iv["lo"], iv["hi"], y[masks["test"]])
        if not cov["is_exact"]:
            raise SystemExit(f"{prop}: target interval is not a union of bins: {cov}")
        cache[prop] = {"y": y, "masks": masks, "binner": binner, "iv": iv,
                       "intervals": {"target": (iv["lo"], iv["hi"])},
                       "y_bin": binner.transform(y), "coverage": cov}

    depths = args.depths or ["final", "mid"]
    variants = args.variants or [v.name for v in PL.POOL_VARIANTS]

    # ---- the ordered task list ------------------------------------------------------
    tasks: list[tuple[str, int, str, str]] = []
    for depth, group in WORK_ORDER:
        if depth not in depths:
            continue
        for vname in group:
            if vname not in variants:
                continue
            for prop in args.properties:
                L = FINAL_PROBE_POINT if depth == "final" else BEST_MID_PROBE_POINT[prop]
                tasks.append((depth, L, prop, vname))
    print(f"{len(tasks)} cells x {len(args.head_seeds)} head seeds", flush=True)

    t_start = time.perf_counter()
    loaded: dict[int, dict[str, np.ndarray]] = {}

    def layer_arrays(L: int) -> dict[str, np.ndarray]:
        if L not in loaded:
            loaded.clear()          # one layer resident at a time; 2.4 GB each
            d = states_dir / f"layer{L}"
            loaded[L] = {
                "stack": np.load(d / "stack.npy", mmap_mode="r"),
                "mean16": np.load(d / "mean16.npy", mmap_mode="r"),
            }
        return loaded[L]

    for depth, L, prop, vname in tasks:
        cell = out_dir / f"cell_{depth}_L{L}_{prop}_{vname}.json"
        if args.resume and cell.exists():
            print(f"resume {cell.name}", flush=True)
            continue
        spec = PL.VARIANTS_BY_NAME[vname]
        c = cache[prop]
        arrays = layer_arrays(L)
        t0 = time.perf_counter()

        per_seed = []
        probs_seed0 = None
        for hseed in args.head_seeds:
            if spec.mode == "attn":
                stack = np.asarray(arrays["stack"])
                entry, probs, head = train_attn_cell(
                    spec, stack, counts_all, c["y"], c["y_bin"], c["binner"], c["masks"],
                    quartile, c["intervals"], cfg["head"], hseed)
                ck = {"state_dict": head.state_dict(), "in_dim": int(head.in_dim),
                      "hidden_dim": int(spec.hidden_dim or cfg["head"]["hidden_dim"]),
                      "n_bins": c["binner"].n_bins,
                      "dropout": float(cfg["head"]["dropout"]),
                      "attn_dim": int(head.attn_dim)}
            else:
                x = PL.pooled_features(spec, np.asarray(arrays["stack"]), counts_all,
                                       mean_wide=np.asarray(arrays["mean16"]))
                head_cfg = dict(cfg["head"])
                if spec.hidden_dim:
                    head_cfg["hidden_dim"] = int(spec.hidden_dim)
                entry, probs, head = P.train_one_probe(
                    x, c["y"], c["y_bin"], c["binner"], c["masks"], quartile,
                    c["intervals"], head_cfg, hseed)
                ck = {"state_dict": head.state_dict(), "in_dim": int(x.shape[1]),
                      "hidden_dim": int(head_cfg["hidden_dim"]),
                      "n_bins": c["binner"].n_bins,
                      "dropout": float(cfg["head"]["dropout"])}
                del x
            per_seed.append(entry)
            if probs_seed0 is None:
                probs_seed0 = probs
            ck.update({"binner": c["binner"].to_dict(), "property": prop,
                       "input": "frozen_state", "head_seed": int(hseed),
                       "probe_point": int(L), "pool_variant": spec.name,
                       "pool_window": int(spec.window), "pool_mode": spec.mode})
            torch.save(ck, out_dir / f"head_{prop}_{vname}_L{L}_seed{hseed}.pt")

        across = P.across_seeds(per_seed)
        record = {
            "depth": depth, "probe_point": int(L), "property": prop,
            "variant": spec.name, "window": int(spec.window), "mode": spec.mode,
            "hidden_dim": int(spec.hidden_dim or cfg["head"]["hidden_dim"]),
            "per_seed": [{k: v for k, v in e.items() if k != "test_by_quartile"}
                         for e in per_seed],
            "test_by_quartile_seed0": per_seed[0]["test_by_quartile"],
            "across_seeds": across,
            "wall_seconds": time.perf_counter() - t0,
        }
        np.save(out_dir / f"probs_{prop}_{vname}_L{L}.npy", probs_seed0)
        write_json(cell, record)
        print(f"{depth:5s} L{L:<3d} {prop:16s} {vname:8s} "
              f"auroc={across['auroc']['mean']:.4f}±{across['auroc']['std']:.4f} "
              f"nll={across['nll']['mean']:.4f} ({record['wall_seconds']:.0f}s)",
              flush=True)

    # ---- assemble ------------------------------------------------------------------
    report: dict = {
        "dataset": args.dataset,
        "states_dir": states_dir.name,
        "head_seeds": list(args.head_seeds),
        "final_probe_point": FINAL_PROBE_POINT,
        "best_mid_probe_point": BEST_MID_PROBE_POINT,
        "bonferroni_family_size": N_COMPARISONS_FOR_BONFERRONI,
        "variants": {v.name: {"window": v.window, "mode": v.mode,
                              "hidden_dim": v.hidden_dim or int(cfg["head"]["hidden_dim"]),
                              "precomputable": v.precomputable}
                     for v in PL.POOL_VARIANTS},
        "cells": {},
    }
    for f in sorted(out_dir.glob("cell_*.json")):
        r = read_json(f)
        report["cells"][f.stem[len("cell_"):]] = r

    # ---- paired bootstrap, variant minus the seed-matched last1 at the same depth ----
    comparisons = []
    for key, r in report["cells"].items():
        if r["variant"] == "last1":
            continue
        prop, L = r["property"], r["probe_point"]
        base_key = f"{r['depth']}_L{L}_{prop}_last1"
        if base_key not in report["cells"]:
            continue
        base = report["cells"][base_key]
        pv = out_dir / f"probs_{prop}_{r['variant']}_L{L}.npy"
        pb = out_dir / f"probs_{prop}_last1_L{L}.npy"
        if not (pv.exists() and pb.exists()):
            continue
        c = cache[prop]
        yt = c["y"][c["masks"]["test"]]
        hit = (yt >= c["iv"]["lo"]) & (yt < c["iv"]["hi"])
        qv = P.target_interval_scores(np.load(pv), c["binner"], c["iv"]["lo"], c["iv"]["hi"])
        qb = P.target_interval_scores(np.load(pb), c["binner"], c["iv"]["lo"], c["iv"]["hi"])
        boot = P.paired_bootstrap_diff(
            lambda s, h: M.auroc(s, h), (qv, hit), (qb, hit), n_boot=args.n_boot,
            alpha=0.05 / N_COMPARISONS_FOR_BONFERRONI)
        boot_nominal = P.paired_bootstrap_diff(
            lambda s, h: M.auroc(s, h), (qv, hit), (qb, hit), n_boot=args.n_boot)
        comparisons.append({
            "depth": r["depth"], "probe_point": L, "property": prop,
            "variant": r["variant"],
            "auroc_variant": r["across_seeds"]["auroc"]["mean"],
            "auroc_last1": base["across_seeds"]["auroc"]["mean"],
            "auroc_margin": (r["across_seeds"]["auroc"]["mean"]
                             - base["across_seeds"]["auroc"]["mean"]),
            "auroc_sd_variant": r["across_seeds"]["auroc"]["std"],
            "auroc_sd_last1": base["across_seeds"]["auroc"]["std"],
            "nll_variant": r["across_seeds"]["nll"]["mean"],
            "nll_last1": base["across_seeds"]["nll"]["mean"],
            "bootstrap_bonferroni": boot,
            "bootstrap_nominal": boot_nominal,
            "clears_material_margin": bool(
                r["across_seeds"]["auroc"]["mean"] - base["across_seeds"]["auroc"]["mean"]
                >= P.MATERIAL_MARGIN),
            "corrected_ci_excludes_zero": bool(boot["lo"] > 0),
        })
        comparisons[-1]["helps"] = bool(
            comparisons[-1]["clears_material_margin"]
            and comparisons[-1]["corrected_ci_excludes_zero"])
    report["comparisons"] = comparisons

    # ---- validity gate C25.0.1 -------------------------------------------------------
    gate_rows = []
    ref13 = OUTPUT_DIR / args.reference_heads / "head_metrics.json"
    ref17 = OUTPUT_DIR / args.reference_c17 / "probe_layer_metrics.json"
    r13 = read_json(ref13) if ref13.exists() else None
    r17 = read_json(ref17) if ref17.exists() else None
    for key, r in report["cells"].items():
        if r["variant"] != "last1":
            continue
        mine = r["across_seeds"]
        if r["probe_point"] == FINAL_PROBE_POINT and r13 is not None:
            ref = r13["properties"][r["property"]]["heads"]["frozen_state"]["across_seeds"]
            gate_rows.append({
                "reference": "pilot_report §13 (pilot_50k_heads_p2/head_metrics.json)",
                "property": r["property"], "probe_point": r["probe_point"],
                "reference_auroc": ref["auroc"]["mean"], "auroc": mine["auroc"]["mean"],
                "auroc_residual": abs(ref["auroc"]["mean"] - mine["auroc"]["mean"]),
                "reference_nll": ref["nll"]["mean"], "nll": mine["nll"]["mean"],
                "nll_residual": abs(ref["nll"]["mean"] - mine["nll"]["mean"])})
        if r17 is not None:
            ref = (r17["properties"][r["property"]]["layers"][str(r["probe_point"])]
                   ["across_seeds"])
            gate_rows.append({
                "reference": "C17 (c17_probe_layers/probe_layer_metrics.json)",
                "property": r["property"], "probe_point": r["probe_point"],
                "reference_auroc": ref["auroc"]["mean"], "auroc": mine["auroc"]["mean"],
                "auroc_residual": abs(ref["auroc"]["mean"] - mine["auroc"]["mean"]),
                "reference_nll": ref["nll"]["mean"], "nll": mine["nll"]["mean"],
                "nll_residual": abs(ref["nll"]["mean"] - mine["nll"]["mean"])})
    max_res = max([max(g["auroc_residual"], g["nll_residual"]) for g in gate_rows],
                  default=float("nan"))
    report["validity_gate"] = {
        "rule": "window size 1 must reproduce the deployed single-position head exactly",
        "per_cell": gate_rows,
        "max_residual": max_res,
        "passes_at_exactly_zero": bool(gate_rows) and max_res == 0.0,
    }

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "pooled_metrics.json", report)
    write_json(out_dir / "config_used.json", cfg)
    write_run_context(out_dir, {"pilot": cfg})
    print(f"\nvalidity gate max residual: {max_res!r} "
          f"({'PASS' if report['validity_gate']['passes_at_exactly_zero'] else 'CHECK'})")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
