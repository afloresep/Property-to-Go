"""C18 step 3b -- retrain the READOUT (not the generator) in two directions.

`docs/HANDOFF.md` §7 and the C18 brief both permit this: the base generator is frozen,
the probe head is not.  Two variants, chosen because they attack the two different
things a readout can be wrong about, and both are deliberately loadable by the
*unmodified* `scripts/05_guided_generation.py`, so the end-to-end measurement needs no
new decoding code and inherits every test that already covers it.

**`wide`** -- the same two-layer MLP with `hidden_dim` 1024 instead of 256.  Pure
capacity.  `pilot_report.md` §8.3 records "one head architecture (two-layer MLP, 256
units)" as an explicit limit on the negative result; this is the cheapest way to find
out whether the limit binds.

**`focused`** -- a three-bin readout whose *middle bin is exactly the target interval*.
The deployed head predicts a 20-way (or up-to-13-way) distribution and guidance then
reads off a marginal, so the head spends its capacity on discriminations the decoder
never consumes.  A three-way below / inside / above head optimises the event guidance
actually scores.  This needs no new binner class: `QuantileBinner` with
`edges = [-inf, lo, hi, inf]` *is* that readout, and its `interval_mask` selects
exactly the middle bin, so the artefact format, the checkpoint loader and
`interval_mask_coverage` all keep working unchanged.

**`wide_focused`** -- both.

Training data is the phase-2 dataset's base-policy prefixes and nothing else.  Training
on guided prefixes would be a second DAgger round, which is forbidden (§9.2.1).

    .venv/bin/python scripts/17_train_head_variants.py --variants wide focused wide_focused
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
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, interval_mask_coverage, interval_probability,
)
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MLPHead, train_head  # noqa: E402
from property_to_go.properties import DISCRETE_PROPERTIES, LOCALITY_BATTERY  # noqa: E402

VARIANTS = {
    #                 hidden_dim   readout
    "baseline_repro": (256, "full"),
    "wide": (1024, "full"),
    "focused": (256, "focused"),
    "wide_focused": (1024, "focused"),
}


def focused_binner(y_train: np.ndarray, lo: float, hi: float) -> QuantileBinner:
    """Three bins: [-inf, lo), [lo, hi), [hi, inf).

    Built directly rather than through `QuantileBinner.fit` because the edges are the
    target interval itself, not quantiles.  `centers` are train medians so
    `expected_value` -- and therefore the binning-invariant MAE control of §11.6 -- keeps
    its meaning; an empty bin falls back to its own edge.
    """
    edges = np.array([-np.inf, float(lo), float(hi), np.inf], dtype=np.float64)
    idx = np.clip(np.searchsorted(edges[1:-1], y_train, side="right"), 0, 2)
    centers = []
    for b, fallback in enumerate((lo, 0.5 * (lo + hi), hi)):
        sel = y_train[idx == b]
        centers.append(float(np.median(sel)) if len(sel) else float(fallback))
    levels = np.array(
        [0.0, float((y_train < lo).mean()), float((y_train < hi).mean()), 1.0]
    )
    return QuantileBinner(edges=edges, centers=np.array(centers), quantile_levels=levels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--config", default="pilot_50k")
    ap.add_argument("--properties", nargs="*", default=list(LOCALITY_BATTERY))
    ap.add_argument("--variants", nargs="*", default=["wide", "focused", "wide_focused"])
    ap.add_argument("--head-seed", type=int, default=1234)
    ap.add_argument("--out-prefix", default="c18_heads_")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    cfg = load_config(args.config)
    intervals_cfg = read_json(data_dir / "target_intervals.json")

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden = np.load(data_dir / "hidden.npy")
    split = meta["split"].to_numpy()
    quartile = meta["quartile"].to_numpy()

    t_start = time.perf_counter()
    summary: dict = {
        "dataset": args.dataset, "head_seed": args.head_seed,
        "training_data": "phase-2 base-policy prefixes only -- NOT guided prefixes, "
                         "which would be a second DAgger round (pilot_report.md 9.2.1)",
        "variants": {},
    }

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}, expected one of {list(VARIANTS)}")
        hidden_dim, readout = VARIANTS[variant]
        out_dir = OUTPUT_DIR / f"{args.out_prefix}{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        vrep: dict = {"hidden_dim": hidden_dim, "readout": readout, "properties": {}}

        for prop in args.properties:
            y = meta[prop].to_numpy().astype(np.float64)
            scored = np.isfinite(y)
            masks = {s: (split == s) & scored for s in ("train", "val", "test")}
            y_train = y[masks["train"]]
            iv = intervals_cfg[prop]
            lo, hi = float(iv["lo"]), float(iv["hi"])

            if readout == "focused":
                binner = focused_binner(y_train, lo, hi)
            elif prop in DISCRETE_PROPERTIES:
                binner = CategoricalBinner(max_value=int(cfg["binning"][f"{prop}_max"]))
            else:
                binner = QuantileBinner.fit(
                    y_train, int(cfg["binning"][f"{prop}_n_bins"]), extra_edges=(lo, hi)
                )
            y_bin = binner.transform(y)

            coverage = interval_mask_coverage(binner, lo, hi, y[masks["test"]])
            if not coverage["is_exact"]:
                raise SystemExit(f"{prop}/{variant}: interval mask is not exact: {coverage}")

            torch.manual_seed(args.head_seed)
            head = MLPHead(
                in_dim=hidden.shape[1], hidden_dim=hidden_dim, n_bins=binner.n_bins,
                dropout=float(cfg["head"]["dropout"]),
            )
            t0 = time.perf_counter()
            tr = train_head(
                head, hidden[masks["train"]], y_bin[masks["train"]],
                hidden[masks["val"]], y_bin[masks["val"]],
                {**cfg["head"], "seed": args.head_seed},
            )
            train_seconds = time.perf_counter() - t0
            head.eval()

            probs_test = head.predict_proba(hidden[masks["test"]])
            yt, ybt = y[masks["test"]], y_bin[masks["test"]]
            q = interval_probability(probs_test, binner, lo, hi)
            hit = (yt >= lo) & (yt < hi)

            ckpt = {
                "state_dict": head.state_dict(), "in_dim": int(hidden.shape[1]),
                "hidden_dim": hidden_dim, "n_bins": binner.n_bins,
                "dropout": float(cfg["head"]["dropout"]), "binner": binner.to_dict(),
                "property": prop, "input": "frozen_state", "head_seed": int(args.head_seed),
                "c18_variant": variant,
            }
            torch.save(ckpt, out_dir / f"head_{prop}_frozen_state.pt")

            ev = M.evaluate(probs_test, yt, ybt, binner, {"target": (lo, hi)})
            byq = {
                str(qz): {
                    "auroc": M.auroc(q[quartile[masks["test"]] == qz],
                                     hit[quartile[masks["test"]] == qz]),
                    "n": int((quartile[masks["test"]] == qz).sum()),
                }
                for qz in (1, 2, 3, 4)
            }
            vrep["properties"][prop] = {
                "n_bins": int(binner.n_bins),
                "n_parameters": int(sum(p.numel() for p in head.parameters())),
                "interval_mask_coverage": coverage,
                "best_epoch": tr.best_epoch, "epochs_run": len(tr.history),
                "train_seconds": train_seconds,
                "test": {
                    "nll": ev["nll"],
                    "expected_value_mae": ev["expected_value_mae"],
                    "target_auroc": ev["intervals"]["target"]["auroc"],
                    "target_brier": ev["intervals"]["target"]["brier"],
                    "target_ece": M.expected_calibration_error(q, hit),
                    "mean_predicted": float(q.mean()),
                    "observed": float(hit.mean()),
                },
                "target_auroc_by_quartile": byq,
                "note_nll_not_comparable_across_readouts": (
                    "NLL and E[y] MAE are computed over a different partition for the "
                    "focused readout (3 bins) than for the full one, so they are NOT "
                    "comparable across variants -- pilot_report.md 11.6 finding 4. "
                    "target_auroc / target_brier / target_ece are on the same event "
                    "and are the comparable columns."
                ),
            }
            print(f"{variant:14s} {prop:16s} auroc={ev['intervals']['target']['auroc']:.4f} "
                  f"ece={vrep['properties'][prop]['test']['target_ece']:.4f} "
                  f"pred={q.mean():.4f} obs={hit.mean():.4f} ({train_seconds:.0f}s)")

        write_json(out_dir / "head_variant_metrics.json",
                   {"variant": variant, **vrep, "dataset": args.dataset})
        write_json(out_dir / "config_used.json", cfg)
        write_run_context(out_dir)
        summary["variants"][variant] = vrep

    summary["wall_seconds_total"] = time.perf_counter() - t_start
    out = OUTPUT_DIR / "c18_head_variants"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "head_variants_summary.json", summary)
    write_run_context(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
