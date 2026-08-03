"""C17 step 4 -- the depth curves.

One panel per property: mean held-out target-interval AUROC against probe point, with
the seed spread as a band, the layer-independent `trivial` head as a horizontal line, and
the final layer (the one every number in `reports/pilot_report.md` uses) marked.

Reads artefacts only; generates nothing and computes no new number, so it cannot
influence a result. Everything it draws is already in `probe_layer_metrics.json`.

    .venv/bin/python scripts/16_layer_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from property_to_go.config import OUTPUT_DIR, read_json, write_run_context  # noqa: E402
from property_to_go.plotting import save  # noqa: E402
from property_to_go.properties import PREDICTED_LOCALITY_ORDER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="c17_probe_layers")
    ap.add_argument("--steering", default="c17_layer_steering")
    ap.add_argument("--out", default="c17_figures")
    args = ap.parse_args()

    sweep = read_json(OUTPUT_DIR / args.sweep / "probe_layer_metrics.json")
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(L) for L in sweep["probe_points"]]
    ref = int(sweep["reference_probe_point"])

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
    for ax, prop in zip(axes.ravel(), PREDICTED_LOCALITY_ORDER):
        rows = sweep["properties"][prop]["layers"]
        mean = np.array([rows[str(L)]["across_seeds"]["auroc"]["mean"] for L in layers])
        lo = np.array([rows[str(L)]["across_seeds"]["auroc"]["min"] for L in layers])
        hi = np.array([rows[str(L)]["across_seeds"]["auroc"]["max"] for L in layers])
        triv = sweep["properties"][prop]["trivial"]["across_seeds"]["auroc"]
        ax.plot(layers, mean, marker="o", ms=3.5, label="frozen state")
        ax.fill_between(layers, lo, hi, alpha=0.25, lw=0)
        ax.axhline(triv["mean"], ls="--", c="grey", lw=1, label="trivial prefix stats")
        ax.axvline(ref, ls=":", c="k", lw=1, label="probed layer (all prior results)")
        ax.set_title(prop, fontsize=10)
        ax.set_ylabel("target AUROC")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("probe point (0 = embedding, 1-12 = layers)")
    axes[0, 0].legend(fontsize=7)
    save(fig, out_dir / "auroc_by_probe_point.png")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
    for ax, prop in zip(axes.ravel(), PREDICTED_LOCALITY_ORDER):
        rows = sweep["properties"][prop]["layers"]
        mean = np.array([rows[str(L)]["across_seeds"]["nll"]["mean"] for L in layers])
        triv = sweep["properties"][prop]["trivial"]["across_seeds"]["nll"]["mean"]
        ax.plot(layers, mean, marker="o", ms=3.5, label="frozen state")
        ax.axhline(triv, ls="--", c="grey", lw=1, label="trivial prefix stats")
        ax.axvline(ref, ls=":", c="k", lw=1)
        ax.set_title(prop, fontsize=10)
        ax.set_ylabel("test NLL (nats)")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("probe point (0 = embedding, 1-12 = layers)")
    axes[0, 0].legend(fontsize=7)
    save(fig, out_dir / "nll_by_probe_point.png")

    steer_path = OUTPUT_DIR / args.steering / "layer_steering_metrics.json"
    if steer_path.exists():
        steer = read_json(steer_path)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for prop in PREDICTED_LOCALITY_ORDER:
            rows = steer["properties"][prop]["layers"]
            ax.plot(layers, [rows[str(L)]["our_head_gain"] for L in layers],
                    marker="o", ms=3, label=prop)
        ax.axvline(ref, ls=":", c="k", lw=1)
        ax.set_xlabel("probe point")
        ax.set_ylabel("per-position hit-rate gain at λ=1")
        ax.set_title("What the probe layer is worth for steering", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        save(fig, out_dir / "steering_gain_by_probe_point.png")

    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
