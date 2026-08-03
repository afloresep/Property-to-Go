"""The two figures `reports/PAPER_WORKSHOP_DRAFT.md` specifies, and nothing else.

Reads only committed summary artefacts -- C27's and C33's equal-information sweeps
(generators 1 and 2), C26's and C31's oracle-selected frontiers, and C28's and C31's
k-sweep cells.  **Generates nothing**: no molecule is sampled, no head is loaded, no
`outputs/c*_summary/` file is written.

Figure 1 (`fig1_oracle_gap_vs_n.png`) is the draft's §3.2 exhibit: the gap between
oracle-selected and equal-information best-of-N against N, generator 1 solid and
generator 2 dashed, one line pair per property.  The visual claim is that the pairs lie on
top of each other while the deployed-arm budget markers do not -- which is exactly why the
share of `pilot_report.md` §25.3 failed to replicate while the curve of §25.2 did.

Figure 2 (`fig2_frontiers.png`) is the draft's §5 exhibit: the two best-of-N frontiers in
processed tokens per molecule with the guided k-sweep cells priced against them, one panel
per generator.

    .venv/bin/python scripts/28_paper_figures.py

Every number drawn is asserted against its artefact by `tests/test_paper_figures.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from property_to_go.config import OUTPUT_DIR, read_json, write_run_context  # noqa: E402

ANCHORS = ["aromatic_rings", "hbd_count", "qed"]
LABELS = {"aromatic_rings": "aromatic rings", "hbd_count": "HBD count", "qed": "QED"}

#: Matplotlib's default cycle, held fixed per property so the two figures agree.
COLOURS = {"aromatic_rings": "C0", "hbd_count": "C1", "qed": "C2"}


# --------------------------------------------------------------------------- inputs


def gap_curves() -> dict[str, dict[str, dict[int, float]]]:
    """gap(N) = oracle_selected(N) - head_selected(N), per generator, per anchor.

    Generator 1 is recomputed from C27's two curves rather than read from
    `E2_price_of_ground_truth.gap_per_n`, so that both generators go through the
    identical subtraction; the two agree, and `tests/test_paper_figures.py` checks it.
    """
    c27 = read_json(OUTPUT_DIR / "c27_summary" / "c27_metrics.json")
    c33 = read_json(OUTPUT_DIR / "c33_summary" / "c33_metrics.json")
    out: dict[str, dict[str, dict[int, float]]] = {"g1": {}, "g2": {}}
    for prop in ANCHORS:
        for tag, metrics in (("g1", c27), ("g2", c33)):
            curves = metrics["properties"][prop]["curves"]
            grid = metrics["properties"][prop]["grid"]
            out[tag][prop] = {
                n: curves["oracle_selected"][str(n)]["hit_rate_mean"]
                - curves["head_selected"][str(n)]["hit_rate_mean"]
                for n in grid
            }
    return out


def deployed_budgets_in_n() -> dict[str, dict[str, float]]:
    """Where each generator's deployed arm sits on the N axis, in units of N.

    The guided arms are priced in processed tokens per molecule; N = 1 costs one
    molecule's worth of tokens, so budget / tokens(N=1) is that budget expressed as an
    equivalent N.  This is the quantity §25.4 says the two generators never matched.
    """
    c27 = read_json(OUTPUT_DIR / "c27_summary" / "c27_metrics.json")
    c33 = read_json(OUTPUT_DIR / "c33_summary" / "c33_metrics.json")
    deployed_g1 = c27["decision_rules"]["E4_deployed_lambda1_arm_vs_head_selected_curve"]
    deployed_g2 = c33["headline"]["per_property"]
    out: dict[str, dict[str, float]] = {"g1": {}, "g2": {}}
    for prop in ANCHORS:
        t1_g1 = c27["properties"][prop]["tokens_per_molecule_actual"][0]
        t1_g2 = c33["properties"][prop]["tokens_per_molecule_actual"][0]
        out["g1"][prop] = deployed_g1[prop]["tokens_per_molecule_actual"] / t1_g1
        out["g2"][prop] = deployed_g2[prop]["tokens_per_molecule_actual"] / t1_g2
    return out


def frontier_and_cells(tag: str) -> dict:
    """The oracle-selected frontier and the guided k-sweep cells for one generator."""
    if tag == "g1":
        best = read_json(OUTPUT_DIR / "c27_summary" / "c27_metrics.json")
        sweep = read_json(OUTPUT_DIR / "c28_summary" / "c28_metrics.json")
        strands = {
            name: s for name, s in sweep["strands"].items() if isinstance(s, dict)
            and "k_grid" in s and s.get("property") in ANCHORS
        }
        cells = [
            {
                "property": s["property"],
                "tokens": tok,
                "hit_rate": hit,
                "layer": s["layer"],
                "lam": s["lam"],
            }
            for s in strands.values()
            for tok, hit in zip(s["tokens_per_molecule_actual"], s["hit_rate_mean"])
        ]
    else:
        best = read_json(OUTPUT_DIR / "c33_summary" / "c33_metrics.json")
        sweep = read_json(OUTPUT_DIR / "c31_summary" / "c31_metrics.json")
        cells = [
            {
                "property": c["property"],
                "tokens": c["tokens_per_molecule_actual"],
                "hit_rate": c["hit_rate_mean"],
                "layer": c["probe_point"],
                "lam": c["lam"],
            }
            for c in sweep["cells"].values()
        ]
    frontiers = {}
    for prop in ANCHORS:
        p = best["properties"][prop]
        grid = p["grid"]
        toks = p["tokens_per_molecule_actual"]
        frontiers[prop] = {
            "tokens": toks,
            "oracle": [p["curves"]["oracle_selected"][str(n)]["hit_rate_mean"] for n in grid],
            "head": [p["curves"]["head_selected"][str(n)]["hit_rate_mean"] for n in grid],
        }
    return {"frontiers": frontiers, "cells": cells}


# --------------------------------------------------------------------------- figures


def fig1(out: Path) -> Path:
    """The draft's §3.2 exhibit: oracle-minus-equal-information gap against N."""
    gaps = gap_curves()
    marks = deployed_budgets_in_n()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for prop in ANCHORS:
        c = COLOURS[prop]
        g1, g2 = gaps["g1"][prop], gaps["g2"][prop]
        ax.plot(list(g1), list(g1.values()), c=c, ls="-", marker="o", ms=3.5, lw=1.6,
                label=f"{LABELS[prop]} — gen 1")
        ax.plot(list(g2), list(g2.values()), c=c, ls="--", marker="s", ms=3.5, lw=1.6,
                label=f"{LABELS[prop]} — gen 2")
    # The deployed guided arms of the two generators sit at different budgets, and that --
    # not the model -- is what breaks the single-number share (pilot_report.md §25.4).
    for tag, note in (("g2", "gen-2 deployed\n~131 tok/mol"),
                      ("g1", "gen-1 deployed\n367–419 tok/mol")):
        lo = min(marks[tag].values())
        hi = max(marks[tag].values())
        ax.axvspan(lo - 0.12, hi + 0.12, color="grey", alpha=0.16, lw=0)
        ax.text(hi + 0.35, 0.012, note, fontsize=7, ha="left", va="bottom",
                color="0.25", linespacing=1.25)
    ax.set_xlabel("N (best-of-N candidate count, matched across generators)")
    ax.set_ylabel("oracle-selected − equal-information hit rate")
    ax.set_title(
        "The oracle's value grows with the baseline's budget, on both generators\n"
        "solid = GP-MoLFormer (46.8M, linear attn); dashed = gpt2_zinc_87m (87.3M, softmax)",
        fontsize=9.5,
    )
    ax.set_xlim(0.5, 33.5)
    ax.set_ylim(-0.02, 0.78)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.95)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig2(out: Path) -> Path:
    """The draft's §5 exhibit: both frontiers, guided cells priced against them."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    titles = {
        "g1": "generator 1 — GP-MoLFormer-Uniq (46.8M, linear attention)",
        "g2": "generator 2 — gpt2_zinc_87m (87.3M, full softmax attention)",
    }
    for ax, tag in zip(axes, ("g1", "g2")):
        data = frontier_and_cells(tag)
        for prop in ANCHORS:
            f = data["frontiers"][prop]
            c = COLOURS[prop]
            ax.plot(f["tokens"], f["oracle"], c=c, ls="-", lw=1.6, marker="o", ms=3,
                    label=f"{LABELS[prop]} — oracle-selected")
            ax.plot(f["tokens"], f["head"], c=c, ls=":", lw=1.4, marker="v", ms=3,
                    label=f"{LABELS[prop]} — equal-information")
        for cell in data["cells"]:
            ax.scatter(cell["tokens"], cell["hit_rate"], c=COLOURS[cell["property"]],
                       marker="x", s=26, lw=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("processed generator tokens per molecule")
        ax.set_title(titles[tag], fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("hit rate (target interval)")
    axes[0].scatter([], [], c="grey", marker="x", s=26, lw=1.2,
                    label="guided cell (k sweep), own budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.suptitle(
        "Guidance crosses the oracle-selected frontier only where best-of-N has drawn "
        "two to four samples", fontsize=10.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "paper_figures"))
    args = ap.parse_args()
    out = Path(args.out)
    made = [
        fig1(out / "fig1_oracle_gap_vs_n.png"),
        fig2(out / "fig2_frontiers.png"),
    ]
    write_run_context(out, {"script": "28_paper_figures.py",
                            "figures": [p.name for p in made]})
    for p in made:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
