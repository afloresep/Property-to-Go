"""Produce the pilot figures from whatever artefacts exist.

Missing inputs are skipped with a printed note rather than faked, so the figure set
always reflects what was actually run.

    python scripts/07_figures.py --dataset pilot_50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import plotting as P  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_run_context  # noqa: E402
from property_to_go.properties import ALL_PROPERTIES  # noqa: E402

HEAD_LABELS = {"frozen_state": "frozen LM state", "trivial": "prefix statistics",
               "combined": "both", "marginal": "marginal (no input)"}
CONDITION_ORDER = ["unguided", "truncation_control", "early", "middle", "late", "throughout"]


def fig_dir(dataset: str) -> Path:
    d = OUTPUT_DIR / f"{dataset}_figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def predictability_from_heads(dataset: str, out: Path) -> list[str]:
    path = OUTPUT_DIR / f"{dataset}_heads" / "head_metrics.json"
    if not path.exists():
        return [f"skip: {path} missing"]
    r = read_json(path)
    made = []
    for prop, pr in r["properties"].items():
        for metric, label, key in [
            ("auroc", "target-interval AUROC", lambda d: d["intervals"]["target"]["auroc"]),
            ("nll", "categorical NLL (nats)", lambda d: d["nll"]),
            ("mae", "expected-value MAE", lambda d: d["expected_value_mae"]),
        ]:
            series = {}
            for head, hd in pr["heads"].items():
                if "test_by_quartile" not in hd:
                    continue
                qs = hd["test_by_quartile"]
                series[HEAD_LABELS.get(head, head)] = [key(qs[str(q)]) for q in (1, 2, 3, 4)]
            if not series:
                continue
            P.predictability_curve(
                [1, 2, 3, 4], series, label,
                f"{prop}: {label} vs generation position\n({dataset}, held-out test)",
                out / f"predictability_{prop}_{metric}.png",
                baseline=0.5 if metric == "auroc" else None, baseline_label="chance",
            )
            made.append(f"predictability_{prop}_{metric}.png")
    return made


def predictability_from_rollouts(dataset: str, out: Path) -> list[str]:
    path = OUTPUT_DIR / f"{dataset}_rollouts" / "rollout_metrics.json"
    if not path.exists():
        return [f"skip: {path} missing"]
    r = read_json(path)
    made = []
    for prop, pr in r["properties"].items():
        for metric, label in [
            ("spearman_vs_empirical_mean", "Spearman rho vs empirical conditional mean"),
            ("brier_vs_rollouts", "Brier score vs rollout outcomes"),
            ("mae_vs_empirical_mean", "MAE vs empirical conditional mean"),
        ]:
            series = {}
            for head, hd in pr["heads"].items():
                vals = [
                    hd["by_quartile"][str(q)][metric] if hd["by_quartile"].get(str(q)) else np.nan
                    for q in (1, 2, 3, 4)
                ]
                series[HEAD_LABELS.get(head, head)] = vals
            if not series:
                continue
            P.predictability_curve(
                [1, 2, 3, 4], series, label,
                f"{prop}: predictability curve ({r['n_rollouts_per_prefix']} rollouts/prefix)",
                out / f"rollout_{prop}_{metric}.png",
            )
            made.append(f"rollout_{prop}_{metric}.png")

        curves = {HEAD_LABELS.get(h, h): hd["reliability"] for h, hd in pr["heads"].items()}
        P.reliability_plot(curves, f"{prop}: reliability vs rollout outcomes",
                           out / f"reliability_{prop}.png")
        made.append(f"reliability_{prop}.png")

        spread = pr["conditional_spread"]
        P.predictability_curve(
            [1, 2, 3, 4],
            {"conditional sd over 32 rollouts": [spread[str(q)] for q in (1, 2, 3, 4)]},
            "sd of final property",
            f"{prop}: how much is still undetermined at each position",
            out / f"conditional_spread_{prop}.png",
            baseline=pr["marginal_spread"], baseline_label="marginal sd",
        )
        made.append(f"conditional_spread_{prop}.png")
    return made


def intervention(dataset: str, out: Path) -> list[str]:
    made = []
    for prop in ALL_PROPERTIES:
        path = OUTPUT_DIR / f"{dataset}_guided_{prop}" / "guidance_metrics.json"
        if not path.exists():
            made.append(f"skip: {path.name} for {prop} missing")
            continue
        r = read_json(path)
        conds = [c for c in CONDITION_ORDER if c in r["conditions"]]
        ref = r["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"]

        for key, label in [
            ("hit_rate", "target hit rate"),
            ("abs_target_error_mean", "mean absolute target error"),
            ("validity", "RDKit validity"),
            ("content_length_mean", "mean content-token length"),
            ("n_heavy_atoms_mean", "mean heavy-atom count"),
        ]:
            means = [r["conditions"][c]["aggregate"][key]["mean"] for c in conds]
            errs = [r["conditions"][c]["aggregate"][key]["std"] for c in conds]
            reference = r["conditions"]["unguided"]["aggregate"][key]["mean"]
            P.intervention_curve(
                conds, means, errs, label,
                f"{prop}: intervention-response ({label})\n"
                f"target [{r['target_interval']['lo']:.2f}, {r['target_interval']['hi']:.2f}), "
                f"base rate {r['target_interval']['base_rate']:.3f}, "
                f"{len(r['seeds'])} seeds",
                out / f"intervention_{prop}_{key}.png",
                reference=reference,
            )
            made.append(f"intervention_{prop}_{key}.png")

        lc = r["length_confound"]
        P.intervention_curve(
            conds,
            [lc[c]["length_matched_hit_rate"] for c in conds],
            [0.0] * len(conds),
            "length-matched hit rate",
            f"{prop}: hit rate after matching the unguided length distribution",
            out / f"length_matched_{prop}.png",
            reference=lc["unguided"]["length_matched_hit_rate"],
        )
        made.append(f"length_matched_{prop}.png")

        mol = read_json(OUTPUT_DIR / f"{dataset}_guided_{prop}" / "molecules.json")
        series = {}
        for c in ("unguided", "throughout"):
            if c in mol:
                series[c] = [
                    rec[prop] for s in mol[c] for rec in mol[c][s] if rec["valid"]
                ]
        if series:
            P.distribution_plot(
                series, (r["target_interval"]["lo"], r["target_interval"]["hi"]),
                prop, f"{prop}: terminal distribution, unguided vs guided throughout",
                out / f"distribution_{prop}.png",
            )
            made.append(f"distribution_{prop}.png")
            lens = {c: [rec["n_content_tokens"] for s in mol[c] for rec in mol[c][s] if rec["valid"]]
                    for c in series}
            P.distribution_plot(
                lens, (0, 0), "content tokens",
                f"{prop}: sequence length, unguided vs guided throughout",
                out / f"lengths_{prop}.png",
            )
            made.append(f"lengths_{prop}.png")
        _ = ref
    return made


def bestofn_fig(dataset: str, out: Path) -> list[str]:
    made = []
    for prop in ALL_PROPERTIES:
        path = OUTPUT_DIR / f"{dataset}_bestofn_{prop}" / "bestofn_metrics.json"
        gpath = OUTPUT_DIR / f"{dataset}_guided_{prop}" / "guidance_metrics.json"
        if not (path.exists() and gpath.exists()):
            continue
        r, g = read_json(path), read_json(gpath)
        labels, means, errs = ["unguided"], [
            g["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"]
        ], [g["conditions"]["unguided"]["aggregate"]["hit_rate"]["std"]]
        for acc, m in r["matches"].items():
            labels.append(f"best-of-{m['n_candidates']}\n({acc})")
            means.append(m["aggregate"]["hit_rate"]["mean"])
            errs.append(m["aggregate"]["hit_rate"]["std"])
        labels.append("guided throughout")
        means.append(g["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"])
        errs.append(g["conditions"]["throughout"]["aggregate"]["hit_rate"]["std"])
        P.intervention_curve(
            labels, means, errs, "target hit rate",
            f"{prop}: compute-matched comparison",
            out / f"compute_matched_{prop}.png",
            reference=means[0],
        )
        made.append(f"compute_matched_{prop}.png")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    args = ap.parse_args()
    out = fig_dir(args.dataset)
    made: list[str] = []
    made += predictability_from_heads(args.dataset, out)
    made += predictability_from_rollouts(args.dataset, out)
    made += intervention(args.dataset, out)
    made += bestofn_fig(args.dataset, out)
    for m in made:
        print(("  " if m.startswith("skip") else "  wrote ") + m)
    write_run_context(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
