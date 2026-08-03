"""Phase 4 -- repeated-continuation evaluation.

Takes held-out prefixes balanced across generation positions, draws 32 independent
base-policy continuations from each, and uses that rollout bank as the empirical
conditional final-property distribution p(y_final | x_<=t).  One bank serves both
properties.

This is what makes the predictability curve an honest measurement: the head is
scored against the distribution the frozen generator actually produces from that
prefix, not against the single completion the prefix happened to come from.

    python scripts/04_rollout_bank.py --dataset pilot_50k --n-prefixes 800 --n-rollouts 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr  # noqa: E402

from property_to_go import generation, metrics as M  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import balanced_position_sample  # noqa: E402
from property_to_go.properties import ALL_PROPERTIES, compute_all_properties  # noqa: E402


def load_head(path: Path) -> tuple[MLPHead, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--heads", default=None)
    ap.add_argument("--n-prefixes", type=int, default=800)
    ap.add_argument("--n-rollouts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / (args.heads or f"{args.dataset}_heads")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    intervals_cfg = read_json(data_dir / "target_intervals.json")

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden = np.load(data_dir / "hidden.npy")
    features = np.load(data_dir / "features.npy")
    prefix_ids_all = read_json(data_dir / "prefix_token_ids.json")

    test = np.flatnonzero(meta["split"].to_numpy() == "test")
    rng = np.random.default_rng(args.seed)
    chosen_local = balanced_position_sample(
        meta["prefix_len"].to_numpy()[test],
        meta["quartile"].to_numpy()[test],
        args.n_prefixes,
        rng,
    )
    chosen = test[chosen_local]
    print(f"{len(chosen)} held-out prefixes, quartile counts: "
          f"{np.bincount(meta['quartile'].to_numpy()[chosen], minlength=5)[1:]}")

    gen = load_generator(model_cfg)
    prefixes = [prefix_ids_all[i] for i in chosen]

    # ---- roll out ------------------------------------------------------------
    meter = ComputeMeter().start()
    conts = generation.continue_from_prefixes(
        gen, prefixes, args.n_rollouts, policy, seed=args.seed, meter=meter,
        batch_size=int(policy["batch_size"]),
    )
    meter.stop()
    print(f"{len(conts)*args.n_rollouts} continuations in {meter.wall_seconds:.1f}s")

    # ---- empirical conditional distributions ---------------------------------
    bank: list[dict] = []
    for local_i, (idx, rows) in enumerate(zip(chosen, conts)):
        smiles = gen.decode(rows)
        props = [compute_all_properties(s) for s in smiles]
        ok = [p for p in props if p is not None]
        entry = {
            "prefix_row": int(idx),
            "quartile": int(meta["quartile"].to_numpy()[idx]),
            "prefix_len": int(meta["prefix_len"].to_numpy()[idx]),
            "relative_position": float(meta["relative_position"].to_numpy()[idx]),
            "n_rollouts": len(rows),
            "n_valid": len(ok),
            "rollout_validity": len(ok) / len(rows) if rows else 0.0,
            "rollout_lengths": [
                len(generation.sequence_content(r, gen.bos_id, gen.eos_id, gen.pad_id)) for r in rows
            ],
        }
        for prop in ALL_PROPERTIES:
            # A parseable molecule can still lack one descriptor (QED alone), so the
            # per-property lists can be shorter than `n_valid`. Length is what every
            # consumer below uses as the denominator, so this stays correct.
            entry[prop] = [float(p[prop]) for p in ok if p.get(prop) is not None]
        bank.append(entry)

    write_json(out_dir / "rollout_bank.json", bank)

    # ---- score each head against the empirical distribution -------------------
    report: dict = {
        "dataset": args.dataset,
        "n_prefixes": len(bank),
        "n_rollouts_per_prefix": args.n_rollouts,
        "seed": args.seed,
        "compute": meter.as_dict(),
        "mean_rollout_validity": float(np.mean([b["rollout_validity"] for b in bank])),
        "properties": {},
    }

    inputs = {"frozen_state": hidden, "trivial": features,
              "combined": np.concatenate([hidden, features], axis=1)}

    for prop in ALL_PROPERTIES:
        iv = intervals_cfg[prop]
        # empirical target from the rollouts themselves
        emp_mean = np.array([np.mean(b[prop]) if b[prop] else np.nan for b in bank])
        emp_rate = np.array(
            [
                float(np.mean((np.array(b[prop]) >= iv["lo"]) & (np.array(b[prop]) < iv["hi"])))
                if b[prop] else np.nan
                for b in bank
            ]
        )
        keep = ~np.isnan(emp_mean)
        q_arr = np.array([b["quartile"] for b in bank])
        pos_arr = np.array([b["relative_position"] for b in bank])

        prop_report: dict = {
            "target_interval": iv,
            "empirical_target_rate_mean": float(np.nanmean(emp_rate)),
            "heads": {},
        }

        for name, x in inputs.items():
            head_path = heads_dir / f"head_{prop}_{name}.pt"
            if not head_path.exists():
                continue
            head, ck = load_head(head_path)
            binner = binner_from_dict(ck["binner"])
            probs = head.predict_proba(x[chosen])
            pred_mean = binner.expected_value(probs)
            pred_q = interval_probability(probs, binner, iv["lo"], iv["hi"])

            def curve(mask):
                if mask.sum() < 20:
                    return None
                return {
                    "n": int(mask.sum()),
                    # error against the empirical conditional mean over 32 rollouts
                    "mae_vs_empirical_mean": float(np.mean(np.abs(pred_mean[mask] - emp_mean[mask]))),
                    "spearman_vs_empirical_mean": float(
                        spearmanr(pred_mean[mask], emp_mean[mask]).statistic
                    ),
                    "spearman_vs_empirical_rate": float(
                        spearmanr(pred_q[mask], emp_rate[mask]).statistic
                    )
                    if len(np.unique(emp_rate[mask])) > 1 else float("nan"),
                    # Brier against every individual rollout outcome
                    "brier_vs_rollouts": _brier_vs_rollouts(pred_q, bank, prop, iv, mask),
                    "mean_predicted_target_prob": float(np.mean(pred_q[mask])),
                    "mean_empirical_target_rate": float(np.nanmean(emp_rate[mask])),
                }

            per_q = {str(q): curve(keep & (q_arr == q)) for q in (1, 2, 3, 4)}
            overall = curve(keep)
            rel = M.reliability(
                np.repeat(pred_q[keep], [len(bank[i][prop]) for i in np.flatnonzero(keep)]),
                np.concatenate(
                    [
                        (np.array(bank[i][prop]) >= iv["lo"]) & (np.array(bank[i][prop]) < iv["hi"])
                        for i in np.flatnonzero(keep)
                    ]
                ),
            )
            prop_report["heads"][name] = {
                "overall": overall,
                "by_quartile": per_q,
                "reliability": rel,
            }
            print(f"{prop:15s} {name:13s} rollout-MAE={overall['mae_vs_empirical_mean']:.4f} "
                  f"spearman={overall['spearman_vs_empirical_mean']:.4f} "
                  f"brier={overall['brier_vs_rollouts']:.4f}")

        # how much of the final property is already fixed by the prefix?
        prop_report["conditional_spread"] = {
            str(q): float(
                np.nanmean([np.std(b[prop]) for b in bank if b["quartile"] == q and b[prop]])
            )
            for q in (1, 2, 3, 4)
        }
        prop_report["marginal_spread"] = float(
            np.std(np.concatenate([b[prop] for b in bank if b[prop]]))
        )
        report["properties"][prop] = prop_report

    np.savez_compressed(
        out_dir / "rollout_arrays.npz",
        prefix_rows=chosen,
        quartile=np.array([b["quartile"] for b in bank]),
        relative_position=np.array([b["relative_position"] for b in bank]),
    )
    write_json(out_dir / "rollout_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy})
    print(f"-> {out_dir}")
    return 0


def _brier_vs_rollouts(pred_q, bank, prop, iv, mask) -> float:
    num, den = 0.0, 0
    for i in np.flatnonzero(mask):
        vals = np.array(bank[i][prop])
        if not len(vals):
            continue
        hit = (vals >= iv["lo"]) & (vals < iv["hi"])
        num += float(np.sum((pred_q[i] - hit) ** 2))
        den += len(vals)
    return num / den if den else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
