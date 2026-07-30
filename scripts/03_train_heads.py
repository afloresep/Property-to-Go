"""Phase 2 -- train the frozen-state head and the trivial-prefix baseline.

Three heads share one training recipe so that only the input representation differs:

    frozen_state  h_t (768-d final-layer state)
    trivial       cheap prefix statistics (tokens.FEATURE_NAMES)
    combined      both, kept as a diagnostic on additivity

`marginal` (the training class frequencies, input-independent) is reported as the
floor.  Every head is evaluated on the same held-out rows, overall and split by
prefix-position quartile, and the kill-gate verdict is computed from a
paired bootstrap rather than from a point estimate.

    python scripts/03_train_heads.py --dataset pilot_10k
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
from property_to_go.binning import CategoricalBinner, QuantileBinner, interval_probability  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MarginalHead, MLPHead, train_head  # noqa: E402
from property_to_go.properties import ALL_PROPERTIES, PRIMARY_PROPERTIES  # noqa: E402

# Kill-gate thresholds, fixed before looking at any result.
GATE_MIN_NLL_GAIN = 0.05  # nats, frozen_state vs trivial, held-out test
GATE_MIN_AUROC_GAIN = 0.02  # target-interval discrimination


def paired_bootstrap_diff(fn, a_args, b_args, n_boot=1000, seed=0, alpha=0.05):
    """95% CI for metric(a) - metric(b) resampling the same rows for both heads."""
    rng = np.random.default_rng(seed)
    n = len(a_args[0])
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(fn(*[x[idx] for x in a_args]) - fn(*[x[idx] for x in b_args]))
        except Exception:
            continue
    if not diffs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "mean": float(np.mean(diffs)),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_10k")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    cfg = load_config(args.config or args.dataset)
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_heads")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden = np.load(data_dir / "hidden.npy")
    features = np.load(data_dir / "features.npy")
    intervals_cfg = read_json(data_dir / "target_intervals.json")
    assert len(meta) == len(hidden) == len(features)

    split = meta["split"].to_numpy()
    quartile = meta["quartile"].to_numpy()
    masks = {s: split == s for s in ("train", "val", "test")}
    print({s: int(m.sum()) for s, m in masks.items()})

    inputs = {
        "frozen_state": hidden,
        "trivial": features,
        "combined": np.concatenate([hidden, features], axis=1),
    }

    report: dict = {
        "dataset": args.dataset,
        "n_rows": int(len(meta)),
        "split_sizes": {s: int(m.sum()) for s, m in masks.items()},
        "gate_thresholds": {"nll_gain": GATE_MIN_NLL_GAIN, "auroc_gain": GATE_MIN_AUROC_GAIN},
        "properties": {},
    }
    t_start = time.perf_counter()

    for prop in ALL_PROPERTIES:
        y = meta[prop].to_numpy().astype(np.float64)
        y_train = y[masks["train"]]

        if prop == "aromatic_rings":
            binner = CategoricalBinner(max_value=int(cfg["binning"]["aromatic_rings_max"]))
        else:
            n_bins = int(cfg["binning"]["clogp_n_bins" if prop == "clogp" else "mol_weight_n_bins"])
            binner = QuantileBinner.fit(y_train, n_bins)
        y_bin = binner.transform(y)

        iv = intervals_cfg[prop]
        intervals = {"target": (iv["lo"], iv["hi"])}

        prop_report: dict = {
            "binner": binner.to_dict(),
            "n_bins": binner.n_bins,
            "target_interval": iv,
            "heads": {},
        }
        preds_store: dict[str, np.ndarray] = {}

        for name, x in inputs.items():
            head = MLPHead(
                in_dim=x.shape[1],
                hidden_dim=int(cfg["head"]["hidden_dim"]),
                n_bins=binner.n_bins,
                dropout=float(cfg["head"]["dropout"]),
            )
            t0 = time.perf_counter()
            tr = train_head(
                head,
                x[masks["train"]],
                y_bin[masks["train"]],
                x[masks["val"]],
                y_bin[masks["val"]],
                cfg["head"],
            )
            train_seconds = time.perf_counter() - t0

            probs_test = head.predict_proba(x[masks["test"]])
            probs_val = head.predict_proba(x[masks["val"]])
            preds_store[name] = probs_test

            prop_report["heads"][name] = {
                "input_dim": int(x.shape[1]),
                "n_parameters": int(sum(p.numel() for p in head.parameters())),
                "best_epoch": tr.best_epoch,
                "epochs_run": len(tr.history),
                "train_seconds": train_seconds,
                "history": tr.history,
                "val": M.evaluate(probs_val, y[masks["val"]], y_bin[masks["val"]], binner, intervals),
                "test": M.evaluate(probs_test, y[masks["test"]], y_bin[masks["test"]], binner, intervals),
                "test_by_quartile": M.evaluate_by_group(
                    probs_test,
                    y[masks["test"]],
                    y_bin[masks["test"]],
                    quartile[masks["test"]],
                    binner,
                    intervals,
                ),
            }
            torch.save(
                {"state_dict": head.state_dict(), "in_dim": int(x.shape[1]),
                 "hidden_dim": int(cfg["head"]["hidden_dim"]), "n_bins": binner.n_bins,
                 "dropout": float(cfg["head"]["dropout"]), "binner": binner.to_dict(),
                 "property": prop, "input": name},
                out_dir / f"head_{prop}_{name}.pt",
            )
            print(
                f"{prop:15s} {name:13s} test nll={prop_report['heads'][name]['test']['nll']:.4f} "
                f"mae={prop_report['heads'][name]['test']['expected_value_mae']:.4f} "
                f"auroc={prop_report['heads'][name]['test']['intervals']['target']['auroc']:.4f}"
            )

        # marginal floor
        marg = MarginalHead(y_bin[masks["train"]], binner.n_bins)
        probs_marg = marg.predict_proba(hidden[masks["test"]])
        preds_store["marginal"] = probs_marg
        prop_report["heads"]["marginal"] = {
            "input_dim": 0,
            "test": M.evaluate(probs_marg, y[masks["test"]], y_bin[masks["test"]], binner, intervals),
        }

        # ---- paired comparison: frozen_state vs trivial on the test split -------
        yt, ybt = y[masks["test"]], y_bin[masks["test"]]
        pf, pt = preds_store["frozen_state"], preds_store["trivial"]
        qf = interval_probability(pf, binner, iv["lo"], iv["hi"])
        qt = interval_probability(pt, binner, iv["lo"], iv["hi"])
        hit = (yt >= iv["lo"]) & (yt < iv["hi"])

        nll_diff = paired_bootstrap_diff(
            lambda p, b: M.categorical_nll(p, b), (pt, ybt), (pf, ybt)
        )  # trivial minus frozen: positive means the frozen state is better
        auroc_diff = paired_bootstrap_diff(
            lambda s, h: M.auroc(s, h), (qf, hit), (qt, hit)
        )
        mae_diff = paired_bootstrap_diff(
            lambda p, v: float(np.mean(np.abs(binner.expected_value(p) - v))), (pt, yt), (pf, yt)
        )
        brier_diff = paired_bootstrap_diff(
            lambda s, h: M.brier(s, h), (qt, hit), (qf, hit)
        )

        passes = (
            nll_diff["lo"] > 0
            and nll_diff["mean"] >= GATE_MIN_NLL_GAIN
            and auroc_diff["lo"] > 0
            and auroc_diff["mean"] >= GATE_MIN_AUROC_GAIN
        )
        prop_report["frozen_vs_trivial"] = {
            "nll_gain_nats": nll_diff,
            "auroc_gain": auroc_diff,
            "expected_value_mae_gain": mae_diff,
            "brier_gain": brier_diff,
            "passes_gate": bool(passes),
            "is_primary": prop in PRIMARY_PROPERTIES,
        }
        print(
            f"{prop:15s} frozen-vs-trivial  nll_gain={nll_diff['mean']:+.4f} "
            f"[{nll_diff['lo']:+.4f},{nll_diff['hi']:+.4f}]  "
            f"auroc_gain={auroc_diff['mean']:+.4f} [{auroc_diff['lo']:+.4f},{auroc_diff['hi']:+.4f}]  "
            f"gate={'PASS' if passes else 'fail'}"
        )

        np.savez_compressed(
            out_dir / f"predictions_{prop}.npz",
            y_true=yt,
            y_bin=ybt,
            quartile=quartile[masks["test"]],
            prefix_len=meta["prefix_len"].to_numpy()[masks["test"]],
            relative_position=meta["relative_position"].to_numpy()[masks["test"]],
            **{f"probs_{k}": v for k, v in preds_store.items()},
        )
        report["properties"][prop] = prop_report

    primary_pass = [
        p for p in PRIMARY_PROPERTIES if report["properties"][p]["frozen_vs_trivial"]["passes_gate"]
    ]
    report["scaling_gate"] = {
        "rule": "frozen_state beats trivial on >=1 primary property, "
                f"nll gain >= {GATE_MIN_NLL_GAIN} nats and target AUROC gain >= {GATE_MIN_AUROC_GAIN}, "
                "with bootstrap 95% CI excluding zero",
        "primary_properties_passing": primary_pass,
        "passes": bool(primary_pass),
    }
    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "head_metrics.json", report)
    write_json(out_dir / "config_used.json", cfg)
    write_run_context(out_dir)

    print(f"\nSCALING GATE: {'PASS' if report['scaling_gate']['passes'] else 'FAIL'} "
          f"(primary properties passing: {primary_pass})")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
