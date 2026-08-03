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
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, interval_mask_coverage, interval_probability,
)
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MarginalHead, MLPHead, train_head  # noqa: E402
from property_to_go.properties import (  # noqa: E402
    ALL_PROPERTIES, DISCRETE_PROPERTIES, PRIMARY_PROPERTIES,
)

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


def _across_seeds(per_seed: list[dict]) -> dict:
    """Spread of the headline metrics over head-training seeds.

    This is the quantity the pilot could not report (`docs/HANDOFF.md` §8.6: the
    paired bootstrap captures test-set sampling variance, not initialisation
    variance), and it is what decides whether a margin under ~0.03 is real.
    """
    def col(fn):
        v = np.array([fn(e) for e in per_seed], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}

    return {
        "n_seeds": len(per_seed),
        "nll": col(lambda e: e["test"]["nll"]),
        "auroc": col(lambda e: e["test"]["intervals"]["target"]["auroc"]),
        "expected_value_mae": col(lambda e: e["test"]["expected_value_mae"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_10k")
    ap.add_argument("--config", default=None)
    ap.add_argument("--head-seeds", type=int, nargs="*", default=None,
                    help="train one head per seed (phase 2). Omit for the pilot's "
                         "single-seed behaviour, which stays bit-identical.")
    ap.add_argument("--legacy-interval-mask", action="store_true",
                    help="do NOT force the target interval's edges to be bin "
                         "boundaries, reproducing the pilot's behaviour. Only useful "
                         "for quantifying what that cost -- see pilot_report.md §11.5.")
    # C17. Layer selection, additive and defaulting to the existing behaviour: with the
    # flag omitted this reads `<dataset>/hidden.npy`, i.e. the states script 02 extracted
    # at `hidden_layer` (-1, the final layer), and nothing about the run changes. Given a
    # path it reads that array instead, which is what makes a probe-layer sweep possible
    # without a second copy of the training recipe. The value used is recorded in
    # head_metrics.json so no checkpoint can be read without knowing which probe point
    # produced it. See scripts/16_probe_layer_sweep.py and
    # reports/section_c17_probe_layers.md.
    ap.add_argument("--hidden-file", default=None,
                    help="alternative frozen-state array (absolute, or relative to the "
                         "dataset directory). Default: <dataset>/hidden.npy.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    cfg = load_config(args.config or args.dataset)
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_heads")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden_path = data_dir / "hidden.npy"
    if args.hidden_file:
        p = Path(args.hidden_file)
        hidden_path = p if p.is_absolute() else data_dir / p
    hidden = np.load(hidden_path)
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

    replicated = args.head_seeds is not None
    head_seeds = list(args.head_seeds) if replicated else [int(cfg["head"]["seed"])]

    report: dict = {
        "dataset": args.dataset,
        "n_rows": int(len(meta)),
        "split_sizes": {s: int(m.sum()) for s, m in masks.items()},
        "gate_thresholds": {"nll_gain": GATE_MIN_NLL_GAIN, "auroc_gain": GATE_MIN_AUROC_GAIN},
        "head_seeds": head_seeds,
        # Which frozen-state array these heads were trained on. `hidden.npy` is the
        # default and is the final layer; anything else is a C17 probe point.
        "hidden_file": str(hidden_path),
        "hidden_file_is_dataset_default": bool(hidden_path == data_dir / "hidden.npy"),
        "seeds_initialisation": replicated,
        "legacy_interval_mask": bool(args.legacy_interval_mask),
        "properties": {},
    }
    t_start = time.perf_counter()

    for prop in ALL_PROPERTIES:
        y = meta[prop].to_numpy().astype(np.float64)
        # A prefix row whose terminal molecule has no value for this property (QED
        # alone can fail on a molecule RDKit parsed) is dropped for this property
        # only, rather than being binned from a NaN.  `n_dropped` is reported so a
        # silently shrinking training set is visible.
        scored = np.isfinite(y)
        masks = {s: (split == s) & scored for s in ("train", "val", "test")}
        y_train = y[masks["train"]]

        iv = intervals_cfg[prop]
        intervals = {"target": (iv["lo"], iv["hi"])}

        # Count properties get one category per observed value; continuous ones get
        # quantile bins fitted on the TRAIN terminal distribution.  Keyed off
        # `DISCRETE_PROPERTIES` and `<prop>_max` / `<prop>_n_bins` rather than a
        # per-property if-chain, so the phase-2 battery uses the identical recipe.
        # The keys resolve to exactly the pilot's for clogp / aromatic_rings /
        # mol_weight.
        #
        # `extra_edges` forces the target interval's edges to be bin boundaries. Without
        # it the head silently learns to predict a *subset* of the target, because
        # `interval_mask` keeps only bins wholly inside [lo, hi) -- which is exactly what
        # happened to the pilot's cLogP head (pilot_report.md §11.5). The interval comes
        # from the full base sample and the binner from the train split, so they never
        # quite agree, and nothing in the pipeline noticed.
        if prop in DISCRETE_PROPERTIES:
            binner = CategoricalBinner(max_value=int(cfg["binning"][f"{prop}_max"]))
        else:
            binner = QuantileBinner.fit(
                y_train,
                int(cfg["binning"][f"{prop}_n_bins"]),
                extra_edges=() if args.legacy_interval_mask else (iv["lo"], iv["hi"]),
            )
        y_bin = binner.transform(y)

        # Verified numerically, not reasoned about: the masked bin sum must equal the
        # empirical rate of the target event. A mismatch means the head is being
        # trained to predict something other than the target.
        coverage = interval_mask_coverage(binner, iv["lo"], iv["hi"], y[masks["test"]])
        if not coverage["is_exact"] and not args.legacy_interval_mask:
            raise SystemExit(
                f"{prop}: target interval [{iv['lo']}, {iv['hi']}) is not a union of "
                f"bins -- the head would predict a {coverage['masked_rate']:.4f}-mass "
                f"event for a {coverage['true_rate']:.4f} target. {coverage}"
            )
        if not coverage["is_exact"]:
            print(f"  !! {prop}: interval mask covers {coverage['masked_rate']:.4f} of a "
                  f"{coverage['true_rate']:.4f} target ({coverage['n_bins_selected']} of "
                  f"{coverage['n_bins']} bins) -- legacy behaviour, reported not fixed")

        prop_report: dict = {
            "binner": binner.to_dict(),
            "n_bins": binner.n_bins,
            "target_interval": iv,
            "interval_mask_coverage": coverage,
            "split_sizes": {s: int(m.sum()) for s, m in masks.items()},
            "n_dropped_unscored": int((~scored).sum()),
            "heads": {},
        }
        preds_store: dict[str, np.ndarray] = {}

        for name, x in inputs.items():
            per_seed: list[dict] = []
            for rank, hseed in enumerate(head_seeds):
                if replicated:
                    # Seed BEFORE constructing the head, not only inside train_head.
                    # `MLPHead.__init__` draws its Linear initialisation from the
                    # ambient torch RNG, so `train_head`'s own manual_seed controls
                    # only the batch shuffling.  The pilot therefore had *one*
                    # architecture-and-shuffle seed but an initialisation that was
                    # merely incidental -- which is exactly the gap docs/HANDOFF.md
                    # §8.6 flags.  Seeding here makes each replicate a genuinely
                    # independent draw over initialisation as well.
                    #
                    # Done only under --head-seeds so the default single-seed path
                    # stays bit-identical to the executed pilot.
                    torch.manual_seed(hseed)
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
                    {**cfg["head"], "seed": hseed},
                )
                train_seconds = time.perf_counter() - t0

                probs_test = head.predict_proba(x[masks["test"]])
                probs_val = head.predict_proba(x[masks["val"]])

                entry = {
                    "head_seed": int(hseed),
                    "input_dim": int(x.shape[1]),
                    "n_parameters": int(sum(p.numel() for p in head.parameters())),
                    "best_epoch": tr.best_epoch,
                    "epochs_run": len(tr.history),
                    "train_seconds": train_seconds,
                    "history": tr.history,
                    "val": M.evaluate(
                        probs_val, y[masks["val"]], y_bin[masks["val"]], binner, intervals
                    ),
                    "test": M.evaluate(
                        probs_test, y[masks["test"]], y_bin[masks["test"]], binner, intervals
                    ),
                    "test_by_quartile": M.evaluate_by_group(
                        probs_test,
                        y[masks["test"]],
                        y_bin[masks["test"]],
                        quartile[masks["test"]],
                        binner,
                        intervals,
                    ),
                }
                per_seed.append(entry)
                ckpt = {
                    "state_dict": head.state_dict(), "in_dim": int(x.shape[1]),
                    "hidden_dim": int(cfg["head"]["hidden_dim"]), "n_bins": binner.n_bins,
                    "dropout": float(cfg["head"]["dropout"]), "binner": binner.to_dict(),
                    "property": prop, "input": name, "head_seed": int(hseed),
                }
                torch.save(ckpt, out_dir / f"head_{prop}_{name}_seed{hseed}.pt")
                if rank == 0:
                    # The unsuffixed name is what script 05 loads by default, so the
                    # first seed in the list is the one guidance steers with unless
                    # --head-seed says otherwise.
                    torch.save(ckpt, out_dir / f"head_{prop}_{name}.pt")
                    preds_store[name] = probs_test

            # The unsuffixed report entry is the first seed's, so every phase-1 key
            # keeps its meaning; replicates are reported alongside it.
            prop_report["heads"][name] = dict(per_seed[0])
            if len(per_seed) > 1:
                prop_report["heads"][name]["seed_replicates"] = [
                    {k: v for k, v in e.items() if k != "history"} for e in per_seed
                ]
                prop_report["heads"][name]["across_seeds"] = _across_seeds(per_seed)
            msg = (
                f"{prop:15s} {name:13s} test nll={per_seed[0]['test']['nll']:.4f} "
                f"mae={per_seed[0]['test']['expected_value_mae']:.4f} "
                f"auroc={per_seed[0]['test']['intervals']['target']['auroc']:.4f}"
            )
            if len(per_seed) > 1:
                a = prop_report["heads"][name]["across_seeds"]
                msg += (f"  | {len(per_seed)} seeds: nll {a['nll']['mean']:.4f}"
                        f"+-{a['nll']['std']:.4f} auroc {a['auroc']['mean']:.4f}"
                        f"+-{a['auroc']['std']:.4f}")
            print(msg)

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
