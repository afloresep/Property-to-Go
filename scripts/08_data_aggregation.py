"""Optional single data-aggregation round (README "Distribution shift").

The head is trained on base-policy prefixes but guided decoding visits prefixes the
base policy rarely produces.  This runs the one inexpensive correction the README
allows, exactly once:

  1. generate guided trajectories;
  2. collect their prefixes and terminal outcomes;
  3. mix them into the head's training data;
  4. retrain the head once;
  5. repeat the held-out guidance test.

No reinforcement learning, no iteration.  If one round does not clearly help, the
result is reported as-is.

    python scripts/08_data_aggregation.py --dataset pilot_50k --property clogp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generation, metrics as M  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer, Windows, guided_sample  # noqa: E402
from property_to_go.heads import MLPHead, train_head  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import select_quartile_prefixes  # noqa: E402
from property_to_go.properties import compute_properties  # noqa: E402
from property_to_go.splits import split_by_group  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_guided = import_module("05_guided_generation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--property", default="clogp")
    ap.add_argument("--n-aggregation-molecules", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / f"{args.dataset}_heads"
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_dagger_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    cfg = load_config(args.dataset)
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])

    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    seeds = args.seeds or list(gcfg["seeds"])
    n_eval = args.n_eval or int(gcfg["n_molecules_per_condition"])

    gen = load_generator(model_cfg)
    ck = torch.load(heads_dir / f"head_{args.property}_frozen_state.pt",
                    map_location="cpu", weights_only=False)
    binner = binner_from_dict(ck["binner"])
    head0 = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head0.load_state_dict(ck["state_dict"])
    head0.eval()
    scorer0 = TargetScorer(head0, binner, lo, hi)

    t_start = time.perf_counter()

    # ---- 1. generate guided trajectories -------------------------------------
    print(f"generating {args.n_aggregation_molecules} guided trajectories for aggregation")
    meter = ComputeMeter().start()
    seqs = guided_sample(
        gen, scorer=scorer0, window_fn=windows.fn("throughout"), policy=policy,
        n_molecules=args.n_aggregation_molecules, seed=7777,
        top_k=int(gcfg["top_k_candidates"]), lam=float(gcfg["lam"]), eps=float(gcfg["eps"]),
        backend=gcfg["candidate_backend"], batch_size=int(gcfg["batch_size"]), meter=meter,
    )
    meter.stop()

    # ---- 2. collect prefixes and terminal outcomes ---------------------------
    smiles = gen.decode(seqs)
    rng = np.random.default_rng(int(cfg["prefix_seed"]) + 1)
    keep_seqs, positions, targets, groups = [], [], [], []
    for ids, smi in zip(seqs, smiles):
        props = compute_properties(smi)
        content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
        if props is None or len(content) < int(cfg["min_content_tokens"]):
            continue
        picks = select_quartile_prefixes(len(content), rng)
        keep_seqs.append(ids)
        positions.append([k for _, k in picks])
        for _ in picks:
            targets.append(props[args.property])
            groups.append(props["canonical_smiles"])
    states = generation.hidden_states_for_positions(
        gen, keep_seqs, positions, layer=int(cfg["hidden_layer"])
    )
    guided_hidden = np.concatenate(states, axis=0).astype(np.float32)
    guided_y = np.array(targets, dtype=np.float64)
    print(f"aggregated {len(guided_hidden)} guided prefixes from {len(keep_seqs)} molecules")

    # guided prefixes get their own grouped split so the retrained head is still
    # evaluated on molecules it has never seen
    guided_split = split_by_group(groups, cfg["split_fractions"], int(cfg["split_seed"]))

    # ---- 3/4. mix in and retrain once ---------------------------------------
    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden = np.load(data_dir / "hidden.npy")
    base_split = meta["split"].to_numpy()
    base_y = meta[args.property].to_numpy().astype(np.float64)

    x_train = np.concatenate([hidden[base_split == "train"], guided_hidden[guided_split == "train"]])
    y_train = np.concatenate([base_y[base_split == "train"], guided_y[guided_split == "train"]])
    x_val = np.concatenate([hidden[base_split == "val"], guided_hidden[guided_split == "val"]])
    y_val = np.concatenate([base_y[base_split == "val"], guided_y[guided_split == "val"]])

    head1 = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    tr = train_head(head1, x_train, binner.transform(y_train), x_val,
                    binner.transform(y_val), cfg["head"])
    head1.eval()
    torch.save({**ck, "state_dict": head1.state_dict()}, out_dir / f"head_{args.property}_dagger.pt")
    print(f"retrained head: best_val_nll={tr.best_val_nll:.4f} at epoch {tr.best_epoch}")

    # how badly was the original head calibrated on guided prefixes?
    q0 = interval_probability(head0.predict_proba(guided_hidden), binner, lo, hi)
    q1 = interval_probability(head1.predict_proba(guided_hidden), binner, lo, hi)
    hit_guided = (guided_y >= lo) & (guided_y < hi)
    shift = {
        "n_guided_prefixes": int(len(guided_hidden)),
        "original_head_on_guided_prefixes": {
            "mean_predicted": float(q0.mean()), "observed": float(hit_guided.mean()),
            "brier": M.brier(q0, hit_guided), "auroc": M.auroc(q0, hit_guided),
            "ece": M.expected_calibration_error(q0, hit_guided),
        },
        "retrained_head_on_guided_prefixes": {
            "mean_predicted": float(q1.mean()), "observed": float(hit_guided.mean()),
            "brier": M.brier(q1, hit_guided), "auroc": M.auroc(q1, hit_guided),
            "ece": M.expected_calibration_error(q1, hit_guided),
        },
        "guided_generation_compute": meter.as_dict(),
    }
    print(f"distribution shift: original head predicted {q0.mean():.3f} on guided prefixes, "
          f"observed {hit_guided.mean():.3f} (ECE {shift['original_head_on_guided_prefixes']['ece']:.3f} "
          f"-> {shift['retrained_head_on_guided_prefixes']['ece']:.3f})")

    # ---- 5. repeat the held-out guidance test --------------------------------
    scorer1 = TargetScorer(head1, binner, lo, hi)
    results = {}
    for label, scorer in [("original_head", scorer0), ("retrained_head", scorer1)]:
        per_seed = []
        for seed in seeds:
            m = ComputeMeter().start()
            s = guided_sample(
                gen, scorer=scorer, window_fn=windows.fn("throughout"), policy=policy,
                n_molecules=n_eval, seed=seed + 50000,
                top_k=int(gcfg["top_k_candidates"]), lam=float(gcfg["lam"]),
                eps=float(gcfg["eps"]), backend=gcfg["candidate_backend"],
                batch_size=int(gcfg["batch_size"]), meter=m,
            )
            m.stop()
            recs = []
            for ids, smi in zip(s, gen.decode(s)):
                p = compute_properties(smi)
                c = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
                recs.append({"smiles": smi, "n_content_tokens": len(c),
                             "valid": p is not None, **(p or {})})
            summ = _guided.summarise(recs, args.property, lo, hi)
            summ["compute"] = m.as_dict()
            per_seed.append(summ)
            print(f"  {label} seed={seed} hit={summ['hit_rate']:.4f} "
                  f"err={summ['abs_target_error_mean']:.4f} val={summ['validity']:.3f}")
        results[label] = {
            "seeds": per_seed,
            "hit_rate_mean": float(np.mean([p["hit_rate"] for p in per_seed])),
            "hit_rate_std": float(np.std([p["hit_rate"] for p in per_seed])),
            "abs_target_error_mean": float(np.mean([p["abs_target_error_mean"] for p in per_seed])),
            "validity_mean": float(np.mean([p["validity"] for p in per_seed])),
        }

    report = {
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": iv,
        "rounds": 1,
        "distribution_shift": shift,
        "held_out_guidance_test": results,
        "improvement_hit_rate": results["retrained_head"]["hit_rate_mean"]
        - results["original_head"]["hit_rate_mean"],
        "wall_seconds_total": time.perf_counter() - t_start,
    }
    write_json(out_dir / "data_aggregation_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg})
    print(f"\nhit rate {results['original_head']['hit_rate_mean']:.4f} -> "
          f"{results['retrained_head']['hit_rate_mean']:.4f} "
          f"({report['improvement_hit_rate']:+.4f})")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
