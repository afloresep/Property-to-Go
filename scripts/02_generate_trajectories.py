"""Phase 2/3 -- trajectory dataset.

Generates N trajectories under the fixed base policy, picks one prefix per
sequence-length quartile, stores the frozen hidden state and the trivial prefix
statistics for each, and computes the terminal properties.

Also freezes, from the base generator's own empirical distributions and before any
guided molecule exists:
  * target_intervals.json  -- the target intervals and their base rates
  * windows.json           -- the early / middle / late token windows

    python scripts/02_generate_trajectories.py --config pilot_10k
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generation, properties  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import RunDir, load_config, write_json  # noqa: E402
from property_to_go.guidance import Windows  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import relative_position, select_quartile_prefixes  # noqa: E402
from property_to_go.splits import check_no_group_leakage, split_by_group  # noqa: E402
from property_to_go.tokens import FEATURE_NAMES, prefix_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_10k")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = load_config("model")
    policy = load_config("base_policy")
    guide_cfg = load_config("guidance")
    run = RunDir.create(
        args.out or cfg["name"],
        {"model": model_cfg, "base_policy": policy, "pilot": cfg, "guidance": guide_cfg},
    )

    gen = load_generator(model_cfg)
    id_to_token = gen.id_to_token()
    t_start = time.perf_counter()

    # ---- generate -------------------------------------------------------------
    meter = ComputeMeter().start()
    seqs = generation.sample_unconditional(
        gen, policy, int(cfg["n_trajectories"]), seed=int(policy["seed"]), meter=meter
    )
    meter.stop()
    print(f"generated {len(seqs)} trajectories in {meter.wall_seconds:.1f}s")

    smiles = gen.decode(seqs)

    # ---- terminal properties --------------------------------------------------
    traj: list[dict] = []
    n_invalid = n_short = 0
    for ids, smi in zip(seqs, smiles):
        props = properties.compute_properties(smi)
        content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
        if props is None:
            n_invalid += 1
            continue
        if len(content) < int(cfg["min_content_tokens"]):
            n_short += 1
            continue
        traj.append(
            {
                "token_ids": ids,
                "content_ids": content,
                "n_content": len(content),
                "smiles": smi,
                **props,
            }
        )
    print(f"kept {len(traj)} trajectories ({n_invalid} invalid, {n_short} too short)")

    # ---- prefix selection -----------------------------------------------------
    rng = np.random.default_rng(int(cfg["prefix_seed"]))
    rows: list[dict] = []
    positions: list[list[int]] = []
    for ti, t in enumerate(traj):
        picks = select_quartile_prefixes(t["n_content"], rng)
        positions.append([k for _, k in picks])
        for q, k in picks:
            toks = [id_to_token[i] for i in t["content_ids"][:k]]
            rows.append(
                {
                    "traj_index": ti,
                    "quartile": q,
                    "prefix_len": k,
                    "relative_position": relative_position(k, t["n_content"]),
                    "n_content": t["n_content"],
                    "canonical_smiles": t["canonical_smiles"],
                    "clogp": t["clogp"],
                    "aromatic_rings": t["aromatic_rings"],
                    "mol_weight": t["mol_weight"],
                    "n_heavy_atoms": t["n_heavy_atoms"],
                    "prefix_token_ids": t["token_ids"][: k + 1],
                    "_features": prefix_features(toks),
                }
            )

    # ---- frozen hidden states -------------------------------------------------
    hs_meter = ComputeMeter().start()
    states = generation.hidden_states_for_positions(
        gen,
        [t["token_ids"] for t in traj],
        positions,
        layer=int(cfg["hidden_layer"]),
        meter=hs_meter,
    )
    hs_meter.stop()
    print(f"hidden states in {hs_meter.wall_seconds:.1f}s ({hs_meter.processed_tokens_actual} tokens)")

    hidden = np.concatenate([s for s in states], axis=0).astype(np.float32)
    features = np.stack([r.pop("_features") for r in rows]).astype(np.float32)
    assert len(hidden) == len(rows) == len(features)

    # ---- grouped splits -------------------------------------------------------
    groups = [r["canonical_smiles"] for r in rows]
    splits = split_by_group(groups, cfg["split_fractions"], int(cfg["split_seed"]))
    group_counts = check_no_group_leakage(np.array(groups), splits)

    # ---- base distributions, target intervals, windows ------------------------
    clogp = np.array([t["clogp"] for t in traj])
    rings = np.array([t["aromatic_rings"] for t in traj])
    mw = np.array([t["mol_weight"] for t in traj])
    lengths = np.array([t["n_content"] for t in traj])

    rule = guide_cfg["target_interval_rule"]
    lo_c, hi_c = (
        float(np.quantile(clogp, rule["clogp"]["lo"])),
        float(np.quantile(clogp, rule["clogp"]["hi"])),
    )
    ring_v = int(rule["aromatic_rings"]["value"])
    lo_m, hi_m = float(np.quantile(mw, 0.85)), float(np.quantile(mw, 0.95))

    intervals = {
        "clogp": {
            "lo": lo_c,
            "hi": hi_c,
            "rule": rule["clogp"],
            "base_rate": float(((clogp >= lo_c) & (clogp < hi_c)).mean()),
        },
        "aromatic_rings": {
            "lo": float(ring_v),
            "hi": float(ring_v + 1),
            "rule": rule["aromatic_rings"],
            "base_rate": float((rings == ring_v).mean()),
        },
        "mol_weight": {
            "lo": lo_m,
            "hi": hi_m,
            "rule": {"kind": "quantile_band", "lo": 0.85, "hi": 0.95},
            "base_rate": float(((mw >= lo_m) & (mw < hi_m)).mean()),
        },
    }
    windows = Windows.from_lengths(lengths, tuple(guide_cfg["window_quantiles"]))

    write_json(run / "target_intervals.json", intervals)
    write_json(run / "windows.json", windows.to_dict())

    # ---- persist --------------------------------------------------------------
    np.save(run / "hidden.npy", hidden)
    np.save(run / "features.npy", features)

    import pandas as pd

    meta = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "prefix_token_ids"} for r in rows]
    )
    meta["split"] = splits
    meta.to_csv(run / "prefix_meta.csv", index=False)
    # prefix token ids are kept separately: the rollout bank replays them verbatim
    write_json(run / "prefix_token_ids.json", [r["prefix_token_ids"] for r in rows])

    write_json(
        run / "trajectories.json",
        [
            {
                "token_ids": t["token_ids"],
                "smiles": t["smiles"],
                "canonical_smiles": t["canonical_smiles"],
                "n_content": t["n_content"],
                "clogp": t["clogp"],
                "aromatic_rings": t["aromatic_rings"],
                "mol_weight": t["mol_weight"],
                "n_heavy_atoms": t["n_heavy_atoms"],
            }
            for t in traj
        ],
    )

    def stats(v):
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "quantiles": {
                str(q): float(np.quantile(v, q)) for q in (0.05, 0.25, 0.33, 0.5, 0.67, 0.75, 0.95)
            },
        }

    summary = {
        "n_requested": int(cfg["n_trajectories"]),
        "n_generated": len(seqs),
        "n_valid_kept": len(traj),
        "n_invalid": n_invalid,
        "n_too_short": n_short,
        "validity": properties.validity(smiles),
        "uniqueness": properties.uniqueness(smiles),
        "n_prefix_examples": len(rows),
        "n_unique_canonical": int(len(set(groups))),
        "feature_names": list(FEATURE_NAMES),
        "hidden_dim": int(hidden.shape[1]),
        "split_row_counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "split_group_counts": group_counts,
        "base_distribution": {
            "clogp": stats(clogp),
            "aromatic_rings": stats(rings),
            "mol_weight": stats(mw),
            "content_length": stats(lengths),
            "aromatic_rings_histogram": {
                str(int(v)): int(c) for v, c in zip(*np.unique(rings, return_counts=True))
            },
        },
        "target_intervals": intervals,
        "windows": windows.to_dict(),
        "compute_generation": meter.as_dict(),
        "compute_hidden_states": hs_meter.as_dict(),
        "wall_seconds_total": time.perf_counter() - t_start,
    }
    write_json(run / "dataset_summary.json", summary)

    print(f"prefix examples: {len(rows)}  splits: {summary['split_row_counts']}")
    print(f"targets: clogp[{lo_c:.3f},{hi_c:.3f}) base={intervals['clogp']['base_rate']:.3f}  "
          f"rings=={ring_v} base={intervals['aromatic_rings']['base_rate']:.3f}")
    print(f"windows: early<{windows.t33}  middle[{windows.t33},{windows.t67})  late>={windows.t67}")
    print(f"-> {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
