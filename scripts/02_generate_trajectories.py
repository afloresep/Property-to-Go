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

from property_to_go import generation, probe_layers, properties  # noqa: E402
from property_to_go.binning import in_interval, resolve_target_interval  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import RunDir, load_config, read_json, write_json  # noqa: E402
from property_to_go.guidance import Windows  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import relative_position, select_quartile_prefixes  # noqa: E402
from property_to_go.splits import check_no_group_leakage, split_by_group  # noqa: E402
from property_to_go.tokens import FEATURE_NAMES, prefix_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_10k")
    ap.add_argument(
        "--inherit-intervals", default=None,
        help="path to an existing target_intervals.json. Any property present there is "
             "copied VERBATIM instead of being re-derived; the rest are derived from "
             "this run's base distribution. Use this whenever a frozen interval must "
             "survive a regeneration -- see docs/HANDOFF.md and pilot_report.md §11.2.",
    )
    # C17. Additive probe-layer selection. Omitted, this script behaves exactly as before:
    # `hidden.npy` holds the states at `cfg["hidden_layer"]` (-1, the final layer) and no
    # other array is written. Given probe points, the SAME forward passes additionally
    # write `hidden_layer<L>.npy` for each -- `output_hidden_states=True` already returns
    # every layer, so the extra probe points cost zero additional processed tokens. The
    # default array is still written unchanged, so no downstream consumer sees a
    # difference. See reports/section_c17_probe_layers.md.
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="extra probe points to store as hidden_layer<L>.npy, at no "
                         "extra token cost. Default: none, i.e. the existing behaviour.")
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
    n_property_unavailable = 0
    for ids, smi in zip(seqs, smiles):
        # `compute_all_properties` extends `compute_properties` with the four phase-2
        # descriptors and, critically, keeps validity decided by RDKit parsing alone,
        # so the kept-trajectory set is identical to the pilot's.
        props = properties.compute_all_properties(smi)
        content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
        if props is None:
            n_invalid += 1
            continue
        if any(props.get(p) is None for p in properties.ALL_PROPERTIES):
            n_property_unavailable += 1
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
                    # Every property in ALL_PROPERTIES, so heads for the phase-2
                    # battery train from the same prefix table as the pilot's three.
                    **{p: t[p] for p in properties.ALL_PROPERTIES},
                    "n_heavy_atoms": t["n_heavy_atoms"],
                    "prefix_token_ids": t["token_ids"][: k + 1],
                    "_features": prefix_features(toks),
                }
            )

    # ---- frozen hidden states -------------------------------------------------
    hs_meter = ComputeMeter().start()
    sequences = [t["token_ids"] for t in traj]
    default_layer = int(cfg["hidden_layer"])
    extra_layers: list[int] = []
    if args.layers:
        # One pass, every requested probe point. `default_layer` is normalised against
        # the returned tuple so that -1 and 12 are recognised as the same probe point and
        # `hidden.npy` is never written twice from two different code paths.
        n_probe = int(gen.model.config.num_hidden_layers) + 1
        canonical_default = default_layer % n_probe
        extra_layers = sorted({int(L) % n_probe for L in args.layers} - {canonical_default})
        multi = probe_layers.hidden_states_all_layers(
            gen, sequences, positions, [canonical_default] + extra_layers, meter=hs_meter,
        )
        states = multi[canonical_default]
    else:
        states = generation.hidden_states_for_positions(
            gen, sequences, positions, layer=default_layer, meter=hs_meter,
        )
        multi = None
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
    lengths = np.array([t["n_content"] for t in traj])
    # A property value can be missing only for QED, and only on a molecule RDKit
    # parsed; those rows are excluded from that property's base distribution rather
    # than coerced, and the count is reported.
    base_values = {
        p: np.array([t[p] for t in traj if t.get(p) is not None], dtype=np.float64)
        for p in properties.ALL_PROPERTIES
    }

    rule = guide_cfg["target_interval_rule"]
    # One code path for all properties, so a newly added interval is derived by exactly
    # the code that derived the existing ones.
    derived = {
        p: resolve_target_interval(rule[p], base_values[p]) for p in properties.ALL_PROPERTIES
    }

    # An interval derived from a *sample* is a property of that sample, not of the base
    # policy alone: the target band is an empirical quantile of ~50k draws, so an
    # independent 50k draw moves it by a couple of standard errors. That is fine when a
    # dataset is built once, and fatal when a frozen interval has to survive a
    # regeneration on hardware whose RNG stream differs. `--inherit-intervals` is how a
    # freeze is made to mean frozen: copy verbatim, do not re-derive.
    intervals = dict(derived)
    inherited: list[str] = []
    if args.inherit_intervals:
        source = read_json(Path(args.inherit_intervals))
        for p in properties.ALL_PROPERTIES:
            if p in source:
                intervals[p] = source[p]
                inherited.append(p)
    interval_provenance = {
        "inherited_from": args.inherit_intervals,
        "inherited_verbatim": inherited,
        "derived_from_this_run": [p for p in properties.ALL_PROPERTIES if p not in inherited],
        "derived_values_for_comparison": {p: derived[p] for p in inherited},
        # An inherited interval carries the base rate of the sample it was cut from, not
        # of this one. Both are needed: the recorded rate is what the freeze fixed, the
        # empirical rate is what a hit actually costs on this dataset. Reporting only one
        # would make some later comparison quietly wrong.
        "empirical_base_rate_on_this_sample": {
            p: float(in_interval(base_values[p], intervals[p]["lo"], intervals[p]["hi"]).mean())
            for p in properties.ALL_PROPERTIES
        },
        "note": (
            "Inherited intervals were frozen before any guided result was inspected and "
            "are copied unchanged. `derived_values_for_comparison` records what THIS "
            "run's base sample would have produced, so the size of the divergence is on "
            "disk rather than only in the report. `empirical_base_rate_on_this_sample` "
            "is the inherited interval's actual rate here, which differs from the "
            "`base_rate` field carried along with the interval."
        ),
    }
    windows = Windows.from_lengths(lengths, tuple(guide_cfg["window_quantiles"]))

    write_json(run / "target_intervals.json", intervals)
    write_json(run / "target_intervals_provenance.json", interval_provenance)
    write_json(run / "windows.json", windows.to_dict())

    # ---- persist --------------------------------------------------------------
    np.save(run / "hidden.npy", hidden)
    np.save(run / "features.npy", features)
    for L in extra_layers:
        np.save(
            run / f"hidden_layer{L}.npy",
            np.concatenate(multi[L], axis=0).astype(np.float32),
        )

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
                **{p: t[p] for p in properties.ALL_PROPERTIES},
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
        # Parseable molecules for which some descriptor (QED only) could not be
        # computed. They are kept -- validity means "RDKit accepted it" throughout
        # this project -- and excluded from that one property's statistics.
        "n_property_unavailable": n_property_unavailable,
        "validity": properties.validity(smiles),
        "uniqueness": properties.uniqueness(smiles),
        "n_prefix_examples": len(rows),
        "n_unique_canonical": int(len(set(groups))),
        "feature_names": list(FEATURE_NAMES),
        "hidden_dim": int(hidden.shape[1]),
        "hidden_layer": default_layer,
        # C17: extra probe points stored alongside, at zero extra token cost.
        "extra_probe_layers": extra_layers,
        "split_row_counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "split_group_counts": group_counts,
        "base_distribution": {
            **{p: stats(base_values[p]) for p in properties.ALL_PROPERTIES},
            "content_length": stats(lengths),
            # Histograms for the count properties: the target-interval base rate of a
            # count is set by one bar of these, so the shape is worth carrying.
            **{
                f"{p}_histogram": {
                    str(int(v)): int(c)
                    for v, c in zip(*np.unique(base_values[p], return_counts=True))
                }
                for p in properties.DISCRETE_PROPERTIES
            },
            "n_scored": {p: int(len(base_values[p])) for p in properties.ALL_PROPERTIES},
        },
        "target_intervals": intervals,
        "target_interval_provenance": interval_provenance,
        "windows": windows.to_dict(),
        "compute_generation": meter.as_dict(),
        "compute_hidden_states": hs_meter.as_dict(),
        "wall_seconds_total": time.perf_counter() - t_start,
    }
    write_json(run / "dataset_summary.json", summary)

    print(f"prefix examples: {len(rows)}  splits: {summary['split_row_counts']}")
    print("target intervals, frozen here before any guided molecule exists:")
    for p in properties.ALL_PROPERTIES:
        iv = intervals[p]
        tag = "INHERITED" if p in inherited else "derived  "
        print(f"  {p:16s} {tag} [{iv['lo']:.4f}, {iv['hi']:.4f})  "
              f"width={iv['hi'] - iv['lo']:.4f}  base_rate={iv['base_rate']:.4f}  "
              f"rule={iv['rule']}")
        if p in inherited:
            d = derived[p]
            print(f"  {'':16s}           this run would have derived "
                  f"[{d['lo']:.4f}, {d['hi']:.4f}) base_rate={d['base_rate']:.4f}")
    print(f"windows: early<{windows.t33}  middle[{windows.t33},{windows.t67})  late>={windows.t67}")
    print(f"-> {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
