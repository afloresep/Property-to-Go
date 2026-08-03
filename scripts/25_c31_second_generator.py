"""C31 -- does the crossing replicate on a second, independent molecular generator?

C28 found 8 of 30 guided cells above the oracle-selected best-of-N frontier at their own
budget, all at k = 2 or k = 4.  C30 replicated 5 of 8 across eight probe seeds.  **Every
one of those numbers is a fact about GP-MoLFormer-Uniq.**  C31 runs the whole pipeline on
`entropy/gpt2_zinc_87m` -- GPT-2, full softmax attention, 87M parameters, byte-level BPE
over SMILES, ~480M ZINC strings -- and asks whether the crossing is a property of the
method or of that one model.

Pre-registration: `outputs/c31_prereg/C31.0_preregistration.md`, frozen with its SHA-256
in `prereg_lock.json` **before the Stage 0 feasibility run**, not merely before the
decision stage.

**Nothing about the method is forked.**  The decoding rule is `guidance.combine_scores`
and the decoder is `guidance.guided_sample`, imported; the properties, binner, interval
semantics, token meter, grouped splitter, probe trainer, best-of-N key and per-condition
summariser are the molecular ones, imported.  What is new is a generator adapter
(`property_to_go.second_generator`) and the standard-attention KV-cache repeat, which is
`generality.repeat_cache_gpt2`, also reused.  A copy of `guided_sample` could agree with
the molecular one today and drift tomorrow, and a gate could then pass for the wrong
reason; C30 established that pattern and C31 follows it.

Stages, in the order a kill loses the least:

    feasibility   G0 (validity floor) + G1 (cached vs full states at all 13 probe points)
    dataset       50k trajectories, prefixes, states at all 13 probe points, frozen
                  target intervals and windows, G5 (union of bins) and G6 (leakage)
    heads         13 probe points x 3 head seeds x property, plus the trivial baseline;
                  selects the mid-network probe point by C31.0.4's PREDICTION rule
    decision-gate G2 (the cached decode makes the same decision as full recomputation)
    bestofn       the oracle-selected frontier, C26's grid and C26's estimator
    ksweep        k in {2,4,8,16,32} on the deployed and mid arms; G4 (cost identity)
    backend-gate  G3 (end-to-end cached vs full, reported as a residual)

Every stage is idempotent per output directory: a completed cell is never regenerated, so
a kill costs at most one cell.

    .venv/bin/python scripts/25_c31_second_generator.py --stage feasibility
    .venv/bin/python scripts/25_c31_second_generator.py --stage dataset
    .venv/bin/python scripts/25_c31_second_generator.py --stage heads
    .venv/bin/python scripts/25_c31_second_generator.py --stage decision-gate
    .venv/bin/python scripts/25_c31_second_generator.py --stage bestofn
    .venv/bin/python scripts/25_c31_second_generator.py --stage ksweep
    .venv/bin/python scripts/25_c31_second_generator.py --stage backend-gate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import generation, metrics as M, probe_layers, properties  # noqa: E402
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, in_interval, interval_mask_coverage,
    resolve_target_interval,
)
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.generality import train_head_on_device  # noqa: E402
from property_to_go.guidance import (  # noqa: E402
    TargetScorer, Windows, combine_scores, guided_sample,
)
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.prefixes import relative_position, select_quartile_prefixes  # noqa: E402
from property_to_go.second_generator import (  # noqa: E402
    load_zinc_generator, trivial_features_from_prefix_ids,
)
from property_to_go.splits import check_no_group_leakage, split_by_group  # noqa: E402
from property_to_go.tokens import FEATURE_NAMES  # noqa: E402

CONFIG = "c31_second_generator"

#: C31.0.2.  `hbd_count` and `aromatic_rings` are the two anchors that carry the crossing
#: on GP-MoLFormer and are REQUIRED; `qed` never crosses there and is the honest test of
#: whether that transfers.  Transcribed, not derived.
REQUIRED_PROPERTIES = ("hbd_count", "aromatic_rings")
OPTIONAL_PROPERTIES = ("qed",)
ALL_C31_PROPERTIES = REQUIRED_PROPERTIES + OPTIONAL_PROPERTIES

#: C31.0.3 G1.  C24's tolerance, reused and not re-chosen.
G1_TOLERANCE = 2e-3
#: C31.0.3 G2.
G2_PROB_TOLERANCE = 1e-3
G2_ARGMAX_AGREEMENT = 0.995
#: C31.0.3 G3.
G3_HIT_RATE_TOLERANCE = 0.05
G3_TOKEN_RELATIVE_TOLERANCE = 0.02

DATASET_DIR = "c31_zinc50k"
STATES_DIR = "c31_layer_states"
HEADS_DIR = "c31_heads"


def _load_module(path: Path, name: str):
    """Import a script whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# `summarise` is scripts/05_guided_generation.py's, imported: the per-condition summary of
# a C31 guidance cell is computed by the identical function that produced every molecular
# guidance number, so a difference between generators cannot be a difference in scoring.
_s05 = _load_module(ROOT / "scripts" / "05_guided_generation.py", "guided_generation_05")
summarise = _s05.summarise
# `score_pool` is scripts/21_n_sweep.py's, for the same reason.
_s21 = _load_module(ROOT / "scripts" / "21_n_sweep.py", "c26_n_sweep")
score_pool = _s21.score_pool


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def load_c31_config() -> dict:
    return load_config(CONFIG)


def model_cfg_of(cfg: dict) -> dict:
    """The generator pins, as `load_zinc_generator` wants them."""
    return {
        "model_repo": cfg["model_repo"],
        "model_revision": cfg["model_revision"],
        "tokenizer_repo": cfg["tokenizer_repo"],
        "tokenizer_revision": cfg["tokenizer_revision"],
        "dtype": cfg["dtype"],
        "device": cfg["device"],
        "clean_up_tokenization_spaces": cfg["clean_up_tokenization_spaces"],
    }


def make_binner(cfg: dict, prop: str, values: np.ndarray, iv: dict):
    """The molecular binner, with the molecular caps, and the target edges forced on.

    `extra_edges` is why `pilot_report.md` section 11.5's bug cannot come back: it makes
    the target interval a union of whole bins by construction rather than by luck.
    """
    b = cfg["binning"]
    if prop in properties.DISCRETE_PROPERTIES:
        return CategoricalBinner(max_value=int(b[f"{prop}_max"]))
    return QuantileBinner.fit(
        values, n_bins=int(b[f"{prop}_n_bins"]), extra_edges=(iv["lo"], iv["hi"])
    )


# ============================================================ stage: feasibility (G0, G1)


def stage_feasibility(args) -> int:
    cfg = load_c31_config()
    out_dir = OUTPUT_DIR / "c31_feasibility"
    if (out_dir / "feasibility.json").exists() and not args.force:
        print(f"[C31] skip {out_dir.name} (already complete)")
        return 0

    gen = load_zinc_generator(model_cfg_of(cfg))
    policy = dict(cfg["base_policy"])
    n = args.n_molecules or 2048

    print(f"[C31] Stage 0: {n} unconditional molecules at the base policy", flush=True)
    meter = ComputeMeter().start()
    seqs = generation.sample_unconditional(gen, policy, n, seed=int(policy["seed"]), meter=meter)
    meter.stop()
    smiles = gen.decode(seqs)

    val = properties.validity(smiles)
    uniq = properties.uniqueness(smiles)
    lengths = [len(generation.sequence_content(s, gen.bos_id, gen.eos_id, gen.pad_id))
               for s in seqs]

    # ------------------------------------------------------------------ gate G1
    # C24's gate, re-run here: candidate states from a shared KV cache versus from
    # re-running the whole extended prefix, at ALL 13 probe points, on real sequences at
    # real prefix positions.  This is the gate that makes the token accounting mean
    # anything -- `actual` charges one token per candidate precisely because a cached
    # step is the same computation as a full recomputation.
    g1 = gate_g1_states(gen, seqs[:args.g1_sequences])

    result = {
        "experiment": "C31",
        "stage": "feasibility",
        "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
        "generator": {
            "repo": cfg["model_repo"],
            "revision": cfg["model_revision"],
            "tokenizer_repo": cfg["tokenizer_repo"],
            "tokenizer_revision": cfg["tokenizer_revision"],
            "fingerprint": gen.fingerprint(),
            "n_probe_points": gen.n_probe_points,
            "n_layers": gen.n_layers,
            "hidden_size": gen.hidden_size,
            "max_position_embeddings": gen.max_length,
            "bos_id": gen.bos_id, "eos_id": gen.eos_id, "pad_id": gen.pad_id,
        },
        "base_policy": policy,
        "n_molecules": n,
        "validity": val,
        "uniqueness": uniq,
        "content_length_mean": float(np.mean(lengths)),
        "content_length_std": float(np.std(lengths)),
        "content_length_max": int(np.max(lengths)),
        "n_at_max_length": int(sum(len(s) >= int(policy["max_length"]) for s in seqs)),
        "compute": meter.as_dict(),
        "example_smiles": smiles[:12],
        "validity_gates": {"G0": {
            "rule": "unconditional RDKit validity >= 0.80 (C31.0.3 G0)",
            "threshold": 0.80, "measured": val, "passes": bool(val >= 0.80),
        }, "G1": g1},
    }
    write_json(out_dir / "feasibility.json", result)
    write_json(out_dir / "sequences.json", {"token_ids": seqs[:args.g1_sequences]})
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})

    print(f"[C31] validity={val:.6f} uniqueness={uniq:.6f} "
          f"len={result['content_length_mean']:.1f}+-{result['content_length_std']:.1f} "
          f"probe_points={gen.n_probe_points}")
    print(f"[C31] G0 passes={result['validity_gates']['G0']['passes']}")
    print(f"[C31] G1 max abs state difference={g1['max_abs_difference']:.3e} "
          f"(tol {G1_TOLERANCE:.0e}) passes={g1['passes']}")
    if not result["validity_gates"]["G0"]["passes"]:
        raise SystemExit("[C31] STOP: G0 failed -- C31.0.7 declares this UNINTERPRETABLE.")
    if not g1["passes"]:
        raise SystemExit("[C31] STOP: G1 failed -- the token accounting is not meaningful.")
    return 0


@torch.no_grad()
def gate_g1_states(gen, seqs: list[list[int]]) -> dict:
    """G1 -- cached candidate states vs full-prefix recomputation, all 13 probe points.

    Not a bit-identity claim.  Standard attention reduces in a different order on the two
    paths, exactly as C24 found on GPT-2; the residual is measured and reported.
    """
    # Positions chosen inside the shortest sequence so every probe is a real prefix.
    shortest = min(len(s) for s in seqs)
    positions = sorted({max(1, int(round(f * (shortest - 2)))) for f in (0.2, 0.4, 0.6, 0.8)})
    per_probe = {L: 0.0 for L in range(gen.n_probe_points)}
    scale = 0.0
    for p in positions:
        ids = torch.tensor([s[:p] for s in seqs], dtype=torch.long, device=gen.device)
        nxt = torch.tensor([[s[p]] for s in seqs], dtype=torch.long, device=gen.device)
        # cached: prefill once, then one token step from the shared cache
        res = gen.model(input_ids=ids, use_cache=True, return_dict=True)
        cout = gen.model(
            input_ids=nxt,
            past_key_values=gen.repeat_cache_fn(res.past_key_values, 1),
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        # full: re-run the whole extended prefix
        ext = torch.cat([ids, nxt], dim=1)
        fout = gen.model(input_ids=ext, use_cache=False, output_hidden_states=True,
                         return_dict=True)
        for L in range(gen.n_probe_points):
            a = cout.hidden_states[L][:, 0, :].float().cpu().numpy()
            b = fout.hidden_states[L][:, -1, :].float().cpu().numpy()
            per_probe[L] = max(per_probe[L], float(np.abs(a - b).max()))
            scale = max(scale, float(np.abs(b).max()))
    worst = max(per_probe.values())
    return {
        "gate": "G1",
        "rule": (f"cached candidate states equal full-prefix recomputation at every probe "
                 f"point to within {G1_TOLERANCE:.0e} absolute (C24's tolerance, reused)"),
        "n_sequences": len(seqs),
        "positions": positions,
        "n_probe_points": gen.n_probe_points,
        "max_abs_difference_by_probe_point": {str(L): v for L, v in per_probe.items()},
        "max_abs_difference": worst,
        "hidden_state_max_abs_value": scale,
        "relative_to_state_scale": worst / scale if scale else float("nan"),
        "tolerance": G1_TOLERANCE,
        "bit_identical": bool(worst == 0.0),
        "passes": bool(worst <= G1_TOLERANCE),
        "note": ("Not bit-identical, and not claimed to be: standard attention reduces in a "
                 "different order on the cached and the full-recompute paths.  The molecular "
                 "linear-attention backends are bit-identical; this one is not, and C24 "
                 "found the same on GPT-2 for text."),
    }


# ==================================================== stage: dataset (Stage 1, G5, G6)


def stage_dataset(args) -> int:
    cfg = load_c31_config()
    data_dir = OUTPUT_DIR / DATASET_DIR
    states_dir = OUTPUT_DIR / STATES_DIR
    if (data_dir / "dataset_metrics.json").exists() and not args.force:
        print(f"[C31] skip {data_dir.name} (already complete)")
        return 0

    gen = load_zinc_generator(model_cfg_of(cfg))
    policy = dict(cfg["base_policy"])
    n_traj = args.n_trajectories or int(cfg["n_trajectories"])
    t_start = time.perf_counter()

    # ---- generate --------------------------------------------------------------
    meter = ComputeMeter().start()
    seqs = generation.sample_unconditional(gen, policy, n_traj, seed=int(policy["seed"]),
                                           meter=meter)
    meter.stop()
    print(f"[C31] generated {len(seqs)} trajectories in {meter.wall_seconds:.1f}s", flush=True)
    smiles = gen.decode(seqs)

    # ---- terminal properties ---------------------------------------------------
    want = frozenset(ALL_C31_PROPERTIES)
    traj: list[dict] = []
    n_invalid = n_short = n_unavailable = 0
    for ids, smi in zip(seqs, smiles):
        props = properties.compute_all_properties(smi, extras=want)
        content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
        if props is None:
            n_invalid += 1
            continue
        if any(props.get(p) is None for p in ALL_C31_PROPERTIES):
            n_unavailable += 1
        if len(content) < int(cfg["min_content_tokens"]):
            n_short += 1
            continue
        traj.append({"token_ids": ids, "content_ids": content, "n_content": len(content),
                     "smiles": smi, **props})
    print(f"[C31] kept {len(traj)} ({n_invalid} invalid, {n_short} too short)", flush=True)

    # ---- prefix selection ------------------------------------------------------
    rng = np.random.default_rng(int(cfg["prefix_seed"]))
    rows: list[dict] = []
    positions: list[list[int]] = []
    row_offsets: list[int] = []
    for ti, t in enumerate(traj):
        picks = select_quartile_prefixes(t["n_content"], rng)
        positions.append([k for _, k in picks])
        row_offsets.append(len(rows))
        for q, k in picks:
            rows.append({
                "traj_index": ti, "quartile": q, "prefix_len": k,
                "relative_position": relative_position(k, t["n_content"]),
                "n_content": t["n_content"],
                "canonical_smiles": t["canonical_smiles"],
                **{p: t[p] for p in ALL_C31_PROPERTIES},
                "n_heavy_atoms": t["n_heavy_atoms"],
                "_prefix_ids": t["token_ids"][: k + 1],
            })
    print(f"[C31] {len(rows)} prefix rows", flush=True)

    # ---- trivial prefix statistics ---------------------------------------------
    # `tokens.prefix_features` unchanged, fed the atom-level re-split of the decoded
    # prefix.  See second_generator.smiles_atom_tokens for why the BPE tokens cannot be
    # handed to it directly and why feeding them would silently weaken the baseline the
    # frozen state has to beat.
    t0 = time.perf_counter()
    features = np.stack([
        trivial_features_from_prefix_ids(gen, r.pop("_prefix_ids")) for r in rows
    ]).astype(np.float32)
    print(f"[C31] trivial features in {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- frozen hidden states at every probe point ------------------------------
    probe = [int(L) for L in cfg["probe_points"]]
    states_dir.mkdir(parents=True, exist_ok=True)
    memmaps = {}
    for L in probe:
        f = states_dir / f"hidden_layer{L}.npy"
        memmaps[L] = np.lib.format.open_memmap(
            f, mode="w+", dtype=np.float32, shape=(len(rows), gen.hidden_size))
    hs_meter = ComputeMeter().start()
    probe_layers.hidden_states_all_layers(
        gen, [t["token_ids"] for t in traj], positions, probe,
        out=memmaps, row_offsets=row_offsets, batch_size=args.state_batch_size,
        meter=hs_meter,
    )
    hs_meter.stop()
    for L in probe:
        memmaps[L].flush()
    print(f"[C31] states at {len(probe)} probe points in {hs_meter.wall_seconds:.1f}s "
          f"({hs_meter.processed_tokens_actual} tokens -- ONE pass serves all 13)", flush=True)

    # ---- grouped splits ---------------------------------------------------------
    groups = [r["canonical_smiles"] for r in rows]
    splits = split_by_group(groups, cfg["split_fractions"], int(cfg["split_seed"]))
    group_counts = check_no_group_leakage(np.array(groups), splits)

    # ---- frozen target intervals and windows ------------------------------------
    lengths = np.array([t["n_content"] for t in traj])
    base_values = {p: np.array([t[p] for t in traj if t.get(p) is not None],
                               dtype=np.float64) for p in ALL_C31_PROPERTIES}
    rule = cfg["target_interval_rule"]
    intervals = {p: resolve_target_interval(rule[p], base_values[p])
                 for p in ALL_C31_PROPERTIES}
    windows = Windows.from_lengths(lengths, tuple(cfg["window_quantiles"]))

    write_json(data_dir / "target_intervals.json", intervals)
    write_json(data_dir / "windows.json", windows.to_dict())

    # ---- gate G5: the interval is a union of binner bins -------------------------
    g5 = {"gate": "G5", "rule": ("the target interval must be an exact union of binner "
                                 "bins, and the binner's top category must sit strictly "
                                 "above the target value (C31.0.3 G5)"),
          "properties": {}}
    binners = {}
    for p in ALL_C31_PROPERTIES:
        iv = intervals[p]
        binner = make_binner(cfg, p, base_values[p], iv)
        binners[p] = binner
        cov = interval_mask_coverage(binner, iv["lo"], iv["hi"], base_values[p])
        row = {**cov, "lo": iv["lo"], "hi": iv["hi"], "base_rate": iv["base_rate"],
               "binner_kind": binner.kind, "n_bins": int(binner.n_bins)}
        if p in properties.DISCRETE_PROPERTIES:
            row["top_category"] = int(binner.max_value)
            row["target_below_top_category"] = bool(iv["lo"] < binner.max_value)
        else:
            row["target_below_top_category"] = True
        row["passes"] = bool(cov["is_exact"] and cov["n_bins_selected"] >= 1
                             and row["target_below_top_category"])
        g5["properties"][p] = row
    g5["passes"] = all(r["passes"] for r in g5["properties"].values())

    g6 = {"gate": "G6",
          "rule": ("no molecule's prefixes straddle train and test, and the frozen "
                   "artefacts are hashed at write time (C31.0.3 G6)"),
          "group_counts": group_counts,
          "no_group_leakage": True,
          "target_intervals_sha256": sha256_file(data_dir / "target_intervals.json"),
          "windows_sha256": sha256_file(data_dir / "windows.json"),
          "passes": True}

    # ---- persist ----------------------------------------------------------------
    np.save(data_dir / "trivial_features.npy", features)
    np.save(data_dir / "splits.npy", splits)
    for p in ALL_C31_PROPERTIES:
        np.save(data_dir / f"y_{p}.npy",
                np.array([r[p] if r[p] is not None else np.nan for r in rows],
                         dtype=np.float64))
    np.save(data_dir / "quartile.npy", np.array([r["quartile"] for r in rows], dtype=np.int64))
    np.save(data_dir / "prefix_len.npy",
            np.array([r["prefix_len"] for r in rows], dtype=np.int64))
    np.save(data_dir / "relative_position.npy",
            np.array([r["relative_position"] for r in rows], dtype=np.float64))
    write_json(data_dir / "binners.json", {p: b.to_dict() for p, b in binners.items()})

    report = {
        "experiment": "C31", "stage": "dataset",
        "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
        "generator": {"repo": cfg["model_repo"], "revision": cfg["model_revision"],
                      "fingerprint": gen.fingerprint()},
        "n_trajectories_requested": n_traj,
        "n_trajectories_kept": len(traj),
        "n_invalid": n_invalid, "n_too_short": n_short,
        "n_property_unavailable": n_unavailable,
        "validity": float(1.0 - n_invalid / n_traj),
        "uniqueness": float(len({t["canonical_smiles"] for t in traj}) / len(traj)),
        "n_prefix_rows": len(rows),
        "probe_points": probe,
        "hidden_size": gen.hidden_size,
        "trivial_feature_names": list(FEATURE_NAMES),
        "split_counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "group_counts": group_counts,
        "content_length": {"mean": float(lengths.mean()), "std": float(lengths.std()),
                           "min": int(lengths.min()), "max": int(lengths.max())},
        "target_intervals": intervals,
        "windows": windows.to_dict(),
        "base_property_summary": {
            p: {"n": int(len(base_values[p])), "mean": float(base_values[p].mean()),
                "std": float(base_values[p].std()),
                "quantiles": {str(q): float(np.quantile(base_values[p], q))
                              for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95)}}
            for p in ALL_C31_PROPERTIES},
        "validity_gates": {"G5": g5, "G6": g6},
        "generation_compute": meter.as_dict(),
        "hidden_state_compute": hs_meter.as_dict(),
        "wall_seconds_total": time.perf_counter() - t_start,
    }
    write_json(data_dir / "dataset_metrics.json", report)
    write_run_context(data_dir, {"c31": cfg, "cli": vars(args)})
    write_run_context(states_dir, {"c31": cfg, "cli": vars(args)})

    for p in ALL_C31_PROPERTIES:
        iv = intervals[p]
        print(f"[C31] {p:15s} target=[{iv['lo']:.4f},{iv['hi']:.4f}) "
              f"base_rate={iv['base_rate']:.4f} exact_bins="
              f"{g5['properties'][p]['is_exact']}")
    print(f"[C31] G5 passes={g5['passes']}  G6 passes={g6['passes']}")
    if not g5["passes"]:
        raise SystemExit("[C31] STOP: G5 failed -- pilot_report.md section 11.5's bug.")
    return 0


# ====================================================== stage: heads (Stage 2, depth curve)


def _masks(splits: np.ndarray) -> dict[str, np.ndarray]:
    return {s: (splits == s) for s in ("train", "val", "test")}


def stage_heads(args) -> int:
    from property_to_go.binning import binner_from_dict, interval_probability

    cfg = load_c31_config()
    data_dir = OUTPUT_DIR / DATASET_DIR
    states_dir = OUTPUT_DIR / STATES_DIR
    heads_dir = OUTPUT_DIR / HEADS_DIR
    heads_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["device"])
    splits = np.load(data_dir / "splits.npy", allow_pickle=True).astype(str)
    masks = _masks(splits)
    quartile = np.load(data_dir / "quartile.npy")
    trivial_x = np.load(data_dir / "trivial_features.npy")
    intervals = read_json(data_dir / "target_intervals.json")
    binners_d = read_json(data_dir / "binners.json")
    head_cfg = dict(cfg["head"])
    head_seeds = [int(s) for s in cfg["head_seeds"]]
    probe = [int(L) for L in cfg["probe_points"]]

    props = args.properties or list(ALL_C31_PROPERTIES)

    def trainer(head, xt, yt, xv, yv, hcfg):
        """`generality.train_head_on_device` -- the GPU copy of `heads.train_head` that
        `tests/test_generality.py` asserts is bit-identical to it on CPU."""
        res = train_head_on_device(head, xt, yt, xv, yv, hcfg, device)
        head.to("cpu")
        return res

    for prop in props:
        out_f = heads_dir / f"depth_{prop}.json"
        if out_f.exists() and not args.force:
            print(f"[C31] skip depth_{prop}.json (already complete)")
            continue
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        binner = binner_from_dict(binners_d[prop])
        y = np.load(data_dir / f"y_{prop}.npy")
        finite = np.isfinite(y)
        y_bin = np.zeros(len(y), dtype=np.int64)
        y_bin[finite] = binner.transform(y[finite])
        # A property value can be missing only for QED and only on a parseable molecule;
        # those rows are dropped from that property's head rather than coerced.
        pm = {s: masks[s] & finite for s in masks}
        target = {"target": (lo, hi)}
        print(f"[C31] {prop}: train={pm['train'].sum()} val={pm['val'].sum()} "
              f"test={pm['test'].sum()} bins={binner.n_bins}", flush=True)

        entry: dict = {
            "experiment": "C31", "property": prop,
            "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
            "target_interval": iv, "n_bins": int(binner.n_bins),
            "head_seeds": head_seeds, "probe_points": probe,
            "head_config": head_cfg,
            "rows": {s: int(pm[s].sum()) for s in pm},
            "by_probe_point": {}, "trivial": {},
        }

        # ---- the trivial prefix-statistics baseline the frozen state must beat ----
        tri_seeds = []
        for hs in head_seeds:
            e, probs_test, _ = probe_layers.train_one_probe(
                trivial_x, y, y_bin, binner, pm, quartile, target, head_cfg, hs,
                trainer=trainer)
            q_val = None
            entry_seed = {"head_seed": hs,
                          "test_target_auroc": e["test"]["intervals"]["target"]["auroc"],
                          "test_nll": e["test"]["nll"]}
            tri_seeds.append(entry_seed)
            del probs_test, q_val
        entry["trivial"] = {
            "n_features": int(trivial_x.shape[1]),
            "feature_names": list(FEATURE_NAMES),
            "per_seed": tri_seeds,
            "test_target_auroc_mean": float(np.mean(
                [s["test_target_auroc"] for s in tri_seeds])),
        }
        print(f"[C31]   trivial test AUROC={entry['trivial']['test_target_auroc_mean']:.4f}",
              flush=True)

        # ---- every probe point, every head seed ----------------------------------
        for L in probe:
            t0 = time.perf_counter()
            x = np.load(states_dir / f"hidden_layer{L}.npy", mmap_mode="r")
            x = np.asarray(x)
            per_seed = []
            for hs in head_seeds:
                e, probs_test, head = probe_layers.train_one_probe(
                    x, y, y_bin, binner, pm, quartile, target, head_cfg, hs,
                    trainer=trainer)
                # C31.0.4: SELECTION is on validation, the depth curve is REPORTED on
                # test, so the probe point is never chosen on the split it is scored on.
                probs_val = head.predict_proba(x[pm["val"]])
                q_val = interval_probability(probs_val, binner, lo, hi)
                hit_val = in_interval(y[pm["val"]], lo, hi)
                e["val_target_auroc"] = float(M.auroc(q_val, hit_val))
                e["test_target_auroc"] = e["test"]["intervals"]["target"]["auroc"]
                per_seed.append(e)
                torch.save({"state_dict": head.state_dict(), "in_dim": int(x.shape[1]),
                            "hidden_dim": int(head_cfg["hidden_dim"]),
                            "n_bins": int(binner.n_bins),
                            "dropout": float(head_cfg["dropout"]),
                            "binner": binner.to_dict(), "head_seed": hs,
                            "input": "frozen_state", "probe_point": L,
                            "property": prop},
                           heads_dir / f"head_{prop}_L{L}_seed{hs}.pt")
                del probs_test, probs_val, head
            del x
            entry["by_probe_point"][str(L)] = {
                "probe_point": L,
                "val_target_auroc_mean": float(np.mean([e["val_target_auroc"] for e in per_seed])),
                "val_target_auroc_values": [e["val_target_auroc"] for e in per_seed],
                "test_target_auroc_mean": float(np.mean([e["test_target_auroc"] for e in per_seed])),
                "test_target_auroc_values": [e["test_target_auroc"] for e in per_seed],
                "test_target_auroc_sd": float(np.std([e["test_target_auroc"] for e in per_seed], ddof=1)),
                "test_nll_mean": float(np.mean([e["test"]["nll"] for e in per_seed])),
                "test_brier_mean": float(np.mean(
                    [e["test"]["intervals"]["target"]["brier"] for e in per_seed])),
                "test_ece_mean": float(np.mean(
                    [e["test"]["intervals"]["target"]["ece"] for e in per_seed])),
                "test_expected_value_mae_mean": float(np.mean(
                    [e["test"]["expected_value_mae"] for e in per_seed])),
                "per_seed": per_seed,
                "wall_seconds": time.perf_counter() - t0,
            }
            r = entry["by_probe_point"][str(L)]
            print(f"[C31]   L{L:>2}  val AUROC={r['val_target_auroc_mean']:.4f}  "
                  f"test AUROC={r['test_target_auroc_mean']:.4f}  "
                  f"({r['wall_seconds']:.0f}s)", flush=True)

        # ---- C31.0.4's selection rule, applied ------------------------------------
        cands = [int(L) for L in cfg["mid_probe_point_candidates"]]
        best = max(cands, key=lambda L: (entry["by_probe_point"][str(L)]["val_target_auroc_mean"],
                                         -L))
        entry["mid_probe_point"] = {
            "rule": ("C31.0.4: argmax over probe points 1..11 of held-out VALIDATION target "
                     "AUROC averaged over the three head seeds; ties to the lower index. "
                     "Selected by prediction, in advance, never by steering outcome."),
            "candidates": cands,
            "selected": best,
            "val_target_auroc": entry["by_probe_point"][str(best)]["val_target_auroc_mean"],
            "test_target_auroc": entry["by_probe_point"][str(best)]["test_target_auroc_mean"],
        }
        # The depth curve is reported on test, and its argmax is D4's quantity.
        test_best = max(probe, key=lambda L: (entry["by_probe_point"][str(L)]["test_target_auroc_mean"], -L))
        entry["test_depth_peak"] = {
            "probe_point": test_best,
            "test_target_auroc": entry["by_probe_point"][str(test_best)]["test_target_auroc_mean"],
            "final_probe_point": max(probe),
            "final_test_target_auroc": entry["by_probe_point"][str(max(probe))]["test_target_auroc_mean"],
            "peaks_before_final": bool(test_best < max(probe)),
        }
        entry["frozen_state_beats_trivial"] = {
            "best_probe_point_test_auroc": entry["test_depth_peak"]["test_target_auroc"],
            "trivial_test_auroc": entry["trivial"]["test_target_auroc_mean"],
            "margin": (entry["test_depth_peak"]["test_target_auroc"]
                       - entry["trivial"]["test_target_auroc_mean"]),
        }
        write_json(out_f, entry)
        print(f"[C31] {prop}: mid probe point M={best} (val AUROC "
              f"{entry['mid_probe_point']['val_target_auroc']:.4f}); test peak at "
              f"L{test_best} ({entry['test_depth_peak']['test_target_auroc']:.4f}), "
              f"final L{max(probe)} "
              f"({entry['test_depth_peak']['final_test_target_auroc']:.4f})", flush=True)

    write_run_context(heads_dir, {"c31": cfg, "cli": vars(args)})
    return 0


# ================================================================= arms and cell layout


def load_arms(cfg: dict, prop: str) -> dict[str, dict]:
    """The two pre-registered arms for one property, with M resolved from disk."""
    d = read_json(OUTPUT_DIR / HEADS_DIR / f"depth_{prop}.json")
    mid = int(d["mid_probe_point"]["selected"])
    return {
        "deployed": {"probe_point": int(cfg["arms"]["deployed"]["probe_point"]),
                     "lam": float(cfg["arms"]["deployed"]["lam"]),
                     "why": "the deployed analogue: the final probe point at lambda = 1"},
        "mid": {"probe_point": mid, "lam": float(cfg["arms"]["mid"]["lam"]),
                "why": ("the mid-network probe point selected by C31.0.4's prediction rule, "
                        "at lambda = 2 -- the family that carries C28's largest margins")},
    }


def cell_dir(prop: str, arm: str, probe_point: int, lam: float, k: int) -> Path:
    return OUTPUT_DIR / (
        f"c31_ksweep_{prop}_{arm}_L{probe_point}_{lam_tag(lam)}_k{k}")


def load_head_ckpt(path: Path):
    """Exactly `scripts/05_guided_generation.py::load_head`, so the head is loaded the
    way every molecular guided run loads one."""
    from property_to_go.binning import binner_from_dict
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"]), ck


def head_path(prop: str, probe_point: int, head_seed: int) -> Path:
    return OUTPUT_DIR / HEADS_DIR / f"head_{prop}_L{probe_point}_seed{head_seed}.pt"


# ============================================================ stage: decision gate (G2)


@torch.no_grad()
def stage_decision_gate(args) -> int:
    """G2 -- the cached decode makes the same DECISION as full recomputation.

    State equality to 2e-3 (G1) is not automatically decision equality: `q` feeds a
    log and then a softmax over k candidates, and a small state difference could in
    principle move the sampling distribution.  This measures the thing that actually
    decides a token.  Deterministic, so it is an equality check and not a noise
    comparison.
    """
    from property_to_go.binning import interval_probability  # noqa: F401

    cfg = load_c31_config()
    data_dir = OUTPUT_DIR / DATASET_DIR
    out_dir = OUTPUT_DIR / "c31_gates"
    gen = load_zinc_generator(model_cfg_of(cfg))
    intervals = read_json(data_dir / "target_intervals.json")
    eps = float(cfg["eps"])

    seq_src = read_json(OUTPUT_DIR / "c31_feasibility" / "sequences.json")["token_ids"]
    n = min(args.g2_prefixes, len(seq_src))
    rng = np.random.default_rng(3131)
    prefixes = []
    for s in seq_src[:n]:
        L = len(s)
        if L < 6:
            continue
        prefixes.append(s[: int(rng.integers(3, L - 1))])

    result = {"gate": "G2",
              "rule": (f"the guided sampling distribution computed from cached candidate "
                       f"states equals the one computed from full-prefix recomputation: "
                       f"max abs probability difference <= {G2_PROB_TOLERANCE:.0e} and "
                       f"argmax identical on >= {G2_ARGMAX_AGREEMENT:.1%} of prefixes "
                       f"(C31.0.3 G2)"),
              "n_prefixes": len(prefixes),
              "prob_tolerance": G2_PROB_TOLERANCE,
              "argmax_agreement_threshold": G2_ARGMAX_AGREEMENT,
              "cells": {}}

    props = args.properties or list(ALL_C31_PROPERTIES)
    worst_p, worst_a = 0.0, 1.0
    for prop in props:
        arms = load_arms(cfg, prop)
        iv = intervals[prop]
        for arm, spec in arms.items():
            L, lam = spec["probe_point"], spec["lam"]
            hp = head_path(prop, L, int(cfg["head_seeds"][0]))
            if not hp.exists():
                continue
            head, binner, _ = load_head_ckpt(hp)
            scorer = TargetScorer(head, binner, float(iv["lo"]), float(iv["hi"]))
            scorer.to(gen.device)
            for k in [int(x) for x in cfg["k_grid"]]:
                max_dp, agree, total = 0.0, 0, 0
                for start in range(0, len(prefixes), args.g2_batch):
                    chunk = prefixes[start:start + args.g2_batch]
                    by_len: dict[int, list] = {}
                    for pfx in chunk:
                        by_len.setdefault(len(pfx), []).append(pfx)
                    for length, group in by_len.items():
                        ids = torch.tensor(group, dtype=torch.long, device=gen.device)
                        b = len(group)
                        out = gen.model(input_ids=ids, use_cache=True, return_dict=True)
                        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                        cand_lp, cand_ids = torch.topk(lp, k, dim=-1)
                        # cached
                        cache = gen.repeat_cache_fn(out.past_key_values, k)
                        cout = gen.model(input_ids=cand_ids.reshape(b * k, 1),
                                         past_key_values=cache, use_cache=True,
                                         output_hidden_states=True, return_dict=True)
                        h_c = cout.hidden_states[L][:, 0, :].reshape(b, k, -1)
                        # full recomputation
                        ext = torch.cat([ids.unsqueeze(1).expand(b, k, length).reshape(b * k, length),
                                         cand_ids.reshape(b * k, 1)], dim=1)
                        fout = gen.model(input_ids=ext, use_cache=False,
                                         output_hidden_states=True, return_dict=True)
                        h_f = fout.hidden_states[L][:, -1, :].reshape(b, k, -1)
                        pc = torch.softmax(combine_scores(
                            cand_lp, scorer(h_c.reshape(-1, h_c.shape[-1])).reshape(b, k),
                            lam, eps), dim=-1)
                        pf = torch.softmax(combine_scores(
                            cand_lp, scorer(h_f.reshape(-1, h_f.shape[-1])).reshape(b, k),
                            lam, eps), dim=-1)
                        max_dp = max(max_dp, float((pc - pf).abs().max()))
                        agree += int((pc.argmax(-1) == pf.argmax(-1)).sum())
                        total += b
                frac = agree / total if total else 0.0
                name = f"{prop}_{arm}_L{L}_{lam_tag(lam)}_k{k}"
                result["cells"][name] = {
                    "property": prop, "arm": arm, "probe_point": L, "lam": lam, "k": k,
                    "n_prefixes": total,
                    "max_abs_probability_difference": max_dp,
                    "argmax_agreement": frac,
                    "passes": bool(max_dp <= G2_PROB_TOLERANCE
                                   and frac >= G2_ARGMAX_AGREEMENT),
                }
                worst_p = max(worst_p, max_dp)
                worst_a = min(worst_a, frac)
                print(f"[C31] G2 {name}: max dp={max_dp:.3e} argmax agreement={frac:.5f}",
                      flush=True)

    result["max_abs_probability_difference"] = worst_p
    result["min_argmax_agreement"] = worst_a
    result["n_cells_checked"] = len(result["cells"])
    result["passes"] = bool(result["cells"]) and all(
        c["passes"] for c in result["cells"].values())
    write_json(out_dir / "g2_decision_equality.json", result)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})
    print(f"[C31] G2: max dp={worst_p:.3e} min agreement={worst_a:.5f} "
          f"passes={result['passes']}")
    if not result["passes"]:
        raise SystemExit("[C31] STOP: G2 failed -- the cached decode is not the same decision.")
    return 0


# ================================================== stage: best-of-N frontier (Stage 3)


def stage_bestofn(args) -> int:
    cfg = load_c31_config()
    data_dir = OUTPUT_DIR / DATASET_DIR
    intervals = read_json(data_dir / "target_intervals.json")
    policy = dict(cfg["base_policy"])
    grid = [int(n) for n in cfg["best_of_n_grid"]]
    n_max = int(cfg["best_of_n_pool_depth"])
    n_mol = args.n_molecules or int(cfg["best_of_n_molecules"])
    seeds = [int(s) for s in cfg["generation_seeds"]]
    props = args.properties or list(ALL_C31_PROPERTIES)

    gen = None
    for prop in props:
        out_dir = OUTPUT_DIR / f"c31_bestofn_{prop}"
        if (out_dir / "n_sweep_metrics.json").exists() and not args.force:
            print(f"[C31] skip {out_dir.name} (already complete)")
            continue
        if gen is None:
            gen = load_zinc_generator(model_cfg_of(cfg))
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        t0 = time.perf_counter()
        per_seed: dict[str, dict] = {}
        for seed in seeds:
            pool_seed = seed * 1000  # scripts/06_best_of_n.py's convention, C26's gate 1
            meter = ComputeMeter().start()
            seqs = generation.sample_unconditional(gen, policy, n_max * n_mol,
                                                   seed=pool_seed, meter=meter)
            meter.stop()
            smiles = gen.decode(seqs)
            keys, cands, tokens = score_pool(smiles, seqs, prop, lo, hi)
            pool = len(keys)
            rows: dict[str, dict] = {}
            for n in grid:
                n_groups = pool // n
                tok = 0
                sel = []
                for i in range(n_groups):
                    block = range(i * n, i * n + n)
                    sel.append(min(block, key=lambda j: keys[j]))
                    tok += sum(tokens[j] for j in block)
                from property_to_go.bestofn import summarise as bon_summarise
                s = bon_summarise([cands[j] for j in sel], prop, lo, hi)
                s["compute"] = {"processed_tokens_actual": int(tok),
                                "molecules_returned": n_groups,
                                "tokens_per_molecule_actual": tok / n_groups}
                s["n_groups"] = n_groups
                rows[str(n)] = s
                print(f"  {prop} seed={seed} N={n:>2} hit={s['hit_rate']:.4f} "
                      f"val={s['validity']:.4f} tok/mol={tok / n_groups:.1f}", flush=True)
            per_seed[str(seed)] = {"pool_seed": pool_seed, "pool_size": pool, "rows": rows,
                                   "pool_compute": meter.as_dict(),
                                   "pool_true_hit_rate": float(np.mean(
                                       [1.0 if (c.get("valid") and c.get(prop) is not None
                                                and lo <= c[prop] < hi) else 0.0
                                        for c in cands]))}
        curve = {}
        for n in grid:
            hits = [per_seed[str(s)]["rows"][str(n)]["hit_rate"] for s in seeds]
            toks = [per_seed[str(s)]["rows"][str(n)]["compute"]["tokens_per_molecule_actual"]
                    for s in seeds]
            vals = [per_seed[str(s)]["rows"][str(n)]["validity"] for s in seeds]
            uniq = [per_seed[str(s)]["rows"][str(n)]["uniqueness"] for s in seeds]
            curve[str(n)] = {
                "n_candidates": n,
                "hit_rate_mean": float(np.mean(hits)),
                "hit_rate_values": [float(h) for h in hits],
                "hit_rate_sd": float(np.std(hits, ddof=1)),
                "tokens_per_molecule_actual": float(np.mean(toks)),
                "validity_mean": float(np.mean(vals)),
                "uniqueness_mean": float(np.mean(uniq)),
            }
        report = {
            "experiment": "C31", "stage": "bestofn", "arm": "oracle_selected",
            "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
            "generator": {"repo": cfg["model_repo"], "revision": cfg["model_revision"],
                          "fingerprint": gen.fingerprint()},
            "property": prop, "target_interval": iv, "grid": grid,
            "n_max": n_max, "n_molecules_per_seed": n_mol, "seeds": seeds,
            "accounting": "actual",
            "selection": ("bestofn.selection_key on the TRUE RDKit property of the finished "
                          "molecule -- the oracle-selected comparator, C26's arm"),
            "grouping_note": ("all disjoint consecutive groups of N over the whole pool, "
                              "C26's corrected estimator"),
            "curve": curve, "per_seed": per_seed,
            "wall_seconds_total": time.perf_counter() - t0,
        }
        write_json(out_dir / "n_sweep_metrics.json", report)
        write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})
        print(f"[C31] -> {out_dir.name}", flush=True)
    return 0


# ============================================================== stage: k sweep (Stage 4)


def run_ksweep_cell(gen, cfg, prop: str, arm: str, spec: dict, k: int,
                    n_mol: int, seeds, force: bool = False) -> Path:
    L, lam = int(spec["probe_point"]), float(spec["lam"])
    out_dir = cell_dir(prop, arm, L, lam, k)
    if (out_dir / "k_cell_metrics.json").exists() and not force:
        print(f"[C31] skip {out_dir.name} (already complete)", flush=True)
        return out_dir

    data_dir = OUTPUT_DIR / DATASET_DIR
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    iv = intervals[prop]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    hs = int(cfg["head_seeds"][0])
    hp = head_path(prop, L, hs)
    if not hp.exists():
        raise SystemExit(f"[C31] missing head checkpoint {hp}")
    head, binner, ck = load_head_ckpt(hp)
    scorer = TargetScorer(head, binner, lo, hi)

    policy = {"temperature": cfg["base_policy"]["temperature"],
              "max_length": cfg["base_policy"]["max_length"]}
    report: dict = {
        "experiment": "C31", "stage": "ksweep",
        "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
        "generator": {"repo": cfg["model_repo"], "revision": cfg["model_revision"]},
        "property": prop, "arm": arm, "arm_why": spec["why"],
        "condition": "throughout", "head_input": "frozen_state",
        "head_checkpoint": hp.name, "head_file": str(hp), "head_seed": hs,
        "probe_point": L, "layer": L, "target_interval": iv,
        "windows": windows.to_dict(), "lambda": lam, "top_k": k,
        "eps": float(cfg["eps"]), "backend": cfg["candidate_backend"],
        "batch_size": int(cfg["guidance_batch_size"]),
        "n_molecules_per_seed": n_mol, "seeds": list(seeds), "seeds_detail": {},
    }
    records_by_seed: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    want = frozenset({prop})

    for seed in seeds:
        meter = ComputeMeter().start()
        seqs = guided_sample(
            gen, scorer=scorer,
            window_fn=windows.fn("throughout"),
            policy=policy, n_molecules=n_mol, seed=seed, top_k=k, lam=lam,
            eps=float(cfg["eps"]), backend=cfg["candidate_backend"],
            batch_size=int(cfg["guidance_batch_size"]),
            layer=(-1 if L == gen.n_layers else L), meter=meter,
        )
        meter.stop()
        smiles = gen.decode(seqs)
        records = []
        for ids, smi in zip(seqs, smiles):
            p = properties.compute_all_properties(smi, extras=want)
            content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
            records.append({"smiles": smi, "n_content_tokens": len(content),
                            "valid": p is not None, **(p or {})})
        records_by_seed[str(seed)] = records
        st = summarise(records, prop, lo, hi)
        st["compute"] = meter.as_dict()
        st["validity_check"] = properties.validity(smiles)
        st["uniqueness_check"] = properties.uniqueness(smiles)
        # G4: the cached backend charges `active` + `active * k` at every guided step, so
        # the actual token count must be divisible by (k + 1) exactly.
        st["cost_identity_tokens_mod_k_plus_1"] = int(
            st["compute"]["processed_tokens_actual"] % (k + 1))
        st["cost_identity_base_steps"] = st["compute"]["processed_tokens_actual"] / (k + 1)
        report["seeds_detail"][str(seed)] = st
        print(f"  {prop} {arm} L{L} lam={lam} k={k:>2} seed={seed} "
              f"hit={st['hit_rate']:.6f} val={st['validity']:.4f} "
              f"len={st['content_length_mean']:.1f} "
              f"tok/mol={st['compute']['tokens_per_molecule_actual']:.4f} "
              f"({meter.wall_seconds:.0f}s)", flush=True)

    keys = ["hit_rate", "abs_target_error_mean", "validity", "uniqueness",
            "property_mean", "content_length_mean", "n_heavy_atoms_mean"]
    agg: dict = {}
    for key in keys:
        v = [report["seeds_detail"][str(sd)][key] for sd in seeds]
        agg[key] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "values": v}
    tot = ComputeMeter()
    for sd in seeds:
        c = report["seeds_detail"][str(sd)]["compute"]
        tot.processed_tokens_actual += c["processed_tokens_actual"]
        tot.processed_tokens_full_recompute += c["processed_tokens_full_recompute"]
        tot.wall_seconds += c["wall_seconds"]
        tot.molecules_returned += c["molecules_returned"]
        tot.forward_calls += c["forward_calls"]
    agg["compute_total"] = tot.as_dict()
    report["aggregate"] = agg
    report["cost_identity_max_residual"] = max(
        report["seeds_detail"][str(sd)]["cost_identity_tokens_mod_k_plus_1"] for sd in seeds)
    report["wall_seconds_total"] = time.perf_counter() - t0

    write_json(out_dir / "k_cell_metrics.json", report)
    write_json(out_dir / "molecules.json", records_by_seed)
    write_run_context(out_dir, {"c31": cfg, "cli": {
        "property": prop, "arm": arm, "probe_point": L, "lam": lam, "top_k": k,
        "head_file": str(hp), "condition": "throughout"}})
    print(f"[C31] -> {out_dir.name}  hit={agg['hit_rate']['mean']:.6f} "
          f"tok/mol={agg['compute_total']['tokens_per_molecule_actual']:.6f}", flush=True)
    return out_dir


def stage_ksweep(args) -> int:
    cfg = load_c31_config()
    seeds = tuple(int(s) for s in cfg["generation_seeds"])
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    ks = [int(x) for x in (args.k or cfg["k_grid"])]
    props = args.properties or list(ALL_C31_PROPERTIES)
    arms_wanted = args.arms or ["deployed", "mid"]

    gen = load_zinc_generator(model_cfg_of(cfg))
    t0 = time.perf_counter()
    n_done = 0
    # cheap k first so a kill loses the least
    for k in sorted(ks):
        for prop in props:
            arms = load_arms(cfg, prop)
            for arm in arms_wanted:
                run_ksweep_cell(gen, cfg, prop, arm, arms[arm], k, n_mol, seeds,
                                force=args.force)
                n_done += 1
                print(f"[C31] {n_done} cells done, {time.perf_counter() - t0:.0f}s",
                      flush=True)
    print(f"[C31] k sweep complete: {n_done} cells in {time.perf_counter() - t0:.0f}s")
    return 0


# ============================================================ stage: backend gate (G3)


def stage_backend_gate(args) -> int:
    """G3 -- end-to-end `cached` vs `full`, reported as a residual with a tolerance.

    The two paths differ at ~1e-3 in `q` (G1), so sampled trajectories may diverge and
    exact equality is NOT required.  Claiming it would be dishonest.  What is required
    is that the residual is small compared with the noise the comparison itself has.
    """
    cfg = load_c31_config()
    out_dir = OUTPUT_DIR / "c31_gates"
    data_dir = OUTPUT_DIR / DATASET_DIR
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])

    prop, L, lam, k, seed = "hbd_count", 12, 1.0, 8, 101
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    iv = intervals[prop]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    gen = load_zinc_generator(model_cfg_of(cfg))
    head, binner, _ = load_head_ckpt(head_path(prop, L, int(cfg["head_seeds"][0])))
    scorer = TargetScorer(head, binner, lo, hi)
    policy = {"temperature": cfg["base_policy"]["temperature"],
              "max_length": cfg["base_policy"]["max_length"]}

    arms = {}
    for backend in ("cached", "full"):
        meter = ComputeMeter().start()
        seqs = guided_sample(gen, scorer=scorer, window_fn=windows.fn("throughout"),
                             policy=policy, n_molecules=n_mol, seed=seed, top_k=k, lam=lam,
                             eps=float(cfg["eps"]), backend=backend,
                             batch_size=int(cfg["guidance_batch_size"]),
                             layer=-1, meter=meter)
        meter.stop()
        smiles = gen.decode(seqs)
        recs = []
        for ids, smi in zip(seqs, smiles):
            p = properties.compute_all_properties(smi, extras=frozenset({prop}))
            content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
            recs.append({"smiles": smi, "n_content_tokens": len(content),
                         "valid": p is not None, **(p or {})})
        st = summarise(recs, prop, lo, hi)
        st["compute"] = meter.as_dict()
        arms[backend] = st
        print(f"[C31] G3 backend={backend}: hit={st['hit_rate']:.6f} "
              f"tok/mol_full_recompute="
              f"{st['compute']['tokens_per_molecule_full_recompute']:.2f}", flush=True)

    dh = arms["cached"]["hit_rate"] - arms["full"]["hit_rate"]
    tc = arms["cached"]["compute"]["tokens_per_molecule_full_recompute"]
    tf = arms["full"]["compute"]["tokens_per_molecule_full_recompute"]
    rel = abs(tc - tf) / tf if tf else float("nan")
    result = {
        "gate": "G3",
        "rule": (f"one cell decoded under both backends: |delta hit rate| <= "
                 f"{G3_HIT_RATE_TOLERANCE} and |delta tokens per molecule under "
                 f"full-recompute accounting| / value <= {G3_TOKEN_RELATIVE_TOLERANCE} "
                 f"(C31.0.3 G3).  Reported as a residual; does not block."),
        "cell": {"property": prop, "probe_point": L, "lam": lam, "k": k, "seed": seed,
                 "n_molecules": n_mol},
        "cached": {kk: arms["cached"][kk] for kk in
                   ("hit_rate", "validity", "uniqueness", "content_length_mean")},
        "full": {kk: arms["full"][kk] for kk in
                 ("hit_rate", "validity", "uniqueness", "content_length_mean")},
        "cached_compute": arms["cached"]["compute"],
        "full_compute": arms["full"]["compute"],
        "hit_rate_residual": dh,
        "hit_rate_tolerance": G3_HIT_RATE_TOLERANCE,
        "token_full_recompute_relative_residual": rel,
        "token_relative_tolerance": G3_TOKEN_RELATIVE_TOLERANCE,
        "binomial_two_se_at_n": float(2 * np.sqrt(0.25 / n_mol)),
        "passes": bool(abs(dh) <= G3_HIT_RATE_TOLERANCE
                       and rel <= G3_TOKEN_RELATIVE_TOLERANCE),
        "note": ("Exact equality is NOT required and is not claimed: G1 shows the two "
                 "paths differ at ~1e-3 in the candidate state, so a sampled trajectory "
                 "can diverge.  The `actual` token counts differ BY DESIGN -- that is "
                 "what the two accounting rules measure -- so the comparison is made "
                 "under the full-recompute rule, which both backends should agree on."),
    }
    write_json(out_dir / "g3_backend_equivalence.json", result)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})
    print(f"[C31] G3: hit residual={dh:+.6f} token relative residual={rel:.6f} "
          f"passes={result['passes']}")
    return 0


# ============================== POST HOC, NOT PRE-REGISTERED: the length-matched control


def stage_length_control(args) -> int:
    """POST HOC, NOT PRE-REGISTERED -- is the advantage a length effect?

    Guided molecules come out longer than base-policy ones, and `hbd_count` and
    `aromatic_rings` both grow with molecule size, so a decoder that only made molecules
    bigger would look like a controller.  `pilot_report.md` section 19 ran exactly this
    control on GP-MoLFormer; C31.0 did **not** pre-register it, and it is reported here as
    a labelled post-hoc addition rather than folded into the pre-registered result.

    `length_matched_hit_rate` is `scripts/05_guided_generation.py`'s, imported: each
    guided cell's per-length-bin hit rate is reweighted by the **unguided** length
    distribution.  The reference is 512 x 3 base-policy molecules at the same generation
    seeds -- which is the N = 1 arm of the best-of-N comparator, regenerated here only
    because the frontier run stores no per-molecule lengths.
    """
    cfg = load_c31_config()
    out_dir = OUTPUT_DIR / "c31_length_control"
    if (out_dir / "length_control.json").exists() and not args.force:
        print(f"[C31] skip {out_dir.name} (already complete)")
        return 0
    gen = load_zinc_generator(model_cfg_of(cfg))
    policy = dict(cfg["base_policy"])
    intervals = read_json(OUTPUT_DIR / DATASET_DIR / "target_intervals.json")
    seeds = [int(s) for s in cfg["generation_seeds"]]
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    want = frozenset(ALL_C31_PROPERTIES)

    reference: list[dict] = []
    for seed in seeds:
        seqs = generation.sample_unconditional(gen, policy, n_mol, seed=seed * 1000)
        for ids, smi in zip(seqs, gen.decode(seqs)):
            p = properties.compute_all_properties(smi, extras=want)
            content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
            reference.append({"smiles": smi, "n_content_tokens": len(content),
                              "valid": p is not None, **(p or {})})
    ref_len = float(np.mean([r["n_content_tokens"] for r in reference]))

    result = {
        "what": "POST HOC, NOT PRE-REGISTERED -- the length-matched control (C31.6)",
        "reference": {"n": len(reference), "seeds": [s * 1000 for s in seeds],
                      "content_length_mean": ref_len,
                      "policy": "unguided base policy, the N = 1 arm of the comparator"},
        "method": ("scripts/05_guided_generation.py::length_matched_hit_rate, imported: "
                   "per-length-bin guided hit rate reweighted by the unguided bin "
                   "frequencies, bin width 5 content tokens"),
        "cells": {},
    }
    for prop in ALL_C31_PROPERTIES:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        result["cells"].setdefault(prop, {})
        ref_summary = summarise(reference, prop, lo, hi)
        result["cells"][prop]["unguided_reference"] = {
            "hit_rate": ref_summary["hit_rate"],
            "content_length_mean": ref_summary["content_length_mean"]}
        arms = load_arms(cfg, prop)
        for arm, spec in arms.items():
            L, lam = int(spec["probe_point"]), float(spec["lam"])
            for k in [int(x) for x in cfg["k_grid"]]:
                d = cell_dir(prop, arm, L, lam, k)
                f = d / "molecules.json"
                if not f.exists():
                    continue
                recs = [r for rows in read_json(f).values() for r in rows]
                st = summarise(recs, prop, lo, hi)
                lm = _s05.length_matched_hit_rate(recs, reference, prop, lo, hi)
                result["cells"][prop][d.name] = {
                    "arm": arm, "k": k, "probe_point": L, "lam": lam,
                    "hit_rate": st["hit_rate"],
                    "content_length_mean": st["content_length_mean"],
                    "content_length_ratio_to_unguided": st["content_length_mean"] / ref_len,
                    **lm,
                    "length_matched_minus_raw": lm["length_matched_hit_rate"] - st["hit_rate"],
                }
                print(f"  {d.name}: raw={st['hit_rate']:.4f} "
                      f"length_matched={lm['length_matched_hit_rate']:.4f} "
                      f"len={st['content_length_mean']:.1f} (unguided {ref_len:.1f})",
                      flush=True)
    write_json(out_dir / "length_control.json", result)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})
    return 0


# ============================================================================== main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["feasibility", "dataset", "heads", "decision-gate",
                             "bestofn", "ksweep", "backend-gate",
                             "length-control"])
    ap.add_argument("--properties", nargs="*", default=None, choices=list(ALL_C31_PROPERTIES))
    ap.add_argument("--arms", nargs="*", default=None, choices=["deployed", "mid"])
    ap.add_argument("--k", type=int, nargs="*", default=None)
    ap.add_argument("--n-molecules", type=int, default=None)
    ap.add_argument("--n-trajectories", type=int, default=None)
    ap.add_argument("--state-batch-size", type=int, default=128)
    ap.add_argument("--g1-sequences", type=int, default=128)
    ap.add_argument("--g2-prefixes", type=int, default=512)
    ap.add_argument("--g2-batch", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    return {
        "feasibility": stage_feasibility,
        "dataset": stage_dataset,
        "heads": stage_heads,
        "decision-gate": stage_decision_gate,
        "bestofn": stage_bestofn,
        "ksweep": stage_ksweep,
        "backend-gate": stage_backend_gate,
        "length-control": stage_length_control,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
