"""C33 -- the head-selected best-of-N control, on the SECOND generator.

C27 showed, on GP-MoLFormer, that most of "best-of-N dominates guided decoding" was the
ground-truth oracle rather than the selection: `bestofn.selection_key` reads the TRUE RDKit
property of the finished molecule, while guidance only ever sees a learned probe.  Restricted
to the same head, the same probe point, the same interval and the same binning that guidance
steers with, the deployed arm's gap collapsed by 0.876 / 0.882 / 0.859 of its size.

That claim rests on one generator.  C33 replicates it on `entropy/gpt2_zinc_87m` -- C31's
generator -- with C31's pool, C31's frozen intervals, C31's heads and C31's seeds.

Pre-registration: `outputs/c33_prereg/C33.0_preregistration.md`, frozen with its SHA-256 in
`prereg_lock.json` before this script produced anything.

Three arms over all disjoint consecutive groups of N on a 16,384-molecule pool per seed:

    oracle_selected        bestofn.selection_key on the true property -- C31's arm verbatim,
                           present as gate G1 (it must reproduce C31 exactly).
    head_selected          argmax of the head's P(y_final in I) read at the LAST CONTENT
                           TOKEN.  No oracle information at all -- invalid candidates are NOT
                           down-ranked, because RDKit validity is oracle information.
    head_selected_at_75pct secondary/diagnostic (C33.0.2 arm 3): the same head read at content
                           position max(1, floor(3n/4)).

Reuse, not re-implementation: `score_pool` is `scripts/21_n_sweep.py`'s, `summarise` is
`bestofn.summarise`, and `content_positions` / `head_probabilities` are imported from
`scripts/22_head_selected_bestofn.py` -- C27's own functions -- so a C33/C27 difference cannot
be an implementation difference in the arms.  The generator adapter is C31's
`second_generator.load_zinc_generator`.

The pool depends only on (seed, base policy, generator), so it is drawn ONCE PER SEED and
shared across the three anchors; G1 is what proves the sharing did not alter it.  The pool and
every per-arm selection are written to disk (C33.0.2) so that no future experiment has to
regenerate them.

    .venv/bin/python scripts/27_c33_oracle_asymmetry_gen2.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import generation, metrics as M  # noqa: E402
from property_to_go.bestofn import summarise  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer  # noqa: E402
from property_to_go.second_generator import load_zinc_generator  # noqa: E402

DATASET_DIR = "c31_zinc50k"
HEADS_DIR = "c31_heads"
ANCHORS = ["aromatic_rings", "hbd_count", "qed"]
ARMS = ["oracle_selected", "head_selected", "head_selected_at_75pct"]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c31 = _load_module(ROOT / "scripts" / "25_c31_second_generator.py", "c31_second_generator")
_c27 = _load_module(ROOT / "scripts" / "22_head_selected_bestofn.py", "c27_head_selected")
score_pool = _c31.score_pool                 # scripts/21_n_sweep.py's, via C31
load_head_ckpt = _c31.load_head_ckpt         # scripts/05_guided_generation.py's, via C31
load_c31_config = _c31.load_c31_config
model_cfg_of = _c31.model_cfg_of
cell_dir = _c31.cell_dir
content_positions = _c27.content_positions   # C27's, imported not copied
head_probabilities = _c27.head_probabilities  # C27's, imported not copied
sha256_file = _c31.sha256_file


def resolve_deployed_head(prop: str, cfg: dict) -> dict:
    """Identify the head C31's DEPLOYED arm used -- from all five k cells, or refuse to guess.

    C33.0.3: every deployed k cell records `arm`, `head_input`, `head_checkpoint`,
    `head_file`, `head_seed`, `probe_point` and `layer`.  All five must agree; anything
    missing, disagreeing or not `frozen_state` stops C33 rather than being guessed at.
    """
    spec = _c31.load_arms(cfg, prop)["deployed"]
    L, lam = int(spec["probe_point"]), float(spec["lam"])
    ks = [int(k) for k in cfg["k_grid"]]
    seen: list[dict] = []
    for k in ks:
        f = cell_dir(prop, "deployed", L, lam, k) / "k_cell_metrics.json"
        if not f.exists():
            raise SystemExit(f"C33 stop: C31 deployed cell {f} not found; head unidentifiable")
        d = read_json(f)
        seen.append({
            "cell": f.parent.name,
            "arm": d.get("arm"),
            "head_input": d.get("head_input"),
            "head_checkpoint": d.get("head_checkpoint"),
            "head_file": d.get("head_file"),
            "head_seed": d.get("head_seed"),
            "probe_point": d.get("probe_point"),
            "layer": d.get("layer"),
            "target_interval": {kk: d["target_interval"][kk] for kk in ("lo", "hi", "base_rate")},
        })
    keys = ("arm", "head_input", "head_checkpoint", "head_file", "head_seed", "probe_point",
            "layer")
    first = seen[0]
    for s in seen[1:]:
        for kk in keys:
            if s[kk] != first[kk]:
                raise SystemExit(
                    f"C33 stop: C31 deployed cells disagree on {kk!r} for {prop}: "
                    f"{first['cell']}={first[kk]!r} vs {s['cell']}={s[kk]!r}")
        if s["target_interval"] != first["target_interval"]:
            raise SystemExit(f"C33 stop: C31 deployed cells disagree on the target interval "
                             f"for {prop}")
    if first["arm"] != "deployed":
        raise SystemExit(f"C33 stop: cell records arm={first['arm']!r}, expected 'deployed'")
    if first["head_input"] != "frozen_state":
        raise SystemExit(f"C33 stop: cell records head_input={first['head_input']!r}")
    if first["layer"] != first["probe_point"]:
        raise SystemExit(f"C33 stop: layer {first['layer']} != probe_point "
                         f"{first['probe_point']}; the probe point is ambiguous")
    path = Path(first["head_file"])
    if not path.exists():
        raise SystemExit(f"C33 stop: head checkpoint {path} does not exist")
    expected = _c31.head_path(prop, int(first["probe_point"]), int(first["head_seed"]))
    if path.resolve() != expected.resolve():
        raise SystemExit(f"C33 stop: recorded head_file {path} != the C31 naming convention "
                         f"{expected}")
    return {
        "resolved_from": [s["cell"] for s in seen],
        "n_cells_agreeing": len(seen),
        "head_checkpoint_name": first["head_checkpoint"],
        "head_file": str(path),
        "head_input": first["head_input"],
        "head_seed": int(first["head_seed"]),
        "probe_point": int(first["probe_point"]),
        "layer": int(first["layer"]),
        "deployed_lambda": lam,
        "target_interval_in_c31_cell": first["target_interval"],
    }


def parameter_sha256(state_dict) -> str:
    """Order-stable hash over parameter CONTENT.

    C27's gate 4 pre-registered a FILE hash and it failed by construction: `torch.save`
    names the zip archive after the output file, so two saves of one dict under two names
    differ in bytes while holding identical tensors.  C33.0.4 G2 therefore hashes the
    tensors, and reports the file hash as evidence rather than as a criterion.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--properties", nargs="*", default=None, choices=ANCHORS)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--grid", type=int, nargs="*", default=None)
    ap.add_argument("--n-molecules", type=int, default=None)
    ap.add_argument("--state-batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_c31_config()
    data_dir = OUTPUT_DIR / DATASET_DIR
    intervals_f = data_dir / "target_intervals.json"
    windows_f = data_dir / "windows.json"
    intervals = read_json(intervals_f)
    props = args.properties or list(ANCHORS)
    seeds = [int(s) for s in (args.seeds or cfg["generation_seeds"])]
    grid = sorted(set(args.grid or [int(n) for n in cfg["best_of_n_grid"]]))
    n_max = int(cfg["best_of_n_pool_depth"])
    n_mol = args.n_molecules or int(cfg["best_of_n_molecules"])
    if max(grid) > n_max:
        raise SystemExit(f"grid max {max(grid)} exceeds pool depth {n_max}")
    policy = dict(cfg["base_policy"])

    pool_dir = OUTPUT_DIR / "c33_pool"
    sel_dir = OUTPUT_DIR / "c33_selections"
    pool_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    # ---- G6: the frozen interval is C31's, and it is hashed rather than trusted ---------
    frozen = {
        "target_intervals_file": str(intervals_f.relative_to(ROOT)),
        "target_intervals_sha256": sha256_file(intervals_f),
        "windows_file": str(windows_f.relative_to(ROOT)),
        "windows_sha256": sha256_file(windows_f),
        "windows": read_json(windows_f),
    }

    # ---- the heads, resolved from C31's deployed cells rather than assumed -------------
    heads: dict[str, dict] = {}
    for prop in props:
        dep = resolve_deployed_head(prop, cfg)
        head, binner, ck = load_head_ckpt(Path(dep["head_file"]))
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        c31_iv = dep["target_interval_in_c31_cell"]
        interval_matches = (abs(c31_iv["lo"] - lo) == 0.0 and abs(c31_iv["hi"] - hi) == 0.0
                            and abs(c31_iv["base_rate"] - float(iv["base_rate"])) == 0.0)
        prov = {
            **dep,
            "head_file_sha256": sha256_file(Path(dep["head_file"])),
            "head_parameter_sha256": parameter_sha256(ck["state_dict"]),
            "head_file_bytes": Path(dep["head_file"]).stat().st_size,
            "checkpoint_property": ck.get("property"),
            "checkpoint_input": ck.get("input"),
            "checkpoint_head_seed": ck.get("head_seed"),
            "in_dim": int(ck["in_dim"]),
            "hidden_dim": int(ck["hidden_dim"]),
            "n_bins": int(ck["n_bins"]),
            "dropout": float(ck["dropout"]),
            "binner_kind": ck["binner"]["kind"],
            "interval_mask_n_bins_selected": int(np.asarray(binner.interval_mask(lo, hi)).sum()),
            "target_interval_used": {"lo": lo, "hi": hi, "base_rate": float(iv["base_rate"])},
            "target_interval_matches_c31_cell": bool(interval_matches),
            "metadata_consistent": bool(
                ck.get("property") == prop and ck.get("input") == "frozen_state"
                and int(ck.get("head_seed", -1)) == dep["head_seed"]),
        }
        # G2's cross-check against C31's own depth sweep.  The layout is
        # `by_probe_point[str(probe_point)]` with a `per_seed` list, one entry per head seed;
        # the held-out target AUROC lives at `test.intervals.target.auroc`.
        depth_f = OUTPUT_DIR / HEADS_DIR / f"depth_{prop}.json"
        if depth_f.exists():
            d = read_json(depth_f)
            cell = (d.get("by_probe_point") or {}).get(str(dep["probe_point"]))
            if cell is not None:
                prov["c31_depth_test_target_auroc_mean_over_head_seeds"] = cell.get(
                    "test_target_auroc_mean")
                for row in cell.get("per_seed", []) or []:
                    if int(row.get("head_seed", -1)) == dep["head_seed"]:
                        prov["c31_depth_test_target_auroc"] = (
                            row.get("test", {}).get("intervals", {})
                            .get("target", {}).get("auroc"))
                        prov["c31_depth_head_input_dim"] = row.get("input_dim")
        heads[prop] = {"prov": prov, "head": head, "binner": binner, "ck": ck,
                       "lo": lo, "hi": hi, "iv": iv}
        if not prov["metadata_consistent"]:
            raise SystemExit(f"C33 stop: checkpoint metadata for {prop} is inconsistent with "
                             f"the C31 deployed cell: {prov}")
        if not interval_matches:
            raise SystemExit(f"C33 stop: G6 -- the frozen interval for {prop} does not match "
                             f"the one recorded in C31's deployed cell")

    # One state pass serves every anchor below, which is only sound if every anchor's
    # deployed head reads the SAME probe point.  On C31 they all read 12; the guard exists
    # so that a future config where they do not stops C33 rather than mis-scoring silently.
    probe_points = sorted({heads[p]["prov"]["probe_point"] for p in props})
    if len(probe_points) != 1:
        raise SystemExit(f"C33 stop: deployed probe points disagree across anchors "
                         f"{probe_points}; the shared state pass would mis-score")

    gen = load_zinc_generator(model_cfg_of(cfg))
    fingerprint = gen.fingerprint()
    scorers = {p: TargetScorer(heads[p]["head"], heads[p]["binner"],
                               heads[p]["lo"], heads[p]["hi"]).to(gen.device)
               for p in props}
    for p in props:
        h = heads[p]["prov"]
        print(f"{p}: target=[{heads[p]['lo']:.4f},{heads[p]['hi']:.4f}) "
              f"base_rate={heads[p]['iv']['base_rate']:.4f} head={h['head_checkpoint_name']} "
              f"probe_point={h['probe_point']} n_bins={h['n_bins']} "
              f"mask_bins={h['interval_mask_n_bins_selected']}", flush=True)

    t0 = time.time()
    per_seed: dict[str, dict[str, dict]] = {p: {} for p in props}
    for seed in seeds:
        pool_seed = seed * 1000  # scripts/06_best_of_n.py's convention; C26/C27/C31's gate
        meter = ComputeMeter().start()
        seqs = generation.sample_unconditional(gen, policy, n_max * n_mol,
                                               seed=pool_seed, meter=meter)
        meter.stop()
        smiles = gen.decode(seqs)
        pool_compute = meter.as_dict()
        print(f"[C33] seed={seed} pool={len(seqs)} "
              f"tokens={pool_compute['processed_tokens_actual']}", flush=True)

        # One state pass serves every anchor: the probe point is the same and the pool is
        # the same, so only the head differs.  The recompute cost is charged once and
        # reported per anchor as the same number, which is what a deployed sampler that
        # scored three properties from one forward pass would pay.
        head_meter = ComputeMeter().start()
        term, p75, n_content = content_positions(seqs, gen.bos_id, gen.eos_id, gen.pad_id)
        positions = [[t, q] for t, q in zip(term, p75)]
        states = generation.hidden_states_for_positions(
            gen, seqs, positions, layer=heads[props[0]]["prov"]["probe_point"],
            batch_size=args.state_batch_size, meter=head_meter)
        head_meter.stop()
        arr = np.stack(states, axis=0)  # (pool, 2, hidden)
        flat = torch.as_tensor(arr.reshape(-1, arr.shape[-1]), dtype=torch.float32)
        probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for prop in props:
            out = []
            with torch.no_grad():
                for i in range(0, len(flat), 8192):
                    out.append(scorers[prop](flat[i:i + 8192].to(gen.device))
                               .float().cpu().numpy())
            pr = np.concatenate(out).reshape(len(seqs), 2).astype(np.float64)
            probs[prop] = (pr[:, 0], pr[:, 1])
        del states, arr, flat

        pool_payload = {
            "token_counts": np.asarray([len(s) for s in seqs], dtype=np.int32),
            "n_content": np.asarray(n_content, dtype=np.int32),
            "terminal_position": np.asarray(term, dtype=np.int32),
            "p75_position": np.asarray(p75, dtype=np.int32),
            "sequence_offsets": np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64),
            "sequence_ids": np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs]),
        }

        for prop in props:
            lo, hi = heads[prop]["lo"], heads[prop]["hi"]
            keys, cands, tokens = score_pool(smiles, seqs, prop, lo, hi)
            p_term, p_75 = probs[prop]
            hit = np.array([bool(c.get("valid") and c.get(prop) is not None
                                 and lo <= c[prop] < hi) for c in cands])
            auroc_term = M.auroc(p_term, hit)
            auroc_75 = M.auroc(p_75, hit)

            pool = len(keys)
            rows: dict[str, dict] = {arm: {} for arm in ARMS}
            selections: dict[str, np.ndarray] = {}
            for n in grid:
                n_groups = pool // n
                tok = 0
                picks = {arm: [] for arm in ARMS}
                for i in range(n_groups):
                    block = range(i * n, i * n + n)
                    # oracle: C26/C31's rule verbatim -- lowest (validity, membership, error).
                    picks["oracle_selected"].append(min(block, key=lambda j: keys[j]))
                    # head arms: highest head probability, ties to the lowest index.  No
                    # oracle information -- invalid candidates are NOT down-ranked (C33.0.2).
                    picks["head_selected"].append(max(block, key=lambda j: (p_term[j], -j)))
                    picks["head_selected_at_75pct"].append(
                        max(block, key=lambda j: (p_75[j], -j)))
                    tok += sum(tokens[j] for j in block)
                for arm in ARMS:
                    sel = picks[arm]
                    selections[f"{arm}_N{n}"] = np.asarray(sel, dtype=np.int32)
                    s = summarise([cands[j] for j in sel], prop, lo, hi)
                    s["compute"] = {
                        "processed_tokens_actual": int(tok),
                        "molecules_returned": n_groups,
                        "tokens_per_molecule_actual": tok / n_groups,
                    }
                    s["n_groups"] = n_groups
                    s["agreement_with_oracle_selection"] = float(
                        np.mean([a == b for a, b in zip(sel, picks["oracle_selected"])]))
                    rows[arm][str(n)] = s
                print(f"  {prop} seed={seed} N={n:>2} "
                      f"oracle={rows['oracle_selected'][str(n)]['hit_rate']:.4f} "
                      f"head={rows['head_selected'][str(n)]['hit_rate']:.4f} "
                      f"head75={rows['head_selected_at_75pct'][str(n)]['hit_rate']:.4f} "
                      f"tok/mol={tok / n_groups:.1f}", flush=True)

            np.savez_compressed(sel_dir / f"{prop}_seed{seed}.npz", **selections)
            hm_ = head_meter.as_dict()
            per_seed[prop][str(seed)] = {
                "pool_seed": pool_seed,
                "pool_size": pool,
                "arms": rows,
                "pool_compute": pool_compute,
                "head_scoring_recompute_compute": hm_,
                "head_scoring_recompute_tokens_per_pool_molecule":
                    hm_["processed_tokens_actual"] / pool,
                "head_auroc_terminal_position": float(auroc_term),
                "head_auroc_75pct_position": float(auroc_75),
                "pool_true_hit_rate": float(hit.mean()),
                "head_prob_terminal_mean": float(p_term.mean()),
                "head_prob_75pct_mean": float(p_75.mean()),
                "content_length_mean": float(np.mean(n_content)),
            }
            print(f"  {prop} seed={seed} AUROC terminal={auroc_term:.4f} 75%={auroc_75:.4f} "
                  f"pool_hit={hit.mean():.4f}", flush=True)

            pool_payload[f"p_terminal_{prop}"] = p_term.astype(np.float32)
            pool_payload[f"p_75pct_{prop}"] = p_75.astype(np.float32)
            pool_payload[f"value_{prop}"] = np.asarray(
                [c.get(prop) if (c.get("valid") and c.get(prop) is not None) else np.nan
                 for c in cands], dtype=np.float64)
            pool_payload[f"valid_{prop}"] = np.asarray(
                [bool(c.get("valid")) for c in cands])
            pool_payload[f"true_hit_{prop}"] = hit

        np.savez_compressed(pool_dir / f"pool_seed{seed}.npz", **pool_payload)
        write_json(pool_dir / f"pool_seed{seed}_smiles.json",
                   {"seed": seed, "pool_seed": pool_seed, "smiles": smiles})

    # ------------------------------------------------------------------ per-anchor report
    for prop in props:
        out_dir = OUTPUT_DIR / f"c33_headsel_{prop}"
        out_dir.mkdir(parents=True, exist_ok=True)
        curves: dict[str, dict] = {}
        for arm in ARMS:
            c = {}
            for n in grid:
                r = [per_seed[prop][str(s)]["arms"][arm][str(n)] for s in seeds]
                hits = [x["hit_rate"] for x in r]
                c[str(n)] = {
                    "n_candidates": n,
                    "hit_rate_mean": float(np.mean(hits)),
                    "hit_rate_values": [float(h) for h in hits],
                    "hit_rate_sd": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
                    "hit_rate_over_all_returned_mean": float(np.mean(
                        [x["hit_rate_over_all_returned"] for x in r])),
                    "tokens_per_molecule_actual": float(np.mean(
                        [x["compute"]["tokens_per_molecule_actual"] for x in r])),
                    "validity_mean": float(np.mean([x["validity"] for x in r])),
                    "uniqueness_mean": float(np.mean([x["uniqueness"] for x in r])),
                    "agreement_with_oracle_selection_mean": float(np.mean(
                        [x["agreement_with_oracle_selection"] for x in r])),
                }
            curves[arm] = c
        rec = float(np.mean([per_seed[prop][str(s)]
                             ["head_scoring_recompute_tokens_per_pool_molecule"]
                             for s in seeds]))
        report = {
            "experiment": "C33",
            "prereg": "outputs/c33_prereg/C33.0_preregistration.md",
            "generator": {"repo": cfg["model_repo"], "revision": cfg["model_revision"],
                          "fingerprint": fingerprint},
            "dataset": DATASET_DIR,
            "property": prop,
            "target_interval": heads[prop]["iv"],
            "frozen_inputs": frozen,
            "arms": ARMS,
            "n_max": n_max,
            "n_molecules_per_seed": n_mol,
            "seeds": seeds,
            "grid": grid,
            "accounting": "actual",
            "head": heads[prop]["prov"],
            "head_scoring_token_charge": 0,
            "head_scoring_token_charge_rationale": (
                "the head reads the final-layer state at a position the generator already "
                "computed -- the same state the LM head reads to emit the next token's "
                "logits -- so a deployed head-selecting sampler pays no extra generator "
                "tokens.  This implementation recomputes them because transformers."
                "generate() does not expose hidden states; that recompute cost is measured "
                "below and is an artefact of the implementation, not of the method "
                "(C33.0.3, sensitivity S1)."),
            "head_scoring_recompute_tokens_per_pool_molecule_mean": rec,
            "grouping_note": ("all disjoint consecutive groups of N over the whole pool, "
                              "C26's corrected estimator"),
            "head_score_position": (
                "terminal content token (full-sequence index n) for `head_selected`; "
                "max(1, floor(3n/4)) for `head_selected_at_75pct`"),
            "pool_artefacts": {
                "pool": [f"outputs/c33_pool/pool_seed{s}.npz" for s in seeds],
                "smiles": [f"outputs/c33_pool/pool_seed{s}_smiles.json" for s in seeds],
                "selections": [f"outputs/c33_selections/{prop}_seed{s}.npz" for s in seeds],
                "note": ("written so that no future experiment has to regenerate the pool, "
                         "which is why C33 had to regenerate it in the first place"),
            },
            "curves": curves,
            "per_seed": per_seed[prop],
            "wall_seconds_total": time.time() - t0,
        }
        write_json(out_dir / "head_selected_metrics.json", report)
        write_run_context(out_dir, {"c31_config": cfg, "cli": vars(args),
                                    "head": heads[prop]["prov"], "frozen_inputs": frozen})
        print(f"[C33] -> {out_dir.name}", flush=True)
    write_run_context(pool_dir, {"c31_config": cfg, "cli": vars(args),
                                 "frozen_inputs": frozen})
    write_run_context(sel_dir, {"c31_config": cfg, "cli": vars(args)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
