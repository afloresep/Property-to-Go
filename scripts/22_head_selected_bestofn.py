"""C27 -- the head-selected best-of-N control.

Every headline in this project compares guided decoding against best-of-N at matched
processed tokens, and guided decoding loses.  That comparison is scoped badly:
`src/property_to_go/bestofn.py::selection_key` is evaluated on the **true RDKit property
of the completed molecule**, so best-of-N selects with the ground-truth oracle while
guidance only ever sees a learned probe.  "Best-of-N dominates" is then partly a
restatement of "ground truth beats an estimate of it at equal token cost".

C27 removes the asymmetry: best-of-N selects with **the same head the deployed lambda=1
guided run steers with**, and both arms are evaluated by the same true RDKit hit rate.
Selection differs; evaluation does not.

Pre-registration: `outputs/c27_prereg/C27.0_preregistration.md`, frozen (with its SHA-256
in `prereg_lock.json`) before this script produced anything.

Three arms over the same disjoint consecutive groups of N, on C26's pool and C26's grid:

    oracle_selected        bestofn.selection_key on the true property -- C26's arm verbatim,
                           present as validity gate 1 (it must reproduce C26 exactly).
    head_selected          argmax of the head's P(y_final in I) read at the LAST CONTENT
                           TOKEN of the completed molecule.  No oracle information at all --
                           in particular invalid candidates are NOT down-ranked, because
                           RDKit validity is oracle information.
    head_selected_at_75pct secondary/diagnostic, pre-registered in C27.0.2 arm 3: the same
                           head read at content position max(1, floor(3n/4)).  Bounds how
                           much of `head_selected` needs the molecule to be finished.

Why the terminal position: `prefixes.select_quartile_prefixes` draws one prefix per quartile
of [1, n] and quartile 4's upper bound is exactly n, so the state at the last content token
is the `relative_position = 1.0` endpoint of the head's own training distribution rather than
an extrapolation off it.  <eos> is deliberately not used -- no training prefix ended there.

Token accounting (`actual`, C26's): all arms share one pool, so generation cost is identical
and must match C26's 44.4 tokens/molecule at N=1 up to 1422.0 at N=32.  Head scoring is
charged **zero generator tokens**: `hidden_states[-1]` at position t is exactly what the LM
head reads to emit the logits for t+1, so it is already materialised by the forward pass that
generated the molecule.  This implementation nevertheless recomputes those states in one
extra pass, because `transformers.generate()` does not expose them; that cost is measured and
published as `head_scoring_recompute_tokens_per_molecule` so the pessimistic accounting
(pre-registered sensitivity S1) can be priced by a reader who rejects the argument.

    .venv/bin/python scripts/22_head_selected_bestofn.py --dataset pilot_50k_p2 \
        --property aromatic_rings
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
from property_to_go.binning import binner_from_dict  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.bestofn import summarise  # noqa: E402
from property_to_go.guidance import TargetScorer  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.generation import sample_unconditional  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402

DEFAULT_GRID = [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]
ARMS = ["oracle_selected", "head_selected", "head_selected_at_75pct"]


def _load_module(path: Path, name: str):
    """Import a script whose filename starts with a digit.

    `score_pool` is imported from `scripts/21_n_sweep.py` rather than copied, so validity
    gate 1 (the oracle arm reproducing C26 exactly) cannot be passed by a re-implementation
    that merely agrees today.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_nsweep = _load_module(ROOT / "scripts" / "21_n_sweep.py", "c26_n_sweep")
score_pool = _nsweep.score_pool


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_head(path: Path):
    """Exactly `scripts/05_guided_generation.py::load_head`, so the head is the deployed one."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"]), ck


def resolve_deployed_head(prop: str, dataset: str, heads_dir: Path) -> dict:
    """Identify the head the deployed lambda=1 guided run used -- or refuse to guess.

    C27.0.3: the deployed artefact records `head_input` and `head_checkpoint` and carries no
    `layer` / `head_file` key, because it predates the C23 edit that added them.  Its probe
    point is therefore `05_guided_generation.py`'s default, -1.  Anything ambiguous raises.
    """
    g = OUTPUT_DIR / f"{dataset}_guided_{prop}" / "guidance_metrics.json"
    if not g.exists():
        raise SystemExit(f"C27 stop: deployed guided run {g} not found; head is unidentifiable")
    r = read_json(g)
    if r.get("head_input") != "frozen_state":
        raise SystemExit(f"C27 stop: deployed run records head_input={r.get('head_input')!r}")
    ckpt_name = r.get("head_checkpoint")
    if not ckpt_name:
        raise SystemExit("C27 stop: deployed run records no head_checkpoint")
    if r.get("head_file") is not None:
        # A later run that used --head-file records the absolute path; honour it rather
        # than reconstructing one.
        path = Path(r["head_file"])
    else:
        path = heads_dir / ckpt_name
    if not path.exists():
        raise SystemExit(f"C27 stop: deployed head checkpoint {path} does not exist")
    layer = int(r.get("layer", -1))
    if "layer" not in r and "layer_source" in r:
        raise SystemExit("C27 stop: deployed run records layer_source without layer")
    return {
        "deployed_run": str(g.parent),
        "head_checkpoint_name": ckpt_name,
        "head_file": str(path),
        "head_input": r["head_input"],
        "layer": layer,
        "layer_source": ("deployed artefact" if "layer" in r
                         else "05_guided_generation.py default (-1); the deployed artefact "
                              "predates the C23 --layer flag and records no layer key"),
        "deployed_lambda": float(r.get("lambda", 1.0)),
        "deployed_hit_rate": float(r["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"]),
        "deployed_tokens_per_molecule_actual": float(
            r["conditions"]["throughout"]["aggregate"]["compute_total"]
            ["tokens_per_molecule_actual"]),
    }


def content_positions(seqs: list[list[int]], bos: int, eos: int, pad: int):
    """(terminal, 75%) full-sequence indices to read the head at, per molecule.

    A sequence is [<bos>, c_1..c_n, <eos>], so content token c_k sits at index k and the
    terminal content position is n.  Empty content falls back to index 0 (<bos>), which is
    the only defensible thing to read when there is no molecule.
    """
    term, p75, n_content = [], [], []
    for s in seqs:
        n = len(generation.sequence_content(s, bos, eos, pad))
        n = min(n, len(s) - 1)  # never index past the sequence
        n_content.append(n)
        term.append(max(n, 0))
        p75.append(max(1, (3 * n) // 4) if n >= 1 else 0)
    return term, p75, n_content


def head_probabilities(gen, seqs, scorer: TargetScorer, layer: int, batch_size: int,
                       meter: ComputeMeter | None):
    """P(y_final in I) at the terminal and the 75% content position of every sequence."""
    term, p75, n_content = content_positions(seqs, gen.bos_id, gen.eos_id, gen.pad_id)
    positions = [[t, q] for t, q in zip(term, p75)]
    states = generation.hidden_states_for_positions(
        gen, seqs, positions, layer=layer, batch_size=batch_size, meter=meter)
    arr = np.stack(states, axis=0)  # (n, 2, hidden)
    flat = torch.as_tensor(arr.reshape(-1, arr.shape[-1]), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), 8192):
            out.append(scorer(flat[i:i + 8192].to(gen.device)).float().cpu().numpy())
    probs = np.concatenate(out).reshape(len(seqs), 2)
    return probs[:, 0].astype(np.float64), probs[:, 1].astype(np.float64), n_content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--heads", default="pilot_50k_heads_p2")
    ap.add_argument("--property", default="aromatic_rings")
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--n-molecules", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--grid", type=int, nargs="*", default=None)
    ap.add_argument("--state-batch-size", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / args.heads
    out_dir = OUTPUT_DIR / (args.out or f"c27_headsel_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    intervals = read_json(data_dir / "target_intervals.json")
    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    seeds = args.seeds or list(gcfg["seeds"])
    n_mol = args.n_molecules
    n_max = args.n_max
    grid = sorted(set(args.grid or DEFAULT_GRID))
    if max(grid) > n_max:
        raise SystemExit(f"grid max {max(grid)} exceeds pool depth {n_max}")

    # ---- the head, identified from the deployed artefact rather than chosen -----------
    dep = resolve_deployed_head(args.property, args.dataset, heads_dir)
    head_path = Path(dep["head_file"])
    head, binner, ck = load_head(head_path)
    scorer = TargetScorer(head, binner, lo, hi)
    seed_twin = heads_dir / f"head_{args.property}_frozen_state_seed{ck.get('head_seed')}.pt"
    provenance_gate = {
        **dep,
        "head_sha256": sha256_file(head_path),
        "head_seed_twin": str(seed_twin),
        "head_seed_twin_sha256": sha256_file(seed_twin) if seed_twin.exists() else None,
        "head_seed": ck.get("head_seed"),
        "head_input_in_checkpoint": ck.get("input"),
        "n_bins": int(ck["n_bins"]),
        "in_dim": int(ck["in_dim"]),
        "hidden_dim": int(ck["hidden_dim"]),
        "binner_kind": ck["binner"]["kind"],
        "interval_mask_n_bins_selected": int(np.asarray(binner.interval_mask(lo, hi)).sum()),
    }
    provenance_gate["sha256_matches_seed_twin"] = (
        provenance_gate["head_sha256"] == provenance_gate["head_seed_twin_sha256"])
    hm = heads_dir / "head_metrics.json"
    if hm.exists():
        h = read_json(hm)["properties"][args.property]["heads"]["frozen_state"]
        provenance_gate["heads_json_head_seed"] = h["head_seed"]
        provenance_gate["heads_json_test_target_auroc"] = float(
            h["test"]["intervals"]["target"]["auroc"])

    gen = load_generator(model_cfg)
    scorer.to(gen.device)
    print(f"property={args.property} target=[{lo:.4f},{hi:.4f}) base_rate={iv['base_rate']:.4f}")
    print(f"head={head_path.name} seed={ck.get('head_seed')} layer={dep['layer']} "
          f"n_bins={ck['n_bins']} mask_bins={provenance_gate['interval_mask_n_bins_selected']}")

    t0 = time.time()
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        pool_seed = seed * 1000  # scripts/06_best_of_n.py's convention; C26 gate 1
        meter = ComputeMeter().start()
        seqs = sample_unconditional(gen, policy, n_max * n_mol, seed=pool_seed, meter=meter)
        meter.stop()
        smiles = gen.decode(seqs)
        keys, cands, tokens = score_pool(smiles, seqs, args.property, lo, hi)

        head_meter = ComputeMeter().start()
        p_term, p_75, n_content = head_probabilities(
            gen, seqs, scorer, dep["layer"], args.state_batch_size, head_meter)
        head_meter.stop()

        # gate 2: does the head's score discriminate true hits in the pool at all?
        hit = np.array([bool(c.get("valid") and c.get(args.property) is not None
                             and lo <= c[args.property] < hi) for c in cands])
        auroc_term = M.auroc(p_term, hit)
        auroc_75 = M.auroc(p_75, hit)

        pool = len(keys)
        rows: dict[str, dict] = {arm: {} for arm in ARMS}
        for n in grid:
            n_groups = pool // n
            tok = 0
            picks = {arm: [] for arm in ARMS}
            for i in range(n_groups):
                block = range(i * n, i * n + n)
                # oracle: C26's rule verbatim -- lowest (validity, membership, distance) key.
                picks["oracle_selected"].append(min(block, key=lambda j: keys[j]))
                # head arms: highest head probability, ties to the lowest index.  No oracle
                # information -- invalid candidates are NOT down-ranked (C27.0.2).
                picks["head_selected"].append(max(block, key=lambda j: (p_term[j], -j)))
                picks["head_selected_at_75pct"].append(max(block, key=lambda j: (p_75[j], -j)))
                tok += sum(tokens[j] for j in block)
            for arm in ARMS:
                sel = picks[arm]
                s = summarise([cands[j] for j in sel], args.property, lo, hi)
                s["compute"] = {
                    "processed_tokens_actual": int(tok),
                    "molecules_returned": n_groups,
                    "tokens_per_molecule_actual": tok / n_groups,
                }
                s["n_groups"] = n_groups
                s["agreement_with_oracle_selection"] = float(
                    np.mean([a == b for a, b in zip(sel, picks["oracle_selected"])]))
                if n_groups >= n_mol:
                    s["first_512_groups_hit_rate"] = float(summarise(
                        [cands[j] for j in sel[:n_mol]], args.property, lo, hi)["hit_rate"])
                rows[arm][str(n)] = s
            print(f"  seed={seed} N={n:>2} oracle={rows['oracle_selected'][str(n)]['hit_rate']:.4f} "
                  f"head={rows['head_selected'][str(n)]['hit_rate']:.4f} "
                  f"head75={rows['head_selected_at_75pct'][str(n)]['hit_rate']:.4f} "
                  f"tok/mol={tok / n_groups:.1f}")

        hm_ = head_meter.as_dict()
        per_seed[str(seed)] = {
            "pool_seed": pool_seed,
            "pool_size": pool,
            "arms": rows,
            "pool_compute": meter.as_dict(),
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
        print(f"  seed={seed} AUROC terminal={auroc_term:.4f} 75%={auroc_75:.4f} "
              f"pool_hit={hit.mean():.4f}")

    curves: dict[str, dict] = {}
    for arm in ARMS:
        c = {}
        for n in grid:
            hits = [per_seed[str(s)]["arms"][arm][str(n)]["hit_rate"] for s in seeds]
            toks = [per_seed[str(s)]["arms"][arm][str(n)]["compute"]
                    ["tokens_per_molecule_actual"] for s in seeds]
            vals = [per_seed[str(s)]["arms"][arm][str(n)]["validity"] for s in seeds]
            uniq = [per_seed[str(s)]["arms"][arm][str(n)]["uniqueness"] for s in seeds]
            allr = [per_seed[str(s)]["arms"][arm][str(n)]["hit_rate_over_all_returned"]
                    for s in seeds]
            agr = [per_seed[str(s)]["arms"][arm][str(n)]["agreement_with_oracle_selection"]
                   for s in seeds]
            c[str(n)] = {
                "n_candidates": n,
                "hit_rate_mean": float(np.mean(hits)),
                "hit_rate_values": [float(h) for h in hits],
                "hit_rate_sd": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
                "hit_rate_over_all_returned_mean": float(np.mean(allr)),
                "tokens_per_molecule_actual": float(np.mean(toks)),
                "validity_mean": float(np.mean(vals)),
                "uniqueness_mean": float(np.mean(uniq)),
                "agreement_with_oracle_selection_mean": float(np.mean(agr)),
            }
        curves[arm] = c

    rec = float(np.mean([per_seed[str(s)]["head_scoring_recompute_tokens_per_pool_molecule"]
                         for s in seeds]))
    report = {
        "experiment": "C27",
        "prereg": "outputs/c27_prereg/C27.0_preregistration.md",
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": iv,
        "arms": ARMS,
        "n_max": n_max,
        "n_molecules_per_seed": n_mol,
        "seeds": seeds,
        "grid": grid,
        "accounting": "actual",
        "head": provenance_gate,
        "head_scoring_token_charge": 0,
        "head_scoring_token_charge_rationale": (
            "the head reads hidden_states[-1] at a position the generator already computed "
            "-- the same state the LM head reads to emit the next token's logits -- so a "
            "deployed head-selecting sampler pays no extra generator tokens.  This "
            "implementation recomputes them because transformers.generate() does not expose "
            "hidden states; that recompute cost is measured below and is an artefact of the "
            "implementation, not of the method (C27.0.4, sensitivity S1)."),
        "head_scoring_recompute_tokens_per_pool_molecule_mean": rec,
        "grouping_note": (
            "all disjoint consecutive groups of N over the whole pool, which is C26's "
            "corrected estimator; the first `n_molecules` groups are exactly "
            "scripts/06_best_of_n.py's"),
        "head_score_position": (
            "terminal content token (full-sequence index n) for `head_selected`; "
            "max(1, floor(3n/4)) for `head_selected_at_75pct`"),
        "curves": curves,
        "per_seed": per_seed,
        "wall_seconds_total": time.time() - t0,
    }
    write_json(out_dir / "head_selected_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                                "cli": vars(args), "head": provenance_gate})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
