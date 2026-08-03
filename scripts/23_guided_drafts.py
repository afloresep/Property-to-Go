"""C28 -- the composition nobody in this project has run: N GUIDED drafts, then rerank.

Best-of-N draws N base-policy molecules and keeps one.  Guided decoding commits at every
token and returns one molecule.  The obvious composition -- draw N **guided** drafts and
rerank them -- has never been run here, and it is the experiment a hostile reviewer will
demand, because it hands guidance a continuous compute axis immediately.

Two rerank rules, both taken verbatim from existing code rather than re-invented:

    oracle_reranked  `bestofn.selection_key` on the true RDKit property -- C26's rule.
    head_reranked    argmax of the DEPLOYED head's P(y_final in I) read at the terminal
                     content token -- C27's rule, with no oracle information at all
                     (invalid drafts are NOT down-ranked, because RDKit validity is oracle
                     information).

Pre-registration: `outputs/c28_prereg/C28.0_preregistration.md` (strand B, gates G5-G7,
decision rule D4).

Token accounting (`actual`): a draft whose full sequence is [<bos>, c_1..c_n, <eos>] is
active for len(seq) - 1 decoding steps and the cached backend charges (k + 1) tokens per
active step, so the draft costs `(k + 1) * (len(seq) - 1)`.  Gate G7 asserts that these
per-draft charges sum exactly to the meter's own total.  Head scoring is charged zero
generator tokens for C27's reason, and the recompute this implementation pays is measured
so a reader who rejects that argument can price it.

Each seed is cached in its own `seed_<s>.json`, so a kill costs at most one seed's pool.

    .venv/bin/python scripts/23_guided_drafts.py --property hbd_count
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import metrics as M  # noqa: E402
from property_to_go.bestofn import summarise  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer, Windows, guided_sample  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402

DATASET = "pilot_50k_p2"
HEADS = "pilot_50k_heads_p2"
SEEDS = (101, 202, 303)
ARMS = ("oracle_reranked", "head_reranked")
DEFAULT_GRID = (1, 2, 4, 8)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_s05 = _load_module(ROOT / "scripts" / "05_guided_generation.py", "guided_generation_05")
_nsweep = _load_module(ROOT / "scripts" / "21_n_sweep.py", "c26_n_sweep")
_headsel = _load_module(ROOT / "scripts" / "22_head_selected_bestofn.py", "c27_headsel")
load_head = _s05.load_head
score_pool = _nsweep.score_pool
head_probabilities = _headsel.head_probabilities


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="hbd_count")
    ap.add_argument("--top-k", type=int, default=8, help="guidance top-k of every draft")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=12, help="12 == the deployed default (-1)")
    ap.add_argument("--head-file", default=None)
    ap.add_argument("--n-max", type=int, default=8, help="drafts per returned molecule")
    ap.add_argument("--n-molecules", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--grid", type=int, nargs="*", default=None)
    ap.add_argument("--state-batch-size", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prop = args.property
    k = int(args.top_k)
    lam = float(args.lam)
    layer_arg = -1 if args.layer == 12 else int(args.layer)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    grid = sorted(set(args.grid or DEFAULT_GRID))
    n_max, n_mol = int(args.n_max), int(args.n_molecules)
    if max(grid) > n_max:
        raise SystemExit(f"grid max {max(grid)} exceeds draft pool depth {n_max}")

    out_dir = OUTPUT_DIR / (args.out or f"c28_guided_drafts_{prop}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = OUTPUT_DIR / DATASET
    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    iv = intervals[prop]
    lo, hi = float(iv["lo"]), float(iv["hi"])

    head_path = (Path(args.head_file) if args.head_file
                 else OUTPUT_DIR / HEADS / f"head_{prop}_frozen_state.pt")
    if not head_path.is_absolute():
        head_path = OUTPUT_DIR / head_path
    head, binner = load_head(head_path)
    scorer = TargetScorer(head, binner, lo, hi)

    # the frozen artefact strand B's pool must reproduce at its first 512 drafts (gate G6)
    dep_dir = OUTPUT_DIR / f"{DATASET}_guided_{prop}"
    dep = read_json(dep_dir / "guidance_metrics.json")
    dep_mols = read_json(dep_dir / "molecules.json")["throughout"]

    gen = load_generator(model_cfg)
    scorer.to(gen.device)
    print(f"property={prop} target=[{lo:.4f},{hi:.4f}) base_rate={iv['base_rate']:.4f} "
          f"k={k} lam={lam} layer={layer_arg} head={head_path.name}", flush=True)

    t0 = time.time()
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        cache = out_dir / f"seed_{seed}.json"
        if cache.exists():
            per_seed[str(seed)] = read_json(cache)
            print(f"[C28] skip seed {seed} (cached)", flush=True)
            continue

        meter = ComputeMeter().start()
        seqs = guided_sample(
            gen, scorer=scorer, window_fn=windows.fn("throughout"), policy=policy,
            n_molecules=n_max * n_mol, seed=seed, top_k=k, lam=lam,
            eps=float(gcfg["eps"]), backend=gcfg["candidate_backend"],
            batch_size=int(gcfg["batch_size"]), layer=layer_arg, meter=meter,
        )
        meter.stop()
        smiles = gen.decode(seqs)
        keys, cands, _unconditional_tokens = score_pool(smiles, seqs, prop, lo, hi)

        # ---- C28.0.3 per-draft token attribution, and gate G7 -----------------------
        tokens = [(k + 1) * (len(ids) - 1) for ids in seqs]
        g7_residual = int(sum(tokens) - meter.processed_tokens_actual)

        # ---- gate G6: the first 512 drafts must be the published deployed 512 --------
        ref = dep_mols[str(seed)]
        g6_mismatch = sum(1 for i in range(min(len(ref), n_mol))
                          if smiles[i] != ref[i]["smiles"])

        head_meter = ComputeMeter().start()
        p_term, p_75, n_content = head_probabilities(
            gen, seqs, scorer, layer_arg, args.state_batch_size, head_meter)
        head_meter.stop()

        hit = np.array([bool(c.get("valid") and c.get(prop) is not None
                             and lo <= c[prop] < hi) for c in cands])
        auroc_term = float(M.auroc(p_term, hit))

        pool = len(keys)
        rows: dict[str, dict] = {arm: {} for arm in ARMS}
        for n in grid:
            n_groups = pool // n
            tok = 0
            picks = {arm: [] for arm in ARMS}
            for i in range(n_groups):
                block = range(i * n, i * n + n)
                picks["oracle_reranked"].append(min(block, key=lambda j: keys[j]))
                picks["head_reranked"].append(max(block, key=lambda j: (p_term[j], -j)))
                tok += sum(tokens[j] for j in block)
            for arm in ARMS:
                sel = picks[arm]
                s = summarise([cands[j] for j in sel], prop, lo, hi)
                s["compute"] = {"processed_tokens_actual": int(tok),
                                "molecules_returned": n_groups,
                                "tokens_per_molecule_actual": tok / n_groups}
                s["n_groups"] = n_groups
                s["agreement_with_oracle_rerank"] = float(
                    np.mean([a == b for a, b in zip(sel, picks["oracle_reranked"])]))
                rows[arm][str(n)] = s
            print(f"  seed={seed} N={n:>2} oracle={rows['oracle_reranked'][str(n)]['hit_rate']:.4f} "
                  f"head={rows['head_reranked'][str(n)]['hit_rate']:.4f} "
                  f"tok/mol={tok / n_groups:.1f}", flush=True)

        hm = head_meter.as_dict()
        entry = {
            "seed": seed, "pool_size": pool, "arms": rows,
            "pool_compute": meter.as_dict(),
            "pool_tokens_per_draft": meter.processed_tokens_actual / pool,
            "gate_G7_per_draft_token_residual": g7_residual,
            "gate_G6_first_512_smiles_mismatches": int(g6_mismatch),
            "gate_G6_reference": str(dep_dir.name),
            "head_scoring_recompute_compute": hm,
            "head_scoring_recompute_tokens_per_pool_molecule":
                hm["processed_tokens_actual"] / pool,
            "head_auroc_terminal_position_on_guided_pool": auroc_term,
            "pool_true_hit_rate": float(hit.mean()),
            "head_prob_terminal_mean": float(p_term.mean()),
            "content_length_mean": float(np.mean(n_content)),
        }
        write_json(cache, entry)
        per_seed[str(seed)] = entry
        print(f"  seed={seed} G6 mismatches={g6_mismatch} G7 residual={g7_residual} "
              f"pool_hit={hit.mean():.4f} AUROC={auroc_term:.4f}", flush=True)

    curves: dict[str, dict] = {}
    for arm in ARMS:
        c = {}
        for n in grid:
            hits = [per_seed[str(s)]["arms"][arm][str(n)]["hit_rate"] for s in seeds]
            toks = [per_seed[str(s)]["arms"][arm][str(n)]["compute"]
                    ["tokens_per_molecule_actual"] for s in seeds]
            vals = [per_seed[str(s)]["arms"][arm][str(n)]["validity"] for s in seeds]
            uniq = [per_seed[str(s)]["arms"][arm][str(n)]["uniqueness"] for s in seeds]
            agr = [per_seed[str(s)]["arms"][arm][str(n)]["agreement_with_oracle_rerank"]
                   for s in seeds]
            c[str(n)] = {
                "n_drafts": n,
                "hit_rate_mean": float(np.mean(hits)),
                "hit_rate_values": [float(h) for h in hits],
                "hit_rate_sd": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
                "tokens_per_molecule_actual": float(np.mean(toks)),
                "validity_mean": float(np.mean(vals)),
                "uniqueness_mean": float(np.mean(uniq)),
                "agreement_with_oracle_rerank_mean": float(np.mean(agr)),
            }
        curves[arm] = c

    rec = float(np.mean([per_seed[str(s)]["head_scoring_recompute_tokens_per_pool_molecule"]
                         for s in seeds]))
    report = {
        "experiment": "C28",
        "prereg": "outputs/c28_prereg/C28.0_preregistration.md",
        "strand": "B",
        "dataset": DATASET,
        "property": prop,
        "target_interval": iv,
        "arms": list(ARMS),
        "grid": grid,
        "n_max": n_max,
        "n_molecules_per_seed": n_mol,
        "seeds": list(seeds),
        "accounting": "actual",
        "top_k": k,
        "lambda": lam,
        "layer": layer_arg,
        "head_file": str(head_path),
        "head_checkpoint": head_path.name,
        "deployed_reference": {
            "run": str(dep_dir.name),
            "hit_rate": dep["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"],
            "tokens_per_molecule_actual": dep["conditions"]["throughout"]["aggregate"]
                                             ["compute_total"]["tokens_per_molecule_actual"],
        },
        "head_scoring_token_charge": 0,
        "head_scoring_token_charge_rationale": (
            "C27's rule verbatim: the head reads hidden_states[-1] at a position the "
            "generator already computed, so a deployed reranking sampler pays no extra "
            "generator tokens.  The recompute this implementation performs is measured "
            "below and priced as sensitivity S1."),
        "head_scoring_recompute_tokens_per_pool_molecule_mean": rec,
        "per_draft_token_rule": "(top_k + 1) * (len(sequence) - 1); gate G7",
        "grouping_note": ("all disjoint consecutive groups of N over the whole guided draft "
                          "pool, C26's corrected estimator"),
        "gates": {
            "G5_n1_arms_identical": {
                "max_abs_residual": max(
                    abs(per_seed[str(s)]["arms"]["head_reranked"]["1"]["hit_rate"]
                        - per_seed[str(s)]["arms"]["oracle_reranked"]["1"]["hit_rate"])
                    for s in seeds) if 1 in grid else None,
            },
            "G6_first_512_smiles_mismatches": {
                str(s): per_seed[str(s)]["gate_G6_first_512_smiles_mismatches"] for s in seeds},
            "G7_per_draft_token_residual": {
                str(s): per_seed[str(s)]["gate_G7_per_draft_token_residual"] for s in seeds},
        },
        "curves": curves,
        "per_seed": per_seed,
        "wall_seconds_total": time.time() - t0,
    }
    write_json(out_dir / "guided_drafts_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                                "cli": vars(args), "head_file": str(head_path)})
    print(f"[C28] G5={report['gates']['G5_n1_arms_identical']['max_abs_residual']} "
          f"G6={report['gates']['G6_first_512_smiles_mismatches']} "
          f"G7={report['gates']['G7_per_draft_token_residual']}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
