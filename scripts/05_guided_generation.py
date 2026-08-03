"""Phase 3 -- Property-to-Go guided decoding and the intervention-response curve.

Conditions (windows frozen in <dataset>/windows.json before any of this ran):

    unguided             base policy, full vocabulary
    throughout           guidance at every step
    early / middle/late  guidance only inside that window; base policy elsewhere
    truncation_control   top-8 restriction throughout with lambda = 0

The truncation control is what separates "the property term moved the molecule"
from "restricting to eight candidates moved the molecule".

Every condition also reports sequence length and heavy-atom count, and a
length-matched hit rate, because a controller that only makes molecules longer
would otherwise look like property control.

    python scripts/05_guided_generation.py --dataset pilot_50k --property clogp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generation  # noqa: E402
from property_to_go.bestofn import INTEGER_PROPERTIES, target_error  # noqa: E402
from property_to_go.binning import binner_from_dict  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer, Windows, guided_sample  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import (  # noqa: E402
    ALL_PROPERTIES, compute_all_properties, uniqueness, validity,
)

LENGTH_BIN = 5


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"])


def summarise(records: list[dict], prop: str, lo: float, hi: float) -> dict:
    n = len(records)
    valid = [r for r in records if r["valid"]]
    # A parseable molecule can still lack one descriptor (QED alone can raise), so
    # `validity` keeps meaning "RDKit accepted it" and the property statistics are
    # computed over the subset that actually has a value.
    scored = [r for r in valid if r.get(prop) is not None]
    if not scored:
        return {"n": n, "validity": len(valid) / n if n else 0.0,
                "n_valid": len(valid), "n_scored": 0}
    vals = np.array([r[prop] for r in scored])
    lens = np.array([r["n_content_tokens"] for r in scored])
    atoms = np.array([r["n_heavy_atoms"] for r in scored])
    hit = (vals >= lo) & (vals < hi)
    dist = np.array([target_error(v, lo, hi, prop in INTEGER_PROPERTIES) for v in vals])
    canon = [r["canonical_smiles"] for r in valid]
    return {
        "n": n,
        "n_valid": len(valid),
        "n_scored": len(scored),
        "validity": len(valid) / n,
        "uniqueness": len(set(canon)) / len(canon),
        "hit_rate": float(hit.mean()),
        "hit_rate_over_all_returned": float(hit.sum() / n),
        "abs_target_error_mean": float(dist.mean()),
        "abs_target_error_median": float(np.median(dist)),
        "property_mean": float(vals.mean()),
        "property_std": float(vals.std()),
        "property_quantiles": {
            str(q): float(np.quantile(vals, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "content_length_mean": float(lens.mean()),
        "content_length_std": float(lens.std()),
        "n_heavy_atoms_mean": float(atoms.mean()),
        "n_heavy_atoms_std": float(atoms.std()),
    }


def length_matched_hit_rate(records: list[dict], reference: list[dict], prop: str,
                            lo: float, hi: float) -> dict:
    """Hit rate this condition would show under the reference length distribution.

    Molecules are binned by content length; the per-bin hit rate is reweighted by
    the reference (unguided) bin frequencies.  If a condition's advantage survives
    this reweighting it is not merely a length effect.
    """
    def binned(rs):
        out: dict[int, list[float]] = {}
        for r in rs:
            if r["valid"] and r.get(prop) is not None:
                out.setdefault(r["n_content_tokens"] // LENGTH_BIN, []).append(r[prop])
        return out

    cond, ref = binned(records), binned(reference)
    total_ref = sum(len(v) for v in ref.values())
    if not total_ref:
        return {"length_matched_hit_rate": float("nan"), "coverage": 0.0}
    num = 0.0
    covered = 0
    for b, ref_vals in ref.items():
        if b not in cond:
            continue
        v = np.array(cond[b])
        rate = float(((v >= lo) & (v < hi)).mean())
        num += rate * len(ref_vals)
        covered += len(ref_vals)
    return {
        "length_matched_hit_rate": float(num / covered) if covered else float("nan"),
        "coverage": float(covered / total_ref),
        "length_bin_tokens": LENGTH_BIN,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--heads", default=None)
    ap.add_argument("--property", default="clogp", choices=list(ALL_PROPERTIES))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--head-seed", type=int, default=None,
                    help="head training seed to steer with; default is the config's")
    ap.add_argument("--lam", type=float, default=None,
                    help="guidance strength; overrides configs/guidance.yaml lam. "
                         "The override is written into configs_used.json so the "
                         "artefact records the value actually used.")
    # C23 (additive, defaults reproduce the pre-edit code path exactly).
    #   --layer      probe point of `res.hidden_states` the scorer reads.  Omitted ->
    #                -1, which is what `guided_sample` already defaulted to.
    #   --head-file  head checkpoint to steer with.  Omitted -> the same
    #                `heads_dir/head_<prop>_frozen_state[_seed<s>].pt` as before.
    # `guided_sample` and both candidate backends already honoured `layer`; nothing
    # below this pair of arguments changes.
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden-state probe point for the head; default -1 (final). "
                         "GP-MoLFormer's hidden_states has 13 entries: 0 is the "
                         "embedding output and 1..12 the transformer layers, so "
                         "--layer 12 is identical to the default -1.")
    ap.add_argument("--head-file", default=None,
                    help="explicit head checkpoint path (absolute, or relative to "
                         "outputs/). Default: heads_dir/head_<prop>_frozen_state"
                         "[_seed<head-seed>].pt, i.e. the pre-edit behaviour.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / (args.heads or f"{args.dataset}_heads")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_guided_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    # A lambda override is folded into the config dict itself rather than kept beside
    # it, so `configs_used.json` -- the artefact a reader trusts -- cannot disagree
    # with the value the run actually used.
    lam_from_config = float(gcfg["lam"])
    if args.lam is not None:
        gcfg = {**gcfg, "lam": float(args.lam), "lam_source": "cli --lam",
                "lam_in_configs_guidance_yaml": lam_from_config}
    lam = float(gcfg["lam"])
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])

    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    n_mol = args.n or int(gcfg["n_molecules_per_condition"])
    seeds = args.seeds or list(gcfg["seeds"])
    conditions = args.conditions or list(gcfg["conditions"])

    gen = load_generator(model_cfg)
    # Guidance scores a *candidate hidden state*, so the decoder can only use the
    # frozen-state head.  That is the method under test; that the trivial features
    # are the better predictor for some properties is a finding, not a knob.
    head_path = heads_dir / f"head_{args.property}_frozen_state.pt"
    if args.head_seed is not None:
        head_path = heads_dir / f"head_{args.property}_frozen_state_seed{args.head_seed}.pt"
    if args.head_file is not None:
        p = Path(args.head_file)
        head_path = p if p.is_absolute() else (OUTPUT_DIR / p)
    layer = -1 if args.layer is None else int(args.layer)
    head, binner = load_head(head_path)
    scorer = TargetScorer(head, binner, lo, hi)

    print(f"property={args.property} target=[{lo:.4f},{hi:.4f}) base_rate={iv['base_rate']:.4f} lam={lam}")
    print(f"windows: early<{windows.t33} middle[{windows.t33},{windows.t67}) late>={windows.t67}")

    report: dict = {
        "dataset": args.dataset,
        "property": args.property,
        "head_input": "frozen_state",
        "head_checkpoint": head_path.name,
        "head_file": str(head_path),
        "head_file_source": "cli --head-file" if args.head_file is not None else "default",
        "layer": layer,
        "layer_source": "cli --layer" if args.layer is not None else "default (-1)",
        "target_interval": iv,
        "windows": windows.to_dict(),
        "lambda": lam,
        "lambda_source": gcfg.get("lam_source", "configs/guidance.yaml"),
        "top_k": int(gcfg["top_k_candidates"]),
        "eps": float(gcfg["eps"]),
        "backend": gcfg["candidate_backend"],
        "n_molecules_per_condition": n_mol,
        "seeds": seeds,
        "conditions": {},
    }
    all_records: dict[str, dict[int, list[dict]]] = {}
    t_start = time.perf_counter()

    for cond in conditions:
        report["conditions"][cond] = {"seeds": {}}
        all_records[cond] = {}
        for seed in seeds:
            meter = ComputeMeter().start()
            seqs = guided_sample(
                gen,
                scorer=None if cond == "unguided" else scorer,
                window_fn=windows.fn(cond),
                policy=policy,
                n_molecules=n_mol,
                seed=seed,
                top_k=int(gcfg["top_k_candidates"]),
                lam=0.0 if cond == "truncation_control" else lam,
                eps=float(gcfg["eps"]),
                backend=gcfg["candidate_backend"],
                batch_size=int(gcfg["batch_size"]),
                layer=layer,
                meter=meter,
            )
            meter.stop()
            smiles = gen.decode(seqs)
            records = []
            for ids, smi in zip(seqs, smiles):
                props = compute_all_properties(smi)
                content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
                records.append(
                    {
                        "smiles": smi,
                        "n_content_tokens": len(content),
                        "valid": props is not None,
                        **(props or {}),
                    }
                )
            all_records[cond][seed] = records
            s = summarise(records, args.property, lo, hi)
            s["compute"] = meter.as_dict()
            s["validity_check"] = validity(smiles)
            s["uniqueness_check"] = uniqueness(smiles)
            report["conditions"][cond]["seeds"][str(seed)] = s
            print(
                f"{cond:20s} seed={seed} hit={s['hit_rate']:.4f} "
                f"err={s['abs_target_error_mean']:.4f} val={s['validity']:.3f} "
                f"len={s['content_length_mean']:.1f} atoms={s['n_heavy_atoms_mean']:.1f} "
                f"tok/mol={s['compute']['tokens_per_molecule_actual']:.0f} "
                f"({meter.wall_seconds:.0f}s)"
            )

        # aggregate across seeds
        keys = ["hit_rate", "abs_target_error_mean", "validity", "uniqueness",
                "property_mean", "content_length_mean", "n_heavy_atoms_mean"]
        agg = {}
        for k in keys:
            v = [report["conditions"][cond]["seeds"][str(s)][k] for s in seeds]
            agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "values": v}
        tot = ComputeMeter()
        for s in seeds:
            c = report["conditions"][cond]["seeds"][str(s)]["compute"]
            tot.processed_tokens_actual += c["processed_tokens_actual"]
            tot.processed_tokens_full_recompute += c["processed_tokens_full_recompute"]
            tot.wall_seconds += c["wall_seconds"]
            tot.molecules_returned += c["molecules_returned"]
            tot.forward_calls += c["forward_calls"]
        agg["compute_total"] = tot.as_dict()
        report["conditions"][cond]["aggregate"] = agg

    # ---- length-confound analysis -------------------------------------------
    ref = [r for s in seeds for r in all_records["unguided"][s]]
    report["length_confound"] = {}
    for cond in conditions:
        pooled = [r for s in seeds for r in all_records[cond][s]]
        lm = length_matched_hit_rate(pooled, ref, args.property, lo, hi)
        raw = summarise(pooled, args.property, lo, hi)
        lm["raw_hit_rate"] = raw.get("hit_rate")
        lm["delta_vs_unguided_raw"] = raw.get("hit_rate", np.nan) - summarise(
            ref, args.property, lo, hi
        ).get("hit_rate", np.nan)
        report["length_confound"][cond] = lm

    ref_sum = summarise(ref, args.property, lo, hi)
    base_lm = length_matched_hit_rate(ref, ref, args.property, lo, hi)
    for cond in conditions:
        report["length_confound"][cond]["delta_vs_unguided_length_matched"] = (
            report["length_confound"][cond]["length_matched_hit_rate"]
            - base_lm["length_matched_hit_rate"]
        )

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "guidance_metrics.json", report)
    write_json(
        out_dir / "molecules.json",
        {c: {str(s): all_records[c][s] for s in seeds} for c in conditions},
    )
    write_run_context(out_dir)
    write_json(out_dir / "configs_used.json",
               {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                "cli": {"layer": layer, "layer_source": report["layer_source"],
                        "head_file": str(head_path),
                        "head_file_source": report["head_file_source"],
                        "head_seed": args.head_seed, "lam": lam}})

    print("\nlength-confound summary (hit rate):")
    for cond in conditions:
        lc = report["length_confound"][cond]
        print(f"  {cond:20s} raw={lc['raw_hit_rate']:.4f} "
              f"length_matched={lc['length_matched_hit_rate']:.4f} "
              f"d_raw={lc['delta_vs_unguided_raw']:+.4f} "
              f"d_matched={lc['delta_vs_unguided_length_matched']:+.4f}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
