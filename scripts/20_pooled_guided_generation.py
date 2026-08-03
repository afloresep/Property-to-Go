"""C25 -- guided decoding with a *pooled* readout.

`scripts/05_guided_generation.py` reads one hidden state per candidate and may not be
edited (C23 already spent the one additive edit it was allowed, and every executed number
in the report comes from it).  `src/property_to_go/guidance.py` may not be edited either.
So the pooled decoder lives in `property_to_go.pooling.pooled_guided_sample`, a
transcription of `guidance.guided_sample` whose only structural difference is a rolling
buffer of committed hidden states, and this script is its runner.

Three things keep the two paths comparable rather than merely similar:

* `tests/test_pooled_readout.py::test_pooled_guided_sample_at_window_one_matches_guided_sample`
  asserts that at window 1 the pooled sampler returns the **same molecules** and the
  **same token counts** as `guided_sample`;
* the summarisation, the length-confound estimator and the artefact layout are script 05's
  own functions, imported from it rather than reimplemented, so `scripts/06_best_of_n.py`
  and `scripts/10_quality_analysis.py` consume this output unchanged;
* the compute meter is the same `ComputeMeter`, and the committed hidden states are free
  (§C25.0.2), so `processed_tokens_actual` is directly comparable with every other run.

    .venv/bin/python scripts/20_pooled_guided_generation.py --dataset pilot_50k_p2 \
        --property hbd_count --layer 4 --variant concat4 --lam 2 \
        --head-file c25_pooled_heads/head_hbd_count_concat4_L4_seed1234.pt \
        --out c25_guided_concat4_L4_lam2_hbd_count
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import generation  # noqa: E402
from property_to_go import pooling as PL  # noqa: E402
from property_to_go.binning import binner_from_dict  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import Windows  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import (  # noqa: E402
    ALL_PROPERTIES, compute_all_properties, uniqueness, validity,
)


def _script05():
    """Import script 05 as a module so its summarisers are *reused*, not reimplemented."""
    path = ROOT / "scripts" / "05_guided_generation.py"
    spec = importlib.util.spec_from_file_location("guided_generation_05", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pooled_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = PL.VARIANTS_BY_NAME[ck["pool_variant"]]
    if ck["pool_mode"] == "attn":
        head = PL.AttnPoolHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"],
                               ck["dropout"], attn_dim=int(ck["attn_dim"]))
    else:
        head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"]), spec, ck


def main() -> int:
    S = _script05()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--property", default="hbd_count", choices=list(ALL_PROPERTIES))
    ap.add_argument("--head-file", required=True,
                    help="a C25 pooled checkpoint (absolute, or relative to outputs/)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=["unguided", "throughout"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    lam_from_config = float(gcfg["lam"])
    if args.lam is not None:
        gcfg = {**gcfg, "lam": float(args.lam), "lam_source": "cli --lam",
                "lam_in_configs_guidance_yaml": lam_from_config}
    lam = float(gcfg["lam"])

    # Frozen before any guided result was inspected; read verbatim, never re-derived.
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    n_mol = args.n or int(gcfg["n_molecules_per_condition"])
    seeds = args.seeds or list(gcfg["seeds"])

    p = Path(args.head_file)
    head_path = p if p.is_absolute() else (OUTPUT_DIR / p)
    head, binner, spec, ck = load_pooled_head(head_path)
    if int(ck["probe_point"]) != int(args.layer):
        raise SystemExit(f"checkpoint probe point {ck['probe_point']} != --layer {args.layer}")
    if ck["property"] != args.property:
        raise SystemExit(f"checkpoint property {ck['property']} != {args.property}")

    gen = load_generator(model_cfg)
    scorer = PL.PooledTargetScorer(head, binner, lo, hi, spec)

    print(f"property={args.property} variant={spec.name} window={spec.window} "
          f"mode={spec.mode} layer={args.layer} lam={lam} "
          f"target=[{lo:.4f},{hi:.4f}) base_rate={iv['base_rate']:.4f}", flush=True)

    report: dict = {
        "dataset": args.dataset,
        "property": args.property,
        "head_input": "frozen_state",
        "head_checkpoint": head_path.name,
        "head_file": str(head_path),
        "head_file_source": "cli --head-file",
        "head_seed": int(ck["head_seed"]),
        "layer": int(args.layer),
        "layer_source": "cli --layer",
        "pool_variant": spec.name,
        "pool_window": int(spec.window),
        "pool_mode": spec.mode,
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

    for cond in args.conditions:
        report["conditions"][cond] = {"seeds": {}}
        all_records[cond] = {}
        for seed in seeds:
            meter = ComputeMeter().start()
            seqs = PL.pooled_guided_sample(
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
                layer=int(args.layer),
                window=int(spec.window),
                meter=meter,
            )
            meter.stop()
            smiles = gen.decode(seqs)
            records = []
            for ids, smi in zip(seqs, smiles):
                props = compute_all_properties(smi)
                content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
                records.append({"smiles": smi, "n_content_tokens": len(content),
                                "valid": props is not None, **(props or {})})
            all_records[cond][seed] = records
            s = S.summarise(records, args.property, lo, hi)
            s["compute"] = meter.as_dict()
            s["validity_check"] = validity(smiles)
            s["uniqueness_check"] = uniqueness(smiles)
            report["conditions"][cond]["seeds"][str(seed)] = s
            print(f"{cond:20s} seed={seed} hit={s['hit_rate']:.4f} "
                  f"val={s['validity']:.3f} len={s['content_length_mean']:.1f} "
                  f"tok/mol={s['compute']['tokens_per_molecule_actual']:.0f} "
                  f"({meter.wall_seconds:.0f}s)", flush=True)

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

    ref = [r for s in seeds for r in all_records["unguided"][s]]
    report["length_confound"] = {}
    for cond in args.conditions:
        pooled = [r for s in seeds for r in all_records[cond][s]]
        lm = S.length_matched_hit_rate(pooled, ref, args.property, lo, hi)
        raw = S.summarise(pooled, args.property, lo, hi)
        lm["raw_hit_rate"] = raw.get("hit_rate")
        lm["delta_vs_unguided_raw"] = raw.get("hit_rate", np.nan) - S.summarise(
            ref, args.property, lo, hi).get("hit_rate", np.nan)
        report["length_confound"][cond] = lm
    base_lm = S.length_matched_hit_rate(ref, ref, args.property, lo, hi)
    for cond in args.conditions:
        report["length_confound"][cond]["delta_vs_unguided_length_matched"] = (
            report["length_confound"][cond]["length_matched_hit_rate"]
            - base_lm["length_matched_hit_rate"])

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "guidance_metrics.json", report)
    write_json(out_dir / "molecules.json",
               {c: {str(s): all_records[c][s] for s in seeds} for c in args.conditions})
    write_run_context(out_dir)
    write_json(out_dir / "configs_used.json",
               {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                "cli": {"layer": int(args.layer), "head_file": str(head_path),
                        "pool_variant": spec.name, "pool_window": int(spec.window),
                        "pool_mode": spec.mode, "head_seed": int(ck["head_seed"]),
                        "lam": lam}})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
