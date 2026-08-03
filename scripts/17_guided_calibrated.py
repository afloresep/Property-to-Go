"""C18 -- guided decoding with a POST-HOC CALIBRATED head, end to end.

`scripts/05_guided_generation.py` is left untouched; its `summarise` and
`length_matched_hit_rate` are imported, so the metrics this writes are computed by the
same code and `guidance_metrics.json` has the same shape -- which is what lets
`scripts/06_best_of_n.py` compute a compute-matched baseline against it with no
changes either.

Two things are measured here and they are different in kind.

**The identity.**  `calibration.py` shows that a power calibration `g(q) = c*q**alpha`
is *exactly* a rescale of lambda, because `lam*log(c*q**alpha)` differs from
`(lam*alpha)*log q` by a candidate-independent constant and a softmax over the eight
candidates cancels it.  With `--eps 0` the two arms are equal in exact arithmetic, so
running `(power-calibrated head, lam)` against `(raw head, lam*alpha)` under identical
seeds must return **the same molecules**.  That is a decisive end-to-end test of the
claim that this whole calibration family is a point on the lambda sweep of §19 rather
than a new experiment.

**The measurement.**  Isotonic and bin-logit-temperature calibration are *not* power
maps, so they have to be run rather than argued about.

    .venv/bin/python scripts/17_guided_calibrated.py --property clogp --arm isotonic
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from property_to_go import calibration as C, generation  # noqa: E402
from property_to_go.binning import binner_from_dict  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import Windows, guided_sample  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import (  # noqa: E402
    ALL_PROPERTIES, compute_all_properties, uniqueness, validity,
)

_guided = import_module("05_guided_generation")

ARMS = ("uncalibrated", "platt", "power_limit", "isotonic", "bin_temperature")


def build_scorer(arm: str, head, binner, lo, hi, cal: dict):
    if arm == "uncalibrated":
        return C.CalibratedTargetScorer(head, binner, lo, hi), {"kind": "identity"}
    if arm == "platt":
        c = C.calibrator_from_dict(cal["platt"])
        return C.CalibratedTargetScorer(head, binner, lo, hi, calibrator=c), c.to_dict()
    if arm == "power_limit":
        c = C.calibrator_from_dict(cal["platt"]).power_limit()
        return C.CalibratedTargetScorer(head, binner, lo, hi, calibrator=c), c.to_dict()
    if arm == "isotonic":
        c = C.calibrator_from_dict(cal["isotonic"])
        return (C.CalibratedTargetScorer(head, binner, lo, hi, calibrator=c),
                {"kind": "isotonic", "n_knots": len(cal["isotonic"]["x"])})
    if arm == "bin_temperature":
        T = float(cal["bin_logit_temperature"] if cal.get("_override_T") is None
                  else cal["_override_T"])
        return (C.CalibratedTargetScorer(head, binner, lo, hi, bin_temperature=T),
                {"kind": "bin_logit_temperature", "temperature": T})
    raise ValueError(f"unknown arm {arm!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--heads", default="pilot_50k_heads_p2")
    ap.add_argument("--calibrators", default="c18_offpolicy_calibration")
    ap.add_argument("--property", default="clogp", choices=list(ALL_PROPERTIES))
    ap.add_argument("--arm", default="isotonic", choices=list(ARMS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=["unguided", "throughout"])
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None,
                    help="override the guidance eps. --eps 0 makes the power "
                         "calibration EXACTLY a lambda rescale, which is how the "
                         "identity is demonstrated rather than approximated.")
    ap.add_argument("--bin-temperature", type=float, default=None,
                    help="override the ECE-selected bin-logit temperature. The "
                         "per-position sweep in c18_per_position shows the "
                         "calibration-optimal T and the decoder-optimal T are "
                         "different; this runs the second one end to end.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / args.heads
    out_dir = OUTPUT_DIR / (args.out or f"c18_guided_{args.arm}_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    if args.lam is not None:
        gcfg = {**gcfg, "lam": float(args.lam), "lam_source": "cli --lam"}
    if args.eps is not None:
        gcfg = {**gcfg, "eps": float(args.eps), "eps_source": "cli --eps"}
    lam, eps = float(gcfg["lam"]), float(gcfg["eps"])

    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    n_mol = args.n or int(gcfg["n_molecules_per_condition"])
    seeds = args.seeds or list(gcfg["seeds"])

    cal = read_json(OUTPUT_DIR / args.calibrators / f"calibrator_{args.property}.json")
    cal["_override_T"] = args.bin_temperature

    gen = load_generator(model_cfg)
    ck = torch.load(heads_dir / f"head_{args.property}_frozen_state.pt",
                    map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    binner = binner_from_dict(ck["binner"])
    scorer, cal_desc = build_scorer(args.arm, head, binner, lo, hi, cal)

    report: dict = {
        "dataset": args.dataset, "property": args.property,
        "head_input": "frozen_state",
        "head_checkpoint": f"head_{args.property}_frozen_state.pt",
        "c18_arm": args.arm, "calibrator": cal_desc,
        "calibrator_source": args.calibrators,
        "target_interval": iv, "windows": windows.to_dict(),
        "lambda": lam, "lambda_source": gcfg.get("lam_source", "configs/guidance.yaml"),
        "top_k": int(gcfg["top_k_candidates"]), "eps": eps,
        "eps_source": gcfg.get("eps_source", "configs/guidance.yaml"),
        "backend": gcfg["candidate_backend"],
        "n_molecules_per_condition": n_mol, "seeds": seeds,
        "not_dagger": (
            "The calibrator is fitted on the head's OUTPUTS on held-out guided "
            "prefixes. No head weight is retrained on guided data, so this is not a "
            "second data-aggregation round (pilot_report.md 9.2.1)."
        ),
        "conditions": {},
    }
    all_records: dict[str, dict[int, list[dict]]] = {}
    t_start = time.perf_counter()

    for cond in args.conditions:
        report["conditions"][cond] = {"seeds": {}}
        all_records[cond] = {}
        for seed in seeds:
            meter = ComputeMeter().start()
            seqs = guided_sample(
                gen, scorer=None if cond == "unguided" else scorer,
                window_fn=windows.fn(cond), policy=policy, n_molecules=n_mol, seed=seed,
                top_k=int(gcfg["top_k_candidates"]),
                lam=0.0 if cond == "truncation_control" else lam, eps=eps,
                backend=gcfg["candidate_backend"], batch_size=int(gcfg["batch_size"]),
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
            s = _guided.summarise(records, args.property, lo, hi)
            s["compute"] = meter.as_dict()
            s["validity_check"] = validity(smiles)
            s["uniqueness_check"] = uniqueness(smiles)
            report["conditions"][cond]["seeds"][str(seed)] = s
            print(f"{cond:20s} seed={seed} hit={s['hit_rate']:.4f} "
                  f"val={s['validity']:.3f} tok/mol="
                  f"{s['compute']['tokens_per_molecule_actual']:.0f} "
                  f"({meter.wall_seconds:.0f}s)")

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

    if "unguided" in all_records:
        ref = [r for s in seeds for r in all_records["unguided"][s]]
        report["length_confound"] = {}
        base_lm = _guided.length_matched_hit_rate(ref, ref, args.property, lo, hi)
        ref_hit = _guided.summarise(ref, args.property, lo, hi).get("hit_rate", float("nan"))
        for cond in args.conditions:
            pooled = [r for s in seeds for r in all_records[cond][s]]
            lm = _guided.length_matched_hit_rate(pooled, ref, args.property, lo, hi)
            raw = _guided.summarise(pooled, args.property, lo, hi)
            lm["raw_hit_rate"] = raw.get("hit_rate")
            lm["delta_vs_unguided_raw"] = raw.get("hit_rate", np.nan) - ref_hit
            lm["delta_vs_unguided_length_matched"] = (
                lm["length_matched_hit_rate"] - base_lm["length_matched_hit_rate"]
            )
            report["length_confound"][cond] = lm

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "guidance_metrics.json", report)
    write_json(out_dir / "molecules.json",
               {c: {str(s): all_records[c][s] for s in seeds} for c in args.conditions})
    write_json(out_dir / "configs_used.json",
               {"model": model_cfg, "base_policy": policy, "guidance": gcfg})
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
