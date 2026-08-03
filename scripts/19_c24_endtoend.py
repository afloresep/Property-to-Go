"""C24 stage 5 -- END TO END: guided generation arms and compute-matched best-of-N.

The text-domain replication of `pilot_report.md` §20.5 (calibration arms), §21.5/C23
(probe depth) and the identity of §20.3.1.  Every arm is checkpointed to its own JSON, so
an interrupted run costs at most one arm.

Arms, per attribute:

    unguided                     the base policy, the control
    throughout_L12               the deployed rule at the final probe point
    throughout_L<best>           the same rule at the best-*predicting* probe point
    platt_L12 / isotonic_L12     post-hoc calibration, end to end
    identity_power_eps0          g(q) = q^alpha at lambda = 1, eps = 0
    identity_lamalpha_eps0       the raw head at lambda = alpha, eps = 0
    identity_power_epsdep        the same pair at the deployed eps = 1e-6
    identity_lamalpha_epsdep

    .venv/bin/python scripts/19_c24_endtoend.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import calibration as C  # noqa: E402
from property_to_go import generality as G  # noqa: E402
from property_to_go.binning import binner_from_dict  # noqa: E402
from property_to_go.compute import ComputeMeter, solve_best_of_n  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402
from property_to_go.guidance import TargetScorer  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402

CFG = {
    "n_per_condition": 512,
    "seeds": [101, 202, 303],
    "lam": 1.0,
    "eps": 1e-6,
    "top_k": 8,
    "n_content": 40,
    "batch_size": 32,
    "head_seed": 1234,
    "bestofn_seed": 5150,
}


def load_head(path: Path, device):
    d = torch.load(path, map_location="cpu")
    head = MLPHead(d["in_dim"], d["hidden_dim"], d["n_bins"], d["dropout"])
    head.load_state_dict(d["state_dict"])
    return head.to(device).eval()


def run_arm(gen, arm, attr, band, binner, hd, device, out):
    """One arm: 3 seeds, `n_per_condition` sequences each.  Idempotent."""
    path = out / f"arm_{attr}_{arm['name']}.json"
    if path.exists():
        return read_json(path)

    lo, hi = float(band["lo"]), float(band["hi"])
    scorer = None
    if arm["kind"] != "unguided":
        head = load_head(
            hd / "heads" / f"{attr}_L{arm['layer']}_seed{CFG['head_seed']}.pt", device
        )
        cal = C.calibrator_from_dict(arm["calibrator"]) if arm.get("calibrator") else None
        scorer = C.CalibratedTargetScorer(head, binner, lo, hi, calibrator=cal)

    per_seed = []
    all_texts: list[str] = []
    meter = ComputeMeter().start()
    for seed in CFG["seeds"]:
        m = ComputeMeter()
        seqs = G.guided_sample_text(
            gen, scorer, n=CFG["n_per_condition"], seed=seed, n_content=CFG["n_content"],
            lam=arm.get("lam", CFG["lam"]), eps=arm.get("eps", CFG["eps"]),
            top_k=CFG["top_k"], layer=arm.get("layer", -1),
            batch_size=CFG["batch_size"], meter=m,
        )
        texts = gen.decode([s[1:] for s in seqs])
        summ = G.summarise_texts(texts, attr, lo, hi)
        summ["seed"] = seed
        summ["compute"] = m.as_dict()
        per_seed.append(summ)
        all_texts.extend(texts)
        meter.merge(m)
    meter.stop()

    hits = np.array([s["hit_rate"] for s in per_seed])
    record = {
        "arm": arm, "attribute": attr, "target": band,
        "per_seed": per_seed,
        "hit_rate_mean": float(hits.mean()),
        "hit_rate_sd": float(hits.std(ddof=1)),
        "hit_rate_values": hits.tolist(),
        "pooled": G.summarise_texts(all_texts, attr, lo, hi),
        "compute": meter.as_dict(),
        "texts_sha256": __import__("hashlib").sha256(
            "\n".join(all_texts).encode()).hexdigest(),
    }
    write_json(path, record)
    write_json(out / f"texts_{attr}_{arm['name']}.json", {"texts": all_texts})
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c24_dataset")
    ap.add_argument("--heads", default="c24_probe_layers")
    ap.add_argument("--calibration", default="c24_calibration")
    ap.add_argument("--out", default="c24_endtoend")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    args = ap.parse_args()
    CFG["batch_size"] = int(args.batch_size)

    ds = OUTPUT_DIR / args.dataset
    hd = OUTPUT_DIR / args.heads
    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"c24_endtoend": CFG, "dataset": args.dataset})

    ti = read_json(ds / "target_intervals.json")
    depth = read_json(hd / "probe_layer_metrics.json")
    cal = read_json(OUTPUT_DIR / args.calibration / "calibration_metrics.json")
    n_probe = int(depth["n_probe_points"])
    final = n_probe - 1
    t0 = time.time()

    gen = G.load_text_generator(args.device)
    summary: dict[str, dict] = {}

    for attr, band in ti["intervals"].items():
        binner = binner_from_dict(ti["binners"][attr])
        best_L = int(depth["attributes"][attr]["depth"]["best_probe_point"])
        alpha = float(cal["attributes"][attr]["platt_slope"])
        platt_d = cal["attributes"][attr]["calibrated"]["platt"]["calibrator"]
        iso_d = cal["attributes"][attr]["calibrated"]["isotonic"]["calibrator"]
        power_d = {"kind": "power", "alpha": alpha, "log_c": 0.0}

        arms = [
            {"name": "unguided", "kind": "unguided"},
            # top-8 restriction with NO property term.  `pilot_report.md` §7.9's control,
            # and on this substrate it is not a formality: restricting GPT-2 to its top 8
            # tokens is itself a large intervention on every attribute, so "guidance moved
            # the attribute" cannot be read off the gap to `unguided` alone.
            {"name": "truncation_control", "kind": "guided", "layer": final, "lam": 0.0},
            {"name": f"throughout_L{final}", "kind": "guided", "layer": final},
            {"name": f"throughout_L{best_L}", "kind": "guided", "layer": best_L},
            {"name": f"platt_L{final}", "kind": "guided", "layer": final,
             "calibrator": platt_d},
            {"name": f"isotonic_L{final}", "kind": "guided", "layer": final,
             "calibrator": iso_d},
            {"name": "identity_power_eps0", "kind": "guided", "layer": final,
             "calibrator": power_d, "lam": 1.0, "eps": 0.0},
            {"name": "identity_lamalpha_eps0", "kind": "guided", "layer": final,
             "lam": alpha, "eps": 0.0},
            {"name": "identity_power_epsdep", "kind": "guided", "layer": final,
             "calibrator": power_d, "lam": 1.0, "eps": CFG["eps"]},
            {"name": "identity_lamalpha_epsdep", "kind": "guided", "layer": final,
             "lam": alpha, "eps": CFG["eps"]},
        ]
        if best_L == final:
            arms = [a for a in arms if a["name"] != f"throughout_L{best_L}"]

        recs = {}
        for arm in arms:
            r = run_arm(gen, arm, attr, band, binner, hd, args.device, out)
            recs[arm["name"]] = r
            print(f"[{time.time()-t0:.0f}s] {attr}/{arm['name']}: hit "
                  f"{r['hit_rate_mean']:.4f} +-{r['hit_rate_sd']:.4f}"
                  f"  tokens/seq {r['compute']['tokens_per_molecule_actual']:.1f}", flush=True)

        # --- the identity, sequence by sequence ---------------------------------
        identity = {}
        for tag in ("eps0", "epsdep"):
            a = read_json(out / f"texts_{attr}_identity_power_{tag}.json")["texts"]
            b = read_json(out / f"texts_{attr}_identity_lamalpha_{tag}.json")["texts"]
            same = sum(x == y for x, y in zip(a, b))
            identity[tag] = {
                "n": len(a),
                "identical": int(same),
                "identical_fraction": float(same / len(a)),
                "hit_rate_power": recs[f"identity_power_{tag}"]["pooled"]["hit_rate"],
                "hit_rate_lamalpha": recs[f"identity_lamalpha_{tag}"]["pooled"]["hit_rate"],
                "hit_rate_difference": recs[f"identity_power_{tag}"]["pooled"]["hit_rate"]
                - recs[f"identity_lamalpha_{tag}"]["pooled"]["hit_rate"],
                "alpha": alpha,
            }
            print(f"[{time.time()-t0:.0f}s] {attr} identity {tag}: "
                  f"{same}/{len(a)} identical, dhit "
                  f"{identity[tag]['hit_rate_difference']:+.6f}", flush=True)

        # --- compute-matched best-of-N, per arm ---------------------------------
        base_tokens = recs["unguided"]["compute"]["tokens_per_molecule_actual"]
        bestofn = {}
        for name, r in recs.items():
            if name == "unguided":
                continue
            gt = r["compute"]["tokens_per_molecule_actual"]
            n_cand = solve_best_of_n(gt, base_tokens)
            key = str(n_cand)
            if key not in bestofn:
                m = ComputeMeter().start()
                sel = G.best_of_n_text(
                    gen, n_sequences=CFG["n_per_condition"] * len(CFG["seeds"]),
                    n_candidates=n_cand, seed=CFG["bestofn_seed"], attribute=attr,
                    lo=float(band["lo"]), hi=float(band["hi"]),
                    n_content=CFG["n_content"], meter=m,
                )
                m.stop()
                bestofn[key] = {
                    "n_candidates": n_cand,
                    **G.summarise_texts([s["text"] for s in sel], attr,
                                        float(band["lo"]), float(band["hi"])),
                    "compute": m.as_dict(),
                }
                print(f"[{time.time()-t0:.0f}s] {attr} best-of-{n_cand}: "
                      f"hit {bestofn[key]['hit_rate']:.4f}", flush=True)
            b = bestofn[key]
            r["best_of_n"] = {
                "n_candidates": n_cand,
                "hit_rate": b["hit_rate"],
                "advantage_guided_minus_bestofn": r["hit_rate_mean"] - b["hit_rate"],
                "guided_tokens_per_sequence": gt,
                "base_tokens_per_sequence": base_tokens,
                "bestofn_tokens_per_sequence":
                    b["compute"]["tokens_per_molecule_actual"],
                "realised_token_ratio":
                    b["compute"]["tokens_per_molecule_actual"] / gt,
            }
            write_json(out / f"arm_{attr}_{name}.json", r)

        summary[attr] = {
            "target": band,
            "best_probe_point": best_L,
            "final_probe_point": final,
            "platt_slope": alpha,
            "arms": {n: {k: v for k, v in r.items() if k != "per_seed"} |
                        {"per_seed_hit_rate": r["hit_rate_values"]}
                     for n, r in recs.items()},
            "identity": identity,
            "best_of_n_runs": bestofn,
        }

    write_json(out / "endtoend_metrics.json", {"config": CFG, "attributes": summary})
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
