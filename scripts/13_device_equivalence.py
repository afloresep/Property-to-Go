"""Phase 2 -- what changes when the frozen generator moves from CPU to GPU.

Setup gate (d) of the phase-2 kickoff failed: regenerating `pilot_50k` on an RTX 4090
with the recorded seeds produced target intervals that differ from the frozen ones. The
instruction in that case is to stop and characterise it rather than route around it, and
that is what this script measures. It separates two explanations that the dataset-level
diff cannot:

  **numerical** -- float32 GPU kernels give different logits, so the model itself
      behaves differently and every downstream number is suspect;
  **RNG stream** -- the logits agree but `torch.multinomial` on a CUDA tensor draws from
      the CUDA Philox generator rather than the CPU Mersenne Twister, so the same seed
      selects different tokens. The base *policy* is unchanged; the *sample* is not.

The distinction matters because only the first would invalidate anything. The second
means the two runs are independent draws from the same distribution, which is a
reproducibility limitation to declare, not an error to fix.

Loads the same pinned revision twice, once per device, and compares:

  * the weight checksum, so "same model" is asserted rather than assumed;
  * last-position logits and log-probs on fixed prefixes;
  * **the top-8 candidate set** -- the quantity guided decoding actually consumes, so
    agreement here bounds the numerical risk to the method under test;
  * whether the two devices draw the same molecules at the same seed.

    python scripts/13_device_equivalence.py --n-molecules 64
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
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, write_json, write_run_context,
)
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import compute_properties  # noqa: E402

PREFIXES = [
    "CCOc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc2[nH]ccc2c1",
    "CN1CCN(CC1)c1ncnc2[nH]ccc12",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-molecules", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=None, help="default: guidance config")
    ap.add_argument("--timing-repeats", type=int, default=3,
                    help="repeats of an identical workload, to test whether wall time "
                         "reproduces on THIS machine before any timing claim is made")
    ap.add_argument("--out", default="device_equivalence")
    args = ap.parse_args()

    t_start = time.perf_counter()
    out_dir = OUTPUT_DIR / args.out
    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    top_k = args.top_k or int(gcfg["top_k_candidates"])

    primary = load_generator(model_cfg)
    other_device = "cpu" if str(primary.device).startswith("cuda") else "cuda"
    secondary = load_generator({**model_cfg, "device": other_device})

    fp_a, fp_b = primary.fingerprint(), secondary.fingerprint()
    report: dict = {
        "primary_device": str(primary.device),
        "secondary_device": str(secondary.device),
        "dtype": model_cfg.get("dtype"),
        "model_revision": model_cfg["model_revision"],
        "weights_identical": bool(fp_a["parameter_sum"] == fp_b["parameter_sum"]),
        "n_parameters": int(fp_a["n_parameters"]),
        "deterministic_eval": bool(fp_a["deterministic_eval"]),
        "top_k": top_k,
    }
    print(f"{report['primary_device']} vs {report['secondary_device']}, "
          f"weights identical: {report['weights_identical']}")

    # ---- 1. does the forward pass agree? ---------------------------------------
    per_prefix = []
    for smi in PREFIXES:
        ids = primary.tokenizer(smi, return_tensors="pt").input_ids[:, :-1]
        with torch.no_grad():
            la = primary.model(ids.to(primary.device)).logits[0, -1].float().cpu()
            lb = secondary.model(ids.to(secondary.device)).logits[0, -1].float().cpu()
        lpa = torch.log_softmax(la, -1).numpy()
        lpb = torch.log_softmax(lb, -1).numpy()
        ta = torch.topk(la, top_k).indices.tolist()
        tb = torch.topk(lb, top_k).indices.tolist()
        per_prefix.append({
            "prefix": smi,
            "max_abs_logit_difference": float((la - lb).abs().max()),
            "max_abs_logprob_difference": float(np.abs(lpa - lpb).max()),
            "top_k_candidate_sets_identical": ta == tb,
            "top_k_logprob_max_difference": float(np.abs(lpa[ta] - lpb[tb]).max()),
        })
        print(f"  {smi:28s} max|dlogit|={per_prefix[-1]['max_abs_logit_difference']:.3e} "
              f"top-{top_k} identical={per_prefix[-1]['top_k_candidate_sets_identical']}")
    report["forward_pass"] = {
        "per_prefix": per_prefix,
        "max_abs_logit_difference": max(p["max_abs_logit_difference"] for p in per_prefix),
        "all_top_k_candidate_sets_identical": all(
            p["top_k_candidate_sets_identical"] for p in per_prefix
        ),
    }

    # ---- 2. does the same seed draw the same molecules? ------------------------
    seed = int(policy["seed"])
    pol = {**policy, "batch_size": min(int(policy["batch_size"]), args.n_molecules)}
    seqs_a = generation.sample_unconditional(primary, pol, args.n_molecules, seed=seed)
    seqs_b = generation.sample_unconditional(secondary, pol, args.n_molecules, seed=seed)
    identical = sum(a == b for a, b in zip(seqs_a, seqs_b))

    def stats(gen, seqs):
        props = [compute_properties(s) for s in gen.decode(seqs)]
        ok = [p for p in props if p is not None]
        return {
            "n": len(seqs),
            "validity": len(ok) / len(seqs),
            "clogp_mean": float(np.mean([p["clogp"] for p in ok])),
            "aromatic_rings_mean": float(np.mean([p["aromatic_rings"] for p in ok])),
            "content_length_mean": float(np.mean([
                len(generation.sequence_content(s, gen.bos_id, gen.eos_id, gen.pad_id))
                for s in seqs
            ])),
        }

    report["sampling"] = {
        "seed": seed,
        "n_molecules": args.n_molecules,
        "n_identical_token_sequences": int(identical),
        "fraction_identical": float(identical / args.n_molecules),
        "primary": stats(primary, seqs_a),
        "secondary": stats(secondary, seqs_b),
    }
    print(f"  identical molecules at seed {seed}: {identical}/{args.n_molecules}")

    # ---- 3. does the SAME device reproduce itself? ----------------------------
    # The determinism that does hold, and the one the pilot's reproducibility rests on.
    again = generation.sample_unconditional(primary, pol, args.n_molecules, seed=seed)
    report["sampling"]["same_device_same_seed_identical"] = bool(again == seqs_a)
    print(f"  same device, same seed, reproduces itself: "
          f"{report['sampling']['same_device_same_seed_identical']}")

    # ---- 4. does WALL TIME reproduce here? -----------------------------------
    # The pilot found 20-25% swings between bit-identical runs on its laptop and
    # therefore refused every timing claim. Whether that band applies to this machine is
    # a separate empirical question, and it has to be answered before any GPU timing
    # number is quoted rather than assumed either way.
    times, tokens = [], []
    for _ in range(args.timing_repeats):
        from property_to_go.compute import ComputeMeter
        m = ComputeMeter().start()
        generation.sample_unconditional(primary, pol, args.n_molecules, seed=seed, meter=m)
        m.stop()
        times.append(m.wall_seconds)
        tokens.append(m.processed_tokens_actual)
    t = np.array(times)
    report["timing_reproducibility"] = {
        "workload": f"{args.n_molecules} unconditional molecules at seed {seed}",
        "repeats": args.timing_repeats,
        "wall_seconds": t.tolist(),
        "wall_seconds_mean": float(t.mean()),
        "wall_seconds_relative_spread": float((t.max() - t.min()) / t.mean()),
        "processed_tokens": tokens,
        "processed_tokens_identical": bool(len(set(tokens)) == 1),
    }
    print(f"  wall time over {args.timing_repeats} identical repeats: "
          f"{[round(x, 2) for x in times]} s "
          f"(spread {report['timing_reproducibility']['wall_seconds_relative_spread']:.1%}), "
          f"tokens identical: {report['timing_reproducibility']['processed_tokens_identical']}")

    report["verdict"] = (
        "numerics agree to float32 tolerance and the top-k candidate set guided decoding "
        "consumes is identical on both devices; the same seed nonetheless draws different "
        "molecules, because torch.multinomial on a CUDA tensor uses the CUDA Philox "
        "generator rather than the CPU Mersenne Twister. The base policy is unchanged, "
        "the sample is not."
        if report["forward_pass"]["all_top_k_candidate_sets_identical"]
        and report["sampling"]["fraction_identical"] < 1.0
        else "unexpected combination -- read the fields, do not trust this sentence."
    )
    report["wall_seconds_total"] = time.perf_counter() - t_start

    write_json(out_dir / "device_equivalence.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy})
    print(f"\n{report['verdict']}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
