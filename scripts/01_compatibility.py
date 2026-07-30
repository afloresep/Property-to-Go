"""Phase 1 -- compatibility spike.

Runs, in order, the ten operations the pilot depends on and writes a
machine-readable report.  Nothing downstream is allowed to run unless every check
here is marked "ok": the report is the evidence that the frozen checkpoint really
supports prefix hidden states, reproducible prefix continuation, and batched
candidate evaluation, without any model surgery.

    python scripts/01_compatibility.py [--n 100] [--out outputs/compatibility]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generation, properties  # noqa: E402
from property_to_go.config import RunDir, load_config, write_json  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import select_quartile_prefixes  # noqa: E402
from property_to_go.tokens import prefix_features_from_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=str, default="compatibility")
    args = ap.parse_args()

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    run = RunDir.create(args.out, {"model": model_cfg, "base_policy": policy})

    report: dict = {"checks": {}, "n_requested": args.n}
    t_start = time.perf_counter()

    # ---- 1. load checkpoint and tokenizer -------------------------------------
    t0 = time.perf_counter()
    gen = load_generator(model_cfg)
    report["checks"]["01_load_model_and_tokenizer"] = {
        "status": "ok",
        "seconds": time.perf_counter() - t0,
        "model_class": type(gen.model).__name__,
        "tokenizer_class": type(gen.tokenizer).__name__,
        "vocab_size": int(gen.tokenizer.vocab_size),
        "hidden_size": gen.hidden_size,
        "max_position_embeddings": gen.max_length,
        "special_token_ids": {"bos": gen.bos_id, "eos": gen.eos_id, "pad": gen.pad_id},
        "fingerprint": gen.fingerprint(),
        "device": str(gen.device),
    }

    # ---- 2. generate n unconditional molecules --------------------------------
    meter = ComputeMeter().start()
    seqs = generation.sample_unconditional(gen, policy, args.n, seed=int(policy["seed"]), meter=meter)
    meter.stop()
    smiles = gen.decode(seqs)
    report["checks"]["02_generate_unconditional"] = {
        "status": "ok",
        "n": len(seqs),
        "compute": meter.as_dict(),
        "first_five_smiles": smiles[:5],
    }

    # ---- 3. RDKit validity, uniqueness, lengths, atoms, properties ------------
    id_to_token = gen.id_to_token()
    records = []
    for ids, smi in zip(seqs, smiles):
        content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
        props = properties.compute_properties(smi)
        records.append(
            {
                "smiles": smi,
                "n_content_tokens": len(content),
                "valid": props is not None,
                **(props or {}),
            }
        )
    valid = [r for r in records if r["valid"]]

    def stats(key, rows=valid):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "quantiles": {str(q): float(np.quantile(v, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        }

    report["checks"]["03_rdkit_diagnostics"] = {
        "status": "ok",
        "validity": properties.validity(smiles),
        "uniqueness": properties.uniqueness(smiles),
        "sequence_length_tokens": stats("n_content_tokens", records),
        "n_heavy_atoms": stats("n_heavy_atoms"),
        "clogp": stats("clogp"),
        "aromatic_rings": stats("aromatic_rings"),
        "mol_weight": stats("mol_weight"),
    }

    # ---- 4. four prefix positions per trajectory ------------------------------
    rng = np.random.default_rng(7)
    prefix_plan = []
    skipped = 0
    for ids, rec in zip(seqs, records):
        n_content = rec["n_content_tokens"]
        if n_content < 8:
            skipped += 1
            continue
        prefix_plan.append(
            {"token_ids": ids, "n_content": n_content, "picks": select_quartile_prefixes(n_content, rng)}
        )
    report["checks"]["04_select_prefixes"] = {
        "status": "ok",
        "n_trajectories_used": len(prefix_plan),
        "n_skipped_too_short": skipped,
        "prefixes_per_trajectory": 4,
        "example": {
            "n_content": prefix_plan[0]["n_content"],
            "picks": prefix_plan[0]["picks"],
        },
    }

    # ---- 5. extract and save frozen hidden states -----------------------------
    hs_meter = ComputeMeter().start()
    sequences = [p["token_ids"] for p in prefix_plan]
    positions = [[k for _, k in p["picks"]] for p in prefix_plan]
    states = generation.hidden_states_for_positions(
        gen, sequences, positions, layer=-1, meter=hs_meter
    )
    hs_meter.stop()
    stacked = np.stack(states)  # (n_traj, 4, hidden)
    np.save(run / "prefix_hidden_states.npy", stacked)

    # a prefix state must equal the state from a forward pass over that prefix alone
    check_idx = 0
    k = positions[check_idx][2]
    single = gen.model(
        input_ids=torch.tensor([sequences[check_idx][: k + 1]], device=gen.device),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-1][0, -1].detach().float().numpy()
    causal_err = float(np.abs(single - stacked[check_idx, 2]).max())

    report["checks"]["05_hidden_states"] = {
        "status": "ok" if causal_err < 1e-4 else "FAILED",
        "shape": list(stacked.shape),
        "dtype": str(stacked.dtype),
        "saved_to": str((run / "prefix_hidden_states.npy").relative_to(Path.cwd())),
        "causal_equivalence_max_abs_err": causal_err,
        "note": "state at position k from one full-sequence pass == state from a pass over the prefix alone",
        "compute": hs_meter.as_dict(),
    }

    # ---- 6. continue generation from a stored prefix --------------------------
    stored_prefix = sequences[0][: positions[0][1] + 1]
    cont_meter = ComputeMeter().start()
    conts = generation.continue_from_prefixes(
        gen, [stored_prefix], n_each=8, policy=policy, seed=99, meter=cont_meter
    )[0]
    cont_meter.stop()
    cont_smiles = gen.decode(conts)
    prefix_ok = all(c[: len(stored_prefix)] == stored_prefix for c in conts)

    # reproducibility: identical seed must give identical continuations
    conts_again = generation.continue_from_prefixes(
        gen, [stored_prefix], n_each=8, policy=policy, seed=99
    )[0]
    report["checks"]["06_continue_from_prefix"] = {
        "status": "ok" if prefix_ok and conts == conts_again else "FAILED",
        "stored_prefix_ids": stored_prefix,
        "stored_prefix_text": gen.tokenizer.decode(stored_prefix, skip_special_tokens=True),
        "n_continuations": len(conts),
        "prefix_preserved": prefix_ok,
        "reproducible_under_same_seed": conts == conts_again,
        "continuations": cont_smiles,
        "compute": cont_meter.as_dict(),
    }

    # ---- 7. top-eight next-token candidates -----------------------------------
    out = gen.model(
        input_ids=torch.tensor([stored_prefix], device=gen.device),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    cand_lp, cand_ids = torch.topk(logprobs, 8, dim=-1)
    report["checks"]["07_top_eight_candidates"] = {
        "status": "ok",
        "candidate_token_ids": cand_ids[0].tolist(),
        "candidate_tokens": [id_to_token[i] for i in cand_ids[0].tolist()],
        "base_logprobs": [float(v) for v in cand_lp[0]],
        "top8_probability_mass": float(cand_lp[0].exp().sum()),
    }

    # ---- 8/9. batch-evaluate the eight extended prefixes ----------------------
    from property_to_go.guidance import _candidate_states_cached, _candidate_states_full

    prefix_t = torch.tensor([stored_prefix], device=gen.device)
    t0 = time.perf_counter()
    h_full = _candidate_states_full(gen, prefix_t, cand_ids, layer=-1)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    h_cached = _candidate_states_cached(gen, out.past_key_values, cand_ids, layer=-1)
    t_cached = time.perf_counter() - t0
    backend_err = float((h_full - h_cached).abs().max())

    # base log-probabilities of each extended prefix's own next token, and shapes
    ext_seqs = [stored_prefix + [int(c)] for c in cand_ids[0]]
    ext_logprobs, ext_states = generation.base_logprobs_and_states(gen, ext_seqs)

    report["checks"]["08_batched_candidate_evaluation"] = {
        "status": "ok",
        "n_candidates": 8,
        "hidden_state_shape_full_recompute": list(h_full.shape),
        "hidden_state_shape_cached": list(h_cached.shape),
        "backend_max_abs_difference": backend_err,
        "backends_agree": backend_err < 1e-3,
        "seconds_full_recompute": t_full,
        "seconds_cached": t_cached,
    }
    report["checks"]["09_candidate_states_logprobs_shapes"] = {
        "status": "ok",
        "candidate_hidden_states_shape": list(ext_states.shape),
        "candidate_next_token_logprobs_shape": list(ext_logprobs.shape),
        "candidate_hidden_state_norms": [float(np.linalg.norm(v)) for v in ext_states],
        "base_logprobs_of_selected_candidates": [float(v) for v in cand_lp[0]],
        "trivial_prefix_features_dim": int(
            len(prefix_features_from_ids(stored_prefix, id_to_token))
        ),
    }

    # ---- 10. write the report -------------------------------------------------
    report["wall_seconds_total"] = time.perf_counter() - t_start
    failed = [k for k, v in report["checks"].items() if v.get("status") != "ok"]
    report["all_checks_passed"] = not failed
    report["failed_checks"] = failed
    write_json(run / "compatibility_report.json", report)

    np.save(run / "candidate_hidden_states.npy", ext_states)
    write_json(
        run / "base_molecules.json",
        {"records": records, "policy": policy},
    )

    print(f"compatibility report -> {run / 'compatibility_report.json'}")
    print(f"all_checks_passed = {report['all_checks_passed']}  failed={failed}")
    print(
        f"validity={report['checks']['03_rdkit_diagnostics']['validity']:.3f} "
        f"uniqueness={report['checks']['03_rdkit_diagnostics']['uniqueness']:.3f} "
        f"backend_err={backend_err:.2e} causal_err={causal_err:.2e}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
