"""Phase 2 -- steering headroom, the head-free and lambda-free ceiling.

The highest-value new measurement (docs/TODO.md C6/C7). At each held-out prefix, for
each of the base model's top-8 candidate next tokens, estimate the expected final
property by base-policy rollouts and take the spread across candidates. Because any
decoding rule at that position can only choose among those candidates, the spread bounds
what *any* such rule could achieve -- so it separates "there is no lever to pull" from
"our head is bad", the question the pilot could not answer.

No new inference machinery: `generation.continue_from_prefixes` already does the work,
called on the extended prefixes `x_{<=t} + a_i` as ordinary prefixes. That matters
beyond convenience -- it means `test_candidate_backends_agree` still covers the
numerics underneath this measurement.

One rollout pass serves every property, exactly as the pilot's Phase 4 bank does: the
continuations are property-free, and the properties are computed from the completed
molecules afterwards.

    python scripts/11_steering_headroom.py --dataset pilot_50k \
        --heads pilot_50k_heads_p2 --n-prefixes 400 --n-rollouts 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generation, headroom as H  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import balanced_position_sample  # noqa: E402
from property_to_go.properties import (  # noqa: E402
    LOCALITY_BATTERY, compute_all_properties,
)


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--heads", default=None)
    ap.add_argument("--n-prefixes", type=int, default=400)
    ap.add_argument("--n-rollouts", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=None, help="default: guidance config")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutation replicates for the noise null")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--min-rollouts", type=int, default=4,
                    help="a candidate needs this many usable rollouts to be scored")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / (args.heads or f"{args.dataset}_heads")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_headroom")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    intervals = read_json(data_dir / "target_intervals.json")
    top_k = args.top_k or int(gcfg["top_k_candidates"])
    lam = float(gcfg["lam"])
    eps = float(gcfg["eps"])

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    prefix_ids_all = read_json(data_dir / "prefix_token_ids.json")

    # Held-out prefixes only, balanced across position quartiles. The seed differs from
    # the Phase 4 rollout bank's (4242) on purpose: an independently drawn sample means
    # headroom and the predictability curve are not two views of the same 800 prefixes.
    test = np.flatnonzero(meta["split"].to_numpy() == "test")
    rng = np.random.default_rng(args.seed)
    chosen = test[
        balanced_position_sample(
            meta["prefix_len"].to_numpy()[test],
            meta["quartile"].to_numpy()[test],
            args.n_prefixes,
            rng,
        )
    ]
    quartile = meta["quartile"].to_numpy()[chosen]
    rel_pos = meta["relative_position"].to_numpy()[chosen]
    n_pref = len(chosen)
    print(f"{n_pref} held-out prefixes, quartile counts "
          f"{np.bincount(quartile, minlength=5)[1:].tolist()}")

    gen = load_generator(model_cfg)
    prefixes = [prefix_ids_all[i] for i in chosen]
    t_start = time.perf_counter()

    # ---- the candidate set the decoder would see -------------------------------
    cand_meter = ComputeMeter().start()
    cand_ids, cand_lp = generation.top_k_next_tokens(
        gen, prefixes, top_k, temperature=float(policy["temperature"]), meter=cand_meter
    )
    cand_meter.stop()
    base_w = H.candidate_weights(cand_lp)
    print(f"top-{top_k} candidates at {n_pref} prefixes "
          f"({cand_meter.processed_tokens_actual} tokens, {cand_meter.wall_seconds:.1f}s)")

    # ---- roll out from every extended prefix -----------------------------------
    extended = [list(prefixes[i]) + [int(cand_ids[i, j])]
                for i in range(n_pref) for j in range(top_k)]
    roll_meter = ComputeMeter().start()
    conts = generation.continue_from_prefixes(
        gen, extended, args.n_rollouts, policy, seed=args.seed, meter=roll_meter,
        batch_size=int(policy["batch_size"]),
    )
    roll_meter.stop()
    print(f"{len(extended) * args.n_rollouts} continuations "
          f"({roll_meter.processed_tokens_actual} tokens, {roll_meter.wall_seconds:.1f}s)")

    # ---- properties of the completed molecules ---------------------------------
    # values[prop][i][j] is the array of usable rollout values for prefix i, candidate j
    values: dict[str, list[list[np.ndarray]]] = {p: [[] for _ in range(n_pref)]
                                                 for p in LOCALITY_BATTERY}
    n_valid = np.zeros((n_pref, top_k), dtype=np.int64)
    n_rollout_total = 0
    for i in range(n_pref):
        for j in range(top_k):
            rows = conts[i * top_k + j]
            n_rollout_total += len(rows)
            props = [compute_all_properties(s) for s in gen.decode(rows)]
            ok = [p for p in props if p is not None]
            n_valid[i, j] = len(ok)
            for prop in LOCALITY_BATTERY:
                vals = [p[prop] for p in ok if p.get(prop) is not None]
                values[prop][i].append(np.asarray(vals, dtype=np.float64))
    print(f"rollout validity {n_valid.sum() / n_rollout_total:.4f}")

    # ---- the head's view of the same candidates, for the capture analysis -------
    # Hidden state of `prefix + a` at its last position: the identical quantity the
    # decoder scores, obtained through the tested right-padded batched path.
    hs_meter = ComputeMeter().start()
    ext_states = generation.hidden_states_for_positions(
        gen, extended, [[len(s) - 1] for s in extended], meter=hs_meter
    )
    hs_meter.stop()
    ext_hidden = np.concatenate(ext_states, axis=0).astype(np.float32)
    print(f"candidate hidden states ({hs_meter.processed_tokens_actual} tokens, "
          f"{hs_meter.wall_seconds:.1f}s)")

    report: dict = {
        "dataset": args.dataset,
        "heads_dir": heads_dir.name,
        "n_prefixes": n_pref,
        "top_k": top_k,
        "n_rollouts_per_candidate": args.n_rollouts,
        "n_continuations": int(n_rollout_total),
        "rollout_validity": float(n_valid.sum() / n_rollout_total),
        "min_rollouts_to_score_a_candidate": args.min_rollouts,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "lambda": lam,
        "eps": eps,
        "quartile_counts": np.bincount(quartile, minlength=5)[1:].tolist(),
        "compute": {
            "candidates": cand_meter.as_dict(),
            "rollouts": roll_meter.as_dict(),
            "candidate_hidden_states": hs_meter.as_dict(),
            "processed_tokens_total": (
                cand_meter.processed_tokens_actual
                + roll_meter.processed_tokens_actual
                + hs_meter.processed_tokens_actual
            ),
        },
        "properties": {},
    }

    arrays: dict[str, np.ndarray] = {
        "prefix_rows": chosen, "quartile": quartile, "relative_position": rel_pos,
        "candidate_ids": cand_ids, "candidate_base_logprobs": cand_lp, "n_valid": n_valid,
    }

    for prop in LOCALITY_BATTERY:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        width = hi - lo

        mu = np.full((n_pref, top_k), np.nan)
        p_hit = np.full((n_pref, top_k), np.nan)
        for i in range(n_pref):
            for j in range(top_k):
                v = values[prop][i][j]
                if len(v) >= args.min_rollouts:
                    mu[i, j] = v.mean()
                    p_hit[i, j] = float(((v >= lo) & (v < hi)).mean())

        # A prefix is usable when at least two candidates were scored: a spread over
        # one candidate is not a spread. Reported, not silently dropped.
        scored = np.isfinite(mu)
        usable = scored.sum(axis=1) >= 2

        spread_value = np.full(n_pref, np.nan)
        spread_prob = np.full(n_pref, np.nan)
        null_value = np.full(n_pref, np.nan)
        null_prob = np.full(n_pref, np.nan)
        null_available = np.full(n_pref, np.nan)
        perm_rng = np.random.default_rng(args.seed + 1)
        for i in range(n_pref):
            if not usable[i]:
                continue
            m = mu[i][scored[i]]
            spread_value[i] = m.max() - m.min()
            ph = p_hit[i][scored[i]]
            spread_prob[i] = ph.max() - ph.min()
            per_cand = [values[prop][i][j] for j in range(top_k) if scored[i, j]]
            # The noise floor in property units...
            null_value[i] = H.permutation_null_spread(per_cand, args.n_perm, perm_rng)
            # ...and in probability units, which needs the hit indicators rather than
            # the values. `max - min` over k noisy rates is biased upward just as the
            # mean spread is, and with K rollouts a rate can only take K+1 values, so
            # the bias is larger here, not smaller.
            per_cand_hit = [((v >= lo) & (v < hi)).astype(np.float64) for v in per_cand]
            null_prob[i] = H.permutation_null_spread(per_cand_hit, args.n_perm, perm_rng)
            # The bias in `max_i p_i - base`, for the capture denominator.
            null_available[i] = H.permutation_null_ceiling(
                per_cand_hit, base_w[i][scored[i]], args.n_perm, perm_rng
            )

        # The head's P(y in I | prefix + a), and hence the weights the decoder would
        # have used. Only affects the *achieved* side; the ceiling stays head-free.
        head, binner = load_head(heads_dir / f"head_{prop}_frozen_state.pt")
        q = interval_probability(head.predict_proba(ext_hidden), binner, lo, hi)
        q = q.reshape(n_pref, top_k)
        guided_w = H.guided_weights(cand_lp, q, lam, eps)

        # Capture is only defined where every candidate has a probability estimate.
        p_hit_filled = np.where(scored, p_hit, 0.0)
        capture_usable = usable & scored.all(axis=1)

        report["properties"][prop] = {
            "target_interval": iv,
            "interval_width": width,
            "n_prefixes_usable": int(usable.sum()),
            "n_prefixes_capture_usable": int(capture_usable.sum()),
            "mean_head_target_prob": float(np.mean(q)),
            "headroom": H.summarise_headroom(spread_value, null_value, width, quartile),
            # Probability units are band-width free, so they are the measure that stays
            # meaningful when two properties' target bands differ in width by 400x
            # (QED 0.046 against TPSA 18.79). Interval width of 1.0 makes the
            # normalisation a no-op, so this is reported alongside rather than instead.
            "headroom_probability_units": H.summarise_headroom(
                spread_prob, null_prob, 1.0, quartile
            ),
            "capture": H.summarise_capture(
                p_hit_filled, base_w, guided_w, quartile, capture_usable,
                available_null=null_available,
            ),
        }

        arrays[f"mu_{prop}"] = mu
        arrays[f"p_hit_{prop}"] = p_hit
        arrays[f"spread_value_{prop}"] = spread_value
        arrays[f"spread_prob_{prop}"] = spread_prob
        arrays[f"null_value_{prop}"] = null_value
        arrays[f"null_prob_{prop}"] = null_prob
        arrays[f"null_available_{prop}"] = null_available
        arrays[f"head_q_{prop}"] = q

        o = report["properties"][prop]["headroom"]["overall"]
        pu = report["properties"][prop]["headroom_probability_units"]["overall"]
        c = report["properties"][prop]["capture"]["overall"]
        cap = c["captured_fraction"]
        print(f"{prop:16s} w={width:7.3f} | rel_headroom raw={o['relative_headroom_raw_mean']:6.3f} "
              f"excess={o['relative_headroom_excess_mean']:6.3f} | "
              f"prob spread raw={pu['headroom_raw_mean']:.3f} excess={pu['headroom_excess_mean']:.3f} | "
              f"base={c['base_policy_target_prob']:.3f} guided={c['guided_target_prob']:.3f} "
              f"best={c['best_candidate_target_prob']:.3f} "
              f"avail={c['available_excess_mean']:+.4f} "
              f"captured={'n/a' if cap is None else f'{cap:+.3f}'}")

    report["wall_seconds_total"] = time.perf_counter() - t_start
    np.savez_compressed(out_dir / "headroom_arrays.npz", **arrays)
    write_json(out_dir / "headroom_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
