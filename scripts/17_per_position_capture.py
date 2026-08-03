"""C18 step 4a -- per-position capture for an arbitrary head or calibrator.

The expensive half of `pilot_report.md` §15.6 is already on disk: 51,200 rollouts gave
`p_hit[i, j]`, the realised P(y_final in I | prefix_i + candidate_j), plus the
permutation nulls, in `outputs/pilot_50k_p2_headroom/headroom_arrays.npz`.  The only
thing that changes when the head changes is `q[i, j]`, and recomputing that needs one
forward pass over 400 x 8 extended prefixes -- no rollouts at all.

So this script rebuilds the *identical* 3,200 extended prefixes script 11 used, checks
that it did so by comparing the recovered candidate ids and base log-probabilities
against the stored arrays bit for bit, and then scores any number of heads and
calibrators against them.  `our_head_gain` and
`our_head_share_of_the_lambda1_optimum` are computed by exactly the expressions in the
`lambda1_ceiling_analysis` block of `scripts/12_locality_scatter.py`, and the baseline
head is always evaluated alongside so the reproduction of §15.6's published value is
visible in the output rather than assumed.

**A per-position improvement is not an end-to-end improvement** (`docs/TODO.md` C22.1).
Nothing here may be extrapolated; `scripts/05` and `scripts/06` measure the end-to-end
quantity and this script writes `MUST_BE_MEASURED_END_TO_END` into its own output as a
standing reminder.

    .venv/bin/python scripts/17_per_position_capture.py --variants baseline wide focused
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import calibration as C, generation, headroom as H  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import balanced_position_sample  # noqa: E402
from property_to_go.properties import LOCALITY_BATTERY  # noqa: E402


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"])


def gain_from_q(
    q: np.ndarray,
    cand_lp: np.ndarray,
    p_filled: np.ndarray,
    base_p: np.ndarray,
    use: np.ndarray,
    lam: float,
    eps: float,
) -> np.ndarray:
    """`achieved` per prefix: the deployed softmax's expected hit probability minus base.

    Identical expression to `ach` in `scripts/12_locality_scatter.py`.
    """
    w = H.guided_weights(cand_lp, q, lam, eps)
    return (w * p_filled).sum(axis=1) - base_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--headroom", default="pilot_50k_p2_headroom")
    ap.add_argument("--locality", default="pilot_50k_p2_locality")
    ap.add_argument("--baseline-heads", default="pilot_50k_heads_p2")
    ap.add_argument("--variants", nargs="*", default=["baseline"],
                    help="`baseline`, or c18 head directories under outputs/")
    ap.add_argument("--calibrators", default="c18_offpolicy_calibration",
                    help="directory holding calibrator_<prop>.json, or 'none'")
    ap.add_argument("--properties", nargs="*", default=list(LOCALITY_BATTERY))
    ap.add_argument("--seed", type=int, default=7777,
                    help="must match scripts/11's prefix seed to hit the same prefixes")
    ap.add_argument("--n-prefixes", type=int, default=400)
    ap.add_argument("--out", default="c18_per_position")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    hr_dir = OUTPUT_DIR / args.headroom
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    lam, eps = float(gcfg["lam"]), float(gcfg["eps"])
    top_k = int(gcfg["top_k_candidates"])
    intervals = read_json(data_dir / "target_intervals.json")
    hr_arrays = np.load(hr_dir / "headroom_arrays.npz", allow_pickle=True)
    hr_metrics = read_json(hr_dir / "headroom_metrics.json")
    locality = read_json(OUTPUT_DIR / args.locality / "locality_metrics.json")["properties"]

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    prefix_ids_all = read_json(data_dir / "prefix_token_ids.json")

    # ---- rebuild script 11's prefix sample, then PROVE it is the same ----------
    test = np.flatnonzero(meta["split"].to_numpy() == "test")
    rng = np.random.default_rng(args.seed)
    chosen = test[
        balanced_position_sample(
            meta["prefix_len"].to_numpy()[test], meta["quartile"].to_numpy()[test],
            args.n_prefixes, rng,
        )
    ]
    stored_rows = hr_arrays["prefix_rows"]
    if not np.array_equal(chosen, stored_rows):
        raise SystemExit(
            "prefix reconstruction does not match headroom_arrays.npz -- refusing to "
            "score a head against rollouts that belong to different prefixes"
        )

    gen = load_generator(model_cfg)
    prefixes = [prefix_ids_all[i] for i in chosen]
    t_start = time.perf_counter()

    cand_meter = ComputeMeter().start()
    cand_ids, cand_lp = generation.top_k_next_tokens(
        gen, prefixes, top_k, temperature=float(policy["temperature"]), meter=cand_meter
    )
    cand_meter.stop()
    reproduction = {
        "candidate_ids_identical": bool(np.array_equal(cand_ids, hr_arrays["candidate_ids"])),
        "base_logprob_max_abs_difference": float(
            np.abs(cand_lp - hr_arrays["candidate_base_logprobs"]).max()
        ),
    }
    if not reproduction["candidate_ids_identical"]:
        raise SystemExit("candidate set does not reproduce; see reproduction block")

    extended = [list(prefixes[i]) + [int(cand_ids[i, j])]
                for i in range(len(prefixes)) for j in range(top_k)]
    hs_meter = ComputeMeter().start()
    ext_states = generation.hidden_states_for_positions(
        gen, extended, [[len(s) - 1] for s in extended], meter=hs_meter
    )
    hs_meter.stop()
    ext_hidden = np.concatenate(ext_states, axis=0).astype(np.float32)
    n_pref = len(prefixes)

    calib_dir = None if args.calibrators == "none" else OUTPUT_DIR / args.calibrators

    report: dict = {
        "dataset": args.dataset,
        "headroom_dir": args.headroom,
        "lambda": lam, "eps": eps, "top_k": top_k,
        "n_prefixes_sampled": n_pref,
        "prefix_reconstruction": reproduction,
        "variants": list(args.variants),
        "calibrators_from": args.calibrators,
        "MUST_BE_MEASURED_END_TO_END": (
            "Everything below is a per-DECODING-POSITION gain with the rest of the "
            "sequence left to the base policy. docs/TODO.md C22.1: end-to-end lift is "
            "20-48x the per-step gain and the ratios do NOT transfer linearly. No "
            "number here may be converted into an end-to-end claim."
        ),
        "compute": {
            "candidates": cand_meter.as_dict(),
            "candidate_hidden_states": hs_meter.as_dict(),
            "processed_tokens_total": (
                cand_meter.processed_tokens_actual + hs_meter.processed_tokens_actual
            ),
        },
        "properties": {},
    }

    for prop in args.properties:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        p_hit = hr_arrays[f"p_hit_{prop}"]
        scored = np.isfinite(p_hit)
        p_filled = np.where(scored, p_hit, 0.0)
        use = (scored.sum(axis=1) >= 2) & scored.all(axis=1)
        bw = H.candidate_weights(cand_lp)
        base_p = (bw * p_filled).sum(axis=1)
        ceil_excess = (p_filled.max(axis=1) - base_p
                       - np.nan_to_num(hr_arrays[f"null_available_{prop}"]))

        published = locality[prop]["lambda1_ceiling_analysis"]
        oracle_gain = float(published["oracle_head_gain"])
        ceiling_gain = float(published["noise_corrected_ceiling_gain"])

        entry: dict = {
            "target_interval": iv,
            "n_prefixes_scored": int(use.sum()),
            "published_reference": {
                "our_head_gain": published["our_head_gain"],
                "oracle_head_gain": oracle_gain,
                "noise_corrected_ceiling_gain": ceiling_gain,
                "our_head_share_of_the_lambda1_optimum":
                    published["our_head_share_of_the_lambda1_optimum"],
                "fraction_of_ceiling_lambda1_permits":
                    published["fraction_of_ceiling_lambda1_permits"],
                "source": f"{args.locality}/locality_metrics.json",
            },
            "arms": {},
        }

        def record(name: str, q: np.ndarray, extra: dict | None = None) -> None:
            ach = gain_from_q(q, cand_lp, p_filled, base_p, use, lam, eps)
            w = H.guided_weights(cand_lp, q, lam, eps)
            rows = np.arange(len(q))
            argmax_p = p_filled.argmax(axis=1)
            # Tie-robust: "does the head give the oracle's best candidate the maximum
            # score", not "does argmax return the same index". Isotonic calibration is
            # only WEAKLY monotone -- it creates flat steps -- so an index comparison
            # would report a reordering that did not happen.
            picks_best = q[rows, argmax_p] >= q.max(axis=1) - 1e-12
            entry["arms"][name] = {
                "our_head_gain": float(ach[use].mean()),
                "our_head_share_of_the_lambda1_optimum": (
                    float(ach[use].sum() / (oracle_gain * int(use.sum())))
                    if oracle_gain > 0 else None
                ),
                "our_head_share_of_the_ceiling": (
                    float(ach[use].sum() / (ceiling_gain * int(use.sum())))
                    if ceiling_gain > 0 else None
                ),
                "mean_head_target_prob": float(q[use].mean()),
                # the only thing a monotone-in-q calibration can never change
                "picks_the_best_candidate_rate": float(picks_best[use].mean()),
                "mean_weight_on_the_best_candidate": float(
                    w[np.arange(len(w)), argmax_p][use].mean()
                ),
                **(extra or {}),
            }

        # ---- baseline and retrained heads -----------------------------------
        for variant in args.variants:
            hd = (OUTPUT_DIR / args.baseline_heads if variant == "baseline"
                  else OUTPUT_DIR / variant)
            path = hd / f"head_{prop}_frozen_state.pt"
            if not path.exists():
                continue
            head, binner = load_head(path)
            q = interval_probability(head.predict_proba(ext_hidden), binner, lo, hi)
            q = q.reshape(n_pref, top_k)
            record(variant, q, {"head_dir": hd.name, "n_bins": int(binner.n_bins),
                                "n_parameters": int(sum(p.numel() for p in head.parameters()))})
            if variant == "baseline":
                q_base_head = q
                head_base, binner_base = head, binner

        # ---- post-hoc calibrators on the BASELINE head ------------------------
        if calib_dir is not None and (calib_dir / f"calibrator_{prop}.json").exists():
            cal = read_json(calib_dir / f"calibrator_{prop}.json")
            platt = C.calibrator_from_dict(cal["platt"])
            iso = C.calibrator_from_dict(cal["isotonic"])
            record("baseline_platt", platt.apply(q_base_head),
                   {"calibrator": cal["platt"],
                    "equivalent_lambda": platt.power_limit().equivalent_lambda(lam)})
            record("baseline_isotonic", iso.apply(q_base_head),
                   {"calibrator_kind": "isotonic", "n_knots": len(cal["isotonic"]["x"])})

            T = float(cal["bin_logit_temperature"])
            mask = np.asarray(binner_base.interval_mask(lo, hi), dtype=bool)
            with torch.no_grad():
                z = head_base(torch.as_tensor(ext_hidden, dtype=torch.float32)).numpy()

            def q_at_temperature(t: float) -> np.ndarray:
                zz = z / float(t)
                zz = zz - zz.max(axis=1, keepdims=True)
                pz = np.exp(zz)
                pz /= pz.sum(axis=1, keepdims=True)
                return pz[:, mask].sum(axis=1).reshape(n_pref, top_k)

            record("baseline_bin_temperature", q_at_temperature(T),
                   {"bin_logit_temperature": T,
                    "selected_by": "lowest ECE on held-out guided prefixes"})

            # Is the ECE-selected temperature the one that helps the DECODER? The two
            # objectives are not the same and there is no reason they should agree, so
            # the whole grid is swept here and the disagreement is reported rather than
            # left implicit in a single selected value.
            sweep = {}
            for t in (0.4, 0.5, 0.625, 0.75, 0.9, 1.0, 1.25, 1.6, 2.0, 3.0, 4.0):
                a = gain_from_q(q_at_temperature(t), cand_lp, p_filled, base_p, use,
                                lam, eps)
                sweep[str(t)] = float(a[use].mean())
            best_t = max(sweep, key=lambda k: sweep[k])
            entry["bin_temperature_sweep"] = {
                "gain_by_temperature": sweep,
                "best_temperature_for_the_decoder": float(best_t),
                "best_gain": sweep[best_t],
                "ece_selected_temperature": T,
                "gain_at_the_ece_selected_temperature": sweep.get(str(T)),
                "note": (
                    "The temperature that minimises calibration error and the one that "
                    "maximises per-position gain are different objectives; where they "
                    "disagree, that disagreement IS the C18 finding."
                ),
            }

            # Platt is a power map to first order, so evaluate the lambda it claims to
            # be equivalent to, exactly, rather than interpolating the sweep below.
            alpha = float(cal["platt"]["a"])
            a_eq = gain_from_q(q_base_head, cand_lp, p_filled, base_p, use,
                               lam * alpha, eps)
            entry["lambda_rescale_at_the_platt_equivalent"] = {
                "equivalent_lambda": lam * alpha,
                "our_head_gain": float(a_eq[use].mean()),
                "platt_calibrated_gain": entry["arms"]["baseline_platt"]["our_head_gain"],
                "difference": float(
                    a_eq[use].mean() - entry["arms"]["baseline_platt"]["our_head_gain"]
                ),
            }

            # the algebraic identity, checked on the real candidate array
            entry["power_calibration_is_a_lambda_rescale"] = C.equivalent_lambda_is_exact(
                cand_lp, q_base_head, float(cal["platt"]["a"]), float(cal["platt"]["b"]),
                lam, eps,
            )

        # ---- what a lambda rescale of the BASELINE head buys, for comparison --
        entry["lambda_rescale_of_the_baseline_head"] = {}
        for lam_alt in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            ach = gain_from_q(q_base_head, cand_lp, p_filled, base_p, use, lam_alt, eps)
            entry["lambda_rescale_of_the_baseline_head"][str(lam_alt)] = {
                "our_head_gain": float(ach[use].mean()),
                "share_of_lambda1_optimum": (
                    float(ach[use].sum() / (oracle_gain * int(use.sum())))
                    if oracle_gain > 0 else None
                ),
            }

        report["properties"][prop] = entry
        arms = entry["arms"]
        print(f"{prop:16s} " + "  ".join(
            f"{k}={v['our_head_gain']:+.4f}" for k, v in arms.items()
        ))

    report["wall_seconds_total"] = time.perf_counter() - t_start
    write_json(out_dir / "per_position_capture.json", report)
    write_json(out_dir / "configs_used.json",
               {"model": model_cfg, "base_policy": policy, "guidance": gcfg})
    write_run_context(out_dir)
    print(f"reproduced script 11's prefix set: {reproduction}")
    print(f"-> {out_dir}")
    del hr_metrics
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
