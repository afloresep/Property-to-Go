"""C17 step 3 -- what a different probe layer would be worth for STEERING, not prediction.

§15.6 decomposes the λ=1 loss: λ=1 permits 32.6–53.2% of the head-free ceiling and our
head collects 11.9–21.5% of what λ=1 permits.  C17's second question is whether a
different probe layer recovers any of that head term.

**No generation happens here.**  The rollouts, the candidate sets, the realised
per-candidate hit rates `p_hit`, `n_valid` and the permutation nulls are all read from
`outputs/pilot_50k_p2_headroom/headroom_arrays.npz`, exactly as
`scripts/12_locality_scatter.py` reads them.  The only thing recomputed is the head's own
view of the eight candidates, `head_q`, which needs one forward pass over the 3,200
extended prefixes `x_{<=t} + a_i` -- and that pass returns every probe point at once.

Everything else is held at phase 2's values: the same 400 prefixes (their row indices are
stored in the npz, so they are not re-derived), the same candidate token ids, λ, ε, and
the same `use` mask (all eight candidates scored) that defines §15.6's 267-prefix set.

Two consistency gates, both checked before anything is reported:

  * the extended-prefix states at probe point 12 must reproduce `head_q_<prop>` in the
    npz to float32 tolerance -- i.e. this script's head_q pipeline is script 11's;
  * `our_head_gain` at probe point 12 must reproduce
    `outputs/pilot_50k_p2_locality/locality_metrics.json` to 1e-9.

`our_head_share_of_the_lambda1_optimum` is `our_head_gain / oracle_head_gain`, and the
oracle term is **head-free**, so it is read from the locality artefact rather than
recomputed: recomputing it would re-draw its permutation null and make the layer columns
differ for a reason that has nothing to do with the layer.

    .venv/bin/python scripts/16_layer_steering_value.py --dataset pilot_50k_p2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import headroom as H  # noqa: E402
from property_to_go import probe_layers as P  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import PREDICTED_LOCALITY_ORDER  # noqa: E402


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--headroom", default=None)
    ap.add_argument("--locality", default=None)
    ap.add_argument("--sweep", default="c17_probe_layers")
    ap.add_argument("--properties", nargs="*", default=list(PREDICTED_LOCALITY_ORDER))
    ap.add_argument("--out", default="c17_layer_steering")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    hr_dir = OUTPUT_DIR / (args.headroom or f"{args.dataset}_headroom")
    loc_path = OUTPUT_DIR / (args.locality or f"{args.dataset}_locality") / "locality_metrics.json"
    sweep_dir = OUTPUT_DIR / args.sweep
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    gcfg = load_config("guidance")
    lam, eps = float(gcfg["lam"]), float(gcfg["eps"])
    intervals_cfg = read_json(data_dir / "target_intervals.json")

    arr = np.load(hr_dir / "headroom_arrays.npz")
    locality = read_json(loc_path)
    sweep = read_json(sweep_dir / "probe_layer_metrics.json")
    layers = [int(L) for L in sweep["probe_points"]]
    reference_layer = int(sweep["reference_probe_point"])

    prefix_rows = arr["prefix_rows"]
    cand_ids = arr["candidate_ids"]
    cand_lp = arr["candidate_base_logprobs"]
    n_pref, top_k = cand_ids.shape

    prefix_ids_all = read_json(data_dir / "prefix_token_ids.json")
    extended = [list(prefix_ids_all[int(prefix_rows[i])]) + [int(cand_ids[i, j])]
                for i in range(n_pref) for j in range(top_k)]

    gen = load_generator(model_cfg)
    meter = ComputeMeter().start()
    t0 = time.perf_counter()
    states = P.hidden_states_all_layers(
        gen, extended, [[len(s) - 1] for s in extended], layers, meter=meter,
    )
    meter.stop()
    ext_hidden = {L: np.concatenate(states[L], axis=0).astype(np.float32) for L in layers}
    print(f"{len(extended)} extended prefixes x {len(layers)} probe points in "
          f"{time.perf_counter() - t0:.1f}s ({meter.processed_tokens_actual} tokens)")

    bw = H.candidate_weights(cand_lp)

    report: dict = {
        "dataset": args.dataset,
        "headroom_dir": hr_dir.name,
        "sweep_dir": sweep_dir.name,
        "n_prefixes_sampled": int(n_pref),
        "top_k": int(top_k),
        "lambda": lam,
        "eps": eps,
        "probe_points": layers,
        "reference_probe_point": reference_layer,
        "compute": meter.as_dict(),
        "properties": {},
        "note": (
            "No generation. p_hit / n_valid / the permutation nulls and the head-free "
            "oracle term are phase 2's; only the head's head_q is recomputed per layer."
        ),
    }

    gates: dict = {"head_q_matches_headroom_npz": {}, "our_head_gain_matches_locality": {}}

    for prop in args.properties:
        iv = intervals_cfg[prop]
        p_hit = arr[f"p_hit_{prop}"]
        scored = np.isfinite(p_hit)
        p_filled = np.where(scored, p_hit, 0.0)
        use = (scored.sum(axis=1) >= 2) & scored.all(axis=1)
        base_p = (bw * p_filled).sum(axis=1)

        ref = locality["properties"][prop]["lambda1_ceiling_analysis"]
        oracle_gain = float(ref["oracle_head_gain"])
        ceiling_gain = float(ref["noise_corrected_ceiling_gain"])

        rows: dict = {}
        for L in layers:
            head, binner = load_head(sweep_dir / f"head_{prop}_frozen_state_L{L}.pt")
            q = interval_probability(
                head.predict_proba(ext_hidden[L]), binner, iv["lo"], iv["hi"]
            ).reshape(n_pref, top_k)
            ach = (H.guided_weights(cand_lp, q, lam, eps) * p_filled).sum(axis=1) - base_p
            gain = float(ach[use].mean())
            rows[str(L)] = {
                "probe_point": int(L),
                "our_head_gain": gain,
                "our_head_share_of_the_lambda1_optimum": (
                    gain / oracle_gain if oracle_gain else None
                ),
                "fraction_of_the_head_free_ceiling": (
                    gain / ceiling_gain if ceiling_gain else None
                ),
                "mean_head_q": float(q.mean()),
                "mean_head_q_spread_across_candidates": float(
                    (q.max(axis=1) - q.min(axis=1)).mean()
                ),
            }
            if L == reference_layer:
                stored_q = arr[f"head_q_{prop}"]
                gates["head_q_matches_headroom_npz"][prop] = {
                    "max_abs_difference": float(np.abs(q - stored_q).max()),
                    "allclose_1e-5": bool(np.allclose(q, stored_q, atol=1e-5, rtol=0)),
                }
                gates["our_head_gain_matches_locality"][prop] = {
                    "reference": ref["our_head_gain"],
                    "recomputed": gain,
                    "abs_difference": abs(gain - float(ref["our_head_gain"])),
                }

        report["properties"][prop] = {
            "n_prefixes_used": int(use.sum()),
            "oracle_head_gain_lambda1": oracle_gain,
            "noise_corrected_ceiling_gain": ceiling_gain,
            "reference_our_head_gain": float(ref["our_head_gain"]),
            "reference_share": ref["our_head_share_of_the_lambda1_optimum"],
            "layers": rows,
        }
        best = max(rows, key=lambda k: rows[k]["our_head_gain"])
        print(f"{prop:16s} L{reference_layer} gain={rows[str(reference_layer)]['our_head_gain']:+.5f}"
              f"  best L{best} gain={rows[best]['our_head_gain']:+.5f}"
              f"  share {rows[str(reference_layer)]['our_head_share_of_the_lambda1_optimum']:.3f}"
              f" -> {rows[best]['our_head_share_of_the_lambda1_optimum']:.3f}")

    report["consistency_gates"] = gates
    print("\nconsistency gates:")
    for prop in args.properties:
        g1 = gates["head_q_matches_headroom_npz"][prop]
        g2 = gates["our_head_gain_matches_locality"][prop]
        print(f"  {prop:16s} head_q Δmax={g1['max_abs_difference']:.2e} "
              f"our_head_gain Δ={g2['abs_difference']:.2e}")

    # ---- the pre-registered C17.0.5 criterion --------------------------------------
    # The layer is selected by PREDICTION (argmax AUROC from the sweep), then evaluated
    # for steering. Selecting it on the steering number would be the free parameter this
    # whole design exists to remove.
    per_prop = {}
    for prop in args.properties:
        l_star = int(sweep["verdicts"][prop]["argmax_layer_by_auroc"])
        g_star = report["properties"][prop]["layers"][str(l_star)]["our_head_gain"]
        g_ref = report["properties"][prop]["layers"][str(reference_layer)]["our_head_gain"]
        per_prop[prop] = {
            "L_star_selected_by_auroc": l_star,
            "our_head_gain_at_L_star": g_star,
            "our_head_gain_at_reference": g_ref,
            "absolute_improvement": g_star - g_ref,
            "relative_improvement": (g_star - g_ref) / g_ref if g_ref else None,
            "share_at_L_star": (
                report["properties"][prop]["layers"][str(l_star)]
                ["our_head_share_of_the_lambda1_optimum"]
            ),
            "share_at_reference": (
                report["properties"][prop]["layers"][str(reference_layer)]
                ["our_head_share_of_the_lambda1_optimum"]
            ),
        }
    rel = [v["relative_improvement"] for v in per_prop.values()
           if v["relative_improvement"] is not None]
    n_better = sum(1 for v in per_prop.values() if v["absolute_improvement"] > 0)
    median_rel = float(np.median(rel)) if rel else None
    report["c17_0_5_criterion"] = {
        "rule": ("material iff our_head_gain at L* (selected by AUROC) exceeds the "
                 "reference layer's for >= 4 of 6 properties AND the median relative "
                 "improvement is >= 0.25"),
        "per_property": per_prop,
        "n_properties_improved": int(n_better),
        "n_properties": len(per_prop),
        "median_relative_improvement": median_rel,
        "material": bool(n_better >= 4 and median_rel is not None and median_rel >= 0.25),
    }
    print(f"\nC17.0.5: improved on {n_better}/{len(per_prop)} properties, "
          f"median relative improvement {median_rel:+.3f} -> "
          f"{'MATERIAL' if report['c17_0_5_criterion']['material'] else 'NOT MATERIAL'}")

    report["wall_seconds_total"] = time.perf_counter() - t0
    write_json(out_dir / "layer_steering_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "guidance": gcfg})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
