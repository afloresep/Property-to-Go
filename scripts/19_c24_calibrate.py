"""C24 stage 4 -- off-policy calibration of the text probe, and the algebraic identity.

The text-domain replication of `pilot_report.md` §20.2-§20.3 (C18).  Calibrators are
fitted on held-out **off-policy** prefixes -- prefixes of guided sequences -- split in
half by completed text so no sequence's prefixes straddle the fit and score halves.

Every fitter is the molecular one (`property_to_go.calibration`), imported.

    .venv/bin/python scripts/19_c24_calibrate.py
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
from property_to_go import metrics as M  # noqa: E402
from property_to_go.binning import binner_from_dict, in_interval, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402
from property_to_go.guidance import TargetScorer  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.splits import split_by_group  # noqa: E402

CFG = {
    "n_guided_sequences": 2000,
    "guided_seed": 91234,
    "lam": 1.0,
    "eps": 1e-6,
    "top_k": 8,
    "probe_point": 12,
    "head_seed": 1234,
    "prefixes_per_sequence": 4,
    "prefix_seed": 913,
    "calibration_split_seed": 55,
    "identity_prefixes": 200,
    "n_content": 40,
    "batch_size": 64,
}


def load_head(path: Path, device):
    d = torch.load(path, map_location="cpu")
    head = MLPHead(d["in_dim"], d["hidden_dim"], d["n_bins"], d["dropout"])
    head.load_state_dict(d["state_dict"])
    return head.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c24_dataset")
    ap.add_argument("--heads", default="c24_probe_layers")
    ap.add_argument("--out", default="c24_calibration")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = OUTPUT_DIR / args.dataset
    hd = OUTPUT_DIR / args.heads
    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"c24_calibration": CFG, "dataset": args.dataset})

    ti = read_json(ds / "target_intervals.json")
    meta = read_json(ds / "dataset_metrics.json")
    rows = np.load(ds / "rows.npz", allow_pickle=False)
    split = rows["row_split"]
    te = split == "test"
    L = CFG["probe_point"]
    Xte = np.ascontiguousarray(np.load(ds / f"layer{L}.npy", mmap_mode="r")[te])
    t0 = time.time()

    gen = G.load_text_generator(args.device)
    meter = ComputeMeter().start()
    results: dict[str, dict] = {}

    for attr, band in ti["intervals"].items():
        lo, hi = float(band["lo"]), float(band["hi"])
        binner = binner_from_dict(ti["binners"][attr])
        head = load_head(hd / "heads" / f"{attr}_L{L}_seed{CFG['head_seed']}.pt", args.device)
        scorer = TargetScorer(head, binner, lo, hi)

        # --- on policy: the dataset's own held-out base-policy prefixes -----------
        probs = G.predict_proba_on_device(head, Xte, args.device)
        q_on = interval_probability(probs, binner, lo, hi)
        hit_on = in_interval(rows[f"value_{attr}"][te], lo, hi)
        on_policy = C.calibration_report(q_on, hit_on)
        on_policy.pop("reliability", None)

        # --- off policy: prefixes of guided sequences ----------------------------
        print(f"[{time.time()-t0:.0f}s] {attr}: {CFG['n_guided_sequences']} guided sequences",
              flush=True)
        gseqs = G.guided_sample_text(
            gen, scorer, n=CFG["n_guided_sequences"], seed=CFG["guided_seed"],
            n_content=CFG["n_content"], lam=CFG["lam"], eps=CFG["eps"],
            top_k=CFG["top_k"], layer=L, batch_size=CFG["batch_size"], meter=meter,
        )
        gtexts = gen.decode([s[1:] for s in gseqs])
        gvals = np.array([G.ATTRIBUTES[attr](t) for t in gtexts], dtype=np.float64)

        rng = np.random.default_rng(CFG["prefix_seed"])
        n_pref = CFG["prefixes_per_sequence"]
        edges = np.linspace(1, CFG["n_content"] - 1, n_pref + 1)
        quart = [(int(np.ceil(edges[i])), int(np.floor(edges[i + 1]))) for i in range(n_pref)]
        gpos = np.stack([np.array([rng.integers(a, b + 1) for a, b in quart])
                         for _ in range(len(gseqs))]).astype(np.int64)
        gstates = G.all_layer_states(gen, gseqs, gpos, meter=meter)[:, :, L, :]
        gstates = gstates.reshape(-1, gen.hidden_size)
        gprobs = G.predict_proba_on_device(head, gstates, args.device)
        q_off = interval_probability(gprobs, binner, lo, hi)
        hit_off = np.repeat(in_interval(gvals, lo, hi), n_pref)
        group = np.repeat(np.array(gtexts), n_pref)
        off_policy = C.calibration_report(q_off, hit_off)
        off_policy.pop("reliability", None)

        # --- fit on half, score on the other half, grouped by completed text -----
        half = split_by_group(list(group), {"train": 0.5, "val": 0.0, "test": 0.5},
                              CFG["calibration_split_seed"])
        fit, score = half == "train", half == "test"
        platt = C.fit_platt(q_off[fit], hit_off[fit])
        iso = C.fit_isotonic(q_off[fit], hit_off[fit])
        power = platt.power_limit()
        power_fit = C.fit_power_approximation(platt, np.clip(q_off[score], 1e-9, None))

        arms = {"uncalibrated": C.IdentityCalibrator(), "platt": platt, "isotonic": iso}
        cal_metrics = {}
        for name, cal in arms.items():
            qc = cal.apply(q_off[score])
            cal_metrics[name] = {
                "ece": M.expected_calibration_error(qc, hit_off[score]),
                "auroc": M.auroc(qc, hit_off[score]),
                "brier": M.brier(qc, hit_off[score]),
                "mean_predicted": float(qc.mean()),
                "observed": float(hit_off[score].mean()),
                "calibrator": cal.to_dict(),
            }

        # --- the identity, on a real candidate array ------------------------------
        gpref = [list(gseqs[i][: int(gpos[i][1]) + 1]) for i in range(CFG["identity_prefixes"])]
        cand_ids, cand_lp, cand_states = G.top_k_candidates(gen, gpref, CFG["top_k"], meter=meter)
        cprobs = G.predict_proba_on_device(
            head, cand_states[:, :, L, :].reshape(-1, gen.hidden_size), args.device
        )
        q_cand = interval_probability(cprobs, binner, lo, hi).reshape(len(gpref), CFG["top_k"])
        identity = {
            "eps_0": C.equivalent_lambda_is_exact(
                cand_lp, q_cand, power.alpha, 0.0, CFG["lam"], 0.0),
            "eps_deployed": C.equivalent_lambda_is_exact(
                cand_lp, q_cand, power.alpha, 0.0, CFG["lam"], CFG["eps"]),
            "min_q_candidate": float(q_cand.min()),
            "n_candidate_rows": int(q_cand.size),
        }

        results[attr] = {
            "target": band,
            "on_policy": on_policy,
            "off_policy": off_policy,
            "off_policy_factor": (off_policy["observed"] / off_policy["mean_predicted"])
            if off_policy["mean_predicted"] > 0 else None,
            "n_off_policy_prefixes": int(len(q_off)),
            "n_fit": int(fit.sum()), "n_score": int(score.sum()),
            "calibrated": cal_metrics,
            "platt_slope": float(platt.a), "platt_intercept": float(platt.b),
            "power_limit": power.to_dict(),
            "power_fit_to_platt": power_fit,
            "equivalent_lambda_at_lam1": power.equivalent_lambda(1.0),
            "auroc_delta_platt": cal_metrics["platt"]["auroc"]
            - cal_metrics["uncalibrated"]["auroc"],
            "auroc_delta_isotonic": cal_metrics["isotonic"]["auroc"]
            - cal_metrics["uncalibrated"]["auroc"],
            "ece_factor_platt": cal_metrics["uncalibrated"]["ece"] / cal_metrics["platt"]["ece"],
            "ece_factor_isotonic": cal_metrics["uncalibrated"]["ece"]
            / cal_metrics["isotonic"]["ece"],
            "identity_numeric": identity,
            "guided_hit_rate": float(in_interval(gvals, lo, hi).mean()),
        }
        print(f"[{time.time()-t0:.0f}s] {attr}: platt a={platt.a:.4f} b={platt.b:+.4f}"
              f" ECE {cal_metrics['uncalibrated']['ece']:.4f}->{cal_metrics['platt']['ece']:.4f}"
              f" dAUROC {results[attr]['auroc_delta_platt']:+.6f}", flush=True)

    meter.stop()
    write_json(out / "calibration_metrics.json", {
        "config": CFG, "attributes": results, "compute": meter.as_dict(),
        "prereg_1a_all_slopes_below_one": all(r["platt_slope"] < 1.0 for r in results.values()),
        "prereg_1b_auroc_unchanged_platt": all(
            abs(r["auroc_delta_platt"]) <= 1e-4 for r in results.values()),
        "prereg_1c_ece_halved_both": all(
            r["ece_factor_platt"] >= 2.0 and r["ece_factor_isotonic"] >= 2.0
            for r in results.values()),
    })
    print(f"[{time.time()-t0:.0f}s] done; {meter.processed_tokens_actual} processed tokens",
          flush=True)


if __name__ == "__main__":
    main()
