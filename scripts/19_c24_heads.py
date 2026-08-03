"""C24 stage 2 -- the probe-depth sweep: a head at every one of the 13 probe points.

The text-domain replication of `pilot_report.md` §21 (C17).  Per-seed values are written
for every cell so the noise floor is legible rather than asserted.

    .venv/bin/python scripts/19_c24_heads.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generality as G  # noqa: E402
from property_to_go import metrics as M  # noqa: E402
from property_to_go.binning import binner_from_dict, in_interval, interval_probability  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402

HEAD_CFG = {
    "hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.01,
    "batch_size": 512, "max_epochs": 60, "patience": 8,
}
HEAD_SEEDS = (1234, 2345, 3456)
CHECKPOINT_SEED = 1234
N_BOOT = 2000
BONFERRONI_FAMILY = 13


def evaluate(probs, y, y_bin, binner, lo, hi):
    q = interval_probability(probs, binner, lo, hi)
    hit = in_interval(y, lo, hi)
    return {
        "target_auroc": M.auroc(q, hit),
        "nll": M.categorical_nll(probs, y_bin),
        "target_ece": M.expected_calibration_error(q, hit),
        "target_brier": M.brier(q, hit),
        "mean_q": float(q.mean()),
        "base_rate": float(hit.mean()),
    }, q, hit


def paired_bootstrap_auroc_diff(q_a, q_b, hit, n_boot, seed, alpha):
    rng = np.random.default_rng(seed)
    n = len(hit)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        h = hit[idx]
        if h.all() or not h.any():
            diffs[r] = 0.0
            continue
        diffs[r] = M.auroc(q_a[idx], h) - M.auroc(q_b[idx], h)
    return {
        "mean": float(diffs.mean()),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c24_dataset")
    ap.add_argument("--out", default="c24_probe_layers")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = OUTPUT_DIR / args.dataset
    out = OUTPUT_DIR / args.out
    (out / "heads").mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"head": HEAD_CFG, "head_seeds": list(HEAD_SEEDS),
                            "dataset": args.dataset, "n_boot": N_BOOT,
                            "bonferroni_family": BONFERRONI_FAMILY})

    rows = np.load(ds / "rows.npz", allow_pickle=False)
    ti = read_json(ds / "target_intervals.json")
    meta = read_json(ds / "dataset_metrics.json")
    n_probe = int(meta["n_probe_points"])
    split = rows["row_split"]
    tr, va, te = split == "train", split == "val", split == "test"
    trivial = rows["trivial"]
    t0 = time.time()

    per_layer_dir = out / "partial"
    per_layer_dir.mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    for attr, band in ti["intervals"].items():
        binner = binner_from_dict(ti["binners"][attr])
        lo, hi = float(band["lo"]), float(band["hi"])
        y = rows[f"value_{attr}"]
        y_bin = binner.transform(y)
        results[attr] = {"target": band, "n_bins": int(binner.n_bins), "probe_points": {},
                         "trivial": {}}
        q_test_mean: dict[str, np.ndarray] = {}

        # --- trivial baseline -----------------------------------------------------
        pfile = per_layer_dir / f"{attr}_trivial.json"
        if pfile.exists():
            results[attr]["trivial"] = read_json(pfile)["metrics"]
            q_test_mean["trivial"] = np.array(read_json(pfile)["q_test_mean"])
        else:
            per_seed, qs = [], []
            for seed in HEAD_SEEDS:
                head = MLPHead(trivial.shape[1], HEAD_CFG["hidden_dim"], binner.n_bins,
                               HEAD_CFG["dropout"])
                cfg = dict(HEAD_CFG, seed=seed)
                G.train_head_on_device(head, trivial[tr], y_bin[tr], trivial[va], y_bin[va],
                                       cfg, args.device)
                probs = G.predict_proba_on_device(head, trivial[te], args.device)
                m, q, hit = evaluate(probs, y[te], y_bin[te], binner, lo, hi)
                m["seed"] = seed
                per_seed.append(m)
                qs.append(q)
            results[attr]["trivial"] = summarise_seeds(per_seed)
            q_test_mean["trivial"] = np.mean(qs, axis=0)
            write_json(pfile, {"metrics": results[attr]["trivial"],
                               "q_test_mean": q_test_mean["trivial"].tolist()})
        print(f"[{time.time()-t0:.0f}s] {attr} trivial "
              f"AUROC {results[attr]['trivial']['target_auroc']['mean']:.4f}", flush=True)

        # --- every probe point ----------------------------------------------------
        for L in range(n_probe):
            pfile = per_layer_dir / f"{attr}_L{L}.json"
            if pfile.exists():
                d = read_json(pfile)
                results[attr]["probe_points"][str(L)] = d["metrics"]
                q_test_mean[str(L)] = np.array(d["q_test_mean"])
                continue
            X = np.load(ds / f"layer{L}.npy", mmap_mode="r")
            Xtr, Xva, Xte = np.ascontiguousarray(X[tr]), np.ascontiguousarray(X[va]), \
                np.ascontiguousarray(X[te])
            per_seed, qs = [], []
            for seed in HEAD_SEEDS:
                head = MLPHead(Xtr.shape[1], HEAD_CFG["hidden_dim"], binner.n_bins,
                               HEAD_CFG["dropout"])
                cfg = dict(HEAD_CFG, seed=seed)
                res = G.train_head_on_device(head, Xtr, y_bin[tr], Xva, y_bin[va], cfg,
                                             args.device)
                probs = G.predict_proba_on_device(head, Xte, args.device)
                m, q, hit = evaluate(probs, y[te], y_bin[te], binner, lo, hi)
                m["seed"] = seed
                m["best_epoch"] = res.best_epoch
                per_seed.append(m)
                qs.append(q)
                if seed == CHECKPOINT_SEED:
                    torch.save({"state_dict": head.state_dict(), "in_dim": Xtr.shape[1],
                                "hidden_dim": HEAD_CFG["hidden_dim"],
                                "n_bins": int(binner.n_bins), "dropout": HEAD_CFG["dropout"]},
                               out / "heads" / f"{attr}_L{L}_seed{seed}.pt")
            summ = summarise_seeds(per_seed)
            results[attr]["probe_points"][str(L)] = summ
            q_test_mean[str(L)] = np.mean(qs, axis=0)
            write_json(pfile, {"metrics": summ, "q_test_mean": q_test_mean[str(L)].tolist()})
            del X, Xtr, Xva, Xte
            print(f"[{time.time()-t0:.0f}s] {attr} L{L} AUROC {summ['target_auroc']['mean']:.4f}"
                  f" NLL {summ['nll']['mean']:.4f}", flush=True)

        # --- depth verdicts, C24.0.7 ---------------------------------------------
        A = {int(L): d["target_auroc"]["mean"] for L, d in results[attr]["probe_points"].items()}
        final = n_probe - 1
        best_L = max(A, key=lambda L: A[L])
        hit_te = in_interval(y[te], lo, hi)
        alpha_bonf = 0.05 / BONFERRONI_FAMILY
        boot = paired_bootstrap_auroc_diff(
            q_test_mean[str(best_L)], q_test_mean[str(final)], hit_te, N_BOOT,
            seed=4242, alpha=alpha_bonf,
        )
        neigh = [L for L in (best_L - 1, best_L + 1) if 0 <= L <= final]
        results[attr]["depth"] = {
            "best_probe_point": int(best_L),
            "auroc_best": A[best_L],
            "auroc_final": A[final],
            "gain_over_final": A[best_L] - A[final],
            "auroc_trivial": results[attr]["trivial"]["target_auroc"]["mean"],
            "margin_over_trivial": A[best_L] - results[attr]["trivial"]["target_auroc"]["mean"],
            "bonferroni_ci_best_minus_final": boot,
            "bonferroni_ci_excludes_zero": bool(boot["lo"] > 0 or boot["hi"] < 0),
            "neighbours": {str(L): A[L] - A[final] for L in neigh},
            "no_isolated_spike": bool(
                (A[best_L] - A[final] >= 0.010)
                and all(A[L] - A[final] >= 0.005 for L in neigh)
            ),
            "final_is_min_over_peak_to_end": bool(
                A[final] == min(A[L] for L in range(best_L, final + 1))
            ),
            "peak_in_first_half": bool(1 <= best_L <= final // 2),
            "nll_best": results[attr]["probe_points"][str(best_L)]["nll"]["mean"],
            "nll_final": results[attr]["probe_points"][str(final)]["nll"]["mean"],
            "nll_argmin": int(min(A, key=lambda L: results[attr]["probe_points"][str(L)]["nll"]["mean"])),
        }
        print(f"[{time.time()-t0:.0f}s] {attr}: best L={best_L} "
              f"({A[best_L]:.4f}) vs final ({A[final]:.4f})", flush=True)

    write_json(out / "probe_layer_metrics.json", {
        "n_probe_points": n_probe,
        "head_seeds": list(HEAD_SEEDS),
        "attributes": results,
        "prereg_2a_peak_before_final_all": all(
            r["depth"]["best_probe_point"] != n_probe - 1 for r in results.values()
        ),
        "prereg_2a_peak_in_first_half_count": sum(
            r["depth"]["peak_in_first_half"] for r in results.values()
        ),
        "prereg_2b_final_is_min_after_peak_all": all(
            r["depth"]["final_is_min_over_peak_to_end"] for r in results.values()
        ),
        "n_attributes": len(results),
    })
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


def summarise_seeds(per_seed: list[dict]) -> dict:
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], float)]
    out = {"per_seed": per_seed, "n_seeds": len(per_seed)}
    for k in keys:
        v = np.array([e[k] for e in per_seed], dtype=np.float64)
        out[k] = {"mean": float(v.mean()),
                  "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "values": v.tolist()}
    return out


if __name__ == "__main__":
    main()
