"""C26 -- assemble the compute-accuracy frontier and score the pre-registration.

Reads only artefacts that already exist: the best-of-N curves from
`scripts/21_n_sweep.py`, and every guidance point on disk (section 16.1's lambda=1 runs,
section 19's lambda grid, and C23's mid-network-layer arms).  Generates nothing.

Scores `outputs/c26_prereg/C26.0_preregistration.md` decision rules D1-D3 and its four
predictions.  Both uncertainty statements C26.0.5 requires are produced: a seed-stratified
bootstrap and a seed-level t interval on n = 3, because the C23 review established that a
molecule-level bootstrap understates the variance that matters when only three seeds exist.

    .venv/bin/python scripts/21_summarise_c26.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import OUTPUT_DIR, read_json, write_json  # noqa: E402

ANCHORS = ["aromatic_rings", "hbd_count", "qed"]
SEEDS = ["101", "202", "303"]


DATASET = "pilot_50k_p2"


def _family(name: str) -> str:
    if name.startswith("c23_gate"):
        return "c23_validity_gate (duplicate of the deployed lambda=1 run)"
    if name.startswith("c23_"):
        return "c23_mid_layer"
    if name.startswith("c18_"):
        return "c18_calibration_or_readout"
    if "_lam" in name:
        return "section19_lambda_sweep"
    return "deployed_lambda1"


def guidance_points(prop: str) -> list[dict]:
    """Every measured `throughout` arm for this property, with its own token cost.

    Filtered to the phase-2 dataset.  Without that filter the glob also catches the
    phase-1 `pilot_50k_*` runs, which are a different 50k sample measured on CPU, and
    putting them on this frontier would compare across both.
    """
    pts = []
    for d in sorted(OUTPUT_DIR.glob("*guided_*")):
        if not d.is_dir() or not d.name.endswith(prop):
            continue
        f = d / "guidance_metrics.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("dataset") != DATASET:
            continue
        cond = r.get("conditions", {}).get("throughout")
        if not cond:
            continue
        agg = cond["aggregate"]
        pts.append({
            "run": d.name,
            "family": _family(d.name),
            "lam": float(r.get("lambda", 1.0)),
            "layer": int(r.get("layer", -1)),
            "head_checkpoint": r.get("head_checkpoint"),
            "hit_rate": float(agg["hit_rate"]["mean"]),
            "hit_rate_values": [float(v) for v in agg["hit_rate"]["values"]],
            "tokens_per_molecule_actual": float(agg["compute_total"]["tokens_per_molecule_actual"]),
            "validity": float(agg["validity"]["mean"]) if isinstance(agg["validity"], dict)
                        else float(agg["validity"]),
            "per_seed": {
                s: {"hit_rate": float(v["hit_rate"]),
                    "tokens_per_molecule_actual": float(v["compute"]["tokens_per_molecule_actual"])}
                for s, v in cond.get("seeds", {}).items()
            },
        })
    return sorted(pts, key=lambda p: p["tokens_per_molecule_actual"])


def interp(curve_tokens: list[float], curve_hits: list[float], budget: float):
    """Linear-in-tokens interpolation of the best-of-N curve at `budget`.

    Returns (hit_rate, bracket_lo_index, bracket_hi_index, extrapolated?).  Linear rather
    than log-N because the reported quantity should be checkable by hand from the two
    bracketing grid points, which are published alongside it.
    """
    if budget <= curve_tokens[0]:
        return curve_hits[0], 0, 0, True
    if budget >= curve_tokens[-1]:
        return curve_hits[-1], len(curve_hits) - 1, len(curve_hits) - 1, True
    for i in range(len(curve_tokens) - 1):
        a, b = curve_tokens[i], curve_tokens[i + 1]
        if a <= budget <= b:
            w = 0.0 if b == a else (budget - a) / (b - a)
            return curve_hits[i] + w * (curve_hits[i + 1] - curve_hits[i]), i, i + 1, False
    return curve_hits[-1], len(curve_hits) - 1, len(curve_hits) - 1, True


#: Two-sided 95% Student t critical values, keyed by degrees of freedom.
#:
#: Extended 2026-08-03 from {2, 3, 4} to cover df 5-15 because C30 averages over **eight**
#: head seeds and the old fallback for an unlisted df was the normal quantile 1.96, which at
#: 7 df is 17% too narrow and would have silently manufactured significance.  Every entry
#: already present is unchanged, so no C26 or C28 number moves: both call this only at
#: n = 3, which reads the same 4.302653 as before.  `tests/test_crossing_head_seeds.py`
#: pins t(7) and re-asserts t(2).
T_CRIT_95 = {
    2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912,
    7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139, 11: 2.200985,
    12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
}


def t_interval(vals: list[float], conf: float = 0.95):
    """Two-sided interval on the mean of n seed-level values.  n = 3 -> t(2) = 4.3027."""
    n = len(vals)
    if n < 2:
        return None
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    tcrit = T_CRIT_95.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "lo": m - half, "hi": m + half,
            "excludes_zero": (m - half) * (m + half) > 0, "n_seeds": n,
            "note": f"seed-level t interval, {n - 1} df"}


def bootstrap_diff(guided_vals: list[float], bon_vals: list[float], n_boot: int, rng):
    """Seed-resampled bootstrap of the mean difference -- and, at n = 3, VACUOUS.

    Resampling SEEDS rather than molecules was the deliberate correction to C23's
    interval, which resampled molecules and so treated the three seed means as fixed.
    That correction was right about *which* variance matters and wrong to think a
    bootstrap could express it at this sample size.

    At n = 3 the percentile bootstrap of a mean is **identically [min, max]** of the
    three values.  The smallest attainable bootstrap mean is `min`, attained when all
    three resampled indices hit the minimum, with probability 1/27 = 0.0370 > 0.025 --
    so the 2.5th percentile *is* the minimum, and symmetrically the 97.5th *is* the
    maximum, for any three numbers whatsoever.  "The CI excludes zero" therefore carries
    exactly the information of "all three seeds share a sign": a three-way sign test with
    null probability 2 * (1/2)**3 = 0.25.  It is n = 3 in a costume.

    The interval is still computed and published, because deleting it silently would hide
    that two C26 decision rules were once read off it.  It is returned alongside the
    degeneracy flag and the sign-test p-value that are the honest summary, and the
    section reports the t interval on 2 df instead.
    """
    g = np.asarray(guided_vals)
    b = np.asarray(bon_vals)
    n = len(g)
    draws = rng.integers(0, n, size=(n_boot, n))
    diffs = (g[draws] - b[draws]).mean(axis=1)
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    per_seed = (g - b).tolist()
    degenerate = n <= 5 and lo == min(per_seed) and hi == max(per_seed)
    return lo, hi, {
        "degenerate_equals_min_max": bool(degenerate),
        "n_seeds": int(n),
        "sign_test_p_two_sided": 2.0 * 0.5 ** n if n > 0 else None,
        "all_seeds_share_sign": bool(all(x > 0 for x in per_seed)
                                    or all(x < 0 for x in per_seed)),
        "note": ("at n=3 the percentile bootstrap of a mean is exactly [min, max]; this "
                 "interval conveys nothing beyond a three-way sign test (p_null = 0.25) "
                 "and must not be read as a confidence interval"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20260731)
    ap.add_argument("--out", default="c26_summary")
    args = ap.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "prereg": "outputs/c26_prereg/C26.0_preregistration.md",
        "accounting": "actual",
        "seeds": SEEDS,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.boot_seed,
                      "unit": "seeds resampled with replacement"},
        "properties": {},
        "decision_rules": {},
        "predictions": {},
    }

    gates: dict = {}
    gate_f = OUTPUT_DIR / "c26_gate_exact_N9_aromatic_rings" / "n_sweep_metrics.json"
    if gate_f.exists():
        g = json.loads(gate_f.read_text())
        pub = read_json(OUTPUT_DIR / "pilot_50k_p2_bestofn_aromatic_rings"
                        / "bestofn_metrics.json")["matches"]["actual"]["seeds"]
        gates["exact_N9_aromatic_rings"] = {
            "per_seed": {
                s: {"gate_hit_rate": g["per_seed"][s]["rows"]["9"]["hit_rate"],
                    "published_hit_rate": pub[s]["hit_rate"],
                    "hit_rate_residual": g["per_seed"][s]["rows"]["9"]["hit_rate"]
                                         - pub[s]["hit_rate"],
                    "gate_tokens_per_molecule":
                        g["per_seed"][s]["rows"]["9"]["compute"]["tokens_per_molecule_actual"],
                    "published_tokens_per_molecule":
                        pub[s]["compute"]["tokens_per_molecule_actual"],
                    "token_residual":
                        g["per_seed"][s]["rows"]["9"]["compute"]["tokens_per_molecule_actual"]
                        - pub[s]["compute"]["tokens_per_molecule_actual"]}
                for s in SEEDS},
        }
        gates["exact_N9_aromatic_rings"]["max_abs_hit_residual"] = max(
            abs(v["hit_rate_residual"]) for v in gates["exact_N9_aromatic_rings"]["per_seed"].values())

    # C26.0.3 gate 2, as an identity rather than a noise comparison.  The sweep evaluates
    # ALL disjoint consecutive groups of N over the pool, so its first `n_molecules` groups
    # are *exactly* `scripts/06_best_of_n.py`'s and must reproduce the published per-seed
    # hit rate bit for bit.  The published N is read from each artefact, never assumed: the
    # compute-matched N is 9 for aromatic_rings and hbd_count but **8** for qed, because
    # qed's guided run is cheaper.  Comparing qed at 9 makes the gate appear to fail by
    # ~0.05 -- which is a bug in the checker, not in the estimator, and is recorded here so
    # the same mistake is not made twice.
    g2: dict = {}
    for prop in ANCHORS:
        f = OUTPUT_DIR / f"c26_nsweep_{prop}" / "n_sweep_metrics.json"
        pf = OUTPUT_DIR / f"pilot_50k_p2_bestofn_{prop}" / "bestofn_metrics.json"
        if not (f.exists() and pf.exists()):
            continue
        sweep = json.loads(f.read_text())
        pm = read_json(pf)["matches"]["actual"]
        n_pub = int(pm["n_candidates"])
        cells = {}
        for s in SEEDS:
            row = sweep["per_seed"][s]["rows"].get(str(n_pub))
            if row is None or "first_512_groups_hit_rate" not in row:
                continue
            cells[s] = {
                "published_n": n_pub,
                "first_512_groups_hit_rate": row["first_512_groups_hit_rate"],
                "published_hit_rate": pm["seeds"][s]["hit_rate"],
                "residual": row["first_512_groups_hit_rate"] - pm["seeds"][s]["hit_rate"],
                "all_groups_hit_rate": row["hit_rate"],
                "n_groups": row["n_groups"],
            }
        if cells:
            g2[prop] = {"published_n": n_pub, "per_seed": cells,
                        "max_abs_residual": max(abs(c["residual"]) for c in cells.values())}
    if g2:
        gates["first_512_groups_identity"] = {
            "rule": "the sweep's first `n_molecules` groups at the published N must "
                    "reproduce scripts/06_best_of_n.py's per-seed hit rate exactly",
            "per_property": g2,
            "max_abs_residual": max(v["max_abs_residual"] for v in g2.values()),
            "passes": all(v["max_abs_residual"] == 0.0 for v in g2.values()),
        }
    report["validity_gates"] = gates

    d1_violations, d2, preds = [], {}, {}
    for prop in ANCHORS:
        f = OUTPUT_DIR / f"c26_nsweep_{prop}" / "n_sweep_metrics.json"
        if not f.exists():
            report["properties"][prop] = {"status": "not run"}
            continue
        sweep = json.loads(f.read_text())
        grid = sweep["grid"]
        curve = sweep["curve"]
        toks = [curve[str(n)]["tokens_per_molecule_actual"] for n in grid]
        hits = [curve[str(n)]["hit_rate_mean"] for n in grid]

        # Concavity of the best-of-N curve in N (prediction 1).
        #
        # The grid 1,2,3,4,6,8,9,12,16,24,32 is NOT uniform -- its spacings run from 1 to 8
        # -- so the textbook second difference h[i+1] - 2h[i] + h[i-1] does not test
        # concavity here.  It reported all three curves non-concave; the curves are in fact
        # strictly concave, with slopes that decrease monotonically at every step.  The
        # correct discrete test on an unequal grid is that the divided differences (secant
        # slopes between consecutive grid points) are non-increasing.  Both are recorded so
        # the discarded statistic is visible rather than silently replaced.
        slopes = [(hits[i + 1] - hits[i]) / (grid[i + 1] - grid[i])
                  for i in range(len(grid) - 1)]
        concave = all(slopes[i + 1] <= slopes[i] + 1e-12 for i in range(len(slopes) - 1))
        second = [hits[i + 1] - 2 * hits[i] + hits[i - 1] for i in range(1, len(hits) - 1)]

        gpts = guidance_points(prop)
        rows = []
        for p in gpts:
            bud = p["tokens_per_molecule_actual"]
            bh, i_lo, i_hi, extrap = interp(toks, hits, bud)
            # per-seed advantage against each seed's own interpolated curve
            per_seed_adv = []
            for s in SEEDS:
                sh = p["per_seed"].get(s)
                if sh is None:
                    continue
                s_toks = [curve[str(n)]["tokens_per_molecule_actual"] for n in grid]
                s_hits = [sweep["per_seed"][s]["rows"][str(n)]["hit_rate"] for n in grid]
                s_bh, *_ = interp(s_toks, s_hits, sh["tokens_per_molecule_actual"])
                per_seed_adv.append(sh["hit_rate"] - s_bh)
            adv = p["hit_rate"] - bh
            row = {
                "run": p["run"], "family": p["family"], "lam": p["lam"], "layer": p["layer"],
                "guided_hit_rate": p["hit_rate"],
                "tokens_per_molecule_actual": bud,
                "best_of_n_interpolated_hit_rate": bh,
                "bracketing_n": [grid[i_lo], grid[i_hi]],
                "extrapolated_beyond_grid": extrap,
                "advantage": adv,
                "advantage_per_seed": per_seed_adv,
                "validity": p["validity"],
            }
            if len(per_seed_adv) == len(SEEDS):
                row["advantage_seed_t_interval"] = t_interval(per_seed_adv)
                gv = [p["per_seed"][s]["hit_rate"] for s in SEEDS]
                bv = [gv[k] - per_seed_adv[k] for k in range(len(SEEDS))]
                # The three-seed percentile bootstrap C26.0.5 pre-registered is NOT
                # reported as an interval: at n = 3 it is identically [min, max] (see
                # `bootstrap_diff`).  It is still computed, so that the degeneracy is
                # demonstrated from the data rather than asserted, and stored under a name
                # that cannot be mistaken for a confidence interval.
                lo, hi, diag = bootstrap_diff(gv, bv, args.n_boot, rng)
                row["advantage_seed_sign_test"] = {
                    "degenerate_bootstrap_lo": lo, "degenerate_bootstrap_hi": hi, **diag}
            if adv > 0:
                d1_violations.append({"property": prop, **row})
            rows.append(row)

        report["properties"][prop] = {
            "n_sweep_run": sweep["mode"],
            "grid": grid,
            "curve": curve,
            "curve_is_concave_in_n": concave,
            "curve_secant_slopes_in_n": slopes,
            "curve_second_differences_uniform_grid_invalid": second,
            "guidance_points": rows,
        }
        preds[prop] = {"concave": concave, "secant_slopes_in_n": slopes}

        for r in rows:
            if "c23_" in r["run"] and prop == "hbd_count" and r["lam"] == 2.0 and r["layer"] == 4:
                d2 = {"property": prop, **r,
                      "note": "C23's Rule B arm, priced against the interpolated best-of-N "
                              "curve at its own exact budget; this removes the integer-"
                              "flooring artefact in both directions"}

    # D2, repeated across C25's head-seed replicates of the same arm.
    #
    # The C26 pre-registration prices D2 on one head seed, because when it was written the
    # replicates did not exist.  C25 produced them, and they are decisive: the whole of
    # C23's Rule B rests on this single arm, and re-training the head with a different seed
    # moves the guided hit rate by 0.09 -- roughly 25x the spread across generation seeds.
    # Pricing every replicate on the same curve at its own budget is what turns a marginal
    # positive into a sign flip, so it is computed here rather than left as prose.
    d2_head_seeds: dict = {}
    hs_runs = {
        "1234": "c23_guided_L4_lam2_hbd_count",
        "2345": "c25_hs2345_L4_lam2_hbd_count_guided",
        "3456": "c25_hs3456_L4_lam2_hbd_count_guided",
    }
    hb = report["properties"].get("hbd_count", {})
    if "curve" in hb:
        grid = hb["grid"]
        c_toks = [hb["curve"][str(n)]["tokens_per_molecule_actual"] for n in grid]
        c_hits = [hb["curve"][str(n)]["hit_rate_mean"] for n in grid]
        sweep_hb = json.loads((OUTPUT_DIR / "c26_nsweep_hbd_count"
                               / "n_sweep_metrics.json").read_text())
        for hs, run in hs_runs.items():
            f = OUTPUT_DIR / run / "guidance_metrics.json"
            if not f.exists():
                continue
            cond = read_json(f)["conditions"]["throughout"]["aggregate"]
            g_mean = cond["hit_rate"]["mean"]
            g_vals = cond["hit_rate"]["values"]
            bud = cond["compute_total"]["tokens_per_molecule_actual"]
            bh, i_lo, i_hi, extrap = interp(c_toks, c_hits, bud)
            ps = []
            for k, s in enumerate(SEEDS):
                s_hits = [sweep_hb["per_seed"][s]["rows"][str(n)]["hit_rate"] for n in grid]
                sb, *_ = interp(c_toks, s_hits, bud)
                ps.append(g_vals[k] - sb)
            d2_head_seeds[hs] = {
                "run": run,
                "guided_hit_rate": g_mean,
                "tokens_per_molecule_actual": bud,
                "best_of_n_interpolated_hit_rate": bh,
                "bracketing_n": [grid[i_lo], grid[i_hi]],
                "extrapolated_beyond_grid": extrap,
                "advantage": g_mean - bh,
                "advantage_per_seed": ps,
            }
        if d2_head_seeds:
            advs = [v["advantage"] for v in d2_head_seeds.values()]
            d2_head_seeds["_across_head_seeds"] = {
                "advantages": advs,
                "mean": float(np.mean(advs)),
                "sd": float(np.std(advs, ddof=1)) if len(advs) > 1 else 0.0,
                "min": float(min(advs)),
                "max": float(max(advs)),
                "sign_flips": any(a > 0 for a in advs) and any(a < 0 for a in advs),
                "guided_hit_rate_span": (
                    max(v["guided_hit_rate"] for k, v in d2_head_seeds.items()
                        if not k.startswith("_"))
                    - min(v["guided_hit_rate"] for k, v in d2_head_seeds.items()
                          if not k.startswith("_"))),
            }

    report["decision_rules"] = {
        "D1_best_of_n_dominates_everywhere": {
            "rule": "upheld iff no measured guidance arm sits above the interpolated "
                    "best-of-N curve at its own budget, for all three anchors",
            "upheld": len(d1_violations) == 0,
            "violations": d1_violations,
        },
        "D2_c23_arm_priced_on_the_curve": d2 or {"status": "arm not found"},
        "D2b_c23_arm_across_c25_head_seeds": d2_head_seeds or {"status": "replicates absent"},
        "D3_best_of_n_saturates": {
            prop: {
                "hit_rate_at_max_n": report["properties"][prop]["curve"][str(max(
                    report["properties"][prop]["grid"]))]["hit_rate_mean"],
                "last_doubling_gain":
                    report["properties"][prop]["curve"][str(max(
                        report["properties"][prop]["grid"]))]["hit_rate_mean"]
                    - report["properties"][prop]["curve"][str(
                        max(report["properties"][prop]["grid"]) // 2)]["hit_rate_mean"],
            }
            for prop in ANCHORS if "curve" in report["properties"].get(prop, {})
        },
    }
    report["predictions"] = preds

    write_json(out_dir / "c26_metrics.json", report)
    print(json.dumps(report["decision_rules"], indent=1)[:3000])
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
