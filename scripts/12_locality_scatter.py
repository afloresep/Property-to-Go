"""Phase 2 -- the central test: locality against steerability, over six properties.

The deliverable is the scatter (docs/TODO.md C8) and the pre-registered predictions it
tests (docs/LEXICAL_LOCALITY.md §4). This script only *reads* artefacts; it generates
nothing, so it is cheap to re-run and cannot influence any measurement.

What it decides:

  P1  does the locality score rank the properties the same way steerability does?
  P2  does relative headroom fall with position for the diffuse properties and stay
      flat for the local count ones?
  P3  does the trivial-feature head's performance correlate with steerability across
      properties?  (the cheap correlational proxy for locality)
  P6  does the size of the guidance-vs-best-of-N gap itself track locality?
  and separately: was the *pre-registered* ordering right, which is much stronger
  evidence than fitting an ordering afterwards.

On the confidence interval. With six properties a bootstrap that resamples properties
is close to meaningless, so it is not what is done here. Instead the *measurement
uncertainty in each property's two coordinates* is propagated: steerability is
resampled over its guidance seeds and headroom over its prefixes, and the rank
correlation is recomputed. The resulting interval says "given these six properties, how
much could measurement noise move the correlation" -- it does **not** cover the
uncertainty from having chosen six properties, which is larger and is not quantifiable
from these data. That limitation is stated in the report rather than hidden in an
interval that looks narrower than it is.

    python scripts/12_locality_scatter.py --dataset pilot_50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr  # noqa: E402

from property_to_go import headroom as H  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.properties import (  # noqa: E402
    LOCALITY_BATTERY, P2_DIFFUSE_PROPERTIES, P2_LOCAL_COUNT_PROPERTIES,
    P2_UNASSIGNED_PROPERTIES, PREDICTED_LOCALITY_ORDER,
)

#: The primary locality score, named here so it appears once. Noise-corrected headroom
#: in units of the target-interval width: "how many target widths can one token choice
#: move the expected final property?" Primary because it is interventional and
#: therefore immune to both circularity risks in docs/LEXICAL_LOCALITY.md §5.
PRIMARY_LOCALITY_KEY = "relative_headroom_excess_mean"

#: The primary steerability score: the pilot's own effect size, so phase 2 is measured
#: on the same axis phase 1 reported. `throughout` hit rate minus `unguided`.
PRIMARY_STEERABILITY = "hit_rate_lift_throughout"


def _rank_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return {"rho": None, "n": int(keep.sum())}
    r = spearmanr(x[keep], y[keep])
    return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": int(keep.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--headroom", default=None)
    ap.add_argument("--heads", default=None,
                    help="head-metrics directory; defaults to <dataset>_heads")
    ap.add_argument("--guided-suffix", default="guided")
    ap.add_argument("--bestofn-suffix", default="bestofn")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm-oracle", type=int, default=200,
                    help="permutation replicates for the in-sample oracle's noise floor")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    hr_dir = OUTPUT_DIR / (args.headroom or f"{ds}_headroom")
    heads_dir = OUTPUT_DIR / (args.heads or f"{ds}_heads")
    out_dir = OUTPUT_DIR / (args.out or f"{ds}_locality")
    out_dir.mkdir(parents=True, exist_ok=True)

    headroom = read_json(hr_dir / "headroom_metrics.json")
    hr_arrays = np.load(hr_dir / "headroom_arrays.npz")
    heads = read_json(heads_dir / "head_metrics.json")
    gcfg = load_config("guidance")
    lam_cfg, eps_cfg = float(gcfg["lam"]), float(gcfg["eps"])

    rows: dict[str, dict] = {}
    # per-property bootstrap ingredients
    boot: dict[str, dict] = {}

    for prop in LOCALITY_BATTERY:
        hp = headroom["properties"][prop]
        hov = hp["headroom"]["overall"]
        if hov is None:
            raise SystemExit(
                f"{prop}: headroom is undefined (too few usable prefixes). Re-run "
                f"scripts/11_steering_headroom.py with more prefixes rather than "
                f"reporting a scatter with a hole in it."
            )
        hov_prob = hp["headroom_probability_units"]["overall"]
        cap = hp["capture"]["overall"]

        guided = read_json(
            OUTPUT_DIR / f"{ds}_{args.guided_suffix}_{prop}" / "guidance_metrics.json"
        )
        gc = guided["conditions"]
        seeds = [str(s) for s in guided["seeds"]]
        thr = np.array([gc["throughout"]["seeds"][s]["hit_rate"] for s in seeds])
        ung = np.array([gc["unguided"]["seeds"][s]["hit_rate"] for s in seeds])
        trunc = (np.array([gc["truncation_control"]["seeds"][s]["hit_rate"] for s in seeds])
                 if "truncation_control" in gc else None)
        windows = {
            w: np.array([gc[w]["seeds"][s]["hit_rate"] for s in seeds])
            for w in ("early", "middle", "late") if w in gc
        }

        base_rate = float(hp["target_interval"]["base_rate"])
        lift = float(thr.mean() - ung.mean())

        bo_path = OUTPUT_DIR / f"{ds}_{args.bestofn_suffix}_{prop}" / "bestofn_metrics.json"
        gaps: dict[str, float | None] = {}
        bo_hit: dict[str, float | None] = {}
        if bo_path.exists():
            for acc, m in read_json(bo_path)["matches"].items():
                gaps[acc] = float(m["comparison_vs_guided_throughout"]["guidance_advantage"])
                bo_hit[acc] = float(m["aggregate"]["hit_rate"]["mean"])

        # Secondary, correlational locality proxy: how well do fixed surface token
        # statistics predict the finished property? Read from the trivial head, averaged
        # over head seeds where replicates exist, so this column matches the table in
        # pilot_report.md §13 rather than quoting one arbitrary seed.
        def head_metric(name: str) -> tuple[float, float]:
            entry = heads["properties"][prop]["heads"][name]
            across = entry.get("across_seeds")
            if across is not None:
                return across["auroc"]["mean"], across["nll"]["mean"]
            return entry["test"]["intervals"]["target"]["auroc"], entry["test"]["nll"]

        trivial_auroc, trivial_nll = head_metric("trivial")
        frozen_auroc, frozen_nll = head_metric("frozen_state")

        rows[prop] = {
            "predicted_locality_rank": PREDICTED_LOCALITY_ORDER.index(prop) + 1,
            "target_interval": hp["target_interval"],
            "interval_width": hp["interval_width"],
            "base_rate": base_rate,
            # --- locality, primary (interventional, head-free, lambda-free) ---
            "locality_score": hov[PRIMARY_LOCALITY_KEY],
            "relative_headroom_raw_mean": hov["relative_headroom_raw_mean"],
            "relative_headroom_excess_median": hov["relative_headroom_excess_median"],
            "headroom_excess_property_units": hov["headroom_excess_mean"],
            "frac_prefixes_excess_above_one_interval": hov[
                "frac_prefixes_excess_above_one_interval"
            ],
            # Band-width-free companion to the primary score: the spread of
            # P(y in I | prefix + a) across candidates, noise-corrected. Reported because
            # the six target bands differ in width by a factor of 400, so dividing by
            # width is a good way to manufacture a large ratio (section 14).
            "headroom_prob_excess": None if hov_prob is None else hov_prob["headroom_excess_mean"],
            "headroom_prob_raw": None if hov_prob is None else hov_prob["headroom_raw_mean"],
            # --- locality, secondary (correlational surface-statistics proxy) ---
            "trivial_head_auroc": trivial_auroc,
            "trivial_head_nll": trivial_nll,
            "frozen_head_auroc": frozen_auroc,
            "frozen_minus_trivial_auroc": frozen_auroc - trivial_auroc,
            "frozen_minus_trivial_nll": frozen_nll - trivial_nll,
            # --- steerability ---
            PRIMARY_STEERABILITY: lift,
            "hit_rate_lift_throughout_sd": float(
                np.std(thr - ung, ddof=1) if len(thr) > 1 else 0.0
            ),
            "hit_rate_unguided": float(ung.mean()),
            "hit_rate_throughout": float(thr.mean()),
            "hit_rate_truncation_control": None if trunc is None else float(trunc.mean()),
            "hit_rate_lift_vs_truncation_control": (
                None if trunc is None else float(thr.mean() - trunc.mean())
            ),
            # Base-rate-adjusted: the share of the room above the base rate that
            # guidance closed. Reported because the count properties' base rates are
            # set by the q=0.90 rule rather than matched, so raw lift is bounded
            # differently for different properties.
            "fraction_of_room_captured": (
                float((thr.mean() - ung.mean()) / (1.0 - ung.mean()))
                if ung.mean() < 1.0 else None
            ),
            "hit_rate_lift_by_window": {w: float(v.mean() - ung.mean())
                                        for w, v in windows.items()},
            "late_share_of_throughout": (
                float((windows["late"].mean() - ung.mean()) / lift)
                if "late" in windows and lift != 0 else None
            ),
            # --- what the pilot's rule actually captured of the ceiling ---
            "captured_fraction_of_headroom": None if cap is None else cap["captured_fraction"],
            # The formula pre-registered in LEXICAL_LOCALITY.md §3 is
            # (guided - base) / (ceiling - base) with no noise correction. The primary
            # number above corrects the denominator, which was a §3.1 estimator decision
            # taken before any headroom value existed but is still a deviation from the
            # literal pre-registered expression -- so the pre-registered value is carried
            # alongside rather than replaced.
            "captured_fraction_of_headroom_preregistered_formula": (
                None if cap is None else cap["captured_fraction_raw"]
            ),
            "n_prefixes_headroom": headroom["properties"][prop]["n_prefixes_usable"],
            "n_prefixes_capture": headroom["properties"][prop]["n_prefixes_capture_usable"],
            "n_prefixes_sampled": headroom["n_prefixes"],
            "available_prob_headroom": None if cap is None else cap["available_mean"],
            "achieved_prob": None if cap is None else cap["achieved_mean"],
            # --- P6: the loss to the compute-matched baseline ---
            "best_of_n_hit_rate": bo_hit,
            "guidance_advantage": gaps,
        }

        # ---- why is capture low? a third explanation the pilot never posed ---
        # The pilot could only ask "no lever" versus "bad head". There is a third
        # possibility: the *ceiling* is attained by picking a candidate the base policy
        # dislikes, and lambda = 1 keeps log p_base at full weight, so the deployed rule
        # structurally cannot get there however good its head is.
        #
        # Separated by substituting an ORACLE head -- the realised rollout hit rate
        # itself -- into the same lambda = 1 softmax the decoder uses. Then:
        #   oracle_at_lambda1 / ceiling  = how much lambda = 1 permits at all
        #   ours / oracle_at_lambda1     = how much of that our head actually gets
        #
        # The oracle is scored on the same rollouts that define it, so it is optimistic.
        # That is the right direction: the argument below is "even an optimistic oracle
        # only reaches X at lambda = 1", so an upper bound strengthens it.
        p_hit = hr_arrays[f"p_hit_{prop}"]
        cand_lp_arr = hr_arrays["candidate_base_logprobs"]
        scored_arr = np.isfinite(p_hit)
        p_filled = np.where(scored_arr, p_hit, 0.0)
        use = (scored_arr.sum(axis=1) >= 2) & scored_arr.all(axis=1)
        bw = H.candidate_weights(cand_lp_arr)
        oracle_w = H.guided_weights(cand_lp_arr, p_filled, lam_cfg, eps_cfg)
        base_p = (bw * p_filled).sum(axis=1)
        oracle_p = (oracle_w * p_filled).sum(axis=1)
        ceil_excess = (p_filled.max(axis=1) - base_p
                       - np.nan_to_num(hr_arrays[f"null_available_{prop}"]))
        ach = (H.guided_weights(cand_lp_arr, hr_arrays[f"head_q_{prop}"], lam_cfg, eps_cfg)
               * p_filled).sum(axis=1) - base_p
        oracle_gain = oracle_p - base_p

        # The in-sample oracle exploits noise in its own estimates, and with ~13 usable
        # rollouts per candidate that bias is large. Corrected by the matching
        # permutation null, reconstructed from hit counts.
        n_valid_arr = hr_arrays["n_valid"]
        hit_counts = np.rint(p_filled * n_valid_arr).astype(np.int64)
        orng = np.random.default_rng(args.seed + 7)
        null_oracle = np.zeros(len(p_hit))
        for i in np.flatnonzero(use):
            null_oracle[i] = H.permutation_null_oracle_gain(
                hit_counts[i], n_valid_arr[i], bw[i], lam_cfg, eps_cfg,
                args.n_perm_oracle, orng,
            )
        oracle_excess = oracle_gain - null_oracle

        rows[prop]["lambda1_ceiling_analysis"] = {
            "lambda": lam_cfg,
            "n_prefixes": int(use.sum()),
            "base_policy_target_prob": float(base_p[use].mean()),
            "our_head_gain": float(ach[use].mean()),
            "oracle_head_gain_raw": float(oracle_gain[use].mean()),
            "oracle_head_gain_null": float(null_oracle[use].mean()),
            "oracle_head_gain": float(oracle_excess[use].mean()),
            "noise_corrected_ceiling_gain": float(ceil_excess[use].mean()),
            # what fraction of the ceiling lambda=1 permits, even with a perfect head
            "fraction_of_ceiling_lambda1_permits": (
                float(oracle_excess[use].sum() / ceil_excess[use].sum())
                if ceil_excess[use].sum() > 0 else None
            ),
            # what fraction of that our head actually achieves
            "our_head_share_of_the_lambda1_optimum": (
                float(ach[use].sum() / oracle_excess[use].sum())
                if oracle_excess[use].sum() > 0 else None
            ),
            "base_policy_weight_on_its_own_top_candidate": float(bw.max(axis=1)[use].mean()),
            "note": (
                "Both ratios use the noise-corrected oracle. Without that correction the "
                "oracle appears ~2x more capable than it is, which would understate what "
                "lambda = 1 costs and overstate what our head fails to capture."
            ),
        }

        # ---- is the ceiling a survivorship artefact? ------------------------
        # Forcing a low-probability candidate lowers validity, so a candidate that
        # usually produces garbage but hits the target on the few molecules that survive
        # would inflate `max_i p_i`. Recomputed with an invalid completion counted as a
        # MISS: p_i * n_valid_i / n_rollouts instead of p_i.
        #
        # Computed here, on exactly the `use` prefixes the capture analysis uses, so the
        # numbers are cross-readable against `lambda1_ceiling_analysis` and against the
        # capture table. An earlier version of this check ran on the wider
        # "at least two candidates scored" set, which made the two tables disagree about
        # the same ceiling by up to 0.024 for no substantive reason.
        n_roll = int(headroom["n_rollouts_per_candidate"])
        p_miss = np.where(scored_arr, p_filled * n_valid_arr / n_roll, 0.0)
        argmax_c = p_filled.argmax(axis=1)
        rows[prop]["survivorship_check"] = {
            "n_prefixes": int(use.sum()),
            "prefix_set": "the capture set: all top-k candidates scored",
            "ceiling_valid_only": float(p_filled.max(axis=1)[use].mean()),
            "ceiling_invalid_counted_as_miss": float(p_miss.max(axis=1)[use].mean()),
            "available_raw_valid_only": float((p_filled.max(axis=1) - base_p)[use].mean()),
            "available_raw_invalid_as_miss": float(
                (p_miss.max(axis=1) - (bw * p_miss).sum(axis=1))[use].mean()
            ),
            "validity_of_the_ceiling_setting_candidate": float(
                (n_valid_arr[np.arange(len(p_hit)), argmax_c] / n_roll)[use].mean()
            ),
            "base_policy_weighted_validity": float(
                ((bw * n_valid_arr / n_roll).sum(axis=1))[use].mean()
            ),
            "note": (
                "`available_raw_*` are NOT noise-corrected, unlike "
                "`noise_corrected_ceiling_gain` above. They are comparable to each other "
                "and must not be read against the corrected figure."
            ),
        }

        # ---- per-step is not end-to-end, and the gap is not small -----------
        # Everything in `lambda1_ceiling_analysis` is a gain in final-hit probability from
        # one token choice, with the rest of the sequence left to the base policy.
        # Guided decoding intervenes at every position, so its end-to-end lift is far
        # larger than its per-step gain. Recorded here so no reader -- and no future
        # section of the report -- converts a per-step ratio into an end-to-end one:
        # multiplying the measured lift by 1 / share gives lifts above the arithmetic
        # maximum (1 - unguided) for most properties, which is how the invalidity of that
        # conversion announces itself.
        ca = rows[prop]["lambda1_ceiling_analysis"]
        e2e = rows[prop]["hit_rate_lift_throughout"]
        share, permits = (ca["our_head_share_of_the_lambda1_optimum"],
                          ca["fraction_of_ceiling_lambda1_permits"])
        rows[prop]["per_step_versus_end_to_end"] = {
            "per_step_our_head_gain": ca["our_head_gain"],
            "end_to_end_lift_throughout": e2e,
            "amplification": (e2e / ca["our_head_gain"]
                              if ca["our_head_gain"] else None),
            "largest_arithmetically_possible_lift": float(
                1.0 - rows[prop]["hit_rate_unguided"]
            ),
            "implied_lift_if_per_step_share_transferred_linearly": (
                e2e / share if share else None
            ),
            "implied_lift_if_per_step_ceiling_transferred_linearly": (
                e2e / (share * permits) if share and permits else None
            ),
            "linear_transfer_is_impossible": bool(
                share and e2e / share > 1.0 - rows[prop]["hit_rate_unguided"]
            ),
        }

        # ---- bootstrap ingredients ------------------------------------------
        spread = hr_arrays[f"spread_value_{prop}"]
        null = hr_arrays[f"null_value_{prop}"]
        excess = (spread - null) / rows[prop]["interval_width"]
        boot[prop] = {
            "excess": excess[np.isfinite(excess)],
            "per_seed_lift": thr - ung,
        }

    # ---- P2: position trend of relative headroom -----------------------------
    p2: dict = {"by_property": {}, "classes": {
        "diffuse": list(P2_DIFFUSE_PROPERTIES),
        "local_count": list(P2_LOCAL_COUNT_PROPERTIES),
        "unassigned_by_p2": list(P2_UNASSIGNED_PROPERTIES),
    }}
    for prop in LOCALITY_BATTERY:
        byq = headroom["properties"][prop]["headroom"]["by_quartile"]
        vals = [None if byq[str(q)] is None else byq[str(q)][PRIMARY_LOCALITY_KEY]
                for q in (1, 2, 3, 4)]
        have = [(q, v) for q, v in zip((1, 2, 3, 4), vals) if v is not None]
        slope = None
        if len(have) >= 3:
            qs = np.array([q for q, _ in have], float)
            vs = np.array([v for _, v in have], float)
            slope = float(np.polyfit(qs, vs, 1)[0])
        p2["by_property"][prop] = {
            "relative_headroom_excess_by_quartile": vals,
            "slope_per_quartile": slope,
            "q4_minus_q1": (None if vals[0] is None or vals[3] is None
                            else float(vals[3] - vals[0])),
            "p2_class": ("diffuse" if prop in P2_DIFFUSE_PROPERTIES
                         else "local_count" if prop in P2_LOCAL_COUNT_PROPERTIES
                         else "unassigned_by_p2"),
        }

    # ---- the correlations the predictions are about --------------------------
    props = list(LOCALITY_BATTERY)
    loc = [rows[p]["locality_score"] for p in props]
    steer = [rows[p][PRIMARY_STEERABILITY] for p in props]
    steer_adj = [rows[p]["fraction_of_room_captured"] for p in props]
    trivial = [rows[p]["trivial_head_auroc"] for p in props]
    predicted = [-rows[p]["predicted_locality_rank"] for p in props]  # rank 1 = most local

    loc_prob = [rows[p]["headroom_prob_excess"] for p in props]

    tests: dict = {
        "P1_locality_vs_steerability": _rank_corr(loc, steer),
        "P1_locality_vs_steerability_base_rate_adjusted": _rank_corr(loc, steer_adj),
        # The same test in band-width-free units. If P1 holds in one unit and not the
        # other, that is the finding rather than a reason to pick a unit.
        "P1b_prob_headroom_vs_steerability": _rank_corr(loc_prob, steer),
        "P1b_prob_headroom_vs_steerability_base_rate_adjusted": _rank_corr(loc_prob, steer_adj),
        "prob_headroom_vs_relative_headroom": _rank_corr(loc_prob, loc),
        "P3_trivial_head_vs_steerability": _rank_corr(trivial, steer),
        "pre_registered_order_vs_measured_locality": _rank_corr(predicted, loc),
        "pre_registered_order_vs_steerability": _rank_corr(predicted, steer),
        "locality_vs_trivial_head": _rank_corr(loc, trivial),
        "steerability_vs_base_rate": _rank_corr(
            [rows[p]["base_rate"] for p in props], steer
        ),
    }
    for acc in ("actual", "full_recompute"):
        g = [rows[p]["guidance_advantage"].get(acc) for p in props]
        if sum(v is not None for v in g) >= 3:
            # P6 predicts the gap is *smaller* (less negative) for local properties,
            # i.e. advantage rises with locality.
            tests[f"P6_locality_vs_guidance_advantage_{acc}"] = _rank_corr(loc, g)
            tests[f"P6b_prob_headroom_vs_guidance_advantage_{acc}"] = _rank_corr(loc_prob, g)

    # ---- how a-priori is the "fully a-priori" ordering, really? --------------
    # PREDICTED_LOCALITY_ORDER is pinned in code and was written before any phase-2
    # measurement, but the hypothesis it encodes exists to explain two results phase 1
    # already had: aromatic rings steered well and cLogP less well. Those two properties
    # sit at predicted ranks 1 and 5, so two of the six ranks restate a known result and
    # calling the ordering free of all post-hoc freedom overstates it.
    #
    # The honest test is to drop them and correlate on the four properties whose ranks
    # could not have been informed by phase 1. Weaker (n = 4) but genuinely out of sample.
    PHASE1_PROPERTIES = ("aromatic_rings", "clogp")
    unseen = [p for p in props if p not in PHASE1_PROPERTIES]
    tests["pre_registered_order_vs_steerability_excluding_phase1_properties"] = {
        **_rank_corr([-rows[p]["predicted_locality_rank"] for p in unseen],
                     [rows[p][PRIMARY_STEERABILITY] for p in unseen]),
        "excluded": list(PHASE1_PROPERTIES),
        "note": (
            "Phase 1 measured steerability for aromatic rings and cLogP, so their "
            "predicted ranks were not blind. This is the same correlation on the four "
            "properties that were."
        ),
    }
    tests["P1b_prob_headroom_vs_steerability_excluding_phase1_properties"] = _rank_corr(
        [rows[p]["headroom_prob_excess"] for p in unseen],
        [rows[p][PRIMARY_STEERABILITY] for p in unseen],
    )

    # P4: does guidance capture a larger share of the available headroom for local
    # properties than diffuse ones?
    capt = [rows[p]["captured_fraction_of_headroom"] for p in props]
    if sum(v is not None for v in capt) >= 3:
        tests["P4_locality_vs_captured_fraction"] = _rank_corr(loc, capt)
        tests["P4b_prob_headroom_vs_captured_fraction"] = _rank_corr(loc_prob, capt)

    # ---- propagate measurement noise into the P1 correlation ----------------
    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.n_boot):
        bl, bs = [], []
        for p in props:
            ex = boot[p]["excess"]
            bl.append(float(rng.choice(ex, size=len(ex), replace=True).mean()))
            sl = boot[p]["per_seed_lift"]
            bs.append(float(rng.choice(sl, size=len(sl), replace=True).mean()))
        r = spearmanr(bl, bs).statistic
        if np.isfinite(r):
            draws.append(float(r))
    if draws:
        d = np.array(draws)
        tests["P1_locality_vs_steerability"]["measurement_noise_ci95"] = [
            float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        ]
        tests["P1_locality_vs_steerability"]["measurement_noise_n_boot"] = len(draws)
        tests["P1_locality_vs_steerability"]["frac_draws_positive"] = float((d > 0).mean())

    report = {
        "dataset": ds,
        "n_properties": len(props),
        "primary_locality_score": PRIMARY_LOCALITY_KEY,
        "primary_steerability_score": PRIMARY_STEERABILITY,
        "pre_registered_locality_order": list(PREDICTED_LOCALITY_ORDER),
        "measured_locality_order": [
            p for p in sorted(props, key=lambda p: -rows[p]["locality_score"])
        ],
        "measured_locality_order_probability_units": [
            p for p in sorted(props, key=lambda p: -(rows[p]["headroom_prob_excess"] or -1e9))
        ],
        "measured_steerability_order": [
            p for p in sorted(props, key=lambda p: -rows[p][PRIMARY_STEERABILITY])
        ],
        "caveat_on_the_interval": (
            "The bootstrap interval propagates measurement noise in each property's two "
            "coordinates (guidance seeds; headroom prefixes) with the six properties "
            "held fixed. It does NOT cover uncertainty from having chosen six "
            "hand-picked properties, which is larger and is not estimable from these "
            "data. n = 6."
        ),
        "properties": rows,
        "P2_position_trend": p2,
        "tests": tests,
    }
    write_json(out_dir / "locality_metrics.json", report)
    write_run_context(out_dir)

    # ---- the scatter, as text ------------------------------------------------
    print(f"\nprimary locality score: {PRIMARY_LOCALITY_KEY}")
    print(f"{'property':<17}{'pred':>5}{'width':>8}{'local':>8}{'locP':>7}{'trivAUC':>9}"
          f"{'steer':>9}{'±sd':>7}{'room':>7}{'cap':>7}{'gapAct':>8}")
    for p in sorted(props, key=lambda p: -rows[p]["locality_score"]):
        r = rows[p]
        cap = r["captured_fraction_of_headroom"]
        gap = r["guidance_advantage"].get("actual")
        room = r["fraction_of_room_captured"]
        lp = r["headroom_prob_excess"]
        print(f"{p:<17}{r['predicted_locality_rank']:>5}{r['interval_width']:>8.3f}"
              f"{r['locality_score']:>8.3f}"
              f"{'    n/a' if lp is None else f'{lp:>7.3f}'}"
              f"{r['trivial_head_auroc']:>9.3f}{r[PRIMARY_STEERABILITY]:>9.4f}"
              f"{r['hit_rate_lift_throughout_sd']:>7.4f}"
              f"{'    n/a' if room is None else f'{room:>7.3f}'}"
              f"{'    n/a' if cap is None else f'{cap:>7.3f}'}"
              f"{'     n/a' if gap is None else f'{gap:>8.4f}'}")

    print(f"\nwhy is capture low? lambda={lam_cfg} versus the head "
          f"(oracle gains are noise-corrected)")
    print(f"{'property':<17}{'base':>8}{'ours':>9}{'oracle@l1':>11}{'ceiling':>9}"
          f"{'l1/ceil':>9}{'ours/l1':>9}{'w(top)':>8}")
    for p in props:
        e = rows[p]["lambda1_ceiling_analysis"]
        f1 = e["fraction_of_ceiling_lambda1_permits"]
        f2 = e["our_head_share_of_the_lambda1_optimum"]
        print(f"{p:<17}{e['base_policy_target_prob']:>8.4f}{e['our_head_gain']:>+9.4f}"
              f"{e['oracle_head_gain']:>+11.4f}{e['noise_corrected_ceiling_gain']:>+9.4f}"
              f"{'   n/a' if f1 is None else f'{f1:>9.3f}'}"
              f"{'   n/a' if f2 is None else f'{f2:>9.3f}'}"
              f"{e['base_policy_weight_on_its_own_top_candidate']:>8.3f}")
    print("  l1/ceil = what an oracle head could reach at lambda=1, as a share of the "
          "head-free ceiling")
    print("  ours/l1 = what our head reaches, as a share of that oracle")

    print(f"\npre-registered order : {list(PREDICTED_LOCALITY_ORDER)}")
    print(f"measured locality    : {report['measured_locality_order']}")
    print(f"measured steerability: {report['measured_steerability_order']}")
    print("\npre-registered predictions:")
    for name, t in tests.items():
        if t.get("rho") is None:
            continue
        ci = t.get("measurement_noise_ci95")
        extra = f"  ci95=[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"  {name:<52} rho={t['rho']:+.3f} p={t['p_value']:.3f} n={t['n']}{extra}")

    print("\nP2 -- position trend of relative headroom (slope per quartile):")
    for p in props:
        e = p2["by_property"][p]
        s = e["slope_per_quartile"]
        print(f"  {p:<17}{e['p2_class']:<18}"
              f"{'slope=n/a' if s is None else f'slope={s:+.4f}'}  "
              f"by-quartile={[None if v is None else round(v, 3) for v in e['relative_headroom_excess_by_quartile']]}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
