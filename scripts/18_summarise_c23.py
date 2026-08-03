"""C23 -- assemble the arms and score the pre-registered decision rules.

Reads only; generates nothing.  Every number `reports/section_c23_layer_end_to_end.md`
quotes comes out of `outputs/c23_summary/c23_metrics.json`, and
`tests/test_layer_end_to_end.py` re-reads that file and requires the numbers to appear in
the section text, so the prose cannot drift from the artefacts.

Three things are done here and nowhere else:

**The validity gate.**  The `--layer 12` replay of the deployed lambda=1 aromatic-ring run
is compared against it per seed and molecule by molecule, and the *residual* is reported
rather than the word "matches".

**Seed-matched comparison.**  Every arm is compared against the deployed-layer run at the
**same lambda** with the **same three seeds** and the same 512 molecules per seed, never
against a differently-seeded mean.

**The bootstrap.**  Two arms generate different molecules, so there is no molecule-level
pairing and none is claimed.  The resampling is seed-stratified: within each seed the
scored molecules are drawn with replacement from each arm independently, each arm's three
per-seed hit rates are averaged with equal weight, and the statistic is the difference of
those two seed-matched means.  Intervals are two-sided Bonferroni-corrected at
alpha = 0.05 / (number of experimental arms run).

    python scripts/18_summarise_c23.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

DATASET = "pilot_50k_p2"
SEEDS = ("101", "202", "303")
N_BOOT = 10000
T_CRIT_2DF = 4.302653  # t_{0.975, 2}; three seeds buy two degrees of freedom
BOOT_SEED = 20260731
ALPHA = 0.05

#: C23.0.3 / C23.0.4, transcribed from the frozen pre-registration.
COMBINATIONS = (
    ("aromatic_rings", 3, "auroc_best"),
    ("hbd_count", 4, "auroc_best"),
    ("qed", 4, "auroc_best"),
    ("aromatic_rings", 6, "per_position_steering_best"),
    ("hbd_count", 6, "per_position_steering_best"),
)
LAMBDAS = {"aromatic_rings": (1.0, 2.0, 0.5),
           "hbd_count": (1.0, 2.0, 0.5),
           "qed": (1.0, 4.0, 2.0)}
S19_OPTIMUM = {"aromatic_rings": 2.0, "hbd_count": 2.0, "qed": 4.0}
#: C23.0.5 bug alarm: `unguided` cannot depend on the layer or on lambda.
UNGUIDED_REFERENCE = {"aromatic_rings": 0.1785, "hbd_count": 0.0837, "qed": 0.0896}
#: C23.0.2, verified against the artefacts before the pre-registration was written.
DEPLOYED_BEST_ADVANTAGE = {"aromatic_rings": -0.2715, "hbd_count": -0.0931,
                           "qed": -0.2829}
DISQUALIFY_VALIDITY_DROP = 0.01
DISQUALIFY_UNIQUENESS_DROP = 0.01
TOKEN_RATIO_CEILING = 1.05


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def arm_dir(prop: str, layer: int, lam: float, kind: str) -> Path:
    return OUTPUT_DIR / f"c23_{kind}_L{layer}_{lam_tag(lam)}_{prop}"


def deployed_dir(prop: str, lam: float, kind: str) -> Path:
    if lam == 1.0:
        return OUTPUT_DIR / f"pilot_50k_p2_{kind}_{prop}"
    return OUTPUT_DIR / f"pilot_50k_p2_{lam_tag(lam)}_{kind}_{prop}"


# ---------------------------------------------------------------- hit vectors

def hit_vectors(guided_dir: Path, prop: str, lo: float, hi: float,
                condition: str = "throughout") -> dict[str, np.ndarray]:
    """Per-seed binary hit indicators over the *scored* molecules of a guided run.

    The denominator is `n_scored`, exactly as `scripts/05_guided_generation.py`
    computes `hit_rate`, so the bootstrap point estimate reproduces the artefact.
    """
    mols = read_json(guided_dir / "molecules.json")
    out = {}
    for seed in SEEDS:
        vals = [r[prop] for r in mols[condition][seed]
                if r["valid"] and r.get(prop) is not None]
        v = np.asarray(vals, dtype=float)
        out[seed] = ((v >= lo) & (v < hi)).astype(np.int8)
    return out


def hit_vectors_from_counts(per_seed: dict[str, dict]) -> dict[str, np.ndarray]:
    """Reconstruct the exact indicator vector from (n_scored, hit_rate).

    A binary vector is determined by its length and its sum, so this is exact rather
    than an approximation; it is used for best-of-N, which does not save molecules.
    """
    out = {}
    for seed in SEEDS:
        s = per_seed[seed]
        n = int(s["n_scored"])
        k = int(round(s["hit_rate"] * n))
        v = np.zeros(n, dtype=np.int8)
        v[:k] = 1
        out[seed] = v
    return out


def seed_mean(vecs: dict[str, np.ndarray]) -> float:
    return float(np.mean([v.mean() for v in vecs.values()]))


def bootstrap_difference(a: dict[str, np.ndarray], b: dict[str, np.ndarray],
                         n_boot: int, alpha: float, rng: np.random.Generator) -> dict:
    """Seed-stratified bootstrap of (seed-matched mean of a) - (seed-matched mean of b)."""
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ma = np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in a.values()])
        mb = np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in b.values()])
        draws[i] = ma - mb
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # The uncorrected 95% interval is carried alongside as a descriptive statistic.
    # The decision rules use the corrected one; both are printed so a reader can see
    # how much of any verdict is the multiplicity correction.
    u_lo, u_hi = np.quantile(draws, [0.025, 0.975])

    # --- the interval this bootstrap does NOT provide -------------------------------
    # Every resample above draws molecules *within* a seed and then averages the three
    # seed means, so the between-seed component is estimated as exactly zero and the
    # width is driven by the ~1536 molecules alone.  C25 later measured the variance that
    # actually matters for these arms -- head seed, sd 0.0513 on the advantage, against a
    # bootstrap SE of about 0.018 -- and overturned Rule B on it.  A seed-level Student t
    # interval on 2 df is reported alongside so the two are visible together.  It is the
    # honest one for a claim about a *population* of runs rather than about these 1536
    # molecules.
    #
    # It is NOT uniformly wider, and assuming so is a trap.  The bootstrap here is
    # Bonferroni-corrected at alpha = 0.05/15, i.e. roughly a 99.7% interval, while this
    # is a plain 95% on 2 df whose width is set by how much the three seeds happen to
    # disagree.  When they agree closely the t interval comes out *narrower* -- and that
    # is a warning rather than a reassurance, because at n = 3 an sd can be small by luck.
    # Both are published for every arm; neither is claimed to dominate.
    per_seed = {s: float(a[s].mean() - b[s].mean()) for s in SEEDS}
    d = np.array(list(per_seed.values()), dtype=float)
    t_mean = float(d.mean())
    t_sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    t_half = T_CRIT_2DF * t_sd / np.sqrt(d.size) if d.size > 1 else 0.0
    t_lo, t_hi = t_mean - t_half, t_mean + t_half

    return {"difference": seed_mean(a) - seed_mean(b),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "ci95_uncorrected_lo": float(u_lo), "ci95_uncorrected_hi": float(u_hi),
            "excludes_zero_uncorrected": bool(u_lo > 0 or u_hi < 0),
            "alpha": alpha, "n_boot": n_boot,
            "bootstrap_unit": "molecules resampled within seed; between-seed variance "
                              "is estimated as zero by construction",
            "seed_level_t_interval": {
                "mean": t_mean, "sd": t_sd, "lo": float(t_lo), "hi": float(t_hi),
                "excludes_zero": bool(t_lo > 0 or t_hi < 0),
                "n_seeds": int(d.size),
                "interval": f"Student t, {d.size - 1} df, t_crit={T_CRIT_2DF}",
                "note": "the interval to quote for a claim about runs rather than about "
                        "these molecules; no three-seed percentile bootstrap is reported "
                        "anywhere, because at n=3 it is identically [min, max]",
            },
            "per_seed_difference": per_seed}


def seed_noise(per_seed_diff: dict[str, float]) -> dict:
    d = np.array(list(per_seed_diff.values()), dtype=float)
    sem = float(d.std(ddof=1) / np.sqrt(d.size))
    return {"sem": sem, "two_sem": 2 * sem,
            "exceeds_seed_noise": bool(abs(d.mean()) > 2 * sem)}


# ---------------------------------------------------------------- the gate

def score_gate(prop: str, lo: float, hi: float) -> dict:
    ref_dir = OUTPUT_DIR / f"pilot_50k_p2_guided_{prop}"
    rep_dir = OUTPUT_DIR / "c23_gate_L12_lam1_aromatic_rings"
    if not (rep_dir / "guidance_metrics.json").exists():
        return {"run": False}
    ref = read_json(ref_dir / "guidance_metrics.json")
    rep = read_json(rep_dir / "guidance_metrics.json")
    ref_m = read_json(ref_dir / "molecules.json")
    rep_m = read_json(rep_dir / "molecules.json")

    per_seed, residuals, n_cmp, identical = {}, [], 0, True
    for cond in ("unguided", "throughout"):
        for seed in SEEDS:
            a = ref["conditions"][cond]["seeds"][seed]["hit_rate"]
            b = rep["conditions"][cond]["seeds"][seed]["hit_rate"]
            residuals.append(abs(a - b))
            per_seed.setdefault(f"{cond}:{seed}", {})
            per_seed[f"{cond}:{seed}"] = {"reference_hit_rate": a, "replay_hit_rate": b,
                                          "residual": abs(a - b)}
            sa = [r["smiles"] for r in ref_m[cond][seed]]
            sb = [r["smiles"] for r in rep_m[cond][seed]]
            n_cmp += len(sa)
            identical = identical and (sa == sb)
    return {
        "run": True,
        "reference_run": ref_dir.name,
        "replay_run": rep_dir.name,
        "replay_layer": rep["layer"],
        "replay_head_file": rep["head_file"],
        "reference_head_file": ref["head_checkpoint"],
        "per_seed": per_seed,
        "reference_throughout_mean":
            ref["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"],
        "replay_throughout_mean":
            rep["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"],
        "reference_unguided_mean":
            ref["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"],
        "replay_unguided_mean":
            rep["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"],
        "max_abs_hit_rate_residual": float(max(residuals)),
        "molecules_identical": bool(identical),
        "n_molecules_compared": n_cmp,
        "passes": bool(max(residuals) == 0.0 and identical),
    }


# ---------------------------------------------------------------- arms

def collect_arm(prop: str, layer: int, why: str, lam: float, lo: float, hi: float,
                rng: np.random.Generator, alpha: float) -> dict | None:
    g = arm_dir(prop, layer, lam, "guided")
    if not (g / "guidance_metrics.json").exists():
        return None
    dep = deployed_dir(prop, lam, "guided")
    gm = read_json(g / "guidance_metrics.json")
    dm = read_json(dep / "guidance_metrics.json")

    arm_hits = hit_vectors(g, prop, lo, hi)
    dep_hits = hit_vectors(dep, prop, lo, hi)
    boot = bootstrap_difference(arm_hits, dep_hits, N_BOOT, alpha, rng)
    noise = seed_noise(boot["per_seed_difference"])

    ga = gm["conditions"]["throughout"]["aggregate"]
    da = dm["conditions"]["throughout"]["aggregate"]
    gu = gm["conditions"]["unguided"]["aggregate"]
    gt = ga["compute_total"]
    dt = da["compute_total"]

    rec = {
        "property": prop, "layer": layer, "layer_reason": why, "lam": lam,
        "guided_run": g.name, "deployed_run": dep.name,
        "head_file": gm["head_file"], "recorded_layer": gm["layer"],
        "throughout_mean": ga["hit_rate"]["mean"],
        "throughout_values": ga["hit_rate"]["values"],
        "deployed_throughout_mean": da["hit_rate"]["mean"],
        "deployed_throughout_values": da["hit_rate"]["values"],
        "lift": ga["hit_rate"]["mean"] - gu["hit_rate"]["mean"],
        "deployed_lift": da["hit_rate"]["mean"] - dm["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"],
        "unguided_mean": gu["hit_rate"]["mean"],
        "unguided_values": gu["hit_rate"]["values"],
        "unguided_reference": UNGUIDED_REFERENCE[prop],
        "diff_vs_deployed": boot["difference"],
        "diff_ci_lo": boot["ci_lo"], "diff_ci_hi": boot["ci_hi"],
        "diff_excludes_zero": boot["excludes_zero"],
        # the between-seed interval the molecule-level bootstrap cannot see
        "diff_seed_level_t": boot["seed_level_t_interval"],
        "diff_per_seed": boot["per_seed_difference"],
        "diff_sem": noise["sem"], "diff_exceeds_seed_noise": noise["exceeds_seed_noise"],
        "tokens_per_molecule_actual": gt["tokens_per_molecule_actual"],
        "tokens_per_molecule_full_recompute": gt["tokens_per_molecule_full_recompute"],
        "deployed_tokens_per_molecule_actual": dt["tokens_per_molecule_actual"],
        "token_ratio_vs_deployed":
            gt["tokens_per_molecule_actual"] / dt["tokens_per_molecule_actual"],
        "validity_mean": ga["validity"]["mean"],
        "deployed_validity_mean": da["validity"]["mean"],
        "uniqueness_mean": ga["uniqueness"]["mean"],
        "deployed_uniqueness_mean": da["uniqueness"]["mean"],
        "content_length_mean": ga["content_length_mean"]["mean"],
        "deployed_content_length_mean": da["content_length_mean"]["mean"],
    }
    rec["validity_delta"] = rec["validity_mean"] - rec["deployed_validity_mean"]
    rec["uniqueness_delta"] = rec["uniqueness_mean"] - rec["deployed_uniqueness_mean"]

    reasons = []
    if rec["validity_delta"] < -DISQUALIFY_VALIDITY_DROP:
        reasons.append(f"validity {rec['validity_delta']:+.4f} vs deployed")
    if rec["uniqueness_delta"] < -DISQUALIFY_UNIQUENESS_DROP:
        reasons.append(f"uniqueness {rec['uniqueness_delta']:+.4f} vs deployed")
    if rec["token_ratio_vs_deployed"] > TOKEN_RATIO_CEILING:
        reasons.append(f"token ratio {rec['token_ratio_vs_deployed']:.4f}")
    rec["disqualified"] = bool(reasons)
    rec["disqualification_reasons"] = reasons

    # compute-matched best-of-N, N re-solved from THIS arm's own tokens
    bo = arm_dir(prop, layer, lam, "bestofn")
    if (bo / "bestofn_metrics.json").exists():
        bm = read_json(bo / "bestofn_metrics.json")["matches"]["actual"]
        bo_hits = hit_vectors_from_counts(bm["seeds"])
        bboot = bootstrap_difference(arm_hits, bo_hits, N_BOOT, alpha, rng)
        rec.update({
            "bestofn_run": bo.name,
            "best_of_n_hit_rate": bm["aggregate"]["hit_rate"]["mean"],
            "best_of_n_values": bm["aggregate"]["hit_rate"]["values"],
            "best_of_n_n_candidates": bm["n_candidates"],
            "best_of_n_tokens_per_molecule_actual":
                bm["aggregate"]["tokens_per_molecule_actual"],
            "realised_token_ratio_guided_over_best_of_n":
                gt["tokens_per_molecule_actual"]
                / bm["aggregate"]["tokens_per_molecule_actual"],
            "advantage_vs_best_of_n": bboot["difference"],
            "advantage_ci_lo": bboot["ci_lo"], "advantage_ci_hi": bboot["ci_hi"],
            "advantage_excludes_zero": bboot["excludes_zero"],
            "advantage_seed_level_t": bboot["seed_level_t_interval"],
            "advantage_ci95_uncorrected_lo": bboot["ci95_uncorrected_lo"],
            "advantage_ci95_uncorrected_hi": bboot["ci95_uncorrected_hi"],
            "advantage_per_seed": bboot["per_seed_difference"],
        })
    else:
        rec["best_of_n_hit_rate"] = None

    # ---- token-conservative comparator -----------------------------------
    # ADDED AFTER THE ARMS WERE RUN, and labelled as such.  `solve_best_of_n` floors,
    # so the matched N can round down and leave best-of-N spending *fewer* tokens per
    # returned molecule than guidance -- section 16.2 says so, and for the hbd_count
    # L4 lambda=2 arm the flooring took N from 9 to 8 and the realised token ratio to
    # 1.09 in guidance's favour.  This check re-runs the comparison against the
    # cheapest already-executed best-of-N run for the same property that spends **at
    # least as many** tokens per returned molecule as the guided arm, so best-of-N is
    # never the poorer-funded side.  It can only make a win harder, never easier.
    cands = []
    for d in sorted(OUTPUT_DIR.glob("*bestofn_*")):
        f = d / "bestofn_metrics.json"
        if not f.exists():
            continue
        bm = read_json(f)
        if bm.get("property") != prop or "actual" not in bm.get("matches", {}):
            continue
        m = bm["matches"]["actual"]
        if list(m["seeds"].keys()) != list(SEEDS):
            continue
        t = m["aggregate"]["tokens_per_molecule_actual"]
        if t >= gt["tokens_per_molecule_actual"]:
            cands.append((t, d.name, m))
    if cands:
        t, name, m = min(cands, key=lambda x: x[0])
        cboot = bootstrap_difference(arm_hits, hit_vectors_from_counts(m["seeds"]),
                                     N_BOOT, alpha, rng)
        rec["conservative_best_of_n"] = {
            "note": "post-hoc robustness check, added after the arms were run; "
                    "best-of-N is given at least as many tokens per returned molecule "
                    "as guidance, removing the flooring slack",
            "run": name, "n_candidates": m["n_candidates"],
            "hit_rate": m["aggregate"]["hit_rate"]["mean"],
            "values": m["aggregate"]["hit_rate"]["values"],
            "tokens_per_molecule_actual": t,
            "realised_token_ratio_guided_over_best_of_n":
                gt["tokens_per_molecule_actual"] / t,
            "advantage": cboot["difference"],
            "ci_lo": cboot["ci_lo"], "ci_hi": cboot["ci_hi"],
            "excludes_zero": cboot["excludes_zero"],
            "seed_level_t_interval": cboot["seed_level_t_interval"],
            "ci95_uncorrected_lo": cboot["ci95_uncorrected_lo"],
            "ci95_uncorrected_hi": cboot["ci95_uncorrected_hi"],
            "excludes_zero_uncorrected": cboot["excludes_zero_uncorrected"],
            "per_seed_difference": cboot["per_seed_difference"],
        }
    else:
        rec["conservative_best_of_n"] = None

    q = arm_dir(prop, layer, lam, "quality")
    if (q / "quality_metrics.json").exists():
        qm = read_json(q / "quality_metrics.json")
        rec["quality_run"] = q.name
        rec["degeneracy_any_guided_hits"] = \
            qm["panels"]["throughout"]["hits"]["degeneracy_rate"]["any"]
        rec["degeneracy_any_base_hits"] = \
            qm["panels"]["unguided"]["hits"]["degeneracy_rate"]["any"]
        rec["quality_significant"] = {
            k: v for k, v in qm["vs_unguided_hits"]["throughout"].items()
            if isinstance(v, dict) and v.get("excludes_zero")
        }
    return rec


# ---------------------------------------------------------------- decision rules

def score_rules(arms: dict) -> dict:
    props = sorted({a["property"] for a in arms.values()})

    # Rule A -- the layer improves guidance
    per_prop = {}
    for p in props:
        cand = [a for a in arms.values() if a["property"] == p]
        eligible = [a for a in cand
                    if a["diff_vs_deployed"] > 0 and a["diff_exceeds_seed_noise"]
                    and a["diff_excludes_zero"] and not a["disqualified"]]
        best = max(cand, key=lambda a: a["diff_vs_deployed"])
        per_prop[p] = {
            "best_arm": f"L{best['layer']}_{lam_tag(best['lam'])}",
            "best_diff_vs_deployed": best["diff_vs_deployed"],
            "best_diff_ci": [best["diff_ci_lo"], best["diff_ci_hi"]],
            "best_diff_excludes_zero": best["diff_excludes_zero"],
            "best_diff_exceeds_seed_noise": best["diff_exceeds_seed_noise"],
            "best_arm_disqualified": best["disqualified"],
            "fires": bool(eligible),
            "qualifying_arms": [f"L{a['layer']}_{lam_tag(a['lam'])}" for a in eligible],
        }
    n_fire = sum(1 for v in per_prop.values() if v["fires"])
    rule_a = {"rule": "best mid-layer arm beats the deployed-layer arm at matched lambda "
                      "for >= 2 of 3 properties, each beyond seed noise with a corrected "
                      "CI excluding 0 and not disqualified on quality or tokens",
              "properties_firing": n_fire, "required": 2,
              "per_property": per_prop, "fires": bool(n_fire >= 2)}

    # Rule B -- the headline falsification
    beat = [a for a in arms.values()
            if a.get("best_of_n_hit_rate") is not None
            and a["advantage_vs_best_of_n"] > 0
            and a["advantage_excludes_zero"] and not a["disqualified"]]
    scored = [a for a in arms.values() if a.get("best_of_n_hit_rate") is not None]
    best_adv = max(scored, key=lambda a: a["advantage_vs_best_of_n"]) if scored else None
    cons = [a for a in arms.values()
            if a.get("conservative_best_of_n") and not a["disqualified"]
            and a["conservative_best_of_n"]["advantage"] > 0
            and a["conservative_best_of_n"]["excludes_zero"]]
    rule_b = {"rule": "some arm beats its own compute-matched best-of-N",
              "fires": bool(beat),
              "survives_token_conservative_check": [
                  f"{a['property']}_L{a['layer']}_{lam_tag(a['lam'])}" for a in cons],
              "arms_beating_best_of_n": [f"{a['property']}_L{a['layer']}_{lam_tag(a['lam'])}"
                                         for a in beat],
              "best_advantage": best_adv["advantage_vs_best_of_n"] if best_adv else None,
              "best_advantage_arm":
                  (f"{best_adv['property']}_L{best_adv['layer']}_{lam_tag(best_adv['lam'])}"
                   if best_adv else None),
              "deployed_best_advantage": DEPLOYED_BEST_ADVANTAGE}

    return {"layer_improves_guidance": rule_a,
            "headline_falsification": rule_b,
            "null": {"rule": "neither A nor B fires",
                     "fires": bool(not rule_a["fires"] and not rule_b["fires"])}}


def score_predictions(arms: dict) -> dict:
    out = {}

    lam1 = [a for a in arms.values() if a["lam"] == 1.0]
    worst = max(lam1, key=lambda a: abs(a["diff_vs_deployed"])) if lam1 else None
    out["P_A_within_0.05_at_lambda1"] = {
        "prediction": "every lambda=1 mid-layer arm within +-0.05 of its deployed arm",
        "n_arms": len(lam1),
        "max_abs_difference": abs(worst["diff_vs_deployed"]) if worst else None,
        "max_abs_difference_arm":
            f"{worst['property']}_L{worst['layer']}" if worst else None,
        "holds": bool(lam1 and all(abs(a["diff_vs_deployed"]) <= 0.05 for a in lam1)),
    }

    pb = {}
    for p in ("aromatic_rings", "hbd_count"):
        a = arms.get(f"{p}_L6_lam1")
        b = arms.get(f"{p}_L{3 if p == 'aromatic_rings' else 4}_lam1")
        if a and b:
            pb[p] = {"steering_best_layer_6": a["throughout_mean"],
                     "auroc_best_layer": b["throughout_mean"],
                     "auroc_best_layer_index": b["layer"],
                     "steering_best_wins": bool(a["throughout_mean"] > b["throughout_mean"])}
    out["P_B_steering_best_layer_transfers"] = {
        "prediction": "probe point 6 beats the AUROC-best probe point end to end at "
                      "lambda=1, for both count properties",
        "per_property": pb,
        "holds": bool(pb and all(v["steering_best_wins"] for v in pb.values())),
    }

    at_opt = [a for a in arms.values() if a["lam"] == S19_OPTIMUM[a["property"]]]
    lower = [a for a in at_opt if a["validity_delta"] < -0.005]
    counts = [(a, arms.get(f"{a['property']}_L{a['layer']}_lam0p5"))
              for a in arms.values()
              if a["lam"] == 2.0 and a["property"] in ("aromatic_rings", "hbd_count")]
    shrink = [(x, y) for x, y in counts if y is not None
              and y["diff_vs_deployed"] > x["diff_vs_deployed"]]
    out["P_C_lambda_curve_shifts_left"] = {
        "prediction": ">=2 combinations lose >0.005 validity at the section-19 optimal "
                      "lambda, and >=1 count arm has a larger advantage at lambda=0.5 "
                      "than at lambda=2",
        "arms_with_validity_drop": [f"{a['property']}_L{a['layer']}" for a in lower],
        "validity_deltas_at_optimum": {f"{a['property']}_L{a['layer']}": a["validity_delta"]
                                       for a in at_opt},
        "count_arms_with_advantage_shrinking_in_lambda":
            [f"{x['property']}_L{x['layer']}" for x, _ in shrink],
        "holds": bool(len(lower) >= 2 and len(shrink) >= 1),
    }

    scored = [a for a in arms.values() if a.get("best_of_n_hit_rate") is not None]
    best = max(scored, key=lambda a: a["advantage_vs_best_of_n"]) if scored else None
    out["P_D_gap_stays_below_minus_0.05"] = {
        "prediction": "best advantage over compute-matched best-of-N stays below -0.05, "
                      "with hbd_count at lambda=2 the closest arm",
        "best_advantage": best["advantage_vs_best_of_n"] if best else None,
        "best_advantage_arm":
            f"{best['property']}_L{best['layer']}_{lam_tag(best['lam'])}" if best else None,
        "closest_is_hbd_lambda2":
            bool(best and best["property"] == "hbd_count" and best["lam"] == 2.0),
        "holds": bool(best is not None and best["advantage_vs_best_of_n"] < -0.05),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c23_summary")
    args = ap.parse_args()
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = read_json(OUTPUT_DIR / DATASET / "target_intervals.json")
    rng = np.random.default_rng(BOOT_SEED)

    planned, arms, not_run = [], {}, []
    for prop, layer, why in COMBINATIONS:
        for lam in LAMBDAS[prop]:
            planned.append({"property": prop, "layer": layer, "lam": lam,
                            "layer_reason": why})
    n_planned = len(planned)

    # alpha is corrected by the number of experimental arms actually run (C23.0.6).
    present = [p for p in planned
               if (arm_dir(p["property"], p["layer"], p["lam"], "guided")
                   / "guidance_metrics.json").exists()]
    n_arms = max(1, len(present))
    alpha = ALPHA / n_arms

    for p in planned:
        prop, layer, lam, why = p["property"], p["layer"], p["lam"], p["layer_reason"]
        iv = intervals[prop]
        rec = collect_arm(prop, layer, why, lam, float(iv["lo"]), float(iv["hi"]),
                          rng, alpha)
        key = f"{prop}_L{layer}_{lam_tag(lam)}"
        if rec is None:
            not_run.append(key)
        else:
            arms[key] = rec

    gate = score_gate("aromatic_rings", float(intervals["aromatic_rings"]["lo"]),
                      float(intervals["aromatic_rings"]["hi"]))

    summary = {
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "prereg_lock": read_json(OUTPUT_DIR / "c23_prereg" / "prereg_lock.json"),
        "n_arms_planned": n_planned,
        "n_arms_run": len(arms),
        "arms_not_run": not_run,
        "multiplicity": {"n_experimental_arms": n_arms, "alpha_family": ALPHA,
                         "alpha_per_arm": alpha, "n_boot": N_BOOT,
                         "bootstrap_seed": BOOT_SEED,
                         "note": "two-sided Bonferroni-corrected interval; the "
                                 "resampling is seed-stratified, not molecule-paired, "
                                 "because the two arms generate different molecules"},
        "validity_gate": gate,
        "arms": arms,
        "decision_rules": score_rules(arms) if arms else {},
        "predictions": score_predictions(arms) if arms else {},
    }
    write_json(out_dir / "c23_metrics.json", summary)
    write_run_context(out_dir)

    print(f"gate: {'PASS' if gate.get('passes') else 'not run / FAIL'} "
          f"residual={gate.get('max_abs_hit_rate_residual')} "
          f"identical={gate.get('molecules_identical')}")
    print(f"arms run {len(arms)}/{n_planned}; alpha/arm = {alpha:.6f}")
    hdr = (f"{'arm':34s} {'through':>8s} {'deploy':>8s} {'diff':>8s} "
           f"{'ci_lo':>8s} {'ci_hi':>8s} {'bo_N':>7s} {'adv':>8s} {'val':>7s} {'tokR':>6s}")
    print(hdr)
    for k, a in arms.items():
        print(f"{k:34s} {a['throughout_mean']:8.4f} {a['deployed_throughout_mean']:8.4f} "
              f"{a['diff_vs_deployed']:+8.4f} {a['diff_ci_lo']:+8.4f} {a['diff_ci_hi']:+8.4f} "
              f"{(a.get('best_of_n_hit_rate') or float('nan')):7.4f} "
              f"{(a.get('advantage_vs_best_of_n') or float('nan')):+8.4f} "
              f"{a['validity_mean']:7.4f} {a['token_ratio_vs_deployed']:6.3f}"
              + ("  DISQUALIFIED " + "; ".join(a["disqualification_reasons"])
                 if a["disqualified"] else ""))
    print(json.dumps(summary["decision_rules"], indent=1)[:1200])
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
