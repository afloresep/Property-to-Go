"""C29 -- assemble every head-seed cell, gate it, and score the pre-registered rules.

Reads only; generates nothing.  Every number `reports/section_c29_head_seeds.md` quotes
comes out of `outputs/c29_summary/c29_metrics.json`, and `tests/test_head_seeds.py`
re-reads that file and requires the numbers to appear in the section text in the exact
printed format, so the prose cannot drift from the artefacts.

The hit-vector reconstruction, the seed-stratified molecule-level bootstrap and the 2-df
seed-level t interval are `scripts/18_summarise_c23.py`'s, **imported** rather than
reimplemented, so a C29 number is computed by the same estimator as the C23 number it is
compared against.  That is what makes validity gate G2 an identity rather than an
approximation.

    .venv/bin/python scripts/23_summarise_c29.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

DATASET = "pilot_50k_p2"
GEN_SEEDS = ("101", "202", "303")
ALPHA = 0.05
BOOT_N = 10000
BOOT_SEED = 20260801          # §C29.0.4.6
MIN_HEAD_SEEDS = 6            # §C29.0.2

#: §C29.0.4.4 -- E[span_n] = d2(n) * sigma for normal data.  A raw span at n = 8 is NOT
#: comparable to a raw span at n = 3; dividing each by its own d2(n) puts them on one scale.
D2 = {2: 1.12838, 3: 1.69257, 4: 2.05875, 5: 2.32593, 6: 2.53441,
      7: 2.70436, 8: 2.84720, 9: 2.97003, 10: 3.07751}

#: §C29.0.6 R3 -- the interval inside which "mid" and "deployed" head-seed sds count as
#: the same size, fixed in the pre-registration.
R3_BAND = (0.5, 2.0)

#: §C29.0.2, transcribed.
HEAD_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901)
MID_ARMS = (("A1", "hbd_count", 4, 2.0),
            ("A2", "aromatic_rings", 3, 1.0),
            ("A3", "qed", 4, 1.0))
ANCHORS = ("aromatic_rings", "hbd_count", "qed")
DEPLOYED_PROBE_POINT = 12

#: §C29.0.1 -- C17's `mean_head_q_spread_across_candidates`, read from the artefact.
STEERING_METRICS = OUTPUT_DIR / "c17_layer_steering" / "layer_steering_metrics.json"

#: The pre-existing deployed lambda grid (§C29.0.2) plus C29's three new points.
COARSE_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NEW_LAMBDAS = (1.25, 1.5, 2.5)

#: §C29.0.3 -- every C23 arm that Rule A rests on, priced raw and effective-lambda-corrected.
C23_ARMS = (
    ("aromatic_rings", 3, 0.5), ("aromatic_rings", 3, 1.0), ("aromatic_rings", 3, 2.0),
    ("aromatic_rings", 6, 0.5), ("aromatic_rings", 6, 1.0), ("aromatic_rings", 6, 2.0),
    ("hbd_count", 4, 0.5), ("hbd_count", 4, 1.0), ("hbd_count", 4, 2.0),
    ("hbd_count", 6, 0.5), ("hbd_count", 6, 1.0), ("hbd_count", 6, 2.0),
    ("qed", 4, 1.0), ("qed", 4, 2.0), ("qed", 4, 4.0),
)

#: C25's published figures, quoted so the section can restate them on a common scale.
C25_HBD_SPAN_N3 = 0.09160960684359803
C25_GEN_SPAN = 0.0037
C25_RATIO = C25_HBD_SPAN_N3 / C25_GEN_SPAN


def _c23_module():
    path = ROOT / "scripts" / "18_summarise_c23.py"
    spec = importlib.util.spec_from_file_location("summarise_c23", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C23 = _c23_module()


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


# ------------------------------------------------------------------ directory maps
# Transcribed from scripts/23_head_seed_variance.py; the driver and the summariser must
# agree on where a cell lives, and neither derives it from the other's output.

def mid_guided_dir(prop: str, layer: int, lam: float, hs: int) -> Path:
    if hs == 1234:
        return OUTPUT_DIR / f"c23_guided_L{layer}_{lam_tag(lam)}_{prop}"
    if hs in (2345, 3456):
        return OUTPUT_DIR / f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_guided"
    return OUTPUT_DIR / f"c29_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_guided"


def mid_bestofn_dir(prop: str, layer: int, lam: float, hs: int) -> Path:
    if hs == 1234:
        return OUTPUT_DIR / f"c23_bestofn_L{layer}_{lam_tag(lam)}_{prop}"
    if hs in (2345, 3456):
        return OUTPUT_DIR / f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_bestofn"
    return OUTPUT_DIR / f"c29_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_bestofn"


def deployed_guided_dir(prop: str, hs: int) -> Path:
    if hs == 1234:
        return OUTPUT_DIR / f"pilot_50k_p2_guided_{prop}"
    return OUTPUT_DIR / f"c29_dep_hs{hs}_lam1_{prop}_guided"


def deployed_lam2_guided_dir(prop: str, hs: int) -> Path:
    """POST-HOC (see `scripts/23_head_seed_variance.py::stage_p4_dep_lam2_hbd`).

    The deployed head at lambda = 2, across head seeds, so that R4's matched-lambda
    pairing exists for A1.  Added after scoring exposed the pre-registration defect; the
    defect is reported, not amended.
    """
    if hs == 1234:
        return OUTPUT_DIR / f"pilot_50k_p2_lam2_guided_{prop}"
    return OUTPUT_DIR / f"c29_dep_hs{hs}_lam2_{prop}_guided"


def deployed_lambda_dir(prop: str, lam: float) -> Path:
    if lam == 1.0:
        return OUTPUT_DIR / f"pilot_50k_p2_guided_{prop}"
    if lam in NEW_LAMBDAS:
        return OUTPUT_DIR / f"c29_deplam{lam_tag(lam)[3:]}_guided_{prop}"
    return OUTPUT_DIR / f"pilot_50k_p2_{lam_tag(lam)}_guided_{prop}"


# ------------------------------------------------------------------ small statistics

def cell_stats(d: Path, prop: str, condition: str = "throughout") -> dict | None:
    f = d / "guidance_metrics.json"
    if not f.exists():
        return None
    m = read_json(f)
    agg = m["conditions"][condition]["aggregate"]
    vals = [float(v) for v in agg["hit_rate"]["values"]]
    return {
        "dir": d.name,
        "hit_rate": float(agg["hit_rate"]["mean"]),
        "hit_rate_by_generation_seed": vals,
        "generation_seed_sd": float(np.std(vals, ddof=1)),
        "generation_seed_span": float(max(vals) - min(vals)),
        "validity": float(agg["validity"]["mean"]),
        "uniqueness": float(agg["uniqueness"]["mean"]),
        "unguided_hit_rate":
            float(m["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"]),
        "tokens_per_molecule_actual":
            float(agg["compute_total"]["tokens_per_molecule_actual"]),
        "lambda": float(m["lambda"]),
        "layer": m.get("layer"),
        "head_file": m.get("head_file"),
        "head_checkpoint": m.get("head_checkpoint"),
    }


def sd_with_chi2_ci(values: np.ndarray, alpha: float = ALPHA) -> dict:
    """Sample sd and its two-sided chi-square interval (§C29.0.4.1)."""
    n = int(values.size)
    s = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return {"n": n, "sd": s, "lo": float("nan"), "hi": float("nan")}
    df = n - 1
    lo = s * float(np.sqrt(df / stats.chi2.ppf(1 - alpha / 2, df)))
    hi = s * float(np.sqrt(df / stats.chi2.ppf(alpha / 2, df)))
    return {"n": n, "sd": s, "lo": float(lo), "hi": float(hi),
            "df": df, "interval": f"chi-square, {df} df, alpha={alpha}"}


def sd_ratio_with_f_ci(s1: float, df1: int, s2: float, df2: int,
                       alpha: float = ALPHA) -> dict:
    """Ratio of two independent sds with its F-based interval (§C29.0.4.3)."""
    r = float(s1 / s2) if s2 > 0 else float("inf")
    lo = r / float(np.sqrt(stats.f.ppf(1 - alpha / 2, df1, df2)))
    hi = r / float(np.sqrt(stats.f.ppf(alpha / 2, df1, df2)))
    return {"ratio": r, "lo": float(lo), "hi": float(hi), "df1": df1, "df2": df2,
            "interval": f"F, ({df1}, {df2}) df, alpha={alpha}"}


def t_interval(values: np.ndarray, alpha: float = ALPHA) -> dict:
    """Student t interval for a mean over head seeds (§C29.0.4.5)."""
    n = int(values.size)
    m = float(values.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": m, "sd": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "excludes_zero": False}
    sd = float(np.std(values, ddof=1))
    tc = float(stats.t.ppf(1 - alpha / 2, n - 1))
    half = tc * sd / np.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "t_crit": tc,
            "lo": float(m - half), "hi": float(m + half),
            "excludes_zero": bool((m - half) > 0 or (m + half) < 0),
            "interval": f"Student t, {n - 1} df"}


def head_seed_bootstrap(values: np.ndarray, alpha: float = ALPHA) -> dict:
    """Percentile bootstrap of the mean over head seeds (§C29.0.4.6).

    Reported only at n >= 6, and always *alongside* the t interval, never instead of it.
    At n = 3 the percentile bootstrap of a mean is identically [min, max]; at n = 8 the
    probability that a resample is constant at the minimum is 8**-8 = 5.96e-8.
    """
    n = int(values.size)
    if n < MIN_HEAD_SEEDS:
        return {"n": n, "computed": False,
                "reason": f"n < {MIN_HEAD_SEEDS}; a percentile bootstrap of a mean at "
                          f"n = {n} carries no more than a sign test"}
    rng = np.random.default_rng(BOOT_SEED)
    draws = values[rng.integers(0, n, (BOOT_N, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"n": n, "computed": True, "n_boot": BOOT_N, "rng_seed": BOOT_SEED,
            "mean": float(values.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "p_constant_resample": float(n ** (-n))}


def span_block(values: np.ndarray) -> dict:
    n = int(values.size)
    span = float(values.max() - values.min()) if n else float("nan")
    d2 = D2.get(n)
    return {"n": n, "span": span, "d2_n": d2,
            "sigma_from_span": float(span / d2) if d2 else None,
            "note": "a raw span is not comparable across different n; sigma_from_span "
                    "= span / d2(n) is"}


# ------------------------------------------------------------------ arms

def collect_family(name: str, prop: str, layer: int, lam: float,
                   dir_fn, with_bestofn: bool) -> dict:
    cells, missing = {}, []
    for hs in HEAD_SEEDS:
        c = cell_stats(dir_fn(prop, hs), prop)
        if c is None:
            missing.append(hs)
        else:
            cells[str(hs)] = c
    present = [hs for hs in HEAD_SEEDS if str(hs) in cells]
    vals = np.array([cells[str(hs)]["hit_rate"] for hs in present], dtype=float)

    rec: dict = {
        "name": name, "property": prop, "probe_point": layer, "lam": lam,
        "head_seeds_present": present, "head_seeds_missing": missing,
        "n_head_seeds": len(present),
        "meets_preregistered_minimum": bool(len(present) >= MIN_HEAD_SEEDS),
        "hit_rate_by_head_seed": {str(hs): cells[str(hs)]["hit_rate"] for hs in present},
        "dirs": {str(hs): cells[str(hs)]["dir"] for hs in present},
        "mean_hit_rate": float(vals.mean()) if vals.size else float("nan"),
        "head_seed_sd": sd_with_chi2_ci(vals),
        "head_seed_span": span_block(vals),
        "hit_rate_by_generation_seed": {
            str(hs): cells[str(hs)]["hit_rate_by_generation_seed"] for hs in present},
        "validity_by_head_seed": {str(hs): cells[str(hs)]["validity"] for hs in present},
        "unguided_by_head_seed": {
            str(hs): cells[str(hs)]["unguided_hit_rate"] for hs in present},
        "tokens_by_head_seed": {
            str(hs): cells[str(hs)]["tokens_per_molecule_actual"] for hs in present},
    }

    # pooled generation-seed sd: sqrt(mean of the within-cell variances), n*(3-1) df
    within = np.array([cells[str(hs)]["generation_seed_sd"] ** 2 for hs in present])
    if within.size:
        sg = float(np.sqrt(within.mean()))
        rec["generation_seed_sd_pooled"] = {
            "sd": sg, "df": int(within.size * (len(GEN_SEEDS) - 1)),
            "n_cells": int(within.size),
            "per_cell_sd": {str(hs): cells[str(hs)]["generation_seed_sd"]
                            for hs in present},
            "per_cell_span": {str(hs): cells[str(hs)]["generation_seed_span"]
                              for hs in present},
            "note": "pooled within-head-seed sd over generation seeds 101/202/303",
        }
        if vals.size > 1 and sg > 0:
            rec["variance_ratio"] = sd_ratio_with_f_ci(
                rec["head_seed_sd"]["sd"], vals.size - 1, sg, within.size * 2)
            rec["variance_ratio"]["c25_reported_span_ratio"] = C25_RATIO
            rec["variance_ratio"]["c25_span_ratio_inside_interval"] = bool(
                rec["variance_ratio"]["lo"] <= C25_RATIO <= rec["variance_ratio"]["hi"])

    if with_bestofn:
        adv, bon = {}, {}
        for hs in present:
            f = mid_bestofn_dir(prop, layer, lam, hs) / "bestofn_metrics.json"
            if not f.exists():
                continue
            m = read_json(f)["matches"]["actual"]
            bon[str(hs)] = {"N": m["n_candidates"],
                            "hit_rate": float(m["aggregate"]["hit_rate"]["mean"]),
                            "tokens_per_molecule_actual":
                                float(m["aggregate"]["tokens_per_molecule_actual"])}
            adv[str(hs)] = cells[str(hs)]["hit_rate"] - bon[str(hs)]["hit_rate"]
        if adv:
            a = np.array(list(adv.values()), dtype=float)
            rec["best_of_n_by_head_seed"] = bon
            rec["advantage_over_best_of_n_by_head_seed"] = adv
            rec["advantage_over_best_of_n"] = t_interval(a)
            rec["advantage_over_best_of_n_bootstrap"] = head_seed_bootstrap(a)
            rec["n_head_seeds_with_positive_advantage"] = int((a > 0).sum())
    return rec


def _c26_module():
    path = ROOT / "scripts" / "21_summarise_c26.py"
    spec = importlib.util.spec_from_file_location("summarise_c26", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def robustness_comparators(fam: dict, prop: str, layer: int, lam: float) -> dict:
    """R6 is not allowed to stand on C23's own comparator alone.

    Two further prices for the same arm, both imported rather than re-implemented:

    * **token-conservative** -- `scripts/18_summarise_c23.py`'s post-hoc check.  N is
      solved by flooring, so the matched best-of-N can end up spending *fewer* tokens per
      returned molecule than guidance (for this arm the flooring took N from 9 to 8).
      This re-prices against the cheapest already-executed best-of-N run for the property
      that spends **at least as many** tokens, so best-of-N is never the poorer-funded side.
    * **C26's corrected curve** -- `scripts/21_summarise_c26.py`'s `interp`, on
      `outputs/c26_nsweep_{prop}/n_sweep_metrics.json`, the 3.6x larger estimator that
      C26 adopted because the slot estimator is optimistic in guidance's favour.
    """
    out: dict = {}
    present = fam["head_seeds_present"]
    if not present:
        return out
    tokens = {str(hs): fam["tokens_by_head_seed"][str(hs)] for hs in present}

    # ---- token-conservative (C23's rule, same candidate set) -------------------------
    cands = []
    for d in sorted(OUTPUT_DIR.glob("*bestofn_*")):
        f = d / "bestofn_metrics.json"
        if not f.exists():
            continue
        bm = read_json(f)
        if bm.get("property") != prop or "actual" not in bm.get("matches", {}):
            continue
        m = bm["matches"]["actual"]
        if list(m["seeds"].keys()) != list(GEN_SEEDS):
            continue
        cands.append((float(m["aggregate"]["tokens_per_molecule_actual"]), d.name, m))
    cons = {}
    for hs in present:
        elig = [c for c in cands if c[0] >= tokens[str(hs)]]
        if not elig:
            continue
        t, name, m = min(elig, key=lambda x: x[0])
        cons[str(hs)] = {
            "run": name, "n_candidates": m["n_candidates"],
            "tokens_per_molecule_actual": t,
            "hit_rate": float(m["aggregate"]["hit_rate"]["mean"]),
            "advantage": fam["hit_rate_by_head_seed"][str(hs)]
            - float(m["aggregate"]["hit_rate"]["mean"]),
            "realised_token_ratio_guided_over_best_of_n": tokens[str(hs)] / t,
        }
    if cons:
        a = np.array([v["advantage"] for v in cons.values()], dtype=float)
        out["token_conservative"] = {
            "note": "best-of-N is given at least as many tokens per returned molecule as "
                    "guidance, removing the flooring slack; imported from C23",
            "per_head_seed": cons, "t": t_interval(a),
            "bootstrap": head_seed_bootstrap(a),
            "n_positive": int((a > 0).sum())}

    # ---- C26's corrected curve --------------------------------------------------------
    f = OUTPUT_DIR / f"c26_nsweep_{prop}" / "n_sweep_metrics.json"
    if f.exists():
        C26 = _c26_module()
        sweep = read_json(f)
        grid = sorted(int(k) for k in sweep["curve"])
        xs = [float(sweep["curve"][str(n)]["tokens_per_molecule_actual"]) for n in grid]
        ys = [float(sweep["curve"][str(n)]["hit_rate_mean"]) for n in grid]
        rows = {}
        for hs in present:
            v, i, j, extrap = C26.interp(xs, ys, tokens[str(hs)])
            rows[str(hs)] = {
                "budget_tokens_per_molecule": tokens[str(hs)],
                "curve_hit_rate_at_budget": float(v),
                "bracket_n": [grid[i], grid[j]], "extrapolated": bool(extrap),
                "advantage": fam["hit_rate_by_head_seed"][str(hs)] - float(v)}
        a = np.array([r["advantage"] for r in rows.values()], dtype=float)
        out["c26_corrected_curve"] = {
            "note": "priced against C26's corrected best-of-N estimator, interpolated "
                    "linearly in tokens at each cell's own realised budget",
            "sweep_run": f.parent.name, "per_head_seed": rows,
            "t": t_interval(a), "bootstrap": head_seed_bootstrap(a),
            "n_positive": int((a > 0).sum())}
    return out


#: §C29.0.2 priority 4 -- the head seeds C27's equal-information comparison is replicated at.
C27_HEAD_SEEDS = (1234, 2345, 3456)
#: C27's published E4 figures, transcribed from reports/section_c27_head_selected_bestofn.md.
C27_E4_PUBLISHED = {"aromatic_rings": -0.0439, "hbd_count": -0.0292, "qed": -0.0522}


def c27_headsel_dir(prop: str, hs: int) -> Path:
    if hs == 1234:
        return OUTPUT_DIR / f"c27_headsel_{prop}"
    return OUTPUT_DIR / f"c29_c27_hs{hs}_{prop}"


def c27_across_head_seeds() -> dict:
    """Priority 4 -- C27's E4 comparison re-run at head seeds 2345 and 3456.

    C27's equal-information result rests on one head seed, and two of its three effects are
    smaller than the head-seed spread C29 measures at the deployed configuration.  Here
    **both** sides move with the head seed: the guided arm is the C29 deployed lambda = 1
    cell at that head seed, and the head-selected curve is built with the same checkpoint.
    The interpolation is `scripts/21_summarise_c26.py::interp`, imported, exactly as C27
    used it.
    """
    out: dict = {"published_e4": C27_E4_PUBLISHED, "per_property": {}}
    C26 = None
    for prop in ANCHORS:
        rows = {}
        for hs in C27_HEAD_SEEDS:
            f = c27_headsel_dir(prop, hs) / "head_selected_metrics.json"
            g = cell_stats(deployed_guided_dir(prop, hs), prop)
            if not f.exists() or g is None:
                continue
            if C26 is None:
                C26 = _c26_module()
            sweep = read_json(f)
            grid = sorted(int(k) for k in sweep["curves"]["head_selected"])
            xs = [float(sweep["curves"]["head_selected"][str(n)]
                        ["tokens_per_molecule_actual"]) for n in grid]
            ys = [float(sweep["curves"]["head_selected"][str(n)]["hit_rate_mean"])
                  for n in grid]
            v, i, j, extrap = C26.interp(xs, ys, g["tokens_per_molecule_actual"])
            ov = [float(sweep["curves"]["oracle_selected"][str(n)]["hit_rate_mean"])
                  for n in grid]
            ovi, _, _, _ = C26.interp(xs, ov, g["tokens_per_molecule_actual"])
            rows[str(hs)] = {
                "sweep_run": f.parent.name,
                "guided_run": g["dir"],
                "guided_hit_rate": g["hit_rate"],
                "guided_tokens_per_molecule_actual": g["tokens_per_molecule_actual"],
                "head_checkpoint": sweep["head"]["head_checkpoint_name"],
                "head_seed_in_checkpoint": sweep["head"]["head_seed"],
                "head_selected_at_budget": float(v),
                "oracle_selected_at_budget": float(ovi),
                "bracket_n": [grid[i], grid[j]], "extrapolated": bool(extrap),
                "advantage_vs_head_selected": g["hit_rate"] - float(v),
                "advantage_vs_oracle_selected": g["hit_rate"] - float(ovi),
            }
        if not rows:
            continue
        a = np.array([r["advantage_vs_head_selected"] for r in rows.values()], dtype=float)
        out["per_property"][prop] = {
            "per_head_seed": rows, "n_head_seeds": int(a.size),
            "mean": float(a.mean()),
            "span": float(a.max() - a.min()) if a.size > 1 else 0.0,
            "n_negative": int((a < 0).sum()),
            "sign_stable": bool(a.size > 1 and (np.all(a < 0) or np.all(a > 0))),
            "published_e4": C27_E4_PUBLISHED[prop],
            "note": "no interval is quoted: n = 3 head seeds, and C29.0.4.6 forbids a "
                    "three-point bootstrap; the per-head-seed values and their span are "
                    "the whole evidence",
        }
    out["all_signs_stable"] = bool(out["per_property"]) and all(
        v["sign_stable"] for v in out["per_property"].values())
    out["n_properties"] = len(out["per_property"])
    return out


def paired_difference(mid: dict, dep: dict) -> dict:
    """§C29.0.6 R4 -- head-seed-paired mid minus deployed, at the same lambda."""
    shared = [hs for hs in mid["head_seeds_present"] if hs in dep["head_seeds_present"]]
    d = np.array([mid["hit_rate_by_head_seed"][str(h)]
                  - dep["hit_rate_by_head_seed"][str(h)] for h in shared], dtype=float)
    out = {"head_seeds_paired": shared, "n_pairs": len(shared),
           "per_head_seed": {str(h): float(v) for h, v in zip(shared, d)},
           "lambda_matched": bool(mid["lam"] == dep["lam"]),
           "mid_lambda": mid["lam"], "deployed_lambda": dep["lam"]}
    if d.size:
        out["t"] = t_interval(d)
        out["bootstrap"] = head_seed_bootstrap(d)
        out["unpaired_difference_of_means"] = float(
            mid["mean_hit_rate"] - dep["mean_hit_rate"])
    return out


# ------------------------------------------------------------------ effective lambda

def envelope(prop: str, lambdas: tuple[float, ...]) -> dict:
    pts = {}
    for lam in sorted(lambdas):
        c = cell_stats(deployed_lambda_dir(prop, lam), prop)
        if c is not None:
            pts[f"{lam:g}"] = {"lam": lam, "hit_rate": c["hit_rate"], "dir": c["dir"],
                               "validity": c["validity"]}
    return pts


def interpolate_log2(points: dict, lam: float) -> dict:
    """Deployed hit rate at `lam`, linear in log2(lambda) between bracketing points.

    log2(lambda) is the axis `scripts/21_summarise_c26.py` uses; §C29.0.3 fixes it.
    """
    xs = np.array(sorted(p["lam"] for p in points.values()), dtype=float)
    ys = np.array([points[f"{x:g}"]["hit_rate"] for x in xs], dtype=float)
    if xs.size < 2:
        return {"ok": False, "reason": "fewer than two envelope points"}
    lx = np.log2(xs)
    t = float(np.log2(lam))
    if t < lx[0] or t > lx[-1]:
        return {"ok": False, "reason": f"lambda={lam:g} outside the measured envelope "
                                       f"[{xs[0]:g}, {xs[-1]:g}]"}
    j = int(np.searchsorted(lx, t))
    if j == 0:
        return {"ok": True, "hit_rate": float(ys[0]), "bracket": [float(xs[0]),
                                                                  float(xs[0])]}
    lo_x, hi_x, lo_y, hi_y = lx[j - 1], lx[j], ys[j - 1], ys[j]
    w = 0.0 if hi_x == lo_x else (t - lo_x) / (hi_x - lo_x)
    return {"ok": True, "hit_rate": float(lo_y + w * (hi_y - lo_y)),
            "bracket": [float(xs[j - 1]), float(xs[j])],
            "bracket_hit_rates": [float(lo_y), float(hi_y)],
            "weight_in_log2": float(w),
            "bracket_width_octaves": float(hi_x - lo_x)}


def effective_lambda_table(spread: dict, coarse: dict, fine: dict) -> list[dict]:
    rows = []
    for prop, layer, lam in C23_ARMS:
        g = mid_guided_dir(prop, layer, lam, 1234)
        c = cell_stats(g, prop)
        if c is None:
            continue
        r = spread[prop]["ratio"][str(layer)]
        lam_eff = lam * r
        dep_raw = cell_stats(deployed_lambda_dir(prop, lam), prop)
        row = {
            "property": prop, "probe_point": layer, "lam": lam,
            "spread_mid": spread[prop]["spread"][str(layer)],
            "spread_deployed": spread[prop]["spread"]["12"],
            "spread_ratio": r, "lambda_effective": lam_eff,
            "mid_hit_rate": c["hit_rate"], "mid_dir": c["dir"],
            "deployed_hit_rate_at_lam": dep_raw["hit_rate"] if dep_raw else None,
            "deployed_dir_at_lam": dep_raw["dir"] if dep_raw else None,
        }
        if dep_raw:
            row["advantage_raw"] = c["hit_rate"] - dep_raw["hit_rate"]
        for tag, env in (("coarse", coarse), ("fine", fine)):
            ip = interpolate_log2(env[prop], lam_eff)
            row[f"envelope_{tag}"] = ip
            if ip.get("ok"):
                row[f"advantage_effective_{tag}"] = c["hit_rate"] - ip["hit_rate"]
                if row.get("advantage_raw") is not None:
                    row[f"confound_share_{tag}"] = (
                        float((row["advantage_raw"] - row[f"advantage_effective_{tag}"])
                              / row["advantage_raw"])
                        if row["advantage_raw"] != 0 else None)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ gates

def tensor_residual(a: Path, b: Path) -> dict:
    """§C29.0.5 G1 -- max |a - b| over parameter tensors.  NOT a file-byte comparison.

    C27's gate 4 failed because `torch.save` names the zip archive after the output path,
    so identical tensors give different file hashes.  C29 compares tensors.
    """
    if not a.exists() or not b.exists():
        return {"comparable": False, "a": str(a), "b": str(b),
                "a_exists": a.exists(), "b_exists": b.exists()}
    ca = torch.load(a, map_location="cpu", weights_only=False)
    cb = torch.load(b, map_location="cpu", weights_only=False)
    sa, sb = ca["state_dict"], cb["state_dict"]
    if set(sa) != set(sb):
        return {"comparable": False, "a": a.name, "b": b.name,
                "reason": "different parameter names"}
    res = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)
    return {"comparable": True, "a": a.name, "b": b.name,
            "max_abs_parameter_residual": res, "identical": bool(res == 0.0),
            "n_tensors": len(sa),
            "binner_identical": bool(ca.get("binner") == cb.get("binner")),
            "n_bins_identical": bool(ca.get("n_bins") == cb.get("n_bins"))}


def gate_g1_g3(heads_dir: str) -> dict:
    rows = []
    for _k, prop, layer, _lam in MID_ARMS:
        rows.append({"role": "G1 mid vs C17", "property": prop, "probe_point": layer,
                     "head_seed": 1234,
                     **tensor_residual(
                         OUTPUT_DIR / heads_dir / f"head_{prop}_last1_L{layer}_seed1234.pt",
                         OUTPUT_DIR / "c17_probe_layers"
                         / f"head_{prop}_frozen_state_L{layer}.pt")})
    for prop in ANCHORS:
        rows.append({"role": "G1 deployed vs C17", "property": prop, "probe_point": 12,
                     "head_seed": 1234,
                     **tensor_residual(
                         OUTPUT_DIR / heads_dir / f"head_{prop}_last1_L12_seed1234.pt",
                         OUTPUT_DIR / "c17_probe_layers"
                         / f"head_{prop}_frozen_state_L12.pt")})
    for hs in (1234, 2345, 3456):
        for prop in ANCHORS:
            # The strongest available identity: the deployed head the published lambda=1
            # runs actually steered with, at the same head seed.
            rows.append({"role": "G1 deployed vs pilot_50k_heads_p2", "property": prop,
                         "probe_point": 12, "head_seed": hs,
                         **tensor_residual(
                             OUTPUT_DIR / heads_dir / f"head_{prop}_last1_L12_seed{hs}.pt",
                             OUTPUT_DIR / "pilot_50k_heads_p2"
                             / f"head_{prop}_frozen_state_seed{hs}.pt")})
        for _k, prop, layer, _lam in MID_ARMS:
            rows.append({"role": "G3 vs C25 pooled", "property": prop,
                         "probe_point": layer, "head_seed": hs,
                         **tensor_residual(
                             OUTPUT_DIR / heads_dir
                             / f"head_{prop}_last1_L{layer}_seed{hs}.pt",
                             OUTPUT_DIR / "c25_pooled_heads"
                             / f"head_{prop}_last1_L{layer}_seed{hs}.pt")})
        for prop in ANCHORS:
            rows.append({"role": "G3 deployed vs C25 pooled", "property": prop,
                         "probe_point": 12, "head_seed": hs,
                         **tensor_residual(
                             OUTPUT_DIR / heads_dir / f"head_{prop}_last1_L12_seed{hs}.pt",
                             OUTPUT_DIR / "c25_pooled_heads"
                             / f"head_{prop}_last1_L12_seed{hs}.pt")})
    comparable = [r for r in rows if r.get("comparable")]
    worst = max((r["max_abs_parameter_residual"] for r in comparable), default=None)
    return {"rule": "the C29 head at a shared seed must equal the published one, tensor "
                    "by tensor; file SHA-256 is deliberately not used (C27 gate 4)",
            "rows": rows, "n_comparable": len(comparable),
            "n_not_comparable": len(rows) - len(comparable),
            "max_abs_parameter_residual": worst,
            "passes": bool(comparable) and worst == 0.0}


def gate_g2() -> dict:
    """§C29.0.5 G2 -- two cheap end-to-end identity runs at head seed 1234."""
    pairs = (("c29_gate_L4_lam1_qed_hs1234", "c23_guided_L4_lam1_qed", "qed",
              "C23 mid arm, probe point 4, lambda = 1"),
             ("c29_gate_L12_lam1_qed_hs1234", "pilot_50k_p2_guided_qed", "qed",
              "the deployed arm, probe point 12, lambda = 1"))
    rows = []
    for rep_name, ref_name, prop, why in pairs:
        rep, ref = OUTPUT_DIR / rep_name, OUTPUT_DIR / ref_name
        if not (rep / "guidance_metrics.json").exists():
            rows.append({"replay": rep_name, "reference": ref_name, "why": why,
                         "run": False})
            continue
        a = read_json(ref / "guidance_metrics.json")
        b = read_json(rep / "guidance_metrics.json")
        am = read_json(ref / "molecules.json")
        bm = read_json(rep / "molecules.json")
        residuals, per_seed, n_cmp, identical = [], {}, 0, True
        for cond in ("unguided", "throughout"):
            for s in GEN_SEEDS:
                x = a["conditions"][cond]["seeds"][s]["hit_rate"]
                y = b["conditions"][cond]["seeds"][s]["hit_rate"]
                residuals.append(abs(x - y))
                per_seed[f"{cond}:{s}"] = {"reference_hit_rate": x, "replay_hit_rate": y,
                                           "residual": abs(x - y)}
                sa = [r["smiles"] for r in am[cond][s]]
                sb = [r["smiles"] for r in bm[cond][s]]
                n_cmp += len(sa)
                identical = identical and (sa == sb)
        rows.append({"replay": rep_name, "reference": ref_name, "why": why, "run": True,
                     "per_seed": per_seed,
                     "max_abs_hit_rate_residual": float(max(residuals)),
                     "molecules_identical": bool(identical),
                     "n_molecules_compared": n_cmp,
                     "replay_head_file": b.get("head_file"),
                     "reference_head_checkpoint": a.get("head_checkpoint"),
                     "passes": bool(max(residuals) == 0.0 and identical)})
    run_rows = [r for r in rows if r.get("run")]
    return {"rule": "the C29 pipeline at head seed 1234 must reproduce the published run "
                    "molecule by molecule",
            "rows": rows, "n_run": len(run_rows),
            "max_abs_hit_rate_residual":
                max((r["max_abs_hit_rate_residual"] for r in run_rows), default=None),
            "all_molecules_identical": bool(run_rows) and all(
                r["molecules_identical"] for r in run_rows),
            "passes": bool(run_rows) and all(r["passes"] for r in run_rows)}


def gate_g4(heads_dir: str) -> dict:
    """§C29.0.5 G4 -- training seed 1234 in a list of 8 must give what a list of 3 gave."""
    rows = []
    for depth, prop, layer in ([("mid", p, L) for _k, p, L, _l in MID_ARMS]
                               + [("final", p, 12) for p in ANCHORS]):
        mine = OUTPUT_DIR / heads_dir / f"cell_{depth}_L{layer}_{prop}_last1.json"
        theirs = OUTPUT_DIR / "c25_pooled_heads" / f"cell_{depth}_L{layer}_{prop}_last1.json"
        if not (mine.exists() and theirs.exists()):
            rows.append({"cell": mine.name, "comparable": False})
            continue
        a = {int(e["head_seed"]): e for e in read_json(mine)["per_seed"]}
        b = {int(e["head_seed"]): e for e in read_json(theirs)["per_seed"]}
        for hs in sorted(set(a) & set(b)):
            ta = a[hs]["test"]["intervals"]["target"]["auroc"]
            tb = b[hs]["test"]["intervals"]["target"]["auroc"]
            rows.append({"cell": mine.name, "head_seed": hs, "comparable": True,
                         "auroc_c29": ta, "auroc_c25": tb,
                         "auroc_residual": abs(ta - tb),
                         "nll_residual": abs(a[hs]["test"]["nll"]
                                             - b[hs]["test"]["nll"])})
    ok = [r for r in rows if r.get("comparable")]
    worst = max((max(r["auroc_residual"], r["nll_residual"]) for r in ok), default=None)
    return {"rule": "per-head-seed AUROC and NLL must not depend on which other seeds "
                    "were in the --head-seeds list",
            "rows": rows, "n_comparable": len(ok), "max_residual": worst,
            "passes": bool(ok) and worst == 0.0}


def gate_g5() -> dict:
    rows = []
    for lam in NEW_LAMBDAS:
        for prop in ANCHORS:
            d = deployed_lambda_dir(prop, lam)
            f = d / "guidance_metrics.json"
            if not f.exists():
                rows.append({"dir": d.name, "run": False})
                continue
            m = read_json(f)
            rows.append({"dir": d.name, "run": True, "property": prop,
                         "lambda": m["lambda"], "lambda_requested": lam,
                         "lambda_residual": abs(float(m["lambda"]) - lam),
                         "head_checkpoint": m.get("head_checkpoint"),
                         "layer": m.get("layer"),
                         "layer_source": m.get("layer_source"),
                         "lambda_source": m.get("lambda_source"),
                         "head_file_source": m.get("head_file_source"),
                         "on_deployed_code_path": bool(
                             m.get("layer") == -1
                             and m.get("layer_source") == "default (-1)"
                             and m.get("head_file_source") == "default"
                             and m.get("head_checkpoint")
                             == f"head_{prop}_frozen_state.pt")})
    run_rows = [r for r in rows if r.get("run")]
    return {"rule": "each new deployed lambda run must be the published deployed code "
                    "path with only lambda changed",
            "rows": rows, "n_run": len(run_rows),
            "max_lambda_residual":
                max((r["lambda_residual"] for r in run_rows), default=None),
            "passes": bool(run_rows) and all(r["on_deployed_code_path"]
                                             for r in run_rows)}


def gate_g6(families: dict) -> dict:
    """§C29.0.5 G6 -- `unguided` cannot depend on head seed, probe point or lambda."""
    per_prop: dict[str, list[float]] = {}
    for fam in families.values():
        for v in fam.get("unguided_by_head_seed", {}).values():
            per_prop.setdefault(fam["property"], []).append(float(v))
    rows = {}
    for prop, vs in per_prop.items():
        a = np.array(vs, dtype=float)
        rows[prop] = {"n_runs": int(a.size), "mean": float(a.mean()),
                      "span": float(a.max() - a.min()),
                      "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0}
    worst = max((r["span"] for r in rows.values()), default=None)
    return {"rule": "the unguided condition is a bug alarm, not a finding: its spread "
                    "across every C29 run of an anchor must be 0.0",
            "per_property": rows, "max_span": worst,
            "passes": bool(rows) and worst == 0.0}


# ------------------------------------------------------------------ rules

def score_rules(fam: dict, paired: dict, eff_rows: list[dict],
                eff_paired: dict) -> dict:
    rules: dict = {}
    a1 = fam.get("A1")

    # R1 -- the head-seed sd is real
    if a1 and a1.get("generation_seed_sd_pooled") and a1["head_seed_sd"]["n"] > 1:
        sd_h, sd_g = a1["head_seed_sd"], a1["generation_seed_sd_pooled"]["sd"]
        rules["R1"] = {
            "rule": "on A1 the lower bound of the 95% chi-square CI on the head-seed sd "
                    "exceeds the pooled generation-seed sd",
            "head_seed_sd": sd_h["sd"], "head_seed_sd_ci": [sd_h["lo"], sd_h["hi"]],
            "generation_seed_sd_pooled": sd_g, "n_head_seeds": sd_h["n"],
            "fires": bool(sd_h["lo"] > sd_g)}

    # R2 -- the variance ratio excludes 1
    if a1 and a1.get("variance_ratio"):
        vr = a1["variance_ratio"]
        rules["R2"] = {
            "rule": "on A1 the 95% F-based CI on sd_head / sd_gen excludes 1",
            "ratio": vr["ratio"], "ci": [vr["lo"], vr["hi"]],
            "c25_reported_span_ratio": C25_RATIO,
            "c25_span_ratio_inside_interval": vr["c25_span_ratio_inside_interval"],
            "fires": bool(vr["lo"] > 1.0 or vr["hi"] < 1.0)}

    # R3 -- mid-network probes, or probes in general?
    q_rows = {}
    for key, prop, _L, _lam in MID_ARMS:
        mid, dep = fam.get(key), fam.get(f"D_{prop}")
        if not (mid and dep):
            continue
        if mid["head_seed_sd"]["n"] < 2 or dep["head_seed_sd"]["n"] < 2:
            continue
        sm, sd_ = mid["head_seed_sd"]["sd"], dep["head_seed_sd"]["sd"]
        q = float(sm / sd_) if sd_ > 0 else float("inf")
        q_rows[prop] = {
            "mid_arm": key, "mid_head_seed_sd": sm, "mid_n": mid["head_seed_sd"]["n"],
            "deployed_head_seed_sd": sd_, "deployed_n": dep["head_seed_sd"]["n"],
            "q": q, "in_band": bool(R3_BAND[0] <= q <= R3_BAND[1]),
            "ratio_ci": sd_ratio_with_f_ci(sm, mid["head_seed_sd"]["n"] - 1,
                                           sd_, dep["head_seed_sd"]["n"] - 1)}
    n_band = sum(1 for v in q_rows.values() if v["in_band"])
    n_above = sum(1 for v in q_rows.values() if v["q"] > R3_BAND[1])
    rules["R3"] = {
        "rule": "q = sd_head(mid) / sd_head(deployed) in [0.5, 2] on >= 2 of 3 anchors "
                "means the finding is about learned probes in general",
        "band": list(R3_BAND), "per_anchor": q_rows,
        "n_in_band": n_band, "n_above_band": n_above,
        "conclusion": ("probes in general" if n_band >= 2
                       else "specific to mid-network probes" if n_above >= 2
                       else "undetermined"),
        "fires": bool(n_band >= 2)}

    # R4 -- does Rule A survive head-seed variation?
    r4 = {}
    for key, prop, _L, _lam in MID_ARMS:
        p = paired.get(key)
        if not p or not p.get("t"):
            continue
        r4[key] = {"property": prop, "n_pairs": p["n_pairs"],
                   "lambda_matched": p["lambda_matched"],
                   "mean": p["t"]["mean"], "ci": [p["t"]["lo"], p["t"]["hi"]],
                   "bootstrap_ci": ([p["bootstrap"]["lo"], p["bootstrap"]["hi"]]
                                    if p["bootstrap"].get("computed") else None),
                   "survives": bool(p["t"]["lo"] > 0 and p["lambda_matched"])}
    n4 = sum(1 for v in r4.values() if v["survives"])
    rules["R4"] = {
        "rule": "head-seed-paired mid minus deployed at matched lambda is strictly "
                "positive on >= 2 of 3 anchors",
        "per_arm": r4, "n_surviving": n4, "required": 2, "fires": bool(n4 >= 2),
        "preregistration_defect": (
            "A1 sits at lambda = 2 but the deployed family was fixed at lambda = 1 in "
            "C29.0.2, so R4 as written is not scoreable at matched lambda on A1. This is "
            "a defect in the pre-registration, discovered when scoring it, and is "
            "reported rather than amended."),
        "arms_not_scoreable_at_matched_lambda":
            [k for k, v in r4.items() if not v["lambda_matched"]]}

    # R5 -- does Rule A survive the effective-lambda correction?
    r5_rows = {k: v for k, v in eff_paired.items()}
    n5 = sum(1 for v in r5_rows.values()
             if v.get("t") and v["t"]["mean"] > 0 and v["t"]["lo"] > 0)
    rules["R5"] = {
        "rule": "head-seed mean of the effective-lambda-corrected advantage is positive "
                "with a t interval excluding 0, on >= 2 of 3 anchors",
        "per_arm": r5_rows, "n_surviving": n5, "required": 2, "fires": bool(n5 >= 2),
        "seed_1234_table": eff_rows}

    # R6 -- Rule B at n >= 6
    if a1 and a1.get("advantage_over_best_of_n"):
        adv = a1["advantage_over_best_of_n"]
        rules["R6"] = {
            "rule": "on A1 the head-seed mean advantage over its own compute-matched "
                    "best-of-N has a t interval strictly above 0",
            "n_head_seeds": adv["n"], "mean": adv["mean"], "sd": adv["sd"],
            "ci": [adv["lo"], adv["hi"]],
            "bootstrap_ci": ([a1["advantage_over_best_of_n_bootstrap"]["lo"],
                              a1["advantage_over_best_of_n_bootstrap"]["hi"]]
                             if a1["advantage_over_best_of_n_bootstrap"].get("computed")
                             else None),
            "n_positive": a1["n_head_seeds_with_positive_advantage"],
            "meets_preregistered_minimum": bool(adv["n"] >= MIN_HEAD_SEEDS),
            "fires": bool(adv["lo"] > 0)}
        # R6 fires or not against C23's own comparator.  Two harder prices are attached
        # here so the verdict cannot be quoted without them.
        rb = a1.get("robustness", {})
        for tag in ("token_conservative", "c26_corrected_curve"):
            if tag in rb:
                rules["R6"][f"also_{tag}"] = {
                    "mean": rb[tag]["t"]["mean"],
                    "ci": [rb[tag]["t"]["lo"], rb[tag]["t"]["hi"]],
                    "n_positive": rb[tag]["n_positive"],
                    "fires": bool(rb[tag]["t"]["lo"] > 0)}
        rules["R6"]["survives_every_comparator"] = bool(
            rules["R6"]["fires"]
            and all(rules["R6"].get(f"also_{t}", {"fires": True})["fires"]
                    for t in ("token_conservative", "c26_corrected_curve")))

    fired = [k for k in ("R1", "R2", "R4", "R5", "R6")
             if rules.get(k, {}).get("fires")]
    rules["R7"] = {"rule": "null: none of R1, R2, R4, R5, R6 fires",
                   "rules_firing": fired, "fires": bool(not fired)}
    return rules


def score_predictions(fam: dict, rules: dict, eff_rows: list[dict]) -> dict:
    out: dict = {}
    a1 = fam.get("A1")

    if a1 and a1["head_seed_sd"]["n"] > 1:
        s = a1["head_seed_sd"]["sd"]
        out["P1_sd_head_A1_in_0.030_0.080"] = {
            "prediction": "sd_head on A1 lands in [0.030, 0.080]",
            "value": s, "holds": bool(0.030 <= s <= 0.080)}
    if a1 and a1.get("generation_seed_sd_pooled"):
        g = a1["generation_seed_sd_pooled"]["sd"]
        out["P2_sd_gen_A1_below_0.010"] = {
            "prediction": "pooled generation-seed sd on A1 is below 0.010",
            "value": g, "holds": bool(g < 0.010)}
    if a1 and a1.get("variance_ratio"):
        vr = a1["variance_ratio"]
        out["P3_ratio_above_5_and_25_not_pinned"] = {
            "prediction": "the sd ratio on A1 is above 5 and its 95% CI contains values "
                          "below 25",
            "ratio": vr["ratio"], "ci": [vr["lo"], vr["hi"]],
            "above_5": bool(vr["ratio"] > 5.0),
            "ci_contains_values_below_25": bool(vr["lo"] < 25.0),
            "holds": bool(vr["ratio"] > 5.0 and vr["lo"] < 25.0)}
    if "R3" in rules:
        out["P4_probes_in_general"] = {
            "prediction": "R3 concludes 'probes in general'",
            "conclusion": rules["R3"]["conclusion"],
            "n_in_band": rules["R3"]["n_in_band"],
            "holds": bool(rules["R3"]["conclusion"] == "probes in general")}
    if "R4" in rules and rules["R4"]["per_arm"]:
        pa = rules["R4"]["per_arm"]
        n_pos = sum(1 for v in pa.values() if v["mean"] > 0)
        n_sig = sum(1 for v in pa.values() if v["ci"][0] > 0)
        out["P5_rule_a_survives_in_sign_not_significance"] = {
            "prediction": "the paired advantage is positive on all 3 anchors but its t "
                          "interval excludes 0 on at most 1, so R4 does not fire",
            "n_arms": len(pa), "n_positive_mean": n_pos, "n_excluding_zero": n_sig,
            "holds": bool(n_pos == len(pa) and n_sig <= 1 and not rules["R4"]["fires"])}

    reviewer = {("aromatic_rings", 3, 1.0): 0.0459,
                ("aromatic_rings", 6, 1.0): 0.0592,
                ("hbd_count", 4, 1.0): 0.0245}
    checks = []
    for row in eff_rows:
        key = (row["property"], row["probe_point"], row["lam"])
        if key not in reviewer:
            continue
        v = row.get("advantage_effective_fine", row.get("advantage_effective_coarse"))
        if v is None:
            continue
        checks.append({"arm": f"{key[0]}_L{key[1]}_{lam_tag(key[2])}",
                       "reviewer_value": reviewer[key], "measured": v,
                       "abs_difference": abs(v - reviewer[key]),
                       "within_0.02": bool(abs(v - reviewer[key]) <= 0.02),
                       "sign_survives": bool(v > 0)})
    if checks:
        out["P6_effective_lambda_removes_about_half"] = {
            "prediction": "the effective-lambda-corrected advantage lands within +-0.02 "
                          "of the reviewer's coarse arithmetic on three named arms, and "
                          "the sign survives on all three",
            "per_arm": checks,
            "n_within_0.02": sum(1 for c in checks if c["within_0.02"]),
            "n_sign_survives": sum(1 for c in checks if c["sign_survives"]),
            "holds": bool(all(c["within_0.02"] for c in checks)
                          and all(c["sign_survives"] for c in checks))}
    if "R6" in rules:
        out["P7_rule_b_does_not_fire"] = {
            "prediction": "the head-seed mean advantage over best-of-N on A1 is positive "
                          "but its t interval contains 0",
            "mean": rules["R6"]["mean"], "ci": rules["R6"]["ci"],
            "holds": bool(rules["R6"]["mean"] > 0 and not rules["R6"]["fires"])}

    sds = {k: fam[k]["head_seed_sd"]["sd"] for k, _p, _L, _l in MID_ARMS
           if k in fam and fam[k]["head_seed_sd"]["n"] > 1}
    if len(sds) == 3:
        out["P8_qed_smallest_hbd_largest"] = {
            "prediction": "qed has the smallest head-seed sd of the three mid anchors "
                          "and hbd_count the largest",
            "sd_by_arm": sds,
            "smallest": min(sds, key=sds.get), "largest": max(sds, key=sds.get),
            "holds": bool(min(sds, key=sds.get) == "A3"
                          and max(sds, key=sds.get) == "A1")}
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="c29_heads")
    ap.add_argument("--out", default="c29_summary")
    args = ap.parse_args()
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- C17's spread table, read verbatim -----------------------------------------
    lsm = read_json(STEERING_METRICS)
    spread: dict = {}
    for prop in ANCHORS:
        layers = lsm["properties"][prop]["layers"]
        ref = float(layers["12"]["mean_head_q_spread_across_candidates"])
        spread[prop] = {
            "source": "c17_layer_steering/layer_steering_metrics.json"
                      " :: mean_head_q_spread_across_candidates",
            "spread": {k: float(v["mean_head_q_spread_across_candidates"])
                       for k, v in layers.items()},
            "ratio": {k: float(v["mean_head_q_spread_across_candidates"]) / ref
                      for k, v in layers.items()},
            "deployed_probe_point": 12}

    # ---- families -------------------------------------------------------------------
    fam: dict = {}
    for key, prop, layer, lam in MID_ARMS:
        fam[key] = collect_family(
            key, prop, layer, lam,
            lambda p, hs, L=layer, lm=lam: mid_guided_dir(p, L, lm, hs),
            with_bestofn=True)
        fam[key]["robustness"] = robustness_comparators(fam[key], prop, layer, lam)
    for prop in ANCHORS:
        fam[f"D_{prop}"] = collect_family(
            f"D_{prop}", prop, DEPLOYED_PROBE_POINT, 1.0,
            lambda p, hs: deployed_guided_dir(p, hs), with_bestofn=False)
    # POST-HOC: the deployed lambda = 2 family on hbd_count, so A1's R4 pairing exists.
    fam["D2_hbd_count"] = collect_family(
        "D2_hbd_count", "hbd_count", DEPLOYED_PROBE_POINT, 2.0,
        lambda p, hs: deployed_lam2_guided_dir(p, hs), with_bestofn=False)
    fam["D2_hbd_count"]["post_hoc"] = (
        "added after scoring exposed the C29.0.2 / R4 lambda mismatch; the defect is "
        "reported in the section rather than amended out of the pre-registration")

    paired = {}
    for key, prop, _L, lam in MID_ARMS:
        dep_key = f"D_{prop}"
        if lam == 2.0 and prop == "hbd_count" and \
                fam["D2_hbd_count"]["n_head_seeds"] > 1:
            dep_key = "D2_hbd_count"
        if key in fam and dep_key in fam:
            paired[key] = paired_difference(fam[key], fam[dep_key])
            paired[key]["deployed_family"] = dep_key
            paired[key]["post_hoc_comparator"] = bool(dep_key == "D2_hbd_count")
    # The as-written R4 comparison for A1 (deployed lambda = 1) is kept alongside, so the
    # pre-registration can be scored verbatim as well as repaired.
    if "A1" in fam and "D_hbd_count" in fam:
        paired["A1_as_preregistered"] = paired_difference(fam["A1"], fam["D_hbd_count"])
        paired["A1_as_preregistered"]["deployed_family"] = "D_hbd_count"

    # ---- effective lambda ------------------------------------------------------------
    coarse = {p: envelope(p, COARSE_LAMBDAS) for p in ANCHORS}
    fine = {p: envelope(p, COARSE_LAMBDAS + NEW_LAMBDAS) for p in ANCHORS}
    eff_rows = effective_lambda_table(spread, coarse, fine)

    # per-head-seed effective-lambda-corrected advantage for the three C29 mid arms.
    # The deployed envelope exists at head seed 1234 only, so the variance here is the
    # mid arm's alone; that is stated in the section and is not hidden.
    eff_paired = {}
    for key, prop, layer, lam in MID_ARMS:
        r = spread[prop]["ratio"][str(layer)]
        lam_eff = lam * r
        ip_fine = interpolate_log2(fine[prop], lam_eff)
        ip_coarse = interpolate_log2(coarse[prop], lam_eff)
        ip = ip_fine if ip_fine.get("ok") else ip_coarse
        used = "fine" if ip_fine.get("ok") else "coarse"
        if not ip.get("ok"):
            eff_paired[key] = {"property": prop, "ok": False, "reason": ip.get("reason")}
            continue
        vals = np.array(list(fam[key]["hit_rate_by_head_seed"].values()), dtype=float)
        d = vals - ip["hit_rate"]
        dep_raw = cell_stats(deployed_lambda_dir(prop, lam), prop)
        raw = (vals - dep_raw["hit_rate"]) if dep_raw else None
        eff_paired[key] = {
            "property": prop, "ok": True, "probe_point": layer, "lam": lam,
            "spread_ratio": r, "lambda_effective": lam_eff,
            "envelope_used": used, "deployed_at_lambda_effective": ip["hit_rate"],
            "envelope_bracket": ip.get("bracket"),
            "deployed_at_lambda": dep_raw["hit_rate"] if dep_raw else None,
            "n_head_seeds": int(vals.size),
            "per_head_seed_effective":
                {k: float(v) for k, v in zip(fam[key]["hit_rate_by_head_seed"], d)},
            "t": t_interval(d), "bootstrap": head_seed_bootstrap(d),
            "raw_t": t_interval(raw) if raw is not None else None,
            "note": "the deployed envelope is head seed 1234 only, so this interval "
                    "carries the mid arm's head-seed variance and not the deployed "
                    "arm's",
        }

    # ---- gates -----------------------------------------------------------------------
    gates = {"G1_G3_checkpoint_identity": gate_g1_g3(args.heads),
             "G2_end_to_end_identity": gate_g2(),
             "G4_seed_list_independence": gate_g4(args.heads),
             "G5_envelope_provenance": gate_g5(),
             "G6_unguided_invariance": gate_g6(fam)}

    rules = score_rules(fam, paired, eff_rows, eff_paired)
    preds = score_predictions(fam, rules, eff_rows)

    summary = {
        "dataset": DATASET,
        "generation_seeds": list(GEN_SEEDS),
        "head_seeds_preregistered": list(HEAD_SEEDS),
        "preregistered_minimum_head_seeds": MIN_HEAD_SEEDS,
        "prereg_lock": read_json(OUTPUT_DIR / "c29_prereg" / "prereg_lock.json"),
        "heads_dir": args.heads,
        "spread_table": spread,
        "families": fam,
        "paired_mid_minus_deployed": paired,
        "effective_lambda": {
            "definition": "lambda_eff = lambda * spread(prop, L) / spread(prop, 12); the "
                          "deployed comparator is interpolated linearly in log2(lambda) "
                          "between bracketing measured points",
            "caveat": "the lambda-rescale identity is pointwise; the spread ratio is a "
                      "scalar moment ratio, so this is a first-order control, not an "
                      "identity",
            "envelope_coarse": coarse, "envelope_fine": fine,
            "seed_1234_table": eff_rows,
            "per_head_seed": eff_paired,
        },
        "c25_reference": {
            "hbd_L4_lam2_span_n3": C25_HBD_SPAN_N3,
            "generation_seed_span": C25_GEN_SPAN,
            "reported_ratio": C25_RATIO,
            "sigma_from_span_n3": C25_HBD_SPAN_N3 / D2[3],
            "note": "C25's 25x is a ratio of a 3-point span to a 3-point span; C29 "
                    "restates it as a ratio of sds",
        },
        "validity_gates": gates,
        "decision_rules": rules,
        "predictions": preds,
        "priority_4_c27_across_head_seeds": c27_across_head_seeds(),
    }
    write_json(out_dir / "c29_metrics.json", summary)
    write_run_context(out_dir)

    # ---- console ---------------------------------------------------------------------
    print("=== C29 validity gates ===")
    for k, g in gates.items():
        print(f"  {k:32s} {'PASS' if g.get('passes') else 'CHECK':6s} "
              f"{ {kk: vv for kk, vv in g.items() if kk.startswith('max_') } }")
    print("\n=== families ===")
    hdr = (f"{'arm':16s} {'prop':16s} {'n':>2s} {'mean':>8s} {'sd_head':>9s} "
           f"{'sd_gen':>9s} {'ratio':>8s} {'ratio_ci':>20s}")
    print(hdr)
    for k, f in fam.items():
        vr = f.get("variance_ratio")
        print(f"{k:16s} {f['property']:16s} {f['n_head_seeds']:2d} "
              f"{f['mean_hit_rate']:8.4f} {f['head_seed_sd']['sd']:9.4f} "
              f"{(f.get('generation_seed_sd_pooled') or {}).get('sd', float('nan')):9.4f} "
              f"{(vr or {}).get('ratio', float('nan')):8.2f} "
              + (f"[{vr['lo']:8.2f},{vr['hi']:8.2f}]" if vr else ""))
    print("\n=== decision rules ===")
    for k, v in rules.items():
        print(f"  {k:4s} {'FIRES' if v.get('fires') else 'does not fire':14s} "
              f"{v['rule'][:96]}")
    print("\n=== predictions ===")
    for k, v in preds.items():
        print(f"  {'HOLDS ' if v.get('holds') else 'FAILS '} {k}")
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
