"""C32 -- decompose C31's crossing into depth, lambda and their interaction.

Reads only artefacts that already exist: C31's cells and oracle-selected curves, C32's two
new 2x2 corners, C32's lambda envelope and C32's `q` spread measurement.  **Generates
nothing** -- no molecule is sampled, no head is trained, no best-of-N pool is drawn.

Every piece of the frontier machinery is imported and unmodified:
`scripts/21_summarise_c26.py::interp` / `::t_interval` and
`scripts/25_summarise_c31.py::price` / `::load_curve`, so a C32 advantage is computed by
exactly the code that produced C26's, C28's, C30's and C31's.

Scores `outputs/c32_prereg/C32.0_preregistration.md`: gates G1-G5, decision rules D1-D7 and
predictions P1-P11, including where they fail.

**No bootstrap.**  C32.0.4: at n = 3 the percentile bootstrap of a mean is identically
[min, max].  Per-seed values and a seed-level t interval on 2 df (t_0.975,2 = 4.302653).

    .venv/bin/python scripts/26_summarise_c32.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_summarise")
_c31s = _load_module(ROOT / "scripts" / "25_summarise_c31.py", "c31_summarise")
_c32 = _load_module(ROOT / "scripts" / "26_c32_depth_vs_lambda.py", "c32_depth_vs_lambda")

interp = _c26.interp
t_interval = _c26.t_interval
price = _c31s.price
load_curve = _c31s.load_curve
CORNERS = _c32.CORNERS
NEW_CORNERS = _c32.NEW_CORNERS
K_GRID = _c32.K_GRID
PRIMARY_K = _c32.PRIMARY_K
ENVELOPE_LAMBDAS = _c32.ENVELOPE_LAMBDAS
ENVELOPE_K = _c32.ENVELOPE_K
corner_dir = _c32.corner_dir
envelope_dir = _c32.envelope_dir
probe_point_of = _c32.probe_point_of
mid_probe_point = _c32.mid_probe_point
ALL_PROPERTIES = _c32.ALL_C31_PROPERTIES

#: C32.0.6.  Every threshold on this page comes from the pre-registration.
DOMINANCE_MARGIN = 0.02
INTERACTION_MARGIN = 0.02
CONFOUND_SHARE_THRESHOLD = 0.50
CROSSING_VALIDITY_FLOOR = 0.80
#: C32.0.8.  More than this many 2x2 cells below the floor voids the run.
MAX_DEGENERATE_CELLS = 6


def cell_stats(d: Path, curve: dict, seeds: list[str]) -> dict | None:
    """One cell's advantage over the fixed C31 oracle curve, priced by C31's own `price`."""
    f = d / "k_cell_metrics.json"
    if not f.exists():
        return None
    j = read_json(f)
    agg = j["aggregate"]
    per_seed = {s: {"hit_rate": j["seeds_detail"][s]["hit_rate"],
                    "tokens_per_molecule_actual":
                        j["seeds_detail"][s]["compute"]["tokens_per_molecule_actual"],
                    "validity": j["seeds_detail"][s]["validity"]}
                for s in seeds if s in j["seeds_detail"]}
    row = {
        "dir": d.name,
        "hit_rate_mean": agg["hit_rate"]["mean"],
        "hit_rate_values": agg["hit_rate"]["values"],
        "tokens_per_molecule_actual": agg["compute_total"]["tokens_per_molecule_actual"],
        "validity_mean": agg["validity"]["mean"],
        "validity_values": agg["validity"]["values"],
        "uniqueness_mean": agg["uniqueness"]["mean"],
        "content_length_mean": agg["content_length_mean"]["mean"],
        "cost_identity_max_residual": j["cost_identity_max_residual"],
        "per_seed": per_seed,
        "probe_point": j["probe_point"],
        "lam": j["lambda"],
        "k": j["top_k"],
    }
    row.update(price(curve, row["hit_rate_mean"], row["tokens_per_molecule_actual"],
                     per_seed, seeds))
    # per-seed advantage, keyed by seed, so the 2x2 contrasts can be paired
    row["advantage_by_seed"] = dict(zip(seeds, row["advantage_vs_oracle_selected_per_seed"]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c32_summary")
    args = ap.parse_args()

    cfg = load_config("c31_second_generator")
    seeds = [str(s) for s in cfg["generation_seeds"]]
    out_dir = OUTPUT_DIR / args.out

    report: dict = {
        "experiment": "C32",
        "question": ("is C31's crossing a depth effect, a lambda effect, or their "
                     "interaction?  C31 ran only two corners of a 2x2 and both factors "
                     "moved together."),
        "prereg": "outputs/c32_prereg/C32.0_preregistration.md",
        "prereg_lock": read_json(OUTPUT_DIR / "c32_prereg" / "prereg_lock.json"),
        "generator": {"repo": cfg["model_repo"], "revision": cfg["model_revision"]},
        "comparator": ("C31's oracle-selected best-of-N curves, read and never regenerated "
                       "(gate G5)"),
        "accounting": "processed generator tokens (actual)",
        "seeds": seeds,
        "uncertainty": ("per-seed values and a seed-level t interval on 2 df "
                        "(t_0.975,2 = 4.302653); no bootstrap anywhere in C32"),
        "validity_gates": {},
        "mid_probe_points": {},
        "cells": {},
        "decomposition": {},
        "spread": {},
        "effective_lambda": {},
        "decision_rules": {},
        "predictions": {},
    }

    for g, f in (("G1", "g1_reproduces_c31.json"), ("G2", "g2_frozen_artefacts.json"),
                 ("G5", "g5_comparator.json")):
        p = OUTPUT_DIR / "c32_gates" / f
        if p.exists():
            report["validity_gates"][g] = read_json(p)

    curves = {p: load_curve(p) for p in ALL_PROPERTIES}
    for p in ALL_PROPERTIES:
        report["mid_probe_points"][p] = {
            "M": mid_probe_point(p),
            "source": ("TRANSCRIBED from outputs/c31_heads/depth_*.json; selected by "
                       "C31.0.4's held-out validation AUROC rule before any steering "
                       "outcome existed, and NOT re-selected in C32"),
        }

    # ------------------------------------------------------------------ the 2x2 cells
    cells: dict[str, dict] = {}
    for prop in ALL_PROPERTIES:
        cv = curves.get(prop)
        if cv is None:
            continue
        for corner in CORNERS:
            for k in K_GRID:
                st = cell_stats(corner_dir(prop, corner, k), cv, seeds)
                if st is None:
                    continue
                st["property"], st["corner"] = prop, corner
                st["source"] = CORNERS[corner]["source"]
                st["depth"] = CORNERS[corner]["depth"]
                st["dropped_on_validity"] = bool(st["validity_mean"] < CROSSING_VALIDITY_FLOOR)
                st["crosses"] = bool(
                    st["advantage_vs_oracle_selected"] > 0
                    and st["advantage_vs_oracle_selected_seed_t_interval"]["lo"] > 0
                    and not st["dropped_on_validity"])
                cells[f"{prop}_{corner}_k{k}"] = st
    report["cells"] = cells

    # -------------------------------------------------------------------- gate G3, G4
    g3 = {n: c["cost_identity_max_residual"] for n, c in cells.items()}
    report["validity_gates"]["G3"] = {
        "gate": "G3", "rule": "processed_tokens_actual mod (k+1) == 0 in every cell",
        "cells": g3, "n_cells_checked": len(g3),
        "max_residual": max(g3.values()) if g3 else None,
        "passes": bool(g3) and max(g3.values()) == 0}

    # ------------------------------------------------------------- the decomposition
    # C32.0.4's arithmetic, transcribed:
    #   a = A(12,1)  b = A(12,2)  c = A(M,1)  d = A(M,2)
    #   depth  = 0.5*((c-a)+(d-b));  lambda = 0.5*((b-a)+(d-c))
    #   inter  = 0.5*((d-c)-(b-a));  and  d-a = depth+lambda+inter
    decomp: dict = {}
    g4_worst = 0.0
    g4_written_worst: list[float] = []
    dropped: list[str] = []
    for prop in ALL_PROPERTIES:
        for k in K_GRID:
            names = {x: f"{prop}_{c}_k{k}" for x, c in
                     (("a", "deployed_l1"), ("b", "deployed_l2"),
                      ("c", "mid_l1"), ("d", "mid_l2"))}
            got = {x: cells.get(n) for x, n in names.items()}
            missing = [names[x] for x, v in got.items() if v is None]
            bad = [names[x] for x, v in got.items() if v is not None and v["dropped_on_validity"]]
            key = f"{prop}_k{k}"
            if missing or bad:
                dropped.extend(bad)
                decomp[key] = {"property": prop, "k": k, "complete": False,
                               "missing": missing, "dropped_on_validity": bad}
                continue
            per_seed = {"depth_main": [], "lambda_main": [], "interaction": [],
                        "total_d_minus_a": []}
            for s in seeds:
                a = got["a"]["advantage_by_seed"][s]
                b = got["b"]["advantage_by_seed"][s]
                c = got["c"]["advantage_by_seed"][s]
                d = got["d"]["advantage_by_seed"][s]
                dep = 0.5 * ((c - a) + (d - b))
                lam = 0.5 * ((b - a) + (d - c))
                inter = 0.5 * ((d - c) - (b - a))
                per_seed["depth_main"].append(dep)
                per_seed["lambda_main"].append(lam)
                per_seed["interaction"].append(inter)
                per_seed["total_d_minus_a"].append(d - a)
                # C32.0.4 claims `d - a == depth + lambda + interaction`.  That claim is
                # ARITHMETICALLY FALSE and the falsity is reported, not amended away (see
                # `preregistration_defects` below).  With the pre-registered 0.5
                # convention the identity that actually holds is
                #     d - a == depth_main + lambda_main
                # and the interaction is a separate contrast, half the conventional
                # `(d-c) - (b-a)`.  G4 checks the true identities.
                g4_worst = max(g4_worst, abs((dep + lam) - (d - a)))
                g4_worst = max(g4_worst, abs(inter - 0.5 * ((d - b) - (c - a))))
                g4_as_written = abs((dep + lam + inter) - (d - a))
                g4_written_worst.append(g4_as_written)
            row = {"property": prop, "k": k, "complete": True,
                   "corners": {x: {"dir": got[x]["dir"],
                                   "probe_point": got[x]["probe_point"],
                                   "lam": got[x]["lam"],
                                   "advantage": got[x]["advantage_vs_oracle_selected"],
                                   "advantage_by_seed": got[x]["advantage_by_seed"],
                                   "hit_rate_mean": got[x]["hit_rate_mean"],
                                   "validity_mean": got[x]["validity_mean"]}
                               for x in ("a", "b", "c", "d")},
                   "primary_k": k in PRIMARY_K}
            for name in ("depth_main", "lambda_main", "interaction", "total_d_minus_a"):
                v = per_seed[name]
                row[name] = {"mean": float(np.mean(v)), "per_seed": v,
                             "t_interval": t_interval(v)}
                row[name]["not_resolved_at_three_seeds"] = bool(
                    row[name]["t_interval"]["sd"] > abs(row[name]["mean"]))
            row["dominant_factor"] = (
                "lambda" if abs(row["lambda_main"]["mean"]) - abs(row["depth_main"]["mean"])
                >= DOMINANCE_MARGIN else
                "depth" if abs(row["depth_main"]["mean"]) - abs(row["lambda_main"]["mean"])
                >= DOMINANCE_MARGIN else "neither_by_the_margin")
            decomp[key] = row
    report["decomposition"] = decomp
    report["validity_gates"]["G4"] = {
        "gate": "G4",
        "rule_as_preregistered": ("C32.0.4: d - a == depth_main + lambda_main + interaction "
                                  "to within 1e-12 -- THIS CLAIM IS ARITHMETICALLY FALSE, "
                                  "see preregistration_defects.D1"),
        "rule_as_scored": ("the identities that actually hold for the pre-registered "
                           "quantities: d - a == depth_main + lambda_main, and "
                           "interaction == 0.5*((d-b)-(c-a)), both to within 1e-12"),
        "max_abs_residual": g4_worst,
        "max_abs_residual_of_the_claim_as_written": (
            max(g4_written_worst) if g4_written_worst else None),
        "n_checked": sum(1 for r in decomp.values() if r.get("complete")) * len(seeds),
        "passes": bool(g4_worst < 1e-12)}
    report["preregistration_defects"] = {
        "D1": {
            "where": "C32.0.4, the closure identity",
            "claim": "d - a = depth_main + lambda_main + interaction",
            "status": "FALSE as written",
            "why": ("with the pre-registered 0.5 convention, depth_main + lambda_main is "
                    "already exactly d - a (the two marginal contrasts partition the "
                    "corner difference), so adding the interaction over-counts.  The "
                    "interaction is a separate contrast, not a third additive component of "
                    "d - a.  Substituting a = 0, b = 1, c = 2, d = 5 gives depth = 3, "
                    "lambda = 2, interaction = 1: depth + lambda = 5 = d - a, while "
                    "depth + lambda + interaction = 6."),
            "handling": ("the defect is REPORTED and the pre-registration is NOT amended, "
                         "following C29 section C29.4's handling of its own R4 defect and "
                         "C27's of its gate 4.  The three effect definitions themselves are "
                         "unchanged and are standard; only the closure claim was wrong, and "
                         "G4 is scored on the identities that do hold.  No decision rule "
                         "reads the closure identity, so no rule changes."),
            "measured_residual_of_the_false_claim": (
                max(g4_written_worst) if g4_written_worst else None),
        }
    }

    # ----------------------------------------------------------------- the spread ratios
    sp = OUTPUT_DIR / "c32_spread" / "spread.json"
    if sp.exists():
        report["spread"] = read_json(sp)

    # ------------------------------------------------- the effective-lambda correction
    env: dict = {}
    for prop in ALL_PROPERTIES:
        cv = curves.get(prop)
        if cv is None:
            continue
        for k in ENVELOPE_K:
            pts = []
            for lam in ENVELOPE_LAMBDAS:
                st = cell_stats(envelope_dir(prop, lam, k), cv, seeds)
                if st is None:
                    continue
                pts.append({"lam": lam, "dir": st["dir"],
                            "advantage": st["advantage_vs_oracle_selected"],
                            "advantage_by_seed": st["advantage_by_seed"],
                            "hit_rate_mean": st["hit_rate_mean"],
                            "validity_mean": st["validity_mean"],
                            "dropped_on_validity": bool(
                                st["validity_mean"] < CROSSING_VALIDITY_FLOOR)})
            env[f"{prop}_k{k}"] = {"property": prop, "k": k, "points": pts,
                                   "n_points": len(pts),
                                   "n_dropped_on_validity": sum(
                                       p["dropped_on_validity"] for p in pts)}
    report["envelope"] = env

    def interpolate_log2(points: list[dict], target: float, seed: str | None = None) -> dict:
        """C29's rule: linear in log2(lambda) between the two bracketing measured points.

        Envelope points below the validity floor are dropped first, which widens the
        bracket rather than inventing a value -- C32.0.8.
        """
        pts = sorted((p for p in points if not p["dropped_on_validity"]),
                     key=lambda p: p["lam"])
        if len(pts) < 2:
            return {"ok": False, "reason": "fewer than two usable envelope points"}
        lo, hi = pts[0]["lam"], pts[-1]["lam"]
        if not (lo <= target <= hi):
            return {"ok": False, "reason": "extrapolated_beyond_envelope",
                    "envelope_range": [lo, hi], "target": target}
        xs = [math.log2(p["lam"]) for p in pts]
        ys = [(p["advantage_by_seed"][seed] if seed else p["advantage"]) for p in pts]
        t = math.log2(target)
        for j in range(1, len(xs)):
            if xs[j - 1] <= t <= xs[j]:
                w = 0.0 if xs[j] == xs[j - 1] else (t - xs[j - 1]) / (xs[j] - xs[j - 1])
                return {"ok": True, "advantage": ys[j - 1] + w * (ys[j] - ys[j - 1]),
                        "bracket": [pts[j - 1]["lam"], pts[j]["lam"]],
                        "bracket_advantages": [ys[j - 1], ys[j]],
                        "weight_in_log2": w}
        return {"ok": False, "reason": "no bracket found"}

    eff: dict = {}
    for prop in ALL_PROPERTIES:
        s = report["spread"].get("properties", {}).get(prop)
        if not s:
            continue
        ratio = s["spread_ratio"]
        for k in ENVELOPE_K:
            pts = env.get(f"{prop}_k{k}", {}).get("points", [])
            if not pts:
                continue
            for lam in (1.0, 2.0):
                mid = cells.get(f"{prop}_{'mid_l1' if lam == 1.0 else 'mid_l2'}_k{k}")
                dep = cells.get(f"{prop}_{'deployed_l1' if lam == 1.0 else 'deployed_l2'}_k{k}")
                if mid is None or dep is None:
                    continue
                lam_eff = lam * ratio
                ip = interpolate_log2(pts, lam_eff)
                row = {"property": prop, "k": k, "lam": lam, "spread_ratio": ratio,
                       "lambda_effective": lam_eff,
                       "mid_dir": mid["dir"], "mid_advantage": mid["advantage_vs_oracle_selected"],
                       "deployed_dir_at_nominal_lam": dep["dir"],
                       "deployed_advantage_at_nominal_lam":
                           dep["advantage_vs_oracle_selected"],
                       "depth_raw": (mid["advantage_vs_oracle_selected"]
                                     - dep["advantage_vs_oracle_selected"]),
                       "envelope": ip,
                       "extrapolated_beyond_envelope": (
                           ip.get("reason") == "extrapolated_beyond_envelope")}
                if ip.get("ok"):
                    row["deployed_advantage_at_lambda_effective"] = ip["advantage"]
                    row["depth_corrected"] = (mid["advantage_vs_oracle_selected"]
                                              - ip["advantage"])
                    row["confound_share"] = (
                        (row["depth_raw"] - row["depth_corrected"]) / row["depth_raw"]
                        if row["depth_raw"] != 0 else None)
                    ps = []
                    for sd in seeds:
                        ipp = interpolate_log2(pts, lam_eff, seed=sd)
                        if ipp.get("ok"):
                            ps.append(mid["advantage_by_seed"][sd] - ipp["advantage"])
                    row["depth_corrected_per_seed"] = ps
                    if len(ps) == len(seeds):
                        row["depth_corrected_t_interval"] = t_interval(ps)
                eff[f"{prop}_k{k}_lam{lam:g}"] = row
    report["effective_lambda"] = {
        "definition": ("lambda_eff = lambda * spread(prop, M) / spread(prop, 12); the "
                       "deployed envelope is interpolated linearly in log2(lambda) between "
                       "bracketing measured points -- C29's rule, reused"),
        "caveat": ("the lambda-rescale identity is pointwise in log q; the spread ratio is a "
                   "scalar moment ratio of q, so this is a FIRST-ORDER CONTROL, not an "
                   "identity -- C29's caveat, inherited"),
        "rows": eff,
    }

    # ------------------------------------------------------------------ decision rules
    dr = report["decision_rules"]
    gates_ok = all(report["validity_gates"].get(g, {}).get("passes")
                   for g in ("G1", "G2", "G3", "G4", "G5"))
    n_degen = sum(1 for c in cells.values() if c["dropped_on_validity"])
    report["uninterpretability"] = {
        "gates_pass": {g: bool(report["validity_gates"].get(g, {}).get("passes"))
                       for g in ("G1", "G2", "G3", "G4", "G5")},
        "all_gates_pass": bool(gates_ok),
        "n_cells": len(cells),
        "n_cells_below_validity_floor": n_degen,
        "cells_dropped_on_validity": sorted(set(dropped)),
        "max_degenerate_cells_before_void": MAX_DEGENERATE_CELLS,
        "experiment_degenerate": bool(n_degen > MAX_DEGENERATE_CELLS),
        "uninterpretable": bool((not gates_ok) or n_degen > MAX_DEGENERATE_CELLS),
    }

    def by_prop_primary(fn):
        """Properties satisfying `fn` at >= 1 primary k, among complete decompositions."""
        hits = set()
        for key, r in decomp.items():
            if not r.get("complete") or not r["primary_k"]:
                continue
            if fn(r):
                hits.add(r["property"])
        return sorted(hits)

    def dominance(diff_fn):
        """C32.0.6's D1/D2 rule IN FULL, including the sign-consistency clause.

        A property counts iff the margin is cleared at >= 1 primary k AND the sign of the
        contrast is the same at *both* primary k wherever both are measured.  The second
        clause is easy to drop by accident and is pre-registered, so it is scored.
        """
        counted, detail = [], {}
        for prop in ALL_PROPERTIES:
            rows = [r for r in decomp.values()
                    if r.get("complete") and r["primary_k"] and r["property"] == prop]
            if not rows:
                continue
            diffs = {r["k"]: diff_fn(r) for r in rows}
            clears = [k for k, v in diffs.items() if v >= DOMINANCE_MARGIN]
            sign_ok = all(v > 0 for v in diffs.values())
            detail[prop] = {"diffs_by_k": diffs, "clears_margin_at_k": clears,
                            "sign_holds_at_every_primary_k": sign_ok,
                            "counts": bool(clears and sign_ok)}
            if clears and sign_ok:
                counted.append(prop)
        return sorted(counted), detail

    d1p, d1d = dominance(lambda r: abs(r["lambda_main"]["mean"])
                         - abs(r["depth_main"]["mean"]))
    d2p, d2d = dominance(lambda r: abs(r["depth_main"]["mean"])
                         - abs(r["lambda_main"]["mean"]))
    d3p = by_prop_primary(lambda r: abs(r["interaction"]["mean"]) >= INTERACTION_MARGIN
                          and r["interaction"]["t_interval"]["excludes_zero"])
    dr["D1"] = {"rule": (f"lambda dominates: |lambda_main| - |depth_main| >= "
                         f"{DOMINANCE_MARGIN} on >= 2 of 3 properties at >= 1 primary k"),
                "properties": d1p, "per_property": d1d, "fires": len(d1p) >= 2}
    dr["D2"] = {"rule": (f"depth dominates: |depth_main| - |lambda_main| >= "
                         f"{DOMINANCE_MARGIN} on >= 2 of 3 properties at >= 1 primary k"),
                "properties": d2p, "per_property": d2d, "fires": len(d2p) >= 2}
    dr["D3"] = {"rule": (f"the interaction is material: |interaction| >= "
                         f"{INTERACTION_MARGIN} with a t interval excluding zero, on >= 2 "
                         f"of 3 properties at >= 1 primary k"),
                "properties": d3p, "fires": len(d3p) >= 2}

    d4p, shares = set(), []
    for key, r in eff.items():
        if r.get("extrapolated_beyond_envelope") or "depth_corrected" not in r:
            continue
        if r.get("confound_share") is not None:
            shares.append(r["confound_share"])
        ti = r.get("depth_corrected_t_interval")
        if r["k"] in PRIMARY_K and r["depth_corrected"] > 0 and ti and ti["lo"] > 0:
            d4p.add(r["property"])
    dr["D4"] = {"rule": ("depth survives the effective-lambda correction: depth_corrected > 0 "
                         "with a t interval excluding zero, on >= 2 of 3 properties at >= 1 "
                         "primary k, among non-extrapolated contrasts"),
                "properties": sorted(d4p), "fires": len(d4p) >= 2}
    med = float(np.median(shares)) if shares else None
    dr["D5"] = {"rule": (f"the correction removes most of the depth effect: median confound "
                         f"share >= {CONFOUND_SHARE_THRESHOLD}"),
                "median_confound_share": med, "n_contrasts": len(shares),
                "all_shares": shares,
                "fires": bool(med is not None and med >= CONFOUND_SHARE_THRESHOLD)}

    d6 = [c["dir"] for n, c in cells.items()
          if c["corner"] == "deployed_l2" and c["k"] in PRIMARY_K and c["crosses"]]
    dr["D6"] = {"rule": ("the crossing is reachable without depth: at least one (12, lambda=2) "
                         "cell at k <= 4 crosses the oracle curve"),
                "cells": sorted(d6), "fires": bool(d6)}

    d7 = {}
    for key, r in decomp.items():
        if not r.get("complete"):
            continue
        for name in ("depth_main", "lambda_main", "interaction"):
            if r[name]["not_resolved_at_three_seeds"]:
                d7[f"{key}:{name}"] = {"mean": r[name]["mean"],
                                       "sd": r[name]["t_interval"]["sd"]}
    dr["D7"] = {"rule": ("the honesty rule: any contrast whose between-seed sd exceeds its "
                         "own absolute mean is reported as NOT RESOLVED at three generation "
                         "seeds"),
                "contrasts": d7, "n": len(d7), "fires": bool(d7)}

    if report["uninterpretability"]["uninterpretable"]:
        verdict, wording = "UNINTERPRETABLE", "a gate failed or the run is degenerate"
    elif dr["D1"]["fires"] and not dr["D2"]["fires"]:
        verdict = "LAMBDA-DOMINATED"
        wording = ("the crossing is mostly a steering-strength effect; the probe-point "
                   "selection step can be dropped from the recipe")
    elif dr["D2"]["fires"] and not dr["D1"]["fires"]:
        verdict = "DEPTH-DOMINATED"
        wording = "C23's Rule A gains a second generator"
    elif dr["D3"]["fires"]:
        verdict = "INTERACTION-DOMINATED"
        wording = "neither factor is interpretable alone"
    else:
        verdict = "MIXED"
        wording = "reported as MIXED and not resolved by prose"
    report["verdict"] = {"verdict": verdict, "committed_wording": wording,
                         "rule": ("C32.0.6: LAMBDA-DOMINATED iff D1 and not D2; "
                                  "DEPTH-DOMINATED iff D2 and not D1; INTERACTION-DOMINATED "
                                  "iff D3 and neither; MIXED otherwise; UNINTERPRETABLE "
                                  "overrides")}

    # ------------------------------------------------------------------- predictions
    P = report["predictions"]

    def pred(name, statement, fires, detail):
        P[name] = {"statement": statement, "outcome": "CONFIRMED" if fires else "FALSIFIED",
                   "detail": detail}

    g1 = report["validity_gates"].get("G1")
    if g1:
        pred("P1", "G1 passes with residual exactly 0.0 on both gate cells, molecules included",
             bool(g1["passes"] and g1["max_abs_hit_rate_residual"] == 0.0
                  and g1["max_abs_token_residual"] == 0.0),
             {"max_abs_hit_rate_residual": g1["max_abs_hit_rate_residual"],
              "max_abs_token_residual": g1["max_abs_token_residual"]})
    sprops = report["spread"].get("properties", {})
    if sprops:
        ratios = {p: v["spread_ratio"] for p, v in sprops.items()}
        pred("P2", "the spread ratio at M exceeds 1.0 for all three properties",
             all(v > 1.0 for v in ratios.values()), ratios)
        pred("P3", ("the spread ratio exceeds C29's largest (1.512) for at least one "
                    "property"),
             any(v > 1.512 for v in ratios.values()),
             {"ratios": ratios, "c29_largest": 1.512})
    prim = [r for r in decomp.values() if r.get("complete") and r["k"] == 2]
    if prim:
        pred("P4", "the lambda main effect is positive at k = 2 on all three properties",
             all(r["lambda_main"]["mean"] > 0 for r in prim),
             {r["property"]: r["lambda_main"]["mean"] for r in prim})
        pred("P5", "the depth main effect is positive at k = 2 on all three properties",
             all(r["depth_main"]["mean"] > 0 for r in prim),
             {r["property"]: r["depth_main"]["mean"] for r in prim})
    if decomp:
        pred("P6", "D3 fires: the interaction is material on >= 2 of 3 properties",
             dr["D3"]["fires"], {"properties": dr["D3"]["properties"]})
    if shares:
        pred("P7", "the median confound share is >= 0.50, i.e. D5 fires",
             dr["D5"]["fires"], {"median": med, "n": len(shares)})
        pred("P8", "D4 fires: depth survives the correction on >= 2 of 3 properties",
             dr["D4"]["fires"], {"properties": dr["D4"]["properties"]})
    if cells:
        pred("P9", ("D6 fires: (12, lambda=2) crosses at k <= 4 on at least one property "
                    "-- STATED IN ADVANCE AS EXPECTED TO FAIL"),
             dr["D6"]["fires"], {"cells": dr["D6"]["cells"]})
        arm_a = {n: c for n, c in cells.items() if c["source"] == "c32" or True}
        pred("P11", ("validity stays >= 0.95 in every 2x2 cell (envelope cells at lambda >= 3 "
                     "are not covered)"),
             all(c["validity_mean"] >= 0.95 for c in cells.values()),
             {"min_validity": min((c["validity_mean"] for c in cells.values()), default=None),
              "n_cells": len(arm_a)})
    if prim:
        depths = {r["property"]: abs(r["depth_main"]["mean"]) for r in prim}
        pred("P10", ("qed shows the smallest depth main effect of the three properties "
                     "at k = 2"),
             bool(depths) and min(depths, key=lambda p: depths[p]) == "qed", depths)

    report["prediction_summary"] = {
        "n": len(P),
        "confirmed": sorted((k for k, v in P.items() if v["outcome"] == "CONFIRMED"),
                            key=lambda s: int(s[1:])),
        "falsified": sorted((k for k, v in P.items() if v["outcome"] == "FALSIFIED"),
                            key=lambda s: int(s[1:])),
    }

    write_json(out_dir / "c32_metrics.json", report)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})

    print(f"[C32] gates {report['uninterpretability']['gates_pass']}")
    for key in sorted(decomp):
        r = decomp[key]
        if not r.get("complete"):
            print(f"  {key:28s} INCOMPLETE {r.get('missing')}")
            continue
        print(f"  {key:28s} depth={r['depth_main']['mean']:+.4f} "
              f"lambda={r['lambda_main']['mean']:+.4f} "
              f"inter={r['interaction']['mean']:+.4f} "
              f"({'primary' if r['primary_k'] else 'secondary'}) -> {r['dominant_factor']}")
    for key in sorted(eff):
        r = eff[key]
        print(f"  {key:28s} ratio={r['spread_ratio']:.4f} lam_eff={r['lambda_effective']:.4f} "
              f"raw={r['depth_raw']:+.4f} corrected="
              f"{r.get('depth_corrected', float('nan')):+.4f} "
              f"share={r.get('confound_share', float('nan')):+.4f}")
    for k, v in dr.items():
        print(f"[C32] {k}: fires={v['fires']}")
    print(f"[C32] VERDICT: {verdict}")
    print(f"[C32] predictions {report['prediction_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
