"""C30 summariser -- scores the pre-registration in `outputs/c30_prereg/`.

Reads only.  Generates nothing.  Every decision rule and prediction in C30.0.5 and C30.0.6
is scored here from artefacts, including the ones that fail.

The comparator is C26's **oracle-selected** best-of-N curve, held fixed across head seeds
because it selects with RDKit and does not depend on the head.  That is the whole design:
the baseline is a constant and only guidance moves, so any spread in the advantage is
attributable to the probe's training seed.

The head-selected curve is priced too and reported as secondary, because it *does* move with
the head seed and so cannot isolate the same thing.  No decision rule reads it.

    .venv/bin/python scripts/24_summarise_c30.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import OUTPUT_DIR, read_json, write_json  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_summarise")
_c28 = _load_module(ROOT / "scripts" / "23_summarise_c28.py", "c28_summarise")
_c30 = _load_module(ROOT / "scripts" / "24_c30_crossing_head_seeds.py", "c30_run")

interp = _c26.interp
t_interval = _c26.t_interval
T_CRIT_95 = _c26.T_CRIT_95
load_curves = _c28.load_curves
price = _c28.price
STRANDS = _c30.STRANDS
STRAND_ORDER = _c30.STRAND_ORDER
K_GRID = _c30.K_GRID
HEAD_SEEDS = _c30.HEAD_SEEDS
cell_dir = _c30.cell_dir

SEEDS = ("101", "202", "303")

#: C30.0.1, transcribed from the pre-registration table and NOT recomputed from C28's
#: artefact, so that a change in C28's output cannot silently redefine which cells C30 was
#: written to test.  `check_winning_cells_match_c28` re-derives them and asserts agreement.
C28_WINNING_CELLS = {
    ("A3", 2): 0.2499, ("A3", 4): 0.2093, ("C2", 4): 0.1690, ("A2", 2): 0.1325,
    ("A1", 2): 0.0846, ("A3", 8): 0.0267, ("A2", 4): 0.0218, ("C2", 2): 0.0007,
}
#: The subset C30 actually re-runs -- A3 k=8 is excluded by C30.0.2 and cited instead.
C30_CELLS = tuple(sorted(
    ((s, k) for (s, k) in C28_WINNING_CELLS if k in K_GRID),
    key=lambda sk: -C28_WINNING_CELLS[sk]))


def load_cell(strand: str, k: int, hs: int) -> dict | None:
    f = cell_dir(strand, k, hs) / "k_cell_metrics.json"
    if not f.exists():
        return None
    d = read_json(f)
    agg = d["aggregate"]
    return {
        "dir": cell_dir(strand, k, hs).name,
        "hit_rate_mean": agg["hit_rate"]["mean"],
        "hit_rate_values": agg["hit_rate"]["values"],
        "validity_mean": agg["validity"]["mean"],
        "uniqueness_mean": agg["uniqueness"]["mean"],
        "tokens_per_molecule_actual": agg["compute_total"]["tokens_per_molecule_actual"],
        "per_seed": {
            s: {"hit_rate": d["seeds_detail"][s]["hit_rate"],
                "tokens_per_molecule_actual":
                    d["seeds_detail"][s]["compute"]["tokens_per_molecule_actual"]}
            for s in SEEDS if s in d["seeds_detail"]},
        "head_checkpoint": d["head_checkpoint"],
        "cost_identity_max_residual": d["cost_identity_max_residual"],
    }


def check_winning_cells_match_c28() -> dict:
    """The transcribed table in C28_WINNING_CELLS must equal what C28's artefact says.

    C30 was designed around eight specific cells.  If C28's summary is ever regenerated and
    a different set of cells comes out above the curve, C30 is testing the wrong thing and
    must say so rather than quietly scoring a stale list.
    """
    f = OUTPUT_DIR / "c28_summary" / "c28_metrics.json"
    if not f.exists():
        return {"checked": False, "reason": "c28_metrics.json absent"}
    c28 = read_json(f)
    derived = {}
    for name, strand in c28["strands"].items():
        for k, cell in (strand.get("cells") or {}).items():
            adv = cell.get("advantage_vs_oracle_selected")
            if adv is not None and adv > 0:
                derived[(name, int(k))] = adv
    agree = sorted(derived) == sorted(C28_WINNING_CELLS)
    return {
        "checked": True,
        "n_derived": len(derived),
        "n_transcribed": len(C28_WINNING_CELLS),
        "same_cells": agree,
        "max_abs_margin_discrepancy": max(
            (abs(derived[c] - C28_WINNING_CELLS[c]) for c in derived if c in C28_WINNING_CELLS),
            default=None),
        "derived": {f"{s}_k{k}": v for (s, k), v in sorted(derived.items())},
    }


def score() -> dict:
    out: dict = {
        "experiment": "C30",
        "prereg": "outputs/c30_prereg/C30.0_preregistration.md",
        "prereg_lock": read_json(OUTPUT_DIR / "c30_prereg" / "prereg_lock.json"),
        "question": ("does C28's crossing -- guided decoding above the oracle-selected "
                     "best-of-N frontier at small budgets -- survive the probe-training seed?"),
        "comparator": ("C26's oracle-selected best-of-N curve, held FIXED across head seeds "
                       "because it selects with RDKit and does not depend on the head"),
        "head_seeds": list(HEAD_SEEDS),
        "generation_seeds": [int(s) for s in SEEDS],
        "t_crit_7df": T_CRIT_95[7],
        "c28_winning_cells_transcribed": {f"{s}_k{k}": v
                                          for (s, k), v in C28_WINNING_CELLS.items()},
        "c28_cell_agreement": check_winning_cells_match_c28(),
        "validity_gates": {},
        "cells": {},
    }

    for gate, fname in (("G1", "g1_checkpoint_identity.json"),
                        ("G2", "g2_reproduces_c28.json"),
                        ("G3", "g3_cost_identity.json")):
        f = OUTPUT_DIR / "c30_gates" / fname
        out["validity_gates"][gate] = read_json(f) if f.exists() else {"present": False}

    curves_cache: dict[str, dict] = {}
    for strand, k in C30_CELLS:
        prop = STRANDS[strand]["property"]
        if prop not in curves_cache:
            c = load_curves(prop)
            if c is None:
                raise SystemExit(f"[C30] stop: no C26 curve for {prop}")
            curves_cache[prop] = c
        curves = curves_cache[prop]

        rows, missing = {}, []
        for hs in HEAD_SEEDS:
            cell = load_cell(strand, k, hs)
            if cell is None:
                missing.append(hs)
                continue
            priced = price(curves, cell["hit_rate_mean"],
                           cell["tokens_per_molecule_actual"], cell["per_seed"])
            rows[str(hs)] = {**cell, **priced}

        adv = [rows[str(h)]["advantage_vs_oracle_selected"]
               for h in HEAD_SEEDS if str(h) in rows]
        hits = [rows[str(h)]["hit_rate_mean"] for h in HEAD_SEEDS if str(h) in rows]
        toks = [rows[str(h)]["tokens_per_molecule_actual"] for h in HEAD_SEEDS if str(h) in rows]
        adv_head = [rows[str(h)].get("advantage_vs_head_selected")
                    for h in HEAD_SEEDS if str(h) in rows]
        adv_head = [a for a in adv_head if a is not None]

        c28_margin = C28_WINNING_CELLS[(strand, k)]
        entry: dict = {
            "strand": strand,
            "k": k,
            "property": prop,
            "probe_point": STRANDS[strand]["layer"],
            "lam": STRANDS[strand]["lam"],
            "is_deployed_configuration": STRANDS[strand]["layer"] == 12,
            "c28_single_seed_margin": c28_margin,
            "n_head_seeds": len(adv),
            "head_seeds_missing": missing,
            "per_head_seed": rows,
            "advantage_by_head_seed": {str(h): rows[str(h)]["advantage_vs_oracle_selected"]
                                       for h in HEAD_SEEDS if str(h) in rows},
        }
        if adv:
            ti = t_interval(adv)
            entry["advantage_mean"] = float(np.mean(adv))
            entry["advantage_sd"] = float(np.std(adv, ddof=1)) if len(adv) > 1 else None
            entry["advantage_t_interval"] = ti
            entry["n_positive"] = int(sum(a > 0 for a in adv))
            entry["all_share_sign"] = bool(len({a > 0 for a in adv}) == 1)
            entry["hit_rate_head_seed_sd"] = (
                float(np.std(hits, ddof=1)) if len(hits) > 1 else None)
            entry["tokens_head_seed_sd"] = (
                float(np.std(toks, ddof=1)) if len(toks) > 1 else None)
            entry["validity_min"] = min(rows[str(h)]["validity_mean"]
                                        for h in HEAD_SEEDS if str(h) in rows)
            entry["extrapolated_any"] = any(
                rows[str(h)].get("extrapolated_beyond_grid_oracle_selected")
                for h in HEAD_SEEDS if str(h) in rows)
            # D6: is the single-seed margin smaller than the spread it was drawn from?
            sd = entry["advantage_sd"]
            entry["d6_not_resolvable_at_one_seed"] = bool(sd is not None and sd > c28_margin)
            entry["c28_margin_over_head_seed_sd"] = (
                c28_margin / sd if sd else None)
            entry["c28_margin_minus_head_seed_mean"] = c28_margin - entry["advantage_mean"]
            if adv_head:
                entry["secondary_advantage_vs_head_selected_mean"] = float(np.mean(adv_head))
                entry["secondary_advantage_vs_head_selected_t"] = t_interval(adv_head)
        out["cells"][f"{strand}_k{k}"] = entry

    scored = [c for c in out["cells"].values() if c.get("n_head_seeds")]
    n_cells = len(C28_WINNING_CELLS)  # 8, per the pre-registration -- includes A3 k=8
    cited = {
        "cell": "A3_k8",
        "why_not_rerun": ("C30.0.2 -- this is C26's D2 arm, already replicated across head "
                          "seeds by C26 D2b (n=3) and C29 R6 (n=8)"),
        "c29_r6_mean": None,
        "c29_r6_ci": None,
    }
    # A1 at k = 4 was RUN -- C30.0.2's "4 strands x 2 k" is 8 combinations and this is the
    # eighth -- but it is not one of C28's winning cells, so it is not in the scored set.
    # Reported here rather than left to vanish from a directory listing: a cell that was
    # generated and then not mentioned is indistinguishable from a cell that was dropped.
    extra = {}
    prop = STRANDS["A1"]["property"]
    curves = curves_cache.get(prop) or load_curves(prop)
    rows = {}
    for hs in HEAD_SEEDS:
        c = load_cell("A1", 4, hs)
        if c is None:
            continue
        rows[str(hs)] = {**c, **price(curves, c["hit_rate_mean"],
                                      c["tokens_per_molecule_actual"], c["per_seed"])}
    if rows:
        adv = [r["advantage_vs_oracle_selected"] for r in rows.values()]
        extra = {
            "cell": "A1_k4",
            "why_not_scored": ("run as part of C30.0.2's 4 strands x 2 k, but not one of "
                               "C28's eight winning cells, so no pre-registered rule "
                               "applies to it"),
            "c28_advantage": None,
            "n_head_seeds": len(adv),
            "advantage_mean": float(np.mean(adv)),
            "advantage_t_interval": t_interval(adv),
            "advantage_by_head_seed": {h: r["advantage_vs_oracle_selected"]
                                       for h, r in rows.items()},
            "hit_rate_by_head_seed": {h: r["hit_rate_mean"] for h, r in rows.items()},
            "validity_min": min(r["validity_mean"] for r in rows.values()),
            "n_positive": int(sum(a > 0 for a in adv)),
        }
    out["run_but_not_scored"] = extra

    f29 = OUTPUT_DIR / "c29_summary" / "c29_metrics.json"
    if f29.exists():
        r6 = read_json(f29)["decision_rules"]["R6"]
        cited["c29_r6_mean"] = r6["mean"]
        cited["c29_r6_ci"] = r6["ci"]
        cited["c29_r6_positive_under_c26_corrected_curve"] = r6["also_c26_corrected_curve"]["fires"]
    out["cited_not_rerun"] = cited

    n_positive_mean = sum(1 for c in scored if c["advantage_mean"] > 0)
    n_excl_zero = sum(1 for c in scored
                      if c["advantage_mean"] > 0 and c["advantage_t_interval"]["lo"] > 0)
    points = [(c, h) for c in scored for h in c["advantage_by_head_seed"].values()]
    n_points_pos = sum(1 for _, h in points if h > 0)

    a1k2 = out["cells"].get("A1_k2", {})
    d3 = bool(a1k2.get("advantage_mean", 0) > 0
              and a1k2.get("advantage_t_interval", {}).get("lo", -1) > 0)

    rules = {
        "D1_crossing_survives": {
            "rule": "head-seed mean advantage over the oracle curve > 0 on >= 5 of 8 cells",
            "n_cells_preregistered": n_cells,
            "n_cells_rerun": len(scored),
            "n_positive_mean": n_positive_mean,
            "threshold": 5,
            "note": ("A3_k8 was not re-run (C30.0.2) and is not counted as positive here; "
                     "the rule is therefore scored conservatively against C30's own claim"),
            "fires": n_positive_mean >= 5,
        },
        "D2_crossing_confirmed": {
            "rule": ">= 3 of 8 cells additionally have a t interval on 7 df strictly above 0",
            "n_excluding_zero": n_excl_zero,
            "threshold": 3,
            "fires": n_excl_zero >= 3,
        },
        "D3_deployed_cell_survives": {
            "rule": "A1 at k=2 -- the only deployed configuration -- has mean > 0 with a t "
                    "interval strictly above 0",
            "mean": a1k2.get("advantage_mean"),
            "ci": ([a1k2["advantage_t_interval"]["lo"], a1k2["advantage_t_interval"]["hi"]]
                   if a1k2.get("advantage_t_interval") else None),
            "c28_single_seed_margin": C28_WINNING_CELLS[("A1", 2)],
            "fires": d3,
        },
        "D4_sign_stability": {
            "rule": ">= 75% of the (cell, head seed) points are positive",
            "n_points": len(points),
            "n_positive": n_points_pos,
            "fraction": (n_points_pos / len(points)) if points else None,
            "threshold": 0.75,
            "fires": bool(points) and (n_points_pos / len(points)) >= 0.75,
        },
        "D5_crossing_refuted": {
            "rule": "fewer than 3 of 8 cells keep a positive head-seed mean",
            "n_positive_mean": n_positive_mean,
            "fires": n_positive_mean < 3,
        },
        "D6_not_resolvable_at_one_seed": {
            "rule": "for each cell, is the head-seed sd of the advantage larger than C28's "
                    "own single-seed margin for that cell?",
            "per_cell": {name: {"c28_margin": c["c28_single_seed_margin"],
                                "head_seed_sd": c.get("advantage_sd"),
                                "not_resolvable": c.get("d6_not_resolvable_at_one_seed")}
                         for name, c in out["cells"].items() if c.get("n_head_seeds")},
            "n_not_resolvable": sum(1 for c in scored if c["d6_not_resolvable_at_one_seed"]),
            "fires": any(c["d6_not_resolvable_at_one_seed"] for c in scored),
        },
    }
    out["decision_rules"] = rules

    # ---------------------------------------------------------------- the validity screen
    # C30.0.8 voids the experiment if "any cell's validity falls below 0.90".  Under C30's
    # own naming a cell is one (strand, k, head seed) directory, so this is scored per head
    # seed, which is the strict reading and the one implemented.  The failing points are
    # enumerated rather than summarised, because a screen that fires on 1 of 56 points and a
    # screen that fires on 30 of 56 are different experiments.
    failures = []
    for name, c in out["cells"].items():
        if not c.get("n_head_seeds"):
            continue
        for hs, r in c["per_head_seed"].items():
            if r["validity_mean"] < 0.90:
                failures.append({
                    "cell": name, "head_seed": int(hs),
                    "validity": r["validity_mean"],
                    "hit_rate": r["hit_rate_mean"],
                    "tokens_per_molecule_actual": r["tokens_per_molecule_actual"],
                    "advantage": r["advantage_vs_oracle_selected"],
                })
    n_points = sum(len(c["per_head_seed"]) for c in scored)
    out["validity_screen"] = {
        "rule": "C30.0.8 -- any cell with validity < 0.90 makes C30 UNINTERPRETABLE and "
                "leaves the decision rules unscored",
        "threshold": 0.90,
        "n_points_checked": n_points,
        "n_points_failing": len(failures),
        "failures": failures,
        "cells_affected": sorted({f["cell"] for f in failures}),
        "passes": not failures,
    }

    gates_ok = all(out["validity_gates"].get(g, {}).get("passes") for g in ("G1", "G2", "G3"))
    validity_ok = out["validity_screen"]["passes"]
    seeds_ok = all(c["n_head_seeds"] == len(HEAD_SEEDS) for c in scored) if scored else False
    if not (gates_ok and validity_ok and seeds_ok):
        verdict = "UNINTERPRETABLE"
    elif rules["D5_crossing_refuted"]["fires"]:
        verdict = "REFUTED"
    elif rules["D1_crossing_survives"]["fires"] and rules["D2_crossing_confirmed"]["fires"]:
        verdict = "CONFIRMED"
    elif rules["D1_crossing_survives"]["fires"]:
        verdict = "SURVIVES, UNDERPOWERED"
    else:
        verdict = "AMBIGUOUS"
    out["verdict"] = {
        "verdict": verdict,
        "gates_pass": gates_ok,
        "all_validity_at_least_0.90": validity_ok,
        "all_cells_have_eight_head_seeds": seeds_ok,
        "committed_wording": {
            "SURVIVES, UNDERPOWERED": ("the crossing's sign replicates across probe seeds "
                                       "and its size is not resolved at eight"),
        }.get(verdict),
    }

    # ------------------------------------------------- POST HOC, NOT PRE-REGISTERED: S1
    # The pre-registered verdict above stands as scored and is not replaced by this block.
    #
    # C30.0.8's screen is written per cell, and it fired on ONE (cell, head seed) point out
    # of 56.  Voiding 56 cells on one is the rule as written, and it is also the least
    # informative possible response to the data.  So the conservative repair is computed and
    # labelled: drop the entire offending CELL -- all eight of its head seeds, not just the
    # failing one, because dropping only the failing seed would be selecting on the outcome
    # -- and re-score D1 through D4 on what remains.
    #
    # Dropping the whole cell is the choice that runs AGAINST C30's own headline: C2_k4's
    # head-seed mean is +0.1052 with an interval excluding zero, so removing it removes a
    # supporting cell.  If the verdict survives that, it survives the screen.
    dropped = set(out["validity_screen"]["cells_affected"])
    kept = [c for c in scored if f"{c['strand']}_k{c['k']}" not in dropped]
    s1_pos = sum(1 for c in kept if c["advantage_mean"] > 0)
    s1_excl = sum(1 for c in kept
                  if c["advantage_mean"] > 0 and c["advantage_t_interval"]["lo"] > 0)
    s1_points = [h for c in kept for h in c["advantage_by_head_seed"].values()]
    s1_d1 = s1_pos >= 5
    s1_d2 = s1_excl >= 3
    s1_verdict = ("REFUTED" if s1_pos < 3
                  else "CONFIRMED" if (s1_d1 and s1_d2)
                  else "SURVIVES, UNDERPOWERED" if s1_d1
                  else "AMBIGUOUS")
    out["sensitivity_S1_drop_cells_failing_the_validity_screen"] = {
        "status": "POST HOC, NOT PRE-REGISTERED -- reported beside the pre-registered "
                  "verdict, not in place of it",
        "cells_dropped": sorted(dropped),
        "why_whole_cell_not_just_the_failing_seed": (
            "dropping only the failing head seed would be selecting on the outcome; the "
            "whole cell goes, including its seven passing seeds"),
        "runs_against_c30s_own_headline": sorted(
            {n for n in dropped
             if out["cells"].get(n, {}).get("advantage_t_interval", {}).get("lo", -1) > 0}),
        "n_cells_remaining": len(kept),
        "n_positive_mean": s1_pos,
        "n_excluding_zero": s1_excl,
        "D1_threshold_unchanged": 5,
        "D2_threshold_unchanged": 3,
        "D1_fires": s1_d1,
        "D2_fires": s1_d2,
        "D3_fires": d3 and "A1_k2" not in dropped,
        "D4_fraction_positive": (sum(1 for h in s1_points if h > 0) / len(s1_points)
                                 if s1_points else None),
        "verdict": s1_verdict,
        "note": ("D1 and D2 are scored against their ORIGINAL thresholds of 5 and 3 out of "
                 "8 pre-registered cells, not rescaled to the smaller denominator -- "
                 "rescaling would make the rule easier to pass after dropping a cell"),
    }

    def cell(name):
        return out["cells"].get(name, {})

    out["predictions"] = {
        "P1_at_least_six_of_eight_positive": {
            "predicted": ">= 6 of 8 cells keep a positive head-seed mean",
            "measured": n_positive_mean,
            "n_rerun": len(scored),
            "holds": n_positive_mean >= 6,
        },
        "P2_largest_margin_survives": {
            "predicted": "A3 k=2 keeps a t interval excluding zero",
            "mean": cell("A3_k2").get("advantage_mean"),
            "excludes_zero": cell("A3_k2").get("advantage_t_interval", {}).get("excludes_zero"),
            "holds": bool(cell("A3_k2").get("advantage_t_interval", {}).get("lo", -1) > 0),
        },
        "P3_deployed_cell_survives": {
            "predicted": "A1 k=2 keeps a t interval excluding zero",
            "mean": cell("A1_k2").get("advantage_mean"),
            "holds": d3,
        },
        "P4_smallest_margin_does_not_survive": {
            "predicted": "C2 k=2 does NOT keep a t interval excluding zero, and fires D6",
            "mean": cell("C2_k2").get("advantage_mean"),
            "excludes_zero": cell("C2_k2").get("advantage_t_interval", {}).get("excludes_zero"),
            "d6": cell("C2_k2").get("d6_not_resolvable_at_one_seed"),
            "holds": bool(not cell("C2_k2").get("advantage_t_interval", {}).get("lo", -1) > 0
                          and cell("C2_k2").get("d6_not_resolvable_at_one_seed")),
        },
        "P5_at_least_one_sign_change": {
            "predicted": "at least one cell's mean advantage changes sign against C28",
            "n_negative_means": sum(1 for c in scored if c["advantage_mean"] < 0),
            "holds": any(c["advantage_mean"] < 0 for c in scored),
        },
        "P6_advantage_sd_exceeds_hit_rate_sd_somewhere": {
            "predicted": "the head-seed sd of the ADVANTAGE exceeds that of the hit rate "
                         "alone for at least one cell, because the budget moves too",
            "per_cell": {n: {"advantage_sd": c.get("advantage_sd"),
                             "hit_rate_sd": c.get("hit_rate_head_seed_sd")}
                         for n, c in out["cells"].items() if c.get("n_head_seeds")},
            "holds": any((c.get("advantage_sd") or 0) > (c.get("hit_rate_head_seed_sd") or 0)
                         for c in scored),
        },
        "P7_sds_inside_c29_band": {
            "predicted": "hit-rate head-seed sd inside C29's measured 0.0142-0.0366 band on "
                         ">= 3 of 4 strands",
            "band": [0.0142, 0.0366],
            "per_strand": {},
            "holds": None,
        },
    }
    by_strand: dict[str, list[float]] = {}
    for c in scored:
        by_strand.setdefault(c["strand"], []).append(c["hit_rate_head_seed_sd"])
    inside = 0
    for s, sds in by_strand.items():
        m = float(np.mean([x for x in sds if x is not None]))
        ok = 0.0142 <= m <= 0.0366
        inside += int(ok)
        out["predictions"]["P7_sds_inside_c29_band"]["per_strand"][s] = {
            "mean_hit_rate_head_seed_sd": m, "inside_band": ok}
    out["predictions"]["P7_sds_inside_c29_band"]["n_inside"] = inside
    out["predictions"]["P7_sds_inside_c29_band"]["holds"] = inside >= 3

    out["predictions_summary"] = {
        "n_total": 7,
        "n_holding": sum(1 for p in out["predictions"].values() if p.get("holds") is True),
        "falsified": [k for k, p in out["predictions"].items() if p.get("holds") is False],
    }
    return out


def main() -> int:
    out = score()
    d = OUTPUT_DIR / "c30_summary"
    write_json(d / "c30_metrics.json", out)

    print(f"\n[C30] PRE-REGISTERED verdict: {out['verdict']['verdict']}")
    vs = out["validity_screen"]
    print(f"      validity screen: {vs['n_points_failing']}/{vs['n_points_checked']} points "
          f"below {vs['threshold']} -> {vs['failures']}")
    s1 = out["sensitivity_S1_drop_cells_failing_the_validity_screen"]
    print(f"      POST-HOC S1 (drop {s1['cells_dropped']} entirely): {s1['verdict']} "
          f"({s1['n_positive_mean']} positive, {s1['n_excluding_zero']} excluding zero)")
    print(f"      gates G1/G2/G3 pass: {out['verdict']['gates_pass']}")
    print(f"      cells scored: {len([c for c in out['cells'].values() if c.get('n_head_seeds')])}"
          f" of {len(C30_CELLS)} re-run ({len(C28_WINNING_CELLS)} pre-registered)")
    print("\n      cell        prop            pp   lam  k   C28      mean(8)     sd      "
          "t interval (7 df)        +/8")
    for name, c in out["cells"].items():
        if not c.get("n_head_seeds"):
            print(f"      {name:<11} -- not run --")
            continue
        ti = c["advantage_t_interval"]
        star = "*" if ti["lo"] > 0 else " "
        print(f"      {name:<11} {c['property']:<15} {c['probe_point']:>2} "
              f"{c['lam']:>4.1f} {c['k']:>2} {c['c28_single_seed_margin']:+.4f} "
              f"{c['advantage_mean']:+.4f} {c['advantage_sd']:.4f} "
              f"[{ti['lo']:+.4f}, {ti['hi']:+.4f}]{star} {c['n_positive']}/8")
    print("\n      decision rules:")
    for k, v in out["decision_rules"].items():
        print(f"        {k:<34} fires={v['fires']}")
    print("\n      predictions:")
    for k, v in out["predictions"].items():
        print(f"        {k:<44} holds={v.get('holds')}")
    print(f"\n[C30] -> {d / 'c30_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
