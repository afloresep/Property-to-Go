"""C31 -- price the second generator's guidance against its own best-of-N frontier.

Reads only artefacts that already exist: the feasibility run, the dataset, the probe depth
curves, the gates, the oracle-selected best-of-N curves and the k cells.  **Generates
nothing** -- no molecule is sampled and no head is trained.

The frontier machinery is `scripts/21_summarise_c26.py::interp` and `::t_interval`,
**imported unmodified**, so a C31 advantage is computed by exactly the code that produced
C26's, C27's, C28's and C30's.  `price` below is the C31 analogue of
`scripts/23_summarise_c28.py::price`, restricted to the oracle-selected arm because that is
the only comparator C31 pre-registered.

Scores `outputs/c31_prereg/C31.0_preregistration.md`: gates G0-G6, decision rules D1-D7,
and predictions P1-P11, including where they fail.

**No bootstrap.**  C31.0.5: at n = 3 the percentile bootstrap of a mean is identically
[min, max], so it conveys only a three-way sign test at null probability 0.25.  Per-seed
values and a seed-level t interval on 2 df (t_0.975,2 = 4.302653) are reported instead.

    .venv/bin/python scripts/25_summarise_c31.py
"""

from __future__ import annotations

import argparse
import importlib.util
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
_c31 = _load_module(ROOT / "scripts" / "25_c31_second_generator.py", "c31_second_generator")
interp = _c26.interp
t_interval = _c26.t_interval
cell_dir = _c31.cell_dir
load_arms = _c31.load_arms
ALL_C31_PROPERTIES = _c31.ALL_C31_PROPERTIES
REQUIRED_PROPERTIES = _c31.REQUIRED_PROPERTIES

#: C31.0.6.  A cell must clear this validity to be allowed to count as a crossing.
CROSSING_VALIDITY_FLOOR = 0.80
#: C31.0.7.  Below this a cell is flagged but still scored.
VALIDITY_FLAG_LEVEL = 0.90
#: C31.0.6 D3, C28's threshold reused rather than re-chosen.
D3_THRESHOLD = 0.02
#: C31.0.7 population-level degeneracy condition.
DEGENERATE_CELL_FRACTION = 0.25


def load_curve(prop: str) -> dict | None:
    """C31's own oracle-selected best-of-N curve, on one token axis."""
    f = OUTPUT_DIR / f"c31_bestofn_{prop}" / "n_sweep_metrics.json"
    if not f.exists():
        return None
    d = read_json(f)
    grid = d["grid"]
    seeds = [str(s) for s in d["seeds"]]
    return {
        "grid": grid,
        "seeds": seeds,
        "tokens_per_molecule_actual": [
            d["curve"][str(n)]["tokens_per_molecule_actual"] for n in grid],
        "oracle_selected": [d["curve"][str(n)]["hit_rate_mean"] for n in grid],
        "oracle_selected_per_seed": {
            s: [d["per_seed"][s]["rows"][str(n)]["hit_rate"] for n in grid] for s in seeds},
        "tokens_per_molecule_actual_per_seed": {
            s: [d["per_seed"][s]["rows"][str(n)]["compute"]["tokens_per_molecule_actual"]
                for n in grid] for s in seeds},
        "validity_mean": [d["curve"][str(n)]["validity_mean"] for n in grid],
        "uniqueness_mean": [d["curve"][str(n)]["uniqueness_mean"] for n in grid],
        "source": f"outputs/c31_bestofn_{prop}/n_sweep_metrics.json",
    }


def price(curves: dict, hit_mean: float, budget: float, per_seed: dict, seeds) -> dict:
    """Advantage of one guidance point against the oracle curve at its own budget.

    The shape of `scripts/23_summarise_c28.py::price`, with `interp` imported unmodified.
    The per-seed advantage interpolates each generation seed's own curve at that seed's
    own budget, so the t interval is over genuinely paired differences.
    """
    toks = curves["tokens_per_molecule_actual"]
    grid = curves["grid"]
    h, i_lo, i_hi, extrap = interp(toks, curves["oracle_selected"], budget)
    row = {
        "oracle_selected_interpolated_hit_rate": h,
        "advantage_vs_oracle_selected": hit_mean - h,
        "bracketing_n_oracle_selected": [grid[i_lo], grid[i_hi]],
        "extrapolated_beyond_grid_oracle_selected": extrap,
    }
    adv = []
    for s in seeds:
        ps = per_seed.get(str(s))
        if ps is None:
            continue
        sb, *_ = interp(curves["tokens_per_molecule_actual_per_seed"][str(s)],
                        curves["oracle_selected_per_seed"][str(s)],
                        ps["tokens_per_molecule_actual"])
        adv.append(ps["hit_rate"] - sb)
    row["advantage_vs_oracle_selected_per_seed"] = adv
    if len(adv) == len(seeds):
        row["advantage_vs_oracle_selected_seed_t_interval"] = t_interval(adv)
    return row


def collect_cells(cfg: dict) -> dict:
    cells: dict[str, dict] = {}
    for prop in ALL_C31_PROPERTIES:
        depth_f = OUTPUT_DIR / "c31_heads" / f"depth_{prop}.json"
        if not depth_f.exists():
            continue
        arms = load_arms(cfg, prop)
        for arm, spec in arms.items():
            L, lam = int(spec["probe_point"]), float(spec["lam"])
            for k in [int(x) for x in cfg["k_grid"]]:
                f = cell_dir(prop, arm, L, lam, k) / "k_cell_metrics.json"
                if not f.exists():
                    continue
                d = read_json(f)
                agg = d["aggregate"]
                seeds = [str(s) for s in d["seeds"]]
                cells[f"{prop}_{arm}_k{k}"] = {
                    "property": prop, "arm": arm, "arm_why": spec["why"],
                    "probe_point": L, "lam": lam, "k": k,
                    "dir": cell_dir(prop, arm, L, lam, k).name,
                    "seeds": seeds,
                    "hit_rate_mean": agg["hit_rate"]["mean"],
                    "hit_rate_values": agg["hit_rate"]["values"],
                    "hit_rate_sd": float(np.std(agg["hit_rate"]["values"], ddof=1)),
                    "tokens_per_molecule_actual":
                        agg["compute_total"]["tokens_per_molecule_actual"],
                    "tokens_per_molecule_full_recompute":
                        agg["compute_total"]["tokens_per_molecule_full_recompute"],
                    "validity_mean": agg["validity"]["mean"],
                    "validity_values": agg["validity"]["values"],
                    "uniqueness_mean": agg["uniqueness"]["mean"],
                    "content_length_mean": agg["content_length_mean"]["mean"],
                    "abs_target_error_mean": agg["abs_target_error_mean"]["mean"],
                    "cost_identity_max_residual": d["cost_identity_max_residual"],
                    "per_seed": {
                        s: {"hit_rate": d["seeds_detail"][s]["hit_rate"],
                            "tokens_per_molecule_actual":
                                d["seeds_detail"][s]["compute"]["tokens_per_molecule_actual"],
                            "validity": d["seeds_detail"][s]["validity"],
                            "uniqueness": d["seeds_detail"][s]["uniqueness"],
                            "cost_identity_tokens_mod_k_plus_1":
                                d["seeds_detail"][s]["cost_identity_tokens_mod_k_plus_1"]}
                        for s in seeds},
                }
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c31_summary")
    args = ap.parse_args()

    cfg = load_config("c31_second_generator")
    out_dir = OUTPUT_DIR / args.out
    seeds = [str(s) for s in cfg["generation_seeds"]]

    report: dict = {
        "experiment": "C31",
        "question": ("does the crossing -- guidance above the oracle-selected best-of-N "
                     "frontier at its own budget -- replicate on a second, independent "
                     "molecular generator?"),
        "prereg": "outputs/c31_prereg/C31.0_preregistration.md",
        "prereg_lock": read_json(OUTPUT_DIR / "c31_prereg" / "prereg_lock.json"),
        "generator": {
            "repo": cfg["model_repo"], "revision": cfg["model_revision"],
            "architecture": "GPT-2, full softmax attention",
            "comparison_generator": ("ibm-research/GP-MoLFormer-Uniq, linear attention, "
                                     "46.8M params -- every C23-C30 number"),
            "serialization": "SMILES (not SAFE, not SELFIES)",
        },
        "accounting": "processed generator tokens (actual); full_recompute also reported",
        "seeds": seeds,
        "uncertainty": ("per-seed values and a seed-level t interval on 2 df "
                        "(t_0.975,2 = 4.302653); no bootstrap anywhere in C31, see C31.0.5"),
        "validity_gates": {},
        "depth_curves": {},
        "curves": {},
        "cells": {},
        "decision_rules": {},
        "predictions": {},
    }

    # ------------------------------------------------------------------- gates
    feas_f = OUTPUT_DIR / "c31_feasibility" / "feasibility.json"
    feas = read_json(feas_f) if feas_f.exists() else None
    if feas:
        report["feasibility"] = {
            "n_molecules": feas["n_molecules"],
            "validity": feas["validity"], "uniqueness": feas["uniqueness"],
            "content_length_mean": feas["content_length_mean"],
            "content_length_std": feas["content_length_std"],
            "content_length_max": feas["content_length_max"],
            "n_at_max_length": feas["n_at_max_length"],
            "generator_fingerprint": feas["generator"]["fingerprint"],
            "n_probe_points": feas["generator"]["n_probe_points"],
        }
        report["validity_gates"]["G0"] = feas["validity_gates"]["G0"]
        report["validity_gates"]["G1"] = feas["validity_gates"]["G1"]

    ds_f = OUTPUT_DIR / "c31_zinc50k" / "dataset_metrics.json"
    if ds_f.exists():
        ds = read_json(ds_f)
        report["dataset"] = {
            k: ds[k] for k in ("n_trajectories_requested", "n_trajectories_kept",
                               "n_invalid", "n_too_short", "n_property_unavailable",
                               "validity", "uniqueness", "n_prefix_rows", "probe_points",
                               "split_counts", "group_counts", "content_length",
                               "target_intervals", "windows", "base_property_summary")}
        report["validity_gates"]["G5"] = ds["validity_gates"]["G5"]
        report["validity_gates"]["G6"] = ds["validity_gates"]["G6"]

    for name, f in (("G2", "g2_decision_equality.json"),
                    ("G3", "g3_backend_equivalence.json")):
        p = OUTPUT_DIR / "c31_gates" / f
        if p.exists():
            report["validity_gates"][name] = read_json(p)

    # ------------------------------------------------------------- depth curves
    for prop in ALL_C31_PROPERTIES:
        f = OUTPUT_DIR / "c31_heads" / f"depth_{prop}.json"
        if not f.exists():
            continue
        d = read_json(f)
        report["depth_curves"][prop] = {
            "target_interval": d["target_interval"],
            "n_bins": d["n_bins"],
            "head_seeds": d["head_seeds"],
            "rows": d["rows"],
            "mid_probe_point": d["mid_probe_point"],
            "test_depth_peak": d["test_depth_peak"],
            "trivial_test_target_auroc_mean": d["trivial"]["test_target_auroc_mean"],
            "frozen_state_beats_trivial": d["frozen_state_beats_trivial"],
            "by_probe_point": {L: {
                "probe_point": r["probe_point"],
                "val_target_auroc_mean": r["val_target_auroc_mean"],
                "val_target_auroc_values": r["val_target_auroc_values"],
                "test_target_auroc_mean": r["test_target_auroc_mean"],
                "test_target_auroc_values": r["test_target_auroc_values"],
                "test_target_auroc_sd": r["test_target_auroc_sd"],
                "test_nll_mean": r["test_nll_mean"],
                "test_brier_mean": r["test_brier_mean"],
                "test_ece_mean": r["test_ece_mean"],
            } for L, r in d["by_probe_point"].items()},
        }

    # ------------------------------------- POST HOC, NOT PRE-REGISTERED: length control
    lc = OUTPUT_DIR / "c31_length_control" / "length_control.json"
    if lc.exists():
        report["length_control_post_hoc"] = read_json(lc)

    # ------------------------------------------------------------------ curves
    cells = collect_cells(cfg)
    props_run = sorted({c["property"] for c in cells.values()})
    curves = {p: load_curve(p) for p in props_run}
    for p, c in curves.items():
        if c:
            report["curves"][p] = c

    # ---------------------------------------------------------- price the cells
    for name, c in cells.items():
        cv = curves.get(c["property"])
        if cv is None:
            c["priced"] = False
        else:
            c.update(price(cv, c["hit_rate_mean"], c["tokens_per_molecule_actual"],
                           c["per_seed"], seeds))
            c["priced"] = True
        ti = c.get("advantage_vs_oracle_selected_seed_t_interval")
        c["validity_flagged"] = bool(c["validity_mean"] < VALIDITY_FLAG_LEVEL)
        c["excluded_from_crossing_on_validity"] = bool(
            c["validity_mean"] < CROSSING_VALIDITY_FLOOR)
        c["crosses"] = bool(
            c.get("priced") and ti is not None
            and c["advantage_vs_oracle_selected"] > 0 and ti["lo"] > 0
            and not c["excluded_from_crossing_on_validity"])
        # D7: is the cell resolved at three generation seeds at all?
        if ti is not None:
            c["advantage_sd"] = ti["sd"]
            c["not_resolved_at_three_generation_seeds"] = bool(
                ti["sd"] > abs(ti["mean"]))
    report["cells"] = cells

    # ------------------------------------------------------------ decision rules
    scored = [c for c in cells.values() if c.get("priced")]
    crossing = [c for c in scored if c["crosses"]]
    dr = report["decision_rules"]

    gates_needed = ["G0", "G1", "G2", "G4", "G5", "G6"]
    g4 = {n: c["cost_identity_max_residual"] for n, c in cells.items()}
    report["validity_gates"]["G4"] = {
        "gate": "G4",
        "rule": ("processed_tokens_actual mod (k + 1) == 0 in every guidance cell and "
                 "every generation seed (C31.0.3 G4)"),
        "cells": g4,
        "n_cells_checked": len(g4),
        "max_residual": max(g4.values()) if g4 else None,
        "passes": bool(g4) and max(g4.values()) == 0,
    }
    gate_status = {g: bool(report["validity_gates"].get(g, {}).get("passes"))
                   for g in gates_needed}
    all_gates_pass = all(gate_status.values())

    # C31.0.7 population-level degeneracy
    n_low = sum(1 for c in scored if c["excluded_from_crossing_on_validity"])
    deployed_low = [c for c in scored
                    if c["arm"] == "deployed" and c["k"] in (2, 4)
                    and c["property"] in REQUIRED_PROPERTIES]
    deployed_all_low = bool(deployed_low) and all(
        c["excluded_from_crossing_on_validity"] for c in deployed_low)
    degenerate = bool(
        scored and (n_low / len(scored) > DEGENERATE_CELL_FRACTION or deployed_all_low))
    report["uninterpretability"] = {
        "gates_pass": gate_status,
        "all_required_gates_pass": all_gates_pass,
        "n_cells_scored": len(scored),
        "n_cells_below_crossing_validity_floor": n_low,
        "fraction_below_floor": (n_low / len(scored)) if scored else None,
        "degenerate_fraction_threshold": DEGENERATE_CELL_FRACTION,
        "all_required_deployed_cheap_cells_below_floor": deployed_all_low,
        "experiment_degenerate": degenerate,
        "uninterpretable": bool((not all_gates_pass) or degenerate),
    }

    d1 = [c for c in crossing if c["k"] in (2, 4)]
    dr["D1"] = {"rule": "the crossing replicates: >= 1 cell crosses at k in {2,4}",
                "cells": sorted(c["dir"] for c in d1), "n": len(d1),
                "fires": bool(d1)}
    dr["D2"] = {"rule": "the crossing is at the cheap end: every crossing cell has k <= 4",
                "crossing_cells": sorted(c["dir"] for c in crossing),
                "crossing_k": sorted({c["k"] for c in crossing}),
                "fires": bool(crossing) and all(c["k"] <= 4 for c in crossing)}

    knob = {}
    for prop in props_run:
        for arm in ("deployed", "mid"):
            a = cells.get(f"{prop}_{arm}_k2")
            b = cells.get(f"{prop}_{arm}_k32")
            if a and b:
                knob[f"{prop}_{arm}"] = {
                    "hit_rate_k2": a["hit_rate_mean"], "hit_rate_k32": b["hit_rate_mean"],
                    "delta": b["hit_rate_mean"] - a["hit_rate_mean"],
                    "tokens_k2": a["tokens_per_molecule_actual"],
                    "tokens_k32": b["tokens_per_molecule_actual"],
                    "token_ratio": (b["tokens_per_molecule_actual"]
                                    / a["tokens_per_molecule_actual"]),
                    "within_threshold": bool(
                        b["hit_rate_mean"] - a["hit_rate_mean"] <= D3_THRESHOLD)}
    dr["D3"] = {"rule": (f"no compute knob: hit rate at k=32 minus k=2 is <= "
                         f"+{D3_THRESHOLD} on every arm"),
                "per_arm": knob,
                "max_delta": max((v["delta"] for v in knob.values()), default=None),
                "fires": bool(knob) and all(v["within_threshold"] for v in knob.values())}

    d4 = {p: report["depth_curves"][p]["test_depth_peak"]
          for p in REQUIRED_PROPERTIES if p in report["depth_curves"]}
    dr["D4"] = {"rule": ("the depth curve replicates: for both required properties the "
                         "probe point maximising held-out TEST target AUROC is strictly "
                         "less than 12"),
                "per_property": d4,
                "fires": bool(d4) and len(d4) == len(REQUIRED_PROPERTIES)
                         and all(v["peaks_before_final"] for v in d4.values())}
    dr["D5"] = {"rule": "the crossing does not replicate: no cell crosses at any k",
                "n_crossing": len(crossing), "fires": bool(scored) and not crossing}

    d6 = [c for c in crossing if c["arm"] == "deployed" and c["k"] == 2]
    dr["D6"] = {"rule": ("the deployed configuration replicates: the deployed arm "
                         "(final probe point, lambda = 1) crosses at k = 2"),
                "cells": sorted(c["dir"] for c in d6), "fires": bool(d6)}

    d7 = {c["dir"]: {"mean_advantage": c["advantage_vs_oracle_selected"],
                     "advantage_sd": c.get("advantage_sd"),
                     "not_resolved": c.get("not_resolved_at_three_generation_seeds")}
          for c in scored if c.get("not_resolved_at_three_generation_seeds")}
    dr["D7"] = {"rule": ("the honesty rule: every cell whose between-generation-seed sd "
                         "exceeds its own mean advantage is reported as NOT RESOLVED at "
                         "three generation seeds"),
                "cells": d7, "n": len(d7), "fires": bool(d7)}

    if report["uninterpretability"]["uninterpretable"]:
        verdict = "UNINTERPRETABLE"
        wording = ("a gate failed or the guidance cells are degenerate; C31.0.7 leaves the "
                   "decision rules unscored as a verdict, though they are computed above")
    elif dr["D1"]["fires"]:
        verdict = "REPLICATES"
        wording = ("the crossing replicates on a second, independent generator: at least "
                   "one guidance cell sits above the oracle-selected best-of-N frontier at "
                   "its own budget at k <= 4, with a seed-level t interval excluding zero")
    elif dr["D5"]["fires"]:
        verdict = "DOES NOT REPLICATE"
        wording = "the crossing is a fact about GP-MoLFormer, not about the method"
    else:
        verdict = "PARTIAL"
        wording = ("some cell has a positive mean advantage but no cell clears the interval "
                   "at k <= 4; reported as PARTIAL and explicitly NOT as a replication")
    report["verdict"] = {"verdict": verdict, "committed_wording": wording,
                         "rule": ("C31.0.6: REPLICATES iff D1; DOES NOT REPLICATE iff D5; "
                                  "PARTIAL otherwise; UNINTERPRETABLE overrides")}

    # -------------------------------------------------------------- predictions
    P = report["predictions"]

    def pred(name, statement, fires, detail):
        P[name] = {"statement": statement, "outcome": "CONFIRMED" if fires else "FALSIFIED",
                   "detail": detail}

    if feas:
        pred("P1", "Stage 0 unconditional RDKit validity >= 0.95",
             feas["validity"] >= 0.95, {"measured": feas["validity"], "threshold": 0.95})
        pred("P2", "Stage 0 uniqueness >= 0.95",
             feas["uniqueness"] >= 0.95, {"measured": feas["uniqueness"], "threshold": 0.95})
    if dr["D4"].get("per_property"):
        pred("P3", ("the depth curve peaks strictly before probe point 12 for both required "
                    "properties (D4 fires)"),
             dr["D4"]["fires"], dr["D4"]["per_property"])
        mids = {p: report["depth_curves"][p]["mid_probe_point"]["selected"]
                for p in REQUIRED_PROPERTIES if p in report["depth_curves"]}
        pred("P4", "the selected mid probe point M is in [3, 8] for both required properties",
             bool(mids) and all(3 <= v <= 8 for v in mids.values()), mids)
        marg = {p: report["depth_curves"][p]["frozen_state_beats_trivial"]
                for p in REQUIRED_PROPERTIES if p in report["depth_curves"]}
        pred("P5", ("the frozen-state head beats the trivial prefix-statistics baseline on "
                    "held-out test target AUROC by >= 0.02 at its best probe point, for both "
                    "required properties"),
             bool(marg) and all(v["margin"] >= 0.02 for v in marg.values()), marg)
    if scored:
        pred("P6", "D1 fires: at least one cell crosses at k in {2,4}",
             dr["D1"]["fires"], {"crossing_cells": dr["D1"]["cells"]})
        pred("P7", "D6 fires: the deployed arm crosses at k = 2",
             dr["D6"]["fires"], {"cells": dr["D6"]["cells"]})
        pred("P8", f"D3 fires: hit rate at k=32 minus k=2 is <= +{D3_THRESHOLD} on every arm",
             dr["D3"]["fires"], knob)
        qed_cells = [c for c in scored if c["property"] == "qed"]
        if qed_cells:
            pred("P9", "qed does not cross at any k, on either arm",
                 not any(c["crosses"] for c in qed_cells),
                 {"crossing": [c["dir"] for c in qed_cells if c["crosses"]],
                  "n_qed_cells": len(qed_cells)})
        k2 = [c for c in scored if c["k"] == 2]
        pred("P10", "guidance mean validity at k = 2 stays >= 0.95 on every arm",
             bool(k2) and all(c["validity_mean"] >= 0.95 for c in k2),
             {c["dir"]: c["validity_mean"] for c in k2})
    mono = {}
    for p, cv in curves.items():
        if cv is None:
            continue
        h = cv["oracle_selected"]
        viol = [(cv["grid"][i], cv["grid"][i + 1], h[i], h[i + 1])
                for i in range(len(h) - 1) if h[i + 1] < h[i]]
        mono[p] = {"monotone": not viol, "violations": viol}
    if mono:
        pred("P11", ("the oracle-selected best-of-N curve is monotone non-decreasing in N "
                     "on the measured grid for every property"),
             all(v["monotone"] for v in mono.values()), mono)

    report["prediction_summary"] = {
        "n": len(P),
        "confirmed": sorted(k for k, v in P.items() if v["outcome"] == "CONFIRMED"),
        "falsified": sorted(k for k, v in P.items() if v["outcome"] == "FALSIFIED"),
    }

    write_json(out_dir / "c31_metrics.json", report)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})

    print(f"[C31] gates: {gate_status}")
    print(f"[C31] {len(scored)} cells priced, {len(crossing)} crossing")
    for name, c in sorted(cells.items()):
        if not c.get("priced"):
            continue
        ti = c.get("advantage_vs_oracle_selected_seed_t_interval") or {}
        print(f"  {c['dir']:<52s} hit={c['hit_rate_mean']:.4f} "
              f"tok={c['tokens_per_molecule_actual']:8.2f} "
              f"adv={c['advantage_vs_oracle_selected']:+.4f} "
              f"[{ti.get('lo', float('nan')):+.4f},{ti.get('hi', float('nan')):+.4f}] "
              f"val={c['validity_mean']:.4f} cross={c['crosses']}")
    for k, v in dr.items():
        print(f"[C31] {k}: fires={v['fires']}")
    print(f"[C31] VERDICT: {verdict}")
    print(f"[C31] predictions confirmed={report['prediction_summary']['confirmed']} "
          f"falsified={report['prediction_summary']['falsified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
