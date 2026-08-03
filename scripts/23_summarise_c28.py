"""C28 -- price the k sweep and the guided-drafts composition against BOTH best-of-N curves.

Reads only artefacts that already exist: the k cells from `scripts/23_k_sweep.py`, the
composition from `scripts/23_guided_drafts.py`, C26's oracle-selected curves and C27's
head-selected curves.  Generates nothing -- no molecule is sampled and no head is trained.

The frontier machinery (`interp`, `t_interval`, `guidance_points`) is **imported** from
`scripts/21_summarise_c26.py`, so C28 prices arms with exactly the code that produced C26's
and C27's numbers; the strand definitions are imported from `scripts/23_k_sweep.py` so the
summariser cannot silently price a different grid from the one that was run.

Scores `outputs/c28_prereg/C28.0_preregistration.md`: validity gates G1-G7, decision rules
D1-D5, and predictions P1-P10, including where they fail.

**No bootstrap.**  C28.0.7: at n = 3 the percentile bootstrap of a mean is identically
[min, max], so it conveys only a three-way sign test at null probability 0.25.  Per-seed
values and a seed-level t interval on 2 df (t(0.975,2) = 4.302653) are reported instead.

    .venv/bin/python scripts/23_summarise_c28.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

SEEDS = ["101", "202", "303"]
#: C26 section C26.4.4: "all 46 arms sit inside a 5.1-17.0% token band", i.e. max/min = 1.170.
C26_BAND_RATIO = 1.17
#: C28.0.6 D1: the ratio above which "guidance has no compute knob" is refuted as stated.
D1_RATIO_THRESHOLD = 2.0
#: C28.0.6 D2: C27's E3 threshold, reused rather than re-chosen.
D2_THRESHOLD = 0.02


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_summarise")
_ks = _load_module(ROOT / "scripts" / "23_k_sweep.py", "c28_k_sweep")
interp = _c26.interp
t_interval = _c26.t_interval
STRANDS = _ks.STRANDS
STRAND_ORDER = _ks.STRAND_ORDER
K_GRID = _ks.K_GRID
cell_dir = _ks.cell_dir
check_gate = _ks.check_gate


def load_curves(prop: str) -> dict | None:
    """C26's oracle-selected curve and C27's head-selected curve, on one token axis."""
    f26 = OUTPUT_DIR / f"c26_nsweep_{prop}" / "n_sweep_metrics.json"
    f27 = OUTPUT_DIR / f"c27_headsel_{prop}" / "head_selected_metrics.json"
    if not f26.exists():
        return None
    d26 = read_json(f26)
    grid = d26["grid"]
    toks = [d26["curve"][str(n)]["tokens_per_molecule_actual"] for n in grid]
    out = {
        "grid": grid,
        "tokens_per_molecule_actual": toks,
        "oracle_selected": [d26["curve"][str(n)]["hit_rate_mean"] for n in grid],
        "oracle_selected_per_seed": {
            s: [d26["per_seed"][s]["rows"][str(n)]["hit_rate"] for n in grid] for s in SEEDS},
        "head_selected": None,
        "head_selected_per_seed": None,
        "sources": {"oracle_selected": f"outputs/c26_nsweep_{prop}/n_sweep_metrics.json"},
    }
    if f27.exists():
        d27 = read_json(f27)
        if d27["grid"] != grid:
            raise SystemExit(f"C28 stop: C26 and C27 grids differ for {prop}")
        out["head_selected"] = [d27["curves"]["head_selected"][str(n)]["hit_rate_mean"]
                                for n in grid]
        out["head_selected_per_seed"] = {
            s: [d27["per_seed"][s]["arms"]["head_selected"][str(n)]["hit_rate"] for n in grid]
            for s in SEEDS}
        out["sources"]["head_selected"] = (
            f"outputs/c27_headsel_{prop}/head_selected_metrics.json")
    return out


def load_extended_curves(prop: str) -> dict | None:
    """POST HOC, NOT PRE-REGISTERED -- best-of-N measured out to N = 80.

    C28.0.4 pre-registered that points beyond C26's grid maximum (1421.98 tokens per
    molecule) are flagged `extrapolated_beyond_grid` and compared against the curve's
    terminal value.  Two C28 points land there: strand A1 at k = 32 (1447.4 tokens) and the
    whole guided-drafts composition (up to 3197 tokens).  Rather than leave those
    comparisons resting on a frozen terminal value, the same unmodified
    `scripts/22_head_selected_bestofn.py` was rerun with `--n-max 80`, which *measures* both
    curves across the composition's budget.  This block is reported separately and labelled
    as a post-hoc addition; the pre-registered comparison is scored as written and is not
    replaced by it.
    """
    f = OUTPUT_DIR / f"c28_bon_extended_{prop}" / "head_selected_metrics.json"
    if not f.exists():
        return None
    d = read_json(f)
    grid = d["grid"]
    return {
        "grid": grid,
        "tokens_per_molecule_actual": [
            d["curves"]["oracle_selected"][str(n)]["tokens_per_molecule_actual"] for n in grid],
        "oracle_selected": [d["curves"]["oracle_selected"][str(n)]["hit_rate_mean"]
                            for n in grid],
        "oracle_selected_per_seed": {
            s: [d["per_seed"][s]["arms"]["oracle_selected"][str(n)]["hit_rate"] for n in grid]
            for s in SEEDS},
        "head_selected": [d["curves"]["head_selected"][str(n)]["hit_rate_mean"] for n in grid],
        "head_selected_per_seed": {
            s: [d["per_seed"][s]["arms"]["head_selected"][str(n)]["hit_rate"] for n in grid]
            for s in SEEDS},
        "sources": {"both": f"outputs/c28_bon_extended_{prop}/head_selected_metrics.json"},
    }


def price(curves: dict, hit_mean: float, budget: float, per_seed: dict) -> dict:
    """Advantage of one guidance point against both best-of-N curves at its own budget."""
    toks = curves["tokens_per_molecule_actual"]
    grid = curves["grid"]
    row: dict = {}
    for arm in ("oracle_selected", "head_selected"):
        if curves[arm] is None:
            continue
        h, i_lo, i_hi, extrap = interp(toks, curves[arm], budget)
        row[f"{arm}_interpolated_hit_rate"] = h
        row[f"advantage_vs_{arm}"] = hit_mean - h
        row[f"bracketing_n_{arm}"] = [grid[i_lo], grid[i_hi]]
        row[f"extrapolated_beyond_grid_{arm}"] = extrap
        adv = []
        for s in SEEDS:
            ps = per_seed.get(s)
            if ps is None:
                continue
            sb, *_ = interp(toks, curves[f"{arm}_per_seed"][s], ps["tokens_per_molecule_actual"])
            adv.append(ps["hit_rate"] - sb)
        row[f"advantage_vs_{arm}_per_seed"] = adv
        if len(adv) == len(SEEDS):
            row[f"advantage_vs_{arm}_seed_t_interval"] = t_interval(adv)
    return row


def collect_k_cells() -> dict:
    cells: dict[str, dict] = {}
    for strand in STRAND_ORDER:
        rows = {}
        for k in K_GRID:
            f = cell_dir(strand, k) / "k_cell_metrics.json"
            if not f.exists():
                continue
            d = read_json(f)
            agg = d["aggregate"]
            rows[str(k)] = {
                "k": k,
                "dir": cell_dir(strand, k).name,
                "hit_rate_mean": agg["hit_rate"]["mean"],
                "hit_rate_values": agg["hit_rate"]["values"],
                "hit_rate_sd": float(np.std(agg["hit_rate"]["values"], ddof=1)),
                "tokens_per_molecule_actual":
                    agg["compute_total"]["tokens_per_molecule_actual"],
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
                        "cost_identity_tokens_mod_k_plus_1":
                            d["seeds_detail"][s]["cost_identity_tokens_mod_k_plus_1"]}
                    for s in SEEDS if s in d["seeds_detail"]},
            }
        if rows:
            cells[strand] = {"property": STRANDS[strand]["property"],
                             "layer": STRANDS[strand]["layer"],
                             "lam": STRANDS[strand]["lam"],
                             "why": STRANDS[strand]["why"],
                             "gate_run": STRANDS[strand]["gate_run"],
                             "k_cells": rows}
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c28_summary")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "experiment": "C28",
        "prereg": "outputs/c28_prereg/C28.0_preregistration.md",
        "accounting": "actual",
        "seeds": SEEDS,
        "uncertainty": ("per-seed values and a seed-level t interval on 2 df "
                        "(t_0.975,2 = 4.302653); no bootstrap anywhere in C28, see C28.0.7"),
        "validity_gates": {},
        "strands": {},
        "decision_rules": {},
        "predictions": {},
    }

    cells = collect_k_cells()
    curves = {p: load_curves(p) for p in sorted({v["property"] for v in cells.values()})}

    # -------------------------------------------------------------- gates G1-G3, G4
    gates = {}
    for strand in cells:
        gates[strand] = check_gate(strand)
    report["validity_gates"]["G1_G3_k8_reproduces_frozen_artefact"] = {
        "rule": ("the k = 8 cell of each strand must reproduce its frozen artefact's "
                 "`throughout` condition exactly: hit-rate residual 0.0 and "
                 "tokens-per-molecule residual 0.0, aggregate and every seed"),
        "per_strand": gates,
        "max_abs_hit_rate_residual": max(
            (abs(g["hit_rate_residual"]) for g in gates.values() if g.get("checked")),
            default=None),
        "max_abs_token_residual": max(
            (abs(g["tokens_residual"]) for g in gates.values() if g.get("checked")),
            default=None),
        "passes": all(g.get("passes") for g in gates.values() if g.get("checked")),
    }
    g4 = {st: {k: r["cost_identity_max_residual"] for k, r in c["k_cells"].items()}
          for st, c in cells.items()}
    report["validity_gates"]["G4_cost_identity"] = {
        "rule": ("the cached backend charges `active` + `active * k` at every guided step, "
                 "so processed_tokens_actual mod (k + 1) must be 0 for every cell and seed"),
        "per_strand": g4,
        "max_residual": max((v for st in g4.values() for v in st.values()), default=None),
        "passes": all(v == 0 for st in g4.values() for v in st.values()),
    }

    # ----------------------------------------------------------------- strand results
    for strand, c in cells.items():
        prop = c["property"]
        cv = curves.get(prop)
        rows = {}
        for k, r in c["k_cells"].items():
            row = dict(r)
            if cv is not None:
                row.update(price(cv, r["hit_rate_mean"], r["tokens_per_molecule_actual"],
                                 r["per_seed"]))
            rows[k] = row
        ks = sorted((int(k) for k in rows), key=int)
        toks = [rows[str(k)]["tokens_per_molecule_actual"] for k in ks]
        hits = [rows[str(k)]["hit_rate_mean"] for k in ks]
        vals = [rows[str(k)]["validity_mean"] for k in ks]
        report["strands"][strand] = {
            "property": prop, "layer": c["layer"], "lam": c["lam"], "why": c["why"],
            "gate_run": c["gate_run"],
            "k_grid": ks,
            "tokens_per_molecule_actual": toks,
            "hit_rate_mean": hits,
            "validity_mean": vals,
            "token_ratio_max_over_min": (max(toks) / min(toks)) if toks else None,
            "cells": rows,
        }

    # ---------------------------------------------------------------- strand B
    b_f = OUTPUT_DIR / "c28_guided_drafts_hbd_count" / "guided_drafts_metrics.json"
    strand_b = None
    if b_f.exists():
        b = read_json(b_f)
        cv = curves.get(b["property"])
        rows = {}
        for arm in b["arms"]:
            arows = {}
            for n in b["grid"]:
                e = b["curves"][arm][str(n)]
                ps = {s: {"hit_rate": b["per_seed"][s]["arms"][arm][str(n)]["hit_rate"],
                          "tokens_per_molecule_actual":
                              b["per_seed"][s]["arms"][arm][str(n)]["compute"]
                              ["tokens_per_molecule_actual"]}
                      for s in SEEDS}
                row = {"n_drafts": n, **e, "per_seed": ps}
                if cv is not None:
                    row.update(price(cv, e["hit_rate_mean"], e["tokens_per_molecule_actual"],
                                     ps))
                arows[str(n)] = row
            rows[arm] = arows
        strand_b = {
            "property": b["property"], "grid": b["grid"], "top_k": b["top_k"],
            "lambda": b["lambda"], "layer": b["layer"],
            "deployed_reference": b["deployed_reference"],
            "head_scoring_recompute_tokens_per_pool_molecule_mean":
                b["head_scoring_recompute_tokens_per_pool_molecule_mean"],
            "head_auroc_terminal_position_on_guided_pool": {
                s: b["per_seed"][s]["head_auroc_terminal_position_on_guided_pool"]
                for s in SEEDS},
            "pool_true_hit_rate": {s: b["per_seed"][s]["pool_true_hit_rate"] for s in SEEDS},
            "arms": rows,
        }
        report["strands"]["B_guided_drafts"] = strand_b
        report["validity_gates"]["G5_n1_arms_identical"] = {
            "rule": "at N = 1 there is one draft and nothing to rerank; both arms must agree",
            "max_abs_residual": b["gates"]["G5_n1_arms_identical"]["max_abs_residual"],
            "passes": b["gates"]["G5_n1_arms_identical"]["max_abs_residual"] == 0.0,
        }
        report["validity_gates"]["G6_pool_provenance"] = {
            "rule": ("the first 512 drafts of each seed must be the published deployed run's "
                     "512, compared as SMILES strings"),
            "mismatches_per_seed": b["gates"]["G6_first_512_smiles_mismatches"],
            "passes": all(v == 0 for v in
                          b["gates"]["G6_first_512_smiles_mismatches"].values()),
        }
        report["validity_gates"]["G7_per_draft_token_attribution"] = {
            "rule": "sum over drafts of (k+1)*(len(seq)-1) equals the meter's own total",
            "residual_per_seed": b["gates"]["G7_per_draft_token_residual"],
            "passes": all(v == 0 for v in b["gates"]["G7_per_draft_token_residual"].values()),
        }

    # ------------------------------------------------------------------ decision rules
    # D1 -- the cost band
    d1 = {}
    for strand, s in report["strands"].items():
        if strand == "B_guided_drafts":
            continue
        d1[strand] = {
            "token_ratio_max_over_min": s["token_ratio_max_over_min"],
            "tokens_min": min(s["tokens_per_molecule_actual"]),
            "tokens_max": max(s["tokens_per_molecule_actual"]),
            "c26_band_ratio": C26_BAND_RATIO,
            "exceeds_threshold": bool(s["token_ratio_max_over_min"] > D1_RATIO_THRESHOLD),
        }
    report["decision_rules"]["D1_cost_band"] = {
        "rule": ("C26's 'no compute knob' is REFUTED AS STATED iff max/min tokens per "
                 f"molecule within a single strand exceeds {D1_RATIO_THRESHOLD}, every cell "
                 "being the same method with the same frozen generator, head and lambda"),
        "per_strand": d1,
        "verdict": ("REFUTED AS STATED" if any(v["exceeds_threshold"] for v in d1.values())
                    else "UPHELD"),
    }

    # D2 -- does the knob buy accuracy?
    d2 = {}
    for strand, s in report["strands"].items():
        if strand == "B_guided_drafts" or "32" not in s["cells"] or "8" not in s["cells"]:
            continue
        a, b8 = s["cells"]["32"], s["cells"]["8"]
        diffs = [a["hit_rate_values"][i] - b8["hit_rate_values"][i] for i in range(3)]
        d = a["hit_rate_mean"] - b8["hit_rate_mean"]
        same_sign = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
        d2[strand] = {
            "hit_rate_k8": b8["hit_rate_mean"], "hit_rate_k32": a["hit_rate_mean"],
            "difference": d, "per_seed_difference": diffs,
            "all_seeds_share_sign": bool(same_sign),
            "seed_t_interval": t_interval(diffs),
            "threshold": D2_THRESHOLD,
            "verdict": ("PRODUCTIVE" if (d > D2_THRESHOLD and same_sign)
                        else "HARMFUL" if (d < -D2_THRESHOLD and same_sign)
                        else "NULL" if abs(d) <= D2_THRESHOLD else "INCONSISTENT"),
        }
        # best k on the grid, and the difference against the cheapest k
        d2[strand]["best_k_by_hit_rate"] = int(
            s["k_grid"][int(np.argmax(s["hit_rate_mean"]))])
        d2[strand]["hit_rate_at_best_k"] = float(max(s["hit_rate_mean"]))
        d2[strand]["hit_rate_at_k2"] = s["cells"]["2"]["hit_rate_mean"] \
            if "2" in s["cells"] else None
    report["decision_rules"]["D2_does_the_knob_buy_accuracy"] = {
        "rule": (f"PRODUCTIVE iff hit_rate(k=32) - hit_rate(k=8) > +{D2_THRESHOLD} with all "
                 f"three seeds sharing the sign; NULL iff |difference| <= {D2_THRESHOLD}; "
                 f"HARMFUL iff below -{D2_THRESHOLD} with a consistent sign"),
        "per_strand": d2,
    }

    # D3 -- the frontier verdict at k = 32
    d3 = {}
    for strand, s in report["strands"].items():
        if strand == "B_guided_drafts" or "32" not in s["cells"]:
            continue
        c32 = s["cells"]["32"]
        d3[strand] = {
            "tokens_per_molecule_actual": c32["tokens_per_molecule_actual"],
            "guided_hit_rate": c32["hit_rate_mean"],
            "oracle_selected_interpolated_hit_rate":
                c32.get("oracle_selected_interpolated_hit_rate"),
            "advantage_vs_oracle_selected": c32.get("advantage_vs_oracle_selected"),
            "advantage_vs_oracle_selected_seed_t_interval":
                c32.get("advantage_vs_oracle_selected_seed_t_interval"),
            "head_selected_interpolated_hit_rate":
                c32.get("head_selected_interpolated_hit_rate"),
            "advantage_vs_head_selected": c32.get("advantage_vs_head_selected"),
            "advantage_vs_head_selected_seed_t_interval":
                c32.get("advantage_vs_head_selected_seed_t_interval"),
            "extrapolated_beyond_grid":
                c32.get("extrapolated_beyond_grid_oracle_selected"),
            "D3a_above_oracle_selected": bool(
                (c32.get("advantage_vs_oracle_selected") or -1.0) > 0),
            "D3b_above_head_selected": bool(
                (c32.get("advantage_vs_head_selected") or -1.0) > 0),
        }
    report["decision_rules"]["D3_frontier_verdict_at_k32"] = {
        "rule": ("D3a: guidance at k = 32 above/below the oracle-selected curve at its own "
                 "budget.  D3b: the same against the head-selected curve.  C26's negative "
                 "result survives in its scoped form iff D3a is 'below' on every strand."),
        "per_strand": d3,
        "D3a_all_below": all(not v["D3a_above_oracle_selected"] for v in d3.values()),
        "D3b_deployed_strand_A1_above":
            d3.get("A1", {}).get("D3b_above_head_selected"),
    }

    # D4 -- the composition
    if strand_b is not None:
        d4 = {}
        for arm, arows in strand_b["arms"].items():
            best = None
            for n, r in arows.items():
                if int(n) == 1:
                    continue
                adv = r.get("advantage_vs_head_selected")
                signs = r.get("advantage_vs_head_selected_per_seed") or []
                consistent = len(signs) == 3 and (all(x > 0 for x in signs)
                                                  or all(x < 0 for x in signs))
                if adv is not None and adv > 0 and consistent and (
                        best is None or adv > best["advantage_vs_head_selected"]):
                    best = {"n_drafts": int(n), **{kk: r[kk] for kk in (
                        "hit_rate_mean", "tokens_per_molecule_actual",
                        "advantage_vs_head_selected", "advantage_vs_oracle_selected")}}
            d4[arm] = {
                "per_n": {n: {kk: r.get(kk) for kk in (
                    "hit_rate_mean", "hit_rate_values", "tokens_per_molecule_actual",
                    "validity_mean",
                    "oracle_selected_interpolated_hit_rate", "advantage_vs_oracle_selected",
                    "head_selected_interpolated_hit_rate", "advantage_vs_head_selected",
                    "extrapolated_beyond_grid_oracle_selected")}
                    for n, r in arows.items()},
                "usable_compute_axis": best is not None,
                "best_violating_n": best,
                "dominates": bool(best is not None
                                  and best.get("advantage_vs_oracle_selected", -1) > 0),
            }
        report["decision_rules"]["D4_composition"] = {
            "rule": ("the composition is a usable compute axis for guidance iff at some N it "
                     "sits above the head-selected curve at its own budget with all three "
                     "seeds sharing the sign; it dominates iff it also sits above the "
                     "oracle-selected curve"),
            "per_arm": d4,
        }

    # D5 -- the defence: is the k profile flat, and is the truncation control still null?
    d5 = {}
    for strand, s in report["strands"].items():
        if strand == "B_guided_drafts":
            continue
        prop = s["property"]
        dep = OUTPUT_DIR / f"pilot_50k_p2_guided_{prop}" / "guidance_metrics.json"
        ref = read_json(dep)["conditions"] if dep.exists() else None
        d5[strand] = {
            "hit_rate_by_k": dict(zip([str(k) for k in s["k_grid"]], s["hit_rate_mean"])),
            "hit_rate_span": max(s["hit_rate_mean"]) - min(s["hit_rate_mean"]),
            "unguided_hit_rate": ref["unguided"]["aggregate"]["hit_rate"]["mean"] if ref else None,
            "truncation_control_hit_rate":
                ref["truncation_control"]["aggregate"]["hit_rate"]["mean"] if ref else None,
            "truncation_control_minus_unguided": (
                ref["truncation_control"]["aggregate"]["hit_rate"]["mean"]
                - ref["unguided"]["aggregate"]["hit_rate"]["mean"]) if ref else None,
        }
    report["decision_rules"]["D5_the_defence"] = {
        "rule": ("if hit rate is flat in k, the null molecular truncation control is the "
                 "explanation and must be reported as such -- a statement about a ~2.4k-token "
                 "SMILES vocabulary, not about FUDGE, which the GPT-2 truncation control "
                 "(47.5-85.7% of base hit rate destroyed) scopes"),
        "per_strand": d5,
    }

    # -------------------------------------------------------------------- predictions
    preds: dict = {}
    a1 = report["strands"].get("A1")
    if a1:
        dep_tok = 401.619140625
        p1 = {}
        for k in a1["k_grid"]:
            expect = dep_tok * (k + 1) / 9.0
            got = a1["cells"][str(k)]["tokens_per_molecule_actual"]
            p1[str(k)] = {"expected": expect, "measured": got,
                          "relative_error": (got - expect) / expect}
        preds["P1_cost_is_k_plus_1_times_base"] = {
            "statement": ("tokens per molecule at k is within +/-10% of 401.619141*(k+1)/9 "
                          "for every cell of strand A1"),
            "per_k": p1,
            "max_abs_relative_error": max(abs(v["relative_error"]) for v in p1.values()),
            "holds": all(abs(v["relative_error"]) <= 0.10 for v in p1.values()),
        }
        preds["P2_hit_rate_non_decreasing_in_k"] = {
            "statement": "hit_rate(k=32) >= hit_rate(k=2) on strand A1",
            "hit_rate_k2": a1["cells"]["2"]["hit_rate_mean"] if "2" in a1["cells"] else None,
            "hit_rate_k32": a1["cells"]["32"]["hit_rate_mean"] if "32" in a1["cells"] else None,
            "holds": bool(a1["cells"]["32"]["hit_rate_mean"]
                          >= a1["cells"]["2"]["hit_rate_mean"])
            if "32" in a1["cells"] and "2" in a1["cells"] else None,
        }
        if "32" in a1["cells"] and "8" in a1["cells"]:
            diff = a1["cells"]["32"]["hit_rate_mean"] - a1["cells"]["8"]["hit_rate_mean"]
            preds["P3_knob_is_weak_here"] = {
                "statement": "hit_rate(k=32) - hit_rate(k=8) on strand A1 is less than +0.10",
                "difference": diff, "holds": bool(diff < 0.10)}
    d3 = report["decision_rules"].get("D3_frontier_verdict_at_k32", {}).get("per_strand", {})
    if d3:
        preds["P4_below_oracle_curve_at_k32"] = {
            "statement": "guidance at k = 32 sits below the oracle-selected curve on every strand",
            "per_strand": {s: v["advantage_vs_oracle_selected"] for s, v in d3.items()},
            "holds": all(not v["D3a_above_oracle_selected"] for v in d3.values())}
        if "A1" in d3:
            preds["P5_deployed_below_head_curve_at_k32"] = {
                "statement": "strand A1 at k = 32 sits below the head-selected curve",
                "advantage": d3["A1"]["advantage_vs_head_selected"],
                "holds": bool(not d3["A1"]["D3b_above_head_selected"])}
        if "A3" in d3:
            preds["P6_strong_arm_above_head_curve_at_k32"] = {
                "statement": "strand A3 at k = 32 sits above the head-selected curve",
                "advantage": d3["A3"]["advantage_vs_head_selected"],
                "holds": bool(d3["A3"]["D3b_above_head_selected"])}
    p7 = {}
    for strand, s in report["strands"].items():
        if strand == "B_guided_drafts" or "32" not in s["cells"] or "2" not in s["cells"]:
            continue
        p7[strand] = {"validity_k2": s["cells"]["2"]["validity_mean"],
                      "validity_k32": s["cells"]["32"]["validity_mean"],
                      "holds": bool(s["cells"]["32"]["validity_mean"]
                                    < s["cells"]["2"]["validity_mean"])}
    if p7:
        preds["P7_validity_falls_with_k"] = {
            "statement": "validity at k = 32 is lower than at k = 2 on every strand",
            "per_strand": p7, "holds": all(v["holds"] for v in p7.values())}
    if strand_b is not None:
        o8 = strand_b["arms"]["oracle_reranked"].get("8")
        h8 = strand_b["arms"]["head_reranked"].get("8")
        if o8:
            preds["P8_oracle_reranked_above_0p90_but_below_curve"] = {
                "statement": ("oracle_reranked at N = 8 exceeds 0.90 hit rate and nevertheless "
                              "sits below the oracle-selected best-of-N curve at its budget"),
                "hit_rate": o8["hit_rate_mean"],
                "tokens_per_molecule_actual": o8["tokens_per_molecule_actual"],
                "advantage_vs_oracle_selected": o8.get("advantage_vs_oracle_selected"),
                "holds": bool(o8["hit_rate_mean"] > 0.90
                              and (o8.get("advantage_vs_oracle_selected") or 0) < 0)}
        if h8:
            preds["P9_head_reranked_above_head_curve_terminal"] = {
                "statement": ("head_reranked at N = 8 sits above the head-selected curve's "
                              "terminal value"),
                "hit_rate": h8["hit_rate_mean"],
                "head_selected_terminal": h8.get("head_selected_interpolated_hit_rate"),
                "advantage_vs_head_selected": h8.get("advantage_vs_head_selected"),
                "holds": bool((h8.get("advantage_vs_head_selected") or -1) > 0)}
    if a1:
        preds["P10_cost_band_refuted"] = {
            "statement": ("the measured max/min token ratio within strand A1 exceeds 3.0, "
                          f"against C26's {C26_BAND_RATIO}"),
            "ratio": a1["token_ratio_max_over_min"],
            "holds": bool(a1["token_ratio_max_over_min"] > 3.0)}
    report["predictions"] = preds

    # ---------------------------------------------- POST HOC: the extended best-of-N curve
    ext = {p: load_extended_curves(p) for p in curves}
    ext = {p: v for p, v in ext.items() if v is not None}
    if ext:
        ph: dict = {
            "status": "POST HOC -- NOT PRE-REGISTERED",
            "why": ("C28.0.4 flags points beyond C26's grid maximum (1421.98 tokens per "
                    "molecule) as extrapolated and prices them against the curve's terminal "
                    "value.  Strand A1 at k = 32 and the whole composition land there.  The "
                    "same unmodified scripts/22_head_selected_bestofn.py was rerun with "
                    "--n-max 80 so those budgets are measured rather than extrapolated.  The "
                    "pre-registered comparison above is scored as written and is not "
                    "replaced."),
            "grid": {p: v["grid"] for p, v in ext.items()},
            "tokens_per_molecule_actual": {p: v["tokens_per_molecule_actual"]
                                           for p, v in ext.items()},
            "oracle_selected": {p: v["oracle_selected"] for p, v in ext.items()},
            "head_selected": {p: v["head_selected"] for p, v in ext.items()},
            "consistency_with_c26_c27_at_n32": {},
            "k_cells": {},
            "composition": {},
        }
        for p, v in ext.items():
            base = curves[p]
            i_new = v["grid"].index(32)
            i_old = base["grid"].index(32)
            ph["consistency_with_c26_c27_at_n32"][p] = {
                "oracle_extended": v["oracle_selected"][i_new],
                "oracle_c26": base["oracle_selected"][i_old],
                "oracle_difference": v["oracle_selected"][i_new] - base["oracle_selected"][i_old],
                "head_extended": v["head_selected"][i_new],
                "head_c27": base["head_selected"][i_old] if base["head_selected"] else None,
                "head_difference": (v["head_selected"][i_new] - base["head_selected"][i_old])
                                   if base["head_selected"] else None,
                "note": ("not an identity: the extended pool is 80 x 512 rather than 32 x 512, "
                         "so N = 32 is estimated over 2.5x more disjoint groups"),
            }
        for strand, s in report["strands"].items():
            if strand == "B_guided_drafts" or s["property"] not in ext:
                continue
            rows = {}
            for k, r in s["cells"].items():
                rows[k] = price(ext[s["property"]], r["hit_rate_mean"],
                                r["tokens_per_molecule_actual"], r["per_seed"])
            ph["k_cells"][strand] = rows
        if strand_b is not None and strand_b["property"] in ext:
            for arm, arows in strand_b["arms"].items():
                ph["composition"][arm] = {
                    n: price(ext[strand_b["property"]], r["hit_rate_mean"],
                             r["tokens_per_molecule_actual"], r["per_seed"])
                    for n, r in arows.items()}
        report["post_hoc_extended_best_of_n_curve"] = ph

    write_json(out_dir / "c28_metrics.json", report)
    write_run_context(out_dir, {
        "cli": vars(args),
        "reads": {
            "k_cells": [c["dir"] for s in cells.values() for c in s["k_cells"].values()],
            "composition": str(b_f.name) if b_f.exists() else None,
            "c26_curves": [f"c26_nsweep_{p}" for p in curves],
            "c27_curves": [f"c27_headsel_{p}" for p in curves],
            "frontier_machinery": "scripts/21_summarise_c26.py (interp, t_interval)",
            "strand_definitions": "scripts/23_k_sweep.py (STRANDS, K_GRID, check_gate)",
        },
        "generates": "nothing -- no molecule is sampled and no head is trained",
    })

    if not args.no_figure:
        try:
            make_figure(report, curves, OUTPUT_DIR / "c28_figures")
        except Exception as exc:  # pragma: no cover - plotting is a convenience
            print(f"[C28] figure skipped: {exc}")

    print(json.dumps({
        "G1_G3_max_abs_hit_residual":
            report["validity_gates"]["G1_G3_k8_reproduces_frozen_artefact"]
            ["max_abs_hit_rate_residual"],
        "G4_max_residual": report["validity_gates"]["G4_cost_identity"]["max_residual"],
        "D1": report["decision_rules"]["D1_cost_band"]["verdict"],
        "D2": {s: v["verdict"] for s, v in
               report["decision_rules"]["D2_does_the_knob_buy_accuracy"]["per_strand"].items()},
        "D3": {s: {"vs_oracle": v["advantage_vs_oracle_selected"],
                   "vs_head": v["advantage_vs_head_selected"]} for s, v in d3.items()},
        "predictions": {k: v.get("holds") for k, v in preds.items()},
    }, indent=1))
    print(f"-> {out_dir}")
    return 0


def make_figure(report: dict, curves: dict, fig_dir: Path) -> None:
    """Guidance and BOTH best-of-N curves on one processed-token x-axis.

    That axis has never existed in this project: guidance had one budget and best-of-N had a
    curve, so they were only ever compared at a single matched point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    props = sorted({s["property"] for k, s in report["strands"].items()
                    if k != "B_guided_drafts"})
    fig, axes = plt.subplots(1, len(props), figsize=(6.2 * len(props), 5.0), squeeze=False)
    for ax, prop in zip(axes[0], props):
        cv = curves.get(prop)
        if cv:
            ax.plot(cv["tokens_per_molecule_actual"], cv["oracle_selected"], "o-",
                    color="#222222", label="best-of-N, oracle-selected (C26)")
            if cv["head_selected"]:
                ax.plot(cv["tokens_per_molecule_actual"], cv["head_selected"], "s--",
                        color="#777777", label="best-of-N, head-selected (C27)")
        ph = report.get("post_hoc_extended_best_of_n_curve")
        if ph and prop in ph["grid"]:
            x = ph["tokens_per_molecule_actual"][prop]
            ax.plot(x, ph["oracle_selected"][prop], "o-", color="#222222", alpha=0.35,
                    linewidth=1.0, markersize=3,
                    label="best-of-N to N=80 (C28.6, post hoc)")
            ax.plot(x, ph["head_selected"][prop], "s--", color="#777777", alpha=0.35,
                    linewidth=1.0, markersize=3)
        colors = {"A1": "#d62728", "A2": "#1f77b4", "A3": "#2ca02c",
                  "C1": "#d62728", "C2": "#2ca02c", "C3": "#d62728"}
        for st, s in report["strands"].items():
            if st == "B_guided_drafts" or s["property"] != prop:
                continue
            ax.plot(s["tokens_per_molecule_actual"], s["hit_rate_mean"], "^-",
                    color=colors.get(st, "#9467bd"),
                    label=f"guidance k sweep {st} (L{s['layer']}, lam={s['lam']:g})")
            for k, x, y in zip(s["k_grid"], s["tokens_per_molecule_actual"],
                               s["hit_rate_mean"]):
                ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(4, -10),
                            fontsize=7, color=colors.get(st, "#9467bd"))
        b = report["strands"].get("B_guided_drafts")
        if b and b["property"] == prop:
            for arm, style in (("oracle_reranked", "v:"), ("head_reranked", "d:")):
                xs = [b["arms"][arm][str(n)]["tokens_per_molecule_actual"] for n in b["grid"]]
                ys = [b["arms"][arm][str(n)]["hit_rate_mean"] for n in b["grid"]]
                ax.plot(xs, ys, style, color="#ff7f0e" if "oracle" in arm else "#8c564b",
                        label=f"guided drafts, {arm}")
        ax.set_xscale("log")
        ax.set_xlabel("processed generator tokens per returned molecule (actual)")
        ax.set_ylabel("hit rate")
        ax.set_title(prop)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"c28_k_sweep_frontier.{ext}", dpi=180)
    plt.close(fig)
    print(f"[C28] figure -> {fig_dir}/c28_k_sweep_frontier.png")


if __name__ == "__main__":
    raise SystemExit(main())
