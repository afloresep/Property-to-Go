"""C27 -- price the 46 existing guidance arms against the HEAD-selected best-of-N curve.

Reads only artefacts that already exist: C27's three-arm sweeps from
`scripts/22_head_selected_bestofn.py`, C26's oracle-selected curves, and every guidance run
on disk.  Generates nothing -- no molecule is sampled here.

The frontier machinery is imported from `scripts/21_summarise_c26.py` rather than copied
(`guidance_points`, `interp`, `t_interval`), so C27 prices arms with exactly the code that
produced C26's numbers and the two frontiers differ only in which curve they are priced
against.

Scores `outputs/c27_prereg/C27.0_preregistration.md`: validity gates 1-5, decision rules
E1-E4, sensitivity S1 and the six predictions.

**No bootstrap.**  C27.0.7: at n = 3 the percentile bootstrap of a mean is identically
[min, max] -- P(all three resamples hit the minimum) = 1/27 = 0.0370 > 0.025 -- so it conveys
nothing beyond a three-way sign test at null probability 0.25.  Per-seed values and a
seed-level t interval on 2 df are reported instead, and where n = 3 cannot support an
inference the section says so.

    .venv/bin/python scripts/22_summarise_c27.py
"""

from __future__ import annotations

import argparse
import hashlib
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

ANCHORS = ["aromatic_rings", "hbd_count", "qed"]
SEEDS = ["101", "202", "303"]
ARMS = ["oracle_selected", "head_selected", "head_selected_at_75pct"]
DATASET = "pilot_50k_p2"
# C27.0.6 E3: an arm that gains less than this from N=1 to N=32 carries no usable ranking
# signal, and E1 must not be read as evidence for guidance on that anchor.
E3_DEGENERACY_THRESHOLD = 0.02


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_summarise")
guidance_points = _c26.guidance_points
interp = _c26.interp
t_interval = _c26.t_interval


def effective_n(grid, head_hits, target: float):
    """Smallest N (linearly interpolated in N) at which head selection reaches `target`.

    None when the head curve never reaches it inside the measured grid; C27 does not
    extrapolate a curve to manufacture an effective N.
    """
    for i in range(len(grid)):
        if head_hits[i] >= target:
            if i == 0:
                return float(grid[0])
            a, b = head_hits[i - 1], head_hits[i]
            w = 0.0 if b == a else (target - a) / (b - a)
            return float(grid[i - 1] + w * (grid[i] - grid[i - 1]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c27_summary")
    args = ap.parse_args()

    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "experiment": "C27",
        "prereg": "outputs/c27_prereg/C27.0_preregistration.md",
        "accounting": "actual",
        "seeds": SEEDS,
        "uncertainty": ("per-seed values and a seed-level t interval on 2 df "
                        "(t_0.975,2 = 4.302653); no bootstrap anywhere in C27, see C27.0.7"),
        "validity_gates": {},
        "properties": {},
        "decision_rules": {},
        "sensitivity_S1_pessimistic_accounting": {},
        "predictions": {},
    }

    sweeps, c26 = {}, {}
    for prop in ANCHORS:
        f = OUTPUT_DIR / f"c27_headsel_{prop}" / "head_selected_metrics.json"
        g = OUTPUT_DIR / f"c26_nsweep_{prop}" / "n_sweep_metrics.json"
        if f.exists():
            sweeps[prop] = read_json(f)
        if g.exists():
            c26[prop] = read_json(g)
    if not sweeps:
        raise SystemExit("no C27 sweep artefacts found")

    # ---------------------------------------------------------------- gate 4: the head
    #
    # C27.0.5 gate 4 pre-registered "the SHA-256 of the loaded checkpoint must equal that
    # of head_<prop>_frozen_state_seed1234.pt".  **That criterion is wrong as written and
    # it fails**, on all three anchors.  `torch.save` writes a zip whose internal archive
    # directory is named after the output file, so `head_qed_frozen_state.pt` and
    # `head_qed_frozen_state_seed1234.pt` differ in file bytes (and in length, by 108-172
    # bytes) while holding identical tensors: `scripts/03_train_heads.py` saves the *same*
    # `ckpt` dict twice under two names.  A file hash tests serialisation, not content.
    #
    # The criterion is replaced here rather than waived, in the direction that makes the
    # gate stricter about the thing it was meant to check: every parameter tensor, every
    # metadata field and the binner are compared element-wise, and an order-stable hash of
    # the parameter bytes is published so the comparison is checkable.  The failed
    # file-hash criterion is kept in the record under `prereg_criterion_file_sha256_*` so
    # the discarded test is visible rather than silently replaced.
    def _content_gate(prop: str) -> dict:
        import torch  # local: the summariser is otherwise import-light

        h = sweeps[prop]["head"]
        a = Path(h["head_file"])
        b = Path(h["head_seed_twin"])
        ca = torch.load(a, map_location="cpu", weights_only=False)
        out = {
            **h,
            "prereg_criterion_file_sha256_equal": bool(h["sha256_matches_seed_twin"]),
            "prereg_criterion_file_sha256_note": (
                "FAILS by construction: torch.save names the zip archive after the output "
                "file, so two saves of one dict under two names differ in bytes; see the "
                "block comment in scripts/22_summarise_c27.py and section C27.2.4"),
            "file_bytes": a.stat().st_size,
            "seed_twin_file_bytes": b.stat().st_size if b.exists() else None,
        }
        hsh = hashlib.sha256()
        for k in sorted(ca["state_dict"]):
            hsh.update(k.encode())
            hsh.update(ca["state_dict"][k].detach().cpu().numpy().tobytes())
        out["parameter_sha256"] = hsh.hexdigest()
        if b.exists():
            cb = torch.load(b, map_location="cpu", weights_only=False)
            out["tensors_identical_to_seed_twin"] = bool(
                set(ca["state_dict"]) == set(cb["state_dict"])
                and all(torch.equal(ca["state_dict"][k], cb["state_dict"][k])
                        for k in ca["state_dict"]))
            out["metadata_identical_to_seed_twin"] = bool(all(
                ca[k] == cb[k] for k in
                ("in_dim", "hidden_dim", "n_bins", "dropout", "property", "input",
                 "head_seed")))
            out["binner_identical_to_seed_twin"] = bool(ca["binner"] == cb["binner"])
        return out

    g4 = {p: _content_gate(p) for p in sweeps}
    report["validity_gates"]["gate_4_head_provenance"] = {
        "rule_as_preregistered": ("the loaded checkpoint must be byte-identical (SHA-256) "
                                  "to its head-seed twin"),
        "rule_as_executed": ("the loaded checkpoint must hold parameter-identical, "
                             "metadata-identical and binner-identical content to its "
                             "head-seed twin; the file-hash criterion tests serialisation "
                             "rather than content and is scored as a pre-registration "
                             "failure in C27.2.4"),
        "per_property": g4,
        "preregistered_file_sha256_criterion_passes": all(
            v["prereg_criterion_file_sha256_equal"] for v in g4.values()),
        "all_tensors_identical_to_seed_twin": all(
            v.get("tensors_identical_to_seed_twin") for v in g4.values()),
        "all_metadata_identical_to_seed_twin": all(
            v.get("metadata_identical_to_seed_twin") for v in g4.values()),
        "all_binners_identical_to_seed_twin": all(
            v.get("binner_identical_to_seed_twin") for v in g4.values()),
        "all_layers_minus_one": all(v["layer"] == -1 for v in g4.values()),
    }

    # ------------------------------------------------- gate 1: the oracle arm is C26's
    g1: dict = {}
    for prop in sweeps:
        if prop not in c26:
            continue
        grid = sweeps[prop]["grid"]
        cells = {}
        worst_h = worst_t = 0.0
        for s in SEEDS:
            for n in grid:
                a = sweeps[prop]["per_seed"][s]["arms"]["oracle_selected"][str(n)]
                b = c26[prop]["per_seed"][s]["rows"][str(n)]
                rh = a["hit_rate"] - b["hit_rate"]
                rt = (a["compute"]["tokens_per_molecule_actual"]
                      - b["compute"]["tokens_per_molecule_actual"])
                cells[f"{s}/N={n}"] = {"c27": a["hit_rate"], "c26": b["hit_rate"],
                                       "hit_rate_residual": rh, "token_residual": rt}
                worst_h = max(worst_h, abs(rh))
                worst_t = max(worst_t, abs(rt))
        g1[prop] = {"max_abs_hit_rate_residual": worst_h,
                    "max_abs_token_residual": worst_t,
                    "n_cells": len(cells), "cells": cells}
    report["validity_gates"]["gate_1_oracle_arm_reproduces_c26"] = {
        "rule": ("`oracle_selected` must reproduce outputs/c26_nsweep_<prop>/"
                 "n_sweep_metrics.json exactly at every grid point and seed"),
        "per_property": g1,
        "max_abs_hit_rate_residual": max((v["max_abs_hit_rate_residual"] for v in g1.values()),
                                         default=None),
        "max_abs_token_residual": max((v["max_abs_token_residual"] for v in g1.values()),
                                      default=None),
        "passes": all(v["max_abs_hit_rate_residual"] == 0.0
                      and v["max_abs_token_residual"] == 0.0 for v in g1.values()),
    }

    # ------------------------------------------------------------- gate 2: head AUROC
    g2 = {}
    for prop in sweeps:
        per = {s: {"terminal": sweeps[prop]["per_seed"][s]["head_auroc_terminal_position"],
                   "at_75pct": sweeps[prop]["per_seed"][s]["head_auroc_75pct_position"],
                   "pool_true_hit_rate": sweeps[prop]["per_seed"][s]["pool_true_hit_rate"]}
               for s in SEEDS}
        term = [per[s]["terminal"] for s in SEEDS]
        p75 = [per[s]["at_75pct"] for s in SEEDS]
        g2[prop] = {
            "per_seed": per,
            "terminal_mean": float(np.mean(term)), "terminal_min": float(np.min(term)),
            "at_75pct_mean": float(np.mean(p75)), "at_75pct_min": float(np.min(p75)),
            "heads_json_pooled_test_auroc": sweeps[prop]["head"].get(
                "heads_json_test_target_auroc"),
            "near_chance": bool(np.min(term) < 0.55),
        }
    report["validity_gates"]["gate_2_head_pool_auroc"] = {
        "rule": ("AUROC of the head's terminal-position P(y_final in I) for discriminating "
                 "true hits among the 16,384 pool molecules; near 0.5 means the arm measures "
                 "nothing and the section must say so"),
        "per_property": g2,
        "any_near_chance": any(v["near_chance"] for v in g2.values()),
    }

    # ------------------------------------------------------- gate 3: N=1 is an identity
    g3 = {}
    for prop in sweeps:
        cells = {}
        worst = 0.0
        for s in SEEDS:
            o = sweeps[prop]["per_seed"][s]["arms"]["oracle_selected"]["1"]["hit_rate"]
            for arm in ARMS[1:]:
                h = sweeps[prop]["per_seed"][s]["arms"][arm]["1"]["hit_rate"]
                cells[f"{s}/{arm}"] = {"oracle": o, "arm": h, "residual": h - o}
                worst = max(worst, abs(h - o))
        g3[prop] = {"max_abs_residual": worst, "cells": cells}
    report["validity_gates"]["gate_3_n1_identical_across_arms"] = {
        "rule": "at N=1 there is one candidate and nothing to select; all arms must agree",
        "per_property": g3,
        "max_abs_residual": max((v["max_abs_residual"] for v in g3.values()), default=None),
        "passes": all(v["max_abs_residual"] == 0.0 for v in g3.values()),
    }

    # ----------------------------------------------- gate 5: token identity across arms
    g5 = {}
    for prop in sweeps:
        grid = sweeps[prop]["grid"]
        worst = 0.0
        for n in grid:
            base = sweeps[prop]["curves"]["oracle_selected"][str(n)][
                "tokens_per_molecule_actual"]
            for arm in ARMS[1:]:
                worst = max(worst, abs(
                    sweeps[prop]["curves"][arm][str(n)]["tokens_per_molecule_actual"] - base))
        g5[prop] = {"max_abs_token_residual": worst,
                    "tokens_per_molecule_at_n1": sweeps[prop]["curves"]["oracle_selected"]
                                                 ["1"]["tokens_per_molecule_actual"],
                    "tokens_per_molecule_at_n32": sweeps[prop]["curves"]["oracle_selected"]
                                                  ["32"]["tokens_per_molecule_actual"]}
    report["validity_gates"]["gate_5_token_identity_across_arms"] = {
        "rule": "all arms select from one pool, so tokens per returned molecule are identical",
        "per_property": g5,
        "passes": all(v["max_abs_token_residual"] == 0.0 for v in g5.values()),
    }

    # ------------------------------------------------------------------ curves and E1/E2
    e1_violations: list[dict] = []
    e4: dict = {}
    preds: dict = {}
    for prop in sweeps:
        sw = sweeps[prop]
        grid = sw["grid"]
        toks = [sw["curves"]["oracle_selected"][str(n)]["tokens_per_molecule_actual"]
                for n in grid]
        arm_hits = {a: [sw["curves"][a][str(n)]["hit_rate_mean"] for n in grid] for a in ARMS}

        # E2 -- the price of not having ground truth
        gaps = [arm_hits["oracle_selected"][i] - arm_hits["head_selected"][i]
                for i in range(len(grid))]
        i9, i32 = grid.index(9), grid.index(32)
        e2 = {
            "gap_per_n": {str(n): gaps[i] for i, n in enumerate(grid)},
            "gap_at_n9": gaps[i9],
            "gap_at_n32": gaps[i32],
            "max_gap": float(max(gaps)),
            "max_gap_at_n": int(grid[int(np.argmax(gaps))]),
            "effective_n_of_head_selection_at_oracle_n9": effective_n(
                grid, arm_hits["head_selected"], arm_hits["oracle_selected"][i9]),
            "oracle_hit_rate_at_n9": arm_hits["oracle_selected"][i9],
        }

        # E3 -- degeneracy
        gain = arm_hits["head_selected"][i32] - arm_hits["head_selected"][0]
        e3 = {"head_selected_at_n1": arm_hits["head_selected"][0],
              "head_selected_at_n32": arm_hits["head_selected"][i32],
              "gain_n1_to_n32": gain,
              "threshold": E3_DEGENERACY_THRESHOLD,
              "degenerate": bool(gain < E3_DEGENERACY_THRESHOLD)}

        # E1 -- every guidance arm priced against the HEAD-selected curve
        rows = []
        for p in guidance_points(prop):
            bud = p["tokens_per_molecule_actual"]
            hh, i_lo, i_hi, extrap = interp(toks, arm_hits["head_selected"], bud)
            oh, *_ = interp(toks, arm_hits["oracle_selected"], bud)
            adv_head = p["hit_rate"] - hh
            per_seed_adv = []
            for s in SEEDS:
                sh = p["per_seed"].get(s)
                if sh is None:
                    continue
                s_hits = [sw["per_seed"][s]["arms"]["head_selected"][str(n)]["hit_rate"]
                          for n in grid]
                sb, *_ = interp(toks, s_hits, sh["tokens_per_molecule_actual"])
                per_seed_adv.append(sh["hit_rate"] - sb)
            row = {
                "run": p["run"], "family": p["family"], "lam": p["lam"], "layer": p["layer"],
                "guided_hit_rate": p["hit_rate"],
                "tokens_per_molecule_actual": bud,
                "head_selected_interpolated_hit_rate": hh,
                "oracle_selected_interpolated_hit_rate": oh,
                "bracketing_n": [grid[i_lo], grid[i_hi]],
                "extrapolated_beyond_grid": extrap,
                "advantage_vs_head_selected": adv_head,
                "advantage_vs_oracle_selected": p["hit_rate"] - oh,
                "advantage_vs_head_selected_per_seed": per_seed_adv,
                "validity": p["validity"],
            }
            if len(per_seed_adv) == len(SEEDS):
                row["advantage_seed_t_interval"] = t_interval(per_seed_adv)
            if adv_head > 0:
                e1_violations.append({"property": prop, **row})
            rows.append(row)
            if p["run"] == f"{DATASET}_guided_{prop}":
                e4[prop] = {**row, "note": "the deployed lambda=1 arm, C27.0.6 E4"}

        # prediction 1 / 6: arm ordering at every N >= 2
        p1 = all(arm_hits["head_selected"][i] > arm_hits["head_selected"][0]
                 and arm_hits["head_selected"][i] <= arm_hits["oracle_selected"][i] + 1e-12
                 for i in range(1, len(grid)))
        p6 = all(arm_hits["head_selected"][0]
                 < arm_hits["head_selected_at_75pct"][i] < arm_hits["head_selected"][i]
                 for i in range(1, len(grid)))
        preds[prop] = {
            "p1_head_between_base_and_oracle_for_all_n_ge_2": bool(p1),
            "p6_75pct_arm_strictly_between_base_and_terminal_arm": bool(p6),
            "p5_pool_auroc_exceeds_pooled_heldout": bool(
                g2[prop]["terminal_min"] > (g2[prop]["heads_json_pooled_test_auroc"] or 1.0)),
            "heads_json_pooled_test_auroc": g2[prop]["heads_json_pooled_test_auroc"],
            "terminal_pool_auroc_min": g2[prop]["terminal_min"],
        }

        report["properties"][prop] = {
            "grid": grid,
            "tokens_per_molecule_actual": toks,
            "curves": sw["curves"],
            "E2_price_of_ground_truth": e2,
            "E3_degeneracy": e3,
            "guidance_points_vs_head_selected": rows,
            "n_guidance_arms": len(rows),
            "n_arms_above_head_selected_curve": sum(
                1 for r in rows if r["advantage_vs_head_selected"] > 0),
            "n_arms_above_oracle_selected_curve": sum(
                1 for r in rows if r["advantage_vs_oracle_selected"] > 0),
        }

        # -------- S1: the pessimistic accounting that charges head scoring in full
        rec = sw["head_scoring_recompute_tokens_per_pool_molecule_mean"]
        s1_toks = [toks[i] + grid[i] * rec for i in range(len(grid))]
        s1_rows = []
        for r in rows:
            hh, i_lo, i_hi, extrap = interp(s1_toks, arm_hits["head_selected"],
                                            r["tokens_per_molecule_actual"])
            s1_rows.append({"run": r["run"],
                            "tokens_per_molecule_actual": r["tokens_per_molecule_actual"],
                            "head_selected_interpolated_hit_rate": hh,
                            "advantage_vs_head_selected": r["guided_hit_rate"] - hh,
                            "bracketing_n": [grid[i_lo], grid[i_hi]],
                            "extrapolated_beyond_grid": extrap})
        report["sensitivity_S1_pessimistic_accounting"][prop] = {
            "recompute_tokens_per_pool_molecule": rec,
            "tokens_per_molecule_charged": s1_toks,
            "n_arms_above_head_selected_curve": sum(
                1 for r in s1_rows if r["advantage_vs_head_selected"] > 0),
            "arms": s1_rows,
        }

    # ----------------------------------------------------------------- decision rules
    report["decision_rules"] = {
        "E1_head_selection_still_beats_steering": {
            "rule": ("upheld iff no measured guidance arm sits above the HEAD-selected "
                     "best-of-N curve interpolated at its own budget, on all three anchors"),
            "n_arms_total": sum(report["properties"][p]["n_guidance_arms"] for p in sweeps),
            "n_arms_above": len(e1_violations),
            "upheld": len(e1_violations) == 0,
            "violations": sorted(e1_violations,
                                 key=lambda r: -r["advantage_vs_head_selected"]),
        },
        "E2_price_of_ground_truth": {
            p: report["properties"][p]["E2_price_of_ground_truth"] for p in sweeps},
        "E3_degeneracy_check": {
            p: report["properties"][p]["E3_degeneracy"] for p in sweeps},
        "E4_deployed_lambda1_arm_vs_head_selected_curve": e4,
    }
    report["predictions"] = preds

    write_json(out_dir / "c27_metrics.json", report)
    write_run_context(out_dir, {
        "cli": vars(args),
        "reads": {
            "c27_sweeps": [f"c27_headsel_{p}" for p in sweeps],
            "c26_curves": [f"c26_nsweep_{p}" for p in c26],
            "frontier_machinery": "scripts/21_summarise_c26.py "
                                  "(guidance_points, interp, t_interval)",
        },
        "generates": "nothing -- no molecule is sampled and no head is trained",
    })
    print(json.dumps({
        "gate_1_max_abs_hit_residual":
            report["validity_gates"]["gate_1_oracle_arm_reproduces_c26"]
            ["max_abs_hit_rate_residual"],
        "gate_2_auroc": {p: report["validity_gates"]["gate_2_head_pool_auroc"]
                         ["per_property"][p]["terminal_mean"] for p in sweeps},
        "gate_3_max_abs_residual":
            report["validity_gates"]["gate_3_n1_identical_across_arms"]["max_abs_residual"],
        "E1": {"n_above": len(e1_violations),
               "upheld": len(e1_violations) == 0},
        "E4": {p: e4[p]["advantage_vs_head_selected"] for p in e4},
    }, indent=1))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
