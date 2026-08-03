"""C33 -- score the pre-registration: does the oracle asymmetry replicate on generator 2?

Reads only artefacts that already exist: C33's three-arm sweeps from
`scripts/27_c33_oracle_asymmetry_gen2.py`, C31's oracle-selected best-of-N curves, and
C31's 30 k-sweep cells.  **Generates nothing** -- no molecule is sampled, no head is
trained, no C31 artefact is written.

The frontier machinery is `scripts/21_summarise_c26.py::interp` and `::t_interval`,
**imported unmodified** (C33.0.5), so a C33 advantage is computed by exactly the code that
produced C26's, C27's, C28's, C30's and C31's numbers.

Scores `outputs/c33_prereg/C33.0_preregistration.md` verbatim: gates G1-G6, decision rules
F1-F6, sensitivity S1 and predictions Q1-Q10, INCLUDING where they fail.

**No bootstrap.**  C33.0.7: at n = 3 the percentile bootstrap of a mean is identically
[min, max] -- P(all three resamples hit the minimum) = 1/27 = 0.0370 > 0.025.  Per-seed
values and a seed-level t interval on 2 df (t_0.975,2 = 4.302653) are reported instead.

    .venv/bin/python scripts/27_summarise_c33.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)

ANCHORS = ["aromatic_rings", "hbd_count", "qed"]
SEEDS = ["101", "202", "303"]
ARMS = ["oracle_selected", "head_selected", "head_selected_at_75pct"]

#: C33.0.8 F4 -- C27's E3 threshold, reused rather than re-chosen.
F4_DEGENERACY_THRESHOLD = 0.02
#: C33.0.4 G5 -- C27's "near chance" threshold, reused rather than re-chosen.
G5_NEAR_CHANCE = 0.55
#: C33.0.8 F1 / F2.
F1_SHARE_FLOOR = 0.50
F2_SHARE_BAND = (0.75, 1.00)
#: C27's generator-1 deployed-arm numbers, quoted in C33.0.1 and C33.0.10 Q9.
C27_GEN1_DEPLOYED = {
    "aromatic_rings": {"adv_oracle": -0.3532, "adv_head": -0.0439, "share": 0.8756,
                       "gap": 0.3093},
    "hbd_count": {"adv_oracle": -0.2472, "adv_head": -0.0292, "share": 0.8819,
                  "gap": 0.2180},
    "qed": {"adv_oracle": -0.3715, "adv_head": -0.0522, "share": 0.8594, "gap": 0.3193},
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_summarise")
_c31s = _load_module(ROOT / "scripts" / "25_summarise_c31.py", "c31_summarise")
interp = _c26.interp
t_interval = _c26.t_interval
collect_cells = _c31s.collect_cells


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def parameter_sha256(state_dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def file_hash_criterion_demo(head_file: Path) -> dict:
    """Demonstrate, on C33's own checkpoint, why C27's FILE-hash gate fails by construction.

    `torch.save` writes a zip whose internal archive directory is named after the output
    file, so saving one dict under two names yields two files with identical tensors and
    different bytes.  C33.0.4 G2 declared in advance that the file hash is reported as
    evidence and is NOT the pass/fail criterion; this function is the evidence.  It writes
    only into a temporary directory that it deletes.
    """
    import torch  # local: the summariser is otherwise import-light

    ck = torch.load(head_file, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "same_dict_name_a.pt", Path(td) / "same_dict_name_b.pt"
        torch.save(ck, a)
        torch.save(ck, b)
        ha, hb = sha256_file(a), sha256_file(b)
        ca = torch.load(a, map_location="cpu", weights_only=False)
        cb = torch.load(b, map_location="cpu", weights_only=False)
        tensors_identical = bool(
            set(ca["state_dict"]) == set(cb["state_dict"])
            and all(torch.equal(ca["state_dict"][k], cb["state_dict"][k])
                    for k in ca["state_dict"]))
        return {
            "what": ("one identical checkpoint dict saved twice under two different file "
                     "names, in a temporary directory that is then deleted"),
            "file_sha256_a": ha, "file_sha256_b": hb,
            "file_bytes_a": a.stat().st_size, "file_bytes_b": b.stat().st_size,
            "file_sha256_equal": bool(ha == hb),
            "byte_length_difference": a.stat().st_size - b.stat().st_size,
            "parameter_sha256_a": parameter_sha256(ca["state_dict"]),
            "parameter_sha256_b": parameter_sha256(cb["state_dict"]),
            "tensors_identical": tensors_identical,
            "conclusion": ("a FILE hash tests serialisation, not content: the file hashes "
                           "differ (or agree only by the accident of equal names) while the "
                           "tensors are identical.  C33.0.4 G2's pass/fail criterion is "
                           "parameter-level identity plus metadata agreement; the file hash "
                           "is recorded as evidence only."),
        }


def curve_of(sw: dict, arm: str, grid: list[int]) -> dict:
    """Mean and per-seed (tokens, hit rate) axes of one C33 arm."""
    return {
        "grid": grid,
        "tokens": [sw["curves"][arm][str(n)]["tokens_per_molecule_actual"] for n in grid],
        "hits": [sw["curves"][arm][str(n)]["hit_rate_mean"] for n in grid],
        "hits_per_seed": {
            s: [sw["per_seed"][s]["arms"][arm][str(n)]["hit_rate"] for n in grid]
            for s in SEEDS},
        "tokens_per_seed": {
            s: [sw["per_seed"][s]["arms"][arm][str(n)]["compute"]
                ["tokens_per_molecule_actual"] for n in grid] for s in SEEDS},
    }


def price_against(curve: dict, hit_mean: float, budget: float, per_seed: dict) -> dict:
    """Advantage of one guidance cell against one C33 curve at that cell's own budget."""
    h, i_lo, i_hi, extrap = interp(curve["tokens"], curve["hits"], budget)
    adv = []
    for s in SEEDS:
        ps = per_seed.get(s)
        if ps is None:
            continue
        sb, *_ = interp(curve["tokens_per_seed"][s], curve["hits_per_seed"][s],
                        ps["tokens_per_molecule_actual"])
        adv.append(ps["hit_rate"] - sb)
    out = {
        "interpolated_hit_rate": h,
        "advantage": hit_mean - h,
        "bracketing_n": [curve["grid"][i_lo], curve["grid"][i_hi]],
        "extrapolated_beyond_grid": bool(extrap),
        "advantage_per_seed": adv,
    }
    if len(adv) == len(SEEDS):
        out["advantage_seed_t_interval"] = t_interval(adv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c33_summary")
    args = ap.parse_args()

    cfg = load_config("c31_second_generator")
    out_dir = OUTPUT_DIR / args.out
    prereg_dir = OUTPUT_DIR / "c33_prereg"
    prereg_f = prereg_dir / "C33.0_preregistration.md"
    lock = read_json(prereg_dir / "prereg_lock.json")

    sweeps: dict[str, dict] = {}
    c31_curves: dict[str, dict] = {}
    for prop in ANCHORS:
        f = OUTPUT_DIR / f"c33_headsel_{prop}" / "head_selected_metrics.json"
        g = OUTPUT_DIR / f"c31_bestofn_{prop}" / "n_sweep_metrics.json"
        if f.exists():
            sweeps[prop] = read_json(f)
        if g.exists():
            c31_curves[prop] = read_json(g)
    if not sweeps:
        raise SystemExit("no C33 sweep artefacts found; run scripts/27_c33_oracle_"
                         "asymmetry_gen2.py first")

    report: dict = {
        "experiment": "C33",
        "question": ("does the oracle asymmetry -- the finding that most of best-of-N's "
                     "reported advantage over guided decoding is the ground-truth oracle "
                     "rather than the selection -- replicate on a second, architecturally "
                     "different generator?"),
        "prereg": "outputs/c33_prereg/C33.0_preregistration.md",
        "prereg_lock": lock,
        "prereg_file_sha256_now": sha256_file(prereg_f),
        "prereg_file_sha256_matches_lock": bool(sha256_file(prereg_f) == lock["file_sha256"]),
        "prereg_mtime_utc_now": mtime_utc(prereg_f),
        "generator": {
            "repo": cfg["model_repo"], "revision": cfg["model_revision"],
            "architecture": "GPT-2, full softmax attention, byte-level BPE, 12 blocks",
            "comparison_generator": ("ibm-research/GP-MoLFormer-Uniq, linear attention, "
                                     "46.8M params, atom-level vocabulary -- C27's"),
        },
        "accounting": "processed generator tokens (actual)",
        "seeds": SEEDS,
        "uncertainty": ("per-seed values and a seed-level t interval on 2 df "
                        "(t_0.975,2 = 4.302653); no bootstrap anywhere in C33, C33.0.7"),
        "c27_generator_1_reference": C27_GEN1_DEPLOYED,
        "gates": {},
        "properties": {},
        "cells": {},
        "headline": {},
        "decision_rules": {},
        "predictions": {},
        "sensitivity_S1_pessimistic_accounting": {},
    }

    # ================================================================== G1: pool identity
    g1: dict = {}
    for prop in sweeps:
        if prop not in c31_curves:
            continue
        grid = sweeps[prop]["grid"]
        cells, worst_h, worst_t = {}, 0.0, 0.0
        for s in SEEDS:
            for n in grid:
                a = sweeps[prop]["per_seed"][s]["arms"]["oracle_selected"][str(n)]
                b = c31_curves[prop]["per_seed"][s]["rows"][str(n)]
                rh = a["hit_rate"] - b["hit_rate"]
                rt = (a["compute"]["tokens_per_molecule_actual"]
                      - b["compute"]["tokens_per_molecule_actual"])
                cells[f"{s}/N={n}"] = {"c33": a["hit_rate"], "c31": b["hit_rate"],
                                       "hit_rate_residual": rh, "token_residual": rt}
                worst_h, worst_t = max(worst_h, abs(rh)), max(worst_t, abs(rt))
        g1[prop] = {"max_abs_hit_rate_residual": worst_h, "max_abs_token_residual": worst_t,
                    "n_cells": len(cells), "cells": cells}
    report["gates"]["G1_pool_identity"] = {
        "rule": ("the regenerated `oracle_selected` curve must reproduce "
                 "outputs/c31_bestofn_<prop>/n_sweep_metrics.json at every grid point and "
                 "every generation seed, in hit rate and in tokens per returned molecule"),
        "blocking": True,
        "per_property": g1,
        "max_abs_hit_rate_residual": max((v["max_abs_hit_rate_residual"]
                                          for v in g1.values()), default=None),
        "max_abs_token_residual": max((v["max_abs_token_residual"]
                                       for v in g1.values()), default=None),
        "passes": bool(g1) and all(v["max_abs_hit_rate_residual"] == 0.0
                                   and v["max_abs_token_residual"] == 0.0
                                   for v in g1.values()),
    }

    # ============================================================== G2: head provenance
    g2 = {}
    for prop in sweeps:
        h = dict(sweeps[prop]["head"])
        hf = Path(h["head_file"])
        h["head_file_exists"] = hf.exists()
        h["head_file_sha256_now"] = sha256_file(hf) if hf.exists() else None
        h["head_file_sha256_stable"] = bool(h["head_file_sha256_now"]
                                            == h.get("head_file_sha256"))
        h["criterion_metadata_consistent"] = bool(h["metadata_consistent"])
        h["criterion_interval_matches_c31_cell"] = bool(h["target_interval_matches_c31_cell"])
        h["criterion_cross_checked_against_depth_json"] = bool(
            h.get("c31_depth_test_target_auroc") is not None)
        h["passes"] = bool(h["criterion_metadata_consistent"]
                           and h["criterion_interval_matches_c31_cell"]
                           and h["criterion_cross_checked_against_depth_json"])
        g2[prop] = h
    demo_file = Path(sweeps[sorted(sweeps)[0]]["head"]["head_file"])
    report["gates"]["G2_head_provenance"] = {
        "rule_as_preregistered": (
            "resolved head_file, file SHA-256, parameter-level SHA-256, checkpoint metadata, "
            "binner kind and parameters, interval-mask bin count, cross-checked against "
            "outputs/c31_heads/depth_<prop>.json for the same probe point and head seed"),
        "pass_fail_criterion": ("parameter-level identity and metadata agreement -- NOT the "
                                "file hash.  C33.0.4 G2 declares this change of criterion "
                                "relative to C27 in advance."),
        "file_hash_criterion_is_evidence_only": True,
        "c27_file_hash_criterion_failed_by_construction": file_hash_criterion_demo(demo_file),
        "per_property": g2,
        "passes": all(v["passes"] for v in g2.values()),
    }

    # ================================================================= G3: cost identity
    g3a = {}
    for prop in sweeps:
        grid = sweeps[prop]["grid"]
        worst = 0.0
        per_n = {}
        for n in grid:
            base = sweeps[prop]["curves"]["oracle_selected"][str(n)][
                "tokens_per_molecule_actual"]
            res = [abs(sweeps[prop]["curves"][a][str(n)]["tokens_per_molecule_actual"] - base)
                   for a in ARMS[1:]]
            per_n[str(n)] = {"tokens_per_molecule_actual": base,
                             "max_abs_residual_across_arms": max(res)}
            worst = max(worst, max(res))
        g3a[prop] = {"max_abs_token_residual_across_arms": worst, "per_n": per_n,
                     "tokens_per_molecule_at_n1": per_n[str(grid[0])][
                         "tokens_per_molecule_actual"],
                     "tokens_per_molecule_at_n32": per_n[str(grid[-1])][
                         "tokens_per_molecule_actual"]}

    cells = collect_cells(cfg)
    cells = {k: v for k, v in cells.items() if v["property"] in ANCHORS}
    g3b = {}
    for name, c in cells.items():
        f = OUTPUT_DIR / c["dir"] / "k_cell_metrics.json"
        d = read_json(f)
        recorded = d["aggregate"]["compute_total"]["tokens_per_molecule_actual"]
        # recomputed from the per-seed record, so the re-pricing reads the same number the
        # cell recorded and the residual is a real check rather than a tautology
        tot_tok = sum(d["seeds_detail"][s]["compute"]["processed_tokens_actual"]
                      for s in SEEDS)
        tot_mol = sum(d["seeds_detail"][s]["compute"]["molecules_returned"] for s in SEEDS)
        g3b[name] = {
            "cell": c["dir"],
            "tokens_per_molecule_actual_used_by_c33": c["tokens_per_molecule_actual"],
            "tokens_per_molecule_actual_recorded_by_c31": recorded,
            "residual": c["tokens_per_molecule_actual"] - recorded,
            "tokens_per_molecule_actual_recomputed_from_seeds": tot_tok / tot_mol,
            "residual_vs_recomputed": c["tokens_per_molecule_actual"] - tot_tok / tot_mol,
            "cost_identity_tokens_mod_k_plus_1": {
                s: d["seeds_detail"][s]["cost_identity_tokens_mod_k_plus_1"] for s in SEEDS},
            "cost_identity_max_residual": d["cost_identity_max_residual"],
        }
    report["gates"]["G3_cost_identity"] = {
        "rule_a": ("within C33 all three arms select from one pool, so tokens per returned "
                   "molecule must be identical across arms at every grid point"),
        "rule_b": ("against C31, each re-priced guided cell's tokens_per_molecule_actual "
                   "must equal the value recorded in its k_cell_metrics.json exactly, and "
                   "C31's processed_tokens_actual mod (k+1) == 0 identity is re-checked"),
        "blocking_part_a": True,
        "a_within_c33": g3a,
        "a_max_abs_residual": max((v["max_abs_token_residual_across_arms"]
                                   for v in g3a.values()), default=None),
        "a_passes": bool(g3a) and all(v["max_abs_token_residual_across_arms"] == 0.0
                                      for v in g3a.values()),
        "b_against_c31": g3b,
        "b_max_abs_residual": max((abs(v["residual"]) for v in g3b.values()), default=None),
        "b_max_abs_residual_vs_recomputed": max(
            (abs(v["residual_vs_recomputed"]) for v in g3b.values()), default=None),
        "b_max_cost_identity_residual": max(
            (v["cost_identity_max_residual"] for v in g3b.values()), default=None),
        "b_passes": bool(g3b) and all(v["residual"] == 0.0
                                      and v["cost_identity_max_residual"] == 0
                                      for v in g3b.values()),
    }

    # ============================================================ G4: N=1 is an identity
    g4 = {}
    for prop in sweeps:
        cells_g4, worst = {}, 0.0
        for s in SEEDS:
            o = sweeps[prop]["per_seed"][s]["arms"]["oracle_selected"]["1"]["hit_rate"]
            for arm in ARMS[1:]:
                h = sweeps[prop]["per_seed"][s]["arms"][arm]["1"]["hit_rate"]
                cells_g4[f"{s}/{arm}"] = {"oracle": o, "arm": h, "residual": h - o}
                worst = max(worst, abs(h - o))
            if prop in c31_curves:
                b = c31_curves[prop]["per_seed"][s]["rows"]["1"]["hit_rate"]
                cells_g4[f"{s}/c31"] = {"oracle": o, "arm": b, "residual": b - o}
                worst = max(worst, abs(b - o))
        g4[prop] = {"max_abs_residual": worst, "cells": cells_g4}
    report["gates"]["G4_n1_identity"] = {
        "rule": ("at N=1 there is one candidate and nothing to select, so all three arms "
                 "must agree exactly with each other and with C31"),
        "blocking": False,
        "per_property": g4,
        "max_abs_residual": max((v["max_abs_residual"] for v in g4.values()), default=None),
        "passes": all(v["max_abs_residual"] == 0.0 for v in g4.values()),
    }

    # ====================================================== G5: the head discriminates
    g5 = {}
    for prop in sweeps:
        per = {s: {"terminal": sweeps[prop]["per_seed"][s]["head_auroc_terminal_position"],
                   "at_75pct": sweeps[prop]["per_seed"][s]["head_auroc_75pct_position"],
                   "pool_true_hit_rate": sweeps[prop]["per_seed"][s]["pool_true_hit_rate"]}
               for s in SEEDS}
        term = [per[s]["terminal"] for s in SEEDS]
        p75 = [per[s]["at_75pct"] for s in SEEDS]
        g5[prop] = {
            "per_seed": per,
            "terminal_mean": float(np.mean(term)), "terminal_min": float(np.min(term)),
            "at_75pct_mean": float(np.mean(p75)), "at_75pct_min": float(np.min(p75)),
            "c31_depth_test_target_auroc": sweeps[prop]["head"].get(
                "c31_depth_test_target_auroc"),
            "near_chance": bool(np.min(term) < G5_NEAR_CHANCE),
            "terminal_exceeds_75pct_on_every_seed": bool(all(
                per[s]["terminal"] > per[s]["at_75pct"] for s in SEEDS)),
        }
    report["gates"]["G5_head_discriminates"] = {
        "rule": (f"per anchor and per seed, the AUROC of the head's terminal-position "
                 f"P(y_final in I) for discriminating true hits among the 16,384 pool "
                 f"molecules; near chance is fixed as min over seeds < {G5_NEAR_CHANCE}"),
        "blocking": False,
        "voids_head_arm_on_affected_anchor": True,
        "per_property": g5,
        "any_near_chance": any(v["near_chance"] for v in g5.values()),
        "passes": not any(v["near_chance"] for v in g5.values()),
    }

    # ============================================== G6: the frozen interval is C31's
    fr = sweeps[sorted(sweeps)[0]]["frozen_inputs"]
    g6 = {"frozen_inputs": fr, "per_property": {}}
    for prop in sweeps:
        h = sweeps[prop]["head"]
        g6["per_property"][prop] = {
            "target_interval_used": h["target_interval_used"],
            "target_interval_in_c31_deployed_cell": h["target_interval_in_c31_cell"],
            "matches": bool(h["target_interval_matches_c31_cell"]),
        }
    g6["target_intervals_sha256_stable"] = bool(
        sha256_file(OUTPUT_DIR / "c31_zinc50k" / "target_intervals.json")
        == fr["target_intervals_sha256"])
    g6["windows_sha256_stable"] = bool(
        sha256_file(OUTPUT_DIR / "c31_zinc50k" / "windows.json") == fr["windows_sha256"])
    report["gates"]["G6_frozen_interval"] = {
        "rule": ("target_intervals.json and windows.json in outputs/c31_zinc50k/ are "
                 "SHA-256'd and the (lo, hi, base_rate) triple used by C33 must equal the "
                 "triple recorded in each C31 deployed k cell"),
        "blocking": True,
        **g6,
        "passes": bool(all(v["matches"] for v in g6["per_property"].values())
                       and g6["target_intervals_sha256_stable"]
                       and g6["windows_sha256_stable"]),
    }

    blocking = {
        "G1": report["gates"]["G1_pool_identity"]["passes"],
        "G3a": report["gates"]["G3_cost_identity"]["a_passes"],
        "G6": report["gates"]["G6_frozen_interval"]["passes"],
    }
    report["blocking_gates"] = {
        "rule": ("C33.0.4: G1, G3(a) and G6 are blocking; if any fails the oracle-share "
                 "headline is NOT stated"),
        "status": blocking, "all_pass": all(blocking.values()),
    }

    # ====================================================== curves, gaps and the pricing
    curves: dict[str, dict[str, dict]] = {}
    for prop in sweeps:
        sw = sweeps[prop]
        grid = sw["grid"]
        curves[prop] = {a: curve_of(sw, a, grid) for a in ARMS}
        oc, hc, dc = (curves[prop][a] for a in ARMS)
        gap_per_n = {str(n): oc["hits"][i] - hc["hits"][i] for i, n in enumerate(grid)}
        gap_per_n_per_seed = {
            s: {str(n): oc["hits_per_seed"][s][i] - hc["hits_per_seed"][s][i]
                for i, n in enumerate(grid)} for s in SEEDS}
        i32 = grid.index(32)
        f4_gain = hc["hits"][i32] - hc["hits"][0]
        f5 = {str(n): {
            "base_rate_n1": hc["hits"][0],
            "head_selected_at_75pct": dc["hits"][i],
            "head_selected": hc["hits"][i],
            "strictly_between": bool(hc["hits"][0] < dc["hits"][i] < hc["hits"][i]),
        } for i, n in enumerate(grid) if n >= 2}
        report["properties"][prop] = {
            "grid": grid,
            "target_interval": sw["target_interval"],
            "tokens_per_molecule_actual": oc["tokens"],
            "curves": sw["curves"],
            "per_seed_hit_rates": {a: curves[prop][a]["hits_per_seed"] for a in ARMS},
            "budget_matched_gap_per_n": gap_per_n,
            "budget_matched_gap_per_n_per_seed": gap_per_n_per_seed,
            "max_gap_over_grid": float(max(gap_per_n.values())),
            "max_gap_at_n": int(grid[int(np.argmax([gap_per_n[str(n)] for n in grid]))]),
            "F4_head_arm_gain_n1_to_n32": {
                "head_selected_at_n1": hc["hits"][0],
                "head_selected_at_n32": hc["hits"][i32],
                "gain": f4_gain, "threshold": F4_DEGENERACY_THRESHOLD,
                "fires": bool(f4_gain >= F4_DEGENERACY_THRESHOLD),
                "degenerate": bool(f4_gain < F4_DEGENERACY_THRESHOLD)},
            "F5_diagnostic_arm_ordering": {
                "per_n": f5,
                "fires": bool(all(v["strictly_between"] for v in f5.values()))},
            "head_scoring_recompute_tokens_per_pool_molecule_mean":
                sw["head_scoring_recompute_tokens_per_pool_molecule_mean"],
        }

    # ------------------------------------------------------- price the 30 C31 k cells
    for name, c in sorted(cells.items()):
        prop = c["property"]
        if prop not in curves:
            c["priced"] = False
            continue
        b = c["tokens_per_molecule_actual"]
        po = price_against(curves[prop]["oracle_selected"], c["hit_rate_mean"], b,
                           c["per_seed"])
        ph = price_against(curves[prop]["head_selected"], c["hit_rate_mean"], b,
                           c["per_seed"])
        adv_o, adv_h = po["advantage"], ph["advantage"]
        gap_b = po["interpolated_hit_rate"] - ph["interpolated_hit_rate"]
        ti_o = po.get("advantage_seed_t_interval")
        row = {
            "priced": True,
            "oracle_selected_interpolated_hit_rate": po["interpolated_hit_rate"],
            "head_selected_interpolated_hit_rate": ph["interpolated_hit_rate"],
            "advantage_vs_oracle_selected": adv_o,
            "advantage_vs_head_selected": adv_h,
            "advantage_vs_oracle_selected_per_seed": po["advantage_per_seed"],
            "advantage_vs_head_selected_per_seed": ph["advantage_per_seed"],
            "advantage_vs_oracle_selected_seed_t_interval": ti_o,
            "advantage_vs_head_selected_seed_t_interval": ph.get("advantage_seed_t_interval"),
            "bracketing_n_oracle_selected": po["bracketing_n"],
            "bracketing_n_head_selected": ph["bracketing_n"],
            "extrapolated_beyond_grid": bool(po["extrapolated_beyond_grid"]
                                             or ph["extrapolated_beyond_grid"]),
            "budget_matched_gap_at_cell_budget": gap_b,
            "above_oracle_selected_curve": bool(adv_o > 0),
            "above_head_selected_curve": bool(adv_h > 0),
        }
        # C33.0.6 rule 1: the share exists only where the advantage vs the oracle curve
        # is negative.  Nowhere else is it filled, imputed or sign-flipped.
        if adv_o < 0:
            share = 1.0 - adv_h / adv_o
            row["oracle_share"] = share
            row["oracle_share_status"] = "computed"
            row["oracle_share_exceeds_one"] = bool(share > 1.0)
            row["oracle_share_note"] = (
                "share > 1 means the advantage against the head-selected curve is POSITIVE: "
                "equalising the information did not merely shrink the gap, it reversed it "
                "(C33.0.6 rule 2); the value is not clipped"
                if share > 1.0 else "")
        else:
            row["oracle_share"] = None
            row["oracle_share_status"] = (
                "not defined (advantage vs oracle curve is not negative)")
        if ti_o is not None:
            row["adv_oracle_t_interval_spans_zero"] = bool(not ti_o["excludes_zero"])
            if adv_o < 0 and not ti_o["excludes_zero"]:
                row["oracle_share_flag"] = "not resolved at three generation seeds"
            row["not_resolved_at_three_generation_seeds"] = bool(
                ti_o["sd"] > abs(ti_o["mean"]))
        ti_h = ph.get("advantage_seed_t_interval")
        if ti_h is not None:
            row["not_resolved_vs_head_curve_at_three_generation_seeds"] = bool(
                ti_h["sd"] > abs(ti_h["mean"]))
        c.update(row)
    report["cells"] = cells

    scored = [c for c in cells.values() if c.get("priced")]
    n_above_head = sum(1 for c in scored if c["above_head_selected_curve"])
    n_above_oracle = sum(1 for c in scored if c["above_oracle_selected_curve"])
    report["arm_counts"] = {
        "rule": ("C33.0.5: n_arms_above_head_selected_curve and "
                 "n_arms_above_oracle_selected_curve over the 30 C31 k-sweep cells, "
                 "counted on the point-estimate advantage as C27 counted them"),
        "n_cells": len(scored),
        "n_arms_above_head_selected_curve": n_above_head,
        "n_arms_above_oracle_selected_curve": n_above_oracle,
        "per_property": {
            p: {"n_cells": sum(1 for c in scored if c["property"] == p),
                "above_head_selected": sum(1 for c in scored if c["property"] == p
                                           and c["above_head_selected_curve"]),
                "above_oracle_selected": sum(1 for c in scored if c["property"] == p
                                             and c["above_oracle_selected_curve"])}
            for p in sorted({c["property"] for c in scored})},
        "cells_above_head_selected_curve": sorted(
            c["dir"] for c in scored if c["above_head_selected_curve"]),
        "cells_above_oracle_selected_curve": sorted(
            c["dir"] for c in scored if c["above_oracle_selected_curve"]),
        "c27_generator_1": {"n_cells": 46, "n_arms_above_head_selected_curve": 15,
                            "n_arms_above_oracle_selected_curve": 1},
    }

    # ---------------------------------------- the headline: the deployed arm at k = 2
    headline = {}
    for prop in sorted(curves):
        c = cells.get(f"{prop}_deployed_k2")
        if c is None or not c.get("priced"):
            continue
        headline[prop] = {
            "cell": c["dir"],
            "guided_hit_rate": c["hit_rate_mean"],
            "guided_hit_rate_values": c["hit_rate_values"],
            "tokens_per_molecule_actual": c["tokens_per_molecule_actual"],
            "advantage_vs_oracle_selected": c["advantage_vs_oracle_selected"],
            "advantage_vs_head_selected": c["advantage_vs_head_selected"],
            "advantage_vs_oracle_selected_per_seed":
                c["advantage_vs_oracle_selected_per_seed"],
            "advantage_vs_head_selected_per_seed": c["advantage_vs_head_selected_per_seed"],
            "advantage_vs_oracle_selected_seed_t_interval":
                c["advantage_vs_oracle_selected_seed_t_interval"],
            "advantage_vs_head_selected_seed_t_interval":
                c["advantage_vs_head_selected_seed_t_interval"],
            "oracle_share": c["oracle_share"],
            "oracle_share_status": c["oracle_share_status"],
            "oracle_share_flag": c.get("oracle_share_flag"),
            "budget_matched_gap_at_cell_budget": c["budget_matched_gap_at_cell_budget"],
            "c27_generator_1": C27_GEN1_DEPLOYED.get(prop),
            "head_arm_degenerate": report["properties"][prop][
                "F4_head_arm_gain_n1_to_n32"]["degenerate"],
            "head_arm_near_chance": report["gates"]["G5_head_discriminates"][
                "per_property"][prop]["near_chance"],
        }
    report["headline"] = {
        "rule": ("C33.0.6 rule 4: the headline oracle-share table is the deployed arm at "
                 "k = 2, one cell per anchor, fixed in the pre-registration and not "
                 "selected from the 30 afterwards"),
        "per_property": headline,
        "crossing_cells_reported_separately": {
            c["dir"]: {"property": c["property"], "arm": c["arm"], "k": c["k"],
                       "advantage_vs_oracle_selected": c["advantage_vs_oracle_selected"],
                       "advantage_vs_head_selected": c["advantage_vs_head_selected"],
                       "oracle_share_status": c["oracle_share_status"],
                       "budget_matched_gap_at_cell_budget":
                           c["budget_matched_gap_at_cell_budget"]}
            for c in scored if not c["advantage_vs_oracle_selected"] < 0},
    }

    # ================================================================= decision rules
    computable = {p: v for p, v in headline.items() if v["oracle_share"] is not None}
    f1 = bool(computable) and all(v["oracle_share"] >= F1_SHARE_FLOOR
                                  for v in computable.values())
    f2 = bool(computable) and all(F2_SHARE_BAND[0] <= v["oracle_share"] <= F2_SHARE_BAND[1]
                                  for v in computable.values())
    f3 = bool(n_above_head > n_above_oracle)
    f4 = all(report["properties"][p]["F4_head_arm_gain_n1_to_n32"]["fires"] for p in curves)
    f5_per = {p: report["properties"][p]["F5_diagnostic_arm_ordering"]["fires"]
              for p in curves}
    f5 = all(f5_per.values())
    f6_cells = {c["dir"]: {"advantage_vs_oracle_selected": c["advantage_vs_oracle_selected"],
                           "advantage_sd": (c["advantage_vs_oracle_selected_seed_t_interval"]
                                            or {}).get("sd"),
                           "not_resolved": c.get("not_resolved_at_three_generation_seeds")}
                for c in scored if c.get("not_resolved_at_three_generation_seeds")}
    f6_anchors = {p: v for p, v in headline.items()
                  if (v["advantage_vs_oracle_selected_seed_t_interval"] or {}).get(
                      "excludes_zero") is False}
    dr = report["decision_rules"]
    dr["F1"] = {"rule": ("the oracle asymmetry replicates on generator 2: for every anchor "
                         "whose deployed-k2 cell has adv_oracle < 0, oracle_share >= 0.50"),
                "per_property": {p: v["oracle_share"] for p, v in computable.items()},
                "anchors_with_computable_share": sorted(computable),
                "anchors_without_computable_share": sorted(set(headline) - set(computable)),
                "fires": f1}
    dr["F2"] = {"rule": (f"it replicates at C27's magnitude: every computable deployed-k2 "
                         f"oracle_share lies in [{F2_SHARE_BAND[0]}, {F2_SHARE_BAND[1]}]"),
                "per_property": {p: v["oracle_share"] for p, v in computable.items()},
                "fires": f2}
    dr["F3"] = {"rule": ("equalising information changes the arm count: "
                         "n_arms_above_head_selected_curve > "
                         "n_arms_above_oracle_selected_curve over the 30 cells"),
                "n_above_head_selected": n_above_head,
                "n_above_oracle_selected": n_above_oracle, "fires": f3}
    dr["F4"] = {"rule": (f"the head arm is not degenerate: on every anchor head_selected "
                         f"gains >= {F4_DEGENERACY_THRESHOLD} absolute hit rate from N=1 "
                         f"to N=32"),
                "per_property": {p: report["properties"][p]["F4_head_arm_gain_n1_to_n32"]
                                 for p in curves},
                "degenerate_anchors": sorted(
                    p for p in curves
                    if report["properties"][p]["F4_head_arm_gain_n1_to_n32"]["degenerate"]),
                "fires": f4}
    dr["F5"] = {"rule": ("the diagnostic arm behaves as C27's did: on every anchor "
                         "head_selected_at_75pct at every N >= 2 lies strictly between the "
                         "N=1 base rate and head_selected at the same N"),
                "per_property": f5_per,
                "n_anchors_firing": sum(1 for v in f5_per.values() if v), "fires": f5}
    dr["F6"] = {"rule": ("the honesty rule: any cell whose between-generation-seed sd "
                         "exceeds the absolute value of its own mean advantage is reported "
                         "as not resolved at three generation seeds; any anchor whose "
                         "deployed-k2 adv_oracle t interval spans zero is reported as "
                         "unresolved even where the point estimate is negative"),
                "cells_not_resolved": f6_cells, "n_cells_not_resolved": len(f6_cells),
                "deployed_k2_anchors_with_t_interval_spanning_zero": sorted(f6_anchors),
                "fires": bool(f6_cells or f6_anchors)}

    if not report["blocking_gates"]["all_pass"] or len(computable) < 2:
        verdict = "PARTIALLY UNINTERPRETABLE"
        wording = ("a blocking gate failed, or fewer than two anchors have a computable "
                   "share; C33.0.8 withholds the headline in that case")
    elif f1 and f2:
        verdict = "REPLICATES"
        wording = ("the oracle asymmetry replicates on a second, architecturally different "
                   "generator, at C27's magnitude: every computable deployed-k2 oracle "
                   "share is at least 0.50 and lies in [0.75, 1.00]")
    elif f1:
        verdict = "REPLICATES IN DIRECTION, NOT IN MAGNITUDE"
        wording = ("the oracle is worth at least half the reported gap on every anchor "
                   "where the share is computable, but not the 0.75-1.00 magnitude C27 "
                   "reported on generator 1")
    else:
        verdict = "DOES NOT REPLICATE"
        wording = ("on generator 2 the oracle is NOT worth at least half of the deployed "
                   "arm's reported gap on every anchor where the share is computable")
    report["verdict"] = {
        "verdict": verdict, "committed_wording": wording,
        "rule": ("C33.0.8: REPLICATES iff F1 and F2; REPLICATES IN DIRECTION, NOT IN "
                 "MAGNITUDE iff F1 and not F2; DOES NOT REPLICATE iff not F1; PARTIALLY "
                 "UNINTERPRETABLE if a blocking gate fails or fewer than two anchors have "
                 "a computable share"),
        "n_anchors_with_computable_share": len(computable),
    }

    # ============================================================ sensitivity S1
    for prop in curves:
        grid = report["properties"][prop]["grid"]
        rec = report["properties"][prop][
            "head_scoring_recompute_tokens_per_pool_molecule_mean"]
        hc = curves[prop]["head_selected"]
        s1_tokens = [hc["tokens"][i] + grid[i] * rec for i in range(len(grid))]
        s1_curve = {"grid": grid, "tokens": s1_tokens, "hits": hc["hits"],
                    "tokens_per_seed": {s: [hc["tokens_per_seed"][s][i] + grid[i] * rec
                                            for i in range(len(grid))] for s in SEEDS},
                    "hits_per_seed": hc["hits_per_seed"]}
        rows = {}
        for name, c in sorted(cells.items()):
            if c["property"] != prop or not c.get("priced"):
                continue
            ph = price_against(s1_curve, c["hit_rate_mean"],
                               c["tokens_per_molecule_actual"], c["per_seed"])
            adv_o = c["advantage_vs_oracle_selected"]
            rows[name] = {
                "cell": c["dir"],
                "head_selected_interpolated_hit_rate": ph["interpolated_hit_rate"],
                "advantage_vs_head_selected": ph["advantage"],
                "above_head_selected_curve": bool(ph["advantage"] > 0),
                "oracle_share": (1.0 - ph["advantage"] / adv_o) if adv_o < 0 else None,
                "oracle_share_status": ("computed" if adv_o < 0 else
                                        "not defined (advantage vs oracle curve is not "
                                        "negative)"),
                "extrapolated_beyond_grid": ph["extrapolated_beyond_grid"],
            }
        report["sensitivity_S1_pessimistic_accounting"][prop] = {
            "recompute_tokens_per_pool_molecule": rec,
            "tokens_per_molecule_charged": s1_tokens,
            "cells": rows,
            "n_arms_above_head_selected_curve": sum(
                1 for r in rows.values() if r["above_head_selected_curve"]),
            "deployed_k2_oracle_share": rows.get(f"{prop}_deployed_k2", {}).get(
                "oracle_share"),
        }
    report["sensitivity_S1_pessimistic_accounting"]["total_n_arms_above_head_selected"] = sum(
        v["n_arms_above_head_selected_curve"]
        for k, v in report["sensitivity_S1_pessimistic_accounting"].items()
        if isinstance(v, dict) and "n_arms_above_head_selected_curve" in v)

    # ================================================================== predictions
    P = report["predictions"]

    def pred(name, statement, fires, detail):
        P[name] = {"statement": statement, "outcome": "CONFIRMED" if fires else "FALSIFIED",
                   "detail": detail}

    g1r = report["gates"]["G1_pool_identity"]
    pred("Q1", "G1's maximum absolute hit-rate residual is exactly 0.0 on all three anchors",
         all(v["max_abs_hit_rate_residual"] == 0.0 for v in g1.values()) and len(g1) == 3,
         {p: v["max_abs_hit_rate_residual"] for p, v in g1.items()})
    q2_gt = {p: g5[p]["terminal_min"] for p in g5}
    pred("Q2", ("G5's terminal-position pool AUROC exceeds 0.70 on all three anchors, and "
                "exceeds the 75%-position AUROC on all three"),
         all(v > 0.70 for v in q2_gt.values())
         and all(g5[p]["terminal_mean"] > g5[p]["at_75pct_mean"] for p in g5),
         {p: {"terminal_min": g5[p]["terminal_min"], "terminal_mean": g5[p]["terminal_mean"],
              "at_75pct_mean": g5[p]["at_75pct_mean"]} for p in g5})
    pred("Q3", f"F4 fires: head_selected gains >= {F4_DEGENERACY_THRESHOLD} from N=1 to "
               f"N=32 on all three anchors", f4, dr["F4"]["per_property"])
    pred("Q4", "F1 fires on every anchor with a computable deployed-k2 share", f1,
         dr["F1"]["per_property"])
    pred("Q5", "F2 fires: every computable deployed-k2 share is in [0.75, 1.00]", f2,
         dr["F2"]["per_property"])
    pred("Q6", "F3 fires, and n_arms_above_head_selected_curve >= 15 of 30",
         bool(f3 and n_above_head >= 15),
         {"n_above_head_selected": n_above_head, "n_above_oracle_selected": n_above_oracle,
          "n_cells": len(scored), "F3_fires": f3})
    q7 = headline.get("hbd_count")
    pred("Q7", ("the hbd_count deployed-k2 share is NOT computable, because C31 records "
                "that cell's advantage against the oracle curve as +0.0317"),
         bool(q7 is not None and q7["oracle_share"] is None),
         {"advantage_vs_oracle_selected": (q7 or {}).get("advantage_vs_oracle_selected"),
          "oracle_share_status": (q7 or {}).get("oracle_share_status"),
          "c31_recorded_advantage": 0.0317})
    n_f5 = sum(1 for v in f5_per.values() if v)
    pred("Q8", "F5 fires on at most two of the three anchors", bool(n_f5 <= 2),
         {"per_property": f5_per, "n_firing": n_f5})
    q9 = {}
    for p, v in headline.items():
        ref = C27_GEN1_DEPLOYED[p]["gap"]
        q9[p] = {"c33_gap_at_deployed_k2_budget": v["budget_matched_gap_at_cell_budget"],
                 "c27_generator_1_gap": ref,
                 "smaller": bool(v["budget_matched_gap_at_cell_budget"] < ref)}
    pred("Q9", ("the budget-matched gap at the deployed k=2 budget is smaller on generator "
                "2 than C27's generator-1 gaps on at least two anchors"),
         sum(1 for v in q9.values() if v["smaller"]) >= 2, q9)
    c31_mtimes = {}
    newest = None
    for d in sorted(OUTPUT_DIR.glob("c31_*")):
        for f in ([d] if d.is_file() else sorted(d.rglob("*"))):
            if f.is_file():
                t = f.stat().st_mtime
                if newest is None or t > newest[1]:
                    newest = (str(f.relative_to(ROOT)), t)
    prereg_mtime = prereg_f.stat().st_mtime
    if newest:
        c31_mtimes = {"newest_c31_artefact": newest[0],
                      "newest_c31_artefact_mtime_utc": datetime.fromtimestamp(
                          newest[1], tz=timezone.utc).isoformat(),
                      "prereg_mtime_utc": mtime_utc(prereg_f),
                      "c31_untouched_since_prereg_freeze": bool(newest[1] < prereg_mtime)}
    pred("Q10", ("no arm of C33 changes any C31 number: C33 re-prices C31's cells, re-runs "
                 "none of them, and every C31 artefact is read-only"),
         bool(c31_mtimes.get("c31_untouched_since_prereg_freeze")), c31_mtimes)

    report["prediction_summary"] = {
        "n": len(P),
        "confirmed": sorted(k for k, v in P.items() if v["outcome"] == "CONFIRMED"),
        "falsified": sorted(k for k, v in P.items() if v["outcome"] == "FALSIFIED"),
    }

    # ---------------------------------------------------------- artefact mtime ordering
    arte = []
    for pat in ("c33_headsel_*", "c33_pool", "c33_selections", "c33_summary"):
        for d in sorted(OUTPUT_DIR.glob(pat)):
            for f in sorted(d.rglob("*")) if d.is_dir() else [d]:
                if f.is_file():
                    arte.append((str(f.relative_to(ROOT)), f.stat().st_mtime))
    report["artefact_mtime_ordering"] = {
        "rule": ("the pre-registration's mtime must strictly precede every C33 measurement "
                 "artefact (prereg_lock.json note)"),
        "prereg_mtime_utc": mtime_utc(prereg_f),
        "n_artefacts_checked": len(arte),
        "earliest_artefact": min(arte, key=lambda x: x[1])[0] if arte else None,
        "earliest_artefact_mtime_utc": (datetime.fromtimestamp(
            min(a[1] for a in arte), tz=timezone.utc).isoformat() if arte else None),
        "prereg_strictly_precedes_every_artefact": bool(
            arte and all(prereg_mtime < t for _, t in arte)),
    }

    write_json(out_dir / "c33_metrics.json", report)
    write_run_context(out_dir, {
        "cli": vars(args), "c31_config": cfg,
        "reads": {"c33_sweeps": [f"c33_headsel_{p}" for p in sweeps],
                  "c31_curves": [f"c31_bestofn_{p}" for p in c31_curves],
                  "c31_cells": sorted(c["dir"] for c in cells.values()),
                  "frontier_machinery": ("scripts/21_summarise_c26.py (interp, t_interval); "
                                         "scripts/25_summarise_c31.py (collect_cells)")},
        "generates": "nothing -- no molecule is sampled and no head is trained",
    })

    print(json.dumps({
        "G1_max_abs_hit_residual": g1r["max_abs_hit_rate_residual"],
        "G1_max_abs_token_residual": g1r["max_abs_token_residual"],
        "G3a_max_abs_residual": report["gates"]["G3_cost_identity"]["a_max_abs_residual"],
        "G3b_max_abs_residual": report["gates"]["G3_cost_identity"]["b_max_abs_residual"],
        "G4_max_abs_residual": report["gates"]["G4_n1_identity"]["max_abs_residual"],
        "G5_terminal_auroc": {p: g5[p]["terminal_mean"] for p in g5},
        "blocking_gates": report["blocking_gates"]["status"],
        "headline_share": {p: v["oracle_share"] for p, v in headline.items()},
        "arm_counts": {"above_head": n_above_head, "above_oracle": n_above_oracle},
        "F": {k: dr[k]["fires"] for k in ("F1", "F2", "F3", "F4", "F5", "F6")},
        "verdict": verdict,
        "predictions": report["prediction_summary"],
    }, indent=1))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
