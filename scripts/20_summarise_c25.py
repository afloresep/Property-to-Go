"""C25 -- assemble every cell and arm and score the pre-registered decision rules.

Reads only; generates nothing.  Every number `reports/section_c25_pooling.md` quotes comes
out of `outputs/c25_summary/c25_metrics.json`, and `tests/test_pooled_readout.py` re-reads
that file and requires the numbers to appear in the section text, so the prose cannot drift
from the artefacts.

The seed-stratified bootstrap, the hit-vector reconstruction and the token accounting are
`scripts/18_summarise_c23.py`'s, **imported** rather than reimplemented, so a C25 end-to-end
number is computed by the same estimator as the C23 number it is compared against.

    .venv/bin/python scripts/20_summarise_c25.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import probe_layers as P  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

DATASET = "pilot_50k_p2"
SEEDS = ("101", "202", "303")
N_BOOT = 10000
BOOT_SEED = 20260731

#: §C25.0.3, transcribed from the frozen pre-registration.
BEST_MID = {"aromatic_rings": 3, "hbd_count": 4, "rotatable_bonds": 4,
            "tpsa": 5, "clogp": 5, "qed": 4}
FINAL = 12
PROPERTIES = ("aromatic_rings", "hbd_count", "rotatable_bonds", "tpsa", "clogp", "qed")
ANCHORS = ("aromatic_rings", "hbd_count", "qed")
S19_OPTIMUM = {"aromatic_rings": 2.0, "hbd_count": 2.0, "qed": 4.0}
UNGUIDED_REFERENCE = {"aromatic_rings": 0.1785, "hbd_count": 0.0837, "qed": 0.0896}
POOLED_VARIANTS = ("mean4", "mean16", "concat4", "attn4")
RULE_PROPERTY_THRESHOLD = 4          # ">= 4 of the 6 properties"
DISQUALIFY_VALIDITY_DROP = 0.01
DISQUALIFY_UNIQUENESS_DROP = 0.01
TOKEN_RATIO_CEILING = 1.05
HEAD_SEED_SPAN_LIMIT = 0.05          # §C25.0.7

#: §C25.0.7 -- the three C23 arms replicated at head seeds 2345 / 3456.
REPLICATION_ARMS = (
    ("aromatic_rings", 3, 1.0, "c23_guided_L3_lam1_aromatic_rings"),
    ("hbd_count", 4, 2.0, "c23_guided_L4_lam2_hbd_count"),
    ("qed", 4, 1.0, "c23_guided_L4_lam1_qed"),
)
REPLICATION_HEAD_SEEDS = (2345, 3456)


def _c23():
    path = ROOT / "scripts" / "18_summarise_c23.py"
    spec = importlib.util.spec_from_file_location("summarise_c23", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def deployed_dir(prop: str, lam: float, kind: str) -> Path:
    if lam == 1.0:
        return OUTPUT_DIR / f"pilot_50k_p2_{kind}_{prop}"
    return OUTPUT_DIR / f"pilot_50k_p2_{lam_tag(lam)}_{kind}_{prop}"


def arm_stats(d: Path, prop: str, condition: str = "throughout") -> dict:
    m = read_json(d / "guidance_metrics.json")
    agg = m["conditions"][condition]["aggregate"]
    tot = agg["compute_total"]
    return {
        "dir": d.name,
        "hit_rate": agg["hit_rate"]["mean"],
        "hit_rate_by_seed": agg["hit_rate"]["values"],
        "validity": agg["validity"]["mean"],
        "uniqueness": agg["uniqueness"]["mean"],
        "content_length": agg["content_length_mean"]["mean"],
        "tokens_per_molecule_actual": tot["tokens_per_molecule_actual"],
        "unguided_hit_rate": m["conditions"]["unguided"]["aggregate"]["hit_rate"]["mean"],
        "lambda": m["lambda"],
        "layer": m["layer"],
        "head_seed": m.get("head_seed"),
        "pool_variant": m.get("pool_variant", "last1"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="c25_pooled_heads")
    ap.add_argument("--out", default="c25_summary")
    args = ap.parse_args()

    C = _c23()
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = read_json(OUTPUT_DIR / args.sweep / "pooled_metrics.json")
    intervals = read_json(OUTPUT_DIR / DATASET / "target_intervals.json")
    rng = np.random.default_rng(BOOT_SEED)

    report: dict = {
        "dataset": DATASET,
        "preregistration": "reports/section_c25_pooling.md §C25.0 "
                           "(frozen at outputs/c25_prereg/)",
        "material_margin": P.MATERIAL_MARGIN,
        "seed_sd_constant": P.SEED_SD,
        "bonferroni_family_size": sweep["bonferroni_family_size"],
        "n_boot_end_to_end": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
    }

    # ---- gates ---------------------------------------------------------------------
    ext = read_json(OUTPUT_DIR / f"c25_window_states_{DATASET}"
                    / "window_states_summary.json")
    report["extraction_gate"] = ext["validity_gate"]
    report["extraction_tokens"] = ext["processed_tokens_actual"]
    report["n_prefix_rows"] = ext["n_prefix_rows"]
    report["validity_gate"] = sweep["validity_gate"]

    # C17 per-head-seed replication: the mid-layer heads C17 trained but never saved.
    c17_path = OUTPUT_DIR / "c17_probe_layers" / "probe_layer_metrics.json"
    per_seed_rows = []
    if c17_path.exists():
        c17 = read_json(c17_path)
        for key, cell in sweep["cells"].items():
            if cell["variant"] != "last1":
                continue
            ref = c17["properties"][cell["property"]]["layers"][str(cell["probe_point"])]
            for mine, theirs in zip(cell["per_seed"], ref["per_seed"]):
                a = mine["test"]["intervals"]["target"]["auroc"]
                b = theirs["test"]["intervals"]["target"]["auroc"]
                per_seed_rows.append({
                    "property": cell["property"], "probe_point": cell["probe_point"],
                    "head_seed": mine["head_seed"], "auroc": a, "c17_auroc": b,
                    "residual": abs(a - b)})
    report["c17_per_head_seed_replication"] = {
        "rows": per_seed_rows,
        "max_residual": max([r["residual"] for r in per_seed_rows], default=None),
    }

    # ---- cells and comparisons -------------------------------------------------------
    report["cells"] = sweep["cells"]
    report["comparisons"] = sweep["comparisons"]

    def helps(c) -> bool:
        return bool(c["clears_material_margin"] and c["corrected_ci_excludes_zero"])

    def hurts(c) -> bool:
        return bool(c["auroc_margin"] <= -P.MATERIAL_MARGIN
                    and c["bootstrap_bonferroni"]["hi"] < 0)

    for c in report["comparisons"]:
        c["verdict"] = "HELPS" if helps(c) else ("HURTS" if hurts(c)
                                                 else "INDISTINGUISHABLE")

    def props_helped(depth: str, variants: tuple[str, ...]) -> dict:
        by_prop = {}
        for prop in PROPERTIES:
            hits = [c for c in report["comparisons"]
                    if c["depth"] == depth and c["property"] == prop
                    and c["variant"] in variants and c["verdict"] == "HELPS"]
            by_prop[prop] = {"helped_by": [h["variant"] for h in hits],
                             "best_margin": max([h["auroc_margin"] for h in hits],
                                                default=None)}
        n = sum(1 for v in by_prop.values() if v["helped_by"])
        return {"by_property": by_prop, "n_properties_helped": n}

    rules: dict = {}
    p1 = props_helped("final", POOLED_VARIANTS)
    rules["Rule P1"] = {"description": "pooling helps at the final layer (probe point 12)",
                        "threshold": RULE_PROPERTY_THRESHOLD, **p1,
                        "fires": bool(p1["n_properties_helped"] >= RULE_PROPERTY_THRESHOLD)}
    p2 = props_helped("mid", POOLED_VARIANTS)
    rules["Rule P2"] = {"description": "pooling helps at the best mid-network probe point",
                        "threshold": RULE_PROPERTY_THRESHOLD, **p2,
                        "fires": bool(p2["n_properties_helped"] >= RULE_PROPERTY_THRESHOLD)}
    p3 = props_helped("mid", ("wide1",))
    rules["Rule P3"] = {"description": "capacity binds at depth (wide1 at the mid point)",
                        "threshold": RULE_PROPERTY_THRESHOLD, **p3,
                        "fires": bool(p3["n_properties_helped"] >= RULE_PROPERTY_THRESHOLD)}

    # Rule P4 -- the aromatic-ring crossover, at probe point 12 only.
    heads13 = read_json(OUTPUT_DIR / "pilot_50k_heads_p2" / "head_metrics.json")
    trivial = (heads13["properties"]["aromatic_rings"]["heads"]["trivial"]
               ["across_seeds"]["auroc"]["mean"])
    ring_cells = {k: v for k, v in sweep["cells"].items()
                  if v["property"] == "aromatic_rings" and v["depth"] == "final"}
    best_ring = max(ring_cells.values(), key=lambda v: v["across_seeds"]["auroc"]["mean"])
    rules["Rule P4"] = {
        "description": "a pooled readout at probe point 12 beats the trivial counter "
                       "for aromatic rings by the material margin",
        "trivial_auroc": trivial,
        "threshold_auroc": trivial + P.MATERIAL_MARGIN,
        "best_variant": best_ring["variant"],
        "best_auroc": best_ring["across_seeds"]["auroc"]["mean"],
        "margin_over_trivial": best_ring["across_seeds"]["auroc"]["mean"] - trivial,
        "fires": bool(best_ring["across_seeds"]["auroc"]["mean"]
                      >= trivial + P.MATERIAL_MARGIN),
    }
    rules["Rule N"] = {
        "description": "null -- none of P1..P4 fires",
        "fires": not any(rules[k]["fires"] for k in ("Rule P1", "Rule P2", "Rule P3",
                                                     "Rule P4")),
    }

    # ---- Trigger T ------------------------------------------------------------------
    qualifying = sorted(
        [c for c in report["comparisons"]
         if c["property"] in ANCHORS and c["variant"] in POOLED_VARIANTS
         and c["verdict"] == "HELPS"],
        key=lambda c: -c["auroc_margin"])
    report["trigger"] = {
        "rule": "anchor property and HELPS at that (property, depth) cell",
        "cap": 6,
        "qualifying": [{"property": c["property"], "depth": c["depth"],
                        "probe_point": c["probe_point"], "variant": c["variant"],
                        "auroc_margin": c["auroc_margin"]} for c in qualifying],
        "n_qualifying": len(qualifying),
    }

    # ---- end-to-end arms actually executed -------------------------------------------
    e2e = []
    for d in sorted(OUTPUT_DIR.glob("c25_guided_*")):
        if not (d / "guidance_metrics.json").exists():
            continue
        m = read_json(d / "guidance_metrics.json")
        prop = m["property"]
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        lam = float(m["lambda"])
        layer = int(m["layer"])
        a = arm_stats(d, prop)
        # seed-matched last1 comparator at the SAME probe point and the SAME lambda
        if layer == FINAL:
            comp_dir = deployed_dir(prop, lam, "guided")
        else:
            comp_dir = OUTPUT_DIR / f"c23_guided_L{layer}_{lam_tag(lam)}_{prop}"
        entry = {"name": d.name, "property": prop, "layer": layer, "lam": lam,
                 "variant": m.get("pool_variant"), "window": m.get("pool_window"),
                 **{k: v for k, v in a.items() if k != "dir"}}
        entry["unguided_reference"] = UNGUIDED_REFERENCE.get(prop)
        entry["unguided_reproduces"] = (
            abs(a["unguided_hit_rate"] - UNGUIDED_REFERENCE[prop]) < 5e-5
            if prop in UNGUIDED_REFERENCE else None)

        if comp_dir.exists() and (comp_dir / "guidance_metrics.json").exists():
            b = arm_stats(comp_dir, prop)
            entry["comparator"] = b
            va = C.hit_vectors(d, prop, lo, hi)
            vb = C.hit_vectors(comp_dir, prop, lo, hi)
            alpha = 0.05 / max(1, len(list(OUTPUT_DIR.glob("c25_guided_*"))))
            entry["vs_comparator"] = C.bootstrap_difference(va, vb, N_BOOT, alpha, rng)
            per_seed_d = np.array(a["hit_rate_by_seed"]) - np.array(b["hit_rate_by_seed"])
            sem = float(per_seed_d.std(ddof=1) / np.sqrt(len(per_seed_d)))
            entry["per_seed_difference"] = per_seed_d.tolist()
            entry["beyond_seed_noise"] = bool(abs(per_seed_d.mean()) > 2 * sem)
            entry["token_ratio_vs_comparator"] = (
                a["tokens_per_molecule_actual"] / b["tokens_per_molecule_actual"])
            entry["disqualified"] = bool(
                a["validity"] < b["validity"] - DISQUALIFY_VALIDITY_DROP
                or a["uniqueness"] < b["uniqueness"] - DISQUALIFY_UNIQUENESS_DROP
                or entry["token_ratio_vs_comparator"] > TOKEN_RATIO_CEILING)

        bon = OUTPUT_DIR / d.name.replace("c25_guided_", "c25_bestofn_")
        if (bon / "bestofn_metrics.json").exists():
            bm = read_json(bon / "bestofn_metrics.json")["matches"]["actual"]
            entry["best_of_n"] = {
                "N": bm["n_candidates"],
                "hit_rate": bm["aggregate"]["hit_rate"]["mean"],
                "hit_rate_by_seed": bm["aggregate"]["hit_rate"]["values"],
                "tokens_per_molecule_actual":
                    bm["aggregate"]["tokens_per_molecule_actual"],
            }
            entry["advantage_over_best_of_n"] = (
                a["hit_rate"] - entry["best_of_n"]["hit_rate"])
            entry["realised_token_ratio_vs_best_of_n"] = (
                a["tokens_per_molecule_actual"]
                / entry["best_of_n"]["tokens_per_molecule_actual"])
        qual = OUTPUT_DIR / d.name.replace("c25_guided_", "c25_quality_")
        if (qual / "quality_metrics.json").exists():
            entry["quality"] = read_json(qual / "quality_metrics.json").get("summary")
        e2e.append(entry)
    report["end_to_end"] = e2e

    fired_e1 = sum(1 for e in e2e if e.get("vs_comparator", {}).get("excludes_zero")
                   and e.get("vs_comparator", {}).get("difference", 0) > 0
                   and e.get("beyond_seed_noise") and not e.get("disqualified"))
    rules["Rule E1"] = {"description": "pooling improves guidance end to end",
                        "n_arms_qualifying": fired_e1, "threshold": 2,
                        "fires": bool(fired_e1 >= 2)}
    e2_arms = [e for e in e2e if e.get("advantage_over_best_of_n", -1) > 0
               and not e.get("disqualified")]
    rules["Rule E2"] = {"description": "a pooled arm beats its own compute-matched "
                                       "best-of-N",
                        "arms": [e["name"] for e in e2_arms],
                        "fires": bool(e2_arms)}
    rules["Rule E3"] = {"description": "null end to end",
                        "fires": bool(e2e) and not (rules["Rule E1"]["fires"]
                                                    or rules["Rule E2"]["fires"])}
    report["rules"] = rules

    # ---- head-seed replication (§C25.0.7) --------------------------------------------
    rep_arms = []
    for prop, layer, lam, ref_dir in REPLICATION_ARMS:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        by_seed = {}
        dirs = {}
        base = OUTPUT_DIR / ref_dir
        if (base / "guidance_metrics.json").exists():
            by_seed["1234"] = arm_stats(base, prop)["hit_rate"]
            dirs["1234"] = base.name
        for hs in REPLICATION_HEAD_SEEDS:
            d = OUTPUT_DIR / f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_guided"
            if (d / "guidance_metrics.json").exists():
                by_seed[str(hs)] = arm_stats(d, prop)["hit_rate"]
                dirs[str(hs)] = d.name
        if len(by_seed) < 2:
            continue
        vals = np.array(list(by_seed.values()))
        arm = {"name": f"{prop} L{layer} lam{lam:g}", "property": prop,
               "probe_point": layer, "lam": lam,
               "hit_rate_by_head_seed": by_seed, "dirs": dirs,
               "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
               "span": float(vals.max() - vals.min()),
               "span_limit": HEAD_SEED_SPAN_LIMIT}
        arm["span_within_limit"] = bool(arm["span"] < HEAD_SEED_SPAN_LIMIT)
        # For the Rule-B arm: does it still beat its own compute-matched best-of-N?
        bon_by_seed = {}
        b0 = OUTPUT_DIR / ref_dir.replace("c23_guided_", "c23_bestofn_")
        if (b0 / "bestofn_metrics.json").exists():
            bm = read_json(b0 / "bestofn_metrics.json")["matches"]["actual"]
            bon_by_seed["1234"] = {"N": bm["n_candidates"],
                                   "hit_rate": bm["aggregate"]["hit_rate"]["mean"],
                                   "tokens_per_molecule_actual":
                                       bm["aggregate"]["tokens_per_molecule_actual"]}
        for hs in REPLICATION_HEAD_SEEDS:
            b = OUTPUT_DIR / f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}_bestofn"
            if (b / "bestofn_metrics.json").exists():
                bm = read_json(b / "bestofn_metrics.json")["matches"]["actual"]
                bon_by_seed[str(hs)] = {"N": bm["n_candidates"],
                                        "hit_rate": bm["aggregate"]["hit_rate"]["mean"],
                                        "tokens_per_molecule_actual":
                                            bm["aggregate"]["tokens_per_molecule_actual"]}
        if bon_by_seed:
            arm["best_of_n_by_head_seed"] = bon_by_seed
            arm["advantage_by_head_seed"] = {
                k: by_seed[k] - bon_by_seed[k]["hit_rate"]
                for k in by_seed if k in bon_by_seed}
            arm["beats_best_of_n_on_all_head_seeds"] = bool(
                arm["advantage_by_head_seed"]
                and all(v > 0 for v in arm["advantage_by_head_seed"].values()))
        qual = {}
        for k, dn in dirs.items():
            qd = OUTPUT_DIR / (dn.replace("_guided", "_quality")
                               if dn.startswith("c25_")
                               else dn.replace("c23_guided_", "c23_quality_"))
            if (qd / "quality_metrics.json").exists():
                qual[k] = qd.name
        arm["quality_dirs"] = qual
        rep_arms.append(arm)
    if rep_arms:
        report["head_seed_replication"] = {
            "rule": "§C25.0.7 -- a span >= 0.05 demotes C23's Rule B to not replicated",
            "arms": rep_arms,
        }

    write_json(out_dir / "c25_metrics.json", report)
    write_run_context(out_dir)

    print("\n=== C25 decision rules ===")
    for k, v in rules.items():
        print(f"  {k:9s} {'FIRES' if v['fires'] else 'does not fire':14s} "
              f"{v['description']}")
    print(f"\nvalidity gate max residual: {report['validity_gate']['max_residual']!r}")
    if per_seed_rows:
        print(f"C17 per-head-seed max residual: "
              f"{report['c17_per_head_seed_replication']['max_residual']!r}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
