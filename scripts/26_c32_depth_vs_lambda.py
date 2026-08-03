"""C32 -- is C31's crossing depth, lambda, or their interaction?

C31 ran two corners of a 2x2 and only two: `(probe point 12, lambda=1)` and
`(mid probe point M, lambda=2)`.  All five of its crossing cells are the second corner, so
"reading mid-network helps" and "steering harder helps" are **observationally identical** in
C31.  C32 runs the two missing corners, and adds the effective-lambda control C29 needed on
GP-MoLFormer -- where correcting for the wider `log q` spread of a mid-network probe removed
54-69% of C23's depth margin.

Pre-registration: `outputs/c32_prereg/C32.0_preregistration.md`, frozen with its SHA-256 in
`prereg_lock.json` before this script produced anything.  C32.0.2 discloses that C31's
results were known when C32 was designed, and which degrees of freedom that creates.

**Nothing is forked.**  `run_ksweep_cell` is **imported** from
`scripts/25_c31_second_generator.py` and pointed at new output directories by a scoped,
restored mutation of that module's `cell_dir` -- C30's pattern.  A copy could agree with C31
today and drift tomorrow, and gate G1 could then pass for the wrong reason.  The probe point
and lambda are the only things that vary.

**The mid probe points are TRANSCRIBED, never re-selected.**  C31 chose them by held-out
validation AUROC before any steering outcome existed (C31.0.4).  Re-selecting them now, after
seeing which arm crossed, is exactly the failure `pilot_report.md` section 21.5.2 exists to
prevent.

Stages:

    gate        G1 (the new code path reproduces C31 exactly, molecules included), G2
                (frozen intervals, windows and head checkpoints unchanged), G5 (the
                comparator is C31's curve and C32 generates no best-of-N molecule)
    grid        Arm A -- the two missing 2x2 corners, 2 x 5 k x 3 properties = 30 cells
    envelope    Arm B -- the lambda envelope at probe point 12 for the effective-lambda
                correction, 5 new lambda x 2 k x 3 properties = 30 cells
    spread      Arm C -- `mean_head_q_spread_across_candidates` at probe point 12 and at M,
                on the SAME prefixes and candidate sets, so the ratio is paired

Every stage is idempotent per output directory: a completed cell is never regenerated.

    .venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage gate
    .venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage grid
    .venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage envelope
    .venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage spread
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.binning import interval_probability  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_c31 = _load_module(ROOT / "scripts" / "25_c31_second_generator.py", "c31_second_generator")
run_ksweep_cell = _c31.run_ksweep_cell
load_zinc_generator = _c31.load_zinc_generator
model_cfg_of = _c31.model_cfg_of
load_head_ckpt = _c31.load_head_ckpt
head_path = _c31.head_path
lam_tag = _c31.lam_tag
ALL_C31_PROPERTIES = _c31.ALL_C31_PROPERTIES
DATASET_DIR = _c31.DATASET_DIR

#: C32.0.3.  The 2x2.  `deployed_l1` and `mid_l2` are C31's own arms and are NOT re-run;
#: they appear here so the table is complete and so G1 can prove C32 reaches them.
DEPLOYED_PROBE_POINT = 12
CORNERS = {
    "deployed_l1": {"depth": "final", "lam": 1.0, "source": "c31"},
    "deployed_l2": {"depth": "final", "lam": 2.0, "source": "c32"},
    "mid_l1": {"depth": "mid", "lam": 1.0, "source": "c32"},
    "mid_l2": {"depth": "mid", "lam": 2.0, "source": "c31"},
}
NEW_CORNERS = ("deployed_l2", "mid_l1")

#: C32.0.3, transcribed from C31's grid.
K_GRID = (2, 4, 8, 16, 32)
#: C32.0.4.  The two budgets at which C31's crossings live.
PRIMARY_K = (2, 4)

#: C32.0.3 Arm B.  Committed BEFORE any spread ratio was measured on this generator, so the
#: grid cannot have been chosen to bracket a convenient lambda_eff.  1.0 and 2.0 are the
#: 2x2's own deployed cells and are reused rather than re-run.
ENVELOPE_LAMBDAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
ENVELOPE_K = (2, 4)

#: C32.0.7 G1.  One cell from each corner C32 reuses rather than re-runs.
GATE_CELLS = (("hbd_count", "deployed_l1", 2), ("aromatic_rings", "mid_l2", 2))


def mid_probe_point(prop: str) -> int:
    """C31's M, TRANSCRIBED from the artefact.  Never re-selected -- see the module docstring."""
    d = read_json(OUTPUT_DIR / "c31_heads" / f"depth_{prop}.json")
    return int(d["mid_probe_point"]["selected"])


def probe_point_of(prop: str, corner: str) -> int:
    return (DEPLOYED_PROBE_POINT if CORNERS[corner]["depth"] == "final"
            else mid_probe_point(prop))


def c32_cell_dir(prop: str, arm: str, probe: int, lam: float, k: int) -> Path:
    return OUTPUT_DIR / f"c32_cell_{prop}_{arm}_L{probe}_{lam_tag(lam)}_k{k}"


def corner_dir(prop: str, corner: str, k: int) -> Path:
    """Where a 2x2 corner's cell lives -- C31's directory for the two reused corners."""
    probe, lam = probe_point_of(prop, corner), CORNERS[corner]["lam"]
    if CORNERS[corner]["source"] == "c31":
        arm = "deployed" if CORNERS[corner]["depth"] == "final" else "mid"
        return _c31.cell_dir(prop, arm, probe, lam, k)
    return c32_cell_dir(prop, corner, probe, lam, k)


def envelope_dir(prop: str, lam: float, k: int) -> Path:
    """Arm B.  lambda = 1 and 2 ARE the 2x2's deployed cells; reused, never duplicated."""
    if lam == 1.0:
        return corner_dir(prop, "deployed_l1", k)
    if lam == 2.0:
        return corner_dir(prop, "deployed_l2", k)
    return c32_cell_dir(prop, "env", DEPLOYED_PROBE_POINT, lam, k)


@contextlib.contextmanager
def cells_redirected_to(fn):
    """Point C31's `cell_dir` at a C32 location, then put it back.

    Scoped rather than permanent so an exception mid-run cannot leave C31's own module
    mutated for anything else importing it in this process.  This is what lets C32 reuse
    `run_ksweep_cell` verbatim instead of copying it.
    """
    saved = _c31.cell_dir
    _c31.cell_dir = fn
    try:
        yield
    finally:
        _c31.cell_dir = saved


def _spec(prop: str, corner: str, why: str) -> dict:
    return {"probe_point": probe_point_of(prop, corner), "lam": CORNERS[corner]["lam"],
            "why": why}


# ================================================================== gates G1, G2, G5


def gate_g1(gen, cfg, n_mol: int, seeds) -> dict:
    """G1 -- the redirected C31 code path must reproduce a C31 cell EXACTLY.

    Run into a fresh directory, then compared against the frozen C31 artefact on hit rate,
    processed tokens and the returned molecule strings.  This is the gate that makes every
    other C32 number comparable to C31's.
    """
    out = {"gate": "G1",
           "rule": ("run_ksweep_cell, imported from C31 and redirected to a fresh C32 "
                    "directory, reproduces its C31 counterpart exactly: hit-rate residual "
                    "0.0 and token residual 0.0 per generation seed, molecules identical"),
           "cells": {}}
    worst_h = worst_t = 0.0
    for prop, corner, k in GATE_CELLS:
        probe, lam = probe_point_of(prop, corner), CORNERS[corner]["lam"]
        ref_dir = corner_dir(prop, corner, k)
        mine_dir = OUTPUT_DIR / f"c32_gate_{prop}_{corner}_L{probe}_{lam_tag(lam)}_k{k}"
        name = f"{prop}_{corner}_k{k}"
        if not (ref_dir / "k_cell_metrics.json").exists():
            out["cells"][name] = {"checked": False, "reason": f"missing {ref_dir}"}
            continue
        with cells_redirected_to(lambda *_a, _d=mine_dir, **_kw: _d):
            run_ksweep_cell(gen, cfg, prop, corner, _spec(prop, corner, "G1 gate cell"),
                            k, n_mol, seeds)
        a, b = read_json(mine_dir / "k_cell_metrics.json"), read_json(ref_dir / "k_cell_metrics.json")
        row = {"checked": True, "c31_dir": ref_dir.name, "c32_dir": mine_dir.name,
               "probe_point": probe, "lam": lam, "k": k, "per_seed": {}}
        for sd in a["seeds"]:
            xa, xb = a["seeds_detail"][str(sd)], b["seeds_detail"][str(sd)]
            rh = xa["hit_rate"] - xb["hit_rate"]
            rt = (xa["compute"]["tokens_per_molecule_actual"]
                  - xb["compute"]["tokens_per_molecule_actual"])
            row["per_seed"][str(sd)] = {"hit_rate_residual": rh, "token_residual": rt}
            worst_h, worst_t = max(worst_h, abs(rh)), max(worst_t, abs(rt))
        row["max_abs_hit_rate_residual"] = max(
            abs(v["hit_rate_residual"]) for v in row["per_seed"].values())
        row["max_abs_token_residual"] = max(
            abs(v["token_residual"]) for v in row["per_seed"].values())
        # molecules, not just summaries -- a summary can agree by accident
        ja, jb = read_json(mine_dir / "molecules.json"), read_json(ref_dir / "molecules.json")
        same = total = 0
        for sd in ja:
            for ra, rb in zip(ja[sd], jb.get(sd, [])):
                total += 1
                same += int(ra["smiles"] == rb["smiles"])
        row["molecules_identical"] = same
        row["molecules_total"] = total
        row["molecules_all_identical"] = bool(same == total and total > 0)
        row["passes"] = bool(row["max_abs_hit_rate_residual"] == 0.0
                             and row["max_abs_token_residual"] == 0.0
                             and row["molecules_all_identical"])
        out["cells"][name] = row
    checked = [r for r in out["cells"].values() if r.get("checked")]
    out["n_cells_checked"] = len(checked)
    out["max_abs_hit_rate_residual"] = worst_h
    out["max_abs_token_residual"] = worst_t
    out["passes"] = bool(checked) and all(r["passes"] for r in checked)
    return out


def gate_g2() -> dict:
    """G2 -- the frozen intervals, windows and head checkpoints are unchanged."""
    data = OUTPUT_DIR / DATASET_DIR
    c31 = read_json(OUTPUT_DIR / "c31_summary" / "c31_metrics.json")
    g6 = c31["validity_gates"]["G6"]
    out = {"gate": "G2",
           "rule": ("the C31 target intervals, windows and head checkpoints C32 steers with "
                    "are unchanged: file hashes match C31's record and every head tensor is "
                    "identical to the one C31 used"),
           "files": {}, "heads": {}}
    ok = True
    for name, key in (("target_intervals.json", "target_intervals_sha256"),
                      ("windows.json", "windows_sha256")):
        got = hashlib.sha256((data / name).read_bytes()).hexdigest()
        out["files"][name] = {"recorded": g6[key], "measured": got, "matches": got == g6[key]}
        ok &= got == g6[key]
    # the head checkpoints: C32 loads the same files C31 steered with, so identity is a
    # file-level statement here; the tensors are compared against a re-load to catch a
    # silently rewritten checkpoint.
    for prop in ALL_C31_PROPERTIES:
        for probe in sorted({DEPLOYED_PROBE_POINT, mid_probe_point(prop)}):
            p = head_path(prop, probe, 1234)
            row = {"file": str(p.relative_to(OUTPUT_DIR)), "exists": p.exists()}
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                row["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                row["head_seed"] = ck.get("head_seed")
                row["probe_point"] = ck.get("probe_point")
                row["property"] = ck.get("property")
                row["binner_edges_or_max"] = ck["binner"].get("edges", ck["binner"].get("max_value"))
                row["matches_requested_probe_point"] = bool(ck.get("probe_point") == probe)
                row["matches_requested_property"] = bool(ck.get("property") == prop)
                ok &= bool(row["matches_requested_probe_point"]
                           and row["matches_requested_property"])
            else:
                ok = False
            out["heads"][f"{prop}_L{probe}"] = row
    out["passes"] = bool(ok)
    return out


def gate_g5() -> dict:
    """G5 -- the comparator is C31's curve, and C32 generates no best-of-N molecule."""
    out = {"gate": "G5",
           "rule": ("the oracle-selected best-of-N curve C32 prices against is byte-identical "
                    "to C31's, and no C32 stage samples a best-of-N pool"),
           "curves": {}}
    ok = True
    for prop in ALL_C31_PROPERTIES:
        f = OUTPUT_DIR / f"c31_bestofn_{prop}" / "n_sweep_metrics.json"
        row = {"file": str(f.relative_to(OUTPUT_DIR)), "exists": f.exists()}
        if f.exists():
            row["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            d = read_json(f)
            row["arm"] = d["arm"]
            row["grid"] = d["grid"]
            row["is_oracle_selected"] = d["arm"] == "oracle_selected"
            ok &= row["is_oracle_selected"]
        else:
            ok = False
        out["curves"][prop] = row
    # Checked on the AST, not on raw text.  A text search over this file would match the
    # names inside this very check -- the first version of this gate failed for exactly that
    # reason -- and would also forbid naming the function in a comment.  The invariant is
    # that C32 never *calls* a pool sampler and never *defines* a best-of-N stage.
    import ast

    tree = ast.parse((ROOT / "scripts" / "26_c32_depth_vs_lambda.py").read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} | \
             {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
    banned = {"sample_unconditional", "best_of_n", "score_pool"}
    out["no_best_of_n_stage_defined"] = not any(
        d.startswith("stage_") and "bestofn" in d for d in defined)
    out["no_pool_sampler_called"] = not (banned & (called | imported))
    out["c32_generates_no_best_of_n"] = bool(out["no_best_of_n_stage_defined"]
                                             and out["no_pool_sampler_called"])
    out["passes"] = bool(ok and out["c32_generates_no_best_of_n"])
    return out


def stage_gate(args) -> int:
    cfg = _c31.load_c31_config()
    seeds = tuple(int(s) for s in cfg["generation_seeds"])
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    gdir = OUTPUT_DIR / "c32_gates"

    g2, g5 = gate_g2(), gate_g5()
    write_json(gdir / "g2_frozen_artefacts.json", g2)
    write_json(gdir / "g5_comparator.json", g5)
    print(f"[C32] G2 frozen artefacts: passes={g2['passes']}")
    print(f"[C32] G5 comparator: passes={g5['passes']}")
    if not (g2["passes"] and g5["passes"]):
        raise SystemExit("[C32] STOP: G2/G5 failed. C32.0.7 forbids scoring past a failed gate.")

    gen = load_zinc_generator(model_cfg_of(cfg))
    g1 = gate_g1(gen, cfg, n_mol, seeds)
    write_json(gdir / "g1_reproduces_c31.json", g1)
    write_run_context(gdir, {"c31": cfg, "cli": vars(args)})
    print(f"[C32] G1: hit residual={g1['max_abs_hit_rate_residual']!r} "
          f"token residual={g1['max_abs_token_residual']!r} passes={g1['passes']}")
    for name, row in g1["cells"].items():
        if row.get("checked"):
            print(f"       {name}: molecules {row['molecules_identical']}/"
                  f"{row['molecules_total']} identical, passes={row['passes']}")
    if not g1["passes"]:
        raise SystemExit("[C32] STOP: G1 failed -- C32's code path does not reproduce C31.")
    return 0


# ==================================================== Arm A: the two missing 2x2 corners


def stage_grid(args) -> int:
    cfg = _c31.load_c31_config()
    seeds = tuple(int(s) for s in cfg["generation_seeds"])
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    ks = [int(x) for x in (args.k or K_GRID)]
    props = args.properties or list(ALL_C31_PROPERTIES)
    corners = args.corners or list(NEW_CORNERS)

    gen = load_zinc_generator(model_cfg_of(cfg))
    t0 = time.perf_counter()
    n = 0
    for k in sorted(ks):  # cheap k first so a kill loses the least
        for prop in props:
            for corner in corners:
                probe, lam = probe_point_of(prop, corner), CORNERS[corner]["lam"]
                why = (f"C32 2x2 corner {corner}: "
                       f"{'final' if CORNERS[corner]['depth'] == 'final' else 'mid'} probe "
                       f"point {probe} at lambda = {lam:g}")
                d = c32_cell_dir(prop, corner, probe, lam, k)
                with cells_redirected_to(lambda *_a, _d=d, **_kw: _d):
                    run_ksweep_cell(gen, cfg, prop, corner, _spec(prop, corner, why),
                                    k, n_mol, seeds, force=args.force)
                n += 1
                print(f"[C32] {n} cells done, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[C32] grid complete: {n} cells in {time.perf_counter() - t0:.0f}s")
    return 0


# ================================== Arm B: the lambda envelope at the final probe point


def stage_envelope(args) -> int:
    cfg = _c31.load_c31_config()
    seeds = tuple(int(s) for s in cfg["generation_seeds"])
    n_mol = args.n_molecules or int(cfg["n_molecules_per_condition"])
    props = args.properties or list(ALL_C31_PROPERTIES)
    lams = [float(x) for x in (args.lam or ENVELOPE_LAMBDAS)]
    ks = [int(x) for x in (args.k or ENVELOPE_K)]

    gen = load_zinc_generator(model_cfg_of(cfg))
    t0 = time.perf_counter()
    n = skipped = 0
    for k in sorted(ks):
        for prop in props:
            for lam in sorted(lams):
                if lam in (1.0, 2.0):
                    # these ARE the 2x2's deployed cells; reused, never duplicated
                    skipped += 1
                    continue
                d = envelope_dir(prop, lam, k)
                spec = {"probe_point": DEPLOYED_PROBE_POINT, "lam": lam,
                        "why": (f"C32 Arm B: lambda envelope at the final probe point, "
                                f"lambda = {lam:g}, for the effective-lambda correction")}
                with cells_redirected_to(lambda *_a, _d=d, **_kw: _d):
                    run_ksweep_cell(gen, cfg, prop, "env", spec, k, n_mol, seeds,
                                    force=args.force)
                n += 1
                print(f"[C32] envelope {n} cells done, {time.perf_counter() - t0:.0f}s",
                      flush=True)
    print(f"[C32] envelope complete: {n} new cells ({skipped} reused from the 2x2) "
          f"in {time.perf_counter() - t0:.0f}s")
    return 0


# ============================== Arm C: the head-q spread, paired across probe points


@torch.no_grad()
def stage_spread(args) -> int:
    """`mean_head_q_spread_across_candidates` at probe point 12 and at M, PAIRED.

    The definition is `scripts/16_layer_steering_value.py`'s, transcribed: the mean over
    prefixes of `max_k q - min_k q` over the base model's top-k candidates.  It is the
    quantity C29's `effective_lambda` block consumes.

    Both probe points are read from the SAME forward pass over the SAME prefixes and the
    SAME candidate sets, so the ratio is a paired quantity and cannot be moved by a
    difference in which prefixes each probe point happened to see.
    """
    cfg = _c31.load_c31_config()
    out_dir = OUTPUT_DIR / "c32_spread"
    gen = load_zinc_generator(model_cfg_of(cfg))
    intervals = read_json(OUTPUT_DIR / DATASET_DIR / "target_intervals.json")

    seq_src = read_json(OUTPUT_DIR / "c31_feasibility" / "sequences.json")["token_ids"]
    rng = np.random.default_rng(3232)
    prefixes = []
    for s in seq_src[: args.n_prefixes]:
        if len(s) >= 6:
            prefixes.append(s[: int(rng.integers(3, len(s) - 1))])

    result = {
        "experiment": "C32", "stage": "spread",
        "prereg": "outputs/c32_prereg/C32.0_preregistration.md",
        "definition": ("mean over prefixes of (max_k q - min_k q) over the base model's top-k "
                       "candidates; scripts/16_layer_steering_value.py's "
                       "`mean_head_q_spread_across_candidates`, transcribed"),
        "pairing": ("probe point 12 and M are read from the same forward pass over the same "
                    "prefixes and the same candidate sets, so the ratio is paired"),
        "n_prefixes": len(prefixes),
        "top_k": args.top_k,
        "properties": {},
    }

    for prop in ALL_C31_PROPERTIES:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        mid = mid_probe_point(prop)
        probes = sorted({DEPLOYED_PROBE_POINT, mid})
        heads = {}
        for L in probes:
            head, binner, _ = load_head_ckpt(head_path(prop, L, 1234))
            heads[L] = (head, binner)
        acc = {L: [] for L in probes}
        mean_q = {L: [] for L in probes}
        by_len: dict[int, list] = {}
        for pfx in prefixes:
            by_len.setdefault(len(pfx), []).append(pfx)
        for length, group in by_len.items():
            for st in range(0, len(group), args.batch):
                chunk = group[st:st + args.batch]
                b = len(chunk)
                ids = torch.tensor(chunk, dtype=torch.long, device=gen.device)
                o = gen.model(input_ids=ids, use_cache=True, return_dict=True)
                lp = torch.log_softmax(o.logits[:, -1, :].float(), dim=-1)
                cand_ids = torch.topk(lp, args.top_k, dim=-1).indices
                co = gen.model(input_ids=cand_ids.reshape(b * args.top_k, 1),
                               past_key_values=gen.repeat_cache_fn(o.past_key_values, args.top_k),
                               use_cache=True, output_hidden_states=True, return_dict=True)
                for L in probes:
                    h = co.hidden_states[L][:, 0, :].float().cpu().numpy()
                    head, binner = heads[L]
                    q = interval_probability(head.predict_proba(h), binner, lo, hi)
                    q = q.reshape(b, args.top_k)
                    acc[L].append(q.max(axis=1) - q.min(axis=1))
                    mean_q[L].append(q.mean(axis=1))
        spread = {L: float(np.concatenate(acc[L]).mean()) for L in probes}
        ref = spread[DEPLOYED_PROBE_POINT]
        result["properties"][prop] = {
            "mid_probe_point": mid,
            "mid_probe_point_source": ("TRANSCRIBED from outputs/c31_heads/depth_*.json, "
                                       "selected by C31.0.4's held-out validation AUROC rule "
                                       "before any steering outcome existed"),
            "final_probe_point": DEPLOYED_PROBE_POINT,
            "spread": {str(L): spread[L] for L in probes},
            "mean_q": {str(L): float(np.concatenate(mean_q[L]).mean()) for L in probes},
            "spread_ratio": spread[mid] / ref if ref else float("nan"),
            "lambda_effective_at_lam1": 1.0 * (spread[mid] / ref if ref else float("nan")),
            "lambda_effective_at_lam2": 2.0 * (spread[mid] / ref if ref else float("nan")),
        }
        r = result["properties"][prop]
        print(f"[C32] {prop:15s} M={mid:<2} spread_mid={spread[mid]:.6f} "
              f"spread_L12={ref:.6f} ratio={r['spread_ratio']:.4f} "
              f"lam_eff(2)={r['lambda_effective_at_lam2']:.4f}", flush=True)

    write_json(out_dir / "spread.json", result)
    write_run_context(out_dir, {"c31": cfg, "cli": vars(args)})
    return 0


# ============================================================================== main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gate", "grid", "envelope", "spread"])
    ap.add_argument("--properties", nargs="*", default=None, choices=list(ALL_C31_PROPERTIES))
    ap.add_argument("--corners", nargs="*", default=None, choices=list(NEW_CORNERS))
    ap.add_argument("--k", type=int, nargs="*", default=None)
    ap.add_argument("--lam", type=float, nargs="*", default=None)
    ap.add_argument("--n-molecules", type=int, default=None)
    ap.add_argument("--n-prefixes", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    return {"gate": stage_gate, "grid": stage_grid, "envelope": stage_envelope,
            "spread": stage_spread}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
