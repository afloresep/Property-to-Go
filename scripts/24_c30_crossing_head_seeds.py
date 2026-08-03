"""C30 -- does C28's crossing survive the probe-training seed?

C28 found 8 of 30 guided cells above the *oracle-selected* best-of-N frontier at their own
budget.  Every one was generated with a single probe-training seed, 1234.  C29 then measured
what a probe-training seed is worth -- a between-head-seed sd of 0.0142 to 0.0366 on
end-to-end hit rate -- which is larger than three of those eight margins.  So the project's
only positive result rests on exactly the protocol the project's own methodological finding
says is inadequate.  C30 re-runs it at all eight of C29's head seeds.

Pre-registration: `outputs/c30_prereg/C30.0_preregistration.md`, frozen with its SHA-256 in
`prereg_lock.json` before this script produced anything.

**Nothing is forked.**  `run_cell` is imported from `scripts/23_k_sweep.py`, so every
molecule comes from the same `guidance.guided_sample` call C28 made, with `top_k` and the
head checkpoint as the only differences.  The one thing this script adds is the ability to
point a strand at a different head file, which is done by a *scoped, restored* mutation of
C28's own STRANDS table rather than by a copy of `run_cell` -- a copy could agree with C28
today and drift tomorrow, and gate G2 could then pass for the wrong reason.

Gates, run before any decision rule (C30.0.3):

    G1  the C29 seed-1234 checkpoint is tensor-by-tensor identical to the one C28 used
    G2  at head seed 1234 every cell reproduces its C28 counterpart exactly -- hit rate
        residual 0.0, token residual 0.0, per generation seed, and identical molecules
    G3  processed_tokens_actual mod (k+1) == 0 in every cell

Usage:

    .venv/bin/python scripts/24_c30_crossing_head_seeds.py --gates      # G1 only, no GPU
    .venv/bin/python scripts/24_c30_crossing_head_seeds.py --all        # the 64 cells
    .venv/bin/python scripts/24_c30_crossing_head_seeds.py --strand A1 --head-seed 1234
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json,
)
from property_to_go.model_io import load_generator  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ks = _load_module(ROOT / "scripts" / "23_k_sweep.py", "c28_k_sweep")
_s05 = _load_module(ROOT / "scripts" / "05_guided_generation.py", "guided_generation_05")
STRANDS = _ks.STRANDS
run_cell = _ks.run_cell
lam_tag = _ks.lam_tag
load_head = _s05.load_head

C29_HEADS = "c29_heads"

#: C30.0.2, transcribed and not derived.  The four strands that produce a C28 winning cell.
#: C1 and C3 produce none and are deliberately absent -- see the pre-registration.
STRAND_ORDER = ("A1", "A2", "A3", "C2")

#: C30.0.2.  k = 8, 16, 32 are not re-run; the only winner above k = 4 is A3 at k = 8, which
#: C26 D2b and C29 R6 already replicated across head seeds.
K_GRID = (2, 4)

#: C30.0.2.  All eight of C29's seeds, none dropped.
HEAD_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901)

#: C30.0.3 G2's reference seed.  Also the seed C28 ran.
GATE_HEAD_SEED = 1234


def c29_head_file(strand: str) -> str:
    """The C29 checkpoint family for a strand, relative to outputs/."""
    s = STRANDS[strand]
    return f"{C29_HEADS}/head_{s['property']}_last1_L{s['layer']}_seed{{hs}}.pt"


def cell_dir(strand: str, k: int, head_seed: int) -> Path:
    s = STRANDS[strand]
    return OUTPUT_DIR / (
        f"c30_hs{head_seed}_{s['property']}_L{s['layer']}_{lam_tag(s['lam'])}_k{k}")


@contextlib.contextmanager
def strand_pointed_at(strand: str, head_seed: int):
    """Point C28's STRANDS entry at a C29 checkpoint, then put it back.

    Scoped rather than permanent so that an exception mid-run cannot leave C28's own table
    mutated for anything else importing it in this process.
    """
    s = STRANDS[strand]
    saved_head, saved_dir = s["head_file"], _ks.cell_dir
    s["head_file"] = c29_head_file(strand).format(hs=head_seed)
    _ks.cell_dir = lambda st, k, _hs=head_seed: cell_dir(st, k, _hs)
    try:
        yield
    finally:
        s["head_file"] = saved_head
        _ks.cell_dir = saved_dir


# --------------------------------------------------------------------------------- gate G1

def gate_g1() -> dict:
    """The C29 seed-1234 checkpoint must be tensor-identical to the one C28 used.

    C29 established this for six families.  C30 re-establishes it for the three it uses,
    because a gate cited is a gate not run.
    """
    out = {"gate": "G1", "rule": "C29 seed-1234 checkpoint identical to C28's, tensor by "
                                 "tensor, max abs diff exactly 0.0", "families": {}}
    seen: set[tuple[str, int]] = set()
    worst = 0.0
    for strand in STRAND_ORDER:
        s = STRANDS[strand]
        key = (s["property"], s["layer"])
        if key in seen:
            continue
        seen.add(key)
        c28_path = OUTPUT_DIR / s["head_file"]
        c29_path = OUTPUT_DIR / c29_head_file(strand).format(hs=GATE_HEAD_SEED)
        row: dict = {"c28_checkpoint": str(c28_path.relative_to(OUTPUT_DIR)),
                     "c29_checkpoint": str(c29_path.relative_to(OUTPUT_DIR))}
        if not c28_path.exists() or not c29_path.exists():
            row["checked"] = False
            row["reason"] = f"missing {c28_path if not c28_path.exists() else c29_path}"
            out["families"][f"{s['property']}_L{s['layer']}"] = row
            continue
        a = torch.load(c28_path, map_location="cpu", weights_only=False)
        b = torch.load(c29_path, map_location="cpu", weights_only=False)
        sa, sb = a["state_dict"], b["state_dict"]
        row["checked"] = True
        row["n_tensors"] = len(sa)
        row["same_keys"] = sorted(sa) == sorted(sb)
        diffs = {}
        for key_t in sorted(sa):
            if key_t not in sb:
                continue
            d = float((sa[key_t].float() - sb[key_t].float()).abs().max())
            diffs[key_t] = d
            worst = max(worst, d)
        row["max_abs_diff_per_tensor"] = diffs
        row["max_abs_diff"] = max(diffs.values()) if diffs else None
        # the binner decides which bins the target interval covers; identical weights with a
        # different binner would still change every q.
        ea = a.get("binner", {}).get("edges")
        eb = b.get("binner", {}).get("edges")
        row["binner_edges_identical"] = (ea == eb)
        row["passes"] = bool(row["same_keys"] and row["max_abs_diff"] == 0.0
                             and row["binner_edges_identical"])
        out["families"][f"{s['property']}_L{s['layer']}"] = row
    checked = [r for r in out["families"].values() if r.get("checked")]
    out["n_families_checked"] = len(checked)
    out["max_abs_diff_over_all_families"] = worst
    out["passes"] = bool(checked) and all(r["passes"] for r in checked)
    return out


# --------------------------------------------------------------------------------- gate G2

def gate_g2() -> dict:
    """At head seed 1234 every cell must reproduce its C28 counterpart exactly."""
    out = {"gate": "G2", "rule": "head seed 1234 reproduces C28 exactly: hit-rate and token "
                                 "residual 0.0 per generation seed, molecules identical",
           "cells": {}}
    worst_h = worst_t = 0.0
    n_ok = n_checked = 0
    for strand in STRAND_ORDER:
        for k in K_GRID:
            mine = cell_dir(strand, k, GATE_HEAD_SEED) / "k_cell_metrics.json"
            theirs = _ks.cell_dir(strand, k) / "k_cell_metrics.json"
            name = f"{strand}_k{k}"
            if not mine.exists() or not theirs.exists():
                out["cells"][name] = {"checked": False,
                                      "reason": f"missing {mine if not mine.exists() else theirs}"}
                continue
            a, b = read_json(mine), read_json(theirs)
            row = {"checked": True,
                   "c28_dir": _ks.cell_dir(strand, k).name,
                   "c30_dir": cell_dir(strand, k, GATE_HEAD_SEED).name,
                   "per_seed": {}}
            for sd in a["seeds"]:
                xa = a["seeds_detail"][str(sd)]
                xb = b["seeds_detail"][str(sd)]
                rh = xa["hit_rate"] - xb["hit_rate"]
                rt = (xa["compute"]["tokens_per_molecule_actual"]
                      - xb["compute"]["tokens_per_molecule_actual"])
                row["per_seed"][str(sd)] = {"hit_rate_residual": rh, "token_residual": rt}
                worst_h = max(worst_h, abs(rh))
                worst_t = max(worst_t, abs(rt))
            row["max_abs_hit_rate_residual"] = max(
                abs(v["hit_rate_residual"]) for v in row["per_seed"].values())
            row["max_abs_token_residual"] = max(
                abs(v["token_residual"]) for v in row["per_seed"].values())
            # molecules, not just summary statistics -- a summary can agree by accident
            ma = cell_dir(strand, k, GATE_HEAD_SEED) / "molecules.json"
            mb = _ks.cell_dir(strand, k) / "molecules.json"
            if ma.exists() and mb.exists():
                ja, jb = read_json(ma), read_json(mb)
                same = total = 0
                for sd in ja:
                    for ra, rb in zip(ja[sd], jb.get(sd, [])):
                        total += 1
                        same += int(ra["smiles"] == rb["smiles"])
                row["molecules_identical"] = same
                row["molecules_total"] = total
                row["molecules_all_identical"] = (same == total and total > 0)
            else:
                row["molecules_all_identical"] = None
            row["passes"] = bool(row["max_abs_hit_rate_residual"] == 0.0
                                 and row["max_abs_token_residual"] == 0.0
                                 and row["molecules_all_identical"] is not False)
            n_checked += 1
            n_ok += int(row["passes"])
            out["cells"][name] = row
    out["n_cells_checked"] = n_checked
    out["n_cells_passing"] = n_ok
    out["max_abs_hit_rate_residual"] = worst_h
    out["max_abs_token_residual"] = worst_t
    out["passes"] = bool(n_checked) and n_ok == n_checked
    return out


# --------------------------------------------------------------------------------- gate G3

def gate_g3() -> dict:
    """Cost identity: processed tokens must be divisible by (k+1) in every cell."""
    out = {"gate": "G3", "rule": "processed_tokens_actual mod (k+1) == 0 in every cell",
           "cells": {}}
    worst = 0
    for strand in STRAND_ORDER:
        for k in K_GRID:
            for hs in HEAD_SEEDS:
                f = cell_dir(strand, k, hs) / "k_cell_metrics.json"
                if not f.exists():
                    continue
                d = read_json(f)
                r = int(d["cost_identity_max_residual"])
                out["cells"][f"{strand}_k{k}_hs{hs}"] = r
                worst = max(worst, r)
    out["n_cells_checked"] = len(out["cells"])
    out["max_residual"] = worst
    out["passes"] = bool(out["cells"]) and worst == 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strand", nargs="*", default=None, choices=list(STRAND_ORDER))
    ap.add_argument("--k", type=int, nargs="*", default=None, choices=list(K_GRID))
    ap.add_argument("--head-seed", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gates", action="store_true",
                    help="run G1 (and G2/G3 over whatever exists) and exit; no GPU needed")
    ap.add_argument("--n-molecules", type=int, default=512)
    args = ap.parse_args()

    gate_dir = OUTPUT_DIR / "c30_gates"

    if args.gates:
        g1 = gate_g1()
        write_json(gate_dir / "g1_checkpoint_identity.json", g1)
        print(f"[C30] G1 checkpoint identity: max abs diff="
              f"{g1['max_abs_diff_over_all_families']!r} passes={g1['passes']}", flush=True)
        for name, row in g1["families"].items():
            print(f"       {name}: {row.get('max_abs_diff')!r} "
                  f"binner={row.get('binner_edges_identical')} passes={row.get('passes')}")
        if not g1["passes"]:
            raise SystemExit("[C30] STOP: G1 failed. C30.0.3 requires diagnosis before any cell.")
        g2, g3 = gate_g2(), gate_g3()
        write_json(gate_dir / "g2_reproduces_c28.json", g2)
        write_json(gate_dir / "g3_cost_identity.json", g3)
        print(f"[C30] G2 over {g2['n_cells_checked']} completed cells: "
              f"hit residual={g2['max_abs_hit_rate_residual']!r} "
              f"token residual={g2['max_abs_token_residual']!r} passes={g2['passes']}")
        print(f"[C30] G3 over {g3['n_cells_checked']} completed cells: "
              f"max residual={g3['max_residual']} passes={g3['passes']}")
        return 0

    strands = list(STRAND_ORDER) if args.all else (args.strand or ["A1"])
    strands = [s for s in STRAND_ORDER if s in strands]
    ks = list(args.k) if args.k else list(K_GRID)
    head_seeds = tuple(args.head_seed) if args.head_seed else HEAD_SEEDS

    # G1 first, always.  It costs no GPU and it is the gate that makes G2 interpretable.
    g1 = gate_g1()
    write_json(gate_dir / "g1_checkpoint_identity.json", g1)
    print(f"[C30] G1: max abs diff={g1['max_abs_diff_over_all_families']!r} "
          f"passes={g1['passes']}", flush=True)
    if not g1["passes"]:
        raise SystemExit("[C30] STOP: G1 failed. C30.0.3 requires diagnosis before any cell.")

    gen = load_generator(load_config("model"))
    t0 = time.perf_counter()
    n_done = 0
    # head seed 1234 first, across every strand and k, so G2 -- the gate that decides whether
    # anything else is comparable to C28 -- is answered before the other seven seeds are run.
    ordered = sorted(head_seeds, key=lambda h: (h != GATE_HEAD_SEED, h))
    for hs in ordered:
        for strand in strands:
            for k in ks:
                with strand_pointed_at(strand, hs):
                    run_cell(gen, strand, k, args.n_molecules, _ks.SEEDS)
                n_done += 1
                print(f"[C30] {n_done} cells done, {time.perf_counter() - t0:.0f}s elapsed",
                      flush=True)
        if hs == GATE_HEAD_SEED:
            g2 = gate_g2()
            write_json(gate_dir / "g2_reproduces_c28.json", g2)
            print(f"[C30] G2 after head seed 1234: hit residual="
                  f"{g2['max_abs_hit_rate_residual']!r} token residual="
                  f"{g2['max_abs_token_residual']!r} "
                  f"{g2['n_cells_passing']}/{g2['n_cells_checked']} pass", flush=True)
            if not g2["passes"]:
                raise SystemExit(
                    "[C30] STOP: G2 failed -- head seed 1234 does not reproduce C28. "
                    "C30.0.3 forbids scoring any decision rule past a failed gate.")

    g3 = gate_g3()
    write_json(gate_dir / "g3_cost_identity.json", g3)
    print(f"[C30] G3: max residual={g3['max_residual']} passes={g3['passes']}", flush=True)
    print(f"[C30] complete: {n_done} cells in {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
