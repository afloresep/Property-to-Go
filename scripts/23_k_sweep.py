"""C28 -- the top-k sweep: guided decoding's compute knob.

`top_k_candidates: 8` is fixed in `configs/guidance.yaml`, is the default of
`guidance.guided_sample`, and has never been swept anywhere in this project.  Guidance's
cost under the `cached` backend is exactly `(k+1)` x the base cost per generated position,
so **k is the compute knob**, it lives inside the method as specified, it changes no weight
and trains nothing.  C26's headline "guided decoding has no compute knob" is therefore, as
of today, a claim about a hyperparameter we froze.  This script turns it.

Pre-registration: `outputs/c28_prereg/C28.0_preregistration.md`, frozen with its SHA-256 in
`prereg_lock.json` before this script produced anything.

Nothing is forked.  Every molecule comes from `guidance.guided_sample` with the same
arguments `scripts/05_guided_generation.py` passes it, and the per-condition summary is
`scripts/05_guided_generation.py::summarise` **imported**, not copied -- so validity gate G1
(k = 8 reproducing the published deployed run exactly) cannot be passed by a
re-implementation that merely agrees today.  The only thing that varies is `top_k`.

Only the `throughout` condition is generated.  `guided_sample` calls `torch.manual_seed(seed)`
on entry and each condition is a separate call, so a run restricted to `throughout` is
bit-identical to the same condition inside a six-condition run; gate G1 asserts exactly that.

Each (property, layer, lambda, k) is its own output directory and a completed cell is never
regenerated, so a kill costs at most one cell.

    .venv/bin/python scripts/23_k_sweep.py --strand A1
    .venv/bin/python scripts/23_k_sweep.py --strand A1 --k 8          # the gate cell alone
    .venv/bin/python scripts/23_k_sweep.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_to_go import generation  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer, Windows, guided_sample  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import compute_all_properties, uniqueness, validity  # noqa: E402

DATASET = "pilot_50k_p2"
HEADS = "pilot_50k_heads_p2"
SEEDS = (101, 202, 303)
#: C28.0.2.  k = 1 is excluded: one candidate makes the softmax degenerate and the property
#: term cannot act, so it would measure greedy-restricted decoding rather than guidance.
K_GRID = (2, 4, 8, 16, 32)
#: C28.0.5.  The k = 8 cell of each strand is a validity gate against a frozen artefact.
GATE_K = 8


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_s05 = _load_module(ROOT / "scripts" / "05_guided_generation.py", "guided_generation_05")
summarise = _s05.summarise
load_head = _s05.load_head


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


#: C28.0.2 -- the strands, transcribed from the pre-registration rather than derived, so that
#: reading this file is enough to see that nothing was chosen after a result was seen.
#: (property, layer, lambda, head file relative to outputs/, gate artefact).
STRANDS: dict[str, dict] = {
    "A1": {
        "property": "hbd_count", "layer": 12, "lam": 1.0,
        "head_file": f"{HEADS}/head_hbd_count_frozen_state.pt",
        "gate_run": "pilot_50k_p2_guided_hbd_count",
        "why": "the deployed configuration; k = 8 is the published run",
    },
    "A2": {
        "property": "hbd_count", "layer": 4, "lam": 1.0,
        "head_file": "c17_probe_layers/head_hbd_count_frozen_state_L4.pt",
        "gate_run": "c23_guided_L4_lam1_hbd_count",
        "why": "the best mid-network layer (C23: L4 beats L6 at every common lambda)",
    },
    "A3": {
        "property": "hbd_count", "layer": 4, "lam": 2.0,
        "head_file": "c17_probe_layers/head_hbd_count_frozen_state_L4.pt",
        "gate_run": "c23_guided_L4_lam2_hbd_count",
        "why": "the strongest hbd_count guidance arm on record (0.5603 at 387.79 tokens)",
    },
    # C28.0.2 strand C -- priority 3, named before strand A was run.
    "C1": {
        "property": "aromatic_rings", "layer": 12, "lam": 1.0,
        "head_file": f"{HEADS}/head_aromatic_rings_frozen_state.pt",
        "gate_run": "pilot_50k_p2_guided_aromatic_rings",
        "why": "strand C: the deployed configuration on the second anchor",
    },
    "C2": {
        "property": "aromatic_rings", "layer": 3, "lam": 2.0,
        "head_file": "c17_probe_layers/head_aromatic_rings_frozen_state_L3.pt",
        "gate_run": "c23_guided_L3_lam2_aromatic_rings",
        "why": "strand C: the strongest aromatic_rings guidance arm on record (0.8159)",
    },
    "C3": {
        "property": "qed", "layer": 12, "lam": 1.0,
        "head_file": f"{HEADS}/head_qed_frozen_state.pt",
        "gate_run": "pilot_50k_p2_guided_qed",
        "why": "strand C: the deployed configuration on the third anchor",
    },
}

#: C28.0.2 run order: cheap cells first so a kill loses the least, gate cell first of all.
RUN_ORDER_K = (8, 2, 4, 16, 32)
STRAND_ORDER = ("A1", "A2", "A3", "C1", "C2", "C3")


def cell_dir(strand: str, k: int) -> Path:
    s = STRANDS[strand]
    return OUTPUT_DIR / (
        f"c28_ksweep_{s['property']}_L{s['layer']}_{lam_tag(s['lam'])}_k{k}")


def run_cell(gen, strand: str, k: int, n_mol: int, seeds, force: bool = False) -> Path:
    s = STRANDS[strand]
    prop, layer, lam = s["property"], s["layer"], float(s["lam"])
    out_dir = cell_dir(strand, k)
    if (out_dir / "k_cell_metrics.json").exists() and not force:
        print(f"[C28] skip {out_dir.name} (already complete)", flush=True)
        return out_dir

    data_dir = OUTPUT_DIR / DATASET
    gcfg = load_config("guidance")
    policy = load_config("base_policy")
    model_cfg = load_config("model")
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    iv = intervals[prop]
    lo, hi = float(iv["lo"]), float(iv["hi"])

    head_path = OUTPUT_DIR / s["head_file"]
    if not head_path.exists():
        raise SystemExit(f"[C28] missing head checkpoint {head_path}")
    head, binner = load_head(head_path)
    scorer = TargetScorer(head, binner, lo, hi)

    report: dict = {
        "experiment": "C28",
        "prereg": "outputs/c28_prereg/C28.0_preregistration.md",
        "strand": strand,
        "strand_why": s["why"],
        "dataset": DATASET,
        "property": prop,
        "condition": "throughout",
        "head_input": "frozen_state",
        "head_checkpoint": head_path.name,
        "head_file": str(head_path),
        "layer": layer,
        "target_interval": iv,
        "windows": windows.to_dict(),
        "lambda": lam,
        "top_k": k,
        "eps": float(gcfg["eps"]),
        "backend": gcfg["candidate_backend"],
        "batch_size": int(gcfg["batch_size"]),
        "n_molecules_per_seed": n_mol,
        "seeds": list(seeds),
        "gate_run": s["gate_run"],
        "seeds_detail": {},
    }
    records_by_seed: dict[str, list[dict]] = {}
    t0 = time.perf_counter()

    for seed in seeds:
        meter = ComputeMeter().start()
        seqs = guided_sample(
            gen,
            scorer=scorer,
            # `throughout` -- the same predicate scripts/05 passes for this condition.
            window_fn=windows.fn("throughout"),
            policy=policy,
            n_molecules=n_mol,
            seed=seed,
            top_k=k,
            lam=lam,
            eps=float(gcfg["eps"]),
            backend=gcfg["candidate_backend"],
            batch_size=int(gcfg["batch_size"]),
            layer=(-1 if layer == 12 else layer),
            meter=meter,
        )
        meter.stop()
        smiles = gen.decode(seqs)
        records = []
        for ids, smi in zip(seqs, smiles):
            props = compute_all_properties(smi)
            content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
            records.append({"smiles": smi, "n_content_tokens": len(content),
                            "valid": props is not None, **(props or {})})
        records_by_seed[str(seed)] = records
        st = summarise(records, prop, lo, hi)
        st["compute"] = meter.as_dict()
        st["validity_check"] = validity(smiles)
        st["uniqueness_check"] = uniqueness(smiles)
        # C28.0.3 / gate G4: the cached backend charges `active` + `active * k` at every
        # guided step, so the actual token count must be divisible by (k + 1) exactly.
        st["cost_identity_tokens_mod_k_plus_1"] = int(
            st["compute"]["processed_tokens_actual"] % (k + 1))
        st["cost_identity_base_steps"] = st["compute"]["processed_tokens_actual"] / (k + 1)
        report["seeds_detail"][str(seed)] = st
        print(f"  {strand} k={k:>2} seed={seed} hit={st['hit_rate']:.6f} "
              f"val={st['validity']:.4f} len={st['content_length_mean']:.1f} "
              f"tok/mol={st['compute']['tokens_per_molecule_actual']:.4f} "
              f"({meter.wall_seconds:.0f}s)", flush=True)

    # aggregate exactly as scripts/05_guided_generation.py does, so the gate residual is a
    # comparison of like with like.
    keys = ["hit_rate", "abs_target_error_mean", "validity", "uniqueness",
            "property_mean", "content_length_mean", "n_heavy_atoms_mean"]
    agg: dict = {}
    for key in keys:
        v = [report["seeds_detail"][str(sd)][key] for sd in seeds]
        agg[key] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "values": v}
    tot = ComputeMeter()
    for sd in seeds:
        c = report["seeds_detail"][str(sd)]["compute"]
        tot.processed_tokens_actual += c["processed_tokens_actual"]
        tot.processed_tokens_full_recompute += c["processed_tokens_full_recompute"]
        tot.wall_seconds += c["wall_seconds"]
        tot.molecules_returned += c["molecules_returned"]
        tot.forward_calls += c["forward_calls"]
    agg["compute_total"] = tot.as_dict()
    report["aggregate"] = agg
    report["cost_identity_max_residual"] = max(
        report["seeds_detail"][str(sd)]["cost_identity_tokens_mod_k_plus_1"] for sd in seeds)
    report["wall_seconds_total"] = time.perf_counter() - t0

    write_json(out_dir / "k_cell_metrics.json", report)
    write_json(out_dir / "molecules.json", records_by_seed)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                                "cli": {"strand": strand, "top_k": k, "layer": layer,
                                        "lam": lam, "head_file": str(head_path),
                                        "condition": "throughout"}})
    print(f"[C28] -> {out_dir.name}  hit={agg['hit_rate']['mean']:.6f} "
          f"tok/mol={agg['compute_total']['tokens_per_molecule_actual']:.6f}", flush=True)
    return out_dir


def check_gate(strand: str) -> dict:
    """G1/G2/G3 -- the k = 8 cell must reproduce its frozen artefact exactly."""
    s = STRANDS[strand]
    cell = cell_dir(strand, GATE_K) / "k_cell_metrics.json"
    ref_f = OUTPUT_DIR / s["gate_run"] / "guidance_metrics.json"
    if not cell.exists() or not ref_f.exists():
        return {"strand": strand, "checked": False,
                "reason": f"missing {cell if not cell.exists() else ref_f}"}
    got = read_json(cell)
    ref = read_json(ref_f)["conditions"]["throughout"]
    res = {
        "strand": strand, "checked": True, "reference": s["gate_run"],
        "hit_rate_reference": ref["aggregate"]["hit_rate"]["mean"],
        "hit_rate_measured": got["aggregate"]["hit_rate"]["mean"],
        "hit_rate_residual": (got["aggregate"]["hit_rate"]["mean"]
                              - ref["aggregate"]["hit_rate"]["mean"]),
        "tokens_reference": ref["aggregate"]["compute_total"]["tokens_per_molecule_actual"],
        "tokens_measured": got["aggregate"]["compute_total"]["tokens_per_molecule_actual"],
        "tokens_residual": (got["aggregate"]["compute_total"]["tokens_per_molecule_actual"]
                            - ref["aggregate"]["compute_total"]["tokens_per_molecule_actual"]),
        "per_seed": {},
    }
    worst_h = worst_t = 0.0
    for sd in got["seeds"]:
        a = got["seeds_detail"][str(sd)]
        b = ref["seeds"][str(sd)]
        rh = a["hit_rate"] - b["hit_rate"]
        rt = (a["compute"]["tokens_per_molecule_actual"]
              - b["compute"]["tokens_per_molecule_actual"])
        res["per_seed"][str(sd)] = {"hit_rate_residual": rh, "token_residual": rt}
        worst_h = max(worst_h, abs(rh))
        worst_t = max(worst_t, abs(rt))
    res["max_abs_seed_hit_rate_residual"] = worst_h
    res["max_abs_seed_token_residual"] = worst_t
    res["passes"] = (res["hit_rate_residual"] == 0.0 and res["tokens_residual"] == 0.0
                     and worst_h == 0.0 and worst_t == 0.0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strand", nargs="*", default=None, choices=list(STRANDS))
    ap.add_argument("--k", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every strand in the pre-registered order")
    ap.add_argument("--n-molecules", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--gate-only", action="store_true",
                    help="report the k=8 gate residuals of completed cells and exit")
    args = ap.parse_args()

    strands = list(STRANDS) if args.all else (args.strand or ["A1"])
    strands = [s for s in STRAND_ORDER if s in strands]
    ks = list(args.k) if args.k else list(RUN_ORDER_K)
    ks = [k for k in RUN_ORDER_K if k in ks]
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    if args.gate_only:
        for st in strands:
            print(check_gate(st))
        return 0

    gen = load_generator(load_config("model"))
    for st in strands:
        for k in ks:
            run_cell(gen, st, k, args.n_molecules, seeds)
            if k == GATE_K:
                g = check_gate(st)
                print(f"[C28] gate {st}: hit residual={g.get('hit_rate_residual')!r} "
                      f"token residual={g.get('tokens_residual')!r} "
                      f"passes={g.get('passes')}", flush=True)
                if not g.get("passes"):
                    raise SystemExit(
                        f"[C28] STOP: validity gate for strand {st} did not return residual "
                        f"0.0 ({g}).  C28.0.5 requires diagnosis before any further cell.")
    print("[C28] k sweep: all requested cells complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
