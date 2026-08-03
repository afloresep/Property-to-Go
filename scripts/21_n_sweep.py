"""C26 -- the N sweep: best-of-N's compute-accuracy frontier.

`docs/HANDOFF.md` E1 asked for a lambda sweep *and* an N sweep; only the lambda half
was run (`pilot_report.md` section 19).  Guidance therefore has six points per anchor and
best-of-N has exactly one -- the compute-matched N.  A frontier needs both curves.

The pre-registration is `outputs/c26_prereg/C26.0_preregistration.md`, written before
this script produced anything.

Design (C26.0.2): for each (property, seed) draw ONE pool of `--n-max` x `--n-molecules`
unconditional molecules and evaluate best-of-N for every N in the grid over ALL disjoint
consecutive groups of N in that pool.  Each group still selects among N i.i.d. base-policy
draws, so every point is an unbiased best-of-N estimate, and the whole pool is used at every
N -- 1820 groups at N=9 against script 06's 512, so the estimator noise is ~1.9x smaller
there.  One generation pass instead of eleven.

The first version selected the first N of each slot of `--n-max`, which discarded 70% of the
pool at N=9 and missed the published best-of-9 by up to 0.038 -- inside its own noise, but
comparable to the effect C26 exists to measure.  C26.0.3 gate 2 required replacing the
estimator rather than caveating it.  Its outputs are kept under `c26_nsweep_v1_nested_*`.

`--exact-n N` instead reproduces `scripts/06_best_of_n.py`'s call signature and grouping
(draw N x n_molecules, group consecutive N), which is validity gate C26.0.3(1).

    .venv/bin/python scripts/21_n_sweep.py --dataset pilot_50k_p2 --property aromatic_rings
    .venv/bin/python scripts/21_n_sweep.py --dataset pilot_50k_p2 --property aromatic_rings \
        --exact-n 9 --out c26_gate_exact_N9_aromatic_rings
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.bestofn import selection_key, summarise  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.generation import sample_unconditional  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.properties import PHASE2_PROPERTIES, compute_all_properties  # noqa: E402

DEFAULT_GRID = [1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32]


def score_pool(smiles: list[str], seqs: list[list[int]], prop: str, lo: float, hi: float):
    """Per-candidate selection key, token cost and property value.

    The key is `06_best_of_n.py`'s exactly: invalid and property-unavailable candidates
    rank worst, so a slot can still return an unusable molecule instead of being silently
    repaired by the selection rule.
    """
    extras = frozenset({prop}) if prop in PHASE2_PROPERTIES else frozenset()
    keys: list[tuple] = []
    cands: list[dict] = []
    tokens: list[int] = []
    for s, ids in zip(smiles, seqs):
        props = compute_all_properties(s, extras=extras)
        if props is None:
            keys.append((1, 1, float("inf")))
            cands.append({"smiles": s, "valid": False})
        elif props.get(prop) is None:
            keys.append((1, 1, float("inf")))
            cands.append({"smiles": s, "valid": True, "property_unavailable": True, **props})
        else:
            keys.append((0, *selection_key(props[prop], lo, hi)))
            cands.append({"smiles": s, "valid": True, **props})
        # `actual` accounting: an unconditional draw processes one token per forward step,
        # so a candidate's cost is its trimmed sequence length -- the same quantity
        # `sample_unconditional` hands the meter.
        tokens.append(len(ids))
    return keys, cands, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--property", default="aromatic_rings")
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--n-molecules", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--grid", type=int, nargs="*", default=None)
    ap.add_argument("--exact-n", type=int, default=None,
                    help="validity gate: reproduce script 06's call signature and grouping "
                         "at this N instead of running the nested sweep.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    out_dir = OUTPUT_DIR / (args.out or f"c26_nsweep_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    intervals = read_json(data_dir / "target_intervals.json")
    iv = intervals[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    seeds = args.seeds or list(gcfg["seeds"])
    n_mol = args.n_molecules

    exact = args.exact_n is not None
    n_max = args.exact_n if exact else args.n_max
    grid = [args.exact_n] if exact else sorted(set(args.grid or DEFAULT_GRID))
    if max(grid) > n_max:
        raise SystemExit(f"grid max {max(grid)} exceeds pool depth {n_max}")

    gen = load_generator(model_cfg)
    print(f"property={args.property} target=[{lo:.4f},{hi:.4f}) base_rate={iv['base_rate']:.4f}")
    print(f"mode={'exact-gate' if exact else 'nested'} n_max={n_max} grid={grid}")

    t0 = time.time()
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        # `scripts/06_best_of_n.py` draws its pool with `seed * 1000`, not `seed`.  Matching
        # that convention is what makes the exact gate a gate: the first version of this
        # script used the raw seed, drew a different (perfectly valid) pool, and missed the
        # published best-of-9 by 0.008 -- which looked like a reproducibility failure in the
        # published artefact and was mine.  Recorded in the section rather than quietly fixed.
        pool_seed = seed * 1000
        meter = ComputeMeter()
        meter.start()
        seqs = sample_unconditional(gen, policy, n_max * n_mol, seed=pool_seed, meter=meter)
        meter.stop()
        smiles = gen.decode(seqs)
        keys, cands, tokens = score_pool(smiles, seqs, args.property, lo, hi)

        rows: dict[str, dict] = {}
        pool = len(keys)
        for n in grid:
            # ALL disjoint consecutive groups of N in the pool, which is script 06's
            # grouping extended to use the whole draw instead of its first N x n_mol
            # molecules.  The first `n_mol` groups at N are therefore *exactly* script
            # 06's, so the gate below is an identity rather than a noise comparison.
            #
            # The first version of this script took the first N of each slot of n_max,
            # which discarded 70% of the pool at N=9 and missed the published best-of-9
            # by up to 0.038 -- inside the estimator's own noise but comparable to the
            # effect C26 exists to measure.  C26.0.3 gate 2 required the estimator to be
            # replaced rather than caveated when that happens, and it was.
            n_groups = pool // n if not exact else n_mol
            selected: list[dict] = []
            tok = 0
            for i in range(n_groups):
                block = range(i * n, i * n + n)
                best = min(block, key=lambda j: keys[j])
                selected.append(cands[best])
                tok += sum(tokens[j] for j in block)
            s = summarise(selected, args.property, lo, hi)
            s["compute"] = {
                "processed_tokens_actual": int(tok),
                "molecules_returned": n_groups,
                "tokens_per_molecule_actual": tok / n_groups,
            }
            s["n_groups"] = n_groups
            # the sub-estimate on script 06's own 512 groups, for the identity gate
            if not exact and n_groups >= n_mol:
                sub = [min(range(i * n, i * n + n), key=lambda j: keys[j]) for i in range(n_mol)]
                s["first_512_groups_hit_rate"] = float(
                    summarise([cands[j] for j in sub], args.property, lo, hi)["hit_rate"])
            rows[str(n)] = s
            print(f"  seed={seed} N={n:>2} hit={s['hit_rate']:.4f} "
                  f"tok/mol={tok / n_mol:.1f} val={s['validity']:.3f}")
        per_seed[str(seed)] = {
            "pool_seed": pool_seed,
            "rows": rows,
            "pool_compute": meter.as_dict(),
        }

    curve = {}
    for n in grid:
        hits = [per_seed[str(s)]["rows"][str(n)]["hit_rate"] for s in seeds]
        toks = [per_seed[str(s)]["rows"][str(n)]["compute"]["tokens_per_molecule_actual"]
                for s in seeds]
        vals = [per_seed[str(s)]["rows"][str(n)]["validity"] for s in seeds]
        uniq = [per_seed[str(s)]["rows"][str(n)]["uniqueness"] for s in seeds]
        curve[str(n)] = {
            "n_candidates": n,
            "hit_rate_mean": float(np.mean(hits)),
            "hit_rate_values": [float(h) for h in hits],
            "hit_rate_sd": float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0,
            "tokens_per_molecule_actual": float(np.mean(toks)),
            "validity_mean": float(np.mean(vals)),
            "uniqueness_mean": float(np.mean(uniq)),
        }

    report = {
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": iv,
        "mode": "exact_gate" if exact else "nested",
        "n_max": n_max,
        "n_molecules_per_seed": n_mol,
        "seeds": seeds,
        "grid": grid,
        "accounting": "actual",
        "nesting_note": (
            "each slot selects over the first N of a pool of n_max i.i.d. base-policy draws, "
            "so every point is an unbiased best-of-N estimate and the points are paired "
            "across N" if not exact else
            "consecutive groups of N, reproducing scripts/06_best_of_n.py's grouping"
        ),
        "curve": curve,
        "per_seed": per_seed,
        "wall_seconds_total": time.time() - t0,
    }
    write_json(out_dir / "n_sweep_metrics.json", report)
    write_run_context(out_dir, {"model": model_cfg, "base_policy": policy, "guidance": gcfg,
                                "cli": vars(args)})
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
