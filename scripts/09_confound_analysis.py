"""Is the guided property shift only a length or size shift?

`05_guided_generation.py` already reports a hit rate matched on **sequence length**
(content tokens).  The specification asks about "sequence length *or molecule size*",
which are not the same variable: a guided run could keep token count fixed and still
win by producing heavier molecules (or vice versa -- branch and ring-closure tokens
cost sequence length without adding heavy atoms).

This script recomputes the confound test from the saved per-molecule records under
four estimators:

  raw            no matching
  length         matched on content-token count
  size           matched on heavy-atom count
  joint          matched on the (length, size) cell

Each is a direct standardisation: the condition's per-stratum hit rate reweighted by
the **unguided** stratum frequencies.  If an advantage survives all four it is not
explained by the guided run drifting to longer or larger molecules.  `coverage` is the
share of unguided mass lying in strata the condition actually visited -- the estimate
is only defined there, and a low coverage is itself evidence of distribution shift.

    python scripts/09_confound_analysis.py --dataset pilot_50k --property clogp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402

LENGTH_BIN = 5
SIZE_BIN = 3


def _scorable(records: list[dict], prop: str) -> list[dict]:
    """Valid molecules that also have a value for this property.

    `valid` means RDKit parsed the SMILES; one descriptor (QED) can still fail on a
    parseable molecule. Applied in one place so the stratum list and the value list
    below cannot drift out of alignment -- they are zipped, so a filter applied to one
    and not the other would silently pair a molecule with another's stratum.
    """
    return [r for r in records if r.get("valid") and r.get(prop) is not None]


def _strata(records: list[dict], kind: str) -> list[tuple]:
    out = []
    for r in records:
        L = r["n_content_tokens"] // LENGTH_BIN
        S = r["n_heavy_atoms"] // SIZE_BIN
        out.append({"raw": (), "length": (L,), "size": (S,), "joint": (L, S)}[kind])
    return out


def standardised_hit_rate(
    records: list[dict], reference: list[dict], prop: str, lo: float, hi: float, kind: str
) -> dict:
    """Condition hit rate reweighted onto the reference's stratum distribution."""
    cond_rows = _scorable(records, prop)
    cond_keys = _strata(cond_rows, kind)
    ref_keys = _strata(_scorable(reference, prop), kind)
    cond_vals = [r[prop] for r in cond_rows]

    by_cell: dict[tuple, list[float]] = {}
    for k, v in zip(cond_keys, cond_vals):
        by_cell.setdefault(k, []).append(v)

    ref_counts: dict[tuple, int] = {}
    for k in ref_keys:
        ref_counts[k] = ref_counts.get(k, 0) + 1
    total_ref = sum(ref_counts.values())
    if not total_ref:
        return {"hit_rate": float("nan"), "coverage": 0.0, "n_strata": 0}

    num = 0.0
    covered = 0
    for cell, n_ref in ref_counts.items():
        if cell not in by_cell:
            continue
        v = np.asarray(by_cell[cell])
        num += float(((v >= lo) & (v < hi)).mean()) * n_ref
        covered += n_ref
    return {
        "hit_rate": float(num / covered) if covered else float("nan"),
        "coverage": float(covered / total_ref),
        "n_strata": len(by_cell),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k")
    ap.add_argument("--property", default="clogp")
    ap.add_argument("--guided", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    guided_dir = OUTPUT_DIR / (args.guided or f"{args.dataset}_guided_{args.property}")
    out_dir = OUTPUT_DIR / (args.out or f"{args.dataset}_confound_{args.property}")
    out_dir.mkdir(parents=True, exist_ok=True)

    iv = read_json(data_dir / "target_intervals.json")[args.property]
    lo, hi = float(iv["lo"]), float(iv["hi"])
    mols = read_json(guided_dir / "molecules.json")

    pooled = {c: [m for seed in per_seed.values() for m in seed] for c, per_seed in mols.items()}
    reference = pooled["unguided"]

    report = {
        "dataset": args.dataset,
        "property": args.property,
        "target_interval": {"lo": lo, "hi": hi, "base_rate": iv["base_rate"]},
        "length_bin_tokens": LENGTH_BIN,
        "size_bin_heavy_atoms": SIZE_BIN,
        "note": (
            "Standardised onto the unguided stratum distribution. 'coverage' is the "
            "share of unguided mass in strata the condition actually visited."
        ),
        "conditions": {},
    }

    ref_raw = standardised_hit_rate(reference, reference, args.property, lo, hi, "raw")["hit_rate"]

    print(f"property={args.property} target=[{lo:.4f},{hi:.4f}) unguided_raw={ref_raw:.4f}")
    header = f"{'condition':20s} " + " ".join(f"{k:>22s}" for k in ("raw", "length", "size", "joint"))
    print(header)

    for cond, recs in pooled.items():
        entry = {"n_molecules": len(recs)}
        valid = [r for r in recs if r.get("valid")]
        entry["length_mean"] = float(np.mean([r["n_content_tokens"] for r in valid]))
        entry["heavy_atoms_mean"] = float(np.mean([r["n_heavy_atoms"] for r in valid]))
        cells = []
        for kind in ("raw", "length", "size", "joint"):
            s = standardised_hit_rate(recs, reference, args.property, lo, hi, kind)
            s["delta_vs_unguided"] = s["hit_rate"] - ref_raw
            entry[kind] = s
            cells.append(f"{s['hit_rate']:.4f}{s['delta_vs_unguided']:+.4f}(c{s['coverage']:.2f})")
        report["conditions"][cond] = entry
        print(f"{cond:20s} " + " ".join(f"{c:>22s}" for c in cells))

    print(
        "\nmean length / heavy atoms per condition:\n  "
        + "\n  ".join(
            f"{c:20s} len={e['length_mean']:.1f} atoms={e['heavy_atoms_mean']:.1f}"
            for c, e in report["conditions"].items()
        )
    )

    write_json(out_dir / "confound_metrics.json", report)
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
