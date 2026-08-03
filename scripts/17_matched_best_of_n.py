"""C18 -- compute-matched best-of-N, once per distinct N rather than once per arm.

`scripts/06_best_of_n.py` solves N from the guided run's measured tokens per returned
molecule and then samples with fixed seeds, so **two arms that solve to the same N
produce a bit-identical baseline**.  Measured: `c18_bestofn_uncalibrated_aromatic_rings`
and `c18_bestofn_platt_aromatic_rings` both returned 0.829427, which is also the central
test's own published value.  Running it once per arm would have added 9,828,336 processed
tokens -- 75% of C18's entire guided budget -- to recompute three numbers.

So the token match is computed here for every arm, arms are grouped by their solved N,
and script 06 is invoked once per distinct N.  The per-arm solved N and the realised
token ratio are written out, so "the baseline is shared" is checkable rather than
asserted -- and if any arm ever solves to a different N it gets its own run
automatically.

    .venv/bin/python scripts/17_matched_best_of_n.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.compute import solve_best_of_n  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")


def arm_dirs(prop: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for arm in ("uncalibrated", "platt", "isotonic", "bin_temperature", "binT0p4"):
        d = OUTPUT_DIR / f"c18_guided_{arm}_{prop}"
        if (d / "guidance_metrics.json").exists():
            out[arm] = d
    for arm in ("wide", "focused", "wide_focused"):
        d = OUTPUT_DIR / f"c18_guided_head_{arm}_{prop}"
        if (d / "guidance_metrics.json").exists():
            out[f"head_{arm}"] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--properties", nargs="*",
                    default=["aromatic_rings", "hbd_count", "clogp"])
    ap.add_argument("--out", default="c18_matched_best_of_n")
    args = ap.parse_args()

    report: dict = {
        "dataset": args.dataset,
        "rule": (
            "N is solved from each arm's own tokens per returned molecule under "
            "`actual` accounting, exactly as scripts/06 does. Arms solving to the "
            "same N share one best-of-N run because that run is deterministic in "
            "(property, N, seeds)."
        ),
        "properties": {},
    }

    for prop in args.properties:
        arms = arm_dirs(prop)
        if not arms:
            continue
        entry: dict = {"arms": {}, "runs": {}}
        by_n: dict[int, list[str]] = {}
        for arm, d in arms.items():
            g = read_json(d / "guidance_metrics.json")
            thr = g["conditions"]["throughout"]["aggregate"]["compute_total"]
            ung = g["conditions"]["unguided"]["aggregate"]["compute_total"]
            n = solve_best_of_n(thr["tokens_per_molecule_actual"],
                                ung["tokens_per_molecule_actual"])
            entry["arms"][arm] = {
                "dir": d.name,
                "guided_tokens_per_molecule_actual": thr["tokens_per_molecule_actual"],
                "base_tokens_per_molecule_actual": ung["tokens_per_molecule_actual"],
                "n_candidates_solved": n,
            }
            by_n.setdefault(n, []).append(arm)

        for n, sharing in sorted(by_n.items()):
            # matched to the first arm with this N; every other arm with the same N
            # would produce the identical run
            anchor = sharing[0]
            out = f"c18_bestofn_N{n}_{prop}"
            if not (OUTPUT_DIR / out / "bestofn_metrics.json").exists():
                subprocess.run(
                    [PY, str(ROOT / "scripts" / "06_best_of_n.py"),
                     "--dataset", args.dataset, "--property", prop,
                     "--guided", arms[anchor].name, "--accounting", "actual",
                     "--out", out],
                    check=True, cwd=ROOT,
                )
            b = read_json(OUTPUT_DIR / out / "bestofn_metrics.json")
            m = b["matches"]["actual"]
            entry["runs"][str(n)] = {
                "dir": out,
                "matched_to_arm": anchor,
                "arms_sharing_this_run": sharing,
                "n_candidates": m["n_candidates"],
                "hit_rate": m["aggregate"]["hit_rate"]["mean"],
                "tokens_per_molecule_actual": m["aggregate"]["tokens_per_molecule_actual"],
            }
            for arm in sharing:
                entry["arms"][arm]["best_of_n_dir"] = out
                entry["arms"][arm]["best_of_n_hit_rate"] = m["aggregate"]["hit_rate"]["mean"]
                entry["arms"][arm]["realised_token_ratio"] = (
                    m["aggregate"]["tokens_per_molecule_actual"]
                    / entry["arms"][arm]["guided_tokens_per_molecule_actual"]
                )
            print(f"{prop:16s} N={n} shared by {sharing} -> hit "
                  f"{m['aggregate']['hit_rate']['mean']:.4f}")
        report["properties"][prop] = entry

    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "matched_best_of_n.json", report)
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
