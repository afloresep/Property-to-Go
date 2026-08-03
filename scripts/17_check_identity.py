"""C18 -- the decisive end-to-end check that a calibration IS a rescale of lambda.

`calibration.py` proves it in two lines of algebra and
`tests/test_head_calibration.py` checks it on constructed candidate arrays.  This
checks it where it actually matters: two full guided decoding runs, same seeds, same
frozen generator, differing only in

  arm A   power-calibrated head `g(q) = c*q**alpha` at lambda = 1
  arm B   the raw head at lambda = alpha

With `eps = 0` the two scores differ by the candidate-independent constant
`log c`, which the softmax over eight candidates annihilates, so **the two runs must
return the same molecules**.  If they do, `docs/HANDOFF.md` E2's "cheap version"
-- temperature-scale the head's interval probability, then re-run guidance -- is not a
new experiment at all; it is a point on the lambda sweep of `pilot_report.md` §19, and
`alpha < 1` says which point.

The same pair is then run at the deployed `eps = 1e-6` to show how much the floor
costs, so the claim is not quietly restricted to a configuration nobody runs.

    .venv/bin/python scripts/17_check_identity.py --property aromatic_rings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")


def run(prop: str, arm: str, lam: float | None, eps: float, out: str) -> None:
    cmd = [PY, str(ROOT / "scripts" / "17_guided_calibrated.py"),
           "--property", prop, "--arm", arm, "--conditions", "throughout",
           "--eps", str(eps), "--out", out]
    if lam is not None:
        cmd += ["--lam", str(lam)]
    subprocess.run(cmd, check=True, cwd=ROOT)


def compare(a: str, b: str, prop: str) -> dict:
    ma = read_json(OUTPUT_DIR / a / "molecules.json")["throughout"]
    mb = read_json(OUTPUT_DIR / b / "molecules.json")["throughout"]
    ga = read_json(OUTPUT_DIR / a / "guidance_metrics.json")
    gb = read_json(OUTPUT_DIR / b / "guidance_metrics.json")
    same, total = 0, 0
    for seed in ma:
        for ra, rb in zip(ma[seed], mb[seed]):
            total += 1
            same += int(ra["smiles"] == rb["smiles"])
    return {
        "identical_molecule_fraction": same / total if total else None,
        "n_molecules": total,
        "hit_rate_a": ga["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"],
        "hit_rate_b": gb["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"],
        "hit_rate_difference": (
            ga["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"]
            - gb["conditions"]["throughout"]["aggregate"]["hit_rate"]["mean"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="aromatic_rings")
    ap.add_argument("--calibrators", default="c18_offpolicy_calibration")
    ap.add_argument("--out", default="c18_identity")
    args = ap.parse_args()

    cal = read_json(OUTPUT_DIR / args.calibrators / f"calibrator_{args.property}.json")
    alpha = float(cal["platt"]["a"])
    p = args.property
    report: dict = {"property": p, "alpha_from_platt_slope": alpha,
                    "claim": ("a power calibration g(q) = c*q**alpha at lambda = 1 "
                              "induces exactly the sampling distribution of the raw "
                              "head at lambda = alpha"),
                    "arms": {}}

    for eps in (0.0, 1e-6):
        tag = "eps0" if eps == 0.0 else "epsdeployed"
        a_dir = f"c18_identity_{tag}_power_{p}"
        b_dir = f"c18_identity_{tag}_lam{alpha:.4f}_{p}".replace(".", "p", 1)
        run(p, "power_limit", None, eps, a_dir)
        run(p, "uncalibrated", alpha, eps, b_dir)
        report["arms"][tag] = {"eps": eps, "power_run": a_dir, "lambda_run": b_dir,
                               **compare(a_dir, b_dir, p)}
        print(tag, report["arms"][tag])

    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "identity_check.json", report)
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
