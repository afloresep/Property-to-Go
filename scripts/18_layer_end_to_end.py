"""C23 -- end-to-end guided decoding with a mid-network head.

C17 measured what a mid-network probe point is worth for steering **per decoding
position** and found it NOT MATERIAL.  A per-position quantity is not an end-to-end one
(`docs/TODO.md` C22.1), and C17 says so about itself in §C17.5.3.  Nobody has ever run
guided decoding with a mid-network head.  This script does exactly that and nothing else.

It is a driver: every generated molecule comes from `scripts/05_guided_generation.py`,
every matched baseline from `scripts/06_best_of_n.py` and every quality number from
`scripts/10_quality_analysis.py`, all unchanged apart from the two additive arguments
`--layer` and `--head-file`.  No logic is forked.

The arm list, the layers, the lambdas and the run order are the pre-registration's
(`reports/section_c23_layer_end_to_end.md` §C23.0, frozen at `outputs/c23_prereg/`) and
are transcribed here rather than derived, so that reading this file is enough to see that
nothing was chosen after a result was seen.

Each arm is its own output directory, so a kill loses at most one arm; re-running the
same command resumes.

    setsid nohup .venv/bin/python scripts/18_layer_end_to_end.py \
        > outputs/c23_run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = str(ROOT / ".venv" / "bin" / "python")

DATASET = "pilot_50k_p2"
HEADS = "pilot_50k_heads_p2"
SEEDS = ("101", "202", "303")

#: C23.0.3 -- layers chosen by C17's frozen numbers.  (property, layer, why).
#: qed's per-position steering-best probe point IS the deployed layer 12, so qed
#: contributes one experimental arm rather than two.  That is C17's number, not a choice.
COMBINATIONS = (
    ("aromatic_rings", 3, "auroc_best"),
    ("hbd_count", 4, "auroc_best"),
    ("qed", 4, "auroc_best"),
    ("aromatic_rings", 6, "per_position_steering_best"),
    ("hbd_count", 6, "per_position_steering_best"),
)

#: C23.0.4 -- lambda, in the pre-registered run order.
LAMBDA_ONE = {"aromatic_rings": 1.0, "hbd_count": 1.0, "qed": 1.0}
LAMBDA_S19_OPTIMUM = {"aromatic_rings": 2.0, "hbd_count": 2.0, "qed": 4.0}
LAMBDA_EXTRA = {"aromatic_rings": 0.5, "hbd_count": 0.5, "qed": 2.0}


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def arm_name(prop: str, layer: int, lam: float, kind: str) -> str:
    return f"c23_{kind}_L{layer}_{lam_tag(lam)}_{prop}"


def deployed_guided_dir(prop: str, lam: float) -> str:
    """The seed-matched deployed-layer (probe point 12) run at the same lambda."""
    if lam == 1.0:
        return f"pilot_50k_p2_guided_{prop}"
    return f"pilot_50k_p2_{lam_tag(lam)}_guided_{prop}"


def run(cmd: list[str], label: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[C23] {label} failed with exit code {r.returncode}")
    print(f"[C23] {label} done in {time.perf_counter() - t0:.0f}s", flush=True)


def guided(prop: str, layer: int, lam: float, out: str, head_file: str) -> None:
    if (OUT / out / "guidance_metrics.json").exists():
        print(f"[C23] skip {out} (already complete)", flush=True)
        return
    run([PY, "scripts/05_guided_generation.py",
         "--dataset", DATASET, "--heads", HEADS, "--property", prop,
         "--lam", f"{lam:g}", "--layer", str(layer), "--head-file", head_file,
         "--conditions", "unguided", "throughout",
         "--seeds", *SEEDS, "--out", out], out)


def best_of_n(prop: str, guided_dir: str, out: str) -> None:
    if (OUT / out / "bestofn_metrics.json").exists():
        print(f"[C23] skip {out} (already complete)", flush=True)
        return
    # N is re-solved from THIS arm's own measured tokens per returned molecule, never
    # inherited from the deployed-layer run (C23.0.6).
    run([PY, "scripts/06_best_of_n.py",
         "--dataset", DATASET, "--property", prop, "--guided", guided_dir,
         "--accounting", "actual", "--out", out], out)


def quality(prop: str, guided_dir: str, out: str) -> None:
    if (OUT / out / "quality_metrics.json").exists():
        print(f"[C23] skip {out} (already complete)", flush=True)
        return
    run([PY, "scripts/10_quality_analysis.py",
         "--dataset", DATASET, "--property", prop, "--guided", guided_dir,
         "--out", out], out)


def gate() -> None:
    """C23.0.1 -- reproduce the deployed lambda=1 aromatic-ring run at --layer 12."""
    out = "c23_gate_L12_lam1_aromatic_rings"
    if (OUT / out / "guidance_metrics.json").exists():
        print(f"[C23] skip {out} (already complete)", flush=True)
        return
    run([PY, "scripts/05_guided_generation.py",
         "--dataset", DATASET, "--heads", HEADS, "--property", "aromatic_rings",
         "--lam", "1", "--layer", "12",
         "--head-file", "c17_probe_layers/head_aromatic_rings_frozen_state_L12.pt",
         "--conditions", "unguided", "throughout",
         "--seeds", *SEEDS, "--out", out], out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gate", "lam1", "optimum", "analysis", "extra", "all"])
    args = ap.parse_args()

    for prop, layer, _why in COMBINATIONS:
        hf = OUT / "c17_probe_layers" / f"head_{prop}_frozen_state_L{layer}.pt"
        if not hf.exists():
            raise SystemExit(f"[C23] missing C17 checkpoint {hf}")

    def head_file(prop: str, layer: int) -> str:
        return f"c17_probe_layers/head_{prop}_frozen_state_L{layer}.pt"

    stages = ["gate", "lam1", "optimum", "analysis", "extra"] if args.stage == "all" \
        else [args.stage]

    # (1) validity gate
    if "gate" in stages:
        gate()

    # (2)+(3) lambda = 1, AUROC-best layer first then steering-best layer
    if "lam1" in stages:
        for prop, layer, _why in COMBINATIONS:
            lam = LAMBDA_ONE[prop]
            guided(prop, layer, lam, arm_name(prop, layer, lam, "guided"),
                   head_file(prop, layer))

    # (4) the section-19 optimal lambda
    if "optimum" in stages:
        for prop, layer, _why in COMBINATIONS:
            lam = LAMBDA_S19_OPTIMUM[prop]
            guided(prop, layer, lam, arm_name(prop, layer, lam, "guided"),
                   head_file(prop, layer))

    # (5) quality and (6) matched best-of-N on everything generated so far
    if "analysis" in stages:
        for prop, layer, _why in COMBINATIONS:
            for lam in (LAMBDA_ONE[prop], LAMBDA_S19_OPTIMUM[prop]):
                g = arm_name(prop, layer, lam, "guided")
                if not (OUT / g / "guidance_metrics.json").exists():
                    continue
                quality(prop, g, arm_name(prop, layer, lam, "quality"))
                best_of_n(prop, g, arm_name(prop, layer, lam, "bestofn"))

    # (7) the extra lambda, named in C23.0.4 before anything was run
    if "extra" in stages:
        for prop, layer, _why in COMBINATIONS:
            lam = LAMBDA_EXTRA[prop]
            g = arm_name(prop, layer, lam, "guided")
            guided(prop, layer, lam, g, head_file(prop, layer))
            quality(prop, g, arm_name(prop, layer, lam, "quality"))
            best_of_n(prop, g, arm_name(prop, layer, lam, "bestofn"))

    print("\n[C23] all requested stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
