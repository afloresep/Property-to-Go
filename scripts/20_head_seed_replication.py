"""C25 priority 4 -- head-seed replication of three C23 arms at a mid-network layer.

C23's end-to-end result rests on **one head seed**.  C17 trained three seeds per probe
point but saved only the seed-1234 checkpoint at non-final layers, so no mid-layer arm has
ever been replicated, including the single arm that fires C23's Rule B (hbd_count, probe
point 4, lambda 2).  §13.2's "head-seed sd <= 0.0041" is a *prediction* quantity and says
nothing about the end-to-end hit rate of a guided run.

`scripts/20_pooled_sweep.py` trains `last1` -- the deployed single-position readout -- at
every property's mid probe point at all three head seeds and saves every checkpoint, so
the replication is a generation job and nothing else.

This is a driver.  Every molecule comes from the **unmodified**
`scripts/05_guided_generation.py` through its C23 `--layer` / `--head-file` arguments;
every matched baseline from `scripts/06_best_of_n.py`; every quality number from
`scripts/10_quality_analysis.py`.  No logic is forked and no existing directory is touched.

The arms, the layers, the lambdas and the head seeds are §C25.0.7's, transcribed rather
than derived.

    setsid nohup .venv/bin/python scripts/20_head_seed_replication.py \
        > outputs/c25_headseed.log 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = str(ROOT / ".venv" / "bin" / "python")

DATASET = "pilot_50k_p2"
HEADS = "pilot_50k_heads_p2"
SEEDS = ("101", "202", "303")
POOLED_HEADS = "c25_pooled_heads"

#: §C25.0.7.  (property, probe point, lambda, the seed-1234 C23 arm it replicates).
ARMS = (
    ("aromatic_rings", 3, 1.0, "c23_guided_L3_lam1_aromatic_rings"),
    ("hbd_count", 4, 2.0, "c23_guided_L4_lam2_hbd_count"),
    ("qed", 4, 1.0, "c23_guided_L4_lam1_qed"),
)
HEAD_SEEDS = (2345, 3456)


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def run(cmd: list[str], label: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[C25] {label} failed with exit code {r.returncode}")
    print(f"[C25] {label} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["guided", "analysis", "all"])
    args = ap.parse_args()
    stages = ["guided", "analysis"] if args.stage == "all" else [args.stage]

    plan = []
    for prop, layer, lam, ref in ARMS:
        for hs in HEAD_SEEDS:
            hf = OUT / POOLED_HEADS / f"head_{prop}_last1_L{layer}_seed{hs}.pt"
            if not hf.exists():
                raise SystemExit(
                    f"[C25] missing {hf} -- run scripts/20_pooled_sweep.py first")
            name = f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}"
            plan.append((prop, layer, lam, hs, name, str(hf.relative_to(OUT)), ref))

    if "guided" in stages:
        for prop, layer, lam, hs, name, hf, _ref in plan:
            g = f"{name}_guided"
            if (OUT / g / "guidance_metrics.json").exists():
                print(f"[C25] skip {g} (already complete)", flush=True)
                continue
            run([PY, "scripts/05_guided_generation.py",
                 "--dataset", DATASET, "--heads", HEADS, "--property", prop,
                 "--lam", f"{lam:g}", "--layer", str(layer), "--head-file", hf,
                 "--conditions", "unguided", "throughout",
                 "--seeds", *SEEDS, "--out", g], g)

    if "analysis" in stages:
        for prop, layer, lam, hs, name, _hf, _ref in plan:
            g = f"{name}_guided"
            if not (OUT / g / "guidance_metrics.json").exists():
                continue
            q = f"{name}_quality"
            if not (OUT / q / "quality_metrics.json").exists():
                run([PY, "scripts/10_quality_analysis.py", "--dataset", DATASET,
                     "--property", prop, "--guided", g, "--out", q], q)
            b = f"{name}_bestofn"
            if not (OUT / b / "bestofn_metrics.json").exists():
                run([PY, "scripts/06_best_of_n.py", "--dataset", DATASET,
                     "--property", prop, "--guided", g,
                     "--accounting", "actual", "--out", b], b)

    print("\n[C25] head-seed replication stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
