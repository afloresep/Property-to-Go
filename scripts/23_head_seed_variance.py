"""C29 -- head-seed replication at n >= 6, and the effective-lambda control.

Two experiments in one directory family because they defend the same claim: that C23's
Rule A -- a mid-network head improves guided generation -- is (a) measured against a
head-seed variance estimated from three seeds and (b) confounded with an effective-lambda
increase that C17 already measured and nobody controlled.

This is a **driver**.  Every head comes from the unmodified `scripts/20_pooled_sweep.py`
(`--variants last1`, which §C25.0.1 established is the identical feature array through the
identical trainer as script 03's `frozen_state` branch).  Every molecule comes from the
unmodified `scripts/05_guided_generation.py` through its C23 `--layer` / `--head-file`
arguments, every matched baseline from `scripts/06_best_of_n.py`, every quality number from
`scripts/10_quality_analysis.py`.  No logic is forked and no existing directory is touched.

The arms, the probe points, the lambdas, the head-seed list and its truncation order are
`outputs/c29_prereg/C29.0_preregistration.md` §C29.0.2, transcribed rather than derived.

Every cell is its own output directory and every stage skips a completed cell, so a kill
costs at most one cell.  Completion is appended to `outputs/c29_progress.jsonl` as it
happens, so a resumed run can see what the killed one achieved.

    setsid nohup .venv/bin/python scripts/23_head_seed_variance.py \
        > outputs/c29_run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = str(ROOT / ".venv" / "bin" / "python")

DATASET = "pilot_50k_p2"
HEADS = "pilot_50k_heads_p2"
SEEDS = ("101", "202", "303")
C29_HEADS = "c29_heads"
PROGRESS = OUT / "c29_progress.jsonl"

#: §C29.0.2 -- the head-seed list IS the truncation order.  The pre-registered minimum is 6.
HEAD_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901)
#: Seeds whose cells already exist and are reused rather than regenerated (subject to G1/G3).
PUBLISHED_MID_SEEDS = (1234, 2345, 3456)

#: §C29.0.2 -- the mid-network family.  (key, property, probe point, lambda).
MID_ARMS = (
    ("A1", "hbd_count", 4, 2.0),
    ("A2", "aromatic_rings", 3, 1.0),
    ("A3", "qed", 4, 1.0),
)
#: §C29.0.2 -- the deployed family, probe point 12, lambda = 1, one per anchor.
ANCHORS = ("qed", "hbd_count", "aromatic_rings")   # cheapest first: truncation order
DEPLOYED_PROBE_POINT = 12

#: §C29.0.2 -- the new deployed lambda points for the envelope.
ENVELOPE_LAMBDAS = (1.25, 1.5, 2.5)


def lam_tag(lam: float) -> str:
    return "lam" + f"{lam:g}".replace(".", "p")


def mid_head_file(prop: str, layer: int, hs: int) -> str:
    return f"{C29_HEADS}/head_{prop}_last1_L{layer}_seed{hs}.pt"


def mid_dirs(prop: str, layer: int, lam: float, hs: int) -> tuple[str, str, str]:
    """(guided, bestofn, quality) directory names for a mid-arm cell.

    Head seeds 1234 / 2345 / 3456 already exist as the published C23 and C25 runs and are
    reused; §C29.0.5 G1 and G3 make that reuse an identity rather than an assumption.
    """
    if hs == 1234:
        base = f"c23_{{k}}_L{layer}_{lam_tag(lam)}_{prop}"
        return (base.format(k="guided"), base.format(k="bestofn"),
                base.format(k="quality"))
    if hs in (2345, 3456):
        base = f"c25_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}"
        return f"{base}_guided", f"{base}_bestofn", f"{base}_quality"
    base = f"c29_hs{hs}_L{layer}_{lam_tag(lam)}_{prop}"
    return f"{base}_guided", f"{base}_bestofn", f"{base}_quality"


def deployed_dir(prop: str, hs: int) -> str:
    """The deployed probe-point-12 lambda=1 cell for one head seed."""
    if hs == 1234:
        return f"pilot_50k_p2_guided_{prop}"
    return f"c29_dep_hs{hs}_lam1_{prop}_guided"


def envelope_dir(prop: str, lam: float) -> str:
    return f"c29_deplam{lam_tag(lam)[3:]}_guided_{prop}"


def note(event: str, **kw) -> None:
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    with PROGRESS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[C29] {event} {kw}", flush=True)


#: The GPU is shared with a concurrent experiment, so a cell can lose a race for memory
#: that it would win a minute later.  A CUDA OOM is a scheduling accident, not a result;
#: retrying it changes no measured quantity because every cell is deterministic in its
#: seeds.  Three attempts, then the stage stops loudly rather than skipping the cell.
RETRIES = 3
RETRY_BACKOFF_SECONDS = 180


def run(cmd: list[str], label: str) -> None:
    for attempt in range(1, RETRIES + 1):
        print(f"\n$ {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT)
        dt = time.perf_counter() - t0
        if r.returncode == 0:
            note("done", cell=label, seconds=round(dt),
                 **({"attempts": attempt} if attempt > 1 else {}))
            return
        note("attempt_failed", cell=label, returncode=r.returncode,
             seconds=round(dt), attempt=attempt)
        if attempt < RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS)
    note("FAILED", cell=label, attempts=RETRIES)
    raise SystemExit(f"[C29] {label} failed {RETRIES} times")


# ------------------------------------------------------------------ unit runners

def guided(prop: str, layer: int | None, lam: float, out: str,
           head_file: str | None) -> bool:
    """One guided run.  Returns True if it was executed, False if already complete."""
    if (OUT / out / "guidance_metrics.json").exists():
        print(f"[C29] skip {out} (already complete)", flush=True)
        return False
    cmd = [PY, "scripts/05_guided_generation.py",
           "--dataset", DATASET, "--heads", HEADS, "--property", prop,
           "--lam", f"{lam:g}"]
    if layer is not None:
        cmd += ["--layer", str(layer)]
    if head_file is not None:
        cmd += ["--head-file", head_file]
    cmd += ["--conditions", "unguided", "throughout", "--seeds", *SEEDS, "--out", out]
    run(cmd, out)
    return True


def best_of_n(prop: str, guided_out: str, out: str) -> None:
    if (OUT / out / "bestofn_metrics.json").exists():
        print(f"[C29] skip {out} (already complete)", flush=True)
        return
    if not (OUT / guided_out / "guidance_metrics.json").exists():
        return
    run([PY, "scripts/06_best_of_n.py", "--dataset", DATASET, "--property", prop,
         "--guided", guided_out, "--accounting", "actual", "--out", out], out)


def quality(prop: str, guided_out: str, out: str) -> None:
    if (OUT / out / "quality_metrics.json").exists():
        print(f"[C29] skip {out} (already complete)", flush=True)
        return
    if not (OUT / guided_out / "guidance_metrics.json").exists():
        return
    run([PY, "scripts/10_quality_analysis.py", "--dataset", DATASET, "--property", prop,
         "--guided", guided_out, "--out", out], out)


# ------------------------------------------------------------------ stages

def stage_heads(threads: int | None) -> None:
    """§C29.0.2 -- one training route for every C29 head, at every depth."""
    cmd = [PY, "scripts/20_pooled_sweep.py",
           "--dataset", DATASET, "--out", C29_HEADS,
           "--variants", "last1", "--depths", "mid", "final",
           "--properties", "aromatic_rings", "hbd_count", "qed",
           "--head-seeds", *[str(s) for s in HEAD_SEEDS]]
    if threads:
        cmd += ["--threads", str(threads)]
    run(cmd, C29_HEADS)


def stage_gate() -> None:
    """§C29.0.5 G2 -- the two cheap end-to-end identity runs, at head seed 1234."""
    guided("qed", 4, 1.0, "c29_gate_L4_lam1_qed_hs1234",
           mid_head_file("qed", 4, 1234))
    guided("qed", 12, 1.0, "c29_gate_L12_lam1_qed_hs1234",
           mid_head_file("qed", 12, 1234))


def stage_p1(seeds: tuple[int, ...]) -> None:
    """Priority 1 -- the decisive arm, hbd_count probe point 4, lambda = 2."""
    _, prop, layer, lam = MID_ARMS[0]
    for hs in seeds:
        if hs in PUBLISHED_MID_SEEDS:
            continue
        g, b, q = mid_dirs(prop, layer, lam, hs)
        guided(prop, layer, lam, g, mid_head_file(prop, layer, hs))
        quality(prop, g, q)
        best_of_n(prop, g, b)


def stage_p2_mid(seeds: tuple[int, ...]) -> None:
    """Priority 2a -- the other two mid anchors, seed-major so truncation is even."""
    for hs in seeds:
        if hs in PUBLISHED_MID_SEEDS:
            continue
        for _key, prop, layer, lam in MID_ARMS[1:]:
            g, b, q = mid_dirs(prop, layer, lam, hs)
            guided(prop, layer, lam, g, mid_head_file(prop, layer, hs))
            quality(prop, g, q)
            best_of_n(prop, g, b)


def stage_p2_dep(seeds: tuple[int, ...]) -> None:
    """Priority 2b -- the deployed probe-point-12 lambda=1 arm across head seeds.

    Seed-major and cheapest-anchor-first, so a truncation leaves the three anchors at the
    same n rather than one anchor complete and two empty.  No best-of-N and no quality
    here: §C29.0.2 fixed that before any result existed.
    """
    for hs in seeds:
        if hs == 1234:
            continue                      # the published deployed run IS head seed 1234
        for prop in ANCHORS:
            guided(prop, DEPLOYED_PROBE_POINT, 1.0, deployed_dir(prop, hs),
                   mid_head_file(prop, DEPLOYED_PROBE_POINT, hs))


def stage_p4_dep_lam2_hbd(seeds: tuple[int, ...]) -> None:
    """POST-HOC, and labelled as such wherever it appears.

    §C29.0.6 R4 asks for a head-seed-paired mid-minus-deployed difference **at matched
    lambda**, but §C29.0.2 fixed the deployed family at lambda = 1 while A1 sits at
    lambda = 2.  R4 is therefore not scoreable as written on A1 -- a defect in the
    pre-registration, found when scoring it.  This stage repairs the *measurement* rather
    than the pre-registration: it runs the deployed head at lambda = 2 on hbd_count across
    the same head seeds, so the pairing exists.  The defect is still reported.
    """
    for hs in seeds:
        if hs == 1234:
            continue                  # `pilot_50k_p2_lam2_guided_hbd_count` is seed 1234
        guided("hbd_count", DEPLOYED_PROBE_POINT, 2.0,
               f"c29_dep_hs{hs}_lam2_hbd_count_guided",
               mid_head_file("hbd_count", DEPLOYED_PROBE_POINT, hs))


def stage_p3() -> None:
    """Priority 3 -- the deployed lambda envelope, filled in at 1.25 / 1.5 / 2.5.

    The **default** code path: no `--layer`, no `--head-file`, exactly as the published
    deployed lambda runs were produced, so only lambda differs (§C29.0.5 G5).
    """
    for lam in ENVELOPE_LAMBDAS:
        for prop in ANCHORS:
            guided(prop, None, lam, envelope_dir(prop, lam), None)


def stage_p1_mid_bestofn(seeds: tuple[int, ...]) -> None:
    """Backfill best-of-N / quality for any mid cell that has a guided run but no
    comparator -- e.g. after a kill between the two."""
    for _key, prop, layer, lam in MID_ARMS:
        for hs in seeds:
            g, b, q = mid_dirs(prop, layer, lam, hs)
            quality(prop, g, q)
            best_of_n(prop, g, b)


STAGES = ("heads", "gate", "p1", "p2mid", "p2dep", "p3", "p4dep2", "backfill")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--head-seeds", type=int, nargs="*", default=list(HEAD_SEEDS))
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    seeds = tuple(args.head_seeds)

    note("start", stages=list(args.stages), head_seeds=list(seeds))
    if "heads" in args.stages:
        stage_heads(args.threads)
    # Only the checkpoints the requested stages will actually load are required, so a
    # GPU stage can start while a CPU stage is still training heads for a later one.
    needed: list[tuple[str, int]] = []
    if "gate" in args.stages:
        needed += [("qed", 4), ("qed", DEPLOYED_PROBE_POINT)]
    if "p1" in args.stages:
        needed += [(MID_ARMS[0][1], MID_ARMS[0][2])]
    if {"p2mid", "backfill"} & set(args.stages):
        needed += [(prop, layer) for _k, prop, layer, _l in MID_ARMS]
    if {"p2dep", "p4dep2"} & set(args.stages):
        needed += [(prop, DEPLOYED_PROBE_POINT) for prop in ANCHORS]
    for hs in seeds:
        for prop, layer in needed:
            f = OUT / mid_head_file(prop, layer, hs)
            if not f.exists():
                raise SystemExit(f"[C29] missing head checkpoint {f}")

    if "gate" in args.stages:
        stage_gate()
    if "p1" in args.stages:
        stage_p1(seeds)
    if "p2mid" in args.stages:
        stage_p2_mid(seeds)
    if "p2dep" in args.stages:
        stage_p2_dep(seeds)
    if "p3" in args.stages:
        stage_p3()
    if "p4dep2" in args.stages:
        stage_p4_dep_lam2_hbd(seeds)
    if "backfill" in args.stages:
        stage_p1_mid_bestofn(seeds)

    note("stages complete", stages=list(args.stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
