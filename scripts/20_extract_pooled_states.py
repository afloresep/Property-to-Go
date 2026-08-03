"""C25 step 1 -- extract the *window* of hidden states around each prefix position.

C17 stored one hidden state per prefix row (`outputs/c17_layer_states_<dataset>/layerL/
hidden.npy`).  A pooled readout needs the states at the neighbouring positions too, and
`prefix_meta.csv` samples only four positions per trajectory, so those neighbours do not
exist on disk and have to be re-extracted.

This script replays the **identical** prefixes of an existing dataset -- same
trajectories, same positions, same splits, same intervals -- and writes, per probe point:

    stack.npy     (n_rows, STACK_WINDOW, hidden)  the last STACK_WINDOW states, left
                                                  padded by index clamping
    mean16.npy    (n_rows, hidden)                the mean over the last 16 *distinct*
                                                  positions
    counts.npy    (n_rows,)                       distinct positions inside STACK_WINDOW
    counts16.npy  (n_rows,)                       distinct positions inside 16

Nothing in the source dataset or in any C17/C18/C23 output directory is written.

Cost, stated in tokens because wall clock does not reproduce (§11.7): one forward pass
per batch serves every probe point *and* every window size, so this processes exactly
the same `processed_tokens_actual` as script 02's single-layer, single-position
extraction.

**Validity gate.**  `stack[:, -1, :]` is the state at the prefix position itself, so it
must come back **bit-identical** to the C17 layer array (and, at probe point 12, to the
dataset's own `hidden.npy`).  The script exits non-zero if it does not, because every
pooled-versus-single comparison downstream would otherwise be measuring the re-extraction
rather than the pooling.

    .venv/bin/python scripts/20_extract_pooled_states.py --dataset pilot_50k_p2 \
        --layers 3 4 5 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import pooling as PL  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.model_io import load_generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--layers", type=int, nargs="*", default=[3, 4, 5, 12])
    ap.add_argument("--stack-window", type=int, default=PL.STACK_WINDOW)
    ap.add_argument("--wide-window", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--c17-states", default=None,
                    help="C17 layer-state dir the bit-identity gate is checked against")
    ap.add_argument("--limit-trajectories", type=int, default=None,
                    help="smoke-test escape hatch; NOT for a reported run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    c17_dir = OUTPUT_DIR / (args.c17_states or f"c17_layer_states_{args.dataset}")
    out_dir = OUTPUT_DIR / (args.out or f"c25_window_states_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    traj = read_json(data_dir / "trajectories.json")
    if args.limit_trajectories:
        traj = traj[: args.limit_trajectories]
        meta = meta[meta["traj_index"] < args.limit_trajectories].reset_index(drop=True)

    # Script 16's reconstruction, unchanged and asserted rather than assumed.
    ti = meta["traj_index"].to_numpy()
    assert np.all(np.diff(ti) >= 0), "prefix_meta is not trajectory-major"
    prefix_len = meta["prefix_len"].to_numpy()
    starts = np.searchsorted(ti, np.arange(len(traj)), side="left")
    ends = np.searchsorted(ti, np.arange(len(traj)), side="right")
    positions = [prefix_len[s:e].tolist() for s, e in zip(starts, ends)]
    sequences = [t["token_ids"] for t in traj]
    n_rows = len(meta)
    assert sum(len(p) for p in positions) == n_rows

    model_cfg = load_config("model")
    gen = load_generator(model_cfg)
    layers = list(args.layers)
    hidden_size = gen.hidden_size
    W = int(args.stack_window)
    print(f"{len(traj)} trajectories, {n_rows} prefix rows, probe points {layers}, "
          f"stack window {W}, wide window {args.wide_window}", flush=True)

    stacks: dict[int, np.ndarray] = {}
    means: dict[int, np.ndarray] = {}
    for L in layers:
        d = out_dir / f"layer{L}"
        d.mkdir(exist_ok=True)
        stacks[L] = np.lib.format.open_memmap(
            d / "stack.npy", mode="w+", dtype=np.float32, shape=(n_rows, W, hidden_size))
        means[L] = np.lib.format.open_memmap(
            d / "mean16.npy", mode="w+", dtype=np.float32, shape=(n_rows, hidden_size))
    counts = np.zeros(n_rows, dtype=np.int16)
    counts16 = np.zeros(n_rows, dtype=np.int16)

    meter = ComputeMeter().start()
    t0 = time.perf_counter()
    PL.extract_window_states(
        gen, sequences, positions, layers,
        stacks=stacks, means=means, counts_out=counts, counts_wide_out=counts16,
        row_offsets=starts.tolist(), stack_window=W, wide_window=int(args.wide_window),
        batch_size=args.batch_size, meter=meter,
    )
    meter.stop()
    for L in layers:
        stacks[L].flush()
        means[L].flush()
    np.save(out_dir / "counts.npy", counts)
    np.save(out_dir / "counts16.npy", counts16)
    print(f"extracted in {time.perf_counter() - t0:.1f}s "
          f"({meter.processed_tokens_actual} processed tokens)", flush=True)

    # ---- validity gate: the last slot must be the C17 single-position state ---------
    gate_rows = []
    ok = True
    if not args.limit_trajectories:
        for L in layers:
            ref_path = c17_dir / f"layer{L}" / "hidden.npy"
            if not ref_path.exists():
                gate_rows.append({"probe_point": int(L), "checked": False})
                continue
            ref = np.load(ref_path, mmap_mode="r")
            got = np.load(out_dir / f"layer{L}" / "stack.npy", mmap_mode="r")
            identical = True
            max_abs = 0.0
            for s in range(0, len(ref), 20000):
                a = np.asarray(ref[s:s + 20000])
                b = np.asarray(got[s:s + 20000, -1, :])
                identical &= bool(np.array_equal(a, b))
                max_abs = max(max_abs, float(np.abs(a - b).max()))
            ok &= identical
            gate_rows.append({"probe_point": int(L), "checked": True,
                              "reference": str(ref_path),
                              "bit_identical": bool(identical),
                              "max_abs_difference": max_abs})
            print(f"gate L{L}: bit_identical={identical} max_abs={max_abs:g}", flush=True)

    summary = {
        "dataset": args.dataset,
        "n_trajectories": len(traj),
        "n_prefix_rows": int(n_rows),
        "hidden_size": int(hidden_size),
        "probe_points": layers,
        "stack_window": W,
        "wide_window": int(args.wide_window),
        "batch_size": args.batch_size,
        "padding_rule": (
            "index clamping: the window at position p is max(0, p-w+1+j) for j=0..w-1, "
            "so a prefix shorter than the window repeats its earliest state. counts.npy "
            "records how many of the w slots are distinct, and every pooling operator "
            "that averages uses only those."
        ),
        "compute": meter.as_dict(),
        "processed_tokens_actual": int(meter.processed_tokens_actual),
        "processed_tokens_if_one_pass_per_layer": int(
            meter.processed_tokens_actual * len(layers)),
        "wall_seconds": time.perf_counter() - t0,
        "validity_gate": {"per_layer": gate_rows, "all_bit_identical": bool(ok)},
        "note": ("Wall time is recorded but is not the cost unit (pilot_report.md §11.7). "
                 "Every probe point and every window size comes from one forward pass."),
    }
    write_json(out_dir / "window_states_summary.json", summary)
    write_run_context(out_dir, {"model": model_cfg})

    if not ok:
        print("VALIDITY GATE FAILED -- the re-extraction does not reproduce C17's states")
        return 1
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
