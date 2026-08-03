"""C17 step 1 -- re-extract the prefix hidden states at EVERY probe point.

The pipeline stores one `hidden.npy` per dataset, taken at `hidden_layer: -1`.  This
script replays the *identical* prefixes of an existing dataset -- same trajectories, same
positions, same splits, same intervals -- and writes one `hidden.npy` per probe point
under `outputs/c17_layer_states_<dataset>/layer<L>/`.

**Nothing in the source dataset is read/written except reads.**  No generation happens:
the trajectories are replayed from `trajectories.json`, so the 50k sample is the phase-2
sample and every downstream head is comparable to §13's.

The efficiency claim, and why it is stated in tokens:  GP-MoLFormer returns all 13
`hidden_states` entries from one forward pass, so a 13-point sweep processes exactly the
same number of tokens as the single-point extraction script 02 already runs.  Wall time
is not the reported unit here (§11.7); `processed_tokens_actual` is, and it is written to
`layer_states_summary.json` next to the naive 13-pass alternative it avoids.

    .venv/bin/python scripts/16_extract_layer_states.py --dataset pilot_50k_p2

Validity gate: probe point 12 must come back **bit-identical** to the dataset's own
`hidden.npy`.  The script exits non-zero if it does not, because every cross-layer
comparison downstream would otherwise be measuring the replay rather than the layer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import probe_layers as P  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.model_io import load_generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=96,
                    help="script 02's default; changing it changes nothing numerically "
                         "(right padding is exact) but does change memory use")
    ap.add_argument("--limit-trajectories", type=int, default=None,
                    help="smoke-test escape hatch; NOT for a reported run")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    out_dir = OUTPUT_DIR / (args.out or f"c17_layer_states_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    traj = read_json(data_dir / "trajectories.json")
    if args.limit_trajectories:
        traj = traj[: args.limit_trajectories]
        meta = meta[meta["traj_index"] < args.limit_trajectories].reset_index(drop=True)

    # Rebuild exactly the (sequence, positions) pairs script 02 passed. `prefix_meta.csv`
    # rows are written trajectory-major in trajectory order, and `prefix_len` IS the
    # position index used for extraction, so the reconstruction is a regrouping and not
    # a re-derivation. Asserted rather than assumed.
    ti = meta["traj_index"].to_numpy()
    assert np.all(np.diff(ti) >= 0), "prefix_meta is not trajectory-major; reconstruction invalid"
    prefix_len = meta["prefix_len"].to_numpy()
    starts = np.searchsorted(ti, np.arange(len(traj)), side="left")
    ends = np.searchsorted(ti, np.arange(len(traj)), side="right")
    positions = [prefix_len[s:e].tolist() for s, e in zip(starts, ends)]
    sequences = [t["token_ids"] for t in traj]
    n_rows = len(meta)
    assert sum(len(p) for p in positions) == n_rows

    model_cfg = load_config("model")
    gen = load_generator(model_cfg)
    layers = P.probe_points(gen)
    print(f"{len(traj)} trajectories, {n_rows} prefix rows, probe points {layers}")

    hidden_size = gen.hidden_size
    out_arrays: dict[int, np.ndarray] = {}
    for L in layers:
        d = out_dir / f"layer{L}"
        d.mkdir(exist_ok=True)
        out_arrays[L] = np.lib.format.open_memmap(
            d / "hidden.npy", mode="w+", dtype=np.float32, shape=(n_rows, hidden_size)
        )

    meter = ComputeMeter().start()
    t0 = time.perf_counter()
    P.hidden_states_all_layers(
        gen, sequences, positions, layers,
        out=out_arrays, row_offsets=starts.tolist(),
        batch_size=args.batch_size, meter=meter,
    )
    meter.stop()
    for L in layers:
        out_arrays[L].flush()
    print(f"extracted {len(layers)} probe points in {time.perf_counter() - t0:.1f}s "
          f"({meter.processed_tokens_actual} processed tokens)")

    # ---- validity gate: probe point 12 must reproduce the dataset's own states -------
    gate: dict = {"checked": False}
    ref_path = data_dir / "hidden.npy"
    if ref_path.exists() and not args.limit_trajectories:
        ref = np.load(ref_path, mmap_mode="r")
        got = np.load(out_dir / f"layer{layers[-1]}" / "hidden.npy", mmap_mode="r")
        assert ref.shape == got.shape, (ref.shape, got.shape)
        max_abs = 0.0
        identical = True
        for s in range(0, len(ref), 20000):
            a = np.asarray(ref[s : s + 20000])
            b = np.asarray(got[s : s + 20000])
            identical &= bool(np.array_equal(a, b))
            max_abs = max(max_abs, float(np.abs(a - b).max()))
        gate = {
            "checked": True,
            "reference": str(ref_path.relative_to(OUTPUT_DIR.parent)),
            "probe_point": int(layers[-1]),
            "bit_identical": bool(identical),
            "max_abs_difference": max_abs,
        }
        print(f"validity gate: probe point {layers[-1]} vs {ref_path.name}: "
              f"bit_identical={identical} max_abs={max_abs:g}")

    summary = {
        "dataset": args.dataset,
        "n_trajectories": len(traj),
        "n_prefix_rows": int(n_rows),
        "hidden_size": int(hidden_size),
        "probe_points": list(layers),
        "n_probe_points": len(layers),
        "batch_size": args.batch_size,
        # The point of the whole design: 13 probe points, one forward pass.
        "compute": meter.as_dict(),
        "processed_tokens_actual": int(meter.processed_tokens_actual),
        "processed_tokens_if_one_pass_per_layer": int(
            meter.processed_tokens_actual * len(layers)
        ),
        "token_saving_factor": float(len(layers)),
        "wall_seconds": time.perf_counter() - t0,
        "validity_gate": gate,
        "note": (
            "Wall time is recorded but is not the cost unit (pilot_report.md §11.7). "
            "The 13 probe points cost the same processed tokens as one, because "
            "`output_hidden_states=True` returns every layer from the single pass."
        ),
    }
    write_json(out_dir / "layer_states_summary.json", summary)
    write_run_context(out_dir, {"model": model_cfg})

    if gate.get("checked") and not gate["bit_identical"]:
        print("VALIDITY GATE FAILED -- the replay does not reproduce hidden.npy")
        return 1
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
