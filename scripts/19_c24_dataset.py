"""C24 stage 1 -- the text dataset, its frozen target intervals, and all 13 probe points.

External-validity check; see `outputs/c24_prereg/prereg.md`.  Nothing here loads
GP-MoLFormer or writes outside `outputs/c24_*`.

    .venv/bin/python scripts/19_c24_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generality as G  # noqa: E402
from property_to_go.binning import (  # noqa: E402
    CategoricalBinner, QuantileBinner, interval_mask_coverage,
)
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import OUTPUT_DIR, write_json, write_run_context  # noqa: E402
from property_to_go.guidance import Windows  # noqa: E402
from property_to_go.splits import check_no_group_leakage, split_by_group  # noqa: E402

CFG = {
    "n_sequences": 20000,
    "n_content": 40,
    "temperature": 1.0,
    "prefixes_per_sequence": 4,
    "generation_seed": 20240001,
    "prefix_seed": 12,
    "split_seed": 11,
    "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    "batch_size": 256,
    "state_batch_size": 128,
    "target_rate": 0.10,
    "base_rate_gate": [0.05, 0.20],
    "continuous_n_bins": 20,
    "count_cap_quantile": 0.995,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="c24_dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"c24_dataset": CFG})

    t0 = time.time()
    gen = G.load_text_generator(args.device)
    meter = ComputeMeter().start()

    print(f"[{time.time()-t0:.0f}s] sampling {CFG['n_sequences']} base sequences", flush=True)
    seqs = G.sample_base(
        gen, CFG["n_sequences"], seed=CFG["generation_seed"], n_content=CFG["n_content"],
        temperature=CFG["temperature"], batch_size=CFG["batch_size"], meter=meter,
    )
    texts = gen.decode([s[1:] for s in seqs])
    assert all(len(s) == CFG["n_content"] + 1 for s in seqs)

    values = {a: np.array([G.ATTRIBUTES[a](t) for t in texts], dtype=np.float64)
              for a in G.ATTRIBUTE_ORDER}

    # --- target intervals, by the C24.0.3 rule, before any head exists ------------
    intervals: dict[str, dict] = {}
    dropped: dict[str, dict] = {}
    for a in G.ATTRIBUTE_ORDER:
        band = G.resolve_target_band(
            values[a], integer_valued=a in G.INTEGER_ATTRIBUTES,
            target_rate=CFG["target_rate"],
        )
        lo_gate, hi_gate = CFG["base_rate_gate"]
        if not (lo_gate <= band["base_rate"] <= hi_gate):
            dropped[a] = band
            print(f"  DROPPED {a}: base rate {band['base_rate']:.4f} outside gate", flush=True)
            continue
        intervals[a] = band
        print(f"  {a}: target [{band['lo']}, {band['hi']}) base rate {band['base_rate']:.4f}"
              f"  (q {band['q_lo']}-{band['q_hi']})", flush=True)
    if len(intervals) < 2:
        raise SystemExit("C24.0.10: fewer than two attributes survive the base-rate gate")

    # --- binners ------------------------------------------------------------------
    binners: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for a, band in intervals.items():
        if a in G.INTEGER_ATTRIBUTES:
            cap = int(np.ceil(np.quantile(values[a], CFG["count_cap_quantile"])))
            cap = max(cap, int(band["hi"]))
            binner = CategoricalBinner(max_value=cap)
        else:
            binner = QuantileBinner.fit(
                values[a], CFG["continuous_n_bins"], extra_edges=(band["lo"], band["hi"])
            )
        binners[a] = binner.to_dict()
        coverage[a] = interval_mask_coverage(binner, band["lo"], band["hi"], values[a])
        assert coverage[a]["is_exact"], (a, coverage[a])

    # --- prefixes -----------------------------------------------------------------
    rng = np.random.default_rng(CFG["prefix_seed"])
    n_pref = CFG["prefixes_per_sequence"]
    edges = np.linspace(1, CFG["n_content"] - 1, n_pref + 1)
    quartiles = [(int(np.ceil(edges[i])), int(np.floor(edges[i + 1]))) for i in range(n_pref)]
    positions = np.stack(
        [np.array([rng.integers(lo, hi + 1) for lo, hi in quartiles]) for _ in range(len(seqs))]
    ).astype(np.int64)

    split_by_seq = split_by_group(texts, CFG["split_fractions"], CFG["split_seed"])
    check_no_group_leakage(np.array(texts), split_by_seq)
    row_split = np.repeat(split_by_seq, n_pref)
    row_seq = np.repeat(np.arange(len(seqs)), n_pref)
    row_pos = positions.reshape(-1)

    # trivial features and prefix texts
    prefix_texts = []
    for i, s in enumerate(seqs):
        for p in positions[i]:
            prefix_texts.append(gen.decode([s[1 : int(p) + 1]])[0])
    trivial = np.stack(
        [G.trivial_features(t, int(p)) for t, p in zip(prefix_texts, row_pos)]
    ).astype(np.float32)

    # --- hidden states at every probe point ---------------------------------------
    n_probe = gen.n_probe_points
    mm = {
        L: np.lib.format.open_memmap(
            out / f"layer{L}.npy", mode="w+", dtype=np.float32,
            shape=(len(seqs) * n_pref, gen.hidden_size),
        )
        for L in range(n_probe)
    }
    chunk = 2000
    for start in range(0, len(seqs), chunk):
        stop = min(start + chunk, len(seqs))
        st = G.all_layer_states(
            gen, seqs[start:stop], positions[start:stop],
            batch_size=CFG["state_batch_size"], meter=meter,
        )  # (b, n_pref, n_probe, H)
        b = stop - start
        st = st.reshape(b * n_pref, n_probe, gen.hidden_size)
        for L in range(n_probe):
            mm[L][start * n_pref : stop * n_pref] = st[:, L, :]
        print(f"[{time.time()-t0:.0f}s] states {stop}/{len(seqs)}", flush=True)
    for L in range(n_probe):
        mm[L].flush()
    meter.stop()

    windows = Windows.from_lengths(
        np.full(len(seqs), CFG["n_content"], dtype=np.int64), (1 / 3, 2 / 3)
    )

    np.savez_compressed(
        out / "rows.npz",
        row_seq=row_seq, row_pos=row_pos, row_split=row_split.astype("U8"),
        trivial=trivial,
        **{f"value_{a}": np.repeat(values[a], n_pref) for a in G.ATTRIBUTE_ORDER},
    )
    write_json(out / "sequences.json", {
        "token_ids": seqs, "texts": texts,
        "split": split_by_seq.tolist(), "positions": positions.tolist(),
    })
    write_json(out / "target_intervals.json", {
        "rule": "C24.0.3 quantile-band search, target base rate 0.10, grid 0.05",
        "intervals": intervals, "dropped": dropped,
        "binners": binners, "interval_mask_coverage": coverage,
        "base_rate_gate": CFG["base_rate_gate"],
    })
    write_json(out / "windows.json", windows.to_dict())
    write_json(out / "dataset_metrics.json", {
        "n_sequences": len(seqs), "n_prefix_rows": int(len(row_seq)),
        "n_probe_points": n_probe, "hidden_size": gen.hidden_size,
        "prefix_quartiles": quartiles,
        "split_counts": {s: int((split_by_seq == s).sum()) for s in ("train", "val", "test")},
        "row_split_counts": {s: int((row_split == s).sum()) for s in ("train", "val", "test")},
        "attribute_summary": {
            a: {"mean": float(values[a].mean()), "std": float(values[a].std()),
                "quantiles": {str(q): float(np.quantile(values[a], q))
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
            for a in G.ATTRIBUTE_ORDER
        },
        "unique_texts": int(len(set(texts))),
        "generator_fingerprint": gen.fingerprint(),
        "compute": meter.as_dict(),
    })
    print(f"[{time.time()-t0:.0f}s] done; {meter.processed_tokens_actual} processed tokens",
          flush=True)


if __name__ == "__main__":
    main()
