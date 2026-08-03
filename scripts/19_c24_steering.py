"""C24 stage 3 -- the PER-POSITION steering value of every probe point.

The text-domain replication of `pilot_report.md` §21.5 (C17 question 2).  At one
decoding position, with the rest of the sequence left to the base policy:

    our_head_gain(L) = E_i [ w_guided(base_lp_i, q_i^L) . p_hit_i  -  w_base(base_lp_i) . p_hit_i ]

where `p_hit_i[j]` is the *true* P(final attribute in target | prefix_i + candidate_j),
estimated by base-policy rollouts, and the weightings are `headroom.candidate_weights`
and `headroom.guided_weights` -- the molecular functions, imported.

**This is a per-position quantity and must never be quoted as an end-to-end one**
(`docs/TODO.md` C22.1).  Stage 5 measures end to end.

    .venv/bin/python scripts/19_c24_steering.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generality as G  # noqa: E402
from property_to_go import headroom as H  # noqa: E402
from property_to_go.binning import binner_from_dict, in_interval, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402

CFG = {
    "n_prefixes": 300,
    "top_k": 8,
    "n_rollouts": 32,
    "rollout_seed": 7777,
    "prefix_sample_seed": 7778,
    "lam": 1.0,
    "eps": 1e-6,
    "head_seed": 1234,
}


def load_head(path: Path, device):
    d = torch.load(path, map_location="cpu")
    head = MLPHead(d["in_dim"], d["hidden_dim"], d["n_bins"], d["dropout"])
    head.load_state_dict(d["state_dict"])
    return head.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c24_dataset")
    ap.add_argument("--heads", default="c24_probe_layers")
    ap.add_argument("--out", default="c24_layer_steering")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = OUTPUT_DIR / args.dataset
    hd = OUTPUT_DIR / args.heads
    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"c24_steering": CFG, "dataset": args.dataset, "heads": args.heads})

    ti = read_json(ds / "target_intervals.json")
    meta = read_json(ds / "dataset_metrics.json")
    seqs = read_json(ds / "sequences.json")
    n_probe = int(meta["n_probe_points"])
    n_content = 40
    t0 = time.time()

    # --- the prefix sample: test-split sequences, positions as the dataset drew them
    split = np.array(seqs["split"])
    positions = np.array(seqs["positions"])
    token_ids = seqs["token_ids"]
    test_idx = np.flatnonzero(split == "test")
    rng = np.random.default_rng(CFG["prefix_sample_seed"])
    pick_seq = rng.choice(test_idx, size=CFG["n_prefixes"], replace=False)
    pick_pos = np.array([positions[i][rng.integers(0, positions.shape[1])] for i in pick_seq])
    prefixes = [list(token_ids[i][: int(p) + 1]) for i, p in zip(pick_seq, pick_pos)]

    gen = G.load_text_generator(args.device)
    meter = ComputeMeter().start()

    print(f"[{time.time()-t0:.0f}s] top-k candidates for {len(prefixes)} prefixes", flush=True)
    cand_ids, cand_lp, cand_states = G.top_k_candidates(
        gen, prefixes, CFG["top_k"], meter=meter
    )

    # --- rollouts: the true p_hit per candidate ---------------------------------
    ext = []
    for i, p in enumerate(prefixes):
        for j in range(CFG["top_k"]):
            ext.extend([p + [int(cand_ids[i, j])]] * CFG["n_rollouts"])
    print(f"[{time.time()-t0:.0f}s] {len(ext)} rollouts", flush=True)
    rolls = G.continue_prefixes(
        gen, ext, n_content=n_content, seed=CFG["rollout_seed"], meter=meter
    )
    texts = gen.decode([r[1:] for r in rolls])
    meter.stop()

    n, k, R = len(prefixes), CFG["top_k"], CFG["n_rollouts"]
    results: dict[str, dict] = {}
    for attr, band in ti["intervals"].items():
        lo, hi = float(band["lo"]), float(band["hi"])
        fn = G.ATTRIBUTES[attr]
        vals = np.array([fn(t) for t in texts], dtype=np.float64).reshape(n, k, R)
        p_hit = ((vals >= lo) & (vals < hi)).mean(axis=2)  # (n, k)

        w_base = H.candidate_weights(cand_lp)
        base_value = float((w_base * p_hit).sum(axis=1).mean())
        oracle_w = H.guided_weights(cand_lp, p_hit, CFG["lam"], CFG["eps"])
        oracle_value = float((oracle_w * p_hit).sum(axis=1).mean())
        ceiling_value = float(p_hit.max(axis=1).mean())

        binner = binner_from_dict(ti["binners"][attr])
        per_layer = {}
        for L in range(n_probe):
            head = load_head(hd / "heads" / f"{attr}_L{L}_seed{CFG['head_seed']}.pt", args.device)
            probs = G.predict_proba_on_device(
                head, cand_states[:, :, L, :].reshape(n * k, -1), args.device
            )
            q = interval_probability(probs, binner, lo, hi).reshape(n, k)
            w = H.guided_weights(cand_lp, q, CFG["lam"], CFG["eps"])
            value = float((w * p_hit).sum(axis=1).mean())
            per_layer[str(L)] = {
                "our_head_value": value,
                "our_head_gain": value - base_value,
                "share_of_oracle_gain": ((value - base_value) / (oracle_value - base_value))
                if oracle_value > base_value else None,
                "mean_q": float(q.mean()),
                "q_spread_mean": float((q.max(axis=1) - q.min(axis=1)).mean()),
                "picks_the_best_candidate_rate": float(
                    (q.argmax(axis=1) == p_hit.argmax(axis=1)).mean()
                ),
            }
        results[attr] = {
            "target": band,
            "base_value": base_value,
            "oracle_head_value": oracle_value,
            "oracle_head_gain": oracle_value - base_value,
            "hard_ceiling_value": ceiling_value,
            "hard_ceiling_gain": ceiling_value - base_value,
            "mean_p_hit": float(p_hit.mean()),
            "per_probe_point": per_layer,
        }
        print(f"[{time.time()-t0:.0f}s] {attr}: base {base_value:.4f} oracle {oracle_value:.4f}"
              f" L12 gain {per_layer[str(n_probe-1)]['our_head_gain']:+.5f}", flush=True)

    np.savez_compressed(out / "steering_arrays.npz", cand_ids=cand_ids, cand_lp=cand_lp,
                        pick_seq=pick_seq, pick_pos=pick_pos)
    write_json(out / "layer_steering_metrics.json", {
        "config": CFG, "n_prefixes": n, "attributes": results,
        "compute": meter.as_dict(),
    })
    print(f"[{time.time()-t0:.0f}s] done; {meter.processed_tokens_actual} processed tokens",
          flush=True)


if __name__ == "__main__":
    main()
