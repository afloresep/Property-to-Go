"""C24 validity gate -- the cached decode path against full-prefix recomputation.

The molecular pipeline asserts that its two candidate backends are *numerically equal*
(`docs/HANDOFF.md` §1.2), because the two-way token accounting is only meaningful if
evaluating a candidate from a shared cache is the same computation as re-running its whole
prefix.  C24 makes the same check on GPT-2 and reports the measured residual rather than
claiming equality: standard attention accumulates in a different order along the two
paths, so the agreement is to float32 tolerance, not to the bit.

Checked at **every one of the 13 probe points**, on real C24 dataset sequences, and at the
positions the heads were trained on.

    .venv/bin/python scripts/19_c24_gate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import generality as G  # noqa: E402
from property_to_go.config import OUTPUT_DIR, read_json, write_json, write_run_context  # noqa: E402

CFG = {"n_sequences": 128, "positions": [5, 12, 20, 28, 36], "tolerance": 2e-3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c24_dataset")
    ap.add_argument("--out", default="c24_gate")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = OUTPUT_DIR / args.dataset
    out = OUTPUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_run_context(out, {"c24_gate": CFG})

    seqdata = read_json(ds / "sequences.json")
    seqs = [seqdata["token_ids"][i] for i in range(CFG["n_sequences"])]
    gen = G.load_text_generator(args.device)

    pos = np.array([CFG["positions"]] * len(seqs))
    train_states = G.all_layer_states(gen, seqs, pos)  # (n, n_pos, n_probe, H)

    per_probe = {L: 0.0 for L in range(gen.n_probe_points)}
    scale = float(np.abs(train_states).max())
    with torch.no_grad():
        for j, p in enumerate(CFG["positions"]):
            ids = torch.tensor([s[:p] for s in seqs], dtype=torch.long, device=gen.device)
            res = gen.model.transformer(input_ids=ids, use_cache=True, return_dict=True)
            nxt = torch.tensor([[s[p]] for s in seqs], dtype=torch.long, device=gen.device)
            cout = gen.model.transformer(
                input_ids=nxt,
                past_key_values=G.repeat_cache_gpt2(res.past_key_values, 1),
                use_cache=True, output_hidden_states=True, return_dict=True,
            )
            for L in range(gen.n_probe_points):
                dec = cout.hidden_states[L][:, 0, :].float().cpu().numpy()
                d = float(np.abs(dec - train_states[:, j, L, :]).max())
                per_probe[L] = max(per_probe[L], d)

    worst = max(per_probe.values())
    result = {
        "config": CFG,
        "n_sequences": len(seqs),
        "max_abs_difference_by_probe_point": {str(L): v for L, v in per_probe.items()},
        "max_abs_difference": worst,
        "hidden_state_max_abs_value": scale,
        "relative_to_state_scale": worst / scale,
        "bit_identical": bool(worst == 0.0),
        "within_tolerance": bool(worst <= CFG["tolerance"]),
        "note": "Not bit-identical, unlike the molecular linear-attention backends: "
                "standard attention reduces in a different order on the cached and the "
                "full-recompute paths. The residual is reported, not asserted away.",
        "generator_fingerprint": gen.fingerprint(),
    }
    write_json(out / "gate.json", result)
    print(f"max abs difference {worst:.3e} over 13 probe points; state scale {scale:.2f}")
    if not result["within_tolerance"]:
        raise SystemExit("C24 validity gate FAILED")


if __name__ == "__main__":
    main()
