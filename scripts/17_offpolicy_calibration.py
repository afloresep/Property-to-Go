"""C18 step 1 -- RE-MEASURE the off-policy miscalibration, then calibrate post hoc.

`docs/TODO.md` C18 is explicit that the premise has to be re-established before it is
fixed: the pilot reported the head under-confident by a factor of 3.5 on guided
prefixes (`pilot_report.md` §8.2, §9.2.1), and §11.6 then showed that roughly half of
that factor was the interval-mask defect in our own code rather than distribution
shift.  The phase-2 heads have that defect fixed, so the factor has to be measured
again on them before any conclusion rests on it.

What this runs, per property:

  1. **on-policy**  the head's target probability against the realised rate on the
     phase-2 dataset's held-out *base-policy* prefixes.  Free -- `hidden.npy` and
     `prefix_meta.csv` are already on disk.
  2. **off-policy** the same two quantities on prefixes GUIDANCE VISITS, generated
     fresh at lambda = 1 with the phase-2 head, prefixes drawn one per position
     quartile exactly as `scripts/02` and `scripts/08` draw them.
  3. **post-hoc calibration**, fitted on half the guided molecules and scored on the
     other half, split by canonical molecule so no molecule's prefixes straddle the
     two halves.  Platt, isotonic, and a bin-logit temperature.

**This is not DAgger.**  The one permitted data-aggregation round is spent
(§9.2.1) and is not repeated: no head is retrained here, no guided prefix enters any
training set.  A calibrator is fitted on the head's *outputs*.

Every calibrator is also fitted with a power law `c * q**alpha` and the residual is
reported, because `calibration.py` shows a power map is *exactly* a rescale of lambda
-- so a calibrator that is well approximated by one is a point on the lambda sweep
already reported in §19 rather than a new experiment.

    .venv/bin/python scripts/17_offpolicy_calibration.py \
        --dataset pilot_50k_p2 --heads pilot_50k_heads_p2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go import calibration as C, generation  # noqa: E402
from property_to_go.binning import binner_from_dict, interval_probability  # noqa: E402
from property_to_go.compute import ComputeMeter  # noqa: E402
from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, load_config, read_json, write_json, write_run_context,
)
from property_to_go.guidance import TargetScorer, Windows, guided_sample  # noqa: E402
from property_to_go.heads import MLPHead  # noqa: E402
from property_to_go.model_io import load_generator  # noqa: E402
from property_to_go.prefixes import select_quartile_prefixes  # noqa: E402
from property_to_go.properties import LOCALITY_BATTERY, compute_all_properties  # noqa: E402
from property_to_go.splits import split_by_group  # noqa: E402


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, binner_from_dict(ck["binner"])


def bin_logits(head: MLPHead, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Pre-softmax bin logits, on whatever device the head currently lives on.

    `TargetScorer` migrates the head to the generator's device the first time it is
    called, so a head that has just steered a guided run is on CUDA while
    `MLPHead.predict_proba` builds CPU tensors.  Reading the device rather than
    assuming it is the difference between a result and a crash halfway through.
    """
    device = next(head.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32).to(device)
            out.append(head(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, head.n_bins))


def q_from_logits(logits: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, mask].sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pilot_50k_p2")
    ap.add_argument("--heads", default="pilot_50k_heads_p2")
    ap.add_argument("--properties", nargs="*", default=list(LOCALITY_BATTERY))
    ap.add_argument("--n-guided-molecules", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=91234)
    ap.add_argument("--out", default="c18_offpolicy_calibration")
    args = ap.parse_args()

    data_dir = OUTPUT_DIR / args.dataset
    heads_dir = OUTPUT_DIR / args.heads
    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model")
    policy = load_config("base_policy")
    gcfg = load_config("guidance")
    cfg = load_config(args.dataset if (Path("configs") / f"{args.dataset}.yaml").exists()
                      else "pilot_50k")
    intervals = read_json(data_dir / "target_intervals.json")
    win_d = read_json(data_dir / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])

    import pandas as pd

    meta = pd.read_csv(data_dir / "prefix_meta.csv")
    hidden = np.load(data_dir / "hidden.npy")
    test_mask = meta["split"].to_numpy() == "test"

    gen = load_generator(model_cfg)
    t_start = time.perf_counter()

    report: dict = {
        "dataset": args.dataset,
        "heads_dir": args.heads,
        "lambda": float(gcfg["lam"]),
        "eps": float(gcfg["eps"]),
        "top_k": int(gcfg["top_k_candidates"]),
        "n_guided_molecules_requested": args.n_guided_molecules,
        "generation_seed": args.seed,
        "hidden_layer": int(cfg["hidden_layer"]),
        "note": (
            "Post-hoc calibration only. No head is retrained and no guided prefix "
            "enters any training set, so this is NOT a second DAgger round "
            "(pilot_report.md section 9.2.1)."
        ),
        "properties": {},
    }
    arrays: dict[str, np.ndarray] = {}

    for prop in args.properties:
        iv = intervals[prop]
        lo, hi = float(iv["lo"]), float(iv["hi"])
        head, binner = load_head(heads_dir / f"head_{prop}_frozen_state.pt")
        mask = np.asarray(binner.interval_mask(lo, hi), dtype=bool)

        # ---- 1. on-policy: held-out base-policy prefixes ----------------------
        y_base = meta[prop].to_numpy().astype(np.float64)
        keep = test_mask & np.isfinite(y_base)
        x_base = hidden[keep]
        q_base = interval_probability(head.predict_proba(x_base), binner, lo, hi)
        hit_base = C.hits_for(y_base[keep], lo, hi)
        on_policy = C.calibration_report(q_base, hit_base)

        # ---- 2. off-policy: prefixes guidance actually visits ------------------
        scorer = TargetScorer(head, binner, lo, hi)
        meter = ComputeMeter().start()
        seqs = guided_sample(
            gen, scorer=scorer, window_fn=windows.fn("throughout"), policy=policy,
            n_molecules=args.n_guided_molecules, seed=args.seed,
            top_k=int(gcfg["top_k_candidates"]), lam=float(gcfg["lam"]),
            eps=float(gcfg["eps"]), backend=gcfg["candidate_backend"],
            batch_size=int(gcfg["batch_size"]), meter=meter,
        )
        meter.stop()

        rng = np.random.default_rng(args.seed + 1)
        keep_seqs, positions, targets, groups, quarts = [], [], [], [], []
        for ids, smi in zip(seqs, gen.decode(seqs)):
            props = compute_all_properties(smi)
            content = generation.sequence_content(ids, gen.bos_id, gen.eos_id, gen.pad_id)
            if props is None or len(content) < int(cfg["min_content_tokens"]):
                continue
            v = props.get(prop)
            if v is None or not np.isfinite(float(v)):
                continue
            picks = select_quartile_prefixes(len(content), rng)
            keep_seqs.append(ids)
            positions.append([k for _, k in picks])
            for qz, _ in picks:
                targets.append(float(v))
                groups.append(props["canonical_smiles"])
                quarts.append(int(qz))
        hs_meter = ComputeMeter().start()
        states = generation.hidden_states_for_positions(
            gen, keep_seqs, positions, layer=int(cfg["hidden_layer"]), meter=hs_meter
        )
        hs_meter.stop()
        x_guided = np.concatenate(states, axis=0).astype(np.float32)
        y_guided = np.asarray(targets, dtype=np.float64)
        quarts = np.asarray(quarts, dtype=np.int64)
        hit_guided = C.hits_for(y_guided, lo, hi)

        logits_guided = bin_logits(head, x_guided)
        head = head.to("cpu")  # TargetScorer left it on the generator's device
        q_guided = interval_probability(
            head.predict_proba(x_guided), binner, lo, hi
        )
        off_policy = C.calibration_report(q_guided, hit_guided)

        # ---- 3. post-hoc calibration, fitted and scored on disjoint molecules --
        # Grouped by canonical molecule so a molecule's four prefixes never straddle
        # the split: ungrouped splitting is the leak `docs/HANDOFF.md` §3.2 warns about.
        gsplit = split_by_group(groups, {"train": 0.5, "val": 0.0, "test": 0.5},
                                int(cfg["split_seed"]) + 500)
        fit_m = gsplit == "train"
        ev_m = gsplit == "test"

        platt = C.fit_platt(q_guided[fit_m], hit_guided[fit_m])
        iso = C.fit_isotonic(q_guided[fit_m], hit_guided[fit_m])

        # bin-logit temperature: the one family that is NOT a function of q alone,
        # so it is the only post-hoc route that can reorder candidates. Selected by
        # ECE on the fit half over a fixed grid, no gradient, nothing to tune.
        grid = np.concatenate([np.linspace(0.2, 1.0, 33)[:-1], np.linspace(1.0, 5.0, 41)])
        from property_to_go import metrics as M
        eces = [
            M.expected_calibration_error(
                q_from_logits(logits_guided[fit_m], mask, T), hit_guided[fit_m]
            )
            for T in grid
        ]
        T_best = float(grid[int(np.argmin(eces))])

        calibrated = {
            "uncalibrated": C.calibration_report(q_guided[ev_m], hit_guided[ev_m]),
            "platt": C.calibration_report(platt.apply(q_guided[ev_m]), hit_guided[ev_m]),
            "isotonic": C.calibration_report(iso.apply(q_guided[ev_m]), hit_guided[ev_m]),
            "bin_logit_temperature": C.calibration_report(
                q_from_logits(logits_guided[ev_m], mask, T_best), hit_guided[ev_m]
            ),
        }

        # ---- how much of each calibrator is just a lambda rescale? -------------
        gridq = np.quantile(q_guided, np.linspace(0.01, 0.99, 199))
        gridq = np.unique(gridq[gridq > 0])
        power_fits = {
            "platt": C.fit_power_approximation(platt, gridq),
            "isotonic": C.fit_power_approximation(iso, gridq),
        }
        power_fits["platt"]["equivalent_lambda_at_lam1"] = (
            platt.power_limit().equivalent_lambda(float(gcfg["lam"]))
        )
        power_fits["isotonic"]["equivalent_lambda_at_lam1"] = (
            float(gcfg["lam"]) * power_fits["isotonic"]["alpha"]
        )

        report["properties"][prop] = {
            "target_interval": iv,
            "n_bins": int(binner.n_bins),
            "n_bins_in_mask": int(mask.sum()),
            "on_policy_base_prefixes": on_policy,
            "off_policy_guided_prefixes": off_policy,
            "off_policy_over_on_policy_factor_ratio": (
                (off_policy["under_confidence_factor"] / on_policy["under_confidence_factor"])
                if on_policy["under_confidence_factor"] else None
            ),
            "n_guided_molecules_kept": int(len(keep_seqs)),
            "n_guided_prefixes": int(len(x_guided)),
            "guided_hit_rate_of_molecules": float(
                C.hits_for(np.array([t for t in y_guided[::4]]), lo, hi).mean()
            ),
            "calibration_fit_split": {
                "n_fit": int(fit_m.sum()), "n_eval": int(ev_m.sum()),
                "grouped_by": "canonical_smiles",
            },
            "post_hoc_calibrated_on_held_out_guided_prefixes": calibrated,
            "fitted_calibrators": {
                "platt": platt.to_dict(),
                "bin_logit_temperature": T_best,
                "isotonic_n_knots": int(len(iso.x)),
            },
            "power_law_approximation": power_fits,
            "off_policy_by_quartile": {
                str(qz): C.calibration_report(q_guided[quarts == qz], hit_guided[quarts == qz])
                for qz in (1, 2, 3, 4) if int((quarts == qz).sum()) > 0
            },
            "compute": {
                "guided_generation": meter.as_dict(),
                "hidden_states": hs_meter.as_dict(),
            },
        }
        write_json(out_dir / f"calibrator_{prop}.json", {
            "property": prop, "target_interval": iv,
            "platt": platt.to_dict(), "isotonic": iso.to_dict(),
            "bin_logit_temperature": T_best,
        })
        arrays[f"q_guided_{prop}"] = q_guided
        arrays[f"hit_guided_{prop}"] = hit_guided
        arrays[f"split_guided_{prop}"] = (gsplit == "test")
        arrays[f"quartile_guided_{prop}"] = quarts

        print(
            f"{prop:16s} on-policy pred {on_policy['mean_predicted']:.4f} vs "
            f"{on_policy['observed']:.4f} (x{on_policy['under_confidence_factor']:.2f}, "
            f"ECE {on_policy['ece']:.4f}) | off-policy pred "
            f"{off_policy['mean_predicted']:.4f} vs {off_policy['observed']:.4f} "
            f"(x{off_policy['under_confidence_factor']:.2f}, ECE {off_policy['ece']:.4f}, "
            f"AUROC {off_policy['auroc']:.4f}) | platt a={platt.a:.3f}"
        )

    report["wall_seconds_total"] = time.perf_counter() - t_start
    np.savez_compressed(out_dir / "calibration_arrays.npz", **arrays)
    write_json(out_dir / "offpolicy_calibration.json", report)
    write_json(out_dir / "configs_used.json",
               {"model": model_cfg, "base_policy": policy, "guidance": gcfg, "dataset": cfg})
    write_run_context(out_dir)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
