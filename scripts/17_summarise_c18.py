"""C18 -- assemble every arm into one artefact the section's tables are read from.

Each run writes its own JSON; this collects them so the end-to-end table is a single
artefact rather than a hand-joined set of twelve, which is the failure mode
`tests/test_report_matches_artifacts.py` exists to catch.

The published lambda = 1 numbers are read from the central test's own directories, not
retyped, so "the calibrated arm against the deployed arm" is a comparison between two
artefacts and never between an artefact and a memory.

    .venv/bin/python scripts/17_summarise_c18.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from property_to_go.config import (  # noqa: E402
    OUTPUT_DIR, read_json, write_json, write_run_context,
)

CAL_ARMS = ("uncalibrated", "platt", "isotonic", "bin_temperature", "binT0p4")
HEAD_ARMS = ("wide", "focused", "wide_focused")


def guided_summary(d: Path, prop: str) -> dict | None:
    f = d / "guidance_metrics.json"
    if not f.exists():
        return None
    g = read_json(f)
    c = g["conditions"]
    ung = c["unguided"]["aggregate"]["hit_rate"]
    thr = c["throughout"]["aggregate"]["hit_rate"]
    return {
        "dir": d.name,
        "hit_rate_unguided": ung["mean"],
        "hit_rate_throughout": thr["mean"],
        "hit_rate_throughout_sd": float(np.std(thr["values"], ddof=1)),
        "lift": thr["mean"] - ung["mean"],
        "validity_throughout": c["throughout"]["aggregate"]["validity"]["mean"],
        "uniqueness_throughout": c["throughout"]["aggregate"]["uniqueness"]["mean"],
        "content_length_throughout": c["throughout"]["aggregate"]["content_length_mean"]["mean"],
        "processed_tokens_actual": (
            c["throughout"]["aggregate"]["compute_total"]["processed_tokens_actual"]
        ),
        "tokens_per_molecule_actual": (
            c["throughout"]["aggregate"]["compute_total"]["tokens_per_molecule_actual"]
        ),
        "seeds": thr["values"],
    }


def bestofn_summary(d: Path) -> dict | None:
    f = d / "bestofn_metrics.json"
    if not f.exists():
        return None
    b = read_json(f)
    m = b["matches"].get("actual")
    if m is None:
        return None
    return {
        "dir": d.name,
        "n_candidates": m["n_candidates"],
        "hit_rate": m["aggregate"]["hit_rate"]["mean"],
        "tokens_per_molecule_actual": m["aggregate"]["tokens_per_molecule_actual"],
    }


def matched_baseline(matched: dict, prop: str, arm: str) -> dict | None:
    """The compute-matched best-of-N for one arm, from the shared-N table.

    `17_matched_best_of_n.py` groups arms by their solved N and runs script 06 once per
    distinct N, because that run is deterministic in (property, N, seeds).
    """
    e = matched.get("properties", {}).get(prop, {}).get("arms", {}).get(arm)
    if e is None or "best_of_n_hit_rate" not in e:
        return None
    return {
        "dir": e["best_of_n_dir"],
        "n_candidates": e["n_candidates_solved"],
        "hit_rate": e["best_of_n_hit_rate"],
        "realised_token_ratio": e["realised_token_ratio"],
        "shared_run": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--properties", nargs="*",
                    default=["aromatic_rings", "hbd_count", "clogp"])
    ap.add_argument("--out", default="c18_summary")
    args = ap.parse_args()

    out_dir = OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = OUTPUT_DIR / "c18_matched_best_of_n" / "matched_best_of_n.json"
    matched = read_json(mf) if mf.exists() else {}
    report: dict = {
        "properties": {},
        "best_of_n_source": "c18_matched_best_of_n/matched_best_of_n.json",
        "framing": (
            "Per-position gains live in c18_per_position/. They are NOT end-to-end "
            "quantities (docs/TODO.md C22.1). The table here is the end-to-end "
            "measurement and is the only one from which an end-to-end claim may be "
            "made."
        ),
    }

    for prop in args.properties:
        entry: dict = {"arms": {}}
        # the deployed lambda = 1 run from the central test, read not retyped
        pub = guided_summary(OUTPUT_DIR / f"pilot_50k_p2_guided_{prop}", prop)
        pub_bo = bestofn_summary(OUTPUT_DIR / f"pilot_50k_p2_bestofn_{prop}")
        entry["published_lambda1"] = pub
        entry["published_lambda1_best_of_n"] = pub_bo
        if pub and pub_bo:
            entry["published_lambda1_advantage"] = (
                pub["hit_rate_throughout"] - pub_bo["hit_rate"]
            )

        for arm in CAL_ARMS:
            g = guided_summary(OUTPUT_DIR / f"c18_guided_{arm}_{prop}", prop)
            if g is None:
                continue
            b = matched_baseline(matched, prop, arm)
            entry["arms"][arm] = {
                "route": "post-hoc calibration", **g,
                "best_of_n": b,
                "guidance_advantage": (g["hit_rate_throughout"] - b["hit_rate"]) if b else None,
            }
        for arm in HEAD_ARMS:
            g = guided_summary(OUTPUT_DIR / f"c18_guided_head_{arm}_{prop}", prop)
            if g is None:
                continue
            b = matched_baseline(matched, prop, f"head_{arm}")
            entry["arms"][f"head_{arm}"] = {
                "route": "retrained readout", **g,
                "best_of_n": b,
                "guidance_advantage": (g["hit_rate_throughout"] - b["hit_rate"]) if b else None,
            }

        if entry["arms"]:
            best = max(entry["arms"].items(),
                       key=lambda kv: kv[1]["hit_rate_throughout"])
            entry["best_arm"] = best[0]
            entry["best_arm_lift"] = best[1]["lift"]
            entry["best_arm_advantage_over_best_of_n"] = best[1]["guidance_advantage"]
            entry["any_arm_beats_compute_matched_best_of_n"] = bool(
                any(a["guidance_advantage"] is not None and a["guidance_advantage"] > 0
                    for a in entry["arms"].values())
            )
        report["properties"][prop] = entry

    report["any_arm_anywhere_beats_compute_matched_best_of_n"] = bool(
        any(e.get("any_arm_beats_compute_matched_best_of_n") for e in report["properties"].values())
    )
    write_json(out_dir / "c18_summary.json", report)
    write_run_context(out_dir)

    for prop, e in report["properties"].items():
        pub = e.get("published_lambda1")
        if pub:
            print(f"## {prop}  published lam=1 throughout={pub['hit_rate_throughout']:.4f} "
                  f"lift={pub['lift']:+.4f} advantage="
                  f"{e.get('published_lambda1_advantage', float('nan')):+.4f}")
        for arm, a in e["arms"].items():
            adv = a["guidance_advantage"]
            print(f"   {arm:20s} throughout={a['hit_rate_throughout']:.4f} "
                  f"lift={a['lift']:+.4f} val={a['validity_throughout']:.3f} "
                  f"advantage={'n/a' if adv is None else f'{adv:+.4f}'}")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
