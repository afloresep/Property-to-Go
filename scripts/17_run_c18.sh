#!/usr/bin/env bash
# C18 -- the end-to-end runs, in the order they must happen.
#
# Stages are separate so a failure does not cost the whole chain, exactly as
# scripts/run_phase2.sh is organised.  Every stage writes into outputs/c18_* and
# overwrites nothing that already exists.
#
#   bash scripts/17_run_c18.sh <stage>
#
# stages: prediction offpolicy heads perposition e2e_calibrated e2e_heads bestofn
#         identity summary all
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ANCHORS=${ANCHORS:-"aromatic_rings hbd_count clogp"}
# Only the readouts that ever improved per-position capture on an anchor are taken to
# the end-to-end stage; `focused` improved none of the three and is reported at the
# per-position stage only. Override with HEAD_VARIANTS to run more.
HEAD_VARIANTS=${HEAD_VARIANTS:-"wide wide_focused"}
stage=${1:-all}

run_prediction() { $PY scripts/17_prediction.py; }

run_offpolicy() { $PY scripts/17_offpolicy_calibration.py; }

run_heads() { $PY scripts/17_train_head_variants.py; }

run_perposition() {
  $PY scripts/17_per_position_capture.py \
      --variants baseline c18_heads_wide c18_heads_focused c18_heads_wide_focused \
      --out c18_per_position
}

# Post-hoc calibration, end to end.  `uncalibrated` is the within-script control:
# `guided_sample` reseeds at every call, so it must reproduce the central test's own
# `throughout` hit rate exactly, and if it does not, nothing else in this block counts.
#
# Best-of-N is NOT run per arm.  It is deterministic in (property, N, seeds) and every
# arm here solves to the same N, so `17_matched_best_of_n.py` runs it once per distinct
# N and records which arms share it.  Running it per arm would have added 9.83M processed
# tokens, 75% of the whole guided budget, to recompute three numbers.
run_e2e_calibrated() {
  for prop in $ANCHORS; do
    for arm in uncalibrated platt isotonic bin_temperature; do
      out="c18_guided_${arm}_${prop}"
      # idempotent: a completed run is never redone, so the chain can be resumed
      [ -f "outputs/$out/guidance_metrics.json" ] && { echo "skip $out"; continue; }
      $PY scripts/17_guided_calibrated.py --property "$prop" --arm "$arm" --out "$out"
    done
  done
}

# Retrained readouts, end to end, through the UNMODIFIED script 05.
run_e2e_heads() {
  for prop in $ANCHORS; do
    for variant in $HEAD_VARIANTS; do
      out="c18_guided_head_${variant}_${prop}"
      [ -f "outputs/$out/guidance_metrics.json" ] && { echo "skip $out"; continue; }
      $PY scripts/05_guided_generation.py --dataset pilot_50k_p2 \
          --heads "c18_heads_${variant}" --property "$prop" \
          --conditions unguided throughout --out "$out"
    done
  done
}

run_bestofn() { $PY scripts/17_matched_best_of_n.py; }

# The identity: a power calibration at lambda = 1 IS the raw head at lambda = alpha.
# Run with --eps 0 so the two are equal in exact arithmetic and the molecules must
# match; alpha is read from the fitted Platt slope by 17_check_identity.py.
run_identity() { $PY scripts/17_check_identity.py; }

case "$stage" in
  prediction) run_prediction ;;
  offpolicy) run_offpolicy ;;
  heads) run_heads ;;
  perposition) run_perposition ;;
  e2e_calibrated) run_e2e_calibrated ;;
  e2e_heads) run_e2e_heads ;;
  bestofn) run_bestofn ;;
  identity) run_identity ;;
  summary) $PY scripts/17_summarise_c18.py ;;
  all)
    run_prediction; run_offpolicy; run_heads; run_perposition
    run_e2e_calibrated; run_e2e_heads; run_bestofn; run_identity
    $PY scripts/17_summarise_c18.py ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
