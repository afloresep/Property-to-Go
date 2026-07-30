#!/usr/bin/env bash
# Serial driver for the parts of the pilot whose wall-clock is a reported metric.
#
# Guidance and compute-matched best-of-N are compared on wall time as well as on
# processed tokens, so they must run one at a time on an otherwise idle machine.
# Anything that only reports token counts can be run concurrently instead.
#
#   bash scripts/run_pilot.sh pilot_50k
set -euo pipefail

DATASET="${1:-pilot_50k}"
PY=.venv/bin/python
LOG=outputs/${DATASET}_pilot.log

exec > >(tee -a "$LOG") 2>&1
echo "=== pilot driver: $DATASET :: $(date -u +%FT%TZ) ==="

echo "--- 03 train heads"
$PY -u scripts/03_train_heads.py --dataset "$DATASET"

for PROP in clogp aromatic_rings; do
  echo "--- 05 guided generation: $PROP"
  $PY -u scripts/05_guided_generation.py --dataset "$DATASET" --property "$PROP"

  echo "--- 06 compute-matched best-of-N: $PROP"
  $PY -u scripts/06_best_of_n.py --dataset "$DATASET" --property "$PROP" --full-recompute-n 64
done

echo "--- 07 figures"
$PY -u scripts/07_figures.py --dataset "$DATASET"

echo "=== done :: $(date -u +%FT%TZ) ==="
