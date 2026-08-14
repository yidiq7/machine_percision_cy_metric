#!/usr/bin/env bash
# Task 1 on the cluster: fit k = 12, 16 and 20, and measure cond(J).
#
# The LM step materializes [J; sqrt(lambda) I], which has shape
# (N_pts + n_theta, n_theta) in FP64. At k = 20 with N_pts = 1e5 that is 228 MB,
# and lineax needs a few copies, so one 80 GB GPU is enough.
#
# Estimated time on one A100, from the laptop measurement:
#   k = 12   n_theta = 56     a few minutes
#   k = 16   n_theta = 135    approximately 20 minutes
#   k = 20   n_theta = 285    approximately 1 hour
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-20260813}
PAIRS=${PAIRS:-50000}      # 4 points per pair, so 2e5 points
TRAIN=${TRAIN:-100000}

uv run python -u scripts/run_fit.py \
    --k 12 16 20 \
    --n-pairs "$PAIRS" \
    --n-train "$TRAIN" \
    --seed "$SEED" \
    --cond \
    --out runs 2>&1 | tee "runs_task1_seed${SEED}.log"
