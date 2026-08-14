#!/usr/bin/env bash
# Task 1 on the cluster: fit k = 12, 16 and 20, and measure cond(J).
#
# The fit set and the hold-out set have different limits.
#   fit: memory. One copy of [J; sqrt(lambda) I] in FP64 takes
#        N_fit * n_theta * 8 bytes, and lineax QR needs about 3. At k = 20 with
#        1e5 that is 0.2 GB, and at k = 38 with 5e5 it is 12 GB. Raise TRAIN to
#        5e5 for k >= 32. More points buy little, because the excess risk is
#        n_theta/N_fit, which is 2.9e-3 at k = 20 with 1e5. That moves sigma by
#        0.15 percent, and it costs time in proportion to N_fit.
#   hold-out: one forward pass in chunks. The standard error of mean abs(f) is
#        1/sqrt(N), so 8e6 points give 0.035 percent. `max abs(f)` also grows
#        with N, and it is a lower bound on sup abs(f).
#
# Estimated time on one A100, from the laptop measurement:
#   k = 12   n_theta = 56     a few minutes
#   k = 16   n_theta = 135    approximately 20 minutes
#   k = 20   n_theta = 285    approximately 1 hour
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-20260813}
TRAIN=${TRAIN:-100000}
EVAL=${EVAL:-8000000}
CHUNK=${CHUNK:-200000}

uv run python -u scripts/run_fit.py \
    --k 12 16 20 \
    --n-train "$TRAIN" \
    --n-eval "$EVAL" \
    --chunk "$CHUNK" \
    --seed "$SEED" \
    --cond \
    --out runs 2>&1 | tee "runs_task1_seed${SEED}.log"
