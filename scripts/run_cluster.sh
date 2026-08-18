#!/usr/bin/env bash
# Task 1 on the cluster: fit k = 12, 16 and 20, and measure cond(J).
#
# The fit set and the hold-out set have different limits.
#   fit: the intermediates of the Jacobian, and not J itself. Forward-mode
#        autodiff nests the points, the 8 real directions of complex_hessian and
#        the n_theta tangents, which is 31 MB for each point at k = 20. BUDGET
#        sets the lax.map block, so the peak follows the block and not TRAIN.
#        A large TRAIN buys little, because the excess risk is n_theta/N_fit:
#        2.9e-3 at k = 20 with 1e5 moves sigma by 0.15 percent, and the time
#        grows in proportion to TRAIN. Raise TRAIN to 5e5 for k >= 32.
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
BUDGET=${BUDGET:-}       # GB for the whole LM solve; empty means 45 percent of the device

# JAX takes 75 percent of the card by default. This gives back the rest.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

uv run python -u scripts/run_fit.py \
    --k 12 16 20 \
    --n-train "$TRAIN" \
    --n-eval "$EVAL" \
    --chunk "$CHUNK" \
    ${BUDGET:+--budget-gb "$BUDGET"} \
    --seed "$SEED" \
    --cond \
    --out runs 2>&1 | tee "runs_task1_seed${SEED}.log"
