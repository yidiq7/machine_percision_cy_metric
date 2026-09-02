#!/usr/bin/env bash
# Fits on the cluster. Pick the stage with STAGE.
#
#   STAGE=task1   k = 12, 16, 20 in FP64. Validated: sigma 2.2018e-07,
#                 3.5096e-09, 6.3276e-11, and the rate is 2.81 and 2.73.
#   STAGE=high    k = 24, 28 with the Gram stage in double-double. In FP64 the
#                 floor on abs(f) at k = 28 is 3.97e-13 and sigma came out
#                 1.71e-13, which is below the floor and so not a measurement.
#                 In dd the floor is 1.61e-14, and the rate predicts 2.5e-14.
#   STAGE=dry     size the blocks and report the memory, then stop.
#
# The fit set and the hold-out set have different limits.
#   fit: the intermediates of the Jacobian. plan_fit measures the real program
#        and sizes both the fit set and the lax.map block against BUDGET.
#   hold-out: one forward pass in chunks. The standard error of mean abs(f) is
#        1/sqrt(N), so 8e6 points give 0.035 percent.
#
# The Gram stage in dd costs 41 times the time and 14 times the memory at
# k = 28, so expect roughly 45 minutes for k = 28 and a few hours for k = 32.
#
# RISK sets the fit set, and the block gets what is left. dd needs 1.1 MB for
# each fit point at k = 28, so the default risk of 5e-3 asks for 212 GB, takes
# every point the budget allows and leaves the block at its floor of 64.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-task1}
SEED=${SEED:-20260813}
TRAIN=${TRAIN:-200000}
EVAL=${EVAL:-8000000}
CHUNK=${CHUNK:-200000}
BUDGET=${BUDGET:-}       # GB for the whole LM solve; empty means 45 percent
K=${K:-}                 # override the degrees of the stage, for a quick check
RISK=${RISK:-}           # excess risk n_theta/N_fit to aim for; sizes the fit set
LIMIT=${LIMIT:-0}        # 1 = --run-to-limit: no convergence test, best so far
WHITEN=${WHITEN:-0}      # 1 whitens the section basis in L2(X)

# JAX takes 75 percent of the card by default. This gives back the rest.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

# `uv run` syncs before it runs, and a sync with no --extra gpu removes
# jax[cuda12]. The run then falls back to the CPU without failing, and a budget
# meant for the card becomes a request for host memory. Carry the extra here.
# Plain strings, not arrays: `set -u` with bash 3.2 rejects an empty array.
if [ "${NO_GPU:-0}" = "1" ]; then
    UV_EXTRA=""
    REQUIRE=""
else
    UV_EXTRA="--extra gpu"
    REQUIRE="--require-gpu"
fi

case "$STAGE" in
  task1) KS="12 16 20"; DD=off;  STAGE_DEF="" ;;
  high)  KS="24 28";    DD=on;   STAGE_DEF="" ;;
  dry)   KS="24 28";    DD=on;   STAGE_DEF="--dry-run" ;;
  *) echo "unknown STAGE '$STAGE'; use task1, high or dry" >&2; exit 2 ;;
esac
KS=${K:-$KS}

# The stage gives the default flags, and FLAGS adds to them. An earlier version
# set STAGE_FLAGS inside the case, so a value from the environment was lost and
# the run silently used the default.
STAGE_FLAGS="$STAGE_DEF ${FLAGS:-}"

if [ "$WHITEN" = "1" ]; then WHITEN_FLAG="--whiten"; else WHITEN_FLAG=""; fi
if [ "$LIMIT" = "1" ]; then RUNLIMIT="--run-to-limit"; else RUNLIMIT=""; fi

# shellcheck disable=SC2086
uv run $UV_EXTRA python -u scripts/run_fit.py \
    $REQUIRE \
    --k $KS \
    --n-train "$TRAIN" \
    --n-eval "$EVAL" \
    --chunk "$CHUNK" \
    --gram-dd "$DD" \
    ${BUDGET:+--budget-gb "$BUDGET"} \
    ${RISK:+--target-risk "$RISK"} \
    $RUNLIMIT \
    $WHITEN_FLAG \
    --max-steps 600 \
    --seed "$SEED" \
    --cond \
    $STAGE_FLAGS \
    --out runs 2>&1 | tee "runs_${STAGE}_seed${SEED}.log"
