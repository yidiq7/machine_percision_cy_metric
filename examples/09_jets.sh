#!/bin/bash
# 09: the derivatives of f for a fit of two stages from 07: grad f, the Laplacian of f and Ricci
#   from Taylor jets, on NPARTS samples of N points, then the L2 norms and the sampled maxima.
#   The jets agree with the fit at D = 36. From D = 40 they miss the fit by about 9 times sigma
#   (the paper), so use D = 36.
# Hardware: one GPU. tmgpu: not necessary.
# Time for the paper: 50 min for each sample of 32,768 points on one H200.
# Variables (default): D (36), BASE, S1, OUT, LEVELS1 as in 07; JETS (runs/jets_D), NPARTS (4), PART,
#   N (32768); the seed of part G is 20271001 + G.
source "$(dirname "$0")/common.sh"
D=${D:-36}
BASE=${BASE:-runs/two_stage_base}
S1=${S1:-runs/two_stage_first}
OUT=${OUT:-runs/two_stage_D$D}
JETS=${JETS:-runs/jets_$D}
NPARTS=${NPARTS:-4}
N=${N:-32768}
LEVELS1=${LEVELS1:-24 23 22 21 20 19 18 17}
K1=$(seq "$D" -1 $((D - 7)) | tr '\n' ' ')
LOAD=$OUT
mkdir -p "$JETS"

jets_part() {
    local seed=$((20271001 + $1))
    [ -e "$JETS/part_$seed.npz" ] && return 0
    $RUN scripts/residual_jets.py compute --k0 12 --theta "$BASE/theta_k0_12.npy" --stage1 "$S1" \
        --stage1-k1 $LEVELS1 --k1 $K1 --load "$LOAD" --run "$OUT" --exact-mul \
        --n "$N" --seed "$seed" --chunk 64 --block 4096 --out "$JETS/part_$seed.npz"
}
parts "$NPARTS" jets_part
if [ -z "${PART:-}" ]; then
    $RUN scripts/residual_jets.py combine "$JETS"/part_*.npz
fi
