#!/bin/bash
# 07: a fit of two stages with the top degree D = 36, 40 or 44 (the fits s2e, s2k and s2o of the
#   paper).
#   Step 1 fits the HN base at k0 = 12 with the levels 36 to 29, and keeps the base and the V+A
#   bases of these levels. Step 2 fits the first stage: the levels 24 to 17 on this base.
#   Step 3 fits the second stage: the levels D to D - 7 with the weights of the first stage, with
#   the cutoff TOL of the joint basis. It builds the bases that BASE does not hold on NBL points.
#   Steps 1 and 2 run only if their output does not exist.
#   The paper, sigma from mpmath on 256 points: 2.71e-28 (D = 36), 9.84e-30 (D = 40), 2.13e-31
#   (D = 44).
# Hardware: one GPU with about 140 GB of memory, and up to 58 GB of host memory at D >= 40.
#   tmgpu: not necessary.
# Time for the paper on one H200: 4.3 h (step 1), 2.2 h (step 2); step 3 5.2 h at D = 40 and
#   7.25 h at D = 44.
# Variables (default):
#   D (36), TOL (1e-20), REFINE (3 at D = 36, 1 above)
#   BASE (runs/f2r), S1 (runs/s1b), OUT (runs/s2_D)   the output directories of the steps
#   NFIT (200000), NBUILD (60000), NBL (20000), NEVAL (400000)
#   LEVELS0 (36 35 ... 29), LEVELS1 (24 23 ... 17)   the levels of the steps 1 and 2
#   PREFIXES0 (1 2 4 8), PREFIXES1 (4 8)             the solutions of the steps 1 and 2
source "$(dirname "$0")/common.sh"
D=${D:-36}
TOL=${TOL:-1e-20}
if [ "$D" -le 36 ]; then REFINE=${REFINE:-3}; else REFINE=${REFINE:-1}; fi
BASE=${BASE:-runs/f2r}
S1=${S1:-runs/s1b}
OUT=${OUT:-runs/s2_$D}
NFIT=${NFIT:-200000}
NBUILD=${NBUILD:-60000}
NBL=${NBL:-20000}
NEVAL=${NEVAL:-400000}
LEVELS0=${LEVELS0:-36 35 34 33 32 31 30 29}
LEVELS1=${LEVELS1:-24 23 22 21 20 19 18 17}
PREFIXES0=${PREFIXES0:-1 2 4 8}
PREFIXES1=${PREFIXES1:-4 8}
K1=$(seq "$D" -1 $((D - 7)) | tr '\n' ' ')
COMMON="--seed 20261001 --n-eval $NEVAL --eval-seed 20261813 --n-build $NBUILD --exact-mul"
mkdir -p "$(dirname "$BASE")" "$(dirname "$S1")" "$(dirname "$OUT")"

if [ ! -e "$BASE/theta_k0_12.npy" ]; then
    $RUN scripts/vpa_dd_levels.py --k0 12 --k1 $LEVELS0 --joint-basis --joint-tol 1e-15 --refine 3 \
        --prefixes $PREFIXES0 --n-fit "$NFIT" $COMMON --out "$BASE"
fi
if [ ! -e "$S1/vpa_levels_coeffs_k0_12_$(echo $LEVELS1 | tr ' ' '_').npy" ]; then
    $RUN scripts/vpa_dd_levels.py --k0 12 --theta "$BASE/theta_k0_12.npy" --k1 $LEVELS1 \
        --joint-basis --joint-tol 1e-20 --refine 3 --prefixes $PREFIXES1 --n-fit "$NFIT" $COMMON --out "$S1"
fi
# OUT holds the bases of all 8 levels: links to those of BASE, and the bases that step 3 builds.
mkdir -p "$OUT"
for k in $K1; do
    for f in "$BASE"/vpa_dd_basis_k0_12_k1_$k.npz "$BASE"/vpa_dd_hlo_k0_12_k1_$k.npy \
             "$BASE"/vpa_dd_gamma_k0_12_k1_$k.npy; do
        if [ -e "$f" ]; then
            ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$OUT/"
        fi
    done
done
$RUN scripts/vpa_dd_stage2.py --k0 12 --theta "$BASE/theta_k0_12.npy" --stage1 "$S1" \
    --stage1-k1 $LEVELS1 --k1 $K1 --load "$OUT" --joint-tol "$TOL" --refine "$REFINE" \
    --prefixes 8 --n-fit "$NFIT" $COMMON --n-build-level "$NBL" --out "$OUT"
