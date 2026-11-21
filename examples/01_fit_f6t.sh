#!/bin/bash
# 01: the fit f6t of the paper, the potential of the strict bound.
#   Step 1 fits the HN (Headrick-Nassar) base at k0 = 16, and builds the V+A bases of the levels.
#   Step 2 fits the 12 levels 24 to 13 together in one joint basis in dd (double-double).
#   Each step writes sigma = mean abs(f) on the hold-out. For f6t it is about 1.7e-22.
# Hardware: one GPU with about 140 GB of memory. tmgpu: not necessary (UV_EXTRA=cuda is enough).
# Time for the paper on one H200: 3.3 h for step 1, and 2.1 h for step 2.
# Variables (default):
#   BASE (runs/f6), OUT (runs/f6t)   the output directories of the steps 1 and 2
#   K0 (16), K1 (24 23 ... 13), PREFIXES (4 8 12), the column of each prefix is a solution
#   NFIT1 (100000), NBUILD1 (30000)  the fit points and the build points of step 1
#   NFIT (200000), NBUILD (60000)    the same for step 2
#   NEVAL (400000)                   the hold-out points
source "$(dirname "$0")/common.sh"
BASE=${BASE:-runs/f6}
OUT=${OUT:-runs/f6t}
K0=${K0:-16}
K1=${K1:-24 23 22 21 20 19 18 17 16 15 14 13}
PREFIXES=${PREFIXES:-4 8 12}
NFIT1=${NFIT1:-100000}
NBUILD1=${NBUILD1:-30000}
NFIT=${NFIT:-200000}
NBUILD=${NBUILD:-60000}
NEVAL=${NEVAL:-400000}
mkdir -p "$(dirname "$BASE")" "$(dirname "$OUT")"

$RUN scripts/vpa_dd_levels.py --k0 "$K0" --k1 $K1 --joint-basis --joint-tol 1e-20 --exact-mul \
    --refine 3 --n-fit "$NFIT1" --seed 20261001 --n-eval "$NEVAL" --eval-seed 20261813 \
    --n-build "$NBUILD1" --out "$BASE"
$RUN scripts/vpa_dd_levels.py --k0 "$K0" --theta "$BASE/theta_k0_$K0.npy" --k1 $K1 \
    --load "$BASE" --joint-basis --joint-tol 1e-23 --exact-mul --refine 3 --prefixes $PREFIXES \
    --n-fit "$NFIT" --seed 20261001 --n-eval "$NEVAL" --eval-seed 20261813 \
    --n-build "$NBUILD" --out "$OUT"
