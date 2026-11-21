#!/bin/bash
# 02: the two files of the potential of the bound, from the fit of 01.
#   level_oracle.py export writes the base in the monomial basis, which makes p0 exactly invariant
#   under S_4, and the data of the levels: f6t_data.npz.
#   level_matrices.py writes each level sum as an invariant Hermitian form N_L: f6t_forms.npz.
#   A new fit gives other bytes. The Zenodo record holds the two files of the paper.
#   The script writes into a new directory OUT_DIR. To use the files in 03 to 05, set DATA_DIR to it.
# Hardware: one GPU, then CPU cores. tmgpu: not necessary.
# Time for the paper: 30 min on one H200, then 2 min on 32 CPU cores.
# Variables (default):
#   BASE (runs/f6), FIT (runs/f6t), K0 (16), K1 (24 23 ... 13)   as in 01
#   COL (3)               the column of the coefficients: 3 is the prefix of all 12 levels
#   OUT_DIR (runs/potential)   the directory of the two output files; they must not exist
#   PROCS (32)            CPU processes of level_matrices.py
source "$(dirname "$0")/common.sh"
BASE=${BASE:-runs/f6}
FIT=${FIT:-runs/f6t}
K0=${K0:-16}
K1=${K1:-24 23 22 21 20 19 18 17 16 15 14 13}
COL=${COL:-3}
PROCS=${PROCS:-32}
OUT_DIR=${OUT_DIR:-runs/potential}
TAG="k0_${K0}_$(echo $K1 | tr ' ' '_')"
for f in f6t_data.npz f6t_forms.npz; do
    if [ -e "$OUT_DIR/$f" ]; then
        echo "$OUT_DIR/$f exists. Set OUT_DIR to a new directory." >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR"

$RUN scripts/level_oracle.py export --k0 "$K0" --theta "$BASE/theta_k0_$K0.npy" --k1 $K1 \
    --load "$BASE" --load-joint "$FIT/joint_map_$TAG.npz" --coeffs "$FIT/vpa_levels_coeffs_$TAG.npy" \
    --exact-mul --invariant-base --data "$OUT_DIR/f6t_data.npz"
JAX_PLATFORMS=cpu $RUN scripts/level_matrices.py --data "$OUT_DIR/f6t_data.npz" --col "$COL" \
    --procs "$PROCS" --out "$OUT_DIR/f6t_forms.npz"
