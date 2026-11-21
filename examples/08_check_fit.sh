#!/bin/bash
# 08: the checks of a fit of two stages from 07.
#   holdout  sigma, rms and max abs(f) on 8e6 hold-out points in 4 slices, joined by
#            vpa_levels_combine.py
#   chart    f at 131,072 points in two charts; the change of f must be near 10 % of sigma or less
#   ref12    f at 12 points against mpmath, at min abs(z) over max abs(z) from 0.95 to 0.05
#   sample   sigma from mpmath on 256 points, with no dd noise
#   basis    the dd error of the level functions of degree D against mpmath
#   seam1    the chart test of the fit of one stage of 01, with seam_levels.py
#   STEPS (holdout chart ref12 sample basis) picks the checks.
# Hardware: one GPU; CPU cores for the mpmath stages. tmgpu: not necessary.
# Time for the paper at D = 36: 293 s of one core for each mpmath point of the sample, with 16
#   cores; 680 to 709 s at D = 40, and 1,740 s at D = 44.
# Variables (default): D, BASE, S1, OUT as in 07; STEPS; PROCS (16);
#   FIT1_BASE (runs/f6) and FIT1 (runs/f6t), the directories of 01 for seam1
source "$(dirname "$0")/common.sh"
D=${D:-36}
BASE=${BASE:-runs/f2r}
S1=${S1:-runs/s1b}
OUT=${OUT:-runs/s2_$D}
STEPS=${STEPS:-holdout chart ref12 sample basis}
PROCS=${PROCS:-16}
LEVELS1=${LEVELS1:-24 23 22 21 20 19 18 17}
K1=$(seq "$D" -1 $((D - 7)) | tr '\n' ' ')
S1TAG="k0_12_$(echo $LEVELS1 | tr ' ' '_')"
TAG="k0_12_s1_$(echo $LEVELS1 | tr ' ' '_')_s2_$(echo $K1 | tr ' ' '_' | sed 's/_$//')"
# The level bases above 36 are in OUT (07 builds them there); the others are in BASE.
LOAD=$BASE
[ "$D" -gt 36 ] && LOAD=$OUT
FIT="--k0 12 --theta $BASE/theta_k0_12.npy --stage1 $S1 --stage1-k1 $LEVELS1 --k1 $K1 --exact-mul"
SAVED="--load $LOAD --load-joint $OUT/joint_map_$TAG.npz --coeffs $OUT/vpa_stage2_coeffs_$TAG.npy
       --record $OUT/vpa_stage2_$TAG.json"
MP="uv run --extra $UV_EXTRA --locked --with gmpy2 python -u"
[ "$UV_EXTRA" = none ] && MP="uv run --locked --with gmpy2 python -u"

for step in $STEPS; do
    case $step in
    holdout)
        for offset in 0 2000000 4000000 6000000; do
            $RUN scripts/vpa_dd_stage2.py $FIT $SAVED --eval-only --eval-offset $offset \
                --eval-total 8000000 --n-eval 2000000 --out "$OUT"
        done
        $RUN scripts/vpa_levels_combine.py "$OUT"/vpa_levels_v_${TAG}_*.npz --k0 12 ;;
    chart)
        $RUN scripts/vpa_dd_stage2.py $FIT $SAVED --seam 131072 --out "$OUT" ;;
    ref12)
        $RUN scripts/stage2_oracle.py export $FIT --load "$LOAD" --run "$OUT" --seed 20264814 \
            --ratios 0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.20 0.15 0.08 0.05 \
            --data "$OUT/oracle_data12.npz"
        $MP scripts/stage2_oracle.py reference --data "$OUT/oracle_data12.npz" --procs "$PROCS" ;;
    sample)
        $RUN scripts/stage2_oracle.py export $FIT --load "$LOAD" --run "$OUT" --seed 20265001 \
            --n 256 --sample --data "$OUT/sample_256_20265001.npz"
        $MP scripts/stage2_oracle.py sample --data "$OUT/sample_256_20265001.npz" --procs "$PROCS" ;;
    basis)
        JAX_PLATFORMS=cpu $RUN scripts/vpa_dd_oracle.py "$LOAD/vpa_dd_basis_k0_12_k1_$D.npz" \
            "$LOAD/vpa_dd_hlo_k0_12_k1_$D.npy" --points 3 ;;
    seam1)
        FIT1_BASE=${FIT1_BASE:-runs/f6}
        FIT1=${FIT1:-runs/f6t}
        K1F=${K1F:-24 23 22 21 20 19 18 17 16 15 14 13}
        T1="k0_16_$(echo $K1F | tr ' ' '_')"
        $RUN scripts/seam_levels.py --k0 16 --theta "$FIT1_BASE/theta_k0_16.npy" --k1 $K1F \
            --load "$FIT1_BASE" --load-joint "$FIT1/joint_map_$T1.npz" \
            --coeffs "$FIT1/vpa_levels_coeffs_$T1.npy" --exact-mul ;;
    *) echo "unknown step $step" >&2; exit 1 ;;
    esac
done
