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
# Variables (default): D, BASE, S1, OUT, LEVELS1 as in 07; STEPS; PROCS (16);
#   NHOLD (8000000), NSEAM (131072), NSAMPLE (256)   the points of holdout, chart and sample;
#   MP (RUN with gmpy2)   the command of the mpmath stages
#   FIT1_BASE (runs/f6), FIT1 (runs/f6t), K0F (16), K1F (24 ... 13)   the fit of 01, for seam1
source "$(dirname "$0")/common.sh"
D=${D:-36}
BASE=${BASE:-runs/f2r}
S1=${S1:-runs/s1b}
OUT=${OUT:-runs/s2_$D}
STEPS=${STEPS:-holdout chart ref12 sample basis}
PROCS=${PROCS:-16}
LEVELS1=${LEVELS1:-24 23 22 21 20 19 18 17}
K1=$(seq "$D" -1 $((D - 7)) | tr '\n' ' ')
TAG="k0_12_s1_$(echo $LEVELS1 | tr ' ' '_')_s2_$(echo $K1 | tr ' ' '_' | sed 's/_$//')"
# 07 puts the bases of all 8 levels into OUT.
LOAD=$OUT
NHOLD=${NHOLD:-8000000}
NSEAM=${NSEAM:-131072}
NSAMPLE=${NSAMPLE:-256}
FIT="--k0 12 --theta $BASE/theta_k0_12.npy --stage1 $S1 --stage1-k1 $LEVELS1 --k1 $K1 --exact-mul"
SAVED="--load $LOAD --load-joint $OUT/joint_map_$TAG.npz --coeffs $OUT/vpa_stage2_coeffs_$TAG.npy
       --record $OUT/vpa_stage2_$TAG.json"
# The mpmath stages run faster with gmpy2, which uv adds for them.
if [ -z "${MP:-}" ]; then
    MP="uv run --extra $UV_EXTRA --locked --with gmpy2 python -u"
    [ "$UV_EXTRA" = none ] && MP="uv run --locked --with gmpy2 python -u"
fi

for step in $STEPS; do
    case $step in
    holdout)
        for part in 0 1 2 3; do
            $RUN scripts/vpa_dd_stage2.py $FIT $SAVED --eval-only --eval-offset $((part * NHOLD / 4)) \
                --eval-total "$NHOLD" --n-eval $((NHOLD / 4)) --out "$OUT"
        done
        $RUN scripts/vpa_levels_combine.py "$OUT"/vpa_levels_v_${TAG}_*.npz --k0 12 ;;
    chart)
        $RUN scripts/vpa_dd_stage2.py $FIT $SAVED --seam "$NSEAM" --out "$OUT" ;;
    ref12)
        $RUN scripts/stage2_oracle.py export $FIT --load "$LOAD" --run "$OUT" --seed 20264814 \
            --ratios 0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.20 0.15 0.08 0.05 \
            --data "$OUT/oracle_data12.npz"
        $MP scripts/stage2_oracle.py reference --data "$OUT/oracle_data12.npz" --procs "$PROCS" ;;
    sample)
        $RUN scripts/stage2_oracle.py export $FIT --load "$LOAD" --run "$OUT" --seed 20265001 \
            --n "$NSAMPLE" --sample --data "$OUT/sample_${NSAMPLE}_20265001.npz"
        $MP scripts/stage2_oracle.py sample --data "$OUT/sample_${NSAMPLE}_20265001.npz" --procs "$PROCS" ;;
    basis)
        JAX_PLATFORMS=cpu $RUN scripts/vpa_dd_oracle.py "$LOAD/vpa_dd_basis_k0_12_k1_$D.npz" \
            "$LOAD/vpa_dd_hlo_k0_12_k1_$D.npy" --points 3 ;;
    seam1)
        FIT1_BASE=${FIT1_BASE:-runs/f6}
        FIT1=${FIT1:-runs/f6t}
        K0F=${K0F:-16}
        K1F=${K1F:-24 23 22 21 20 19 18 17 16 15 14 13}
        T1="k0_${K0F}_$(echo $K1F | tr ' ' '_')"
        $RUN scripts/seam_levels.py --k0 "$K0F" --theta "$FIT1_BASE/theta_k0_$K0F.npy" --k1 $K1F \
            --load "$FIT1_BASE" --load-joint "$FIT1/joint_map_$T1.npz" \
            --coeffs "$FIT1/vpa_levels_coeffs_$T1.npy" --exact-mul ;;
    *) echo "unknown step $step" >&2; exit 1 ;;
    esac
done
