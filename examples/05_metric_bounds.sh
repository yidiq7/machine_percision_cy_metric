#!/bin/bash
# 05: the bounds of the metric of the paper: C, kappa, lambda, Lambda, K~, c~1, c~2, the diameters
#   and lambda_1.
#   cert_metric.py bounds the metric in each chart of the atlas on the cells of the cover, in
#   NPARTS parts, with the lists of reference centers of 04. cert_partition.py --metric checks the
#   records. cert_atlas.py fs and transitions give the bounds that need no potential, and
#   cert_atlas.py combine gives the constants from the records of 04 and of this script.
#   The paper: C <= 2.1765, kappa <= 4.4449e-13, lambda >= 0.016109, Lambda <= 0.64047, K~ <= 59.999.
# Hardware: GPUs, then CPU cores. tmgpu: necessary.
# Time for the paper: 1.5 to 1.7 h for each of 4 parts on one H200; on 48 CPU cores, 52 min for
#   fs and 25 min for transitions.
# Variables (default):
#   DATA_DIR, COVER, QUARTIC_LOCAL_FORMS, MANIFEST   as in 04
#   BOUND (runs/bound)    the output directory of 04: the lists of reference centers and the records
#   OUT (runs/metric)     the records metric_<G>.npz, the logs and the files of cert_atlas.py
#   NPARTS (4), PART (all parts in sequence)
#   PROCS (48)            CPU processes of cert_partition.py and cert_atlas.py
source "$(dirname "$0")/common.sh"
BOUND=${BOUND:-runs/bound}
OUT=${OUT:-runs/metric}
NPARTS=${NPARTS:-4}
PROCS=${PROCS:-48}
DATA="$DATA_DIR/f6t_data.npz"
FORMS="$DATA_DIR/f6t_forms.npz"
need_tmgpu
mkdir -p "$OUT"

metric_part() {
    local g=$1
    [ -e "$OUT/metric_$g.npz" ] && return 0
    if [ ! -s "$BOUND/refs_$g.txt" ]; then
        echo "part $g has no reference center"
        return 0
    fi
    $RUN scripts/cert_metric.py --cover "$COVER" --local "$QUARTIC_LOCAL_FORMS" \
        --refs $(cat "$BOUND/refs_$g.txt") --k-target 60 --depth 3 --order 20 --order-split 12 \
        --a 0.12 --batch 64 --out "$OUT/metric_$g.npz" > "$OUT/log_$g.txt" 2>&1
    grep -h "^done" "$OUT/log_$g.txt" || true
}
parts "$NPARTS" metric_part
if [ -n "${PART:-}" ]; then
    echo "Part $PART is done. Run this script with no PART when all parts are done."
    exit 0
fi

METRIC=$(ls "$OUT"/metric_*.npz)
$RUN scripts/cert_partition.py --metric --run $METRIC --cover "$COVER" --manifest "$MANIFEST" \
    --data "$DATA" --forms "$FORMS" --procs "$PROCS" ${REFS:+--refs $REFS}
[ -e "$OUT/atlas_fs.npz" ] || $RUN scripts/cert_atlas.py fs --tol 0.3 --procs "$PROCS" --out "$OUT/atlas_fs.npz"
[ -e "$OUT/atlas_tr_a.npz" ] || $RUN scripts/cert_atlas.py transitions --kinds d2 d3 --tol 0.2 \
    --charts 01 02 03 10 12 13 --procs "$PROCS" --out "$OUT/atlas_tr_a.npz"
[ -e "$OUT/atlas_tr_b.npz" ] || $RUN scripts/cert_atlas.py transitions --kinds d2 d3 --tol 0.2 \
    --charts 20 21 23 30 31 --procs "$PROCS" --out "$OUT/atlas_tr_b.npz"
$RUN scripts/cert_atlas.py combine --runs $(ls "$BOUND"/run_*.npz) --metric $METRIC \
    --fs "$OUT/atlas_fs.npz" --transitions "$OUT/atlas_tr_a.npz" "$OUT/atlas_tr_b.npz" \
    --out "$OUT/atlas_combine.npz"
