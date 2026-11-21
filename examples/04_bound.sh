#!/bin/bash
# 04: the strict bound of sup abs(f) and sup abs(grad f) on X, for the potential of 02.
#   The reference centers go to NPARTS lists of about equal cost. cert_run.py bounds the final
#   boxes of each list. When all parts exist, cert_partition.py checks that the boxes cover F and
#   that each record uses a file of the manifest, and cert_check.py checks the cover with random
#   points and 8 random boxes against arb.
#   The paper: sup abs(f) <= 4.9985e-20 and sup abs(grad f) <= 3.3654e-16, from 134,552 boxes.
# Hardware: GPUs, then CPU cores. tmgpu: necessary.
# Time for the paper: 3.0 to 3.4 h for each of 4 parts on one H200; 1 h for cert_check.
# Variables (default):
#   DATA_DIR, COVER, QUARTIC_LOCAL_FORMS, MANIFEST   as in common.sh (MANIFEST: the manifest of the
#                         local forms; after a new 03, set it to the manifest of 03)
#   OUT (runs/bound)      the lists of reference centers, the records run_<G>.npz and the logs
#   NPARTS (4), PART (all parts in sequence)
#   REFS (all)            only these reference centers, for a test
#   POINTS (3000), BOXES (8)   the random points of F and the random boxes of cert_check.py; with
#                         REFS, set POINTS=0, because the boxes then cover only a part of F
#   PROCS (16)            CPU processes of cert_partition.py
source "$(dirname "$0")/common.sh"
OUT=${OUT:-runs/bound}
NPARTS=${NPARTS:-4}
PROCS=${PROCS:-16}
DATA="$DATA_DIR/f6t_data.npz"
FORMS="$DATA_DIR/f6t_forms.npz"
need_tmgpu
mkdir -p "$OUT"

# The lists of reference centers: a cell costs about 0.26 s at rho >= 0.3 and 0.16 s below.
if [ ! -e "$OUT/refs_0.txt" ]; then
    $RUN - "$COVER" "$OUT" "$NPARTS" ${REFS:-} <<'PY'
import sys
import numpy as np
cov = np.load(sys.argv[1])
out, nparts = sys.argv[2], int(sys.argv[3])
cen = cov['centers']
t0 = cen[:, 0, 0] + 1j * cen[:, 0, 2]
t1 = cen[:, 1, 0] + 1j * cen[:, 1, 2]
z2 = np.abs(1 + t0 ** 4 + t1 ** 4) ** 0.25
a = np.stack([np.abs(t0), np.abs(t1), z2, np.ones(len(t0))], 1)
rho = a.min(1) / a.max(1)
cost = np.where(rho >= 0.3, 0.26, 0.16)
refs = [int(r) for r in sys.argv[4:]] or [int(r) for r in np.unique(cov['ref'])]
per = {r: float(cost[cov['ref'] == r].sum()) for r in refs}
bins, load = [[] for _ in range(nparts)], [0.0] * nparts
for r in sorted(per, key=per.get, reverse=True):
    g = int(np.argmin(load))
    bins[g].append(r)
    load[g] += per[r]
for g in range(nparts):
    open(f'{out}/refs_{g}.txt', 'w').write(' '.join(str(r) for r in sorted(bins[g])))
print(len(refs), 'reference centers; estimated GPU seconds of each part:', [round(x) for x in load])
PY
fi

run_part() {
    local g=$1
    [ -e "$OUT/run_$g.npz" ] && return 0
    if [ ! -s "$OUT/refs_$g.txt" ]; then
        echo "part $g has no reference center"
        return 0
    fi
    $RUN scripts/cert_run.py --cover "$COVER" --local "$QUARTIC_LOCAL_FORMS" \
        --refs $(cat "$OUT/refs_$g.txt") --target 5e-20 --verbose --out "$OUT/run_$g.npz" \
        > "$OUT/log_$g.txt" 2>&1
    grep -h "^done" "$OUT/log_$g.txt" || true
}
parts "$NPARTS" run_part
if [ -n "${PART:-}" ]; then
    echo "Part $PART is done. Run this script with no PART when all parts are done."
    exit 0
fi

RUNS=$(ls "$OUT"/run_*.npz)
$RUN scripts/cert_partition.py --run $RUNS --cover "$COVER" --manifest "$MANIFEST" \
    --data "$DATA" --forms "$FORMS" --procs "$PROCS" ${REFS:+--refs $REFS}
$RUN scripts/cert_check.py --run $RUNS --cover "$COVER" --local "$QUARTIC_LOCAL_FORMS" \
    --data "$DATA" --forms "$FORMS" --points ${POINTS:-3000} --boxes ${BOXES:-8}
