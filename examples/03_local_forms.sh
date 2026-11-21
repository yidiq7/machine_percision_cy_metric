#!/bin/bash
# 03: the cover of the fundamental domain F, and the local forms of its 332 reference centers.
#   cert_cover.py writes the cells of side 0.03 and their reference centers, if COVER does not exist.
#   tm_local_forms.py moves the forms of the potential to each reference center, in arb and in td.
#   cert_local_values.py checks each file of local forms against the potential, and writes a
#   manifest with the SHA-256 of each file. The script then compares these SHA-256 values with
#   the manifest of the paper, if MANIFEST exists.
# Hardware: CPU cores; 211 GB of disk for the local forms. tmgpu: necessary.
# Time for the paper: about 1.5 h on 48 CPU cores.
# Variables (default):
#   DATA_DIR (runs_tm)    the directory of f6t_data.npz and f6t_forms.npz (02, or the Zenodo record)
#   COVER ($DATA_DIR/cover_s0.03_c6_v3.npz), QUARTIC_LOCAL_FORMS ($DATA_DIR/tm_local_f6t_td_v3)
#   MANIFEST ($DATA_DIR/f6t_local_manifest_v3.txt)   the manifest of the paper, for the comparison
#   NEW_MANIFEST ($QUARTIC_LOCAL_FORMS/manifest.txt)  the manifest that this script writes
#   ONLY (all)            only these reference centers, for a test, for example ONLY=297
#   PROCS (48)            CPU processes
source "$(dirname "$0")/common.sh"
NEW_MANIFEST=${NEW_MANIFEST:-$QUARTIC_LOCAL_FORMS/manifest.txt}
PROCS=${PROCS:-48}
need_tmgpu

if [ ! -e "$COVER" ]; then
    $RUN scripts/cert_cover.py --side 0.03 --coarse 6 --out "$COVER" --procs "$PROCS"
fi
$RUN scripts/tm_local_forms.py --data "$DATA_DIR/f6t_data.npz" --forms "$DATA_DIR/f6t_forms.npz" \
    --centers "$COVER" --out "$QUARTIC_LOCAL_FORMS" --procs "$PROCS" ${ONLY:+--only $ONLY}
$RUN scripts/cert_local_values.py --cover "$COVER" --local "$QUARTIC_LOCAL_FORMS" \
    --data "$DATA_DIR/f6t_data.npz" --forms "$DATA_DIR/f6t_forms.npz" --points 2 --procs "$PROCS" \
    --manifest "$NEW_MANIFEST" ${ONLY:+--refs $ONLY}

if [ -e "$MANIFEST" ]; then
    # Rows of a manifest: reference center, SHA-256 of its file, largest gap over its bound.
    same=$(awk 'NR == FNR && !/^#/ {sha[$1] = $2; next} !/^#/ && ($1 in sha) {n++; if (sha[$1] == $2) s++}
                END {printf "%d of %d", s, n}' "$MANIFEST" "$NEW_MANIFEST")
    echo "reference centers with the SHA-256 of $MANIFEST: $same"
fi
