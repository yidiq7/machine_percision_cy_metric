#!/bin/bash
# 06: the checks of the published files of the proof, with no GPU and no local forms.
#   The script downloads each file of the Zenodo record that is not in DATA_DIR, and checks:
#   the SHA-256 values of SHA256SUMS; the exact symmetry of the two files of the potential; the
#   sign of p0 on X; the partition of the cover by the records of the bound and of the metric
#   pass, with the manifest of the local forms; the constants of cert_atlas.py combine against
#   the published atlas_combine_v3.npz; and the cover of F by the boxes, at random points.
# Hardware: CPU cores. tmgpu: necessary, except for the first two checks.
# Time: 2.5 min on 8 CPU cores with POINTS=300; the cover check takes longer with more points.
# Variables (default):
#   DATA_DIR (runs/zenodo)    the directory of the files of the record
#   ZENODO_URL (<ZENODO-URL>) the URL of the record
#   OUT (runs/check)          the output of cert_atlas.py combine
#   POINTS (3000), PROCS (16)
DATA_DIR=${DATA_DIR:-runs/zenodo}
source "$(dirname "$0")/common.sh"
ZENODO_URL=${ZENODO_URL:-<ZENODO-URL>}
OUT=${OUT:-runs/check}
POINTS=${POINTS:-3000}
PROCS=${PROCS:-16}
FILES="SHA256SUMS README.txt f6t_data.npz f6t_forms.npz cover_s0.03_c6_v3.npz f6t_local_manifest_v3.txt
       run_0.npz run_1.npz run_2.npz run_3.npz metric_0.npz metric_1.npz metric_2.npz metric_3.npz
       atlas_fs_v3_tol03.npz atlas_trd_v1_010203101213.npz atlas_trd_v1_2021233031.npz
       atlas_combine_v3.npz"
mkdir -p "$DATA_DIR" "$OUT"
for f in $FILES; do
    if [ ! -e "$DATA_DIR/$f" ]; then
        case $ZENODO_URL in
            http*) ;;
            *) echo "$DATA_DIR/$f does not exist: set ZENODO_URL to the URL of the record" >&2; exit 1 ;;
        esac
        curl -fL --retry 3 -o "$DATA_DIR/$f" "$ZENODO_URL/files/$f?download=1"
    fi
done

# 1. The SHA-256 values.
if command -v sha256sum > /dev/null; then
    (cd "$DATA_DIR" && sha256sum -c SHA256SUMS)
else
    (cd "$DATA_DIR" && shasum -a 256 -c SHA256SUMS)
fi
D="$DATA_DIR"
# 2. Each stored matrix is exactly symmetric, and the base blocks are exact S_4 copies.
$RUN scripts/check_file_symmetry.py "$D/f6t_data.npz" "$D/f6t_forms.npz"

need_tmgpu
RUNS="$D/run_0.npz $D/run_1.npz $D/run_2.npz $D/run_3.npz"
METRIC="$D/metric_0.npz $D/metric_1.npz $D/metric_2.npz $D/metric_3.npz"
# 3. The sign of p0 on X.
$RUN scripts/check_p0_sign.py --data "$D/f6t_data.npz" --forms "$D/f6t_forms.npz"
# 4. The partition of the cover, the radii and the bounds of each record, and the manifest.
$RUN scripts/cert_partition.py --run $RUNS --cover "$D/cover_s0.03_c6_v3.npz" \
    --manifest "$D/f6t_local_manifest_v3.txt" --data "$D/f6t_data.npz" --forms "$D/f6t_forms.npz" \
    --procs "$PROCS"
$RUN scripts/cert_partition.py --metric --run $METRIC --cover "$D/cover_s0.03_c6_v3.npz" \
    --manifest "$D/f6t_local_manifest_v3.txt" --data "$D/f6t_data.npz" --forms "$D/f6t_forms.npz" \
    --procs "$PROCS"
# 5. The constants of the paper from the records, against the published file.
$RUN scripts/cert_atlas.py combine --runs $RUNS --metric $METRIC --fs "$D/atlas_fs_v3_tol03.npz" \
    --transitions "$D/atlas_trd_v1_010203101213.npz" "$D/atlas_trd_v1_2021233031.npz" \
    --out "$OUT/atlas_combine.npz"
$RUN - "$OUT/atlas_combine.npz" "$D/atlas_combine_v3.npz" <<'PY'
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
def equal(x, y):
    return np.array_equal(x, y, equal_nan=x.dtype.kind in 'fc' and y.dtype.kind in 'fc')
same = sorted(a.files) == sorted(b.files) and all(equal(a[k], b[k]) for k in a.files)
print('the constants equal those of the published file:', same)
sys.exit(0 if same else 1)
PY
# 6. Random points of F lie in a final box of the bound.
$RUN scripts/cert_check.py --run $RUNS --cover "$D/cover_s0.03_c6_v3.npz" --data "$D/f6t_data.npz" \
    --forms "$D/f6t_forms.npz" --points "$POINTS"
echo "06: all checks pass"
