#!/bin/bash
# The test suites of the repo, in one run.
#   The 15 suites of the fits need no tmgpu. The 9 suites of the strict bound need tmgpu and a
#   GPU; without tmgpu, the script names them as skipped. Some of them also need data files:
#   the script names each suite that it skips for this reason, and the file that it needs.
#   Exit status 1 if a suite fails.
# Hardware: one GPU (the suites of the fits also run on CPU cores, slowly). tmgpu: for the
#   suites of the strict bound.
# Time: about 30 min on one H200.
# Variables: UV_EXTRA (gpu), DATA_DIR (runs_tm), QUARTIC_LOCAL_FORMS, SUITES (all).
source "$(dirname "$0")/common.sh"
FIT_SUITES="test_analytic test_arnoldi test_arnoldi_dd test_blocks test_correction test_dd
    test_dd_metric test_invariant test_joint_dd test_ozaki test_prepare test_sections test_tjets
    test_volume test_whiten"
CERT_SUITES="test_cert_imports test_cert_records test_cert_compare test_cert_atlas test_cert_steps
    test_tm_host test_box_reference test_box_steps test_metric_bounds"
SUITES=${SUITES:-$FIT_SUITES $CERT_SUITES}
R=runs_tm
V3="$R/full_f6t_v3/run_0.npz $R/full_f6t_v3/run_1.npz $R/full_f6t_v3/run_2.npz $R/full_f6t_v3/run_3.npz"
L297="$QUARTIC_LOCAL_FORMS/center_297.npz"

# The data files that a suite needs, and how to make them.
needs() {
    case $1 in
    test_tm_host) echo "$R/cover_s0.03_c6_v2.npz" ;;
    test_box_reference) echo "$R/f6t_data.npz $R/f6t_forms.npz $R/cover_s0.03_c6_v2.npz $R/steps_b1/B1_x1.npz" ;;
    test_box_steps) echo "$R/cover_s0.03_c6_v3.npz $R/f6t_data.npz $R/f6t_forms.npz $V3 $L297" ;;
    test_metric_bounds) echo "$R/cover_s0.03_c6_v3.npz $L297" ;;
    esac
}
HOW="The cover: scripts/cert_cover.py (examples/03_local_forms.sh). The local forms:
examples/03_local_forms.sh. The records of the run v3 and the two files of the potential: the
Zenodo record (examples/06_check_published.sh). The box B1: scripts/cert_steps.py export."

# True if $1 is a suite of the strict bound.
is_cert() {
    local c
    for c in $CERT_SUITES; do
        [ "$c" = "$1" ] && return 0
    done
    return 1
}

tmgpu=1
$RUN -c "import tmgpu" > /dev/null 2>&1 || tmgpu=0
failed=() skipped=() passed=()
for t in $SUITES; do
    if is_cert "$t" && [ $tmgpu = 0 ]; then
        skipped+=("$t (no tmgpu)")
        continue
    fi
    lack=""
    for f in $(needs "$t"); do
        [ -e "$f" ] || lack="$lack $f"
    done
    if [ -n "$lack" ]; then
        skipped+=("$t (no file$lack)")
        continue
    fi
    echo "=== $t"
    s0=$SECONDS
    if $RUN -m "tests.$t" 2>&1 | grep -v -E "UserWarning|warnings.warn" | tail -3; [ "${PIPESTATUS[0]}" = 0 ]; then
        passed+=("$t")
        echo "$t: pass, $((SECONDS - s0)) s"
    else
        failed+=("$t")
        echo "$t: FAIL, $((SECONDS - s0)) s"
    fi
done
echo
echo "passed: ${#passed[@]}"
for s in ${skipped[@]+"${skipped[@]}"}; do echo "skipped: $s"; done
if [ ${#skipped[@]} -gt 0 ]; then echo "$HOW"; fi
for s in ${failed[@]+"${failed[@]}"}; do echo "failed: $s"; done
if [ ${#failed[@]} -gt 0 ]; then exit 1; fi
echo "RESULT: PASS"
