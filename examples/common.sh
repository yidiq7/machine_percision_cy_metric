# The values and functions that all examples share. Each example sources this file.
# It stops at the first failed command, and it changes to the repo root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# UV_EXTRA: gpu (all steps, with tmgpu), cuda (the fits on a GPU, no tmgpu) or none (no GPU).
# RUN: the command that starts Python. Set it to use an environment of your own.
UV_EXTRA=${UV_EXTRA:-gpu}
if [ -n "${RUN:-}" ]; then
    :
elif [ "$UV_EXTRA" = none ]; then
    RUN="uv run --locked python -u"
else
    RUN="uv run --extra $UV_EXTRA --locked python -u"
fi

# JAX takes this part of the memory of the GPU. The large fits need almost all of it.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
# CuPy compiles its kernels with the libraries of the CUDA toolkit in CUDA_PATH, if it is set.
if [ -n "${CUDA_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# The files of the potential, the cover and the manifest, and the directory of the local forms.
DATA_DIR=${DATA_DIR:-runs_tm}
export QUARTIC_LOCAL_FORMS=${QUARTIC_LOCAL_FORMS:-$DATA_DIR/tm_local_f6t_td_v3}
COVER=${COVER:-$DATA_DIR/cover_s0.03_c6_v3.npz}
MANIFEST=${MANIFEST:-$DATA_DIR/f6t_local_manifest_v3.txt}

# Stop with a clear message if the package tmgpu does not import.
need_tmgpu() {
    if ! $RUN -c "import tmgpu" > /dev/null 2>&1; then
        echo "This step needs the package tmgpu, which is not public yet (see README.md)." >&2
        echo "The fits, the jets and the checks of a fit do not need it." >&2
        exit 1
    fi
}

# parts N CMD ...: run "CMD ... G" for G = 0 to N - 1 in sequence, or only for G = PART.
parts() {
    local n=$1 g
    shift
    if [ -n "${PART:-}" ]; then
        "$@" "$PART"
    else
        for ((g = 0; g < n; g++)); do
            "$@" "$g"
        done
    fi
}
