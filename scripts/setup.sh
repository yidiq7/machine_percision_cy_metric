#!/usr/bin/env bash
# Build the environment from uv.lock, so that every machine gets the same
# versions. The cuda12 wheels are Linux only, so the GPU extra is optional.
#
#   cluster:  ./scripts/setup.sh
#   laptop:   NO_GPU=1 ./scripts/setup.sh
#
# **Every `uv run` must carry the same extras as the sync.** `uv run` syncs the
# environment before it runs, and a `uv run` with no `--extra gpu` removes
# jax[cuda12] again. That is silent: JAX then prints "a CUDA-enabled jaxlib is
# not installed" and falls back to the CPU, and a memory budget meant for an
# 80 GB card becomes a request for 70 GB of host memory.
set -euo pipefail
cd "$(dirname "$0")/.."

# A plain string, not an array: `set -u` with bash 3.2 rejects an empty array.
if [ "${NO_GPU:-0}" = "1" ]; then
    EXTRA=""
else
    EXTRA="--extra gpu"
fi

# shellcheck disable=SC2086
uv sync --locked $EXTRA

# shellcheck disable=SC2086
uv run $EXTRA python -c "
import jax, optimistix, optax, flax, MLGeometry
print('jax', jax.__version__, '| optimistix', optimistix.__version__,
      '| optax', optax.__version__, '| flax', flax.__version__)
print('MLGeometry from', MLGeometry.__file__)
dev = jax.devices()[0]
print('devices:', jax.devices())
stats = (dev.memory_stats() or {}) if dev.platform == 'gpu' else {}
print('device memory:', round(stats.get('bytes_limit', 0) / 1e9, 1), 'GB')
"

echo
echo "Run everything with the same extras, for example:"
if [ "${NO_GPU:-0}" = "1" ]; then
    echo "  uv run python -m tests.test_blocks"
else
    echo "  uv run --extra gpu python -m tests.test_blocks"
    echo "  ./scripts/run_cluster.sh          # does this for you"
fi
