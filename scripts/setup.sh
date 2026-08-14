#!/usr/bin/env bash
# Build the environment from uv.lock, so that every machine gets the same
# versions. The cuda12 wheels are Linux only, so the GPU extra is optional.
#
#   cluster:  ./scripts/setup.sh
#   laptop:   NO_GPU=1 ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${NO_GPU:-0}" = "1" ]; then
    uv sync --locked
else
    uv sync --locked --extra gpu
fi

uv run python -c "
import jax, optimistix, optax, flax, MLGeometry
print('jax', jax.__version__, '| optimistix', optimistix.__version__,
      '| optax', optax.__version__, '| flax', flax.__version__)
print('MLGeometry from', MLGeometry.__file__)
print('devices:', jax.devices())
"

echo
echo "Now run the tests:"
echo "  uv run python -m tests.test_blocks"
echo "  uv run python -m tests.test_sections"
