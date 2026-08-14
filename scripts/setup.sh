#!/usr/bin/env bash
# Build the virtual environment for the Fermat quartic fit. Run this once.
# On a machine with no GPU, set NO_GPU=1.
set -euo pipefail
cd "$(dirname "$0")/.."

JAX_SPEC="jax[cuda12]"
if [ "${NO_GPU:-0}" = "1" ]; then
    JAX_SPEC="jax"
fi

uv venv --python 3.12 .venv

# Install jax first, so that the GPU build wins over the plain dependency that
# MLGeometry-JAX declares.
uv pip install --python .venv/bin/python "$JAX_SPEC"
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -c "
import jax, optimistix, optax, flax, MLGeometry
print('jax', jax.__version__, '| optimistix', optimistix.__version__,
      '| optax', optax.__version__)
print('MLGeometry from', MLGeometry.__file__)
print('devices:', jax.devices())
"

echo
echo "Now run the tests:"
echo "  .venv/bin/python -m tests.test_blocks"
echo "  .venv/bin/python -m tests.test_sections"
