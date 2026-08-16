# Machine-precision Calabi-Yau metric on the Fermat quartic

Stage 1 of a Ricci-flat metric on `Σ_a z_a^4 = 0` in CP^3, with the
algebraic-metric ansatz of Headrick-Nassar
([arXiv:0908.2635](https://arxiv.org/abs/0908.2635)). The goal is to decrease
`mean abs(f)` to approximately 1e-15 in FP64, where `f = 1 − η` and
`η = μ_ω / μ̂`.

`CLAUDE.md` holds the full specification, the conventions, and every
measurement. Read it first.

## Install

`uv` manages the environment, and `uv.lock` is committed, so every machine gets
the same versions. The point sampler, the restriction matrix and `ΩΩ̄` come from
[MLGeometry](https://github.com/yidiq7/MLGeometry), which the lock pins as
`MLGeometry-JAX 2.2.2`. This repository does not contain a copy.

```bash
./scripts/setup.sh              # cluster: uv sync --locked --extra gpu
NO_GPU=1 ./scripts/setup.sh     # laptop:  uv sync --locked
```

The `gpu` extra holds `jax[cuda12]`. Those wheels are Linux only, so the laptop
must leave the extra out. `uv sync` makes `.venv`, and `uv run` uses it.

## Test

```bash
uv run python -m tests.test_blocks     # the symmetry structure, ~10 s
uv run python -m tests.test_sections   # the potential p, ~30 s
```

`test_blocks` checks `n_θ` against an independent character-theory count, and
`test_sections` checks that `p = (Σ_a abs(z_a)²)^k` at the Fubini-Study point.

## Run

```bash
./scripts/run_cluster.sh                                  # k = 12, 16, 20
uv run python -u scripts/run_fit.py --k 12 --n-train 100000 \
    --n-eval 8000000 --chunk 200000 --seed 20260813 --cond
```

Each fit writes `runs/theta_k<k>_seed<seed>.npy` and a JSON record with every
reported number. `run_cluster.sh` takes `SEED`, `TRAIN`, `EVAL` and `CHUNK`.

The fit set and the hold-out set are two independent samples, because they have
different limits:

* **fit set: memory.** LM materializes `[J; sqrt(λ)I]`, which is
  `N_fit · n_θ · 8` bytes, and lineax QR needs about 3 copies. A large fit set
  also buys little, because the excess risk is only `n_θ / N_fit`. Use 1e5 to
  k = 20, and 5e5 at k ≥ 32.
* **hold-out set: one forward pass in chunks.** The standard error of
  `mean abs(f)` is `1/sqrt(N)`, so 8e6 points give 0.035 percent. `max abs(f)`
  also grows with N.

## The analytic path, not yet merged

`quartic/analytic.py` gives `φ` from the Gram sums of the holomorphic section
vector and its 4 derivatives, and `∂v/∂θ` from Jacobi's formula. The present
`metric.py` and `fit.py` do not use it, so no result changes.

On the laptop CPU the Jacobian became 10 times faster at k = 6 and 110 times
faster at k = 16, with the same answer to 1.5e-15. A GPU has a different balance
of arithmetic against bandwidth, so measure it there before you trust it:

```bash
uv run python -u scripts/bench_analytic.py --k 6 8 12 16 20 \
    --n-points 20000 --budget-gb 20 --out bench_gpu.json
uv run python -u scripts/bench_analytic.py --k 8 --fit      # end to end, slower
```

The script reports the device, the time and the memory of each configuration,
and how far the analytic result sits from the present one. Each configuration is
guarded, so an out-of-memory case prints `OOM` and the table continues.

## Results so far

`mean abs(f)` on a hold-out set, against σ from the earlier fits of this group,
which used a different code and a first-order method. The last column is the
standard error of the reported number.

| k | n_θ | steps | `mean abs(f)` | earlier σ | ratio | `cond(J)` | N | ± |
|---|-----|-------|---------------|-----------|-------|-----------|---|---|
| 2 | 2 | 36 | 1.9070e-2 | 1.90e-2 | 1.00 | 1.0 | 8e3 | 1.1% |
| 4 | 5 | 59 | 1.6655e-3 | 1.68e-3 | 0.99 | 3.1e1 | 2e6 | 0.07% |
| 6 | 10 | 65 | 1.6268e-4 | 1.60e-4 | 1.02 | 3.7e2 | 2e6 | 0.07% |
| 8 | 19 | 73 | 1.6278e-5 | 1.69e-5 | 0.96 | 2.3e4 | 8e3 | 1.1% |
| 10 | 33 | 80 | 1.8202e-6 | 1.85e-6 | 0.98 | 1.9e5 | 8e3 | 1.1% |

`h → λh` leaves the metric unchanged, so `J` has one exact null direction.
`fit.condition_number` drops it before it reports `cond(J)`.

## Layout

| Path | Contents |
|---|---|
| `quartic/blocks.py` | the 64 residue blocks, the S₄ action, the invariant basis of `h` |
| `quartic/sections.py` | the potential `p` at degree k, and the Fubini-Study point |
| `quartic/metric.py` | `v = det(g\|_X)/ΩΩ̄`, the residual `f`, and the reported numbers |
| `quartic/data.py` | point sets from the MLGeometry sampler, with a seed |
| `quartic/fit.py` | the LM fit on MSPE, and `cond(J)` |
| `notes/symmetry_param_counts_s4.py` | the exact count of `n_θ` from character theory |
| `scripts/` | setup, one fit, and the cluster job |
| `tests/` | the tests above |
| `pyproject.toml`, `uv.lock` | the environment; `uv sync --locked` reproduces it |

## Language

Every document, comment and commit message in this repository uses ASD-STE100
Simplified Technical English. `CLAUDE.md` gives the rules.
