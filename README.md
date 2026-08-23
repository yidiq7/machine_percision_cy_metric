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
uv run python -m tests.test_sections   # the potential p, ~40 s
uv run python -m tests.test_dd         # the double-double arithmetic, ~30 s
uv run python -m tests.test_analytic   # the analytic phi and Jacobian, ~5 min
```

* `test_blocks` checks `n_θ` against an independent character-theory count.
* `test_sections` checks that `p = (Σ_a abs(z_a)²)^k` at the Fubini-Study point,
  to k = 38.
* `test_dd` checks the error-free transformations against mpmath, **through
  `jax.jit`**. Keep it that way: the fault it guards against does not appear
  outside `jax.jit`.
* `test_analytic` holds the analytic path against nested autodiff, and holds a
  whole fit against a whole fit.

## Run

```bash
STAGE=dry   ./scripts/run_cluster.sh    # size the blocks, report memory, stop
STAGE=task1 ./scripts/run_cluster.sh    # k = 12, 16, 20 in FP64
STAGE=high  ./scripts/run_cluster.sh    # k = 24, 28 with the Gram stage in dd
```

`STAGE=dry` first: it compiles the real solve and prints the fit-set size, the
block size and the measured peak, so an out-of-memory run costs seconds and not
the job. `K=28`, `SEED`, `TRAIN`, `EVAL`, `BUDGET` all override. One fit takes
about a minute in FP64 at k = 20, and about 45 minutes at k = 28 in dd.

For one degree directly:

```bash
uv run python -u scripts/run_fit.py --k 28 --n-train 200000 \
    --n-eval 8000000 --gram-dd on --max-steps 600 --seed 20260813 --cond
```

Each fit writes `runs/theta_k<k>_seed<seed>.npy` and a JSON record with every
reported number, including `gram_dd` and the sizes used.

The fit set and the hold-out set are two independent samples, because they have
different limits:

* **fit set: the intermediates of the Jacobian.** `fit.plan_fit` compiles the
  real solve and sizes both the fit set and the `lax.map` block against the
  budget. It takes only as many fit points as `--target-risk` needs, which is
  `n_θ / N_fit` and defaults to 5e-3, and gives the rest of the budget to the
  block. That order matters: at k = 20 a planner that maximized the fit set took
  141,948 points, was left with a block of 64, ran 2.5 times longer and moved σ
  by 0.04 percent. Use `--target-risk 1e-3` for a number you intend to publish.
  The planner refuses any plan below `4 n_θ` points rather than return a fit that
  carries no information.
* **hold-out set: one forward pass in chunks.** The standard error of
  `mean abs(f)` is `1/sqrt(N)`, so 8e6 points give 0.035 percent. `max abs(f)`
  also grows with N, and it is a lower bound on `sup abs(f)`.

`h → λh` leaves the metric unchanged, and `θ` is scaled so that the Fubini-Study
point is ±1. Without that scaling `θ_FS` carries the multinomial coefficients,
which reach 5.1e11 at k = 24, and LM stops after 2 steps because its damping and
its step test are both absolute.

## The analytic Jacobian

`quartic/analytic.py` gives `φ` from the Gram sums of the holomorphic section
vector and its 4 derivatives, and `∂v/∂θ` from Jacobi's formula. **The fit uses
it by default.** `--autodiff-jac` selects the old nested-autodiff path, and the
tests hold the two against each other.

Jacobian time on one A100, at 20,000 points:

| k | 6 | 8 | 12 | 16 | 20 |
|---|---|---|----|----|----|
| autodiff | 0.070 s | 0.192 s | 0.902 s | 3.405 s | 12.158 s |
| analytic | 0.005 s | 0.010 s | 0.021 s | 0.047 s | 0.096 s |
| gain | 14.7x | 19.4x | 42.1x | 72.3x | **126x** |
| memory | 3.11→0.41 GB | 5.62→0.91 GB | 32.1→1.44 GB | 26.2→0.73 GB | 22.8→0.40 GB |

`optimistix` keeps its own trust region: the residual carries a `jax.custom_jvp`
whose costly part depends on the primal alone, so `jax.vmap` over the basis of θ
computes it once. Reproduce the table with:

```bash
uv run python -u scripts/bench_analytic.py --k 6 8 12 16 20 \
    --n-points 20000 --budget-gb 20 --out bench_gpu.json
```

## Precision

`quartic/dd.py` gives double-double arithmetic, and `--gram-dd on` puts the Gram
stage of the evaluation in it. `scripts/oracle_precision.py` measures the floor
under `mean abs(f)` against mpmath at 50 digits, and
`scripts/oracle_stages.py` says which stage of the evaluation carries the error.

| k | floor, FP64 | floor, `--gram-dd on` |
|---|---|---|
| 24 | 6.04e-14 | 4.33e-15 |
| 28 | 3.97e-13 | **1.61e-14** |
| 32 | 5.07e-12 | 3.03e-14 |

**Report no σ that sits below the floor.** At k = 28 in FP64 the fit returned
1.7077e-13 against a floor of 3.97e-13, which means it fitted round-off and the
number is not a measurement.

The cost of `--gram-dd on` is 41 times the time and 14 times the memory at
k = 28. The floor still grows with k, by 1.28 for each unit, so the useful limit
moves from k = 27.1 to k = 28.6. Reaching 1e-15 needs stages 1 to 3 of the
evaluation in dd as well, and the oracle says those give a floor of 1.6e-15 that
does not grow. `analytic.section_vectors_closed` is the piece that work needs,
because `dd.split` carries no derivative.

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
| 12 | 56 | 103 | 2.2018e-7 | 2.96e-7 | 0.74 | 2.2e7 | 8e6 | 0.04% |
| 16 | 135 | 112 | 3.5096e-9 | 8.5e-9 | 0.41 | 2.1e10 | 8e6 | 0.04% |
| 20 | 285 | 140 | 6.3276e-11 | 6.3e-10 | 0.10 | 2.3e13 | 8e6 | 0.04% |
| 24 | 542 | 117 | 1.2566e-12 | — | — | 5.4e16 | 8e6 | 0.04% |

Reproduced after the scaling of θ and the multinomial fix, on 8e6 hold-out
points: 2.1950e-07, 3.5140e-09 and 6.3524e-11 at k = 12, 16 and 20, in 71, 92
and 106 LM steps against 103, 112 and 140 before.

The rate is 2.81, 2.73 and 2.67 across 12 → 16 → 20 → 24, against the
Headrick-Nassar prediction of 2.83. k = 28 needs `--gram-dd on`, because in FP64
its σ falls below the floor.

`J` has one exact null direction, from the scale of `h`.
`fit.condition_number` drops it before it reports `cond(J)`.

## Layout

| Path | Contents |
|---|---|
| `quartic/blocks.py` | the 64 residue blocks, the S₄ action, the invariant basis of `h` |
| `quartic/sections.py` | the potential `p` at degree k, and the Fubini-Study point |
| `quartic/metric.py` | `v = det(g\|_X)/ΩΩ̄`, the residual `f`, and the reported numbers |
| `quartic/data.py` | point sets from the MLGeometry sampler, with a seed |
| `quartic/analytic.py` | `φ` from Gram sums, `∂v/∂θ` from Jacobi's formula |
| `quartic/dd.py` | double-double arithmetic that survives the XLA simplifier |
| `quartic/fit.py` | the LM fit on MSPE, `plan_fit`, and `cond(J)` |
| `notes/symmetry_param_counts_s4.py` | the exact count of `n_θ` from character theory |
| `scripts/setup.sh`, `run_cluster.sh`, `run_fit.py` | environment and fits |
| `scripts/oracle_precision.py` | the floor on `abs(f)`, against mpmath |
| `scripts/oracle_stages.py` | which stage of the evaluation carries the error |
| `scripts/bench_analytic.py` | the analytic path against nested autodiff |
| `tests/` | the tests above |
| `pyproject.toml`, `uv.lock` | the environment; `uv sync --locked` reproduces it |

## Language

Every document, comment and commit message in this repository uses ASD-STE100
Simplified Technical English. `CLAUDE.md` gives the rules.
