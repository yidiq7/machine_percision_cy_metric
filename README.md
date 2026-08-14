# Machine-precision Calabi-Yau metric on the Fermat quartic

Stage 1 of a Ricci-flat metric on `Σ_a z_a^4 = 0` in CP^3, with the
algebraic-metric ansatz of Headrick-Nassar
([arXiv:0908.2635](https://arxiv.org/abs/0908.2635)). The goal is to decrease
`mean abs(f)` to approximately 1e-15 in FP64, where `f = 1 − η` and
`η = μ_ω / μ̂`.

`CLAUDE.md` holds the full specification, the conventions, and every
measurement. Read it first.

## Install

The point sampler, the restriction matrix and `ΩΩ̄` come from
[MLGeometry](https://github.com/yidiq7/MLGeometry), which `setup.sh` installs
from PyPI as `MLGeometry-JAX`. This repository does not contain a copy.

```bash
./scripts/setup.sh              # GPU build of JAX
NO_GPU=1 ./scripts/setup.sh     # a machine with no GPU
```

## Test

```bash
.venv/bin/python -m tests.test_blocks     # the symmetry structure, ~10 s
.venv/bin/python -m tests.test_sections   # the potential p, ~30 s
```

`test_blocks` checks `n_θ` against an independent character-theory count, and
`test_sections` checks that `p = (Σ_a abs(z_a)²)^k` at the Fubini-Study point.

## Run

```bash
./scripts/run_cluster.sh                                  # k = 12, 16, 20
.venv/bin/python -u scripts/run_fit.py --k 12 --n-pairs 50000 \
    --n-train 100000 --seed 20260813 --cond
```

Each fit writes `runs/theta_k<k>_seed<seed>.npy` and a JSON record with every
reported number. `run_cluster.sh` takes `SEED`, `PAIRS` and `TRAIN`.

## Results so far

`mean abs(f)` on a hold-out set of 8,000 points, against σ from the earlier fits
of this group, which used a different code and a first-order method:

| k | n_θ | steps | `mean abs(f)` | earlier σ | ratio | `cond(J)` |
|---|-----|-------|---------------|-----------|-------|-----------|
| 2 | 2 | 36 | 1.9070e-2 | 1.90e-2 | 1.00 | 1.0 |
| 4 | 5 | 60 | 1.6713e-3 | 1.68e-3 | 0.99 | 3.1e1 |
| 6 | 10 | 58 | 1.6250e-4 | 1.60e-4 | 1.02 | 3.7e2 |
| 8 | 19 | 73 | 1.6278e-5 | 1.69e-5 | 0.96 | 2.3e4 |
| 10 | 33 | 80 | 1.8202e-6 | 1.85e-6 | 0.98 | 1.9e5 |

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

## Language

Every document, comment and commit message in this repository uses ASD-STE100
Simplified Technical English. `CLAUDE.md` gives the rules.
