# Machine-precision Calabi-Yau metric on the Fermat quartic

This repository computes an approximate Ricci-flat metric on the Fermat quartic
X = {Σ_a z_a⁴ = 0} in CP³, in the Kähler class k₀ H, and measures its residual. For
one fit it also proves a strict bound of the residual and of its gradient on all of
X, with Taylor models on a graphics processing unit (GPU).

```
g = ∂∂̄Φ on X, in the chart coordinates t
v = det(g) abs(∂P/∂z_m)²         the volume form of g over the measure of Ω
v̄ = 512 π⁴ k₀² / Γ(1/4)⁸         the exact mean of v (quartic/volume.py)
f = v / v̄ − 1                    the residual; 1 − 1/η with η = v / v̄ is equal to first order
σ = Σ_i w_i abs(f_i),  w_i = mass_i / Σ mass, on a hold-out sample
ρ = min_a abs(z_a) / max_a abs(z_a), the distance of a point from the curves z_a = 0
```

Abbreviations: dd is double-double (two FP64 values, about 32 digits), td is
triple-double (three FP64 values, about 48 digits), FP64 is the 64-bit binary
floating-point format, HN is Headrick-Nassar
([arXiv:0908.2635](https://arxiv.org/abs/0908.2635)), V+A is Vandermonde with
Arnoldi, LM is Levenberg-Marquardt, and MSPE is the mean squared percentage error.

## Results

### The final fit, s2e

s2e has the top degree 36 and two stages (see "The method"). The collaborators
asked for ‖f‖_{H¹} ≤ 1e-23 and [f]_{0,α} ≤ 1e-8. s2e meets both.

| quantity | value | sample |
|---|---|---|
| σ, mpmath | 2.7104e-28 ± 1.6e-29 | 256 points |
| σ, dd | 2.7374e-28 | 8e6 hold-out points |
| rms(f) = ‖f‖_L² | 4.346e-28 | 8e6 hold-out points |
| max abs(f) | 1.363e-25 | 8e6 hold-out points; the 12 largest have ρ ≤ 0.018 |
| ‖grad f‖_L² | 1.2231e-25 | 4 × 32,768 points, from Taylor jets |
| ‖Δf‖_L² | 3.5842e-23 | the same |
| ‖Ric‖_L² | 2.5582e-23 | the same |

* The norms of the derivatives use the metric of volume 1. So ‖f‖_{H¹} ≈ 1.2e-25.
* Between two charts, the mean abs change of f is 6.9 percent of σ. At 12 points
  from ρ = 0.95 to 0.05, the mean abs difference of dd and mpmath is 6.1 percent of σ.
* A sample maximum is a lower bound of sup abs(f). It is not a bound.

### Higher top degrees

| fit | top degree | σ, mpmath (256 points) |
|---|---|---|
| s2k | 40 | 9.8412e-30 ± 6.2e-31 |
| s2o | 44 | 2.1273e-31 ± 1.2e-32 |

From the top degree 40 the Taylor jets miss f by about 9 times σ, so these fits
have no derivatives. At the top degree 44 the dd noise has the size of σ, and only
the mpmath sample gives σ.

### The strict bound, f6t

f6t is a fit with one stage and the top degree 24: an HN base at k₀ = 16 and the
levels 24 down to 13. Its σ is 1.7259e-22 on 400,000 hold-out points. For f6t, on all
of X:

```
sup abs(f)      ≤ 4.9984e-20
sup abs(grad f) ≤ 3.7497e-16    in the metric of volume 1
```

* The bound holds for the potential in two files of the run, with these SHA-256
  values:
  * `f6t_data.npz`: 69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1
  * `f6t_forms.npz`: 2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2
* The largest abs(f) at a box center is 4.063e-21, so the bound is within a factor
  12 of the true sup.
* The box code does not have the second stage yet, so s2e has no strict bound.
* The run took 38,215 GPU-seconds on H200 GPUs (2.9 hours on 4 GPUs).

The steps of the proof:

1. The potential is exactly invariant under S₄, the phases z_a → i z_a and the
   complex conjugation. So a bound on a fundamental region F in the chart z₃ = 1 is
   a bound on X.
2. `scripts/cert_cover.py` covers F with 113,357 cubes of side 0.03, which share 332
   reference centers. Interval arithmetic proves that each dropped cube misses F.
3. For each box, `scripts/cert_run.py` computes a Taylor model of f in four complex
   variables (τ and conj(τ) as independent variables), of order 28 or 32. The forms
   of the potential come from arb at the reference centers
   (`scripts/tm_local_forms.py`), stored in td. Each round-off on the GPU goes into
   the remainder with a strict bound. A box above the target 5e-20 takes order 32,
   and then splits.
4. A Cauchy estimate gives g from the model of Φ on a larger polydisc. Then f, grad f
   and their bounds follow. 134,888 final boxes cover F.
5. Independent checks:
   * `scripts/cert_partition.py` proves that the final boxes fill each cell of the
     cover, that each dropped cell misses F, and that the radii of all records hold
     again; with `--manifest` it ties the records to checked files of local forms;
   * `scripts/cert_local_values.py` checks the 332 files of local forms against the
     potential through another path;
   * `scripts/cert_check.py` checks random points of F, compares the model of f at
     random boxes with arb, and computes boxes again in new batches.
   Each check exits with status 1 if it fails.
6. The code of 2026-10-13 gives records equal bit for bit to those of commit 8f0c1e8,
   on the 4,459 final boxes of 4 reference centers. The box code of 8f0c1e8 differs from
   that of the run (commit 7c07f0d) only in comments.

## The method

### The potential

```
p₀ = s(z)* H s(z)            the HN base at degree k₀; s(z) the sections, H real symmetric and S₄-invariant
G  = W_L q,   W_L = (abs(z)² p₀^(−1/k₀))^L,   q from the V+A recurrence of level L
Φ₁ = log p₀ + ψ₁,   ψ₁ = Σ c_t G_t  over the levels S₁ = {24, …, 17}
Φ  = Φ₁ + ψ₂,       ψ₂ = Σ c'_t abs(z)^(2L) exp(−(L/k₀) Φ₁) q_t  over S₂ = {36, …, 29}
```

Each G is homogeneous of degree 0 and invariant under the symmetry group, so ψ₁ and
ψ₂ are functions on X, and g = ∂∂̄Φ is in the class k₀ H. For s2e, k₀ = 12.

### The steps

1. **The HN base.** An LM fit of H on the MSPE (`quartic/fit.py`), which
   `scripts/vpa_dd_levels.py` runs when it gets no `--theta`.
2. **The basis of a level.** The V+A recurrence in dd, with Ozaki matrix products
   (`quartic/arnoldi_dd.py`, `quartic/vpa_dd.py`, `quartic/ozaki.py`).
3. **The joint basis of several levels.** A pivoted Gram-Schmidt, top level first,
   with a cutoff on the new part of each function (`quartic/joint_dd.py`). ψ_N, the
   projection of 1, takes its own column.
4. **The solve.** v is linear in the coefficients to second order in δg/g, so one
   linear least-squares solve gives them. Correction steps from the dd residual
   finish it.
5. **The second stage** (`scripts/vpa_dd_stage2.py`) does steps 2 to 4 again with
   the weights of Φ₁.
6. **The evaluation** is in dd from end to end, at points that one dd Newton step
   puts on X (`quartic/dd_metric.py`). The references come from mpmath
   (`scripts/level_oracle.py`, `scripts/stage2_oracle.py`).
7. **The derivatives** come from Taylor jets of bidegree (2, 2) in complex dd
   (`quartic/tjets.py`, `scripts/residual_jets.py`).

## Install

`uv` manages the environment, and `uv.lock` is committed, so every machine gets the
same versions. The point sampler, the restriction matrix and ΩΩ̄ come from
[MLGeometry](https://github.com/yidiq7/MLGeometry), which the lock pins as
`MLGeometry-JAX 2.2.2`. This repository does not contain a copy.

```bash
./scripts/setup.sh              # Linux with an NVIDIA GPU: uv sync --locked --extra gpu
NO_GPU=1 ./scripts/setup.sh     # other machines: uv sync --locked
```

* The `gpu` extra holds `jax[cuda12]`. Those wheels are for Linux only.
* Each `uv run` must have the same extras as the sync. A `uv run` with no
  `--extra gpu` removes the CUDA build of JAX again.
* The strict bound also needs python-flint and CuPy. The lock fixes python-flint 0.9.0
  (a dependency) and cupy-cuda12x 14.2.0 (in the `gpu` extra), the versions of the full
  run, because the host numbers of the bound depend on the rounding of python-flint.
* CuPy compiles the kernels with the CUDA 12 toolkit. Set `CUDA_PATH` to the
  toolkit, and add `$CUDA_PATH/lib64` to `LD_LIBRARY_PATH`. Without it, CuPy cannot
  open `libnvrtc-builtins`.

## Test

Run the tests on a Linux machine with an NVIDIA GPU:

```bash
for t in blocks sections dd analytic volume invariant ozaki arnoldi arnoldi_dd \
         joint_dd dd_metric tjets correction gradient gradient_dd prepare whiten; do
    uv run --extra gpu python -m tests.test_$t
done
for t in tests.test_tm_host tests.test_cert_compare tests.test_cert_imports \
         tests.test_cert_records tmgpu.tests.test_tm tmgpu.tests.test_tmtd \
         tmgpu.tests.test_gemm tmgpu.tests.test_host tmgpu.tests.test_boundary; do
    uv run --extra gpu python -m $t
done
```

* `test_dd` and `test_ozaki` run the dd code under `jax.jit`, because the XLA
  simplifier can remove the error terms of dd. Keep it that way.
* The tests of `tmgpu` check the Taylor-model operations and their error bounds
  against mpmath and exact rationals. `test_rounding_bounds` builds cases with no
  truncation, so that the remainder is the bound of the round-off alone, and the same
  check must fail when the rounding constant is 0.
* `test_boundary` checks that `tmgpu` imports only the standard library, numpy,
  cupy, mpmath, python-flint and itself. `test_cert_imports` checks that the
  certificate code loads no JAX.

## Run

Each script gives its usage in its docstring.

**The fit:**

1. `scripts/vpa_dd_levels.py`: the HN base, then the first stage.
2. `scripts/vpa_dd_stage2.py`: the second stage.
3. The checks: `scripts/seam_levels.py` (f in two charts), `scripts/stage2_oracle.py`
   (mpmath at 12 points and at the 256-point sample), and `scripts/residual_jets.py`
   (the derivatives).

**The strict bound:**

1. `scripts/level_oracle.py export --invariant-base`: the data of a fit with one
   stage.
2. `scripts/level_matrices.py`: the forms N_L of the levels as Hermitian matrices.
3. `scripts/cert_cover.py`: the cover of F.
4. `scripts/tm_local_forms.py --centers <cover>`: the local forms at the reference
   centers, in arb, stored in td.
5. `scripts/cert_run.py`: the boxes on the GPU, with the bounds of abs(f) and of
   abs(grad f), and a file of records with the git commit and the SHA-256 of each file
   of local forms. A run on several GPUs takes `--refs` for each part, and the checks
   take the files of all parts.
6. `scripts/cert_local_values.py --manifest <file>`, `scripts/cert_partition.py
   --manifest <file> --data <data> --forms <forms>` and `scripts/cert_check.py`: the
   checks. The manifest ties the files of local forms to the files of the potential, and
   cert_partition ties the records to the manifest.
7. `scripts/cert_compare.py A.npz B.npz --exact`: a bit-for-bit comparison of two
   files of records.

## Layout

| path | content |
|---|---|
| `quartic/blocks.py`, `sections.py`, `invariant.py` | the symmetry blocks, the HN potential, and the exactly invariant base |
| `quartic/dd.py`, `ozaki.py` | dd arithmetic that survives the XLA simplifier, and dd matrix products from FP64 slices |
| `quartic/arnoldi_dd.py`, `vpa_dd.py`, `joint_dd.py` | the V+A bases of the levels, and the joint basis, in dd |
| `quartic/dd_metric.py`, `metric.py`, `volume.py` | v and f in dd and in FP64, and the exact volumes |
| `quartic/tjets.py`, `gradient.py`, `gradient_dd.py` | the derivatives of f |
| `quartic/fit.py`, `analytic.py`, `data.py` | the LM fit of the HN base, the analytic Jacobian, and the point sets |
| `quartic/arnoldi.py`, `correction.py`, `whiten.py` | the earlier FP64 forms of the levels and of the base |
| `quartic/cert/` | the strict bound for the quartic: the frames of the boxes, the forms, the box pipeline and the cover |
| `tmgpu/` | a general library: Taylor models in dd and td on the GPU (CuPy), with a strict bound of each round-off |
| `scripts/vpa_dd_levels.py`, `vpa_dd_stage2.py`, `vpa_dd_fit.py`, `vpa_levels_combine.py` | the fits |
| `scripts/level_oracle.py`, `stage2_oracle.py`, `seam_levels.py`, `vpa_dd_oracle.py` | the checks of the fits against mpmath and in two charts |
| `scripts/residual_jets.py`, `residual_derivatives.py`, `residual_regions.py`, `vpa_f_stats.py` | the derivatives and the distribution of f |
| `scripts/cert_*.py`, `tm_local_forms.py`, `level_matrices.py` | the strict bound |
| `scripts/tm_box.py` | the first Taylor model in arb; `cert_check.py` takes f at a point from it |
| `scripts/oracle_*.py`, `correction_*.py`, `lm_*.py`, `vpa_fit.py`, `vpa_oracle.py`, `vpa_stability.py`, `arnoldi_test.py`, `basis_condition.py`, `bench_analytic.py`, `gradient_dd_report.py` | studies of the precision and of earlier designs |
| `scripts/run_fit.py`, `run_cluster.sh`, `report_theta.py`, `export_h.py`, `lift.py` | the HN fits in FP64 of the first period, and tools for a saved θ |
| `scripts/setup.sh`, `pyproject.toml`, `uv.lock` | the environment |
| `tests/`, `tmgpu/tests/` | the tests |
| `notes/symmetry_param_counts_s4.py` | the exact count of the invariant parameters n_θ |
| `notes/quartic_results.tex` | a write-up of 2026-09-02; its results are out of date |
| `docs/superpowers/` | the design documents of the strict bound and of the split into `tmgpu` |

## The path to the final fit

| date | method | best σ |
|---|---|---|
| 2026-08 | HN, LM in FP64 | 1.2e-12 at k = 24 |
| 2026-09 | HN, LM with the evaluation in dd | 3.2e-14 at k = 28 |
| 2026-09-30 | one correction level in the product basis | 2.2e-15 |
| 2026-10-02 | one V+A level in dd, with ψ_N | 1.1e-17 |
| 2026-10-03 | levels 44, 40 and 36 in one joint basis | 1.9e-25 |
| 2026-10-05 | 8 levels with a degree step of 1, one stage | 3.5e-27 at the top degree 36 |
| 2026-10-07 | two stages, s2e | 2.7e-28 |
| 2026-10-10 | the strict bound of f6t | sup abs(f) ≤ 5.0e-20 |

## Language

Every document, comment and commit message in this repository uses ASD-STE100
Simplified Technical English (STE).
