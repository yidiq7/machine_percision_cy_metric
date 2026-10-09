# A strict bound of sup abs(f) with Taylor models on the GPU

Date: 2026-10-09. Status: approved in the chat (approach A); built in stages. Revised on
2026-10-09 after the measurements of `scripts/tm_ref.py` (see "Measurements").

## Purpose

The draft (`notes/draft-main.tex`) needs `‖f‖_Y = ‖f‖'_{C^{0,α}} + a R^(−2−n/q) ‖f‖_{L²}`.
Its work plan asks to "rigorously bound the residuals (for now just C⁰)". One strict
bound of sup abs(f) gives both terms, because `‖f‖_{L²} ≤ vol^(1/2) sup abs(f)`. The
C^{0,α} term also needs `sup abs(grad f) ≲ 1e-6 / R`, which is a coarse bound.

**The output** is two numbers with a record of every box:

* `B_f ≥ sup over X of abs(f)`,
* `B_g ≥ sup over X of abs(grad f)_G`, with G the metric of volume 1.

**The first fit** is D = 20 (`r20`, its coefficients on the invariant base). If its
`B_f` is not enough for the L² term and the run takes less than one GPU-hour, the
next fit is D = 24.

## The certified potential

```
Φ = log p̄₀ + Σ_L N_L p̄₀^(−L/k₀)
```

* `p̄₀` is the HN base in the monomial basis of each block (`quartic/invariant.py`),
  exactly invariant under S₄, the phases and the conjugation.
* `N_L` is the form of level L from `scripts/level_matrices.py`: a sum over S₄ of one
  block of each orbit, with the matrices C_o rounded to dd.
* The stored dd numbers define the potential. Every other number of the computation
  carries a strict bound of its distance to the exact value for this potential.

## The domain

* The chart is `z₃ = 1` with `y = (z₀, z₁, z₂)` and `P(y) = y₀⁴ + y₁⁴ + y₂⁴ + 1`.
* `F = {abs(t₁) ≤ abs(t₀) ≤ abs(z₂) ≤ 1, arg t₀ ∈ [0, π/4], arg t₁ ∈ [0, π/2)}` with
  `t = (y₀, y₁)`. S₄, the phases and the conjugation map every point of X into F. f is
  invariant, so sup over X = sup over F. On F, `abs(z₂)⁴ ≥ 1/3`.
* The chart forms keep `z₂` only as `z₂^e`, e < 4, with `z₂⁴ = −1 − t₀⁴ − t₁⁴`. Where
  abs(z₂) is small this cancels strongly, so every box must lie in F.

## The box: tangent coordinates

* A box has a center p in F and the coordinates τ ∈ C² of the tangent plane:
  `y(τ) = p + E τ + h(τ) n`, with n the unit normal `conj(∇P(p)) / abs(∇P(p))` and
  `E = dy/dt` at p, so that `τ = t − c` to first order.
* h is a fixed point of `h ↦ h − P(y(h)) / c₁`, `c₁ = ∇P(p)·n`. A contraction on the
  ball `abs(h − P_h) ≤ ρ` gives the strict remainder ρ.
* `σ = conj(τ)` is the second pair of independent variables. A Taylor model is a
  polynomial of total degree ≤ N in (τ, σ) and a remainder on the polydisc
  `abs(τ_i), abs(σ_i) ≤ δ`.
* Three nested polydiscs: Φ on `δ_Φ`, f on `δ_f`, the box δ, with
  `δ_f = δ (1 + a)`, `δ_Φ = δ_f (1 + a)`, a = 2/N.

## The computation on one box

1. **Models in τ alone.** h, y(τ), the monomials `(t(τ) − c)^α y₂(τ)^e` and `∂P/∂h`, to the
   order `N_y = N + 4` or more. These are cheap.
2. **Local forms.** For each reference center c_r of a coarse grid, the chart forms
   Q_e of the base and of each N_L move to c_r: `Q^(r) = T_r Q T_rᴴ`. The global chart
   polynomials cancel by up to 2e5 at ρ ≈ 0.9; the local ones by less than 1.3. So
   this step needs about 150 bits (Ozaki products with 8 slices), and every later
   step needs only dd.
3. **Composition.** `W = Σ_e Y_e Q_e^(r) Y_eᴴ` gives the bidegree coefficients of a form up
   to (N_y, N_y). The model in (τ, σ) takes the terms of total degree ≤ N; the tail
   comes from the exact entries of W. This is done for p̄₀ and for each N_L.
4. **The base.** `r = abs(y)² = 1 + Σ y_k ȳ_k` (tensors of the τ models),
   `p̃ = p̄₀ r^(−k₀)`, `log p̄₀ = k₀ log r + log p̃`, from binomial and log series in
   `r / r(c) − 1` and `p̃ / p̃(c) − 1`. (The quotient `D = p̄₀ / (A Ā)` with
   `A = p̄₀(τ, σ = 0)` was measured too: A varies by 36 percent over a box of
   δ = 0.03, so its series in τ converge slowly, and it gains nothing.)
5. **The level sum in the affine chart.** `x = p̄₀^(−1/k₀) = r^(−1) p̃^(−1/k₀)`,
   `ξ = x / x_c − 1`, and `ψ = Σ_k F_k ξ^k`, `F_k = Σ_L C(L, k) x_c^L N_L`. The F_k are
   formed from the composed W of each level, so the cancellation of the levels
   happens in the coefficients and not in the remainders. No factor abs(y)^(±2 top)
   enters.
6. **The metric.** `Φ = log p̄₀ + ψ`. g is the exact derivative of the polynomial part of
   Φ; the remainder of g comes from a Cauchy estimate on `δ_Φ`. det g,
   `v = det(g) abs(∂P/∂h)² / abs(det U)²` with `U = [E, n]`, and `f = v / v̄ − 1` follow
   on `δ_f`.
7. **The bounds.** `sup abs(f) ≤ abs(f(c)) + B(P_f − f(c)) + R_f` on δ. grad f comes
   from the derivative of P_f and a Cauchy estimate from `δ_f` to δ.

## Rounding on the GPU (approach A)

* Midpoints are complex dd. Each dd operation has a relative error of at most `8u²`,
  `u = 2⁻⁵³` (accurate `two_sum` addition, the FMA product).
* The products of Taylor models run in a CUDA kernel (CuPy `RawModule`), with
  `__dadd_rn`, `__fma_rn` and `--fmad=false`, so no compiler can merge or reorder the
  error terms. The probe of 2026-10-09: two_sum and two_prod exact on 20,000 cases,
  the kernel equal to `quartic.tm_gpu.mul` to 2e-32, 9e10 pairs per second.
* A product of Taylor models with at most w terms in each output coefficient adds
  `γ_w B(A) B(B)` to the remainder, `γ_w = (w + 20) · 8u²`. Each sum, matrix product
  and shift gets the same treatment.
* Bounds, tails and Lagrange terms are FP64 sums and products of non-negative numbers,
  each multiplied by `(1 + 1e-9)`. Each subtraction such as `1 − η` takes a lower bound
  of the first term and an upper bound of the second.

## The cover

* Cells of a Cartesian grid in the 4 real coordinates of t, each inside the polydisc
  of its box (with `E = dy/dt`, a cell of side √2 δ fits up to O(δ²)). A cell is
  dropped only when arb proves that it does not meet F.
* A box that misses the target remainder, or has `η ≥ 1` in a series, splits into 16
  cells of half the size.
* The run stops with a failure if a box falls below a minimum size.

## The checks

1. **The arithmetic.** Each GPU operation against arb on random Taylor models, with
   cases that cancel strongly. The arb value must lie inside the GPU enclosure.
2. **The forms.** The chart polynomials against direct arb values of the forms at
   points of the box (`tm_ref.py`: 1e-33 to 1e-46 relative).
3. **The boxes.** `tm_ref.py` (arb, CPU) recomputes a random sample of the boxes. The
   GPU coefficients must agree within their radii.
4. **The points.** f from `tm_box.py` (chart, chain rule, arb) at random points of
   random boxes must lie inside the box bound.
5. **The cover.** An independent script maps random points of X into F and finds a box
   that contains each one. It also recomputes the test of each dropped cell.
6. **The record.** The inputs are fixed by their hashes, the box list repeats, and
   each box record is saved.

## Measurements that set the design (r20, order 16, `scripts/tm_ref.py`)

| change | ρ = 0.9, δ = 0.01 | ρ = 0.4, δ = 0.01 |
|---|---|---|
| chart coordinates, chain rule (`tm_box.py`) | 1.1e-7 | — |
| tangent coordinates, M_L = N_L abs(y)^(2(20−L)) | 4.5e-8 | 2.4e-11 |
| the same, models in τ to order N + 10 | 3.4e-8 | 2.4e-11 |
| affine sum of N_L x^L | 2.1e-10 | 2.3e-13 |
| the same at order 20 | 1.9e-16 | 6.2e-20 |

* The tangent coordinates raise the radius of convergence at ρ ≈ 0.9 from 0.13 to
  0.28; with the affine sum it is 0.38 at ρ = 0.9 and 0.53 at ρ = 0.4.
* `abs(y)^(±40)` vary by 30 percent over a box and cancel in their product, and the
  tail of a product adds absolute values. The affine sum removes them.
* The local forms (`--shift`) reduce the estimated dd rounding in g from 1e-13 to
  1e-19.

## The stages of the build

1. `quartic/tm_gpu.py`: Taylor-model arithmetic in complex dd. Check 1. Done; the
   product moves to the CUDA kernel.
2. `quartic/tm_forms.py`: the chart forms in arb. Check 2. Done.
3. `quartic/tm_cert.py`: the computation of one batch of boxes as in "The computation
   on one box". Check 3.
4. `scripts/cert_run.py` and `scripts/cert_check.py`: the cover, the adaptive run and
   the independent checks. Checks 4, 5 and 6.
5. The run at D = 20, and the report.
