# A strict bound of sup abs(f) with Taylor models on the GPU

Date: 2026-10-09. Status: approved in the chat (approach A); built in stages.

## Purpose

The draft (`notes/draft-main.tex`) needs `‖f‖_Y = ‖f‖'_{C^{0,α}} + a R^(−2−n/q) ‖f‖_{L²}`.
Its work plan asks to "rigorously bound the residuals (for now just C⁰)". One strict
bound of sup abs(f) gives both terms, because `‖f‖_{L²} ≤ vol^(1/2) sup abs(f)`. The
C^{0,α} term also needs `sup abs(grad f) ≲ 1e-6 / R`, which is a coarse bound.

**The output** is two numbers with a record of every box:

* `B_f ≥ sup over X of abs(f)`,
* `B_g ≥ sup over X of abs(grad f)_G`, with G the metric of volume 1.

**The first fit** is D = 20 (`r20`, fitted again on the invariant base). If its
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

* The chart is `z₃ = 1`, `m = 2`, free coordinates `t = (z₀, z₁)`, and
  `z₂⁴ = u_m = −(1 + t₀⁴ + t₁⁴)`.
* `F = {abs(t₁) ≤ abs(t₀) ≤ abs(z₂) ≤ 1, arg t₀ ∈ [0, π/4], arg t₁ ∈ [0, π/2)}`. S₄,
  the phases and the conjugation map every point of X into F. f is invariant, so
  sup over X = sup over F. On F, `abs(z₂)⁴ ≥ 1/3`, so the chart is valid.
* One branch of z₂ is enough, because the phase of z₂ is a symmetry.

## The box

* A box is the polydisc `abs(t_i − c_i) ≤ δ`, `abs(w_i − c̄_i) ≤ δ`, with `w = t̄`
  taken as independent variables. A Taylor model is a polynomial of total degree ≤ N
  in `(s, r) = (t − c, w − c̄)` with complex dd coefficients, and a remainder bound
  on the polydisc.
* Three nested polydiscs: Φ on `δ_Φ`, f on `δ_f`, the certified box `δ`, with
  `δ_f = δ (1 + 1/N)` and `δ_Φ = δ_f (1 + 1/N)`.

## The computation on one box

1. **The forms.** Each form is `Σ_{e=0..3} z₂(t)^e w₂(w)^e Q_e(t, w)` with exact
   polynomials Q_e. Their global coefficients in the chart are formed once on the CPU
   in arb, for the base and for `M_L = N_L abs(z)^(2(20−L))`, and stored as dd with a
   radius. On the GPU, each box shifts them to its center: `T_c Q T_c̄ᵀ`, with dd
   matrix products.
2. **The ξ expansion.** `ψ = abs(z)^(−40) Σ_k F_k ξ^k`, with `x = abs(z)² p̄₀^(−1/k₀)`,
   `ξ = x / x_c − 1`, and `F_k = Σ_L C(L, k) x_c^L M_L` formed per box from the shifted
   M_L. This is an exact identity. The cancellation between the levels of r20 (terms
   of 1e6 that sum to a variation of 1e-8) then happens in dd coefficients, and not
   in the remainders.
3. **The powers.** `abs(z)² = 1 + t₀w₀ + t₁w₁ + z₂w₂`, `p̃ = p̄₀ abs(z)^(−2k₀)`, and the
   binomial and logarithm series act on abs(z)² and p̃, which vary slowly.
   `log p̄₀ = k₀ log abs(z)² + log p̃`.
4. **The metric.** g is the exact derivative of the polynomial part of Φ. The
   remainder of g comes from a Cauchy estimate on `δ_Φ`. det g, v and
   `f = v / v̄ − 1` follow on `δ_f`.
5. **The bounds.** `sup abs(f) ≤ abs(f(c)) + B(P_f − f(c)) + R_f` on δ. grad f comes
   from the derivative of P_f and a Cauchy estimate from `δ_f` to δ, and
   `abs(grad f)_G` from a lower bound of the eigenvalues of g on the box.

## Rounding on the GPU (approach A)

* Midpoints are complex dd. Each dd operation has a relative error of at most
  `8u²`, with `u = 2⁻⁵³` (accurate `two_sum` addition, the exact `dd.mul`).
* A product of Taylor models with at most n terms in each output coefficient adds
  `γ_n B(A) B(B)` to the remainder, with `γ_n = (n + 10) · 8u²`, because the error
  of each coefficient is at most `γ_n Σ abs(a_i) abs(b_j)`. Each sum, matrix product
  and shift gets the same treatment.
* Bounds, tails and Lagrange terms are FP64 sums and products of non-negative numbers.
  Each result is multiplied by `(1 + 1e-10)`, which is strict for fewer than 1e5
  operations. Each subtraction such as `1 − η` takes a lower bound of the first
  term and an upper bound of the second.
* Code that holds a dd error-free transformation runs only under `jax.jit` tests.

## The cover

* Cells of a Cartesian grid in the 4 real coordinates of t, each inside the polydisc
  of its box. A cell is dropped only when arb proves that it does not meet F.
* A box that misses the target remainder, or has `η ≥ 1` in a series, splits into 16
  cells of half the size.
* The run stops with a failure if a box falls below a minimum size.

## The checks

1. **The arithmetic.** Each GPU operation against arb on random Taylor models, with
   cases that cancel strongly. The arb value must lie inside the GPU enclosure.
2. **The forms.** The global chart polynomials against direct arb values of the forms
   at random points of F.
3. **The boxes.** `tm_box.py` (arb, CPU) recomputes a random sample of the boxes. The
   GPU coefficients must agree within their radii.
4. **The points.** f in arb at random points of random boxes must lie inside the box
   bound.
5. **The cover.** An independent script maps random points of X into F and finds a box
   that contains each one. It also recomputes the test of each dropped cell.
6. **The record.** The inputs are fixed by their hashes, the box list repeats, and
   each box record is saved.

## The stages of the build

1. `quartic/tm_gpu.py`: Taylor-model arithmetic in complex dd on the GPU, with the
   remainder and the rounding bound. Check 1.
2. `quartic/tm_forms.py`: the global chart forms in arb. Check 2.
3. `quartic/tm_cert.py`: the computation of one batch of boxes. Check 3.
4. `scripts/cert_run.py` and `scripts/cert_check.py`: the cover, the adaptive run and
   the independent checks. Checks 4, 5 and 6.
5. The run at D = 20, and the report.
