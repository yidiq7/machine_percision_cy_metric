# Machine-precision Calabi-Yau metric on the Fermat quartic

## Language rule

Write all text in ASD-STE100 Simplified Technical English (STE). This rule
applies to your replies in the chat and to this file. It also applies to all
other documents, to code comments, and to commit messages.

Keep to these STE rules:

* Use only these verb forms:
  * the infinitive
  * the imperative
  * the simple present tense
  * the simple past tense
  * the simple future tense
  * the past participle, as an adjective only
* Do not use a word that ends in `-ing`. Do not use a gerund.
* Use the active voice. Use the passive voice only if the active voice is not
  possible.
* Write short sentences. A descriptive sentence must have not more than 25
  words. An instruction must have not more than 20 words.
* Give one instruction in each sentence.
* Write not more than 6 sentences in each paragraph.
* Keep to one topic in each paragraph.
* Keep the articles `a`, `an`, and `the`. Do not remove them.
* Do not put more than 3 nouns together in a noun cluster.
* Use a vertical list for complex material.
* Use the same word for the same thing each time.
* Do not use idioms, slang, or metaphor. Write the direct meaning.
* Use `must` for a requirement, `can` for a possibility, and `will` for the
  future. Do not use `should` or `may`.
* Define each abbreviation at its first use.
* Do not use the oblique stroke `/`. Units, file paths, and code are exceptions.

The STE dictionary has approximately 900 approved words. Use an approved word if
one exists. Technical Names and Technical Verbs from mathematics and from
software are correct STE. Examples are `metric`, `monomial`, `block diagonal`,
`Jacobian`, and `matrix-vector product`. Write mathematics and code as usual.

## Abbreviations

| Abbreviation | Full term |
|---|---|
| FP64 | 64-bit binary floating-point format |
| GN | Gauss-Newton |
| GPU | graphics processing unit |
| HN | Headrick-Nassar |
| LM | Levenberg-Marquardt |
| MAC | multiply-accumulate operation |
| MAPE | mean absolute percentage error |
| MSPE | mean squared percentage error |
| STE | ASD-STE100 Simplified Technical English |
| dd | double-double, two FP64 values that emulate 128-bit precision |

---

## Goal

Calculate the Ricci-flat metric on the **Fermat quartic** `Σ_a z_a^4 = 0` in
CP^3. Use the algebraic-metric ansatz of HN
([arXiv:0908.2635](https://arxiv.org/abs/0908.2635)). Write the code in JAX.

**This is an independent task. The single goal is to make the residual small.**
You do not need the theory that consumes the result.

```
η = μ_ω / μ̂        f = 1 − 1/η        therefore |f| ≈ |η−1|
```

**Stage 1 is the whole of the present work. Decrease `mean abs(f)` to
approximately 1e-15 in FP64.** That value is the FP64 limit.

Stage 2 will decrease `mean abs(f)` to approximately 2e-26 with dd. Do not write
dd code now. Keep to two rules only: keep the arithmetic swappable, and do not
report a number that stage 2 must retract. The section "Output" gives the rules.

Stage 1 is small. The target is k = 32 to 38, with `n_θ ≤ 3.2e3`. One GPU is
enough. The laptop can run k ≤ 20 with 1e5 points.

### The loss

**Minimize `weighted_MSPE` with the LM method. Report `weighted_MAPE`.** Both
functions are in `MLGeometry/src/MLGeometry/loss.py`:

```
MSPE = Σ_i w_i (η_i/⟨η⟩ − 1)²        MAPE = Σ_i w_i |1 − η_i/⟨η⟩|
w_i = mass_i / Σ mass
```

Each function is an energy functional `E_F` of HN. Each one is convex and has
its minimum at η = 1. Therefore each one is a correct member of their family.

We measured the choice on a test problem with the shape of `v`, which is a ratio
of two linear forms with an exact zero minimum. The table gives `rms abs(f)`
after convergence, at 60 parameters and 5,000 points.

| `cond(J)` | LM on MSPE | L-BFGS on MSPE | L-BFGS on MAPE |
|---|---|---|---|
| 1.0e2 | 7.9e-16 | 1.1e-15 | 5.3e-14 |
| 1.0e5 | 6.0e-16 | 4.5e-07 | 4.8e-07 |
| 1.0e8 | 5.1e-16 | 2.3e-07 | 2.8e-07 |
| 1.0e11 | 4.3e-16 | 1.5e-07 | 2.0e-07 |

MAPE is the worst column at every condition number. The corner points of the L1
function break the secant condition of a quasi-Newton method.

The earlier rule "use MAPE and not MSPE" was correct for a first-order method.
The L1 subgradient `sign(η−1)·∂η` keeps its magnitude when η goes to 1, and an
L2 gradient of approximately 1e-15 hides under round-off. **That argument does
not hold for LM.** LM solves `(JᵀJ + λI)δ = −Jᵀr`. The residual scale sits on
the right side and inside `J` on the left side, so `δ` stays proportional to the
distance to the minimum at `r ≈ 1e-15`.

A small MSPE value is safe. At `rms abs(f) = 5.1e-16` the loss value is 2.6e-31,
and numpy FP64, `math.fsum`, and mpmath at 50 digits agree to all 17 digits.
Machine precision is relative, and the smallest normal FP64 value is 2.2e-308.
`weighted_MSPE` at `loss.py:36` forms `η − 1` first and then squares it, so it
adds no cancellation. A form such as `⟨η²⟩ − ⟨η⟩²` would cancel. Do not use one.

---

## Status

* **Complete:** FP64 support in MLGeometry. We pushed this code.
* **Complete:** the symmetry structure, for the quartic and for the quintic.
* **Complete:** task 3, the exact count of `n_θ` for the quartic. See "Parameter
  counts and cost".
* **Complete:** the readiness measurements of 2026-08-12. The sampler, the
  restriction matrix, and `ΩΩ̄` each hold 1e-15. See "The metric evaluation".
* **Complete:** fits to k = 20 on the quartic, in FP64, with a first-order
  method. See "Convergence rate". **No fit is saved.** `draft/` is not in this
  directory, and no checkpoint file is here.
* **Complete:** the model, the solver and the analytic Jacobian. The fits at
  k = 2 to 10 reproduce the earlier sigma values to 4 percent.
* **Complete:** task 1. The optimizer stopped early, and the rate does not fall.
  See "Convergence rate".
* **Not started:** task 2 needs `R = ‖∇f‖_{L²}/‖f‖_{L²}`; the other two ratios
  are in the table. The compensated accumulator is not written.

---

## First tasks

Task 3 is complete. Tasks 1 and 2 need the model first, because no fit is saved
and no model above k = 4 exists. So the order is:

0. **Build the model and the solver.** The model is the section evaluator at
   degree k, with the 64 residue blocks and the S₄-invariant parametrization of
   `h`. The solver is `optimistix.LevenbergMarquardt` on MSPE.
1. **Test the convergence rate.** Complete. The rate holds at 2.73 to 2.90
   against the HN prediction of 2.83, so stage 1 needs k = 31 and not 38. See
   "Convergence rate".
2. **Measure three ratios** on the fits at k = 12, 16, and 20:
   * `rms(f) / mean abs(f)`.
   * `max abs(f) / rms(f)` on the sample.
   * `R = ‖∇f‖_{L²} / ‖f‖_{L²}`. Nobody measured this one. It sets the target
     for stage 2, and it decides the cost of the later rigorous work.
3. **Calculate `n_θ` exactly for the quartic.** Complete.
   `notes/symmetry_param_counts_s4.py` holds the calculation, and "Parameter
   counts and cost" holds the result.

Use forward-mode autodiff for `∇f`. X is a graph over 2 local complex
coordinates `t`, so the pass needs 4 real tangent directions. The restriction
matrix is the Jacobian of that graph, and it is enough for `∇f`, for `φ_{ij̄}`,
and for the Ricci form.

For the LM Jacobian `∂r/∂θ` use `jax.jacfwd` first. Replace it with the analytic
Jacobian only if the step time becomes a problem. `p` is linear in θ, so
`∂p/∂θ_α` is the feature itself, and Jacobi's formula gives
`∂v/∂θ_α = v·tr(g|_X^{-1} ∂g|_X/∂θ_α)`.

---

## Symmetry structure

The symmetry group is Γ = (Z₄³ ⋊ S₄) × Z₂, and the Z₂ factor is complex
conjugation. The holomorphic part has order **1536**. The diagonal phase acts
trivially on CP^3, so the phase group is Z₄³ and not Z₄⁴.

**Phase invariance makes `h_AB = 0`, unless A ≡ B (mod 4).** Therefore `h` is
block diagonal in the residue `r = A mod 4`. There are 64 blocks, and they have
`|r| ≡ k (mod 4)`. In one block, `A = r + 4C`. Put **`u_a ≡ z_a^4`**. Then

```
z^A = z^r · u^C        p = Σ_r |z^r|² · q_r(u, ū)
```

and the defining equation becomes exactly **Σ_a u_a = 0**. Therefore each block
has one linear relation, and that relation is the complete effect of the fact
that the monomials are not independent sections.

* The block section space is `Sec_m ≅ Sym^m(std)`. Here `std` is the
  3-dimensional standard representation of S₄, which is the hyperplane Σu = 0.
  Its dimension is `C(m+2,2)`, and `m_r = (k−|r|)/4`.
* `Σ_r C(m_r+2,2) = 2k²+2` exactly. We verified this at k = 4, 8, 12, 32, and
  38.
* S₄ makes a permutation of the blocks. Therefore there is one parameter set for
  each S₄ orbit of `r`. The stabilizer `Stab(r)` constrains the set.
* The Z₂ conjugation makes `h` **real symmetric**.
* `n_θ = Σ dim Sym²(Sec_{m_r})^{Stab(r)}`, and the sum is over the S₄ orbits of
  `r`. Calculate it exactly with character theory.

### Parameter counts and cost

**Every number below is exact.** `notes/symmetry_param_counts_s4.py` calculates
them with character theory, and it agrees with our own fits at k = 4, 6, 8, 10,
and 12.

| k | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 36 | 38 |
|---|---|---|----|----|----|----|----|----|----|----|----|
| sections | 34 | 74 | 130 | 202 | 290 | 514 | 802 | 1,154 | 2,050 | 2,594 | 2,890 |
| n_θ | 5 | 10 | 19 | 33 | 56 | 135 | 285 | 542 | 1,559 | 2,432 | 2,987 |
| MACs | 40 | 134 | 346 | 766 | 1,496 | 4,450 | 10,552 | 21,530 | 66,946 | 106,760 | 132,310 |

The old estimates at k ≥ 20 were high: 300 against 285, 1.7e3 against 1,559, and
3.2e3 against 2,987. The exact value sits 6 to 20 percent above the leading term
`Σ_r d_r(d_r+1)/2 / 24`. The check `Σ_r C(m_r+2,2) = 2k²+2` holds at every k in
the table.

The MACs row is `Σ_r d_r²` for one matrix-vector product. Each point needs 6
columns, which are `p`, `∂p`, and `∂∂̄p`. **The code must calculate the features
at each step. Do not store them.** All 64 blocks use the same `u^C` vectors, and
only the scalar factor `|z^r|²` is different.

Put the blocks into groups by `m`. The measured pattern is:

* k ≡ 0 (mod 4) gives 4 values of `m`, with block counts {1, 31, 31, 1}. At
  k = 32 that is `m` = 5, 6, 7, 8 with `d` = 21, 28, 36, 45.
* k ≡ 2 (mod 4) gives 3 values of `m`, with block counts {10, 44, 10}. At k = 38
  that is `m` = 7, 8, 9 with `d` = 36, 45, 55.

Then each group needs one batch matrix-vector product.

**The intermediates of the LM Jacobian are the memory limit, and not `J`
itself.** `J` has shape `(N_pts, n_θ)`, which is only 228 MB at k = 20 with
`N_pts = 1e5`. But forward-mode autodiff nests three things: the points, the 8
real directions inside `complex_hessian`, and the `n_θ` tangents. One fused pass
therefore holds this much for each point:

| k | n_θ | memory for each point | `N_pts = 1e5`, one pass |
|---|---|---|---|
| 12 | 56 | 2.58 MB | 258 GB |
| 16 | 135 | 8.79 MB | 879 GB |
| 20 | 285 | 31.26 MB | 3,126 GB |

A run at k = 12 with `N_pts = 1e5` failed on an 80 GB A100 for this reason.

**The fix is `lax.map` over the points inside the residual.** One block frees its
intermediates before the next block starts, so the peak follows the block and
not `N_pts`. We measured the peak as flat at 0.646 GB for `N_pts` of 2e3, 8e3 and
2e4 at k = 12 with a block of 250. Only `J` grows. The result changes by 1.3e-16,
which is the order of the sum and nothing more.

* `metric.volumes(lay, theta, batch, chunk)` puts `lax.map` in that path.
* `fit.solve_memory` lowers the whole `least_squares` and reads the peak from the
  compiled program. One Jacobian is not the peak, because the same program holds
  `J`, the stacked `[J; sqrt(λ)I]`, its QR and the candidates of the trust
  region.
* `fit.plan_fit` sizes the fit set **and** the block against a budget, from
  `peak = a·N_fit + b·chunk + c`. **It spends the budget on the fit set first
  and the block second**, because the fit set sets the excess risk `n_θ/N_fit`
  and the block only sets the speed. An earlier version shrank only the fit set,
  and with the Gram stage in dd, where the block term dominates, it reported
  failure where a plan existed. It refuses any plan below `4 n_θ` points and
  raises `MemoryError`, because an earlier version answered a small budget with
  a fit set of 1 point and then reported σ = 0.23 as a result.

Do **not** use mini-batches for the fit. LM solves `min ‖Jδ + r‖² + λ‖δ‖²`, and a
mini-batch makes `J` and `r` stochastic. The trust region compares the predicted
reduction against the true one, so it would then accept or reject a step on
noise. `lax.map` gives the exact full-batch step instead.

### Fermat quintic, secondary

The quintic in CP^4 has the same structure with 5 in place of 4: Γ has order
150,000, there are 625 blocks, `Sec_m ≅ Sym^m(std)` with `std` of dimension 4,
and `Σ_r C(m_r+3,3) = (5/6)k(k²+5)`. Exact counts of `n_θ` are 5, 7, 10, and 14
at k = 4, 5, 6, and 7, and 6,484 at k = 32. `notes/symmetry_param_counts.py` has
the calculation.

**A simple ansatz makes `h` diagonal in the monomials, with one coefficient for
each orbit. That ansatz is complete for k ≤ 6, but it is wrong at k = 7.** It
gives 13 real parameters at k = 7 against the correct 14.
`SLagSearch/slag_condition.py` uses the simple ansatz. Do not use it above
k = 6.

---

## The metric evaluation

Use the MLGeometry path for stage 1: the sub-patches by the largest gradient,
and the restriction matrix. That code exists and it is tested. The restriction
matrix is a per-point constant, so `dataset.py` calculates it once. The
optimizer then sees one batched contraction with no branch.

`v = det g|_X / ΩΩ̄` and `η = v / ⟨v⟩`. Equation (6.7) of HN calculates the same
`v` from a bordered determinant of `p`, `∂p`, `∂∂̄p`, and `∂P`, with no
sub-patch. We verified that it is gauge invariant for `p → p + gP + ḡP̄`, and it
holds for any smooth hypersurface. **Use it once as an independent test oracle.
Do not make it a code path in stage 1.**

Two tests protect every number below 1e-10:

1. **The seam test.** In the restriction path the value of η depends on which
   sub-patch caught the point, and only round-off must make a difference. Take
   points where two gradient components are nearly equal. Evaluate η under both
   sub-patch choices. Require agreement to approximately 1e-15.
2. **The oracle test.** Compare η from the restriction path and from equation
   (6.7) at k = 6. This test validates the restriction matrix, `ΩΩ̄`, the
   conventions, and the normalization together.

The tests of HN and of MLGeometry live at 1e-3 to 1e-7. An inconsistency of
1e-12 is invisible there, and it puts a floor under every fit below that level.

**The seam test passes.** We ran it on 2026-08-12 with the Fubini-Study metric at
k = 1, so no fit was necessary. We compared `v` from the two coordinates with the
largest gradient, on 25,138 points.

| `g2/g1` band | count | median rel. difference | max |
|---|---|---|---|
| ≥ 0.999 | 21 | 2.6e-16 | 7.2e-16 |
| 0.99 to 0.999 | 194 | 2.4e-16 | 9.6e-16 |
| all points | 25,138 | 2.3e-16 | 2.3e-15 |

So the restriction matrix and `ΩΩ̄` each hold 1e-15. The symbolic `jac.inv()` in
`get_restriction` is accurate enough for stage 1. Replace it for speed, and not
for accuracy.

### Two traps in the evaluation of φ

Both cost many digits. We measured both.

1. **Use integer exponents for the sections.** A float exponent array sends
   `z ** A` through `exp` and `log`. The relative error of φ at k = 32 was then
   21. Integer exponents give 3.4e-15.
2. **Do not write `φ_{ij̄} = (p_{ij̄}p − p_i p̄_j)/p²`.** The two terms cancel by
   a factor of 21.8 at k = 32. Use `complex_math.complex_hessian`, or the
   centered Gram form: put `t_i = ∂_i s − (⟨∂_i s, s⟩/p)·s`, and then
   `φ_{ij̄} = ⟨t_i,t_j⟩/p`.
3. **Keep the multinomial coefficients in Python integers.** `numpy` wraps int64
   without a warning. For `A = (20, 4, 0, 0)` the coefficient `k!/∏A_a!` is
   5.839e19, and `np.prod` returns 3.049e18, which is 19 times too small. Every
   `A_a!` fits in int64, so only the product overflows. The relative error of `p`
   at `θ_FS` was then 1.5e-5 at k = 24, 0.199 at k = 26 and 95 at k = 28, and at
   k = 28 `min p` went to −2.5e14 before the fit took one step. `math.prod` with
   an exact integer division fixes it, and `Σ_a A_a = k` makes the division
   exact. **Test the model at the largest k you intend to run.** Our test stopped
   at k = 20 and so it missed this for a long time.

Relative error of φ against an exact reference, worst of 8 random points:

| k | 1 | 12 | 20 | 32 | 38 |
|---|---|----|----|----|----|
| difference form | 2.4e-16 | 5.3e-15 | 2.6e-14 | 1.4e-13 | 2.8e-13 |
| centered Gram | 3.0e-16 | 2.1e-15 | 3.7e-15 | 6.3e-15 | 1.3e-14 |
| `complex_hessian` | 5.9e-16 | 2.3e-15 | 7.5e-15 | 3.4e-15 | 8.4e-15 |

---

## Convergence rate

`σ = mean abs(η−1)` from our own fits on the quartic. The values to k = 12 come
from `draft/atlas_bounds.tex`. The values at k = 16 and k = 20 come from
`draft/quartic_inputs.tex`, and from a different run.

| k | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|----|----|----|----|----|
| σ | 1.90e-2 | 1.68e-3 | 1.60e-4 | 1.69e-5 | 1.85e-6 | 2.96e-7 | 8.5e-9 | 6.3e-10 |
| rate | — | 3.37 | 3.24 | 3.08 | 3.02 | 2.50 | 2.43 | 1.92 |

### Task 1 is done. The optimizer stopped early

The LM fits of 2026-08-16, on 1e5 fit points with an 8e6 hold-out set, give:

| k | 12 | 16 | 20 |
|---|----|----|----|
| σ, LM | **2.2018e-07** | **3.5096e-09** | **6.3276e-11** |
| σ, first-order | 2.96e-07 | 8.5e-09 | 6.3e-10 |
| ratio | 0.74 | 0.41 | **0.10** |
| standard error | 0.04 % | 0.04 % | 0.04 % |
| `cond(J)` | 2.24e7 | 2.14e10 | 2.34e13 |
| `rms(f)/mean abs(f)` | 1.695 | 1.905 | 2.096 |
| `max abs(f)/rms(f)` | 29.9 | 41.6 | 54.8 |
| `min λ(φ_{ij̄})` | 1.498 | 1.998 | 2.497 |

**The rate no longer falls.** With the same definition as the table above:

| span | 10 → 12 | 12 → 16 | 16 → 20 | 20 → 24 | 24 → 28 |
|---|---|---|---|---|---|
| first-order | 2.50 | 2.43 | 1.92 | — | — |
| LM | **2.90** | **2.81** | **2.73** | **2.67** | **2.06** |

The 24 → 28 entry is provisional. Its two ends come from different code:
k = 24 predates the canonical basis and the global scale of θ, and k = 28
carries both. See "k = 28 after the canonical basis and the global scale".

### The fits with the Gram stage in dd, 2026-08-24

| k | σ | fit points | excess risk | `cond(J)` | dd floor | σ over floor |
|---|---|---|---|---|---|---|
| 24 | 1.2529e-12 | 69,076 | 7.8e-3 | 5.77e16 | 4.33e-15 | 289 |
| 28 | **2.0566e-13** | 37,346 | 2.5e-2 | 8.73e18 | 1.61e-14 | **12.8** |

* **k = 24 is confirmed.** dd gives 1.2529e-12 against the 1.2566e-12 of FP64,
  0.29 percent apart, which is right because σ there sits 21 times above the FP64
  floor. Two arithmetics agree, so the number is solid.
* **k = 28 is now a measurement.** σ sits 12.8 times above the dd floor. In FP64
  it read 1.7077e-13 against a floor of 3.97e-13, so that number was not one.
* **The rate for 24 → 28 is 1.57**, and σ at k = 28 is 8.3 times above the
  2.47e-14 that a rate of 2.67 predicts.

**Do not yet call this a slowdown of the ansatz.** Three things can hold σ at
2.06e-13, and only the first is settled:

1. The floor of the residual. Ruled out at `θ_FS`, where it is 1.61e-14. **It is
   not ruled out at the θ of the fit.** `scripts/oracle_precision.py --theta`
   measures it there, and that is the first test to run.
2. The accuracy of `J`. The rank scan showed that the σ reached is approximately
   the relative cutoff, and the cutoff that FP64 imposes is the relative error of
   `J` itself. `J` uses the dd Gram stage, and the rest of it, the matrices `W`
   and the contraction with `K`, is FP64. `cond(J) = 8.7e18` is 1,900 times past
   `1/ε`, against 13 times at k = 24.
3. The excess risk, which is 2.5e-2 at k = 28 and the largest of the series. It
   predicts only 1.25 percent, so it cannot carry the whole factor of 8.3, but a
   run at `BUDGET=70` gives approximately 73,000 fit points and settles it.

None of the three was the answer. The floor at the θ of the fit is 1.899e-15,
which is 108 times below σ, so explanation 1 is out. The next section holds the
cause, which is a fourth thing: the scale of θ. Their own estimate of a converged
fit at k = 20 was 4.5e-11 against the 6.33e-11 we measured.

The k = 20 fit hit the 200-step limit, so its σ is an upper bound. A lower value
would raise the 16 → 20 rate, and the small decline that is left can be that.

**Stage 1 needs k ≈ 31, and not 38.** At a rate of 2.73 to 2.83, σ = 1e-15 comes
at k = 30.6 to 31.0, and k = 32 gives 2.4e-16 to 3.7e-16.

### k = 28 after the canonical basis and the global scale, 2026-08-30

Three runs at k = 28, as each fix landed. The seed is 20260813 in each one.

| fit points | basis | scale of θ | `cond(J)` | σ |
|---|---|---|---|---|
| 37,346 | SVD | for each component | 8.73e18 | 2.0566e-13 |
| 90,134 | Gram-Schmidt | for each component | 8.14e30 | 8.6215e-14 |
| 88,993 | Gram-Schmidt | **global** | **7.72e17** | **6.8949e-14** |

A second seed at 91,113 points gave 7.4086e-14, which is 15 percent from the
8.6215e-14 of the first. The three runs before those spanned 70 percent, so the
small fit sets carried that spread and not the arithmetic.

* **The scale of θ was the cause of `cond(J) = 8.1e30`, and one global scale
  removes it.** 13 orders of magnitude, and σ fell by 20 percent with it.
* σ sits 36 times above the dd floor of 1.899e-15 at the fitted θ. **k = 28 is a
  measurement**, and it is not floor limited.
* `min λ(φ) = 3.496` and `min p = 6.62e9`, so the margin above zero is large.
* The fit converged in 123 steps and 3,189 s. It did not hit the step limit.

**The rate for 24 → 28 is 2.06**, against 1.57 before. That number is not yet
solid, because k = 24 is still the 1.2496e-12 of the old convention. **Both ends
of a rate must come from the same code.** Refit k = 24 first.

The planner left the block at 64, because dd needs 1.115 MB for each fit point at
k = 28. A target risk of 5e-3 asks for 189,800 points, which is 212 GB, so the
budget went to the points and the block got its floor. `RISK` in
`scripts/run_cluster.sh` sets that trade-off. Excess risk of 1e-2 moves σ by
approximately 0.5 percent, which is far below the 15 percent that two seeds give,
so a larger block and fewer points is the faster plan.

### The binding constraint is the arithmetic, and FP64 stops at k = 27

`scripts/oracle_precision.py` measures the FP64 error of the evaluation against
mpmath at 50 digits. The block matrices `H_r` go in as exact input, so the test
covers what the per-point path does: the monomials of `u`, the frame of the
sections, the Gram sums, and the centered form of φ. The relative error of `v`
is the floor under `mean abs(f)`.

| k | 12 | 20 | 24 | 28 | 32 |
|---|----|----|----|----|----|
| error of φ | 6.2e-16 | 3.2e-15 | 2.4e-14 | 9.9e-14 | 1.6e-12 |
| **floor on `abs(f)`** | <1e-16 | **4.4e-15** | **6.0e-14** | **4.0e-13** | **5.1e-12** |
| σ measured | 2.2018e-07 | 6.3276e-11 | 1.2566e-12 | 1.7077e-13 | — |
| σ over the floor | huge | 14,391 | **20.8** | **0.43** | — |

The floor grows by 1.80 for each unit of k, and σ falls by 2.66, so the ratio
falls by 2.64 for each unit of k. **It reaches 1 at k = 27.1, with the floor at
2.6e-13.**

Three conclusions follow:

* **σ at k = 20 and k = 24 is real.** It sits 14,391 and 20.8 times above the
  floor, so the rates 2.81, 2.73 and 2.67 stand.
* **σ at k = 28 is not a measurement.** 1.7077e-13 is below the floor of
  3.969e-13, so the optimizer drove its own computed residual under the error of
  the arithmetic. The apparent fall of the rate to 1.65 across 20 → 28 is that
  artifact. **The ansatz does not slow down.**
* **Plain FP64 cannot reach 1e-15.** It stops near σ = 4e-13 at k = 27, which is
  400 times short. At the rate of 2.67, σ = 1e-15 comes at k = 31.3, and the
  floor there is above 1e-11.

**Therefore the compensated accumulator is the whole of the remaining work for
stage 1, and it is needed from k = 24 and not from k = 28.** An earlier estimate
of the floor gave 5e-15 at k = 32. That measurement used a diagonal `h` in the
plain monomial basis, where every term of the sum is positive. The real model
carries the frame and the 64 blocks, and it is 1,000 times worse. Do not
estimate this floor again; measure it with the oracle.

### Where the error is born

`scripts/oracle_stages.py` runs any subset of the 5 stages of the evaluation in
mpmath and the rest in FP64. The stages are:

1. `u = z^4` and the monomials `u^C` of each group,
2. `psi = Qᵀ psi_full`, the frame of the sections,
3. `σ = z^r psi` and its 4 holomorphic derivatives,
4. `H σ`, `H ∂σ`, and the Gram sums `p`, `∂p`, `∂∂̄p`,
5. the centered form of φ.

Relative error of φ, worst of 6 points:

| k | all FP64 | dd on 4 | dd on 5 | dd on 1-3 | dd on 4+5 | dd on 1-4 |
|---|---|---|---|---|---|---|
| 24 | 1.94e-14 | 3.67e-15 | 1.96e-14 | 1.81e-14 | 3.36e-15 | 4.4e-16 |
| 28 | 6.88e-14 | 3.39e-15 | 6.89e-14 | 2.12e-13 | 3.46e-15 | 2.7e-16 |
| 32 | 4.44e-13 | **1.06e-14** | 4.43e-13 | 5.10e-13 | 1.06e-14 | **5.1e-16** |

* **Stage 4 carries almost all of it.** dd there alone gives 42 times at k = 32,
  and it takes the floor on `abs(f)` from 5.1e-12 to approximately 3.4e-14.
* **Stage 5 gives nothing on its own**, and stages 1 to 3 give nothing on their
  own. They matter only after stage 4 is fixed.
* **dd on stages 1 to 4 puts the error of φ at 5e-16, and flat in k.** The floor
  on `abs(f)` is then approximately 1.6e-15, at every k.

So the fix that "Precision budget" proposes, a two-term accumulator on `p`, `∂p`
and `∂∂̄p`, **is aimed correctly**, and stage 4 is exactly those sums together
with the matrix-vector products that make them. But stage 4 alone stops at
3.4e-14. **To reach 1e-15 the accumulator must cover stages 1 to 4, and stage 5
as well for a margin.**

### Stage 4 is done. `analytic.GRAM_DD` selects it

`quartic/dd.py` holds the arithmetic and `analytic._gram_parts` uses it. Measured
against the oracle, and the gain matches what the staged test predicted:

| k | φ error, FP64 | φ error, dd | floor on `abs(f)`, FP64 | floor, dd |
|---|---|---|---|---|
| 24 | 2.37e-14 | 8.79e-16 | 6.04e-14 | **4.33e-15** |
| 28 | 9.93e-14 | 3.24e-15 | 3.97e-13 | **1.61e-14** |
| 32 | 1.56e-12 | 1.04e-14 | 5.07e-12 | **3.03e-14** |

**The cost is 41 times at k = 28, and not the 3 to 4 flops for each MAC that
"Precision budget" estimates.** Correct that line. dd gives up the
matrix-vector product: FP64 calls one matmul kernel, and dd must broadcast and
reduce by hand.

| k | Jacobian, FP64 | Jacobian, dd | memory, FP64 | memory, dd |
|---|---|---|---|---|
| 20 | 0.54 s | 30.1 s | 0.12 GB | 2.70 GB |
| 28 | 2.66 s | 109.5 s | 0.60 GB | 8.21 GB |

`GRAM_DD = 'inner'` puts only the inner products in dd. That is enough at k = 20,
where it matches full dd at 1.47e-15 for a sixth of the cost, and it is **not**
enough higher up: at k = 28 it gives 3.03e-14 against 3.24e-15. The
matrix-vector products carry the error exactly where it matters.

**Two traps in the arithmetic, both measured.** See "Notes for stage 2" for the
XLA simplifier. The second is the size of the program: a Python loop over the
groups, the 4 derivative directions and the 4 partial products made the graph 48
times the FP64 one, and XLA took 79 s to compile k = 6 and 339 s to compile
k = 8. One reduction tree shared by all 10 columns brings that to 13 s and 63 s
at k = 6 and k = 20, with the graph 11 to 15 times the FP64 one.

**What stage 4 buys.** The floor still grows with k, by 1.28 for each unit, while
σ falls by 2.67. The ratio σ over the floor is 290 at k = 24 and 1.5 at k = 28,
so it reaches 1 near **k = 28.6**. Stage 4 therefore moves the useful limit from
k = 27.1 to k = 28.6, and σ there is approximately 2e-14. **k = 28 becomes a real
measurement, and 1e-15 still needs stages 1 to 3.** The oracle says those give a
floor of 1.6e-15 that does not grow with k.

A count of cancellation factors does not find this. The sum over the 64 blocks
cancels by only 1.9 at k = 32, and the frame by 133, so that count points at the
frame. The cancellation is inside the matrix-vector product `H σ`, which the
count does not see, because `H` has entries of mixed sign and `σ` spans a large
dynamic range. Trust the oracle and not the count.

`cond(J)` grows by 5.65 for each unit of k. It passes `1/ε = 4.5e15` at
**k = 23**, so `J` is numerically singular in FP64 from k = 24.

### The parametrization must be scaled

The first fits at k = 24 and k = 28 failed. At k = 24 LM took 2 steps and
returned the start point. At k = 28 it left the region where `p` is positive,
with `min p = -3.1e14`. `cond(J) > 1/ε` looked like the cause, and it is not.

**The cause was the scale of θ.** The invariant basis is orthonormal in the
Frobenius norm, so `θ_FS` carries the multinomial coefficients: its components
run from 1.8e8 to 5.1e11 at k = 24, and `σ_max(J)` is only 1.1e-7. LM damps with
`λI` and tests the step in absolute size, and neither is invariant under a
change of scale. A trust-region step of order 1 was therefore a relative change
of 1e-11, and the fit did not move.

`sections.layout` now divides θ by `|θ_FS|`, so the fit starts at ±1 and
`lay.scale` holds the factors. `prepare` applies the scale, so a tangent gets it
too and the chain rule is automatic. The loss after 60 LM steps, on 2,800
points:

| k | parametrization | steps | loss at the start | loss at the end |
|---|---|---|---|---|
| 20 | raw | 60 | 9.08e-02 | 4.15e-12 |
| 20 | **scaled** | 60 | 9.08e-02 | **1.71e-20** |
| 24 | raw | 2 | 9.09e-02 | 9.09e-02 |
| 24 | **scaled** | 60 | 9.09e-02 | **6.01e-24** |

A whole fit at k = 24 then converged in 116 steps, with `cond(J) = 4.71e16`,
which is 10 times past `1/ε`:

| k | σ | fit points | excess risk | `cond(J)` | `min λ(φ)` | `min p` |
|---|---|---|---|---|---|---|
| 20 | 6.4043e-11 | 12,000 | 2.4e-2 | 5.48e13 | 2.541 | 5.93e6 |
| 24 | **1.2818e-12** | 5,981 | 9.1e-2 | 4.71e16 | 3.049 | 1.76e8 |

**So the rate holds past `1/ε`, and the ill-conditioning of `J` does not block
the fit.** The rate for 20 → 24 is 2.65, or 2.68 after a correction for the
excess risk of the small fit set. With 12,000 points the k = 20 fit gave
6.4043e-11 against the 6.3276e-11 of the 100,000-point run, which is the 1.2
percent that the excess risk predicts.

`lx.SVD(rcond)` is available through `fit.fit(rcond=...)`, and it is **not**
necessary. With the scale fixed, plain QR does better: 6.01e-24 against 9.07e-21
at `rcond = 1e-13`. A cutoff throws away the small directions that carry the
last digits, and a scan at k = 16 and k = 20 shows that the σ reached is
approximately `rcond`. Use it only to diagnose.

**Report the scale with `h`.** `lay.scale` is part of the basis convention that
"Output" asks for.

---

## Precision budget

* **The floor is above 1e-16, and it grows with k.** The code calculates η from a
  ratio of order 1, so `η−1` carries the absolute error of `v`. `v` is a 2×2
  determinant of φ, so its relative error is at least 2 times that of φ, and more
  where `g|_X` is ill-conditioned. With the best FP64 form of φ, `η−1` has an
  absolute error near 7e-15 at k = 32 and near 1.7e-14 at k = 38. Therefore plain
  FP64 stops `mean abs(f)` near 5e-15 at k = 32, and near 1.4e-14 at k = 38.
* **A smaller k does not avoid the floor.** At k = 16 the floor is near 5e-15,
  and σ is 8.5e-9. The floor only matters where σ approaches 1e-15, and that is
  k = 32 or higher.
* **The fix is a compensated sum, and it waits.** The sum over 2k² monomials
  accumulates the round-off, so the centered Gram form still gives 6.3e-15 at
  k = 32. A two-term (Kahan or Neumaier) accumulator on `p`, `∂p`, and `∂∂̄p`
  removes it for 3 to 4 extra flops per MAC. **Do not write it now.** Keep the
  three sums inside one small function, so that the swap stays a one-function
  change. That is rule 2 of "Output".
* **Use two independent point sets, because the fit and the report have
  different limits.** Always report on the hold-out set.
  * **The hold-out set.** The standard error of `mean abs(f)` is
    `mean abs(f)/sqrt(N)`. We measured this at k = 6 on 6e5 points, in disjoint
    subsets: the spread was 1.18, 0.45 and 0.20 percent at N = 8e3, 4e4 and 2e5,
    against `1/sqrt(N)` of 1.12, 0.50 and 0.22 percent. The mean did not move.
    **The error is relative and not absolute, so a small N does not stop
    `mean abs(f)` from reaching 1e-15. It only limits the digits you can report.**
    Use `N = 8e6`, which gives 0.035 percent. The cost is one forward pass in
    chunks.
  * **`max abs(f)` needs the points.** At k = 6 it read 2.25e-3 on 8e3 points and
    3.00e-3 on 2e6 points. It is a lower bound on `sup abs(f)`, and it grows with
    N. Never report it from a small sample.
  * **The fit set is bounded by memory.** LM materializes `[J; sqrt(λ)I]`, which
    is `N_fit · n_θ · 8` bytes in FP64, and lineax QR needs approximately 3
    copies.

| `N_fit` | k = 12 | k = 16 | k = 20 | k = 32 | k = 38 |
|---|---|---|---|---|---|
| 1e5 | 0.04 GB | 0.11 GB | 0.23 GB | 1.2 GB | 2.4 GB |
| 5e5 | 0.2 GB | 0.5 GB | 1.1 GB | 6.2 GB | 12 GB |
| 1e7 | 4.5 GB | 11 GB | 23 GB | **125 GB** | **239 GB** |

* A large fit set buys little. The excess risk is `n_θ / N_fit`, which is 2.9e-3
  at k = 20 with `N_fit = 1e5`, so σ moves by 0.15 percent. The fit time grows in
  proportion to `N_fit`. **Use `N_fit = 1e5` to k = 20, and 5e5 at k ≥ 32.**
* The points must satisfy `P(z) = 0` to full working precision.
  `hypersurface.refine_points` already does the Newton refinement, and we
  measured `max abs(P)/scale = 3.3e-16` with a median of 7.0e-17. This is done.
* HN show that all fixed positive measures give a correct energy functional.
  Therefore the `mass` weights need an accuracy of only approximately 1e-8. Only
  the points need full precision.

---

## Decisions

* **The Jacobian is analytic, and `quartic/analytic.py` holds it.** Nested
  autodiff pays two factors that the structure does not need:
  * one residual costs approximately 50 evaluations of `p`, because
    `complex_hessian` differentiates through an 8-dimensional real split, and
  * one `jacfwd` costs 1.5 to 1.9 times `n_θ` residuals.

  Instead, `p`, `∂p` and `∂∂̄p` are Gram sums of the holomorphic section vector
  and its 4 derivatives, and Jacobi's formula plus the contraction with
  `K = R† g⁻¹ R` reduces `∂v/∂θ_α` to one Frobenius product for each block.
  Measured on one A100, Jacobian time only:

| k | 6 | 8 | 12 | 16 | 20 |
|---|---|---|----|----|----|
| autodiff | 0.070 s | 0.192 s | 0.902 s | 3.405 s | 12.158 s |
| analytic | 0.005 s | 0.010 s | 0.021 s | 0.047 s | 0.096 s |
| gain | 14.7x | 19.4x | 42.1x | 72.3x | **126x** |
| memory | 3.11→0.41 GB | 5.62→0.91 GB | 32.1→1.44 GB | 26.2→0.73 GB | 22.8→0.40 GB |

  `analytic.weighted_residual_fast` carries a `jax.custom_jvp`. `lineax`
  materializes the LM operator with `jax.vmap` over the basis of θ, and `vmap`
  only batches an operation whose input carries a batch axis. So `v` and the
  block matrices `W` come out once, outside the batch, and only the Frobenius
  product runs for each direction. **`optimistix` therefore keeps its own trust
  region and convergence tests.** `fit.fit(analytic_jac=False)` gives the old
  path, and the tests hold the two against each other.
* **Optimizer: `optimistix.LevenbergMarquardt` on MSPE.** `optax` and
  `jax.scipy.optimize` have no LM. `optimistix` 0.1.0 works with JAX 0.11.0, and
  it has `LevenbergMarquardt`, `IndirectLevenbergMarquardt`, `GaussNewton`,
  `Dogleg`, and `BestSoFarLeastSquares`. `jaxopt` also has LM, but `optimistix`
  supersedes it.
  * LM took 28 to 87 steps to reach the FP64 floor, at every condition number
    from 1e2 to 1e11. GN reached the same floor, but it always ran to the step
    limit, because its convergence test did not fire.
  * A large condition number costs θ, and not the residual. At `cond(J) = 1e11`
    the LM solution had a relative error of 1.3e-7 in θ, and `rms abs(f)` was
    still 4.3e-16. **The target is the residual, so a large condition number does
    not block stage 1.**
  * Do not expect L-BFGS to help. At `memory_size = 300` against 60 parameters,
    which is enough memory for the full Hessian, L-BFGS stalled at 1.9e-11
    against 5.1e-16 for LM. It builds curvature from differences of gradients,
    and near the minimum those differences are round-off. LM takes `JᵀJ`, which
    is the exact Hessian in the limit of a zero residual.
  * Run one L-BFGS pass on MAPE at the end, because the L1 and L2 minimizers are
    different points. Expect little from it. Report which functional made each
    number.
* **`v` is not linear in θ.** `p` is linear in θ, so each entry of
  `p_{ij̄}p − p_i p̄_j` is quadratic in θ. `R` does not depend on θ, so the 2×2
  determinant is quartic over `p⁴`. Therefore `v` is a ratio of two quartics, and
  no single linear solve exists.
* **Point samples: use the method of `MLGeometry/hypersurface.py`.** Make a
  random line in CP^3, and find the intersection of the line and X. Each line
  gives **4 roots**. The mass is `mass = ΩΩ̄ / vol_FS`.

## Output

Save these, and report each number on a **hold-out** point set:

* `h` at full precision, with the basis convention: block order, monomial order,
  and the scale of `h`. Say how many digits of `h` are correct.
* `mean abs(f)`, `rms(f)`, and `max abs(f)` on the sample. Call `max abs(f)` a
  lower bound on `sup abs(f)`, because a point sample cannot bound a supremum.
* `‖f‖_{L²}` and `‖∇f‖_{L²}`, with the volume normalization for each.
* `min λ(φ_{ij̄})` and `min p`. A margin above zero must exist.
* The point generator and its seed.

Three rules protect the value of the result:

1. **Keep `h` exactly in the invariant subspace.** Do not let the optimizer
   leave it. The symmetry is the reason that `n_θ` is small.
   * **The basis must be canonical, and not whatever LAPACK returns.** Both
     bases come from a degenerate subspace: `section_frame` is a null space,
     where every singular value is zero, and `invariant_basis` is the image of a
     projector. An SVD then gives *an* orthonormal basis, which repeats on one
     machine and not across nodes. A θ written on an A100 node and read on an
     H200 node reported `mean abs(f) = 1.34` at k = 28, in place of 2.06e-13.
     `blocks.orthonormalize` does Gram-Schmidt in a fixed order instead, so the
     order is the convention. Gram-Schmidt combines nearly dependent columns and
     the normalization amplifies their round-off, which put the invariance
     residual at 1.5e-9, so `invariant_basis` then re-projects with `Π` a few
     times and reaches 2.7e-15.
   * **Save `blocks.fingerprint` with every θ.** `run_fit.py` writes it beside
     the θ and refuses a restart whose fingerprint differs. Without that check a
     θ from another convention fits from a wrong start and still reports a
     number.
2. **Keep the arithmetic swappable.** Put the scalar operations behind one small
   interface, so that stage 2 can change the type and not the algorithm.
3. **State the normalization of η.** `loss.py:133` divides by a sampled mean.
   That is correct in stage 1. Stage 2 must use the exact period
   `Vol(X, μ_Ω) = Γ(1/4)^8/(64π²) = 47.268180032308…` against
   `Vol(X, μ_ω) = 2π²`, so say which one each number uses.

The `jax.lax.stop_gradient` on that mean, at `loss.py:136`, is **correct**. The
exact `⟨v⟩` is `Vol(X, μ_ω)/Vol(X, μ_Ω)`, which is topological. Therefore it does
not depend on θ, and the true gradient has no term from it. Do not "repair" this.

## Open questions

1. **The sampler. Closed on 2026-08-12.** `solve_points` is vectorized and
   chunked, and it does not use `multiprocessing`. It made 1e6 points in
   approximately 2 s on the laptop. `refine_points` runs 2 Newton steps, and
   `max abs(P)/scale` was 3.3e-16 with a median of 7.0e-17, where
   `scale = Σ_a abs(z_a)^4`. **You do not need a new generator.** One change is
   necessary: `generate_random_projective` calls the global `np.random`, so pass
   a seed the way `SLagSearch/points_gen/points_generation.py:44` does.
2. **The normalization for the reported norms.** `Vol = 2π²`, or volume one?
   `‖∇f‖` carries units of 1 over length, so the choice changes that number.
3. **`cond(J)` at k = 12, 16, and 20.** Nobody measured it. It decides whether
   `optimistix` can hold `J` at high k, and it tests the first explanation of the
   fall in rate. Measure it at the LM solution.

## Notes for stage 2

Two facts that are expensive to rediscover:

* The algebraic simplifier of XLA can change `(a+b)-a` to `b`. That change
  destroys the error-free transformations of dd. **`optimization_barrier` is not
  enough.** We measured this on a dot product that cancels by 7.9e16. A product
  from Veltkamp splitting gave a relative error of 1.1e-32 outside `jax.jit` and
  9.2e-17 inside it, with barriers on every intermediate, on every returned pair,
  and at every level of the reduction. The composition also came out correct as
  soon as the intermediates escaped as outputs, which makes it a fault that only
  appears when nothing observes it.
  * **The cure is to remove the identity, and not to fence it.** `dd.split` masks
    the low 27 mantissa bits with a bitcast, so `hi` is exact and no subtraction
    is left to simplify, and the partial products of two split values are exact
    in FP64. That version gives 1.1e-32 inside `jax.jit` and outside it.
  * `two_sum` does survive `jax.jit`, so the reduction still uses it.
  * **Run the test through `jax.jit`.** A test that does not cannot see this.
* `get_restriction` uses a symbolic `jac.inv()` and `lambdify`, and both are
  FP64 numpy. `J` has a closed form: `∂w_1/∂t_i = −(∂P/∂t_i)/(∂P/∂w_1)`, with
  the identity on the other rows. `ΩΩ̄` has one too.

---

## Environment

* The laptop has **no GPU**. Do small tests only on the laptop.
* The cluster has **4 A100 GPUs**, and each GPU has 80 GB of memory.
* **Use `uv` for the environment. `pyproject.toml` and `uv.lock` define it, and
  the lock is committed.** Do not use `uv pip install` or `uv venv` by hand,
  because that leaves the lock behind.
  * cluster: `./scripts/setup.sh`, which runs `uv sync --locked --extra gpu`.
  * laptop: `NO_GPU=1 ./scripts/setup.sh`, which runs `uv sync --locked`.
  * Run every command with `uv run python ...`.
  * To add a package, put it in `pyproject.toml` and then run `uv lock`.
* The `gpu` extra holds `jax[cuda12]`. Those wheels are Linux only, so the laptop
  must leave the extra out.
* The lock pins JAX **0.11.0**, `optimistix` **0.1.0**, `optax` **0.2.8**, `flax`
  **0.12.8**, and `MLGeometry-JAX` **2.2.2**. Every measurement used these.
  `optimistix` pulls in `equinox`, `lineax`, and `jaxtyping`. Drop `jaxopt`,
  because `optimistix` supersedes it.
* Python 3.12 is necessary. The `pyproject.toml` of MLGeometry requires it, and
  the system Python of the laptop is 3.9.6.
* **MLGeometry comes from PyPI as `MLGeometry-JAX`, and version 2.2.2 is the same
  as the git HEAD `501c357`.** The repository of this project holds no copy. Do
  not put the vendored `MLGeometry/` directory on `sys.path`, because then the
  laptop and the cluster run different code.
* `SLagSearch/`: you can copy this code, or use it directly. It is not a
  dependency of stage 1.

## Conventions

* Do **not** put a `Co-Authored-By: Claude` trailer in a commit message in these
  repositories.
* Write each code comment so that it is complete without other information. Do
  not compare a change to a version that the reader did not see. Put that
  comparison in the commit message. Write 1 or 2 lines only.
