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
* `fit.jacobian_memory` reports the peak from the compiled program, with no run.
* `fit.suggest_chunk` picks the block from a memory budget. `run_fit.py` calls it
  by default, through `--budget-gb`.

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

| span | 10 → 12 | 12 → 16 | 16 → 20 |
|---|---|---|---|
| first-order | 2.50 | 2.43 | 1.92 |
| LM | **2.90** | **2.81** | **2.73** |

The HN law predicts 2.83, so the first explanation was the correct one, and the
ansatz does not slow down. Their own estimate of a converged fit at k = 20 was
4.5e-11 against the 6.33e-11 we measured. **The largest risk in the project is
closed.**

The k = 20 fit hit the 200-step limit, so its σ is an upper bound. A lower value
would raise the 16 → 20 rate, and the small decline that is left can be that.

**Stage 1 needs k ≈ 31, and not 38.** At a rate of 2.73 to 2.83, σ = 1e-15 comes
at k = 30.6 to 31.0, and k = 32 gives 2.4e-16 to 3.7e-16.

### The binding constraint is now the arithmetic

σ falls below the FP64 floor of "Precision budget" before k = 32:

| k | 24 | 28 | 30 | 32 |
|---|----|----|----|----|
| σ predicted | 9.9e-13 | 1.5e-14 | 1.9e-15 | 2.4e-16 |

The floor is approximately 5e-15. So **σ meets it near k = 28, and past that the
number is round-off and not the ansatz.** The compensated accumulator on `p`,
`∂p` and `∂∂̄p` therefore decides whether stage 1 reaches 1e-15. It is no longer
an item for the end of the project.

`cond(J)` grows by 5.65 for each unit of k. It passes `1/ε = 4.5e15` at
**k = 23**, so `J` is numerically singular in FP64 from k = 24. A large condition
number costs θ and not the residual, which we measured to `cond(J) = 1e11`. Past
`1/ε` the QR cannot resolve the smallest directions at all. **Run k = 24 next and
compare σ against the rate. That test is cheap and it decides the endgame.**

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
  destroys the error-free transformations of dd. Use
  `jax.lax.optimization_barrier`, and test this danger early.
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
