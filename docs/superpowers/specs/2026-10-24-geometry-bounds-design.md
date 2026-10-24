# The geometric bounds of the draft for f6t: design

The collaborators need these bounds for the approximate metric ω of f6t (the "% for Yidi" marks
of `notes/draft-main.tex`):

1. the metric ratio `ω ≤ C ω₀` (`equation:metric-ratio`), with ω₀ the Fubini-Study (FS) form;
2. the Ricci bound `Ric_g ≥ −3κ g` (`eqn:ricci-bound`);
3. the atlas bounds from `equation:parent-chart-margin` to
   `equation:parent-transition-third-derivative-bound`: ρ, λ ≤ G_a ≤ Λ on V_a, K̃, c̃1 and c̃2.

The user's rules (2026-10-24): one full run with the Arb-rounded code gives these bounds and the
bounds of f together; do the work correctly, with no overdesign.

## Conventions

* Both forms have volume 1: ω = c i∂∂̄Φ with c = 1 / (√2 π k₀), and ω₀ = FS ÷ (π√2) (the
  diameter section of the draft). The Riemannian metric is G(u, u) = c g_{ij̄} uⁱ ūʲ for
  u ∈ C² ≅ R⁴, with g = ∂∂̄Φ. So the real 4 × 4 matrix G_a has the eigenvalues of c g, each two
  times. In the same way G₀(u, u) = g_FS,ij̄ uⁱ ūʲ ÷ (π√2), with g_FS = ∂∂̄ log(1 + abs(y)²).
* Each tensor norm is the real Frobenius norm (the conventions of the draft). For a
  complex-multilinear map B of k arguments, the real norm is 2^(k/2) times the complex norm
  (each of the 2^k choices of 1 or i in the arguments gives abs(B_I)).
* The chart of F (z₃ = 1, z₂ solved, t = (z₀, z₁)) is a parent chart of the draft, with X₀ = z₃
  and X₁ = z₂. G, complex conjugation and the coordinate phases are isometries of ω and ω₀ and
  keep the atlas, so a sup over F and over the chart of F gives the sup over all charts.

## The bounds of each box (GPU, `quartic/cert/box.py`)

A new function `metric_bounds` runs after `gradient_bounds` in `box_f`. All models exist there;
the function adds models but changes no model of f. Its outputs are new record columns. The
products take g and g_FS at the order METRIC_ORDER = 10 (the rest goes to the remainder).

* **Eigenvalue bounds.** For Hermitian 2 × 2 models, λ = (a + d ∓ √((a − d)² + 4|b|²)) ÷ 2 with
  the bounds of Re a, Re d, |a − d| and |b| (`_eig_bounds`). The bound λ_min ≥ det ÷ tr loses a
  factor up to 2 for near-equal eigenvalues, so the new bounds do not take it.
* **Normalization at the center.** Any constant P gives exact inequalities. P = conj(U) W^(−1/2)
  with g(c) = U W Uᴴ in FP64 makes Pᵀ g P̄ near I (`_near_inv_sqrt`, `_congruent`).

1. **The ratio μ = eig(g_FS⁻¹ g) ÷ k₀** (coordinate-free). With P from g_FS(c), A = Pᵀ g_FS P̄
   and B = Pᵀ g P̄: μ ∈ [λ_min(B) ÷ (k₀ λ_max(A)), λ_max(B) ÷ (k₀ λ_min(A))]. Columns `mu_up`,
   `mu_low`.
2. **Ricci.** Ric_{ij̄} = −∂ᵢ∂ⱼ̄ log(1 + f), because 1 + f = det g_τ abs(∇P·n)² vol, with a
   holomorphic factor. abs(Ric)_F ≤ abs(∂∂̄f)_F ÷ (1 − abs(f)) + abs(∂f)² ÷ (1 − abs(f))², and
   each eigenvalue of Ric against G has abs at most (2 ÷ c) abs(Ric)_F ÷ λ_min(g). Column `ric`.
3. **The derivative of the metric.** In coordinates z = P w, with g_P = Pᵀ g P̄ and
   Γ_P[c] = Σ_k P_kc ∂_k g_P, the real norm of DG normalized at the point is at most
   √8 abs(Γ_P)_F ÷ (√c λ_min(g_P)^(3/2)): the normalizing matrix at x is P M with
   abs(M)² = 1 ÷ (c λ_min(g_P)), and a complex 3-tensor Γ of c ∂g gives the real norm √8 abs(Γ)_F.
   This form takes the sup of abs(Γ_P) and the inf of λ_min from different points. The second
   form (`_k_square`) bounds S = Σ Γ[c′, (i′, j′)] conj(Γ[c″, (i″, j″)]) M[c″, c′] M[i″, i′] M[j′, j″]
   with M = g_P⁻¹ (the adjugate and a series for 1 ÷ det) as one model: S is the square of the
   norm with each slot normalized by g_P at the point, so the bound is √(8 B(S) ÷ c). `_k_bound`
   takes the smaller of the two. The second form is about 2 times better on the cells of half
   the side of a box of the run, and much worse on some full boxes, where the series of 1 ÷ det
   has large terms. Column `k_tau` (in τ).
4. **In the chart t.** t = p₀₁ + τ + n₀₁ h(τ), J = dt/dτ = I + n₀₁ ∇hᵀ, J⁻¹ = I − n₀₁ ∇hᵀ ÷
   (1 + ∇h·n₀₁) (a series in sp2), g_t = J⁻ᵀ g J̄⁻¹ as models, and ∂_t = J⁻ᵀ ∂_τ, so that
   Γ_P[c] = Σ_a (J⁻¹ P)_ac ∂_τa g_t,P. The same bound as in 3 gives column `k_t`.

5. **In each chart of the atlas.** `cover.chart_mask` (mpmath.iv on the cell) gives the charts
   (i, j) whose domain D(i, j) can meet the cell. For each of them, t_a = (z_k ÷ z_i, z_l ÷ z_i)
   from the models of the point y (a series for 1 ÷ z_i when i ≠ 3), J_a = dt_a/dτ by Cauchy,
   J_a⁻¹ by the adjugate and a series for 1 ÷ det, and g_a = J_a⁻ᵀ g J̄_a⁻¹; the same bound as in 3
   gives k_a, and the eigenvalues of G_a = c g_a come from A = Pᵀ P̄ and B = Pᵀ g_a P̄ as in 1.
   Outputs `k_atlas` (the largest k_a), `lam_atlas` and `Lam_atlas` (the smallest and the largest
   eigenvalue). The chart (3, 2) always enters, since F lies in its domain, so k_atlas ≥ k_t; a
   cell at the edge of F that chart_mask proves outside D(3, 2) holds no point of F. A chart a and a point x ∈ U_a go to a point of F and a chart that meets its cell
   by an element of G, which is an isometry and maps charts to charts by linear maps, so these
   outputs bound K̃, λ and Λ of the draft directly.

Each FP64 operation of these bounds uses `tmgpu.mag` (up for upper bounds, down for lower
bounds). A box with no proof (a lower bound ≤ 0) gets inf for an upper bound and 0 for a lower
bound. `cert_partition` checks that each column of the run is finite and that mu_low > 0.

## The metric pass (GPU, `scripts/cert_metric.py`)

On a box of the run (side 0.03, radius about 0.024) the bounds 3 to 5 are loose in a distorted
chart, because the bound of abs(Γ_P) and of λ_min(g_P) over the box are separate. The probes of
2026-10-24 measured it:

With the first form of 3 alone:

| box | k_t, the record | at the center | largest bound, cells of side s ÷ 2 | side s ÷ 4 |
|---|---|---|---|---|
| the 12 boxes of v3 with k_t > 400 (corner abs(t₀) ≈ abs(t₁) ≈ abs(z₂) ≈ 0.8) | 400 to 439 | 48 to 55 | 95 to 131 | 65 to 84 |

On 210 of the 4,402 boxes of the references 0, 243, 297 and 231 (all near ρ = 1, where all 12
charts can meet the cell) a chart gives no proof; on the cells of half the side every chart gives
one, and at the centers k_a is 38 to 61 in every chart.

So `cert_metric` computes the bounds 1 to 5 on cells of its own: the cells of the cover, each
split into the cells of half the side that may meet F (`cover.halves`) while k_atlas > --k-target
or a bound has no proof, at most --depth times. The bounds take g at METRIC_ORDER = 10, so a low
order is enough:
* on the 27 cells of side s ÷ 4 of the box with the largest k_t, the bounds at order 12 equal
  those at order 32 to 4 digits, in 55 ms for each box against 308 ms (order 10 loses 5 percent);
* on full boxes, orders 20, 24 and 32 give the same bounds, and order 12 gives no proof at large ρ.
So the cells of the cover run at --order 20 and the smaller cells at --order-split 12.

The pilot (jobs 55135390 to 55135399, the 2,789 cells of the references 139, 243, 231, 0 and 297,
depth 2, both forms of 3) and its trace give the cost of each target of k_atlas:

| target | boxes computed | largest k_atlas |
|---|---|---|
| 60 | 29,020 | 60.00 |
| 70 | 18,533 | 69.97 |
| 80 | 16,094 | 80.00 |
| 100 | 14,302 | 99.98 |

A box takes 57 to 70 ms at order 12 and about 90 ms at order 20 on one H200. The full pass takes
the target 60. Its
records (`records.METRIC_COLUMNS`) do not bound abs(f); their ric is useless, because f at these
orders has a large remainder, and combine takes the ric of the run.
`cert_partition --metric` checks that they make a partition of each cell of the cover, their
radii and lam_big, that each bound is finite and not below 0, that mu_low > 0 and lam_atlas > 0,
and that k_atlas ≥ k_t. The run of f keeps the 24 columns of v3, so the run v3 stays the run of
the bound. With the second form of 3, a new cert_run equals v3 bit for bit in the columns before
k_tau, and its k_tau and k_t are at most those of v3.

## The combination (`scripts/cert_atlas.py combine`)

From the records of the run, the records of cert_metric (`--metric`) and the two CPU results,
in arb with outward rounding:

* C = max mu_up; μ_min = min mu_low; κ = max ric ÷ 3; each is the better of the two sets of
  records, since each set covers F;
* λ = the larger of μ_min λ₀ and min lam_atlas; Λ = the smaller of C Λ₀ and max Lam_atlas;
* c̃1 = √C c̃1⁰ ÷ μ_min and c̃2 = √C c̃2⁰ ÷ μ_min^(3/2), from abs(Q_a v) ≤ √C abs(Q₀_a v) and
  abs(Q₀_b Q_b⁻¹) ≤ μ_min^(−1/2); or c̃1 = Λ^(1/2) λ⁻¹ sup abs(D²F) and c̃2 = Λ^(1/2) λ^(−3/2)
  sup abs(D³F); the smaller of the two;
* K̃ = the smaller of max k_atlas and max k_t + 2 c̃1 (the chain rule from the chart of F to any
  chart, with the term of c̃1). In the v3 run the second gave 9,341 (c̃1 = 4,451 from the plain
  norms), so the first route is the one that counts;
* the C¹ term of the residual in the parent charts: Λ^(1/2) max grad_g;
* the diameters sqrt(C) diam(·, ω₀) from the Proposition of the draft.

ρ > 1/200 has a proof in the draft (Proposition `prop:atlas-rho-bound`), so it needs no code.

## Tests

* `tests/test_metric_bounds.py` (GPU): synthetic polynomial models with an mpmath reference from
  the definitions in real coordinates (μ, the eigenvalues of Ric against G, the real norm of
  DG normalized at the point, and DG in t from finite differences of G_t): each bound is at or
  above the value at random points, and within a factor of the largest value. On a real box:
  g_FS at the center equals Eᵀ h_FS(p) Ē in arb, and mu_low ≤ μ(center) ≤ mu_up.
* `tests/test_cert_atlas.py` (CPU): the jets against mpmath derivatives, the FS metric against
  mpmath, the real norm of D²F against the tensor in real coordinates, the FS normalization of a
  transition in both charts against the form with DF⁻¹, and the result of the branch-and-bound
  against random points of the domain.
* The run: on the references 0, 243, 297 and 231 the first 19 columns equal those of
  `base_all_v020.npz` bit for bit, and the time for each box is measured.
