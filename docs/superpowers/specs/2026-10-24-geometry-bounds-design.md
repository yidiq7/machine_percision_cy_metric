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
   Column `k_tau` (in τ).
4. **In the chart t.** t = p₀₁ + τ + n₀₁ h(τ), J = dt/dτ = I + n₀₁ ∇hᵀ, J⁻¹ = I − n₀₁ ∇hᵀ ÷
   (1 + ∇h·n₀₁) (a series in sp2), g_t = J⁻ᵀ g J̄⁻¹ as models, and ∂_t = J⁻ᵀ ∂_τ, so that
   Γ_P[c] = Σ_a (J⁻¹ P)_ac ∂_τa g_t,P. The same bound as in 3 gives column `k_t`.

5. **In each chart of the atlas.** `cover.chart_mask` (mpmath.iv on the cell) gives the charts
   (i, j) whose domain D(i, j) can meet the cell. For each of them, t_a = (z_k ÷ z_i, z_l ÷ z_i)
   from the models of the point y (a series for 1 ÷ z_i when i ≠ 3), J_a = dt_a/dτ by Cauchy,
   J_a⁻¹ by the adjugate and a series for 1 ÷ det, and g_a = J_a⁻ᵀ g J̄_a⁻¹; the same bound as in 3
   gives k_a, and the eigenvalues of G_a = c g_a come from A = Pᵀ P̄ and B = Pᵀ g_a P̄ as in 1.
   Columns `k_atlas` (the largest k_a), `lam_atlas` and `Lam_atlas` (the smallest and the largest
   eigenvalue). A chart a and a point x ∈ U_a go to a point of F and a chart that meets its cell
   by an element of G, which is an isometry and maps charts to charts by linear maps, so these
   columns bound K̃, λ and Λ of the draft directly.

Each FP64 operation of these bounds uses `tmgpu.mag` (up for upper bounds, down for lower
bounds). A box with no proof (a lower bound ≤ 0) gets inf for an upper bound and 0 for a lower
bound. `cert_partition` checks that each new column is finite, that mu_low > 0 and
lam_atlas > 0, and that k_atlas ≥ k_t.

## The bounds of FS and of the chart maps (CPU, `scripts/cert_atlas.py`, arb)

Chart b = (3, 2): z = (t₀, t₁, w, 1), w⁴ = −(1 + t₀⁴ + t₁⁴). Its domain is
D = {1 > 0.99 max(abs(t₀), abs(t₁), abs(w)), abs(w) > 0.98 max(abs(t₀), abs(t₁))}, with no window
of the argument (a superset of each V_a). The symmetry reduces it to arg t₀ ∈ [0, π/4] and
arg t₁ ∈ [0, π/2].

1. `fs`: λ₀ = inf over D of λ_min(g_FS) ÷ (π√2) and Λ₀ = sup of λ_max(g_FS) ÷ (π√2). The values
   do not depend on the branch of w.
2. `transitions`: for each chart a = (i, j) ≠ (3, 2), F_a(t) = (z_k ÷ z_i, z_l ÷ z_i), and the
   sups c̃1⁰ and c̃2⁰ over D ∩ D(i, j) of the real norm of Q₀_a DⁿF_a[Q₀_b⁻¹ ·] (n = 2, 3) with
   the FS forms at the same point. FS is one metric, so abs(Q₀_a w) = abs(Q₀_b DF⁻¹ w), and the
   norm is at most √λ_max λ_min^(−n/2) abs(DF⁻¹ DⁿF) with the FS eigenvalues of chart b.
   D(i, j) = {abs(z_i) > 0.99 max of the others, abs(z_j) > 0.98 max of the other two}.
3. **The bound of a box** of side s and center c (abs(x − c)₂ ≤ s) is the better of the ball
   value and a second-order Taylor form from jets at c and on the box:
   * E = DF⁻¹ DⁿF from holomorphic jets of F of degree 5: abs(E(x)) ≤ abs(E(c)) +
     abs(DE(c)) s + sup abs(D²E) s² ÷ 2;
   * the FS eigenvalues: the potential log(1 + t·σ + w(t) w̃(σ)) in (t, σ), σ = conj(t) on the
     slice, as jets of degree 4. abs(g(x) − g(c))_F is at most the linear terms at c plus the
     terms of degree 2 on the box, with abs(xᵢ − cᵢ) ≤ s ÷ √2, and Weyl's inequality moves each
     eigenvalue by at most that.
   The first-order forms (ball partials) left gaps of 5 to 34 percent at the side R/256, because
   the partials on a box grow as much as the values.
4. A branch-and-bound on boxes of (Re t₀, Im t₀, Re t₁, Im t₁), in rounds of one side so that each
   round knows the best point value: a box that cannot meet the domain goes out; a box whose
   bound is within the tolerance of the best point value, or of the smallest side, gives its
   bound. The result is the largest (smallest) bound of a box. For the transitions a floor from
   random points of all 11 charts lets a chart stop early, since combine takes only the maximum.
5. The FS eigenvalues of a ball also come from two closed forms with no root of a ball (s g =
   I + a aᴴ − b bᴴ / s, and the same in the basis a / abs(a), which has no cancellation near the
   corner abs(t₀) ≈ abs(t₁) ≈ abs(w) ≈ 0.8).
6. Two routes for the transitions: `c1 c2` (the FS form at the same point, normalized at the box
   center by S = Q_c⁻¹) and `d2 d3` (the plain real norms of D²F and D³F, with the factors
   Λ^(1/2) λ⁻¹ and Λ^(1/2) λ^(−3/2)). combine takes the smaller valid bound. On 2026-10-24 only
   `d2 d3` converged in the time of a run: near the overlaps of near-equal moduli the jets on
   balls lose the cancellations of the formulas, so `c1 c2` needs the side R/512 there.

## The combination (`scripts/cert_atlas.py combine`)

From the records of the run and the two CPU results, in arb with outward rounding:

* C = max mu_up; μ_min = min mu_low; κ = max ric ÷ 3;
* λ = the larger of μ_min λ₀ and min lam_atlas; Λ = the smaller of C Λ₀ and max Lam_atlas;
* c̃1 = √C c̃1⁰ ÷ μ_min and c̃2 = √C c̃2⁰ ÷ μ_min^(3/2), from abs(Q_a v) ≤ √C abs(Q₀_a v) and
  abs(Q₀_b Q_b⁻¹) ≤ μ_min^(−1/2); or c̃1 = Λ^(1/2) λ⁻¹ sup abs(D²F) and c̃2 = Λ^(1/2) λ^(−3/2)
  sup abs(D³F); the smaller of the two;
* K̃ = the smaller of max k_atlas and max k_t + 2 c̃1 (the chain rule from the chart of F to any
  chart, with the term of c̃1). In the v3 run the second gave 9,341, and k_t alone is at most
  438.7, so the first route is the one that counts;
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
