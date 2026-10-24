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
the function adds models but changes no model of f. Its outputs are new record columns.

1. **The ratio μ = eig(g_FS⁻¹ g) ÷ k₀** (coordinate-free, so τ is correct). With g_FS from
   `cauchy_mixed` of log_r2: s = tr(adj(g_FS) g) ÷ (k₀ det g_FS) and p = det g ÷ (k₀² det g_FS)
   bound the sum and the product of the two values. μ_max ≤ (s⁺ + √(s⁺² − 4p⁻)) ÷ 2 and
   μ_min ≥ p⁻ ÷ μ_max⁺, from the upper bound s⁺ and the lower bound p⁻ on the box.
   Columns `mu_up`, `mu_low`.
2. **Ricci.** Ric_{ij̄} = −∂ᵢ∂ⱼ̄ log(1 + f), because 1 + f = det g_τ abs(∇P·n)² vol, with a
   holomorphic factor. abs(Ric)_F ≤ abs(∂∂̄f)_F ÷ (1 − abs(f)) + abs(∂f)² ÷ (1 − abs(f))², and
   each eigenvalue of Ric against G has abs at most (2 ÷ c) abs(Ric)_F ÷ λ_min(g). Column `ric`.
3. **The derivative of the metric in τ.** With Γ_k = c ∂_k g in the coordinates where c g = I,
   the real tensor DG[u](v, w) has the norm √8 abs(Γ)_F. So
   K_τ ≤ √8 abs(∂g)_F ÷ (√c λ_min(g)^(3/2)). Column `k_tau`.
4. **From τ to the chart t.** t = p₀₁ + τ + n₀₁ h(τ). The chain rule for G_t = Dτᵀ G_τ Dτ gives
   K_t ≤ K_τ + 2N with N ≤ 2 √λ_max(g) abs(J⁻¹) abs(n₀₁) abs(∂²h)_F ÷ (√c λ_min(g)),
   J = I + n₀₁ ∇hᵀ and abs(J⁻¹) ≤ 1 ÷ (1 − abs(n₀₁) abs(∇h)). Column `k_t`.

Each FP64 operation of these bounds uses `tmgpu.mag` (up for upper bounds, down for lower
bounds). A box with no proof (a lower bound ≤ 0) gets inf for an upper bound and 0 for a lower
bound. `cert_partition` checks that each new column is finite and that mu_low > 0.

## The bounds of FS and of the chart maps (CPU, `scripts/cert_atlas.py`, arb)

Chart b = (3, 2): z = (t₀, t₁, w, 1), w⁴ = −(1 + t₀⁴ + t₁⁴). Its domain is
D = {1 > 0.99 max(abs(t₀), abs(t₁), abs(w)), abs(w) > 0.98 max(abs(t₀), abs(t₁))}, with no window
of the argument (a superset of each V_a). The symmetry reduces it to arg t₀ ∈ [0, π/4] and
arg t₁ ∈ [0, π/2].

1. `fs`: λ₀ = inf over D of λ_min(g_FS) ÷ (π√2) and Λ₀ = sup of λ_max(g_FS) ÷ (π√2). The values
   do not depend on the branch of w.
2. `transitions`: for each chart a = (i, j) ≠ (3, 2), F_a(t) = (z_k ÷ z_i, z_l ÷ z_i); the sups
   over D ∩ D(i, j) of abs(D²F_a) and abs(D³F_a) (real norms), from Taylor jets of degree 3 in
   acb. D(i, j) = {abs(z_i) > 0.99 max of the others, abs(z_j) > 0.98 max of the other two}. The
   norms do not depend on the branch of w.
3. A branch-and-bound on boxes of (Re t₀, Im t₀, Re t₁, Im t₁): a box that cannot meet the
   domain goes out; a box whose bound is within the tolerance of the best point value, or of
   the smallest side, gives its bound. The result is the largest (smallest) bound of a box.

## The combination (`scripts/cert_atlas.py combine`)

From the records of the run and the two CPU results, in arb with outward rounding:

* C = max mu_up; μ_min = min mu_low; κ = max ric ÷ 3;
* λ = μ_min λ₀ and Λ = C Λ₀;
* c̃1 = Λ^(1/2) λ⁻¹ max_a sup abs(D²F_a), c̃2 = Λ^(1/2) λ^(−3/2) max_a sup abs(D³F_a);
* K̃ = max k_t + 2 c̃1 (the chain rule from the chart of F to any chart, with the term of c̃1);
* the C¹ term of the residual in the parent charts: Λ^(1/2) max grad_g;
* the diameters sqrt(C) diam(·, ω₀) from the Proposition of the draft.

ρ > 1/200 has a proof in the draft (Proposition `prop:atlas-rho-bound`), so it needs no code.

## Tests

* `tests/test_metric_bounds.py` (GPU): synthetic polynomial models with an mpmath reference from
  the definitions in real coordinates (μ, the eigenvalues of Ric against G, the real norm of
  DG normalized at the point, and DG in t from finite differences of G_t): each bound is at or
  above the value at random points, and within a factor of the largest value. On a real box:
  g_FS at the center equals Eᵀ h_FS(p) Ē in arb, and mu_low ≤ μ(center) ≤ mu_up.
* `tests/test_cert_atlas.py` (CPU): the jets against mpmath differences, the FS metric against
  mpmath, the real norm of D²F against the tensor in real coordinates, and the result of the
  branch-and-bound against random points of the domain.
* The run: on the references 0, 243, 297 and 231 the first 19 columns equal those of
  `base_all_v020.npz` bit for bit, and the time for each box is measured.
