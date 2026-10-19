# Tests of the steps of the box code against arb

Date: 2026-10-19. Status: design for review.

## Intent

The formal proof in Lean will prove the mathematics of `notes/strict_bound_proof.tex`, but
not the code. So tests must show that the code of the strict bound computes what the
document describes. The user's aim (2026-10-19): close each gap in the tests of the box code.

Each step of the box (`quartic/cert/box.py`) now has one of these checks:

* a unit test of its own (the offsets, the constant of Δ, the weights of the re-expansion,
  `may_meet_f`, `_real_low`, and the Cauchy estimates in `tmgpu`);
* only the f test of `scripts/cert_check.py`: f of the GPU model against f of
  `scripts/tm_box.py` in arb, at points of 10 boxes on the slice σ = conj(τ). The largest
  gap is 3.5e-7 of the remainder, so this test finds only large errors;
* no check (κ, and the gradient bound).

This work adds a test of each step that has only the f test or no check.

## Goals

1. Each step in the table of "The steps and their references" gets a comparison with an
   independent reference in arb, at points of real boxes of the run.
2. Each comparison shows that the model of the step holds the reference value inside its
   remainder.
3. Each comparison shows which parts of the remainder it can see (see "Requirements").
4. The gradient bound gets its first comparison with a reference value of grad f.

## Non-goals

* No test of a step that has a unit test now (see "Intent"). The user decided this on
  2026-10-19.
* No MPFR. arb is the reference (the user, 2026-10-19).
* The margin of the `tmgpu` tests (a constant up to approximately 25 times too small
  passes `test_rounding_bounds`) is not in this work.
* No change of a bound and no new run of the bound. If a test finds an error, the work
  stops (see "If a test fails").
* No Lean code.

## Anti-goals

These results make the work a failure, also if each test passes:

* A test that cannot fail. Each requirement below has a way to show that the test sees
  the part that it checks.
* A reference that copies the algorithm of `box.py`. Such a reference repeats the errors
  of the code. The reference starts from the definitions of the document and the two files
  of the potential.
* A change of one bit of the run.
* A reference at a point that is not the point of the model. (Three false results in the
  history of this project came from a reference at an FP64 point.)

## Constraints

* **The conventions of `CLAUDE.md`.** STE in all text. Explicit paths in `git add`. No
  `Co-Authored-By` trailer. Code comments of 1 or 2 lines, complete without other
  information. No computation on the laptop: `python3 -m py_compile` there, then commit,
  push, and `git pull` on the cluster.
* **The jobs.** A GPU job: `-p mweber_gpu --gres=gpu:1 --constraint=h200 -c 1`. A CPU job:
  `mweber_compute`, with `-c 16` only when the job uses 16 processes. `uv run --extra gpu
  --locked`, and the `LD_LIBRARY_PATH` line of "The code of the strict bound" in
  `CLAUDE.md`.
* **No change of a bit.** A change of `box.py` for the tests (for example an argument that
  keeps the intermediate models) must not change a number. `submit_gpu_tests.sh` (the
  reference 297) and `submit_equiv_all.sh` (the references 0, 243, 297 and 231) must pass
  bit for bit after the change.
* **The independence of the reference.** The reference imports no module of
  `quartic.cert` and no module of `tmgpu`. It takes from the export only data: the dd
  numbers of the frame (p, E, n, c1inv, vol), the free dd numbers s_r, s_p and ζ, the
  numbers c_r, x₀ and s₂ of the file of local forms, the box data (center, d, d_f, d_φ,
  order, a, h_max), and the models under test. These numbers are definitions in the
  document (Definition "the local forms of a reference center", Lemmas "the frame", "the
  base" and "the level sum"), so the reference can take them as given.
* **`tm_box.py` as a reference.** The builder can add return values to `run_box` (g_t and
  the coefficients of degree 1 of f), if its present outputs do not change. A test checks the import rule, as
  `tests/test_boundary.py` of `tmgpu` does for its package.
* **The inputs.** `runs_tm/f6t_data.npz` and `runs_tm/f6t_forms.npz` (with their SHA-256
  values of `CLAUDE.md`), the local forms in `tm_local_f6t_td` (with the SHA-256 of
  `runs_tm/f6t_local_manifest.txt`), the records `runs_tm/run_f6t_v2.npz` and the cover
  `runs_tm/cover_s0.03_c6_v2.npz`. The scripts check the SHA-256 values before they start.
* **The precision of the reference.** arb (python-flint 0.9.0) at 300 bits or more. The
  test converts each dd or td coefficient of a model to an exact arb number (the sum of its
  limbs), and evaluates the polynomial of the model in arb.

## The steps and their references

The models of `box_f` are functions of (τ, σ) on the polydisc D₄(r) (Lemma "polarization").
On the slice σ = conj(τ) they are the real functions of the document. For the reference,
let y(τ) = p + E τ + h(τ) n be the point of X, and let ȳ(σ) = conj(y(conj(σ))). Then the
polarized form of a function in y and conj(y) takes y(τ) and ȳ(σ) in their places.

| # | step | models of `box_f` | reference in arb | points |
|---|---|---|---|---|
| 1 | implicit h | `h_td`, `h_dd`, `ys` (`implicit_h_td`) | the root h of P(p + E τ + h n) = 0, from an interval Newton step (Krawczyk) at 300 bits in a ball around the model value | τ in D₂(d_φ) |
| 2 | Δ | `dls[n]` (`delta_model`) | y₂(τ) ȳ₂(σ) − s₂ | D₄(d_φ) |
| 3 | base | `p0` (`delta_form_td` on the base) | the polarized p₀ from `f6t_data.npz` by Definition "the base p0": the method of `global_values` in `scripts/cert_local_values.py`, with abs(z^r)² replaced by z(τ)^r z̄(σ)^r and u^T C conj(u) by u(τ)^T C ū(σ) | D₄(d_φ) |
| 4 | abs(y)² | `r2`, `ir2`, `log_r2` | r₂ = 1 + Σ_k y_k(τ) ȳ_k(σ), 1 / r₂, log(s_r r₂) | D₄(d_φ) |
| 5 | p̃ | `pt`, `xh`, `log_pt` | p̃ = p̃₀ r₂^(−k₀), (s_p p̃)^(−1/k₀) and log(s_p p̃), principal branches | D₄(d_φ) |
| 6 | ξ | `xw`, `xi` | x = r₂⁻¹ s_p^(1/k₀) (s_p p̃)^(−1/k₀), ξ = x / x_c − 1, x_c = x₀ (1 + ζ) | D₄(d_φ) |
| 7 | F_k, k = 0 to top | `fk[k]` (`reexpand_td`, `monomial_mats_td`, `mats_at`, `compose_td`, `delta_form_td`) | F̃_k = Σ_L C(L, k) x_c^L Ñ_L, with Ñ_L the polarized N_L from `f6t_forms.npz` (the method of `global_values`) | D₄(d_φ) |
| 8 | level sum | `psi` (Horner's rule in ξ) | ψ̃ = Σ_L Ñ_L x^L | D₄(d_φ) |
| 9 | potential | `phi` | k₀ log(s_r r₂) + log(s_p p̃) + ψ̃ | D₄(d_φ) |
| 10 | metric | `g[i, j]`, `det` | g_τ = Jᵀ g_t conj(J) at a point of the slice, with g_t from `scripts/tm_box.py` (order 2, chart z₃ = 1, m = 2) and J = dt/dτ from ∂h/∂τ = −(∇P · E_i) / (∇P · n) | slice, abs(τ_i) ≤ d_f |
| 11 | volume and f | `q4`, `v`, `f` | abs(∇P(y) · n)², v / v̄ and f from `tm_box.py` | slice, abs(τ_i) ≤ d_f |
| 12 | gradient | `dfs` (`deriv1`), `grad_tau`, `det_low`, `g00_low`, `tr_up`, `lam_low`, `grad_g` (`gradient_bounds`) | ∂f/∂τ from the coefficients of degree 1 of f of `tm_box.py` at order 3, times J; abs(grad f)_G from `4 (∂f)† g⁻¹ ∂f / c` | slice, abs(τ_i) ≤ d |
| 13 | frame (host) | `ha_td`, `a6_td`, `a4_td`, `kappa_td`, `h_err_td`, `vol`, `vol_err` (`frame.frame`) | mpmath at 300 digits from the dd numbers p, E, n and c1inv of the frame | the test boxes and 64 random centers |
| 14 | κ (host) | `kappa` | s_p^(1/k₀) / (x₀ (1 + ζ)) in mpmath at 300 digits | the test boxes and 64 random inputs |

Notes on the table:

* Step 7 tests the GPU chain of the forms as one unit: the re-expansion in ζ, the monomial
  models, the composition, the cut of the order for k > 3, and Horner's rule in Δ. The
  stored local forms have their own check (`cert_local_values`).
* The identity F̃_k = Σ_L C(L, k) x_c^L Ñ_L follows from the code formula F_k = (1 + ζ)^k
  Σ_{m≥k} C(m, k) ζ^(m−k) M̃_m with M̃_m = Σ_L C(L, m) x₀^L N_L, and
  C(L, m) C(m, k) = C(L, k) C(L − k, m − k). So the reference of step 7 does not use the
  re-expansion.
* `tm_box.py` is in arb and works in the chart coordinates t with the chain rule. It does
  not import `quartic.cert`. Steps 10 to 12 use it as the reference. Its chart is the
  chart of the frame (z₃ = 1, t = (z₀, z₁)).
* Steps 1 to 9 test points off the slice, because each model claims an enclosure on all of
  D₄. Steps 10 to 12 test only the slice. Off the slice, the models of steps 10 and 11
  follow from step 9 and from the Cauchy estimates, which `tmgpu` tests.
* At each box, half of the points lie at 0.999 of the radius in each variable, and half lie
  at random in the polydisc, as in the tests of `tmgpu`.

## The test boxes

The test boxes are final boxes of the records `runs_tm/run_f6t_v2.npz` (my call):

| name | box | why |
|---|---|---|
| B1 | the largest bound of abs(f) (ρ = 0.298, order 28) | the bound of the run |
| B2 | the largest bound of abs(grad f) (ρ = 0.887) | the gradient bound of the run |
| B3 | a box of the reference 0 (ρ ≈ 0.02) | the strongest cancellation of the forms |
| B4 | a box of the reference 243 (the largest ρ) | the largest δ |
| B5 | a sub-cell of depth 1 | the split |
| B6 | a box of the reference 297 | the reference of the unit tests |
| B7, B8 | two random final boxes | no selection |

Each test box runs with the order, a, δ and h_max of its record. Each one also runs with δ
times 1e-3 and δ times 1e-9, where the round-off and the errors of the host numbers are the
largest parts of the remainder. If its bound stays finite, it also runs with δ times 1.5,
where the truncation is the largest part. These scaled boxes keep the center, the order, a
and h_max.

## The batches (my call)

`box_f` takes the number of terms of each series from the largest η of its batch
(`tmgpu.tools.series_order`). So a box gives the bits of its record only in the batch of the
run. Also, a batch of one cannot show an error along the batch axis, for example data of box
j in the place of box i.

* **At the radius of the record**, each test box runs in the batch of the run. The export
  replays the queue of `cert_run` for the reference center of the box, with the retries at
  order 32 and the splits, up to that batch. The hook keeps only the slices of the test box.
  The sup bound and `grad_g` of the test box must then equal the record bit for bit. The
  replay costs a few GPU minutes for each reference center (on average about 400 boxes at
  0.275 s each).
* **At a scaled radius**, the whole batch of the run is scaled by the same factor, and the
  hook again keeps only the slices of the test box. There is no record to compare with.
* `tmgpu` does not change. Each new part of a remainder in `info` comes from `box.py`.

## Requirements

* **R1, enclosure.** At each point x, the gap abs(P(x) − ref(x)) must be at most the
  remainder R of the model. P(x) and the gap are arb balls, and the test uses the upper
  bound of the gap. A NaN counts as a failure.
* **R2, the parts that the test sees.** `box_f` gives the parts of many remainders in
  `info` (for example `parts_p0`, `psi_parts`, `xi_parts`, `fk_rem`). For each part π of
  the remainder of a step, the report gives the largest value of gap / (R − π) over the
  points and the radii. The test sees π when this value exceeds 1 at some point. Each part
  must be seen, or the report must list it with its largest value. The user then decides
  about each part that the tests do not see. The builder can add parts to `info`, for
  example the two terms of ρ in `implicit_h_td`, if no bit of the run changes.
* **R3, the scalar bounds of the gradient.** At each point of the slice:
  abs(∂f/∂τ_i) ≤ the bound of `dfs[i]`, det g ≥ `det_low`, g₀₀ ≥ `g00_low`,
  tr g ≤ `tr_up`, λ_min(g) ≥ `lam_low`, and abs(grad f)_G ≤ `grad_g`. The report gives the
  ratio of each.
* **R4, the host numbers.** Each td or dd number of steps 13 and 14 must lie within its
  error bound of the mpmath value. The test also runs with each error bound set to 0, and
  then it must fail, so that the test sees each bound. A number that is exact in td or dd
  (for example `a4_td` = 4 p_k) cannot fail this way; the report lists each such number.
* **R5, the report.** For each step and each box, the report gives the largest gap / R,
  the table of R2 and the ratios of R3. The exit status is 1 if R1, R3 or R4 fails.

## The structure (my call)

* **A hook in `box_f`.** An optional argument, for example `keep=None`. With a dict,
  `box_f` puts each model of the table into it. It changes no number.
* **`scripts/cert_steps.py`**, with two stages, as `scripts/residual_jets.py` has:
  * `export` (one H200): for each test box and each radius, it runs `box_f` with the hook
    and writes the models, the remainders, the parts, the frame, s_r, s_p, ζ, κ and the
    SHA-256 of the inputs into a new directory. It stops if the directory exists.
  * `compare` (`mweber_compute`, `-c 16`): the reference, the comparisons and the report.
* **The reference module**, outside `quartic/cert` (for example `tests/box_reference.py`).
  It holds steps 1 to 9 and calls `tm_box.py` for steps 10 to 12.
* **The unit test `tests/test_box_steps.py`.** A short form for `submit_gpu_tests.sh`:
  the box B6, the run radius, 4 points, steps 1 to 9. The target is less than 15 minutes on
  the GPU node.
* **The host tests of steps 13 and 14** go into `tests/test_tm_host.py`.

## The order of the work

1. The hook, the export and the reference of steps 1 to 9 on B1 at the run radius. Then
   measure the time of the reference for each point and step, and the values of R1 and R2.
   Give the user a table of the cost and the benefit before the work continues to the
   other boxes and steps.
2. Steps 13 and 14 (host tests).
3. Steps 10 to 12 with `tm_box.py`. First measure the time of `tm_box.py` at order 3 for one
   point.
4. All test boxes and radii, and the report.
5. The unit test, and its entry in `submit_gpu_tests.sh`.
6. After the user accepts the results: a Check line for each lemma in
   `notes/strict_bound_proof.tex` (the frame, the contraction, the base, the level sum, the
   metric and f, the gradient), and the results in `CLAUDE.md`.

## If a test fails

The work stops and reports the step, the box, the point and the numbers. A test can fail
because of an error in the reference. So the first action is a second, independent
computation of the reference value at that point. A change of `box.py` that changes a bit
of the run needs the decision of the user and a new full run of the bound.

## What is left to the builder

* The names of the files and of the functions, and the form of the export.
* The number of points for each box (at least 16 for each step and radius in the full
  check).
* The method of the interval Newton step, and the arb precision (300 bits or more).
* The details of J and of the change of coordinates for steps 10 to 12.
* The parallel layout of the `compare` stage.
