# Tests of the steps of the box code against arb: the plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to do this plan task by task. Each step has a
> checkbox (`- [ ]`) to record its progress.

**Goal:** Compare each step of `quartic/cert/box.py` that has no unit test of its own with an
independent reference in arb, at points of 8 real boxes of the run and at scaled radii, and
compare the gradient bound with a reference value of grad f for the first time.

**Architecture:** A hook in `box_f` keeps the intermediate models, and a hook in the queue of
`cert_run` replays the batch of the run that holds a test box. `scripts/cert_steps.py export`
(one H200) writes the models of each test box to a new directory. `scripts/cert_steps.py
compare` (`mweber_compute`, 16 cores) evaluates each model in arb and compares it with the
reference of `tests/box_reference.py`, which imports no `quartic.cert` and no `tmgpu`. Steps 10
to 12 of the design take g, f and ∂f from `scripts/tm_box.py`.

**Tech stack:** Python 3.12, CuPy, python-flint 0.9.0 (arb), mpmath, numpy, SLURM.

**Spec:** `docs/superpowers/specs/2026-10-19-box-step-tests-design.md`. The step numbers 1 to 14
and the requirements R1 to R5 below are those of the spec.

## Global constraints

* Write all text in STE: code comments, docstrings, commit messages, reports and documents.
  See the language rule of `CLAUDE.md`.
* No `Co-Authored-By` trailer and no other line of attribution in a commit.
* Stage explicit paths. Never `git add -A` or `git add .`. Check the `??` lines before each
  commit.
* Work on the branch `box-step-tests` from `main` at the commit of this plan. On the cluster, `git checkout
  box-step-tests` in `~/machine_percision_cy_metric` only when `squeue -u yidqi` shows no job
  of the bound. Merge into `main` as a fast-forward after Task 9.
* No computation on the laptop: only `python3 -m py_compile`, `git`, `grep`, and `pdflatex` in
  the scratchpad. Run every test and every stage on the cluster.
* A GPU job: `-p mweber_gpu --gres=gpu:1 --constraint=h200 -c 1`. A CPU job: `-p
  mweber_compute`, with `-c 16` only for a stage with `--procs 16`. Each job runs
  `uv run --extra gpu --locked python -u ...`, after `export CUDA_PATH=...` and the
  `LD_LIBRARY_PATH` line of `submit_gpu_tests.sh`.
* **No change of a bit.** After a change of `box.py` or `cert_run.py`, `sbatch
  submit_gpu_tests.sh` and `sbatch submit_equiv_all.sh` must both print `RESULT: PASS`.
* `tmgpu` does not change. `pyproject.toml` and `uv.lock` do not change.
* `tests/box_reference.py` imports only the standard library, numpy, flint and
  `scripts/tm_box.py`. It sets `flint.ctx.prec = 320`.
* The inputs: `runs_tm/f6t_data.npz` (SHA-256
  69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1), `runs_tm/f6t_forms.npz`
  (2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2), the local forms in
  `/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td` (the SHA-256 of
  `runs_tm/f6t_local_manifest.txt`), the records `runs_tm/run_f6t_v2.npz` and the cover
  `runs_tm/cover_s0.03_c6_v2.npz`. Each stage checks these SHA-256 values and stops if one
  differs.
* Each stage that writes files needs a new directory, and stops if the directory exists.
* Code comments: 1 or 2 lines, complete without other information. A comparison with an
  earlier version goes in the commit message.

## Review focus

1. **The reference at another point.** The model and the reference must take the same exact
   point: `compare` builds each point once as arb balls from FP64 numbers, and gives that
   object to both `poly_at` and the reference. Task 4, Step 1 has a test that a reference at a
   point moved by 1e-25 gives a failure at the factor 1e-9.
2. **The wrong box of a batch.** An error in the slice of the batch gives the models of another
   box. The export asserts that the sliced frame center equals the center of the record, and
   that the sliced sup bound and `grad_g` equal the record bit for bit (Task 3, Step 1).
3. **A remainder that claims nothing.** An infinite remainder passes R1 with no test. At the
   radius of the record, an infinite or NaN remainder is a failure; at a scaled radius, the
   report lists the model as "no claim" and does not count it as a pass (Task 4, Step 1).
4. **A wide reference ball.** A reference ball with a radius above 1e-3 of R makes the gap
   meaningless. `compare` stops with an error for such a point (Task 4, Step 1).
5. **The branch off the slice.** Off the slice, s_p p̃ is complex. The reference must take
   the principal power and the principal logarithm, as Lemmas "the base" and "the metric and
   f" do. Task 2, Step 1 tests the principal branch at a point where arg(s_p p̃) is near 0.4.

---

### Task 1: The hooks, with no change of a bit

**Files:**
- Modify: `quartic/cert/box.py` (`implicit_h_td`, `box_f`, `gradient_bounds`)
- Modify: `scripts/cert_run.py` (the loop of `main` moves into `run_reference`)
- Create: `tests/test_box_steps.py`

**Interfaces:**
- Produces:
  - `implicit_h_td(sp, fr, delta, hmax, parts=None)`. With a dict, it holds the CuPy arrays
    (B,) `residual` (the bound of T(h) − h), `host` (eps), `lam0` and `rho`. Its return value
    does not change.
  - `box_f(fr, local, delta_box, n, ny, a, hmax, grad=False, keep=None)`. `info['h_parts']`
    always holds the dict of `implicit_h_td`. With a dict `keep`, `box_f` puts into it:
    `d`, `d_f`, `d_phi`; the models `h_td` (TMT), `h_dd`, `ys` (list of 3), `delta`, `p0`,
    `r2`, `ir2`, `log_r2`, `pt`, `xh`, `log_pt`, `xw`, `xi`, `fk` (list of top + 1),
    `psi`, `phi`, `g` (dict with keys (i, j)), `det`, `q4`, `v`, `f`; `orders` (list of int);
    the free numbers `s_r`, `s_p` (CuPy (4, B)), `zeta` (list of B dd lists), `kappa` (CuPy
    (4, B)), `kap_err` (CuPy (B,)); the parts `xi_kap` = `kap_err * tm.bound(sp4, xw)` and
    `v_vol` = `fr.vol_err * abs(vol) * tm.bound(sp4, dq)`.
  - `gradient_bounds(sp4, f, g, det, d, k0, keep=None)`. With a dict: `dfs` (list of 2 TM),
    `df_bounds` (list of 2 CuPy (B,), the values of `tm.bound` of `dfs`), `det_low`,
    `g00_low`, `tr_up`.
  - `run_reference(cov, r, local, settings, args, on_batch=None) -> list[tuple]` in
    `scripts/cert_run.py`: the records of the reference center r, as `main` makes them now.
    `args` holds `order`, `order_high`, `rho_high`, `target`, `depth`, `batch`, `verbose`.
    `on_batch(batch, order)` gets the list of queue items `(lo, side, center, delta, hmax,
    depth, order)` and returns a dict or None. With a dict, `box_f` gets it as `keep`, and
    after the call the dict also holds `batch`, `fr`, `info`, `sup` (the first output of
    `sup_bound`), `a`, `hmaxs` and `deltas`. If the dict then holds `stop=True`,
    `run_reference` returns after that batch.

- [ ] **Step 1: Write the test `test_keep_same_bits` in `tests/test_box_steps.py`**

  On the first batch of the reference center 297 (the queue of `run_reference` with
  `on_batch` that returns `{'stop': True}` for the first call), run `box_f` again with
  `keep=None` on the same inputs. Assert that `f.c`, `f.rem`, `info['grad_g']` and the sup
  bound are equal bit for bit (`np.array_equal` on the host arrays, NaN at the same places) to
  those of the call with `keep`. Assert that `keep` has each key of the Interfaces block. The
  module has a `main()` that runs each `test_*` function and prints `all box step tests pass`,
  as `tests/test_tm_host.py` does.

- [ ] **Step 2: Check the syntax, commit, push, pull on the cluster, and run the test**

  Run on a GPU node: `uv run --extra gpu --locked python -u -m tests.test_box_steps`.
  Expected: FAIL, because `run_reference` and `keep` do not exist.

- [ ] **Step 3: Add the hooks of the Interfaces block**

  `box_f` writes into `keep` only references to objects that it makes anyway. The parts
  `xi_kap` and `v_vol` are the terms that the code adds to the remainders now; compute each
  one once and use the same array in the remainder and in `keep`, so that the order of the
  FP64 operations does not change. `main` of `cert_run.py` calls `run_reference` for each
  reference center and keeps its output, its file format and its prints.

- [ ] **Step 4: Run the test and the two bit checks**

  Run `uv run --extra gpu --locked python -u -m tests.test_box_steps`: expected `all box
  step tests pass`. Run `sbatch submit_gpu_tests.sh` and `sbatch submit_equiv_all.sh`:
  expected `RESULT: PASS` in both.

- [ ] **Step 5: Commit**

  `git add quartic/cert/box.py scripts/cert_run.py tests/test_box_steps.py`. The message
  names the jobs of Step 4.

### Task 2: The reference of steps 1 to 9

**Files:**
- Create: `tests/box_reference.py`
- Create: `tests/test_box_reference.py`
- Modify: `tests/test_cert_imports.py` (one test of the import rule)

**Interfaces:**
- Produces (all in `tests/box_reference.py`, all values acb at 320 bits):
  - `load(data_path, forms_path) -> Potential`, a dataclass with `k0`, `top`, `base`, `levels`,
    read from the two files by Definitions "the base p0" and "the level forms N_L".
  - `exact(limbs) -> acb`: the exact sum of 4 (dd) or 6 (td) limbs, in the layouts of
    `tmgpu.tm` (re_hi, re_lo, im_hi, im_lo) and `tmgpu.tmtd` (re0, re1, re2, im0, im1, im2).
  - `poly_at(coefs, mons, x) -> acb`: the polynomial with the exact coefficients `coefs`
    (limbs, S) and the exponents `mons` (S, nvars) at the point x (list of acb).
  - `Box`, a dataclass with the fields `p`, `e`, `n`, `c1inv`, `vol`, `s_r`, `s_p`, `zeta`,
    `kappa`, `c_r`, `x0`, `s2` (exact acb, or lists of them) and `k0`, `top` (int). The
    constructor takes them directly; `Box.from_npz(z)` takes them from an export file of
    Task 3.
  - `ppow(x: acb, a: Fraction) -> acb` and `plog(x: acb) -> acb`: the principal power
    exp(a log x) and the principal logarithm. `step_values` takes every power and logarithm
    from these two functions.
  - `solve_h(box, tau, h0, rad=1e-3) -> acb`: the unique root of P(p + E τ + h n) = 0 in the
    disc abs(h − h0) ≤ rad, by Krawczyk steps in arb until the radius stops to fall; it raises
    `ValueError` if a step does not prove the root or if the final radius is not below 1e-80.
  - `polar_point(box, tau, sigma, h_tau, h_conj_sigma) -> tuple[list, list]`: z = (y(τ), 1)
    and zb = (ȳ(σ), 1), with ȳ(σ) = conj(y(conj(σ))) and the two roots of `solve_h`.
  - `forms_polar(pot, z, zb) -> tuple[acb, dict]`: p̃₀ and {L: Ñ_L}, with abs(w^r)² replaced
    by w^r wb^r and u^T C conj(u) by u(w)^T C u(wb).
  - `step_values(pot, box, z, zb) -> dict[str, acb]`: the keys `delta`, `p0`, `r2`, `ir2`,
    `log_r2`, `pt`, `xh`, `log_pt`, `xw`, `xi`, `fk0` to `fk<top>`, `psi`, `phi`, by the
    formulas of the table of the spec (steps 2 to 9), with principal branches.

- [ ] **Step 1: Write the tests in `tests/test_box_reference.py`**

  * `test_slice_equals_global`: at 3 points of X (z₃ = 1, ρ = 0.9, 0.4, 0.05), with zb =
    conj(z), `forms_polar` overlaps p₀ and each N_L of `global_values` of
    `scripts/cert_local_values.py` (balls that overlap, radius below 1e-60 of the value).
  * `test_hermitian`: at 3 points off the slice, `forms_polar(pot, z, zb)` equals the
    conjugate of `forms_polar(pot, conj(zb), conj(z))` within the ball radii.
  * `test_identity_fk`: at one point off the slice, Σ_k `fk<k>` ξ^k overlaps `psi`.
  * `test_principal_branch`: with x = 0.9 exp(0.4 i), `ppow(x, Fraction(-1, 16))` holds the
    principal x^(−1/16) of mpmath at 300 digits, and `plog(x)` the principal logarithm; the
    same with x = 0.9 exp(−3.1 i), near the cut.
  * `test_solve_h`: at 4 random τ with abs(τ_i) ≤ 0.02 of a box of the cover, P of the point
    of `solve_h` holds 0, the radius of h is below 1e-80, and a start value off by 10 times
    `rad` raises `ValueError`.
  * In `tests/test_cert_imports.py`, `test_reference_independent`: a subprocess that imports
    `tests.box_reference` has no module `quartic.cert` and no module `tmgpu` in
    `sys.modules`.

  The tests that need a box take its frame from `quartic.cert.frame.frame` in the test (not
  in the reference module), and put it into a `Box` by its constructor.

- [ ] **Step 2: Run the tests on the cluster**

  Run `uv run --extra gpu --locked python -u -m tests.test_box_reference` on a GPU node.
  Expected: FAIL, because `tests/box_reference.py` does not exist.

- [ ] **Step 3: Write `tests/box_reference.py` with the Interfaces block**

  The loader follows `setup` of `scripts/cert_local_values.py`, but the module has its own
  code. F̃_k = Σ_L C(L, k) x_c^L Ñ_L with x_c = x₀ (1 + ζ); ψ̃ = Σ_L Ñ_L x^L with
  x = r₂⁻¹ s_p^(1/k₀) (s_p p̃)^(−1/k₀); κ = s_p^(1/k₀) / x_c exact, and ξ = κ x_w − 1.

- [ ] **Step 4: Run the tests**

  Run the command of Step 2 and `uv run --extra gpu --locked python -u -m
  tests.test_cert_imports`. Expected: `all ... tests pass` in both.

- [ ] **Step 5: Commit**

  `git add tests/box_reference.py tests/test_box_reference.py tests/test_cert_imports.py`.

### Task 3: The export stage

**Files:**
- Create: `scripts/cert_steps.py` (the stage `export`; Task 4 adds `compare`)

**Interfaces:**
- Consumes: `run_reference` and `keep` of Task 1.
- Produces:
  - `select_boxes(rec) -> dict[str, int]`: the rows of the records for B1 (the largest
    `bound`), B2 (the largest `grad_g`), B3 (the largest `bound` of the reference 0), B4 (the
    largest `bound` of the reference 243), B5 (the largest `bound` of depth 1), B6 (the first
    row of the reference 297), B7 and B8 (two other rows from
    `np.random.default_rng(20261019)`).
  - The command `python scripts/cert_steps.py export --run R --cover C --local L --data D
    --forms F --manifest M --boxes B1 ... --factors 1 1e-3 1e-9 1.5 --out DIR`.
  - One file `DIR/<name>_x<factor>.npz` for each box and factor (for example `B1_x1.npz`,
    `B1_x0.001.npz`), with these keys:
    * for each model `<m>` (`h`, `y0`, `y1`, `y2`, `delta`, `p0`, `r2`, `ir2`, `log_r2`, `pt`,
      `xh`, `log_pt`, `xw`, `xi`, `fk<k>`, `psi`, `phi`, `g<i><j>`, `det`, `q4`, `v`, `f`,
      `df<i>`): `m_<m>_c` (limbs, S), `m_<m>_rem`, `m_<m>_delta`, `m_<m>_mons` (S, nvars),
      and `m_<m>_part_<key>` for each part of `.parts` of that model;
    * `h_parts_<key>` for `residual`, `host`, `lam0`, `rho`, and `rho_host` and
      `rho_residual`, the shares of ρ: ρ less ρ computed again with that term 0;
    * `psi_part_<key>` (the parts of `info['psi_parts']`), `xi_kap`, `v_vol`;
    * the frame `fr_p`, `fr_e`, `fr_n`, `fr_c1inv`, `fr_vol`, `fr_vol_err`; `s_r`, `s_p`,
      `zeta`, `kappa`, `kap_err`; `c_r`, `x0`, `s2`, `k0`, `top`;
    * `d`, `d_f`, `d_phi`, `order`, `ny`, `a`, `hmax`, `factor`, `record` (the row), `sup`,
      `grad_tau`, `grad_g`, `lam_low`, `det_low`, `g00_low`, `tr_up`, `df_bound0`,
      `df_bound1`;
    * `sha_data`, `sha_forms`, `sha_local`, `sha_run`, `sha_cover`, and `commit`.

- [ ] **Step 1: Write the checks of the export into `export`**

  At the factor 1: the sliced `fr_p` of the test box equals the center of its record cell
  (`cover.center` of the exact corner), and `sup` and `grad_g` equal the record bit for bit.
  The replay finds exactly one batch that holds the queue item with the corner, side, depth
  and order of the record, and its `on_batch` sets `stop=True` there; zero or two matches
  raise `SystemExit`. At a scaled factor: the whole batch runs again with `deltas * factor`
  and the same `hmaxs`, and only the test box is written.

- [ ] **Step 2: Run the export of B1 at the factor 1**

  Run on one H200: `uv run --extra gpu --locked python -u scripts/cert_steps.py export ...
  --boxes B1 --factors 1 --out runs_tm/steps_b1`. Expected: one file `B1_x1.npz`, and the
  print `B1: sup and grad_g equal the record bit for bit`.

- [ ] **Step 3: Commit**

  `git add scripts/cert_steps.py`. The message names the job of Step 2.

### Task 4: The compare stage for steps 1 to 9

**Files:**
- Modify: `scripts/cert_steps.py` (the stage `compare`)
- Create: `tests/test_cert_steps.py`

**Interfaces:**
- Consumes: the export files of Task 3, `tests/box_reference.py` of Task 2.
- Produces:
  - `Model`, a dataclass in `scripts/cert_steps.py` with `c` (limbs, S), `rem` (float),
    `delta` (float), `mons` (S, nvars) and `parts` (dict of float); `load_models(z) ->
    dict[str, Model]` from the keys `m_<m>_*` of an export file.
  - `check_point(model: Model, x: list, ref: acb, factor: float) -> dict` with `gap`,
    `ratio`, `r2` (dict: part → value) and `status` (`pass`, `fail` or `no claim`).
  - The command `python scripts/cert_steps.py compare --steps DIR --data D --forms F
    --points 16 --procs 16`. It writes `DIR/report.txt` and `DIR/compare.npz` (for each file,
    model and point: the gap, R, the reference radius, and for each part the value of R2), and
    prints the report. The exit status is 1 if R1 fails.

- [ ] **Step 1: Write the tests of the comparison in `tests/test_cert_steps.py`**

  These tests use small `Model` objects that the test makes, and `exact` and `poly_at` of
  `tests/box_reference.py`.

  * `test_point_moved`: for the model y₀ = c + τ₀ (exact, remainder 1e-40, factor 1e-9), a
    reference at τ₀ + 1e-25 gives `fail`, and a reference at τ₀ gives `pass`.
  * `test_no_claim`: a model with an infinite remainder gives `fail` at the factor 1 and `no
    claim` at the factor 1e-3; a NaN remainder gives `fail` at each factor.
  * `test_wide_reference`: a reference ball with a radius of 1e-2 R raises `ValueError`.
  * `test_part_seen`: a model with R = 1.0, a part 0.6 and a gap of 0.5 gives the R2 value
    0.5 / 0.4 = 1.25 for that part.

- [ ] **Step 2: Run the tests; expected FAIL, because `check_point` does not exist**

- [ ] **Step 3: Write `compare` and `check_point`**

  The points of each file come from `np.random.default_rng(20261019)` and the name of the
  file. The models in 4 variables take 16 points on D₄(d_φ): the even points have each
  coordinate at 0.999 d_φ with a random phase, the odd points a random modulus below d_φ.
  The points 0, 4, 8 and 12 lie on the slice σ = conj(τ); the others do not. The models h
  and y_k take τ of each point. The gap is the upper bound of abs(P(x) − ref). For R2, the
  value for a part π is gap / (R − π), and `inf` when R − π ≤ 0. The parts are those of
  `m_<m>_part_<key>` (`p0`, `fk<k>`), `psi_part_<key>`, `rho_host` and `rho_residual` (h),
  `xi_kap` (xi) and `v_vol` (v). The report gives, for each file and model, the largest
  gap / R, and for each part the largest R2 value with `seen` or `not seen`, and the time of
  the reference for each point and step.

- [ ] **Step 4: Run the tests, then `compare` on `runs_tm/steps_b1`**

  Run the tests: expected PASS. Run on `mweber_compute` with `-c 16`: `uv run --extra gpu
  --locked python -u scripts/cert_steps.py compare --steps runs_tm/steps_b1 ... --points 16
  --procs 16`. Expected: exit status 0, and a report with a row for each model of steps 1 to
  9.

- [ ] **Step 5: Commit**

  `git add scripts/cert_steps.py tests/test_cert_steps.py`. The message names the job and
  gives the largest gap / R.

### Task 5: The measurement on B1, and the decision of the user

**Files:** none.

- [ ] **Step 1: Make the table of the cost and the benefit**

  From the report of Task 4 and the times of the two jobs: the GPU seconds of the export of
  one box and factor, the CPU seconds of the reference for each point and step, the largest
  gap / R for each model, and the parts that the test sees and does not see. Add the
  estimate of the full check: 8 boxes, 4 factors, 16 points.

- [ ] **Step 2: Give the table to the user, and stop**

  The work continues with Task 6 only after the user accepts the table, or after a change of
  the plan that the user gives.

### Task 6: The host tests of the frame and of κ (steps 13 and 14)

**Files:**
- Modify: `tests/test_tm_host.py` (`check_frame`, `check_kappa`, and both in `main`)

**Interfaces:**
- Consumes: `quartic.cert.frame.frame` and `quartic.cert.frame.kappa`.

- [ ] **Step 1: Write `check_frame` and `check_kappa`**

  * `check_frame`: for 64 random dd centers t with abs(t₁) ≤ abs(t₀) ≤ 1, and the 332
    reference centers of `runs_tm/cover_s0.03_c6_v2.npz`, take the dd numbers p, E, n and
    c1inv of `frame(t, 16)` as exact, and compute in mpmath at `mp.dps = 300`: ha₀ =
    c1inv (P(p)), ha_i = c1inv (∇P · E_i), 6 p_k², 4 p_k, κ = 1 − c1inv (∇P · n), and
    vol = 1 / (abs(det [E₀, E₁, n])² v̄) with v̄ = 512 π⁴ k₀² / Γ(1/4)⁸. Assert that each td
    number lies within its entry of `h_err_td` of the mpmath value, and that `vol` lies within
    `vol_err` times the mpmath value. Then assert that with each error bound set to 0, at
    least one center fails; the numbers that are exact in td (each entry of `a4_td`) are
    exempt, and the test prints their list.
  * `check_kappa`: for 64 random inputs (s real in (1e-10, 1e10), k₀ = 16, x₀ real in (0.1,
    10), ζ complex with abs(ζ) < 1e-3, each as exact dd numbers), `kappa` lies within its
    error bound of s^(1/k₀) / (x₀ (1 + ζ)) in mpmath at 300 digits, and with the bound set to
    0 at least one input fails.

- [ ] **Step 2: Run on a GPU node; expected `all host tests pass`**

  Run `uv run --extra gpu --locked python -u -m tests.test_tm_host`.

- [ ] **Step 3: Commit**

  `git add tests/test_tm_host.py`.

### Task 7: Steps 10 to 12 with `tm_box.py`

**Files:**
- Modify: `scripts/tm_box.py` (new keys of the result of `run_box`)
- Modify: `tests/box_reference.py` (`metric_values`)
- Modify: `scripts/cert_steps.py` (the slice points of steps 10 to 12 and R3)
- Modify: `tests/test_box_reference.py` (`test_tm_box_orders`)

**Interfaces:**
- Produces:
  - `run_box` also returns `g_center` (dict (a, b) → acb, the constant coefficient of each
    g[a, b]), `f_acb` (the constant coefficient of f) and `df_center` (list of 2 acb, the
    coefficients of s₀ and s₁ of f). Its other keys do not change.
  - `metric_values(box, z, tb) -> dict[str, acb]` in `tests/box_reference.py`, with `tb` the
    module `tm_box` after `tm_box.load` at `prec = 320`. It puts the exact point z into
    `tb.D` (patch [2, 0, 1]), runs `run_box` at order 2 with radius 1e-4, and returns `g00`,
    `g01`, `g10`, `g11` (g_τ = Jᵀ g_t conj(J)), `det`, `q4` (abs(4 Σ_k n_k y_k³)²), `v`
    (f + 1), `f`, `df0`, `df1` (∂f/∂τ_i = Σ_a ∂f/∂t_a J[a][i]), `tr`, `lam_min` and `grad_G`
    (sqrt((4/c) (∂_t f)ᴴ g_t⁻¹ ∂_t f), c = 1 / (sqrt(2) π k₀)). J[a][i] = δ_ai + n_a ∂_i h,
    ∂_i h = −(∇P · E_i) / (∇P · n) at y.

- [ ] **Step 1: Measure the time of `run_box` at order 2 and 3 for one point**

  Run one job on `mweber_compute` with `-c 1`. Record the seconds in the commit message of
  Step 5.

- [ ] **Step 2: Write `test_tm_box_orders`**

  At one point of B1, `run_box` at order 2 and at order 3 give balls that overlap for
  `f_acb`, for each `g_center` and for each `df_center`. This shows that the coefficients of
  degree 0 and 1 do not depend on the truncation.

- [ ] **Step 3: Add the keys of `run_box`, `metric_values`, and the points of steps 10 to 12**

  The slice points of g, det, q4, v and f lie in abs(τ_i) ≤ d_f, those of `df<i>` and R3 in
  abs(τ_i) ≤ d; 8 points for each file. R3 tests each relation of the spec and gives each
  ratio: abs(`df<i>`) against `df_bound<i>`, det against `det_low`, Re g₀₀ against
  `g00_low`, tr against `tr_up`, `lam_min` against `lam_low`, `grad_G` against `grad_g`. A
  bound that is infinite counts as "no claim", as in R1. The exit status is 1 if R1 or R3
  fails.

- [ ] **Step 4: Run the test, and `compare` on `runs_tm/steps_b1`**

  Expected: the test passes; `compare` exits with status 0, and the report has rows for steps
  10 to 12 and the ratios of R3.

- [ ] **Step 5: Commit**

  `git add scripts/tm_box.py tests/box_reference.py scripts/cert_steps.py
  tests/test_box_reference.py`. The message gives the times of Step 1.

### Task 8: The full check

**Files:** none (the outputs go into `runs_tm/steps_full`, on the cluster).

- [ ] **Step 1: Run the export of all boxes and factors**

  `scripts/cert_steps.py export --boxes B1 B2 B3 B4 B5 B6 B7 B8 --factors 1 1e-3 1e-9 1.5
  --out runs_tm/steps_full` on one H200. Expected: 32 files, or fewer at the factor 1.5 when a
  bound is not finite (the export prints each such box), and the bit-for-bit print for each
  box at the factor 1.

- [ ] **Step 2: Run `compare` on `runs_tm/steps_full`**

  On `mweber_compute` with `-c 16` and `--points 16`. Expected: exit status 0.

- [ ] **Step 3: Give the report to the user**

  For each step: the largest gap / R, the parts that the tests see and do not see, and the
  ratios of R3. The user decides about each part that the tests do not see. If R1 or R3
  fails, follow "If a test fails" of the spec, and stop.

### Task 9: The unit test for each run of the test suites

**Files:**
- Modify: `tests/test_box_steps.py` (`test_steps_b6`)
- Modify on the cluster (untracked): `submit_gpu_tests.sh`; keep the old form as
  `submit_gpu_tests.sh.bak-2026-10-19`

- [ ] **Step 1: Write `test_steps_b6`**

  The box B6 at the factor 1 in its batch of the run, 4 points, steps 1 to 9, through
  `check_point` of `scripts/cert_steps.py`. Assert R1 at each point. Assert that the test
  sees one part at B6: the part with the largest R2 value at B6 in the report of Task 8.

- [ ] **Step 2: Add `tests.test_box_reference`, `tests.test_cert_steps` and
  `tests.test_box_steps` to the loop of `submit_gpu_tests.sh`, and set `#SBATCH -t 90`**

- [ ] **Step 3: Run `sbatch submit_gpu_tests.sh`**

  Expected: `RESULT: PASS`, and `tests.test_box_steps` in less than 15 minutes.

- [ ] **Step 4: Commit, and merge into `main`**

  `git add tests/test_box_steps.py`. Then `git checkout main`, `git merge --ff-only
  box-step-tests`, `git push`, and `git checkout main` on the cluster.

### Task 10: The documents

**Files:**
- Modify (untracked): `notes/strict_bound_proof.tex` and its PDF
- Modify (ignored by git): `CLAUDE.md`

- [ ] **Step 1: Wait for the new full run, and for the user to accept its report**

  The user decided on 2026-10-20 that the dd kernels of `tmgpu.tm` get a posteriori bounds
  in the style of Arb, as a separate project after Task 9. That project ends with a new full
  run and Tasks 8 and 9 again on it. This task writes the documents once, for that run.

- [ ] **Step 2: Add a Check line to each lemma that the tests cover**

  Lemmas "the frame", "the contraction", "the base", "the level sum", "the metric and f" and
  "the gradient bound": the step, the boxes, the largest gap / R, and the job. Compile with
  pdflatex three times in the scratchpad, then copy the .tex and the .pdf to `notes/`.

- [ ] **Step 3: Replace the statements of `CLAUDE.md` that the results change**

  Add the tests to "The code of the strict bound", with the jobs and the largest ratios.
  Replace old statements; do not add corrections below them. `CLAUDE.md` is in `.gitignore`,
  so it has no commit.
