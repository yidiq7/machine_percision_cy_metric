# Round-off bounds in the style of Arb: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**A change after the plan ran (the user, 2026-10-24):** the dd GEMM `gemm.zgemm` with its kernel and
`tests/test_gemm.py`, `mag.powi_down`, `host.up_sum`, the direction tests at the call sites, and the
switch checks beyond one of u and one of η for each kernel family were removed again (tmgpu 32863d7,
this repository 1f738d7). The tasks below keep their text as they ran.

**Goal:** Every round-off bound of the strict bound follows the rule of Arb (u·abs(result) + η for
each rounded operation, every bound operation rounded up or down), with no SAFE, ERR_SAFE, TINY,
a priori constant or host factor.

**Architecture:** `tmgpu` v0.2.0 gets device helpers for the per-operation rule, error outputs of
every kernel, and a new module `tmgpu.mag` of directed bound arithmetic on the GPU. The bound code
of `quartic/cert` then uses `mag` and the directed host conversions of `tmgpu.host`.

**Tech Stack:** Python 3.12, CuPy (RawModule, ElementwiseKernel, ReductionKernel, CUDA intrinsics
`_rn`, `_ru`, `_rd`), python-flint (arb), `fractions.Fraction`.

**Spec:** docs/superpowers/specs/2026-10-23-arb-rounding-design.md (the binding authority). The
inventory of every bound with file:line references:
`.superpowers/sdd/2026-10-23-arb-rounding/arb_inventory.md`.

## Global Constraints

* **Run nothing** (the user's rule of 2026-10-23): no cluster job, no test, no script, no `uv`.
  On the laptop, use only `python3 -m py_compile <files>`, `git` and `grep`. The tests are written
  now and run later.
* The arithmetic contract of the spec (rules 1 to 6) binds every task.
* The midpoint operations keep the same intrinsics in the same order. The one change:
  `quick_two_sum` becomes `two_sum` in `dd_mul`.
* `tmgpu` stays general: no cluster, no project, no file of a run in its code, tests or documents.
* All text (comments, docstrings, commit messages, README) in ASD-STE100 Simplified Technical
  English: no words that end in -ing, active voice, short sentences, no "should" or "may".
  Comments 1 or 2 lines, complete without other information.
* No `Co-Authored-By` or other attribution line in a commit. Stage explicit paths; never
  `git add -A`. `rm` is disabled on the laptop.
* Branches: `arb-rounding` in `/Users/yidiqi/Documents/Research/tmgpu` (from `main` at f805d84)
  and `arb-rounding` in this repository (from `main` at c92b9d9). Commit locally. Do not push,
  and do not tag.
* `pyproject.toml` and `uv.lock` of this repository do not change in this plan.

## Review Focus

1. A bound sum that still rounds to nearest (a CuPy `+`, `*`, `**`, `cp.sum`, `cp.matmul`,
   `cp.einsum` or a Python float sum on a bound path) — the reviewer must find every one.
2. A product of an upper bound with a value that can be negative in `mag.mul` or `mag.matmul`
   (an upward product is an upper bound only for non-negative factors).
3. A midpoint operation that changed order or intrinsic (the bits of the run must stay).
4. The η terms: each rounded product, FMA, division and square root, and each FMA error pair,
   adds η; a sum needs none.
5. A kernel cache that keeps an old compiled source after a test changes `cuda.ERR_U`.

## Interfaces (fixed for all tasks)

* `tmgpu.cuda`:
  * `ERR_U = 2.0 ** -53` and `ERR_ETA = 2.0 ** -1074` (module floats); `err_defines()` returns
    the CUDA lines `#define TM_U <ERR_U as a hex float>` and `#define TM_ETA <...>`.
  * `module(code)` prepends `err_defines()` and caches by (code, ERR_U, ERR_ETA), so a test that
    sets `ERR_U = 0.0` gets new kernels.
  * Device functions in `DD_PRIMITIVES`: `two_sum`, `dd_add`, `dd_mul` (with `two_sum` in place
    of `quick_two_sum`), `err_sum(double &E, double r)` (E ← RU(E + TM_U·abs(r))),
    `err_prod(double &E, double r)` (E ← RU(E + TM_U·abs(r) + TM_ETA)),
    `err_pair(double &E)` (E ← RU(E + TM_ETA)), `dd_add_e(ah, al, bh, bl, &sh, &sl, &E)` and
    `dd_mul_e(ah, al, bh, bl, &ph, &pl, &E)`, with the same midpoint operations as `dd_add` and
    `dd_mul`.
* `tmgpu.mag` (new; float64 CuPy arrays in and out; broadcast like CuPy):
  * up: `add(*xs)`, `sum(x, axis)` (any sign: an upper bound of the exact sum of the given
    floats); `mul(a, b)`, `fma(a, b, c)`, `div(a, b)`, `sqrt(a)`, `powi(x, k)`,
    `powers(x, n)` (shape (n + 1,) + x.shape), `matmul(a, b)`, `horner(g, x)` (Σ_d g[d] x^d
    along axis 0), `conv_tail(ga, gb, x, n)` (Σ over k + l > n of ga[k] gb[l] x^(k+l), axis 0
    of ga and gb), `abs_sum(c, axis)` (Σ abs(c)) — the products need non-negative inputs;
  * down: `add_down(*xs)`, `sub_down(a, b)`, `mul_down(a, b)`, `div_down(a, b)`,
    `sqrt_down(a)`, `powi_down(x, k)`.
* `tmgpu.tm`: the public functions keep their names and signatures (inventory, section 17),
  except that `U2`, `SAFE` and `TINY` go away. New: `err_bound(sp, err, delta)`
  (Σ_k err[k] δ^deg(k), upward). The kernels return Err arrays of shape (S, B) (abs(Δre) +
  abs(Δim) of each coefficient). `mul(...).parts` keeps the keys tail, round, bx, by.
* `tmgpu.tmtd`: names and signatures stay; `ERR_SAFE` goes away; `mul(..., return_err=True)`
  returns the kernel Err with no factor.
* `tmgpu.gemm`: `zgemm_tt(a, b, mode=0, out_td=True)` returns (C, Err); `zgemm(a, b, mode=0)`
  returns (C, Err); `gamma` goes away.
* `tmgpu.host`: `upper(x)`, `lower(x)` (same names, directed exact conversion; `lower` is 0 only
  when the lower endpoint is ≤ 0); new `up(x)`, `down(x)` for arb, acb (its abs), Fraction and
  int; `up_sum(values)` for a list of floats; `to_dd`, `to_td`, `frac_dd` keep their returns
  (limbs, err) with an exact upward err.

---

### Task 1: The device helpers and `tmgpu.mag`

**Files:** Modify `tmgpu/cuda.py`. Create `tmgpu/mag.py`, `tests/test_mag.py`. Modify
`tests/test_boundary.py` if it lists the modules.

- [ ] **Step 1: Create the branch** `git checkout -b arb-rounding` in the tmgpu repository.
- [ ] **Step 2: Write `tests/test_mag.py`.** For each function of `mag`: random inputs over many
  binades (1e-300 to 1e300, subnormals, zeros), the result against the exact value in Fractions:
  an up result ≥ exact, a down result ≤ exact, and each within 4 ulp of the exact value for one
  operation (within (n + 1) ulp relative for a sum of n terms). A planted sum where `cp.sum` gives
  a value below the exact sum (for example 1 followed by 1e6 terms of 2^-54), and `mag.sum` must
  be ≥ the exact sum. `matmul` against Fraction products, also with broadcast shapes.
  `conv_tail` and `horner` against Fractions. The test prints one line for each function, and the
  file ends with `main()` and `all mag tests pass`, as the other test files of tmgpu do.
- [ ] **Step 3: Write `tmgpu/cuda.py`** with the interface above. The module docstring states
  rules 2 and 3 of the spec. `module()` keeps the compiler options `--fmad=false`, `-std=c++14`.
- [ ] **Step 4: Write `tmgpu/mag.py`.** One short docstring with the contract (an upward chain of
  non-negative terms is an upper bound in any order). Use `cp.ElementwiseKernel` for the
  elementwise forms, a `RawKernel` or `ReductionKernel` with `__dadd_ru` for `sum`, a `RawKernel`
  for `matmul` (each thread one output, a loop of `__fma_ru` over the inner index; broadcast the
  batch dimensions in Python), `horner` and `conv_tail`.
- [ ] **Step 5: `python3 -m py_compile tmgpu/cuda.py tmgpu/mag.py tests/test_mag.py`.**
- [ ] **Step 6: Commit** `git add tmgpu/cuda.py tmgpu/mag.py tests/test_mag.py` (and
  `tests/test_boundary.py` if changed).

### Task 2: `tmgpu.tm` with error outputs

**Files:** Modify `tmgpu/tm.py`, `tests/test_tm.py`.

- [ ] **Step 1: Kernels.** `tm_mul`, `c_axpy`, `c_scale`, `c_mulint`, `c_add0` write Err (S, B)
  with `dd_mul_e`, `dd_add_e` and `err_sum`; each midpoint operation stays as it is.
- [ ] **Step 2: Bound functions** (inventory rows TM-1 to TM-20): `abs_ub`, `graded`,
  `poly_bound_g`, `poly_bound`, `bound`, `tail_g`, `err_bound`, `_axpy`, `scale`, `add_const`,
  `mul`, `series`, `power_series`, `log1p_series`, `cauchy_mixed`, `deriv1`, `tensor`,
  `truncate` with `mag`; `cconst` returns the residual of each coefficient (exact, rounded up)
  for `series`. Remove `U2`, `SAFE`, `TINY`, `gamma_deg` and the count table. The Lagrange rests
  as in the spec. The module docstring describes the new bounds; no a priori constant remains in
  it.
- [ ] **Step 3: Tests.** `test_rounding_bounds` checks the Err of each kernel against mpmath with
  no truncation, for normal, reversed and equal limb orders and for results near 1e-310; with
  `cuda.ERR_U = 0.0` the check must fail (restore it after). Add the branch p ≥ 0 of
  `power_series` to `test_series`. Keep the other tests and their names.
- [ ] **Step 4: `python3 -m py_compile tmgpu/tm.py tests/test_tm.py`**, then commit the 2 files.

### Task 3: `tmgpu.tmtd` and `tmgpu.gemm`

**Files:** Modify `tmgpu/tmtd.py`, `tmgpu/gemm.py`, `tests/test_tmtd.py`, `tests/test_gemm.py`.

- [ ] **Step 1:** The running bounds of `acc_tt`, `levels_dd`, `tmt_mul`, `tmt_axpy`,
  `tmt_scale` and `zgemm_tt` sum upward and add η for each rounded product, FMA and FMA pair;
  the dropped products are bounded upward. `ERR_SAFE` goes away. `to_dd`, `abs_ub`, `graded`,
  `bound`, `err_bound`, `_axpy`, `scale`, `mul` use `mag` (inventory rows TD-1 to TD-10).
- [ ] **Step 2:** `zgemm_dd` and `zgemm` get the running bound with `dd_mul_e`, `dd_add_e`;
  `zgemm` returns (C, Err); `gamma` goes away. The docstrings follow.
- [ ] **Step 3: Tests.** `check_tm_ops` and `check_gemm` keep their checks and add results near
  underflow and the switch `cuda.ERR_U = 0.0` (must fail). `tests/test_gemm.py` checks the Err of
  `zgemm` against mpmath in place of `gamma`.
- [ ] **Step 4: `python3 -m py_compile`** on the 4 files, then commit them.

### Task 4: `tmgpu.tools`, `tmgpu.host`, the README and the version

**Files:** Modify `tmgpu/tools.py`, `tmgpu/host.py`, `tests/test_host.py`, `README.md`,
`pyproject.toml` (version 0.2.0 only), `tmgpu/__init__.py` if it lists modules or a version.

- [ ] **Step 1:** `sup_bound` with `mag` (inventory TL-1). `host` as in the interfaces (HO-1 to
  HO-7). No factor and no floor remain.
- [ ] **Step 2: Tests.** `test_host.py`: `up`, `down`, `upper`, `lower`, `up_sum`, `to_dd`, `to_td`,
  `frac_dd` against exact values, on balls whose midpoints round the wrong way under `float()`,
  on exact values, and on values near the limits of FP64.
- [ ] **Step 3:** The README describes the contract and the module `mag`, in STE and with no
  project information. `grep -rn "SAFE\|TINY\|U2\|1e-12\|1e-9\|gamma" tmgpu tests README.md` must
  find no use on a bound path (a comment that says the constants are gone is acceptable only in
  the README history line, if any).
- [ ] **Step 4: `python3 -m py_compile`**, then commit.

### Task 5: `quartic/cert/box.py` and its copies

**Files:** Modify `quartic/cert/box.py`, `scripts/cert_steps.py` (the copy of the radius of
h), and the tests that read the changed parts: `tests/test_tm_host.py` (`check_real_low`),
`tests/test_cert_steps.py`, `tests/test_box_steps.py`, `tests/box_reference.py` only if an
interface changes.

- [ ] **Step 1: Create the branch** `git checkout -b arb-rounding` in this repository.
- [ ] **Step 2:** Each row BX-01 to BX-85 of the inventory with target UP-SUM, UP-MM, DOWN or
  HOST-X changes as the spec says; each `* SAFE`, `+ TINY`, `cp.power` on a bound and factor goes
  away; the einsum and matmul of bounds become `mag` calls with the same shapes. The midpoint code
  does not change.
- [ ] **Step 3:** The rows of the spec: `v_vol` with vol_err / (1 − vol_err) rounded up;
  `_real_low` with the down forms; `lam_low`, `c` down; `grad_tau`, `grad_g` up; the trial radius
  of `implicit_h_td` free and its checks up; `psi_parts` and the `radius` copy in
  `scripts/cert_steps.py` follow the new formulas.
- [ ] **Step 4:** `check_real_low` of `tests/test_tm_host.py` tests the new `_real_low`, and a
  planted round to nearest must fail it. `tests/test_cert_steps.py` follows any changed key.
- [ ] **Step 5: `python3 -m py_compile`** on each changed file, then commit.

### Task 6: frame, cover, forms and the check scripts

**Files:** Modify `quartic/cert/frame.py`, `quartic/cert/cover.py`, `quartic/cert/forms.py`,
`scripts/cert_local_values.py`, `tests/host_reference.py` and `tests/test_tm_host.py` if the
errors of the frame change their form.

- [ ] **Step 1:** frame.py (FR-2, FR-3, FR-5), cover.py (CV-05 to CV-12), forms.py (FO-4, FO-5)
  as in the spec. `cert_local_values.upper` imports `tmgpu.host.upper`.
- [ ] **Step 2:** The R4 helpers of `tests/host_reference.py` still compare with the new bounds.
- [ ] **Step 3: `python3 -m py_compile`**, then commit.

### Task 7: The report

- [ ] **Step 1:** `grep` both repositories for SAFE, ERR_SAFE, TINY, U2, `1e-12`, `1e-9`,
  `1e-15`, `8e-16`, `1e-300`, `1e-60`, `cp.power`, `cp.matmul`, `cp.einsum` and `cp.sum` on bound
  paths, and list what remains with a reason for each.
- [ ] **Step 2:** Write the list of the runs of the spec ("The order of the work", item 3) as the
  plan for the run phase, with the expected checks.
