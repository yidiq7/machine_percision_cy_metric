# Round-off bounds in the style of Arb, with no SAFE constant

Date: 2026-10-23. Item 4 of "Open work" in `CLAUDE.md`.

## Intent

The strict bound of f6t (sup abs(f) ≤ 4.9984e-20) is correct now, but some of its round-off
bounds rest on a count of operations or on a condition of the inputs:

* each FP64 bound sum or product rounds to nearest and then takes the factor SAFE = 1 + 1e-9,
  which is strict only for fewer than 1e6 operations;
* the dd kernels use a priori constants (7u²(1 + 3u) for `dd_mul`, 3.1u² for `dd_add`,
  (6.2c + 3.9)u² for a sum of c products), which need normalized dd inputs;
* the td kernels sum their running bounds in round to nearest and take ERR_SAFE = 1 + 1e-9;
* TINY = 1e-290 covers underflow with the count "fewer than 1e9 products with cofactors
  below 1e6";
* the host conversions take the factors 1 ± 1e-12 (and 1 + 1e-9 in two places).

The user's decision (2026-10-20 and 2026-10-21): use the rule of Arb everywhere, and remove
the SAFE constant everywhere. Then each bound holds operation by operation, and no global count
or condition can fail. This is the safer method; it is not a fix of a known error.

## The arithmetic contract (binding)

1. **Midpoints.** Each midpoint operation keeps its present FP64 algorithm, with the same
   intrinsics in the same order (`--fmad=false`). The one exception: `quick_two_sum` in
   `dd_mul` becomes `two_sum`. When the precondition abs(a) ≥ abs(b) holds, the two give the same
   bits (inventory, section 2), so the midpoints of the run do not change.
2. **The bound of one rounded operation.** For a result r = RN(x) of one FP64 operation that is
   not error-free:
   * a sum or a difference: abs(x − r) ≤ u·abs(r), u = 2^-53 (a sum with a subnormal result
     is exact);
   * a product, an FMA, a division or a square root: abs(x − r) ≤ u·abs(r) + η, η = 2^-1074.
   * An error-free pair (`two_sum`; p = RN(ab) with e = fma(a, b, −p)) adds 0, except that
     the pair (p, e) adds η, because e can underflow.
3. **Bound arithmetic.** Every FP64 operation on a bound (a radius, a remainder, a sum of
   magnitudes, a product of bounds, a power of δ) rounds toward +∞. Every lower bound rounds
   toward −∞. On the GPU this uses the intrinsics `__dadd_ru`, `__dmul_ru`, `__fma_ru`,
   `__ddiv_ru`, `__dsqrt_ru` and their `_rd` forms. On the host it uses exact arithmetic
   (`Fraction` or arb) and one directed conversion to FP64 at the end.
4. **Directed conversion on the host.** A float g is an upper bound of an exact number x only
   after an exact comparison: g = float(x), then g = nextafter(g, +∞) while g < x in exact
   arithmetic. The same with −∞ for a lower bound. `float()` of an arb is not trusted to round in
   a known direction (inventory, F6).
5. **No constants.** No SAFE, ERR_SAFE, TINY, U2, a priori constant of `dd_mul`, `dd_add` or of
   a GEMM, normalization condition, or factor 1 ± 1e-12, 1 ± 1e-15, 1 + 8e-16, 1 + 1e-9 or
   1 − 1e-12 of a sum remains in the code of the bound. A floor such as 1e-300 or 1e-60 also
   goes away: directed arithmetic covers underflow by itself.
6. **Overflow and NaN.** An infinite or NaN bound stays infinite or NaN, and every check keeps
   its rule that such a box fails.

## Goals

* `tmgpu` v0.2.0 implements the contract in all its modules (`cuda`, `tm`, `tmtd`, `gemm`,
  `tools`, `host`), and adds a general module `tmgpu.mag` for directed bound arithmetic on the
  GPU.
* `quartic/cert/box.py`, `frame.py`, `cover.py` and `forms.py` implement the contract in all
  their bounds.
* The tests of both repositories cover the new bounds, with a test that sees each kind of rule
  (a planted change of u or of the direction of rounding must make a test fail).
* Later, when the user allows runs: the bits of the midpoints equal those of the run, the new
  bounds compare box by box with the old ones, then a new full run and the tests again.

## Non-goals

* No change of a midpoint algorithm (except `quick_two_sum`), of an order, of a radius rule, of
  the cover cells, or of the choice of boxes.
* No change of the truncation bounds (tails, Lagrange rests, Cauchy estimates) except that
  their FP64 arithmetic rounds up.
* No new speed work. A slower kernel is acceptable if it stays below 2 times the old time; the
  report gives the measured factor.
* The documents (`notes/strict_bound_proof.tex`, Task 10 of the test plan, `CLAUDE.md`) wait
  for the new run.

## Anti-goals

* A bound that is smaller than the true error in any case, for example a sum with round to
  nearest that the contract does not allow.
* A change of the midpoint bits with no reason.
* Any project information in `tmgpu` (it must stay general, for a public release).

## The decisions (my calls; the user did not take part in them)

### tmgpu

* **`tmgpu.cuda`.** The source adds device helpers for the contract:
  `err_sum(E, r)` adds u·abs(r) upward; `err_prod(E, r)` adds u·abs(r) + η upward. `dd_add`
  and `dd_mul` get forms with an error output (for example `dd_add_e(..., double &E)`) that do
  the same midpoint operations and add the bound of rule 2: `dd_add` adds
  u(abs(v₁) + abs(v₂)); `dd_mul` adds u(abs(t₁) + abs(t₂) + abs(t₃) + abs(v)) + 4η. The
  module docstring states the contract.
* **`tmgpu.mag`** (new, general): directed arithmetic of non-negative FP64 arrays on the GPU,
  with CuPy `ElementwiseKernel`, `ReductionKernel` or `RawKernel` and the `_ru` and `_rd`
  intrinsics. At least: `add`, `mul`, `fma`, `div`, `sqrt`, `powi` (integer power by
  products), `sum` along an axis, `matmul` (batched, with broadcast, non-negative inputs,
  each output a chain of `__fma_ru`), `horner` (Σ g_d δ^d by `__fma_ru`), `conv_tail`
  (Σ over k + l > n of a_k b_l δ^(k+l)), and the lower forms `sub_down`, `mul_down`,
  `div_down`, `sqrt_down`. A sum of non-negative terms with each partial sum rounded up is an
  upper bound of the exact sum, so the order of the sum does not matter.
* **`tmgpu.tm`.** Each kernel (`tm_mul`, `c_axpy`, `c_scale`, `c_mulint`, `c_add0`) returns an
  error array Err with one bound for each output coefficient (abs(Δre) + abs(Δim)). Each
  operation (`add`, `sub`, `scale`, `add_const`, `mul`, `tensor`, `deriv1`, `cauchy_mixed`,
  `series`, `power_series`, `log1p_series`, `truncate`) takes its round-off part from Err,
  weighted by δ^deg through `mag`, and forms its remainder with `mag`. `graded`,
  `poly_bound_g`, `poly_bound`, `bound`, `tail_g` and `abs_ub` use `mag`. `U2`, `SAFE`,
  `TINY`, `gamma_deg` and the table of counts go away. The Lagrange rests use integer powers:
  for p < 0, (1 − e)^p ≤ (1 − e)^⌊p⌋ with 1 − e rounded down; for p ≥ 0, (1 + e)^⌈p⌉ rounded
  up; abs(C(a, k + 1)) from its exact Fraction, rounded up. The gap of `deriv1` and
  `cauchy_mixed` rounds down. `cconst` also returns the exact residual of each coefficient,
  rounded up, and `series` uses it in place of 2u² abs(q). The keys of `mul(...).parts`
  (tail, round, bx, by) stay.
* **`tmgpu.tmtd`.** The running bounds of `acc_tt`, `levels_dd`, `tmt_mul`, `tmt_axpy` and
  `tmt_scale` sum upward and add η for each rounded product and each FMA pair. `ERR_SAFE` goes
  away. `to_dd` keeps its 6 midpoint operations and bounds u·abs(lo) upward. `abs_ub`,
  `graded`, `bound`, `err_bound` use `mag`.
* **`tmgpu.gemm`.** `zgemm_tt` returns its kernel bound with no factor. `zgemm` (dd), its
  kernel and `gamma` go away, because the run does not use them (the user, 2026-10-24). This is
  a change of the API, so the version is 0.2.0.
* **`tmgpu.tools`.** `sup_bound` sums upward with `mag`. `series_order` does not change.
* **`tmgpu.host`.** `upper`, `lower`, `to_dd`, `to_td` and `frac_dd` follow rule 4: the
  endpoint of the ball (or the exact Fraction residual) is converted with an exact comparison.
  The sum of the real and the imaginary residual is formed in arb (or as Fractions) and
  converted once. `lower` gives 0 only when the lower endpoint is ≤ 0. New helpers: `up(x)`
  and `down(x)` for arb and Fraction values.
* **Tests (written now, run later).** A new `tests/test_mag.py`: each function against exact
  Fractions; a planted sum where round to nearest falls below the exact sum, which `mag.sum`
  must not do; the lower forms below the exact value. `test_tm.py` and `test_tmtd.py`: the Err
  of each kernel bounds the error against mpmath, also with inputs that are not normalized and
  with results near underflow (abs about 1e-310); with u set to 0 in the error helpers (a test
  switch of the module source), the tests of the round-off must fail. One switch check of u and
  one of η for each kernel family is enough (the user, 2026-10-24); a test that only checks that
  a call site uses the right `mag` function is not necessary.
  `test_rounding_bounds` changes from the a priori constants to the Err outputs. `test_host.py`:
  the directed conversions on values where `float()` rounds the wrong way.
* `pyproject.toml` of `tmgpu`: version 0.2.0. The tag `v0.2.0` waits until its tests pass.

### This repository

* **`quartic/cert/box.py`**: each row of the inventory with the target UP-SUM, UP-MM, DOWN or
  HOST-X changes to `mag` or to the host helpers; each `* SAFE`, `+ TINY` and factor goes away;
  `cp.power` on bounds becomes `mag.powi` or `mag.horner`; the einsum and matmul of bounds
  become `mag.matmul` or `mag.sum` with the same shapes. Also:
  * `v_vol` gets the explicit term of vol_err²: the factor vol_err / (1 − vol_err), rounded up
    (inventory, F3);
  * `_real_low` forms its lower bound with the `_rd` forms;
  * `lam_low` and `c` round down; `grad_tau` and `grad_g` round up;
  * in `implicit_h_td`, the trial radius r is a free number, and only the checks round up;
  * the copies of bound formulas follow the code: `psi_parts` (box.py:576-584) and
    `scripts/cert_steps.py` (`radius`, 210-219);
  * `kap_err` comes from frame.py with no factor.
* **`quartic/cert/frame.py`**: `vol_err` as a quotient in arb, converted up once; the errors of
  `to_dd` and `to_td` from the new host functions.
* **`quartic/cert/cover.py`**: the host arithmetic of `box_radius` (cover.py:105-116) in arb,
  converted once, with no factor.
* **`quartic/cert/forms.py`**: `to_td_complex` forms `_rad` and `_abs` in arb and converts
  them up once.
* **The check scripts** (`cert_partition`, `cert_check`, `cert_local_values`) keep their
  tolerances, which are not bounds. `cert_local_values.upper` imports `tmgpu.host.upper` in place
  of its copy.
* **Tests:** `tests/test_tm_host.py` and the tests of the box steps follow the new API; a test
  of `_real_low` with the `_rd` forms; the R4 helpers use the new errors of the frame.
* `pyproject.toml` and `uv.lock` change to the tag `v0.2.0` only in the run phase.

### The order of the work

1. Write the code and the tests in both repositories, on the branch `arb-rounding` of each.
   Run nothing (the user's rule of 2026-10-23); check the syntax with `python3 -m py_compile`.
2. Report to the user.
3. When the user allows runs: the tests of `tmgpu`; the tests of this repository with a
   temporary path source; the midpoint bits and the bounds against the old run on the 4
   references of `submit_equiv_all.sh`; the time of each kernel; the tag `v0.2.0`; new local
   forms and a new cover with the new host code; the new full run; `cert_partition`,
   `cert_check`, and the full check of the box steps again.

## What is left to the builder

* The names of the device helpers and of the `mag` functions beyond the list above, and the
  kernel layout of `mag.matmul`.
* Whether the td kernels keep their sums in units of u (and add η/u) or change to absolute
  units, as long as the contract holds.
* The test inputs, if each test can fail for the rule that it covers.
