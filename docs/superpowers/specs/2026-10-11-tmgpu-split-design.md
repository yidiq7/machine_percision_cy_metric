# Split of the Taylor-model core into the package `tmgpu`

Date: 2026-10-11. Status: design for review.

## Goal

The strict bound of sup abs(f) uses two kinds of code that are mixed today:

* a general core: Taylor models in dd and td on the GPU, with a strict bound of each
  round-off, and matrix products with bounds;
* the quartic certificate: the frame of a box, the forms of the potential, the box
  pipeline, the cover of the region F.

This work does three things:

1. It moves the general core into one folder, `tmgpu/`, that imports no quartic code
   and no JAX. A later step puts this folder into its own GitHub repository.
2. It moves the quartic certificate code into the subpackage `quartic/cert/`, which
   imports `tmgpu`.
3. It removes unused code, the superseded paths of the box, and comments that compare
   with older versions.

## Constraints

* **No numerical change.** The work moves code, changes imports and removes code that
  the run does not use. It changes no algorithm and no bound. On the same inputs, the
  new code must give the same floats, bit for bit, as `main` at commit 8f0c1e8.
* **Isolation.** The work runs in a git worktree on the branch `tm-library`, from
  `main` at 8f0c1e8. It merges into `main` only after the equivalence test passes.
* **The cluster.** No `git pull` on the cluster while a certificate job runs, because
  `ddgemm` and `tm_cert` import modules inside functions.
* **Conventions.** STE in all text. Explicit paths in `git add`. No `Co-Authored-By`
  trailer. Code comments of 1 or 2 lines, complete without other information. No
  computation on the laptop: `python3 -m py_compile` there, and tests on a GPU node.

## The layout

```
tmgpu/                      the core; imports only numpy, cupy, mpmath, flint and itself
  __init__.py
  cuda.py                   the CUDA source of the dd primitives (two_sum, dd_add, dd_mul)
                            and the kernel loader, public
  tm.py                     Taylor models in complex dd            <- quartic/tmc.py
  tmtd.py                   Taylor models in complex td            <- quartic/tmtd.py
  gemm.py                   batched dd and td matrix products with bounds
                                                                   <- quartic/ddgemm.py
  host.py                   (CPU only) Arb balls to dd or td with an error bound; exact
                            rationals and mpmath numbers to and from dd or td
                                                                   <- tm_frame.py, tm_cert.py
  tools.py                  space, sup_bound, series_order, power, conj, and CuPy dd
                            arrays to and from mpmath (host_dd, gpu_dd) <- tm_cert.py
  tests/
    test_tm.py              <- tests/test_tmc.py
    test_tmtd.py            <- tests/test_tmtd.py
    test_gemm.py            <- tests/test_ddgemm.py
    test_boundary.py        new: tmgpu imports only the modules in the list above
    test_host.py            new: check_frac_dd and check_lower of tests/test_tm_host.py
quartic/cert/               the quartic certificate; imports tmgpu
  __init__.py
  frame.py                  the frame of a box and kappa           <- quartic/tm_frame.py
  forms.py                  the forms N_L, the local forms, the Delta split
                                                                   <- quartic/tm_forms.py
  box.py                    the box pipeline (td chain) and the gradient bound
                                                                   <- quartic/tm_cert.py
  cover.py                  cell, may_meet_f, box_radius, dd       <- scripts/cert_cover.py
scripts/                    the command-line tools stay here, with new imports
tests/test_tm_host.py       stays; it tests the quartic host numbers, with new imports
```

## What moves where

| from | to | content |
|---|---|---|
| `quartic/tmc.py` | `tmgpu/tm.py` | all, except `_SRC` and the kernel loader |
| `_SRC` and `_kernels` of `quartic/tmc.py` | `tmgpu/cuda.py` | the dd primitives and the dd kernels |
| `quartic/tmtd.py` | `tmgpu/tmtd.py` | all; it takes the dd primitives from `tmgpu.cuda` |
| `quartic/ddgemm.py` | `tmgpu/gemm.py` | all, except the two unused functions below |
| `precision`, `exact`, `upper`, `lower`, `to_dd`, `to_td` of `quartic/tm_frame.py` | `tmgpu/host.py` | general ball helpers |
| `mp_dd`, `dd_mp`, `dd_frac`, `frac_dd` of `quartic/tm_cert.py` | `tmgpu/host.py` | general conversions |
| `sup_bound`, `_space`, `_series_order`, `_power`, `conj_tm`, `_host_dd`, `_to_dd` of `quartic/tm_cert.py` | `tmgpu/tools.py` | general; `series_order` takes `eps` as an argument, and the quartic box passes 1e-34; `_host_dd` and `_to_dd` become `host_dd` and `gpu_dd` |
| `frame`, `kappa` of `quartic/tm_frame.py` | `quartic/cert/frame.py` | quartic |
| `quartic/tm_forms.py` | `quartic/cert/forms.py` | quartic, except the unused functions below |
| `quartic/tm_cert.py` (the rest) | `quartic/cert/box.py` | quartic, except the removed paths below |
| `cell`, `may_meet_f`, `box_radius`, `dd` and their helpers of `scripts/cert_cover.py` | `quartic/cert/cover.py` | quartic; the script keeps only its `main` |

The plan fixes the exact list of helpers by a check of the call graph.

## What goes

| code | reason |
|---|---|
| `ddgemm.td_of_dd`, `ddgemm.tcabs`, `tm_forms.chart_forms`, `tm_forms.save_gpu`, `tm_cert.tm_const`, `tm_cert._bnorm_low` | no code uses them |
| `quartic/tm_gpu.py`, `scripts/bench_tm_gpu.py`, `tests/test_tm_gpu.py` | the XLA prototype; only its own test and benchmark use it |
| the dd chain of the box: `implicit_h`, `offset_models`, `monomial_mats`, `compose`, `reexpand`, `delta_form`, `_batched_mul`, and each helper that only they use | quartic only; the run uses the td chain |
| `tm_cert.neg`, `tm_cert.p_value`, `tm_cert.tm_const`; the dd frame constants `ha`, `a6`, `a4`, `h_err` of `tm_frame.frame` and of `Frames` | no code on the td path uses them |
| the Q_e form of the local forms (tag `e`, `mode_j` false) in the box, in `tm_local_forms.py` and in `cert_box.py` | quartic only; the Delta form replaced it |
| the option `--td` of `cert_run.py` and `cert_box.py`; the dd and Q_e modes of `tm_local_forms.py`, which then always writes the Delta form in td | they serve only the dd chain |
| `scripts/tm_ref.py` | quartic only; neither the run nor the checks use it |
| comments that compare with older versions | the convention puts them in commit messages |

The general dd matrix products (`zgemm`, `gamma`, `gamma_td`) stay in `tmgpu`, with
their tests, although the quartic code no longer uses them.

The tag `pre-tmgpu` on 8f0c1e8 keeps every removed path reachable.

## Other changes

* `quartic/__init__.py` stops its eager import of `blocks` and `sections`. Then
  `quartic.cert` loads no JAX, and the scripts drop `JAX_PLATFORMS=cpu`. The plan
  first checks that no code uses `quartic.sections` or `quartic.blocks` without an
  explicit import.
* The cert scripts import `quartic.cert.cover` and no longer put `scripts/` on
  `sys.path`. `cert_check.py` still imports `tm_box` from `scripts/`, because the
  independent reference must not share code with the box.
* The docstrings name the new modules.
* `cert_box.py` reads the six parts of the td local forms in its check of p0.
* `box_f` stops with a clear `ValueError` if the local forms are not in Delta and in td.

## Not in this work

* The fit pipeline: the JAX modules and the scripts `oracle_*`, `correction_*`,
  `vpa_*`, `lm_*` and others. The appendices of the report depend on them.
* Changes of floats: `dd_mul` in the published order, and powers without the CUDA
  `pow`. They follow after the merge, as separate commits, each with its own test.
* Changes of the interface: a Taylor model that carries its Space, real models, an
  FP64 mode, new functions.
* A `pyproject.toml` for `tmgpu`. The scripts find it through the repository root, as
  they find `quartic` today.
* The deletion of the dd local forms on netscratch (159 GB). Only the user decides it.

## Verification

1. **On the laptop:** `python3 -m py_compile` on every changed file; a search for each
   removed name; `tests/test_boundary.py` reads the imports of `tmgpu` as text.
2. **On a GPU node, after each task:** the unit tests, the paths of the imported packages,
   and cert_run.py with --grad on reference center 297, compared bit for bit with the
   records of main.
3. **Before the merge:** cert_run.py with --grad on 4 reference centers (0, 243, 297, and
   the center with the most split cells in run_f6t_v2.npz), bit for bit; the sha256 of
   the four CUDA sources against main; cert_partition.py --grad on the full run_f6t_v2.npz
   with the code of main and of the branch, with the same output; cert_check.py with
   --points 300 --dropped 300 on the branch. cert_local_values.py imports no moved code,
   so it is not part of the test.
4. **After the merge:** the tag `pre-tmgpu`, a section of `CLAUDE.md` on the new layout,
   and a pull on the cluster when no certificate job runs.
