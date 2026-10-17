# Move of `tmgpu` into its own repository

Date: 2026-10-17. Status: approved by the user on 2026-10-17.

## Goal

`tmgpu/` is a general library: Taylor models in complex dd and td on the GPU, with a
strict bound of each round-off, and batched matrix products with bounds. Today it is a
folder of this repo. This work moves it into its own private GitHub repository,
`yidiq7/tmgpu`, as a standalone package. This repo then uses `tmgpu` as an installed
dependency, only through its public interface.

The reason: the user will develop `tmgpu` as a standalone package, and will publish it on
the Python Package Index (PyPI) later. So the two code bases must be decoupled.

## The decisions of the user

* The name stays `tmgpu`. On 2026-10-17 the name was free on PyPI and on GitHub.
* The new repository is private for now. Later the user publishes the package on PyPI, so
  that any user can install it with `pip install tmgpu` or `uv pip install tmgpu`.
* Until that release, this repo installs `tmgpu` from the git repository.
* Only the user installs this repo before the PyPI release. So `tmgpu` goes into the main
  dependencies of this repo, and every install needs read access to `yidiq7/tmgpu`.
* No commit in either repository has a `Co-Authored-By` trailer or any other line of
  co-authorship. This also applies to the first commit and to the tag of `tmgpu`.
* The quartic code uses `tmgpu` only as an API (application programming interface).
* This repo trusts the release of `tmgpu` that it installs. The work adds no check of the
  installed `tmgpu` to the records, to `cert_run` or to `cert_partition`. `uv.lock`
  records the exact commit of `tmgpu`, and that is sufficient.
* The new repository starts with a clean history. Its first commit names the source
  commit 71d8297 of this repo.
* `tmgpu` holds no information about the cluster or about this project: no cluster
  paths, partitions, user names or file names of our runs. Its code and documents must
  not assume that a user runs on a cluster (the rule of 2026-10-11 and 2026-10-15).

## Goals

1. A private repository `yidiq7/tmgpu` with the package, its tests, `pyproject.toml`,
   `uv.lock`, `README.md` and the tag `v0.1.0`. A local `CLAUDE.md` stays out of git.
2. This repo depends on `tmgpu` v0.1.0 from that repository, and the folder `tmgpu/`
   goes away from this repo.
3. The bound code of this repo gives the same floats, bit for bit, as before the move.

## Non-goals

* No change to an algorithm, a bound, or a name of the interface. v0.1.0 holds the code
  of `tmgpu/` at 71d8297. The only code changes are in the tests: the import paths, and
  the name in the planted import of `test_boundary.py`.
* No release on PyPI in this work, and no choice of a license. The user must choose a
  license before the public release.
* No change to the format of the records or to the checks of the bound.
* No continuous integration service. The tests need an NVIDIA GPU.
* No change to the earlier design documents in `docs/superpowers/`. They describe the
  history.

## Anti-goals

These results make the work a failure, also if the code runs:

* `tmgpu` imports `quartic`, JAX, or any module outside its boundary, or it names the
  cluster or this project.
* The records of the reference 297 differ by one bit from `runs_tm/equiv/base_297.npz`.
* The environment of this repo imports `tmgpu` from a local folder and not from the
  installed release.

## The interface

The quartic code uses 55 public names of `tmgpu` and no name that starts with an underscore.
These names are the interface. A change to one of them needs a new version of `tmgpu`.

| module | names that the quartic code uses |
|---|---|
| `tmgpu.tm` | `TM`, `Space`, `zeros`, `const`, `cconst`, `add`, `sub`, `add_const`, `mul`, `scale`, `tensor`, `truncate`, `restrict`, `embed`, `split_index`, `powers`, `power_series`, `log1p_series`, `deriv1`, `cauchy_mixed`, `bound`, `abs_ub`, `poly_bound`, `U2`, `TINY`, `SAFE` |
| `tmgpu.tmtd` | `TMT`, `const`, `tconst`, `add`, `sub`, `add_const`, `mul`, `scale`, `bound`, `abs_ub`, `to_dd` |
| `tmgpu.gemm` | `zgemm_tt` |
| `tmgpu.host` | `precision`, `exact`, `lower`, `upper`, `to_dd`, `to_td`, `mp_dd`, `dd_mp`, `dd_frac`, `frac_dd` |
| `tmgpu.tools` | `space`, `sup_bound`, `series_order`, `power`, `conj`, `host_dd`, `gpu_dd` |

The users are `quartic/cert/box.py`, `cover.py`, `frame.py`, the scripts `cert_box.py`,
`cert_check.py`, `cert_cover.py`, `cert_run.py`, `tm_local_forms.py`, and the tests
`tests/test_tm_host.py` and `tests/test_cert_imports.py`. Their import lines stay the
same, because the package name stays the same.

## The new repository

### The layout

```
tmgpu/                       the repository root
  pyproject.toml
  uv.lock
  README.md
  CLAUDE.md                  a local file, in .gitignore, not in git
  .gitignore
  tmgpu/                     the package: the 7 files of tmgpu/ at 71d8297, unchanged
    __init__.py  cuda.py  tm.py  tmtd.py  gemm.py  host.py  tools.py
  tests/                     the 5 test files of tmgpu/tests/ at 71d8297
    __init__.py  test_boundary.py  test_host.py  test_tm.py  test_tmtd.py  test_gemm.py
```

* The package keeps the flat layout (my decision). Then each path inside the package,
  for example `tmgpu/tm.py`, is the same as in this repo at 71d8297. The document
  `notes/strict_bound_proof.tex` cites the code by these paths.
* The tests go out of the package, to `tests/` at the root. So an install of the package
  holds no tests.
* The local clone on the laptop is `~/Documents/Research/tmgpu` (my decision).

### The changes to the tests

* Remove the `sys.path.insert(...)` lines. The tests import the package from the
  environment of the new repository.
* `test_boundary.py` finds the package folder from its new place (`parents[1] / 'tmgpu'`).
* The planted import in `test_boundary.py` uses a neutral name, for example
  `from otherpkg import box`, in place of `from quartic.cert import box`.
* Each test file keeps its `if __name__ == '__main__'` runner. The command is
  `python -m tests.test_tm` from the repository root.

### `pyproject.toml`

* `name = "tmgpu"`, `version = "0.1.0"`, `requires-python = ">=3.12"` (the only version
  that the tests ran on).
* The dependencies: numpy, mpmath and python-flint, each with the version of the run of
  2026-10-10 as the lower bound and no exact pin. This repo keeps its exact pins.
* CuPy goes in an optional extra `cuda12 = ["cupy-cuda12x>=14.2"]`, because `tmgpu.host`
  runs on the CPU alone, and a user must choose the CuPy build for the CUDA version.
* A build backend, so that `pip` and `uv` can build a wheel. The builder picks it.
* `uv.lock` is committed, for the development environment of the new repository.

### `README.md`

In STE (ASD-STE100 Simplified Technical English):

* what the package does, and the model of the error bounds in a few lines;
* the install from git now, and from PyPI after the release;
* the requirements: an NVIDIA GPU with FP64, a CUDA 12 toolkit, and CuPy. CuPy compiles the
  kernels with NVRTC, so the dynamic loader must find `libnvrtc-builtins` of the toolkit.
  Give the general values: `CUDA_PATH` and `$CUDA_PATH/lib64` in `LD_LIBRARY_PATH`;
* the modules, each in one line;
* the tests and how to run them.

### `CLAUDE.md`

`CLAUDE.md` is a local file, as in this repo. Commit 568a372 of this repo removed
it from git with the reason "It is a local working file and it does not belong on the
remote". So the `.gitignore` of the new repository lists `CLAUDE.md`, and no commit holds
it. My guess, which the user did not correct: the local file gets these rules from the
`CLAUDE.md` of this repo:

* the language rule (STE) and its list of rules;
* the abbreviations that `tmgpu` uses (dd, td, FP64, GPU, STE);
* the commit rules: no `Co-Authored-By: Claude` trailer, explicit paths in `git add`,
  a check of the `??` lines before a commit;
* code comments of 1 or 2 lines, complete without other information;
* the rule of generality above;
* "Measure, and do not assert";
* the tests need an NVIDIA GPU.

It holds no cluster information.

## The changes in this repo

* `pyproject.toml`: add `tmgpu` to the dependencies, with the source
  `[tool.uv.sources] tmgpu = { git = "ssh://git@github.com/yidiq7/tmgpu.git", tag = "v0.1.0" }`.
  Keep the pins `python-flint==0.9.0` and `cupy-cuda12x==14.2.0` in this repo. Remove the
  words "of tmgpu" from the comment on CuPy, or change them to tell that `tmgpu` needs it.
* `uv.lock`: run `uv lock` after the change. The lock then holds the commit of the tag.
* Remove the folder `tmgpu/` with `git rm -r tmgpu`.
* `README.md`: the install note, the test loop without the 5 `tmgpu` suites, and the
  layout table, where `tmgpu` becomes a dependency from `yidiq7/tmgpu`.
* `CLAUDE.md`: the section "The code of the strict bound" tells that `tmgpu` is now a
  separate repository at v0.1.0, how this repo gets it, and where its tests run. Replace
  the old statements; do not add a correction below them.
* `notes/strict_bound_proof.tex` (untracked): one sentence that `tmgpu` v0.1.0 holds the
  code of `tmgpu/` at 71d8297 with the same paths (my decision). Compile it as
  `CLAUDE.md` tells.
* After the PyPI release, a later change replaces the git source with a version pin.

### Development across the two repositories

To test a change of `tmgpu` with the quartic code before a tag, a developer can point
`[tool.uv.sources]` to the local clone for a short time. That change is never committed,
and no run of the bound uses it. A new version of `tmgpu` enters this repo when the user
changes the tag, runs `uv lock`, and the bit-for-bit check passes again. If a new version
changes the bits of a bound, the bound needs a new full run.

## The cluster

* The compute nodes get the private repository over SSH, with the key in `~/.ssh` of the
  user. `uv` runs `git`, and `git` uses that key. The user decided that no separate test
  of this access is necessary; the sync job of step 7 shows it.
* Never run `uv` on the login node (see `CLAUDE.md`). The sync of this repo after the
  change is a job with `uv sync --locked --extra gpu` on `mweber_compute`.
* A clone `~/tmgpu` on the cluster runs the tests of `tmgpu`, with its own `.venv`. Its
  sync is also a job on `mweber_compute`.
* The submit scripts stay untracked in `~/machine_percision_cy_metric` (my decision), so
  the clone `~/tmgpu` holds no cluster file:
  * `submit_gpu_tests.sh`: remove the 5 `tmgpu` suites. It keeps the 4 suites of this
    repo and the bit-for-bit check of the reference 297.
  * a new `submit_tmgpu_tests.sh`: on one H200, it goes to `~/tmgpu`, sets `CUDA_PATH`
    and `LD_LIBRARY_PATH`, and runs the 5 suites with `uv run --extra cuda12 --locked`.

## The order of the work and the checks

1. Make the new repository on the laptop from `git archive 71d8297 tmgpu`, with the
   changes to the tests and the new files. Commit with a message that names 71d8297.
2. Check the rule of generality: no cluster word, path, user name or project name in the
   new repository. A `git grep` over the tracked files for `mweber`, `yidqi`, `/n/`,
   `netscratch`, `harvard`, `slurm`, `sbatch`, `quartic` and `f6t` must find no match.
3. Make the private GitHub repository `yidiq7/tmgpu` and push.
4. Clone it to `~/tmgpu` on the cluster, sync, and run `submit_tmgpu_tests.sh`. The 5
   suites must pass.
5. Tag `v0.1.0` and push the tag.
6. In this repo, on a branch: the changes of "The changes in this repo". Commit, push, and
   `git pull` on the cluster.
7. Sync this repo on `mweber_compute`, then run `submit_gpu_tests.sh`. The 4 suites must
   pass, and `cert_compare --exact` of the reference 297 against
   `runs_tm/equiv/base_297.npz` must pass.
8. In the environment of this repo, `tmgpu.__file__` must be in `.venv`, and not in a
   folder of this repo.
9. Merge the branch into `main`, after the user agrees.

## Left to the builder

* the build backend, and the `.gitignore` of the new repository;
* the exact lower bounds of the dependency versions, from the lock of this repo;
* the words of the two README files, in STE;
* whether `tmgpu/__init__.py` gets a `__version__`. If it does, the version in
  `pyproject.toml` and in `__version__` must agree.
