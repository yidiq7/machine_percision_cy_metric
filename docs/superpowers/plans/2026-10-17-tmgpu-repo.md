# Move of `tmgpu` into its own repository: the plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to do this plan task by task. Each step has a
> checkbox (`- [ ]`) to record its progress.

**Goal:** Move `tmgpu/` at commit 71d8297 into the private repository `yidiq7/tmgpu` as the
package v0.1.0, and make this repo install it from git, with no change of one bit in the
records of the bound.

**Architecture:** The new repository holds the 7 modules unchanged, in a flat layout, with the
tests at the root. This repo removes its folder `tmgpu/`, and gets the package through a
`[tool.uv.sources]` git entry on the tag `v0.1.0`; `uv.lock` holds the commit. The import
lines of the quartic code do not change.

**Tech stack:** Python 3.12, uv, hatchling, CuPy (CUDA 12), mpmath, python-flint, git, GitHub
(`gh`), SLURM on the cluster.

**Spec:** `docs/superpowers/specs/2026-10-17-tmgpu-repo-design.md`

## Global constraints

* Write all text in STE (ASD-STE100 Simplified Technical English): README files, `CLAUDE.md`,
  code comments, commit messages and tag messages. See the language rule of `CLAUDE.md`.
* No `Co-Authored-By` trailer and no other line of co-authorship or attribution in any commit,
  tag, repository description or file, in either repository.
* Stage explicit paths. Never use `git add -A` or `git add .`. Check the `??` lines before each
  commit.
* No computation on the laptop. On the laptop, use only `python3 -m py_compile`, `git`, `gh`,
  `uv lock`, `cmp` and `grep`. Run every test on the cluster.
* Never run `uv` on the login node. Each `uv sync` is a job on `mweber_compute`. Each GPU job
  asks for `-p mweber_gpu --gres=gpu:1 --constraint=h200 -c 1`.
* The 7 modules of the new repository are byte for byte those of `tmgpu/` at 71d8297.
* No file of the new repository names the cluster or this project. `CLAUDE.md` is in its
  `.gitignore`, and no commit holds it.
* `requires-python = ">=3.12"`. Dependencies: `numpy>=2.5`, `mpmath>=1.3`, `python-flint>=0.9`;
  extra `cuda12 = ["cupy-cuda12x>=14.2"]`. This repo keeps `python-flint==0.9.0` and
  `cupy-cuda12x==14.2.0`.
* The git source in this repo: `tmgpu = { git = "ssh://git@github.com/yidiq7/tmgpu.git", tag = "v0.1.0" }`.
* No change to the earlier documents in `docs/superpowers/`.

## Review focus

1. A folder `tmgpu/` that is left in the root of this repo (for example `__pycache__` after the
   removal) must not supply the package: `tests/test_cert_imports.py::test_paths` asserts that
   `tmgpu` comes from the environment (Task 3, Step 2).
2. The installed copy of `tmgpu` must be byte for byte the code of 71d8297: a `cmp` of the 7
   modules on the laptop (Task 1, Step 6) and a `sha256sum` of the installed files on the
   cluster (Task 4, Step 4).
3. The wheel must hold the 7 modules and no test: the sync job lists the wheel (Task 2, Step 4).
4. The commit in `uv.lock` must be the commit of the tag `v0.1.0` (Task 3, Step 7).
5. `test_boundary` must not pass with an empty scan when its path to the package is wrong: it
   asserts that the scan saw the 7 modules (Task 1, Step 3).

---

### Task 1: The new repository on the laptop

**Files:**
- Create: `~/Documents/Research/tmgpu/` with `tmgpu/` (7 modules), `tests/` (6 files),
  `pyproject.toml`, `uv.lock`, `README.md`, `.gitignore`, and the local `CLAUDE.md`

**Interfaces:**
- Produces: the package `tmgpu` 0.1.0 with the same 55 public names as at 71d8297; the test
  modules `tests.test_boundary`, `tests.test_host`, `tests.test_tm`, `tests.test_tmtd`,
  `tests.test_gemm`, each with its `__main__` runner.

- [ ] **Step 1: Copy the files of 71d8297**

```bash
mkdir ~/Documents/Research/tmgpu && cd ~/Documents/Research/tmgpu && git init -b main
git -C ~/Documents/Research/Machine_percision_CYmetric archive 71d8297 tmgpu | tar -x
mv tmgpu/tests tests
```

Expected: `tmgpu/` holds `__init__.py cuda.py gemm.py host.py tm.py tmtd.py tools.py`, and
`tests/` holds `__init__.py test_boundary.py test_gemm.py test_host.py test_tm.py test_tmtd.py`.

- [ ] **Step 2: Remove the `sys.path` lines of the tests**

In `tests/test_gemm.py`, `test_host.py`, `test_tm.py` and `test_tmtd.py`, remove the line
`sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`, and the `# noqa: E402` on the
`from tmgpu import ...` line after it. Remove `import sys` and `from pathlib import Path` only
if no other line of the file uses them.

- [ ] **Step 3: Change `tests/test_boundary.py`**

* `PKG = Path(__file__).resolve().parents[1] / 'tmgpu'`.
* In `test_boundary`, collect the names of the scanned files, and add after the loop:

```python
    assert {'__init__.py', 'cuda.py', 'tm.py', 'tmtd.py', 'gemm.py', 'host.py',
            'tools.py'} <= scanned, f'the scan missed modules of tmgpu: {sorted(scanned)}'
```

* In `test_boundary_finds_a_planted_import`, the planted line is `from otherpkg import box`,
  and the assertion is `assert imported(p) - ALLOWED == {'otherpkg'}`.
* The skip of the paths under `tests` can go, because the package holds no tests now.

- [ ] **Step 4: Write `pyproject.toml`, `.gitignore`, `README.md` and the local `CLAUDE.md`**

`pyproject.toml`:

```toml
[project]
name = "tmgpu"
version = "0.1.0"
description = "Taylor models in double-double and triple-double on the GPU, with a strict bound of each round-off."
readme = "README.md"
requires-python = ">=3.12"
dependencies = ["numpy>=2.5", "mpmath>=1.3", "python-flint>=0.9"]

[project.optional-dependencies]
cuda12 = ["cupy-cuda12x>=14.2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`.gitignore`: `CLAUDE.md`, `__pycache__/`, `.venv/`, `dist/`, `build/`, `*.egg-info/`.

`README.md`, in STE, with these sections: what the package does, and the model of the error
bounds in a few lines; the install from git
(`uv add "tmgpu[cuda12] @ git+ssh://git@github.com/yidiq7/tmgpu.git" --tag v0.1.0`, or
`pip install "tmgpu[cuda12] @ git+ssh://git@github.com/yidiq7/tmgpu.git@v0.1.0"`), and from
PyPI after the release; the requirements (an NVIDIA GPU with FP64, a CUDA 12 toolkit, CuPy;
`CUDA_PATH` and `$CUDA_PATH/lib64` in `LD_LIBRARY_PATH`, so that the loader finds
`libnvrtc-builtins`); the 7 modules, one line each, from the module docstrings; the tests
(`uv run --extra cuda12 python -m tests.test_tm` from the root, on a machine with an NVIDIA GPU).

`CLAUDE.md` (local, not in git): from the `CLAUDE.md` of this repo, copy the language rule with
its list, the abbreviations dd, td, FP64, GPU and STE, the commit rules (no `Co-Authored-By`
trailer, explicit paths, the check of the `??` lines), the rule on code comments, the rule of
generality (no cluster or project information; do not assume a cluster), "Measure, and do not
assert", and "the tests need an NVIDIA GPU". It holds no cluster information.

- [ ] **Step 5: Lock and compile**

Run: `uv lock && python3 -m py_compile tmgpu/*.py tests/*.py`
Expected: `uv.lock` exists, and no error. If `uv lock` fails on `cupy-cuda12x` for macOS, add
`; sys_platform == 'linux'` to the requirement of the extra and run it again.

- [ ] **Step 6: Check the modules against 71d8297**

Run: `for f in __init__ cuda tm tmtd gemm host tools; do git -C ~/Documents/Research/Machine_percision_CYmetric show 71d8297:tmgpu/$f.py | cmp - tmgpu/$f.py && echo "$f same"; done`
Expected: 7 lines `<name> same`, and no output of `cmp`.

Run: `for f in tests/test_*.py; do git -C ~/Documents/Research/Machine_percision_CYmetric show 71d8297:tmgpu/$f | diff - $f; done`
Expected: only the changes of Steps 2 and 3.

- [ ] **Step 7: Check the rule of generality**

Run: `git add tmgpu tests pyproject.toml uv.lock README.md .gitignore && git status --short && git grep -n -i -E "mweber|yidqi|/n/|netscratch|harvard|fasrc|slurm|sbatch|quartic|f6t"`
Expected: `git status` lists only the staged files and no `CLAUDE.md`; `git grep` finds no match.

- [ ] **Step 8: Commit**

```bash
git commit -m "First version of tmgpu

The 7 modules are those of the folder tmgpu/ at commit 71d8297 of the repository where the
package started, byte for byte. The tests moved to tests/ at the root and import the
installed package."
```

Expected: `git log -1 --format=%B` shows no `Co-Authored-By` line.

---

### Task 2: The private GitHub repository, the tests on the cluster, and the tag

**Files:**
- Create (cluster, untracked): `~/machine_percision_cy_metric/submit_tmgpu_venv.sh`,
  `~/machine_percision_cy_metric/submit_tmgpu_tests.sh`
- Create (cluster): the clone `~/tmgpu`

**Interfaces:**
- Consumes: the commit of Task 1.
- Produces: the tag `v0.1.0` on `yidiq7/tmgpu`, which Task 3 locks.

- [ ] **Step 1: Make the repository and push**

```bash
cd ~/Documents/Research/tmgpu
gh repo create yidiq7/tmgpu --private --source . --push \
  --description "Taylor models in double-double and triple-double on the GPU, with a strict bound of each round-off."
```

Expected: `gh repo view yidiq7/tmgpu --json visibility -q .visibility` prints `PRIVATE`.

- [ ] **Step 2: Clone it on the cluster**

On the login node, with `git` only: `git clone git@github.com:yidiq7/tmgpu.git ~/tmgpu`.

- [ ] **Step 3: Write the two submit scripts**

`submit_tmgpu_venv.sh`: the header of `submit_venv.sh` (`-p mweber_compute`, `-c 2`,
`--mem=8000`, `-t 60`) with `-J tmgpu_venv`, then:

```bash
set -euo pipefail
cd ~/tmgpu
uv sync --locked --extra cuda12
.venv/bin/python -c "import tmgpu, numpy, mpmath, flint; print(tmgpu.__file__, numpy.__version__, mpmath.__version__, flint.__version__)"
W=/tmp/tmgpu_wheel_$SLURM_JOB_ID
uv build --wheel --out-dir $W
.venv/bin/python -m zipfile -l $W/tmgpu-0.1.0-py3-none-any.whl
```

`submit_tmgpu_tests.sh`: the header of `submit_gpu_tests.sh` with `-J tmgputests`, the same
`CUDA_PATH`, `LD_LIBRARY_PATH`, `filt` and `fail` lines, `cd ~/tmgpu`,
`RUN="uv run --extra cuda12 --locked python -u"`, a loop over `tests.test_boundary
tests.test_host tests.test_tm tests.test_tmtd tests.test_gemm`, and the `RESULT` line.

- [ ] **Step 4: Sync and list the wheel**

Run: `sbatch submit_tmgpu_venv.sh`, and watch it with a background command.
Expected: the `tmgpu.__file__` is in `~/tmgpu/tmgpu/`; the wheel holds `tmgpu/__init__.py`,
`cuda.py`, `gemm.py`, `host.py`, `tm.py`, `tmtd.py`, `tools.py` and the `dist-info` files, and
no file under `tests/`.

- [ ] **Step 5: Run the tests**

Run: `sbatch submit_tmgpu_tests.sh`.
Expected: `status 0` for each of the 5 suites, and `RESULT: PASS`.

- [ ] **Step 6: Tag and push the tag**

```bash
cd ~/Documents/Research/tmgpu
git tag -a v0.1.0 -m "tmgpu 0.1.0: the code of tmgpu/ at commit 71d8297"
git push origin v0.1.0
```

Expected: `git ls-remote --tags origin v0.1.0` prints one line.

---

### Task 3: This repo uses the installed package

**Files:**
- Modify: `pyproject.toml` (the dependencies, the CuPy comment, a new `[tool.uv.sources]`)
- Modify: `uv.lock`
- Modify: `tests/test_cert_imports.py` (`test_paths` and the docstring)
- Modify: `README.md` (lines 138 to 185 and 217 to 241)
- Delete: `tmgpu/`

**Interfaces:**
- Consumes: the tag `v0.1.0` of Task 2.
- Produces: the branch `tmgpu-repo`, pushed, with `uv.lock` on the commit of `v0.1.0`.

- [ ] **Step 1: Make the branch**

Run: `git switch -c tmgpu-repo` in this repo.

- [ ] **Step 2: Change `test_paths`**

```python
def test_paths():
    import quartic
    import tmgpu
    assert Path(quartic.__file__).resolve().is_relative_to(ROOT), quartic.__file__
    path = Path(tmgpu.__file__).resolve()
    assert not path.is_relative_to(ROOT / 'tmgpu'), path
    assert path.is_relative_to(Path(sys.prefix).resolve()), path
```

The module docstring: "The certificate code loads no JAX, quartic comes from this repository,
and tmgpu from the environment."

- [ ] **Step 3: Change `pyproject.toml`**

* In `dependencies`, add `"tmgpu"` with a comment of 1 or 2 lines: Taylor models with strict
  bounds on the GPU, from the repository `yidiq7/tmgpu`.
* The comment on CuPy in the `gpu` extra: "CuPy for the kernels of tmgpu. The full run of
  2026-10-10 used 14.2.0."
* A new table at the end:

```toml
[tool.uv.sources]
tmgpu = { git = "ssh://git@github.com/yidiq7/tmgpu.git", tag = "v0.1.0" }
```

- [ ] **Step 4: Lock**

Run: `uv lock`
Expected: `git diff uv.lock` adds the package `tmgpu` 0.1.0 with its git source, and changes no
other version.

- [ ] **Step 5: Remove the folder and change the README**

Run: `git rm -r -q tmgpu`.

`README.md`:
* "Install": one bullet that the lock takes `tmgpu` 0.1.0 from the private repository
  `yidiq7/tmgpu` over SSH, so the install needs a GitHub key with read access to it.
* "Test": remove the 5 `tmgpu.tests.*` suites from the loop, and tell that the tests of `tmgpu`
  are in its repository. The bullets on the `tmgpu` tests and on `test_boundary` go; the bullet
  on `test_cert_imports` also tells that `tmgpu` comes from the environment.
* "Layout": the row `tmgpu/` becomes "`tmgpu` (a dependency, repository `yidiq7/tmgpu`)", and
  the row `tests/`, `tmgpu/tests/` becomes `tests/`.

- [ ] **Step 6: Compile and check the names**

Run: `python3 -m py_compile tests/test_cert_imports.py && git grep -n "tmgpu.tests\|tmgpu/tests" -- ':!docs/superpowers'`
Expected: no error, and no match.

- [ ] **Step 7: Check the lock against the tag**

Run: `git ls-remote --tags git@github.com:yidiq7/tmgpu.git 'v0.1.0^{}'` and
`grep -A3 'name = "tmgpu"' uv.lock`
Expected: the commit in the `source` line of the lock equals the commit of `v0.1.0^{}`.

- [ ] **Step 8: Commit and push**

```bash
git add pyproject.toml uv.lock tests/test_cert_imports.py README.md
git status --short | grep -v '^??'
git commit -m "Install tmgpu 0.1.0 from its own repository

The folder tmgpu/ moved to the private repository yidiq7/tmgpu, whose tag v0.1.0 holds
the same 7 modules as commit 71d8297. uv.lock fixes the commit of the tag.
test_paths now requires tmgpu from the environment and not from a folder of this repo."
git push -u origin tmgpu-repo
```

Expected: the `git status` lines show the 4 files and the deletions under `tmgpu/`, and no
other file; no `Co-Authored-By` line.

---

### Task 4: The bit-for-bit check on the cluster

**Files:**
- Modify (cluster, untracked): `~/machine_percision_cy_metric/submit_gpu_tests.sh`

**Interfaces:**
- Consumes: the branch `tmgpu-repo` of Task 3.

- [ ] **Step 1: Check the jobs**

Run: `squeue -u yidqi -h -o "%i %j %Z"`
Expected: no job with the work directory `~/machine_percision_cy_metric`. If one runs, wait for
its end, because the checkout removes `tmgpu/`.

- [ ] **Step 2: Check out the branch, and look at the folder that stays**

On the login node: `cd ~/machine_percision_cy_metric && git fetch && git switch tmgpu-repo`,
then `find tmgpu -type f | head`.
Expected: only files under `__pycache__`, or no folder. If only those, remove the folder with
`rm -r tmgpu`. If other files are there, stop and ask the user.

- [ ] **Step 3: Sync the environment**

Run: `sbatch submit_venv.sh`
Expected: the sync installs `tmgpu==0.1.0` from the git source, and the job ends with status 0.

- [ ] **Step 4: Compare the installed files with 71d8297**

On the login node:

```bash
for f in __init__ cuda tm tmtd gemm host tools; do
  a=$(git show 71d8297:tmgpu/$f.py | sha256sum | cut -d' ' -f1)
  b=$(sha256sum .venv/lib/python3.12/site-packages/tmgpu/$f.py | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "$f same" || echo "$f DIFFERS"
done
```

Expected: 7 lines `<name> same`.

- [ ] **Step 5: Change and run `submit_gpu_tests.sh`**

Remove the 5 suites `tmgpu.tests.test_boundary tmgpu.tests.test_host tmgpu.tests.test_tm
tmgpu.tests.test_tmtd tmgpu.tests.test_gemm` from its loop, and change its comment to "The unit
tests of quartic.cert in one GPU job, then cert_run on the reference center 297 bit for bit
against runs_tm/equiv/base_297.npz (the records of commit 8f0c1e8)."

Run: `sbatch submit_gpu_tests.sh`
Expected: `status 0` for `tests.test_cert_imports`, `tests.test_cert_records`,
`tests.test_cert_compare` and `tests.test_tm_host`; `compare 297 status 0`; `RESULT: PASS`.

---

### Task 5: The documents and the merge

**Files:**
- Modify (local, untracked): `CLAUDE.md`
- Modify (untracked): `notes/strict_bound_proof.tex`, `notes/strict_bound_proof.pdf`

- [ ] **Step 1: Change `CLAUDE.md`**

Replace the old statements; do not add a correction below them:
* "The code of the strict bound": `tmgpu` is the private repository `yidiq7/tmgpu`, tag
  v0.1.0, with the 7 modules of 71d8297; this repo gets it through `[tool.uv.sources]` and
  `uv.lock`; its tests are in `tests/` of that repository, and `tmgpu/tests/test_boundary.py`
  becomes `tests/test_boundary.py` there.
* The bullet on `submit_gpu_tests.sh`: the 4 suites of this repo and the check of 297; the new
  `submit_tmgpu_tests.sh` and `submit_tmgpu_venv.sh` for the clone `~/tmgpu`.
* The line "Run the unit tests with ...": the commands for both repositories.
* The user's rule on `tmgpu`: the separate repository now exists; the rule of generality stays.
* "The cluster": the clone `~/tmgpu`, and its sync as a job on `mweber_compute`.
* How to test a change of `tmgpu` with the quartic code before a tag: a temporary path
  entry in `[tool.uv.sources]` to the local clone, never committed, and never used by a run.
* The results of Tasks 2 and 4: the job numbers, `RESULT: PASS`, and the 7 equal modules.

- [ ] **Step 2: Add the sentence to the proof document**

In `notes/strict_bound_proof.tex`, after the sentence on `main` at commit 71d8297 (line 77):
"The modules \code{tmgpu/*.py} are also those of the package \code{tmgpu} 0.1.0, with the same
paths; its tests are in \code{tests/} of its repository."

Compile with pdflatex three times in the scratchpad, then copy the .tex and the .pdf to `notes/`.
Expected: no undefined reference in the last log.

- [ ] **Step 3: Merge, after the user agrees**

Ask the user. Then:

```bash
git switch main && git merge --ff-only tmgpu-repo && git push origin main
```

On the cluster: `git switch main && git pull`.
Expected: `main` and `origin/main` are at the commit of Task 3.
