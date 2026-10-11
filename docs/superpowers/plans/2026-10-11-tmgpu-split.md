# tmgpu Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Taylor-model core into `tmgpu/` and the quartic certificate into `quartic/cert/`, remove the unused and superseded code, and show on the GPU that the records of the certificate stay the same, bit for bit.

**Architecture:** First the removals, on the present files, so that each edit below matches the code as it is today. Then the moves, in three steps: the GPU core, the host helpers, the quartic package. After each task, one GPU job on the cluster runs the unit tests and compares the records of reference center 297 with the baseline of `main`. The last task compares 4 reference centers, the hashes of the CUDA sources and the output of the partition check, and then merges.

**Tech Stack:** Python 3.12, CuPy (CUDA kernels through NVRTC), python-flint (Arb), mpmath, numpy, uv, Slurm on FASRC (H200 GPUs).

**Spec:** `docs/superpowers/specs/2026-10-11-tmgpu-split-design.md`

## Global Constraints

- No numerical change: on the same inputs the new code gives the same floats, bit for bit, as `main` at 8f0c1e8.
- Work in a git worktree on the branch `tm-library`. Merge into `main` only after Task 6 passes.
- STE in all prose: comments, docstrings, commit messages. No `Co-Authored-By` trailer in a commit message.
- `git add` with explicit paths only. Never `git add -A`. Read the `??` lines of `git status` before each commit.
- Code comments of 1 or 2 lines, complete without other information, and with no comparison with an older version.
- The laptop does no computation: only `python3 -m py_compile`, `grep`, `uvx pyflakes` and the edit helper of Task 1. All tests run on a GPU node.
- Cluster login: `ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu`. `git` on the login node is allowed; `uv` on the login node is not.
- GPU jobs: `-p mweber_gpu --gres=gpu:1 --constraint=h200`. CPU jobs: `-p mweber_compute`, at most one node.
- No `git pull` in `~/machine_percision_cy_metric` on the cluster while a certificate job runs.
- `tmgpu` imports only the standard library, numpy, cupy, mpmath, flint and itself.

## Review Focus

1. **JAX loads through a path that the edits missed.** JAX then takes 75 % of the GPU, and a batch of 64 boxes at order 32 runs out of memory. Expected: an import of `quartic.cert.box`, `quartic.cert.cover`, `quartic.cert.forms` and `quartic.cert.frame` loads no JAX. Test: `tests/test_cert_imports.py::test_no_jax` (Task 5).
2. **The cluster job imports the code of the main checkout instead of the worktree.** The comparison then says "equal", but it compares the old code with itself. Expected: `quartic.__file__` and `tmgpu.__file__` lie under the worktree. Test: the first step of `submit_tmgpu_check.sh` (Task 1) and `tests/test_cert_imports.py::test_paths` (Task 5).
3. **The split of the CUDA source changes one character.** The kernels and the floats then change. Expected: the four CUDA source texts are identical to those of `main`. Test: `HASHES=1` in the check job (Task 3 and Task 6).
4. **The GPU run is not deterministic.** The bit-for-bit test then fails for equal code. Expected: two runs of `main` give equal records. Test: the baseline job runs reference 297 twice (Task 1).
5. **Old local-form files (4 parts, or Q_e keys) after the removal.** Without a guard they give a KeyError deep in the box. Expected: a clear `ValueError` at the start of `box_f`. Test: `check_old_local_forms` in `tests/test_tm_host.py` (Task 2).

---

## File structure after the work

```
tmgpu/__init__.py              package docstring, no imports
tmgpu/cuda.py                  DD_PRIMITIVES (CUDA source of two_sum, quick_two_sum, dd_add, dd_mul), OPTIONS, module()
tmgpu/tm.py                    <- quartic/tmc.py
tmgpu/tmtd.py                  <- quartic/tmtd.py
tmgpu/gemm.py                  <- quartic/ddgemm.py
tmgpu/host.py                  CPU only: Arb balls and rationals to dd or td, mpmath to and from dd
tmgpu/tools.py                 space, sup_bound, series_order, power, conj, host_dd, gpu_dd
tmgpu/tests/__init__.py
tmgpu/tests/test_boundary.py   new
tmgpu/tests/test_tm.py         <- tests/test_tmc.py
tmgpu/tests/test_tmtd.py       <- tests/test_tmtd.py
tmgpu/tests/test_gemm.py       <- tests/test_ddgemm.py
tmgpu/tests/test_host.py       new; holds check_frac_dd and check_lower of tests/test_tm_host.py
quartic/__init__.py            docstring only
quartic/cert/__init__.py       docstring only
quartic/cert/frame.py          <- quartic/tm_frame.py
quartic/cert/forms.py          <- quartic/tm_forms.py
quartic/cert/box.py            <- quartic/tm_cert.py
quartic/cert/cover.py          the library part of scripts/cert_cover.py
scripts/cert_compare.py        new
tests/test_cert_compare.py     new
tests/test_cert_imports.py     new
tests/test_tm_host.py          kept, with new imports and check_old_local_forms
deleted: quartic/tm_gpu.py, scripts/bench_tm_gpu.py, tests/test_tm_gpu.py, scripts/tm_ref.py
```

Names used below:

- `WT`: the root of the laptop worktree of `tm-library`.
- `SCRATCH`: `/private/tmp/claude-501/-Users-yidiqi-Documents-Research-Machine-percision-CYmetric/2617e71c-1447-4196-8b82-8b44f06c140a/scratchpad`.
- On the cluster: `MAIN=~/machine_percision_cy_metric` (the main checkout, at 8f0c1e8 until Task 6) and `CWT=~/mpcm_tmgpu` (a worktree of `tm-library`).

---

### Task 1: Worktree, comparison tool, spec update, baseline on the cluster

**Files:**
- Create: `scripts/cert_compare.py`, `tests/test_cert_compare.py`
- Modify: `docs/superpowers/specs/2026-10-11-tmgpu-split-design.md`
- Scratch, not committed: `$SCRATCH/refactor_tools.py`
- Cluster, untracked: `$MAIN/submit_tmgpu_base.sh`, `$MAIN/submit_tmgpu_check.sh`, `$MAIN/submit_tmgpu_part.sh`

**Interfaces:**
- Produces: `refactor_tools.replace(path, old, new, count=1)`, `refactor_tools.replace_all(path, old, new) -> int`, `refactor_tools.remove_defs(path, names)`; `cert_compare.compare(a, b) -> (cols, rows)` with rows `(name, max_abs_diff, max_rel_diff, equal: bool)`; on the cluster, `$MAIN/runs_tm/equiv/{refs.txt, base_all.npz, base_297.npz, base_297_again.npz, base_hashes.txt}`.

- [ ] **Step 1: Create the worktree**

Use superpowers:using-git-worktrees. Create the branch `tm-library` from `main` (HEAD 5be57da, which is 8f0c1e8 plus two commits of documentation).

Run: `git -C $WT log --oneline -3`
Expected: the top line is `5be57da State the cluster constraint of the tmgpu design without the cancelled job 51893738`.

- [ ] **Step 2: Write the edit helper (scratch)**

Write `$SCRATCH/refactor_tools.py`:

```python
"""Edit helpers for the tmgpu split; each edit checks how often its text occurs."""
import ast
import re
from pathlib import Path


def replace(path, old, new, count=1):
    """Replace old by new in a file; old must occur exactly count times."""
    p = Path(path)
    s = p.read_text()
    k = s.count(old)
    if k != count:
        raise SystemExit(f'{path}: {old[:70]!r} occurs {k} times, not {count}')
    p.write_text(s.replace(old, new))


def replace_all(path, old, new):
    """Replace every occurrence of old by new, and return the count."""
    p = Path(path)
    s = p.read_text()
    k = s.count(old)
    p.write_text(s.replace(old, new))
    return k


def remove_defs(path, names):
    """Remove the top-level functions, classes and simple assignments with these names."""
    p = Path(path)
    src = p.read_text()
    lines = src.split('\n')
    spans, found = [], set()
    for node in ast.parse(src).body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
        if name in names:
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            spans.append((first - 1, node.end_lineno))
            found.add(name)
    missing = set(names) - found
    if missing:
        raise SystemExit(f'{path}: not found: {sorted(missing)}')
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    p.write_text(re.sub(r'\n{4,}', '\n\n\n', '\n'.join(lines)))
```

Every later edit step runs as `python3 - <<'EOF' ... EOF` from `$WT`, and starts with these two lines:

```python
import sys; sys.path.insert(0, '/private/tmp/claude-501/-Users-yidiqi-Documents-Research-Machine-percision-CYmetric/2617e71c-1447-4196-8b82-8b44f06c140a/scratchpad')
from refactor_tools import replace, replace_all, remove_defs
```

- [ ] **Step 3: Write the test of the comparison tool**

Write `tests/test_cert_compare.py`:

```python
"""Test of scripts/cert_compare.py on small record files, on the CPU."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import cert_compare                                          # noqa: E402


def _rows(recs_a, recs_b):
    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / 'a.npz', Path(d) / 'b.npz'
        for path, recs in ((pa, recs_a), (pb, recs_b)):
            np.savez(path, records=np.array(recs, dtype=float), target=5e-20,
                     columns=np.array(['ref', 'bound']))
        return cert_compare.compare(np.load(pa), np.load(pb))[1]


def test_equal_with_nan():
    rows = _rows([[0, 1e-20], [1, np.nan]], [[0, 1e-20], [1, np.nan]])
    assert all(r[3] for r in rows)


def test_one_ulp_differs():
    x = 1e-20
    rows = _rows([[0, x]], [[0, np.nextafter(x, 1.0)]])
    assert rows[0][3] and not rows[1][3] and rows[1][1] > 0


def test_nan_against_number_differs():
    rows = _rows([[0, np.nan]], [[0, 1e-20]])
    assert not rows[1][3]


def test_different_length_fails():
    try:
        _rows([[0, 1e-20]], [[0, 1e-20], [1, 2e-20]])
    except ValueError:
        return
    raise AssertionError('files of different length passed')


if __name__ == '__main__':
    for fn in (test_equal_with_nan, test_one_ulp_differs, test_nan_against_number_differs,
               test_different_length_fails):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all cert_compare tests pass')
```

The laptop runs no tests. The test runs on the cluster in Step 11. Until Step 4 it fails with `ModuleNotFoundError: No module named 'cert_compare'`.

- [ ] **Step 4: Write the comparison tool**

Write `scripts/cert_compare.py`:

```python
"""Compare two record files of scripts/cert_run.py, column by column.

For each column the script prints the largest absolute difference and the largest
difference relative to abs(A). Two NaN in the same place count as equal. With --exact the
exit status is 0 only if the targets and every column agree bit for bit.

Usage:
    uv run --extra gpu python scripts/cert_compare.py A.npz B.npz --exact
"""
import argparse
import sys

import numpy as np


def compare(a, b):
    """The column names, and for each column (name, largest absolute difference, largest
    relative difference, equal bit for bit with NaN equal to NaN)."""
    ra, rb = a['records'], b['records']
    cols = [str(c) for c in a['columns']]
    if [str(c) for c in b['columns']] != cols:
        raise ValueError('the files have different columns')
    if ra.shape != rb.shape:
        raise ValueError(f'the files have {len(ra)} and {len(rb)} records')
    rows = []
    for j, name in enumerate(cols):
        x, y = ra[:, j], rb[:, j]
        both = np.isfinite(x) & np.isfinite(y)
        diff = np.abs(x[both] - y[both])
        rel = diff / np.maximum(np.abs(x[both]), 1e-300)
        rows.append((name, float(diff.max(initial=0.0)), float(rel.max(initial=0.0)),
                     bool(np.array_equal(x, y, equal_nan=True))))
    return cols, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--exact', action='store_true')
    args = ap.parse_args()
    a, b = np.load(args.a), np.load(args.b)
    try:
        _, rows = compare(a, b)
    except ValueError as err:
        print(f'not comparable: {err}')
        sys.exit(1)
    same_target = float(a['target']) == float(b['target'])
    for name, d, r, same in rows:
        print(f'{name:12s} {"equal" if same else "DIFFERS":8s} largest difference {d:.3e}, '
              f'relative {r:.3e}')
    ok = same_target and all(row[3] for row in rows)
    print(f'target {"equal" if same_target else "DIFFERS"}; {len(a["records"])} records; '
          f'{"all equal bit for bit" if ok else "NOT EQUAL"}', flush=True)
    if args.exact and not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Update the spec to the facts found while the plan was written**

Make these edits in `docs/superpowers/specs/2026-10-11-tmgpu-split-design.md`:

1. In the layout block, replace the two lines of `box.py` under `tmgpu/` with:
```
  tools.py                  space, sup_bound, series_order, power, conj, and CuPy dd
                            arrays to and from mpmath (host_dd, gpu_dd) <- tm_cert.py
```
   Also add `(CPU only)` to the line of `host.py`. Under `tests/`, add `test_host.py  new: check_frac_dd and check_lower of tests/test_tm_host.py`.
2. In the table "What moves where", replace `tmgpu/box.py` with `tmgpu/tools.py`. Move `_host_dd` and `_to_dd` from the `host.py` row to the `tools.py` row, as `host_dd` and `gpu_dd`.
3. In the table "What goes", remove `lin_tm` and `p_value` from the dd-chain row, and add the row:
   `| tm_cert.neg, tm_cert.p_value, tm_cert.tm_const; the dd frame constants ha, a6, a4, h_err of tm_frame.frame and of Frames | no code on the td path uses them |`
4. Add to "Other changes": `* cert_box.py reads the six parts of the td local forms in its check of p0.` and `* box_f stops with a clear ValueError if the local forms are not in Delta and in td.`
5. Replace the GPU part of "Verification" with:
```
2. **On a GPU node, after each task:** the unit tests, the paths of the imported packages,
   and cert_run.py with --grad on reference center 297, compared bit for bit with the
   records of main.
3. **Before the merge:** cert_run.py with --grad on 4 reference centers (0, 243, 297, and
   the center with the most split cells in run_f6t_v2.npz), bit for bit; the sha256 of
   the four CUDA sources against main; cert_partition.py --grad on the full run_f6t_v2.npz
   with the code of main and of the branch, with the same output; cert_check.py with
   --points 300 --dropped 300 on the branch. cert_local_values.py imports no moved code,
   so it is not part of the test.
```
   Renumber the item "After the merge" to 4.

- [ ] **Step 6: Static checks**

Run: `cd $WT && python3 -m py_compile scripts/cert_compare.py tests/test_cert_compare.py && uvx pyflakes scripts/cert_compare.py tests/test_cert_compare.py`
Expected: no output.

- [ ] **Step 7: Commit and push**

```bash
cd $WT
git status --short
git add scripts/cert_compare.py tests/test_cert_compare.py docs/superpowers/specs/2026-10-11-tmgpu-split-design.md
git commit -m "Add cert_compare.py, which compares two record files of cert_run column by column

The split of the Taylor-model core needs a test that the records stay the same bit for
bit. The design now names tmgpu/tools.py, keeps lin_tm, and lists the checks on the GPU."
git push -u origin tm-library
```

- [ ] **Step 8: Create the cluster worktree**

Run on the login node:
```bash
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'git -C ~/machine_percision_cy_metric fetch origin tm-library && git -C ~/machine_percision_cy_metric worktree add --detach ~/mpcm_tmgpu origin/tm-library && git -C ~/mpcm_tmgpu log --oneline -1 && git -C ~/machine_percision_cy_metric log --oneline -1'
```
Expected: the worktree shows the commit of Step 7, and the main checkout shows `8f0c1e8`.

- [ ] **Step 9: Write the three cluster scripts**

Write `$MAIN/submit_tmgpu_base.sh` on the cluster:

```bash
#!/bin/bash
#SBATCH -p mweber_gpu
#SBATCH -c 1
#SBATCH --mem=96000
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH -t 180
#SBATCH -J tmgbase
# The baseline of the tmgpu split, with the code of main: the records of 4 reference
# centers, a second run of reference 297 for the determinism, the hashes of the CUDA sources.
set -uo pipefail
MAIN=~/machine_percision_cy_metric
CWT=~/mpcm_tmgpu
EQ=$MAIN/runs_tm/equiv
mkdir -p $EQ
cd $MAIN
git log --oneline -1
export CUDA_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/12.9.1-fasrc01/cuda
export JAX_PLATFORMS=cpu
RUN="uv run --extra gpu --with cupy-cuda12x --with python-flint python -u"
RUNW="uv run --project $MAIN --extra gpu --with cupy-cuda12x --with python-flint python -u"
COV=runs_tm/cover_s0.03_c6_v2.npz
L=/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td
$RUN - > $EQ/refs.txt <<'PY'
import numpy as np
rec = np.load('runs_tm/run_f6t_v2.npz')['records']
refs, counts = np.unique(rec[rec[:, 1] >= 1, 0].astype(int), return_counts=True)
order = [int(r) for r in refs[np.argsort(-counts)] if int(r) not in (0, 243, 297)]
print(0, 243, 297, order[0])
PY
echo "reference centers: $(cat $EQ/refs.txt)"
$RUN scripts/cert_run.py --cover $COV --local $L --refs $(cat $EQ/refs.txt) --target 5e-20 --grad --out $EQ/base_all.npz
$RUN scripts/cert_run.py --cover $COV --local $L --refs 297 --target 5e-20 --grad --out $EQ/base_297.npz
$RUN scripts/cert_run.py --cover $COV --local $L --refs 297 --target 5e-20 --grad --out $EQ/base_297_again.npz
$RUN - > $EQ/base_hashes.txt <<'PY'
import hashlib
from quartic import tmc, tmtd, ddgemm
for s in (tmc._SRC, tmtd.TD_SRC, ddgemm._SRC, ddgemm._TT_SRC):
    print(hashlib.sha256(s.encode()).hexdigest())
PY
cat $EQ/base_hashes.txt
cd $CWT
$RUNW scripts/cert_compare.py $EQ/base_297.npz $EQ/base_297_again.npz --exact && echo DETERMINISTIC || echo "NOT DETERMINISTIC"
$RUNW -m tests.test_cert_compare
```

Write `$MAIN/submit_tmgpu_check.sh` on the cluster:

```bash
#!/bin/bash
#SBATCH -p mweber_gpu
#SBATCH -c 1
#SBATCH --mem=96000
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH -t 120
#SBATCH -J tmgchk
# A check of the branch tm-library in ~/mpcm_tmgpu: the path of the imported package, the
# unit tests TESTS, with HASHES=1 the CUDA sources against main, the records of REFS
# against base_TAG.npz, and with FULL=1 a short run of cert_check.py.
set -uo pipefail
MAIN=~/machine_percision_cy_metric
CWT=~/mpcm_tmgpu
EQ=$MAIN/runs_tm/equiv
cd $CWT
git log --oneline -1
export CUDA_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/12.9.1-fasrc01/cuda
RUN="uv run --project $MAIN --extra gpu --with cupy-cuda12x --with python-flint python -u"
COV=$MAIN/runs_tm/cover_s0.03_c6_v2.npz
L=/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td
fail=0
$RUN -c "import os, quartic; p = os.path.realpath(quartic.__file__); print(p); assert p.startswith(os.path.realpath('$CWT'))" || fail=1
for t in $TESTS; do
    echo "=== $t"
    $RUN -m $t || fail=1
done
if [ "${HASHES:-0}" = 1 ]; then
    $RUN - > $EQ/branch_hashes.txt <<'PY'
import hashlib
from tmgpu import tm, tmtd, gemm
for s in (tm._SRC, tmtd.TD_SRC, gemm._SRC, gemm._TT_SRC):
    print(hashlib.sha256(s.encode()).hexdigest())
PY
    diff $EQ/base_hashes.txt $EQ/branch_hashes.txt && echo "CUDA SOURCES EQUAL" || { echo "CUDA SOURCES DIFFER"; fail=1; }
fi
$RUN scripts/cert_run.py --cover $COV --local $L --refs $REFS --target 5e-20 --grad --out $EQ/branch_$TAG.npz || fail=1
$RUN scripts/cert_compare.py $EQ/base_$TAG.npz $EQ/branch_$TAG.npz --exact || fail=1
if [ "${FULL:-0}" = 1 ]; then
    $RUN scripts/cert_check.py --run $MAIN/runs_tm/run_f6t_v2.npz --cover $COV --points 300 --dropped 300 || fail=1
fi
echo "RESULT: $([ $fail = 0 ] && echo PASS || echo FAIL)"
```

Write `$MAIN/submit_tmgpu_part.sh` on the cluster:

```bash
#!/bin/bash
#SBATCH -p mweber_compute
#SBATCH -c 16
#SBATCH --mem=64000
#SBATCH -t 120
#SBATCH -J tmgpart
# cert_partition.py --grad on the full run_f6t_v2.npz, with the code of main and of the
# branch; the two outputs must agree, apart from the times.
set -uo pipefail
MAIN=~/machine_percision_cy_metric
CWT=~/mpcm_tmgpu
EQ=$MAIN/runs_tm/equiv
RUN="uv run --project $MAIN --extra gpu --with python-flint python -u"
for side in main branch; do
    if [ $side = main ]; then cd $MAIN; else cd $CWT; fi
    git log --oneline -1
    JAX_PLATFORMS=cpu $RUN scripts/cert_partition.py --run $MAIN/runs_tm/run_f6t_v2.npz \
        --cover $MAIN/runs_tm/cover_s0.03_c6_v2.npz --depth 2 --grad --procs 16 2>&1 \
        | grep -v -E "Warning|return asarray|fork" > $EQ/partition_$side.txt
    sed -E 's/[0-9][0-9.]*s\b//g' $EQ/partition_$side.txt > $EQ/partition_${side}_cut.txt
done
diff $EQ/partition_main_cut.txt $EQ/partition_branch_cut.txt && echo "PARTITION OUTPUT EQUAL" || echo "PARTITION OUTPUT DIFFERS"
cat $EQ/partition_branch.txt
```

Note: in the branch, `cert_partition.py` must run with no CuPy. The cover reaches `tmgpu.host`, which is CPU only.

- [ ] **Step 10: Submit the baseline and watch it**

```bash
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/machine_percision_cy_metric && sbatch submit_tmgpu_base.sh'
```
Watch it with a background loop on `sacct -j <id>`, because a foreground `sleep` is blocked.

- [ ] **Step 11: Read the baseline**

Run: `ssh ... 'cd ~/machine_percision_cy_metric && tail -40 slurm-<id>.out && ls -la runs_tm/equiv'`
Expected:
- 4 reference centers in `refs.txt`;
- `base_all.npz`, `base_297.npz`, `base_297_again.npz`, and 4 lines in `base_hashes.txt`;
- `DETERMINISTIC`, and `all cert_compare tests pass`.

If the output says `NOT DETERMINISTIC`, stop the plan and report it. The bit-for-bit test then cannot show equal code.

---

### Task 2: Remove the dd chain, the Q_e form, the unused code and the XLA prototype

**Files:**
- Modify: `quartic/tm_cert.py`, `quartic/tm_frame.py`, `quartic/tm_forms.py`, `quartic/ddgemm.py`, `scripts/cert_run.py`, `scripts/cert_box.py`, `scripts/tm_local_forms.py`, `tests/test_tm_host.py`
- Delete: `quartic/tm_gpu.py`, `scripts/bench_tm_gpu.py`, `tests/test_tm_gpu.py`, `scripts/tm_ref.py`

**Interfaces:**
- Consumes: `replace`, `replace_all`, `remove_defs` (Task 1).
- Produces:
  - `box_f(fr, local, delta_box, n, ny, k_max=20, a=None, k_full=3, hmax=None, grad=False)`, with no `td` argument;
  - `_check_local(local)`, which raises `ValueError`;
  - `reexpand_td(local, zeta_dd, k_max, top)` and `reexpand_abs(local, zeta_host, k, top)`, with no `tag` argument;
  - `tm_forms.local_forms(cf, c, x0, s2)`, which always writes the Delta form in td;
  - `Frames` with only the td constants of the map for h;
  - `tm_frame.frame` returns no `ha`, `a6`, `a4` or `h_err`.

- [ ] **Step 1: Write the test of the guard, which fails first**

In `tests/test_tm_host.py`, add this function above `def main():`:

```python
def check_old_local_forms():
    for bad in ({'parts': 4, 's2': np.zeros(4)}, {'parts': 6}):
        try:
            tc._check_local(bad)
        except ValueError:
            continue
        raise AssertionError(f'local forms with keys {sorted(bad)} passed the check')
    tc._check_local({'parts': 6, 's2': np.zeros(4)})
    print('  _check_local: local forms that are not in Delta and td fail', flush=True)
```

Then add the call `    check_old_local_forms()` in `main()`, after `check_real_low()`. Until Step 4 the test fails with `AttributeError: module 'quartic.tm_cert' has no attribute '_check_local'`.

- [ ] **Step 2: Remove the dd functions of tm_cert**

```python
remove_defs('quartic/tm_cert.py', ['neg', 'tm_const', 'p_value', 'implicit_h', 'offset_models',
            'monomial_mats', '_batched_mul', 'compose', '_add_mats', '_bnorm', '_bnorm_low',
            'reexpand', 'delta_form'])
```

Keep `lin_tm`: `implicit_h_td` and `offset_models_td` call it.

- [ ] **Step 3: Remove the dd frame constants from Frames and make_frames**

```python
replace('quartic/tm_cert.py', """    host: list             # mpmath values of each box, for checks
    ha: cp.ndarray = None  # (3, 4, B): c1inv P(p), c1inv grad.E_0, c1inv grad.E_1
    a6: cp.ndarray = None  # (3, 4, B): 6 p_k^2
    a4: cp.ndarray = None  # (3, 4, B): 4 p_k
    h_err: cp.ndarray = None   # (6, B): the errors of ha (3), abs(1 - c1inv c1), and the
                               # largest errors of a6 and a4
    ha_td: cp.ndarray = None   # (3, 6, B), and the next three: the same constants in td,
    a6_td: cp.ndarray = None   # with kappa = 1 - c1inv c1 and the errors of the td roundings
    a4_td: cp.ndarray = None
    kappa_td: cp.ndarray = None
    h_err_td: cp.ndarray = None
""", """    host: list             # mpmath values of each box, for checks
    ha_td: cp.ndarray = None       # (3, 6, B): c1inv P(p), c1inv grad.E_0, c1inv grad.E_1
    a6_td: cp.ndarray = None       # (3, 6, B): 6 p_k^2
    a4_td: cp.ndarray = None       # (3, 6, B): 4 p_k
    kappa_td: cp.ndarray = None    # (6, B): kappa = 1 - c1inv c1
    h_err_td: cp.ndarray = None    # (6, B): the errors of ha_td (3) and kappa_td, and the
                                   # largest errors of a6_td and a4_td
""")
replace('quartic/tm_cert.py', """    keys = ('p', 'e', 'n', 'c1inv', 'vol', 'ha', 'a6', 'a4', 'ha_td', 'a6_td', 'a4_td',
            'kappa_td')
    cols = {k: [] for k in keys}
    errs, host, herr, herr_td = [], [], [], []
    for t in centers:
        f = frame(t, k0)
        for k in keys:
            cols[k].append(f[k])
        errs.append(f['vol_err'])
        herr.append(f['h_err'])
        herr_td.append(f['h_err_td'])
""", """    keys = ('p', 'e', 'n', 'c1inv', 'vol', 'ha_td', 'a6_td', 'a4_td', 'kappa_td')
    cols = {k: [] for k in keys}
    errs, host, herr_td = [], [], []
    for t in centers:
        f = frame(t, k0)
        for k in keys:
            cols[k].append(f[k])
        errs.append(f['vol_err'])
        herr_td.append(f['h_err_td'])
""")
replace('quartic/tm_cert.py', """                  vol_err=cp.asarray(np.array(errs)), host=host, ha=to('ha'), a6=to('a6'),
                  a4=to('a4'), h_err=cp.asarray(np.array(herr).T), ha_td=to('ha_td'),
                  a6_td=to('a6_td'), a4_td=to('a4_td'), kappa_td=to('kappa_td'),
""", """                  vol_err=cp.asarray(np.array(errs)), host=host, ha_td=to('ha_td'),
                  a6_td=to('a6_td'), a4_td=to('a4_td'), kappa_td=to('kappa_td'),
""")
```

- [ ] **Step 4: Make box_f take the td chain only, with the guard**

```python
replace('quartic/tm_cert.py', """def box_f(fr, local, delta_box, n, ny, k_max=20, a=None, k_full=3, td=False, hmax=None,
          grad=False):""", """def _check_local(local):
    \"\"\"Stop with a clear message if the local forms are not in Delta and in td.\"\"\"
    if int(local.get('parts', 0)) != 6 or 's2' not in local:
        raise ValueError('the box needs local forms in Delta and in td, as '
                         'scripts/tm_local_forms.py writes them')


def box_f(fr, local, delta_box, n, ny, k_max=20, a=None, k_full=3, hmax=None, grad=False):""")
replace('quartic/tm_cert.py', """    td: the compositions with the forms take the triple-double products of ddgemm.
    hmax: the bound of abs(h) over the cell of each box, for the check in implicit_h.
""", """    hmax: the bound of abs(h) over the cell of each box, for the check in implicit_h_td.
""")
replace('quartic/tm_cert.py', """    # The forms are sums over e of abs(y_2)^(2e) Q_e, or over j of Delta^j R_j; local
    # forms stored in td (6 parts) take the td chain.
    mode_j = 's2' in local
    tag = 'j' if mode_j else 'e'
    chain = int(local.get('parts', 4)) == 6
    if chain and not mode_j:
        raise ValueError('the td chain needs the forms in Delta (tm_local_forms.py --delta)')
    cref = [list(cp.asnumpy(cp.asarray(local['c']))[i]) for i in range(2)]
    if chain:
        ys, rho, lam, h_td, hm, lam_big = implicit_h_td(sp2, fr, d_phi, hmax)
        base_t, v2 = offset_models_td(sp2, fr, h_td, hm, cref, d_phi)
    else:
        ys, rho, lam, hm, lam_big = implicit_h(sp2, fr, d_phi, hmax)
        base_t, v2 = offset_models(sp2, fr, hm, cref, d_phi)
    info['rho_h'], info['lam_h'], info['lam_big'] = rho, lam, lam_big
    yc = [conj_tm(y) for y in ys]
    clock('h')
    if chain:
        mats = monomial_mats_td(sp2, sp_low, base_t, top)
    else:
        mats = monomial_mats(sp2, sp_low, base_t, None if mode_j else ys[2], top)
    dls, bdl = {}, None
    if mode_j:
        dls[n] = delta_model(sp4, sp2, fr, v2, local, d_phi)
        bdl = tmc.bound(sp4, dls[n])
        info['bound_delta'] = bdl
    clock('monomials')

    def form(sp4k, splk, mk, q, rr, aa):
        if not mode_j:
            return compose(sp4k, splk, mk, q, rr, aa, d_phi, td)
        if sp4k.n not in dls:
            dls[sp4k.n] = tmc.truncate(sp4k, sp4, dls[n])
        if chain:
            return delta_form_td(sp4k, splk, mk, q, rr, aa, d_phi, dls[sp4k.n], bdl)
        return delta_form(sp4k, splk, mk, q, rr, aa, d_phi, dls[sp4k.n], bdl, td)
""", """    # Each form is a sum over j of Delta^j R_j, with the R_j of the local forms in td.
    _check_local(local)
    cref = [list(cp.asnumpy(cp.asarray(local['c']))[i]) for i in range(2)]
    ys, rho, lam, h_td, hm, lam_big = implicit_h_td(sp2, fr, d_phi, hmax)
    base_t, v2 = offset_models_td(sp2, fr, h_td, hm, cref, d_phi)
    info['rho_h'], info['lam_h'], info['lam_big'] = rho, lam, lam_big
    yc = [conj_tm(y) for y in ys]
    clock('h')
    mats = monomial_mats_td(sp2, sp_low, base_t, top)
    dls = {n: delta_model(sp4, sp2, fr, v2, local, d_phi)}
    bdl = tmc.bound(sp4, dls[n])
    info['bound_delta'] = bdl
    clock('monomials')

    def form(sp4k, splk, mk, q, rr, aa):
        if sp4k.n not in dls:
            dls[sp4k.n] = tmc.truncate(sp4k, sp4, dls[n])
        return delta_form_td(sp4k, splk, mk, q, rr, aa, d_phi, dls[sp4k.n], bdl)
""")
replace('quartic/tm_cert.py', """    fq = (reexpand_td(local, zeta, k_max, top, tag) if chain
          else reexpand(local, zeta, k_max, top, tag, td))
""", """    fq = reexpand_td(local, zeta, k_max, top)
""")
replace('quartic/tm_cert.py', """reexpand_abs(local, [dd_mp(z) for z in zeta], k, top, tag),""",
        """reexpand_abs(local, [dd_mp(z) for z in zeta], k, top),""")
replace('quartic/tm_cert.py', "delta=dls.get(n))", "delta=dls[n])")
replace('quartic/tm_cert.py', "def reexpand_td(local, zeta_dd, k_max, top, tag='j'):",
        "def reexpand_td(local, zeta_dd, k_max, top):")
replace('quartic/tm_cert.py', "def reexpand_abs(local, zeta_host, k, top, tag='e'):",
        "def reexpand_abs(local, zeta_host, k, top):")
assert replace_all('quartic/tm_cert.py', '_{tag}{e}', '_j{e}') == 8
```

The 8 replacements are 3 in `box_f` (base), 4 in `reexpand_td` and 1 in `reexpand_abs`. The order of the GPU operations on the td path does not change.

- [ ] **Step 5: Make the docstrings of the td functions complete without the dd functions**

```python
replace('quartic/tm_cert.py', '''    """`implicit_h` with the map T(h) = kappa h - A(tau) - c1inv N(w) in td arithmetic.

    The residual of the td model of h, with the errors of the td constants of the frame,
    gives rho; the contraction factors come from the dd models, as bounds. Returns the dd
    models ys, rho, lam, the td model of h, its dd model, and lam_big.
    """''', '''    """The models of y(tau) = p + E tau + h n on X, with a strict remainder of h, in td.

    T(h) = h - c1inv P(y(h)). If T maps the ball abs(h - P_h) <= rho into itself with a
    contraction factor lam < 1, the root of each tau lies in that ball. With w = E tau + h n,
    T(h) = kappa h - A(tau) - c1inv N(w): kappa = 1 - c1inv c1, A = c1inv (P(p) + grad.E tau)
    and N(w) = sum over k of w_k^2 (6 p_k^2 + w_k (4 p_k + w_k)). The host gives kappa, A,
    6 p_k^2 and 4 p_k as td numbers with their errors, so the round-off acts only on the
    small terms of N. The residual of the td model of h, with those errors, gives rho; the
    contraction factors come from the dd models, as bounds.

    With hmax (B,), a bound of abs(h) for every point of X over the cell of the box (from
    the cover), T must also contract on the ball of radius R = hmax + B(h) + rho around
    the model, which holds both that point and the root: then they are the same point.
    A box whose bound lam_big of T' on that ball is not below 1 gets rho = inf. Returns the
    dd models ys, rho, lam, the td model of h, its dd model, and lam_big (NaN without hmax).
    """''')
replace('quartic/tm_cert.py', '''    """`offset_models` with td models of t_i - ref_i; v = y_2 - p_2 stays dd."""''',
        '''    """The td models of t_i - ref_i (i = 0, 1) and the dd model of v = y_2 - p_2.

    The constant p_i - ref_i comes from the host, rounded to td with its error, so no
    difference of two near values rounds on the GPU. ref: the two dd lists of c_r.
    """''')
replace('quartic/tm_cert.py', '''    """`monomial_mats` (one block, e = 0) with td models: Y is (B, 6, S_low, n_top)."""''',
        '''    """The td coefficients of the monomials (t - c_r)^alpha, alpha of degree <= top.

    base_t: the td models of t_i - c_r,i (offset_models_td). Returns a list with one
    block (Y, b_low, b_high, rem, gdeg): Y (B, 6, S_low, n_top) for the degrees <= sp_low.n
    in tau; (n_top, B) bounds of that part, of the rest of the polynomial and of the
    remainder; gdeg, the bound of each degree.
    """''')
replace('quartic/tm_cert.py', '''    """`reexpand` with the td weights of `reexpand_weights` and td local forms; returns td
    matrices (B, 6, n, n) of F_k with rad and abs. The error of the product has the bound
    that the kernel of `ddgemm.zgemm_tt` sums as it runs.''', '''    """Per box: the td matrices of F_k = (1 + zeta)^k sum over m >= k of C(m, k)
    zeta^(m - k) M~_m, for k <= k_max and each block j, with rad and abs.

    For each j this is one product of the td weights of `reexpand_weights` with the stacked
    td local forms. The error of the product has the bound that the kernel of
    `ddgemm.zgemm_tt` sums as it runs.''')
replace('quartic/tm_cert.py', '''    """`compose` for one block with td models and a td form: T = Y Q in td, W = T Y^H to dd.
''', '''    """The model in four variables of Y Q Y^H for one block, with all remainders.

    T = Y Q in td, W = T Y^H to dd. The true form differs from the polynomial of total
    degree <= n by the entries of W of higher total degree (their exact sizes; a majorant
    from abs(Q) is far too large, because those entries cancel strongly), by the parts of
    degree above sp_low.n of the monomial models, by their remainders, by the radii of Q,
    and by the round-off of the two products.
''')
replace('quartic/tm_cert.py', '''    """`delta_form` with compose_td."""''',
        '''    """sum over j of Delta^j W_j with W_j = Y R_j Y^H (compose_td), by Horner's rule in Delta.

    The parts of the remainder are those of the W_j, with the factor bdl^j.
    """''')
replace('quartic/tm_cert.py', "of `reexpand` in ball arithmetic", "of `reexpand_td` in ball arithmetic")
replace('quartic/tm_cert.py', '''    """abs bounds of the matrices of F_k only, (B, n, n) for each e."""''',
        '''    """abs bounds of the matrices of F_k only, (B, n, n) for each block j."""''')
```

- [ ] **Step 6: Rewrite the module docstring of tm_cert**

```python
replace('quartic/tm_cert.py', '''The design is docs/superpowers/specs/2026-10-09-gpu-taylor-certificate-design.md; the
arb reference of the same steps is scripts/tm_ref.py (--affine). All boxes of a batch
share one reference center c_r and its local forms (scripts/tm_local_forms.py).

1. Host: each box has a center p on X (t = (y_0, y_1), y_2 from P = 0), the frame
   E = dy/dt, the unit normal n and 1/c1 = 1/(grad P . n), all exact dd numbers.
2. GPU, models in tau alone (order n_y): h from the fixed point h -> h - P(y(h)) / c1 with
   a contraction bound, y(tau) = p + E tau + h n, the monomials (t - c_r)^alpha y_2^e.
3. Composition W = sum over e of Y_e Q_e Y_e^H with the local forms, up to the degree n
   in tau and in sigma; the parts above n go to the remainder through bounds.''', '''The design is docs/superpowers/specs/2026-10-09-gpu-taylor-certificate-design.md. All
boxes of a batch share one reference center c_r and its local forms
(scripts/tm_local_forms.py), stored in td, each form as sum over j of Delta^j R_j with
Delta = abs(y_2)^2 - s2.

1. Host: each box has a center p on X (t = (y_0, y_1), y_2 from P = 0), the frame
   E = dy/dt, the unit normal n and 1/c1 = 1/(grad P . n), all exact dd numbers.
2. GPU, models in tau alone (order n_y), in td: h from the fixed point
   h -> h - P(y(h)) / c1 with a contraction bound, y(tau) = p + E tau + h n, and the
   monomials (t - c_r)^alpha.
3. Composition W_j = Y R_j Y^H with the local forms, up to the degree n in tau and in
   sigma, summed by Horner's rule in Delta; the parts above n go to the remainder through
   bounds.''')
```

- [ ] **Step 7: Remove the dd frame constants from tm_frame.frame**

```python
replace('quartic/tm_frame.py', "* c1inv = 1 / (grad P . n) and the constants of `tm_cert.implicit_h` with their errors;",
        "* c1inv = 1 / (grad P . n) and the td constants of `tm_cert.implicit_h_td` with their errors;")
replace('quartic/tm_frame.py', """    Returns a dict of dd lists (p, e, n, c1inv, vol, ha, a6, a4), the exact acb values
    of the frame (key 'acb'), the relative error bound vol_err of vol, and h_err =
    [errors of the three ha, abs(1 - c1inv c1), largest error of a6, of a4].
""", """    Returns a dict of dd lists (p, e, n, c1inv, vol); the td constants of the map for h
    (ha_td, a6_td, a4_td, kappa_td) with h_err_td = [errors of the three ha_td, of kappa_td,
    largest error of a6_td, of a4_td]; the exact acb values of the frame (key 'acb'); and
    the relative error bound vol_err of vol.
""")
replace('quartic/tm_frame.py', """        ha_dd = [to_dd(v) for v in ha]
        a6 = [to_dd(6 * c ** 2) for c in p]
        a4 = [to_dd(4 * c) for c in p]
        h_err = [x[1] for x in ha_dd] + [upper(1 - c1inv * c1), max(x[1] for x in a6),
                                         max(x[1] for x in a4)]
        # The same constants in td, and kappa = 1 - c1inv c1, for the td solve of h.
""", """        # The constants of the td solve of h, with kappa = 1 - c1inv c1.
""")
replace('quartic/tm_frame.py', """                    vol=vol_dd, vol_err=vol_err, ha=[x[0] for x in ha_dd],
                    a6=[x[0] for x in a6], a4=[x[0] for x in a4], h_err=h_err,
                    ha_td""", """                    vol=vol_dd, vol_err=vol_err, ha_td""")
```

- [ ] **Step 8: Make tm_forms write only the Delta form in td**

```python
replace('quartic/tm_forms.py', '''def local_forms(cf, c, x0, s2=None, td=False):
    """The base and the M~_m at x0, moved to t = c in arb, as dd parts, radii, abs bounds.''',
    '''def local_forms(cf, c, x0, s2):
    """The base and the M~_m at x0, moved to t = c in arb, as td parts, radii, abs bounds.''')
replace('quartic/tm_forms.py', """    With an exact real s2, each form is written as R_j of `delta_split` (keys
    `<name>_j<j>`) and not as Q_e (keys `<name>_e<e>`). td stores three limbs (6 parts).
""", """    With the exact real s2, each form is written as the R_j of `delta_split` (keys
    `<name>_j<j>`), each entry in three limbs (6 parts).
""")
replace('quartic/tm_forms.py', """    tag = 'e' if s2 is None else 'j'
    for name, qs in named:
        if s2 is not None:
            qs = delta_split(qs, acb(s2))
        for e, q in enumerate(qs):
            parts, rad, absb = (to_td_complex if td else to_dd_complex)(q)
            out[f'{name}_{tag}{e}'] = parts
            out[f'{name}_{tag}{e}_rad'] = rad
            out[f'{name}_{tag}{e}_abs'] = absb
""", """    for name, qs in named:
        qs = delta_split(qs, acb(s2))
        for e, q in enumerate(qs):
            parts, rad, absb = to_td_complex(q)
            out[f'{name}_j{e}'] = parts
            out[f'{name}_j{e}_rad'] = rad
            out[f'{name}_j{e}_abs'] = absb
""")
replace('quartic/tm_forms.py', "in the order of tm_gpu.Space(2, d).", "in the order of tmgpu.tm.Space(2, d).")
remove_defs('quartic/tm_forms.py', ['abs_bounds', 'chart_forms', 'combine_levels', 'save_gpu',
            'to_dd', 'to_dd_complex'])
remove_defs('quartic/ddgemm.py', ['td_of_dd', 'tcabs'])
```

- [ ] **Step 9: Make tm_local_forms always write the Delta form in td**

```python
replace('scripts/tm_local_forms.py', """exact dd numbers c and x0. With --delta, each form is the four R_j of
`quartic.tm_forms.delta_split`, and the file also holds the exact dd number s2.""",
        """exact dd numbers c, x0 and s2. Each form is the four R_j of
`quartic.tm_forms.delta_split`, with each entry in td.""")
replace('scripts/tm_local_forms.py', """    s2_dd = dd_of(exact(acb(abs(y2) ** 2))) if G['delta'] else None
    s2 = from_dd(s2_dd).real if s2_dd is not None else None
    out = tm_forms.local_forms(G['cf'], c, x0, s2, td=G['td'])
    out['c'] = np.array([t[0], t[1]])
    out['x0'] = np.array(x0_dd)
    if s2 is not None:
        out['s2'] = np.array(s2_dd)
    out['parts'] = 6 if G['td'] else 4
""", """    s2_dd = dd_of(exact(acb(abs(y2) ** 2)))
    s2 = from_dd(s2_dd).real
    out = tm_forms.local_forms(G['cf'], c, x0, s2)
    out['c'] = np.array([t[0], t[1]])
    out['x0'] = np.array(x0_dd)
    out['s2'] = np.array(s2_dd)
    out['parts'] = 6
""")
replace('scripts/tm_local_forms.py', """    ap.add_argument('--delta', action='store_true',
                    help='write each form as R_j in Delta = abs(y2)^2 - s2 (tm_forms.delta_split)')
    ap.add_argument('--td', action='store_true', help='store three limbs (td) of each entry')
""", "")
replace('scripts/tm_local_forms.py', "    G['prec'], G['out'], G['delta'], G['td'] = args.prec, args.out, args.delta, args.td",
        "    G['prec'], G['out'] = args.prec, args.out")
```

- [ ] **Step 10: Remove the td option from cert_run and cert_box, and read td forms in cert_box**

```python
replace('scripts/cert_run.py', "Usage, on one GPU node (local forms in Delta and td, tm_local_forms.py --delta --td):",
        "Usage, on one GPU node:")
replace('scripts/cert_run.py', "    ap.add_argument('--td', action='store_true', help='td GEMMs for dd local forms')\n", "")
replace('scripts/cert_run.py', "k_max=kmax, a=a, td=args.td, hmax=hmaxs,", "k_max=kmax, a=a, hmax=hmaxs,")

replace('scripts/cert_box.py', """of abs(f) on the box and the parts of the remainder, as `scripts/tm_ref.py` does.""",
        """of abs(f) on the box and the parts of the remainder.""")
replace('scripts/cert_box.py', "local forms; and f(c) against --fref (values from tm_box.py or tm_ref.py).",
        "local forms; and f(c) against --fref (values from tm_box.py).")
replace('scripts/cert_box.py', """        --local runs_tm/local_r20i --centers 0 1 2 --orders 16 20 24 --ny-extra 6 \\\\
        --deltas 0.005 0.01 0.02 0.03 --fref=-3.03e-19,-1.33e-18,-2.23e-19""",
        """        --local /n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td --centers 0 \\\\
        --orders 28 --kmax 24 --deltas 0.01 0.02""")
replace('scripts/cert_box.py', """    # Each block e has the factor abs(y_2)^(2e), or Delta^j with Delta = abs(y_2)^2 - s2.
    if 's2' in local.files:
        tag, factor = 'j', abs(y[2]) ** 2 - tc.dd_mp(local['s2']).real
    else:
        tag, factor = 'e', abs(y[2]) ** 2
    for e in range(4):
        q = local[f'base_{tag}{e}']""", """    # Block j has the factor Delta^j, with Delta = abs(y_2)^2 - s2.
    factor = abs(y[2]) ** 2 - tc.dd_mp(local['s2']).real
    for e in range(4):
        q = local[f'base_j{e}']""")
replace('scripts/cert_box.py', """                v = mp.mpc(mp.mpf(q[0, i, j]) + mp.mpf(q[1, i, j]),
                           mp.mpf(q[2, i, j]) + mp.mpf(q[3, i, j]))""",
        """                v = mp.mpc(mp.mpf(q[0, i, j]) + mp.mpf(q[1, i, j]) + mp.mpf(q[2, i, j]),
                           mp.mpf(q[3, i, j]) + mp.mpf(q[4, i, j]) + mp.mpf(q[5, i, j]))""")
replace('scripts/cert_box.py', """    ap.add_argument('--td', type=int, nargs='+', default=[0],
                    help='1: the triple-double products for the compositions')
""", "")
replace('scripts/cert_box.py', """        for n, a, td in [(n, a, td) for n in args.orders for a in args.a for td in args.td]:
            a, td = a or None, bool(td)""", """        for n, a in [(n, a) for n in args.orders for a in args.a]:
            a = a or None""")
replace('scripts/cert_box.py', "args.kmax, a, td=td, hmax=hmax)", "args.kmax, a, hmax=hmax)")
replace('scripts/cert_box.py', """                                                      n + args.ny_extra, args.kmax, a, td=td,
                                                      hmax=hmax)""", """                                                      n + args.ny_extra, args.kmax, a,
                                                      hmax=hmax)""")
replace('scripts/cert_box.py', "f'{a or 2 / n:.3f}, td {td}{extra}: f(c) '", "f'{a or 2 / n:.3f}{extra}: f(c) '")
```

The p0 check of `cert_box.py` read only 4 limbs, but all the local forms that `box_f` now accepts have 6. The edit above reads all six.

- [ ] **Step 11: Delete the XLA prototype and tm_ref**

Run: `cd $WT && git rm -q quartic/tm_gpu.py scripts/bench_tm_gpu.py tests/test_tm_gpu.py scripts/tm_ref.py`

- [ ] **Step 12: Static checks**

```bash
cd $WT
F="quartic/tm_cert.py quartic/tm_frame.py quartic/tm_forms.py quartic/ddgemm.py scripts/cert_run.py scripts/cert_box.py scripts/tm_local_forms.py tests/test_tm_host.py"
python3 -m py_compile $F
uvx pyflakes $F
grep -n -E "\b(implicit_h|offset_models|monomial_mats|compose|reexpand|delta_form|p_value|tm_const|mode_j|chain|tm_ref|tm_gpu|to_dd_complex|td_of_dd|tcabs)\b|_\{tag\}|args\.td|td=td|\.ha\b|\.a6\b|\.a4\b|\.h_err\b" $F scripts/cert_check.py
```
Expected:
- `py_compile` gives no output;
- `pyflakes` reports no `undefined name`;
- `grep` finds only function names that end in `_td` (for example `implicit_h_td`), or the word "compose" in prose.

Remove each `imported but unused` that this task caused. For example, `upper` in `tm_frame.py` stays in use, but check it.

- [ ] **Step 13: Commit, push, and check on the GPU**

```bash
cd $WT
git status --short
git add quartic/tm_cert.py quartic/tm_frame.py quartic/tm_forms.py quartic/ddgemm.py scripts/cert_run.py scripts/cert_box.py scripts/tm_local_forms.py tests/test_tm_host.py
git commit -m "Remove the dd chain and the Q_e form of the box, which the run does not use

The box keeps only the td chain in Delta, which made the certificate. The td functions
now hold the full description that the dd functions had. box_f stops with a clear error
on local forms that are not in Delta and td. tm_forms and tm_local_forms write only
that form. Six unused functions, the dd frame constants, the XLA prototype tm_gpu.py and
scripts/tm_ref.py go; the tag pre-tmgpu keeps them. cert_box reads all six parts of the
td forms in its check of p0."
git push
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/mpcm_tmgpu && git fetch origin tm-library && git checkout -q --detach origin/tm-library && git log --oneline -1 && cd ~/machine_percision_cy_metric && TESTS="tests.test_tmc tests.test_tmtd tests.test_ddgemm tests.test_tm_host tests.test_cert_compare" REFS=297 TAG=297 sbatch --export=ALL submit_tmgpu_check.sh'
```
Watch the job with a background loop. Expected at the end of `slurm-<id>.out`:
- the path line under `~/mpcm_tmgpu`;
- each test suite passes, and `check_old_local_forms` prints its line;
- `all equal bit for bit`;
- `RESULT: PASS`.

---

### Task 3: Move the GPU core into tmgpu

**Files:**
- Create: `tmgpu/__init__.py`, `tmgpu/cuda.py`, `tmgpu/tests/__init__.py`, `tmgpu/tests/test_boundary.py`
- Move: `quartic/tmc.py` to `tmgpu/tm.py`, `quartic/tmtd.py` to `tmgpu/tmtd.py`, `quartic/ddgemm.py` to `tmgpu/gemm.py`, `tests/test_tmc.py` to `tmgpu/tests/test_tm.py`, `tests/test_tmtd.py` to `tmgpu/tests/test_tmtd.py`, `tests/test_ddgemm.py` to `tmgpu/tests/test_gemm.py`
- Modify: `quartic/tm_cert.py`, `scripts/cert_box.py`, `tests/test_tm_host.py`

**Interfaces:**
- Produces:
  - `tmgpu.cuda.DD_PRIMITIVES` (str), `tmgpu.cuda.OPTIONS` (tuple) and `tmgpu.cuda.module(code) -> cupy.RawModule`;
  - the modules `tmgpu.tm`, `tmgpu.tmtd` and `tmgpu.gemm` with the same functions as before. `tm._SRC`, `tmtd.TD_SRC`, `gemm._SRC` and `gemm._TT_SRC` hold text that is identical, character for character, to the text of `main`.

- [ ] **Step 1: Write the boundary test, which checks no module yet**

Write `tmgpu/__init__.py`:

```python
"""Taylor models in double-double and triple-double on the GPU, with a strict bound of
each round-off, and batched matrix products with bounds.

The package imports no submodule by itself; import each one by name, for example
`from tmgpu import tm`.
"""
```

Write an empty `tmgpu/tests/__init__.py`, and write `tmgpu/tests/test_boundary.py`:

```python
"""tmgpu imports only the standard library, numpy, cupy, mpmath, flint and itself."""
import ast
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ALLOWED = set(sys.stdlib_module_names) | {'numpy', 'cupy', 'mpmath', 'flint', 'tmgpu'}


def imported(path):
    """The top-level names of the absolute imports of one file."""
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split('.')[0])
    return names


def test_boundary():
    bad = {}
    for path in sorted(PKG.rglob('*.py')):
        if 'tests' in path.relative_to(PKG).parts:
            continue
        extra = imported(path) - ALLOWED
        if extra:
            bad[str(path.relative_to(PKG))] = sorted(extra)
    assert not bad, f'tmgpu imports outside its boundary: {bad}'


def test_boundary_finds_a_planted_import():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'x.py'
        p.write_text('import numpy\nfrom quartic.cert import box\nfrom . import tm\n')
        assert imported(p) - ALLOWED == {'quartic'}


if __name__ == '__main__':
    for fn in (test_boundary, test_boundary_finds_a_planted_import):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all boundary tests pass')
```

On the cluster, before Step 2, `test_boundary` would pass with no module to check, and `test_boundary_finds_a_planted_import` would pass. The test starts to check real modules when they arrive in Step 2.

- [ ] **Step 2: Move the files**

```bash
cd $WT
git mv quartic/tmc.py tmgpu/tm.py
git mv quartic/tmtd.py tmgpu/tmtd.py
git mv quartic/ddgemm.py tmgpu/gemm.py
git mv tests/test_tmc.py tmgpu/tests/test_tm.py
git mv tests/test_tmtd.py tmgpu/tests/test_tmtd.py
git mv tests/test_ddgemm.py tmgpu/tests/test_gemm.py
```

- [ ] **Step 3: Split the dd primitives into tmgpu/cuda.py with identical text**

```python
from pathlib import Path
s = Path('tmgpu/tm.py').read_text()
head = "_SRC = r'''"
start = s.index(head) + len(head)
marker = s.index('// Z[k] = sum over t', start)
prefix = s[start:marker]
Path('tmgpu/cuda.py').write_text('''"""The CUDA source of the dd primitives, and the compiler options of every tmgpu kernel.

two_sum and quick_two_sum are exact; the bounds of dd_add and dd_mul are in the docstring
of `tmgpu.tm`. --fmad=false keeps the compiler from a merge of a product and a sum into
one FMA, which would change the error term that two_sum computes.
"""
import cupy as cp

OPTIONS = ('--fmad=false', '-std=c++14')

DD_PRIMITIVES = r\'\'\'''' + prefix + """'''


def module(code):
    \"\"\"The CuPy module of CUDA source compiled with OPTIONS.\"\"\"
    return cp.RawModule(code=code, options=OPTIONS)
""")
s = s[:s.index(head)] + "_SRC = DD_PRIMITIVES + r'''" + s[marker:]
Path('tmgpu/tm.py').write_text(s)
replace('tmgpu/tm.py', "import cupy as cp\n", "import cupy as cp\n\nfrom .cuda import DD_PRIMITIVES, module\n")
replace('tmgpu/tm.py', "        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))",
        "        _MOD = module(_SRC)")
```

Read the head of `tmgpu/cuda.py` and check that `DD_PRIMITIVES = r'''` starts on its own line and the source follows on the next line.

- [ ] **Step 4: Point tmtd and gemm at tmgpu.cuda**

```python
replace('tmgpu/tmtd.py', "from . import tmc\nfrom .tmc import _SRC as _DD_SRC, SAFE\n",
        "from . import tm\nfrom .cuda import DD_PRIMITIVES, module\nfrom .tm import SAFE\n")
replace('tmgpu/tmtd.py', "TD_SRC = _DD_SRC.split('// Z[k] = sum over t')[0] + r'''", "TD_SRC = DD_PRIMITIVES + r'''")
replace('tmgpu/tmtd.py', "        _MOD = cp.RawModule(code=TD_SRC, options=('--fmad=false', '-std=c++14'))",
        "        _MOD = module(TD_SRC)")
replace('tmgpu/gemm.py', "from .tmc import _SRC as _DD_SRC\n",
        "from .cuda import DD_PRIMITIVES, module\nfrom .tmtd import ERR_SAFE, TD_SRC\n")
replace('tmgpu/gemm.py', "_SRC = _DD_SRC.split('// Z[k] = sum over t')[0] + r'''", "_SRC = DD_PRIMITIVES + r'''")
replace('tmgpu/gemm.py', """        from .tmtd import TD_SRC
        _TT_MOD = cp.RawModule(code=TD_SRC + _TT_SRC, options=('--fmad=false', '-std=c++14'))""",
        "        _TT_MOD = module(TD_SRC + _TT_SRC)")
replace('tmgpu/gemm.py', "    from .tmtd import ERR_SAFE\n", "")
replace('tmgpu/gemm.py', "        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))",
        "        _MOD = module(_SRC)")
```

`tmtd` does not import `gemm`, so the imports at the top of `gemm` make no cycle.

- [ ] **Step 5: Rename the references in every user**

```python
files = ['tmgpu/tm.py', 'tmgpu/tmtd.py', 'tmgpu/gemm.py', 'tmgpu/tests/test_tm.py',
         'tmgpu/tests/test_tmtd.py', 'tmgpu/tests/test_gemm.py', 'quartic/tm_cert.py',
         'quartic/tm_frame.py', 'quartic/tm_forms.py', 'scripts/cert_box.py',
         'scripts/cert_run.py', 'scripts/cert_check.py', 'tests/test_tm_host.py']
pairs = [('quartic/tmc.py', 'tmgpu/tm.py'), ('quartic/tmtd.py', 'tmgpu/tmtd.py'),
         ('quartic/ddgemm.py', 'tmgpu/gemm.py'), ('quartic.tmc', 'tmgpu.tm'),
         ('quartic.tmtd', 'tmgpu.tmtd'), ('quartic.ddgemm', 'tmgpu.gemm'),
         ('tmc.', 'tm.'), ('ddgemm.', 'gemm.')]
for f in files:
    for old, new in pairs:
        replace_all(f, old, new)
replace('quartic/tm_cert.py', "from . import tmc, tmtd, ddgemm\n", "from tmgpu import tm, tmtd, gemm\n")
replace('tmgpu/tests/test_tm.py', "from quartic import tmc ", "from tmgpu import tm ")
replace('tmgpu/tests/test_tmtd.py', "from quartic import tmc, tmtd, ddgemm ", "from tmgpu import tm, tmtd, gemm ")
replace('tmgpu/tests/test_gemm.py', "from quartic import ddgemm ", "from tmgpu import gemm ")
for f in ['tmgpu/tests/test_tm.py', 'tmgpu/tests/test_tmtd.py', 'tmgpu/tests/test_gemm.py']:
    replace(f, "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
            "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))")
replace('scripts/cert_box.py', "from quartic import tmc, tm_cert as tc ", "from quartic import tm_cert as tc ")
replace('tests/test_tm_host.py', "from quartic import tmc, tm_cert as tc, tm_frame ",
        "from quartic import tm_cert as tc, tm_frame ")
replace('tests/test_tm_host.py', "import cert_cover as cc ",
        "from tmgpu import tm                                       # noqa: E402\nimport cert_cover as cc ")
```

The import lines keep their `# noqa: E402` comments, because each `replace` matches only the text before the spaces. If `replace` stops with a count of 0, read the import line and copy its exact text into the call.

- [ ] **Step 6: Static checks**

```bash
cd $WT
F="tmgpu/*.py tmgpu/tests/*.py quartic/tm_cert.py quartic/tm_frame.py quartic/tm_forms.py scripts/cert_box.py scripts/cert_run.py scripts/cert_check.py tests/test_tm_host.py"
python3 -m py_compile $F
uvx pyflakes $F
grep -n -E "\btmc\b|ddgemm|_DD_SRC|RawModule" $F
```
Expected:
- no `undefined name`;
- `RawModule` appears only in `tmgpu/cuda.py`;
- no `tmc`, `ddgemm` or `_DD_SRC`.

- [ ] **Step 7: Commit, push, and check on the GPU, with the hashes**

```bash
cd $WT
git status --short
git add tmgpu quartic/tm_cert.py scripts/cert_box.py tests/test_tm_host.py
git status --short
git commit -m "Move the Taylor-model core into the package tmgpu

tmc, tmtd and ddgemm become tmgpu.tm, tmgpu.tmtd and tmgpu.gemm, with their tests. The
CUDA source of the dd primitives is the public constant tmgpu.cuda.DD_PRIMITIVES; tmtd
and gemm cut it from the source of tmc before. Every kernel compiles through
tmgpu.cuda.module with the same options. A test checks that tmgpu imports only the
standard library, numpy, cupy, mpmath and flint."
git push
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/mpcm_tmgpu && git fetch origin tm-library && git checkout -q --detach origin/tm-library && git log --oneline -1 && cd ~/machine_percision_cy_metric && TESTS="tmgpu.tests.test_boundary tmgpu.tests.test_tm tmgpu.tests.test_tmtd tmgpu.tests.test_gemm tests.test_tm_host tests.test_cert_compare" REFS=297 TAG=297 HASHES=1 sbatch --export=ALL submit_tmgpu_check.sh'
```
Expected: all test suites pass, `CUDA SOURCES EQUAL`, `all equal bit for bit`, and `RESULT: PASS`.

---

### Task 4: The host helpers and the tools of tmgpu

**Files:**
- Create: `tmgpu/host.py`, `tmgpu/tools.py`, `tmgpu/tests/test_host.py`
- Modify: `quartic/tm_frame.py`, `quartic/tm_cert.py`, `scripts/cert_cover.py`, `scripts/cert_run.py`, `scripts/cert_check.py`, `scripts/cert_box.py`, `tests/test_tm_host.py`

**Interfaces:**
- Produces:
  - `tmgpu.host` (CPU only): `PREC`, `precision`, `exact`, `upper`, `lower`, `to_dd`, `to_td`, `mp_dd`, `dd_mp`, `dd_frac`, `frac_dd(re, im, limbs=2)`;
  - `tmgpu.tools`: `space(nvars, n)`, `sup_bound(sp4, f, d) -> (sup, c0, pb)`, `series_order(eta, n, eps)`, `power(sp, x, m, cache)`, `conj(x)`, `host_dd(x)`, `gpu_dd(vals)`.

- [ ] **Step 1: Write the test of the host helpers, which fails first**

Write `tmgpu/tests/test_host.py`, with the two checks of `tests/test_tm_host.py`:

```python
"""Test of tmgpu.host: exact rationals to dd and td, and the bounds of an Arb ball."""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmgpu import host                                       # noqa: E402

RNG = np.random.default_rng(20261010)


def test_frac_dd():
    worst = [0.0, 0.0]
    for _ in range(2000):
        re = Fraction(float(RNG.normal())) + Fraction(float(RNG.normal())) * Fraction(1, 2 ** 70)
        im = Fraction(float(RNG.normal())) * Fraction(1, 3)
        for k, limbs in enumerate((2, 3)):
            v, err = host.frac_dd(re, im, limbs)
            gap = (abs(re - sum(Fraction(x) for x in v[:limbs]))
                   + abs(im - sum(Fraction(x) for x in v[limbs:])))
            assert gap <= Fraction(err), (limbs, float(gap), err)
            worst[k] = max(worst[k], float(gap) / err)
    print(f'  frac_dd: largest error over its bound {worst[0]:.2f} (dd), {worst[1]:.2f} (td)',
          flush=True)


def test_lower_upper():
    from flint import arb
    with host.precision():
        assert host.lower(arb(1, 0.6)) == 0.0
        assert host.lower(arb(-1, 0.6)) == 0.0
        x = arb(3, 1e-20)
        assert 0 < host.lower(x) <= 3 - 1e-20
        assert host.upper(x) >= 3 + 1e-20
    print('  lower: 0 for a wide ball, a lower bound for a narrow one', flush=True)


if __name__ == '__main__':
    for fn in (test_frac_dd, test_lower_upper):
        print(fn.__name__, flush=True)
        fn()
    print('all host tests pass')
```

Then remove the two old checks from `tests/test_tm_host.py`:

```python
remove_defs('tests/test_tm_host.py', ['check_frac_dd', 'check_lower'])
replace('tests/test_tm_host.py', "    check_frac_dd()\n", "")
replace('tests/test_tm_host.py', "    check_lower()\n", "")
```

- [ ] **Step 2: Write tmgpu/host.py**

Write `tmgpu/host.py`. The functions are those of `quartic/tm_frame.py` and `quartic/tm_cert.py`, with no change of any expression:

```python
"""Host numbers for the GPU models, on the CPU: Arb balls and exact rationals to dd or td
with a bound of each error, and mpmath numbers to and from dd.

A dd list is [re_hi, re_lo, im_hi, im_lo] and a td list is [re0, re1, re2, im0, im1, im2].
After the rounding, the dd or td number is the exact value that the GPU uses, and the
bound of its distance to the true number goes into a remainder.
"""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp
from flint import acb, arb, ctx

# The default precision of `precision`, in bits.
PREC = 512


class precision:
    """`with precision():` runs ball arithmetic at PREC bits and restores the old setting."""

    def __init__(self, prec=PREC):
        self.prec = prec

    def __enter__(self):
        self.old = ctx.prec
        ctx.prec = self.prec

    def __exit__(self, *exc):
        ctx.prec = self.old


def exact(v):
    """A dd list [re_hi, re_lo, im_hi, im_lo] as an exact acb (inside `precision`)."""
    return acb(arb(v[0]) + arb(v[1]), arb(v[2]) + arb(v[3]))


def upper(x):
    """An upper bound of abs(x) for an arb or acb ball, as a float."""
    a = abs(x)
    return (abs(float(a.mid())) + float(a.rad())) * (1 + 1e-12) + 1e-300


def lower(x):
    """A lower bound of abs(x) for an arb or acb ball, as a float (0 if it may vanish).

    With mid > 2 rad the FP64 round-off of mid - rad is below 3 u of it, which the
    factor 1 - 1e-12 covers; a wider ball gives 0.
    """
    a = abs(x)
    m, r = float(a.mid()), float(a.rad())
    if not m > 2 * r:
        return 0.0
    return (m - r) * (1 - 1e-12)


def to_dd(x):
    """The dd rounding of the midpoint of an acb ball, and a bound of its distance to x."""
    out, err = [], 0.0
    for part in (x.real, x.imag):
        mid = part.mid()
        h = float(mid)
        lo = float(mid - arb(h))
        out += [h, lo]
        err += upper(part - arb(h) - arb(lo))
    return out, err


def to_td(x):
    """The td rounding of the midpoint of an acb ball (6 floats), and a bound of its
    distance to x."""
    out, err = [], 0.0
    for part in (x.real, x.imag):
        mid = part.mid()
        h = float(mid)
        m = float(mid - arb(h))
        lo = float(mid - arb(h) - arb(m))
        out += [h, m, lo]
        err += upper(part - arb(h) - arb(m) - arb(lo))
    return out, err


def mp_dd(z):
    """An mpmath complex as dd parts [re_hi, re_lo, im_hi, im_lo]."""
    z = mp.mpc(z)
    rh, ih = float(z.real), float(z.imag)
    return [rh, float(z.real - rh), ih, float(z.imag - ih)]


def dd_mp(v):
    return mp.mpc(mp.mpf(v[0]) + mp.mpf(v[1]), mp.mpf(v[2]) + mp.mpf(v[3]))


def dd_frac(v):
    """A dd list [re_hi, re_lo, im_hi, im_lo] as the exact rationals (re, im)."""
    return Fraction(float(v[0])) + Fraction(float(v[1])), Fraction(float(v[2])) + Fraction(float(v[3]))


def frac_dd(re, im, limbs=2):
    """The exact complex rational re + i im rounded to dd (limbs 2) or td (limbs 3), as
    [re limbs, im limbs], and an upper bound of the distance (abs re + abs im)."""
    out, err = [], Fraction(0)
    for x in (re, im):
        rest = x
        for _ in range(limbs):
            v = float(rest)
            out.append(v)
            rest -= Fraction(v)
        err += abs(rest)
    return out, float(err) * (1 + 1e-12) + 1e-300
```

Before the commit, compare each function with its source, for example with `diff <(sed -n '/^def lower/,/^def to_dd/p' quartic/tm_frame.py) <(sed -n '/^def lower/,/^def to_dd/p' tmgpu/host.py)`. Do this before Step 4 removes the functions from `tm_frame.py`. The only differences must be blank lines.

- [ ] **Step 3: Write tmgpu/tools.py**

```python
"""Helpers on batches of Taylor models of `tmgpu.tm`: one Space for each size, the bound
of a model on its box, the order of a series, cached powers, the conjugate model, and CuPy
dd arrays to and from mpmath."""
from __future__ import annotations

import math

import numpy as np
import cupy as cp

from . import tm
from .host import dd_mp, mp_dd

_SPACES = {}


def space(nvars, n):
    """One Space for each (nvars, n), so that its tables are built once."""
    if (nvars, n) not in _SPACES:
        _SPACES[nvars, n] = tm.Space(nvars, n)
    return _SPACES[nvars, n]


def sup_bound(sp4, f, d):
    """abs(f) <= abs(f(c)) + B(f - f(c)) on the box delta d + the remainder of f."""
    c0 = tm.abs_ub(f.c[:, :1, :])[0]
    var = f.c.copy()
    var[:, 0, :] = 0
    pb = tm.poly_bound(sp4, tm.TM(var, f.rem, d), d)
    return (c0 + pb + f.rem) * tm.SAFE + tm.TINY, c0, pb


def series_order(eta, n, eps):
    """The number of terms with eta^(k+1) below eps for the largest eta of the batch: at
    least 4 and at most n, and n if the largest eta is not in (0, 0.5)."""
    e = float(cp.max(eta))
    if not 0 < e < 0.5:
        return n
    return int(min(n, max(4, math.ceil(math.log(eps) / math.log(e)) - 1)))


def power(sp, x, m, cache):
    """x^m by squares; cache is a dict that holds {1: x} and keeps every power."""
    if m not in cache:
        half = power(sp, x, m // 2, cache)
        cache[m] = (tm.mul(sp, half, half) if m % 2 == 0
                    else tm.mul(sp, power(sp, x, m - 1, cache), x))
    return cache[m]


def conj(x):
    """The model with conjugate coefficients: for a model of g(tau), the model of conj(g)
    as a function of sigma = conj(tau), on the same monomials."""
    c = x.c.copy()
    c[2:] = -c[2:]
    return tm.TM(c, x.rem.copy(), x.delta)


def host_dd(x):
    """A CuPy dd array (4, B) as a list of mpmath complex numbers."""
    h = cp.asnumpy(x)
    return [dd_mp(h[:, i]) for i in range(h.shape[1])]


def gpu_dd(vals):
    """A list of mpmath complex numbers as a CuPy dd array (4, B)."""
    return cp.asarray(np.array([mp_dd(v) for v in vals]).T)
```

`sup_bound` uses `tm.SAFE` and `tm.TINY`. In `quartic/tm_cert.py`, `SAFE` and `TINY` are the same objects, so the floats do not change.

- [ ] **Step 4: Make tm_frame use tmgpu.host**

```python
remove_defs('quartic/tm_frame.py', ['PREC', 'precision', 'exact', 'upper', 'lower', 'to_dd', 'to_td'])
replace('quartic/tm_frame.py', """# t^4 of a dd center needs 424 bits, so at 512 bits the center values are exact balls;
# then a center with -(1 + t_0^4 + t_1^4) on the negative real axis keeps a narrow root.
""", "")
replace('quartic/tm_frame.py', "from flint import acb, acb_mat, arb, ctx\n", """from flint import acb, acb_mat, arb, ctx

from tmgpu.host import PREC, exact, lower, precision, to_dd, to_td

# t^4 of a dd center needs 424 bits, so at PREC = 512 bits the center values are exact
# balls; a center with -(1 + t_0^4 + t_1^4) on the negative real axis keeps a narrow root.
""")
```

If `pyflakes` (Step 7) reports a name of this import as unused, remove it. If it reports an undefined name, add it.

- [ ] **Step 5: Make tm_cert use tmgpu.host and tmgpu.tools**

```python
remove_defs('quartic/tm_cert.py', ['mp_dd', 'dd_mp', 'dd_frac', 'frac_dd', '_host_dd', '_to_dd',
            '_series_order', '_power', 'conj_tm', '_SPACES', '_space', 'sup_bound'])
replace('quartic/tm_cert.py', "from tmgpu import tm, tmtd, gemm\n", """from tmgpu import tm, tmtd, gemm
from tmgpu.host import (dd_frac, dd_mp, exact, frac_dd, mp_dd, precision, to_dd as ball_dd,
                        to_td as ball_td, upper)
from tmgpu.tools import conj, gpu_dd, host_dd, power, series_order, space
""")
replace('quartic/tm_cert.py', "    from .tm_frame import exact, precision, to_dd as ball_dd, to_td as ball_td, upper\n", "")
replace('quartic/tm_cert.py', "# from ball arithmetic (quartic.tm_frame), so no bound", "# from ball arithmetic (tmgpu.host), so no bound")
replace('quartic/tm_cert.py', "TINY = tm.TINY\n", """TINY = tm.TINY
# The series of r and p~ stop at eta^(k+1) < SERIES_EPS: psi takes the error of x about
# 2e3 times, and the Cauchy estimate of g about 1e5 times.
SERIES_EPS = 1e-34
""")
for old, new in [('_series_order(eta_r, n)', 'series_order(eta_r, n, SERIES_EPS)'),
                 ('_series_order(eta_p, n)', 'series_order(eta_p, n, SERIES_EPS)'),
                 ('_space(', 'space('), ('conj_tm(', 'conj('), ('_power(', 'power('),
                 ('_host_dd(', 'host_dd('), ('_to_dd(', 'gpu_dd(')]:
    print(old, replace_all('quartic/tm_cert.py', old, new))
```

Read the printed counts. Each count must be at least 1, and the two `_series_order` counts must be exactly 1.

- [ ] **Step 6: Update the scripts and tests/test_tm_host.py**

```python
replace('scripts/cert_run.py', "from quartic import tm_cert as tc ",
        "from tmgpu.tools import sup_bound                           # noqa: E402\nfrom quartic import tm_cert as tc ")
replace('scripts/cert_run.py', "tc.sup_bound(", "sup_bound(")
replace('scripts/cert_check.py', "import cert_cover as cc ",
        "from tmgpu.tools import sup_bound                           # noqa: E402\nimport cert_cover as cc ")
assert replace_all('scripts/cert_check.py', "tc.sup_bound(", "sup_bound(") == 2
replace('scripts/cert_box.py', "from quartic import tm_cert as tc ",
        "from tmgpu.host import dd_mp                                # noqa: E402\nfrom tmgpu.tools import sup_bound                           # noqa: E402\nfrom quartic import tm_cert as tc ")
replace('scripts/cert_box.py', "tc.sup_bound(", "sup_bound(")
print('dd_mp in cert_box:', replace_all('scripts/cert_box.py', "tc.dd_mp(", "dd_mp("))
replace('scripts/cert_cover.py', "    from quartic.tm_frame import frame, lower, precision, upper\n",
        "    from quartic.tm_frame import frame\n    from tmgpu.host import lower, precision, upper\n")
replace('tests/test_tm_host.py', "from tmgpu import tm ", "from tmgpu import host, tm ")
for old, new in [('tc.dd_frac(', 'host.dd_frac('), ('tc.frac_dd(', 'host.frac_dd('),
                 ('tc.dd_mp(', 'host.dd_mp('), ('tm_frame.lower(', 'host.lower('),
                 ('tm_frame.upper(', 'host.upper('), ('tm_frame.precision(', 'host.precision(')]:
    print(old, replace_all('tests/test_tm_host.py', old, new))
```

`cert_cover.py` must not import `tmgpu.tools`, because the cover runs on CPU nodes with no CuPy.

- [ ] **Step 7: Static checks**

```bash
cd $WT
F="tmgpu/*.py tmgpu/tests/*.py quartic/tm_frame.py quartic/tm_cert.py scripts/cert_cover.py scripts/cert_run.py scripts/cert_check.py scripts/cert_box.py tests/test_tm_host.py"
python3 -m py_compile $F
uvx pyflakes $F
grep -n -E "\b(_space|_power|conj_tm|_host_dd|_to_dd|_series_order|_SPACES)\b|tc\.(sup_bound|dd_mp|dd_frac|frac_dd)|tm_frame\.(lower|upper|precision)" $F
grep -n "cupy" tmgpu/host.py
```
Expected: no `undefined name`, and neither `grep` finds a line.

- [ ] **Step 8: Commit, push, and check on the GPU**

```bash
cd $WT
git status --short
git add tmgpu/host.py tmgpu/tools.py tmgpu/tests/test_host.py quartic/tm_frame.py quartic/tm_cert.py scripts/cert_cover.py scripts/cert_run.py scripts/cert_check.py scripts/cert_box.py tests/test_tm_host.py
git commit -m "Move the host conversions and the model helpers into tmgpu.host and tmgpu.tools

tmgpu.host holds the conversions of Arb balls and exact rationals to dd and td, with
their error bounds, and runs on the CPU only. tmgpu.tools holds the Space cache, the
bound of a model on its box, the series order, cached powers, the conjugate model and
the CuPy dd arrays to and from mpmath. The quartic code imports them; the series order
of the box takes SERIES_EPS = 1e-34 as an argument. The tests of frac_dd and lower move
to tmgpu/tests/test_host.py."
git push
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/mpcm_tmgpu && git fetch origin tm-library && git checkout -q --detach origin/tm-library && git log --oneline -1 && cd ~/machine_percision_cy_metric && TESTS="tmgpu.tests.test_boundary tmgpu.tests.test_tm tmgpu.tests.test_tmtd tmgpu.tests.test_gemm tmgpu.tests.test_host tests.test_tm_host tests.test_cert_compare" REFS=297 TAG=297 sbatch --export=ALL submit_tmgpu_check.sh'
```
Expected: all suites pass, `all equal bit for bit`, and `RESULT: PASS`.

---

### Task 5: The package quartic.cert, with the cover, and no JAX

**Files:**
- Create: `quartic/cert/__init__.py`, `quartic/cert/cover.py`, `tests/test_cert_imports.py`
- Move: `quartic/tm_frame.py` to `quartic/cert/frame.py`, `quartic/tm_forms.py` to `quartic/cert/forms.py`, `quartic/tm_cert.py` to `quartic/cert/box.py`
- Modify: `quartic/__init__.py`, `scripts/cert_cover.py`, `scripts/cert_run.py`, `scripts/cert_check.py`, `scripts/cert_box.py`, `scripts/cert_partition.py`, `scripts/tm_local_forms.py`, `tests/test_tm_host.py`

**Interfaces:**
- Produces:
  - `quartic.cert.frame` (`frame`, `kappa`);
  - `quartic.cert.forms` (`Poly2`, `local_forms`, `delta_split`, `chart_forms_arb`, `to_td_complex`);
  - `quartic.cert.box` (`Frames`, `make_frames`, `box_f`, `gradient_bounds`, `local_on_gpu`, `offset_consts`, `reexpand_weights`, `delta_const`, `_real_low`, `_check_local`);
  - `quartic.cert.cover` (`cpow4`, `may_meet_f`, `box_radius`, `dd`, `cell`);
  - `quartic/__init__.py` imports no submodule.

- [ ] **Step 1: Write the test of the imports, which fails first**

Write `tests/test_cert_imports.py`:

```python
"""The certificate code loads no JAX, and the packages come from this repository."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_no_jax():
    import quartic.cert.box                                   # noqa: F401
    import quartic.cert.cover                                 # noqa: F401
    import quartic.cert.forms                                 # noqa: F401
    import quartic.cert.frame                                 # noqa: F401
    assert 'jax' not in sys.modules, 'the certificate code loads JAX'


def test_paths():
    import quartic
    import tmgpu
    for mod in (quartic, tmgpu):
        assert Path(mod.__file__).resolve().is_relative_to(ROOT), mod.__file__


if __name__ == '__main__':
    for fn in (test_no_jax, test_paths):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all import tests pass')
```

Before Step 2 it fails with `ModuleNotFoundError: No module named 'quartic.cert'`.

- [ ] **Step 2: Move the modules, and make quartic/__init__.py import no submodule**

```bash
cd $WT
mkdir -p quartic/cert
git mv quartic/tm_frame.py quartic/cert/frame.py
git mv quartic/tm_forms.py quartic/cert/forms.py
git mv quartic/tm_cert.py quartic/cert/box.py
```

Write `quartic/cert/__init__.py`:

```python
"""The strict bound of sup abs(f) for the Fermat quartic: the frame of a box, the forms of
the potential, the box pipeline and the cover of the region F. The Taylor models come
from `tmgpu`."""
```

Write `quartic/__init__.py`:

```python
"""Machine-precision Calabi-Yau metric on the Fermat quartic.

The package imports no submodule by itself, so `quartic.cert` loads no JAX. Import each
submodule by name, for example `from quartic import sections`.
"""
```

No code uses `quartic.sections` or `quartic.blocks` without an explicit import. A search for the plan confirmed it, and `sections.py` builds no JAX array at import.

- [ ] **Step 3: Fix the imports inside the moved modules**

```python
replace('quartic/cert/forms.py', "from . import blocks\n", "from .. import blocks\n")
replace('quartic/cert/box.py', "from tmgpu.tools import conj, gpu_dd, host_dd, power, series_order, space\n",
        "from tmgpu.tools import conj, gpu_dd, host_dd, power, series_order, space\n\nfrom .forms import Poly2\nfrom .frame import frame, kappa\n")
replace('quartic/cert/box.py', "    from .tm_frame import frame\n", "")
assert replace_all('quartic/cert/box.py', "    from .tm_forms import Poly2\n", "") >= 1
replace('quartic/cert/box.py', "    from .tm_frame import kappa as _kappa\n", "")
replace('quartic/cert/box.py', "_kappa(", "kappa(")
```

- [ ] **Step 4: Move the library part of cert_cover into quartic/cert/cover.py**

```python
import ast
from pathlib import Path
src = Path('scripts/cert_cover.py').read_text()
lines = src.split('\n')
names = ['cpow4', 'may_meet_f', '_to_arb', '_ball', 'box_radius', 'dd', 'cell']
blocks = []
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        blocks.append('\n'.join(lines[node.lineno - 1:node.end_lineno]))
assert len(blocks) == len(names)
body = '\n\n\n'.join(blocks)
body = body.replace("    from quartic.tm_frame import frame\n    from tmgpu.host import lower, precision, upper\n", "")
Path('quartic/cert/cover.py').write_text('''"""The cells of the cover of the region F, for the strict bound of sup abs(f).

F = {abs(t_1) <= abs(t_0) <= abs(z_2) <= 1, arg t_0 in [0, pi/4], arg t_1 in [0, pi/2)}
in the chart z_3 = 1, with abs(z_2)^4 = abs(1 + t_0^4 + t_1^4). A cell is a cube of side s
in (Re t_0, Im t_0, Re t_1, Im t_1) with the exact corner idx s. `may_meet_f` is False
only if interval arithmetic (mpmath.iv) proves that the cell misses F. `box_radius`
proves, in ball arithmetic, the radius delta and the bound hmax of the box at the center
of a cell in the frame of `quartic.cert.frame`.
"""
import mpmath as mp
from mpmath import iv

from tmgpu.host import lower, precision, upper

from .frame import frame

# The corners and centers are short binary numbers, exact at 30 digits; a caller with a
# higher precision keeps it.
mp.mp.dps = max(mp.mp.dps, 30)
iv.dps = 30


''' + body + '\n')
remove_defs('scripts/cert_cover.py', names)
replace('scripts/cert_cover.py', """# The corners and centers are short binary numbers, exact at 30 digits; a caller with a
# higher precision keeps it.
mp.mp.dps = max(mp.mp.dps, 30)
iv.dps = 30
""", "\nfrom quartic.cert.cover import box_radius, cell, dd, may_meet_f      # noqa: E402\n")
replace('scripts/cert_cover.py', "in the frame of `quartic.tm_frame` (the frame of", "in the frame of `quartic.cert.frame` (the frame of")
```

`cover.py` keeps the global precision settings of mpmath that `cert_cover.py` had at import. The scripts that import the cover therefore still run at 30 digits or more, and the free numbers of the box (zeta, s_r, s_p) stay the same.

- [ ] **Step 5: Update the scripts and tests/test_tm_host.py**

```python
# cert_run: the box and the cover from quartic.cert; no scripts path; no JAX setting.
replace('scripts/cert_run.py', "from quartic import tm_cert as tc ", "from quartic.cert import box as tc ")
replace('scripts/cert_run.py', "import cert_cover as cc ", "from quartic.cert import cover as cc ")
replace('scripts/cert_run.py', "sys.path.insert(0, str(ROOT / 'scripts'))\n", "")
# cert_check keeps scripts/ on its path for tm_box, the independent reference.
replace('scripts/cert_check.py', "import cert_cover as cc ", "from quartic.cert import cover as cc ")
assert replace_all('scripts/cert_check.py', "from quartic import tm_cert as tc", "from quartic.cert import box as tc") == 3
replace('scripts/cert_box.py', "from quartic import tm_cert as tc ", "from quartic.cert import box as tc ")
replace('scripts/cert_box.py', "from quartic.tm_forms import Poly2 ", "from quartic.cert.forms import Poly2 ")
replace('scripts/cert_partition.py', "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))",
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))")
replace('scripts/cert_partition.py', "import cert_cover as cc ", "from quartic.cert import cover as cc ")
replace('scripts/tm_local_forms.py', "from quartic import tm_forms ", "from quartic.cert import forms as tm_forms ")
replace('tests/test_tm_host.py', "from quartic import tm_cert as tc, tm_frame ", "from quartic.cert import box as tc ")
replace('tests/test_tm_host.py', "import cert_cover as cc ", "from quartic.cert import cover as cc ")
replace('tests/test_tm_host.py', "sys.path.insert(0, str(ROOT / 'scripts'))\n", "")
for f in ['scripts/cert_run.py', 'scripts/cert_box.py', 'scripts/cert_check.py',
          'scripts/cert_partition.py', 'scripts/cert_cover.py', 'tests/test_tm_host.py']:
    replace_all(f, "# The quartic package imports JAX, which would take most of the GPU memory from CuPy.\n", "")
    replace(f, "os.environ.setdefault('JAX_PLATFORMS', 'cpu')\n", "")
```

After this step, run `uvx pyflakes` (Step 7). It reports `'os' imported but unused` and `'iv' imported but unused` where the removed lines were the last users. Remove those imports. If `tests/test_tm_host.py` still uses `tm_frame.` anywhere, import `from quartic.cert import frame as tm_frame` there.

- [ ] **Step 6: Point the docstrings at the new modules**

```bash
cd $WT
grep -rn -E "quartic[./](tm_cert|tm_frame|tm_forms|tmc|tmtd|ddgemm)|\btm_cert\.|\btm_frame\.|\btm_forms\.|scripts/cert_cover\.py|cert_cover\.(may_meet_f|box_radius|cell)" tmgpu quartic scripts tests docs/superpowers/specs/2026-10-11-tmgpu-split-design.md
```

Edit each hit in code and docstrings to the new name:
- `quartic.tm_cert` becomes `quartic.cert.box`; `quartic.tm_frame` becomes `quartic.cert.frame`; `quartic.tm_forms` becomes `quartic.cert.forms`;
- `cert_cover.may_meet_f`, `cert_cover.box_radius` and `cert_cover.cell` become `quartic.cert.cover.<name>`;
- `scripts/cert_cover.py` stays, where the text means the script that writes the cover file.

Leave the 2026-10-09 design spec and the spec of this work as they are: they describe the old names on purpose.

- [ ] **Step 7: Static checks**

```bash
cd $WT
F="quartic/__init__.py quartic/cert/*.py scripts/cert_*.py scripts/tm_local_forms.py tests/test_tm_host.py tests/test_cert_imports.py"
python3 -m py_compile $F
uvx pyflakes $F
grep -n -E "JAX_PLATFORMS|import cert_cover|quartic import tm_(cert|frame|forms)" $F
```
Expected: no `undefined name`, and `grep` finds no line.

- [ ] **Step 8: Commit, push, and check on the GPU**

```bash
cd $WT
git status --short
git add quartic/__init__.py quartic/cert scripts/cert_cover.py scripts/cert_run.py scripts/cert_check.py scripts/cert_box.py scripts/cert_partition.py scripts/tm_local_forms.py tests/test_tm_host.py tests/test_cert_imports.py
git status --short
git commit -m "Move the quartic certificate into the package quartic.cert

tm_frame, tm_forms and tm_cert become quartic.cert.frame, forms and box, and the cell
functions of scripts/cert_cover.py become quartic.cert.cover. The scripts import them
from the package and no longer put scripts/ on the path for the cover. quartic/__init__
imports no submodule, so the certificate code loads no JAX, and the scripts no longer
set JAX_PLATFORMS. A test checks both facts."
git push
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/mpcm_tmgpu && git fetch origin tm-library && git checkout -q --detach origin/tm-library && git log --oneline -1 && cd ~/machine_percision_cy_metric && TESTS="tests.test_cert_imports tmgpu.tests.test_boundary tmgpu.tests.test_tm tmgpu.tests.test_tmtd tmgpu.tests.test_gemm tmgpu.tests.test_host tests.test_tm_host tests.test_cert_compare" REFS=297 TAG=297 sbatch --export=ALL submit_tmgpu_check.sh'
```
Expected: `test_no_jax passes` and `test_paths passes` first, then all suites pass, `all equal bit for bit`, and `RESULT: PASS`.

---

### Task 6: The full equivalence test, the merge, the tag and the notes

**Files:**
- Modify, after the merge: `CLAUDE.md` in the main laptop checkout (it is ignored by git)
- Cluster: no new files

**Interfaces:**
- Consumes: the baseline of Task 1 and the scripts `submit_tmgpu_check.sh` and `submit_tmgpu_part.sh`.

- [ ] **Step 1: The full GPU check**

```bash
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/mpcm_tmgpu && git fetch origin tm-library && git checkout -q --detach origin/tm-library && git log --oneline -1 && cd ~/machine_percision_cy_metric && TESTS="tests.test_cert_imports tmgpu.tests.test_boundary tmgpu.tests.test_tm tmgpu.tests.test_tmtd tmgpu.tests.test_gemm tmgpu.tests.test_host tests.test_tm_host tests.test_cert_compare" REFS="$(cat runs_tm/equiv/refs.txt)" TAG=all HASHES=1 FULL=1 sbatch --export=ALL submit_tmgpu_check.sh'
```
Expected:
- all suites pass;
- `CUDA SOURCES EQUAL`;
- `all equal bit for bit` for the 4 reference centers;
- the cert_check lines report that all 300 points lie in a final box and all 300 dropped cells miss F again;
- `RESULT: PASS`.

- [ ] **Step 2: The partition check on the CPU**

```bash
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'cd ~/machine_percision_cy_metric && sbatch submit_tmgpu_part.sh'
```
Expected: `PARTITION OUTPUT EQUAL`, and the branch output reports a valid partition with the gradient check passed. The branch run also shows that the cover needs no CuPy.

- [ ] **Step 3: If a check fails, stop**

Do not merge. Report the line that failed and the job number. For a `DIFFERS` line of `cert_compare`, give its largest differences.

- [ ] **Step 4: Merge, tag and push**

```bash
cd /Users/yidiqi/Documents/Research/Machine_percision_CYmetric
git status --short
git checkout main
git merge --ff-only tm-library
git tag pre-tmgpu 8f0c1e8
git push origin main
git push origin pre-tmgpu
git log --oneline -8
```

- [ ] **Step 5: Update the cluster checkout and remove the cluster worktree**

Run this only when no certificate job runs (`squeue -u yidqi`):

```bash
ssh -o BatchMode=yes yidqi@login.rc.fas.harvard.edu 'squeue -u yidqi -h; cd ~/machine_percision_cy_metric && git pull --ff-only && git log --oneline -1 && git worktree remove ~/mpcm_tmgpu && git worktree list'
```

Then remove the laptop worktree through superpowers:finishing-a-development-branch.

- [ ] **Step 6: Record the layout in CLAUDE.md**

In the main laptop checkout, add this section to `CLAUDE.md` after the section "How to run the code" and before the subsection "The cluster":

```markdown
### The code of the strict bound

* `tmgpu/` is the general core, with no quartic code and no JAX: `tm` (dd Taylor
  models), `tmtd` (td Taylor models), `gemm` (dd and td matrix products with bounds),
  `cuda` (the dd primitives and the compiler options), `host` (Arb balls and rationals
  to dd or td, CPU only), `tools` (Space cache, sup_bound, series order, powers). Its
  tests are in `tmgpu/tests/`. It moves to its own repository later.
* `quartic/cert/` is the quartic certificate: `frame`, `forms`, `box` (the td chain
  in Delta) and `cover`. The scripts `cert_run.py`, `cert_check.py`,
  `cert_partition.py`, `cert_box.py`, `cert_cover.py` and `tm_local_forms.py` use it.
* `tm_local_forms.py` always writes the Delta form in td, so it has no `--delta` or
  `--td` options. `box_f` accepts only such files.
* The tag `pre-tmgpu` (8f0c1e8) holds the dd chain, the Q_e form, `tm_gpu.py` and
  `tm_ref.py`.
* `scripts/cert_compare.py A B --exact` compares two record files of `cert_run.py`.
```

Also replace each path `quartic/tmc.py`, `quartic/tmtd.py`, `quartic/ddgemm.py`, `quartic/tm_cert.py`, `quartic/tm_frame.py` and `quartic/tm_forms.py` in `CLAUDE.md` with its new path. Find them with `grep -n -E "quartic/(tmc|tmtd|ddgemm|tm_cert|tm_frame|tm_forms)|tm_gpu|tm_ref" CLAUDE.md`.

- [ ] **Step 7: Report**

Report to the user:
- the job numbers and the result of each check;
- the line counts, from `git diff --stat pre-tmgpu..main`;
- the two follow-up changes that change floats and wait for a decision: `dd_mul` in the published order, and powers without the CUDA `pow`.
