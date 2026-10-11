"""Compare two record files of scripts/cert_run.py, column by column.

For each column the script prints the largest absolute difference and the largest
difference relative to abs(A). Equal means the same bits: -0.0 differs from +0.0, and two
NaN are equal only with the same bits. With --exact the exit status is 0 only if the
targets and every column agree bit for bit.

Usage:
    uv run --extra gpu python scripts/cert_compare.py A.npz B.npz --exact
"""
import argparse
import sys

import numpy as np


def compare(a, b):
    """The column names, and for each column (name, largest absolute difference, largest
    relative difference, equal bit for bit)."""
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
                     bool(x.dtype == y.dtype and np.array_equal(
                         np.ascontiguousarray(x).view(np.uint8),
                         np.ascontiguousarray(y).view(np.uint8)))))
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
