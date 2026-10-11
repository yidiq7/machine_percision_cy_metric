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
