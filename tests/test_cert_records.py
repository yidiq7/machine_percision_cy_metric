"""Test of quartic.cert.records.load on small record files, on the CPU."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quartic.cert import records as rs                     # noqa: E402


def part(path, refs, target=5e-20, columns=rs.COLUMNS, commit='abc', hashes=True):
    rec = np.zeros((2 * len(refs), len(columns)))
    rec[:, 0] = np.repeat(refs, 2)
    extra = dict(commit=commit)
    if hashes:
        extra.update(local_refs=np.array(refs), local_sha=np.array([f'h{r}' for r in refs]))
    np.savez(path, records=rec, target=target, columns=np.array(columns), **extra)
    return str(path)


def fails(paths):
    try:
        rs.load(paths)
    except ValueError:
        return True
    return False


def test_join():
    with tempfile.TemporaryDirectory() as d:
        a, b = part(Path(d) / 'a.npz', [0, 1]), part(Path(d) / 'b.npz', [2])
        rec, target, commit, shas = rs.load([a, b])
        assert rec.shape == (6, len(rs.COLUMNS)) and target == 5e-20 and commit == 'abc'
        assert shas == {0: 'h0', 1: 'h1', 2: 'h2'}


def test_old_file():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / 'a.npz'
        np.savez(a, records=np.zeros((1, len(rs.COLUMNS))), target=5e-20,
                 columns=np.array(rs.COLUMNS))
        _, _, commit, shas = rs.load([str(a)])
        assert commit is None and shas == {}


def test_bad_parts():
    with tempfile.TemporaryDirectory() as d:
        a = part(Path(d) / 'a.npz', [0, 1])
        assert fails([a, part(Path(d) / 'b.npz', [1, 2])]), 'a reference center in two parts'
        assert fails([a, part(Path(d) / 'c.npz', [3], target=1e-19)]), 'two targets'
        assert fails([a, part(Path(d) / 'e.npz', [4], commit='xyz')]), 'two commits'
        cols = rs.COLUMNS[:7] + ('bound', 'delta') + rs.COLUMNS[9:]
        assert fails([part(Path(d) / 'f.npz', [5], columns=cols)]), 'columns in another order'


if __name__ == '__main__':
    for fn in (test_join, test_old_file, test_bad_parts):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all records tests pass')
