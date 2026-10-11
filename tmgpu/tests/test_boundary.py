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
