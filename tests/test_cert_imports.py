"""The certificate code loads no JAX, quartic comes from this repository, and tmgpu from the environment.

The arb reference tests/box_reference.py loads no module of quartic.cert, tmgpu or cupy."""
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
    assert Path(quartic.__file__).resolve().is_relative_to(ROOT), quartic.__file__
    path = Path(tmgpu.__file__).resolve()
    assert not path.is_relative_to(ROOT / 'tmgpu'), path
    assert path.is_relative_to(Path(sys.prefix).resolve()), path


def test_reference_independent():
    import subprocess
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import tests.box_reference; '
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('tmgpu', 'cupy') "
            "or m == 'quartic.cert' or m.startswith('quartic.cert.')); "
            "print(' '.join(bad)); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, '-c', code, str(ROOT)], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, ('tests.box_reference loads', out.stdout, out.stderr)


if __name__ == '__main__':
    for fn in (test_no_jax, test_paths, test_reference_independent):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all import tests pass')
