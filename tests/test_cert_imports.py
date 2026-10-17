"""The certificate code loads no JAX, quartic comes from this repository, and tmgpu from the environment."""
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


if __name__ == '__main__':
    for fn in (test_no_jax, test_paths):
        fn()
        print(fn.__name__, 'passes', flush=True)
    print('all import tests pass')
