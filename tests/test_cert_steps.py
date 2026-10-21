"""Tests of the stage `compare` of scripts/cert_steps.py.

1. A reference at a point moved by 1e-25 fails at the factor 1e-9, and the reference at
   the point of the model passes.
2. An infinite remainder fails at the factor 1 and gives "no claim" at another factor; a
   NaN remainder fails at each factor.
3. A reference ball with a radius of 1e-2 of the remainder raises ValueError.
4. R2 is gap / (R - part): 1.25 for R = 1, a part 0.6 and a gap 0.5; inf when R - part <= 0.
5. load_models takes the parts of R2 that the stage `compare` uses, and no other key.
6. make_points: the even points at 0.999 r, the odd points inside the polydisc, the points
   0, 4, 8, 12 on the slice sigma = conj(tau), and the same points for the same file name.
7. The import of scripts/cert_steps.py loads no CuPy, tmgpu or quartic.cert.box.

Run on a GPU node: uv run --extra gpu --locked python -u -m tests.test_cert_steps
"""
import math
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from flint import acb, arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from tests import box_reference as br                     # noqa: E402
import cert_steps as cs                                    # noqa: E402

# A dd number c with a low limb, so that the exact sum of the limbs matters.
C_LIMBS = [0.3, 2.0 ** -60, -0.7, 2.0 ** -62]
X = [acb(0.004, -0.003), acb(0.002, 0.001)]


def linear_model(rem):
    """The model y0 = c + tau_0 in 2 variables, with the remainder rem."""
    coefs = np.array([C_LIMBS, [1.0, 0.0, 0.0, 0.0]]).T
    return cs.Model(c=coefs, rem=rem, delta=0.01, mons=np.array([[0, 0], [1, 0]]), parts={})


def test_point_moved():
    m = linear_model(1e-40)
    c = br.exact(C_LIMBS)
    good = cs.check_point(m, X, c + X[0], 1e-9)
    assert good['status'] == 'pass', good
    assert good['gap'] <= 1e-80, good
    moved = cs.check_point(m, X, c + (X[0] + arb(1e-25)), 1e-9)
    assert moved['status'] == 'fail', moved
    assert abs(moved['gap'] / 1e-25 - 1) < 1e-6 and moved['ratio'] > 1e14, moved
    print(f"  the point of the model: gap {good['gap']:.1e}, pass; moved by 1e-25: gap "
          f"{moved['gap']:.3e}, fail", flush=True)


def test_no_claim():
    c = br.exact(C_LIMBS)
    inf = linear_model(math.inf)
    assert cs.check_point(inf, X, c + X[0], 1.0)['status'] == 'fail'
    assert cs.check_point(inf, X, c + X[0], 1e-3)['status'] == 'no claim'
    nan = linear_model(math.nan)
    for factor in (1.0, 1e-3, 1.5):
        out = cs.check_point(nan, X, c + X[0], factor)
        assert out['status'] == 'fail', (factor, out)
    print('  infinite R: fail at the factor 1, no claim at 1e-3; NaN R: fail at each factor',
          flush=True)


def test_wide_reference():
    m = linear_model(1e-20)
    ref = br.exact(C_LIMBS) + X[0]
    narrow = ref + acb(arb(0, 1e-24))
    assert cs.check_point(m, X, narrow, 1.0)['status'] == 'pass'
    wide = ref + acb(arb(0, 1e-22))
    try:
        cs.check_point(m, X, wide, 1.0)
    except ValueError as e:
        print(f'  a reference radius of 1e-2 R raises ValueError ({e}); 1e-4 R passes',
              flush=True)
        return
    raise AssertionError('a reference radius of 1e-2 R passed')


def test_part_seen():
    zero = np.zeros((4, 1))
    m = cs.Model(c=zero, rem=1.0, delta=0.01, mons=np.array([[0, 0]]),
                 parts={'seen': 0.6, 'hidden': 0.4, 'all': 1.0})
    out = cs.check_point(m, X, acb(0.5), 1.0)
    assert out['status'] == 'pass' and out['gap'] == 0.5 and out['ratio'] == 0.5, out
    r2 = out['r2']
    assert sorted(r2) == ['all', 'hidden', 'seen'], r2
    assert abs(r2['seen'] - 1.25) < 1e-12, r2
    assert abs(r2['hidden'] - 0.5 / 0.6) < 1e-12, r2
    assert r2['all'] == math.inf, r2
    print(f"  R2: part 0.6 gives {r2['seen']}, part 0.4 gives {r2['hidden']:.4f}, part 1.0 "
          f"gives {r2['all']}", flush=True)


def test_load_models():
    arrays = {}
    vals = [1e-22, 2e-22, 3e-22, 4e-22, 5e-22, 6e-22]

    def model(name, limbs, nvars, parts=()):
        arrays[f'm_{name}_c'] = np.ones((limbs, 3))
        arrays[f'm_{name}_rem'] = 1e-20
        arrays[f'm_{name}_delta'] = 0.04
        arrays[f'm_{name}_mons'] = np.zeros((3, nvars), dtype=np.int64)
        for i, key in enumerate(parts):
            arrays[f'm_{name}_part_{key}'] = vals[i]

    compose = ('abs', 'high', 'models', 'radius', 'round', 'sums')
    model('h', 6, 2)
    model('p0', 4, 4, compose)
    model('fk3', 4, 4, compose)
    model('fk12', 4, 4, compose)
    model('pt', 4, 4, ('bx', 'by', 'round', 'tail'))
    model('log_r2', 4, 4)
    for name in ('psi', 'xi', 'v', 'g01'):
        model(name, 4, 4)
    psi = {'F_k': 1e-23, 'rounding': 2e-23, 'tails': 3e-23, 'xi': 4e-23, 'xi_cut': 5e-23}
    arrays.update({f'psi_part_{key}': val for key, val in psi.items()})
    arrays.update(h_parts_rho_host=1e-31, h_parts_rho_residual=2e-31, h_parts_rho=3e-31,
                  xi_kap=4e-31, v_vol=5e-31)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'B0_x1.npz'
        np.savez(path, **arrays)
        models = cs.load_models(np.load(path))
    assert sorted(models) == sorted(['h', 'p0', 'fk3', 'fk12', 'pt', 'log_r2', 'psi', 'xi',
                                     'v', 'g01']), sorted(models)
    # The keys abs and sums of a composition, and bx and by of a product, are not parts.
    want = {'round': vals[4], 'high': vals[1], 'models': vals[2], 'radius': vals[3]}
    for name in ('p0', 'fk3', 'fk12'):
        assert models[name].parts == want, (name, models[name].parts)
    assert models['pt'].parts == {'round': vals[2], 'tail': vals[3]}, models['pt'].parts
    assert models['psi'].parts == psi, models['psi'].parts
    assert models['h'].parts == {'host': 1e-31, 'residual': 2e-31}, models['h'].parts
    assert models['xi'].parts == {'kap': 4e-31}, models['xi'].parts
    assert models['v'].parts == {'vol': 5e-31}, models['v'].parts
    assert models['log_r2'].parts == {} and models['g01'].parts == {}
    h = models['h']
    assert h.c.shape == (6, 3) and h.mons.shape == (3, 2) and h.rem == 1e-20
    assert h.delta == 0.04 and isinstance(h.rem, float)
    print('  load_models: 10 models; p0 and fk: round, high, models, radius; pt: tail, '
          'round; psi: 5 parts; h: host, residual; xi: kap; v: vol', flush=True)


def test_points():
    r = 0.0389
    pts = cs.make_points('B1_x1.npz', r, 16)
    assert pts.shape == (16, 4, 2) and pts.dtype == np.float64, pts.shape
    assert np.array_equal(pts, cs.make_points('B1_x1.npz', r, 16))
    assert not np.array_equal(pts, cs.make_points('B2_x1.npz', r, 16))
    mods = np.hypot(pts[..., 0], pts[..., 1])
    for i in range(16):
        on_slice = (np.array_equal(pts[i, 2:, 0], pts[i, :2, 0])
                    and np.array_equal(pts[i, 2:, 1], -pts[i, :2, 1]))
        assert on_slice == (i % 4 == 0), (i, pts[i])
        if i % 2 == 0:
            assert np.allclose(mods[i], 0.999 * r, rtol=1e-14, atol=0), (i, mods[i])
        else:
            assert np.all(mods[i] < r), (i, mods[i])
    assert mods[1::2].min() < 0.9 * r, mods[1::2].min()
    # The exact acb of each coordinate lies inside the polydisc of radius r.
    for i in range(16):
        for re, im in pts[i]:
            assert abs(acb(float(re), float(im))) < arb(r), (i, re, im)
    print(f'  make_points: 16 points, 4 on the slice, smallest odd modulus over r '
          f'{mods[1::2].min() / r:.3f}', flush=True)


def test_import_no_gpu():
    code = ('import sys; sys.path.insert(0, sys.argv[1] + "/scripts"); import cert_steps; '
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('cupy', 'tmgpu') "
            "or m in ('quartic.cert.box', 'cert_run', 'cert_check')); "
            "print(' '.join(bad)); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, '-c', code, str(ROOT)], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, ('scripts/cert_steps.py loads', out.stdout, out.stderr)
    print('  the import of scripts/cert_steps.py loads no cupy, tmgpu or quartic.cert.box',
          flush=True)


def main():
    assert ctx.prec == 320, ctx.prec
    failed = []
    for fn in (test_point_moved, test_no_claim, test_wide_reference, test_part_seen,
               test_load_models, test_points, test_import_no_gpu):
        try:
            fn()
        except Exception:
            failed.append(fn.__name__)
            print(f'{fn.__name__} FAILS', flush=True)
            traceback.print_exc()
            continue
        print(f'{fn.__name__} passes', flush=True)
    if failed:
        print(f'{len(failed)} cert_steps tests fail: {" ".join(failed)}', flush=True)
        sys.exit(1)
    print('all cert_steps tests pass', flush=True)


if __name__ == '__main__':
    main()
