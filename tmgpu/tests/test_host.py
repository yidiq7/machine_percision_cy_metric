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
