"""The exact volumes of `quartic.volume` against mpmath and against a sample.

The sampled mean of v over the measure of the masses must agree with the exact mean
of the class within its standard error. A wrong factor in the conventions, such as
the 4 between int Omega ^ Omegabar and int abs(dP/dz_m)^-2 d^4t, fails this test.

Run: uv run python -m tests.test_volume
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

from quartic import data, metric, sections, volume          # noqa: E402


def test_mean_v_dd_is_exact():
    """The dd pair of the exact mean against 512 pi^4 k0^2 / Gamma(1/4)^8 in mpmath."""
    mp.mp.dps = 50
    worst = 0.0
    for k0 in (1, 4, 28, 44):
        hi, lo = volume.mean_v_dd(k0)
        want = 512 * mp.pi ** 4 * k0 ** 2 / mp.gamma(mp.mpf(1) / 4) ** 8
        worst = max(worst, float(abs((mp.mpf(hi) + mp.mpf(lo)) / want - 1)))
    assert worst < 1e-31, worst
    scale = volume.unit_scale(28) ** 2 * float(volume.omega_volume(28))
    assert abs(scale - 1.0) < 1e-15, scale
    print(f"  the exact mean of v as a dd pair, to {worst:.1e}")


def test_sampled_mean_matches_the_exact_mean():
    """The weighted mean of v for the Fubini-Study metric, against the exact mean."""
    batch = data.make(25000, 5)
    w = np.asarray(batch['mass'])
    for k in (1, 4):
        lay = sections.layout(k)
        v = np.asarray(metric.volumes(lay, sections.fubini_study_theta(lay), batch))
        mean = float(np.sum(w * v) / np.sum(w))
        err = float(np.sqrt(np.sum(w ** 2 * (v - mean) ** 2)) / np.sum(w))
        exact = float(volume.mean_v(k))
        assert abs(mean - exact) < 5 * err, (k, mean, exact, err)
        print(f"  k = {k}: sampled mean {mean:.6f}, exact {exact:.6f}, "
              f"difference {abs(mean - exact) / err:.2f} standard errors")


if __name__ == "__main__":
    for fn in (test_mean_v_dd_is_exact, test_sampled_mean_matches_the_exact_mean):
        print(fn.__name__)
        fn()
    print("\nall volume tests pass")
