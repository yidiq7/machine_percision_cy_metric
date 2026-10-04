"""The exact volumes of the Fermat quartic X in the conventions of v.

v = det(g) abs(dP/dz_m)^2 in the chart coordinates t, with g = ddbar Phi. For Phi
in the class of k0 times the hyperplane class:

    int_X det(g) d^4t           = 2 pi^2 k0^2,          the volume of (i/2) g dt dtbar,
    int_X abs(dP/dz_m)^-2 d^4t  = Gamma(1/4)^8 / (256 pi^2),

so the mean of v over the measure abs(dP/dz_m)^-2 d^4t is 512 pi^4 k0^2 / Gamma(1/4)^8,
and f = v / that mean - 1 needs no sampled mean. int Omega ^ Omegabar is 4 times the
second volume, Gamma(1/4)^8 / (64 pi^2). The metric g / (sqrt(2) pi k0) has volume 1.
"""
import mpmath as mp

__all__ = ['omega_volume', 'holomorphic_volume', 'mean_v', 'mean_v_dd', 'unit_scale']


def omega_volume(k0, dps=50):
    """int_X det(g) d^4t for Phi in the class of k0 times the hyperplane class."""
    with mp.workdps(dps):
        return 2 * mp.pi ** 2 * k0 ** 2


def holomorphic_volume(dps=50):
    """int_X abs(dP/dz_m)^-2 d^4t, the volume of the measure that the masses sample."""
    with mp.workdps(dps):
        return mp.gamma(mp.mpf(1) / 4) ** 8 / (256 * mp.pi ** 2)


def mean_v(k0, dps=50):
    """The exact mean of v over X: omega_volume / holomorphic_volume."""
    with mp.workdps(dps):
        return omega_volume(k0, dps) / holomorphic_volume(dps)


def mean_v_dd(k0):
    """`mean_v` as a dd pair of floats, exact to 2^-106."""
    with mp.workdps(50):
        value = mean_v(k0)
        hi = float(value)
        return hi, float(value - hi)


def unit_scale(k0):
    """The factor that gives g volume 1: 1 / sqrt(omega_volume)."""
    with mp.workdps(50):
        return float(1 / mp.sqrt(omega_volume(k0)))
