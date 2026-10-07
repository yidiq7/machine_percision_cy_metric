"""The Taylor jets of bidegree (2, 2) must be exact to dd and agree with the 13-row jets.

Run: uv run --extra gpu python -m tests.test_tjets
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from quartic import analytic, data, dd, dd_metric, sections, tjets as tj   # noqa: E402

analytic.EXACT_MUL = True
BATCH = data.make(20, 11)
M = tj.M


def _cdd(x):
    return dd.cto_dd(jnp.asarray(x))


def _poly_hol(c):
    """A dict multi-index -> coefficient from a holomorphic jet array (6,)."""
    return {M[a]: c[a] for a in range(6)}


def _mul_hol_ref(f, g):
    out = np.zeros(6, complex)
    for a, ma in enumerate(M):
        for a1, m1 in enumerate(M):
            for a2, m2 in enumerate(M):
                if (m1[0] + m2[0], m1[1] + m2[1]) == ma:
                    out[a] += f[a1] * g[a2]
    return out


def _mul_ref(f, g):
    out = np.zeros((6, 6), complex)
    for a in range(6):
        for b in range(6):
            for a1 in range(6):
                for a2 in range(6):
                    if (M[a1][0] + M[a2][0], M[a1][1] + M[a2][1]) != M[a]:
                        continue
                    for b1 in range(6):
                        for b2 in range(6):
                            if (M[b1][0] + M[b2][0], M[b1][1] + M[b2][1]) == M[b]:
                                out[a, b] += f[a1, b1] * g[a2, b2]
    return out


def test_products_match_polynomials():
    """hol_mul and mul are the truncated products of polynomials."""
    rng = np.random.default_rng(1)
    f = rng.normal(size=6) + 1j * rng.normal(size=6)
    g = rng.normal(size=6) + 1j * rng.normal(size=6)
    got = dd.cto_fp(jax.jit(tj.hol_mul)(_cdd(f), _cdd(g)))
    err_h = np.max(np.abs(np.asarray(got) - _mul_hol_ref(f, g)))
    F = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    G = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    got = dd.cto_fp(jax.jit(tj.mul)(_cdd(F), _cdd(G)))
    err_b = np.max(np.abs(np.asarray(got) - _mul_ref(F, G)))
    assert err_h < 1e-13 and err_b < 1e-12, (err_h, err_b)
    print(f'  products against polynomials: {err_h:.1e} holomorphic, {err_b:.1e} bidegree')


def _nil_jet(rng, scale=0.3):
    x = scale * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    x = 0.5 * (x + x.conj().T)
    x[0, 0] = 0.0
    return _cdd(x)


def test_series_identities():
    """exp(log(1 + x)) = 1 + x and (1 + x)^-3 (1 + x)^3 = 1, to dd."""
    rng = np.random.default_rng(2)
    x = _nil_jet(rng)

    def run(x):
        a = tj.exp_nil(tj.log1p_nil(x))
        b = tj.mul(tj.pow_nil(x, -3), tj.pow_nil(x, 3))
        return a, b
    a, b = jax.jit(run)(x)
    one_x = tj._add_value(x, dd.to_dd(jnp.ones(())))
    err_a = max(float(jnp.max(jnp.abs(dd.to_fp(dd.add(a[i], dd.neg(one_x[i]))))))
                for i in range(2))
    one = np.zeros((6, 6))
    one[0, 0] = 1.0
    err_b = max(float(jnp.max(jnp.abs(dd.to_fp(b[0]) - one))),
                float(jnp.max(jnp.abs(dd.to_fp(b[1])))))
    assert err_a < 1e-29 and err_b < 1e-29, (err_a, err_b)
    print(f'  exp(log1p(x)) = 1 + x to {err_a:.1e}, (1+x)^-3 (1+x)^3 = 1 to {err_b:.1e}')


def _dd_point(i):
    z = BATCH['points'][i]
    m = dd_metric.newton_index(z)
    patch = int(np.argmin(np.abs(np.asarray(z) - 1.0)))
    free = tuple(sorted(set(range(4)) - {patch, m}))
    z_dd, _ = jax.jit(lambda w: dd_metric.refine(w, m, 1))(z)
    return z_dd, m, free


def test_chart_jets_stay_on_X():
    """sum_a z_a(t)^4 vanishes to second order: every coefficient is near 1e-32."""
    worst = 0.0
    for i in range(4):
        z_dd, m, free = _dd_point(i)

        def run(z_dd):
            ch = tj.chart_jets(z_dd, m, free)
            four = analytic._cindex(tj.power_table(ch.hol, 4), 4)       # (4, 6)
            total = (dd.sum_dd(*four[0], axis=0), dd.sum_dd(*four[1], axis=0))
            size = jnp.sum(jnp.abs(dd.cto_fp(four)))
            return jnp.max(jnp.abs(dd.cto_fp(total))) / size
        worst = max(worst, float(jax.jit(run)(z_dd)))
    assert worst < 1e-30, worst
    print(f'  P(z(t)) vanishes to second order to {worst:.1e} of its terms')


def test_base_jet_at_fubini_study():
    """At theta_FS, p0 = |z|^(2k), so log p0 and k log |z|^2 have the same jet."""
    worst = 0.0
    for k in (4, 8):
        lay = sections.layout(k)
        theta = sections.fubini_study_theta(lay)
        parts = sections.prepare_parts(lay, jnp.asarray(theta))
        for i in range(3):
            z_dd, m, free = _dd_point(i)

            def run(z_dd):
                ch = tj.chart_jets(z_dd, m, free)
                table = tj.power_table(ch.hol, max(4 * int(np.max(g.exps)) for g in lay.groups)
                                       + 3)
                p = tj.base_jet(lay, parts, ch, table)
                r2 = tj.radius_jet(ch)
                lp = tj.log1p_nil(tj._scale(tj.nil(p), dd.reciprocal(tj.value(p))))
                lr = tj.log1p_nil(tj._scale(tj.nil(r2), dd.reciprocal(tj.value(r2))))
                lr = analytic._cscale(lr, jnp.asarray(float(k)))
                diff = dd.cto_fp(dd.csub(lp, lr))
                return jnp.max(jnp.abs(diff)) / jnp.max(jnp.abs(dd.cto_fp(lr)))
            worst = max(worst, float(jax.jit(run)(z_dd)))
    assert worst < 1e-28, worst
    print(f'  log p0 at theta_FS equals k log |z|^2 to {worst:.1e}, k = 4 and 8')


def test_metric_value_matches_volume_dd():
    """eta mean = v of `dd_metric.volume_dd` at theta_FS, to dd."""
    worst = 0.0
    k = 6
    lay = sections.layout(k)
    theta = sections.fubini_study_theta(lay)
    parts = sections.prepare_parts(lay, jnp.asarray(theta))
    mats = sections.prepare(lay, jnp.asarray(theta))
    for i in range(3):
        z_dd, m, free = _dd_point(i)

        def run(z_dd):
            ch = tj.chart_jets(z_dd, m, free)
            table = tj.power_table(ch.hol, 4 * max(int(np.max(g.exps)) for g in lay.groups)
                                   + 3)
            p = tj.base_jet(lay, parts, ch, table)
            lp = tj.log1p_nil(tj._scale(tj.nil(p), dd.reciprocal(tj.value(p))))
            one = dd.to_dd(jnp.ones(()))
            f0 = tj.metric_parts(lp, ch, one)[0]
            v = dd_metric.volume_dd(lay, mats, z_dd, m, free)
            return dd.to_fp(dd.add(f0, one)), dd.to_fp(v)
        a, b = jax.jit(run)(z_dd)
        worst = max(worst, abs(float(a) / float(b) - 1.0))
    # volume_dd takes H rounded to FP64, and the jets take it in dd.
    assert worst < 1e-15, worst
    print(f'  det(g) |dP/dz_m|^2 from the jets equals volume_dd to {worst:.1e}')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(name)
            fn()
    print('all passed')
