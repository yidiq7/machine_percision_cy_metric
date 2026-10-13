"""The frame of a box of the strict bound, in ball arithmetic (python-flint), on the host.

A box has its center p on X in the chart z_3 = 1: t = (p_0, p_1) is the dd center of a
cell, and p_2 is the dd rounding of the principal fourth root of -(1 + t_0^4 + t_1^4).
Each frame number is the dd rounding of an arb ball, and from then on that dd number is
the exact definition of the frame. The error of each rounding is bounded from the ball:

* the tangent frame E (dy/dt at p, with E_2 = -p_i^3 / p_2^3) and the unit normal n;
* c1inv = 1 / (grad P . n) and the td constants of `quartic.cert.box.implicit_h_td` with their errors;
* vol = 1 / (abs(det U)^2 vbar), U = [E, n], with vbar = 512 pi^4 k0^2 / Gamma(1/4)^8,
  and a bound of its relative error.

Both `quartic.cert.box.make_frames` (the boxes on the GPU) and
`quartic.cert.cover.box_radius` (the radius of each cell) take the frame from `frame`, so
they work with the same numbers.
"""
from __future__ import annotations

from flint import acb, acb_mat, arb

from tmgpu.host import exact, lower, precision, to_dd, to_td

# t^4 of a dd center needs 424 bits, so at PREC = 512 bits the center values are exact
# balls; a center with -(1 + t_0^4 + t_1^4) on the negative real axis keeps a narrow root.



def frame(c_dd, k0):
    """The frame of the box at the dd center t = c_dd (two dd lists), with error bounds.

    k0 = None skips vol (the cover needs only the geometry).

    Returns a dict of dd lists (p, e, n, c1inv, vol); the td constants of the map for h
    (ha_td, a6_td, a4_td, kappa_td) with h_err_td = [errors of the three ha_td, of kappa_td,
    largest error of a6_td, of a4_td]; the exact acb values of the frame (key 'acb'); and
    the relative error bound vol_err of vol.
    """
    with precision():
        t0, t1 = exact(c_dd[0]), exact(c_dd[1])
        y2 = (-(1 + t0 ** 4 + t1 ** 4)).root(4)
        p2_dd, _ = to_dd(y2)
        p = [t0, t1, exact(p2_dd)]
        one, zero = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
        e = [[acb(1), acb(0)], [acb(0), acb(1)]]
        e_dd = [[one, zero], [zero, one]]
        for i in range(2):
            d_dd, _ = to_dd(-p[i] ** 3 / p[2] ** 3)
            e[i].append(exact(d_dd))
            e_dd[i].append(d_dd)
        grad = [4 * c ** 3 for c in p]
        norm = sum(abs(g) ** 2 for g in grad).sqrt()
        n_dd = [to_dd(g.conjugate() / norm)[0] for g in grad]
        n = [exact(v) for v in n_dd]
        c1 = sum(g * m for g, m in zip(grad, n))
        c1inv_dd, _ = to_dd(1 / c1)
        c1inv = exact(c1inv_dd)
        vol_dd, vol_err = None, None
        if k0 is not None:
            u = acb_mat([[e[0][k], e[1][k], n[k]] for k in range(3)])
            vbar = 512 * arb.pi() ** 4 * k0 ** 2 / arb(0.25).gamma() ** 8
            w = 1 / (abs(u.det()) ** 2 * vbar)
            vol_dd, vol_abs_err = to_dd(acb(w))
            vol_err = vol_abs_err / lower(w) * (1 + 1e-9) + 1e-60
        ha = [c1inv * (sum(c ** 4 for c in p) + 1),
              c1inv * sum(g * x for g, x in zip(grad, e[0])),
              c1inv * sum(g * x for g, x in zip(grad, e[1]))]
        # The constants of the td solve of h, with kappa = 1 - c1inv c1.
        ha_td = [to_td(v) for v in ha]
        a6_td = [to_td(6 * c ** 2) for c in p]
        a4_td = [to_td(4 * c) for c in p]
        kap_td = to_td(1 - c1inv * c1)
        h_err_td = [x[1] for x in ha_td] + [kap_td[1], max(x[1] for x in a6_td),
                                            max(x[1] for x in a4_td)]
        return dict(p=[list(c_dd[0]), list(c_dd[1]), p2_dd], e=e_dd, n=n_dd, c1inv=c1inv_dd,
                    vol=vol_dd, vol_err=vol_err, ha_td=[x[0] for x in ha_td], a6_td=[x[0] for x in a6_td],
                    a4_td=[x[0] for x in a4_td], kappa_td=kap_td[0], h_err_td=h_err_td,
                    acb=dict(p=p, e=e, n=n, y2=y2))


def kappa(s_dd, k0, x0_dd, zeta_dd):
    """s^(1/k0) / (x0 (1 + zeta)) for a real s > 0 and dd numbers, as dd and an error bound."""
    with precision():
        root = (exact(s_dd).real.log() / k0).exp()
        return to_dd(root / (exact(x0_dd).real * (1 + exact(zeta_dd))))
