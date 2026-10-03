"""The joint map of several levels: the joint functions are orthonormal, and a
function that the earlier levels already span does not enter.

Run: uv run python -m tests.test_joint_dd
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MLGeometry as mlg                                    # noqa: E402
mlg.set_precision(64)

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from quartic import dd, joint_dd                            # noqa: E402


def sample():
    """Two 'levels' on [-1, 1]: x^j and x^j + x^(j+1) / 8 for j < 8, the second as
    exact dd sums, so 7 of the second lie in the span of the first and one adds x^8."""
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, 2000)
    a = np.stack([x ** j for j in range(8)], axis=1)
    up = np.stack([x ** (j + 1) for j in range(8)], axis=1) * 0.125
    b = dd.add(dd.to_dd(jnp.asarray(a)), dd.to_dd(jnp.asarray(up)))
    return x, a, b


def test_joint_functions_are_orthonormal():
    x, a, b = sample()
    mu = np.full(len(x), 1.0 / len(x))
    jm = joint_dd.build([dd.to_dd(jnp.asarray(a)), b], mu, tol=1e-24, block=4)
    entered = [int(np.sum(jm.level == i)) for i in range(2)]
    assert entered == [8, 1], entered
    # The map acts on Hessians; values in the (0, 0) entry give the joint functions.
    re_hi = np.zeros((len(x), 16, 2, 2))
    re_lo = np.zeros((len(x), 16, 2, 2))
    re_hi[:, :8, 0, 0], re_hi[:, 8:, 0, 0] = a, np.asarray(b[0])
    re_lo[:, 8:, 0, 0] = np.asarray(b[1])
    p = ((jnp.asarray(re_hi), jnp.asarray(re_lo)), dd.to_dd(jnp.zeros((len(x), 16, 2, 2))))
    out = jax.jit(jax.vmap(lambda q: joint_dd.transform(q, jm)))(p)
    hi, lo = np.asarray(out[0][0][:, :, 0, 0]), np.asarray(out[0][1][:, :, 0, 0])
    mp.mp.dps = 40
    n = hi.shape[1]
    g = [[mp.fsum(mp.mpf(float(mu[i])) * (mp.mpf(float(hi[i, s])) + mp.mpf(float(lo[i, s])))
                  * (mp.mpf(float(hi[i, t])) + mp.mpf(float(lo[i, t])))
                  for i in range(0, len(x))) for t in range(n)] for s in range(n)]
    gap = max(float(abs(g[s][t] - (1 if s == t else 0))) for s in range(n) for t in range(n))
    assert gap < 1e-26, gap
    print(f"  {entered} functions enter; the joint functions are orthonormal to {gap:.1e}")
    # 1 = x^0 lies in the span, so psi_N = sum gamma_t Gt_t is 1 to dd accuracy.
    psi = [mp.fsum((mp.mpf(float(jm.gamma[0][t])) + mp.mpf(float(jm.gamma[1][t])))
                   * (mp.mpf(float(hi[i, t])) + mp.mpf(float(lo[i, t]))) for t in range(n))
           for i in range(0, len(x), 50)]
    off = max(float(abs(v - 1)) for v in psi)
    assert off < 1e-26 and jm.gap < 1e-12, (off, jm.gap)
    print(f"  psi_N is 1 to {off:.1e}, and the reported gap is {jm.gap:.1e}")
    # A saved map gives the same Hessians.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        joint_dd.save(tmp + '/map.npz', jm)
        back = joint_dd.load(tmp + '/map.npz')
    again = jax.jit(jax.vmap(lambda q: joint_dd.transform(q, back)))(p)
    same = max(float(jnp.max(jnp.abs(x - y))) for x, y in
               zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(again)))
    assert same == 0.0, same
    print("  a saved and loaded map gives the same joint functions")


if __name__ == "__main__":
    test_joint_functions_are_orthonormal()
    print("\nall joint_dd tests pass")
