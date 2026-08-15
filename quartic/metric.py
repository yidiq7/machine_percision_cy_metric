"""The residual f on the Fermat quartic, through the MLGeometry restriction path.

v = det(g|_X) / OmegaOmegabar, and f = v/<v> - 1. The normalization <v> is the
mass-weighted mean over the sample, which is the convention of `loss.py`. The
exact <v> is Vol(X, mu_omega)/Vol(X, mu_Omega), which is topological. Every
reported number must say which of the two it used.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from MLGeometry import complex_math

from . import sections

__all__ = ['restricted_metric', 'volumes', 'residual', 'weighted_residual',
           'report']


def restricted_metric(lay, mats, z, restriction):
    """g|_X = R phi R^dag, with phi the complex Hessian of log p.

    The Hessian goes through `complex_hessian`, and not through the difference
    form (p_ij p - p_i p_j)/p^2, which cancels by a factor of k.
    """
    phi = complex_math.complex_hessian(
        lambda w: sections.log_potential_at(lay, mats, w), z)
    return restriction @ phi @ jnp.conj(restriction).T


def metrics_on(lay, theta, batch):
    """g|_X at every point. The block matrices are built once, not once a point."""
    mats = sections.prepare(lay, theta)
    return jax.vmap(lambda z, r: restricted_metric(lay, mats, z, r))(
        batch['points'], batch['restriction'])


def volumes(lay, theta, batch, chunk=None):
    """v = det(g|_X) / OmegaOmegabar at every point of the batch.

    With `chunk` set, `lax.map` walks the points in blocks of that size. The
    intermediates of one block are then free before the next block starts, so
    the peak memory follows `chunk` and not the number of points. That matters
    under forward-mode autodiff: the Jacobian nests the points, the 8 real
    directions of the Hessian, and the n_theta tangents, so one fused pass takes
    approximately 2.6 MB for each point at k = 12.
    """
    if chunk is None:
        metrics = metrics_on(lay, theta, batch)
        return jnp.real(jax.vmap(jnp.linalg.det)(metrics)) / batch['Omega_Omegabar']

    mats = sections.prepare(lay, theta)

    def one(args):
        z, restriction, omega = args
        metric_at = restricted_metric(lay, mats, z, restriction)
        return jnp.real(jnp.linalg.det(metric_at)) / omega

    return jax.lax.map(one, (batch['points'], batch['restriction'],
                             batch['Omega_Omegabar']), batch_size=chunk)


def _chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def volumes_chunked(lay, theta, batch, chunk=100_000):
    """v at every point, a chunk at a time, so a large hold-out set fits.

    This path takes no derivative, so a Python loop over jitted pieces is enough
    and it avoids one scan. The result is one array of N doubles, which is 80 MB
    at N = 1e7.
    """
    n = int(batch['points'].shape[0])
    if n <= chunk:
        return volumes(lay, theta, batch)

    mats = sections.prepare(lay, theta)

    @jax.jit
    def piece(points, restriction, omega):
        metrics = jax.vmap(lambda z, r: restricted_metric(lay, mats, z, r))(
            points, restriction)
        return jnp.real(jax.vmap(jnp.linalg.det)(metrics)) / omega

    return jnp.concatenate([
        piece(batch['points'][s], batch['restriction'][s],
              batch['Omega_Omegabar'][s]) for s in _chunks(n, chunk)])


def residual(lay, theta, batch, chunk=None):
    """f = v/<v> - 1, with <v> the mass-weighted mean over the batch."""
    v = volumes(lay, theta, batch, chunk)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return v / jnp.sum(weights * v) - 1.0


def weighted_residual(lay, theta, batch, chunk=None):
    """sqrt(w) f, so that the sum of squares is exactly the MSPE.

    LM differentiates this vector, so the mean stays inside the derivative. The
    residual then always satisfies sum_i w_i f_i = 0, which is correct: the
    sample mean is not a free parameter of the model.
    """
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return jnp.sqrt(weights) * residual(lay, theta, batch, chunk)


def report(lay, theta, batch, chunk=100_000):
    """The numbers that "Output" asks for, on one point set.

    The standard error of `mape` is approximately `mape / sqrt(N)`, because the
    spread of abs(f) is close to its mean. So N sets how many digits of the
    reported residual are real, and it does not set the residual itself.
    """
    v = volumes_chunked(lay, theta, batch, chunk)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    f = v / jnp.sum(weights * v) - 1.0

    mats = sections.prepare(lay, theta)

    @jax.jit
    def margins(points, restriction):
        metrics = jax.vmap(lambda z, r: restricted_metric(lay, mats, z, r))(
            points, restriction)
        herm = 0.5 * (metrics + jnp.conj(jnp.swapaxes(metrics, -1, -2)))
        return (jnp.min(jnp.linalg.eigvalsh(herm)),
                jnp.min(jax.vmap(lambda z: sections.potential_at(lay, mats, z))(points)))

    pairs = [margins(batch['points'][s], batch['restriction'][s])
             for s in _chunks(int(f.shape[0]), chunk)]

    mape = float(jnp.sum(weights * jnp.abs(f)))
    return {
        'mape': mape,
        'mape_stderr': mape / float(jnp.sqrt(f.shape[0])),
        'rms': float(jnp.sqrt(jnp.sum(weights * f ** 2))),
        'max_abs': float(jnp.max(jnp.abs(f))),
        'mspe': float(jnp.sum(weights * f ** 2)),
        'min_eig': float(min(float(a) for a, _ in pairs)),
        'min_p': float(min(float(b) for _, b in pairs)),
        'n_points': int(f.shape[0]),
    }
