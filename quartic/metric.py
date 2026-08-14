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


def volumes(lay, theta, batch):
    """v = det(g|_X) / OmegaOmegabar at every point of the batch."""
    metrics = metrics_on(lay, theta, batch)
    return jnp.real(jax.vmap(jnp.linalg.det)(metrics)) / batch['Omega_Omegabar']


def residual(lay, theta, batch):
    """f = v/<v> - 1, with <v> the mass-weighted mean over the batch."""
    v = volumes(lay, theta, batch)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return v / jnp.sum(weights * v) - 1.0


def weighted_residual(lay, theta, batch):
    """sqrt(w) f, so that the sum of squares is exactly the MSPE.

    LM differentiates this vector, so the mean stays inside the derivative. The
    residual then always satisfies sum_i w_i f_i = 0, which is correct: the
    sample mean is not a free parameter of the model.
    """
    weights = batch['mass'] / jnp.sum(batch['mass'])
    return jnp.sqrt(weights) * residual(lay, theta, batch)


def report(lay, theta, batch):
    """The numbers that "Output" asks for, on one point set."""
    v = volumes(lay, theta, batch)
    weights = batch['mass'] / jnp.sum(batch['mass'])
    f = v / jnp.sum(weights * v) - 1.0

    metrics = metrics_on(lay, theta, batch)
    eig = jnp.linalg.eigvalsh(0.5 * (metrics + jnp.conj(jnp.swapaxes(metrics, -1, -2))))
    mats = sections.prepare(lay, theta)
    potentials = jax.vmap(lambda z: sections.potential_at(lay, mats, z))(batch['points'])

    return {
        'mape': float(jnp.sum(weights * jnp.abs(f))),
        'rms': float(jnp.sqrt(jnp.sum(weights * f ** 2))),
        'max_abs': float(jnp.max(jnp.abs(f))),
        'mspe': float(jnp.sum(weights * f ** 2)),
        'min_eig': float(jnp.min(eig)),
        'min_p': float(jnp.min(potentials)),
        'n_points': int(f.shape[0]),
    }
