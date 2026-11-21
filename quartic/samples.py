"""Samples of points on X with their dd points, and a table of f by regions of min abs(z) over
max abs(z).
"""
import math

from . import data, dd_metric


def load_points(n, seed):
    """n points of the sampler with `seed`, with the dd points of `point_extras`."""
    batch = data.make(math.ceil(n / 4), seed)
    batch = {key: val[:n] for key, val in batch.items()}
    batch.update(dd_metric.point_extras(batch['points']))
    return batch


EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def regions_text(rep):
    lines = []
    for row in rep['regions']:
        lines.append(f"    min/max abs(z) in {row['bin'][0]:.2f} to {row['bin'][1]:.2f}: "
                     f"points {row['points']:.4f}, weight {row['weight']:.4f}, loss "
                     f"{row['loss_share']:.3f}, mean abs(f) {row['mape']:.3e}, rms "
                     f"{row['rms']:.3e}")
    return '\n'.join(lines)
