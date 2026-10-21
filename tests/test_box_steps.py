"""Test of the hooks of `quartic.cert.box.box_f` and of `scripts/cert_run.run_reference`.

1. A call of box_f with a dict `keep` gives the same bits as a call with no dict, and the
   dict holds each key of the hooks.

Run on a GPU node, with the cover and the local forms of the reference center 297:
    uv run --extra gpu --locked python -u -m tests.test_box_steps
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from tmgpu.tools import sup_bound                           # noqa: E402
from quartic.cert import box as tc                          # noqa: E402
import cert_run                                             # noqa: E402

COVER = ROOT / 'runs_tm' / 'cover_s0.03_c6_v2.npz'
LOCAL = Path('/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td/center_297.npz')
REF = 297

KEEP_KEYS = ['d', 'd_f', 'd_phi', 'h_td', 'h_dd', 'ys', 'delta', 'p0', 'r2', 'ir2', 'log_r2',
             'pt', 'xh', 'log_pt', 'xw', 'xi', 'fk', 'psi', 'phi', 'g', 'det', 'q4', 'v', 'f',
             'orders', 's_r', 's_p', 'zeta', 'kappa', 'kap_err', 'xi_kap', 'v_vol',
             'dfs', 'df_bounds', 'det_low', 'g00_low', 'tr_up',
             'batch', 'fr', 'info', 'sup', 'a', 'hmaxs', 'deltas']


def same(x, y):
    """True if the host arrays are equal bit for bit, with NaN at the same places."""
    x, y = np.asarray(cp.asnumpy(x)), np.asarray(cp.asnumpy(y))
    return x.shape == y.shape and np.array_equal(x, y, equal_nan=True)


def test_keep_same_bits():
    cov = np.load(COVER)
    local = tc.local_on_gpu(str(LOCAL))
    args = Namespace(order=28, order_high=32, rho_high=0.3, target=5e-20, depth=2, batch=64,
                     verbose=False)
    settings = {28: 0.35, 32: 0.12}
    keeps = []

    def on_batch(batch, order):
        keeps.append({'stop': True})
        return keeps[-1]

    cert_run.run_reference(cov, REF, local, settings, args, on_batch=on_batch)
    assert len(keeps) == 1, f'run_reference called on_batch {len(keeps)} times'
    keep = keeps[0]
    missing = [k for k in KEEP_KEYS if k not in keep]
    assert not missing, f'keep has no key {missing}'
    assert len(keep['ys']) == 3 and len(keep['fk']) == int(local['top']) + 1
    assert len(keep['dfs']) == 2 and len(keep['df_bounds']) == 2
    assert set(keep['g']) == {(i, j) for i in range(2) for j in range(2)}
    assert all(k in keep['info']['h_parts'] for k in ('residual', 'host', 'lam0', 'rho'))
    order = keep['orders'][0]
    f, (d, _, _), info, parts = tc.box_f(keep['fr'], local, keep['deltas'], order,
                                         order + tc.NY_EXTRA, keep['a'], keep['hmaxs'], grad=True)
    sup, _, _ = sup_bound(parts['sp4'], f, d)
    assert same(f.c, keep['f'].c), 'f.c differs'
    assert same(f.rem, keep['f'].rem), 'f.rem differs'
    assert same(info['grad_g'], keep['info']['grad_g']), 'grad_g differs'
    assert same(sup, keep['sup']), 'the sup bound differs'
    print(f'  keep: {len(keep)} keys; f, grad_g and the sup bound of {len(keep["batch"])} boxes '
          f'equal bit for bit', flush=True)


def main():
    test_keep_same_bits()
    print('all box step tests pass', flush=True)


if __name__ == '__main__':
    main()
