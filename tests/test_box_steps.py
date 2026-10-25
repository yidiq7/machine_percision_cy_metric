"""Test of the hooks of `quartic.cert.box.box_f` and of `scripts/cert_run.run_reference`.

1. A call of box_f with a dict `keep` gives the same bits as a call with no dict, and the
   dict holds each key of the hooks.
2. The box B6 of the run v3 at the factor 1: the replay gives the sup and grad_g of its record
   bit for bit, 4 points pass R1 for each model of the steps 1 to 9, and the test detects the
   part `residual` of h (R2 above 1 at a point, and no R2 of it NaN).

Run on a GPU node, with the cover v3, the local forms v3 of the reference center 297 and the
records of the run v3:
    uv run --extra gpu --locked python -u -m tests.test_box_steps
"""
import math
import sys
import tempfile
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
import cert_steps as cs                                     # noqa: E402
from quartic.cert import records as rs                      # noqa: E402

COVER = ROOT / 'runs_tm' / 'cover_s0.03_c6_v3.npz'
LOCAL = Path('/n/netscratch/mweber_lab/Lab/yidqi/tm_local_f6t_td_v3/center_297.npz')
REF = 297
RUN = [ROOT / 'runs_tm' / 'full_f6t_v3' / f'run_{g}.npz' for g in range(4)]
DATA = ROOT / 'runs_tm' / 'f6t_data.npz'
FORMS = ROOT / 'runs_tm' / 'f6t_forms.npz'
POINTS = 4

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


def test_steps_b6():
    """The box B6 of the records RUN (the four parts of the run v3). The replay of its sup and
    grad_g must equal the record bit for bit."""
    cs._gpu_imports()
    rec, _, _, _ = rs.load([str(p) for p in RUN])
    row_index = cs.select_boxes(rec)['B6']
    row = rec[row_index]
    assert int(row[cs.COL['ref']]) == REF
    cover_file = np.load(COVER)
    cov = {k: cover_file[k] for k in cover_file.files}
    local = tc.local_on_gpu(str(LOCAL))
    shas = dict(sha_data=cs.SHA_DATA, sha_forms=cs.SHA_FORMS, sha_run='', sha_cover='')
    with tempfile.TemporaryDirectory() as tmp:
        # export_box raises SystemExit if the replay differs from the record in sup or grad_g.
        cs.export_box('B6', row, cov, float(cov['side']), local, rs.sha256(LOCAL), [1.0], tmp,
                      shas, 'test')
        path = Path(tmp) / 'B6_x1.npz'
        with np.load(path) as z:
            assert float(z['factor']) == 1.0 and float(z['sup']) == row[cs.COL['bound']]
            assert float(z['grad_g']) == row[cs.COL['grad_g']]
            d_phi, top = float(z['d_phi']), int(z['top'])
        cs._init_worker(str(DATA), str(FORMS))
        points = cs.make_points(path.name, d_phi, POINTS)
        r2s, bad = [], []
        for i in range(POINTS):
            _, _, out, times = cs._check_file_point((str(path), i, points[i]))
            bad += [f'point {i} model {n}: {r["status"]}, gap/R {r["ratio"]:.3e}'
                    for n, r in out.items() if r['status'] != 'pass']
            r2s.append(out['h']['r2']['residual'])
            print(f'  point {i}: largest gap / R {max(r["ratio"] for r in out.values()):.3e}, '
                  f'{sum(times):.1f} s', flush=True)
    assert not bad, 'R1 fails: ' + '; '.join(bad)
    # _largest gives NaN if one R2 is NaN, and a NaN R2 fails.
    worst = cs._largest(r2s)
    assert not math.isnan(worst), f'an R2 of the part residual of h is NaN: {r2s}'
    assert worst > 1, f'the largest R2 of the part residual of h is {worst:.3e}, not above 1'
    print(f'  B6: {len(cs.step_models(top))} models x {POINTS} points pass; '
          f'largest R2 of h residual {worst:.3e}', flush=True)


def main():
    test_keep_same_bits()
    test_steps_b6()
    print('all box step tests pass', flush=True)


if __name__ == '__main__':
    main()
