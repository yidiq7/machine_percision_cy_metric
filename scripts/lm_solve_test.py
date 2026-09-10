"""Does a more accurate LM linear solve lower the TRUE residual that LM reaches?

**The answer is no, and that is why no double-double linear solve exists in this
project.** Measured at cond(J)*eps up to 220, which is worse than the 148 of the
fit at k = 28, the gain is 0.84 to 1.04 times, which is noise.

The reason is the trust region. A step from an inaccurate solve is a candidate
only. When it fails, lambda grows and the step turns toward the gradient
direction, which is accurate, so LM loses speed and not accuracy. That extends
the earlier measurement on this same test problem, which reached cond(J) = 1e11.

The test problem has the shape of `v`: a ratio of two linear forms, with an
exact zero minimum. Only the linear solve differs between the two columns.

**Report the true residual and not the computed one.** LM minimizes the residual
that it computes, and that value can go below the error of the arithmetic: here
it reads 3.2e-18 where the true value at the same theta is 7.2e-19. The same
artifact put sigma at k = 28 below its own floor in plain FP64.

Usage:
    uv run python scripts/lm_solve_test.py
"""
import numpy as np
from mpmath import mp, mpf, matrix as mpmatrix, lu_solve, sqrt as mpsqrt
mp.dps = 60


def make(n, m, cond, seed):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(m, n)))
    vt, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = u @ np.diag(np.logspace(0, -np.log10(cond), n)) @ vt.T
    c = rng.normal(size=(m, n)) * (0.05 / np.sqrt(n))
    star = rng.normal(size=n)
    target = (A @ star) / (1.0 + c @ star)

    def full(th):
        den = 1.0 + c @ th
        val = (A @ th) / den
        jac = A / den[:, None] - (val / den)[:, None] * c
        return val - target, jac

    Am = [[mpf(float(x)) for x in row] for row in A]
    cm = [[mpf(float(x)) for x in row] for row in c]
    sm = [mpf(float(x)) for x in star]

    def true_rms(th, rows=400):
        """rms of the exact residual at th, on the first `rows` residuals."""
        tm = [mpf(float(x)) for x in th]
        acc = mpf(0)
        for i in range(rows):
            num = sum(Am[i][j] * tm[j] for j in range(n))
            den = mpf(1) + sum(cm[i][j] * tm[j] for j in range(n))
            nums = sum(Am[i][j] * sm[j] for j in range(n))
            dens = mpf(1) + sum(cm[i][j] * sm[j] for j in range(n))
            f = num / den - nums / dens
            acc += f * f
        return float(mpsqrt(acc / rows))
    return full, star, true_rms


def lm(full, th0, steps, exact_solve):
    th = th0.copy()
    r, J = full(th)
    loss = 0.5 * r @ r
    lam = 1e-3 * np.max(np.diag(J.T @ J))
    for _ in range(steps):
        Hd = J.T @ J + lam * np.eye(len(th))
        g = J.T @ r
        if exact_solve:
            Am = mpmatrix([[mpf(float(x)) for x in row] for row in Hd])
            bm = mpmatrix([mpf(float(-x)) for x in g])
            step = np.array([float(x) for x in lu_solve(Am, bm)])
        else:
            step = np.linalg.solve(Hd, -g)
        rc, Jc = full(th + step)
        lc = 0.5 * rc @ rc
        if lc < loss:
            th, r, J, loss = th + step, rc, Jc, lc
            lam = max(lam / 10.0, 1e-300)
        else:
            lam *= 10.0
            if lam > 1e30:
                break
    return th, float(np.sqrt(np.mean(r * r)))


n, m, steps = 60, 4000, 400
print(f'{n} parameters, {m} residuals, {steps} LM steps')
print('`computed` is what LM sees. `true` is mpmath at 60 digits at the same theta.')
print(f"\n{'cond(J)':>9} {'cond*eps':>9} | {'FP64 solve':>21} | {'exact solve':>21} | {'gain':>6}")
print(f"{'':>9} {'':>9} | {'computed':>10} {'true':>10} | {'computed':>10} {'true':>10} |")
for cond in (1e10, 1e14, 1e16, 1e18):
    full, star, true_rms = make(n, m, cond, 7)
    th0 = star + np.random.default_rng(3).normal(size=n) * 0.05
    ta, ca = lm(full, th0, steps, False)
    tb, cb = lm(full, th0, steps, True)
    va, vb = true_rms(ta), true_rms(tb)
    print(f'{cond:>9.0e} {cond*2.22e-16:>9.1e} | {ca:>10.3e} {va:>10.3e} | '
          f'{cb:>10.3e} {vb:>10.3e} | {va/max(vb,1e-300):>5.2f}x', flush=True)
