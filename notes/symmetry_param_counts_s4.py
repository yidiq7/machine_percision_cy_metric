"""Exact count of Gamma-invariant parameters on the Fermat quartic (task 3).

Adapted from symmetry_param_counts.py by the change S_5 -> S_4:
  * the phases Z_4^3 force h_AB = 0 unless A = B mod 4, so h is block diagonal
    in r = A mod 4 in {0..3}^4 with |r| = k mod 4, which gives 64 blocks,
  * inside a block A = r + 4C and u_a := z_a^4, and P = 0 is exactly sum u = 0,
    so Sec_m = Sym^m(std) with std the 3-dim standard rep of S_4, dim C(m+2,2),
  * S_4 permutes the blocks, so one orbit representative carries the parameters,
    and Stab(r) constrains them,
  * the Z_2 conjugation makes h real symmetric, so the count uses Sym^2.

  n_theta(k) = sum over S_4-orbits of r  of  dim ( Sym^2( Sec_{m_r} ) )^{Stab(r)}
"""
import itertools
from collections import Counter
from fractions import Fraction
from math import comb

N = 4
SN = list(itertools.permutations(range(N)))


def cycle_type(perm):
    """Sorted cycle lengths of perm, where perm[i] is the image of i."""
    seen = [False] * N
    out = []
    for i in range(N):
        if seen[i]:
            continue
        length, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        out.append(length)
    return tuple(sorted(out))


def sym_power_chars(ct, mmax):
    """chi_{Sym^m(std)}(sigma) for m = 0..mmax, from (1-t) / prod_j (1 - t^{c_j})."""
    coeffs = ([1, -1] + [0] * mmax)[: mmax + 1]
    for c in ct:
        out = [0] * (mmax + 1)
        for i in range(mmax + 1):
            if coeffs[i]:
                for j in range(i, mmax + 1, c):
                    out[j] += coeffs[i]
        coeffs = out
    return coeffs


def compose(p, q):
    """(p o q)(i) = p[q[i]]."""
    return tuple(p[q[i]] for i in range(N))


def sym2_invariant_dim(subgroup, m, char_cache):
    """dim ( Sym^2 Sec_m )^H = (1/|H|) sum_h (chi(h)^2 + chi(h^2)) / 2."""
    total = Fraction(0)
    for h in subgroup:
        c1 = char_cache[cycle_type(h)][m]
        c2 = char_cache[cycle_type(compose(h, h))][m]
        total += Fraction(c1 * c1 + c2, 2)
    total /= len(subgroup)
    assert total.denominator == 1, total
    return int(total)


def residue_orbits(k):
    """S_4-orbits of r in {0..3}^4 with |r| = k (mod 4) and |r| <= k."""
    out, seen = [], set()
    for r in itertools.product(range(N), repeat=N):
        if r in seen:
            continue
        s = sum(r)
        if (k - s) % N or k - s < 0:
            continue
        orbit = {tuple(r[p[i]] for i in range(N)) for p in SN}
        seen |= orbit
        out.append((r, len(orbit), (k - s) // N))
    return out


def stabilizer(r):
    return [p for p in SN if tuple(r[p[i]] for i in range(N)) == tuple(r)]


def char_cache_for(k):
    return {ct: sym_power_chars(ct, max(k // N, 1))
            for ct in {cycle_type(p) for p in SN}}


def n_theta(k, char_cache=None):
    """Exact parameter count at degree k, with the per-orbit detail."""
    char_cache = char_cache or char_cache_for(k)
    total, detail = 0, []
    for r, osize, m in residue_orbits(k):
        d = sym2_invariant_dim(stabilizer(r), m, char_cache)
        total += d
        detail.append((r, osize, m, d))
    return total, detail


def block_stats(k):
    """Total sections, the MACs of one matrix-vector product, and the m groups."""
    secs = macs = 0
    sizes = Counter()
    for _r, osize, m in residue_orbits(k):
        d = comb(m + 2, 2)
        secs += osize * d
        macs += osize * d * d
        sizes[m] += osize
    return secs, macs, dict(sorted(sizes.items()))


if __name__ == "__main__":
    KNOWN = {4: 5, 6: 10, 8: 19, 10: 33, 12: 56}
    print(f"{'k':>3} {'sections':>9} {'2k^2+2':>8} {'ok':>6} {'n_theta':>8} "
          f"{'own fits':>9} {'sum d_r^2':>10}  blocks (m: count)")
    for k in (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 36, 38):
        nt, _ = n_theta(k)
        secs, macs, sizes = block_stats(k)
        print(f"{k:>3} {secs:>9} {2 * k * k + 2:>8} {str(secs == 2 * k * k + 2):>6} "
              f"{nt:>8} {str(KNOWN.get(k, '')):>9} {macs:>10,}  {sizes}")

    print("\nblocks at k=32, one line for each S_4 orbit of r:")
    nt, detail = n_theta(32)
    for r, osize, m, d in sorted(detail, key=lambda t: (sum(t[0]), t[0])):
        print(f"  r={r} |r|={sum(r):>2} orbit={osize:>2} m={m:>2} "
              f"dim Sec={comb(m + 2, 2):>3} params={d:>4}")
    print(f"  total n_theta = {nt}")
