"""The CUDA source of the dd primitives, and the compiler options of every tmgpu kernel.

two_sum and quick_two_sum are exact; the bounds of dd_add and dd_mul are in the docstring
of `tmgpu.tm`. --fmad=false keeps the compiler from a merge of a product and a sum into
one FMA, which would change the error term that two_sum computes.
"""
import cupy as cp

OPTIONS = ('--fmad=false', '-std=c++14')

DD_PRIMITIVES = r'''
__device__ __forceinline__ void two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    double bb = __dsub_rn(s, a);
    e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
}
__device__ __forceinline__ void quick_two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    e = __dsub_rn(b, __dsub_rn(s, a));
}
__device__ __forceinline__ void dd_add(double ah, double al, double bh, double bl,
                                       double &sh, double &sl) {
    double s1, s2, t1, t2;
    two_sum(ah, bh, s1, s2);
    two_sum(al, bl, t1, t2);
    two_sum(s1, __dadd_rn(s2, t1), s1, s2);
    two_sum(s1, __dadd_rn(s2, t2), sh, sl);
}
__device__ __forceinline__ void dd_mul(double ah, double al, double bh, double bl,
                                       double &ph, double &pl) {
    double p = __dmul_rn(ah, bh);
    double e = __fma_rn(ah, bh, -p);
    double t = __fma_rn(al, bl, __dmul_rn(ah, bl));
    t = __fma_rn(al, bh, t);
    quick_two_sum(p, __dadd_rn(e, t), ph, pl);
}
'''


def module(code):
    """The CuPy module of CUDA source compiled with OPTIONS."""
    return cp.RawModule(code=code, options=OPTIONS)
