# Studies

These scripts give numbers that the paper cites. The metric and the bound do not need them.

| script | number of the paper | place in the paper |
|---|---|---|
| `run_fit.py` | fits of the Headrick-Nassar (HN) type stop near a mean residual of 3e-14 at degree 28 | Section 2.1 |
| `lift.py` | the map from the coefficients of H to the values of p₀ has a condition number of 1.9e20 at degree 32 | Section 2.1 |
| `vpa_oracle.py` | in FP64 (64-bit floating point), the error of the recurrence grows by about 3.2 for each degree | Appendix C.3 |
| `vpa_fit.py` | one level of degree 36 with the basis in FP64 gives mean abs(f) = 3.5e-16 | Appendix C.3 |
| `vpa_dd_fit.py` | the same level with the basis in dd (double-double) arithmetic gives 1.1e-17 | Appendix C.3 |

The studies use the same environment as the other scripts. The docstring of each script gives a command.
