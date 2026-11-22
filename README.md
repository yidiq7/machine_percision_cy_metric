# A machine-precision Calabi–Yau metric on the Fermat quartic

This repository holds the code of the paper

> M. R. Douglas, J. Gómez-Serrano, F. Lehmann, D. Platt and Y. Qi, *A constructive
> computer-assisted existence proof for a Calabi–Yau metric*, 2026.

The code fits approximate Calabi–Yau metrics on the Fermat quartic M = {Σ_a z_a⁴ = 0} in CP³.
Each fit is a Kähler potential, which the code evaluates in double-double (dd) arithmetic. For
the fit f6t, the code proves strict bounds of η − 1 and of ∇η on all of M (η as in the paper),
with Taylor models on a graphics processing unit (GPU), and the bounds of the metric that the
proof needs. The files of the proof are in the Zenodo record https://zenodo.org/records/24420573.

## Results

The bounds for f6t on all of M are sup abs(η − 1) ≤ 4.9985e-20 and sup abs(∇η)_g ≤ 3.3654e-16
(in the metric of volume 1). The bounds of its metric, from `atlas_combine_v3.npz`, are
C ≤ 2.17652, ω ≥ 0.59669 ω₀, κ ≤ 4.4450e-13, λ ≥ 0.016109, Λ ≤ 0.64047, K̃ ≤ 59.9992,
c̃₁ ≤ 4,396.5, c̃₂ ≤ 2.969e7, diam(M, ω) ≤ 3.1638, diam(M/G, ω) ≤ 1.4712 and λ₁,G ≥ 4.5605.
The paper rounds each value in the safe direction, for example K̃ ≤ 60 and diam(M, ω) ≤ 3.165.

The mean of abs(f) on samples of points:

| fit | top degree | sample | mean abs(f) |
|---|---|---|---|
| f6t, one stage | 24 | 400,000 points, dd | 1.73e-22 (rms 2.38e-22, max 3.29e-21) |
| s2e, two stages | 36 | 8e6 points, dd | 2.74e-28 |
| s2e | 36 | 256 points, mpmath | (2.71 ± 0.16)e-28 |
| s2k | 40 | 256 points, mpmath | (9.84 ± 0.62)e-30 |
| s2o | 44 | 256 points, mpmath | (2.13 ± 0.12)e-31 |

The paper does not give s2k and s2o. At the degree 44 the dd error of f has the size of f.

## tmgpu is not public yet

The strict bound uses `tmgpu`, a library of Taylor models with strict error bounds on the GPU.
`tmgpu` is not public yet, so the install of all steps fails without access to it.
`quartic/cert/`, `scripts/cert_*.py`, `scripts/tm_local_forms.py`, `scripts/check_p0_sign.py`,
the examples 03 to 06 and 9 test suites need it. All other parts run without it: the fits, their
checks, the jets, the export of the potential (`level_oracle.py`, `level_matrices.py`),
`tm_box.py` (f at a point in arb), `check_file_symmetry.py` and the studies.

## Install

Install [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock` fixes the version of each package.

```
uv sync --locked                 # no GPU: tm_box.py, level_matrices.py, check_file_symmetry.py
uv sync --locked --extra cuda    # a GPU with CUDA 12, no tmgpu
uv sync --locked --extra gpu     # all steps, with CuPy and tmgpu
```

The fits need a GPU. The examples take the extra from `UV_EXTRA` (`gpu`, `cuda` or `none`). The
default `gpu` needs access to tmgpu: without it, set `UV_EXTRA=cuda`. CuPy compiles its kernels
with the CUDA toolkit: set `CUDA_PATH`, and the examples add `$CUDA_PATH/lib64` to
`LD_LIBRARY_PATH`. The runs of the paper used NVIDIA H200 GPUs with 141 GB of memory.

## Data

The Zenodo record https://zenodo.org/records/24420573 holds the files that the proof used (license CC-BY-4.0).
`examples/06_check_published.sh` downloads them and checks them.

| file | content | SHA-256 |
|---|---|---|
| `f6t_data.npz` | the potential: the base p₀ and the data of the levels | `69878fbcaa3d4ea44955ef341b162788c7e34831452f13332ad6200cbbf3dbc1` |
| `f6t_forms.npz` | the forms N_L of the levels | `2fecc9289368a192c6e4f809a61aef55291150922fd6273e14386a4704bb3cf2` |
| `cover_s0.03_c6_v3.npz` | the cover of the fundamental domain F | `323195788563e91b4561b803ebd7de05158021f7db94fd462d5a5bf367f1071a` |
| `f6t_local_manifest_v3.txt` | the SHA-256 of the 332 files of local forms | `b04bb801006a7e892e3f57e9f5918e2d56f658ecaf8b798de0cb28ea59713e2b` |
| `run_0.npz` | the records of the bound, part 0 | `3c60ece6fb396898b500191a3b2045851b84d1e2598d532ad3ffd1e54699a598` |
| `run_1.npz` | part 1 | `7216b7e6c1322103fe8a5a1d67cc525f431203c60287892d619910b1c3eec6cb` |
| `run_2.npz` | part 2 | `c4805a49516e4fd5563fc8211e8dc53a45b4c1b6abdc6c4b3d346201eb18e1cd` |
| `run_3.npz` | part 3 | `6b9fbec7344517748715b5d54217b1c4e1cdd2ca1929b3e37f5730162c02978e` |
| `metric_0.npz` | the records of the bounds of the metric, part 0 | `48de78875c00176a1e708b0e0b7311af84c4db5696628ff15db0ce669a682efa` |
| `metric_1.npz` | part 1 | `b544766a5f30af33a75b1a75bf4a6c992287999d78fd913cdd1d67a92238d68e` |
| `metric_2.npz` | part 2 | `080d206713f8646a278469f7a26c057586ac324a50f56393d1ee686a741c2da5` |
| `metric_3.npz` | part 3 | `9dd23a41c1f77b51521d57d542ea98292d38a6442316e888ab7da75bf36737b5` |
| `atlas_fs_v3_tol03.npz` | the bounds of the Fubini–Study metric | `af9f76e5e688232ec900044e273fe7929f125b2695de58d61746e5a37a781e41` |
| `atlas_trd_v1_010203101213.npz` | the bounds of the transition maps, charts 01 to 13 | `8499cb126489db6f091394b91b6bc1388e182112ff756c505ec3556aba9bcc04` |
| `atlas_trd_v1_2021233031.npz` | the bounds of the transition maps, charts 20 to 31 | `b9daae5c795bf51eca82805fd4ad0c6c6903f6fe4086fa41f6172d503102fee5` |
| `atlas_combine_v3.npz` | the constants of the paper | `ffbd622835dd104cb2ab5c2d93fcc13cd4c828053b25015e0241bc898ae6f162` |

The local forms of the 332 reference centers (211 GB) are not in the record.
`examples/03_local_forms.sh` makes them again from `f6t_data.npz` and `f6t_forms.npz`.
A new build of the reference centers 0, 243 and 297 gave the SHA-256 values of the manifest.

## Reproduce

Each script of `examples/` gives its hardware, its time for the paper and its variables.

| script | step |
|---|---|
| `01_fit_f6t.sh` | the fit f6t: the HN base at k₀ = 16 and the levels 24 to 13 |
| `02_export_potential.sh` | the two files of the potential, `f6t_data.npz` and `f6t_forms.npz` |
| `03_local_forms.sh` | the cover of F and the local forms of its reference centers |
| `04_bound.sh` | the strict bounds of f and of grad f, and their checks |
| `05_metric_bounds.sh` | the bounds of the metric and the constants of the paper |
| `06_check_published.sh` | the checks of the files of the Zenodo record, with no GPU |
| `07_fit_two_stage.sh` | a fit of two stages with the top degree 36, 40 or 44 |
| `08_check_fit.sh` | the checks of a fit: hold-out, second chart, mpmath references |
| `09_jets.sh` | the derivatives of f from Taylor jets |

`examples/run_tests.sh` runs all test suites. With no `tmgpu`, it skips the 9 suites of the bound.
`scripts/studies/` holds the scripts of the other numbers of the paper.

## Layout

| folder | content |
|---|---|
| `quartic/` | the potential, the dd arithmetic, the bases of the levels, the fits and the jets |
| `quartic/cert/` | the parts of the strict bound that are special to the quartic (needs `tmgpu`) |
| `scripts/` | the command-line programs of the fits, the checks, the export and the bound, and the exact count of the invariant functions (`symmetry_param_counts_s4.py`) |
| `scripts/studies/` | the scripts of other numbers of the paper |
| `examples/` | one script for each step, and `run_tests.sh` |
| `tests/` | the test suites |

## License and citation

The code has the MIT license (`LICENSE`). The files of the Zenodo record have the license
CC-BY-4.0. If you use the code or the data, cite the paper. `CITATION.cff` gives the reference.
