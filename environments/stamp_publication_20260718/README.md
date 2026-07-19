# STAMP publication environment (2026-07-18)

This is the frozen inventory for the isolated Study 04 STAMP environment.
It was created as a full-copy clone of the existing site-local `sctm`
environment; the source environment was not modified.

The retained scientific stack is scTM 0.1.3, NumPy 1.26.4, PyTorch
2.6.0+cu124, and CUDA 12.4. The unused conflicting SpatialData branch
(`s3fs`, `xarray-dataclass`, `spatialdata`,
`multiscale-spatial-image`, and `spatial-image`) was removed. Squidpy was
replaced with 1.2.2, the minimum declared by scTM, together with leidenalg
0.12.0, igraph 1.0.0, and texttable 1.7.0.

The four offline wheels were verified before installation:

| Distribution | SHA-256 |
|---|---|
| squidpy 1.2.2 | `136a460d35cbdd52099cc3184b9e48c0b304f05e48a3afe32f9365a296c6645a` |
| leidenalg 0.12.0 | `fcbd89655243cc739d28ba23fe894587d8d7235124a9248ba375819585e70090` |
| igraph 1.0.0 | `2d04c2c76f686fb1f554ee35dfd3085f5e73b7965ba6b4cf06d53e66b1955522` |
| texttable 1.7.0 | `72227d592c82b3d7f672731ae73e4d1f88cd8e2ef5b075a7a7f01a23a3743917` |

Validation completed on 2026-07-18:

- `pip check`: no broken requirements.
- Import smoke test: scTM 0.1.3, NumPy 1.26.4, PyTorch 2.6.0+cu124.
- CUDA execution: a real tensor operation completed on one NVIDIA RTX A6000.
- FEAST is not imported or executed in this external method environment.

`pip-list.txt` is the complete Python distribution inventory.
`conda-explicit.txt` records the conda-managed base packages.

