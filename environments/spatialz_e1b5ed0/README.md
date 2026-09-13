# SpatialZ external-method environment

This records the isolated environment prepared for the Study 06 SpatialZ
interpolation baseline. FEAST is not installed or run in this environment.

- Environment prefix: `/path/to/envs/spatialz`
- Python: 3.9.19
- SpatialZ repository: `https://github.com/senlin-lin/SpatialZ.git`
- SpatialZ commit: `e1b5ed01e933729149e08399febca08db16ced14`
- Source checkout: `$CONDA_PREFIX/SpatialZ-source`
- PyTorch: 1.13.0+cu117
- Verification GPU: NVIDIA RTX A6000

The source repository has no Python package metadata. `SpatialZ.py` and
`Synthesize.py` are therefore symlinked from the pinned checkout into the
environment's `site-packages` directory.

The environment follows the upstream Python 3.9 and pinned dependency
requirements. Its complete observed Python package inventory is in
`pip-list.txt`.

Verification completed on 2026-08-31:

- `pip check`: no broken requirements;
- imports: NumPy, pandas, AnnData, SciPy, scikit-learn, PyTorch, POT, Scanpy,
  Squidpy, Open3D, MENDER, and SpatialZ;
- CUDA tensor operation: passed on the RTX A6000;
- in-memory `Generate_spatialz` CUDA smoke test: generated a finite 6-by-3
  virtual slice with the expected spatial and cell-class fields.

Activate with:

```bash
conda activate /path/to/envs/spatialz
```

The smoke test establishes environment and API viability only. It does not
validate the Study 06 scientific baseline or select its consequential SpatialZ
parameters.
