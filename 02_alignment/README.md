# Study 02: alignment

This directory regenerates the 20 centered, identity-preserving rotation inputs
and then runs Spateo and PASTE for all 40 method/condition rows. It does not use
the legacy grid-snapping `transform_sequencing` path or any prior method output.

The only scientific input is the correct raw reference slice (`151675.h5ad`).
`prepare.py` first creates four fresh FEAST base simulations (baseline, mean,
variance, and sparsity) with global SciPy assignment, and then creates their 20
rotations. It never accepts an existing FEAST base H5AD as input.

## Run

```bash
FEAST_PY=/path/to/supported-feast-env/bin/python
SPATEO_PY=/path/to/spateo-env/bin/python
PASTE_PY=/path/to/paste-env/bin/python
REFERENCE=/path/to/151675.h5ad
FEAST_COMMIT=$(git -C /path/to/FEAST rev-parse HEAD)
OUTPUT=/path/to/new/study02_run

$FEAST_PY prepare.py \
  --reference "$REFERENCE" \
  --feast-commit "$FEAST_COMMIT" \
  --output-dir "$OUTPUT/rotations"

$FEAST_PY run.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --output-dir "$OUTPUT/methods" \
  --spateo-python "$SPATEO_PY" \
  --paste-python "$PASTE_PY"

$FEAST_PY score.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --methods-dir "$OUTPUT/methods" \
  --output-dir "$OUTPUT/scores"

$FEAST_PY validate.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --methods-dir "$OUTPUT/methods" \
  --scores-dir "$OUTPUT/scores" \
  --output "$OUTPUT/validation.csv"
```

All output roots must be new. A failed method keeps its own failure/log files,
but the workflow exits nonzero. PASTE is fail-closed: finite output alone is not
enough; every inner EMD call and the outer conditional-gradient solve must have
positive convergence evidence.

After an interruption, repeat the method command with `--resume`. Only rows
whose recorded input and complete artifact hashes validate are skipped;
incomplete rows are preserved below the method root's `failures/` directory.

Use `run.py --dry-run ...` to validate the 40-job matrix without creating an
output directory or starting a method.
