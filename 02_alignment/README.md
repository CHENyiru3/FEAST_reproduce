# Study 02: alignment

This directory regenerates 28 fixed-plate, partial-overlap rotation inputs and
then runs Spateo and PASTE2 for all 56 method/condition rows. It does not use the
legacy grid-snapping `transform_sequencing` path or any prior method output.

The only scientific input is the correct raw reference slice (`151675.h5ad`).
`prepare.py` first creates four fresh FEAST base simulations (baseline, mean,
variance, and sparsity) with global SciPy assignment, and then creates their 28
rotations. It never accepts an existing FEAST base H5AD as input.

The fixed plate is the inclusive axis-aligned bounding box of the original
reference coordinates. Each slice rotates around `plate center + 0.5 × plate
half-extent`; spots outside the unchanged plate are removed after rotation. The
angles are 5°, 15°, 25°, 35°, 45°, 60°, and 75°, retaining 3,532, 3,397, 3,144,
2,918, 2,691, 2,384, and 2,058 of the original 3,565 spots, respectively. Each
moving slice therefore remains an ordered subset of the full reference support.

## Run

```bash
FEAST_PY=/path/to/supported-feast-env/bin/python
SPATEO_PY=/path/to/spateo-env/bin/python
PASTE2_PY=/path/to/paste2-env/bin/python
REFERENCE=/path/to/151675.h5ad
FEAST_COMMIT=6b10a227563856552d534bc185341ee0d2d9c5b0
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
  --methods spateo

$FEAST_PY run_paste2_grid.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --python "$PASTE2_PY" \
  --output-dir "$OUTPUT/paste2"

$FEAST_PY score.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --methods-dir "$OUTPUT/methods" \
  --methods spateo \
  --output-dir "$OUTPUT/spateo_scores_v3"

$FEAST_PY score_paste2.py \
  --reference "$REFERENCE" \
  --rotation-dir "$OUTPUT/rotations" \
  --run-dir "$OUTPUT/paste2" \
  --output-dir "$OUTPUT/paste2/scores_v3"

$FEAST_PY validate_metrics_v3.py \
  --spateo-scores "$OUTPUT/spateo_scores_v3" \
  --paste2-scores "$OUTPUT/paste2/scores_v3" \
  --rotation-dir "$OUTPUT/rotations" \
  --output "$OUTPUT/metrics_v3_validation.csv"
```

All output roots must be new. A failed method keeps its own failure/log files,
but the workflow exits nonzero. The primary cross-method metrics use the full
rigid-alignment layer: every retained moving spot is assigned to its nearest
reference coordinate. These metrics are `coordinate_nn_spot_accuracy`,
normalized spatial error, coordinate-NN region accuracy/ARI using the
`ground_truth` annotation, and coordinate-NN expression correlation.

The partial coupling is evaluated separately as an auxiliary transport layer.
All such fields are explicitly named `transport_*` (or transport coverage and
reference-support diagnostics). Zero-mass PASTE2 rows and columns remain
abstentions and are never converted into artificial complete pairings by
`argmax`.

After an interruption, repeat the method command with `--resume`. Only rows
whose recorded input and complete artifacts validate are skipped;
incomplete rows are preserved below the method root's `failures/` directory.

Use `run.py --methods spateo --dry-run ...` to validate the 28-job Spateo
matrix without creating an output directory or starting a method. PASTE2 has a
separate 28-job grid runner and scorer as shown above.
