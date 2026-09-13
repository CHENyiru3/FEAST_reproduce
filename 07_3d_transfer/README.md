# Study 07: DevCCF expression transfer

The current generation and postprocessing code is in `two_reference_float32/`.
It produced `outputs/generative_transfer_two_reference_float32_v1` using two
references per modeling region and float32 transport.

## Setup

Use the recorded Python 3.11 environment with NumPy 1.26.4, POT 0.9.7, CUDA
PyTorch, and the FEAST local-generative source used for this experiment.
This workflow imports local FEAST source and records runtime details; it does
not perform the immutable-wheel verification of the older full-reference workflow.
The environment name alone is not an exact FEAST source version.

Edit the path entries in `two_reference_float32/config.yaml` for your FEAST checkout, processed data,
expression-free blueprints, and fresh output/work directories. Paths resolve
relative to the configuration file. The default layout expects `FEAST/` and
`Datasets/Processed/` alongside `FEAST_reproduce/`. Input identities remain in
`data/input_checksums.csv`. Existing blueprints must be provided; the older
`prepare.py` documents their construction.

## Generation and evaluation

From the repository root, after installing dependencies and making the recorded
FEAST source importable:

```bash
FEAST_PY=/path/to/feast-environment/bin/python
FEAST_SOURCE=/path/to/recorded/FEAST
export PYTHONPATH="$FEAST_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=07_3d_transfer/two_reference_float32/config.yaml
CODE=07_3d_transfer/two_reference_float32
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

"$FEAST_PY" "$CODE/run.py" --config "$CONFIG" --age E15.5 \
  --shard-id production_0 --start-index 0 --stop-index 158
"$FEAST_PY" "$CODE/run.py" --config "$CONFIG" --age E18.5 \
  --shard-id production_0 --start-index 0 --stop-index 202
"$FEAST_PY" "$CODE/consolidate.py" --config "$CONFIG" --age E15.5
"$FEAST_PY" "$CODE/consolidate.py" --config "$CONFIG" --age E18.5
"$FEAST_PY" "$CODE/validate.py" --config "$CONFIG"
"$FEAST_PY" "$CODE/evaluate.py" --config "$CONFIG"
```

Generation requires CUDA and refuses an existing shard directory. Use fresh
configured roots for a new run. Assignment randomness remains the recorded
approved estimates: 0.30 for E15.5 and 0.50 for E18.5. The latter derives from a
superseded-model estimate, not a completed calibration of this model. The
configuration records that distinction.

## Other retained code

The files directly in this directory retain the older full-reference,
verified-wheel workflow and its tests. The current two-reference code stays in
`two_reference_float32/` to preserve the different validation/import requirements.
`conditional_resampling_baseline.py` retains the all-eligible-reference,
within-region whole-spot resampling control; supply a validated matching
configuration and a fresh output directory. It is not a two-reference ablation.

## Figures

Run from the repository root after validation.

```bash
FEAST_PY=/path/to/feast-environment/bin/python
INPUT=07_3d_transfer/outputs/generative_transfer_two_reference_float32_v1
FIGURES=visualization/07_3d_transfer/figures/reproduced
for script in plot_resampling_control plot_continuity_resampling_benchmark plot_expression_summaries plot_3d_gene_distribution plot_3d_region_four_views plot_axis_cross_sections; do
  "$FEAST_PY" "visualization/07_3d_transfer/$script.py" \
    --input-root "$INPUT" --output-dir "$FIGURES"
done
```

These commands require the completed validation/evaluation artifacts. The first
two also read the existing resampling control under
`07_3d_transfer/outputs/final/baselines/conditional_whole_spot_resampling/`;
that control must be supplied or reproduced separately. Atlas-based plots also
require the configured processed DevCCF volumes. Existing files are protected
by the entry points' overwrite checks; use a fresh figure directory.

The six commands produce coverage/continuity, expression summaries, 3D marker
and region views, and selected cross-sections. E15.5 has 158 levels and E18.5
has 202, both with 550 genes. Target expression is unobserved, so these are
descriptive displays, not biological-accuracy estimates. The existing control
uses all eligible references and ten resampling runs; its 5–95% range is not
a confidence interval. Cross-sections are independently selected per age,
not asserted to be homologous.
