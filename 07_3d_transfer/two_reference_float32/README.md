# Study 07: two-reference float32 reproduction

These files are the existing generation and postprocessing code used for
`outputs/generative_transfer_two_reference_float32_v1`, recovered from the
local September 6 run directory. Only filesystem paths were made relative to
the repository layout. Scientific settings and validation logic are retained.

## Setup

Use the recorded Python 3.11 environment with NumPy 1.26.4, POT 0.9.7, CUDA
PyTorch, and the FEAST local-generative source used for this experiment.
This workflow imports local FEAST source and records runtime details; it does
not perform the immutable-wheel verification of the older parent workflow.
The environment name alone is not an exact FEAST source version.

Edit the path entries in `config.yaml` for your FEAST checkout, processed data,
expression-free blueprints, and fresh output/work directories. Paths resolve
relative to the configuration file. The default layout expects `FEAST/` and
`Datasets/Processed/` alongside `FEAST_reproduce/`. Input identities remain in
`../data/input_checksums.csv`. Existing blueprints must be provided; the older
`../prepare.py` documents their construction.

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

Build the report using the [Study 07 plotting commands](../../visualization/07_3d_transfer/README.md).
No input datasets, generated results, caches, or logs are shipped with this code.
