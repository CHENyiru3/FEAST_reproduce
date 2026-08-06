# Study 03: deconvolution

This workflow regenerates six FEAST simulations and then runs RCTD and
Cell2location on the exact same H5AD files. No previous FEAST simulation,
prediction, score, or truth artifact is used; fresh truth is generated from
the declared aggregation during this run.

The default input directory must contain the three raw, integer-count references named
`Zhuang-ABCA-1.007.h5ad`, `Zhuang-ABCA-1.050.h5ad`, and
`Zhuang-ABCA-1.100.h5ad`. The `cell_type` annotation and `obsm['spatial']` are
required. Local hardlinks may be placed in `data/local/`; their expected hashes
and verified input contracts are recorded in `data/input_checksums.csv`.

## Zhuang-ABCA-1.120 pre-Cell2location diagnostic

`config_120.yaml` declares a new, internally consistent two-resolution input
set for `Zhuang-ABCA-1.120`. It must not be combined with simulation, truth, or
prediction artifacts from the default 007/050/100 run. The diagnostic stage
creates the two FEAST aggregations and their ground-truth composition matrices,
but deliberately starts **no** external deconvolution method:

```bash
FEAST_PY=/path/to/supported-feast-env/bin/python
INPUTS=$(pwd)/data/local
OUTPUT=$(pwd)/outputs/diagnostic_120_20260806_v1
FEAST_COMMIT=68816e5c1862a6fa2a49bc30609d617c7fa4b449

# Isolate the two FEAST calls. This avoids retaining the first large reference
# in memory while the second resolution is generated.
$FEAST_PY run.py --input-dir "$INPUTS" --output-dir "$OUTPUT" \
  --feast-commit "$FEAST_COMMIT" --config config_120.yaml \
  --simulate-only --only-resolution 0.1 --resume
$FEAST_PY run.py --input-dir "$INPUTS" --output-dir "$OUTPUT" \
  --feast-commit "$FEAST_COMMIT" --config config_120.yaml \
  --simulate-only --only-resolution 0.25 --resume
$FEAST_PY run.py --input-dir "$INPUTS" --output-dir "$OUTPUT" \
  --feast-commit "$FEAST_COMMIT" --config config_120.yaml \
  --simulate-only --resume

$FEAST_PY ../visualization/03_deconvolution/prepare_120_diagnostic.py
```

The last command writes a common-coordinate single-cell/centre preview,
canonical 30-class alignment audit, average ground-truth proportions, and a
fixed ROI record under `visualization/03_deconvolution/figures/`. Empty
aggregate locations have a zero ground-truth row by construction; every
non-empty location must have a canonical proportion row sum of one. Do not
start Cell2location until this preview, its spot counts, and its fixed ROI have
been reviewed.

## Run

```bash
FEAST_PY=/path/to/supported-feast-env/bin/python
CELL2LOC_PY=/path/to/cell2location-env/bin/python
RSCRIPT=/path/to/rctd-env/bin/Rscript
FEAST_COMMIT=$(git -C /path/to/FEAST rev-parse HEAD)
INPUTS=$(pwd)/data/local
OUTPUT=/path/to/new/study03_run
HISTORICAL_SCORES=/path/to/frozen/deconvolution_benchmark_results.csv
PRIOR_SCORES=/path/to/frozen/deconvolution_same_expression_scores.csv
DIRECTION_AUDIT=/path/to/frozen/headline_direction_assessment.csv

$FEAST_PY run.py \
  --input-dir "$INPUTS" \
  --output-dir "$OUTPUT" \
  --feast-commit "$FEAST_COMMIT" \
  --rscript "$RSCRIPT" \
  --cell2location-python "$CELL2LOC_PY"

$FEAST_PY score.py \
  --run-dir "$OUTPUT" \
  --output-dir "$OUTPUT/scores" \
  --historical-scores "$HISTORICAL_SCORES" \
  --prior-scores "$PRIOR_SCORES" \
  --direction-audit "$DIRECTION_AUDIT"

$FEAST_PY validate.py \
  --run-dir "$OUTPUT" \
  --scores-dir "$OUTPUT/scores" \
  --output "$OUTPUT/validation.csv" \
  --historical-scores "$HISTORICAL_SCORES" \
  --prior-scores "$PRIOR_SCORES" \
  --direction-audit "$DIRECTION_AUDIT"
```

`run.py` requires a new output root and stops on the first method failure while
preserving its logs and failure metadata. RCTD allows six hours for the known
slow full-mode jobs and uses the declared writable cache. The Cell2location environment may use
an unsupported NumPy version; its metadata states explicitly that FEAST is not
imported or executed in that external method process.

Cell2location must run on CUDA. A successful row records the actual CUDA
device and a positive peak GPU allocation. Its exported abundance columns use
the package's exact `<summary>cell_abundance_w_sf_<cell_type>` prefix; the
wrapper validates that complete ordered schema before removing the prefix and
writing the named cell-type table. Positional or partial column matching is not
accepted.

After an interruption, rerun the same command with `--resume`. A simulation or
method row is skipped only when its configuration, FEAST build, public seed,
input hash, output hashes, and validated-success metadata still match. Any
stale partial artifacts are moved below the run's `failures/` directory before
that row is regenerated.

Scoring requires exact spot identity/order, raw integer counts, common named
cell types plus `__other__`, and distinct prediction/truth hashes. Zero-library
spots are retained in Cell2location output but excluded from both methods'
biological scores because RCTD legitimately omits them.

RCTD receives exact reference and spatial library sizes; the wrapper never
inflates low-UMI reference cells. Cell2location's `N_cells_per_location` is the
mean number of high-resolution source cells assigned to each aggregate
location during the fresh simulation—not mean UMI count. This construction
quantity uses geometry only, not cell-type labels, and is recorded in every
simulation and method artifact. The score stage emits side-by-side atomic
metrics, a six-pair support audit, and an exact 12-key old-versus-new table. The
comparison binds the frozen article scores, the prior repaired same-expression
scores, and the fresh rerun by SHA-256. Its deltas are classified as mixed
workflow changes and are not attributed to a single cause. The historical
aggregate majority is context only; the gate itself is atomic and stops if any
declared metric direction changes, ties, or is indeterminate. It never
authorizes an aggregate method-ranking or publication claim. `provenance.json`
binds the scorer, config, manifests, historical inputs, every scored input, and
every scientific output by SHA-256.

Both `score.py` and `validate.py` require explicit `--historical-scores`,
`--prior-scores`, and `--direction-audit` paths. This keeps the reproduction
repository portable while each file must still match its declared frozen
SHA-256 before it can enter comparison or provenance.

This study explicitly sets FEAST's public `clip_overshoot_factor=0.0`. The
optional post-decoding 1.1x clip can create fractional maxima (for example,
19.8) even though the count decoder itself returns integers. Disabling that
second clip preserves the raw-integer input contract required by both
deconvolution methods; FEAST's decoder-level boundary constraint remains
active. No rounding or silent count coercion is performed downstream.

Use `run.py --dry-run ...` to check paths and the 6-simulation/12-method matrix
without creating output or starting a method.
