# Study 00: simulator benchmark

This clean workflow creates 12 new FEAST reference-rank simulations and
compares them with four retained external simulators. The historical FEAST OT
arm is deliberately absent, so the fresh atomic metric table has 60 rows:
12 samples times 5 simulators.

## Inputs

Materialize files below `data/local/` using the standardized paths declared in
`data/input_checksums.csv`:

```text
data/local/reference/<sample>.h5ad
data/local/external/<simulator>/<sample>.h5ad
```

The 12 reference datasets and 48 external outputs must match the recorded
SHA-256 values. No earlier FEAST output is a permitted input. The external
outputs are read-only; the scripts never modify them.

Verify the materialized inputs before running:

```bash
python validate.py --stage inputs --data-root data/local
```

## Run

Use the supported FEAST environment and a new output path:

```bash
python run.py \
  --config config.yaml \
  --reference-dir data/local/reference \
  --output-dir outputs/release_candidate

python score.py \
  --config config.yaml \
  --data-root data/local \
  --simulation-dir outputs/release_candidate \
  --output-dir outputs/release_candidate_metrics

python validate.py \
  --stage all \
  --data-root data/local \
  --simulation-dir outputs/release_candidate \
  --metrics-dir outputs/release_candidate_metrics
```

`run.py --dry-run` prints the 12-job matrix without importing FEAST or writing
files. Every scientific output directory must initially be new. If a run is
interrupted, repeat the command with `--resume`; only outputs matching their
recorded configuration, runner-source, input, and output hashes are skipped,
while invalid partial H5ADs are
preserved under the run's `failures/` directory.

The FEAST call is fixed to public `FEAST.simulate()` with seed 2026,
`spatial_mode="reference_rank"`, global SciPy assignment, and no assignment
blocking. There is no OT configuration or internal simulator shortcut.

## Outputs

The simulation root contains 12 H5AD files plus `simulation_manifest.csv` and
`provenance.json`. Scoring produces `simulator_quality_metrics.csv`,
`moran_gene_panel.csv`, and `provenance.json`. Metrics use exact gene joins,
exact named spot support where available, one ordered reference-derived Moran
panel, and no composite score.

## Pairwise-support repair

The frozen external H5AD files are not modified. When observation identifiers
do not overlap, scoring may use spatial coordinates only if the coordinates are
finite and form a complete, unique bijection at the configured eight-decimal
precision. Partial support and duplicate coordinates are rejected.

The bounded repair command is:

```bash
python repair_pairwise_support.py
```

It writes only to the new
`outputs/final_rerun_20260718_metrics_final/` root. Twenty of the 21 formerly
unsupported rows have verified coordinate bijections. `scCube/Slideseq_001`
remains unscored because 35,054 simulated rows contain only 34,598 unique
coordinates. Pairwise aggregation uses the same 11 complete datasets across
all five simulators. The prior all-five-method pairwise ranking was
indeterminate, so publication and figure promotion remain stopped for author
review.
