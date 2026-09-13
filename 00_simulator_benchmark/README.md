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

The 12 reference datasets and 48 external outputs must satisfy the recorded
identifiers, dimensions, and semantic input contracts. No earlier FEAST output is a permitted input. The external
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
recorded configuration, runner-source, input, and output paths are skipped,
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

## Retained benchmark sources

The current figure source is `outputs/final_metrics/simulator_quality_metrics.csv`,
assembled by `build_final_metrics.py` using the fresh identity-preserving scCube
Slide-seq run. The external scCube environment uses Python 3.8/NumPy 1.23.5;
FEAST is not imported there. `repair_pairwise_support.py` records an earlier
coordinate-bijection repair and does not replace the final table.

The overwritten predecessor-table lineage remains unresolved. No composite,
overall winner ranking, or figure promotion is authorized without author review.

## Figures

Run from the repository root after validation.

```bash
python visualization/00_simulator_benchmark/plot.py
python visualization/00_simulator_benchmark/plot_delta_spatial.py
python visualization/00_simulator_benchmark/plot_slice_panel.py
python visualization/00_simulator_benchmark/plot_spatial_comparison.py
```

For a new metric run, pass `--metrics-csv`, `--decision-json`, and `--output-dir`
to `plot.py`. Metric-specific median ordering does not define an overall ranking.
Spatial expression and delta views use their recorded within-row/column scales;
scales are not interchangeable between datasets.
