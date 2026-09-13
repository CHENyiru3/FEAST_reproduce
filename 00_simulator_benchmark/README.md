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

## Pairwise-support repair and fresh scCube completion

The frozen external H5AD files are not modified. When observation identifiers
do not overlap, scoring may use spatial coordinates only if the coordinates are
finite and form a complete, unique bijection at the configured eight-decimal
precision. Partial support and duplicate coordinates are rejected.

The earlier bounded coordinate-support repair command was:

```bash
python repair_pairwise_support.py
```

It wrote only to `outputs/final_rerun_20260718_metrics_final/`. Twenty of the
21 formerly unsupported rows had verified coordinate bijections. The remaining
`scCube/Slideseq_001` artifact could not be repaired because its 35,054 rows
contained only 34,598 unique coordinates.

A fresh, identity-preserving scCube Slide-seq run was therefore scored into the
new `outputs/final_metrics/` root by `build_final_metrics.py`. Its 35,054 spots,
23,197 genes, identifiers, spatial rows, and gene order match the reference
exactly. The final table contains all 60 simulator/sample rows and has no
missing cosine-divergence values. The external scCube worker used Python 3.8
and NumPy 1.23.5, which are outside FEAST's supported environment; FEAST was not
imported in that worker.

This closes the missing pairwise values, but not publication authorization. An
ignored predecessor `metrics_v2` file was overwritten in place and its
expected original bytes are unavailable. The fresh provenance records that
source-lineage break. No all-method ranking, composite score, figure promotion,
or simulator-winner claim is authorized until the author dispositions it.
