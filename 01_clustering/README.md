# Subtask: Clustering Benchmark

Validates FEAST empirical-mode simulation quality by running three clustering methods
(GraphST, STAGATE+mclust, Leiden) on simulated DLPFC Visium slices and measuring ARI,
NMI, AMI, CHAOS, and PAS against ground-truth cortical layer labels.

## Pipeline

```
run_simulation.py  →  run_hvg.py  →  run_pipeline.sh  →  benchmark.py  →  plot_metrics.py
```

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Simulate | `run_simulation.py` | Raw h5ad (151508, 151670, 151676) | 69 `.h5ad` (3 slices × 23 alterations) |
| 2. Select HVGs | `run_hvg.py` | Simulation `.h5ad` files | 3000-gene `.h5ad` per simulation |
| 3. Cluster | `run_pipeline.sh` | HVG `.h5ad` files | `clusters.csv` per method/slice/simulation |
| 4. Benchmark | `benchmark.py` | Method outputs + simulations | `clustering_benchmark_results.csv` |
| 5. Plot | `plot_metrics.py` | Benchmark CSV | Figure 2 clustering panels |

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# Generate simulations (4-6 hours)
python run_simulation.py \
  --input-dir $FEAST_DATA_ROOT/spatialLIBD_DLPFC_Visium/h5ad \
  --slices 151508,151670,151676 \
  --output-dir outputs/simulations \
  --simulation-mode empirical \
  --seed 2026

# Select HVGs (5-10 min)
python run_hvg.py \
  --simulation-manifest outputs/simulation_manifest.csv \
  --output-dir outputs/hvg_inputs \
  --n-top-genes 3000

# Run methods (3-6 hours)
FIG2_CLUSTER_METHODS="GraphST STAGATE_mclust Leiden" bash run_pipeline.sh

# Benchmark (5 min)
python benchmark.py \
  --input-dir outputs/methods \
  --simulation-manifest outputs/simulation_manifest.csv \
  --output outputs/clustering_benchmark_results.csv

# Plot
python plot_metrics.py --benchmark outputs/clustering_benchmark_results.csv
```

## Results

| Method | Mean ARI | N |
|--------|----------|---|
| STAGATE_mclust | 0.324 | 75 |
| GraphST | 0.310 | 75 |
| Leiden | 0.162 | 75 |

## Environments

| Script | Conda Env |
|--------|-----------|
| `run_simulation.py`, `run_hvg.py`, `benchmark.py`, `plot_metrics.py`, `leiden.py` | `feast-py311-conda` |
| `stagate_mclust.py` | `STAGATE` (Python 3.8 + TF 1.x) |
| `graphst.py` | `GraphST` |

## Known Issue

`mean_0_10` alteration (fold_change=0.1) produces near-zero expression, causing scanpy HVG
to fail with singularity errors. Workaround: select top-3000 genes by raw variance.
See `run_hvg.py --flavor seurat` or bypass scanpy entirely for this edge case.
