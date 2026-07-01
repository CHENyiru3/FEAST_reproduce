# Subtask: 3D Semi-Reference Stack Reconstruction

Reconstructs intermediate target slices between real reference slices in a 3D tissue
stack. Uses the current FEAST de novo conditional API (`fit_reference` +
`simulate_from_reference`) to generate expression at target z-levels from two
bracketing reference slices. `assignment_randomness` is auto-selected once per
density from representative reference slices and then held fixed across all
target slices in that density.

## Dataset

Allen_Zhuang ABCA-1 MERFISH: 150 coronal slices (001–150), ~10k spots each, 1122 genes.
Label key: `class`.

## Density levels

| Density | Gap | Radius | Targets | References |
|---------|-----|--------|---------|------------|
| Dense | 3 | 1 | 49 | 95 |
| Medium | 5 | 2 | 29 | 57 |
| Sparse | 10 | 5 | 15 | 16 |

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# Run all densities
python run.py \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --densities 3 5 10 \
  --label-key class \
  --seed 2026 \
  --output-dir outputs/

# Run a single density
python run.py \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --densities 3 \
  --label-key class \
  --seed 2026 \
  --output-dir outputs/

# Run with a fixed AR override instead of auto-selection
python run.py \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --densities 3 \
  --label-key class \
  --assignment-randomness 0.3 \
  --seed 2026 \
  --output-dir outputs/

# With z-regularization
python run.py \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --densities 3 5 10 \
  --label-key class \
  --seed 2026 \
  --z-regularize \
  --output-dir outputs/
```

## Outputs per target

Each target directory contains:
- `generated.h5ad` — generated expression + spatial_3d coordinates
- `per_gene_metrics.csv` — per-gene Pearson, Spearman, Moran's I
- `moran_metrics.csv` — Moran's I comparison
- `per_class_metrics.csv` — per-class mean expression
- `panel_summary.json` — aggregate metrics

Each density directory also writes `metadata.json` with the selected
`assignment_randomness` and the per-reference auto-AR estimates used to choose it.

## Previous Results

The checked-in `outputs/` summaries were produced before this conditional-API
refactor. Rerun this subtask to refresh the values below under the new backend.

| Density | Targets | mean_corr | moran_corr | z_coherence |
|---------|---------|-----------|------------|-------------|
| Dense (gap=3) | 49 | 0.962 | 0.760 | 0.237 |
| Medium (gap=5) | 29 | 0.963 | 0.747 | 0.322 |
| Sparse (gap=10) | 15 | 0.964 | 0.729 | 0.370 |

Cross-density summary: `results/cross_density_summary.csv`

## Environment

`feast-py311-conda`
