# Subtask: 2D Conditional Transfer

Benchmarks FEAST de novo conditional reference generator on real 2D spatial slices.
A reference model is fitted on a source slice (expression + coordinates + labels) and
used to generate a target slice from geometry and labels alone.

## Modes

- **Cross-slice**: Train on slice A, generate slice B from blueprint only
- **Mask-complete**: Mask half of a slice, reconstruct the masked half

## Datasets

| Dataset | Slices | Genes | Label Key |
|---------|--------|-------|-----------|
| DLPFC Visium | 151675 ↔ 151676 | 3000 HVG / full | `ground_truth` |
| MERFISH Zhuang-ABCA-1 | 006 ↔ 007 | 1122 panel | `class` |

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# DLPFC cross-slice (3000 HVG)
python run.py \
  --mode cross_slice \
  --data-dir $FEAST_DATA_ROOT/spatialLIBD_DLPFC_Visium/h5ad \
  --annotation-key ground_truth \
  --directions 151675:151676 151676:151675 \
  --n-top-genes 3000 \
  --seed 2026 \
  --output-dir outputs/dlpfc/cross_3000

# DLPFC mask-complete
python run.py \
  --mode mask_complete \
  --data-dir $FEAST_DATA_ROOT/spatialLIBD_DLPFC_Visium/h5ad \
  --annotation-key ground_truth \
  --slices 151675 151676 \
  --n-top-genes 3000 \
  --seed 2026 \
  --output-dir outputs/dlpfc/mask_3000

# MERFISH cross-slice (full panel)
python run.py \
  --mode cross_slice \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --annotation-key class \
  --directions Zhuang-ABCA-1.006:Zhuang-ABCA-1.007 Zhuang-ABCA-1.007:Zhuang-ABCA-1.006 \
  --seed 2026 \
  --output-dir outputs/merfish/cross_full

# MERFISH mask-complete
python run.py \
  --mode mask_complete \
  --data-dir $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/h5ad \
  --annotation-key class \
  --slices Zhuang-ABCA-1.006 Zhuang-ABCA-1.007 \
  --seed 2026 \
  --output-dir outputs/merfish/mask_full
```

## Results

### DLPFC (3000 HVG)

| Mode | mean_corr | moran_corr |
|------|-----------|------------|
| Cross-slice | 0.999 | 0.82 |
| Mask-complete | 0.996 | 0.64 |

### MERFISH (1122 genes)

| Mode | mean_corr | moran_corr |
|------|-----------|------------|
| Cross-slice | 0.985 | 0.65 |
| Mask-complete | 0.995 | 0.65 |

## Environment

`feast-py311-conda`
