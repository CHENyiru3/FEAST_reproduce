# Subtask: Deconvolution Benchmark

Benchmarks Cell2location and RCTD deconvolution methods on FEAST-simulated pseudo-bulk
spots from Allen_Zhuang ABCA-1 MERFISH data (slices 007, 050, 100). Measures JSD,
Pearson correlation, and RMSE between predicted and ground-truth cell-type proportions.

## Pipeline

```
run_deconvolution.py  →  benchmark.py  →  visualization.py
```

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Simulate | `run_deconvolution.py` | Raw h5ad + scRNA-seq reference | Pseudo-bulk spots (2 resolutions × 3 slices) |
| 2. Deconvolve | `run_deconvolution.py` (--cell2loc) | Pseudo-bulk spots | Cell2location proportions (25000 epochs) |
| 3. Benchmark | `benchmark.py` | Proportions + ground truth | `deconvolution_benchmark_results.csv` |
| 4. Plot | `visualization.py` | Benchmark CSV | Figure 2 deconvolution panels |

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# Generate pseudo-bulk spots (1-2 hours)
python run_deconvolution.py --seed 2026 --output-dir outputs/ --simulate-only

# Run Cell2location (4-10 hours — longest step)
for slice in 007 050 100; do
  for res in 0.1 0.25; do
    conda run -p /path/to/cell2loc_env python run_deconvolution.py \
      --input outputs/simulations/${slice}/resolution_${res}.h5ad \
      --reference $FEAST_DATA_ROOT/Allen_Zhuang_ABCA_1/sc_ref.h5ad \
      --cell-type-key class \
      --output outputs/cell2location/${slice}/resolution_${res}_proportions.csv \
      --seed 2026
  done
done

# Benchmark (1 min)
python benchmark.py \
  --truth-dir outputs/ground_truth \
  --prediction-dirs outputs/cell2location \
  --slices 007 050 100 \
  --resolutions 0.1 0.25 \
  --output outputs/deconvolution_benchmark_results.csv
```

## Results

| Method | Mean JSD | Mean RMSE | N |
|--------|----------|-----------|---|
| cell2location | 0.917 | 0.032 | 6 |
| RCTD | 0.784 | 0.066 | 6 |

## Environments

| Script | Conda Env |
|--------|-----------|
| `run_deconvolution.py`, `benchmark.py`, `visualization.py` | `feast-py311-conda` |
| Cell2location training | `cell2loc_env` |
| RCTD | R in `feast-py311-conda` |

## Known Issues

**Cell2location column prefix**: `export_posterior` prepends `meanscell_abundance_w_sf_`
to cell-type column names. `benchmark.py` strips this prefix before comparison.

**Low Pearson**: Both methods produce near-zero Pearson — cell-type proportions don't
linearly correlate with ground truth at this resolution. The Allen_Zhuang ABCA-1 reference
has 574–1000+ cell types; pseudo-bulk spots at 0.1–0.25 downsampling may not preserve
enough signal.
