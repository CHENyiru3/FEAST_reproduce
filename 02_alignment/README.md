# Subtask: Alignment Benchmark

Tests spatial alignment methods (Spateo, PASTE) on FEAST-simulated DLPFC Visium
slices rotated at 1°, 5°, 10°, 30°, and 45°. Reference = raw Visium data. Moving =
FEAST-simulated with expression perturbation, then rotated. Measures nn_region_accuracy,
spatial error, and gene expression correlation.

## Pipeline

```
run_simulation.py  →  run_methods.sh  →  benchmark.py  →  plot_metrics.py
```

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Simulate | `run_simulation.py` | Raw 151675.h5ad | base + 4 alterations × 5 rotations = 25 `.h5ad` |
| 2. Align | `run_methods.sh` | Simulation `.h5ad` files | Aligned coordinates per method/alt/angle |
| 3. Benchmark | `benchmark.py` | Method outputs | `alignment_benchmark_results.csv` |
| 4. Plot | `plot_metrics.py` | Benchmark CSV | Figure 2 alignment panels |

## Quick Run

```bash
export FEAST_DATA_ROOT=/path/to/processed_datasets
export PYTHONPATH=$(pwd)/../../FEAST/src:$PYTHONPATH

# Generate simulations (2-3 hours)
python run_simulation.py --seed 2026 --output-dir outputs/

# Run methods
FIG2_ALIGNMENT_METHODS="spateo paste" \
FIG2_ALIGNMENT_ANGLES="1 5 10 30 45" \
  bash run_methods.sh

# Benchmark (1 min)
python benchmark.py \
  --simulation-manifest outputs/simulation_manifest.csv \
  --methods-dir outputs/methods \
  --output outputs/alignment_benchmark_results.csv

# Plot
python plot_metrics.py --benchmark outputs/alignment_benchmark_results.csv
```

## Results

Current production defaults compare Spateo against PASTE. Previous SPACEL
outputs were archived because the SPACEL runner used inherited layer labels,
making spatial metrics insensitive to FEAST expression alterations.

## Environments

| Script | Conda Env |
|--------|-----------|
| `run_simulation.py`, `benchmark.py`, `plot_metrics.py` | `feast-py311-conda` |
| Spateo via `run_methods.sh` | `feast-py311-conda` |
| PASTE via `run_methods.sh` | `feast-py311-conda` with `paste-bio` |

## Design Notes

**v2 redesign**: v1 generated one simulated slice and rotated it — reference and moving
had byte-identical expression, trivializing spot matching. v2 uses raw Visium as reference,
testing actual alignment under expression perturbation.

**Critical flags**:
- Spateo: `nn_init=False` (skip expression-based MNN init — maps to wrong spots)
- PASTE: `alpha=0.1`, `dissimilarity=euclidean`, `use_rep=X_pca`
- PASTE: current `paste-bio` uses an older POT line-search callback signature;
  `run_paste.py` applies a local compatibility shim before calling
  `paste.pairwise_align`.
