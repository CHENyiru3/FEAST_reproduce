# FEAST Reproduction Experiments

Reproducibility package for the FEAST spatial transcriptomics simulator benchmark
(Figure 2) and de novo conditional transfer experiments (2D cross-slice / mask-complete,
3D semi-reference stack).

## Quick Start

### Prerequisites

- Linux x86_64
- Conda (Miniconda or Anaconda)
- ~100 GB disk space (simulation outputs)
- No GPU required (all experiments run CPU-only)

### Setup

```bash
# 1. Create FEAST environment
conda env create -f environments/feast-py311-conda.yml

# 2. Install FEAST
cd /path/to/FEAST
conda run -p /path/to/feast-py311-conda python -m pip install --no-deps -e .

# 3. Create external method environments
conda env create -f environments/stagate.yml      # Python 3.8 + TF 1.x
conda env create -f environments/graphst.yml
conda env create -f environments/spacel.yml
conda env create -f environments/cell2loc.yml
```

### Reproduce All Experiments

```bash
# Set data root
export FEAST_DATA_ROOT=/path/to/processed_datasets

# Run everything
make all

# Or run individual pipelines
make subtask_01
make subtask_02
make subtask_03
make subtask_04
make subtask_05
make subtask_06
```

---

## Repository Structure

```
feast-reproduce/
├── README.md
├── Makefile
├── environments/                    # Conda environment exports
├── configs/                         # YAML experiment configs
├── data/
│   └── manifests/                   # Required dataset manifests
├── 01_2d_conditional_transfer/      # Subtask: cross-slice + mask-complete
│   ├── README.md
│   ├── run.py
│   └── results/{dlpfc,merfish}/
├── 02_3d_stack/                     # Subtask: 3D semi-reference stack
│   ├── README.md
│   ├── run.py
│   └── results/
├── 03_clustering/                   # Subtask: Figure 2 clustering
│   ├── README.md
│   ├── run_simulation.py
│   ├── run_hvg.py
│   ├── methods/                     # GraphST, STAGATE_mclust, Leiden
│   ├── run_pipeline.sh
│   ├── benchmark.py
│   ├── plot_metrics.py
│   └── results/
├── 04_alignment/                    # Subtask: Figure 2 alignment
│   ├── README.md
│   ├── run_simulation.py
│   ├── run_methods.sh               # Spateo + SPACEL
│   ├── benchmark.py
│   ├── plot_metrics.py
│   └── results/
├── 05_deconvolution/                # Subtask: Figure 2 deconvolution
│   ├── README.md
│   ├── run_deconvolution.py
│   ├── benchmark.py
│   └── results/
├── 06_simulator_benchmark/          # Subtask: Multi-simulator quality
│   ├── README.md
│   ├── run.sh
│   ├── scripts/
│   ├── tests/
│   └── results/
└── SPEC/                            # Design documents (reference)
    └── Reproduce_SPEC/
```

## Results Summary

### 3D Stack Reconstruction (Zhuang-ABCA-1, 150 slices)

| Density | Targets | mean_corr | moran_corr |
|---------|---------|-----------|------------|
| dense (gap=3) | 49 | 0.962 | 0.760 |
| medium (gap=5) | 29 | 0.963 | 0.747 |
| sparse (gap=10) | 15 | 0.964 | 0.729 |

### 2D De Novo — DLPFC (151675↔151676, 3000 HVG)

| Mode | mean_corr | moran_corr |
|------|-----------|------------|
| Cross-slice | 0.999 | 0.82 |
| Mask-complete | 0.996 | 0.64 |

### 2D De Novo — MERFISH (Zhuang-ABCA-1 006↔007, 1122 genes)

| Mode | mean_corr | moran_corr |
|------|-----------|------------|
| Cross-slice | 0.985 | 0.65 |
| Mask-complete | 0.995 | 0.65 |

### Figure 2 — Clustering (DLPFC, 3 slices × 23 alterations)

| Method | Mean ARI |
|--------|----------|
| STAGATE_mclust | 0.324 |
| GraphST | 0.310 |
| Leiden | 0.162 |

### Figure 2 — Alignment (DLPFC 151675, 5 angles)

| Angle | SPACEL nn_region_acc | Spateo nn_region_acc |
|-------|---------------------|---------------------|
| 1° | 0.95 | **1.00** |
| 5° | **0.90** | 0.89 |
| 10° | **0.94** | 0.66 |
| 30° | **0.87** | 0.04 |
| 45° | **0.89** | 0.00 |

### Figure 2 — Deconvolution (Allen_Zhuang ABCA-1, 3 slices)

| Method | Mean JSD | Mean RMSE |
|--------|----------|-----------|
| cell2location | 0.917 | 0.032 |
| RCTD | 0.784 | 0.066 |

### Simulator Quality Benchmark (8 datasets)

| Simulator | Composite | Zero Pres. | Novelty |
|-----------|-----------|------------|---------|
| FEAST_Rank | **0.816** | 0.941 | **0.893** |
| Splatter | 0.640 | 0.858 | 0.501 |
| FEAST_OT | 0.556 | 0.729 | 0.068 |
| SRTsim | 0.539 | **0.984** | 0.000 |

## Datasets Required

Processed `.h5ad` files must be available at `$FEAST_DATA_ROOT`:

| Dataset | Slices | Key |
|---------|--------|-----|
| spatialLIBD DLPFC Visium | 151508, 151670, 151675, 151676 | `ground_truth` |
| Allen_Zhuang ABCA-1 (MERFISH) | 001–150 (3D), 006, 007 (2D), 007, 050, 100 (deconv) | `class` |
| Allen_Zhuang ABCA-1 scRNA-seq ref | Single-cell reference | `class` |

See `data/manifests/` for exact file lists.

## Citation

If you use these experiments in your research, please cite the FEAST paper and this repository.

## License

MIT
