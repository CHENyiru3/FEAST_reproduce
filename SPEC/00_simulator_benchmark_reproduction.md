# 00_simulator_benchmark — Reproduction Plan

**Date:** 2026-06-22
**Working directory:** `/maiziezhou_lab2/yiru/FEAST_experiments/00_simulator_benchmark`
**FEAST package:** `/maiziezhou_lab2/yiru/FEAST` (main branch, commit `cc53121`)

---

## 1. Pipeline (7 stages)

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `scripts/prepare_simulator_inputs.py` | Materialize input .h5ad files → `exper_data/` |
| 2 | `scripts/external_simulators/build_r_split_inputs.py` | Convert h5ad → R-friendly CSVs/MTX → `exper_data/R_split/` |
| 3 | `scripts/ParameterCloud_sim.py` | Run FEAST simulation (2 modes × N samples) |
| 4 | `scripts/external_simulators/{SRTsim,splat,splatSimple}_run.R`, `sccube_run.py`, `reconstruct_external_h5ad.py` | Run external simulators |
| 5 | `scripts/build_simulator_inventory.py` | Scan outputs → `simulator_inventory.csv` |
| 6 | `scripts/build_simulator_quality_metrics.py` | Compute 10 atomic + 4 composite metrics |
| 7 | `scripts/plot_simulator_quality.py` | Generate boxplot + heatmap PNGs |

Run Stages 1→7 sequentially. Stages 1-2 are idempotent (can re-run safely). Stages 3-4 have per-sample skip-if-output-exists logic. Stages 5-7 always re-run on the full inventory.

---

## 2. Environments

| Name | Path | Used for |
|------|------|----------|
| `feast-py311-conda` | `/maiziezhou_lab2/yiru/envs/feast-py311-conda` | FEAST, R_split builder, h5ad reconstruction, inventory, metrics, viz |
| `srtsim-r` | `/maiziezhou_lab2/yiru/envs/srtsim-r/bin/Rscript` | SRTsim (R) |
| `splatter-r46` | `/maiziezhou_lab2/yiru/envs/splatter-r46/bin/Rscript` | Splatter, Splatter_Simple (R) |
| `sccube` | `/maiziezhou_lab2/yiru/envs/sccube/bin/python` | scCube (Python) |

```bash
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=/maiziezhou_lab2/yiru/FEAST/src:$PYTHONPATH

FEAST_ENV=/maiziezhou_lab2/yiru/envs/feast-py311-conda
PYTHON=${FEAST_ENV}/bin/python
RSCRIPT_SRTSIM=/maiziezhou_lab2/yiru/envs/srtsim-r/bin/Rscript
RSCRIPT_SPLATTER=/maiziezhou_lab2/yiru/envs/splatter-r46/bin/Rscript
PYTHON_SCCUBE=/maiziezhou_lab2/yiru/envs/sccube/bin/python
SEED=2026
```

---

## 3. Input Datasets (13)

All symlinked under `exper_data/`. Each has `spatial` in `obsm` and `ground_truth` in `obs.columns`.

| Alias | Technology | Source | Spots × Genes |
|-------|-----------|--------|---------------|
| DLPFC_151670 | 10x Visium | spatialLIBD | 3,480 × 18,326 |
| DLPFC_151675 | 10x Visium | spatialLIBD | 3,565 × 18,641 |
| DLPFC_151676 | 10x Visium | spatialLIBD | 3,431 × 18,661 |
| MERFISH_006 | MERFISH | Allen Brain Atlas | 10,442 × 1,122 |
| MERFISH_007 | MERFISH | Allen Brain Atlas | 9,693 × 1,122 |
| MERFISH_120 | MERFISH | Allen Brain Atlas | 22,270 × 1,122 |
| OpenST_005 | OpenST | GEO:GSE251926 | 47,217 × 28,943 |
| OpenST_006 | OpenST | GEO:GSE251926 | 38,555 × 28,943 |
| Stereoseq_E9_5_E2S2 | Stereo-seq | MOSTA | 4,356 × 21,052 |
| Stereoseq_E10_5_E2S1 | Stereo-seq | MOSTA | 8,494 × 22,385 |
| Stereoseq_E12_5_E2S1 | Stereo-seq | MOSTA | 23,640 × 24,620 |
| Slideseq_001 | Slide-seqV2 | SODB/Tencent | 35,054 × 23,197 |
| Xenium_LymphNode | Xenium | 10x Genomics | 377,957 × 523 |

Defined in `scripts/prepare_simulator_inputs.py` → `ALIAS_REGISTRY`.

---

## 4. Stage Commands

### Stage 1 — Prepare Inputs

```bash
mkdir -p outputs/benchmarks
${PYTHON} scripts/prepare_simulator_inputs.py \
    --datasets-root /maiziezhou_lab2/yiru/Datasets \
    --output-dir exper_data \
    --manifest outputs/benchmarks/input_manifest.csv \
    --seed ${SEED}
```

### Stage 2 — Build R_split

```bash
${PYTHON} scripts/external_simulators/build_r_split_inputs.py \
    --input-dir exper_data \
    --output-dir exper_data/R_split \
    --manifest outputs/benchmarks/r_split_manifest.csv
```

### Stage 3 — FEAST Simulation

For each sample × each of 2 modes:

```bash
# FEAST_Rank:  parameter_mode=hungarian  spatial_mode=reference_rank  assignment=hybrid
# FEAST_OT_Spatial: parameter_mode=hungarian  spatial_mode=ot_spatial  assignment=hybrid

${PYTHON} scripts/ParameterCloud_sim.py \
    --input exper_data/${SAMPLE}.h5ad \
    --output outputs/simulation_saved/${MODE}/${SAMPLE}.h5ad \
    --parameter-mode hungarian \
    --spatial-mode ${SPATIAL_MODE} \
    --assignment-method hybrid \
    --annotation-key auto \
    --seed ${SEED}
```

**FEAST internals** (passed to `simulate_single_slice()`):
```python
simulate_single_slice(
    adata=adata,
    annotation_key=annotation_key,       # auto-detected → "ground_truth"
    parameter_mode="hungarian",          # copula-based fitting + Hungarian assignment
    spatial_mode="reference_rank"|"ot_spatial",
    target_adata=adata,                  # same-slice for ot_spatial (identity transport)
    assignment_method="hybrid",          # copula-rank + OT hybrid
    random_seed=2026,
    verbose=True,
    clip_overshoot_factor=0.0,           # no clipping
    boundary_multiplier=1.1,             # 110% of reference max
    use_heuristic_search=False,          # full search
)
```

### Stage 4 — External Simulators

**SRTsim** (R, `srtsim-r` env):
```bash
for SAMPLE in ...; do
    TMP=outputs/simulation_saved/srtsim_updated/_tmp/${SAMPLE}
    mkdir -p "${TMP}"
    ${RSCRIPT_SRTSIM} scripts/external_simulators/SRTsim_run.R \
        --cellinfo exper_data/R_split/${SAMPLE}/cellinfo.csv \
        --geneinfo exper_data/R_split/${SAMPLE}/geneinfo.csv \
        --matrix exper_data/R_split/${SAMPLE}/sparse_matrix.mtx \
        --spatial_loc exper_data/R_split/${SAMPLE}/spatial.csv \
        --output_dir "${TMP}"
    ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${TMP}/SRTsim_count.mtx" \
        --gene_path "${TMP}/SRTsim_genes.csv" \
        --location_path "${TMP}/SRTsim_loc.csv" \
        --save_adata_path outputs/simulation_saved/srtsim_updated/${SAMPLE}.h5ad
done
```

**Splatter** (R, `splatter-r46` env):
```bash
for SAMPLE in ...; do
    TMP=outputs/simulation_saved/splat/_tmp/${SAMPLE}
    mkdir -p "${TMP}"
    ${RSCRIPT_SPLATTER} scripts/external_simulators/splat_run.R \
        --cellinfo exper_data/R_split/${SAMPLE}/cellinfo.csv \
        --geneinfo exper_data/R_split/${SAMPLE}/geneinfo.csv \
        --matrix exper_data/R_split/${SAMPLE}/sparse_matrix.mtx \
        --spatial_loc exper_data/R_split/${SAMPLE}/spatial.csv \
        --output_dir "${TMP}"
    ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${TMP}/splat_count.mtx" \
        --gene_path "${TMP}/splat_genes.csv" \
        --location_path "${TMP}/splat_loc.csv" \
        --save_adata_path outputs/simulation_saved/splat/${SAMPLE}.h5ad
done
```

**Splatter Simple** (R, `splatter-r46` env) — uses default params, no estimation from input:
```bash
for SAMPLE in ...; do
    TMP=outputs/simulation_saved/splatSimple/_tmp/${SAMPLE}
    mkdir -p "${TMP}"
    ${RSCRIPT_SPLATTER} scripts/external_simulators/splatSimple_run.R \
        --cellinfo exper_data/R_split/${SAMPLE}/cellinfo.csv \
        --geneinfo exper_data/R_split/${SAMPLE}/geneinfo.csv \
        --matrix exper_data/R_split/${SAMPLE}/sparse_matrix.mtx \
        --spatial_loc exper_data/R_split/${SAMPLE}/spatial.csv \
        --output_dir "${TMP}"
    ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${TMP}/splatSimple_count.mtx" \
        --gene_path "${TMP}/splatSimple_genes.csv" \
        --location_path "${TMP}/splatSimple_loc.csv" \
        --save_adata_path outputs/simulation_saved/splatSimple/${SAMPLE}.h5ad
done
```

**scCube** (Python, `sccube` env):
```bash
for SAMPLE in ...; do
    # Auto-resolve label column
    CELL_KEY=$(${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('exper_data/${SAMPLE}.h5ad')
for c in ['ground_truth', 'cell_type', 'annotation', 'region', 'cluster']:
    if c in a.obs.columns:
        print(c)
        break
")
    ${PYTHON_SCCUBE} scripts/external_simulators/sccube_run.py \
        --h5ad_file exper_data/${SAMPLE}.h5ad \
        --output_dir outputs/simulation_saved/sccube/${SAMPLE}_sccube_sim.h5ad \
        --cell_key ${CELL_KEY:-ground_truth}
done
```

**Known exclusions:**
- `Stereoseq_E9_5_E2S2` excluded from SRTsim and Splatter (simulator limitations)

### Stage 5 — Build Inventory

```bash
${PYTHON} scripts/build_simulator_inventory.py \
    --output-base outputs/simulation_saved \
    --exper-data exper_data \
    --inventory outputs/benchmarks/simulator_inventory.csv
```

### Stage 6 — Build Quality Metrics

```bash
${PYTHON} scripts/build_simulator_quality_metrics.py \
    --inventory outputs/benchmarks/simulator_inventory.csv \
    --exper-data exper_data \
    --metrics outputs/benchmarks/simulator_quality_metrics.csv \
    --metadata outputs/benchmarks/simulator_quality_metric_metadata.json
```

10 atomic metrics: `input_mean_corr`, `input_variance_corr`, `zero_mask_jaccard`, `cosine_divergence`, `relative_error_mean`, `gene_zero_fraction_wasserstein`, `gene_mean_wasserstein`, `gene_variance_wasserstein`, `library_size_wasserstein`, `moran_i_correlation`.

4 composite scores: `zero_preservation_score`, `statistical_structure_score`, `structured_novelty_score`, `composite_quality_score`.

Identity flag thresholds: `mean_corr >= 0.995`, `variance_corr >= 0.95`, `zero_jaccard >= 0.95`.

### Stage 7 — Visualization

```bash
${PYTHON} scripts/plot_simulator_quality.py \
    --metrics outputs/benchmarks/simulator_quality_metrics.csv \
    --output-dir outputs/quality_visualization
```

---

## 5. Output Directory Structure

```
outputs/simulation_saved/
├── FEAST_Rank/           # 13 .h5ad
├── FEAST_OT_Spatial/     # 13 .h5ad
├── srtsim_updated/       # 12 .h5ad (excl. Stereoseq_E9_5_E2S2)
├── splat/                # 12 .h5ad (excl. Stereoseq_E9_5_E2S2)
├── splatSimple/          # 13 .h5ad
└── sccube/               # 13 .h5ad

outputs/benchmarks/
├── input_manifest.csv
├── r_split_manifest.csv
├── simulator_inventory.csv
├── simulator_quality_metrics.csv
└── simulator_quality_metric_metadata.json

outputs/quality_visualization/
├── simulator_quality_boxplots.png
└── simulator_quality_heatmap.png
```

---

## 6. Current State (2026-06-22)

| Stage | Status |
|-------|--------|
| 1 — Inputs | **DONE** — 13 files in `exper_data/` |
| 2 — R_split | **DONE** — 13 dirs in `exper_data/R_split/` |
| 3 — FEAST | 22 done (original 11 × 2 modes), **2 remaining: DLPFC_151675, DLPFC_151676** |
| 4 — External | Copied from reproduce for original 11 samples; **need to run for DLPFC_151675, DLPFC_151676** |
| 5 — Inventory | Needs re-run after 3-4 complete |
| 6 — Metrics | Needs re-run after 5 |
| 7 — Viz | Needs re-run after 6 |
