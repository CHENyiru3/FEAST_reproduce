#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# Figure 2 Simulator Benchmark — Remaining Production (DLPFC_151675 + 151676)
#
# Stages 1-2 already done. This completes:
#   Stage 3: FEAST for DLPFC_151675, DLPFC_151676 (2 modes each = 4 runs)
#   Stage 4: External sims for DLPFC_151675, DLPFC_151676 (4 sims each = 8 runs)
#   Stages 5-7: Inventory → Metrics → Visualization (full set)
#
# Run inside tmux:  bash run_remaining.sh
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

FEAST_ENV=/maiziezhou_lab2/yiru/envs/feast-py311-conda
PYTHON=${FEAST_ENV}/bin/python
RSCRIPT_SRTSIM=/maiziezhou_lab2/yiru/envs/srtsim-r/bin/Rscript
RSCRIPT_SPLATTER=/maiziezhou_lab2/yiru/envs/splatter-r46/bin/Rscript
PYTHON_SCCUBE=/maiziezhou_lab2/yiru/envs/sccube/bin/python
SEED=2026

EXPER_DATA=exper_data
R_SPLIT=exper_data/R_split
OUTPUTS=outputs/simulation_saved
BENCHMARKS=outputs/benchmarks
VIZ_DIR=outputs/quality_visualization
LOG_DIR=logs

mkdir -p "${LOG_DIR}" "${BENCHMARKS}" "${VIZ_DIR}"

SAMPLES=("DLPFC_151675" "DLPFC_151676")

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
banner() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; echo; }

# ============================================================================
# Stage 3: FEAST
# ============================================================================
banner "STAGE 3: FEAST for ${SAMPLES[*]}"

for sample in "${SAMPLES[@]}"; do
    for mode_label in "FEAST_Rank:reference_rank" "FEAST_OT_Spatial:ot_spatial"; do
        label="${mode_label%%:*}"
        spatial="${mode_label##*:}"
        out_dir="${OUTPUTS}/${label}"
        output_path="${out_dir}/${sample}.h5ad"
        mkdir -p "${out_dir}"

        if [[ -f "${output_path}" ]]; then
            log "SKIP ${label}:${sample} — exists"
            continue
        fi

        log "RUN  ${label}:${sample}"
        sample_log="${LOG_DIR}/feast_${label}_${sample}.log"
        if ${PYTHON} scripts/ParameterCloud_sim.py \
            --input "${EXPER_DATA}/${sample}.h5ad" \
            --output "${output_path}" \
            --parameter-mode hungarian \
            --spatial-mode "${spatial}" \
            --assignment-method hybrid \
            --annotation-key auto \
            --seed ${SEED} > "${sample_log}" 2>&1; then
            log "OK   ${label}:${sample}"
        else
            log "FAIL ${label}:${sample} — see ${sample_log}"
        fi
    done
done

# ============================================================================
# Stage 4: External Simulators
# ============================================================================
banner "STAGE 4: External Simulators for ${SAMPLES[*]}"

# --- SRTsim ---
for sample in "${SAMPLES[@]}"; do
    out_dir="${OUTPUTS}/srtsim_updated"
    output_path="${out_dir}/${sample}.h5ad"
    [[ -f "${output_path}" ]] && { log "SKIP SRTsim:${sample} — exists"; continue; }
    tmp_dir="${out_dir}/_tmp/${sample}"
    mkdir -p "${tmp_dir}" "${out_dir}"
    log "RUN  SRTsim:${sample}"
    sample_log="${LOG_DIR}/ext_SRTsim_${sample}.log"
    if ${RSCRIPT_SRTSIM} scripts/external_simulators/SRTsim_run.R \
        --cellinfo "${R_SPLIT}/${sample}/cellinfo.csv" \
        --geneinfo "${R_SPLIT}/${sample}/geneinfo.csv" \
        --matrix "${R_SPLIT}/${sample}/sparse_matrix.mtx" \
        --spatial_loc "${R_SPLIT}/${sample}/spatial.csv" \
        --output_dir "${tmp_dir}" > "${sample_log}" 2>&1 && \
       ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${tmp_dir}/SRTsim_count.mtx" \
        --gene_path "${tmp_dir}/SRTsim_genes.csv" \
        --location_path "${tmp_dir}/SRTsim_loc.csv" \
        --save_adata_path "${output_path}" >> "${sample_log}" 2>&1; then
        log "OK   SRTsim:${sample}"
    else
        log "FAIL SRTsim:${sample} — see ${sample_log}"
    fi
done

# --- Splatter ---
for sample in "${SAMPLES[@]}"; do
    out_dir="${OUTPUTS}/splat"
    output_path="${out_dir}/${sample}.h5ad"
    [[ -f "${output_path}" ]] && { log "SKIP Splatter:${sample} — exists"; continue; }
    tmp_dir="${out_dir}/_tmp/${sample}"
    mkdir -p "${tmp_dir}" "${out_dir}"
    log "RUN  Splatter:${sample}"
    sample_log="${LOG_DIR}/ext_Splatter_${sample}.log"
    if ${RSCRIPT_SPLATTER} scripts/external_simulators/splat_run.R \
        --cellinfo "${R_SPLIT}/${sample}/cellinfo.csv" \
        --geneinfo "${R_SPLIT}/${sample}/geneinfo.csv" \
        --matrix "${R_SPLIT}/${sample}/sparse_matrix.mtx" \
        --spatial_loc "${R_SPLIT}/${sample}/spatial.csv" \
        --output_dir "${tmp_dir}" > "${sample_log}" 2>&1 && \
       ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${tmp_dir}/splat_count.mtx" \
        --gene_path "${tmp_dir}/splat_genes.csv" \
        --location_path "${tmp_dir}/splat_loc.csv" \
        --save_adata_path "${output_path}" >> "${sample_log}" 2>&1; then
        log "OK   Splatter:${sample}"
    else
        log "FAIL Splatter:${sample} — see ${sample_log}"
    fi
done

# --- Splatter Simple ---
for sample in "${SAMPLES[@]}"; do
    out_dir="${OUTPUTS}/splatSimple"
    output_path="${out_dir}/${sample}.h5ad"
    [[ -f "${output_path}" ]] && { log "SKIP Splatter_Simple:${sample} — exists"; continue; }
    tmp_dir="${out_dir}/_tmp/${sample}"
    mkdir -p "${tmp_dir}" "${out_dir}"
    log "RUN  Splatter_Simple:${sample}"
    sample_log="${LOG_DIR}/ext_SplatterSimple_${sample}.log"
    if ${RSCRIPT_SPLATTER} scripts/external_simulators/splatSimple_run.R \
        --cellinfo "${R_SPLIT}/${sample}/cellinfo.csv" \
        --geneinfo "${R_SPLIT}/${sample}/geneinfo.csv" \
        --matrix "${R_SPLIT}/${sample}/sparse_matrix.mtx" \
        --spatial_loc "${R_SPLIT}/${sample}/spatial.csv" \
        --output_dir "${tmp_dir}" > "${sample_log}" 2>&1 && \
       ${PYTHON} scripts/external_simulators/reconstruct_external_h5ad.py \
        --count_path "${tmp_dir}/splatSimple_count.mtx" \
        --gene_path "${tmp_dir}/splatSimple_genes.csv" \
        --location_path "${tmp_dir}/splatSimple_loc.csv" \
        --save_adata_path "${output_path}" >> "${sample_log}" 2>&1; then
        log "OK   Splatter_Simple:${sample}"
    else
        log "FAIL Splatter_Simple:${sample} — see ${sample_log}"
    fi
done

# --- scCube ---
for sample in "${SAMPLES[@]}"; do
    out_dir="${OUTPUTS}/sccube"
    output_path="${out_dir}/${sample}_sccube_sim.h5ad"
    [[ -f "${output_path}" ]] && { log "SKIP scCube:${sample} — exists"; continue; }
    mkdir -p "${out_dir}"
    cell_key=$(${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('${EXPER_DATA}/${sample}.h5ad')
for c in ['ground_truth', 'cell_type', 'annotation', 'region', 'cluster']:
    if c in a.obs.columns:
        print(c)
        break
" 2>/dev/null)
    [[ -z "${cell_key}" ]] && cell_key="ground_truth"
    log "RUN  scCube:${sample}"
    sample_log="${LOG_DIR}/ext_scCube_${sample}.log"
    if ${PYTHON_SCCUBE} scripts/external_simulators/sccube_run.py \
        --h5ad_file "${EXPER_DATA}/${sample}.h5ad" \
        --output_dir "${output_path}" \
        --cell_key "${cell_key}" > "${sample_log}" 2>&1; then
        log "OK   scCube:${sample}"
    else
        log "FAIL scCube:${sample} — see ${sample_log}"
    fi
done

# ============================================================================
# Stage 5: Inventory
# ============================================================================
banner "STAGE 5: Build Inventory"
${PYTHON} scripts/build_simulator_inventory.py \
    --output-base "${OUTPUTS}" \
    --exper-data "${EXPER_DATA}" \
    --inventory "${BENCHMARKS}/simulator_inventory.csv"

# ============================================================================
# Stage 6: Quality Metrics
# ============================================================================
banner "STAGE 6: Build Quality Metrics"
${PYTHON} scripts/build_simulator_quality_metrics.py \
    --inventory "${BENCHMARKS}/simulator_inventory.csv" \
    --exper-data "${EXPER_DATA}" \
    --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
    --metadata "${BENCHMARKS}/simulator_quality_metric_metadata.json"

# ============================================================================
# Stage 7: Visualization
# ============================================================================
banner "STAGE 7: Visualization"
${PYTHON} scripts/plot_simulator_quality.py \
    --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
    --output-dir "${VIZ_DIR}"

# ============================================================================
# Summary
# ============================================================================
banner "FINAL SUMMARY"
log "Inventory:"
${PYTHON} -c "
import pandas as pd
df = pd.read_csv('${BENCHMARKS}/simulator_inventory.csv')
counts = df.groupby(['simulator','status']).size().unstack(fill_value=0)
print(counts.to_string())
"
log ""
log "Output counts:"
for d in FEAST_Rank FEAST_OT_Spatial srtsim_updated splat splatSimple sccube; do
    n=$(find "${OUTPUTS}/${d}" -name "*.h5ad" -type f | wc -l)
    log "  ${d}: ${n} .h5ad files"
done
log ""
log "ALL DONE"
