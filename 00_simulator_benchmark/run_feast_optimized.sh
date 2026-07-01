#!/usr/bin/env bash
set -euo pipefail

# FEAST-only optimized rerun.
# Existing output .h5ad files are skipped, so this can be resumed safely.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

FEAST_ENV=/maiziezhou_lab2/yiru/envs/feast-py311-conda
PYTHON=${FEAST_ENV}/bin/python
SEED=2026

EXPER_DATA=exper_data
OUTPUTS=outputs/simulation_saved
BENCHMARKS=outputs/benchmarks
VIZ_DIR=outputs/quality_visualization
LOG_DIR=logs

RUN_ID="feast_optimized_$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${LOG_DIR}" "${BENCHMARKS}" "${VIZ_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

log "FEAST optimized rerun"
log "Run ID: ${RUN_ID}"
log "Seed: ${SEED}"
log "Python: ${PYTHON}"

mapfile -t SAMPLES < <(find "${EXPER_DATA}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" | sort | sed 's/[.]h5ad$//')

for sample in "${SAMPLES[@]}"; do
    for mode_label in "FEAST_Rank:reference_rank" "FEAST_OT_Spatial:ot_spatial"; do
        label="${mode_label%%:*}"
        spatial="${mode_label##*:}"
        out_dir="${OUTPUTS}/${label}"
        output_path="${out_dir}/${sample}.h5ad"
        sample_log="${LOG_DIR}/feast_${label}_${sample}.log"
        mkdir -p "${out_dir}"

        if [[ -f "${output_path}" ]]; then
            log "SKIP ${label}:${sample} - exists"
            continue
        fi

        log "RUN  ${label}:${sample}"
        if "${PYTHON}" scripts/ParameterCloud_sim.py \
            --input "${EXPER_DATA}/${sample}.h5ad" \
            --output "${output_path}" \
            --parameter-mode hungarian \
            --spatial-mode "${spatial}" \
            --assignment-method hybrid \
            --annotation-key auto \
            --seed "${SEED}" \
            --ppf-method interp \
            --beta-n-jobs 4 \
            --beta-early-stopping-patience 2 \
            --assignment-solver scipy \
            --assignment-blocks \
            --assignment-block-multiplier 8 \
            --convert-n-jobs 4 > "${sample_log}" 2>&1; then
            log "OK   ${label}:${sample}"
        else
            log "FAIL ${label}:${sample} - see ${sample_log}"
        fi
    done
done

log "Build inventory"
"${PYTHON}" scripts/build_simulator_inventory.py \
    --output-base "${OUTPUTS}" \
    --exper-data "${EXPER_DATA}" \
    --inventory "${BENCHMARKS}/simulator_inventory.csv" 2>&1 | tee -a "${RUN_LOG}"

log "Build quality metrics"
"${PYTHON}" scripts/build_simulator_quality_metrics.py \
    --inventory "${BENCHMARKS}/simulator_inventory.csv" \
    --exper-data "${EXPER_DATA}" \
    --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
    --metadata "${BENCHMARKS}/simulator_quality_metric_metadata.json" 2>&1 | tee -a "${RUN_LOG}"

log "Build visualization"
"${PYTHON}" scripts/plot_simulator_quality.py \
    --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
    --output-dir "${VIZ_DIR}" 2>&1 | tee -a "${RUN_LOG}"

log "Output counts"
for d in FEAST_Rank FEAST_OT_Spatial srtsim_updated splat splatSimple sccube; do
    n=0
    [[ -d "${OUTPUTS}/${d}" ]] && n=$(find "${OUTPUTS}/${d}" -name "*.h5ad" -type f | wc -l)
    log "  ${d}: ${n} .h5ad files"
done

log "ALL DONE"
