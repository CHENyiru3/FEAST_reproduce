#!/usr/bin/env bash
# Re-run cell2location + RCTD on 05_deconvolution_rerun simulations
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CELL2LOC_ENV="/maiziezhou_lab2/yiru/envs/cell2loc_env"
RCTD_ENV="/maiziezhou_lab2/yiru/envs/rctd_bioc"
FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"

SIM_DIR="${PROJECT_ROOT}/outputs/05_deconvolution_rerun"
REF_DIR="/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad"
C2L_DIR="${SIM_DIR}/cell2location"
RCTD_DIR="${SIM_DIR}/rctd"
BENCH_DIR="${SIM_DIR}/benchmarks"

SLICES=(007 050 100)
RESOLUTIONS=(0.1 0.25)
CELL_TYPE_KEY="cell_type"
SEED=2026

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =========================================================================
# Stage 1: Cell2location
# =========================================================================
log "=== STAGE 1: Cell2location ==="
for slice in "${SLICES[@]}"; do
    ref="${REF_DIR}/Zhuang-ABCA-1.${slice}.h5ad"
    for res in "${RESOLUTIONS[@]}"; do
        sim="${SIM_DIR}/simulations/${slice}/resolution_${res}.h5ad"
        out="${C2L_DIR}/${slice}/resolution_${res}_proportions.csv"
        if [[ -f "${out}" ]]; then
            log "SKIP cell2location ${slice}/res=${res}"
            continue
        fi
        mkdir -p "$(dirname "${out}")"
        log "RUN  cell2location ${slice}/res=${res}"
        "${CELL2LOC_ENV}/bin/python" "${SCRIPT_DIR}/methods/run_cell2loc.py" \
            --input "${sim}" \
            --reference "${ref}" \
            --cell-type-key "${CELL_TYPE_KEY}" \
            --output "${out}" \
            --seed "${SEED}" \
            &> "$(dirname "${out}")/run.log"
        log "DONE cell2location ${slice}/res=${res}"
    done
done

# =========================================================================
# Stage 2: RCTD
# =========================================================================
log "=== STAGE 2: RCTD ==="
for slice in "${SLICES[@]}"; do
    ref="${REF_DIR}/Zhuang-ABCA-1.${slice}.h5ad"
    for res in "${RESOLUTIONS[@]}"; do
        sim="${SIM_DIR}/simulations/${slice}/resolution_${res}.h5ad"
        out="${RCTD_DIR}/${slice}/resolution_${res}_proportions.csv"
        if [[ -f "${out}" ]]; then
            log "SKIP RCTD ${slice}/res=${res}"
            continue
        fi
        mkdir -p "$(dirname "${out}")"
        log "RUN  RCTD ${slice}/res=${res}"
        "${FEAST_ENV}/bin/python" "${SCRIPT_DIR}/methods/run_rctd.py" \
            --input "${sim}" \
            --reference "${ref}" \
            --cell-type-key "${CELL_TYPE_KEY}" \
            --rscript "${SCRIPT_DIR}/methods/run_rctd_backend.R" \
            --output "${out}" \
            --cache-dir /tmp/rctd_bioc_cache \
            --force \
            &> "$(dirname "${out}")/run.log"
        log "DONE RCTD ${slice}/res=${res}"
    done
done

# =========================================================================
# Stage 3: Benchmark
# =========================================================================
log "=== STAGE 3: Benchmark ==="
"${FEAST_ENV}/bin/python" "${SCRIPT_DIR}/benchmark.py" \
    --truth-dir "${SIM_DIR}/ground_truth" \
    --prediction-dirs "${C2L_DIR}" "${RCTD_DIR}" \
    --slices "${SLICES[@]}" \
    --resolutions "${RESOLUTIONS[@]}" \
    --output "${BENCH_DIR}/deconvolution_benchmark_results.csv" \
    --overwrite

log "Done. Output: ${BENCH_DIR}/deconvolution_benchmark_results.csv"
