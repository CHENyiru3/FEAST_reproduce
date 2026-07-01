#!/usr/bin/env bash
# ============================================================================
# Quick test: Run Spateo + Spacel on the FIXED simulation (sequencing mode).
# Baseline alteration only, 6 angles.
# ============================================================================
set -euo pipefail

FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
SPACEL_ENV="${SPACEL_ENV:-/maiziezhou_lab2/yiru/envs/SPACEL}"
SIM_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/outputs/04_alignment_fixed"
METHOD_DIR="${SIM_DIR}/methods"
ANGLES="1 5 10 30 45 60"

mkdir -p "${METHOD_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for angle in ${ANGLES}; do
    # --- Spateo ---
    OUT="${METHOD_DIR}/baseline/spateo/angle_${angle}"
    if [[ -f "${OUT}/aligned_coordinates.csv" ]]; then
        log "SKIP spateo angle_${angle} (exists)"
    else
        mkdir -p "${OUT}"
        log "RUN  spateo angle_${angle}"
        conda run -p "${FEAST_ENV}" python scripts/methods/run_spateo.py \
            --reference "${SIM_DIR}/reference.h5ad" \
            --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
            --output-dir "${OUT}" \
            &> "${OUT}/run.log" &
    fi

    # --- Spacel ---
    OUT="${METHOD_DIR}/baseline/spacel/angle_${angle}"
    if [[ -f "${OUT}/aligned_coordinates.csv" ]]; then
        log "SKIP spacel angle_${angle} (exists)"
    else
        mkdir -p "${OUT}"
        log "RUN  spacel angle_${angle}"
        conda run -p "${SPACEL_ENV}" python scripts/methods/run_spacel.py \
            --reference "${SIM_DIR}/reference.h5ad" \
            --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
            --output-dir "${OUT}" \
            --layer-key sce.layer_guess \
            &> "${OUT}/run.log" &
    fi
done

log "Waiting for all jobs to complete..."
wait
log "All done."
