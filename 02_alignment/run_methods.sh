#!/usr/bin/env bash
# ============================================================================
# Stage 2: Run alignment methods (Spateo + PASTE) on all rotation angles
# for one or all expression alteration variants.
#
# Usage:
#   FIG2_ALTERATION="mean_0.5" bash run_alignment_methods.sh   # single alteration
#   bash run_alignment_methods.sh                               # all alterations
#   FIG2_FORCE=1 bash run_alignment_methods.sh                  # rerun all
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# --- config ---
FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"

METHODS="${FIG2_ALIGNMENT_METHODS:-spateo paste}"
ANGLES="${FIG2_ALIGNMENT_ANGLES:-1 5 10 30 45}"
SKIP_EXISTING="${FIG2_ALIGNMENT_SKIP_EXISTING:-1}"
FORCE="${FIG2_FORCE:-0}"
PASTE_ALPHA="${PASTE_ALPHA:-0.1}"
PASTE_DISSIMILARITY="${PASTE_DISSIMILARITY:-euclidean}"
PASTE_N_PCS="${PASTE_N_PCS:-50}"
PASTE_NUM_ITER_MAX="${PASTE_NUM_ITER_MAX:-200}"

REFERENCE="outputs/04_alignment/reference.h5ad"
METHOD_DIR="outputs/04_alignment/methods"
LOG_DIR="logs"

# All known alterations (fallback if manifest can't be read)
ALL_ALTERATIONS="${FIG2_ALTERATIONS:-baseline mean_0.5 variance_2.0 sparsity_0.5}"

mkdir -p "${LOG_DIR}"

# --- helpers ---
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/methods.log"; }

run_one() {
    local alteration="$1"
    local method="$2"
    local angle="$3"
    local out_dir="${METHOD_DIR}/${alteration}/${method}/angle_${angle}"

    # skip if outputs exist
    if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" ]]; then
        if [[ -f "${out_dir}/aligned_coordinates.csv" ]]; then
            log "SKIP  ${alteration}/${method}/angle_${angle} (already done)"
            return 0
        fi
    fi

    local moving="outputs/04_alignment/${alteration}_rotated_${angle}.h5ad"
    if [[ ! -f "${moving}" ]]; then
        log "SKIP  ${alteration}/${method}/angle_${angle} (moving file missing: ${moving})"
        return 1
    fi

    mkdir -p "${out_dir}"

    local env_path script_path extra_args=""
    if [[ "${method}" == "spateo" ]]; then
        env_path="${FEAST_ENV}"
        script_path="scripts/methods/run_spateo.py"
    elif [[ "${method}" == "paste" ]]; then
        env_path="${FEAST_ENV}"
        script_path="scripts/methods/run_paste.py"
        extra_args="--alpha ${PASTE_ALPHA} --dissimilarity ${PASTE_DISSIMILARITY} --n-pcs ${PASTE_N_PCS} --num-iter-max ${PASTE_NUM_ITER_MAX}"
    else
        log "ERROR unknown method: ${method}"
        return 1
    fi

    if [[ ! -d "${env_path}" ]]; then
        log "ERROR ${method}: env not found: ${env_path}"
        return 1
    fi

    log "RUN   ${alteration}/${method}/angle_${angle}"

    set +e
    conda run -p "${env_path}" python "${script_path}" \
        --reference "${REFERENCE}" \
        --moving "${moving}" \
        --output-dir "${out_dir}" \
        ${extra_args} \
        &> "${LOG_DIR}/${alteration}_${method}_angle_${angle}.log"
    local rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
        log "OK    ${alteration}/${method}/angle_${angle}"
        return 0
    else
        log "FAIL  ${alteration}/${method}/angle_${angle} (rc=${rc})"
        return 1
    fi
}

# --- resolve alterations ---
if [[ -n "${FIG2_ALTERATION:-}" ]]; then
    # Single alteration (env var override)
    ALTERATIONS="${FIG2_ALTERATION}"
else
    ALTERATIONS="${ALL_ALTERATIONS}"
fi

# --- main ---
log "=== Alignment Methods ==="
log "Alterations: ${ALTERATIONS}"
log "Methods: ${METHODS}"
log "Angles: ${ANGLES}"
log "Reference: ${REFERENCE}"
log "FEAST_ENV: ${FEAST_ENV}"
log "PASTE_ALPHA: ${PASTE_ALPHA}"

if [[ ! -f "${REFERENCE}" ]]; then
    log "ERROR: reference not found: ${REFERENCE}"
    exit 1
fi

TOTAL=0
OK=0
FAIL=0
SKIP=0

for alteration in ${ALTERATIONS}; do
    for method in ${METHODS}; do
        for angle in ${ANGLES}; do
            TOTAL=$((TOTAL + 1))
            if run_one "${alteration}" "${method}" "${angle}"; then
                OK=$((OK + 1))
            else
                FAIL=$((FAIL + 1))
            fi
        done
    done
done

log "=== Summary: ${OK}/${TOTAL} ok, ${FAIL}/${TOTAL} failed ==="
