#!/usr/bin/env bash
# Re-run deconvolution benchmark with fixed FEAST (parameter_cloud v2)
# Simulation only — methods need separate envs and GPU.
set -euo pipefail

FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
SCRIPT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/03_deconvolution"
OUT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/outputs/05_deconvolution_rerun"
DATA_DIR="/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad"

SLICES=(007 050 100)
RESOLUTIONS=(0.1 0.25)
CT_KEY="cell_type"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =========================================================================
# Stage 1: Simulation (3 slices × 2 resolutions = 6 datasets)
# =========================================================================
log "=== STAGE 1: Simulation ==="
rm -rf "${OUT_DIR}"

for slice in "${SLICES[@]}"; do
    input="${DATA_DIR}/Zhuang-ABCA-1.${slice}.h5ad"
    for res in "${RESOLUTIONS[@]}"; do
        sim_out="${OUT_DIR}/simulations/${slice}/resolution_${res}.h5ad"
        truth_out="${OUT_DIR}/ground_truth/${slice}/resolution_${res}.csv"
        mkdir -p "$(dirname "${sim_out}")" "$(dirname "${truth_out}")"

        log "SIM  slice=${slice} res=${res}"
        ${FEAST_ENV}/bin/python -u "${SCRIPT_DIR}/run_deconvolution.py" \
            --input "${input}" \
            --output-dir "${OUT_DIR}/simulations/${slice}" \
            --ground-truth-dir "${OUT_DIR}/ground_truth/${slice}" \
            --slices "${slice}" \
            --resolutions "${res}" \
            --cell-type-key "${CT_KEY}" \
            --seed 2026
    done
done

log "Simulation done."
find "${OUT_DIR}/simulations" -name "*.h5ad" | sort | while read f; do
    echo "  $(basename $(dirname $(dirname $f)))/$(basename $(dirname $f))/$(basename $f)"
done

# =========================================================================
# Stage 2: Methods (Cell2location + RCTD) - SKIP for now, re-use existing?
# =========================================================================
log "=== STAGE 2: Methods ==="
log "NOTE: Methods not run automatically. Existing predictions are from old simulation."
log "Re-run manually:"
log "  Cell2location: bash ${SCRIPT_DIR}/run_methods.sh"
log "  RCTD: python ${SCRIPT_DIR}/methods/run_rctd.py"

log "=== PIPELINE DONE (simulation only) ==="
log "Output: ${OUT_DIR}"
