#!/usr/bin/env bash
# Re-run deconvolution simulation with fixed FEAST (parameter_cloud v2)
set -euo pipefail

FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
SCRIPT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/03_deconvolution"
OUT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/outputs/05_deconvolution_rerun"
DATA_DIR="/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/data"

# Symlink data files so run_deconvolution.py can find {input_dir}/Zhuang-ABCA-1.{slice}.h5ad
for slice in 007 050 100; do
    ln -sf "${DATA_DIR}/Zhuang-ABCA-1.${slice}.h5ad" "${OUT_DIR}/data/Zhuang-ABCA-1.${slice}.h5ad"
done

log "=== Deconvolution Simulation Rerun ==="
log "Slices: 007, 050, 100   Resolutions: 0.1, 0.25"

${FEAST_ENV}/bin/python -u "${SCRIPT_DIR}/run_deconvolution.py" \
    --input-dir "${OUT_DIR}/data" \
    --output-dir "${OUT_DIR}/simulations" \
    --slices "007,050,100" \
    --resolutions 0.1 0.25 \
    --cell-type-key "cell_type" \
    --seed 2026

log ""
log "=== Simulation complete ==="
find "${OUT_DIR}/simulations" -name "*.h5ad" -exec ls -lh {} \;
echo ""
log "Ground truth: ${OUT_DIR}/ground_truth/"
log "Next: re-run methods (cell2location + RCTD), then benchmark + visualize"
