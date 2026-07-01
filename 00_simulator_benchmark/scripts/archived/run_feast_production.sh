#!/bin/bash
# FEAST production benchmark runner.
# Runs FEAST with generative + rank decode + reference_rank quantile calibration
# on all simulator benchmark datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

FEAST_PYTHON="${FEAST_PYTHON:-python}"

# --- Configuration ---
# FEAST_OT:  generative mode, quantile decode, raw quantile calibration (default)
# FEAST_Rank: generative mode, rank decode, reference_rank quantile calibration
DECODE_METHOD="${DECODE_METHOD:-rank}"
QUANTILE_CALIBRATION="${QUANTILE_CALIBRATION:-reference_rank}"
SIMULATION_MODE="${SIMULATION_MODE:-generative}"

DATASETS=(
    sim_dlpfc
    sim_merfish_006
    sim_merfish_007
    sim_openst_005
    sim_openst_006
    sim_stereoseq
    sim_slideseq
    sim_xenium
)

echo "=== FEAST Production Benchmark ==="
echo "Mode:       ${SIMULATION_MODE}"
echo "Decode:     ${DECODE_METHOD}"
echo "Quantile:   ${QUANTILE_CALIBRATION}"
echo "Datasets:   ${#DATASETS[@]}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

for dataset in "${DATASETS[@]}"; do
    output_h5ad="${OUTPUT_DIR}/${dataset}_${DECODE_METHOD}.h5ad"
    echo "[$(date '+%H:%M:%S')] Running ${dataset} ..."
    ${FEAST_PYTHON} "${SCRIPT_DIR}/ParameterCloud_sim.py" \
        --input "${dataset}" \
        --output "${output_h5ad}" \
        --simulation-mode "${SIMULATION_MODE}" \
        --quantile-calibration "${QUANTILE_CALIBRATION}" \
        --decode-method "${DECODE_METHOD}"
    echo "[$(date '+%H:%M:%S')] → ${output_h5ad}"
done

echo ""
echo "=== Done. Outputs in ${OUTPUT_DIR} ==="
