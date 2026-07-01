#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "${SCRIPT_DIR}")"

FEAST_ENV="${FIG2_FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON_BIN="${FIG2_PYTHON:-${FEAST_ENV}/bin/python}"
DRY_RUN="${FIG2_DRY_RUN:-0}"

INVENTORY="${SIM_ROOT}/outputs/benchmarks/simulator_inventory.csv"
METRICS="${SIM_ROOT}/outputs/benchmarks/simulator_quality_metrics.csv"
VIZ_DIR="${SIM_ROOT}/outputs/quality_visualization"
EXPER_DATA="${SIM_ROOT}/exper_data"
SIM_SAVED="${SIM_ROOT}/outputs/simulation_saved"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

echo "=== Simulator Quality Visualization ==="

# Inventory
echo "[1/3] Building inventory..."
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  + ${PYTHON_BIN} ${SCRIPT_DIR}/build_simulator_inventory.py ..."
else
    mkdir -p "$(dirname "${INVENTORY}")"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_simulator_inventory.py" \
        --output-base "${SIM_SAVED}" \
        --exper-data "${EXPER_DATA}" \
        --inventory "${INVENTORY}"
fi

# Metrics
echo "[2/3] Building quality metrics..."
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  + ${PYTHON_BIN} ${SCRIPT_DIR}/build_simulator_quality_metrics.py ..."
else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_simulator_quality_metrics.py" \
        --inventory "${INVENTORY}" \
        --exper-data "${EXPER_DATA}" \
        --metrics "${METRICS}"
fi

# Plots
echo "[3/3] Generating plots..."
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  + ${PYTHON_BIN} ${SCRIPT_DIR}/plot_simulator_quality.py ..."
else
    mkdir -p "${VIZ_DIR}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_simulator_quality.py" \
        --metrics "${METRICS}" \
        --output-dir "${VIZ_DIR}"
fi

echo "=== Visualization complete ==="
