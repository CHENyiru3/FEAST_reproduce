#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "${SCRIPT_DIR}")"

FEAST_ENV="${FIG2_FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON_BIN="${FIG2_PYTHON:-${FEAST_ENV}/bin/python}"
DATASETS_ROOT="${FIG2_DATASETS_ROOT:-/maiziezhou_lab2/yiru/Datasets}"

DRY_RUN="${FIG2_DRY_RUN:-0}"
FORCE="${FIG2_FORCE:-0}"
FAIL_FAST="${FIG2_FAIL_FAST:-0}"

EXPER_DATA="${SIM_ROOT}/exper_data"
R_SPLIT="${SIM_ROOT}/exper_data/R_split"
MANIFEST_DIR="${SIM_ROOT}/outputs/benchmarks"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

echo "============================================"
echo " Figure 2 Simulator Benchmark — Full Pipeline"
echo "============================================"
echo "SIM_ROOT: ${SIM_ROOT}"
echo "Dry run: ${DRY_RUN}"
echo "Force: ${FORCE}"
echo ""

run_stage() {
    local name="$1"; shift
    echo ""
    echo "==== Stage: ${name} ===="
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: $*"
        return 0
    fi
    if "$@"; then
        echo "==== ${name}: OK ===="
    else
        echo "==== ${name}: FAILED ===="
        if [[ "${FAIL_FAST}" == "1" ]]; then
            exit 1
        fi
    fi
}

# Stage 1: Prepare inputs
run_stage "prepare_simulator_inputs" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_simulator_inputs.py" \
        --datasets-root "${DATASETS_ROOT}" \
        --output-dir "${EXPER_DATA}" \
        --manifest "${MANIFEST_DIR}/input_manifest.csv" \
        --seed 2026

# Stage 2: Build R_split inputs
run_stage "build_r_split_inputs" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/external_simulators/build_r_split_inputs.py" \
        --input-dir "${EXPER_DATA}" \
        --output-dir "${R_SPLIT}" \
        --manifest "${MANIFEST_DIR}/r_split_manifest.csv"

# Stage 3: FEAST benchmark
run_stage "feast_simulator_benchmark" \
    bash "${SCRIPT_DIR}/run_feast_simulator_benchmark.sh"

# Stage 4: External simulators
run_stage "external_simulators" \
    bash "${SCRIPT_DIR}/external_simulators/run_external_simulators.sh"

# Stage 5-7: Quality metrics and visualization
run_stage "quality_visualization" \
    bash "${SCRIPT_DIR}/run_simulator_quality_visualization.sh"

echo ""
echo "============================================"
echo " Full pipeline complete"
echo "============================================"
echo ""
echo "Output locations:"
echo "  Inputs:      ${EXPER_DATA}"
echo "  R_split:     ${R_SPLIT}"
echo "  Simulations: ${SIM_ROOT}/outputs/simulation_saved/"
echo "  Benchmarks:  ${MANIFEST_DIR}/"
echo "  Plots:       ${SIM_ROOT}/outputs/quality_visualization/"
echo ""
