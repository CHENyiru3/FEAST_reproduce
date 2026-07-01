#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "${SCRIPT_DIR}")"

FEAST_ENV="${FIG2_FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON_BIN="${FIG2_PYTHON:-${FEAST_ENV}/bin/python}"
# FEAST_Rank = hungarian parameter mode with reference_rank spatial assignment.
METHODS="${FIG2_FEAST_QUANTILE_MODES:-reference_rank}"
SAMPLES="${FIG2_SIM_SAMPLES:-}"
DRY_RUN="${FIG2_DRY_RUN:-0}"
SKIP_EXISTING="${FIG2_SKIP_EXISTING:-1}"
FORCE="${FIG2_FORCE:-0}"

EXPER_DATA="${SIM_ROOT}/exper_data"
OUTPUT_BASE="${SIM_ROOT}/outputs/simulation_saved"
PARAM_CLOUD="${SCRIPT_DIR}/ParameterCloud_sim.py"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

discover_samples() {
    if [[ -n "${SAMPLES}" ]]; then
        echo "${SAMPLES}"
        return 0
    fi
    find "${EXPER_DATA}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" \
        | sed 's/[.]h5ad$//' | sort
}

echo "=== FEAST Simulator Benchmark ==="
echo "Quantile modes: ${METHODS}"
echo "Input dir: ${EXPER_DATA}"
echo "Output base: ${OUTPUT_BASE}"
echo "Dry run: ${DRY_RUN}"

mapfile -t SAMPLE_LIST < <(discover_samples)
echo "Samples: ${#SAMPLE_LIST[@]}"

for sample in "${SAMPLE_LIST[@]}"; do
    input_path="${EXPER_DATA}/${sample}.h5ad"
    if [[ ! -f "${input_path}" ]]; then
        echo "[SKIP] ${sample}: input not found at ${input_path}"
        continue
    fi

    label="FEAST_Rank"
    output_dir="${OUTPUT_BASE}/${label}"
    output_path="${output_dir}/${sample}.h5ad"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
        echo "[SKIP] ${label}:${sample} — output exists"
        continue
    fi

    cmd=(
        "${PYTHON_BIN}" "${PARAM_CLOUD}"
        --input "${input_path}"
        --output "${output_path}"
        --parameter-mode hungarian
        --spatial-mode reference_rank
        --assignment-method copula_rank
        --annotation-key auto
        --seed 2026
    )

    echo "[RUN] ${label}:${sample}"
    echo "  + ${cmd[*]}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    mkdir -p "${output_dir}"
    if "${cmd[@]}"; then
        echo "  OK ${label}:${sample}"
    else
        echo "  FAILED ${label}:${sample}"
    fi
done

echo "=== FEAST Simulator Benchmark complete ==="
