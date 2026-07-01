#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPER_DATA="${SIM_ROOT}/exper_data"
R_SPLIT_DIR="${FIG2_R_SPLIT_DIR:-${SIM_ROOT}/exper_data/R_split}"
OUTPUT_BASE="${FIG2_SIM_OUTPUT_DIR:-${SIM_ROOT}/outputs/simulation_saved}"

TOOLS="${FIG2_EXTERNAL_TOOLS:-srtsim splat splatSimple sccube}"
SAMPLES="${FIG2_EXTERNAL_SAMPLES:-}"
DRY_RUN="${FIG2_DRY_RUN:-0}"
SKIP_EXISTING="${FIG2_EXTERNAL_SKIP_EXISTING:-1}"

PYTHON_BIN="${FIG2_PYTHON:-/maiziezhou_lab2/yiru/envs/feast-py311-conda/bin/python}"
RSCRIPT_BIN="${FIG2_RSCRIPT:-Rscript}"

SRTSIM_ENV="${FIG2_SRTSIM_CONDA_ENV:-/maiziezhou_lab2/yiru/envs/srtsim-r}"
SPLATTER_ENV="${FIG2_SPLATTER_CONDA_ENV:-/maiziezhou_lab2/yiru/envs/splatter-r46}"
SCCUBE_ENV="${FIG2_SCCUBE_CONDA_ENV:-/maiziezhou_lab2/yiru/envs/sccube}"
SCCUBE_CELL_KEY="${FIG2_SCCUBE_CELL_KEY:-CellType}"
SCCUBE_SUFFIX="${FIG2_SCCUBE_OUTPUT_SUFFIX:-_sccube_sim}"

RECONSTRUCT_PY="${SCRIPT_DIR}/reconstruct_external_h5ad.py"

# Known exclusions from recovered benchmark
EXCLUDE_SRTSIM="${FIG2_EXTERNAL_EXCLUDE_SRTSIM:-Stereoseq_E9_5_E2S2}"
EXCLUDE_SPLAT="${FIG2_EXTERNAL_EXCLUDE_SPLAT:-Stereoseq_E9_5_E2S2}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

contains_word() {
    local needle="$1" haystack="$2" item
    for item in ${haystack}; do
        [[ "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

maybe_run() {
    echo "+ $*"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    "$@"
}

sample_dir() { printf "%s/%s" "${R_SPLIT_DIR}" "$1"; }

sample_h5ad() {
    local sample="$1"
    local path="${R_SPLIT_DIR}/${sample}/${sample}.h5ad"
    [[ -f "${path}" ]] && { printf "%s" "${path}"; return 0; }
    path="${EXPER_DATA}/${sample}.h5ad"
    [[ -f "${path}" ]] && { printf "%s" "${path}"; return 0; }
    return 1
}

discover_samples() {
    if [[ -n "${SAMPLES}" ]]; then
        printf "%s\n" ${SAMPLES}
        return 0
    fi
    if [[ -d "${R_SPLIT_DIR}" ]]; then
        find "${R_SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort
        return 0
    fi
    find "${EXPER_DATA}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" \
        | sed 's/[.]h5ad$//' | sort
}

require_r_split_files() {
    local sample="$1" dir file
    dir="$(sample_dir "${sample}")"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    for file in cellinfo.csv geneinfo.csv sparse_matrix.mtx spatial.csv; do
        [[ -r "${dir}/${file}" ]] || { echo "ERROR: missing ${dir}/${file}" >&2; return 1; }
    done
}

reconstruct_intermediate() {
    local prefix="$1" tmp_dir="$2" output_h5ad="$3"
    "${PYTHON_BIN}" "${RECONSTRUCT_PY}" \
        --count_path "${tmp_dir}/${prefix}_count.mtx" \
        --gene_path "${tmp_dir}/${prefix}_genes.csv" \
        --location_path "${tmp_dir}/${prefix}_loc.csv" \
        --save_adata_path "${output_h5ad}"
}

run_r_intermediate_tool() {
    local sample="$1" tool="$2" script="$3" prefix="$4" output_dir="$5" env_name="$6"

    require_r_split_files "${sample}"
    [[ "${DRY_RUN}" != "1" ]] && mkdir -p "${output_dir}"

    local output_h5ad="${output_dir}/${sample}.h5ad"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_h5ad}" ]]; then
        echo "[${tool}:${sample}] skip existing ${output_h5ad}"
        return 0
    fi

    local tmp_dir="${output_dir}/_tmp/${sample}"
    [[ "${DRY_RUN}" != "1" ]] && mkdir -p "${tmp_dir}"

    echo "[${tool}:${sample}] R simulation"
    maybe_run "${RSCRIPT_BIN}" "${script}" \
        --cellinfo "$(sample_dir "${sample}")/cellinfo.csv" \
        --geneinfo "$(sample_dir "${sample}")/geneinfo.csv" \
        --matrix "$(sample_dir "${sample}")/sparse_matrix.mtx" \
        --spatial_loc "$(sample_dir "${sample}")/spatial.csv" \
        --output_dir "${tmp_dir}"

    echo "[${tool}:${sample}] reconstruct ${output_h5ad}"
    reconstruct_intermediate "${prefix}" "${tmp_dir}" "${output_h5ad}"
}

run_srtsim() {
    local sample="$1"
    if contains_word "${sample}" "${EXCLUDE_SRTSIM}"; then
        echo "[srtsim:${sample}] skip excluded sample"
        return 0
    fi
    run_r_intermediate_tool \
        "${sample}" "srtsim" "${SCRIPT_DIR}/SRTsim_run.R" "SRTsim" \
        "${OUTPUT_BASE}/srtsim_updated" "${SRTSIM_ENV}"
}

run_splat() {
    local sample="$1"
    if contains_word "${sample}" "${EXCLUDE_SPLAT}"; then
        echo "[splat:${sample}] skip excluded sample"
        return 0
    fi
    run_r_intermediate_tool \
        "${sample}" "splat" "${SCRIPT_DIR}/splat_run.R" "splat" \
        "${OUTPUT_BASE}/splat" "${SPLATTER_ENV}"
}

run_splat_simple() {
    local sample="$1"
    run_r_intermediate_tool \
        "${sample}" "splatSimple" "${SCRIPT_DIR}/splatSimple_run.R" "splatSimple" \
        "${OUTPUT_BASE}/splatSimple" "${SPLATTER_ENV}"
}

run_sccube() {
    local sample="$1"
    local input_h5ad
    if ! input_h5ad="$(sample_h5ad "${sample}")"; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            input_h5ad="${R_SPLIT_DIR}/${sample}/${sample}.h5ad"
        else
            echo "ERROR: missing h5ad for scCube sample ${sample}" >&2
            return 1
        fi
    fi

    local output_dir="${OUTPUT_BASE}/sccube"
    [[ "${DRY_RUN}" != "1" ]] && mkdir -p "${output_dir}"
    local output_h5ad="${output_dir}/${sample}${SCCUBE_SUFFIX}.h5ad"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_h5ad}" ]]; then
        echo "[sccube:${sample}] skip existing ${output_h5ad}"
        return 0
    fi

    echo "[sccube:${sample}] simulate"
    maybe_run "${PYTHON_BIN}" "${SCRIPT_DIR}/sccube_run.py" \
        --h5ad_file "${input_h5ad}" \
        --output_dir "${output_h5ad}" \
        --cell_key "${SCCUBE_CELL_KEY}"
}

echo "=== Figure 2 External Simulators ==="
echo "Exper data: ${EXPER_DATA}"
echo "R split dir: ${R_SPLIT_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo "Tools: ${TOOLS}"
echo "Dry run: ${DRY_RUN}"

mapfile -t SAMPLE_LIST < <(discover_samples)
[[ "${#SAMPLE_LIST[@]}" -eq 0 ]] && { echo "ERROR: no samples found" >&2; exit 1; }

for sample in "${SAMPLE_LIST[@]}"; do
    for tool in ${TOOLS}; do
        case "${tool}" in
            srtsim) run_srtsim "${sample}" ;;
            splat) run_splat "${sample}" ;;
            splatSimple) run_splat_simple "${sample}" ;;
            sccube) run_sccube "${sample}" ;;
            *) echo "ERROR: unknown tool '${tool}'" >&2 ;;
        esac
    done
done

echo "=== External simulator run complete ==="
