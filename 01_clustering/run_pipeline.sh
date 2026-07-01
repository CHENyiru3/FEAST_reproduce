#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# Clustering method pipeline runner — runs all methods on all HVG inputs.
#
# Usage:
#   bash scripts/methods/run_clust_pipeline.sh
#   FIG2_CLUSTER_METHODS="STAGATE_mclust GraphST" bash scripts/methods/run_clust_pipeline.sh
#   FIG2_FORCE=1 bash scripts/methods/run_clust_pipeline.sh
#
# Controls:
#   FIG2_CLUSTER_METHODS      space-separated method names (default: all three)
#   FIG2_CLUSTER_SKIP_EXISTING skip if clusters.csv exists (default: 1)
#   FIG2_FORCE                rerun all, ignore skip (default: 0)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CLUSTER_ROOT}"

FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
STAGATE_ENV="/maiziezhou_lab2/yiru/envs/STAGATE"
GRAPHST_ENV="/maiziezhou_lab2/yiru/envs/GraphST"

PYTHON_FEAST="${FEAST_ENV}/bin/python"
PYTHON_STAGATE="${STAGATE_ENV}/bin/python"
PYTHON_GRAPHST="${GRAPHST_ENV}/bin/python"

METHODS="${FIG2_CLUSTER_METHODS:-STAGATE_mclust GraphST Leiden}"
SKIP_EXISTING="${FIG2_CLUSTER_SKIP_EXISTING:-1}"
FORCE="${FIG2_FORCE:-0}"

CKPT_DIR="outputs/.checkpoints"
LOG_DIR="logs"
HVG_MANIFEST="outputs/03_clustering/hvg_inputs/hvg_manifest.csv"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() { echo ""; echo "==== $* ===="; }
fail_record() { echo "${1}:${2}:${3}" >> "${CKPT_DIR}/failures_${4}.txt"; }
fail_clear() { rm -f "${CKPT_DIR}/failures_$1.txt"; }
fail_list() { [[ -f "${CKPT_DIR}/failures_$1.txt" ]] && cat "${CKPT_DIR}/failures_$1.txt" || true; }

# ---------------------------------------------------------------------------
run_stagate() {
    local slice="$1" sim_id="$2" input_path="$3"
    local out_dir="outputs/03_clustering/methods/STAGATE_mclust/${slice}/${sim_id}"
    local clusters="${out_dir}/clusters.csv"

    if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" && -f "${clusters}" ]]; then
        log "  [SKIP] STAGATE:${slice}/${sim_id} — exists"
        return 0
    fi

    local log_file="${LOG_DIR}/stagate_${slice}_${sim_id}.log"
    log "  [RUN]  STAGATE:${slice}/${sim_id}"

    if "${PYTHON_STAGATE}" methods/stagate_mclust.py \
        --input "${input_path}" \
        --output-dir "${out_dir}" \
        --rad-cutoff 150 \
        --n-epochs 500 \
        --n-clusters auto > "${log_file}" 2>&1; then
        log "  [OK]   STAGATE:${slice}/${sim_id}"
    else
        log "  [FAIL] STAGATE:${slice}/${sim_id} — see ${log_file}"
        fail_record "${slice}" "${sim_id}" "STAGATE_mclust" "methods"
    fi
}

run_graphst() {
    local slice="$1" sim_id="$2" input_path="$3"
    local out_dir="outputs/03_clustering/methods/GraphST/${slice}/${sim_id}"
    local clusters="${out_dir}/clusters.csv"

    if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" && -f "${clusters}" ]]; then
        log "  [SKIP] GraphST:${slice}/${sim_id} — exists"
        return 0
    fi

    local log_file="${LOG_DIR}/graphst_${slice}_${sim_id}.log"
    log "  [RUN]  GraphST:${slice}/${sim_id}"

    if "${PYTHON_GRAPHST}" methods/graphst.py \
        --input "${input_path}" \
        --output-dir "${out_dir}" \
        --n-clusters auto \
        --device cpu > "${log_file}" 2>&1; then
        log "  [OK]   GraphST:${slice}/${sim_id}"
    else
        log "  [FAIL] GraphST:${slice}/${sim_id} — see ${log_file}"
        fail_record "${slice}" "${sim_id}" "GraphST" "methods"
    fi
}

run_leiden() {
    local slice="$1" sim_id="$2" input_path="$3"
    local out_dir="outputs/03_clustering/methods/Leiden/${slice}/${sim_id}"
    local clusters="${out_dir}/clusters.csv"

    if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" && -f "${clusters}" ]]; then
        log "  [SKIP] Leiden:${slice}/${sim_id} — exists"
        return 0
    fi

    local log_file="${LOG_DIR}/leiden_${slice}_${sim_id}.log"
    log "  [RUN]  Leiden:${slice}/${sim_id}"

    if "${PYTHON_FEAST}" methods/leiden.py \
        --input "${input_path}" \
        --output-dir "${out_dir}" > "${log_file}" 2>&1; then
        log "  [OK]   Leiden:${slice}/${sim_id}"
    else
        log "  [FAIL] Leiden:${slice}/${sim_id} — see ${log_file}"
        fail_record "${slice}" "${sim_id}" "Leiden" "methods"
    fi
}

# ---------------------------------------------------------------------------
run_method() {
    local method="$1" slice="$2" sim_id="$3" input_path="$4"
    case "${method}" in
        STAGATE_mclust) run_stagate "${slice}" "${sim_id}" "${input_path}" ;;
        GraphST)         run_graphst  "${slice}" "${sim_id}" "${input_path}" ;;
        Leiden)          run_leiden   "${slice}" "${sim_id}" "${input_path}" ;;
        *) log "  [SKIP] Unknown method: ${method}" ;;
    esac
}

# ---------------------------------------------------------------------------
main() {
    log_section "CLUSTERING METHOD PIPELINE"
    log "Methods: ${METHODS}"
    log "Skip existing: ${SKIP_EXISTING}"
    log "Force: ${FORCE}"
    log ""

    if [[ ! -f "${HVG_MANIFEST}" ]]; then
        log "ERROR: HVG manifest not found at ${HVG_MANIFEST}"
        log "Run HVG refresh first: make subtask_03_hvg"
        exit 1
    fi

    fail_clear "methods"

    local total_jobs=0
    local done_jobs=0
    local skipped_jobs=0
    local failed_jobs=0

    # Count total
    while IFS=, read -r slice_id sim_id hvg_file n_hvgs status; do
        [[ "${status}" == "ok" ]] || continue
        for m in ${METHODS}; do
            total_jobs=$((total_jobs + 1))
        done
    done < <(tail -n +2 "${HVG_MANIFEST}")

    log "Total jobs: ${total_jobs}"

    while IFS=, read -r slice_id sim_id hvg_file n_hvgs status; do
        [[ "${status}" == "ok" ]] || continue

        for method in ${METHODS}; do
            done_jobs=$((done_jobs + 1))
            run_method "${method}" "${slice_id}" "${sim_id}" "${hvg_file}"
        done
    done < <(tail -n +2 "${HVG_MANIFEST}")

    # --- summary ---
    log ""
    log_section "METHOD PIPELINE SUMMARY"

    local fail_count=$(fail_list "methods" | wc -l)
    log "  Total jobs: ${done_jobs}"
    log "  Failures:   ${fail_count}"

    if [[ "${fail_count}" -gt 0 ]]; then
        log ""
        log "Failures:"
        fail_list "methods" | while read line; do log "  ${line}"; done
        exit 1
    fi

    log "Done."
}

main
