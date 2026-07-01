#!/usr/bin/env bash
# ============================================================================
# Alignment Benchmark — Full Production Runner
#
# Break-resume pipeline across 7 stages with checkpoint files.
#
# Usage:
#   bash run_production.sh              # run all, skip completed
#   bash run_production.sh              # resume after interrupt (same command)
#   FIG2_FORCE=1 bash run_production.sh # force rerun all
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- config ---
FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
DATASETS="${DATASETS:-/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad}"

METHODS="${FIG2_ALIGNMENT_METHODS:-spateo paste}"
ANGLES="${FIG2_ALIGNMENT_ANGLES:-1 5 10 30 45}"
ALTERATIONS="${FIG2_ALTERATIONS:-baseline mean_0.5 variance_2.0 sparsity_0.5}"
FORCE="${FIG2_FORCE:-0}"
SKIP_SIM="${FIG2_ALIGNMENT_SKIP_SIM:-1}"
SKIP_METHODS="${FIG2_ALIGNMENT_SKIP_EXISTING:-1}"
SEED="${FIG2_SEED:-2026}"
FEAST_PPF_METHOD="${FEAST_PPF_METHOD:-interp}"
FEAST_BETA_N_JOBS="${FEAST_BETA_N_JOBS:-4}"
FEAST_BETA_EARLY_STOPPING_PATIENCE="${FEAST_BETA_EARLY_STOPPING_PATIENCE:-2}"
FEAST_ASSIGNMENT_SOLVER="${FEAST_ASSIGNMENT_SOLVER:-scipy}"
FEAST_ASSIGNMENT_BLOCKS="${FEAST_ASSIGNMENT_BLOCKS:-1}"
FEAST_ASSIGNMENT_BLOCK_SIZE="${FEAST_ASSIGNMENT_BLOCK_SIZE:-}"
FEAST_ASSIGNMENT_BLOCK_MULTIPLIER="${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER:-8}"
FEAST_CONVERT_N_JOBS="${FEAST_CONVERT_N_JOBS:-4}"
PASTE_ALPHA="${PASTE_ALPHA:-0.1}"
PASTE_DISSIMILARITY="${PASTE_DISSIMILARITY:-euclidean}"
PASTE_N_PCS="${PASTE_N_PCS:-50}"
PASTE_NUM_ITER_MAX="${PASTE_NUM_ITER_MAX:-200}"

SIM_DIR="outputs/simulations"
METHOD_DIR="outputs/methods"
BENCH_DIR="outputs/benchmarks"
PLOT_DIR="outputs/plots"
CKPT_DIR="outputs/.checkpoints"
LOG_DIR="logs"
RESULT_DIR="results"
CONFIG_DIR="../configs"
METHOD_SCRIPT_DIR="../scripts/methods"

CURRENT_STAGE=""

# --- helpers ---
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/production.log"; }
ckpt_done() { mkdir -p "${CKPT_DIR}"; touch "${CKPT_DIR}/${1}"; }
ckpt_skip() {
    if [[ "${FORCE}" == "1" ]]; then return 1; fi
    [[ -f "${CKPT_DIR}/${1}" ]]
}
fail_record() {
    local f="${CKPT_DIR}/failures_${1}.txt"
    mkdir -p "$(dirname "$f")"
    echo "${2}" >> "$f"
}
cleanup() {
    log "INTERRUPTED at stage ${CURRENT_STAGE:-unknown}"
    exit 1
}
trap cleanup INT TERM

# ===================================================================
# Stage 1: Link reference data
# ===================================================================
stage_link_data() {
    log "=== Stage 1: Link reference data ==="
    mkdir -p data

    local src="${DATASETS}/151675.h5ad"
    local dst="data/151675.h5ad"

    if [[ ! -f "${src}" ]]; then
        log "FATAL: reference not found: ${src}"
        exit 1
    fi

    if [[ -L "${dst}" ]] || [[ -f "${dst}" ]]; then
        log "  data/151675.h5ad already exists"
    else
        ln -s "${src}" "${dst}"
        log "  linked ${src} -> ${dst}"
    fi

    # Validate required fields
    log "  validating AnnData fields ..."
    ${FEAST_ENV}/bin/python -c "
import scanpy as sc
a = sc.read_h5ad('${dst}')
assert a.X is not None, 'X missing'
assert 'spatial' in a.obsm, 'obsm.spatial missing'
print(f'  OK: {a.shape[0]} spots, {a.shape[1]} genes')
"
    ckpt_done "01_link_data.done"
    log "  done."
}

# ===================================================================
# Stage 2: Install alignment dependencies into FEAST environment
# ===================================================================
stage_install_alignment_deps() {
    log "=== Stage 2: Install alignment dependencies ==="

    if ${FEAST_ENV}/bin/python -c "import spateo, paste" 2>/dev/null; then
        log "  spateo already installed"
        log "  paste already installed"
        ckpt_done "02_install_alignment_deps.done"
        return
    fi

    if ! ${FEAST_ENV}/bin/python -c "import spateo" 2>/dev/null; then
        log "  pip install spateo-release ..."
        ${FEAST_ENV}/bin/pip install spateo-release
    fi

    if ! ${FEAST_ENV}/bin/python -c "import paste" 2>/dev/null; then
        log "  pip install paste-bio ..."
        ${FEAST_ENV}/bin/pip install paste-bio
    fi

    if ${FEAST_ENV}/bin/python -c "import spateo, paste; print('spateo', spateo.__version__); print('paste ok')"; then
        log "  alignment dependencies installed successfully"
        ckpt_done "02_install_alignment_deps.done"
    else
        log "  FATAL: alignment dependency import failed after install"
        exit 1
    fi
}

# ===================================================================
# Stage 3: Run alignment simulations
# ===================================================================
stage_simulations() {
    log "=== Stage 3: FEAST alignment simulations (v2) ==="

    # Skip if reference and all rotated files exist
    local all_done=true
    if [[ ! -f "${SIM_DIR}/reference.h5ad" ]]; then
        all_done=false
    fi
    for alt in ${ALTERATIONS}; do
        for angle in ${ANGLES}; do
            if [[ ! -f "${SIM_DIR}/${alt}_rotated_${angle}.h5ad" ]]; then
                all_done=false
                break
            fi
        done
    done

    if [[ "${all_done}" == "true" && "${SKIP_SIM}" == "1" && "${FORCE}" != "1" ]]; then
        log "  all simulation outputs exist, skipping"
        ckpt_done "03_simulations.done"
        return
    fi

    log "  Alterations: ${ALTERATIONS}"
    log "  Angles: ${ANGLES}"
    log "  FEAST knobs: ppf=${FEAST_PPF_METHOD}, beta_n_jobs=${FEAST_BETA_N_JOBS}, beta_patience=${FEAST_BETA_EARLY_STOPPING_PATIENCE}, blocks=${FEAST_ASSIGNMENT_BLOCKS}, block_multiplier=${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER}, convert_n_jobs=${FEAST_CONVERT_N_JOBS}"

    local overwrite_args=()
    [[ "${FORCE}" == "1" ]] && overwrite_args+=(--overwrite)

    local assignment_block_args=()
    if [[ "${FEAST_ASSIGNMENT_BLOCKS}" == "1" ]]; then
        assignment_block_args+=(--assignment-blocks)
    else
        assignment_block_args+=(--no-assignment-blocks)
    fi

    local assignment_block_size_args=()
    if [[ -n "${FEAST_ASSIGNMENT_BLOCK_SIZE}" ]]; then
        assignment_block_size_args+=(--assignment-block-size "${FEAST_ASSIGNMENT_BLOCK_SIZE}")
    fi

    ${FEAST_ENV}/bin/python run_simulation.py \
        --input data/151675.h5ad \
        --output-dir "${SIM_DIR}" \
        --angles ${ANGLES} \
        --seed "${SEED}" \
        --config "${CONFIG_DIR}/alignment.yaml" \
        --ppf-method "${FEAST_PPF_METHOD}" \
        --beta-n-jobs "${FEAST_BETA_N_JOBS}" \
        --beta-early-stopping-patience "${FEAST_BETA_EARLY_STOPPING_PATIENCE}" \
        --assignment-solver "${FEAST_ASSIGNMENT_SOLVER}" \
        "${assignment_block_args[@]}" \
        "${assignment_block_size_args[@]}" \
        --assignment-block-multiplier "${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER}" \
        --convert-n-jobs "${FEAST_CONVERT_N_JOBS}" \
        "${overwrite_args[@]}"

    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        cp "${SIM_DIR}/simulation_manifest.csv" "${RESULT_DIR}/simulation_manifest.csv"
        ckpt_done "03_simulations.done"
        log "  done."
    else
        fail_record "simulations" "run_alignment_simulation.py exited with rc=${rc}"
        log "  FAILED (rc=${rc})"
    fi
}

# ===================================================================
# Stage 4: Spateo alignment methods
# ===================================================================
stage_spateo_methods() {
    log "=== Stage 4: Spateo alignment methods ==="

    if ! echo "${METHODS}" | grep -q "spateo"; then
        log "  spateo not in METHODS, skipping"
        ckpt_done "04_spateo_methods.done"
        return
    fi

    local all_ok=true
    for alt in ${ALTERATIONS}; do
        for angle in ${ANGLES}; do
            local out_dir="${METHOD_DIR}/${alt}/spateo/angle_${angle}"
            if [[ "${SKIP_METHODS}" == "1" && "${FORCE}" != "1" && -f "${out_dir}/aligned_coordinates.csv" ]]; then
                log "  ${alt}/spateo/angle_${angle} already done, skip"
                continue
            fi

            mkdir -p "${out_dir}"
            log "  running ${alt}/spateo/angle_${angle} ..."

            set +e
            ${FEAST_ENV}/bin/python "${METHOD_SCRIPT_DIR}/run_spateo.py" \
                --reference "${SIM_DIR}/reference.h5ad" \
                --moving "${SIM_DIR}/${alt}_rotated_${angle}.h5ad" \
                --output-dir "${out_dir}" \
                &> "${LOG_DIR}/${alt}_spateo_angle_${angle}.log"
            local rc=$?
            set -e

            if [[ ${rc} -ne 0 ]]; then
                log "    FAILED (rc=${rc})"
                fail_record "spateo" "${alt}/angle_${angle} rc=${rc}"
                all_ok=false
            else
                log "    ok"
            fi
        done
    done

    if ${all_ok}; then
        ckpt_done "04_spateo_methods.done"
    fi
}

# ===================================================================
# Stage 5: PASTE alignment methods
# ===================================================================
stage_paste_methods() {
    log "=== Stage 5: PASTE alignment methods ==="

    if ! echo "${METHODS}" | grep -q "paste"; then
        log "  paste not in METHODS, skipping"
        ckpt_done "05_paste_methods.done"
        return
    fi

    local all_ok=true
    for alt in ${ALTERATIONS}; do
        for angle in ${ANGLES}; do
            local out_dir="${METHOD_DIR}/${alt}/paste/angle_${angle}"
            if [[ "${SKIP_METHODS}" == "1" && "${FORCE}" != "1" && -f "${out_dir}/aligned_coordinates.csv" ]]; then
                log "  ${alt}/paste/angle_${angle} already done, skip"
                continue
            fi

            mkdir -p "${out_dir}"
            log "  running ${alt}/paste/angle_${angle} ..."

            set +e
            MPLCONFIGDIR=/tmp/matplotlib ${FEAST_ENV}/bin/python "${METHOD_SCRIPT_DIR}/run_paste.py" \
                --reference "${SIM_DIR}/reference.h5ad" \
                --moving "${SIM_DIR}/${alt}_rotated_${angle}.h5ad" \
                --output-dir "${out_dir}" \
                --alpha "${PASTE_ALPHA}" \
                --dissimilarity "${PASTE_DISSIMILARITY}" \
                --n-pcs "${PASTE_N_PCS}" \
                --num-iter-max "${PASTE_NUM_ITER_MAX}" \
                &> "${LOG_DIR}/${alt}_paste_angle_${angle}.log"
            local rc=$?
            set -e

            if [[ ${rc} -ne 0 ]]; then
                log "    FAILED (rc=${rc})"
                fail_record "paste" "${alt}/angle_${angle} rc=${rc}"
                all_ok=false
            else
                log "    ok"
            fi
        done
    done

    if ${all_ok}; then
        ckpt_done "05_paste_methods.done"
    fi
}

# ===================================================================
# Stage 6: Benchmark
# ===================================================================
stage_benchmark() {
    log "=== Stage 6: Benchmark ==="

    ${FEAST_ENV}/bin/python benchmark.py \
        --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
        --methods-dir "${METHOD_DIR}" \
        --reference "${SIM_DIR}/reference.h5ad" \
        --output "${BENCH_DIR}/alignment_benchmark_results.csv"

    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        cp "${BENCH_DIR}/alignment_benchmark_results.csv" "${RESULT_DIR}/alignment_benchmark_results.csv"
        cp "${BENCH_DIR}/alignment_benchmark_summary.csv" "${RESULT_DIR}/alignment_benchmark_summary.csv"
        ckpt_done "06_benchmark.done"
        log "  done."
    else
        fail_record "benchmark" "alignment_benchmark.py exited with rc=${rc}"
        log "  FAILED (rc=${rc})"
    fi
}

# ===================================================================
# Stage 7: Plots
# ===================================================================
stage_plots() {
    log "=== Stage 7: Plots ==="

    local bench_csv="${BENCH_DIR}/alignment_benchmark_results.csv"
    if [[ ! -f "${bench_csv}" ]]; then
        log "  benchmark CSV not found, skipping plots"
        return
    fi

    ${FEAST_ENV}/bin/python plot_metrics.py \
        --benchmark-csv "${bench_csv}" \
        --output-dir "${PLOT_DIR}"

    ${FEAST_ENV}/bin/python visualize_3d.py \
        --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
        --methods-dir "${METHOD_DIR}" \
        --output-dir "${PLOT_DIR}/alignment_3d"

    ckpt_done "07_plots.done"
    log "  done."
}

# ===================================================================
# Final summary
# ===================================================================
print_final_summary() {
    log ""
    log "========================================"
    log "  Alignment Benchmark Pipeline Complete"
    log "========================================"

    local n_alts=$(echo "${ALTERATIONS}" | wc -w)
    local expected_sim=$((1 + n_alts * (1 + $(echo "${ANGLES}" | wc -w))))  # ref + alts * (1 unrotated + N angles)
    local expected_methods=$((n_alts * $(echo "${ANGLES}" | wc -w)))
    local expected_bench=$((n_alts * $(echo "${ANGLES}" | wc -w) * 2))

    local n_sim=$(ls "${SIM_DIR}"/*.h5ad 2>/dev/null | wc -l)
    local n_spateo=$(find "${METHOD_DIR}" -path "*/spateo/angle_*" -name "aligned_coordinates.csv" 2>/dev/null | wc -l)
    local n_paste=$(find "${METHOD_DIR}" -path "*/paste/angle_*" -name "aligned_coordinates.csv" 2>/dev/null | wc -l)
    local n_bench=0
    if [[ -f "${BENCH_DIR}/alignment_benchmark_results.csv" ]]; then
        n_bench=$(tail -n +2 "${BENCH_DIR}/alignment_benchmark_results.csv" | wc -l)
    fi
    local n_plots=$(find "${PLOT_DIR}" -name "*.pdf" 2>/dev/null | wc -l)

    log "Simulations:      ${n_sim}/${expected_sim} (ref + ${n_alts} alts x (1 + $(echo "${ANGLES}" | wc -w) angles))"
    log "Spateo methods:   ${n_spateo}/${expected_methods}"
    log "PASTE methods:    ${n_paste}/${expected_methods}"
    log "Benchmark rows:   ${n_bench}/${expected_bench}"
    log "Plots generated:  ${n_plots} PDFs"

    # Count failures
    local n_fail=0
    for f in "${CKPT_DIR}"/failures_*.txt; do
        if [[ -f "${f}" ]]; then
            n_fail=$((n_fail + $(wc -l < "${f}")))
        fi
    done
    log "Failed items:     ${n_fail}"
    log "========================================"
}

# ===================================================================
# Main
# ===================================================================
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SIM_DIR}" "${METHOD_DIR}" "${BENCH_DIR}" "${PLOT_DIR}" "${RESULT_DIR}"

log "Alignment Benchmark Pipeline — $(date)"
log "FEAST_ENV=${FEAST_ENV}"
log "PASTE_ALPHA=${PASTE_ALPHA} PASTE_DISSIMILARITY=${PASTE_DISSIMILARITY} PASTE_N_PCS=${PASTE_N_PCS}"
log "FORCE=${FORCE} SEED=${SEED}"
log ""

CURRENT_STAGE="01_link_data";      ckpt_skip "01_link_data.done"      || stage_link_data
CURRENT_STAGE="02_install_deps";   ckpt_skip "02_install_alignment_deps.done" || stage_install_alignment_deps
CURRENT_STAGE="03_simulations";    ckpt_skip "03_simulations.done"    || stage_simulations
CURRENT_STAGE="04_spateo";         ckpt_skip "04_spateo_methods.done" || stage_spateo_methods
CURRENT_STAGE="05_paste";          ckpt_skip "05_paste_methods.done"  || stage_paste_methods
CURRENT_STAGE="06_benchmark";      ckpt_skip "06_benchmark.done"      || stage_benchmark
CURRENT_STAGE="07_plots";          ckpt_skip "07_plots.done"          || stage_plots

print_final_summary
