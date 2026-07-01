#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# Figure 2 Deconvolution Benchmark — Resume Runner
#
# Orchestrates the full deconvolution pipeline: simulation, Cell2location,
# RCTD, benchmark, and visualization. Skips already-completed stages by
# default, with force/dry-run controls.
#
# Usage:
#   bash scripts/run_deconvolution_resume.sh           # run missing stages
#   FIG2_FORCE=1 bash scripts/run_deconvolution_resume.sh   # rerun all
#   FIG2_DRY_RUN=1 bash scripts/run_deconvolution_resume.sh # inventory only
#   FIG2_FORCE_RCTD_RERUN=1 bash scripts/run_deconvolution_resume.sh
#
# Environment:
#   FIG2_FORCE=1              Rebuild all outputs
#   FIG2_DRY_RUN=1            Print inventory without executing
#   FIG2_FORCE_RCTD_RERUN=1   Force rebuild of RCTD only
#   FIG2_SLICE_LIST           Override default slices (space-separated)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DVC_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${DVC_ROOT}"

# ---- Paths ----
FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
CELL2LOC_ENV="/maiziezhou_lab2/yiru/envs/cell2loc_env"
RCTD_ENV="/maiziezhou_lab2/yiru/envs/rctd_bioc"
RCTD_RSCRIPT="${RCTD_ENV}/bin/Rscript"
RCTD_CACHE_DIR="/tmp/rctd_bioc_cache"

PYTHON_FEAST="${FEAST_ENV}/bin/python"
PYTHON_C2L="${CELL2LOC_ENV}/bin/python"

INPUT_DIR="${DVC_ROOT}/data"
CONFIG="${DVC_ROOT}/configs/deconvolution.yaml"
CELL_TYPE_KEY="cell_type"
SEED=2026

SIM_OUT="${DVC_ROOT}/outputs/simulations"
TRUTH_OUT="${DVC_ROOT}/outputs/ground_truth"
C2L_OUT="${DVC_ROOT}/outputs/cell2location"
RCTD_OUT="${DVC_ROOT}/outputs/rctd"
BENCH_OUT="${DVC_ROOT}/outputs/benchmarks"
PLOT_OUT="${DVC_ROOT}/outputs/plots"
CKPT_DIR="${DVC_ROOT}/outputs/.checkpoints"
LOG_DIR="${DVC_ROOT}/logs"
RESULT_DIR="${SCRIPT_DIR}/results"

SLICES_DEFAULT=("007" "050" "100")
RESOLUTIONS=("0.1" "0.25")

if [[ -n "${FIG2_SLICE_LIST:-}" ]]; then
    read -ra SLICES <<< "${FIG2_SLICE_LIST}"
else
    SLICES=("${SLICES_DEFAULT[@]}")
fi

FORCE="${FIG2_FORCE:-0}"
DRY_RUN="${FIG2_DRY_RUN:-0}"
FORCE_RCTD="${FIG2_FORCE_RCTD_RERUN:-0}"
FEAST_PPF_METHOD="${FEAST_PPF_METHOD:-interp}"
FEAST_BETA_N_JOBS="${FEAST_BETA_N_JOBS:-4}"
FEAST_BETA_EARLY_STOPPING_PATIENCE="${FEAST_BETA_EARLY_STOPPING_PATIENCE:-2}"
FEAST_ASSIGNMENT_SOLVER="${FEAST_ASSIGNMENT_SOLVER:-scipy}"
FEAST_ASSIGNMENT_BLOCKS="${FEAST_ASSIGNMENT_BLOCKS:-1}"
FEAST_ASSIGNMENT_BLOCK_SIZE="${FEAST_ASSIGNMENT_BLOCK_SIZE:-}"
FEAST_ASSIGNMENT_BLOCK_MULTIPLIER="${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER:-8}"
FEAST_CONVERT_N_JOBS="${FEAST_CONVERT_N_JOBS:-4}"

#=============================================================================
# Helpers
#=============================================================================
_log()  { echo "[$(date '+%H:%M:%S')] $*"; }
_warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
_fail() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

_file_valid() { [[ -f "$1" ]] && [[ -s "$1" ]]; }

_csv_valid() {
    # Check CSV file exists, is non-empty, and has at least one data row
    [[ -f "$1" ]] && [[ -s "$1" ]] && [[ $(wc -l < "$1") -ge 2 ]]
}

_remove_truncated_csvs() {
    # Remove CSV files that exist but are header-only (corrupt/truncated)
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        while IFS= read -r -d '' f; do
            if [[ -f "$f" && -s "$f" ]] && [[ $(wc -l < "$f") -lt 2 ]]; then
                _warn "Removing truncated CSV: ${f}"
                rm -f "$f"
            fi
        done < <(find "${dir}" -name "*.csv" -print0 2>/dev/null)
    fi
}

ckpt_done() { touch "${CKPT_DIR}/$1.done"; }
ckpt_skip() { [[ "${FORCE}" != "1" && -f "${CKPT_DIR}/$1.done" ]]; }

cleanup() {
    _log ""
    _log "==== INTERRUPTED ===="
    _log "Progress saved. Re-run 'bash scripts/run_deconvolution_resume.sh' to resume."
    exit 130
}
trap cleanup INT TERM

#=============================================================================
# Inventory
#=============================================================================
_inventory() {
    local total=0 missing=0
    _log "=== Pipeline Inventory ==="
    _log "Slices: ${SLICES[*]}"
    _log ""

    _log "Stage 1 — Simulation (.h5ad):"
    for slice in "${SLICES[@]}"; do
        for res in "${RESOLUTIONS[@]}"; do
            local f="${SIM_OUT}/${slice}/resolution_${res}.h5ad"
            total=$((total + 1))
            if _file_valid "${f}"; then
                _log "  ok  ${f}"
            else
                _log "  --- ${f}  [MISSING]"
                missing=$((missing + 1))
            fi
        done
    done

    _log ""
    _log "Stage 1 — Ground Truth (CSV):"
    for slice in "${SLICES[@]}"; do
        for res in "${RESOLUTIONS[@]}"; do
            local f="${TRUTH_OUT}/${slice}/resolution_${res}_proportions.csv"
            total=$((total + 1))
            if _file_valid "${f}"; then
                _log "  ok  ${f}"
            else
                _log "  --- ${f}  [MISSING]"
                missing=$((missing + 1))
            fi
        done
    done

    _log ""
    _log "Stage 2 — Cell2location:"
    for slice in "${SLICES[@]}"; do
        for res in "${RESOLUTIONS[@]}"; do
            local f="${C2L_OUT}/${slice}/resolution_${res}_proportions.csv"
            total=$((total + 1))
            if _file_valid "${f}"; then
                _log "  ok  ${f}"
            else
                _log "  --- ${f}  [MISSING]"
                missing=$((missing + 1))
            fi
        done
    done

    _log ""
    _log "Stage 3 — RCTD:"
    for slice in "${SLICES[@]}"; do
        for res in "${RESOLUTIONS[@]}"; do
            local f="${RCTD_OUT}/${slice}/resolution_${res}_proportions.csv"
            total=$((total + 1))
            if _file_valid "${f}"; then
                _log "  ok  ${f}"
            else
                _log "  --- ${f}  [MISSING]"
                missing=$((missing + 1))
            fi
        done
    done

    _log ""
    _log "Stage 4 — Benchmark:"
    for f in "deconvolution_benchmark_results.csv" "deconvolution_benchmark_summary.csv"; do
        total=$((total + 1))
        if _file_valid "${BENCH_OUT}/${f}"; then
            _log "  ok  ${f}"
        else
            _log "  --- ${f}  [MISSING]"
            missing=$((missing + 1))
        fi
    done

    _log ""
    _log "Stage 5 — Visualization:"
    for f in "deconvolution_metrics.pdf" "deconvolution_metrics.png"; do
        total=$((total + 1))
        if _file_valid "${PLOT_OUT}/${f}"; then
            _log "  ok  ${f}"
        else
            _log "  --- ${f}  [MISSING]"
            missing=$((missing + 1))
        fi
    done

    _log ""
    _log "Summary: $((total - missing))/${total} present, ${missing} missing"

    return "${missing}"
}

#=============================================================================
# Stage runner
#=============================================================================
_run_stage() {
    local name="$1" description="$2" cmd="$3" check_path="$4"

    _log ""
    _log "==== STAGE: ${name} ===="
    _log "  ${description}"

    # Use _csv_valid for CSV outputs (catches header-only corrupt files)
    local skip=0
    if [[ "${FORCE}" != "1" ]]; then
        if [[ "${check_path}" == *.csv ]] && _csv_valid "${check_path}"; then
            skip=1
        elif [[ "${check_path}" != *.csv ]] && _file_valid "${check_path}"; then
            skip=1
        fi
    fi

    if [[ "${skip}" == "1" ]]; then
        _log "  SKIP — output exists: ${check_path}"
        return 0
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        _log "  [DRY RUN] Would execute: ${cmd}"
        return 0
    fi

    _log "  Running..."
    if eval "${cmd}"; then
        _log "  OK"
    else
        _fail "Stage '${name}' FAILED (exit code $?)"
        return 1
    fi
}

#=============================================================================
# Setup
#=============================================================================
_setup() {
    mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${RESULT_DIR}"
    mkdir -p "${SIM_OUT}" "${TRUTH_OUT}" "${C2L_OUT}" "${RCTD_OUT}" \
             "${BENCH_OUT}" "${PLOT_OUT}"

    # Create data symlinks if missing
    local src_base="/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad"
    for slice in "${SLICES[@]}"; do
        local src="${src_base}/Zhuang-ABCA-1.${slice}.h5ad"
        local dst="${INPUT_DIR}/${slice}.h5ad"
        local named_dst="${INPUT_DIR}/Zhuang-ABCA-1.${slice}.h5ad"
        if ! _file_valid "${dst}"; then
            if ! _file_valid "${src}"; then
                _fail "Source data not found: ${src}"
                return 1
            fi
            mkdir -p "${INPUT_DIR}"
            ln -sf "${src}" "${dst}"
            _log "Symlinked: ${dst} -> ${src}"
        fi
        if ! _file_valid "${named_dst}"; then
            ln -sf "${src}" "${named_dst}"
            _log "Symlinked: ${named_dst} -> ${src}"
        fi
    done

    if [[ "${DRY_RUN}" != "1" ]]; then
        RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
        STAGE_LOG="${LOG_DIR}/${RUN_ID}.log"
        exec > >(tee -a "${STAGE_LOG}") 2>&1
    fi
}

#=============================================================================
# MAIN
#=============================================================================
main() {
    _log "=========================================="
    _log "Figure 2 Deconvolution Pipeline — $(date)"
    _log "Working dir: ${DVC_ROOT}"
    _log "Slices:  ${SLICES[*]}"
    _log "Resolutions: ${RESOLUTIONS[*]}"
    _log "FORCE=${FORCE}  DRY_RUN=${DRY_RUN}  FORCE_RCTD=${FORCE_RCTD}"
    _log "=========================================="
    _log ""

    _setup

    # --- Stage 1: Simulation ---
    local sim_overwrite=""
    [[ "${FORCE}" == "1" ]] && sim_overwrite="--overwrite"
    local assignment_block_args=""
    if [[ "${FEAST_ASSIGNMENT_BLOCKS}" == "1" ]]; then
        assignment_block_args="--assignment-blocks"
    else
        assignment_block_args="--no-assignment-blocks"
    fi
    local assignment_block_size_args=""
    if [[ -n "${FEAST_ASSIGNMENT_BLOCK_SIZE}" ]]; then
        assignment_block_size_args="--assignment-block-size ${FEAST_ASSIGNMENT_BLOCK_SIZE}"
    fi
    local slices_str
    slices_str="$(IFS=,; echo "${SLICES[*]}")"
    _log "FEAST knobs: ppf=${FEAST_PPF_METHOD}, beta_n_jobs=${FEAST_BETA_N_JOBS}, beta_patience=${FEAST_BETA_EARLY_STOPPING_PATIENCE}, blocks=${FEAST_ASSIGNMENT_BLOCKS}, block_multiplier=${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER}, convert_n_jobs=${FEAST_CONVERT_N_JOBS}"

    _run_stage \
        "Simulation" \
        "FEAST deconvolution simulation (${#SLICES[@]} slices x ${#RESOLUTIONS[@]} resolutions)" \
        "${PYTHON_FEAST} ${SCRIPT_DIR}/run_deconvolution.py \
           --input-dir ${INPUT_DIR} \
           --slices ${slices_str} \
           --output-dir ${SIM_OUT} \
           --resolutions ${RESOLUTIONS[*]} \
           --cell-type-key ${CELL_TYPE_KEY} \
           --seed ${SEED} \
           --config ${CONFIG} \
           --ppf-method ${FEAST_PPF_METHOD} \
           --beta-n-jobs ${FEAST_BETA_N_JOBS} \
           --beta-early-stopping-patience ${FEAST_BETA_EARLY_STOPPING_PATIENCE} \
           --assignment-solver ${FEAST_ASSIGNMENT_SOLVER} \
           ${assignment_block_args} \
           ${assignment_block_size_args} \
           --assignment-block-multiplier ${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER} \
           --convert-n-jobs ${FEAST_CONVERT_N_JOBS} \
           ${sim_overwrite}" \
        "${SIM_OUT}/../simulation_manifest.csv" \
        || exit 1

    ckpt_done "01_simulation"
    if [[ "${DRY_RUN}" != "1" ]]; then
        _log "Simulation manifest:"
        "${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${SIM_OUT}/../simulation_manifest.csv')
ok = (df['status'] == 'ok').sum()
sk = (df['status'] == 'skipped').sum()
fl = (df['status'].str.startswith('failed')).sum()
print(f'  {ok} ok, {sk} skipped, {fl} failed')
"
    fi

    # --- Stage 2: Cell2location ---
    _remove_truncated_csvs "${C2L_OUT}"
    for slice in "${SLICES[@]}"; do
        local ref_data="${INPUT_DIR}/${slice}.h5ad"
        if ! _file_valid "${ref_data}"; then
            _warn "Skipping Cell2location for ${slice} — reference not found"
            continue
        fi

        for res in "${RESOLUTIONS[@]}"; do
            local sim_input="${SIM_OUT}/${slice}/resolution_${res}.h5ad"
            local c2l_output="${C2L_OUT}/${slice}/resolution_${res}_proportions.csv"
            local c2l_overwrite=""
            [[ "${FORCE}" == "1" ]] && c2l_overwrite="--overwrite"

            if ! _file_valid "${sim_input}"; then
                _warn "Skipping Cell2location for ${slice}/resolution_${res} — simulation missing"
                continue
            fi

            _run_stage \
                "Cell2location ${slice}/res=${res}" \
                "Cell2location deconvolution" \
                "${PYTHON_C2L} ${SCRIPT_DIR}/methods/run_cell2loc.py \
                   --input ${sim_input} \
                   --reference ${ref_data} \
                   --cell-type-key ${CELL_TYPE_KEY} \
                   --output ${c2l_output} \
                   --seed ${SEED} \
                   ${c2l_overwrite}" \
                "${c2l_output}" \
                || exit 1
        done
    done
    ckpt_done "02_cell2location"

    # --- Stage 3: RCTD ---
    _remove_truncated_csvs "${RCTD_OUT}"
    for slice in "${SLICES[@]}"; do
        local ref_data="${INPUT_DIR}/${slice}.h5ad"
        if ! _file_valid "${ref_data}"; then
            _warn "Skipping RCTD for ${slice} — reference not found"
            continue
        fi

        for res in "${RESOLUTIONS[@]}"; do
            local sim_input="${SIM_OUT}/${slice}/resolution_${res}.h5ad"
            local rctd_output="${RCTD_OUT}/${slice}/resolution_${res}_proportions.csv"
            local rctd_overwrite=""
            [[ "${FORCE}" == "1" || "${FORCE_RCTD}" == "1" ]] && rctd_overwrite="--force"

            if ! _file_valid "${sim_input}"; then
                _warn "Skipping RCTD for ${slice}/resolution_${res} — simulation missing"
                continue
            fi

            _run_stage \
                "RCTD ${slice}/res=${res}" \
                "RCTD full-mode deconvolution" \
                "${PYTHON_FEAST} ${SCRIPT_DIR}/methods/run_rctd.py \
                   --input ${sim_input} \
                   --reference ${ref_data} \
                   --cell-type-key ${CELL_TYPE_KEY} \
                   --rscript ${RCTD_RSCRIPT} \
                   --output ${rctd_output} \
                   --cache-dir ${RCTD_CACHE_DIR} \
                   ${rctd_overwrite}" \
                "${rctd_output}" \
                || exit 1
        done
    done
    ckpt_done "03_rctd"

    # --- Stage 4: Benchmark ---
    local bench_overwrite=""
    [[ "${FORCE}" == "1" ]] && bench_overwrite="--overwrite"
    local pred_dirs="${C2L_OUT} ${RCTD_OUT}"

    _run_stage \
        "Benchmark" \
        "Compute deconvolution benchmark metrics" \
        "${PYTHON_FEAST} ${SCRIPT_DIR}/benchmark.py \
           --truth-dir ${TRUTH_OUT} \
           --prediction-dirs ${pred_dirs} \
           --slices ${SLICES[*]} \
           --resolutions ${RESOLUTIONS[*]} \
           --output ${BENCH_OUT}/deconvolution_benchmark_results.csv \
           ${bench_overwrite}" \
        "${BENCH_OUT}/deconvolution_benchmark_results.csv" \
        || exit 1
    ckpt_done "04_benchmark"
    if [[ "${DRY_RUN}" != "1" ]]; then
        cp "${BENCH_OUT}/deconvolution_benchmark_results.csv" "${RESULT_DIR}/deconvolution_benchmark_results.csv"
        if _file_valid "${BENCH_OUT}/deconvolution_benchmark_summary.csv"; then
            cp "${BENCH_OUT}/deconvolution_benchmark_summary.csv" "${RESULT_DIR}/deconvolution_benchmark_summary.csv"
        fi
    fi

    # Print benchmark summary
    if [[ "${DRY_RUN}" != "1" ]] && _file_valid "${BENCH_OUT}/deconvolution_benchmark_summary.csv"; then
        _log "Benchmark summary:"
        "${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${BENCH_OUT}/deconvolution_benchmark_summary.csv')
for _, row in df.iterrows():
    print(f\"  {row['metric']}: mean={row['mean']:.4f}, best={row['best_method']}/{row['best_slice']}\")
"
    fi

    # --- Stage 5: Visualization ---
    _run_stage \
        "Visualization" \
        "Generate deconvolution benchmark plots" \
        "${PYTHON_FEAST} ${SCRIPT_DIR}/visualization.py \
           --benchmark-csv ${BENCH_OUT}/deconvolution_benchmark_results.csv \
           --truth-dir ${TRUTH_OUT} \
           --prediction-dirs ${pred_dirs} \
           --simulation-dir ${SIM_OUT} \
           --slices ${SLICES[*]} \
           --resolutions ${RESOLUTIONS[*]} \
           --output-dir ${PLOT_OUT}" \
        "${PLOT_OUT}/deconvolution_metrics.pdf" \
        || exit 1
    ckpt_done "05_visualization"

    # --- Final summary ---
    _log ""
    _log "=========================================="
    _log "Pipeline complete — $(date)"
    _log "=========================================="
    _log "Stages completed:"
    for stage in 01_simulation 02_cell2location 03_rctd 04_benchmark 05_visualization; do
        if [[ -f "${CKPT_DIR}/${stage}.done" ]]; then
            _log "  [DONE] ${stage}"
        else
            _log "  [----] ${stage}"
        fi
    done

    _inventory
    _log ""
    _log "Run ID: ${RUN_ID:-dry-run}"
    [[ -n "${STAGE_LOG:-}" ]] && _log "Log:    ${STAGE_LOG}"
    _log "All done."
}

if [[ "${DRY_RUN}" == "1" ]]; then
    _log "DRY RUN MODE — no commands will be executed."
    _setup
    _inventory
else
    main
fi
