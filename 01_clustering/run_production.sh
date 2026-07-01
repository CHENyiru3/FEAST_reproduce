#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# Figure 2 Clustering Benchmark — Full Production Runner with Break-Resume
#
# Usage:
#   bash run_production.sh           # run all, skip completed stages
#   bash run_production.sh           # resume after interrupt (same command)
#   FIG2_FORCE=1 bash run_production.sh  # force rerun all stages
#
# Stages:
#   1. Link reference data
#   2. FEAST empirical simulations (3 slices x 23 alterations = 69 jobs)
#   3. HVG refresh
#   4. STAGATE + mclust clustering
#   5. GraphST clustering
#   6. Leiden clustering
#   7. Benchmark metrics
#   8. Plots
#
# Checkpoints: outputs/.checkpoints/
# Logs:        logs/
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- inline config ---------------------------------------------------------
FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
STAGATE_ENV="/maiziezhou_lab2/yiru/envs/STAGATE"
GRAPHST_ENV="/maiziezhou_lab2/yiru/envs/GraphST"
DATASETS="/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad"
SEED=2026

PYTHON_FEAST="${FEAST_ENV}/bin/python"
PYTHON_STAGATE="${STAGATE_ENV}/bin/python"
PYTHON_GRAPHST="${GRAPHST_ENV}/bin/python"

SLICES="${FIG2_SLICE_LIST:-151508 151670 151676}"
FORCE="${FIG2_FORCE:-0}"
FEAST_PPF_METHOD="${FEAST_PPF_METHOD:-interp}"
FEAST_BETA_N_JOBS="${FEAST_BETA_N_JOBS:-4}"
FEAST_BETA_EARLY_STOPPING_PATIENCE="${FEAST_BETA_EARLY_STOPPING_PATIENCE:-2}"
FEAST_ASSIGNMENT_SOLVER="${FEAST_ASSIGNMENT_SOLVER:-scipy}"
FEAST_ASSIGNMENT_BLOCKS="${FEAST_ASSIGNMENT_BLOCKS:-1}"
FEAST_ASSIGNMENT_BLOCK_SIZE="${FEAST_ASSIGNMENT_BLOCK_SIZE:-}"
FEAST_ASSIGNMENT_BLOCK_MULTIPLIER="${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER:-8}"
FEAST_CONVERT_N_JOBS="${FEAST_CONVERT_N_JOBS:-4}"

SIM_DIR="outputs/simulations"
HVG_DIR="outputs/hvg_inputs"
METHOD_DIR="outputs/methods"
BENCH_DIR="outputs/benchmarks"
PLOT_DIR="outputs/plots"
CKPT_DIR="outputs/.checkpoints"
LOG_DIR="logs"
CONFIG_DIR="../configs"
DATA_DIR="data"
RESULT_DIR="results"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib
#=============================================================================

RUN_ID="prod_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${BENCH_DIR}" "${PLOT_DIR}" "${DATA_DIR}" "${RESULT_DIR}"

STAGE_LOG="${LOG_DIR}/${RUN_ID}.log"

# --- helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${STAGE_LOG}"; }
log_section() { log ""; log "==== $* ===="; }
ckpt_done() { touch "${CKPT_DIR}/$1"; }
ckpt_skip() { [[ "${FORCE}" != "1" && -f "${CKPT_DIR}/$1" ]]; }
fail_record() { echo "$*" >> "${CKPT_DIR}/failures_$1.txt"; }
fail_list() { [[ -f "${CKPT_DIR}/failures_$1.txt" ]] && cat "${CKPT_DIR}/failures_$1.txt" || true; }
fail_clear() { rm -f "${CKPT_DIR}/failures_$1.txt"; }

cleanup() {
    log ""
    log "==== INTERRUPTED ===="
    log "Progress saved. Re-run 'bash run_production.sh' to resume."
    exit 130
}
trap cleanup INT TERM

print_status() {
    log_section "CURRENT PROGRESS"
    for f in "${CKPT_DIR}"/*.done; do
        [[ -f "$f" ]] && log "  [DONE] $(basename "$f")"
    done
    for f in "${CKPT_DIR}"/failures_*.txt; do
        if [[ -f "$f" && -s "$f" ]]; then
            local n=$(wc -l < "$f")
            log "  [FAIL] $(basename "$f") — ${n} failures"
        fi
    done
}

# --- stage 1: link reference data -------------------------------------------
stage_1() {
    log_section "STAGE 1/8: Link reference data"
    if ckpt_skip "01_link_data.done"; then
        log "  Skip — already done"
        return 0
    fi

    local all_ok=1
    for slice in ${SLICES}; do
        local src="${DATASETS}/${slice}.h5ad"
        local dst="${DATA_DIR}/${slice}.h5ad"

        if [[ -f "${dst}" ]]; then
            log "  [OK] ${slice}.h5ad — exists"
            continue
        fi

        if [[ ! -f "${src}" ]]; then
            log "  [FAIL] ${slice}.h5ad — source not found: ${src}"
            all_ok=0
            continue
        fi

        ln -sf "${src}" "${dst}"
        log "  [OK] ${slice}.h5ad — linked"

        # Validate required fields
        "${PYTHON_FEAST}" -c "
import anndata as ad
a = ad.read_h5ad('${dst}')
assert a.obsm.get('spatial') is not None, 'missing spatial'
assert 'ground_truth' in a.obs.columns, 'missing ground_truth'
print(f'  -> {a.n_obs} spots x {a.n_vars} genes, domains: {list(a.obs.ground_truth.unique())}')
" 2>&1 | tee -a "${STAGE_LOG}" || {
            log "  [FAIL] ${slice}.h5ad — validation failed"
            all_ok=0
        }
    done

    if [[ "${all_ok}" == "1" ]]; then
        ckpt_done "01_link_data.done"
        log "  Stage 1 complete"
    else
        log "  Stage 1: some slices failed validation"
    fi
}

# --- stage 2: FEAST empirical simulations -----------------------------------
stage_2() {
    log_section "STAGE 2/8: FEAST empirical simulations"
    if ckpt_skip "02_simulations.done"; then
        log "  Skip — already done"
        return 0
    fi

    fail_clear "02_simulations"

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

    log "  Running FEAST simulations for slices: ${SLICES}"
    log "  FEAST knobs: ppf=${FEAST_PPF_METHOD}, beta_n_jobs=${FEAST_BETA_N_JOBS}, beta_patience=${FEAST_BETA_EARLY_STOPPING_PATIENCE}, blocks=${FEAST_ASSIGNMENT_BLOCKS}, block_multiplier=${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER}, convert_n_jobs=${FEAST_CONVERT_N_JOBS}"

    "${PYTHON_FEAST}" run_simulation.py \
        --input-dir "${DATA_DIR}" \
        --slices "$(echo ${SLICES} | tr ' ' ',')" \
        --output-dir "${SIM_DIR}" \
        --parameter-mode hungarian \
        --seed "${SEED}" \
        --config "${CONFIG_DIR}/clustering.yaml" \
        --ppf-method "${FEAST_PPF_METHOD}" \
        --beta-n-jobs "${FEAST_BETA_N_JOBS}" \
        --beta-early-stopping-patience "${FEAST_BETA_EARLY_STOPPING_PATIENCE}" \
        --assignment-solver "${FEAST_ASSIGNMENT_SOLVER}" \
        "${assignment_block_args[@]}" \
        "${assignment_block_size_args[@]}" \
        --assignment-block-multiplier "${FEAST_ASSIGNMENT_BLOCK_MULTIPLIER}" \
        --convert-n-jobs "${FEAST_CONVERT_N_JOBS}" \
        "${overwrite_args[@]}" 2>&1 | tee -a "${STAGE_LOG}"

    # Check manifest for failures
    local manifest="${SIM_DIR}/simulation_manifest.csv"
    if [[ -f "${manifest}" ]]; then
        local ok_count=$("${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${manifest}')
print(df['status'].isin(['ok', 'skipped']).sum())
")
        local total=$("${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${manifest}')
print(len(df))
")
        log "  Simulations: ${ok_count}/${total} usable"

        if [[ "${ok_count}" == "${total}" ]]; then
            cp "${manifest}" "${RESULT_DIR}/simulation_manifest.csv"
            ckpt_done "02_simulations.done"
            log "  Stage 2 complete"
        else
            log "  Stage 2: $((${total} - ${ok_count})) failures — re-run to retry"
        fi
    fi
}

# --- stage 3: HVG refresh ---------------------------------------------------
stage_3() {
    log_section "STAGE 3/8: HVG refresh"
    if ckpt_skip "03_hvg.done"; then
        log "  Skip — already done"
        return 0
    fi

    local overwrite=""
    [[ "${FORCE}" == "1" ]] && overwrite="--overwrite"

    "${PYTHON_FEAST}" run_hvg.py \
        --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
        --output-dir "${HVG_DIR}" \
        --n-top-genes 3000 \
        ${overwrite} 2>&1 | tee -a "${STAGE_LOG}"

    local manifest="${HVG_DIR}/hvg_manifest.csv"
    if [[ -f "${manifest}" ]]; then
        local ok_count=$("${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${manifest}')
print((df['status'] == 'ok').sum())
")
        local total=$("${PYTHON_FEAST}" -c "
import pandas as pd
df = pd.read_csv('${manifest}')
print(len(df))
")
        log "  HVG: ${ok_count}/${total} ok"

        if [[ "${ok_count}" == "${total}" ]]; then
            cp "${manifest}" "${RESULT_DIR}/hvg_manifest.csv"
            ckpt_done "03_hvg.done"
            log "  Stage 3 complete"
        fi
    fi
}

# --- stage 4: STAGATE + mclust ----------------------------------------------
stage_4() {
    log_section "STAGE 4/8: STAGATE + mclust"
    if ckpt_skip "04_stagate.done"; then
        log "  Skip — already done"
        return 0
    fi

    _run_method_jobs "STAGATE_mclust" "${PYTHON_STAGATE}" \
        "methods/stagate_mclust.py" \
        "stagate" "04_stagate"
}

# --- stage 5: GraphST -------------------------------------------------------
stage_5() {
    log_section "STAGE 5/8: GraphST"
    if ckpt_skip "05_graphst.done"; then
        log "  Skip — already done"
        return 0
    fi

    _run_method_jobs "GraphST" "${PYTHON_GRAPHST}" \
        "methods/graphst.py" \
        "graphst" "05_graphst"
}

# --- stage 6: Leiden --------------------------------------------------------
stage_6() {
    log_section "STAGE 6/8: Leiden"
    if ckpt_skip "06_leiden.done"; then
        log "  Skip — already done"
        return 0
    fi

    _run_method_jobs "Leiden" "${PYTHON_FEAST}" \
        "methods/leiden.py" \
        "leiden" "06_leiden"
}

# --- generic method job runner ----------------------------------------------
_run_method_jobs() {
    local method="$1" python="$2" script="$3" tag="$4" ckpt_name="$5"
    local hvg_manifest="${HVG_DIR}/hvg_manifest.csv"

    if [[ ! -f "${hvg_manifest}" ]]; then
        log "  ERROR: HVG manifest not found — run stage 3 first"
        return 1
    fi

    fail_clear "${ckpt_name}"

    local total=0 done_count=0

    while IFS=, read -r slice_id sim_id hvg_file n_hvgs status; do
        [[ "${status}" == "ok" ]] || continue
        total=$((total + 1))
    done < <(tail -n +2 "${hvg_manifest}")

    log "  Method: ${method} — ${total} jobs"

    while IFS=, read -r slice_id sim_id hvg_file n_hvgs status; do
        [[ "${status}" == "ok" ]] || continue

        done_count=$((done_count + 1))
        local out_dir="${METHOD_DIR}/${method}/${slice_id}/${sim_id}"
        local clusters="${out_dir}/clusters.csv"
        local log_file="${LOG_DIR}/${tag}_${slice_id}_${sim_id}.log"

        if [[ -f "${clusters}" && "${FORCE}" != "1" ]]; then
            log "  [${done_count}/${total}] SKIP ${method}:${slice_id}/${sim_id} — exists"
            continue
        fi

        log "  [${done_count}/${total}] RUN  ${method}:${slice_id}/${sim_id}"

        if "${python}" "${script}" \
            --input "${hvg_file}" \
            --output-dir "${out_dir}" > "${log_file}" 2>&1; then
            log "  [${done_count}/${total}] OK   ${method}:${slice_id}/${sim_id}"
        else
            log "  [${done_count}/${total}] FAIL ${method}:${slice_id}/${sim_id} — see ${log_file}"
            fail_record "${slice_id}" "${sim_id}" "${ckpt_name}"
        fi
    done < <(tail -n +2 "${hvg_manifest}")

    local fail_count=$(fail_list "${ckpt_name}" | wc -l)
    local ok_count=$(find "${METHOD_DIR}/${method}" -name "clusters.csv" -type f 2>/dev/null | wc -l)
    log "  Results: ${ok_count} clusters.csv files, ${fail_count} failures recorded"

    if [[ "${fail_count}" -eq 0 && "${ok_count}" -gt 0 ]]; then
        ckpt_done "${ckpt_name}.done"
        log "  Stage complete — ${total} ok"
    else
        log "  Stage: ${fail_count} failures — re-run to retry"
    fi
}

# --- stage 7: benchmark -----------------------------------------------------
stage_7() {
    log_section "STAGE 7/8: Benchmark metrics"
    if ckpt_skip "07_benchmark.done"; then
        log "  Skip — already done"
        return 0
    fi

    "${PYTHON_FEAST}" benchmark.py \
        --input-dir "${METHOD_DIR}" \
        --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
        --label-column ground_truth \
        --output "${BENCH_DIR}/clustering_benchmark_results.csv" 2>&1 | tee -a "${STAGE_LOG}"

    if [[ -f "${BENCH_DIR}/clustering_benchmark_results.csv" ]]; then
        cp "${BENCH_DIR}/clustering_benchmark_results.csv" "${RESULT_DIR}/clustering_benchmark_results.csv"
        ckpt_done "07_benchmark.done"
        log "  Stage 7 complete"
    fi
}

# --- stage 8: plots ---------------------------------------------------------
stage_8() {
    log_section "STAGE 8/8: Plots"
    if ckpt_skip "08_plots.done"; then
        log "  Skip — already done"
        return 0
    fi

    "${PYTHON_FEAST}" plot_metrics.py \
        --benchmark "${BENCH_DIR}/clustering_benchmark_results.csv" \
        --output-dir "${PLOT_DIR}" 2>&1 | tee -a "${STAGE_LOG}"

    "${PYTHON_FEAST}" plot_spatial_maps.py \
        --benchmark "${BENCH_DIR}/clustering_benchmark_results.csv" \
        --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
        --output-dir "${PLOT_DIR}/spatial_maps" 2>&1 | tee -a "${STAGE_LOG}"

    ckpt_done "08_plots.done"
    log "  Stage 8 complete"
}

# --- final summary ----------------------------------------------------------
print_final_summary() {
    log ""
    log_section "FINAL SUMMARY"
    log ""

    # Checkpoint status
    log "Stages completed:"
    for f in 01_link_data 02_simulations 03_hvg 04_stagate 05_graphst 06_leiden 07_benchmark 08_plots; do
        [[ -f "${CKPT_DIR}/${f}.done" ]] && log "  [DONE] ${f}" || log "  [----] ${f}"
    done

    # Benchmark summary
    local bench_csv="${BENCH_DIR}/clustering_benchmark_results.csv"
    if [[ -f "${bench_csv}" ]]; then
        log ""
        log "Benchmark summary:"
        "${PYTHON_FEAST}" -c "
import pandas as pd; import numpy as np
df = pd.read_csv('${bench_csv}')
ok = df[df['status'] == 'ok']
if len(ok) > 0:
    for method in ok['method'].unique():
        m = ok[ok['method'] == method]
        print(f'  {method}: ARI={m[\"ARI\"].mean():.3f} ± {m[\"ARI\"].std():.3f} (n={len(m)})')
print(f'  Total rows: {len(df)} ({len(ok)} ok, {(df[\"status\"]!=\"ok\").sum()} not ok)')
"
    fi

    # Failures
    local any_fails=0
    for f in "${CKPT_DIR}"/failures_*.txt; do
        if [[ -f "$f" && -s "$f" ]]; then
            any_fails=1
            log ""
            log "Failures in $(basename "$f"):"
            cat "$f" | while read line; do log "  $line"; done
        fi
    done
    [[ "${any_fails}" == "0" ]] && log "No failures recorded."

    log ""
    log "Run ID: ${RUN_ID}"
    log "Log:    ${STAGE_LOG}"
    log "All done."
}

#=============================================================================
# MAIN
#=============================================================================
log_section "FIGURE 2 CLUSTERING BENCHMARK — PRODUCTION RUN"
log "Run ID:  ${RUN_ID}"
log "Log:     ${STAGE_LOG}"
log "Seed:    ${SEED}"
log "Slices:  ${SLICES}"
log "Force:   ${FORCE}"

print_status

stage_1
stage_2
stage_3
stage_4
stage_5
stage_6
stage_7
stage_8

print_final_summary
