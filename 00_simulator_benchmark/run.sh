#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# Figure 2 Simulator Benchmark — Production Runner with Checkpoint Resume
#
# Usage:
#   bash run_production.sh           # run all, skip completed stages
#   bash run_production.sh           # resume after interrupt (same command)
#
# Checkpoints are stored in outputs/.checkpoints/
# Logs are written to logs/
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- inline config ---------------------------------------------------------
FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
PYTHON="${FEAST_ENV}/bin/python"
RSCRIPT_SRTSIM="/maiziezhou_lab2/yiru/envs/srtsim-r/bin/Rscript"
RSCRIPT_SPLATTER="/maiziezhou_lab2/yiru/envs/splatter-r46/bin/Rscript"
RSCRIPT_SCCUBE="${FEAST_ENV}/bin/python"
PYTHON_SCCUBE="/maiziezhou_lab2/yiru/envs/sccube/bin/python"
SEED=2026

EXPER_DATA="exper_data"
R_SPLIT="exper_data/R_split"
OUTPUTS="outputs/simulation_saved"
BENCHMARKS="outputs/benchmarks"
VIZ_DIR="outputs/quality_visualization"
CKPT_DIR="outputs/.checkpoints"
LOG_DIR="logs"

# Two FEAST spatial modes for the benchmark
declare -A FEAST_MODES=(
    ["FEAST_Rank"]="hungarian reference_rank hybrid"
    ["FEAST_OT_Spatial"]="hungarian ot_spatial hybrid"
)

# External simulators: folder -> env
declare -A EXT_SIM=(
    ["srtsim_updated"]="SRTsim"
    ["splat"]="Splatter"
    ["splatSimple"]="Splatter_Simple"
    ["sccube"]="scCube"
)

# Known exclusions (simulator limitations documented in inventory)
EXCLUDE_SRTSIM=""
EXCLUDE_SPLAT=""

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib
#=============================================================================

RUN_ID="prod_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${BENCHMARKS}" "${VIZ_DIR}"

STAGE_LOG="${LOG_DIR}/${RUN_ID}.log"

# --- helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${STAGE_LOG}"; }
log_section() { log ""; log "==== $* ===="; }
ckpt_done() { touch "${CKPT_DIR}/$1"; }
ckpt_skip() { [[ -f "${CKPT_DIR}/$1" ]]; }

fail_record() {
    local stage="$1" label="$2" sample="$3"
    echo "${label}:${sample}" >> "${CKPT_DIR}/failures_${stage}.txt"
}

fail_list() {
    local stage="$1"
    [[ -f "${CKPT_DIR}/failures_${stage}.txt" ]] && cat "${CKPT_DIR}/failures_${stage}.txt" || true
}

fail_clear() { rm -f "${CKPT_DIR}/failures_$1.txt"; }

# --- cleanup on interrupt ---------------------------------------------------
cleanup() {
    log ""
    log "==== INTERRUPTED ===="
    log "Progress saved. Re-run 'bash run_production.sh' to resume."
    exit 130
}
trap cleanup INT TERM

# --- stage: status check -----------------------------------------------------
print_status() {
    log_section "CURRENT PROGRESS"
    local total=0 matched=0
    for folder in FEAST_Rank FEAST_OT_Spatial srtsim_updated splat splatSimple sccube; do
        local dir="${OUTPUTS}/${folder}"
        local n=0
        [[ -d "${dir}" ]] && n=$(find "${dir}" -name "*.h5ad" -type f | wc -l)
        log "  ${folder}: ${n} outputs"
        total=$((total + 1))
        [[ $n -gt 0 ]] && matched=$((matched + 1))
    done
    for f in "${CKPT_DIR}"/*.done; do
        [[ -f "$f" ]] && log "  [DONE] $(basename "$f")"
    done
}

# --- stage 1: prepare inputs ------------------------------------------------
stage_1() {
    log_section "STAGE 1/7: Prepare simulator inputs"
    if ckpt_skip "01_inputs.done"; then
        log "  Skip — already done ($(ls exper_data/*.h5ad 2>/dev/null | wc -l) inputs)"
        return 0
    fi

    "${PYTHON}" scripts/prepare_simulator_inputs.py \
        --datasets-root /maiziezhou_lab2/yiru/Datasets \
        --output-dir "${EXPER_DATA}" \
        --manifest "${BENCHMARKS}/input_manifest.csv" \
        --seed "${SEED}" 2>&1 | tee -a "${STAGE_LOG}"

    log "  Stage 1 complete"
    ckpt_done "01_inputs.done"
}

# --- stage 2: build R_split -------------------------------------------------
stage_2() {
    log_section "STAGE 2/7: Build R_split inputs"
    if ckpt_skip "02_rsplit.done"; then
        log "  Skip — already done"
        return 0
    fi

    "${PYTHON}" scripts/external_simulators/build_r_split_inputs.py \
        --input-dir "${EXPER_DATA}" \
        --output-dir "${R_SPLIT}" \
        --manifest "${BENCHMARKS}/r_split_manifest.csv" 2>&1 | tee -a "${STAGE_LOG}"

    log "  Stage 2 complete"
    ckpt_done "02_rsplit.done"
}

# --- stage 3: FEAST benchmark -----------------------------------------------
stage_3() {
    log_section "STAGE 3/7: FEAST simulator benchmark"

    if ckpt_skip "03_feast.done"; then
        log "  Skip — already done"
        return 0
    fi

    # Collect samples
    mapfile -t SAMPLES < <(find "${EXPER_DATA}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" | sort | sed 's/[.]h5ad$//')
    local total_samples=${#SAMPLES[@]}

    # Check for previous failures to retry first
    local retry_set=""
    for entry in $(fail_list "03_feast"); do
        retry_set="${retry_set} ${entry}"
    done
    fail_clear "03_feast"

    local overall_done=0
    local overall_total=$((total_samples * 2))

    for mode_label in "${!FEAST_MODES[@]}"; do
        local mode_config="${FEAST_MODES[$mode_label]}"
        read -r param_mode spatial_mode assign_method <<< "${mode_config}"

        local out_dir="${OUTPUTS}/${mode_label}"
        mkdir -p "${out_dir}"

        log ""
        log "--- FEAST mode: ${mode_label} (param=${param_mode}, spatial=${spatial_mode}, assign=${assign_method}) ---"

        local done_count=0
        for sample in "${SAMPLES[@]}"; do
            local input_path="${EXPER_DATA}/${sample}.h5ad"
            local output_path="${out_dir}/${sample}.h5ad"

            # Skip if output already exists and appears valid
            if [[ -f "${output_path}" ]]; then
                log "  [SKIP] ${mode_label}:${sample} — exists"
                done_count=$((done_count + 1))
                overall_done=$((overall_done + 1))
                continue
            fi

            local sample_log="${LOG_DIR}/feast_${mode_label}_${sample}.log"
            log "  [RUN]  ${mode_label}:${sample} [$(($done_count + 1))/${total_samples}]"

            if "${PYTHON}" scripts/ParameterCloud_sim.py \
                --input "${input_path}" \
                --output "${output_path}" \
                --parameter-mode "${param_mode}" \
                --spatial-mode "${spatial_mode}" \
                --assignment-method "${assign_method}" \
                --annotation-key auto \
                --seed "${SEED}" > "${sample_log}" 2>&1; then
                log "  [OK]   ${mode_label}:${sample}"
                done_count=$((done_count + 1))
                overall_done=$((overall_done + 1))
            else
                log "  [FAIL] ${mode_label}:${sample} — see ${sample_log}"
                fail_record "03_feast" "${mode_label}" "${sample}"
                done_count=$((done_count + 1))
                overall_done=$((overall_done + 1))
            fi
        done
    done

    local remaining=$(fail_list "03_feast" | wc -l)
    if [[ ${remaining} -eq 0 ]]; then
        log "  Stage 3 complete — ${overall_total} outputs"
        ckpt_done "03_feast.done"
    else
        log "  Stage 3: ${remaining} failures remain — fix and re-run to retry"
    fi
}

# --- stage 4: external simulators -------------------------------------------
stage_4() {
    log_section "STAGE 4/7: External simulators"

    if ckpt_skip "04_external.done"; then
        log "  Skip — already done"
        return 0
    fi

    mapfile -t SAMPLES < <(find "${EXPER_DATA}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" | sed 's/[.]h5ad$//' | sort)

    # --- helper: run R-based simulator ---
    run_r_sim() {
        local sample="$1" tool_name="$2" r_script="$3" prefix="$4"
        local out_dir="$5" exclude="$6" rscript_bin="$7"

        [[ -n "${exclude}" && "${sample}" == "${exclude}" ]] && {
            log "  [SKIP] ${tool_name}:${sample} — excluded"
            return 0
        }

        local output_h5ad="${out_dir}/${sample}.h5ad"
        [[ -f "${output_h5ad}" ]] && {
            log "  [SKIP] ${tool_name}:${sample} — exists"
            return 0
        }

        local tmp_dir="${out_dir}/_tmp/${sample}"
        mkdir -p "${tmp_dir}" "${out_dir}"

        local sample_log="${LOG_DIR}/ext_${tool_name}_${sample}.log"
        log "  [RUN]  ${tool_name}:${sample}"

        local r_split_sample="${R_SPLIT}/${sample}"
        if [[ ! -d "${r_split_sample}" ]]; then
            log "  [FAIL] ${tool_name}:${sample} — R_split dir missing"
            fail_record "04_external" "${tool_name}" "${sample}"
            return 0
        fi

        if ! "${rscript_bin}" "${r_script}" \
            --cellinfo "${r_split_sample}/cellinfo.csv" \
            --geneinfo "${r_split_sample}/geneinfo.csv" \
            --matrix "${r_split_sample}/sparse_matrix.mtx" \
            --spatial_loc "${r_split_sample}/spatial.csv" \
            --output_dir "${tmp_dir}" > "${sample_log}" 2>&1; then
            log "  [FAIL] ${tool_name}:${sample} — see ${sample_log}"
            fail_record "04_external" "${tool_name}" "${sample}"
            return 0
        fi

        # Reconstruct h5ad
        if "${PYTHON}" scripts/external_simulators/reconstruct_external_h5ad.py \
            --count_path "${tmp_dir}/${prefix}_count.mtx" \
            --gene_path "${tmp_dir}/${prefix}_genes.csv" \
            --location_path "${tmp_dir}/${prefix}_loc.csv" \
            --save_adata_path "${output_h5ad}" >> "${sample_log}" 2>&1; then
            log "  [OK]   ${tool_name}:${sample}"
        else
            log "  [FAIL] ${tool_name}:${sample} — reconstruction failed"
            fail_record "04_external" "${tool_name}" "${sample}"
        fi
    }

    # --- helper: run scCube ---
    run_sccube() {
        local sample="$1"
        local out_dir="${OUTPUTS}/sccube"
        mkdir -p "${out_dir}"
        local output_h5ad="${out_dir}/${sample}_sccube_sim.h5ad"

        [[ -f "${output_h5ad}" ]] && {
            log "  [SKIP] scCube:${sample} — exists"
            return 0
        }

        local input_h5ad="${EXPER_DATA}/${sample}.h5ad"
        [[ ! -f "${input_h5ad}" ]] && {
            log "  [FAIL] scCube:${sample} — input missing"
            fail_record "04_external" "scCube" "${sample}"
            return 0
        }

        # Auto-resolve label column
        local cell_key
        cell_key=$("${PYTHON}" -c "
import anndata as ad
a = ad.read_h5ad('${input_h5ad}')
for c in ['ground_truth', 'cell_type', 'annotation', 'region', 'cluster']:
    if c in a.obs.columns:
        print(c)
        break
" 2>/dev/null)
        [[ -z "${cell_key}" ]] && cell_key="ground_truth"

        local sample_log="${LOG_DIR}/ext_scCube_${sample}.log"
        log "  [RUN]  scCube:${sample} (cell_key=${cell_key})"

        if "${PYTHON_SCCUBE}" scripts/external_simulators/sccube_run.py \
            --h5ad_file "${input_h5ad}" \
            --output_dir "${output_h5ad}" \
            --cell_key "${cell_key}" > "${sample_log}" 2>&1; then
            log "  [OK]   scCube:${sample}"
        else
            log "  [FAIL] scCube:${sample} — see ${sample_log}"
            fail_record "04_external" "scCube" "${sample}"
        fi
    }

    fail_clear "04_external"

    for sample in "${SAMPLES[@]}"; do
        # SRTsim
        run_r_sim "${sample}" "SRTsim" \
            "scripts/external_simulators/SRTsim_run.R" "SRTsim" \
            "${OUTPUTS}/srtsim_updated" "${EXCLUDE_SRTSIM}" \
            "${RSCRIPT_SRTSIM}"

        # Splatter
        run_r_sim "${sample}" "Splatter" \
            "scripts/external_simulators/splat_run.R" "splat" \
            "${OUTPUTS}/splat" "${EXCLUDE_SPLAT}" \
            "${RSCRIPT_SPLATTER}"

        # Splatter Simple
        run_r_sim "${sample}" "Splatter_Simple" \
            "scripts/external_simulators/splatSimple_run.R" "splatSimple" \
            "${OUTPUTS}/splatSimple" "" \
            "${RSCRIPT_SPLATTER}"

        # scCube
        run_sccube "${sample}"
    done

    local remaining=$(fail_list "04_external" | wc -l)
    if [[ ${remaining} -eq 0 ]]; then
        log "  Stage 4 complete"
        ckpt_done "04_external.done"
    else
        log "  Stage 4: ${remaining} failures remain"
    fi
}

# --- stage 5: inventory -----------------------------------------------------
stage_5() {
    log_section "STAGE 5/7: Build simulator inventory"

    if ckpt_skip "05_inventory.done"; then
        log "  Skip — already done"
        return 0
    fi

    "${PYTHON}" scripts/build_simulator_inventory.py \
        --output-base "${OUTPUTS}" \
        --exper-data "${EXPER_DATA}" \
        --inventory "${BENCHMARKS}/simulator_inventory.csv" 2>&1 | tee -a "${STAGE_LOG}"

    ckpt_done "05_inventory.done"
}

# --- stage 6: quality metrics -----------------------------------------------
stage_6() {
    log_section "STAGE 6/7: Build quality metrics"

    if ckpt_skip "06_metrics.done"; then
        log "  Skip — already done"
        return 0
    fi

    "${PYTHON}" scripts/build_simulator_quality_metrics.py \
        --inventory "${BENCHMARKS}/simulator_inventory.csv" \
        --exper-data "${EXPER_DATA}" \
        --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
        --metadata "${BENCHMARKS}/simulator_quality_metric_metadata.json" 2>&1 | tee -a "${STAGE_LOG}"

    ckpt_done "06_metrics.done"
}

# --- stage 7: visualization + summary ---------------------------------------
stage_7() {
    log_section "STAGE 7/7: Visualization + Summary"

    if ckpt_skip "07_viz.done"; then
        log "  Skip — already done"
    else
        "${PYTHON}" scripts/plot_simulator_quality.py \
            --metrics "${BENCHMARKS}/simulator_quality_metrics.csv" \
            --output-dir "${VIZ_DIR}" 2>&1 | tee -a "${STAGE_LOG}"
        ckpt_done "07_viz.done"
    fi

    # --- final summary ---
    log ""
    log_section "FINAL SUMMARY"
    log ""

    # Inventory
    if [[ -f "${BENCHMARKS}/simulator_inventory.csv" ]]; then
        log "Inventory:"
        "${PYTHON}" -c "
import pandas as pd
df = pd.read_csv('${BENCHMARKS}/simulator_inventory.csv')
counts = df.groupby(['simulator','status']).size().unstack(fill_value=0)
print(counts.to_string())
" 2>&1 | tee -a "${STAGE_LOG}"
    fi

    # Checkpoint status
    log ""
    log "Checkpoints:"
    for f in "${CKPT_DIR}"/*.done; do
        [[ -f "$f" ]] && log "  [DONE] $(basename "$f")"
    done

    # Failures
    for f in "${CKPT_DIR}"/failures_*.txt; do
        if [[ -f "$f" && -s "$f" ]]; then
            log ""
            log "Failures in $(basename "$f"):"
            cat "$f" | while read line; do log "  $line"; done
        fi
    done

    log ""
    log "Log: ${STAGE_LOG}"
    log "All done."
}

#=============================================================================
# MAIN
#=============================================================================
log_section "FIGURE 2 SIMULATOR BENCHMARK — PRODUCTION RUN"
log "Run ID:  ${RUN_ID}"
log "Log:     ${STAGE_LOG}"
log "Seed:    ${SEED}"

print_status

stage_1
stage_2
stage_3
stage_4
stage_5
stage_6
stage_7
