#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# Deconvolution Methods Runner
#
# Runs Cell2location (and optionally RCTD) on FEAST-simulated pseudo-bulk
# spots for all slice/resolution combinations. Skips already-completed outputs.
#
# Usage:
#   bash 05_deconvolution/run_methods.sh
#
# Environment variables:
#   FORCE=1            Re-run even if outputs exist
#   SLICES="007 050 100"   Override default slices
#   RESOLUTIONS="0.1 0.25"  Override default resolutions
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---- Paths ----
CELL2LOC_ENV="/maiziezhou_lab2/yiru/envs/cell2loc_env"
CELL2LOC_PYTHON="${CELL2LOC_ENV}/bin/python"
CELL2LOC_SCRIPT="${SCRIPT_DIR}/methods/run_cell2loc.py"

REF_DATA_DIR="/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad"
SIM_DIR="${PROJECT_ROOT}/outputs/05_deconvolution"
C2L_DIR="${PROJECT_ROOT}/outputs/05_deconvolution/cell2location"

CELL_TYPE_KEY="cell_type"
SEED=2026

# ---- Config ----
FORCE="${FORCE:-0}"
SLICES=(${SLICES:-"007" "050" "100"})
RESOLUTIONS=(${RESOLUTIONS:-"0.1" "0.25"})
MAX_EPOCHS="${MAX_EPOCHS:-25000}"

mkdir -p "${C2L_DIR}"

echo "=========================================="
echo "Deconvolution Methods Runner"
echo "=========================================="
echo "Slices:      ${SLICES[*]}"
echo "Resolutions: ${RESOLUTIONS[*]}"
echo "Max epochs:  ${MAX_EPOCHS}"
echo "Force:       ${FORCE}"
echo ""

total=$(( ${#SLICES[@]} * ${#RESOLUTIONS[@]} ))
count=0
failed=0
skipped=0

for slice in "${SLICES[@]}"; do
    ref_path="${REF_DATA_DIR}/Zhuang-ABCA-1.${slice}.h5ad"
    if [[ ! -f "${ref_path}" ]]; then
        echo "ERROR: Reference not found for slice ${slice}: ${ref_path}"
        failed=$((failed + ${#RESOLUTIONS[@]}))
        continue
    fi

    for res in "${RESOLUTIONS[@]}"; do
        count=$((count + 1))
        sim_path="${SIM_DIR}/${slice}/resolution_${res}.h5ad"
        out_path="${C2L_DIR}/${slice}/resolution_${res}_proportions.csv"

        if [[ ! -f "${sim_path}" ]]; then
            echo "[${count}/${total}] SKIP ${slice}/resolution_${res} — simulation missing: ${sim_path}"
            failed=$((failed + 1))
            continue
        fi

        if [[ -f "${out_path}" && "${FORCE}" != "1" ]]; then
            echo "[${count}/${total}] SKIP ${slice}/resolution_${res} — output exists"
            skipped=$((skipped + 1))
            continue
        fi

        overwrite_flag=""
        [[ "${FORCE}" == "1" ]] && overwrite_flag="--overwrite"

        echo ""
        echo "=== [${count}/${total}] Cell2location: ${slice} / resolution_${res} ==="
        echo "  Input:     ${sim_path}"
        echo "  Reference: ${ref_path}"
        echo "  Output:    ${out_path}"
        echo ""

        if ${CELL2LOC_PYTHON} "${CELL2LOC_SCRIPT}" \
            --input "${sim_path}" \
            --reference "${ref_path}" \
            --cell-type-key "${CELL_TYPE_KEY}" \
            --output "${out_path}" \
            --seed "${SEED}" \
            --max-epochs "${MAX_EPOCHS}" \
            ${overwrite_flag}; then
            echo "  => OK"
        else
            echo "  => FAILED (exit code $?)"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Done: ${total} total, $((total - failed - skipped)) ok, ${skipped} skipped, ${failed} failed"
echo "=========================================="

exit $(( failed > 0 ? 1 : 0 ))
