#!/bin/bash
# run_pipeline.sh — GSE269617 → DevCCF 3D transfer pipeline
# Phases 1–3 run sequentially.  Each phase stops on first error.
set -euo pipefail

ROOT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/06_3d_transfer"
CONDA_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
DATA_DIR="/maiziezhou_lab2/yiru/Datasets/Processed"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$ROOT_DIR/logs/transfer_3d_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  GSE269617 → DevCCF 3D Transfer Pipeline"
echo "  Started: $(date)"
echo "  Log:     $LOG_FILE"
echo "============================================================"

cd "$ROOT_DIR"

# ------------------------------------------------------------------
# Phase 1: Extract full blueprints (all z-levels, z-step=1)
# ------------------------------------------------------------------
echo ""
echo "=== Phase 1: Blueprint extraction ==="
echo ""

echo "[1a] Extracting E15.5 blueprints..."
conda run -p "$CONDA_ENV" python scripts/extract_devccf_blueprints.py \
    --volume "$DATA_DIR/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/E15.5_broad_region_annotations.nii.gz" \
    --region-schema "$DATA_DIR/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv" \
    --age E15.5 \
    --output outputs/E15.5_blueprints_full.json \
    --mask-other
echo "[1a] E15.5 blueprints done."

echo ""
echo "[1b] Extracting E18.5 blueprints..."
conda run -p "$CONDA_ENV" python scripts/extract_devccf_blueprints.py \
    --volume "$DATA_DIR/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/E18.5_broad_region_annotations.nii.gz" \
    --region-schema "$DATA_DIR/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv" \
    --age E18.5 \
    --output outputs/E18.5_blueprints_full.json \
    --mask-other
echo "[1b] E18.5 blueprints done."

# ------------------------------------------------------------------
# Phase 2: Transfer — Test run first, then full runs
# ------------------------------------------------------------------
echo ""
echo "=== Phase 2: Transfer (test run) ==="
echo ""

echo "[2a] Test run: E14M_1 → E15.5 (few z-levels)..."
conda run -p "$CONDA_ENV" python scripts/transfer_to_devccf.py \
    --references "$DATA_DIR/GSE269617/h5ad_region_annotated/" \
    --reference-pattern "E14M_1" \
    --blueprints outputs/E15.5_blueprints_full.json \
    --label-key region \
    --output-dir outputs/E15.5_test/ \
    --seed 2026 \
    --z-subsample 20 \
    --limit 5
echo "[2a] Test run complete."

echo ""
echo "=== Phase 2: Transfer (full E15.5 run) ==="
echo ""

echo "[2b] Full run: E14* → E15.5 (all z-levels)..."
conda run -p "$CONDA_ENV" python scripts/transfer_to_devccf.py \
    --references "$DATA_DIR/GSE269617/h5ad_region_annotated/" \
    --reference-pattern "E14*" \
    --blueprints outputs/E15.5_blueprints_full.json \
    --label-key region \
    --output-dir outputs/E15.5_full/ \
    --seed 2026
echo "[2b] E15.5 full run complete."

echo ""
echo "=== Phase 2: Transfer (full E18.5 run) ==="
echo ""

echo "[2c] Full run: E18M* → E18.5 (all z-levels)..."
conda run -p "$CONDA_ENV" python scripts/transfer_to_devccf.py \
    --references "$DATA_DIR/GSE269617/h5ad_region_annotated/" \
    --reference-pattern "E18M*" \
    --blueprints outputs/E18.5_blueprints_full.json \
    --label-key region \
    --output-dir outputs/E18.5_full/ \
    --seed 2026
echo "[2c] E18.5 full run complete."

# ------------------------------------------------------------------
# Phase 3: Spot smoothing (z-regularization)
# ------------------------------------------------------------------
echo ""
echo "=== Phase 3: Spot smoothing ==="
echo ""

echo "[3a] Smoothing E15.5 slices..."
conda run -p "$CONDA_ENV" python scripts/z_regularize.py \
    --input-dir outputs/E15.5_full/ \
    --output-dir outputs/E15.5_smoothed/ \
    --sigma 1.0
echo "[3a] E15.5 smoothing done."

echo ""
echo "[3b] Smoothing E18.5 slices..."
conda run -p "$CONDA_ENV" python scripts/z_regularize.py \
    --input-dir outputs/E18.5_full/ \
    --output-dir outputs/E18.5_smoothed/ \
    --sigma 1.0
echo "[3b] E18.5 smoothing done."

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"
echo "ALL DONE"
