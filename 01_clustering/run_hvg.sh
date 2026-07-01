#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# HVG Refresh — select highly variable genes from clustering simulations
#
# Usage:
#   bash scripts/run_hvg_clustering_refresh.sh
#   FIG2_FORCE=1 bash scripts/run_hvg_clustering_refresh.sh
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON="${FEAST_ENV}/bin/python"

OVERWRITE=""
[[ "${FIG2_FORCE:-0}" == "1" ]] && OVERWRITE="--overwrite"

exec "${PYTHON}" scripts/run_hvg_clustering_refresh.py \
    --simulation-manifest outputs/simulation_manifest.csv \
    --output-dir outputs/hvg_inputs \
    --n-top-genes 3000 \
    ${OVERWRITE}
