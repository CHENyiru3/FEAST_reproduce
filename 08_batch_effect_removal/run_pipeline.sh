#!/usr/bin/env bash
# ============================================================================
# Batch Effect Correction Benchmark — Full Pipeline
# ============================================================================
# 1. Generate batch-effect simulations (if needed)
# 2. Run batch correction methods (GraphST, scVI, STAMP)
# 3. Compute benchmark metrics
#
# Usage:
#   ./run_pipeline.sh                          # Run all steps
#   ./run_pipeline.sh --step simulate          # Only step 1
#   ./run_pipeline.sh --step methods           # Only step 2
#   ./run_pipeline.sh --step benchmark         # Only step 3
#   ./run_pipeline.sh --method scVI            # Only scVI in step 2
#   ./run_pipeline.sh --dry-run                # Print commands only
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FEAST_ROOT="/maiziezhou_lab2/yiru/FEAST"
FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
BATCH_EVAL_ENV="/maiziezhou_lab2/yiru/miniconda3/envs/batch-eval"

cd "$SCRIPT_DIR"

DO_SIMULATE=1
DO_METHODS=1
DO_BENCHMARK=1
METHOD_FILTER=""
DRY_RUN=""
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            DO_SIMULATE=0; DO_METHODS=0; DO_BENCHMARK=0
            case "$2" in
                simulate) DO_SIMULATE=1 ;;
                methods)  DO_METHODS=1 ;;
                benchmark) DO_BENCHMARK=1 ;;
                *) echo "Unknown step: $2"; exit 1 ;;
            esac
            shift 2 ;;
        --method)
            METHOD_FILTER="--method $2"
            shift 2 ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift ;;
        --config)
            CONFIG="$2"
            shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

echo "=== FEAST Batch Effect Correction Benchmark ==="
echo "Config: $CONFIG"
echo "Steps: simulate=$DO_SIMULATE methods=$DO_METHODS benchmark=$DO_BENCHMARK"
echo

# ---------------------------------------------------------------------------
# Step 1: Generate batch-effect simulations
# ---------------------------------------------------------------------------
if [[ $DO_SIMULATE -eq 1 ]]; then
    echo "--- [1/3] Generating Batch-Effect Simulations ---"
    cd effect_simulation
    conda run -p "$FEAST_ENV" python run_simulation.py \
        --config config.yaml
    cd "$SCRIPT_DIR"
    echo "Simulation done."
    echo
fi

# ---------------------------------------------------------------------------
# Step 2: Run batch correction methods
# ---------------------------------------------------------------------------
if [[ $DO_METHODS -eq 1 ]]; then
    echo "--- [2/3] Running Batch Correction Methods ---"
    conda run -p "$FEAST_ENV" python run_methods.py \
        --config "$CONFIG" \
        $METHOD_FILTER \
        $DRY_RUN
    echo "Methods done."
    echo
fi

# ---------------------------------------------------------------------------
# Step 3: Compute benchmark metrics
# ---------------------------------------------------------------------------
if [[ $DO_BENCHMARK -eq 1 ]]; then
    echo "--- [3/3] Computing Benchmark Metrics ---"
    conda run -p "$BATCH_EVAL_ENV" python benchmark.py \
        --config "$CONFIG"
    echo "Benchmark done."
    echo
fi

echo "=== Pipeline Complete ==="
