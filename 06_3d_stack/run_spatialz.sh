#!/usr/bin/env bash
# Run all three native SpatialZ density arms, then the matched slice evaluation.
# Usage: bash 06_3d_stack/run_spatialz.sh <completed-feast-root> <fresh-spatialz-root>
set -euo pipefail
study_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$study_dir/../.." && pwd)
feast_root=${1:?Provide the completed FEAST output root}
spatialz_root=${2:?Provide a fresh SpatialZ output root}
spatialz_python=${SPATIALZ_PYTHON:-$workspace_dir/envs/spatialz/bin/python}
evaluation_python=${EVALUATION_PYTHON:-$workspace_dir/envs/feast-study06-1.0.6-precision/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/spatialz-matplotlib}
mkdir -p "$spatialz_root"
exec > >(tee -a "$spatialz_root/workflow.log") 2>&1
date -Is
for gap in 3 5 10; do
    "$spatialz_python" -u "$study_dir/spatialz_generate.py" \
        --feast-root "$feast_root" --output-root "$spatialz_root" \
        --gap "$gap" --device cuda:0 --threads 2 --mender-workers 2
done
"$evaluation_python" -u "$study_dir/spatialz_evaluate.py" \
    --feast-root "$feast_root" --spatialz-root "$spatialz_root" \
    --output-dir "$spatialz_root/evaluation"
date -Is
