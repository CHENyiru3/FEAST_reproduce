#!/usr/bin/env bash
set -euo pipefail

FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
SPACEL_ENV="/maiziezhou_lab2/yiru/envs/SPACEL"
SIM_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/outputs/04_alignment_fixed"
METHOD_DIR="${SIM_DIR}/methods"
ANGLES=(1 5 10 30 45 60)
SCRIPT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/02_alignment"
SCRIPTS_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/scripts/methods"
INPUT="/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad/151675.h5ad"
CONFIG="/maiziezhou_lab2/yiru/FEAST_experiments/configs/alignment_test_baseline.yaml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =========================================================================
# Stage 1: Simulation
# =========================================================================
log "=== STAGE 1: Simulation (baseline, 6 angles, sequencing mode) ==="
rm -rf "${SIM_DIR}"/*
${FEAST_ENV}/bin/python -u "${SCRIPT_DIR}/run_simulation.py" \
  --input "${INPUT}" \
  --output-dir "${SIM_DIR}" \
  --config "${CONFIG}" \
  --angles 1 5 10 30 45 60 \
  --seed 2026
log "Simulation done. Files:"
ls -lh "${SIM_DIR}/"
echo "---"
cat "${SIM_DIR}/simulation_manifest.csv"

# =========================================================================
# Stage 2: Run alignment methods (all 6 angles, parallel per angle)
# =========================================================================
log "=== STAGE 2: Alignment Methods ==="

run_spateo() {
    local angle=$1
    local out="${METHOD_DIR}/baseline/spateo/angle_${angle}"
    mkdir -p "${out}"
    log "  RUN  spateo angle_${angle}"
    ${FEAST_ENV}/bin/python "${SCRIPTS_DIR}/run_spateo.py" \
        --reference "${SIM_DIR}/reference.h5ad" \
        --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
        --output-dir "${out}" \
        &> "${out}/run.log"
    log "  DONE spateo angle_${angle}"
}

run_spacel() {
    local angle=$1
    local out="${METHOD_DIR}/baseline/spacel/angle_${angle}"
    mkdir -p "${out}"
    log "  RUN  spacel angle_${angle}"
    conda run -p "${SPACEL_ENV}" python "${SCRIPTS_DIR}/run_spacel.py" \
        --reference "${SIM_DIR}/reference.h5ad" \
        --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
        --output-dir "${out}" \
        --layer-key sce.layer_guess \
        &> "${out}/run.log"
    log "  DONE spacel angle_${angle}"
}

for angle in "${ANGLES[@]}"; do
    run_spateo "${angle}" &
    run_spacel "${angle}" &
done
wait
log "All methods done."

# =========================================================================
# Stage 3: Benchmark
# =========================================================================
log "=== STAGE 3: Benchmark ==="
BENCH_OUT="${SIM_DIR}/alignment_benchmark_results.csv"
${FEAST_ENV}/bin/python "${SCRIPT_DIR}/benchmark.py" \
  --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
  --methods-dir "${METHOD_DIR}" \
  --reference "${SIM_DIR}/reference.h5ad" \
  --output "${BENCH_OUT}"
log "Benchmark done."
cat "${BENCH_OUT}"

# =========================================================================
# Stage 4: Regenerate metrics CSV & plot
# =========================================================================
log "=== STAGE 4: Regenerate metrics CSV & plot ==="
VIZ_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/07_Visualization"

# Point prepare script at the new benchmark CSV
${FEAST_ENV}/bin/python -c "
import pandas as pd
from pathlib import Path

COLUMN_MAP = {
    'nn_accuracy': 'Accuracy',
    'morpho_precision': 'Precision',
    'morpho_f1': 'F1 Score',
    'ge_correlation': 'Gene Expression Correlation',
    'nn_region_accuracy': 'Adjusted Region Accuracy',
}

df = pd.read_csv('${BENCH_OUT}')
df = df[df['alteration_id'] == 'baseline'].copy()
df['method'] = df['method'].str.capitalize()

records = []
for _, row in df.iterrows():
    for src, display in COLUMN_MAP.items():
        s = row[src]
        if not pd.isna(s):
            records.append({'method': row['method'], 'angle': row['angle'], 'metric': display, 'score': s})

out = pd.DataFrame(records)
out_path = Path('${VIZ_DIR}') / 'alignment_alteration_metrics.csv'
out.to_csv(out_path, index=False)
print(f'Wrote {len(out)} rows to {out_path}')
print(out.groupby('metric').agg(n=('score','count'), min=('score','min'), max=('score','max')).to_string())
"

${FEAST_ENV}/bin/python "${VIZ_DIR}/visualization_alignment.py"
log "Visualization done."

log "=== PIPELINE COMPLETE ==="
log "CSV: ${SIM_DIR}/alignment_benchmark_results.csv"
log "Figures: /maiziezhou_lab2/yiru/Reproduce/Visualization/figures/alignment_alteration_curve_panel.*"
