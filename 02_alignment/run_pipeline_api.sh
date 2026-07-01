#!/usr/bin/env bash
# Full pipeline using FEAST built-in API for simulation
set -euo pipefail

FEAST_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
SPACEL_ENV="/maiziezhou_lab2/yiru/envs/SPACEL"
SIM_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/outputs/04_alignment_api"
METHOD_DIR="${SIM_DIR}/methods"
ANGLES=(1 5 10 30 45 60)
SCRIPT_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/02_alignment"
SCRIPTS_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/scripts/methods"
INPUT="/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad/151675.h5ad"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =========================================================================
# Stage 1: Simulation via FEAST API
# =========================================================================
log "=== STAGE 1: Simulation (FEAST API, sequencing, filter_edge_spots) ==="
rm -rf "${SIM_DIR}"
${FEAST_ENV}/bin/python -u "${SCRIPT_DIR}/run_simulation_api.py" \
  --input "${INPUT}" \
  --output-dir "${SIM_DIR}" \
  --seed 2026

log "Simulation done. Spot counts:"
cat "${SIM_DIR}/simulation_manifest.csv" | column -t -s ','

# =========================================================================
# Stage 2: Run Spateo + Spacel on all 6 angles (parallel)
# =========================================================================
log "=== STAGE 2: Alignment Methods ==="

for angle in "${ANGLES[@]}"; do
    out="${METHOD_DIR}/baseline/spateo/angle_${angle}"
    mkdir -p "${out}"
    log "  RUN  spateo angle_${angle}"
    ${FEAST_ENV}/bin/python "${SCRIPTS_DIR}/run_spateo.py" \
        --reference "${SIM_DIR}/reference.h5ad" \
        --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
        --output-dir "${out}" &> "${out}/run.log" &
done

for angle in "${ANGLES[@]}"; do
    out="${METHOD_DIR}/baseline/spacel/angle_${angle}"
    mkdir -p "${out}"
    log "  RUN  spacel angle_${angle}"
    conda run -p "${SPACEL_ENV}" python "${SCRIPTS_DIR}/run_spacel.py" \
        --reference "${SIM_DIR}/reference.h5ad" \
        --moving "${SIM_DIR}/baseline_rotated_${angle}.h5ad" \
        --output-dir "${out}" \
        --layer-key sce.layer_guess &> "${out}/run.log" &
done

wait
log "Methods done. Checking status:"
for angle in "${ANGLES[@]}"; do
    for m in spateo spacel; do
        d="${METHOD_DIR}/baseline/${m}/angle_${angle}"
        if [[ -f "${d}/aligned_coordinates.csv" ]]; then
            log "  OK    ${m} angle_${angle}"
        else
            log "  FAIL  ${m} angle_${angle} — $(tail -1 ${d}/run.log 2>/dev/null | cut -c1-80)"
        fi
    done
done

# =========================================================================
# Stage 3: Benchmark
# =========================================================================
log "=== STAGE 3: Benchmark ==="
${FEAST_ENV}/bin/python "${SCRIPT_DIR}/benchmark.py" \
  --simulation-manifest "${SIM_DIR}/simulation_manifest.csv" \
  --methods-dir "${METHOD_DIR}" \
  --reference "${SIM_DIR}/reference.h5ad" \
  --output "${SIM_DIR}/alignment_benchmark_results.csv"
cat "${SIM_DIR}/alignment_benchmark_results.csv"

# =========================================================================
# Stage 4: Visualization
# =========================================================================
log "=== STAGE 4: Regenerate metrics CSV & plot ==="
VIZ_DIR="/maiziezhou_lab2/yiru/FEAST_experiments/07_Visualization"

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

df = pd.read_csv('${SIM_DIR}/alignment_benchmark_results.csv')
df = df[df['alteration_id'] == 'baseline'].copy()
df['method'] = df['method'].str.capitalize()

records = []
for _, row in df.iterrows():
    for src, display in COLUMN_MAP.items():
        s = row[src]
        if not pd.isna(s):
            records.append({'method': row['method'], 'angle': row['angle'], 'metric': display, 'score': s})

out = pd.DataFrame(records)
out.to_csv(Path('${VIZ_DIR}') / 'alignment_alteration_metrics.csv', index=False)
print(f'Wrote {len(out)} rows')
for m in out['metric'].unique():
    sub = out[out['metric'] == m]
    print(f'  {m}: {len(sub)} rows [{sub[\"score\"].min():.4f}, {sub[\"score\"].max():.4f}]')
"

${FEAST_ENV}/bin/python "${VIZ_DIR}/visualization_alignment.py"

log "=== PIPELINE COMPLETE ==="
log "CSV: ${SIM_DIR}/alignment_benchmark_results.csv"
log "Fig: /maiziezhou_lab2/yiru/Reproduce/Visualization/figures/alignment_alteration_curve_panel.*"
