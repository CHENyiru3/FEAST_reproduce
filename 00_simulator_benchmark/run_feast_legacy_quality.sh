#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# FEAST legacy-quality rerun — exact-match mode
#
# Regenerates FEAST outputs with approximation knobs disabled, computes metrics,
# then splices FEAST results onto archived non-FEAST rows and recomputes
# composite scores on the combined table.
#
# Usage:
#   bash run_feast_legacy_quality.sh           # fresh run
#   FORCE=1 bash run_feast_legacy_quality.sh   # overwrite existing outputs
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON="${PYTHON:-${FEAST_ENV}/bin/python}"
SEED="${SEED:-2026}"

ARCHIVED_METRICS="${SCRIPT_DIR}/outputs/archived_20260623_110615/simulator_quality_metrics.csv"
EXPER_DATA="${SCRIPT_DIR}/exper_data"

RUN_ID="${RUN_ID:-feast_legacy_quality_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${SCRIPT_DIR}/outputs/${RUN_ID}"
OUT_BASE="${RUN_ROOT}/simulation_saved"
BENCH="${RUN_ROOT}/benchmarks"
LOG_DIR="${RUN_ROOT}/logs"
VIZ_DIR="${RUN_ROOT}/quality_visualization"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${OUT_BASE}" "${BENCH}" "${LOG_DIR}" "${VIZ_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

log "FEAST legacy-quality rerun (exact-match mode)"
log "Run ID:      ${RUN_ID}"
log "Archived:    ${ARCHIVED_METRICS}"
log "Python:      ${PYTHON}"
log "Seed:        ${SEED}"
log "NumPy seed:  $(NUMBA_DISABLE_JIT=1 ${PYTHON} -c 'import numpy; print(id(numpy.random.get_state()))')" || true

# ---------------------------------------------------------------------------
# Extract FEAST sample list from archived metrics
# ---------------------------------------------------------------------------
mapfile -t SAMPLES < <("${PYTHON}" - "${ARCHIVED_METRICS}" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
samples = df[df["simulator"].str.startswith("FEAST")]["sample"].drop_duplicates()
for sample in samples:
    print(sample)
PY
)

log "Samples from archived FEAST table: ${#SAMPLES[@]}"
for s in "${SAMPLES[@]}"; do log "  ${s}"; done

# ---------------------------------------------------------------------------
# Stage 1: Regenerate FEAST outputs with exact-match mode
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 1: FEAST simulation (exact-mode knobs) ===="

for sample in "${SAMPLES[@]}"; do
    input_path="${EXPER_DATA}/${sample}.h5ad"
    if [[ ! -f "${input_path}" ]]; then
        log "FAIL missing input: ${input_path}"
        exit 1
    fi

    for mode_label in "FEAST_Rank:reference_rank" "FEAST_OT_Spatial:ot_spatial"; do
        label="${mode_label%%:*}"
        spatial="${mode_label##*:}"
        out_dir="${OUT_BASE}/${label}"
        output_path="${out_dir}/${sample}.h5ad"
        sample_log="${LOG_DIR}/feast_${label}_${sample}.log"
        mkdir -p "${out_dir}"

        if [[ -f "${output_path}" && "${FORCE:-0}" != "1" ]]; then
            log "SKIP ${label}:${sample} — exists"
            continue
        fi

        log "RUN  ${label}:${sample}"
        if "${PYTHON}" scripts/ParameterCloud_sim.py \
            --input "${input_path}" \
            --output "${output_path}" \
            --parameter-mode hungarian \
            --spatial-mode "${spatial}" \
            --assignment-method hybrid \
            --annotation-key auto \
            --seed "${SEED}" \
            --ppf-method exact \
            --beta-n-jobs 1 \
            --beta-early-stopping-patience 0 \
            --assignment-solver scipy \
            --no-assignment-blocks \
            --convert-n-jobs 1 > "${sample_log}" 2>&1; then
            log "OK   ${label}:${sample}"
        else
            log "FAIL ${label}:${sample} — see ${sample_log}"
        fi
    done
done

# ---------------------------------------------------------------------------
# Stage 2: Build FEAST-only inventory
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 2: Build FEAST-only inventory ===="

"${PYTHON}" - "${ARCHIVED_METRICS}" "${OUT_BASE}" "${BENCH}/feast_inventory.csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

archive = pd.read_csv(sys.argv[1])
out_base = Path(sys.argv[2])
inventory_path = Path(sys.argv[3])

rows = []
folders = {"FEAST_Rank": "FEAST_Rank", "FEAST_OT_Spatial": "FEAST_OT_Spatial"}
for _, rec in archive[archive["simulator"].str.startswith("FEAST")][["simulator", "sample"]].iterrows():
    sim = rec["simulator"]
    sample = rec["sample"]
    path = out_base / folders[sim] / f"{sample}.h5ad"
    rows.append({
        "simulator": sim,
        "sample": sample,
        "file_path": str(path),
        "status": "matched" if path.exists() else "missing_simulation",
    })

inv = pd.DataFrame(rows)
inv.to_csv(inventory_path, index=False)
print(inv["status"].value_counts().to_string())
PY

# ---------------------------------------------------------------------------
# Stage 3: Compute FEAST atomic metrics
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 3: Compute FEAST atomic metrics ===="

"${PYTHON}" scripts/build_simulator_quality_metrics.py \
    --inventory "${BENCH}/feast_inventory.csv" \
    --exper-data "${EXPER_DATA}" \
    --metrics "${BENCH}/feast_quality_metrics_raw.csv" \
    --metadata "${BENCH}/feast_quality_metric_metadata.json" 2>&1 | tee -a "${RUN_LOG}"

# ---------------------------------------------------------------------------
# Stage 4: Merge FEAST metrics with archived non-FEAST, recompute composites
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 4: Merge and recompute composite scores ===="

"${PYTHON}" - "${ARCHIVED_METRICS}" "${BENCH}/feast_quality_metrics_raw.csv" "${BENCH}/simulator_quality_metrics.csv" <<'PY'
import sys
import numpy as np
import pandas as pd

archive_path, feast_path, output_path = sys.argv[1:4]

archive = pd.read_csv(archive_path)
feast = pd.read_csv(feast_path)

# Verify FEAST coverage
is_feast = archive["simulator"].str.startswith("FEAST")
expected = set(map(tuple, archive.loc[is_feast, ["simulator", "sample"]].to_numpy()))
got = set(map(tuple, feast[["simulator", "sample"]].to_numpy()))
missing = sorted(expected - got)
extra = sorted(got - expected)

if missing:
    print(f"WARNING: {len(missing)} FEAST metric rows missing (will be dropped)")
    for m in missing[:5]:
        print(f"  missing: {m}")
if extra:
    print(f"INFO: {len(extra)} unexpected FEAST rows (ignored)")

# Non-FEAST rows from archive (keep as-is for atomic, composite will be recomputed)
external = archive.loc[~is_feast].copy()

# FEAST rows from fresh metrics (drop composite columns from feast — they were
# normalized only across FEAST rows, so they're invalid)
atomic_cols = [
    "simulator", "sample", "identity_flag",
    "input_mean_corr", "input_variance_corr",
    "cosine_divergence", "relative_error_mean", "zero_mask_jaccard",
    "gene_zero_fraction_wasserstein", "gene_mean_wasserstein",
    "gene_variance_wasserstein", "library_size_wasserstein",
    "moran_i_correlation",
]
feast_atomic = feast[feast[["simulator", "sample"]].apply(tuple, axis=1).isin(expected)].copy()
# Only keep atomic columns (build_simulator_quality_metrics also writes composites)
feast_atomic = feast_atomic[[c for c in atomic_cols if c in feast_atomic.columns]]

# External rows: also strip composites for consistent recomputation
external_atomic = external[[c for c in atomic_cols if c in external.columns]]

# Combine and recompute composite scores on the full table
combined = pd.concat([feast_atomic, external_atomic], ignore_index=True)
m = combined.copy()
m["identity_flag"] = m["identity_flag"].astype(str).str.lower().isin(["true", "1"])

# --- Composite scoring (matching build_simulator_quality_metrics.py logic) ---
m["gene_zero_preservation"] = np.clip(
    1.0 - m["gene_zero_fraction_wasserstein"] / max(m["gene_zero_fraction_wasserstein"].quantile(0.90), 1e-10),
    0, 1,
)
m["spot_zero_preservation"] = m["zero_mask_jaccard"]
m["zero_preservation_score"] = 0.5 * m["gene_zero_preservation"] + 0.5 * m["spot_zero_preservation"]

m["mean_structure_score"] = np.clip(m["input_mean_corr"], 0, 1)
m["variance_structure_score"] = np.clip(m["input_variance_corr"], 0, 1)
lib_max = max(m["library_size_wasserstein"].quantile(0.90), 1e-10)
m["library_structure_score"] = np.clip(1.0 - m["library_size_wasserstein"] / lib_max, 0, 1)
m["statistical_structure_score"] = (
    0.40 * m["mean_structure_score"]
    + 0.35 * m["variance_structure_score"]
    + 0.25 * m["library_structure_score"]
)

m["raw_novelty"] = np.clip(1.0 - m["input_mean_corr"], 0, 1)
structure_retention = 0.5 * m["input_variance_corr"] + 0.5 * m["spot_zero_preservation"]
m["structured_novelty_score"] = m["raw_novelty"] * structure_retention
m["structured_novelty_score"] *= np.where(m["identity_flag"], 0.05, 1.0)

m["composite_quality_score"] = (
    0.30 * m["zero_preservation_score"]
    + 0.35 * m["statistical_structure_score"]
    + 0.35 * m["structured_novelty_score"]
)

# Sort: FEAST first (matching archive convention), then by simulator name
sim_order = {"FEAST_Rank": 0, "FEAST_OT_Spatial": 1}
m["_sort"] = m["simulator"].map(sim_order).fillna(2)
m = m.sort_values(["_sort", "simulator", "sample"]).drop(columns=["_sort"])

output_cols = atomic_cols + [
    "zero_preservation_score", "statistical_structure_score",
    "structured_novelty_score", "composite_quality_score",
]
m[output_cols].to_csv(output_path, index=False)

# Quick summary
feast_combined = m[m["simulator"].str.startswith("FEAST")]
print(f"\nCombined: {len(m)} rows ({len(feast_combined)} FEAST, {len(external_atomic)} external)")
print(f"FEAST composite scores:")
print(feast_combined.groupby("simulator")[
    ["input_mean_corr", "input_variance_corr", "zero_mask_jaccard",
     "moran_i_correlation", "composite_quality_score"]
].mean().round(4).to_string())
PY

# ---------------------------------------------------------------------------
# Stage 5: Build visualization
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 5: Build visualization ===="

"${PYTHON}" scripts/plot_simulator_quality.py \
    --metrics "${BENCH}/simulator_quality_metrics.csv" \
    --output-dir "${VIZ_DIR}" 2>&1 | tee -a "${RUN_LOG}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log ""
log "==== DONE ===="
log "Combined metrics: ${BENCH}/simulator_quality_metrics.csv"
log "Visualization:    ${VIZ_DIR}/"
log "Log:              ${RUN_LOG}"