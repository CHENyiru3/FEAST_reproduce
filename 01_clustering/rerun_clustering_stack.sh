#!/usr/bin/env bash
set -euo pipefail

# Rerun the Figure 2 clustering stack after method/preprocessing fixes.
# This intentionally starts at HVG refresh and does not rerun FEAST simulations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON_FEAST="${FEAST_ENV}/bin/python"

SIM_MANIFEST="${FIG2_SIMULATION_MANIFEST:-outputs/simulation_manifest.csv}"
METHODS="${FIG2_CLUSTER_METHODS:-STAGATE_mclust GraphST Leiden}"
BENCH_OUT="${FIG2_BENCHMARK_OUT:-outputs/benchmarks/clustering_benchmark_results.csv}"
PLOT_DIR="${FIG2_PLOT_DIR:-outputs/plots}"
RUN_PLOTS="${FIG2_RUN_PLOTS:-1}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p "${MPLCONFIGDIR}" outputs/benchmarks "${PLOT_DIR}" logs

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        log "ERROR: missing required file: $1"
        exit 1
    fi
}

require_file "${PYTHON_FEAST}"
require_file "${SIM_MANIFEST}"

log "Figure 2 clustering rerun"
log "Simulation manifest: ${SIM_MANIFEST}"
log "Methods: ${METHODS}"
log "Benchmark output: ${BENCH_OUT}"
log "Note: FEAST simulations are not rerun by this script."

log "Step 1/5: refresh HVG inputs"
"${PYTHON_FEAST}" run_hvg.py \
    --simulation-manifest "${SIM_MANIFEST}" \
    --output-dir outputs/hvg_inputs \
    --n-top-genes 3000 \
    --overwrite

"${PYTHON_FEAST}" - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/hvg_inputs/hvg_manifest.csv")
ok = int((df["status"] == "ok").sum())
total = len(df)
print(f"HVG manifest: {ok}/{total} ok")
if ok != total:
    failed = df[df["status"] != "ok"]
    print("WARNING: some HVG rows failed (proceeding without them):")
    for _, row in failed.iterrows():
        print(f"  {row['slice_id']}/{row['simulation_id']}: {row['status']}")
    print(f"Proceeding with {ok}/{total} HVG inputs.")
PY

log "Step 2/5: rerun clustering methods"
FIG2_CLUSTER_METHODS="${METHODS}" FIG2_FORCE=1 \
    bash run_pipeline.sh

log "Step 3/5: rebuild benchmark"
"${PYTHON_FEAST}" benchmark.py \
    --input-dir outputs/methods \
    --simulation-manifest "${SIM_MANIFEST}" \
    --label-column ground_truth \
    --output "${BENCH_OUT}"

log "Benchmark summary"
"${PYTHON_FEAST}" - <<PY
import pandas as pd
df = pd.read_csv("${BENCH_OUT}")
print(df["status"].value_counts(dropna=False).to_string())
ok = df[df["status"] == "ok"]
if len(ok):
    print(ok.groupby("method")["ARI"].agg(["count", "mean", "std", "min", "max"]).to_string())
n_ok = len(ok)
if n_ok < 150:
    print(f"WARNING: only {n_ok} ok benchmark rows — something may be wrong")
if (df["status"] != "ok").any():
    n_bad = (df["status"] != "ok").sum()
    print(f"WARNING: benchmark contains {n_bad} non-ok rows")
PY

if [[ "${RUN_PLOTS}" == "1" ]]; then
    log "Step 4/5: regenerate metric plots"
    "${PYTHON_FEAST}" plot_metrics.py \
        --benchmark "${BENCH_OUT}" \
        --output-dir "${PLOT_DIR}"

    log "Step 5/5: regenerate spatial maps"
    "${PYTHON_FEAST}" plot_spatial_maps.py \
        --benchmark "${BENCH_OUT}" \
        --simulation-manifest "${SIM_MANIFEST}" \
        --output-dir "${PLOT_DIR}/spatial_maps"
else
    log "Step 4/5 and 5/5: plots skipped because FIG2_RUN_PLOTS=${RUN_PLOTS}"
fi

log "Done."
