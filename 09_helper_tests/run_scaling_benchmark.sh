#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# FEAST Scaling Benchmark — gene-level vs spot-level performance curves
#
# Design:
#   Gene scaling:  OpenST_005    (all 47,217 spots, genes: 2k→5k→10k→20k→full 28,943)
#   Spot scaling:  Xenium_LN     (all 523 genes, spots: 5k→10k→25k→50k→100k→200k→full 377,957)
#   Anchor:        DLPFC_151675  (full 3,565 spots × 18,641 genes)
#
# Uses optimized FEAST settings (matches run_feast_optimized.sh):
#   ppf_method=interp, beta_n_jobs=4, assignment_blocks=True, convert_n_jobs=4
#
# Tracks: wall time (via /usr/bin/time) + max RSS memory
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

FEAST_ENV="${FEAST_ENV:-/maiziezhou_lab2/yiru/envs/feast-py311-conda}"
PYTHON="${PYTHON:-${FEAST_ENV}/bin/python}"
SEED="${SEED:-2026}"
PREP_SEED="${PREP_SEED:-42}"

EXPER_DATA="/maiziezhou_lab2/yiru/FEAST_experiments/00_simulator_benchmark/exper_data"
SIM_SCRIPT="/maiziezhou_lab2/yiru/FEAST_experiments/00_simulator_benchmark/scripts/ParameterCloud_sim.py"

INPUTS_DIR="${SCRIPT_DIR}/inputs"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

RUN_ID="${RUN_ID:-scaling_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"
TIMING_CSV="${RESULTS_DIR}/${RUN_ID}_timing.csv"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

mkdir -p "${INPUTS_DIR}" "${OUTPUTS_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

# ---------------------------------------------------------------------------
# Stage 0: Prepare downsampled inputs
# ---------------------------------------------------------------------------
log "==== STAGE 0: Prepare downsampled inputs ===="
log "Prep seed: ${PREP_SEED}"

"${PYTHON}" - "${INPUTS_DIR}" "${EXPER_DATA}" "${PREP_SEED}" <<'PY'
import sys
from pathlib import Path
import anndata as ad
import numpy as np

inputs_dir = Path(sys.argv[1])
exper_data = Path(sys.argv[2])
prep_seed = int(sys.argv[3])
rng = np.random.default_rng(prep_seed)

# --- OpenST_005: gene downsampling (all 47,217 spots, no spot subsampling) ---
openst = ad.read_h5ad(str(exper_data / "OpenST_005.h5ad"))
n_spots, n_genes = openst.n_obs, openst.n_vars
print(f"OpenST_005 raw: {n_spots} spots x {n_genes} genes")

gene_levels = [2000, 5000, 10000, 20000, n_genes]
for g in gene_levels:
    label = f"openst_g{g:05d}_s{n_spots:05d}"
    out_path = inputs_dir / f"{label}.h5ad"
    if out_path.exists():
        print(f"  SKIP {label} — exists")
        continue
    if g >= n_genes:
        openst.write_h5ad(str(out_path), compression="gzip")
    else:
        # Pick top-g highly variable genes across all spots
        X = openst.X.toarray() if hasattr(openst.X, 'toarray') else np.asarray(openst.X)
        gene_vars = np.var(X, axis=0)
        top_idx = np.argsort(gene_vars)[-g:]
        top_idx.sort()
        sub = openst[:, top_idx].copy()
        sub.write_h5ad(str(out_path), compression="gzip")
    print(f"  WROTE {label}: {n_spots}s x {min(g, n_genes)}g")

# --- Xenium_LymphNode: spot downsampling (all 523 genes) ---
xenium = ad.read_h5ad(str(exper_data / "Xenium_LymphNode.h5ad"))
n_spots, n_genes = xenium.n_obs, xenium.n_vars
print(f"Xenium_LymphNode raw: {n_spots} spots x {n_genes} genes")

spot_levels = [5000, 10000, 25000, 50000, 100000, 200000, n_spots]
for s in spot_levels:
    label = f"xenium_s{s:06d}_g{n_genes:04d}"
    out_path = inputs_dir / f"{label}.h5ad"
    if out_path.exists():
        print(f"  SKIP {label} — exists")
        continue
    if s >= n_spots:
        xenium.write_h5ad(str(out_path), compression="gzip")
    else:
        spot_idx = rng.choice(n_spots, size=s, replace=False)
        spot_idx.sort()
        sub = xenium[spot_idx].copy()
        sub.write_h5ad(str(out_path), compression="gzip")
    print(f"  WROTE {label}: {min(s, n_spots)}s x {n_genes}g")

# --- DLPFC_151675 anchor (full) ---
dlpfc_path = inputs_dir / "dlpfc_151675_full.h5ad"
if not dlpfc_path.exists():
    dlpfc = ad.read_h5ad(str(exper_data / "DLPFC_151675.h5ad"))
    dlpfc.write_h5ad(str(dlpfc_path), compression="gzip")
    print(f"  WROTE dlpfc_151675_full: {dlpfc.n_obs}s x {dlpfc.n_vars}g")
else:
    print(f"  SKIP dlpfc_151675_full — exists")

print("\nAll inputs prepared.")
PY

# ---------------------------------------------------------------------------
# Discover inputs
# ---------------------------------------------------------------------------
mapfile -t INPUT_FILES < <(find "${INPUTS_DIR}" -maxdepth 1 -name "*.h5ad" -printf "%f\n" | sort)
log "Input files: ${#INPUT_FILES[@]}"
for f in "${INPUT_FILES[@]}"; do log "  ${f}"; done

# ---------------------------------------------------------------------------
# Stage 1: Run FEAST on each input × 2 spatial modes
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 1: FEAST scaling runs ===="

# Write timing CSV header
echo "config,spatial_mode,n_spots,n_genes,wall_seconds,max_rss_mb,status" > "${TIMING_CSV}"

for h5ad_file in "${INPUT_FILES[@]}"; do
    config="${h5ad_file%.h5ad}"
    input_path="${INPUTS_DIR}/${h5ad_file}"

    for mode_label in "FEAST_Rank:reference_rank" "FEAST_OT_Spatial:ot_spatial"; do
        label="${mode_label%%:*}"
        spatial="${mode_label##*:}"
        out_dir="${OUTPUTS_DIR}/${label}"
        output_path="${out_dir}/${config}.h5ad"
        run_log="${LOG_DIR}/${RUN_ID}_${label}_${config}.log"
        time_log="${LOG_DIR}/${RUN_ID}_${label}_${config}.time"
        mkdir -p "${out_dir}"

        if [[ -f "${output_path}" && "${FORCE:-0}" != "1" ]]; then
            log "SKIP ${label}:${config} — exists"
            continue
        fi

        # Read dimensions for logging
        read n_spots n_genes < <("${PYTHON}" -c "
import anndata as ad
a = ad.read_h5ad('${input_path}')
print(a.n_obs, a.n_vars)
")
        log "RUN  ${label}:${config} (${n_spots}s × ${n_genes}g)"

        # /usr/bin/time -f '%e\n%M': line 1 = wall seconds, line 2 = max RSS KB
        if "${TIME_BIN}" -o "${time_log}" -f '%e\n%M' \
            "${PYTHON}" "${SIM_SCRIPT}" \
            --input "${input_path}" \
            --output "${output_path}" \
            --parameter-mode hungarian \
            --spatial-mode "${spatial}" \
            --assignment-method hybrid \
            --annotation-key auto \
            --seed "${SEED}" \
            --ppf-method interp \
            --beta-n-jobs 4 \
            --beta-early-stopping-patience 2 \
            --assignment-solver scipy \
            --assignment-blocks \
            --assignment-block-multiplier 8 \
            --convert-n-jobs 4 > "${run_log}" 2>&1; then
            status="OK"
        else
            status="FAIL"
        fi

        # Parse wall time (line 1) and max RSS KB (line 2) from time output
        wall=$(head -1 "${time_log}" 2>/dev/null || echo "0")
        rss_kb=$(sed -n '2p' "${time_log}" 2>/dev/null || echo "0")
        rss_mb=$(echo "scale=1; ${rss_kb} / 1024" | bc)

        echo "${config},${label},${n_spots},${n_genes},${wall},${rss_mb},${status}" >> "${TIMING_CSV}"
        log "${status}  ${label}:${config} — ${wall}s  max_rss=${rss_mb}MB"
    done
done

# ---------------------------------------------------------------------------
# Stage 2: Summary
# ---------------------------------------------------------------------------
log ""
log "==== STAGE 2: Summary ===="

"${PYTHON}" "${TIMING_CSV}" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print(f"\nTotal runs: {len(df)}")
print(f"  OK:    {(df['status']=='OK').sum()}")
print(f"  FAIL:  {(df['status']=='FAIL').sum()}")

ok = df[df['status'] == 'OK']
if len(ok):
    print(f"  Total wall time: {ok['wall_seconds'].sum():.0f}s ({ok['wall_seconds'].sum()/3600:.1f}h)")
    print(f"  Peak RSS range:  {ok['max_rss_mb'].min():.0f} – {ok['max_rss_mb'].max():.0f} MB")

    print("\nPer-config timing (OK runs):")
    summary = ok.groupby(['config', 'spatial_mode']).agg(
        spots=('n_spots', 'first'),
        genes=('n_genes', 'first'),
        wall_s=('wall_seconds', 'mean'),
        rss_mb=('max_rss_mb', 'mean'),
    ).round(1)
    print(summary.to_string())

    # Gene scaling analysis
    print("\n--- Gene scaling (OpenST, all 47,217 spots) ---")
    gene_runs = ok[ok['config'].str.startswith('openst')]
    for mode in ['FEAST_Rank', 'FEAST_OT_Spatial']:
        sub = gene_runs[gene_runs['spatial_mode'] == mode].sort_values('n_genes')
        if len(sub):
            print(f"\n{mode}:")
            for _, r in sub.iterrows():
                print(f"  {int(r.n_genes):>6}g → {r.wall_seconds:>8.1f}s  {r.max_rss_mb:>8.0f} MB")

    # Spot scaling analysis
    print("\n--- Spot scaling (Xenium, 523 genes fixed) ---")
    spot_runs = ok[ok['config'].str.startswith('xenium')]
    for mode in ['FEAST_Rank', 'FEAST_OT_Spatial']:
        sub = spot_runs[spot_runs['spatial_mode'] == mode].sort_values('n_spots')
        if len(sub):
            print(f"\n{mode}:")
            for _, r in sub.iterrows():
                print(f"  {int(r.n_spots):>7}s → {r.wall_seconds:>8.1f}s  {r.max_rss_mb:>8.0f} MB")
PY

log ""
log "==== DONE ===="
log "Timing CSV: ${TIMING_CSV}"
log "Log:        ${RUN_LOG}"
