#!/usr/bin/env bash
set -euo pipefail

ROOT="/maiziezhou_lab2/yiru/FEAST_experiments"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/rerun_02_03_$(date +%Y%m%d_%H%M%S).log"
SKIP_FEAST_OPT_WAIT="${SKIP_FEAST_OPT_WAIT:-0}"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

wait_for_tmux_session() {
    local session="$1"
    while tmux has-session -t "${session}" 2>/dev/null; do
        log "Waiting for tmux session '${session}' to finish..."
        sleep 300
    done
}

log "Queued rerun for 02_alignment and 03_deconvolution"
log "Log: ${LOG_FILE}"

if [[ "${SKIP_FEAST_OPT_WAIT}" == "1" ]]; then
    log "Skipping wait for feast_opt_rerun because SKIP_FEAST_OPT_WAIT=1"
else
    wait_for_tmux_session "feast_opt_rerun"
fi
wait_for_tmux_session "feast_cluster_rerun"

log "Starting 02_alignment rerun"
cd "${ROOT}/02_alignment"
FIG2_FORCE=1 bash run_production.sh 2>&1 | tee -a "${LOG_FILE}"
log "02_alignment rerun finished"

log "Starting 03_deconvolution rerun"
cd "${ROOT}/03_deconvolution"
FIG2_FORCE=1 bash run_resume.sh 2>&1 | tee -a "${LOG_FILE}"
log "03_deconvolution rerun finished"

log "Queued rerun complete"
