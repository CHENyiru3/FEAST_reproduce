#!/usr/bin/env bash
set -euo pipefail

ROOT="/maiziezhou_lab2/yiru/FEAST_experiments"
CONDA_ENV="/maiziezhou_lab2/yiru/envs/feast-py311-conda"
PYTHON="${CONDA_ENV}/bin/python"
FEAST_SRC="/maiziezhou_lab2/yiru/FEAST/src"
LOG_DIR="${ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/rerun_04_05_06_08_${TIMESTAMP}.log"
SKIP_FEAST_OPT_WAIT="${SKIP_FEAST_OPT_WAIT:-0}"
SKIP_02_03_WAIT="${SKIP_02_03_WAIT:-0}"

mkdir -p "${LOG_DIR}" /tmp/matplotlib
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="${FEAST_SRC}:${PYTHONPATH:-}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_tmux() {
  local session="$1"
  while tmux has-session -t "${session}" 2>/dev/null; do
    log "Waiting for tmux session '${session}' to finish..."
    sleep 300
  done
}

run_04() {
  log "Starting 04_2d_conditional_transfer rerun"
  cd "${ROOT}/04_2d_conditional_transfer"
  for ar in 0.0 0.1 0.2 0.3 0.5; do
    log "04 cross_slice assignment_randomness=${ar}"
    "${PYTHON}" run.py \
      --mode cross_slice \
      --data-dir /maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad \
      --annotation-key ground_truth \
      --directions 151675:151676 151676:151675 \
      --n-top-genes 17924 \
      --assignment-randomness "${ar}" \
      --seed 2026 \
      --output-dir "results/ar_${ar}" \
      --overwrite \
      --ppf-method interp \
      --beta-n-jobs 1 \
      --beta-early-stopping-patience 2 \
      --assignment-solver scipy \
      --assignment-blocks \
      --assignment-block-multiplier 8 \
      --convert-n-jobs 1
  done
  log "Finished 04_2d_conditional_transfer rerun"
}

run_05() {
  log "Starting 05_3d_stack rerun"
  cd "${ROOT}/05_3d_stack"
  "${PYTHON}" run.py \
    --data-dir /maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad \
    --densities 3 5 10 \
    --label-key class \
    --seed 2026 \
    --output-dir outputs \
    --ppf-method interp \
    --beta-n-jobs 1 \
    --beta-early-stopping-patience 2 \
    --assignment-solver scipy \
    --assignment-blocks \
    --assignment-block-multiplier 8 \
    --convert-n-jobs 1
  log "Finished 05_3d_stack rerun"
}

run_06() {
  log "Starting 06_3d_transfer rerun"
  cd "${ROOT}/06_3d_transfer"
  bash run_pipeline.sh
  log "Finished 06_3d_transfer rerun"
}

run_08() {
  log "Starting 08_batch_effect_removal rerun"

  log "08 effect_verification assessment"
  cd "${ROOT}/08_batch_effect_removal/effect_verification"
  "${PYTHON}" run_assessment.py --config config.yaml

  log "08 effect_simulation ladder"
  cd "${ROOT}/08_batch_effect_removal/effect_simulation"
  "${PYTHON}" run_simulation.py --config config.yaml
  latest="$(ls -td outputs/*/ | head -1)"
  test -f "${latest}/manifest.csv"
  "${PYTHON}" evaluate_simulation.py "${latest}"

  log "Finished 08_batch_effect_removal rerun"
}

log "Queue log: ${LOG_FILE}"
log "FEAST commit: $(git -C /maiziezhou_lab2/yiru/FEAST rev-parse HEAD)"
if [[ "${SKIP_FEAST_OPT_WAIT}" == "1" ]]; then
  log "Skipping wait for feast_opt_rerun because SKIP_FEAST_OPT_WAIT=1"
else
  wait_for_tmux feast_opt_rerun
fi
if [[ "${SKIP_02_03_WAIT}" == "1" ]]; then
  log "Skipping wait for feast_02_03_queue because SKIP_02_03_WAIT=1"
else
  wait_for_tmux feast_02_03_queue
fi

run_04
run_05
run_06
run_08

log "All requested reruns finished"
