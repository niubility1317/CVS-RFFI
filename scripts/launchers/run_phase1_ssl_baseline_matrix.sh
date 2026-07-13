#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERIC_LAUNCHER="${SCRIPT_DIR}/run_cvs_baseline_queue.sh"

METHODS="${METHODS:-cvcnn_ce,riei_fd,drift}"
GPU_IDS="${GPU_IDS:-0,1,2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-713101}"
STANDARD_EPOCHS="${STANDARD_EPOCHS:-200}"
RUN_ID="${RUN_ID:-cvs_phase1_ssl_seed${SEED}}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-${REPO_ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-${REPO_ROOT}/paper_reproduction/logs/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

LABELED_RATIO="${LABELED_RATIO:-0.1}"
UNLABELED_RATIO="${UNLABELED_RATIO:-0.6}"
SOURCE_VAL_RATIO="${SOURCE_VAL_RATIO:-0.3}"

SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
SAT_TRAIN_SCENARIOS="${SAT_TRAIN_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
SAT_VIEW_PROB="${SAT_VIEW_PROB:-1.0}"
SAT_VIEW_SEED="${SAT_VIEW_SEED:-2027}"

PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-150}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.95}"
PSEUDO_MARGIN="${PSEUDO_MARGIN:-0.0}"
LAMBDA_PSEUDO="${LAMBDA_PSEUDO:-1.0}"

CONSISTENCY_START_EPOCH="${CONSISTENCY_START_EPOCH:-1}"
CONSISTENCY_TEMPERATURE="${CONSISTENCY_TEMPERATURE:-1.0}"
LAMBDA_CONSISTENCY="${LAMBDA_CONSISTENCY:-1.0}"

run_line() {
  local mode="$1"
  env \
    METHODS="${METHODS}" \
    GPU_IDS="${GPU_IDS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SEED="${SEED}" \
    STANDARD_EPOCHS="${STANDARD_EPOCHS}" \
    SSL_MODE="${mode}" \
    LABELED_RATIO="${LABELED_RATIO}" \
    UNLABELED_RATIO="${UNLABELED_RATIO}" \
    SOURCE_VAL_RATIO="${SOURCE_VAL_RATIO}" \
    SAT_EVAL=1 \
    SAT_SCENARIOS="${SAT_SCENARIOS}" \
    SAT_VIEW_AUG=1 \
    SAT_TRAIN_SCENARIOS="${SAT_TRAIN_SCENARIOS}" \
    SAT_VIEW_PROB="${SAT_VIEW_PROB}" \
    SAT_VIEW_SEED="${SAT_VIEW_SEED}" \
    PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH}" \
    PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD}" \
    PSEUDO_MARGIN="${PSEUDO_MARGIN}" \
    LAMBDA_PSEUDO="${LAMBDA_PSEUDO}" \
    CONSISTENCY_START_EPOCH="${CONSISTENCY_START_EPOCH}" \
    CONSISTENCY_TEMPERATURE="${CONSISTENCY_TEMPERATURE}" \
    LAMBDA_CONSISTENCY="${LAMBDA_CONSISTENCY}" \
    RUN_ROOT="${RUN_ROOT_BASE}/${mode}" \
    LOG_ROOT="${LOG_ROOT_BASE}/${mode}" \
    DRY_RUN="${DRY_RUN}" \
    SKIP_DONE=0 \
    bash "${GENERIC_LAUNCHER}"
}

echo "[PHASE1-SSL] run_id=${RUN_ID} methods=${METHODS} gpus=${GPU_IDS}"
echo "[PHASE1-SSL] split labeled=${LABELED_RATIO} unlabeled=${UNLABELED_RATIO} val=${SOURCE_VAL_RATIO}"
echo "[PHASE1-SSL] routes=pseudo_label,augmentation_consistency sat_view_aug=1 scenarios=${SAT_TRAIN_SCENARIOS}"

if [ "${DRY_RUN}" = "1" ]; then
  run_line pseudo_label
  run_line augmentation_consistency
  exit 0
fi

mkdir -p "${RUN_ROOT_BASE}" "${LOG_ROOT_BASE}"
run_line pseudo_label &
pseudo_pid=$!
run_line augmentation_consistency &
consistency_pid=$!
echo "[PHASE1-SSL] pseudo_scheduler_pid=${pseudo_pid} consistency_scheduler_pid=${consistency_pid}"

status=0
wait "${pseudo_pid}" || status=$?
wait "${consistency_pid}" || status=$?
exit "${status}"
