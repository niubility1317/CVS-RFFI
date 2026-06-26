#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}" || exit 1

RUN_ID="${RUN_ID:-riei_drift_ratio010_080_sat_20260610_165754}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"

TRAIN_RATIOS="${TRAIN_RATIOS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}"
METHODS="${METHODS:-riei_fixed_sat,drift_fixed_sat}"
GPU_IDS="${GPU_IDS:-0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-4}"
QUEUE_SLOT_POLL_SECONDS="${QUEUE_SLOT_POLL_SECONDS:-30}"

WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
PAPER_EVAL_LAST_N="${PAPER_EVAL_LAST_N:-1}"
SAT_TRAIN_AUG="${SAT_TRAIN_AUG:-1}"
SAT_EVAL="${SAT_EVAL:-1}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:-0}"

export TRAIN_RATIOS METHODS GPU_IDS WISIG_PKL
export BASELINE_EPOCHS PAPER_EVAL_LAST_N SAT_TRAIN_AUG SAT_EVAL SAT_EVAL_MAX_BATCHES
export QUEUE_SLOT_POLL_SECONDS

bash "${ROOT}/run_cvs_fixed_riei_drift_ratio_sweep.sh" \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --ratios "${TRAIN_RATIOS}" \
  --methods "${METHODS}" \
  --gpu-ids "${GPU_IDS}" \
  --max-train-per-gpu "${MAX_TRAIN_PER_GPU}" \
  "$@"
