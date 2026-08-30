#!/usr/bin/env bash
set -euo pipefail

RUN_ID="phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r1"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
RELEASE_ROOT="${PROJECT_ROOT}/releases/phase1_bicad_xr_quick24_20260831_r1"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs"
DISPATCH_LOG="${LOG_ROOT}/${RUN_ID}.dispatcher.log"
PID_FILE="${LOG_ROOT}/${RUN_ID}.dispatcher.pid"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
WISIG_PKL="${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl"
LAUNCHER="${RELEASE_ROOT}/code/scripts/launch_phase1_bicad_xr_matrix_20260830.py"

test -x "${PYTHON}"
test -f "${WISIG_PKL}"
test -f "${LAUNCHER}"
test ! -e "${RUN_ROOT}"
test ! -e "${DISPATCH_LOG}"
test ! -e "${PID_FILE}"
mkdir -p "${LOG_ROOT}"

cd "${RELEASE_ROOT}"
nohup "${PYTHON}" -u "${LAUNCHER}" \
  --stage quick \
  --formal \
  --run-id "${RUN_ID}" \
  --output-root "${PROJECT_ROOT}/runs" \
  --code-root "${RELEASE_ROOT}" \
  --python "${PYTHON}" \
  --wisig-pkl "${WISIG_PKL}" \
  --max-jobs-per-gpu 3 \
  >"${DISPATCH_LOG}" 2>&1 </dev/null &
dispatcher_pid=$!
printf '%s\n' "${dispatcher_pid}" >"${PID_FILE}"
printf 'dispatcher_pid=%s\nrun_root=%s\ndispatch_log=%s\n' \
  "${dispatcher_pid}" "${RUN_ROOT}" "${DISPATCH_LOG}"
