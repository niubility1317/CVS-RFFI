#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/szu2070436088/2510044040/CV-SincNet
CODE_ROOT="${ROOT}/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v1"
RUN_ID=phase1_adv3b02_fcr_r1r8_s392002_20260902_v1
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs"
BASE_LAUNCHER="${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_fcr_20260901.sh"

[[ -f "${BASE_LAUNCHER}" ]] || { echo "missing base launcher: ${BASE_LAUNCHER}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing existing run root: ${RUN_ROOT}" >&2; exit 3; }
mkdir -p "${RUN_ROOT}/jobs" "${LOG_ROOT}"
cd "${CODE_ROOT}"

for index in 1 2 3 4 5 6 7 8; do
  row="R${index}"
  gpu="$((index - 1))"
  job_root="${RUN_ROOT}/jobs/${row}"
  log_path="${LOG_ROOT}/${RUN_ID}.${row}.launcher.out"
  pid_path="${LOG_ROOT}/${RUN_ID}.${row}.launcher.pid"
  [[ ! -e "${job_root}" ]] || { echo "refusing existing row job root: ${job_root}" >&2; exit 4; }
  nohup env \
    RUN_ID="${RUN_ID}" \
    OUTPUT_ROOT="${job_root}" \
    ROOT="${ROOT}" \
    CODE_ROOT="${CODE_ROOT}" \
    GPU="${gpu}" \
    SEED=392002 \
    bash "${BASE_LAUNCHER}" "--row=${row}" \
    > "${log_path}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\n' "${pid}" > "${pid_path}"
  printf 'LAUNCHED row=%s gpu=%s pid=%s log=%s\n' "${row}" "${gpu}" "${pid}" "${log_path}"
done
