#!/usr/bin/env bash
set -euo pipefail

RUN_ID="d92_e0_continuous_session_k10new5_20260817_v1"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
RUNTIME_ROOT="${RUN_ROOT}/runtime"
OUTPUT_ROOT="${RUN_ROOT}/output"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
ARCHIVE_PATH="${RUN_ROOT}/release/d92_e0_continuous_session_v1.tar.gz"
MANIFEST_PATH="${OUTPUT_ROOT}/matrix_manifest.json"
RUNNER="${RUNTIME_ROOT}/code/scripts/run_d92_e0_continuous_session.py"
CONFIG="${RUNTIME_ROOT}/configs/stage2_d92_e0_continuous_session_v1.json"

mkdir -p "${RUNTIME_ROOT}" "${RUN_ROOT}/release" "${LOG_ROOT}"
tar -xzf "${ARCHIVE_PATH}" -C "${RUNTIME_ROOT}" --strip-components=1
export PYTHONPATH="${RUNTIME_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${RUNNER}" prepare --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" \
  >"${LOG_ROOT}/prepare.out" 2>&1
"${PYTHON}" "${RUNNER}" prepare-deltas --manifest "${MANIFEST_PATH}" \
  >"${LOG_ROOT}/prepare-deltas.out" 2>&1

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${RUNNER}" smoke --manifest "${MANIFEST_PATH}" --device cuda:0 \
  >"${LOG_ROOT}/smoke_gpu0.out" 2>&1

declare -a PIDS=()
declare -a JOB_IDS=(
  "rx_20_1__seed_713106__k_10__new_5__continuous_session"
  "rx_3_19__seed_713106__k_10__new_5__continuous_session"
  "rx_7_14__seed_713106__k_10__new_5__continuous_session"
  "rx_7_7__seed_713106__k_10__new_5__continuous_session"
  "rx_8_8__seed_713106__k_10__new_5__continuous_session"
)

for index in "${!JOB_IDS[@]}"; do
  job_id="${JOB_IDS[$index]}"
  CUDA_VISIBLE_DEVICES="${index}" "${PYTHON}" "${RUNNER}" run-job \
    --manifest "${MANIFEST_PATH}" --job-id "${job_id}" --device cuda:0 \
    >"${LOG_ROOT}/${job_id}.out" 2>&1 &
  PIDS+=("$!")
done

set +e
overall_rc=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}"
  rc=$?
  printf 'job_index=%s pid=%s rc=%s\n' "${index}" "${PIDS[$index]}" "${rc}" >>"${LOG_ROOT}/formal_wait_receipt.txt"
  if [[ "${rc}" -ne 0 ]]; then overall_rc="${rc}"; fi
done
set -e

"${PYTHON}" "${RUNNER}" status --manifest "${MANIFEST_PATH}" \
  >"${LOG_ROOT}/status.out" 2>&1 || status_rc=$?
status_rc="${status_rc:-0}"
if [[ "${overall_rc}" -ne 0 ]]; then exit "${overall_rc}"; fi
exit "${status_rc}"
