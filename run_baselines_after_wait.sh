#!/usr/bin/env bash
set -euo pipefail

# Wait for current experiments to finish, then launch the CVS baseline queue.
#
# Recommended server usage:
#   WAIT_PATTERN=run_final_best_sgc_queue.sh \
#   WAIT_GPU_IDLE=1 \
#   GPU_IDS=0,1,2,3,4,5,6,7 \
#   nohup bash run_baselines_after_wait.sh > baseline_logs/wait_then_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# If you know the old launcher PID(s), prefer WAIT_PIDS:
#   WAIT_PIDS=12345,12346 nohup bash run_baselines_after_wait.sh > baseline_logs/wait_then_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-run_cvs_baseline_queue.sh}"
WAIT_PATTERN="${WAIT_PATTERN:-run_final_best_sgc_queue.sh}"
WAIT_PIDS="${WAIT_PIDS:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
WAIT_GPU_IDLE="${WAIT_GPU_IDLE:-0}"
GPU_IDLE_INTERVAL_SECONDS="${GPU_IDLE_INTERVAL_SECONDS:-300}"
GPU_IDLE_MIN_CHECKS="${GPU_IDLE_MIN_CHECKS:-2}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-1024}"

mkdir -p baseline_logs

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

pid_alive() {
  local pid="$1"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

wait_for_pids() {
  local pids_csv="$1"
  IFS=',' read -r -a pids <<< "${pids_csv}"
  while true; do
    local alive=()
    for raw_pid in "${pids[@]}"; do
      local pid
      pid="$(trim "${raw_pid}")"
      if pid_alive "${pid}"; then
        alive+=("${pid}")
      fi
    done
    if [ "${#alive[@]}" -eq 0 ]; then
      echo "[$(timestamp)] all WAIT_PIDS have finished."
      return 0
    fi
    echo "[$(timestamp)] waiting for PIDs: ${alive[*]}"
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
}

wait_for_pattern() {
  local pattern="$1"
  while true; do
    mapfile -t matches < <(pgrep -af "${pattern}" | grep -v "run_baselines_after_wait.sh" || true)
    if [ "${#matches[@]}" -eq 0 ]; then
      echo "[$(timestamp)] no process matched WAIT_PATTERN='${pattern}'."
      return 0
    fi
    echo "[$(timestamp)] waiting for process pattern '${pattern}':"
    printf '  %s\n' "${matches[@]}"
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
}

gpu_idle_once() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(timestamp)] nvidia-smi not found; skipping GPU idle wait."
    return 0
  fi
  local rows
  rows="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)"
  while IFS=',' read -r util mem; do
    util="$(trim "${util}")"
    mem="$(trim "${mem}")"
    if [ "${util:-0}" -gt "${GPU_UTIL_THRESHOLD}" ] || [ "${mem:-0}" -gt "${GPU_MEM_THRESHOLD_MB}" ]; then
      return 1
    fi
  done <<< "${rows}"
  return 0
}

wait_for_gpu_idle() {
  local checks=0
  while [ "${checks}" -lt "${GPU_IDLE_MIN_CHECKS}" ]; do
    if gpu_idle_once; then
      checks=$((checks + 1))
      echo "[$(timestamp)] GPU idle check ${checks}/${GPU_IDLE_MIN_CHECKS} passed."
    else
      checks=0
      echo "[$(timestamp)] GPUs still busy; waiting."
    fi
    if [ "${checks}" -lt "${GPU_IDLE_MIN_CHECKS}" ]; then
      sleep "${GPU_IDLE_INTERVAL_SECONDS}"
    fi
  done
}

echo "[$(timestamp)] wait-then-baseline launcher started."
if [ -n "$(trim "${WAIT_PIDS}")" ]; then
  wait_for_pids "${WAIT_PIDS}"
else
  wait_for_pattern "${WAIT_PATTERN}"
fi

if [ "${WAIT_GPU_IDLE}" = "1" ]; then
  wait_for_gpu_idle
fi

if [ ! -f "${BASELINE_SCRIPT}" ]; then
  echo "[$(timestamp)] missing baseline script: ${BASELINE_SCRIPT}" >&2
  exit 1
fi

echo "[$(timestamp)] launching baseline queue: ${BASELINE_SCRIPT}"
PYTHON_BIN="${PYTHON_BIN}" bash "${BASELINE_SCRIPT}"
