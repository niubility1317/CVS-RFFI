#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAIT_FOR_RUN="${WAIT_FOR_RUN:-paper_repro_repaired_riei_drift_seed1337_20260714_103000}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MODE="dry-run"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --wait) MODE="wait"; shift ;;
    --wait-for-run) WAIT_FOR_RUN="$2"; shift 2 ;;
    --timeout-seconds) WAIT_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

FIXOPT_LAUNCHER="${ROOT}/code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh"
LAUNCH_CMD=(bash "${FIXOPT_LAUNCHER}" --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2)

echo "[DEFER] mode=${MODE} wait_for_run=${WAIT_FOR_RUN} timeout=${WAIT_TIMEOUT_SECONDS} poll=${POLL_SECONDS}"
printf '[DEFER] launch_cmd='; printf '%q ' "${LAUNCH_CMD[@]}"; printf '\n'
if [[ "${MODE}" == "dry-run" ]]; then exit 0; fi

[[ -x "${FIXOPT_LAUNCHER}" || -f "${FIXOPT_LAUNCHER}" ]] || { echo "ERROR: missing fixopt launcher" >&2; exit 3; }
start_epoch="$(date +%s)"
while true; do
  set +e
  active_count="$(ps -eo args= | grep -F "${WAIT_FOR_RUN}" | grep -E 'baselines\.(drift|riei_fd)\.train|queues/gpu_[0-7]\.sh|run_wisig_paper_scope_queue\.sh' | grep -v 'grep -F' | wc -l | tr -d ' ')"
  set -e
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  echo "[DEFER-WAIT] elapsed=${elapsed} active_target_processes=${active_count}"
  if (( active_count == 0 )); then break; fi
  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "ERROR: timed out waiting for ${WAIT_FOR_RUN}; no launch performed" >&2
    exit 4
  fi
  sleep "${POLL_SECONDS}"
done

echo "[DEFER-READY] target run exited; executing capacity-gated fixopt launcher"
exec "${LAUNCH_CMD[@]}"
