#!/usr/bin/env bash
set -uo pipefail

# 8-GPU dynamic queue launcher for FJMP v2 experiments.
#
# It keeps GPUs 0-7 busy: one experiment per GPU, and when one exits the next
# pending experiment starts automatically on the freed GPU.
#
# Examples:
#   bash scripts/run_fjmp_v2_8gpu.sh --base-ckpt runs/cvs_rffi_staged/B3b_stable_sat07_cls_only/latest_model.pth --plan A
#   bash scripts/run_fjmp_v2_8gpu.sh --base-ckpt /path/to/latest_model.pth --plan L0,L1,L2 --gpu-ids 0,1,2,3,4,5,6,7
#   bash scripts/run_fjmp_v2_8gpu.sh --base-ckpt /path/to/latest_model.pth --plan all --dry-run
#
# Useful env:
#   GPU_IDS=0,1,2,3,4,5,6,7
#   PLAN=A                  # A-F, L0-L6, comma-separated, or all
#   PYTHON_BIN=python
#   BASE_CKPT=/path/to/baseline.pth
#   RUN_ROOT=runs/fjmp_v2
#   LOG_ROOT=logs/fjmp_v2
#   STOP_ON_FAIL=0|1
#   SKIP_DONE=1|0           # skip experiments with metrics_epoch.csv
#   EXTRA_ARGS="--eval_max_batches 0 --amp true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PLAN="${PLAN:-A}"
PYTHON_BIN="${PYTHON_BIN:-}"
BASE_CKPT="${BASE_CKPT:-}"
RUN_ROOT="${RUN_ROOT:-runs/fjmp_v2}"
LOG_ROOT="${LOG_ROOT:-logs/fjmp_v2}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  sed -n '1,32p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids)
      GPU_IDS_CSV="$2"
      shift 2
      ;;
    --plan)
      PLAN="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --base-ckpt)
      BASE_CKPT="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    --no-skip-done)
      SKIP_DONE=0
      shift
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "${BASE_CKPT}" ]; then
  echo "ERROR: --base-ckpt is required, or set BASE_CKPT=/path/to/baseline.pth." >&2
  exit 2
fi

if [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: baseline checkpoint not found: ${BASE_CKPT}" >&2
  exit 2
fi

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python python3 py python.exe; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: empty GPU list." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
SCHED_LOG="${LOG_ROOT}/scheduler_$(date +%Y%m%d_%H%M%S).log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN//,/}_$(date +%Y%m%d_%H%M%S).tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

generate_queue() {
  PLAN_VALUE="${PLAN}" RUN_ROOT_VALUE="${RUN_ROOT}" BASE_CKPT_VALUE="${BASE_CKPT}" "${PYTHON_BIN}" - <<'PY' > "${QUEUE_FILE}"
import os
import shlex
from FJMP.experiment_manifest import build_experiment_manifest

plan = os.environ["PLAN_VALUE"].strip()
run_root = os.environ["RUN_ROOT_VALUE"].strip()
base_ckpt = os.environ["BASE_CKPT_VALUE"].strip()
layers = None if plan.lower() == "all" else [item.strip() for item in plan.split(",") if item.strip()]
for row in build_experiment_manifest(layers):
    exp_id = str(row["id"])
    args = row.get("args") or {}
    parts = [
        "--baseline_ckpt", base_ckpt,
        "--output_dir", f"{run_root}/{exp_id}",
    ]
    for key, value in args.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        parts.extend([f"--{key}", str(value)])
    quoted = " ".join(shlex.quote(str(part)) for part in parts)
    layer = str(row.get("layer", "") or "-").replace("|", "/")
    batch = str(row.get("batch", "") or "-").replace("|", "/")
    purpose = str(row.get("purpose", "") or "-").replace("|", "/")
    print(f"{exp_id}|{layer}|{batch}|{purpose}|{quoted}")
PY
}

generate_queue

TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: no experiments selected for PLAN=${PLAN}" >&2
  exit 2
fi

log_msg "================================"
log_msg "FJMP v2 8-GPU launcher"
log_msg "PLAN=${PLAN}"
log_msg "GPU_IDS=${GPU_IDS_CSV}"
log_msg "BASE_CKPT=${BASE_CKPT}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "QUEUE_FILE=${QUEUE_FILE}"
log_msg "TOTAL_JOBS=${TOTAL_JOBS}"
log_msg "SKIP_DONE=${SKIP_DONE}"
log_msg "DRY_RUN=${DRY_RUN}"
log_msg "================================"

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
FREE_GPUS=("${GPU_LIST[@]}")
OVERALL_STATUS=0
NEXT_INDEX=0
COMPLETED=0
FAILED=0
SKIPPED=0

queue_line_at() {
  local index="$1"
  sed -n "$((index + 1))p" "${QUEUE_FILE}"
}

launch_one() {
  local gpu_id="$1"
  local exp_id="$2"
  local layer="$3"
  local batch="$4"
  local purpose="$5"
  local arg_string="$6"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local stamp log cmd pid

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics_epoch.csv" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out=${out_dir}"
    SKIPPED=$((SKIPPED + 1))
    FREE_GPUS+=("${gpu_id}")
    return 0
  fi

  mkdir -p "${out_dir}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_ROOT}/${exp_id}_${stamp}.log"
  cmd="${PYTHON_BIN} -u -m FJMP.train_fjmp ${arg_string} ${EXTRA_ARGS}"

  {
    echo "================================"
    echo "EXP_ID=${exp_id}"
    echo "LAYER=${layer}"
    echo "BATCH=${batch}"
    echo "PURPOSE=${purpose}"
    echo "GPU=${gpu_id}"
    echo "OUT_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
    echo "================================"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} cmd=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  pid=$!
  RUNNING_PIDS+=("${pid}")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} pid=${pid} log=${log}"
}

start_until_full() {
  local gpu_id line exp_id layer batch purpose arg_string
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS='|' read -r exp_id layer batch purpose arg_string <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${layer}" "${batch}" "${purpose}" "${arg_string}"
  done
}

reap_one() {
  local status done_pid done_tag done_gpu i
  wait -n
  status=$?

  for i in "${!RUNNING_PIDS[@]}"; do
    if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      done_pid="${RUNNING_PIDS[$i]}"
      done_tag="${RUNNING_TAGS[$i]}"
      done_gpu="${RUNNING_GPUS[$i]}"
      unset 'RUNNING_PIDS[i]' 'RUNNING_TAGS[i]' 'RUNNING_GPUS[i]'
      RUNNING_PIDS=("${RUNNING_PIDS[@]}")
      RUNNING_TAGS=("${RUNNING_TAGS[@]}")
      RUNNING_GPUS=("${RUNNING_GPUS[@]}")
      FREE_GPUS+=("${done_gpu}")
      if [ "${status}" -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        log_msg "[DONE] exp=${done_tag} pid=${done_pid} freed_gpu=${done_gpu} completed=${COMPLETED}"
      else
        FAILED=$((FAILED + 1))
        OVERALL_STATUS=1
        log_msg "[FAIL] exp=${done_tag} pid=${done_pid} freed_gpu=${done_gpu} failed=${FAILED}"
        if [ "${STOP_ON_FAIL}" = "1" ]; then
          log_msg "[STOP] STOP_ON_FAIL=1, terminating remaining children."
          for pid in "${RUNNING_PIDS[@]}"; do
            kill "${pid}" 2>/dev/null || true
          done
          exit 1
        fi
      fi
      return 0
    fi
  done

  OVERALL_STATUS=1
  log_msg "[WARN] wait -n returned but no finished child was matched."
}

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    reap_one
  fi
  start_until_full
done

log_msg "================================"
log_msg "FJMP v2 queue finished."
log_msg "completed=${COMPLETED} failed=${FAILED} skipped=${SKIPPED} total=${TOTAL_JOBS}"
log_msg "scheduler_log=${SCHED_LOG}"
log_msg "================================"

exit "${OVERALL_STATUS}"
