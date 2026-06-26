#!/usr/bin/env bash
set -uo pipefail

# 8-GPU dynamic queue launcher for SGV-BP-FJMP experiments.
#
# Examples:
#   bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt runs/cvs_rffi_staged/B3b_stable_sat07_cls_only/latest_model.pth
#   bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt /path/to/latest_model.pth --plan FULL --gpu-ids 0,1,2,3,4,5,6,7
#   bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt /path/to/latest_model.pth --plan CORE --dry-run
#
# Main command includes:
#   --model_name SGV-BP-FJMP
#   --selection_metric best_proxy_safe_score

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PLAN="${PLAN:-FULL}"
PYTHON_BIN="${PYTHON_BIN:-}"
BASE_CKPT="${BASE_CKPT:-}"
RUN_ROOT="${RUN_ROOT:-runs/fjmp_sgv_bp}"
LOG_ROOT="${LOG_ROOT:-logs/fjmp_sgv_bp}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---amp false}"

usage() {
  sed -n '1,24p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --base-ckpt) BASE_CKPT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${BASE_CKPT}" ]; then
  echo "ERROR: --base-ckpt is required." >&2
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
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
SCHED_LOG="${LOG_ROOT}/scheduler_$(date +%Y%m%d_%H%M%S).log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN//,/}_$(date +%Y%m%d_%H%M%S).tsv"

generate_queue() {
  PLAN_VALUE="${PLAN}" RUN_ROOT_VALUE="${RUN_ROOT}" BASE_CKPT_VALUE="${BASE_CKPT}" "${PYTHON_BIN}" - <<'PY' > "${QUEUE_FILE}"
import os
import shlex
from FJMP.experiment_manifest import build_experiment_manifest

plan = os.environ["PLAN_VALUE"].strip()
run_root = os.environ["RUN_ROOT_VALUE"].strip()
base_ckpt = os.environ["BASE_CKPT_VALUE"].strip()
plan_upper = plan.upper()
if plan_upper.startswith("EXP-"):
    layers = ["SGV-BP"]
    wanted = {plan_upper}
else:
    layers = ["SGV-BP"] if plan.upper() in {"EXP", "SGV-BP", "FULL", "ALL"} else [p.strip() for p in plan.split(",") if p.strip()]
    wanted = None
selected_rows = build_experiment_manifest(layers)
if plan.upper() in {"FULL", "ALL"}:
    core = [row for row in selected_rows if str(row.get("batch", "")).upper() == "CORE"]
    rest = [row for row in selected_rows if str(row.get("batch", "")).upper() != "CORE"]
    selected_rows = core + rest
for row in selected_rows:
    exp_id = str(row["id"])
    if wanted is not None and exp_id.upper() not in wanted:
        continue
    args = row.get("args") or {}
    parts = ["--baseline_ckpt", base_ckpt, "--output_dir", f"{run_root}/{exp_id}"]
    for key, value in args.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        parts.extend([f"--{key}", str(value)])
    print(f"{exp_id}|{str(row.get('purpose', '-')).replace('|', '/')}|{' '.join(shlex.quote(str(p)) for p in parts)}")
PY
}

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

generate_queue
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: no SGV-BP experiments selected for PLAN=${PLAN}" >&2
  exit 2
fi

log_msg "SGV-BP-FJMP launcher PLAN=${PLAN} GPU_IDS=${GPU_IDS_CSV} TOTAL_JOBS=${TOTAL_JOBS}"

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
FREE_GPUS=("${GPU_LIST[@]}")
NEXT_INDEX=0
STATUS=0

queue_line_at() {
  sed -n "$(($1 + 1))p" "${QUEUE_FILE}"
}

launch_one() {
  local gpu_id="$1" exp_id="$2" purpose="$3" arg_string="$4"
  local out_dir="${RUN_ROOT}/${exp_id}"
  if [ "${SKIP_DONE}" = "1" ] && { [ -f "${out_dir}/metrics_epoch.csv" ] || compgen -G "${LOG_ROOT}/${exp_id}_*_metrics_epoch.csv" >/dev/null; }; then
    log_msg "[SKIP-DONE] exp=${exp_id}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi
  mkdir -p "${out_dir}"
  local log="${LOG_ROOT}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  local metrics_csv="${log%.log}_metrics_epoch.csv"
  local metrics_arg
  metrics_arg="$(printf '%q' "${metrics_csv}")"
  local cmd="${PYTHON_BIN} -u -m FJMP.train_fjmp ${arg_string} --metrics_csv ${metrics_arg} ${EXTRA_ARGS}"
  echo "EXP_ID=${exp_id} PURPOSE=${purpose} GPU=${gpu_id} CMD=${cmd}" > "${log}"
  echo "METRICS_CSV=${metrics_csv}" >> "${log}"
  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} cmd=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id purpose arg_string
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS='|' read -r exp_id purpose arg_string <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${purpose}" "${arg_string}"
  done
}

reap_one() {
  wait -n
  local status=$?
  local i
  for i in "${!RUNNING_PIDS[@]}"; do
    if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      local gpu="${RUNNING_GPUS[$i]}" tag="${RUNNING_TAGS[$i]}"
      unset 'RUNNING_PIDS[i]' 'RUNNING_TAGS[i]' 'RUNNING_GPUS[i]'
      RUNNING_PIDS=("${RUNNING_PIDS[@]}")
      RUNNING_TAGS=("${RUNNING_TAGS[@]}")
      RUNNING_GPUS=("${RUNNING_GPUS[@]}")
      FREE_GPUS+=("${gpu}")
      [ "${status}" -eq 0 ] || STATUS=1
      log_msg "[FINISHED] exp=${tag} status=${status} freed_gpu=${gpu}"
      return
    fi
  done
}

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  [ "${#RUNNING_PIDS[@]}" -gt 0 ] && reap_one
  start_until_full
done

log_msg "SGV-BP-FJMP queue finished status=${STATUS}"
exit "${STATUS}"
