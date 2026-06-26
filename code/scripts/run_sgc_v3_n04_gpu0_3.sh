#!/usr/bin/env bash
set -uo pipefail

# SGC v3 experiments on the N04 base checkpoint.
#
# Defaults:
#   - GPUs: 0,1,2,3
#   - Base checkpoint: /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth
#   - Output: runs/sgc_v3_n04
#   - Logs: logs/sgc_v3_n04
#
# Examples:
#   bash code/scripts/run_sgc_v3_n04_gpu0_3.sh --dry-run
#   bash code/scripts/run_sgc_v3_n04_gpu0_3.sh
#   bash code/scripts/run_sgc_v3_n04_gpu0_3.sh --gpu-ids 0,1 --plan CORE
#   STREAM_LOGS=1 bash code/scripts/run_sgc_v3_n04_gpu0_3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
PLAN="${PLAN:-FULL}"
PYTHON_BIN="${PYTHON_BIN:-}"
BASE_CKPT="${BASE_CKPT:-/home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/sgc_v3_n04}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/sgc_v3_n04}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  sed -n '1,20p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids 0,1,2,3
  --plan FULL|CORE|TARGET|SMOKE
  --base-ckpt PATH
  --run-root PATH
  --log-root PATH
  --python PATH
  --no-skip-done
  --stop-on-fail
  --dry-run
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --base-ckpt) BASE_CKPT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
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

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: base checkpoint not found: ${BASE_CKPT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

append_rows_for_plan() {
  local plan_name
  plan_name="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  case "${plan_name}" in
    CORE)
      cat <<'EOF' >> "${QUEUE_FILE}"
SGCV3-10_blrc_mixed|CORE|BLRC-only top-3 logit residual; safest first check for boundary calibration gains.|--mode blrc_only --config SGC/configs/sgc_v3_blrc_only.yaml --epochs 40 --lr_sgc 1e-4 --sat_train_scenario mixed_orbit
SGCV3-12_ipfa_mixed|CORE|IPFA-only low-rank feature residual with norm clamp; checks feature-space safety without logit edits.|--mode ipfa_only --config SGC/configs/sgc_v3_ipfa_only.yaml --epochs 40 --lr_sgc 1e-4 --sat_train_scenario mixed_orbit
SGCV3-14_ipfa_blrc_mixed|CORE|Main source-only SGC v3 path: PSC + SEE + IPFA + BLRC + gate.|--mode ipfa_blrc --config SGC/configs/sgc_v3_safe.yaml --epochs 50 --lr_sgc 1e-4 --sat_train_scenario mixed_orbit
EOF
      ;;
    TARGET)
      cat <<'EOF' >> "${QUEUE_FILE}"
SGCV3-20_target_mixed|TARGET|Target pseudo-label adaptation path using SGC-processed satellite samples as unlabeled target stream.|--mode target_adapt --config SGC/configs/sgc_v3_target_adapt.yaml --epochs 30 --lr_sgc 5e-5 --sat_train_scenario mixed_orbit
EOF
      ;;
    SMOKE)
      cat <<'EOF' >> "${QUEUE_FILE}"
SGCV3-SMOKE_blrc|SMOKE|Short BLRC-only parser/data/model smoke check.|--mode blrc_only --config SGC/configs/sgc_v3_blrc_only.yaml --epochs 1 --lr_sgc 1e-4 --sat_train_scenario mixed_orbit --eval_max_batches 1 --wisig_max_train_per_combo 2 --wisig_max_val_per_combo 2 --wisig_max_test_per_combo 2
EOF
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use FULL,CORE,TARGET,SMOKE." >&2
      exit 2
      ;;
  esac
}

generate_queue() {
  : > "${QUEUE_FILE}"
  local plan_upper
  plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
  if [ "${plan_upper}" = "FULL" ] || [ "${plan_upper}" = "ALL" ]; then
    append_rows_for_plan CORE
    append_rows_for_plan TARGET
  else
    IFS=',' read -r -a plans <<< "${PLAN}"
    for p in "${plans[@]}"; do
      append_rows_for_plan "${p}"
    done
  fi
}

generate_queue
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

BASE_ARGS=(
  --teacher_ckpt "${BASE_CKPT}"
  --freeze_teacher true
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --wisig_train_ratio 0.2
  --wisig_out_len 256
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --prefetch_factor 2
  --seed 1337
  --device cuda:0
)

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
  local gpu_id="$1" exp_id="$2" group="$3" purpose="$4" extra_args="$5"
  local out_dir="${RUN_ROOT}/${exp_id}"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_sgc_v3.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local log="${LOG_ROOT}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u -m SGC.v3.train_sgc_v3 "${BASE_ARGS[@]}" --run_name "${exp_id}" --output_dir "${out_dir}")"
  cmd="${cmd}${extra_args} ${EXTRA_ARGS}"

  {
    echo "EXP_ID=${exp_id}"
    echo "GROUP=${group}"
    echo "PURPOSE=${purpose}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "BASE_CKPT=${BASE_CKPT}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 PYTHONPATH=${CODE_ROOT} ${cmd}"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} group=${group} cmd=PYTHONPATH=${CODE_ROOT} ${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  if [ "${STREAM_LOGS}" = "1" ]; then
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}" bash -lc "${cmd}" 2>&1
      status="$?"
      echo "__EXIT_STATUS__=${status}"
      exit "${status}"
    ) | sed -u "s/^/[${exp_id}|GPU${gpu_id}] /" | tee -a "${log}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}" bash -lc "${cmd}" >> "${log}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} group=${group} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id group purpose extra_args
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS='|' read -r exp_id group purpose extra_args <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${group}" "${purpose}" "${extra_args}"
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
      if [ "${status}" -ne 0 ]; then
        STATUS=1
      fi
      log_msg "[FINISHED] exp=${tag} status=${status} freed_gpu=${gpu}"
      return
    fi
  done
}

log_msg "SGC v3 N04 launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "BASE_CKPT=${BASE_CKPT}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "QUEUE_FILE=${QUEUE_FILE}"
log_msg "DRY_RUN=${DRY_RUN} SKIP_DONE=${SKIP_DONE} STOP_ON_FAIL=${STOP_ON_FAIL} STREAM_LOGS=${STREAM_LOGS}"

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    reap_one
    if [ "${STATUS}" -ne 0 ] && [ "${STOP_ON_FAIL}" = "1" ]; then
      log_msg "Stopping early because a job failed and STOP_ON_FAIL=1."
      exit 1
    fi
  fi
  start_until_full
done

log_msg "SGC v3 N04 queue finished status=${STATUS}"
exit "${STATUS}"
