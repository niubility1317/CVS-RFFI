#!/usr/bin/env bash
set -uo pipefail

# Few-shot target-domain adaptation launcher for the strongest BEX02 baseline.
#
# The trainer loads BEX02_fishr002_mixed_e170/latest_model.pth, keeps the
# backbone mostly frozen, and adapts on target samples that are already
# satellite-channel views. It schedules labeled and unlabeled target lines.
#
# Examples:
#   bash code/scripts/run_target_adapt_bex02_6gpu.sh --plan SMOKE --dry-run
#   TEACHER_CKPT=/path/to/latest_model.pth WISIG_PKL=/path/to/ManySig.pkl \
#     bash code/scripts/run_target_adapt_bex02_6gpu.sh --plan CORE --gpu-ids 6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-6,7}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-${WORKSPACE_ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/target_adapt_bex02}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/target_adapt_bex02}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
TARGET_SAMPLES_CSV="${TARGET_SAMPLES:-}"
SEEDS_CSV="${SEEDS:-}"
TARGET_LABEL_MODES_CSV="${TARGET_LABEL_MODES:-labeled,unlabeled}"

usage() {
  sed -n '1,11p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV              GPUs to use, default 6,7
  --plan NAME                SMOKE, CORE, or FULL
  --teacher-ckpt PATH        BEX02 checkpoint, default $TEACHER_CKPT
  --wisig-pkl PATH           Dataset_WigSig/ManySig.pkl path
  --target-loader NAME       Named target loader, default test_unseen_day_unseen_rx
  --target-samples CSV       target samples per target receiver; overrides plan defaults
  --target-label-modes CSV   labeled,unlabeled, or both; default labeled,unlabeled
  --seeds CSV                seed list; overrides plan defaults
  --python PATH              Python executable
  --run-root PATH            Output checkpoint root
  --log-root PATH            Log root
  --no-skip-done             Re-run even when best_target_adapt.pth exists
  --stop-on-fail             Stop queue after first failure
  --stream-logs              Stream job logs to scheduler stdout
  --dry-run                  Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --teacher-ckpt) TEACHER_CKPT="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --target-loader) TARGET_LOADER="$2"; shift 2 ;;
    --target-samples) TARGET_SAMPLES_CSV="$2"; shift 2 ;;
    --target-label-modes) TARGET_LABEL_MODES_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
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

if [ "${DRY_RUN}" != "1" ] && { [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; }; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${TEACHER_CKPT}" ]; then
  echo "ERROR: TEACHER_CKPT not found: ${TEACHER_CKPT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
if [ -z "${TARGET_SAMPLES_CSV}" ]; then
  case "${plan_upper}" in
    SMOKE|CORE|FULL) TARGET_SAMPLES_CSV="5,10" ;;
    *) echo "ERROR: unknown plan: ${PLAN}" >&2; exit 2 ;;
  esac
fi
if [ -z "${SEEDS_CSV}" ]; then
  case "${plan_upper}" in
    SMOKE|CORE) SEEDS_CSV="1337" ;;
    FULL) SEEDS_CSV="1337,2027,42" ;;
  esac
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

generate_queue() {
  local mode samples seed exp_id desc
  : > "${QUEUE_FILE}"
  IFS=',' read -r -a sample_list <<< "${TARGET_SAMPLES_CSV}"
  IFS=',' read -r -a seed_list <<< "${SEEDS_CSV}"
  IFS=',' read -r -a mode_list <<< "${TARGET_LABEL_MODES_CSV}"
  for mode in "${mode_list[@]}"; do
    if [ "${mode}" != "labeled" ] && [ "${mode}" != "unlabeled" ]; then
      echo "ERROR: unknown target label mode: ${mode}" >&2
      return 2
    fi
    for samples in "${sample_list[@]}"; do
      for seed in "${seed_list[@]}"; do
        exp_id="BEX02_fishr002_mixed_e170_target_adapt_${mode}_${TARGET_LOADER}_rxn${samples}_seed${seed}"
        desc="config=BEX02_fishr002_mixed_e170/latest_model; method=provided_satellite_${mode}_finetune; target_loader=${TARGET_LOADER}; target_samples_per_rx=${samples}; seed=${seed}"
        printf '%s\t%s\t%s\t%s\t%s\n' "${exp_id}" "${mode}" "${samples}" "${seed}" "${desc}" >> "${QUEUE_FILE}"
      done
    done
  done
}

generate_queue
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

BASE_ARGS=(
  --teacher_ckpt "${TEACHER_CKPT}"
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_train_ratio 0.2
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --wisig_max_day123_per_combo 0
  --wisig_max_train_per_combo 0
  --wisig_max_val_per_combo 0
  --wisig_max_test_per_combo 0
  --batch_size 64
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor 2
  --target_loader "${TARGET_LOADER}"
  --target_channel_view provided_satellite
  --epochs 20
  --adapt_steps_per_epoch 20
  --lr_adapt 1e-4
  --update_norm true
  --entropy_weight 1.0
  --consistency_weight 0.0
  --pseudo_weight 0.5
  --anchor_weight 0.05
  --conf_threshold 0.90
  --margin_threshold 0.20
  --eval_sat_channel false
  --eval_sat_on "${TARGET_LOADER}"
  --sat_eval_max_batches -1
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
  local gpu_id="$1" exp_id="$2" label_mode="$3" samples="$4" seed="$5" desc="$6"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local log="${LOG_ROOT}/${exp_id}_${STAMP}.log"
  local short_tag="${label_mode}:rxn${samples}:s${seed}|G${gpu_id}"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_target_adapt.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi
  mkdir -p "${out_dir}"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train_target_adapt.py \
    "${BASE_ARGS[@]}" \
    --target_label_mode "${label_mode}" \
    --update_classifier "$([ "${label_mode}" = "labeled" ] && echo true || echo false)" \
    --target_num_samples "${samples}" \
    --target_samples_per_rx "${samples}" \
    --seed "${seed}" \
    --run_name "${exp_id}" \
    --output_dir "${out_dir}")"
  {
    echo "EXP_ID=${exp_id}"
    echo "SHORT_TAG=${short_tag}"
    echo "DESCRIPTION=${desc}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"
  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] tag=${short_tag} exp=${exp_id} ${desc}"
    log_msg "CMD=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi
  if [ "${STREAM_LOGS}" = "1" ]; then
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" 2>&1
      status="$?"
      echo "__EXIT_STATUS__=${status}"
      exit "${status}"
    ) | sed -u "s/^/[${short_tag}] /" | tee -a "${log}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id label_mode samples seed desc
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS=$'\t' read -r exp_id label_mode samples seed desc <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${label_mode}" "${samples}" "${seed}" "${desc}"
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

log_msg "BEX02 target-domain adaptation launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "TEACHER_CKPT=${TEACHER_CKPT}"
log_msg "WISIG_PKL=${WISIG_PKL}"
log_msg "TARGET_LOADER=${TARGET_LOADER} TARGET_SAMPLES=${TARGET_SAMPLES_CSV} SEEDS=${SEEDS_CSV}"
log_msg "TARGET_LABEL_MODES=${TARGET_LABEL_MODES_CSV}"
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

log_msg "BEX02 target-domain adaptation queue finished status=${STATUS}"
exit "${STATUS}"
