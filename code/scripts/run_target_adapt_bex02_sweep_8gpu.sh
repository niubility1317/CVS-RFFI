#!/usr/bin/env bash
set -uo pipefail

# 8-GPU target-domain adaptation sweep for BEX02_fishr002_mixed_e170.
# Each GPU pulls the next experiment as soon as its current run exits.
#
# The sweep fixes the two gaps exposed by 5.20-adapt-logs:
# - target/test metrics are emitted every epoch with detailed split lines;
# - satellite-channel evaluation is enabled every epoch.
#
# Examples:
#   bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan SMOKE --dry-run
#   bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7
#   STREAM_LOGS=1 bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan CORE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-${WORKSPACE_ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/target_adapt_bex02_sweep_8gpu}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/target_adapt_bex02_sweep_8gpu}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
TARGET_LOADERS_CSV="${TARGET_LOADERS:-}"
TARGET_CHANNEL_VIEW="${TARGET_CHANNEL_VIEW:-provided_satellite}"
EXP_PREFIX="${EXP_PREFIX:-BEX02_tadapt}"
TARGET_SAMPLES_CSV="${TARGET_SAMPLES:-}"
TARGET_SAMPLES_PER_RX_TX_CSV="${TARGET_SAMPLES_PER_RX_TX:-}"
SEEDS_CSV="${SEEDS:-}"
TARGET_LABEL_MODES_CSV="${TARGET_LABEL_MODES:-labeled,unlabeled}"
EPOCHS_CSV="${EPOCHS:-}"
ADAPT_WEIGHTS_CSV="${ADAPT_WEIGHTS:-}"
EVAL_DETAIL_EVERY="${EVAL_DETAIL_EVERY:-1}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SAT_SCENARIOS="${SAT_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"

usage() {
  sed -n '1,17p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV              GPUs to use, default 0,1,2,3,4,5,6,7
  --plan NAME                SMOKE, CORE, or FULL
  --teacher-ckpt PATH        BEX02 checkpoint, default $TEACHER_CKPT
  --wisig-pkl PATH           Dataset_WigSig/ManySig.pkl path
  --target-loader NAME       Named target loader, default test_unseen_day_unseen_rx
  --target-loaders CSV       Multiple named target loaders; overrides --target-loader
  --target-channel-view NAME clean or provided_satellite, default provided_satellite
  --exp-prefix NAME          Experiment id prefix, default BEX02_tadapt
  --target-samples CSV       Target samples per target receiver
  --target-samples-per-rx-tx CSV
                              Target samples per transmitter inside each target receiver
  --target-label-modes CSV   labeled,unlabeled, or both
  --epochs CSV               Adaptation epochs, default CORE/FULL: 20,50,100
  --adapt-weights CSV        safe,base,strong, or custom triples name:lr:anchor
  --seeds CSV                Seed list
  --sat-scenarios CSV        Satellite scenarios for per-epoch evaluation
  --eval-detail-every N      Print detailed TEST-SPLIT/SAT lines every N epochs, default 1
  --sat-eval-max-batches N   Satellite eval cap, default -1 meaning follow eval_max_batches
  --eval-max-batches N       Main eval cap, default 0/full
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
    --target-loaders) TARGET_LOADERS_CSV="$2"; shift 2 ;;
    --target-channel-view) TARGET_CHANNEL_VIEW="$2"; shift 2 ;;
    --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
    --target-samples) TARGET_SAMPLES_CSV="$2"; shift 2 ;;
    --target-samples-per-rx-tx) TARGET_SAMPLES_PER_RX_TX_CSV="$2"; shift 2 ;;
    --target-label-modes) TARGET_LABEL_MODES_CSV="$2"; shift 2 ;;
    --epochs) EPOCHS_CSV="$2"; shift 2 ;;
    --adapt-weights) ADAPT_WEIGHTS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --sat-scenarios) SAT_SCENARIOS="$2"; shift 2 ;;
    --eval-detail-every) EVAL_DETAIL_EVERY="$2"; shift 2 ;;
    --sat-eval-max-batches) SAT_EVAL_MAX_BATCHES="$2"; shift 2 ;;
    --eval-max-batches) EVAL_MAX_BATCHES="$2"; shift 2 ;;
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
    SMOKE) TARGET_SAMPLES_CSV="5" ;;
    CORE|FULL) TARGET_SAMPLES_CSV="5,10" ;;
    *) echo "ERROR: unknown plan: ${PLAN}" >&2; exit 2 ;;
  esac
fi
if [ -z "${SEEDS_CSV}" ]; then
  case "${plan_upper}" in
    SMOKE|CORE) SEEDS_CSV="1337" ;;
    FULL) SEEDS_CSV="1337,2027,42" ;;
  esac
fi
if [ -z "${EPOCHS_CSV}" ]; then
  case "${plan_upper}" in
    SMOKE) EPOCHS_CSV="1" ;;
    CORE|FULL) EPOCHS_CSV="20,50,100" ;;
  esac
fi
if [ -z "${ADAPT_WEIGHTS_CSV}" ]; then
  case "${plan_upper}" in
    SMOKE) ADAPT_WEIGHTS_CSV="base" ;;
    CORE|FULL) ADAPT_WEIGHTS_CSV="safe,base,strong" ;;
  esac
fi

adapt_weight_fields() {
  case "$1" in
    safe) printf '%s\t%s\t%s' "safe" "5e-5" "0.10" ;;
    base) printf '%s\t%s\t%s' "base" "1e-4" "0.05" ;;
    strong) printf '%s\t%s\t%s' "strong" "2e-4" "0.02" ;;
    *:*:*)
      local name lr anchor
      IFS=':' read -r name lr anchor <<< "$1"
      printf '%s\t%s\t%s' "${name}" "${lr}" "${anchor}"
      ;;
    *)
      echo "ERROR: unknown adapt weight '$1'. Use safe,base,strong, or name:lr:anchor." >&2
      return 2
      ;;
  esac
}

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

generate_queue() {
  local target_loader mode samples seed epochs weight_token weight_name lr_adapt anchor_weight exp_id desc fields_status budget_kind budget_tag
  : > "${QUEUE_FILE}"
  local target_loader_source="${TARGET_LOADER}"
  if [ -n "${TARGET_LOADERS_CSV}" ]; then
    target_loader_source="${TARGET_LOADERS_CSV}"
  fi
  IFS=',' read -r -a target_loader_list <<< "${target_loader_source}"
  if [ -n "${TARGET_SAMPLES_PER_RX_TX_CSV}" ]; then
    IFS=',' read -r -a sample_list <<< "${TARGET_SAMPLES_PER_RX_TX_CSV}"
    budget_kind="rx_tx"
  else
    IFS=',' read -r -a sample_list <<< "${TARGET_SAMPLES_CSV}"
    budget_kind="rx"
  fi
  IFS=',' read -r -a seed_list <<< "${SEEDS_CSV}"
  IFS=',' read -r -a mode_list <<< "${TARGET_LABEL_MODES_CSV}"
  IFS=',' read -r -a epoch_list <<< "${EPOCHS_CSV}"
  IFS=',' read -r -a weight_list <<< "${ADAPT_WEIGHTS_CSV}"
  for target_loader in "${target_loader_list[@]}"; do
    for mode in "${mode_list[@]}"; do
      if [ "${mode}" != "labeled" ] && [ "${mode}" != "unlabeled" ]; then
        echo "ERROR: unknown target label mode: ${mode}" >&2
        return 2
      fi
      for samples in "${sample_list[@]}"; do
        for epochs in "${epoch_list[@]}"; do
          for weight_token in "${weight_list[@]}"; do
            fields_status="$(adapt_weight_fields "${weight_token}")" || return 2
            IFS=$'\t' read -r weight_name lr_adapt anchor_weight <<< "${fields_status}"
            for seed in "${seed_list[@]}"; do
              if [ "${budget_kind}" = "rx_tx" ]; then
                budget_tag="rxtx${samples}"
                desc="teacher=BEX02_fishr002_mixed_e170; label_mode=${mode}; target_loader=${target_loader}; target_channel_view=${TARGET_CHANNEL_VIEW}; samples_per_rx_tx=${samples}; epochs=${epochs}; weight=${weight_name}; lr_adapt=${lr_adapt}; anchor=${anchor_weight}; eval_sat=on; seed=${seed}"
              else
                budget_tag="rxn${samples}"
                desc="teacher=BEX02_fishr002_mixed_e170; label_mode=${mode}; target_loader=${target_loader}; target_channel_view=${TARGET_CHANNEL_VIEW}; samples_per_rx=${samples}; epochs=${epochs}; weight=${weight_name}; lr_adapt=${lr_adapt}; anchor=${anchor_weight}; eval_sat=on; seed=${seed}"
              fi
              exp_id="${EXP_PREFIX}_${mode}_${target_loader}_${budget_tag}_e${epochs}_${weight_name}_seed${seed}"
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${exp_id}" "${target_loader}" "${mode}" "${budget_kind}" "${samples}" "${epochs}" "${weight_name}" "${lr_adapt}" "${anchor_weight}" "${seed}" >> "${QUEUE_FILE}"
            done
          done
        done
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
  --target_channel_view "${TARGET_CHANNEL_VIEW}"
  --adapt_steps_per_epoch 20
  --weight_decay 0.0
  --grad_clip 1.0
  --update_norm true
  --entropy_weight 1.0
  --consistency_weight 0.0
  --pseudo_weight 0.5
  --conf_threshold 0.90
  --margin_threshold 0.20
  --eval_max_batches "${EVAL_MAX_BATCHES}"
  --eval_sat_channel
  --eval_sat_scenarios "${SAT_SCENARIOS}"
  --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}"
  --eval_detail_every "${EVAL_DETAIL_EVERY}"
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
  local gpu_id="$1" exp_id="$2" target_loader="$3" label_mode="$4" budget_kind="$5" samples="$6" epochs="$7" weight_name="$8" lr_adapt="$9" anchor_weight="${10}" seed="${11}"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local log="${LOG_ROOT}/${exp_id}_${STAMP}.log"
  local budget_tag="rxn${samples}"
  local target_sample_args=(--target_num_samples "${samples}" --target_samples_per_rx "${samples}" --target_samples_per_rx_tx 0)
  if [ "${budget_kind}" = "rx_tx" ]; then
    budget_tag="rxtx${samples}"
    target_sample_args=(--target_num_samples 0 --target_samples_per_rx 0 --target_samples_per_rx_tx "${samples}")
  fi
  local short_tag="${label_mode}:${budget_tag}:e${epochs}:${weight_name}:s${seed}|G${gpu_id}"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_target_adapt.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi
  mkdir -p "${out_dir}"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train_target_adapt.py \
    "${BASE_ARGS[@]}" \
    --target_loader "${target_loader}" \
    --eval_sat_on "${target_loader}" \
    --epochs "${epochs}" \
    --lr_adapt "${lr_adapt}" \
    --anchor_weight "${anchor_weight}" \
    --target_label_mode "${label_mode}" \
    --update_classifier "$([ "${label_mode}" = "labeled" ] && echo true || echo false)" \
    "${target_sample_args[@]}" \
    --seed "${seed}" \
    --run_name "${exp_id}" \
    --output_dir "${out_dir}")"
  {
    echo "EXP_ID=${exp_id}"
    echo "SHORT_TAG=${short_tag}"
    echo "DESCRIPTION=teacher=BEX02_fishr002_mixed_e170 target_loader=${target_loader} label_mode=${label_mode} target_channel_view=${TARGET_CHANNEL_VIEW} budget_kind=${budget_kind} samples=${samples} epochs=${epochs} weight=${weight_name} lr_adapt=${lr_adapt} anchor_weight=${anchor_weight} eval_sat=on seed=${seed}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"
  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] tag=${short_tag} exp=${exp_id}"
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
    local line exp_id target_loader label_mode budget_kind samples epochs weight_name lr_adapt anchor_weight seed
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS=$'\t' read -r exp_id target_loader label_mode budget_kind samples epochs weight_name lr_adapt anchor_weight seed <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${target_loader}" "${label_mode}" "${budget_kind}" "${samples}" "${epochs}" "${weight_name}" "${lr_adapt}" "${anchor_weight}" "${seed}"
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

log_msg "BEX02 target-domain adaptation 8GPU sweep launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "TEACHER_CKPT=${TEACHER_CKPT}"
log_msg "WISIG_PKL=${WISIG_PKL}"
log_msg "TARGET_LOADER=${TARGET_LOADER} TARGET_LOADERS=${TARGET_LOADERS_CSV} TARGET_CHANNEL_VIEW=${TARGET_CHANNEL_VIEW} EXP_PREFIX=${EXP_PREFIX} TARGET_SAMPLES=${TARGET_SAMPLES_CSV} TARGET_SAMPLES_PER_RX_TX=${TARGET_SAMPLES_PER_RX_TX_CSV} SEEDS=${SEEDS_CSV}"
log_msg "TARGET_LABEL_MODES=${TARGET_LABEL_MODES_CSV} EPOCHS=${EPOCHS_CSV} ADAPT_WEIGHTS=${ADAPT_WEIGHTS_CSV}"
log_msg "SAT_SCENARIOS=${SAT_SCENARIOS} EVAL_DETAIL_EVERY=${EVAL_DETAIL_EVERY} SAT_EVAL_MAX_BATCHES=${SAT_EVAL_MAX_BATCHES}"
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

log_msg "BEX02 target-domain adaptation 8GPU sweep finished status=${STATUS}"
exit "${STATUS}"
