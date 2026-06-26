#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fsl_inference_10h_20260614}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
MAX_SCHEDULER_SECONDS="${MAX_SCHEDULER_SECONDS:-36000}"
POLL_SECONDS="${POLL_SECONDS:-45}"
DRY_RUN="${DRY_RUN:-0}"

SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"
SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"
SFE_SHOTS="${SFE_SHOTS:-20}"
SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"
SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"
SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"
SFE_SEED="${SFE_SEED:-1457}"
SFE_SAT_SEED="${SFE_SAT_SEED:-2368}"

TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
TARGET_SCENARIOS="${TARGET_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
FTRC_EPOCHS="${FTRC_EPOCHS:-20}"
FTRC_STEPS="${FTRC_STEPS:-10}"
FTRC_TARGET_BATCH_SIZE="${FTRC_TARGET_BATCH_SIZE:-32}"
FTRC_SEED="${FTRC_SEED:-1337}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -z "${UNKNOWN_TX_IDS}" ]]; then
  echo "[ERROR] UNKNOWN_TX_IDS must not be empty for this open-set batch" >&2
  exit 2
fi

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
SCHED_LOG="${LOG_ROOT}/scheduler.out"
EVENTS_TSV="${LOG_ROOT}/scheduler_events.tsv"
: > "${SCHED_LOG}"
: > "${EVENTS_TSV}"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

gpu_process_count() {
  local gpu="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo 0
    return 0
  fi
  nvidia-smi pmon -c 1 2>/dev/null \
    | awk -v g="${gpu}" '$1 == g && $2 ~ /^[0-9]+$/ && $3 == "C" {count += 1} END {print count + 0}'
}

run_sfe_openset_bundle() {
  local cid="$1"
  local channel_view="$2"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[SFE-BUNDLE-BEGIN] cid=${cid} source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS} shots=${SFE_SHOTS} target_new_channel_view=${channel_view}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --new_wisig_pkl "${NEW_WISIG_PKL}" \
    --out_npz "${out_dir}/features.npz" \
    --feature_name z_id \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --new_tx_ids "${NEW_TX_IDS}" \
    --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
    --wisig_equalized 1 \
    --wisig_domain rx_day \
    --wisig_out_len 256 \
    --max_samples_per_combo 0 \
    --max_samples_per_tx "${SFE_MAX_SAMPLES_PER_TX}" \
    --batch_size "${SFE_EXPORT_BATCH_SIZE}" \
    --device cuda:0 \
    --seed "${SFE_SEED}" \
    --target_new_channel_view "${channel_view}" \
    --target_new_sat_scenarios "${TARGET_SCENARIOS}" \
    --target_new_sat_seed "${SFE_SAT_SEED}"

  run_sfe_eval "${cid}" "combined_t070_m005_mh8" "combined" "0.70" "0.05" "8.0" "1.0"
  run_sfe_eval "${cid}" "combined_t065_m003_mh8" "combined" "0.65" "0.03" "8.0" "1.0"
  run_sfe_eval "${cid}" "combined_t075_m007_mh8" "combined" "0.75" "0.07" "8.0" "1.0"
  run_sfe_eval "${cid}" "openmax_t070_q095" "openmax" "0.70" "" "" "0.95"
  run_sfe_eval "${cid}" "mahal_t070_mh6" "mahalanobis" "0.70" "" "6.0" "0.95"
  run_sfe_eval "${cid}" "mahal_t070_mh10" "mahalanobis" "0.70" "" "10.0" "0.95"
  echo "[SFE-BUNDLE-END] cid=${cid}"
}

run_sfe_eval() {
  local cid="$1"
  local suffix="$2"
  local gate="$3"
  local threshold="$4"
  local margin="$5"
  local max_mahal="$6"
  local openmax_q="$7"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=(
    "${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py"
    --protocol sfe
    --feature_npz "${out_dir}/features.npz"
    --output_json "${out_dir}/metrics_${suffix}.json"
    --manifest_json "${out_dir}/manifest_${suffix}.json"
    --score_table_csv "${out_dir}/score_table_${suffix}.csv"
    --source_tx_ids "${SOURCE_TX_IDS}"
    --new_tx_ids "${NEW_TX_IDS}"
    --unknown_tx_ids "${UNKNOWN_TX_IDS}"
    --shots "${SFE_SHOTS}"
    --source_proto_per_tx "${SFE_SOURCE_PROTO_PER_TX}"
    --source_query_per_tx "${SFE_SOURCE_QUERY_PER_TX}"
    --query_per_tx "${SFE_QUERY_PER_TX}"
    --unknown_threshold "${threshold}"
    --gate_mode "${gate}"
    --openmax_tail_size 20
    --openmax_quantile "${openmax_q}"
    --openmax_min_threshold 0.02
    --seed "${SFE_SEED}"
  )
  if [[ -n "${margin}" ]]; then
    cmd+=(--min_margin "${margin}")
  fi
  if [[ -n "${max_mahal}" ]]; then
    cmd+=(--max_mahalanobis "${max_mahal}")
  fi
  echo "[SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin:-n/a} max_mahalanobis=${max_mahal:-n/a}"
  "${cmd[@]}"
}

run_ftrc_candidate() {
  local cid="$1"
  local adapter="$2"
  local k="$3"
  local lr="$4"
  local anchor="$5"
  local alpha="$6"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  local update_norm="false"
  local update_classifier="false"
  if [[ "${adapter}" == "logit_calibration" ]]; then
    update_norm="false"
    update_classifier="false"
  fi
  echo "[FTRC-BEGIN] cid=${cid} adapter=${adapter} k=${k} lr=${lr} anchor=${anchor} alpha=${alpha}"
  "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --output_dir "${out_dir}" \
    --dataset wisig \
    --wisig_pkl "${WISIG_PKL}" \
    --wisig_equalized 1 \
    --wisig_domain rx_day \
    --wisig_train_ratio 0.1 \
    --wisig_guard_gap 8 \
    --wisig_train_days 0,1 \
    --wisig_test_days 2,3 \
    --wisig_train_rxs 0,1,2,3,4,5,6 \
    --wisig_test_rxs 7,8,9,10,11 \
    --target_loader "${TARGET_LOADER}" \
    --target_channel_view satellite \
    --target_label_mode labeled \
    --target_samples_per_rx_tx "${k}" \
    --target_train_scenarios "${TARGET_SCENARIOS}" \
    --epochs "${FTRC_EPOCHS}" \
    --adapt_steps_per_epoch "${FTRC_STEPS}" \
    --target_batch_size "${FTRC_TARGET_BATCH_SIZE}" \
    --lr_adapt "${lr}" \
    --entropy_weight 0 \
    --consistency_weight 0 \
    --pseudo_weight 0 \
    --anchor_weight "${anchor}" \
    --eval_detail_every 5 \
    --target_adapter_type "${adapter}" \
    --adapter_rank 4 \
    --adapter_bottleneck 16 \
    --adapter_alpha "${alpha}" \
    --adapter_dropout 0.0 \
    --freeze_base_stats true \
    --update_norm "${update_norm}" \
    --update_classifier "${update_classifier}" \
    --rollback_enabled true \
    --eval_sat_channel true \
    --eval_sat_on "${TARGET_LOADER}" \
    --eval_sat_scenarios "${TARGET_SCENARIOS}" \
    --sat_eval_max_batches 0 \
    --eval_max_batches 0 \
    --seed "${FTRC_SEED}" \
    --run_name "${cid}"
  echo "[FTRC-END] cid=${cid}"
}

CAND_ID=()
CAND_GPU=()
CAND_KIND=()
CAND_CHANNEL=()
CAND_ADAPTER=()
CAND_K=()
CAND_LR=()
CAND_ANCHOR=()
CAND_ALPHA=()
CAND_STATUS=()
CAND_PID=()

add_sfe() {
  CAND_ID+=("$1"); CAND_GPU+=("$2"); CAND_KIND+=("sfe"); CAND_CHANNEL+=("$3"); CAND_ADAPTER+=(""); CAND_K+=(""); CAND_LR+=(""); CAND_ANCHOR+=(""); CAND_ALPHA+=(""); CAND_STATUS+=("queued"); CAND_PID+=("")
}

add_ftrc() {
  CAND_ID+=("$1"); CAND_GPU+=("$2"); CAND_KIND+=("ftrc"); CAND_CHANNEL+=(""); CAND_ADAPTER+=("$3"); CAND_K+=("$4"); CAND_LR+=("$5"); CAND_ANCHOR+=("$6"); CAND_ALPHA+=("$7"); CAND_STATUS+=("queued"); CAND_PID+=("")
}

add_sfe "SFE_OPENSET_K20_GATE_SWEEP_CLEAN_U2" "0" "clean"
add_sfe "SFE_OPENSET_K20_GATE_SWEEP_SAT_TARGET_U2" "1" "satellite"
add_ftrc "FTRC_CAL_K5_SAFE" "2" "logit_calibration" "5" "5e-5" "0.10" "1.0"
add_ftrc "FTRC_CAL_K10_SAFE" "3" "logit_calibration" "10" "5e-5" "0.10" "1.0"
add_ftrc "FTRC_CAL_K20_SAFE" "4" "logit_calibration" "20" "5e-5" "0.10" "1.0"
add_ftrc "FTRC_FEAT_K5_SAFER" "5" "feature_residual" "5" "2e-5" "0.10" "0.5"
add_ftrc "FTRC_FEAT_K10_SAFER" "6" "feature_residual" "10" "2e-5" "0.10" "0.5"
add_ftrc "FTRC_FEAT_K20_SAFER" "6" "feature_residual" "20" "2e-5" "0.10" "0.5"
add_ftrc "FTRC_LORA_K5_SAFER" "7" "logit_lora" "5" "2e-5" "0.10" "0.5"
add_ftrc "FTRC_LORA_K10_SAFER" "7" "logit_lora" "10" "2e-5" "0.10" "0.5"
add_ftrc "FTRC_LORA_K20_SAFER" "7" "logit_lora" "20" "2e-5" "0.10" "0.5"

launch_candidate() {
  local i="$1"
  local cid="${CAND_ID[$i]}"
  local gpu="${CAND_GPU[$i]}"
  local kind="${CAND_KIND[$i]}"
  local log_path="${LOG_ROOT}/${cid}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] cid=${cid} gpu=${gpu} kind=${kind} log=${log_path}" | tee -a "${EVENTS_TSV}"
    CAND_STATUS[$i]="dry_run"
    return 0
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"
  if [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_openset_bundle "${cid}" "${CAND_CHANNEL[$i]}" > "${log_path}" 2>&1) &
  else
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "${CAND_ADAPTER[$i]}" "${CAND_K[$i]}" "${CAND_LR[$i]}" "${CAND_ANCHOR[$i]}" "${CAND_ALPHA[$i]}" > "${log_path}" 2>&1) &
  fi
  local pid="$!"
  CAND_PID[$i]="${pid}"
  CAND_STATUS[$i]="running"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "${cid}" "LAUNCHED" "gpu=${gpu}" "pid=${pid}" "log=${log_path}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"
}

reap_finished() {
  local i pid cid gpu
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_STATUS[$i]}" != "running" ]]; then
      continue
    fi
    pid="${CAND_PID[$i]}"
    cid="${CAND_ID[$i]}"
    gpu="${CAND_GPU[$i]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      CAND_STATUS[$i]="finished"
      printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "${cid}" "FINISHED_OR_EXITED" "gpu=${gpu}" "pid=${pid}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"
    fi
  done
}

has_queued() {
  local i
  for i in "${!CAND_ID[@]}"; do
    [[ "${CAND_STATUS[$i]}" == "queued" ]] && return 0
  done
  return 1
}

has_running() {
  local i
  for i in "${!CAND_ID[@]}"; do
    [[ "${CAND_STATUS[$i]}" == "running" ]] && return 0
  done
  return 1
}

planned_running_count() {
  local gpu="$1"
  local i count=0
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_STATUS[$i]}" == "running" && "${CAND_GPU[$i]}" == "${gpu}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

try_launch_queued() {
  local i gpu count planned effective
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_STATUS[$i]}" != "queued" ]]; then
      continue
    fi
    gpu="${CAND_GPU[$i]}"
    count="$(gpu_process_count "${gpu}")"
    planned="$(planned_running_count "${gpu}")"
    effective=$((count + planned))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "${CAND_ID[$i]}" "CAPACITY" "gpu=${gpu}" "compute_count=${count}" "planned_running=${planned}" "effective_count=${effective}" | tee -a "${EVENTS_TSV}" >/dev/null
    if (( effective < MAX_ACTIVE_PER_GPU )); then
      launch_candidate "${i}"
    else
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "${CAND_ID[$i]}" "DEFERRED_CAPACITY" "gpu=${gpu}" "compute_count=${count}" "planned_running=${planned}" "effective_count=${effective}" | tee -a "${EVENTS_TSV}" >/dev/null
    fi
  done
}

log_msg "[SPACEBORNE-FSL-10H] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} max_seconds=${MAX_SCHEDULER_SECONDS}"
log_msg "[SPACEBORNE-FSL-10H-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
log_msg "[SPACEBORNE-FSL-10H-TARGET] loader=${TARGET_LOADER} scenarios=${TARGET_SCENARIOS} ftrc_epochs=${FTRC_EPOCHS} ftrc_steps=${FTRC_STEPS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[SPACEBORNE-FSL-10H-CANDIDATE] idx=${i} id=${CAND_ID[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} channel=${CAND_CHANNEL[$i]:-n/a} adapter=${CAND_ADAPTER[$i]:-n/a} k=${CAND_K[$i]:-n/a} lr=${CAND_LR[$i]:-n/a} anchor=${CAND_ANCHOR[$i]:-n/a} alpha=${CAND_ALPHA[$i]:-n/a}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  try_launch_queued
  log_msg "[SPACEBORNE-FSL-10H-DRYRUN-COMPLETE]"
  exit 0
fi

start_ts="$(date +%s)"
while true; do
  reap_finished
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed <= MAX_SCHEDULER_SECONDS )); then
    try_launch_queued
  fi
  if ! has_queued && ! has_running; then
    log_msg "[SPACEBORNE-FSL-10H-COMPLETE] all candidates reached terminal state"
    exit 0
  fi
  if (( elapsed > MAX_SCHEDULER_SECONDS )) && ! has_running; then
    log_msg "[SPACEBORNE-FSL-10H-STOP] scheduler window expired with queued candidates remaining"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
