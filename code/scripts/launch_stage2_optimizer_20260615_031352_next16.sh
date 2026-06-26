#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next16_20260615_031352}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
TARGET_SCENARIOS="${TARGET_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -z "${UNKNOWN_TX_IDS}" ]]; then
  echo "[ERROR] UNKNOWN_TX_IDS must not be empty for stage2 open-set validation" >&2
  exit 2
fi

SCHED_LOG="${LOG_ROOT}/scheduler.out"
EVENTS_TSV="${LOG_ROOT}/scheduler_events.tsv"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  : > "${SCHED_LOG}"
  : > "${EVENTS_TSV}"
fi

log_msg() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "$@"
  else
    echo "$@" | tee -a "${SCHED_LOG}"
  fi
}

event_row() {
  local row
  row="$(printf "%s\t%s\t%s\t%s\t%s\t%s" "$(date -Is)" "$1" "$2" "$3" "$4" "$5")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${row}"
  else
    echo "${row}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"
  fi
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

run_sfe_eval() {
  local cid="$1"
  local suffix="$2"
  local gate="$3"
  local threshold="$4"
  local margin="$5"
  local max_mahal="$6"
  local openmax_q="$7"
  local shots="$8"
  local source_proto="$9"
  local query_per_tx="${10}"
  local seed="${11}"
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
    --shots "${shots}"
    --source_proto_per_tx "${source_proto}"
    --source_query_per_tx 20
    --query_per_tx "${query_per_tx}"
    --unknown_threshold "${threshold}"
    --gate_mode "${gate}"
    --openmax_tail_size 20
    --openmax_quantile "${openmax_q}"
    --openmax_min_threshold 0.02
    --seed "${seed}"
  )
  if [[ -n "${margin}" ]]; then
    cmd+=(--min_margin "${margin}")
  fi
  if [[ -n "${max_mahal}" ]]; then
    cmd+=(--max_mahalanobis "${max_mahal}")
  fi
  echo "[S2-SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin:-n/a} max_mahalanobis=${max_mahal:-n/a}"
  "${cmd[@]}"
}

run_sfe_bundle() {
  local cid="$1"
  local channel_view="$2"
  local shots="$3"
  local seed="$4"
  local sat_seed="$5"
  local max_samples_per_tx="$6"
  local source_proto="$7"
  local query_per_tx="$8"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-SFE-BEGIN] cid=${cid} channel=${channel_view} shots=${shots} source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
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
    --max_samples_per_tx "${max_samples_per_tx}" \
    --batch_size 512 \
    --device cuda:0 \
    --seed "${seed}" \
    --target_new_channel_view "${channel_view}" \
    --target_new_sat_scenarios "${TARGET_SCENARIOS}" \
    --target_new_sat_seed "${sat_seed}"

  run_sfe_eval "${cid}" "mahal_t070_mh6" "mahalanobis" "0.70" "" "6.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "combined_t065_m003_mh8" "combined" "0.65" "0.03" "8.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "combined_t070_m005_mh8" "combined" "0.70" "0.05" "8.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "openmax_t070_q095" "openmax" "0.70" "" "" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  echo "[S2-SFE-END] cid=${cid}"
}

run_ftrc_candidate() {
  local cid="$1"
  local adapter="$2"
  local k="$3"
  local lr="$4"
  local anchor="$5"
  local alpha="$6"
  local rank="$7"
  local epochs="$8"
  local steps="$9"
  local eval_detail="${10}"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-FTRC-BEGIN] cid=${cid} adapter=${adapter} k=${k} lr=${lr} anchor=${anchor} alpha=${alpha} rank=${rank} epochs=${epochs} steps=${steps}"
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
    --epochs "${epochs}" \
    --adapt_steps_per_epoch "${steps}" \
    --target_batch_size 32 \
    --lr_adapt "${lr}" \
    --entropy_weight 0 \
    --consistency_weight 0 \
    --pseudo_weight 0 \
    --anchor_weight "${anchor}" \
    --anchor_temperature 2.5 \
    --eval_detail_every "${eval_detail}" \
    --target_adapter_type "${adapter}" \
    --adapter_rank "${rank}" \
    --adapter_bottleneck 16 \
    --adapter_alpha "${alpha}" \
    --adapter_dropout 0.0 \
    --freeze_base_stats true \
    --update_norm false \
    --update_classifier false \
    --rollback_enabled true \
    --eval_sat_channel true \
    --eval_sat_on "${TARGET_LOADER}" \
    --eval_sat_scenarios "${TARGET_SCENARIOS}" \
    --sat_eval_max_batches 0 \
    --eval_max_batches 0 \
    --seed 2026 \
    --run_name "${cid}"
  echo "[S2-FTRC-END] cid=${cid}"
}

declare -a CAND_ID=(
  "S2_SFE_K20_CLEAN_GATE_BASE_A"
  "S2_SFE_K50_CLEAN_PROTO_CAL_B"
  "S2_SFE_K20_SAT_GATE_CAL_A"
  "S2_SFE_K50_SAT_OPENSET_CAL_B"
  "S2_SFE_K50_SAT_PROTO_CAL_A"
  "S2_SFE_K20_SAT_SUPPORT_FILTER_B"
  "S2_SFE_SCENARIO_BALANCED_SUPPORT_A"
  "S2_SFE_CHANNEL_AWARE_SCORE_FUSION_B"
  "S2_FTRC_LOGIT_CAL_K20_SAFE_A"
  "S2_FTRC_LOGIT_CAL_K50_ANCHOR_B"
  "S2_FTRC_FEATURE_RESIDUAL_K20_SAFER_A"
  "S2_FTRC_FEATURE_RESIDUAL_PROTO_REPLAY_B"
  "S2_FTRC_LOGIT_LORA_K20_LOW_ALPHA_A"
  "S2_FTRC_EWC_OLD_GUARD_LORA_B"
  "S2_DEPLOY_FORGET_TELEMETRY_PROFILE_A"
  "S2_SCORETABLE_MANIFEST_DIAGNOSTIC_B"
)
declare -a CAND_GPU=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)
declare -a CAND_KIND=("sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "sfe")
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do
  CAND_STATUS+=("queued")
  CAND_PID+=("")
done

describe_candidate() {
  local i="$1"
  local cid="${CAND_ID[$i]}"
  case "${cid}" in
    S2_SFE_K20_CLEAN_GATE_BASE_A)
      echo "sfe channel=clean shots=20 seed=2031 gates=mahal6,combined065,combined070,openmax095"
      ;;
    S2_SFE_K50_CLEAN_PROTO_CAL_B)
      echo "sfe channel=clean shots=50 seed=2032 prototype-calibration control"
      ;;
    S2_SFE_K20_SAT_GATE_CAL_A)
      echo "sfe channel=satellite shots=20 seed=2031 paired with GPU0/A"
      ;;
    S2_SFE_K50_SAT_OPENSET_CAL_B)
      echo "sfe channel=satellite shots=50 seed=2032 open-set/prototype calibration"
      ;;
    S2_SFE_K50_SAT_PROTO_CAL_A)
      echo "sfe channel=satellite shots=50 seed=2033 source_proto_per_tx=30"
      ;;
    S2_SFE_K20_SAT_SUPPORT_FILTER_B)
      echo "sfe channel=satellite shots=20 seed=2034 max_samples_per_tx=120 support-filter proxy"
      ;;
    S2_SFE_SCENARIO_BALANCED_SUPPORT_A)
      echo "sfe channel=satellite shots=20 seed=2035 all-scenario support diagnostic"
      ;;
    S2_SFE_CHANNEL_AWARE_SCORE_FUSION_B)
      echo "sfe channel=satellite shots=20 seed=2036 hard-scenario score-fusion proxy"
      ;;
    S2_FTRC_LOGIT_CAL_K20_SAFE_A)
      echo "ftrc adapter=logit_calibration k=20 lr=1e-5 anchor=0.30 alpha=1.0 rank=4 epochs=20 steps=10"
      ;;
    S2_FTRC_LOGIT_CAL_K50_ANCHOR_B)
      echo "ftrc adapter=logit_calibration k=50 lr=1e-5 anchor=0.45 alpha=1.0 rank=4 epochs=20 steps=10"
      ;;
    S2_FTRC_FEATURE_RESIDUAL_K20_SAFER_A)
      echo "ftrc adapter=feature_residual k=20 lr=5e-6 anchor=0.25 alpha=0.10 rank=4 epochs=20 steps=10"
      ;;
    S2_FTRC_FEATURE_RESIDUAL_PROTO_REPLAY_B)
      echo "ftrc adapter=feature_residual k=50 lr=5e-6 anchor=0.45 alpha=0.10 rank=4 epochs=20 steps=10 proto-replay-proxy"
      ;;
    S2_FTRC_LOGIT_LORA_K20_LOW_ALPHA_A)
      echo "ftrc adapter=logit_lora k=20 lr=5e-6 anchor=0.30 alpha=0.10 rank=2 epochs=20 steps=10"
      ;;
    S2_FTRC_EWC_OLD_GUARD_LORA_B)
      echo "ftrc adapter=logit_lora k=50 lr=5e-6 anchor=0.55 alpha=0.10 rank=2 epochs=20 steps=10 ewc-proxy"
      ;;
    S2_DEPLOY_FORGET_TELEMETRY_PROFILE_A)
      echo "ftrc adapter=logit_calibration k=20 lr=1e-5 anchor=0.30 alpha=1.0 rank=4 epochs=3 steps=5 telemetry-profile"
      ;;
    S2_SCORETABLE_MANIFEST_DIAGNOSTIC_B)
      echo "sfe channel=satellite shots=20 seed=2037 max_samples_per_tx=100 query=30 diagnostic score-table"
      ;;
  esac
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

launch_candidate() {
  local i="$1"
  local cid="${CAND_ID[$i]}"
  local gpu="${CAND_GPU[$i]}"
  local log_path="${LOG_ROOT}/${cid}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[S2-DRY-RUN] cid=${cid} gpu=${gpu} $(describe_candidate "$i") log=${log_path}"
    CAND_STATUS[$i]="dry_run"
    return 0
  fi
  if [[ "${cid}" == "S2_SFE_K20_CLEAN_GATE_BASE_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "clean" "20" "2031" "3201" "200" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_K50_CLEAN_PROTO_CAL_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "clean" "50" "2032" "3202" "260" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_K20_SAT_GATE_CAL_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "satellite" "20" "2031" "3203" "200" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_K50_SAT_OPENSET_CAL_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "satellite" "50" "2032" "3204" "260" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_K50_SAT_PROTO_CAL_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "satellite" "50" "2033" "3205" "320" "30" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_K20_SAT_SUPPORT_FILTER_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "satellite" "20" "2034" "3206" "120" "20" "40" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_SCENARIO_BALANCED_SUPPORT_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; TARGET_SCENARIOS="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" run_sfe_bundle "${cid}" "satellite" "20" "2035" "3207" "180" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SFE_CHANNEL_AWARE_SCORE_FUSION_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; TARGET_SCENARIOS="storm_mp,mixed_orbit" run_sfe_bundle "${cid}" "satellite" "20" "2036" "3208" "220" "20" "50" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_LOGIT_CAL_K20_SAFE_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "logit_calibration" "20" "1e-5" "0.30" "1.0" "4" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_LOGIT_CAL_K50_ANCHOR_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "logit_calibration" "50" "1e-5" "0.45" "1.0" "4" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_FEATURE_RESIDUAL_K20_SAFER_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "feature_residual" "20" "5e-6" "0.25" "0.10" "4" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_FEATURE_RESIDUAL_PROTO_REPLAY_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "feature_residual" "50" "5e-6" "0.45" "0.10" "4" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_LOGIT_LORA_K20_LOW_ALPHA_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "logit_lora" "20" "5e-6" "0.30" "0.10" "2" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_FTRC_EWC_OLD_GUARD_LORA_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "logit_lora" "50" "5e-6" "0.55" "0.10" "2" "20" "10" "5" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_DEPLOY_FORGET_TELEMETRY_PROFILE_A" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "logit_calibration" "20" "1e-5" "0.30" "1.0" "4" "3" "5" "1" > "${log_path}" 2>&1) &
  elif [[ "${cid}" == "S2_SCORETABLE_MANIFEST_DIAGNOSTIC_B" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "satellite" "20" "2037" "3209" "100" "20" "30" > "${log_path}" 2>&1) &
  else
    echo "[ERROR] unknown candidate ${cid}" >&2
    exit 2
  fi
  local pid="$!"
  CAND_PID[$i]="${pid}"
  CAND_STATUS[$i]="running"
  event_row "${cid}" "LAUNCHED" "gpu=${gpu}" "pid=${pid}" "log=${log_path}"
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
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[S2-CAPACITY] cid=${CAND_ID[$i]} gpu=${gpu} compute_count=${count} planned_running=${planned} effective_count=${effective}"
    else
      event_row "${CAND_ID[$i]}" "CAPACITY" "gpu=${gpu}" "compute_count=${count}" "effective_count=${effective}"
    fi
    if (( effective < MAX_ACTIVE_PER_GPU )); then
      launch_candidate "${i}"
    else
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[S2-DEFERRED-CAPACITY] cid=${CAND_ID[$i]} gpu=${gpu}"
      else
        event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${gpu}" "compute_count=${count}" "effective_count=${effective}"
      fi
    fi
  done
}

log_msg "[S2-SCHEDULER] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU}"
log_msg "[S2-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} $(describe_candidate "$i")"
done

try_launch_queued

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[S2-DRY-RUN-COMPLETE]"
  exit 0
fi

status=0
for i in "${!CAND_ID[@]}"; do
  if [[ "${CAND_STATUS[$i]}" != "running" ]]; then
    continue
  fi
  pid="${CAND_PID[$i]}"
  cid="${CAND_ID[$i]}"
  gpu="${CAND_GPU[$i]}"
  if wait "${pid}"; then
    event_row "${cid}" "COMPLETE" "gpu=${gpu}" "pid=${pid}" "status=0"
  else
    rc=$?
    event_row "${cid}" "FAILED" "gpu=${gpu}" "pid=${pid}" "status=${rc}"
    status="${rc}"
  fi
done
log_msg "[S2-SCHEDULER-END] status=${status}"
exit "${status}"
