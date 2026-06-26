#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next16b_20260615_051132}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
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

log_msg() { if [[ "${DRY_RUN}" == "1" ]]; then echo "$@"; else echo "$@" | tee -a "${SCHED_LOG}"; fi; }
event_row() { local row; row="$(printf "%s\t%s\t%s\t%s\t%s\t%s" "$(date -Is)" "$1" "$2" "$3" "$4" "$5")"; if [[ "${DRY_RUN}" == "1" ]]; then echo "${row}"; else echo "${row}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"; fi; }
gpu_process_count() { local gpu="$1"; if [[ "${DRY_RUN}" == "1" ]]; then echo 0; return 0; fi; nvidia-smi pmon -c 1 2>/dev/null | awk -v g="${gpu}" '$1 == g && $2 ~ /^[0-9]+$/ && $3 == "C" {count += 1} END {print count + 0}'; }

run_sfe_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" shots="$8" source_proto="$9" query_per_tx="${10}" seed="${11}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol sfe --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --shots "${shots}" --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2-SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold}"
  "${cmd[@]}"
}

run_sfe_bundle() {
  local cid="$1" channel_view="$2" shots="$3" seed="$4" sat_seed="$5" max_samples_per_tx="$6" source_proto="$7" query_per_tx="$8" scenarios="$9"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-SFE-BEGIN] cid=${cid} channel=${channel_view} shots=${shots} source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS} scenarios=${scenarios}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" --ckpt "${TEACHER_CKPT}" --wisig_pkl "${WISIG_PKL}" --new_wisig_pkl "${NEW_WISIG_PKL}" --out_npz "${out_dir}/features.npz" --feature_name z_id --source_tx_ids "${SOURCE_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo 0 --max_samples_per_tx "${max_samples_per_tx}" --batch_size 512 --device cuda:0 --seed "${seed}" --target_new_channel_view "${channel_view}" --target_new_sat_scenarios "${scenarios}" --target_new_sat_seed "${sat_seed}"
  run_sfe_eval "${cid}" "mahal_t075_mh6" "mahalanobis" "0.75" "" "6.0" "0.97" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "combined_t070_m005_mh6" "combined" "0.70" "0.05" "6.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "combined_t075_m005_mh6" "combined" "0.75" "0.05" "6.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  run_sfe_eval "${cid}" "openmax_t075_q097" "openmax" "0.75" "" "" "0.97" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  echo "[S2-SFE-END] cid=${cid}"
}

run_ftrc_candidate() {
  local cid="$1" adapter="$2" k="$3" lr="$4" anchor="$5" alpha="$6" rank="$7" epochs="$8" steps="$9" eval_detail="${10}" seed="${11}"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-FTRC-BEGIN] cid=${cid} adapter=${adapter} k=${k} lr=${lr} anchor=${anchor} alpha=${alpha} rank=${rank} epochs=${epochs} steps=${steps} seed=${seed}"
  "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" --teacher_ckpt "${TEACHER_CKPT}" --output_dir "${out_dir}" --dataset wisig --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --wisig_train_ratio 0.1 --wisig_guard_gap 8 --wisig_train_days 0,1 --wisig_test_days 2,3 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_test_rxs 7,8,9,10,11 --target_loader "${TARGET_LOADER}" --target_channel_view satellite --target_label_mode labeled --target_samples_per_rx_tx "${k}" --target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --epochs "${epochs}" --adapt_steps_per_epoch "${steps}" --target_batch_size 32 --lr_adapt "${lr}" --entropy_weight 0 --consistency_weight 0 --pseudo_weight 0 --anchor_weight "${anchor}" --anchor_temperature 2.5 --eval_detail_every "${eval_detail}" --target_adapter_type "${adapter}" --adapter_rank "${rank}" --adapter_bottleneck 16 --adapter_alpha "${alpha}" --adapter_dropout 0.0 --freeze_base_stats true --update_norm false --update_classifier false --rollback_enabled true --eval_sat_channel true --eval_sat_on "${TARGET_LOADER}" --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches 0 --eval_max_batches 0 --seed "${seed}" --run_name "${cid}"
  echo "[S2-FTRC-END] cid=${cid}"
}

declare -a CAND_ID=("S2N17_SFE_K50_CLEAN_OPENMAX_A" "S2N17_SFE_K100_CLEAN_PROTO_B" "S2N17_SFE_K20_SAT_STRICT_GATE_A" "S2N17_SFE_K50_SAT_OPENSET_B" "S2N17_SFE_K20_SAT_FILTER_A" "S2N17_SFE_K50_SAT_FILTER_B" "S2N17_SFE_SCEN_BALANCE_A" "S2N17_SFE_HARD_SCENE_FUSION_B" "S2N17_FTRC_LOGIT_CAL_K50_CONFIRM_A" "S2N17_FTRC_LOGIT_CAL_K100_ANCHOR_B" "S2N17_FTRC_FEATURE_RESIDUAL_K50_A" "S2N17_FTRC_FEATURE_RESIDUAL_K100_B" "S2N17_FTRC_LORA_K50_R4_A" "S2N17_FTRC_LORA_K100_OLDGUARD_B" "S2N17_DEPLOY_TELEMETRY_K50_A" "S2N17_SCORETABLE_STRICT_DIAG_B")
declare -a CAND_GPU=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)
declare -a CAND_KIND=("sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "sfe")
declare -a CAND_SLOT=("GPU0/A" "GPU0/B" "GPU1/A" "GPU1/B" "GPU2/A" "GPU2/B" "GPU3/A" "GPU3/B" "GPU4/A" "GPU4/B" "GPU5/A" "GPU5/B" "GPU6/A" "GPU6/B" "GPU7/A" "GPU7/B")
declare -a CAND_DESC=("K50 clean open-set confirm" "K100 clean prototype calibration" "satellite strict gate calibration" "satellite K50 strict open-set" "support quality filtering proxy" "K50 support filtering" "scenario-balanced support diagnostic" "channel-aware hard-scene proxy" "multi-seed logit calibration confirm" "K100 logit calibration" "feature residual stronger diagnostic" "K100 feature residual" "LoRA rank/alpha stress" "K100 LoRA old-guard proxy" "deployment telemetry refresh" "score-table strict diagnostic")
declare -a SFE_CHANNEL=("clean" "clean" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "-" "-" "-" "-" "-" "-" "-" "satellite")
declare -a SFE_SHOTS=(50 100 20 50 20 50 20 20 0 0 0 0 0 0 0 50)
declare -a SFE_SEED=(2038 2039 2038 2039 2040 2041 2042 2043 0 0 0 0 0 0 0 2045)
declare -a SFE_SAT_SEED=(3301 3302 3303 3304 3305 3306 3307 3308 0 0 0 0 0 0 0 3309)
declare -a SFE_MAX_SAMPLES=(260 420 200 280 120 160 180 220 0 0 0 0 0 0 0 100)
declare -a SFE_SOURCE_PROTO=(20 30 20 20 20 30 20 20 0 0 0 0 0 0 0 20)
declare -a SFE_QUERY=(50 50 50 50 40 40 50 50 0 0 0 0 0 0 0 30)
declare -a SFE_SCENARIOS=("clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "storm_mp,mixed_orbit" "-" "-" "-" "-" "-" "-" "-" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
declare -a FTRC_ADAPTER=("-" "-" "-" "-" "-" "-" "-" "-" "logit_calibration" "logit_calibration" "feature_residual" "feature_residual" "logit_lora" "logit_lora" "logit_calibration" "-")
declare -a FTRC_K=(0 0 0 0 0 0 0 0 50 100 50 100 50 100 50 0)
declare -a FTRC_LR=("-" "-" "-" "-" "-" "-" "-" "-" "1e-5" "1e-5" "1e-5" "1e-5" "1e-5" "5e-6" "1e-5" "-")
declare -a FTRC_ANCHOR=("-" "-" "-" "-" "-" "-" "-" "-" "0.45" "0.40" "0.35" "0.30" "0.45" "0.60" "0.35" "-")
declare -a FTRC_ALPHA=("-" "-" "-" "-" "-" "-" "-" "-" "1.0" "1.0" "0.20" "0.20" "0.20" "0.15" "1.0" "-")
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 20 20 20 20 20 20 3 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 10 10 10 10 10 10 5 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 5 5 5 5 5 5 1 0)
declare -a FTRC_SEED=(0 0 0 0 0 0 0 0 2038 2039 2040 2041 2042 2043 2044 0)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

planned_running_count() { local gpu="$1" i count=0; for i in "${!CAND_ID[@]}"; do if [[ "${CAND_STATUS[$i]}" == "running" && "${CAND_GPU[$i]}" == "${gpu}" ]]; then count=$((count + 1)); fi; done; echo "${count}"; }

launch_candidate() {
  local i="$1"
  local cid="${CAND_ID[$i]}"
  local gpu="${CAND_GPU[$i]}"
  local kind="${CAND_KIND[$i]}"
  local log_path="${LOG_ROOT}/${cid}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[S2-DRY-RUN] cid=${cid} slot=${CAND_SLOT[$i]} gpu=${gpu} kind=${kind} desc=${CAND_DESC[$i]} log=${log_path}"; CAND_STATUS[$i]="dry_run"; return 0; fi
  if [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "${SFE_CHANNEL[$i]}" "${SFE_SHOTS[$i]}" "${SFE_SEED[$i]}" "${SFE_SAT_SEED[$i]}" "${SFE_MAX_SAMPLES[$i]}" "${SFE_SOURCE_PROTO[$i]}" "${SFE_QUERY[$i]}" "${SFE_SCENARIOS[$i]}" > "${log_path}" 2>&1) &
  else
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "${FTRC_ADAPTER[$i]}" "${FTRC_K[$i]}" "${FTRC_LR[$i]}" "${FTRC_ANCHOR[$i]}" "${FTRC_ALPHA[$i]}" "${FTRC_RANK[$i]}" "${FTRC_EPOCHS[$i]}" "${FTRC_STEPS[$i]}" "${FTRC_EVAL_DETAIL[$i]}" "${FTRC_SEED[$i]}" > "${log_path}" 2>&1) &
  fi
  local pid="$!"; CAND_PID[$i]="${pid}"; CAND_STATUS[$i]="running"; event_row "${cid}" "LAUNCHED" "gpu=${gpu}" "pid=${pid}" "log=${log_path}"
}

try_launch_queued() {
  local i gpu count planned effective
  for i in "${!CAND_ID[@]}"; do
    [[ "${CAND_STATUS[$i]}" != "queued" ]] && continue
    gpu="${CAND_GPU[$i]}"; count="$(gpu_process_count "${gpu}")"; planned="$(planned_running_count "${gpu}")"; effective=$((count + planned))
    if [[ "${DRY_RUN}" == "1" ]]; then echo "[S2-CAPACITY] cid=${CAND_ID[$i]} gpu=${gpu} compute_count=${count} planned_running=${planned} effective_count=${effective}"; else event_row "${CAND_ID[$i]}" "CAPACITY" "gpu=${gpu}" "compute_count=${count}" "effective_count=${effective}"; fi
    if (( effective < MAX_ACTIVE_PER_GPU )); then launch_candidate "${i}"; else if [[ "${DRY_RUN}" == "1" ]]; then echo "[S2-DEFERRED-CAPACITY] cid=${CAND_ID[$i]} gpu=${gpu}"; else event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${gpu}" "compute_count=${count}" "effective_count=${effective}"; fi; fi
  done
}

log_msg "[S2-SCHEDULER] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU}"
log_msg "[S2-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
for i in "${!CAND_ID[@]}"; do log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} desc=${CAND_DESC[$i]}"; done
try_launch_queued
if [[ "${DRY_RUN}" == "1" ]]; then echo "[S2-DRY-RUN-COMPLETE]"; exit 0; fi
status=0
for i in "${!CAND_ID[@]}"; do
  [[ "${CAND_STATUS[$i]}" != "running" ]] && continue
  pid="${CAND_PID[$i]}"; cid="${CAND_ID[$i]}"; gpu="${CAND_GPU[$i]}"
  if wait "${pid}"; then event_row "${cid}" "COMPLETE" "gpu=${gpu}" "pid=${pid}" "status=0"; else rc=$?; event_row "${cid}" "FAILED" "gpu=${gpu}" "pid=${pid}" "status=${rc}"; status="${rc}"; fi
done
log_msg "[S2-SCHEDULER-END] status=${status}"
exit "${status}"
