#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_r8_qknn8_riskgate_20260706}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
PHASE1_RUN_ROOT="${PHASE1_RUN_ROOT:-${ROOT}/runs/phase1_epoc_r8_paog_20260706}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
SOURCE_RXS="${SOURCE_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_RECEIVERS="${TARGET_RECEIVERS:-20-1,3-19,7-14,7-7,8-8}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-12-7,13-14,13-19,13-3,13-7,14-11,14-12,14-13}"

K_SHOT="${K_SHOT:-8}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-20}"
QKNN_K="${QKNN_K:-8}"
MAX_SAMPLES_PER_COMBO="${MAX_SAMPLES_PER_COMBO:-80}"
MAX_SAMPLES_PER_TX="${MAX_SAMPLES_PER_TX:-0}"
MAX_EVENT_BYTES="${MAX_EVENT_BYTES:-1152}"
MAX_EVENT_LATENCY_MS="${MAX_EVENT_LATENCY_MS:-20}"
COLLAB_GROUP_POLICY="${COLLAB_GROUP_POLICY:-available_up_to_k}"
PARTIAL_COLLAB_MIN_RECEIVERS="${PARTIAL_COLLAB_MIN_RECEIVERS:-1}"
EVENT_ALIGNMENT_POLICY="${EVENT_ALIGNMENT_POLICY:-receiver_domain_ranked}"
SEED="${SEED:-4070618}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

case_config() {
  case "$1" in
    RADIUS)
      echo "EPOC_R8_PAOG_RADIUS_ENERGY|2|706801|${PHASE1_RUN_ROOT}/EPOC_R8_PAOG_RADIUS_ENERGY/best_joint_safe_ssdg.pth"
      ;;
    SHELL)
      echo "EPOC_R8_PAOG_SHELL_BALANCED|3|706811|${PHASE1_RUN_ROOT}/EPOC_R8_PAOG_SHELL_BALANCED/best_joint_safe_ssdg.pth"
      ;;
    *) echo "[ERROR] unknown case: $1" >&2; exit 2 ;;
  esac
}

cases=()
case "${ONLY}" in
  all) cases=(RADIUS SHELL) ;;
  RADIUS|SHELL) cases=("${ONLY}") ;;
  *) echo "[ERROR] --only must be all, RADIUS, or SHELL" >&2; exit 2 ;;
esac

echo "[R8-QKNN8-RISKGATE] run_id=${RUN_ID} dry_run=${DRY_RUN} only=${ONLY}"
echo "[R8-QKNN8-RISKGATE] target_receivers=${TARGET_RECEIVERS}"
echo "[R8-QKNN8-RISKGATE] qknn_k=${QKNN_K} k_shot=${K_SHOT} query_per_class=${QUERY_PER_CLASS}"
echo "[R8-QKNN8-RISKGATE] fusion_policy=old_protected_unknown_confirm_cvs unknown_gate_mode=support_envelope_full"
echo "[R8-QKNN8-RISKGATE] risk_components=score,radius,margin,mahalanobis,evt,oldness,virtual_unknown,class_negative,class_shell"
echo "[R8-QKNN8-RISKGATE] protocol=Stage2-C unknown_query_eval_only=true ground_training_unknown_seen=false"
echo "[R8-QKNN8-RISKGATE] stage2_success_claim=0 deployment_success_claim=0"

launch_case() {
  local short_name="$1"
  local config case_id gpu seed ckpt out_dir feature_npz output_json evidence_csv log_path
  config="$(case_config "${short_name}")"
  IFS='|' read -r case_id gpu seed ckpt <<< "${config}"
  out_dir="${RUNS_ROOT}/${case_id}"
  feature_npz="${out_dir}/features_stage2c_leo_multirx.npz"
  output_json="${out_dir}/qknn8_riskgate.json"
  evidence_csv="${out_dir}/qknn8_riskgate_evidence.csv"
  log_path="${LOG_ROOT}/${case_id}.out"

  local export_cmd=(
    env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"
    --ckpt "${ckpt}"
    --wisig_pkl "${WISIG_PKL}"
    --new_wisig_pkl "${NEW_WISIG_PKL}"
    --out_npz "${feature_npz}"
    --feature_name z_id
    --source_tx_ids "${SOURCE_TX_IDS}"
    --source_rxs "${SOURCE_RXS}"
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}"
    --target_old_rxs "${TARGET_RECEIVERS}"
    --target_old_channel_view satellite
    --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --new_tx_ids "${TARGET_NEW_TX_IDS}"
    --new_rxs "${TARGET_RECEIVERS}"
    --target_new_channel_view satellite
    --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --unknown_tx_ids "${UNKNOWN_TX_IDS}"
    --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}"
    --proxy_unknown_rxs "${SOURCE_RXS}"
    --proxy_unknown_channel_view satellite
    --proxy_unknown_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --star_ground_channel_impl simplified_leo_residual
    --wisig_equalized 1
    --wisig_domain rx_day
    --wisig_out_len 256
    --sample_rate_hz 25000000
    --sat_fs_hz 25000000
    --sat_fc_hz 2462000000
    --max_samples_per_combo "${MAX_SAMPLES_PER_COMBO}"
    --max_samples_per_tx "${MAX_SAMPLES_PER_TX}"
    --batch_size 512
    --device cuda:0
    --seed "${seed}"
  )

  local eval_cmd=(
    env PYTHONPATH="${ROOT}/code:${ROOT}/code/scripts:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u "${ROOT}/code/scripts/phase2_collaborative_open_set_qknn_eval.py"
    --feature_npz "${feature_npz}"
    --output_json "${output_json}"
    --output_evidence_csv "${evidence_csv}"
    --collab_counts all
    --collab_group_policy "${COLLAB_GROUP_POLICY}"
    --partial_collab_min_receivers "${PARTIAL_COLLAB_MIN_RECEIVERS}"
    --k_shot "${K_SHOT}"
    --query_per_class "${QUERY_PER_CLASS}"
    --qknn_k "${QKNN_K}"
    --support_selection_policy stable_first
    --support_calibration_mode leave_one_out
    --event_alignment_policy "${EVENT_ALIGNMENT_POLICY}"
    --unknown_gate_mode support_envelope_full
    --fusion_policy old_protected_unknown_confirm_cvs
    --prototype_score_blend 0.35
    --mahalanobis_score_blend 0.35
    --mahalanobis_quantile 0.90
    --mahalanobis_slack 0.02
    --evt_tail_quantile 0.80
    --evt_tail_slack 0.02
    --radius_quantile 0.92
    --radius_slack 0.02
    --score_quantile 0.05
    --margin_quantile 0.05
    --virtual_unknown_calibration_enabled
    --virtual_unknown_samples_per_class 4
    --virtual_unknown_risk_enabled
    --virtual_unknown_risk_samples_per_class 4
    --class_negative_risk_enabled
    --class_negative_samples_per_class 4
    --class_negative_combine_mode weak_evidence
    --class_negative_weak_margin 0.02
    --class_negative_weak_pvalue 0.05
    --class_shell_unknown_risk_enabled
    --class_shell_radius_scale 1.18
    --unknown_risk_threshold 0.82
    --accept_margin_threshold 0.06
    --consensus_gap_threshold 0.00
    --consensus_score_threshold 0.00
    --scorer_component_vote_threshold 0.50
    --latency_budget_ms "${MAX_EVENT_LATENCY_MS}"
    --max_event_bytes "${MAX_EVENT_BYTES}"
    --max_event_latency_ms "${MAX_EVENT_LATENCY_MS}"
    --evidence_packet_bytes 56
    --seed "${seed}"
  )

  echo "[R8-QKNN8-RISKGATE-CASE] case=${case_id} gpu=${gpu} ckpt=${ckpt}"
  printf "[R8-QKNN8-RISKGATE-EXPORT-CMD] "
  printf "%q " "${export_cmd[@]}"
  printf "\n"
  printf "[R8-QKNN8-RISKGATE-EVAL-CMD] "
  printf "%q " "${eval_cmd[@]}"
  printf "\n"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    return 3
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}"
  (
    set -euo pipefail
    "${export_cmd[@]}"
    "${eval_cmd[@]}"
    echo "[R8-QKNN8-RISKGATE-DONE] case=${case_id} output_json=${output_json}"
  ) > "${log_path}" 2>&1 &
  echo "[R8-QKNN8-RISKGATE-LAUNCHED] case=${case_id} pid=$! gpu=${gpu} log=${log_path} out_dir=${out_dir}"
}

for case_name in "${cases[@]}"; do
  launch_case "${case_name}"
done
