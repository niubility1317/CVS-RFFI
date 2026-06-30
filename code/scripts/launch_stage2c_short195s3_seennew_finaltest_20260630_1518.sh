#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2c_short195s3_seennew_finaltest_20260630_1518}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_RECEIVER_LABEL="${TARGET_RECEIVER_LABEL:-20-1}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-1-16,1-18,1-14}"
OA_MSE_UNKNOWN_TX_IDS="${OA_MSE_UNKNOWN_TX_IDS:-10-1,10-10,10-11}"
SAMPLE_RATE_HZ="${SAMPLE_RATE_HZ:-25000000}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[STAGE2C-SHORT195S3] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-SHORT195S3] teacher=${TEACHER_CKPT}"
echo "[STAGE2C-SHORT195S3] policy=stage2c_old_plus_seen_new_enrollment adapter_train_then_final_test_only no_alpha_eval_sweep"
echo "[STAGE2C-SHORT195S3] target_receiver=${TARGET_RECEIVER_LABEL} new_tx=${TARGET_NEW_TX_IDS} unknown_tx=${OA_MSE_UNKNOWN_TX_IDS}"

run_case() {
  local name="$1"
  local gpu="$2"
  local seed="$3"
  local multiproto="$4"
  local out_dir="${RUNS_ROOT}/${name}"
  local log_path="${LOG_ROOT}/${name}.out"
  local multiproto_flags=()

  if [[ "${multiproto}" == "1" ]]; then
    multiproto_flags=(
      --oa_mse_multiproto_score
      --multiproto_topk 5
      --multiproto_temperature 0.12
      --multiproto_score_weight 0.65
      --oa_mse_mixture_consistency_gate
      --mixture_consistency_min_cos 0.25
      --mixture_consistency_max_residual 1.20
      --mixture_consistency_min_margin -0.08
      --mixture_consistency_action uncertain
    )
  fi

  echo "[STAGE2C-SHORT195S3-CANDIDATE] id=${name} gpu=${gpu} protocol=Stage2-C k_old=10 k_new=10 multiproto=${multiproto}"
  local cmd=(
    env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" bash -lc
    "set -euo pipefail; \
mkdir -p '${out_dir}'; \
'${PYTHON}' -u '${ROOT}/code/export_spaceborne_features.py' \
  --ckpt '${TEACHER_CKPT}' \
  --wisig_pkl '${WISIG_PKL}' \
  --new_wisig_pkl '${NEW_WISIG_PKL}' \
  --out_npz '${out_dir}/features.npz' \
  --feature_name z_id \
  --source_tx_ids '${SOURCE_TX_IDS}' \
  --source_rxs '${CEN51_TRAIN_RXS}' \
  --target_old_tx_ids '${TARGET_OLD_TX_IDS}' \
  --target_old_rxs '${TARGET_RECEIVER_LABEL}' \
  --target_old_channel_view satellite \
  --new_tx_ids '${TARGET_NEW_TX_IDS}' \
  --new_rxs '${TARGET_RECEIVER_LABEL}' \
  --unknown_tx_ids '${OA_MSE_UNKNOWN_TX_IDS}' \
  --star_ground_channel_impl simplified_leo_residual \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --sample_rate_hz '${SAMPLE_RATE_HZ}' \
  --max_samples_per_combo 0 \
  --max_samples_per_tx 120 \
  --batch_size 512 \
  --device cuda:0 \
  --seed '${seed}'; \
'${PYTHON}' -u '${ROOT}/code/eval_spaceborne_fewshot.py' \
  --protocol sfe \
  --feature_npz '${out_dir}/features.npz' \
  --output_json '${out_dir}/metrics.json' \
  --manifest_json '${out_dir}/manifest.json' \
  --score_table_csv '${out_dir}/score_table.csv' \
  --source_tx_ids '${SOURCE_TX_IDS}' \
  --target_old_tx_ids '${TARGET_OLD_TX_IDS}' \
  --target_old_support_per_tx 10 \
  --target_old_query_per_tx 30 \
  --new_tx_ids '${TARGET_NEW_TX_IDS}' \
  --unknown_tx_ids '${OA_MSE_UNKNOWN_TX_IDS}' \
  --shots 10 \
  --source_proto_per_tx 48 \
  --source_query_per_tx 40 \
  --query_per_tx 30 \
  --unknown_threshold 0.96 \
  --gate_mode oa_mse \
  --openmax_tail_size 20 \
  --openmax_quantile 1.0 \
  --openmax_min_threshold 0.1 \
  --oa_mse_adapter_rank 2 \
  --oa_mse_adapter_kind low_rank \
  --oa_mse_adapter_steps 80 \
  --oa_mse_adapter_selection_policy final \
  --oa_mse_source_anchor_weight 0.05 \
  --oa_mse_source_ce_weight 1.10 \
  --oa_mse_unknown_moat_weight 0.05 \
  --oa_mse_unknown_moat_margin 0.30 \
  --pseudo_unknown_samples_per_pair 4 \
  --pseudo_unknown_offset_scale 0.15 \
  --pseudo_unknown_source_boundary_samples_per_pair 8 \
  --pseudo_unknown_source_boundary_offset_scale 0.18 \
  --pseudo_unknown_target_shift_samples_per_class 4 \
  --pseudo_unknown_target_shift_offset_scale 0.22 \
  --pseudo_unknown_target_halo_samples_per_class 4 \
  --pseudo_unknown_target_halo_offset_scale 0.32 \
  --pseudo_unknown_target_ring_samples_per_class 6 \
  --pseudo_unknown_target_ring_offset_scale 0.38 \
  --oa_mse_old_bridge_weight 0.18 \
  --old_bridge_samples_per_class 4 \
  --old_bridge_max_mix 0.78 \
  --oa_mse_support_contrast_weight 0.05 \
  --oa_mse_support_center_ce_weight 0.18 \
  --support_center_temperature 0.35 \
  --oa_mse_soft_proto_weight 0.12 \
  --soft_proto_topk 3 \
  --soft_proto_temperature 0.20 \
  --oa_mse_soft_proto_boundary_weight 0.05 \
  --soft_proto_boundary_margin 0.10 \
  --oa_mse_anchor_density_gate \
  --anchor_density_topk 3 \
  --anchor_density_temperature 0.10 \
  --anchor_density_min_quantile 0.02 \
  --anchor_density_margin_quantile 0.02 \
  --anchor_density_gate_action uncertain \
  --oa_mse_class_envelope_gate \
  --class_envelope_evidence_quantile 0.03 \
  --class_envelope_residual_quantile 0.98 \
  --class_envelope_score_quantile 0.03 \
  --class_envelope_margin_quantile 0.03 \
  --class_envelope_evidence_slack 0.08 \
  --class_envelope_residual_slack 0.08 \
  --class_envelope_score_slack 0.10 \
  --class_envelope_margin_slack 0.08 \
  --class_envelope_min_failures 2 \
  --class_envelope_gate_action uncertain \
  --old_acc_target 0.80 \
  --seen_new_acc_target 0.75 \
  --seed '${seed}' \
  ${multiproto_flags[*]}"
  )
  printf "[STAGE2C-SHORT195S3-CMD] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" > "${log_path}" 2>&1 &
    echo "[STAGE2C-SHORT195S3-LAUNCHED] id=${name} pid=$! gpu=${gpu} log=${log_path}"
  fi
}

run_case "SHORT195S3_STAGE2C_FINALTEST_BASELINE_K10NEW10" "0" "362101" "0"
run_case "SHORT195S3_STAGE2C_FINALTEST_MULTIPROTO_K10NEW10" "1" "362101" "1"

if [[ "${DRY_RUN}" != "1" ]]; then
  wait
fi
