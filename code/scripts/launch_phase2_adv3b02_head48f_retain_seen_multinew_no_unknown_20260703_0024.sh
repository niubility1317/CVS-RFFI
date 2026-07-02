#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
ROLLBACK_POLICY_JSON="${ROLLBACK_POLICY_JSON:-${ROOT}/code/configs/phase2_old80_first_rollback_policy.json}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_RECEIVER_LABEL="${TARGET_RECEIVER_LABEL:-20-1}"
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

echo "[ADV3B02-HEAD48F-RETAIN-SEEN] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[ADV3B02-HEAD48F-RETAIN-SEEN] teacher=${TEACHER_CKPT}"
echo "[ADV3B02-HEAD48F-RETAIN-SEEN] target_receiver=${TARGET_RECEIVER_LABEL}"
echo "[ADV3B02-HEAD48F-RETAIN-SEEN] unknown_policy=excluded_actual_unknown_tx"
echo "[ADV3B02-HEAD48F-RETAIN-SEEN] old80_policy=replace_all_except_seen_new_override"

run_case() {
  local name="$1"
  local gpu="$2"
  local seed="$3"
  local new_tx_ids="$4"
  local seen_evidence_delta="$5"
  local seen_anchor_delta="$6"
  local seen_affinity_delta="$7"
  local seen_residual_delta="$8"
  local seen_score_margin="$9"
  local out_dir="${RUNS_ROOT}/${name}"
  local log_path="${LOG_ROOT}/${name}.out"

  echo "[ADV3B02-HEAD48F-RETAIN-SEEN-CANDIDATE] id=${name} gpu=${gpu} seed=${seed} new_tx=${new_tx_ids}"
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
  --new_tx_ids '${new_tx_ids}' \
  --new_rxs '${TARGET_RECEIVER_LABEL}' \
  --star_ground_channel_impl simplified_leo_residual \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --sample_rate_hz '${SAMPLE_RATE_HZ}' \
  --max_samples_per_combo 0 \
  --max_samples_per_tx 80 \
  --batch_size 512 \
  --device cuda:0 \
  --seed '${seed}'; \
'${PYTHON}' -u '${ROOT}/code/eval_spaceborne_fewshot.py' \
  --protocol sfe \
  --rollback_policy_json '${ROLLBACK_POLICY_JSON}' \
  --feature_npz '${out_dir}/features.npz' \
  --output_json '${out_dir}/metrics.json' \
  --manifest_json '${out_dir}/manifest.json' \
  --score_table_csv '${out_dir}/score_table.csv' \
  --source_tx_ids '${SOURCE_TX_IDS}' \
  --target_old_tx_ids '${TARGET_OLD_TX_IDS}' \
  --target_old_support_per_tx 10 \
  --target_old_query_per_tx 40 \
  --new_tx_ids '${new_tx_ids}' \
  --shots 10 \
  --source_proto_per_tx 24 \
  --source_query_per_tx 40 \
  --query_per_tx 40 \
  --unknown_threshold 0.0 \
  --gate_mode oa_mse \
  --openmax_tail_size 20 \
  --openmax_quantile 1.0 \
  --openmax_min_threshold 0.1 \
  --oa_mse_adapter_rank 2 \
  --oa_mse_adapter_kind low_rank \
  --oa_mse_adapter_steps 36 \
  --oa_mse_adapter_selection_policy identity_preserving_cv \
  --oa_mse_adapter_alpha_eval_sweep \
  --oa_mse_source_anchor_weight 0.05 \
  --oa_mse_source_ce_weight 1.28 \
  --oa_mse_unknown_moat_weight 0.0 \
  --oa_mse_unknown_moat_margin 0.2 \
  --pseudo_unknown_samples_per_pair 4 \
  --pseudo_unknown_offset_scale 0.15 \
  --pseudo_unknown_source_boundary_samples_per_pair 14 \
  --pseudo_unknown_source_boundary_offset_scale 0.24 \
  --pseudo_unknown_target_shift_samples_per_class 6 \
  --pseudo_unknown_target_shift_offset_scale 0.22 \
  --pseudo_unknown_target_halo_samples_per_class 12 \
  --pseudo_unknown_target_halo_offset_scale 0.32 \
  --pseudo_unknown_target_ring_samples_per_class 18 \
  --pseudo_unknown_target_ring_offset_scale 0.38 \
  --oa_mse_old_bridge_weight 0.16 \
  --old_bridge_samples_per_class 4 \
  --old_bridge_max_mix 0.78 \
  --oa_mse_support_contrast_weight 0.04 \
  --old_support_contrast_negative_margin 0.78 \
  --old_support_contrast_positive_margin 0.88 \
  --oa_mse_support_center_ce_weight 0.18 \
  --support_center_temperature 0.46 \
  --support_center_margin 0.06 \
  --oa_mse_soft_proto_weight 0.0 \
  --soft_proto_topk 6 \
  --soft_proto_temperature 0.45 \
  --oa_mse_soft_proto_boundary_weight 0.0 \
  --soft_proto_boundary_margin 0.104 \
  --oa_mse_three_way_head_weight 0.072 \
  --three_way_head_temperature 0.102 \
  --three_way_head_known_margin 0.104 \
  --three_way_head_background_margin 0.126 \
  --three_way_head_support_ce_weight 1.3 \
  --three_way_head_pseudo_ce_weight 0.18 \
  --three_way_head_support_background_margin_weight 0.7 \
  --three_way_head_pseudo_margin_weight 0.22 \
  --oa_mse_multiproto_score \
  --multiproto_topk 6 \
  --multiproto_temperature 0.47 \
  --multiproto_score_weight 0.23 \
  --oa_mse_mixture_consistency_gate \
  --mixture_consistency_min_cos 0.52 \
  --mixture_consistency_max_residual 0.404 \
  --mixture_consistency_min_margin 0.045 \
  --mixture_consistency_action uncertain \
  --oa_mse_known_coverage_weight 0.24 \
  --known_coverage_margin 0.04 \
  --known_coverage_min_affinity 0.18 \
  --known_coverage_max_samples 256 \
  --oa_mse_seen_new_registration_override \
  --seen_new_override_min_evidence_delta '${seen_evidence_delta}' \
  --seen_new_override_min_anchor_delta '${seen_anchor_delta}' \
  --seen_new_override_min_affinity_delta '${seen_affinity_delta}' \
  --seen_new_override_min_residual_delta '${seen_residual_delta}' \
  --seen_new_override_min_score_margin '${seen_score_margin}' \
  --seen_new_override_min_seen_vs_old_evidence_margin 0.0 \
  --seen_new_override_max_background_score 1.0 \
  --seen_new_override_max_background_margin 1.0 \
  --oa_mse_old80_head_mode support_cv_select \
  --old80_head_apply_policy replace_all_except_seen_new_override \
  --old80_head_fusion_rho 0.85 \
  --old80_head_knn_k 3 \
  --old_anchor_override_min_quality 0.6 \
  --old_retention_quantile 0.9 \
  --old_acc_target 0.80 \
  --seen_new_acc_target 0.65 \
  --seed '${seed}'"
  )
  printf "[ADV3B02-HEAD48F-RETAIN-SEEN-CMD] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" > "${log_path}" 2>&1 &
    echo "[ADV3B02-HEAD48F-RETAIN-SEEN-LAUNCHED] id=${name} pid=$! gpu=${gpu} log=${log_path}"
  fi
}

run_case "ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_STRICT" "0" "402501" "1-16,1-18" "0.0" "-0.01" "-0.02" "-0.02" "0.0"
run_case "ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_RELAX" "1" "402502" "1-16,1-18" "-0.28" "-0.16" "-0.16" "-0.16" "0.0"
run_case "ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_STRICT" "2" "402503" "1-16,1-18,1-14" "0.0" "-0.01" "-0.02" "-0.02" "0.0"
run_case "ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_RELAX" "3" "402504" "1-16,1-18,1-14" "-0.28" "-0.16" "-0.16" "-0.16" "0.0"

if [[ "${DRY_RUN}" != "1" ]]; then
  wait
fi
