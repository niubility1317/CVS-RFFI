#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-1-16,1-18}"
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

echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] teacher=${TEACHER_CKPT}"
echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] target_new=${TARGET_NEW_TX_IDS}"
echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] target_receivers=3-19,7-14,7-7,8-8"
echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] unknown_policy=excluded_from_export_eval_and_success_metrics"
echo "[ADV3B02-RXSWEEP-2NEW-NOUNK] success_target=old_acc>=0.80 seen_new_acc>=0.65 multi_new_classes=2"

run_case() {
  local name="$1"
  local gpu="$2"
  local seed="$3"
  local target_receiver="$4"
  local old80_mode="$5"
  local old80_policy="$6"
  local out_dir="${RUNS_ROOT}/${name}"
  local log_path="${LOG_ROOT}/${name}.out"
  local old80_flags=()

  if [[ "${old80_mode}" != "disabled" ]]; then
    old80_flags=(
      --oa_mse_old80_head_mode "${old80_mode}"
      --old80_head_apply_policy "${old80_policy}"
      --old80_head_fusion_rho 0.75
      --old80_head_knn_k 3
    )
  fi

  echo "[ADV3B02-RXSWEEP-2NEW-NOUNK-CANDIDATE] id=${name} gpu=${gpu} seed=${seed} target_receiver=${target_receiver} old80_mode=${old80_mode} old80_policy=${old80_policy} target_new=${TARGET_NEW_TX_IDS}"
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
  --target_old_rxs '${target_receiver}' \
  --target_old_channel_view satellite \
  --new_tx_ids '${TARGET_NEW_TX_IDS}' \
  --new_rxs '${target_receiver}' \
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
  --target_old_query_per_tx 40 \
  --new_tx_ids '${TARGET_NEW_TX_IDS}' \
  --shots 10 \
  --source_proto_per_tx 48 \
  --source_query_per_tx 40 \
  --query_per_tx 40 \
  --unknown_threshold 0.0 \
  --gate_mode oa_mse \
  --openmax_tail_size 20 \
  --openmax_quantile 1.0 \
  --openmax_min_threshold 0.0 \
  --oa_mse_adapter_rank 2 \
  --oa_mse_adapter_kind low_rank \
  --oa_mse_adapter_steps 72 \
  --oa_mse_adapter_selection_policy final \
  --oa_mse_source_anchor_weight 0.04 \
  --oa_mse_source_ce_weight 0.85 \
  --oa_mse_unknown_moat_weight 0.0 \
  --oa_mse_unknown_moat_margin 0.0 \
  --pseudo_unknown_samples_per_pair 0 \
  --pseudo_unknown_source_boundary_samples_per_pair 0 \
  --pseudo_unknown_target_shift_samples_per_class 0 \
  --pseudo_unknown_target_halo_samples_per_class 0 \
  --pseudo_unknown_target_ring_samples_per_class 0 \
  --oa_mse_old_bridge_weight 0.32 \
  --old_bridge_samples_per_class 4 \
  --old_bridge_max_mix 0.80 \
  --oa_mse_support_contrast_weight 0.12 \
  --old_support_contrast_negative_margin 0.72 \
  --old_support_contrast_positive_margin 0.86 \
  --oa_mse_support_center_ce_weight 0.42 \
  --support_center_temperature 0.18 \
  --support_center_margin 0.10 \
  --oa_mse_soft_proto_weight 0.32 \
  --soft_proto_topk 3 \
  --soft_proto_temperature 0.12 \
  --oa_mse_soft_proto_boundary_weight 0.10 \
  --soft_proto_boundary_margin 0.12 \
  --oa_mse_known_coverage_weight 0.22 \
  --known_coverage_margin 0.08 \
  --known_coverage_min_affinity 0.20 \
  --known_coverage_max_samples 256 \
  --oa_mse_seen_new_registration_override \
  --seen_new_override_min_evidence_delta -0.24 \
  --seen_new_override_min_anchor_delta -0.18 \
  --seen_new_override_min_affinity_delta -0.18 \
  --seen_new_override_min_residual_delta -0.18 \
  --seen_new_override_min_score_margin -0.22 \
  --seen_new_override_min_seen_vs_old_evidence_margin -0.08 \
  --seen_new_override_max_background_score 1.0 \
  --seen_new_override_max_background_margin 1.0 \
  --old_acc_target 0.80 \
  --seen_new_acc_target 0.65 \
  --seed '${seed}' \
  ${old80_flags[*]}"
  )
  printf "[ADV3B02-RXSWEEP-2NEW-NOUNK-CMD] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" > "${log_path}" 2>&1 &
    echo "[ADV3B02-RXSWEEP-2NEW-NOUNK-LAUNCHED] id=${name} pid=$! gpu=${gpu} log=${log_path}"
  fi
}

run_case "ADV3B02_RXSWEEP_2NEW_RX3_19_BALANCED" "0" "412601" "3-19" "disabled" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX3_19_OLDRESCUE" "1" "412602" "3-19" "support_cv_select" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX7_14_BALANCED" "2" "412603" "7-14" "disabled" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE" "3" "412604" "7-14" "support_cv_select" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX7_7_BALANCED" "4" "412605" "7-7" "disabled" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX7_7_OLDRESCUE" "5" "412606" "7-7" "support_cv_select" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX8_8_BALANCED" "6" "412607" "8-8" "disabled" "rescue_rejected"
run_case "ADV3B02_RXSWEEP_2NEW_RX8_8_OLDRESCUE" "7" "412608" "8-8" "support_cv_select" "rescue_rejected"

if [[ "${DRY_RUN}" != "1" ]]; then
  wait
fi
