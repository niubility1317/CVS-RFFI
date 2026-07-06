#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-5}"
PROFILE="${PROFILE:-HP08}"
HARD_PAIR_WEIGHT="${HARD_PAIR_WEIGHT:-0.08}"
HARD_OLD_WEIGHT="${HARD_OLD_WEIGHT:-0.04}"
EXPORT_SEED="${EXPORT_SEED:-4070391}"
EXPORT_REFERENCE_NPZ="${EXPORT_REFERENCE_NPZ:-}"
SATELLITE_TTA_POLICY="${SATELLITE_TTA_POLICY:-none}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/phase2_qknn_hardpair_n20_20260706}"

OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2"
SOURCE_RXS="0,1,2,3,4,5,6"
TARGET_RX="7-14"
PROXY_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
PROXY_TX="1-8,10-11,10-17,10-4,10-7,11-1,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-4,19-6,19-7,19-8,19-9,2-12,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,4-1,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-7,8-8,9-1,9-20,9-7"
HARD_PAIRS="19-19:10-7,19-19:3,17-10:10-7,17-10:3,19-19:17-10,16-5:19-19,16-5:3,10-7:19-19,10-7:3,10-7:17-10,10-7:1,19-19:14-11,19-19:1,19-19:16-16,14-11:16-16,14-11:1,16-5:14-11,16-5:1,5-5:19-19,5-5:3,5-5:16-5,5-5:10-7,5-5:17-10,5-5:14-11,5-1:19-19,5-1:3,5-5:1,5-5:16-16,5-5:3-13,5-5:2-6,5-1:14-11,5-1:1"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/${PROFILE}"
cd "${ROOT}"

REFERENCE_ARGS=()
if [[ -n "${EXPORT_REFERENCE_NPZ}" ]]; then
  REFERENCE_ARGS=(--export_reference_npz "${EXPORT_REFERENCE_NPZ}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u code/scripts/train_apply_phase1_iq_preadapter_20260703.py \
  --ckpt "${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth" \
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl" \
  --new_wisig_pkl "${ROOT}/Dataset_WigSig/ManyTx.pkl" \
  --runs_root "${RUN_ROOT}" \
  --out_subdir "ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_${PROFILE}_N20" \
  --out_name "features_hardpair_${PROFILE}_n20.npz" \
  --clean_out_name "features_clean_hardpair_${PROFILE}_n20.npz" \
  "${REFERENCE_ARGS[@]}" \
  --cells "MANYNEW20_HARDPAIR_${PROFILE}:${TARGET_RX}:${NEW_TX}" \
  --source_tx_ids "${OLD_TX}" \
  --target_old_tx_ids "${OLD_TX}" \
  --source_rxs "${SOURCE_RXS}" \
  --proxy_unknown_tx_ids "${PROXY_TX}" \
  --proxy_unknown_rxs "${PROXY_RXS}" \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --max_samples_per_combo 0 \
  --max_source_samples_per_tx 1000 \
  --max_proxy_unknown_train_samples_per_tx 500 \
  --max_proxy_unknown_samples_per_combo 0 \
  --max_export_samples_per_tx 80 \
  --num_old_classes 6 \
  --sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --satellite_tta_policy "${SATELLITE_TTA_POLICY}" \
  --star_ground_channel_impl simplified_leo_residual \
  --batch_size 384 \
  --epochs 60 \
  --no-input_adapter_enabled \
  --model_adapter_mode id_norm_late_feature \
  --input_repair raw \
  --clean_input_repair_mode raw \
  --lr 8e-4 \
  --weight_decay 1e-4 \
  --mse_weight 1.0 \
  --cos_weight 2.0 \
  --proto_ce_weight 0.2 \
  --logit_ce_weight 0.0 \
  --clean_identity_weight 22.0 \
  --clean_cos_weight 1.0 \
  --feature_margin_weight 4.5 \
  --clean_feature_margin_weight 7.5 \
  --feature_margin_tolerance 0.01 \
  --proxy_unknown_separation_weight 0.1 \
  --proxy_unknown_max_cos 0.05 \
  --proxy_unknown_supcon_weight 0.16 \
  --proxy_unknown_supcon_temperature 0.07 \
  --proxy_unknown_proto_ce_weight 0.12 \
  --proxy_unknown_proto_temperature 0.07 \
  --proxy_unknown_pair_margin_weight 0.14 \
  --proxy_unknown_pair_margin 0.07 \
  --proxy_unknown_old_margin_weight 0.12 \
  --proxy_unknown_old_margin 0.05 \
  --proxy_unknown_hard_pair_ids "${HARD_PAIRS}" \
  --proxy_unknown_hard_pair_margin_weight "${HARD_PAIR_WEIGHT}" \
  --proxy_unknown_hard_pair_margin 0.08 \
  --proxy_unknown_hard_old_margin_weight "${HARD_OLD_WEIGHT}" \
  --proxy_unknown_hard_old_margin 0.05 \
  --teacher_logit_distill_weight 0.16 \
  --distill_temperature 2.0 \
  --residual_weight 0.0 \
  --proto_temperature 0.07 \
  --grad_clip 5.0 \
  --log_every 5 \
  --device cuda:0 \
  --seed "${EXPORT_SEED}"

FEATURE_NPZ="${RUN_ROOT}/MANYNEW20_HARDPAIR_${PROFILE}/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_${PROFILE}_N20/features_hardpair_${PROFILE}_n20.npz"
EVAL_DIR="${RUN_ROOT}/${PROFILE}/qknn_eval"
mkdir -p "${EVAL_DIR}"

"${PYTHON}" -u code/scripts/phase2_support_metric_qknn_probe.py \
  --feature_npz "${FEATURE_NPZ}" \
  --output_json "${EVAL_DIR}/n20_k10_coreproto_hardpair_${PROFILE}.json" \
  --output_csv "${EVAL_DIR}/n20_k10_coreproto_hardpair_${PROFILE}.csv" \
  --old_tx_ids "${OLD_TX}" \
  --new_tx_ids "${NEW_TX}" \
  --new_role target_unknown --seed_start 421029 --seed_count 1 \
  --k_old 10 --k_new 10 --query_per_old 70 --query_per_new 70 \
  --pool_per_old 10 --pool_per_new 10 --exclude_pool_from_query \
  --policies stable_first --transform_modes diag_fisher --transform_strengths 0.5 \
  --topm_grid 4 --proto_mix_grid 0.25 --radius_norm_grid 0 \
  --old_bias_grid 0.001 --neg_lambda_grid 0.7 --neg_threshold_grid 0.75 --neg_margin_grid 0.01 --mutual_only_grid true \
  --pair_gaussian_similarity_grid 0.95 --pair_gaussian_weight_grid 0.005 --pair_gaussian_clip_grid 2.0 \
  --pair_fisher_similarity_grid 0.9 --pair_fisher_weight_grid 0.01 --pair_fisher_alpha_grid 1.0 --pair_fisher_clip_grid 2.0 \
  --ridge_head_weight_grid 0.015 --ridge_head_alpha_grid 0.01 --ridge_head_clip_grid 2.0 \
  --core_proto_weight_grid 0,0.1 --core_proto_count_grid 2 --core_proto_topm_grid 2 --core_proto_mode_grid axis \
  --scenario_aware --balanced_assignment

"${PYTHON}" -u code/scripts/phase2_support_metric_qknn_probe.py \
  --feature_npz "${FEATURE_NPZ}" \
  --output_json "${EVAL_DIR}/n20_k5_sourceguard_hardpair_${PROFILE}.json" \
  --output_csv "${EVAL_DIR}/n20_k5_sourceguard_hardpair_${PROFILE}.csv" \
  --old_tx_ids "${OLD_TX}" \
  --new_tx_ids "${NEW_TX}" \
  --new_role target_unknown --seed_start 421037 --seed_count 1 \
  --k_old 5 --k_new 5 --query_per_old 75 --query_per_new 75 \
  --pool_per_old 5 --pool_per_new 5 --exclude_pool_from_query \
  --policies stable_first --transform_modes diag_whiten_fisher --transform_strengths 0.1 \
  --topm_grid 4 --proto_mix_grid 0.4 --radius_norm_grid 0 \
  --old_bias_grid 0.001 --neg_lambda_grid 0.7 --neg_threshold_grid 0.75 --neg_margin_grid 0.01 --mutual_only_grid true \
  --pair_gaussian_similarity_grid 0.85 --pair_gaussian_weight_grid 0.02 --pair_gaussian_clip_grid 2.0 \
  --pair_fisher_similarity_grid 0.9 --pair_fisher_weight_grid 0.01 --pair_fisher_alpha_grid 1.0 --pair_fisher_clip_grid 2.0 \
  --source_guard_mode_grid add_old --source_guard_weight_grid 0.05 --source_guard_conf_min_grid 0 --source_guard_margin_min_grid 0 \
  --scenario_aware --balanced_assignment
