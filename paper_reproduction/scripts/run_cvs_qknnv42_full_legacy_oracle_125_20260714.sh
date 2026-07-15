#!/usr/bin/env bash
set -euo pipefail

echo "BLOCKED: role Oracle and class-quota inference are prohibited by the 2026-07-15 target-query protocol. Historical artifacts remain audit-only." >&2
exit 2

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-3}"
RUN_ID="${RUN_ID:-cvs_qknnv42_full_legacy_oracle_125_20260714}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/runs/cvs_qknnv42_full_adapter5_fft96_20260714}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
CVS_CONFIG="${CVS_CONFIG:-${ROOT}/paper_reproduction/configs/cvs_qknnv42_full_legacy_oracle_stage2c_20260714_n607.json}"

OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="1-16,1-18"
UNKNOWN_TX="1-1"
SOURCE_RXS="0,1,2,3,4,5,6"
PROXY_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
PROXY_TX="1-8,10-11,10-17,10-4,10-7,11-1,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-4,19-6,19-7,19-8,19-9,2-12,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,4-1,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-7,8-8,9-1,9-20,9-7"
HARD_PAIRS="19-19:10-7,19-19:3,17-10:10-7,17-10:3,19-19:17-10,16-5:19-19,16-5:3,10-7:19-19,10-7:3,10-7:17-10,10-7:1,19-19:14-11,19-19:1,19-19:16-16,14-11:16-16,14-11:1,16-5:14-11,16-5:1,5-5:19-19,5-5:3,5-5:16-5,5-5:10-7,5-5:17-10,5-5:14-11,5-1:19-19,5-1:3,5-5:1,5-5:16-16,5-5:3-13,5-5:2-6,5-1:14-11,5-1:1"
CELLS="FULL_RX_20-1:20-1:${NEW_TX}:${UNKNOWN_TX};FULL_RX_3-19:3-19:${NEW_TX}:${UNKNOWN_TX};FULL_RX_7-14:7-14:${NEW_TX}:${UNKNOWN_TX};FULL_RX_7-7:7-7:${NEW_TX}:${UNKNOWN_TX};FULL_RX_8-8:8-8:${NEW_TX}:${UNKNOWN_TX}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "$FEATURE_ROOT" "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u code/scripts/train_apply_phase1_iq_preadapter_20260703.py \
  --ckpt "${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth" \
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl" \
  --new_wisig_pkl "${ROOT}/Dataset_WigSig/ManyTx.pkl" \
  --runs_root "$FEATURE_ROOT" \
  --out_subdir ADV3B02_FULL_ADAPTER5_FFT96 \
  --out_name features_full_adapter5_fft96.npz \
  --no-export_clean_control \
  --no-export_identity \
  --cells "$CELLS" \
  --feature_name z_id \
  --aux_fft_logmag_dim 96 \
  --export_all_scenarios_per_sample \
  --source_tx_ids "$OLD_TX" \
  --target_old_tx_ids "$OLD_TX" \
  --source_rxs "$SOURCE_RXS" \
  --proxy_unknown_tx_ids "$PROXY_TX" \
  --proxy_unknown_rxs "$PROXY_RXS" \
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
  --satellite_tta_policy rx_light5 \
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
  --proxy_unknown_hard_pair_ids "$HARD_PAIRS" \
  --proxy_unknown_hard_pair_margin_weight 0.08 \
  --proxy_unknown_hard_pair_margin 0.08 \
  --proxy_unknown_hard_old_margin_weight 0.04 \
  --proxy_unknown_hard_old_margin 0.05 \
  --teacher_logit_distill_weight 0.16 \
  --distill_temperature 2.0 \
  --residual_weight 0.0 \
  --proto_temperature 0.07 \
  --grad_clip 5.0 \
  --log_every 5 \
  --device cuda:0 \
  --seed 4070391

"$PYTHON" -u -m paper_reproduction.scripts.run_cvs_publication_matrix \
  --phase stage2c \
  --config "${ROOT}/paper_reproduction/configs/cvs_stage2c_publication_base_n607.json" \
  --cvs-config "$CVS_CONFIG" \
  --output-root "$OUT_ROOT" \
  --log-root "$LOG_ROOT" \
  --methods cvs_qknnv42 \
  --execute

echo "[FULL-LEGACY-ORACLE-125-DONE] run_root=${OUT_ROOT}"
