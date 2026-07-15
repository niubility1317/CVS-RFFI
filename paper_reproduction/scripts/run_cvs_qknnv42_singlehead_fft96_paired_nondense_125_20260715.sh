#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-6}"
SWEEP_ID="${SWEEP_ID:-qknnv42_nondense_adapter_epoch_sweep_20260715_104409}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/runs/qknnv42_singlehead_fft96_paired_features_20260715}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${SWEEP_ID}/singlehead_fft96_paired}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${SWEEP_ID}/singlehead_fft96_paired}"
CVS_CONFIG="${CVS_CONFIG:-${ROOT}/paper_reproduction/configs/cvs_qknnv42_singlehead_fft96_paired_nondense_stage2c_20260715_n607.json}"

OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="1-16,1-18"
UNUSED_UNKNOWN_TX="1-1"
CELLS="FULL_RX_20-1:20-1:${NEW_TX}:${UNUSED_UNKNOWN_TX};FULL_RX_3-19:3-19:${NEW_TX}:${UNUSED_UNKNOWN_TX};FULL_RX_7-14:7-14:${NEW_TX}:${UNUSED_UNKNOWN_TX};FULL_RX_7-7:7-7:${NEW_TX}:${UNUSED_UNKNOWN_TX};FULL_RX_8-8:8-8:${NEW_TX}:${UNUSED_UNKNOWN_TX}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "$FEATURE_ROOT" "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u code/scripts/train_apply_phase1_iq_preadapter_20260703.py \
  --ckpt "${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth" \
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl" --new_wisig_pkl "${ROOT}/Dataset_WigSig/ManyTx.pkl" \
  --runs_root "$FEATURE_ROOT" --out_subdir ADV3B02_SINGLEHEAD_FFT96_PAIRED \
  --out_name features_singlehead_fft96_paired.npz --export_role_scope qknn_registered_only \
  --no-export_clean_control --no-export_identity --cells "$CELLS" --feature_name z_id \
  --aux_fft_logmag_dim 96 --export_all_scenarios_per_sample \
  --target_old_tx_ids "$OLD_TX" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 \
  --max_samples_per_combo 0 --max_export_samples_per_tx 80 --num_old_classes 6 \
  --sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --satellite_tta_policy none \
  --star_ground_channel_impl simplified_leo_residual --batch_size 384 --skip_adapter_training \
  --no-input_adapter_enabled --model_adapter_mode none --input_repair raw --device cuda:0 --seed 4070391

"$PYTHON" -u -m paper_reproduction.scripts.run_cvs_publication_matrix \
  --phase stage2c --config "${ROOT}/paper_reproduction/configs/cvs_stage2c_publication_base_n607.json" \
  --cvs-config "$CVS_CONFIG" --output-root "$OUT_ROOT" --log-root "$LOG_ROOT" \
  --methods cvs_qknnv42 --execute

echo "[QKNN-SINGLEHEAD-FFT96-PAIRED-NONDENSE-DONE] run_root=${OUT_ROOT}"
