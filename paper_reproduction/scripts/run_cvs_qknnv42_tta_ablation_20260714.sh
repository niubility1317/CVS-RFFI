#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-3}"
RUN_ID="${RUN_ID:-cvs_qknnv42_frozen_adv3b02_full_history_tta_20260714}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/runs/cvs_qknnv42_frozen_adv3b02_tta_features_20260714}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
HISTORICAL_REFERENCE_ROOT="${HISTORICAL_REFERENCE_ROOT:-${ROOT}/runs/cvs_qknnv42_full_legacy_oracle_strict125_20260714_183556}"
CKPT="${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
EXPECTED_CHECKPOINT_SHA256="$(sha256sum "$CKPT" | awk '{print $1}')"

OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="1-16,1-18"
UNKNOWN_TX="1-1"
SOURCE_RXS="0,1,2,3,4,5,6"
PROXY_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
PROXY_TX="1-8,10-11,10-17,10-4,10-7,11-1,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-4,19-6,19-7,19-8,19-9,2-12,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,4-1,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-7,8-8,9-1,9-20,9-7"
HARD_PAIRS="19-19:10-7,19-19:3,17-10:10-7,17-10:3,19-19:17-10,16-5:19-19,16-5:3,10-7:19-19,10-7:3,10-7:17-10,10-7:1,19-19:14-11,19-19:1,19-19:16-16,14-11:16-16,14-11:1,16-5:14-11,16-5:1,5-5:19-19,5-5:3,5-5:16-5,5-5:10-7,5-5:17-10,5-5:14-11,5-1:19-19,5-1:3,5-5:1,5-5:16-16,5-5:3-13,5-5:2-6,5-1:14-11,5-1:1"
CELLS="FULL_RX_20-1:20-1:${NEW_TX}:${UNKNOWN_TX};FULL_RX_3-19:3-19:${NEW_TX}:${UNKNOWN_TX};FULL_RX_7-14:7-14:${NEW_TX}:${UNKNOWN_TX};FULL_RX_7-7:7-7:${NEW_TX}:${UNKNOWN_TX};FULL_RX_8-8:8-8:${NEW_TX}:${UNKNOWN_TX}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
for guarded_root in "$FEATURE_ROOT" "$OUT_ROOT" "$LOG_ROOT"; do
  if [[ -d "$guarded_root" ]] && [[ -n "$(find "$guarded_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[REFUSE-OVERWRITE] non-empty path=${guarded_root}" >&2
    exit 2
  fi
done
mkdir -p "$FEATURE_ROOT" "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u code/scripts/train_apply_phase1_iq_preadapter_20260703.py \
  --ckpt "$CKPT" \
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl" \
  --new_wisig_pkl "${ROOT}/Dataset_WigSig/ManyTx.pkl" \
  --runs_root "$FEATURE_ROOT" \
  --out_subdir ADV3B02_FROZEN_QKNN_FFT96 \
  --out_name features_frozen_adv3b02_fft96.npz \
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
  --export_tta_policies none,rx_shift3,rx_cfo3,rx_light5 \
  --export_tta_subdir_template '{base}_{policy}' \
  --star_ground_channel_impl simplified_leo_residual \
  --batch_size 384 \
  --epochs 0 \
  --skip_adapter_training \
  --no-input_adapter_enabled \
  --model_adapter_mode none \
  --input_repair raw \
  --clean_input_repair_mode raw \
  --device cuda:0 \
  --seed 4070391 2>&1 | tee "$LOG_ROOT/frozen_adv3b02_export.out"

"$PYTHON" -u -m paper_reproduction.scripts.benchmark_qknnv42_tta_policies \
  --template-config "${ROOT}/paper_reproduction/configs/cvs_qknnv42_full_legacy_oracle_strict_stage2c_20260714_n607.json" \
  --historical-reference-root "$HISTORICAL_REFERENCE_ROOT" \
  --head-profile full_legacy_oracle \
  --feature-root "$FEATURE_ROOT" \
  --out-root "$OUT_ROOT/dense" \
  --policies none rx_shift3 rx_cfo3 rx_light5 \
  --feature-subdir-base ADV3B02_FROZEN_QKNN_FFT96 \
  --feature-subdir-template '{base}_{policy}' \
  --feature-name features_frozen_adv3b02_fft96.npz \
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
  --seed-grid 713101 713102 713103 713104 713105 \
  --k-grid 1 2 5 10 20 \
  --expected-runs 500 2>&1 | tee "$LOG_ROOT/tta_benchmark.out"

"$PYTHON" -u -m paper_reproduction.scripts.benchmark_qknnv42_tta_policies \
  --template-config "${ROOT}/paper_reproduction/configs/cvs_qknnv42_full_legacy_oracle_strict_stage2c_20260714_n607.json" \
  --historical-reference-root "$HISTORICAL_REFERENCE_ROOT" \
  --head-profile full_legacy_oracle_prototype \
  --feature-root "$FEATURE_ROOT" \
  --out-root "$OUT_ROOT/prototype" \
  --policies none rx_shift3 rx_cfo3 rx_light5 \
  --feature-subdir-base ADV3B02_FROZEN_QKNN_FFT96 \
  --feature-subdir-template '{base}_{policy}' \
  --feature-name features_frozen_adv3b02_fft96.npz \
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
  --seed-grid 713101 713102 713103 713104 713105 \
  --k-grid 1 2 5 10 20 \
  --expected-runs 500 2>&1 | tee "$LOG_ROOT/tta_benchmark_prototype.out"

echo "[QKNN-FROZEN-ADV3B02-TTA-DONE] run_root=${OUT_ROOT} feature_root=${FEATURE_ROOT} historical_root=${HISTORICAL_REFERENCE_ROOT}"
