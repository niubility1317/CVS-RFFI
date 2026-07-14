#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CKPT="${CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/cvs_qknnv42_extreme_light_20new_features_20260714_v2_day0_eq1}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/cvs_qknnv42_extreme_light_20new_features_20260714_v2_day0_eq1}"
WISIG="${WISIG:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
MANYTX="${MANYTX:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
TARGET_RXS="${TARGET_RXS:-20-1,3-19,7-14,7-7,8-8}"
NEW_TX_IDS="${NEW_TX_IDS:-1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6}"
SEED="${SEED:-713101}"
GPUS="${GPUS:-0,1,2}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a gpu_array <<<"$GPUS"
scenarios=(leo_clear_weak leo_low_elev_weak leo_rain_weak)
if [[ "${#gpu_array[@]}" -ne "${#scenarios[@]}" ]]; then
  echo "GPUS must contain exactly three comma-separated GPU IDs" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT_ROOT" "$LOG_ROOT"
fi

run_one() {
  local scenario="$1"
  local gpu="$2"
  local out_npz="${OUT_ROOT}/${scenario}.npz"
  local log="${LOG_ROOT}/${scenario}.log"
  local cmd=(
    "$PYTHON" -u "${ROOT}/code/export_spaceborne_features.py"
    --ckpt "$CKPT"
    --wisig_pkl "$WISIG"
    --new_wisig_pkl "$MANYTX"
    --out_npz "$out_npz"
    --feature_name z_id
    --aux_fft_logmag_dim 96
    --source_tx_ids 0,1,2,3,4,5
    --target_old_tx_ids 0,1,2,3,4,5
    --new_tx_ids "$NEW_TX_IDS"
    --source_days 0,1
    --source_rxs 0,1,2,3,4,5,6
    --target_old_days 0
    --target_old_rxs "$TARGET_RXS"
    --new_days 0
    --new_rxs "$TARGET_RXS"
    --wisig_equalized 1
    --wisig_domain rx_day
    --wisig_out_len 256
    --max_samples_per_combo 0
    --max_samples_per_tx 400
    --batch_size 512
    --source_channel_view clean
    --target_old_channel_view satellite
    --target_old_sat_scenarios "$scenario"
    --target_old_sat_seed "$((SEED + 811))"
    --target_new_channel_view satellite
    --target_new_sat_scenarios "$scenario"
    --target_new_sat_seed "$((SEED + 911))"
    --satellite_tta_policy none
    --star_ground_channel_impl simplified_leo_residual
    --sat_fs_hz 25e6
    --sat_fc_hz 2.462e9
    --device cuda:0
    --seed "$SEED"
  )
  printf '[EXTREME-LIGHT-20NEW-EXPORT] scenario=%s gpu=%s out=%s\n' "$scenario" "$gpu" "$out_npz"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${cmd[@]}" >"$log" 2>&1
  test -s "$out_npz"
}

pids=()
for i in "${!scenarios[@]}"; do
  run_one "${scenarios[$i]}" "${gpu_array[$i]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" != "0" ]]; then
  echo "one or more extreme-light feature exports failed" >&2
  exit 3
fi
echo "[EXTREME-LIGHT-20NEW-EXPORT-DONE] out_root=${OUT_ROOT}"
