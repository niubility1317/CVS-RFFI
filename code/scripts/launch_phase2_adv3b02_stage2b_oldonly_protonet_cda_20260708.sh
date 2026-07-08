#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
CKPT="${CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SOURCE_RXS="${SOURCE_RXS:-1-1,1-19,14-7,18-2,19-2,2-1,2-19}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_RECEIVERS="${TARGET_RECEIVERS:-20-1,3-19,7-14,7-7,8-8}"
SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
K_VALUES="${K_VALUES:-5,10}"
MAX_SAMPLES_PER_COMBO="${MAX_SAMPLES_PER_COMBO:-80}"
MAX_SAMPLES_PER_TX="${MAX_SAMPLES_PER_TX:-0}"
SEED="${SEED:-4070802}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

CASE_ID="ADV3B02_CORE90_SOFT_E200_STAGE2B_OLDONLY_PROTONET_CDA"
OUT_DIR="${RUNS_ROOT}/${CASE_ID}"
FEATURE_NPZ="${OUT_DIR}/features_target_old_leo.npz"
METRICS_JSON="${OUT_DIR}/target_old_protonet_cda_metrics.json"
SUMMARY_CSV="${OUT_DIR}/target_old_protonet_cda_summary.csv"
DETAIL_CSV="${OUT_DIR}/target_old_protonet_cda_detail.csv"
LOG_PATH="${LOG_ROOT}/${CASE_ID}.out"

EXPORT_CMD=(
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"
  --ckpt "${CKPT}"
  --wisig_pkl "${WISIG_PKL}"
  --out_npz "${FEATURE_NPZ}"
  --feature_name z_id
  --source_tx_ids "${SOURCE_TX_IDS}"
  --source_rxs "${SOURCE_RXS}"
  --target_old_tx_ids "${TARGET_OLD_TX_IDS}"
  --target_old_rxs "${TARGET_RECEIVERS}"
  --target_old_channel_view satellite
  --target_old_sat_scenarios "${SAT_SCENARIOS}"
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
  --seed "${SEED}"
)

EVAL_CMD=(
  env PYTHONPATH="${ROOT}/code:${ROOT}/code/scripts:${ROOT}:${PYTHONPATH:-}"
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_target_old_only_upper_bound.py"
  --feature_npz "${FEATURE_NPZ}"
  --target_old_tx_ids "${TARGET_OLD_TX_IDS}"
  --k_values "${K_VALUES}"
  --output_json "${METRICS_JSON}"
  --summary_csv "${SUMMARY_CSV}"
  --detail_csv "${DETAIL_CSV}"
)

echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA] base_model=ADV3B02_CORE90_SOFT_E200 ckpt=${CKPT}"
echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA] protocol=Stage2-B target_old_only=true target_new_enabled=false unknown_rejection_enabled=false"
echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA] support_query=target_old_R_t_only target_receivers=${TARGET_RECEIVERS}"
echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA] target_channel=satellite/LEO scenarios=${SAT_SCENARIOS} k_values=${K_VALUES}"
printf "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-EXPORT-CMD] "
printf "%q " "${EXPORT_CMD[@]}"
printf "\n"
printf "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-EVAL-CMD] "
printf "%q " "${EVAL_CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-DRY-RUN-DONE]"
  exit 0
fi

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
(
  set -euo pipefail
  "${EXPORT_CMD[@]}"
  "${EVAL_CMD[@]}"
  echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-DONE] run_id=${RUN_ID} out_dir=${OUT_DIR}"
) > "${LOG_PATH}" 2>&1 &

echo "[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-LAUNCHED] pid=$! gpu=${GPU} log=${LOG_PATH} out_dir=${OUT_DIR}"
