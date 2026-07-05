#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_epoc_b_ospr_qknn_collab_20260705}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
CKPT="${CKPT:-${ROOT}/runs/phase1_epoc_adv3b02_distill_20260705/EPOC_DISTILL_B_KDHI/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
SOURCE_RXS="${SOURCE_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_RECEIVERS="${TARGET_RECEIVERS:-20-1,3-19,7-14,7-7,8-8}"

# Confirmed on N607 ManyTx: every listed TX has >=200 rows on each target receiver.
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
OSPR_DEVICE="${OSPR_DEVICE:-cpu}"
EVENT_ALIGNMENT_POLICY="${EVENT_ALIGNMENT_POLICY:-receiver_domain_ranked}"
SEED="${SEED:-4070505}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

select_gpu() {
  if [[ -n "${GPU:-}" ]]; then
    echo "${GPU}"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "0"
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2,2n \
    | head -n 1 \
    | cut -d',' -f1 \
    | tr -d ' '
}

GPU_SELECTED="$(select_gpu)"
CASE_ID="EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5"
OUT_DIR="${RUNS_ROOT}/${CASE_ID}"
FEATURE_NPZ="${OUT_DIR}/features_stage2c_leo_multirx.npz"
OSPR_OUT="${OUT_DIR}/ospr_ci"
LOG_PATH="${LOG_ROOT}/${CASE_ID}.out"

echo "[EPOC-B-OSPR-QKNN-COLLAB] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] ckpt=${CKPT}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] target_receivers=${TARGET_RECEIVERS}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] target_new_tx_ids=${TARGET_NEW_TX_IDS}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] unknown_tx_ids=${UNKNOWN_TX_IDS}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] proxy_unknown_tx_ids=${PROXY_UNKNOWN_TX_IDS}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] gpu_selected=${GPU_SELECTED} ospr_device=${OSPR_DEVICE}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] collab_counts=all qknn_k=${QKNN_K} k_shot=${K_SHOT} query_per_class=${QUERY_PER_CLASS}"
echo "[EPOC-B-OSPR-QKNN-COLLAB] protocol=Stage2-C unknown_query_eval_only=true ground_training_unknown_seen=false"
echo "[EPOC-B-OSPR-QKNN-COLLAB] event_alignment_policy=${EVENT_ALIGNMENT_POLICY} verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC unless strict_event_key succeeds"

EXPORT_CMD=(
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_SELECTED}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"
  --ckpt "${CKPT}"
  --wisig_pkl "${WISIG_PKL}"
  --new_wisig_pkl "${NEW_WISIG_PKL}"
  --out_npz "${FEATURE_NPZ}"
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
  --seed "${SEED}"
)

OSPR_CMD=(
  env PYTHONPATH="${ROOT}/code:${ROOT}/code/scripts:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_SELECTED}"
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_ospr_ci_eval.py"
  --feature_npz "${FEATURE_NPZ}"
  --output_dir "${OSPR_OUT}"
  --backend both
  --collab_counts all
  --collab_group_policy same_max_budget
  --partial_collab_min_receivers 1
  --k_shot "${K_SHOT}"
  --query_per_class "${QUERY_PER_CLASS}"
  --qknn_k "${QKNN_K}"
  --source_holdout_per_class 32
  --support_selection_policy stable_first
  --event_alignment_policy "${EVENT_ALIGNMENT_POLICY}"
  --device "${OSPR_DEVICE}"
  --max_event_bytes "${MAX_EVENT_BYTES}"
  --max_event_latency_ms "${MAX_EVENT_LATENCY_MS}"
  --target_old_acc 0.99
  --target_min_old 0.95
  --target_seen_new_acc 0.97
  --target_min_seen 0.93
  --target_unknown_reject 0.99
  --seed "${SEED}"
)

printf "[EPOC-B-OSPR-QKNN-COLLAB-EXPORT-CMD] "
printf "%q " "${EXPORT_CMD[@]}"
printf "\n"
printf "[EPOC-B-OSPR-QKNN-COLLAB-OSPR-CMD] "
printf "%q " "${OSPR_CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUT_DIR}" "${OSPR_OUT}" "${LOG_ROOT}"
(
  set -euo pipefail
  "${EXPORT_CMD[@]}"
  "${OSPR_CMD[@]}"
  echo "[EPOC-B-OSPR-QKNN-COLLAB-DONE] run_id=${RUN_ID} out_dir=${OUT_DIR}"
) > "${LOG_PATH}" 2>&1 &

echo "[EPOC-B-OSPR-QKNN-COLLAB-LAUNCHED] pid=$! gpu=${GPU_SELECTED} log=${LOG_PATH} out_dir=${OUT_DIR}"
