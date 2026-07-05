#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_frozen_manytx_unknown_diag_20260706}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
CKPT="${CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SOURCE_RXS="${SOURCE_RXS:-1-1,1-19,14-7,18-2,19-2,2-1,2-19}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_RECEIVERS="${TARGET_RECEIVERS:-20-1,3-19,7-14,7-7,8-8}"

# Confirmed Stage2-C ManyTx candidates from 项目.md; target_new and
# target_unknown are mutually exclusive and target_unknown remains eval-only.
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-12-7,13-14,13-19,13-3,13-7,14-11,14-12,14-13}"
PROXY_UNKNOWN_RXS="${PROXY_UNKNOWN_RXS:-1-1,1-19,14-7,18-2,19-2,2-1}"

K_SHOT="${K_SHOT:-8}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-20}"
QKNN_K="${QKNN_K:-8}"
MAX_SAMPLES_PER_COMBO="${MAX_SAMPLES_PER_COMBO:-80}"
MAX_SAMPLES_PER_TX="${MAX_SAMPLES_PER_TX:-0}"
MAX_EVENT_BYTES="${MAX_EVENT_BYTES:-1152}"
MAX_EVENT_LATENCY_MS="${MAX_EVENT_LATENCY_MS:-20}"
EVENT_ALIGNMENT_POLICY="${EVENT_ALIGNMENT_POLICY:-receiver_domain_ranked}"
COLLAB_GROUP_POLICY="${COLLAB_GROUP_POLICY:-available_up_to_k}"
PARTIAL_COLLAB_MIN_RECEIVERS="${PARTIAL_COLLAB_MIN_RECEIVERS:-1}"
SEED="${SEED:-4070606}"
DRY_RUN="${DRY_RUN:-0}"
DIAG_INCLUDE_EVIDENCE="${DIAG_INCLUDE_EVIDENCE:-0}"

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
CASE_ID="ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5"
OUT_DIR="${RUNS_ROOT}/${CASE_ID}"
FEATURE_NPZ="${OUT_DIR}/features_stage2c_leo_multirx.npz"
DIAG_JSON="${OUT_DIR}/frozen_manytx_diag.json"
DIAG_SUMMARY="${OUT_DIR}/frozen_manytx_summary.csv"
DIAG_EVIDENCE="${OUT_DIR}/frozen_manytx_evidence.csv"
LOG_PATH="${LOG_ROOT}/${CASE_ID}.out"

echo "[ADV3B02-FROZEN-MANYTX-DIAG] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] base_model=ADV3B02_CORE90_SOFT_E200 ckpt=${CKPT}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] source_tx_ids=${SOURCE_TX_IDS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] source_receivers=${SOURCE_RXS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] target_receivers=${TARGET_RECEIVERS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] target_new_tx_ids=${TARGET_NEW_TX_IDS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] unknown_tx_ids=${UNKNOWN_TX_IDS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] proxy_unknown_tx_ids=${PROXY_UNKNOWN_TX_IDS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] proxy_unknown_receivers=${PROXY_UNKNOWN_RXS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] gpu_selected=${GPU_SELECTED}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] collab_counts=all collab_group_policy=${COLLAB_GROUP_POLICY} partial_collab_min_receivers=${PARTIAL_COLLAB_MIN_RECEIVERS} qknn_k=${QKNN_K} k_shot=${K_SHOT} query_per_class=${QUERY_PER_CLASS}"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] protocol=Stage2-C unknown_query_eval_only=true ground_training_unknown_seen=false"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] target_goals old_acc=0.99 min_old=0.95 seen_new_acc=0.97 min_seen=0.93 unknown_reject=0.99"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] event_alignment_policy=${EVENT_ALIGNMENT_POLICY} verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC unless strict_event_key succeeds"
echo "[ADV3B02-FROZEN-MANYTX-DIAG] resource_proxy=max_event_bytes=${MAX_EVENT_BYTES} max_event_latency_ms=${MAX_EVENT_LATENCY_MS}"

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
  --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}"
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

DIAG_CMD=(
  env PYTHONPATH="${ROOT}/code:${ROOT}/code/scripts:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_SELECTED}"
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py"
  --feature_npz "${FEATURE_NPZ}"
  --output_json "${DIAG_JSON}"
  --output_summary_csv "${DIAG_SUMMARY}"
  --collab_counts all
  --collab_group_policy "${COLLAB_GROUP_POLICY}"
  --partial_collab_min_receivers "${PARTIAL_COLLAB_MIN_RECEIVERS}"
  --k_shot "${K_SHOT}"
  --query_per_class "${QUERY_PER_CLASS}"
  --qknn_k "${QKNN_K}"
  --support_selection_policy stable_first
  --event_alignment_policy "${EVENT_ALIGNMENT_POLICY}"
  --max_event_bytes "${MAX_EVENT_BYTES}"
  --max_event_latency_ms "${MAX_EVENT_LATENCY_MS}"
  --seed "${SEED}"
)

if [[ "${DIAG_INCLUDE_EVIDENCE}" == "1" ]]; then
  DIAG_CMD+=(--output_evidence_csv "${DIAG_EVIDENCE}")
fi

printf "[ADV3B02-FROZEN-MANYTX-DIAG-EXPORT-CMD] "
printf "%q " "${EXPORT_CMD[@]}"
printf "\n"
printf "[ADV3B02-FROZEN-MANYTX-DIAG-EVAL-CMD] "
printf "%q " "${DIAG_CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUT_DIR}" "${LOG_ROOT}"
(
  set -euo pipefail
  "${EXPORT_CMD[@]}"
  "${DIAG_CMD[@]}"
  echo "[ADV3B02-FROZEN-MANYTX-DIAG-DONE] run_id=${RUN_ID} out_dir=${OUT_DIR}"
) > "${LOG_PATH}" 2>&1 &

echo "[ADV3B02-FROZEN-MANYTX-DIAG-LAUNCHED] pid=$! gpu=${GPU_SELECTED} log=${LOG_PATH} out_dir=${OUT_DIR}"
