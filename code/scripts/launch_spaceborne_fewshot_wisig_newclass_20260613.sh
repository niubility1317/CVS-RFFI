#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fewshot_wisig_newclass_20260613}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-}"
SFE_MAX_SAMPLES_PER_COMBO="${SFE_MAX_SAMPLES_PER_COMBO:-0}"
SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"
SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"
SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"
SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"
SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=1"
PIDS=()
NAMES=()

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_NEW_TX_K5_STRICT protocol=CVS-SFE k=5 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"${SFE_MAX_SAMPLES_PER_COMBO}\" --max_samples_per_tx \"${SFE_MAX_SAMPLES_PER_TX}\" --batch_size \"${SFE_EXPORT_BATCH_SIZE}\" --device cuda:0 --seed 1342; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"${SFE_SOURCE_PROTO_PER_TX}\" --source_query_per_tx \"${SFE_SOURCE_QUERY_PER_TX}\" --query_per_tx \"${SFE_QUERY_PER_TX}\" --unknown_threshold 0.70 --seed 1342")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_NEW_TX_K5_STRICT")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_NEW_TX_K5_STRICT pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_NEW_TX_K5_STRICT.out"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  STATUS=0
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[${idx}]}"; then
      echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=0"
    else
      rc=$?
      echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=${rc}" >&2
      STATUS=${rc}
    fi
  done
  exit "${STATUS}"
fi
