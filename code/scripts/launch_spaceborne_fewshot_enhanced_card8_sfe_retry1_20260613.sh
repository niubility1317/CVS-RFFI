#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fewshot_enhanced_card8_20260613}"
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
echo "[SPACEBORNE-FSDA-RETRY1] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=6"

PIDS=()
NAMES=()

launch_sfe() {
  local cid="$1"
  local gpu="$2"
  local shots="$3"
  local seed="$4"
  local gate_mode="$5"
  local openmax_quantile="$6"
  local extra_eval_args="$7"
  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.retry1.out"
  local inner

  echo "[SPACEBORNE-FSDA-RETRY1-CANDIDATE] id=${cid} protocol=CVS-SFE k=${shots} gpu=${gpu} gate=${gate_mode}"
  inner="set -euo pipefail; mkdir -p \"${out_dir}\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${out_dir}/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"${SFE_MAX_SAMPLES_PER_COMBO}\" --max_samples_per_tx \"${SFE_MAX_SAMPLES_PER_TX}\" --batch_size \"${SFE_EXPORT_BATCH_SIZE}\" --device cuda:0 --seed ${seed}; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${out_dir}/features.npz\" --output_json \"${out_dir}/metrics.json\" --manifest_json \"${out_dir}/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots ${shots} --source_proto_per_tx \"${SFE_SOURCE_PROTO_PER_TX}\" --source_query_per_tx \"${SFE_SOURCE_QUERY_PER_TX}\" --query_per_tx \"${SFE_QUERY_PER_TX}\" --unknown_threshold 0.7 --gate_mode ${gate_mode} --openmax_tail_size 20 --openmax_quantile ${openmax_quantile} --openmax_min_threshold 0.02 ${extra_eval_args} --seed ${seed}"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "${inner}")
  printf "[SPACEBORNE-FSDA-RETRY1-CMD] "
  printf "%q " "${CMD[@]}"
  printf "\n"

  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${out_dir}"
    ("${CMD[@]}" > "${log_path}" 2>&1) &
    local pid="$!"
    PIDS+=("${pid}")
    NAMES+=("${cid}")
    echo "[SPACEBORNE-FSDA-RETRY1-LAUNCHED] id=${cid} pid=${pid} gpu=${gpu} log=${log_path}"
  fi
}

launch_sfe "SFE_WISIG_GATE_COSINE_K5" "0" "5" "1342" "cosine" "0.95" ""
launch_sfe "SFE_WISIG_GATE_MARGIN_K5" "1" "5" "1343" "combined" "0.95" "--min_margin 0.05"
launch_sfe "SFE_WISIG_GATE_MAHAL_K5" "2" "5" "1344" "mahalanobis" "0.95" "--max_mahalanobis 8.0"
launch_sfe "SFE_WISIG_GATE_OPENMAX_K5" "3" "5" "1345" "openmax" "1.0" ""
launch_sfe "SFE_WISIG_GATE_COMBINED_K10" "4" "10" "1347" "combined" "1.0" "--min_margin 0.05 --max_mahalanobis 8.0"
launch_sfe "SFE_WISIG_GATE_COMBINED_K20" "5" "20" "1357" "combined" "1.0" "--min_margin 0.05 --max_mahalanobis 8.0"

if [[ "${DRY_RUN}" != "1" ]]; then
  STATUS=0
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[${idx}]}"; then
      echo "[SPACEBORNE-FSDA-RETRY1-COMPLETE] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=0"
    else
      rc=$?
      echo "[SPACEBORNE-FSDA-RETRY1-FAILED] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=${rc}" >&2
      STATUS=${rc}
    fi
  done
  exit "${STATUS}"
fi
