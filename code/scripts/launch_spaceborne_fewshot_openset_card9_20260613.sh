#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fewshot_openset_card9_20260613}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
GPU="${GPU:-3}"
CID="${CID:-SFE_WISIG_GATE_COMBINED_K20_OPENSET_U2}"
SFE_MAX_SAMPLES_PER_COMBO="${SFE_MAX_SAMPLES_PER_COMBO:-0}"
SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"
SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"
SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"
SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"
SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"
SFE_SHOTS="${SFE_SHOTS:-20}"
SEED="${SEED:-1457}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -z "${UNKNOWN_TX_IDS}" ]]; then
  echo "[ERROR] UNKNOWN_TX_IDS must not be empty for open-set Card9" >&2
  exit 2
fi

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
echo "[SPACEBORNE-FSDA-OPENSET-CARD9] run_id=${RUN_ID} dry_run=${DRY_RUN} cid=${CID} gpu=${GPU}"
echo "[SPACEBORNE-FSDA-OPENSET-CARD9-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"

out_dir="${RUNS_ROOT}/${CID}"
log_path="${LOG_ROOT}/${CID}.out"
inner="set -euo pipefail; mkdir -p \"${out_dir}\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${out_dir}/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"${SFE_MAX_SAMPLES_PER_COMBO}\" --max_samples_per_tx \"${SFE_MAX_SAMPLES_PER_TX}\" --batch_size \"${SFE_EXPORT_BATCH_SIZE}\" --device cuda:0 --seed \"${SEED}\"; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${out_dir}/features.npz\" --output_json \"${out_dir}/metrics.json\" --manifest_json \"${out_dir}/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots \"${SFE_SHOTS}\" --source_proto_per_tx \"${SFE_SOURCE_PROTO_PER_TX}\" --source_query_per_tx \"${SFE_SOURCE_QUERY_PER_TX}\" --query_per_tx \"${SFE_QUERY_PER_TX}\" --unknown_threshold 0.7 --gate_mode combined --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --min_margin 0.05 --max_mahalanobis 8.0 --seed \"${SEED}\""
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "${inner}")
printf "[SPACEBORNE-FSDA-OPENSET-CARD9-CMD] "
printf "%q " "${CMD[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "${out_dir}"
("${CMD[@]}" > "${log_path}" 2>&1) &
pid="$!"
echo "[SPACEBORNE-FSDA-OPENSET-CARD9-LAUNCHED] id=${CID} pid=${pid} gpu=${GPU} log=${log_path}"
wait "${pid}"
rc="$?"
if [[ "${rc}" == "0" ]]; then
  echo "[SPACEBORNE-FSDA-OPENSET-CARD9-COMPLETE] id=${CID} pid=${pid} status=0"
else
  echo "[SPACEBORNE-FSDA-OPENSET-CARD9-FAILED] id=${CID} pid=${pid} status=${rc}" >&2
fi
exit "${rc}"
