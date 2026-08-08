#!/usr/bin/env bash
set -euo pipefail

# Final bounded repair entry: reuse the completed v2 checkpoints, export each
# frozen candidate exactly once, then run the two frozen energy-only scorers.
RUN_ID="${RUN_ID:-phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${PROJECT_ROOT}/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
MANYTX_PKL="${MANYTX_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManyTx.pkl}"
EXPORTER="${CODE_ROOT}/export_spaceborne_features.py"
SCORER="${CODE_ROOT}/scripts/eval_phase1_logits_open_set_reject.py"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -f "${EXPORTER}" ]] || { echo "missing exporter: ${EXPORTER}" >&2; exit 2; }
[[ -f "${SCORER}" ]] || { echo "missing scorer: ${SCORER}" >&2; exit 2; }

SOURCE_DAYS="2021_03_01,2021_03_08"
SOURCE_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
PROXY_TX="20-20,20-16,19-3,1-19,3-18,19-12,5-20,7-14,12-7,7-9,17-11,20-3,12-20,16-1,18-7,2-3,19-10,18-9,2-4,15-6"

CANDIDATES=(
  F1C_ManyTxRealOE12 F5G_ManyTxRealOE12
  F1G_ManyTxRealOE12 F5C_ManyTxRealOE12
  F2C_ManyTxRealOE12 F6G_ManyTxRealOE12
  F2G_ManyTxRealOE12 F6C_ManyTxRealOE12
  F3C_ManyTxRealOE12 F3G_ManyTxRealOE12
  F4C_ManyTxRealOE12 F4G_ManyTxRealOE12
)
GPUS=(0 0 1 1 2 2 3 3 4 5 6 7)
SOURCE_TX=(
  "14-7,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-15,20-19,8-20"
  "14-7,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-15,20-19,8-20"
  "14-10,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-15,20-19,6-15"
  "14-10,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-15,20-19,6-15"
  "14-10,14-7,20-19,6-15,8-20"
  "14-10,14-7,20-19,6-15,8-20"
  "14-10,14-7,20-15,6-15,8-20"
  "14-10,14-7,20-15,6-15,8-20"
)
HELD_TX=(14-10 6-15 14-10 6-15 14-7 8-20 14-7 8-20 20-15 20-15 20-19 20-19)

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing ManySig: ${WISIG_PKL}" >&2; exit 2; }
  [[ -f "${MANYTX_PKL}" ]] || { echo "missing ManyTx: ${MANYTX_PKL}" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  for candidate in "${CANDIDATES[@]}"; do
    [[ -f "${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth" ]] || {
      echo "missing frozen checkpoint: ${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth" >&2
      exit 2
    }
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

declare -a export_pids=()

export_one() {
  local index="$1"
  local candidate="${CANDIDATES[index]}"
  local gpu="${GPUS[index]}"
  local source_tx="${SOURCE_TX[index]}"
  local held_tx="${HELD_TX[index]}"
  local out_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.export.out"
  local -a command=(
    "${PYTHON}" -u "${EXPORTER}"
    --ckpt "${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth"
    --wisig_pkl "${WISIG_PKL}"
    --new_wisig_pkl "${MANYTX_PKL}"
    --out_npz "${out_dir}/features.npz"
    --feature_name z_id
    --source_tx_ids "${source_tx}"
    --target_old_tx_ids "${held_tx}"
    --proxy_unknown_tx_ids "${PROXY_TX}"
    --source_days "${SOURCE_DAYS}" --source_rxs "${SOURCE_RXS}"
    --target_old_days "${SOURCE_DAYS}" --target_old_rxs "${SOURCE_RXS}"
    --proxy_unknown_days "${SOURCE_DAYS}" --proxy_unknown_rxs "${SOURCE_RXS}"
    --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
    --max_samples_per_combo 0 --max_samples_per_tx 400
    --batch_size 512 --device cuda:0 --seed 7281105
    --source_channel_view clean --target_old_channel_view clean
    --proxy_unknown_channel_view clean
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN-EXPORT] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${gpu}" "${CODE_ROOT}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
  export_pids+=("$!")
}

for index in "${!CANDIDATES[@]}"; do
  export_one "${index}"
done

if [[ "${DRY_RUN}" == "0" ]]; then
  printf 'candidate|pid|physical_gpu|checkpoint|output_npz|log_path|exit_code\n' >"${LOG_ROOT}/export_completion.tsv"
  export_failed=0
  for index in "${!CANDIDATES[@]}"; do
    rc=0
    if wait "${export_pids[index]}"; then rc=0; else rc=$?; export_failed=1; fi
    candidate="${CANDIDATES[index]}"
    printf '%s|%s|%s|%s|%s|%s|%s\n' \
      "${candidate}" "${export_pids[index]}" "${GPUS[index]}" \
      "${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth" \
      "${RUN_ROOT}/${candidate}/features.npz" \
      "${LOG_ROOT}/${candidate}.export.out" "${rc}" >>"${LOG_ROOT}/export_completion.tsv"
  done
  [[ "${export_failed}" == "0" ]] || exit 8
fi

score_one() {
  local index="$1"
  local kind="$2"
  local candidate="${CANDIDATES[index]}"
  local source_tx="${SOURCE_TX[index]}"
  local unknown_tx unknown_role
  case "${kind}" in
    proxy) unknown_tx="${PROXY_TX}"; unknown_role="proxy_unknown" ;;
    held) unknown_tx="${HELD_TX[index]}"; unknown_role="target_old" ;;
    *) echo "unsupported score kind: ${kind}" >&2; return 2 ;;
  esac
  local out_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.${kind}.out"
  local -a command=(
    "${PYTHON}" -u "${SCORER}"
    --feature_npz "${out_dir}/features.npz"
    --source_tx_ids "${source_tx}"
    --unknown_tx_ids "${unknown_tx}"
    --known_query_roles source --unknown_query_roles "${unknown_role}"
    --calibration_roles source --conf_quantile 0.05
    --margin_quantile 0.05 --energy_quantile 0.95
    --disable_conf_gate --disable_margin_gate --unknown_far_target 0.05
    --output_json "${out_dir}/${kind}_metrics.json"
    --score_table_csv "${out_dir}/${kind}_scores.csv"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN-SCORE]'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  rc=0
  if PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1; then rc=0; else rc=$?; fi
  printf '%s|%s|%s|%s|%s|%s\n' \
    "${candidate}" "${kind}" "${unknown_role}" \
    "${out_dir}/${kind}_metrics.json" "${out_dir}/${kind}_scores.csv" "${rc}" \
    >>"${LOG_ROOT}/score_completion.tsv"
  return "${rc}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for index in "${!CANDIDATES[@]}"; do
    score_one "${index}" proxy
    score_one "${index}" held
  done
  exit 0
fi

printf 'candidate|kind|unknown_role|metrics_json|score_csv|exit_code\n' >"${LOG_ROOT}/score_completion.tsv"
score_failed=0
for index in "${!CANDIDATES[@]}"; do
  score_one "${index}" proxy || score_failed=1
  score_one "${index}" held || score_failed=1
done
[[ "${score_failed}" == "0" ]] || exit 8
