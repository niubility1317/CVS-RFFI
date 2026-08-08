#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-phase1_wrc_nct_leo6x3_20260809_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2}"
CLEAN_ROOT="${CLEAN_ROOT:-${PROJECT_ROOT}/runs/phase1_wrc_nct_clean6_20260809_v2}"
GI_ROOT="${GI_ROOT:-${PROJECT_ROOT}/runs/phase1_gi_epior_clean6_20260809_v3}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
EXPORTER="${CODE_ROOT}/export_spaceborne_features.py"
SCORER="${CODE_ROOT}/scripts/eval_phase1_wrc_nct_leo.py"
SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SOURCE_DAYS="2021_03_01,2021_03_08"
SOURCE_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

FOLDS=(F1 F2 F3 F4 F5 F6)
CANDIDATES=(F1C_ManyTxRealOE12 F2C_ManyTxRealOE12 F3C_ManyTxRealOE12 F4C_ManyTxRealOE12 F5C_ManyTxRealOE12 F6C_ManyTxRealOE12)
GPUS=(0 1 2 3 4 5)
SOURCE_TX=(
  "14-7,20-15,20-19,6-15,8-20"
  "14-10,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-19,6-15,8-20"
  "14-10,14-7,20-15,6-15,8-20"
  "14-10,14-7,20-15,20-19,8-20"
  "14-10,14-7,20-15,20-19,6-15"
)

[[ -f "${EXPORTER}" ]] || { echo "missing exporter: ${EXPORTER}" >&2; exit 2; }
[[ -f "${SCORER}" ]] || { echo "missing LEO scorer: ${SCORER}" >&2; exit 2; }
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  [[ -f "${WISIG_PKL}" ]] || { echo "missing ManySig: ${WISIG_PKL}" >&2; exit 2; }
  for index in "${!FOLDS[@]}"; do
    [[ -f "${TRAIN_ROOT}/${CANDIDATES[index]}/final_ssdg.pth" ]] || exit 2
    [[ -f "${CLEAN_ROOT}/${FOLDS[index]}/wrc_nct_readout.json" ]] || exit 2
    [[ -f "${CLEAN_ROOT}/${FOLDS[index]}/clean_scores.csv" ]] || exit 2
    [[ -f "${GI_ROOT}/${FOLDS[index]}/gi_epior_bundle.npz" ]] || exit 2
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

export_one() {
  local index="$1" fold="${FOLDS[index]}" candidate="${CANDIDATES[index]}" gpu="${GPUS[index]}"
  local -a command=(
    "${PYTHON}" -u "${EXPORTER}"
    --ckpt "${TRAIN_ROOT}/${candidate}/final_ssdg.pth"
    --wisig_pkl "${WISIG_PKL}"
    --source_only_export
    --out_npz "${RUN_ROOT}/${fold}/leo_features.npz"
    --feature_name z_id --source_tx_ids "${SOURCE_TX[index]}"
    --source_days "${SOURCE_DAYS}" --source_rxs "${SOURCE_RXS}"
    --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
    --max_samples_per_combo 0 --max_samples_per_tx 400
    --batch_size 32 --device cuda:0 --seed 7281105
    --source_channel_view satellite --source_sat_scenarios "${SCENARIOS}"
    --source_sat_seed 7281718 --star_ground_channel_impl simplified_leo_residual
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN-LEO-EXPORT] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${gpu}" "${CODE_ROOT}"
    printf ' %q' "${command[@]}"; printf '\n'; return 0
  fi
  mkdir -p "${RUN_ROOT}/${fold}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${LOG_ROOT}/${fold}.export.out" 2>&1
}

score_one() {
  local index="$1" fold="${FOLDS[index]}"
  local -a command=(
    "${PYTHON}" -u "${SCORER}"
    --feature-npz "${RUN_ROOT}/${fold}/leo_features.npz"
    --gi-bundle "${GI_ROOT}/${fold}/gi_epior_bundle.npz"
    --wrc-readout-json "${CLEAN_ROOT}/${fold}/wrc_nct_readout.json"
    --clean-scores-csv "${CLEAN_ROOT}/${fold}/clean_scores.csv"
    --source-tx-ids "${SOURCE_TX[index]}"
    --expected-scenarios "${SCENARIOS}"
    --output-metrics-json "${RUN_ROOT}/${fold}/leo_metrics.json"
    --output-scores-csv "${RUN_ROOT}/${fold}/leo_scores.csv"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN-LEO-SCORE] OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=%q' "${CODE_ROOT}"
    printf ' %q' "${command[@]}"; printf '\n'; return 0
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${LOG_ROOT}/${fold}.score.out" 2>&1
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for index in "${!FOLDS[@]}"; do export_one "${index}"; done
  for index in "${!FOLDS[@]}"; do score_one "${index}"; done
  exit 0
fi

printf 'fold|candidate|pid|gpu|checkpoint|npz|log|exit_code\n' >"${LOG_ROOT}/export_completion.tsv"
declare -a export_pids=()
for index in "${!FOLDS[@]}"; do export_one "${index}" & export_pids+=("$!"); done
export_failed=0
for index in "${!FOLDS[@]}"; do
  rc=0; if wait "${export_pids[index]}"; then rc=0; else rc=$?; export_failed=1; fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' "${FOLDS[index]}" "${CANDIDATES[index]}" "${export_pids[index]}" \
    "${GPUS[index]}" "${TRAIN_ROOT}/${CANDIDATES[index]}/final_ssdg.pth" "${RUN_ROOT}/${FOLDS[index]}/leo_features.npz" \
    "${LOG_ROOT}/${FOLDS[index]}.export.out" "${rc}" >>"${LOG_ROOT}/export_completion.tsv"
done
[[ "${export_failed}" == "0" ]] || exit 8

printf 'fold|pid|readout|clean_scores|metrics|scores|log|exit_code\n' >"${LOG_ROOT}/score_completion.tsv"
declare -a score_pids=()
for index in "${!FOLDS[@]}"; do score_one "${index}" & score_pids+=("$!"); done
score_failed=0
for index in "${!FOLDS[@]}"; do
  rc=0; if wait "${score_pids[index]}"; then rc=0; else rc=$?; score_failed=1; fi
  fold="${FOLDS[index]}"
  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' "${fold}" "${score_pids[index]}" \
    "${CLEAN_ROOT}/${fold}/wrc_nct_readout.json" "${CLEAN_ROOT}/${fold}/clean_scores.csv" \
    "${RUN_ROOT}/${fold}/leo_metrics.json" "${RUN_ROOT}/${fold}/leo_scores.csv" "${LOG_ROOT}/${fold}.score.out" "${rc}" \
    >>"${LOG_ROOT}/score_completion.tsv"
done
[[ "${score_failed}" == "0" ]] || exit 8
