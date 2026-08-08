#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-phase1_wrc_nct_clean6_20260809_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${PROJECT_ROOT}/runs/phase1_gi_epior_clean6_20260809_v3}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
ENTRY="${CODE_ROOT}/scripts/eval_phase1_wrc_nct.py"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -f "${ENTRY}" ]] || { echo "missing WRC-NCT entry: ${ENTRY}" >&2; exit 2; }

FOLDS=(F1 F2 F3 F4 F5 F6)
CANDIDATES=(
  F1C_ManyTxRealOE12 F2C_ManyTxRealOE12 F3C_ManyTxRealOE12
  F4C_ManyTxRealOE12 F5C_ManyTxRealOE12 F6C_ManyTxRealOE12
)
SOURCE_TX=(
  "14-7,20-15,20-19,6-15,8-20"
  "14-10,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-19,6-15,8-20"
  "14-10,14-7,20-15,6-15,8-20"
  "14-10,14-7,20-15,20-19,8-20"
  "14-10,14-7,20-15,20-19,6-15"
)
HELD_TX=(14-10 14-7 20-15 20-19 6-15 8-20)

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  for index in "${!FOLDS[@]}"; do
    [[ -f "${INPUT_ROOT}/${CANDIDATES[index]}/features.npz" ]] || {
      echo "missing frozen input NPZ: ${INPUT_ROOT}/${CANDIDATES[index]}/features.npz" >&2
      exit 2
    }
    [[ -f "${BUNDLE_ROOT}/${FOLDS[index]}/gi_epior_bundle.npz" ]] || {
      echo "missing frozen GI bundle: ${BUNDLE_ROOT}/${FOLDS[index]}/gi_epior_bundle.npz" >&2
      exit 2
    }
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

score_one() {
  local index="$1" fold="${FOLDS[index]}" candidate="${CANDIDATES[index]}"
  local out_dir="${RUN_ROOT}/${fold}" log_path="${LOG_ROOT}/${fold}.out"
  local -a command=(
    "${PYTHON}" -u "${ENTRY}"
    --feature-npz "${INPUT_ROOT}/${candidate}/features.npz"
    --gi-bundle "${BUNDLE_ROOT}/${fold}/gi_epior_bundle.npz"
    --source-tx-ids "${SOURCE_TX[index]}"
    --held-tx-ids "${HELD_TX[index]}"
    --source-role source --held-role target_old --proxy-role proxy_unknown
    --view-name clean
    --output-readout-json "${out_dir}/wrc_nct_readout.json"
    --output-torchscript "${out_dir}/wrc_nct_runtime.ts"
    --output-metrics-json "${out_dir}/clean_metrics.json"
    --output-scores-csv "${out_dir}/clean_scores.csv"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN-WRC] OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=%q' "${CODE_ROOT}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "${out_dir}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for index in "${!FOLDS[@]}"; do score_one "${index}"; done
  exit 0
fi

printf 'fold|candidate|pid|feature_npz|gi_bundle|readout|runtime|metrics|scores|log|exit_code\n' >"${LOG_ROOT}/completion.tsv"
declare -a pids=()
for index in "${!FOLDS[@]}"; do
  score_one "${index}" &
  pids+=("$!")
done

failed=0
for index in "${!FOLDS[@]}"; do
  rc=0
  if wait "${pids[index]}"; then rc=0; else rc=$?; failed=1; fi
  fold="${FOLDS[index]}" candidate="${CANDIDATES[index]}"
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${fold}" "${candidate}" "${pids[index]}" "${INPUT_ROOT}/${candidate}/features.npz" \
    "${BUNDLE_ROOT}/${fold}/gi_epior_bundle.npz" "${RUN_ROOT}/${fold}/wrc_nct_readout.json" \
    "${RUN_ROOT}/${fold}/wrc_nct_runtime.ts" "${RUN_ROOT}/${fold}/clean_metrics.json" \
    "${RUN_ROOT}/${fold}/clean_scores.csv" "${LOG_ROOT}/${fold}.out" "${rc}" \
    >>"${LOG_ROOT}/completion.tsv"
done
[[ "${failed}" == "0" ]] || exit 8
