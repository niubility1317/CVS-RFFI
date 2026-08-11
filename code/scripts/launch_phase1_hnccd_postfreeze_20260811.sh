#!/usr/bin/env bash
set -euo pipefail

# Invoke explicitly as: bash scripts/launch_phase1_hnccd_postfreeze_20260811.sh [--dry-run]
# Frozen P1-HNCCD postfreeze closure: 12 clean exports, 12 source-LEO/binding
# exports, 12 fixed-400 proxy bindings and six same-fold C/G pair scores.
# This launcher is mechanical; receipt and pair semantics are validated by the
# HNCCD exporter/evaluator, not selected here.
POSTFREEZE_RUN_ID="${POSTFREEZE_RUN_ID:-phase1_hnccd_postfreeze_20260811_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${PROJECT_ROOT}/runs/phase1_hnccd12_20260811_v1}"
POSTFREEZE_ROOT="${POSTFREEZE_ROOT:-${PROJECT_ROOT}/runs/${POSTFREEZE_RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${POSTFREEZE_RUN_ID}}"
HNCCD_CLEAN_EXPORT_SCRIPT="${CODE_ROOT}/export_phase1_hnccd_features.py"
HNCCD_LEO_EXPORT_SCRIPT="${CODE_ROOT}/export_phase1_hnccd_leo_features.py"
LOGITS_REJECT_SCRIPT="${CODE_ROOT}/scripts/eval_phase1_logits_open_set_reject.py"
PAIR_EVAL_SCRIPT="${CODE_ROOT}/evaluate_phase1_hnccd_postfreeze_pair.py"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DAYS="2021_03_01,2021_03_08"
SOURCE_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
SAT_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SOURCE_SAT_SEED="7281718"
EXPORT_SEED="7281105"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
MAX_PER_TX="400"
EXPORT_BATCH="32"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}
[[ "${POSTFREEZE_RUN_ID}" == "phase1_hnccd_postfreeze_20260811_v1" ]] || {
  echo "POSTFREEZE_RUN_ID is frozen to phase1_hnccd_postfreeze_20260811_v1" >&2
  exit 3
}
[[ "$(basename "${TRAIN_RUN_ROOT}")" == "phase1_hnccd12_20260811_v1" ]] || {
  echo "TRAIN_RUN_ROOT leaf must be phase1_hnccd12_20260811_v1" >&2
  exit 3
}
[[ "$(basename "${POSTFREEZE_ROOT}")" == "phase1_hnccd_postfreeze_20260811_v1" ]] || {
  echo "POSTFREEZE_ROOT leaf must bind the frozen HNCCD matrix" >&2
  exit 3
}
[[ "${POSTFREEZE_ROOT}" != "${TRAIN_RUN_ROOT}" ]] || {
  echo "postfreeze root must differ from immutable training root" >&2
  exit 3
}
for required in "${HNCCD_CLEAN_EXPORT_SCRIPT}" "${HNCCD_LEO_EXPORT_SCRIPT}" "${LOGITS_REJECT_SCRIPT}" "${PAIR_EVAL_SCRIPT}"; do
  [[ -f "${required}" ]] || { echo "missing HNCCD postfreeze script: ${required}" >&2; exit 2; }
done

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing ManySig dataset: ${WISIG_PKL}" >&2; exit 2; }
  [[ ! -e "${POSTFREEZE_ROOT}" ]] || { echo "refusing to overwrite postfreeze root: ${POSTFREEZE_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite postfreeze log root: ${LOG_ROOT}" >&2; exit 3; }
  mkdir -p "${POSTFREEZE_ROOT}" "${LOG_ROOT}"
fi

print_command() {
  local stage="$1"; local device="$2"; shift 2
  printf '[DRY-RUN][%s] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${stage}" "${device}" "${CODE_ROOT}:${PROJECT_ROOT}"
  printf ' %q' "$@"
  printf '\n'
}

declare -a pids folds arms gpus candidates logs

run_candidate() {
  local fold="$1"; local arm="$2"; local gpu="$3"
  local fold_index=$((fold - 1))
  local candidate="F${fold}${arm}_HNCCD12"
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local known_tx="${FOLD_KNOWN_VAL_TX[fold_index]}"
  local proxy_tx="${FOLD_PROXY_TX[fold_index]}"
  local checkpoint="${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth"
  local candidate_dir="${POSTFREEZE_ROOT}/${candidate}"
  local clean_npz="${candidate_dir}/icmt_clean_l_v_proxy_final_only.npz"
  local leo_npz="${candidate_dir}/source_leo_final_only.npz"
  local leo_binding="${candidate_dir}/source_leo_binding.json"
  local proxy_metrics="${candidate_dir}/proxy_logits_open_set_metrics.json"
  local proxy_scores="${candidate_dir}/proxy_logits_open_set_scores.csv"
  local -a clean_command=(
    "${PYTHON}" -u "${HNCCD_CLEAN_EXPORT_SCRIPT}"
    --ckpt "${checkpoint}" --wisig_pkl "${WISIG_PKL}" --out_npz "${clean_npz}"
    --source_tx_ids "${train_tx}" --known_validation_tx_ids "${known_tx}" --proxy_unknown_tx_ids "${proxy_tx}"
    --expected-wisig-sha256 "${WISIG_SHA256}" --batch_size 256 --device cuda:0
  )
  local -a leo_command=(
    "${PYTHON}" -u "${HNCCD_LEO_EXPORT_SCRIPT}"
    --ckpt "${checkpoint}" --wisig-pkl "${WISIG_PKL}" --out-npz "${leo_npz}" --binding-json "${leo_binding}"
    --training-run-root "${TRAIN_RUN_ROOT}" --postfreeze-output-root "${POSTFREEZE_ROOT}"
    --candidate-id "${candidate}" --fold-index "${fold}" --arm "${arm}" --feature-name z_id --source_only_export
    --source-tx-ids "${train_tx}" --source-days "${SOURCE_DAYS}" --source-rxs "${SOURCE_RXS}"
    --source-channel-view satellite --source-sat-scenarios "${SAT_SCENARIOS}" --source-sat-seed "${SOURCE_SAT_SEED}"
    --star-ground-channel-impl simplified_leo_residual --satellite-tta-policy none --wisig-equalized 1
    --wisig-domain rx_day --wisig-out-len 256 --max-samples-per-combo 0 --max_samples_per_tx "${MAX_PER_TX}"
    --batch_size "${EXPORT_BATCH}" --seed "${EXPORT_SEED}" --expected-wisig-sha256 "${WISIG_SHA256}" --device cuda:0
  )
  local -a proxy_command=(
    "${PYTHON}" -u "${LOGITS_REJECT_SCRIPT}" --feature_npz "${clean_npz}" --source_tx_ids "${train_tx}"
    --known_query_roles source_validation_known --unknown_query_roles proxy_unknown --calibration_roles source_validation_known
    --unknown_far_target 0.05 --output_json "${proxy_metrics}" --score_table_csv "${proxy_scores}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command HNCCD_CLEAN_EXPORT "${gpu}" "${clean_command[@]}"
    print_command HNCCD_LEO_EXPORT_AND_BIND "${gpu}" "${leo_command[@]}"
    print_command FROZEN_LOGITS_PROXY_BINDING "${gpu}" "${proxy_command[@]}"
    return 0
  fi
  [[ -f "${checkpoint}" ]] || { echo "missing HNCCD final-only checkpoint: ${checkpoint}" >&2; return 2; }
  mkdir -p "${candidate_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${clean_command[@]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${leo_command[@]}"
  PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${proxy_command[@]}"
}

launch_candidate() {
  local fold="$1"; local arm="$2"; local gpu="$3"
  local candidate="F${fold}${arm}_HNCCD12"
  local log_path="${LOG_ROOT}/${candidate}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_candidate "${fold}" "${arm}" "${gpu}"
    return 0
  fi
  run_candidate "${fold}" "${arm}" "${gpu}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); arms+=("${arm}"); gpus+=("${gpu}"); candidates+=("${candidate}"); logs+=("${log_path}")
}

# Established six-fold C/G GPU mapping; candidate-internal work is serial.
launch_candidate 1 C 0
launch_candidate 5 G 0
launch_candidate 1 G 1
launch_candidate 5 C 1
launch_candidate 2 C 2
launch_candidate 6 G 2
launch_candidate 2 G 3
launch_candidate 6 C 3
launch_candidate 3 C 4
launch_candidate 3 G 5
launch_candidate 4 C 6
launch_candidate 4 G 7

run_pair() {
  local fold="$1"; local fold_index=$((fold - 1)); local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local c_candidate="F${fold}C_HNCCD12"; local g_candidate="F${fold}G_HNCCD12"
  local c_checkpoint="${TRAIN_RUN_ROOT}/${c_candidate}/final_ssdg.pth"; local g_checkpoint="${TRAIN_RUN_ROOT}/${g_candidate}/final_ssdg.pth"
  local output_json="${POSTFREEZE_ROOT}/F${fold}_C_vs_G_pair_metrics.json"
  local -a pair_command=(
    "${PYTHON}" -u "${PAIR_EVAL_SCRIPT}"
    --c-clean-npz "${POSTFREEZE_ROOT}/${c_candidate}/icmt_clean_l_v_proxy_final_only.npz"
    --g-clean-npz "${POSTFREEZE_ROOT}/${g_candidate}/icmt_clean_l_v_proxy_final_only.npz"
    --c-leo-npz "${POSTFREEZE_ROOT}/${c_candidate}/source_leo_final_only.npz"
    --g-leo-npz "${POSTFREEZE_ROOT}/${g_candidate}/source_leo_final_only.npz"
    --c-leo-binding-json "${POSTFREEZE_ROOT}/${c_candidate}/source_leo_binding.json"
    --g-leo-binding-json "${POSTFREEZE_ROOT}/${g_candidate}/source_leo_binding.json"
    --c-final-checkpoint "${c_checkpoint}" --g-final-checkpoint "${g_checkpoint}"
    --c-proxy-metrics-json "${POSTFREEZE_ROOT}/${c_candidate}/proxy_logits_open_set_metrics.json"
    --g-proxy-metrics-json "${POSTFREEZE_ROOT}/${g_candidate}/proxy_logits_open_set_metrics.json"
    --c-proxy-scores-csv "${POSTFREEZE_ROOT}/${c_candidate}/proxy_logits_open_set_scores.csv"
    --g-proxy-scores-csv "${POSTFREEZE_ROOT}/${g_candidate}/proxy_logits_open_set_scores.csv"
    --source-tx-ids "${train_tx}" --candidate-pair "F${fold}_C_vs_G" --fold-index "${fold}"
    --postfreeze-matrix-id "${POSTFREEZE_RUN_ID}" --postfreeze-output-root "${POSTFREEZE_ROOT}"
    --training-run-root "${TRAIN_RUN_ROOT}" --expected-scenarios "${SAT_SCENARIOS}"
    --expected-source-days "${SOURCE_DAYS}" --expected-source-rxs "${SOURCE_RXS}"
    --source-sat-seed "${SOURCE_SAT_SEED}" --expected-source-count 1600 --expected-proxy-count 400
    --output-metrics-json "${output_json}"
  )
  if [[ "${fold}" == "6" ]]; then
    pair_command+=(--aggregate-prior-pair-metrics-json
      "${POSTFREEZE_ROOT}/F1_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F2_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F3_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F4_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F5_C_vs_G_pair_metrics.json")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command HNCCD_PAIR_SCORE CPU "${pair_command[@]}"
    return 0
  fi
  [[ -f "${c_checkpoint}" && -f "${g_checkpoint}" ]] || { echo "missing final-only HNCCD checkpoint for fold ${fold}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${pair_command[@]}" >"${LOG_ROOT}/F${fold}_C_vs_G_pair.out" 2>&1
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do run_pair "${fold}"; done
  exit 0
fi

printf 'pid|fold|arm|physical_gpu|candidate|log_path\n' >"${LOG_ROOT}/candidate_pids.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s\n' "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" "${candidates[index]}" "${logs[index]}" >>"${LOG_ROOT}/candidate_pids.tsv"
done
status=0
for index in "${!pids[@]}"; do wait "${pids[index]}" || status=8; done
[[ "${status}" == "0" ]] || exit "${status}"
for fold in 1 2 3 4 5 6; do run_pair "${fold}"; done
