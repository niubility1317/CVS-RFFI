#!/usr/bin/env bash
set -euo pipefail

# Final-only postfreeze closure for the completed P1-CP-SFCE C/G matrix.
# It exports immutable checkpoint features, runs the fixed source-proxy
# diagnostic, and evaluates six CPU-serial C/G pairs.  It never trains, fits,
# calibrates, sweeps, or selects a checkpoint.
POSTFREEZE_RUN_ID="${POSTFREEZE_RUN_ID:-phase1_cp_sfce_postfreeze_20260809_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${PROJECT_ROOT}/runs/phase1_cp_sfce12_20260809_v2}"
POSTFREEZE_ROOT="${POSTFREEZE_ROOT:-${PROJECT_ROOT}/runs/${POSTFREEZE_RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${POSTFREEZE_RUN_ID}}"
FEATURE_EXPORT_SCRIPT="${CODE_ROOT}/export_spaceborne_features.py"
LOGITS_REJECT_SCRIPT="${CODE_ROOT}/scripts/eval_phase1_logits_open_set_reject.py"
PAIR_EVAL_SCRIPT="${CODE_ROOT}/scripts/eval_phase1_cp_sfce_pair.py"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DAYS="2021_03_01,2021_03_08"
SOURCE_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
SAT_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SOURCE_SAT_SEED="7281718"
EXPORT_SEED="7281105"
MAX_PER_TX="400" # train4=1600; held/proxy singleton TX each=400
EXPORT_BATCH="32"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$(basename "${TRAIN_RUN_ROOT}")" == "phase1_cp_sfce12_20260809_v2" ]] || {
  echo "TRAIN_RUN_ROOT leaf must be phase1_cp_sfce12_20260809_v2" >&2
  exit 3
}
[[ "${POSTFREEZE_ROOT}" != "${TRAIN_RUN_ROOT}" ]] || { echo "postfreeze root must differ from immutable training root" >&2; exit 3; }
for required in "${FEATURE_EXPORT_SCRIPT}" "${LOGITS_REJECT_SCRIPT}" "${PAIR_EVAL_SCRIPT}"; do
  [[ -f "${required}" ]] || { echo "missing postfreeze script: ${required}" >&2; exit 2; }
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
  local stage="$1"
  local device="$2"
  shift 2
  printf '[DRY-RUN][%s] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${stage}" "${device}" "${CODE_ROOT}:${PROJECT_ROOT}"
  printf ' %q' "$@"
  printf '\n'
}

declare -a pids folds arms gpus candidates logs

run_candidate() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local fold_index=$((fold - 1))
  local candidate="F${fold}${arm}_CP_SFCE12"
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local known_tx="${FOLD_KNOWN_VAL_TX[fold_index]}"
  local proxy_tx="${FOLD_PROXY_TX[fold_index]}"
  local checkpoint="${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth"
  local candidate_dir="${POSTFREEZE_ROOT}/${candidate}"
  local clean_npz="${candidate_dir}/clean_development.npz"
  local leo_npz="${candidate_dir}/source_leo_final_only.npz"
  local proxy_metrics="${candidate_dir}/proxy_logits_open_set_metrics.json"
  local proxy_scores="${candidate_dir}/proxy_logits_open_set_scores.csv"
  local -a clean_command=(
    "${PYTHON}" -u "${FEATURE_EXPORT_SCRIPT}"
    --ckpt "${checkpoint}"
    --wisig_pkl "${WISIG_PKL}"
    --out_npz "${clean_npz}"
    --feature_name z_id
    --source_tx_ids "${train_tx}"
    --target_old_tx_ids "${known_tx}"
    --proxy_unknown_tx_ids "${proxy_tx}"
    --source_days "${SOURCE_DAYS}"
    --source_rxs "${SOURCE_RXS}"
    --target_old_days "${SOURCE_DAYS}"
    --target_old_rxs "${SOURCE_RXS}"
    --proxy_unknown_days "${SOURCE_DAYS}"
    --proxy_unknown_rxs "${SOURCE_RXS}"
    --source_channel_view clean
    --target_old_channel_view clean
    --proxy_unknown_channel_view clean
    --max_samples_per_tx "${MAX_PER_TX}"
    --batch_size "${EXPORT_BATCH}"
    --seed "${EXPORT_SEED}"
    --device cuda:0
  )
  local -a leo_command=(
    "${PYTHON}" -u "${FEATURE_EXPORT_SCRIPT}"
    --ckpt "${checkpoint}"
    --wisig_pkl "${WISIG_PKL}"
    --out_npz "${leo_npz}"
    --feature_name z_id
    --source_only_export
    --source_tx_ids "${train_tx}"
    --source_days "${SOURCE_DAYS}"
    --source_rxs "${SOURCE_RXS}"
    --source_channel_view satellite
    --source_sat_scenarios "${SAT_SCENARIOS}"
    --source_sat_seed "${SOURCE_SAT_SEED}"
    --star_ground_channel_impl simplified_leo_residual
    --satellite_tta_policy none
    --max_samples_per_tx "${MAX_PER_TX}"
    --batch_size "${EXPORT_BATCH}"
    --seed "${EXPORT_SEED}"
    --device cuda:0
  )
  local -a proxy_command=(
    "${PYTHON}" -u "${LOGITS_REJECT_SCRIPT}"
    --feature_npz "${clean_npz}"
    --source_tx_ids "${train_tx}"
    --known_query_roles source
    --unknown_query_roles proxy_unknown
    --calibration_roles source
    --unknown_far_target 0.05
    --output_json "${proxy_metrics}"
    --score_table_csv "${proxy_scores}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command CLEAN_EXPORT "${gpu}" "${clean_command[@]}"
    print_command LEO_EXPORT "${gpu}" "${leo_command[@]}"
    print_command PROXY_SCORE "${gpu}" "${proxy_command[@]}"
    return 0
  fi
  [[ -f "${checkpoint}" ]] || { echo "missing final-only checkpoint: ${checkpoint}" >&2; return 2; }
  mkdir -p "${candidate_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${clean_command[@]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${leo_command[@]}"
  PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${proxy_command[@]}"
}

launch_candidate() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local candidate="F${fold}${arm}_CP_SFCE12"
  local log_path="${LOG_ROOT}/${candidate}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_candidate "${fold}" "${arm}" "${gpu}"
    return 0
  fi
  run_candidate "${fold}" "${arm}" "${gpu}" >"${log_path}" 2>&1 &
  pids+=("$!")
  folds+=("${fold}")
  arms+=("${arm}")
  gpus+=("${gpu}")
  candidates+=("${candidate}")
  logs+=("${log_path}")
}

# Same fixed mapping as continuation training; at most two export pipelines
# are active per physical GPU.
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
  local fold="$1"
  local fold_index=$((fold - 1))
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local c_candidate="F${fold}C_CP_SFCE12"
  local g_candidate="F${fold}G_CP_SFCE12"
  local c_checkpoint="${TRAIN_RUN_ROOT}/${c_candidate}/final_ssdg.pth"
  local g_checkpoint="${TRAIN_RUN_ROOT}/${g_candidate}/final_ssdg.pth"
  local output_json="${POSTFREEZE_ROOT}/F${fold}_C_vs_G_pair_metrics.json"
  local -a pair_command=(
    "${PYTHON}" -u "${PAIR_EVAL_SCRIPT}"
    --c-clean-npz "${POSTFREEZE_ROOT}/${c_candidate}/clean_development.npz"
    --g-clean-npz "${POSTFREEZE_ROOT}/${g_candidate}/clean_development.npz"
    --c-leo-npz "${POSTFREEZE_ROOT}/${c_candidate}/source_leo_final_only.npz"
    --g-leo-npz "${POSTFREEZE_ROOT}/${g_candidate}/source_leo_final_only.npz"
    --c-final-checkpoint "${c_checkpoint}"
    --g-final-checkpoint "${g_checkpoint}"
    --c-proxy-metrics-json "${POSTFREEZE_ROOT}/${c_candidate}/proxy_logits_open_set_metrics.json"
    --g-proxy-metrics-json "${POSTFREEZE_ROOT}/${g_candidate}/proxy_logits_open_set_metrics.json"
    --candidate-pair "F${fold}_C_vs_G"
    --fold-index "${fold}"
    --postfreeze-matrix-id "${POSTFREEZE_RUN_ID}"
    --postfreeze-output-root "${POSTFREEZE_ROOT}"
    --training-run-root "${TRAIN_RUN_ROOT}"
    --source-tx-ids "${train_tx}"
    --expected-scenarios "${SAT_SCENARIOS}"
    --expected-source-days "${SOURCE_DAYS}"
    --expected-source-rxs "${SOURCE_RXS}"
    --source-sat-seed "${SOURCE_SAT_SEED}"
    --expected-source-count 1600
    --expected-target-old-count 400
    --expected-proxy-count 400
    --output-metrics-json "${output_json}"
  )
  if [[ "${fold}" == "6" ]]; then
    pair_command+=(
      --aggregate-prior-pair-metrics-json
      "${POSTFREEZE_ROOT}/F1_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F2_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F3_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F4_C_vs_G_pair_metrics.json,${POSTFREEZE_ROOT}/F5_C_vs_G_pair_metrics.json"
    )
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command PAIR_SCORE CPU "${pair_command[@]}"
    return 0
  fi
  [[ -f "${c_checkpoint}" && -f "${g_checkpoint}" ]] || { echo "missing final-only C/G checkpoint for fold ${fold}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" "${pair_command[@]}" >"${LOG_ROOT}/F${fold}_C_vs_G_pair.out" 2>&1
}

if [[ "${DRY_RUN}" == "1" ]]; then
  run_pair 1
  run_pair 2
  run_pair 3
  run_pair 4
  run_pair 5
  run_pair 6
  exit 0
fi

printf 'pid|fold|arm|physical_gpu|candidate|log_path\n' >"${LOG_ROOT}/candidate_pids.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${candidates[index]}" "${logs[index]}" >>"${LOG_ROOT}/candidate_pids.tsv"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    status=8
  fi
done
[[ "${status}" == "0" ]] || exit "${status}"

# Pair scoring is CPU-serial; F6 seals the six-fold aggregate only after the
# immutable F1--F5 per-fold JSONs exist in the same postfreeze root.
run_pair 1
run_pair 2
run_pair 3
run_pair 4
run_pair 5
run_pair 6
