#!/usr/bin/env bash
set -euo pipefail

# Frozen zero-training source-only density readout matrix over the six C-arm
# postfreeze NPZs. DRY_RUN=1 prints all 24 score commands without writes.
RUN_ID="${RUN_ID:-phase1_loto_c_density_readout12_20260808_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/runs/phase1_loto_clsgeo12_20260808_v1/postfreeze_audit_v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
PROTOTYPE_EVAL="${CODE_ROOT}/scripts/eval_phase1_prototype_reject.py"
KNN_EVAL="${CODE_ROOT}/scripts/eval_phase1_knn_reject.py"
DRY_RUN="${DRY_RUN:-0}"

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -f "${PROTOTYPE_EVAL}" ]] || { echo "missing evaluator: ${PROTOTYPE_EVAL}" >&2; exit 2; }
[[ -f "${KNN_EVAL}" ]] || { echo "missing evaluator: ${KNN_EVAL}" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)
FOLD_SECONDARY_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PRIMARY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -d "${INPUT_ROOT}" ]] || { echo "missing input root: ${INPUT_ROOT}" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
  printf 'candidate|fold|readout|kind|source_tx|unknown_tx|unknown_role|exit_code\n' >"${LOG_ROOT}/completion.tsv"
fi

run_score() {
  local fold="$1"
  local readout="$2"
  local kind="$3"
  local index=$((fold - 1))
  local source_tx="${FOLD_TRAIN_TX[index]}"
  local unknown_tx unknown_role
  if [[ "${kind}" == "primary" ]]; then
    unknown_tx="${FOLD_PRIMARY_TX[index]}"
    unknown_role="proxy_unknown"
  elif [[ "${kind}" == "secondary" ]]; then
    unknown_tx="${FOLD_SECONDARY_TX[index]}"
    unknown_role="target_old"
  else
    echo "unsupported kind: ${kind}" >&2
    return 2
  fi

  local candidate="F${fold}C_LOTO_CLSGeo12"
  local input_npz="${INPUT_ROOT}/${candidate}/features.npz"
  local output_dir="${RUN_ROOT}/${candidate}"
  local output_json="${output_dir}/${readout}_${kind}_metrics.json"
  local output_csv="${output_dir}/${readout}_${kind}_scores.csv"
  local log_path="${LOG_ROOT}/${candidate}.${readout}.${kind}.out"
  local evaluator
  local -a method_args

  if [[ "${readout}" == "prototype" ]]; then
    evaluator="${PROTOTYPE_EVAL}"
    method_args=(
      --metric cosine
      --confidence_weight 0
      --entropy_weight 0
      --margin_weight 0
    )
  elif [[ "${readout}" == "knn5" ]]; then
    evaluator="${KNN_EVAL}"
    method_args=(
      --feature_reduce mean
      --distance cosine
      --knn_k 5
      --exclude_self
      --class_conditional_threshold
    )
  else
    echo "unsupported readout: ${readout}" >&2
    return 2
  fi

  local -a command=(
    "${PYTHON}" -u "${evaluator}"
    --feature_npz "${input_npz}"
    --source_tx_ids "${source_tx}"
    --unknown_tx_ids "${unknown_tx}"
    --train_known_roles source
    --proxy_unknown_roles __disabled__
    --known_query_roles source
    --unknown_query_roles "${unknown_role}"
    --train_known_correct_only
    --source_incorrect_as_proxy
    --threshold_policy source_accept
    --source_accept_quantile 0.98
    --proxy_far_quantile 0.05
    --unknown_far_target 0.05
    --max_old_drop_pp 2.0
    --output_json "${output_json}"
    --score_table_csv "${output_csv}"
    "${method_args[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  [[ -f "${input_npz}" ]] || { echo "missing input NPZ: ${input_npz}" >&2; return 2; }
  mkdir -p "${output_dir}"
  local rc=0
  if "${command[@]}" >"${log_path}" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${candidate}" "${fold}" "${readout}" "${kind}" "${source_tx}" \
    "${unknown_tx}" "${unknown_role}" "${rc}" >>"${LOG_ROOT}/completion.tsv"
  return "${rc}"
}

status=0
for fold in 1 2 3 4 5 6; do
  for readout in prototype knn5; do
    for kind in primary secondary; do
      if ! run_score "${fold}" "${readout}" "${kind}"; then
        status=8
        break 3
      fi
    done
  done
done
exit "${status}"
