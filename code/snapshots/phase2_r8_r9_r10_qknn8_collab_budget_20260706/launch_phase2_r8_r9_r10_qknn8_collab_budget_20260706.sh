#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_r8_r9_r10_qknn8_collab_20260706}"
RUN_ID="${RUN_ID:-phase2_r8_r9_r10_qknn8_collab_budget_20260706}"
FEATURE_RUNS_ROOT="${FEATURE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"

K_SHOT="${K_SHOT:-8}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-20}"
QKNN_K="${QKNN_K:-8}"
MAX_EVENT_BYTES="${MAX_EVENT_BYTES:-1152}"
MAX_EVENT_LATENCY_MS="${MAX_EVENT_LATENCY_MS:-20}"
COLLAB_GROUP_POLICY="${COLLAB_GROUP_POLICY:-available_up_to_k}"
PARTIAL_COLLAB_MIN_RECEIVERS="${PARTIAL_COLLAB_MIN_RECEIVERS:-1}"
EVENT_ALIGNMENT_POLICY="${EVENT_ALIGNMENT_POLICY:-receiver_domain_ranked}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

case_config() {
  case "$1" in
    R8_RADIUS) echo "R8_RADIUS|0|706801" ;;
    R8_SHELL) echo "R8_SHELL|1|706811" ;;
    R9_ANCHOR) echo "R9_ANCHOR|2|706901" ;;
    R9_GENTLE) echo "R9_GENTLE|3|706911" ;;
    R10_BOUNDARY) echo "R10_BOUNDARY|4|7061001" ;;
    R10_GENTLE) echo "R10_GENTLE|5|7061011" ;;
    *) echo "[ERROR] unknown case: $1" >&2; exit 2 ;;
  esac
}

cases=()
case "${ONLY}" in
  all) cases=(R8_RADIUS R8_SHELL R9_ANCHOR R9_GENTLE R10_BOUNDARY R10_GENTLE) ;;
  R8_RADIUS|R8_SHELL|R9_ANCHOR|R9_GENTLE|R10_BOUNDARY|R10_GENTLE) cases=("${ONLY}") ;;
  *) echo "[ERROR] --only must be all or one known case" >&2; exit 2 ;;
esac

echo "[R8R9R10-QKNN8-COLLAB-BUDGET] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN} only=${ONLY}"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] cases=${cases[*]}"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] qknn_k=${QKNN_K} k_shot=${K_SHOT} query_per_class=${QUERY_PER_CLASS}"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] collab_counts=all collab_group_policy=${COLLAB_GROUP_POLICY} partial_collab_min_receivers=${PARTIAL_COLLAB_MIN_RECEIVERS}"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] protocol=Stage2-C unknown_query_eval_only=true proxy_unknown_real_tx_calibration=0 stage2_success_claim=0 deployment_success_claim=0"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] target_channel_view=leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
echo "[R8R9R10-QKNN8-COLLAB-BUDGET] note=uses_existing_feature_npz_from_strict_exact_attempt; records actual_receiver_count_histogram"

launch_case() {
  local short_name="$1"
  local config case_id gpu seed feature_npz out_dir output_json evidence_csv log_path
  config="$(case_config "${short_name}")"
  IFS='|' read -r case_id gpu seed <<< "${config}"
  feature_npz="${FEATURE_RUNS_ROOT}/${case_id}/features_stage2c_leo_multirx.npz"
  out_dir="${RUNS_ROOT}/${case_id}"
  output_json="${out_dir}/qknn8_collab_budget.json"
  evidence_csv="${out_dir}/qknn8_collab_budget_evidence.csv"
  log_path="${LOG_ROOT}/${case_id}.out"

  local eval_cmd=(
    env PYTHONPATH="${ROOT}/code:${ROOT}/code/scripts:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u "${ROOT}/code/scripts/phase2_collaborative_open_set_qknn_eval.py"
    --feature_npz "${feature_npz}"
    --output_json "${output_json}"
    --output_evidence_csv "${evidence_csv}"
    --collab_counts all
    --collab_group_policy "${COLLAB_GROUP_POLICY}"
    --partial_collab_min_receivers "${PARTIAL_COLLAB_MIN_RECEIVERS}"
    --k_shot "${K_SHOT}"
    --query_per_class "${QUERY_PER_CLASS}"
    --qknn_k "${QKNN_K}"
    --support_selection_policy stable_first
    --event_alignment_policy "${EVENT_ALIGNMENT_POLICY}"
    --latency_budget_ms "${MAX_EVENT_LATENCY_MS}"
    --max_event_bytes "${MAX_EVENT_BYTES}"
    --max_event_latency_ms "${MAX_EVENT_LATENCY_MS}"
    --evidence_packet_bytes 40
    --seed "${seed}"
  )

  echo "[R8R9R10-QKNN8-COLLAB-BUDGET-CASE] case=${case_id} gpu=${gpu} seed=${seed} feature_npz=${feature_npz} log=${log_path} out_dir=${out_dir}"
  printf "[R8R9R10-QKNN8-COLLAB-BUDGET-EVAL-CMD] "
  printf "%q " "${eval_cmd[@]}"
  printf "\n"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${feature_npz}" ]]; then
    echo "[ERROR] feature npz not found: ${feature_npz}" >&2
    return 3
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}"
  (
    set -euo pipefail
    "${eval_cmd[@]}"
    echo "[R8R9R10-QKNN8-COLLAB-BUDGET-DONE] case=${case_id} output_json=${output_json}"
  ) > "${log_path}" 2>&1 &
  echo "[R8R9R10-QKNN8-COLLAB-BUDGET-LAUNCHED] case=${case_id} pid=$! gpu=${gpu} log=${log_path} out_dir=${out_dir}"
}

for case_name in "${cases[@]}"; do
  launch_case "${case_name}"
done
