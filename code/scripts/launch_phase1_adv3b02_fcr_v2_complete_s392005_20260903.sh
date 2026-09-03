#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
SEED="${SEED:-392005}"
SOURCE_DAYS='1,2,3'
SOURCE_RXS='1,3,4,6,8'
TARGET_DAYS='0,1,2,3'
TARGET_RXS='0,2,5,7,9,10,11'
C0_CHECKPOINT="${C0_CHECKPOINT:-/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth}"
TRAIN_ROWS=(C1 C2 C3 S0 S1 S2 S3 S4 M1 M2 M3 M4 M5 M6)
WAVE1_ROWS=(C1 C2 C3 S0 S1 S2 S3 S4)
WAVE2_ROWS=(M1 M2 M3 M4 M5 M6)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

[[ "${SEED}" == "392005" ]] || { echo "[FCRV2-ERROR] seed must be 392005" >&2; exit 2; }
if [[ "${DRY_RUN}" != "1" && -e "${OUTPUT_ROOT}" ]]; then
  echo "[FCRV2-ERROR] refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 3
fi

TARGET_INPUT_ROOT="${OUTPUT_ROOT}/target_inputs"
TARGET_TRUTH_ROOT="${OUTPUT_ROOT}/target_truth"
C0_ROOT="${OUTPUT_ROOT}/C0"
C0_PREDICTIONS="${C0_ROOT}/target_prediction/predictions.json"
C0_SCORE_JSON="${C0_ROOT}/target_prediction/score.json"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_ROWS=()
declare -a ACTIVE_FINALS=()
declare -a ACTIVE_DIAGNOSTICS=()

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --seed "${SEED}"
  --device cuda:0
  --epochs 200
  --wisig_equalized 1
  --wisig_train_days "${SOURCE_DAYS}"
  --wisig_train_rxs "${SOURCE_RXS}"
  --wisig_test_days "${TARGET_DAYS}"
  --wisig_test_rxs "${TARGET_RXS}"
  --wisig_allow_day_overlap_if_receiver_disjoint
  --use_meta_ssl_cvs
  --ssl_labeled_ratio 0.07
  --ssl_unlabeled_ratio 0.63
  --ssl_val_ratio 0.30
  --test_eval_policy never
  --init_checkpoint "${C0_CHECKPOINT}"
  --init_checkpoint_expected_seed 392005
  --init_checkpoint_expected_epoch 200
  --init_checkpoint_expected_candidate ADV3B02_CORE90_SOFT_E200
  --init_checkpoint_require_mature_identity_complete
  --defer_target_evaluation
)

probe_training_contract() {
  local row cmd
  for row in C1 M6; do
    cmd="$(build_train_command "${row}")"
    bash -lc "${cmd} --fcr_config_dry_run" > /dev/null 2>&1
  done
}

row_gpu() {
  case "$1" in
    C1) echo 0 ;;
    C2) echo 1 ;;
    C3) echo 2 ;;
    S0) echo 3 ;;
    S1) echo 4 ;;
    S2) echo 5 ;;
    S3) echo 6 ;;
    S4) echo 7 ;;
    M1) echo 0 ;;
    M2) echo 1 ;;
    M3) echo 2 ;;
    M4) echo 3 ;;
    M5) echo 4 ;;
    M6) echo 5 ;;
    C0) echo 0 ;;
    *) echo "[FCRV2-ERROR] unknown row: $1" >&2; exit 8 ;;
  esac
}

row_wave() {
  case "$1" in
    C1|C2|C3|S0|S1|S2|S3|S4) echo 1 ;;
    M1|M2|M3|M4|M5|M6) echo 2 ;;
    C0) echo 0 ;;
    *) echo "[FCRV2-ERROR] unknown row: $1" >&2; exit 8 ;;
  esac
}

row_root() {
  if [[ "$1" == "C0" ]]; then
    echo "${C0_ROOT}"
  else
    echo "${OUTPUT_ROOT}/$1"
  fi
}

emit_row_contract() {
  local row="$1"
  local gpu wave rr final_checkpoint diagnostics_path
  gpu="$(row_gpu "${row}")"
  wave="$(row_wave "${row}")"
  rr="$(row_root "${row}")"
  final_checkpoint="${rr}/final.pth"
  diagnostics_path="${rr}/fcr_diagnostics.json"
  printf '[FCRV2-ROW] row=%s wave=%s gpu=%s epochs=200 checkpoint_selection=final_only init_checkpoint=%s output=%s final_checkpoint=%s diagnostics=%s\n' \
    "${row}" "${wave}" "${gpu}" "${C0_CHECKPOINT}" "${rr}" "${final_checkpoint}" "${diagnostics_path}"
}

build_train_command() {
  local row="$1"
  local rr
  rr="$(row_root "${row}")"
  local -a command=(
    env "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=$(row_gpu "${row}")"
    "${PYTHON}" -u "${CODE_ROOT}/code/train.py"
    "${COMMON_ARGS[@]}"
    --phase1_method adv3b02_fcr
    --use_fcr
    --fcr_ablation_row "${row}"
    --run_name "${RUN_ID}_${row}"
    --final_save_path "${rr}/final.pth"
    --log_dir "${rr}/logs"
    --fcr_diagnostics_path "${rr}/fcr_diagnostics.json"
    --fcr_predictions_path "${rr}/fcr_predictions.json"
  )
  printf '%q ' "${command[@]}"
  printf '\n'
}

launch_row_async() {
  local row="$1"
  local rr final_checkpoint diagnostics_path
  rr="$(row_root "${row}")"
  final_checkpoint="${rr}/final.pth"
  diagnostics_path="${rr}/fcr_diagnostics.json"
  mkdir -p "${rr}" "${rr}/logs"
  local cmd
  cmd="$(build_train_command "${row}")"
  bash -lc "${cmd}" > "${rr}/train.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_ROWS+=("${row}")
  ACTIVE_FINALS+=("${final_checkpoint}")
  ACTIVE_DIAGNOSTICS+=("${diagnostics_path}")
}

wait_training_rows() {
  local label="$1"
  local status=0
  for idx in "${!ACTIVE_PIDS[@]}"; do
    local pid row final_checkpoint diagnostics_path
    pid="${ACTIVE_PIDS[$idx]}"
    row="${ACTIVE_ROWS[$idx]}"
    final_checkpoint="${ACTIVE_FINALS[$idx]}"
    diagnostics_path="${ACTIVE_DIAGNOSTICS[$idx]}"
    if ! wait "${pid}"; then
      echo "[FCRV2-ERROR] training failed label=${label} row=${row}" >&2
      status=1
      continue
    fi
    [[ -s "${final_checkpoint}" ]] || { echo "[FCRV2-ERROR] missing final checkpoint row=${row} path=${final_checkpoint}" >&2; status=1; }
    [[ -s "${diagnostics_path}" ]] || { echo "[FCRV2-ERROR] missing diagnostics row=${row} path=${diagnostics_path}" >&2; status=1; }
  done
  ACTIVE_PIDS=()
  ACTIVE_ROWS=()
  ACTIVE_FINALS=()
  ACTIVE_DIAGNOSTICS=()
  [[ "${status}" == "0" ]] || exit 6
}

emit_prepare_contract() {
  printf '[FCRV2-PREPARE] rows_ready=14 input_package=%s truth_sidecar=%s\n' \
    "${TARGET_INPUT_ROOT}" "${TARGET_TRUTH_ROOT}/truth_sidecar.json"
}

emit_predict_contract() {
  local row="$1"
  local checkpoint="$2"
  local rr
  rr="$(row_root "${row}")"
  printf '[FCRV2-PREDICT] row=%s checkpoint=%s output=%s\n' \
    "${row}" "${checkpoint}" "${rr}/target_prediction/predictions.json"
}

emit_score_contract() {
  local row="$1"
  local rr
  rr="$(row_root "${row}")"
  printf '[FCRV2-SCORE] row=%s predictions=%s output=%s\n' \
    "${row}" "${rr}/target_prediction/predictions.json" "${rr}/target_prediction/score.json"
}

prepare_truth_last() {
  emit_prepare_contract
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${LOG_ROOT}"
  env "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -u "${CODE_ROOT}/code/scripts/predict_phase1_truth_last.py" \
    --mode prepare \
    --checkpoint "${C0_CHECKPOINT}" \
    --wisig-pkl "${WISIG_PKL}" \
    --output-root "${TARGET_TRUTH_ROOT}" \
    --input-package "${TARGET_INPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --row-id SHARED \
    --device cpu \
    > "${LOG_ROOT}/target_prepare.out" 2>&1
}

predict_row() {
  local row="$1"
  local checkpoint="$2"
  local rr gpu
  rr="$(row_root "${row}")"
  gpu="$(row_gpu "${row}")"
  emit_predict_contract "${row}" "${checkpoint}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  [[ -s "${checkpoint}" ]] || { echo "[FCRV2-ERROR] checkpoint missing for predict row=${row} path=${checkpoint}" >&2; exit 7; }
  mkdir -p "${rr}"
  env "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}" \
    "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON}" -u "${CODE_ROOT}/code/scripts/predict_phase1_truth_last.py" \
    --mode predict \
    --checkpoint "${checkpoint}" \
    --output-root "${rr}/target_prediction" \
    --input-package "${TARGET_INPUT_ROOT}" \
    --run-id "${RUN_ID}_${row}" \
    --row-id "${row}" \
    --device cuda:0 \
    > "${rr}/predict.log" 2>&1
}

score_row() {
  local row="$1"
  local rr
  rr="$(row_root "${row}")"
  emit_score_contract "${row}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  env "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -u "${CODE_ROOT}/code/scripts/score_phase1_truth_last.py" \
    --predictions "${rr}/target_prediction/predictions.json" \
    --truth "${TARGET_TRUTH_ROOT}/truth_sidecar.json" \
    --output "${rr}/target_prediction/score.json" \
    > "${rr}/score.log" 2>&1
}

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[FCRV2-C0] row=C0 checkpoint=%s train=0 score_json=%s\n' \
    "${C0_CHECKPOINT}" "${C0_SCORE_JSON}"
  for row in "${TRAIN_ROWS[@]}"; do
    emit_row_contract "${row}"
  done
  emit_prepare_contract
  emit_predict_contract C0 "${C0_CHECKPOINT}"
  emit_score_contract C0
  for row in "${TRAIN_ROWS[@]}"; do
    emit_predict_contract "${row}" "$(row_root "${row}")/final.pth"
    emit_score_contract "${row}"
  done
  exit 0
fi

if ! probe_training_contract; then
  echo "[FCRV2-ERROR] C1..M6 training route is not integrated in train.py yet; dry-run contract is available but real launch is blocked pending Task6 landing." >&2
  exit 9
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
for row in "${WAVE1_ROWS[@]}"; do
  launch_row_async "${row}"
done
wait_training_rows "wave1"

for row in "${WAVE2_ROWS[@]}"; do
  launch_row_async "${row}"
done
wait_training_rows "wave2"

prepare_truth_last
predict_row C0 "${C0_CHECKPOINT}"
score_row C0
for row in "${TRAIN_ROWS[@]}"; do
  predict_row "${row}" "$(row_root "${row}")/final.pth"
  score_row "${row}"
done

printf '[FCRV2-COMPLETE] run_id=%s rows_ready=14 scored=15\n' "${RUN_ID}"
