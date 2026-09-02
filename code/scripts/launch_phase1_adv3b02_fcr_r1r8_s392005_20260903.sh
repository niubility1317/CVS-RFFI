#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
SEED="${SEED:-392005}"
SOURCE_DAYS='1,2,3'
SOURCE_RXS='1,3,4,6,8'
TARGET_DAYS='0,1,2,3'
TARGET_RXS='0,2,5,7,9,10,11'
BASE_CANDIDATE='ADV3B02_CORE90_SOFT_E200'
ROWS=(R1 R2 R3 R4 R5 R6 R7 R8)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

[[ "${SEED}" == "392005" ]] || { echo "[RELEASE-ERROR] seed must be 392005" >&2; exit 2; }
if [[ "${DRY_RUN}" != "1" && -e "${OUTPUT_ROOT}" ]]; then
  echo "[RELEASE-ERROR] refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 3
fi

COMMON_ARGS="--seed 392005 --wisig_equalized 1 --wisig_train_days ${SOURCE_DAYS} --wisig_train_rxs ${SOURCE_RXS} --wisig_test_days ${TARGET_DAYS} --wisig_test_rxs ${TARGET_RXS} --wisig_allow_day_overlap_if_receiver_disjoint true --labeled_ratio 0.07 --unlabeled_ratio 0.63 --source_val_ratio 0.30 --phase1_source_role_protocol legacy_l_u_v --checkpoint_selection final_only --best_metric clean_val_tx --enable_joint_safe_guard false --test_eval_policy never --phase1_external_final_eval true"

printf '[RELEASE] stage=ADV3B02_CORE90_SOFT_E200 seed=%s source_days=%s source_rxs=%s target_days=%s target_rxs=%s\n' \
  "${SEED}" "${SOURCE_DAYS}" "${SOURCE_RXS}" "${TARGET_DAYS}" "${TARGET_RXS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  ROOT="${ROOT}" PYTHON="${PYTHON}" WISIG_PKL="${WISIG_PKL}" RUN_ID="${RUN_ID}_ADV3B02" \
    RUNS_ROOT="${OUTPUT_ROOT}/ADV3B02" LOG_ROOT="${LOG_ROOT}/ADV3B02" \
    EXTRA_ARGS="${COMMON_ARGS}" DRY_RUN=1 VALIDATE_TRAIN_DRY_RUN=1 \
    "${ROOT}/code/scripts/launch_phase1_adv3_mechanism32_queue_20260701.sh" \
  --only="${BASE_CANDIDATE}" --dry-run
  printf '[RELEASE] stage=R1-R8 waits_for=%s\n' "${OUTPUT_ROOT}/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth"
  for row in "${ROWS[@]}"; do
    printf '[RELEASE-ROW] row=%s gpu=%s checkpoint=final_ssdg.pth test_eval_policy=never final_checkpoint=final.pth\n' \
      "${row}" "$(( ${row#R} - 1 ))"
  done
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
ROOT="${ROOT}" PYTHON="${PYTHON}" WISIG_PKL="${WISIG_PKL}" RUN_ID="${RUN_ID}_ADV3B02" \
  RUNS_ROOT="${OUTPUT_ROOT}/ADV3B02" LOG_ROOT="${LOG_ROOT}/ADV3B02" \
  EXTRA_ARGS="${COMMON_ARGS}" STAGE2_MAX_ACTIVE_PER_GPU=999 \
  "${ROOT}/code/scripts/launch_phase1_adv3_mechanism32_queue_20260701.sh" \
  --only="${BASE_CANDIDATE}"

BASE_CHECKPOINT="${OUTPUT_ROOT}/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth"
[[ -s "${BASE_CHECKPOINT}" ]] || { echo "[RELEASE-ERROR] ADV3B02 final checkpoint missing" >&2; exit 5; }

FCR_ARGS="--wisig_equalized 1 --wisig_train_days ${SOURCE_DAYS} --wisig_train_rxs ${SOURCE_RXS} --wisig_test_days ${TARGET_DAYS} --wisig_test_rxs ${TARGET_RXS} --wisig_allow_day_overlap_if_receiver_disjoint --ssl_labeled_ratio 0.07 --ssl_unlabeled_ratio 0.63 --ssl_val_ratio 0.30 --test_eval_policy never --init_checkpoint_expected_seed 392005 --init_checkpoint_expected_epoch 200 --init_checkpoint_expected_candidate ADV3B02_CORE90_SOFT_E200"
pids=()
for row in "${ROWS[@]}"; do
  gpu=$(( ${row#R} - 1 ))
  row_output="${OUTPUT_ROOT}/FCR_${row}"
  ROOT="${ROOT}" CODE_ROOT="${ROOT}" PYTHON="${PYTHON}" WISIG_PKL="${WISIG_PKL}" \
    INIT_CHECKPOINT="${BASE_CHECKPOINT}" RUN_ID="${RUN_ID}_${row}" OUTPUT_ROOT="${row_output}" \
    GPU="${gpu}" SEED="${SEED}" EXPECTED_SEED="${SEED}" EXTERNAL_TARGET_EVAL=1 \
    FCR_EXTRA_ARGS="${FCR_ARGS}" \
    "${ROOT}/code/scripts/launch_phase1_adv3b02_fcr_20260901.sh" --row="${row}" \
    > "${LOG_ROOT}/${row}_launcher.out" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == "0" ]] || { echo "[RELEASE-ERROR] one or more FCR rows failed" >&2; exit 6; }

env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/predict_phase1_truth_last.py" \
  --mode prepare --checkpoint "${BASE_CHECKPOINT}" --wisig-pkl "${WISIG_PKL}" \
  --output-root "${OUTPUT_ROOT}/target_truth" --input-package "${OUTPUT_ROOT}/target_inputs" \
  --run-id "${RUN_ID}" --row-id SHARED --device cpu \
  > "${LOG_ROOT}/target_prepare.out" 2>&1

eval_pids=()
for row in "${ROWS[@]}"; do
  gpu=$(( ${row#R} - 1 ))
  row_root="${OUTPUT_ROOT}/FCR_${row}/${row}"
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/predict_phase1_truth_last.py" \
    --mode predict \
    --checkpoint "${row_root}/final.pth" \
    --output-root "${row_root}/target_prediction" --input-package "${OUTPUT_ROOT}/target_inputs" \
    --run-id "${RUN_ID}_${row}" --row-id "${row}" --device cuda:0 \
    > "${row_root}/predict.log" 2>&1 &
  eval_pids+=("$!")
done
status=0
for pid in "${eval_pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == "0" ]] || { echo "[RELEASE-ERROR] one or more prediction jobs failed" >&2; exit 7; }

for row in "${ROWS[@]}"; do
  row_root="${OUTPUT_ROOT}/FCR_${row}/${row}"
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/score_phase1_truth_last.py" \
    --predictions "${row_root}/target_prediction/predictions.json" \
    --truth "${OUTPUT_ROOT}/target_truth/truth_sidecar.json" \
    --output "${row_root}/target_prediction/score.json" \
    > "${row_root}/score.log" 2>&1
done

BASE_ROW_ROOT="${OUTPUT_ROOT}/ADV3B02/ADV3B02_CORE90_SOFT_E200"
env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" -u "${ROOT}/code/scripts/predict_phase1_truth_last.py" \
  --mode predict \
  --checkpoint "${BASE_CHECKPOINT}" \
  --output-root "${BASE_ROW_ROOT}/target_prediction" --input-package "${OUTPUT_ROOT}/target_inputs" \
  --run-id "${RUN_ID}_ADV3B02" \
  --row-id ADV3B02 --device cuda:0 \
  > "${OUTPUT_ROOT}/ADV3B02/ADV3B02_CORE90_SOFT_E200/predict.log" 2>&1
env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/score_phase1_truth_last.py" \
  --predictions "${BASE_ROW_ROOT}/target_prediction/predictions.json" \
  --truth "${OUTPUT_ROOT}/target_truth/truth_sidecar.json" \
  --output "${BASE_ROW_ROOT}/target_prediction/score.json" \
  > "${OUTPUT_ROOT}/ADV3B02/ADV3B02_CORE90_SOFT_E200/score.log" 2>&1

printf '[RELEASE] artifacts_complete=1 rows=ADV3B02,R1,R2,R3,R4,R5,R6,R7,R8\n'
