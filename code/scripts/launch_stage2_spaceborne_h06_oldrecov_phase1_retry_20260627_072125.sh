#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_h06_oldrecov_phase1_retry_20260627_072125}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
NEW_TX_IDS="${NEW_TX_IDS:-1-16,1-18}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-}"
OA_MSE_UNKNOWN_TX_IDS="${OA_MSE_UNKNOWN_TX_IDS:-10-1,10-10}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-20-1}"
SFE_MAX_SAMPLES_PER_COMBO="${SFE_MAX_SAMPLES_PER_COMBO:-0}"
SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"
SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"
SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"
SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"
SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-3}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8"
PIDS=()
NAMES=()
GPUS=()
STATUS=0

reap_finished() {
  local idx pid rc
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[${idx}]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${pid} status=0"
      else
        rc=$?
        echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${pid} status=${rc}" >&2
        STATUS=${rc}
      fi
      PIDS[${idx}]=""; NAMES[${idx}]=""; GPUS[${idx}]=""
    fi
  done
}

active_for_gpu() {
  local gpu="$1" idx pid count=0
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[${idx}]}"
    if [[ -n "${pid}" && "${GPUS[${idx}]}" == "${gpu}" ]] && kill -0 "${pid}" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

wait_for_gpu_slot() {
  local gpu="$1" active
  while true; do
    reap_finished
    active="$(active_for_gpu "${gpu}")"
    if (( active < STAGE2_MAX_ACTIVE_PER_GPU )); then
      break
    fi
    echo "[SPACEBORNE-FSDA-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 5
  done
}

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260627)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU0_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260628)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU1_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260629)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU2_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260630)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU3_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260631)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU4_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260632)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU5_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260633)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU6_A.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260634)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GROUND_PROTO_MASK_OA_MSE_H06_OLDRECOV48_GPU7_A.out"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  for idx in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[${idx}]}" ]] && wait "${PIDS[${idx}]}"; then
      echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=0"
    else
      rc=$?
      if [[ -n "${PIDS[${idx}]}" ]]; then
        echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=${rc}" >&2
        STATUS=${rc}
      fi
    fi
  done
  exit "${STATUS}"
fi
