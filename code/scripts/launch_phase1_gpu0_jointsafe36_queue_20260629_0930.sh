#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_gpu0_jointsafe36_queue_20260629_0930}"
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
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
INCLUDE_EXTERNAL_GPU_PROCS="${INCLUDE_EXTERNAL_GPU_PROCS:-1}"
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
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=36"
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

is_own_pid() {
  local probe="$1" idx
  for idx in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[${idx}]}" && "${PIDS[${idx}]}" == "${probe}" ]]; then
      return 0
    fi
  done
  return 1
}

external_active_for_gpu() {
  local gpu="$1" pid count=0
  if [[ "${INCLUDE_EXTERNAL_GPU_PROCS}" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  while read -r pid; do
    if [[ -n "${pid}" ]] && ! is_own_pid "${pid}"; then
      count=$((count + 1))
    fi
  done < <(nvidia-smi pmon -c 1 2>/dev/null | awk -v g="${gpu}" '$1 == g && $2 ~ /^[0-9]+$/ {print $2}')
  echo "${count}"
}

wait_for_gpu_slot() {
  local gpu="$1" active external total
  while true; do
    reap_finished
    active="$(active_for_gpu "${gpu}")"
    external="$(external_active_for_gpu "${gpu}")"
    total=$((active + external))
    if (( total < STAGE2_MAX_ACTIVE_PER_GPU )); then
      break
    fi
    echo "[SPACEBORNE-FSDA-WAIT] gpu=${gpu} active=${active} external=${external} total=${total} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 5
  done
}

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260700)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260701)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.93 --tau_max 0.985 --pseudo_quantile 0.88 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260702)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3" --epochs 200 --label_epochs 195 --pseudo_epochs 5 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.16 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260703)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4" --epochs 200 --label_epochs 188 --pseudo_epochs 12 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.18 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260704)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.52 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260705)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 0.85 --lambda_adv 0.3 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260706)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1.1 --lambda_adv 0.35 --lambda_group_ce 0.15 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260707)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.2 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.03 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260708)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260709)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260710)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.87 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260711)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3" --epochs 200 --label_epochs 195 --pseudo_epochs 5 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.21 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260712)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4" --epochs 200 --label_epochs 188 --pseudo_epochs 12 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.23 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260713)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.57 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260714)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.3 --lambda_group_ce 0.16 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260715)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.25 --lambda_adv 0.35 --lambda_group_ce 0.15 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260716)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.65 --lambda_sat_cons 0.005 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.03 --tau_min 0.91 --tau_max 0.97 --pseudo_quantile 0.85 --use_ema_teacher true --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260717)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260718)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260719)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.91 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260720)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3" --epochs 200 --label_epochs 195 --pseudo_epochs 5 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.21 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260721)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4" --epochs 200 --label_epochs 188 --pseudo_epochs 12 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.23 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260722)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.47 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260723)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 0.85 --lambda_adv 0.3 --lambda_group_ce 0.18 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260724)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.1 --lambda_adv 0.35 --lambda_group_ce 0.17 --lambda_fishr 0.04 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260725)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8" --epochs 200 --label_epochs 185 --pseudo_epochs 15 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.55 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.16 --lambda_fishr 0.03 --tau_min 0.9 --tau_max 0.97 --pseudo_quantile 0.84 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260726)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260727)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260728)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.93 --tau_max 0.985 --pseudo_quantile 0.88 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260729)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3" --epochs 200 --label_epochs 195 --pseudo_epochs 5 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.21 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260730)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4" --epochs 200 --label_epochs 188 --pseudo_epochs 12 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.23 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260731)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.52 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260732)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 0.85 --lambda_adv 0.3 --lambda_group_ce 0.14 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260733)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1.1 --lambda_adv 0.35 --lambda_group_ce 0.13 --lambda_fishr 0.035 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260734)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8 protocol=Safe-SSDG-CVS-R01 k=0 target_visibility=source_only_ground_training_no_target_receiver label_set_relation=Y_old_source_only"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8" --epochs 200 --label_epochs 190 --pseudo_epochs 10 --from_scratch true --best_metric joint_safe --enable_joint_safe_guard true --one_epoch_drop_guard_pp 2.0 --paic_guard_enabled true --paic_guard_sat_ce_delta 0.12 --paic_guard_grad_delta 3.0 --paic_guard_reliable_drop 0.01 --paic_guard_cooldown_epochs 1 --paic_guard_sat_scale 0.75 --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --use_tx_rx_balanced_sampler false --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --sat_cons_start_epoch 80 --lambda_sat_cls 0.6 --lambda_sat_cons 0 --lambda_u 0.25 --lambda_ent 0.01 --lambda_domain 1 --lambda_adv 0.35 --lambda_group_ce 0.12 --lambda_fishr 0.025 --tau_min 0.92 --tau_max 0.985 --pseudo_quantile 0.86 --use_ema_teacher false --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 260735)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8"
  "${CMD[@]}" > "${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8.out" 2>&1 &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8")
  GPUS+=("${GPU}")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8.out"
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
