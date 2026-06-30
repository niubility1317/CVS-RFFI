#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_h06_phase1_floorrepair_20260628_221940}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
export ROOT PYTHON RUN_ID RUNS_ROOT LOG_ROOT WISIG_PKL
DRY_RUN="${DRY_RUN:-0}"
PHASE1_MAX_ACTIVE_PER_GPU="${PHASE1_MAX_ACTIVE_PER_GPU:-1}"
for arg in "$@"; do case "${arg}" in --dry-run) DRY_RUN=1 ;; *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;; esac; done
if [[ "${DRY_RUN}" != "1" ]]; then mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"; fi
echo "[PHASE1-FLOOR-REPAIR] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8"
PIDS=()
NAMES=()
GPUS=()
STATUS=0
active_for_gpu() { local gpu="$1" idx pid count=0; for idx in "${!PIDS[@]}"; do pid="${PIDS[$idx]}"; if [[ -n "${pid}" && "${GPUS[$idx]}" == "${gpu}" ]] && kill -0 "${pid}" 2>/dev/null; then count=$((count+1)); fi; done; echo "${count}"; }
reap_finished() { local idx pid rc; for idx in "${!PIDS[@]}"; do pid="${PIDS[$idx]}"; if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then if wait "${pid}"; then echo "[PHASE1-FLOOR-REPAIR-COMPLETE] id=${NAMES[$idx]} pid=${pid} status=0"; else rc=$?; echo "[PHASE1-FLOOR-REPAIR-FAILED] id=${NAMES[$idx]} pid=${pid} status=${rc}" >&2; STATUS=${rc}; fi; PIDS[$idx]=""; NAMES[$idx]=""; GPUS[$idx]=""; fi; done; }
wait_for_gpu_slot() { local gpu="$1" active; while true; do reap_finished; active="$(active_for_gpu "${gpu}")"; if (( active < PHASE1_MAX_ACTIVE_PER_GPU )); then break; fi; echo "[PHASE1-FLOOR-REPAIR-WAIT] gpu=${gpu} active=${active} max=${PHASE1_MAX_ACTIVE_PER_GPU}"; sleep 5; done; }
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="0"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806290 --label_epochs 188 --pseudo_epochs 12 --tau_min 0.93 --tau_max 0.995 --pseudo_quantile 0.89 --lambda_u 0.16 --mixstyle_p 0.14 --mixstyle_strength 0.30 --mixstyle_late_start 115 --mixstyle_late_min_strength 0.12 --lambda_group_ce 0.24 --group_ce_min_domains 6 --lambda_fishr 0.055 --lambda_sat_cls 0.36 --lambda_sat_cons 0.00 --sat_cons_start_epoch 145 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_MIXSTYLE_STRICTUDU_FLOOR_TUNE_GPU0_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="1"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806291 --label_epochs 186 --pseudo_epochs 14 --tau_min 0.93 --tau_max 0.995 --pseudo_quantile 0.89 --lambda_u 0.18 --lambda_group_ce 0.24 --group_ce_min_domains 6 --lambda_fishr 0.055 --lambda_domain 1.28 --lambda_adv 0.22 --lambda_sat_cls 0.34 --lambda_sat_cons 0.00 --sat_cons_start_epoch 150 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_SATLATE_STRICTUDU_RECEIVER_TUNE_GPU1_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="2"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806292 --label_epochs 194 --pseudo_epochs 6 --tau_min 0.95 --tau_max 0.995 --pseudo_quantile 0.92 --lambda_u 0.08 --lambda_cons 0.02 --lambda_group_ce 0.24 --group_ce_min_domains 6 --lambda_fishr 0.06 --lambda_sat_cls 0.30 --lambda_sat_cons 0.00 --sat_cons_start_epoch 155 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOLOW_GROUPFLOOR_CONTROL_GPU2_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="3"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806293 --label_epochs 190 --pseudo_epochs 10 --tau_min 0.94 --tau_max 0.995 --pseudo_quantile 0.90 --lambda_u 0.14 --lambda_domain 1.35 --lambda_adv 0.22 --lambda_orth 0.12 --lambda_group_ce 0.18 --group_ce_min_domains 6 --lambda_fishr 0.075 --lambda_sat_cls 0.34 --lambda_sat_cons 0.00 --sat_cons_start_epoch 145 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_FISHR_ORTH_RECEIVER_LIFT_GPU3_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="4"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806294 --label_epochs 188 --pseudo_epochs 12 --tau_min 0.93 --tau_max 0.995 --pseudo_quantile 0.88 --lambda_u 0.16 --lambda_group_ce 0.26 --group_ce_min_domains 6 --lambda_fishr 0.055 --lambda_sat_cls 0.26 --lambda_sat_cons 0.00 --sat_cons_start_epoch 155 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PAIC_LOWCLS_RECEIVER_BALANCE_GPU4_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="5"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806295 --label_epochs 190 --pseudo_epochs 10 --tau_min 0.95 --tau_max 0.998 --pseudo_quantile 0.92 --lambda_u 0.10 --lambda_cons 0.03 --lambda_group_ce 0.20 --group_ce_min_domains 6 --lambda_fishr 0.055 --lambda_sat_cls 0.42 --lambda_sat_cons 0.00 --sat_cons_start_epoch 135 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_STRICTUDU_SATRETENTION_GUARD_GPU5_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="6"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806296 --label_epochs 200 --pseudo_epochs 0 --use_unlabeled false --lambda_u 0.0 --lambda_ent 0.0 --lambda_group_ce 0.24 --group_ce_min_domains 6 --lambda_fishr 0.065 --lambda_sat_cls 0.28 --lambda_sat_cons 0.00 --sat_cons_start_epoch 160 --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_PSEUDOOOFF_FISHR_FLOOR_GUARD_GPU6_A.out"
fi
echo "[PHASE1-FLOOR-REPAIR-CANDIDATE] id=PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A category=oldrecov_ridge_head route_family=CEN51_REFRESH_CONTROL"
GPU="7"
export GPU
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 2806297 --label_epochs 200 --pseudo_epochs 0 --use_unlabeled false --lambda_u 0.0 --lambda_ent 0.0 --lambda_group_ce 0.14 --lambda_fishr 0.04 --lambda_sat_cls 0.0 --lambda_sat_cons 0.0 --no_use_sat_consistency --no_use_concat_sat_channel_aug --no_concat_sat_ce_only --best_metric clean_val_tx')
printf "[PHASE1-FLOOR-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-FLOOR-REPAIR-LAUNCHED] id=PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_FLOOR_REPAIR_CEN51_REFRESH_CONTROL_SEED3_GPU7_A.out"
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  for idx in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[$idx]}" ]] && wait "${PIDS[$idx]}"; then
      echo "[PHASE1-FLOOR-REPAIR-COMPLETE] id=${NAMES[$idx]} pid=${PIDS[$idx]} status=0"
    else
      rc=$?
      if [[ -n "${PIDS[$idx]}" ]]; then echo "[PHASE1-FLOOR-REPAIR-FAILED] id=${NAMES[$idx]} pid=${PIDS[$idx]} status=${rc}" >&2; STATUS=${rc}; fi
    fi
  done
  exit "${STATUS}"
fi
