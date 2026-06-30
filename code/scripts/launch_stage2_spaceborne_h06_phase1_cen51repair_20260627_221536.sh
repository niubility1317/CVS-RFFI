#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_h06_phase1_cen51repair_20260627_221536}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
PHASE1_MAX_ACTIVE_PER_GPU="${PHASE1_MAX_ACTIVE_PER_GPU:-1}"
for arg in "$@"; do case "${arg}" in --dry-run) DRY_RUN=1 ;; *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;; esac; done
if [[ "${DRY_RUN}" != "1" ]]; then mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"; fi
echo "[PHASE1-CEN51-REPAIR] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8"
PIDS=()
NAMES=()
GPUS=()
STATUS=0
active_for_gpu() { local gpu="$1" idx pid count=0; for idx in "${!PIDS[@]}"; do pid="${PIDS[$idx]}"; if [[ -n "${pid}" && "${GPUS[$idx]}" == "${gpu}" ]] && kill -0 "${pid}" 2>/dev/null; then count=$((count+1)); fi; done; echo "${count}"; }
reap_finished() { local idx pid rc; for idx in "${!PIDS[@]}"; do pid="${PIDS[$idx]}"; if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then if wait "${pid}"; then echo "[PHASE1-CEN51-REPAIR-COMPLETE] id=${NAMES[$idx]} pid=${pid} status=0"; else rc=$?; echo "[PHASE1-CEN51-REPAIR-FAILED] id=${NAMES[$idx]} pid=${pid} status=${rc}" >&2; STATUS=${rc}; fi; PIDS[$idx]=""; NAMES[$idx]=""; GPUS[$idx]=""; fi; done; }
wait_for_gpu_slot() { local gpu="$1" active; while true; do reap_finished; active="$(active_for_gpu "${gpu}")"; if (( active < PHASE1_MAX_ACTIVE_PER_GPU )); then break; fi; echo "[PHASE1-CEN51-REPAIR-WAIT] gpu=${gpu} active=${active} max=${PHASE1_MAX_ACTIVE_PER_GPU}"; sleep 5; done; }
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="0"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270627 --label_epochs 185 --pseudo_epochs 15 --tau_min 0.90 --tau_max 0.985 --pseudo_quantile 0.82 --lambda_u 0.35 --lambda_group_ce 0.18 --group_ce_top_frac 0.50 --lambda_fishr 0.04 --lambda_domain 1.15 --lambda_adv 0.35 --lambda_sat_cls 0.70 --lambda_sat_cons 0.01 --sat_cons_start_epoch 80 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="1"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270628 --label_epochs 190 --pseudo_epochs 10 --tau_min 0.93 --tau_max 0.995 --pseudo_quantile 0.88 --lambda_u 0.20 --lambda_cons 0.06 --lambda_group_ce 0.14 --lambda_fishr 0.035 --lambda_sat_cls 0.60 --lambda_sat_cons 0.00 --sat_cons_start_epoch 100 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_STRICTUDU_STRICTPSEUDO_GPU1_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="2"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270629 --label_epochs 200 --pseudo_epochs 0 --use_unlabeled false --lambda_u 0.0 --lambda_ent 0.0 --lambda_group_ce 0.16 --lambda_fishr 0.04 --lambda_sat_cls 0.55 --lambda_sat_cons 0.00 --sat_cons_start_epoch 120 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_PSEUDOOFF_PAIC_BALANCE_GPU2_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="3"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270630 --label_epochs 180 --pseudo_epochs 20 --tau_min 0.90 --pseudo_quantile 0.84 --lambda_u 0.25 --lambda_group_ce 0.20 --group_ce_min_domains 5 --lambda_fishr 0.05 --lambda_sat_cls 0.50 --lambda_sat_cons 0.00 --sat_cons_start_epoch 120 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_SATLATE_RECEIVER_FLOOR_GPU3_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="4"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270631 --label_epochs 185 --pseudo_epochs 15 --tau_min 0.91 --pseudo_quantile 0.85 --lambda_u 0.25 --mixstyle_p 0.24 --mixstyle_strength 0.55 --mixstyle_late_start 80 --mixstyle_late_min_strength 0.25 --lambda_group_ce 0.16 --lambda_fishr 0.035 --lambda_sat_cls 0.65 --lambda_sat_cons 0.01 --sat_cons_start_epoch 90 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_MIXSTYLE_FLOOR_GUARD_GPU4_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A category=oldrecov_ridge_head route_family=SAFE_SSDG_CVS_R01"
GPU="5"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270632 --label_epochs 190 --pseudo_epochs 10 --tau_min 0.92 --pseudo_quantile 0.86 --lambda_u 0.20 --lambda_domain 1.30 --lambda_adv 0.30 --lambda_orth 0.08 --lambda_group_ce 0.12 --lambda_fishr 0.06 --lambda_sat_cls 0.55 --lambda_sat_cons 0.00 --sat_cons_start_epoch 100 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_FISHR_DOMAIN_GUARD_GPU5_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A category=oldrecov_proto_bridge route_family=SAFE_SSDG_CVS_R01"
GPU="6"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270633 --label_epochs 180 --pseudo_epochs 20 --tau_min 0.90 --pseudo_quantile 0.84 --lambda_u 0.25 --lambda_group_ce 0.14 --lambda_fishr 0.035 --lambda_sat_cls 0.85 --lambda_sat_cons 0.00 --sat_cons_start_epoch 85 --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_SATFLOOR_NO_CONS_GPU6_A.out"
fi
echo "[PHASE1-CEN51-REPAIR-CANDIDATE] id=PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A category=oldrecov_ridge_head route_family=CEN51_REFRESH_CONTROL"
GPU="7"
CMD=(bash -lc 'env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py" --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 --output_dir "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A" --epochs 200 --from_scratch true --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_train_scenario leo_clear_weak --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_eval_max_batches -1 --device cuda:0 --seed 270634 --label_epochs 200 --pseudo_epochs 0 --use_unlabeled false --lambda_u 0.0 --lambda_ent 0.0 --lambda_group_ce 0.12 --lambda_fishr 0.035 --lambda_sat_cls 0.0 --lambda_sat_cons 0.0 --no_use_sat_consistency --no_use_concat_sat_channel_aug --no_concat_sat_ce_only --best_metric clean_val_tx')
printf "[PHASE1-CEN51-REPAIR-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  wait_for_gpu_slot "${GPU}"
  mkdir -p "${RUNS_ROOT}/PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A"
  ("${CMD[@]}" > "${LOG_ROOT}/PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A")
  GPUS+=("${GPU}")
  echo "[PHASE1-CEN51-REPAIR-LAUNCHED] id=PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A pid=${pid} gpu=${GPU} log=${LOG_ROOT}/PHASE1_CEN51_REPAIR_CEN51_REFRESH_CONTROL_EXEMPT_GPU7_A.out"
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  for idx in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[$idx]}" ]] && wait "${PIDS[$idx]}"; then
      echo "[PHASE1-CEN51-REPAIR-COMPLETE] id=${NAMES[$idx]} pid=${PIDS[$idx]} status=0"
    else
      rc=$?
      if [[ -n "${PIDS[$idx]}" ]]; then echo "[PHASE1-CEN51-REPAIR-FAILED] id=${NAMES[$idx]} pid=${PIDS[$idx]} status=${rc}" >&2; STATUS=${rc}; fi
    fi
  done
  exit "${STATUS}"
fi
