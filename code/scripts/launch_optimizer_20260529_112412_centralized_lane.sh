#!/usr/bin/env bash
set -euo pipefail

# One-run centralized lane candidate selected by optimizer_20260529_112412.
# Parent evidence anchor: SA64_floor_lowrain_ce1p45_eval81_r010.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260529_112412_centralized_lane}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="CEN_SA75_sa64_swad_lowrain_ce1p42_eval81_r010"

COMMON_ARGS=(
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --primary_udu_weight 0.70
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 81
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --lambda_fishr 0.02
  --fishr_min_domains 4
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_ce_weight 1.42
  --sat_train_scenarios low_elev_leo,rain_leo
  --concat_sat_start_epoch 1
  --sat_view_prob 1.00
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --use_swad_ckpt
  --swad_start_epoch 81
  --swad_interval 1
  --swad_tolerance 1.5
  --seed 1337
  --run_name "${RUN_NAME}"
)

cd "${ROOT}"

CMD=(
  env "CUDA_VISIBLE_DEVICES=${GPU}" PYTHONPATH=. "${PYTHON}" -u train.py
  "${COMMON_ARGS[@]}"
  --latest_save_path "${RUNS_ROOT}/${RUN_NAME}/latest_model.pth"
  --best_save_path "${RUNS_ROOT}/${RUN_NAME}/best_val_model.pth"
  --best_primary_save_path "${RUNS_ROOT}/${RUN_NAME}/best_primary_ood_model.pth"
  --best_unseen_day_unseen_rx_save_path "${RUNS_ROOT}/${RUN_NAME}/best_strict_udu_model.pth"
  --swad_save_path "${RUNS_ROOT}/${RUN_NAME}/swad_model.pth"
)

echo "[CENTRALIZED-LANE] root=${ROOT} run_id=${RUN_ID} run=${RUN_NAME} gpu=${GPU} dry_run=${DRY_RUN}"
printf '[CENTRALIZED-LANE-CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ ! -f "${ROOT}/train.py" ]]; then
  echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
  exit 2
fi

active_pids="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
if [[ -n "${active_pids}" ]]; then
  echo "[ERROR] gpu=${GPU} already has compute process(es): ${active_pids}" >&2
  exit 4
fi

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
PID="$!"
printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${RUNS_ROOT}/${RUN_NAME}" | tee "${LOG_ROOT}/launch_pids.tsv"
