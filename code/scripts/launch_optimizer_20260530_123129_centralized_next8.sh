#!/usr/bin/env bash
set -euo pipefail

# Centralized next8 after CEN_C33-C36/A37-A40 completed.
# Evidence anchors: A40 is best clean/strict, C35 is strongest final-primary
# satellite, A39 is strongest final-best satellite, and C33 is best
# risk-adjusted. This launcher only uses existing train.py knobs.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260530_123129_centralized_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
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
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
)

MIXSTYLE_ARGS=(
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
)

CONCAT_SAT_ARGS=(
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
)

STACK_ARGS=(
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --use_proto_memory
  --lambda_proto 0.015
  --proto_momentum 0.95
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --generalization_feature z_id
)

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

launch() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    "$@"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )

  echo "[CENTRALIZED-NEXT8] candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CENTRALIZED-NEXT8-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${ROOT}/train.py" ]]; then
    echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
    exit 2
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch CEN_C41 \
  CEN_C41_a40_ce130_pudu074_mixstop_r010 \
  0 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --mixstyle_stop_epoch 150
sleep 2

launch CEN_C42 \
  CEN_C42_c35_satfloor_sched_ce130_r010 \
  1 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;111@1.00:low_elev_leo,rain_leo,storm_mp'
sleep 2

launch CEN_C43 \
  CEN_C43_c36_gce007_cap062_cleanrisk_r010 \
  2 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.07 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.30 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.62 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.02 \
  --supcon_temp 0.12 \
  --generalization_feature z_id
sleep 2

launch CEN_C44 \
  CEN_C44_c33_swa_jointguard_ce130_r010 \
  3 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --use_swa_ckpt \
  --swa_start_epoch 120 \
  --swa_interval 2
sleep 2

launch CEN_A45 \
  CEN_A45_a40_stormmix_ce134_r010 \
  4 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.34 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 0.95 \
  --sat_view_schedule '1@0.85:clear_leo,low_elev_leo,rain_leo;91@1.00:mixed_orbit,storm_mp,low_elev_leo,rain_leo' \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --mixstyle_stop_epoch 150
sleep 2

launch CEN_A46 \
  CEN_A46_a40_mixstop140_swa_r010 \
  5 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.28 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --mixstyle_stop_epoch 140 \
  --use_swa_ckpt \
  --swa_start_epoch 115 \
  --swa_interval 2
sleep 2

launch CEN_A47 \
  CEN_A47_a38_softsatcons_cls035_r010 \
  6 \
  "${MIXSTYLE_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.70 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 0.75 \
  --use_sat_consistency \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.035 \
  --lambda_sat_cons 0.004 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4
sleep 2

launch CEN_A48 \
  CEN_A48_a40_proto025_supcon030_r010 \
  7 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.28 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_proto 0.025 \
  --lambda_supcon_id 0.030 \
  --supcon_temp 0.10 \
  --lambda_fishr 0.008 \
  --fishr_min_domains 4
