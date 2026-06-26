#!/usr/bin/env bash
set -euo pipefail

# Centralized next8 after CEN_C09-C12/A13-A16 completed.
# Evidence anchors: CEN_A13 best clean/risk-adjusted, CEN_C11 best satellite/joint,
# CEN_A14 useful proto/SupCon satellite floor, CEN_A16 rx-chain collapse warning.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260530_003055_centralized_next8}"
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
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
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

launch CEN_C17 \
  CEN_C17_a13_groupdro_fishr0_ce1p22_r010 \
  0 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65
sleep 2

launch CEN_C18 \
  CEN_C18_c11_sat_lightgdro_ce1p44_r010 \
  1 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.44 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.04 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.55 \
  --groupdro_cap 0.60 \
  --use_swad_ckpt \
  --swad_start_epoch 81 \
  --swad_interval 1 \
  --swad_tolerance 1.20
sleep 2

launch CEN_C19 \
  CEN_C19_a14_proto_only_ce1p34_r010 \
  2 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.34 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --proto_momentum 0.95 \
  --generalization_feature z_id
sleep 2

launch CEN_C20 \
  CEN_C20_a14_supcon_only_ce1p34_r010 \
  3 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.34 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_supcon_id 0.02 \
  --supcon_temp 0.12 \
  --generalization_feature z_id
sleep 2

launch CEN_A21 \
  CEN_A21_a13_dualworst_gce012_r010 \
  4 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.12 \
  --group_ce_mode dual_worst \
  --group_ce_top_frac 0.30 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.75
sleep 2

launch CEN_A22 \
  CEN_A22_a13_proto_supcon_stack_r010 \
  5 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.02 \
  --supcon_temp 0.12 \
  --generalization_feature z_id
sleep 2

launch CEN_A23 \
  CEN_A23_satcurric_gdro_ce1p34_r010 \
  6 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.34 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule '1@0.85:clear_leo,low_elev_leo,rain_leo;81@1.00:clear_leo,low_elev_leo,rain_leo;121@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65
sleep 2

launch CEN_A24 \
  CEN_A24_a13_mixstyle_stop150_r010 \
  7 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --mixstyle_stop_epoch 150
