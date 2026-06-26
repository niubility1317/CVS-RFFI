#!/usr/bin/env bash
set -euo pipefail

# Centralized next8 after CEN_C73-C76/A77-A80 completed.
# Evidence anchors: C76 is best clean/risk-adjusted, C74 is best satellite,
# and A79 is best joint but shows satellite-floor collapse risk.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260531_175432_centralized_next8}"
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
  --seed 1337
)

CONCAT_SAT_ARGS=(
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
)

MIXSTYLE_LIGHT_ARGS=(
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.36
  --mixstyle_p 0.07
  --mixstyle_late_start 95
  --mixstyle_late_ramp_epochs 35
  --mixstyle_late_min_p 0.012
  --mixstyle_late_min_strength 0.08
)

PHASE_JOINT_ARGS=(
  --id_time_stability_mode phase_delta
  --domain_time_stability_mode off
  --time_stability_channels 2
  --use_proto_memory
  --lambda_proto 0.020
  --proto_momentum 0.95
  --lambda_supcon_id 0.024
  --supcon_temp 0.10
  --generalization_feature id_feat_joint
)

SMOOTH_DRO_ARGS=(
  --domain_freq_stability_mode off
  --lambda_group_ce 0.055
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.28
  --groupdro_tau 0.42
  --groupdro_cap 0.60
)

DUAL_WORST_ARGS=(
  --domain_freq_stability_mode off
  --lambda_group_ce 0.062
  --group_ce_mode dual_worst
  --group_ce_top_frac 0.32
  --groupdro_tau 0.38
  --groupdro_cap 0.58
)

LATE_STABLE_ARGS=(
  --late_stable_start 105
  --late_stable_ramp_epochs 30
  --late_adv_min_scale 0.70
  --late_cons_min_scale 0.50
  --late_group_ce_min_scale 0.70
  --late_aug_min_scale 0.70
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

  if [[ -n "${ONLY_CANDIDATE:-}" && "${candidate_id}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

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
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
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

launch CEN_C81 \
  CEN_C81_c76_satfloor_bridge_r010 \
  0 \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.78 \
  --concat_sat_ce_weight 1.26 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.95 \
  --sat_view_schedule '1@0.95:clear_leo,low_elev_leo,rain_leo;120@0.86:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --collapse_guard_min_epoch 35 \
  --collapse_guard_best_margin 12.0 \
  --collapse_guard_max_skipped_delta 2 \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 90 \
  --swad_tolerance 0.8
sleep 2

launch CEN_C82 \
  CEN_C82_c74_clean_recover_ema_r010 \
  1 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.76 \
  --concat_sat_ce_weight 1.27 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.98 \
  --sat_view_schedule '1@0.92:clear_leo,low_elev_leo,rain_leo;120@0.82:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 92 \
  --swad_tolerance 0.9
sleep 2

launch CEN_C83 \
  CEN_C83_a78_fishr_floor_soft_r010 \
  2 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.75 \
  --concat_sat_ce_weight 1.26 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.95 \
  --sat_view_schedule '1@0.90:clear_leo,low_elev_leo,rain_leo;115@0.82:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --use_swad_ckpt \
  --swad_start_epoch 92 \
  --swad_tolerance 1.0
sleep 2

launch CEN_C84 \
  CEN_C84_a79_joint_sat_recover_r010 \
  3 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.10 \
  --use_sat_consistency \
  --lambda_sat_cls 0.040 \
  --lambda_sat_cons 0.002 \
  --sat_cons_start_epoch 130 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.45 \
  --sat_view_schedule '1@0.35:clear_leo,low_elev_leo,rain_leo;130@0.30:clear_leo,low_elev_leo,rain_leo;150@0.22:clear_leo,low_elev_leo,rain_leo' \
  --use_proto_memory \
  --proto_momentum 0.95 \
  --lambda_proto 0.010 \
  --lambda_supcon_id 0.010 \
  --supcon_temp 0.10 \
  --generalization_feature id_feat_joint \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --use_swad_ckpt \
  --swad_start_epoch 100 \
  --swad_tolerance 1.2
sleep 2

launch CEN_A85 \
  CEN_A85_c76_rxchain_ema_stress_r010 \
  4 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.77 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.92 \
  --aug_p_rx_chain 0.14 \
  --aug_rx_chain_envs 8 \
  --aug_rx_chain_p_lowpass 0.70 \
  --aug_rx_chain_p_multipath 0.70 \
  --lambda_fishr 0.000 \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 92 \
  --swad_tolerance 1.0
sleep 2

launch CEN_A86 \
  CEN_A86_c74_highsat_fishr_dsq_r010 \
  5 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --id_freq_stability_mode dsq \
  --domain_freq_stability_mode off \
  --freq_stability_channels 2 \
  --primary_udu_weight 0.73 \
  --concat_sat_ce_weight 1.34 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo;110@0.86:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.004 \
  --fishr_min_domains 4 \
  --use_swad_ckpt \
  --swad_start_epoch 92 \
  --swad_tolerance 1.0
sleep 2

launch CEN_A87 \
  CEN_A87_c76_clean_floor_nosat_r010 \
  6 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --id_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --primary_udu_weight 0.82 \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --collapse_guard_min_epoch 35 \
  --collapse_guard_best_margin 12.0 \
  --collapse_guard_max_skipped_delta 2 \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 90 \
  --swad_tolerance 0.8
sleep 2

launch CEN_A88 \
  CEN_A88_c76_satmax_swad_r010 \
  7 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.36 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo;105@0.90:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --collapse_guard_min_epoch 35 \
  --collapse_guard_best_margin 12.0 \
  --collapse_guard_max_skipped_delta 2 \
  --use_swad_ckpt \
  --swad_start_epoch 90 \
  --swad_tolerance 0.8
