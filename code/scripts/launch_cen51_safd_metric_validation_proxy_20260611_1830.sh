#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RUN_ID="cen51_safd_metric_validation_proxy_20260611_1830"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
TRAIN="${ROOT}/code/train.py"
export PYTHONPATH="${ROOT}/code:${ROOT}/tools:${ROOT}"

DRY_RUN=0
MAX_ACTIVE_PER_GPU=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --max-active-per-gpu)
      MAX_ACTIVE_PER_GPU="$2"
      shift 2
      ;;
    *)
      echo "[SAFD-METRIC] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF { n += 1 } END { print n + 0 }'
}

print_cmd() {
  local -a cmd=("$@")
  printf '%q ' "${cmd[@]}"
  printf '\n'
}

COMMON_ARGS=(
  --train_mode centralized
  --dataset wisig
  --wisig_protocol cvs_day_rx
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_val_ratio -1
  --wisig_guard_gap 8
  --wisig_split_strategy random
  --wisig_cap_strategy random
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --eval_batch_size 256
  --test_eval_policy interval_final
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --arch_family cvsincnet
  --slim_group none
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --pa_orders 1,3,5
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 12
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --swad_interval 1
  --primary_udu_weight 0.84
  --label_smoothing 0.0
  --generalization_feature z_id
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl"
)

LOW_NO_AUG_ARGS=(
  --no_use_aug
  --no_use_mixstyle
  --no_enable_pa_aux
  --no_enable_dac_aux
  --no_aug_enable_pa_normal
  --aug_p_pa 0.0
  --aug_p_dac 0.0
  --lambda_cls_pa 0.0
  --lambda_pa_joint_inv 0.0
  --lambda_pa_kl 0.0
  --lambda_pa_reg 0.0
)

MID_AUG_ARGS=(
  --use_aug
  --aug_scale_min 0.08
  --aug_scale_max 0.26
  --late_aug_min_scale 0.12
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_p 0.025
  --mixstyle_strength 0.22
  --mixstyle_late_start 80
  --mixstyle_late_ramp_epochs 35
  --mixstyle_late_min_p 0.018
  --mixstyle_late_min_strength 0.14
  --no_enable_pa_aux
  --no_enable_dac_aux
  --no_aug_enable_pa_normal
  --aug_p_pa 0.0
  --aug_p_dac 0.0
  --lambda_cls_pa 0.0
  --lambda_pa_joint_inv 0.0
  --lambda_pa_kl 0.0
  --lambda_pa_reg 0.0
)

FULLDG_SAT_K5_ARGS=(
  --use_concat_sat_channel_aug
  --no_use_sat_consistency
  --lambda_sat_cls 0.0
  --lambda_sat_cons 0.0
  --concat_sat_start_epoch 90
  --sat_view_prob 0.060
  --sat_train_scenarios clear_leo,mixed_orbit
  --sat_view_schedule "1@0.000:clear_leo,mixed_orbit;90@0.060:clear_leo,mixed_orbit"
)

FULLDG_SAT_K10_ARGS=(
  --use_concat_sat_channel_aug
  --no_use_sat_consistency
  --lambda_sat_cls 0.0
  --lambda_sat_cons 0.0
  --concat_sat_start_epoch 70
  --sat_view_prob 0.100
  --sat_train_scenarios clear_leo,mixed_orbit
  --sat_view_schedule "1@0.000:clear_leo,mixed_orbit;70@0.100:clear_leo,mixed_orbit"
)

FULLDG_SAT_K30_ARGS=(
  --use_concat_sat_channel_aug
  --no_use_sat_consistency
  --lambda_sat_cls 0.0
  --lambda_sat_cons 0.0
  --concat_sat_start_epoch 35
  --sat_view_prob 0.180
  --sat_train_scenarios clear_leo,low_elev_leo,mixed_orbit
  --sat_view_schedule "1@0.000:clear_leo,low_elev_leo,mixed_orbit;35@0.180:clear_leo,low_elev_leo,mixed_orbit"
)

NO_SAT_ARGS=(
  --no_use_concat_sat_channel_aug
  --no_use_sat_consistency
  --lambda_sat_cls 0.0
  --lambda_sat_cons 0.0
  --sat_view_prob 0.0
  --concat_sat_start_epoch 999
)

launch_train() {
  local candidate="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  local count
  count="$(gpu_process_count "${gpu}")"
  if [[ "${count}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[SAFD-METRIC] skip_capacity candidate=${candidate} gpu=${gpu} active=${count} max=${MAX_ACTIVE_PER_GPU}"
    return 0
  fi

  local out_dir="${RUN_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${candidate}.log"
  local pid_path="${LOG_ROOT}/launch_pids.tsv"
  local -a cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${PYTHONPATH}"
    "${PYTHON}" -u "${TRAIN}"
    "${COMMON_ARGS[@]}"
    "$@"
    --run_name "${run_name}"
    --output_dir "${out_dir}"
    --latest_save_path "${out_dir}/latest_model.pth"
    --best_save_path "${out_dir}/best_val_model.pth"
    --best_primary_save_path "${out_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${out_dir}/best_strict_udu_model.pth"
    --best_worst_rx_save_path "${out_dir}/best_worst_rx_model.pth"
    --ema_save_path "${out_dir}/ema_model.pth"
    --swa_save_path "${out_dir}/swa_model.pth"
    --swad_save_path "${out_dir}/swad_model.pth"
  )

  echo "[SAFD-METRIC] candidate=${candidate} run=${run_name} gpu=${gpu} active_before=${count} dry_run=${DRY_RUN}"
  echo -n "[SAFD-METRIC-CMD] "
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if [[ -d "${out_dir}" ]] && find "${out_dir}" -mindepth 1 -maxdepth 1 | grep -q .; then
    echo "[SAFD-METRIC] skip_existing_output run=${run_name} out_dir=${out_dir}"
    return 0
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}"
  if [[ ! -f "${pid_path}" ]]; then
    printf 'candidate\trun_name\tgpu\tpid\tlog_path\tout_dir\n' > "${pid_path}"
  fi
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${candidate}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${out_dir}" >> "${pid_path}"
  echo "[SAFD-METRIC] launched candidate=${candidate} pid=${pid} log=${log_path}"
  sleep 4
}

echo "[SAFD-METRIC] run_id=${RUN_ID} dry_run=${DRY_RUN} max_active_per_gpu=${MAX_ACTIVE_PER_GPU}"

# GPU0: K5 satellite contrast, seed 1337.
launch_train K005_NOSAT_S1337 CEN51_SAFDMV_K005_NOSAT_s1337 0 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 5 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.85 \
  --seed 1337 --sat_view_seed 91337 \
  "${LOW_NO_AUG_ARGS[@]}" "${NO_SAT_ARGS[@]}" \
  --lambda_dom 0.38 --lambda_adv 0.08 --grl_lambda 1.0 --lambda_orth 0.010 --lambda_cons 0.002 \
  --lambda_group_ce 0.004 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.12 \
  --groupdro_tau 0.34 --groupdro_cap 0.40 --use_proto_memory --lambda_proto 0.0005 \
  --lambda_supcon_id 0.0005 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00045 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

launch_train K005_FULLDG_S1337 CEN51_SAFDMV_K005_FULLDG_s1337 0 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 5 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.85 \
  --seed 1337 --sat_view_seed 91337 \
  "${LOW_NO_AUG_ARGS[@]}" "${FULLDG_SAT_K5_ARGS[@]}" \
  --lambda_dom 0.38 --lambda_adv 0.08 --grl_lambda 1.0 --lambda_orth 0.010 --lambda_cons 0.002 \
  --lambda_group_ce 0.004 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.12 \
  --groupdro_tau 0.34 --groupdro_cap 0.40 --use_proto_memory --lambda_proto 0.0005 \
  --lambda_supcon_id 0.0005 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00045 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

# GPU1: K10 satellite contrast, seed 1337.
launch_train K010_NOSAT_S1337 CEN51_SAFDMV_K010_NOSAT_s1337 1 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 10 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.80 \
  --seed 1337 --sat_view_seed 92337 \
  "${LOW_NO_AUG_ARGS[@]}" "${NO_SAT_ARGS[@]}" \
  --lambda_dom 0.42 --lambda_adv 0.12 --grl_lambda 1.0 --lambda_orth 0.015 --lambda_cons 0.004 \
  --lambda_group_ce 0.006 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.16 \
  --groupdro_tau 0.35 --groupdro_cap 0.42 --use_proto_memory --lambda_proto 0.0008 \
  --lambda_supcon_id 0.0008 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00035 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

launch_train K010_FULLDG_S1337 CEN51_SAFDMV_K010_FULLDG_s1337 1 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 10 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.80 \
  --seed 1337 --sat_view_seed 92337 \
  "${LOW_NO_AUG_ARGS[@]}" "${FULLDG_SAT_K10_ARGS[@]}" \
  --lambda_dom 0.42 --lambda_adv 0.12 --grl_lambda 1.0 --lambda_orth 0.015 --lambda_cons 0.004 \
  --lambda_group_ce 0.006 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.16 \
  --groupdro_tau 0.35 --groupdro_cap 0.42 --use_proto_memory --lambda_proto 0.0008 \
  --lambda_supcon_id 0.0008 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00035 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

# GPU2: stability repeat for K5/K10 full-DG, seed 2028.
launch_train K005_FULLDG_S2028 CEN51_SAFDMV_K005_FULLDG_s2028 2 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 5 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.85 \
  --seed 2028 --sat_view_seed 92028 \
  "${LOW_NO_AUG_ARGS[@]}" "${FULLDG_SAT_K5_ARGS[@]}" \
  --lambda_dom 0.38 --lambda_adv 0.08 --grl_lambda 1.0 --lambda_orth 0.010 --lambda_cons 0.002 \
  --lambda_group_ce 0.004 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.12 \
  --groupdro_tau 0.34 --groupdro_cap 0.40 --use_proto_memory --lambda_proto 0.0005 \
  --lambda_supcon_id 0.0005 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00045 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

launch_train K010_FULLDG_S2028 CEN51_SAFDMV_K010_FULLDG_s2028 2 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 10 --batch_size 128 --epochs 125 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 50 --swad_tolerance 0.80 \
  --seed 2028 --sat_view_seed 93028 \
  "${LOW_NO_AUG_ARGS[@]}" "${FULLDG_SAT_K10_ARGS[@]}" \
  --lambda_dom 0.42 --lambda_adv 0.12 --grl_lambda 1.0 --lambda_orth 0.015 --lambda_cons 0.004 \
  --lambda_group_ce 0.006 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.16 \
  --groupdro_tau 0.35 --groupdro_cap 0.42 --use_proto_memory --lambda_proto 0.0008 \
  --lambda_supcon_id 0.0008 --supcon_temp 0.12 --lambda_fishr 0.0 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00035 --feature_norm_guard_mode hinge --feature_norm_guard_target 8.5

# GPU3: K30 transition-zone satellite contrast.
launch_train K030_NOSAT_S1337 CEN51_SAFDMV_K030_NOSAT_s1337 3 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 30 --batch_size 128 --epochs 135 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 60 --swad_tolerance 0.70 \
  --seed 1337 --sat_view_seed 94337 \
  "${MID_AUG_ARGS[@]}" "${NO_SAT_ARGS[@]}" \
  --lambda_dom 0.50 --lambda_adv 0.20 --grl_lambda 1.0 --lambda_orth 0.024 --lambda_cons 0.012 \
  --lambda_group_ce 0.018 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.20 \
  --groupdro_tau 0.39 --groupdro_cap 0.52 --use_proto_memory --lambda_proto 0.0025 \
  --lambda_supcon_id 0.0025 --supcon_temp 0.12 --lambda_fishr 0.0002 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00004 --feature_norm_guard_mode l2 --feature_norm_guard_target 0.0

launch_train K030_FULLDG_S1337 CEN51_SAFDMV_K030_FULLDG_s1337 3 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 30 --batch_size 128 --epochs 135 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 60 --swad_tolerance 0.70 \
  --seed 1337 --sat_view_seed 94337 \
  "${MID_AUG_ARGS[@]}" "${FULLDG_SAT_K30_ARGS[@]}" \
  --lambda_dom 0.50 --lambda_adv 0.20 --grl_lambda 1.0 --lambda_orth 0.024 --lambda_cons 0.012 \
  --lambda_group_ce 0.018 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.20 \
  --groupdro_tau 0.39 --groupdro_cap 0.52 --use_proto_memory --lambda_proto 0.0025 \
  --lambda_supcon_id 0.0025 --supcon_temp 0.12 --lambda_fishr 0.0002 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00004 --feature_norm_guard_mode l2 --feature_norm_guard_target 0.0

# GPU4: K50 transition-zone satellite contrast with slightly relaxed regularization.
launch_train K050_NOSAT_S1337 CEN51_SAFDMV_K050_NOSAT_s1337 4 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 50 --batch_size 160 --epochs 145 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 65 --swad_tolerance 0.60 \
  --seed 1337 --sat_view_seed 95337 \
  "${MID_AUG_ARGS[@]}" "${NO_SAT_ARGS[@]}" \
  --lambda_dom 0.55 --lambda_adv 0.26 --grl_lambda 1.0 --lambda_orth 0.028 --lambda_cons 0.018 \
  --lambda_group_ce 0.028 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.20 \
  --groupdro_tau 0.39 --groupdro_cap 0.52 --use_proto_memory --lambda_proto 0.0040 \
  --lambda_supcon_id 0.0040 --supcon_temp 0.12 --lambda_fishr 0.0004 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00002 --feature_norm_guard_mode l2 --feature_norm_guard_target 0.0

launch_train K050_FULLDG_S1337 CEN51_SAFDMV_K050_FULLDG_s1337 4 \
  --wisig_train_ratio 0.1 --wisig_max_train_per_combo 50 --batch_size 160 --epochs 145 \
  --test_eval_start_epoch 31 --test_eval_interval 20 --swad_start_epoch 65 --swad_tolerance 0.60 \
  --seed 1337 --sat_view_seed 95337 \
  "${MID_AUG_ARGS[@]}" "${FULLDG_SAT_K30_ARGS[@]}" \
  --lambda_dom 0.55 --lambda_adv 0.26 --grl_lambda 1.0 --lambda_orth 0.028 --lambda_cons 0.018 \
  --lambda_group_ce 0.028 --group_ce_mode smooth_dro_capped --group_ce_min_domains 2 --group_ce_top_frac 0.20 \
  --groupdro_tau 0.39 --groupdro_cap 0.52 --use_proto_memory --lambda_proto 0.0040 \
  --lambda_supcon_id 0.0040 --supcon_temp 0.12 --lambda_fishr 0.0004 --fishr_min_domains 2 \
  --lambda_feature_norm_guard 0.00002 --feature_norm_guard_mode l2 --feature_norm_guard_target 0.0

# GPU5: high-sample anchor pair. R0.4 full-DG R04 schedule vs no-sat control.
launch_train R040_NOSAT_S1337 CEN51_SAFDMV_R040_NOSAT_s1337 5 \
  --wisig_train_ratio 0.4 --batch_size 256 --epochs 200 \
  --test_eval_start_epoch 1 --test_eval_interval 10 --swad_start_epoch 70 --swad_tolerance 0.34 \
  --seed 1337 "${NO_SAT_ARGS[@]}" \
  --use_aug --use_mixstyle --mixstyle_layers time_down,t1 --mixstyle_mix same_tx_crossdomain --mixstyle_fallback skip \
  --mixstyle_strength 0.70 --mixstyle_p 0.18 --mixstyle_late_start 110 --mixstyle_late_ramp_epochs 40 \
  --mixstyle_late_min_p 0.05 --mixstyle_late_min_strength 0.32 \
  --domain_freq_stability_mode dsq --freq_stability_channels 2 --lambda_dom 1.0 --lambda_adv 0.70 \
  --grl_lambda 1.0 --lambda_orth 1.0 --lambda_cons 0.45 --lambda_group_ce 0.088 \
  --group_ce_mode smooth_dro_capped --group_ce_min_domains 4 --group_ce_top_frac 0.20 --groupdro_tau 0.37 \
  --groupdro_cap 0.48 --use_proto_memory --lambda_proto 0.016 --proto_momentum 0.97 \
  --lambda_supcon_id 0.022 --supcon_temp 0.12 --lambda_fishr 0.002 --fishr_min_domains 4

launch_train R040_FULLDG_R04_S1337 CEN51_SAFDMV_R040_FULLDG_R04_s1337 5 \
  --wisig_train_ratio 0.4 --batch_size 256 --epochs 200 \
  --test_eval_start_epoch 1 --test_eval_interval 10 --swad_start_epoch 70 --swad_tolerance 0.34 \
  --seed 1337 --use_aug --use_mixstyle --mixstyle_layers time_down,t1 --mixstyle_mix same_tx_crossdomain \
  --mixstyle_fallback skip --mixstyle_strength 0.70 --mixstyle_p 0.18 --mixstyle_late_start 110 \
  --mixstyle_late_ramp_epochs 40 --mixstyle_late_min_p 0.05 --mixstyle_late_min_strength 0.32 \
  --use_concat_sat_channel_aug --no_use_sat_consistency --concat_sat_start_epoch 1 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 \
  --sat_view_schedule "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq --freq_stability_channels 2 --lambda_dom 1.0 --lambda_adv 0.70 \
  --grl_lambda 1.0 --lambda_orth 1.0 --lambda_cons 0.45 --lambda_group_ce 0.088 \
  --group_ce_mode smooth_dro_capped --group_ce_min_domains 4 --group_ce_top_frac 0.20 --groupdro_tau 0.37 \
  --groupdro_cap 0.48 --use_proto_memory --lambda_proto 0.016 --proto_momentum 0.97 \
  --lambda_supcon_id 0.022 --supcon_temp 0.12 --lambda_fishr 0.002 --fishr_min_domains 4 \
  --lambda_sat_cls 0.0 --lambda_sat_cons 0.0

echo "[SAFD-METRIC] submitted run_id=${RUN_ID}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "[SAFD-METRIC] launch_pids=${LOG_ROOT}/launch_pids.tsv"
fi
