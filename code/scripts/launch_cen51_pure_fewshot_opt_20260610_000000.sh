#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_pure_fewshot_opt_20260610_000000}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

print_cmd() { printf '%q ' "$@"; printf '\n'; }

should_skip() {
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A LAUNCHED_BY_GPU=()

run_candidate() {
  local candidate_id="$1" run_name="$2" shots="$3" gpu="$4"
  shift 4
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${BASE_ARGS[@]}" "$@"
    --run_name "${run_name}"
    --wisig_train_shots_per_class "${shots}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth")

  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  echo "[CEN51-PURE-FS] candidate=${candidate_id} run=${run_name} shots_per_class=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
  printf '[CEN51-PURE-FS-CMD]'
  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

BASE_ARGS=(
  --train_mode
  centralized
  --eval_batch_size
  256
  --no_train_drop_last
  --dataset
  wisig
  --wisig_protocol
  cvs_day_rx
  --wisig_domain
  rx_day
  --wisig_train_ratio
  0.5
  --wisig_split_strategy
  random
  --wisig_cap_strategy
  random
  --wisig_max_train_per_combo
  0
  --test_eval_policy
  interval_final
  --test_eval_start_epoch
  1
  --test_eval_interval
  10
  --eval_sat_channel
  --eval_sat_on
  test_unseen_day_unseen_rx
  --eval_sat_scenarios
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches
  -1
  --arch_family
  cvsincnet
  --slim_group
  none
  --branch_ablation
  no_dac
  --domain_branch_ablation
  no_stats
  --domain_enhancer
  rcn_stats
  --domain_enhancer_strength
  0.35
  --exp_group
  s3_rxrobust_no_dac
  --model_variant
  lite_d
  --seed
  2028
  --collapse_guard
  --collapse_guard_min_epoch
  35
  --collapse_guard_best_margin
  12.0
  --collapse_guard_max_skipped_delta
  2
  --use_ema_ckpt
  --ema_decay
  0.999
  --use_swad_ckpt
  --swad_interval
  1
  --primary_udu_weight
  0.84
)

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi
cd "${ROOT}"
echo "[CEN51-PURE-FS] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

run_candidate FS005_R04_RANDOMSEL CEN51_PFS_FS005_R04_RANDOMSEL_seed2028 5 0 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  random

run_candidate FS005_R04_DOMBAL CEN51_PFS_FS005_R04_DOMBAL_seed2028 5 1 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS005_RXGRL_HINGE CEN51_PFS_FS005_RXGRL_HINGE_seed2028 5 2 \
  --epochs \
  220 \
  --swad_start_epoch \
  60 \
  --swad_tolerance \
  0.8 \
  --no_use_aug \
  --no_use_mixstyle \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --aug_p_pa \
  0.0 \
  --aug_p_dac \
  0.0 \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.35 \
  --lambda_adv \
  0.12 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.0 \
  --lambda_group_ce \
  0.0 \
  --lambda_proto \
  0.0 \
  --lambda_supcon_id \
  0.0 \
  --lambda_fishr \
  0.0 \
  --lambda_feature_norm_guard \
  0.001 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --no_use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS005_RXGRL_WEAKSAT CEN51_PFS_FS005_RXGRL_WEAKSAT_seed2028 5 3 \
  --epochs \
  220 \
  --swad_start_epoch \
  60 \
  --swad_tolerance \
  0.8 \
  --no_use_aug \
  --no_use_mixstyle \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --aug_p_pa \
  0.0 \
  --aug_p_dac \
  0.0 \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.35 \
  --lambda_adv \
  0.1 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.0 \
  --lambda_group_ce \
  0.0 \
  --lambda_proto \
  0.0 \
  --lambda_supcon_id \
  0.0 \
  --lambda_fishr \
  0.0 \
  --lambda_feature_norm_guard \
  0.0008 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.1 \
  --sat_view_prob \
  0.1 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.10:clear_leo,mixed_orbit \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS010_R04_RANDOMSEL CEN51_PFS_FS010_R04_RANDOMSEL_seed2028 10 4 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  random

run_candidate FS010_R04_DOMBAL CEN51_PFS_FS010_R04_DOMBAL_seed2028 10 5 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS010_RXGRL_HINGE CEN51_PFS_FS010_RXGRL_HINGE_seed2028 10 6 \
  --epochs \
  210 \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.8 \
  --no_use_aug \
  --no_use_mixstyle \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --aug_p_pa \
  0.0 \
  --aug_p_dac \
  0.0 \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.35 \
  --lambda_adv \
  0.18 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.0 \
  --lambda_group_ce \
  0.0 \
  --lambda_proto \
  0.0 \
  --lambda_supcon_id \
  0.0 \
  --lambda_fishr \
  0.0 \
  --lambda_feature_norm_guard \
  0.0007 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --no_use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS010_RXGRL_WEAKSAT CEN51_PFS_FS010_RXGRL_WEAKSAT_seed2028 10 7 \
  --epochs \
  210 \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.8 \
  --no_use_aug \
  --no_use_mixstyle \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --aug_p_pa \
  0.0 \
  --aug_p_dac \
  0.0 \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.35 \
  --lambda_adv \
  0.16 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.0 \
  --lambda_group_ce \
  0.0 \
  --lambda_proto \
  0.0 \
  --lambda_supcon_id \
  0.0 \
  --lambda_fishr \
  0.0 \
  --lambda_feature_norm_guard \
  0.0006 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.12 \
  --sat_view_prob \
  0.12 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.12:clear_leo,mixed_orbit \
  --batch_size \
  16 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS020_R04_DOMBAL CEN51_PFS_FS020_R04_DOMBAL_seed2028 20 0 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS020_SHOTAWARE_CLEAN CEN51_PFS_FS020_SHOTAWARE_CLEAN_seed2028 20 1 \
  --epochs \
  210 \
  --swad_start_epoch \
  80 \
  --swad_tolerance \
  0.65 \
  --use_aug \
  --aug_scale_min \
  0.03 \
  --aug_scale_max \
  0.18 \
  --late_aug_min_scale \
  0.14 \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_p \
  0.04 \
  --mixstyle_strength \
  0.18 \
  --mixstyle_late_start \
  125 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.02 \
  --mixstyle_late_min_strength \
  0.12 \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.45 \
  --lambda_adv \
  0.2 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.008 \
  --lambda_group_ce \
  0.012 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.18 \
  --groupdro_tau \
  0.35 \
  --groupdro_cap \
  0.45 \
  --use_proto_memory \
  --lambda_proto \
  0.0015 \
  --proto_momentum \
  0.95 \
  --lambda_supcon_id \
  0.0015 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.0 \
  --fishr_min_domains \
  2 \
  --lambda_feature_norm_guard \
  0.0001 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --no_use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS020_SHOTAWARE_SATGATE CEN51_PFS_FS020_SHOTAWARE_SATGATE_seed2028 20 2 \
  --epochs \
  210 \
  --swad_start_epoch \
  80 \
  --swad_tolerance \
  0.65 \
  --use_aug \
  --aug_scale_min \
  0.03 \
  --aug_scale_max \
  0.18 \
  --late_aug_min_scale \
  0.14 \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_p \
  0.04 \
  --mixstyle_strength \
  0.18 \
  --mixstyle_late_start \
  125 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.02 \
  --mixstyle_late_min_strength \
  0.12 \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.45 \
  --lambda_adv \
  0.18 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.008 \
  --lambda_group_ce \
  0.01 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.18 \
  --groupdro_tau \
  0.35 \
  --groupdro_cap \
  0.45 \
  --use_proto_memory \
  --lambda_proto \
  0.0015 \
  --proto_momentum \
  0.95 \
  --lambda_supcon_id \
  0.0015 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.0 \
  --fishr_min_domains \
  2 \
  --lambda_feature_norm_guard \
  8e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.14 \
  --sat_view_prob \
  0.14 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  '1@0.14:clear_leo,mixed_orbit;150@0.08:clear_leo,mixed_orbit' \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS030_R04_DOMBAL CEN51_PFS_FS030_R04_DOMBAL_seed2028 30 3 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS030_SHOTAWARE_CLEAN CEN51_PFS_FS030_SHOTAWARE_CLEAN_seed2028 30 4 \
  --epochs \
  220 \
  --swad_start_epoch \
  90 \
  --swad_tolerance \
  0.65 \
  --use_aug \
  --aug_scale_min \
  0.03 \
  --aug_scale_max \
  0.22 \
  --late_aug_min_scale \
  0.16 \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_p \
  0.06 \
  --mixstyle_strength \
  0.24 \
  --mixstyle_late_start \
  135 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.027 \
  --mixstyle_late_min_strength \
  0.132 \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.5 \
  --lambda_adv \
  0.24 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.012 \
  --lambda_group_ce \
  0.02 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.22 \
  --groupdro_tau \
  0.42 \
  --groupdro_cap \
  0.55 \
  --use_proto_memory \
  --lambda_proto \
  0.0025 \
  --proto_momentum \
  0.95 \
  --lambda_supcon_id \
  0.0025 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.0003 \
  --fishr_min_domains \
  2 \
  --lambda_feature_norm_guard \
  8e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --no_use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS030_SHOTAWARE_SATGATE CEN51_PFS_FS030_SHOTAWARE_SATGATE_seed2028 30 5 \
  --epochs \
  220 \
  --swad_start_epoch \
  90 \
  --swad_tolerance \
  0.65 \
  --use_aug \
  --aug_scale_min \
  0.03 \
  --aug_scale_max \
  0.22 \
  --late_aug_min_scale \
  0.16 \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_p \
  0.06 \
  --mixstyle_strength \
  0.24 \
  --mixstyle_late_start \
  135 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.027 \
  --mixstyle_late_min_strength \
  0.132 \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.5 \
  --lambda_adv \
  0.22 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.012 \
  --lambda_group_ce \
  0.018 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.22 \
  --groupdro_tau \
  0.42 \
  --groupdro_cap \
  0.55 \
  --use_proto_memory \
  --lambda_proto \
  0.0025 \
  --proto_momentum \
  0.95 \
  --lambda_supcon_id \
  0.0025 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.0003 \
  --fishr_min_domains \
  2 \
  --lambda_feature_norm_guard \
  6e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.18 \
  --sat_view_prob \
  0.18 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  '1@0.18:clear_leo,mixed_orbit;150@0.11:clear_leo,mixed_orbit' \
  --batch_size \
  32 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS050_R04_DOMBAL CEN51_PFS_FS050_R04_DOMBAL_seed2028 50 6 \
  --epochs \
  200 \
  --use_aug \
  --use_concat_sat_channel_aug \
  --concat_sat_start_epoch \
  1 \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.006 \
  --sat_cons_start_epoch \
  118 \
  --use_sat_consistency \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob \
  1.0 \
  --concat_sat_ce_weight \
  1.19 \
  --sat_view_schedule \
  '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_strength \
  0.7 \
  --mixstyle_p \
  0.18 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --domain_freq_stability_mode \
  dsq \
  --freq_stability_channels \
  2 \
  --lambda_group_ce \
  0.088 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  4 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.37 \
  --groupdro_cap \
  0.48 \
  --use_proto_memory \
  --lambda_proto \
  0.016 \
  --proto_momentum \
  0.97 \
  --lambda_supcon_id \
  0.022 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.002 \
  --fishr_min_domains \
  4 \
  --generalization_feature \
  z_id \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.34 \
  --pa_orders \
  1,3,5 \
  --batch_size \
  64 \
  --wisig_train_shot_strategy \
  domain_balanced

run_candidate FS050_SHOTAWARE_SATGATE CEN51_PFS_FS050_SHOTAWARE_SATGATE_seed2028 50 7 \
  --epochs \
  230 \
  --swad_start_epoch \
  100 \
  --swad_tolerance \
  0.65 \
  --use_aug \
  --aug_scale_min \
  0.03 \
  --aug_scale_max \
  0.26 \
  --late_aug_min_scale \
  0.18 \
  --use_mixstyle \
  --mixstyle_layers \
  time_down,t1 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_p \
  0.1 \
  --mixstyle_strength \
  0.32 \
  --mixstyle_late_start \
  145 \
  --mixstyle_late_ramp_epochs \
  40 \
  --mixstyle_late_min_p \
  0.045000000000000005 \
  --mixstyle_late_min_strength \
  0.17600000000000002 \
  --no_enable_pa_aux \
  --no_enable_dac_aux \
  --no_aug_enable_pa_normal \
  --lambda_cls_pa \
  0.0 \
  --lambda_pa_joint_inv \
  0.0 \
  --lambda_pa_kl \
  0.0 \
  --lambda_pa_reg \
  0.0 \
  --lambda_dom \
  0.55 \
  --lambda_adv \
  0.26 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.016 \
  --lambda_group_ce \
  0.028 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.25 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.62 \
  --use_proto_memory \
  --lambda_proto \
  0.004 \
  --proto_momentum \
  0.95 \
  --lambda_supcon_id \
  0.004 \
  --supcon_temp \
  0.12 \
  --lambda_fishr \
  0.0005 \
  --fishr_min_domains \
  2 \
  --lambda_feature_norm_guard \
  4e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --domain_freq_stability_mode \
  off \
  --pa_orders \
  1,3,5 \
  --no_use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --lambda_sat_cls \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_weight \
  0.24 \
  --sat_view_prob \
  0.24 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  '1@0.24:clear_leo,mixed_orbit;150@0.14:clear_leo,mixed_orbit' \
  --batch_size \
  64 \
  --wisig_train_shot_strategy \
  domain_balanced

