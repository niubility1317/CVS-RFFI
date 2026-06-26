#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_fulldg_shotaware_lacsr_fd_20260610_161500}"
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
  if [[ "${DRY_RUN}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' ' || echo 0
}

print_cmd() { printf '%q ' "$@"; printf '\n'; }

should_skip() {
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A INITIAL_BY_GPU=()
declare -A LAUNCHED_BY_GPU=()
snapshot_capacity() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    INITIAL_BY_GPU[${gpu}]="$(gpu_process_count "${gpu}")"
    LAUNCHED_BY_GPU[${gpu}]=0
  done
}

run_candidate() {
  local candidate_id="$1" run_name="$2" shots="$3" gpu="$4"
  shift 4
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${BASE_ARGS[@]}" "$@"
    --run_name "${run_name}"
    --wisig_max_train_per_combo "${shots}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth")

  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  echo "[CEN51-FDLACSRFD] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
  printf '[CEN51-FDLACSRFD-CMD]'
  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  local initial_count local_count
  initial_count="${INITIAL_BY_GPU[${gpu}]:-0}"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s initial_count=%s local_count=%s max=%s\n" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${initial_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU[${gpu}]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

BASE_ARGS=(
  --train_mode
  centralized
  --eval_batch_size
  256
  --dataset
  wisig
  --wisig_protocol
  cvs_day_rx
  --wisig_domain
  rx_day
  --wisig_equalized
  1
  --wisig_train_ratio
  0.1
  --wisig_val_ratio
  -1.0
  --wisig_split_strategy
  random
  --wisig_cap_strategy
  random
  --wisig_train_days
  0,1
  --wisig_test_days
  2,3
  --wisig_train_rxs
  0,1,2,3,4,5,6
  --wisig_test_rxs
  7,8,9,10,11
  --test_eval_policy
  interval_final
  --test_eval_start_epoch
  31
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
  --model_variant
  lite_d
  --branch_ablation
  no_dac
  --domain_branch_ablation
  no_stats
  --domain_enhancer
  rcn_stats
  --domain_enhancer_strength
  0.35
  --id_time_stability_mode
  off
  --id_freq_stability_mode
  off
  --domain_time_stability_mode
  off
  --domain_freq_stability_mode
  off
  --exp_group
  s3_rxrobust_no_dac
  --pa_orders
  1,3,5
  --collapse_guard
  --collapse_guard_min_epoch
  35
  --collapse_guard_best_margin
  10.0
  --collapse_guard_max_skipped_delta
  2
  --use_ema_ckpt
  --ema_decay
  0.999
  --use_swad_ckpt
  --swad_interval
  1
  --swad_tolerance
  0.85
  --primary_udu_weight
  0.82
  --label_smoothing
  0.0
)

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi
cd "${ROOT}"
snapshot_capacity
echo "[CEN51-FDLACSRFD] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
echo "[CEN51-FDLACSRFD] initial_gpu_counts: ${INITIAL_BY_GPU[*]}"

run_candidate FS005_SATMIN_HINGE_1337 CEN51_FDLACSRFD_FS005_SATMIN_HINGE_1337 5 0 \
  --batch_size \
  128 \
  --epochs \
  195 \
  --swad_start_epoch \
  55 \
  --seed \
  1337 \
  --sat_view_seed \
  9256 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.080 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.080:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.36 \
  --lambda_adv \
  0.14 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.004 \
  --lambda_group_ce \
  0.004 \
  --lambda_proto \
  0.0005 \
  --lambda_supcon_id \
  0.0005 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00075 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.160 \
  --groupdro_tau \
  0.350 \
  --groupdro_cap \
  0.420 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS005_SATMIN_HINGE_2029 CEN51_FDLACSRFD_FS005_SATMIN_HINGE_2029 5 1 \
  --batch_size \
  128 \
  --epochs \
  195 \
  --swad_start_epoch \
  55 \
  --seed \
  2029 \
  --sat_view_seed \
  9948 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.080 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.080:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.36 \
  --lambda_adv \
  0.14 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.004 \
  --lambda_group_ce \
  0.004 \
  --lambda_proto \
  0.0005 \
  --lambda_supcon_id \
  0.0005 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00075 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.160 \
  --groupdro_tau \
  0.350 \
  --groupdro_cap \
  0.420 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS005_RXGUARD_LATE_2028 CEN51_FDLACSRFD_FS005_RXGUARD_LATE_2028 5 2 \
  --batch_size \
  128 \
  --epochs \
  195 \
  --swad_start_epoch \
  60 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.100 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.100:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  55 \
  --lambda_dom \
  0.5 \
  --lambda_adv \
  0.16 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.02 \
  --lambda_cons \
  0.006 \
  --lambda_group_ce \
  0.006 \
  --lambda_proto \
  0.0005 \
  --lambda_supcon_id \
  0.0005 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00012 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.140 \
  --groupdro_tau \
  0.320 \
  --groupdro_cap \
  0.400 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS005_SATMIN_HINGE_2028 CEN51_FDLACSRFD_FS005_SATMIN_HINGE_2028 5 3 \
  --batch_size \
  128 \
  --epochs \
  195 \
  --swad_start_epoch \
  55 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.080 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.080:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.36 \
  --lambda_adv \
  0.14 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.01 \
  --lambda_cons \
  0.004 \
  --lambda_group_ce \
  0.004 \
  --lambda_proto \
  0.0005 \
  --lambda_supcon_id \
  0.0005 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00075 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.160 \
  --groupdro_tau \
  0.350 \
  --groupdro_cap \
  0.420 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS010_RXGUARD_HINGE_2028 CEN51_FDLACSRFD_FS010_RXGUARD_HINGE_2028 10 4 \
  --batch_size \
  128 \
  --epochs \
  200 \
  --swad_start_epoch \
  60 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.120 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.120:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.52 \
  --lambda_adv \
  0.16 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.025 \
  --lambda_cons \
  0.006 \
  --lambda_group_ce \
  0.01 \
  --lambda_proto \
  0.001 \
  --lambda_supcon_id \
  0.001 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.0005 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.5 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.140 \
  --groupdro_tau \
  0.320 \
  --groupdro_cap \
  0.400 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS010_RXGUARD_HINGE_2029 CEN51_FDLACSRFD_FS010_RXGUARD_HINGE_2029 10 5 \
  --batch_size \
  128 \
  --epochs \
  200 \
  --swad_start_epoch \
  60 \
  --seed \
  2029 \
  --sat_view_seed \
  9948 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.120 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.120:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.52 \
  --lambda_adv \
  0.16 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.025 \
  --lambda_cons \
  0.006 \
  --lambda_group_ce \
  0.01 \
  --lambda_proto \
  0.001 \
  --lambda_supcon_id \
  0.001 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.0005 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.5 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.140 \
  --groupdro_tau \
  0.320 \
  --groupdro_cap \
  0.400 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS010_LACSRFD_BAL_1337 CEN51_FDLACSRFD_FS010_LACSRFD_BAL_1337 10 6 \
  --batch_size \
  128 \
  --epochs \
  200 \
  --swad_start_epoch \
  65 \
  --seed \
  1337 \
  --sat_view_seed \
  9256 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.140 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.140:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  35 \
  --lambda_dom \
  0.44 \
  --lambda_adv \
  0.18 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.02 \
  --lambda_cons \
  0.008 \
  --lambda_group_ce \
  0.014 \
  --lambda_proto \
  0.0015 \
  --lambda_supcon_id \
  0.0015 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00011 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.180 \
  --groupdro_tau \
  0.360 \
  --groupdro_cap \
  0.450 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.020 \
  --aug_scale_max \
  0.120 \
  --late_aug_min_scale \
  0.120 \
  --no_use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.025 \
  --mixstyle_strength \
  0.120 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  85 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.020 \
  --mixstyle_late_min_strength \
  0.100 \
  --use_proto_memory

run_candidate FS010_IDFIRST_LATE_2030 CEN51_FDLACSRFD_FS010_IDFIRST_LATE_2030 10 7 \
  --batch_size \
  128 \
  --epochs \
  195 \
  --swad_start_epoch \
  65 \
  --seed \
  2030 \
  --sat_view_seed \
  9949 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.100 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.100:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  70 \
  --lambda_dom \
  0.42 \
  --lambda_adv \
  0.12 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.015 \
  --lambda_cons \
  0.004 \
  --lambda_group_ce \
  0.006 \
  --lambda_proto \
  0.0008 \
  --lambda_supcon_id \
  0.0008 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  0.00045 \
  --feature_norm_guard_mode \
  hinge \
  --feature_norm_guard_target \
  8.5 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.160 \
  --groupdro_tau \
  0.340 \
  --groupdro_cap \
  0.420 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.000 \
  --aug_scale_max \
  0.000 \
  --late_aug_min_scale \
  0.000 \
  --no_use_aug \
  --no_use_mixstyle \
  --use_proto_memory

run_candidate FS020_LACSRFD_RELAX_2028 CEN51_FDLACSRFD_FS020_LACSRFD_RELAX_2028 20 0 \
  --batch_size \
  256 \
  --epochs \
  205 \
  --swad_start_epoch \
  70 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.160 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  35 \
  --lambda_dom \
  0.45 \
  --lambda_adv \
  0.2 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.02 \
  --lambda_cons \
  0.01 \
  --lambda_group_ce \
  0.014 \
  --lambda_proto \
  0.002 \
  --lambda_supcon_id \
  0.002 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  8e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.180 \
  --groupdro_tau \
  0.360 \
  --groupdro_cap \
  0.480 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.030 \
  --aug_scale_max \
  0.180 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.040 \
  --mixstyle_strength \
  0.160 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  90 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.020 \
  --mixstyle_late_min_strength \
  0.104 \
  --use_proto_memory

run_candidate FS020_RIEIFD_FDGATE_2029 CEN51_FDLACSRFD_FS020_RIEIFD_FDGATE_2029 20 1 \
  --batch_size \
  256 \
  --epochs \
  205 \
  --swad_start_epoch \
  70 \
  --seed \
  2029 \
  --sat_view_seed \
  9948 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.140 \
  --sat_train_scenarios \
  clear_leo,mixed_orbit \
  --sat_view_schedule \
  1@0.140:clear_leo,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.46 \
  --lambda_adv \
  0.18 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.02 \
  --lambda_cons \
  0.008 \
  --lambda_group_ce \
  0.012 \
  --lambda_proto \
  0.0015 \
  --lambda_supcon_id \
  0.0015 \
  --lambda_fishr \
  0 \
  --lambda_feature_norm_guard \
  7e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.180 \
  --groupdro_tau \
  0.350 \
  --groupdro_cap \
  0.460 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.030 \
  --aug_scale_max \
  0.160 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.035 \
  --mixstyle_strength \
  0.140 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  90 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.020 \
  --mixstyle_late_min_strength \
  0.100 \
  --use_proto_memory

run_candidate FS020_LATE_REPAIR_1337 CEN51_FDLACSRFD_FS020_LATE_REPAIR_1337 20 2 \
  --batch_size \
  256 \
  --epochs \
  205 \
  --swad_start_epoch \
  70 \
  --seed \
  1337 \
  --sat_view_seed \
  9256 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.200 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.200:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  75 \
  --lambda_dom \
  0.48 \
  --lambda_adv \
  0.22 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.025 \
  --lambda_cons \
  0.01 \
  --lambda_group_ce \
  0.018 \
  --lambda_proto \
  0.0025 \
  --lambda_supcon_id \
  0.0025 \
  --lambda_fishr \
  0.0002 \
  --lambda_feature_norm_guard \
  6e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.200 \
  --groupdro_tau \
  0.380 \
  --groupdro_cap \
  0.500 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.030 \
  --aug_scale_max \
  0.180 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.045 \
  --mixstyle_strength \
  0.160 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  90 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.022 \
  --mixstyle_late_min_strength \
  0.104 \
  --use_proto_memory

run_candidate FS030_LACSRFD_BAL_2028 CEN51_FDLACSRFD_FS030_LACSRFD_BAL_2028 30 3 \
  --batch_size \
  256 \
  --epochs \
  210 \
  --swad_start_epoch \
  75 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.220 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.220:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  35 \
  --lambda_dom \
  0.5 \
  --lambda_adv \
  0.24 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.025 \
  --lambda_cons \
  0.012 \
  --lambda_group_ce \
  0.024 \
  --lambda_proto \
  0.0035 \
  --lambda_supcon_id \
  0.0035 \
  --lambda_fishr \
  0.0003 \
  --lambda_feature_norm_guard \
  5e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.220 \
  --groupdro_tau \
  0.400 \
  --groupdro_cap \
  0.550 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.040 \
  --aug_scale_max \
  0.220 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.060 \
  --mixstyle_strength \
  0.200 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  95 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.030 \
  --mixstyle_late_min_strength \
  0.130 \
  --use_proto_memory

run_candidate FS030_RXFLOOR_2029 CEN51_FDLACSRFD_FS030_RXFLOOR_2029 30 4 \
  --batch_size \
  256 \
  --epochs \
  210 \
  --swad_start_epoch \
  75 \
  --seed \
  2029 \
  --sat_view_seed \
  9948 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.200 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.200:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  1 \
  --lambda_dom \
  0.56 \
  --lambda_adv \
  0.22 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.03 \
  --lambda_cons \
  0.01 \
  --lambda_group_ce \
  0.022 \
  --lambda_proto \
  0.003 \
  --lambda_supcon_id \
  0.003 \
  --lambda_fishr \
  0.0003 \
  --lambda_feature_norm_guard \
  5e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.180 \
  --groupdro_tau \
  0.360 \
  --groupdro_cap \
  0.480 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.040 \
  --aug_scale_max \
  0.200 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.050 \
  --mixstyle_strength \
  0.180 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  95 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.025 \
  --mixstyle_late_min_strength \
  0.117 \
  --use_proto_memory

run_candidate FS050_LACSRFD_MONO_2028 CEN51_FDLACSRFD_FS050_LACSRFD_MONO_2028 50 5 \
  --batch_size \
  256 \
  --epochs \
  215 \
  --swad_start_epoch \
  80 \
  --seed \
  2028 \
  --sat_view_seed \
  9947 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.260 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.260:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  45 \
  --lambda_dom \
  0.54 \
  --lambda_adv \
  0.28 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.03 \
  --lambda_cons \
  0.014 \
  --lambda_group_ce \
  0.032 \
  --lambda_proto \
  0.005 \
  --lambda_supcon_id \
  0.005 \
  --lambda_fishr \
  0.0005 \
  --lambda_feature_norm_guard \
  3.5e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.240 \
  --groupdro_tau \
  0.440 \
  --groupdro_cap \
  0.580 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.050 \
  --aug_scale_max \
  0.240 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.080 \
  --mixstyle_strength \
  0.220 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  100 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.040 \
  --mixstyle_late_min_strength \
  0.143 \
  --use_proto_memory

run_candidate FS050_LACSRFD_MONO_2029 CEN51_FDLACSRFD_FS050_LACSRFD_MONO_2029 50 6 \
  --batch_size \
  256 \
  --epochs \
  215 \
  --swad_start_epoch \
  80 \
  --seed \
  2029 \
  --sat_view_seed \
  9948 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.260 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.260:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  45 \
  --lambda_dom \
  0.54 \
  --lambda_adv \
  0.28 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.03 \
  --lambda_cons \
  0.014 \
  --lambda_group_ce \
  0.032 \
  --lambda_proto \
  0.005 \
  --lambda_supcon_id \
  0.005 \
  --lambda_fishr \
  0.0005 \
  --lambda_feature_norm_guard \
  3.5e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.240 \
  --groupdro_tau \
  0.440 \
  --groupdro_cap \
  0.580 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.050 \
  --aug_scale_max \
  0.240 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.080 \
  --mixstyle_strength \
  0.220 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  100 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.040 \
  --mixstyle_late_min_strength \
  0.143 \
  --use_proto_memory

run_candidate FS050_CAP_RELAX_1337 CEN51_FDLACSRFD_FS050_CAP_RELAX_1337 50 7 \
  --batch_size \
  256 \
  --epochs \
  215 \
  --swad_start_epoch \
  80 \
  --seed \
  1337 \
  --sat_view_seed \
  9256 \
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
  --use_concat_sat_channel_aug \
  --no_use_sat_consistency \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --sat_view_prob \
  0.220 \
  --sat_train_scenarios \
  clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule \
  1@0.220:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --concat_sat_start_epoch \
  25 \
  --lambda_dom \
  0.52 \
  --lambda_adv \
  0.24 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.025 \
  --lambda_cons \
  0.012 \
  --lambda_group_ce \
  0.026 \
  --lambda_proto \
  0.004 \
  --lambda_supcon_id \
  0.004 \
  --lambda_fishr \
  0.0003 \
  --lambda_feature_norm_guard \
  3e-05 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0 \
  --group_ce_mode \
  smooth_dro_capped \
  --group_ce_min_domains \
  2 \
  --group_ce_top_frac \
  0.220 \
  --groupdro_tau \
  0.400 \
  --groupdro_cap \
  0.550 \
  --fishr_min_domains \
  2 \
  --aug_scale_min \
  0.050 \
  --aug_scale_max \
  0.220 \
  --late_aug_min_scale \
  0.160 \
  --use_aug \
  --use_mixstyle \
  --mixstyle_p \
  0.070 \
  --mixstyle_strength \
  0.200 \
  --mixstyle_mix \
  same_tx_crossdomain \
  --mixstyle_fallback \
  skip \
  --mixstyle_late_start \
  100 \
  --mixstyle_late_ramp_epochs \
  35 \
  --mixstyle_late_min_p \
  0.035 \
  --mixstyle_late_min_strength \
  0.130 \
  --use_proto_memory

