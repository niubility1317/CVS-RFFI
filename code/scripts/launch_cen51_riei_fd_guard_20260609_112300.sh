#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_riei_fd_guard_20260609_112300}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-5}"
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
  echo "[CEN51-RIEI-FD] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
  printf '[CEN51-RIEI-FD-CMD]'
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
  --dataset
  wisig
  --wisig_protocol
  cvs_day_rx
  --wisig_domain
  rx_day
  --wisig_train_ratio
  0.1
  --wisig_split_strategy
  random
  --wisig_cap_strategy
  random
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
  1.0
  --primary_udu_weight
  0.78
  --label_smoothing
  0.0
  --seed
  2028
)

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi
cd "${ROOT}"
echo "[CEN51-RIEI-FD] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

run_candidate FS005_CE_FN_L2 CEN51_RIEIFD_FS005_CE_FN_L2_seed2028 5 0 \
  --batch_size \
  128 \
  --epochs \
  150 \
  --swad_start_epoch \
  45 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.0 \
  --lambda_adv \
  0.0 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.0 \
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
  0.0002 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0

run_candidate FS005_FORCE_GRLLOW_L2 CEN51_RIEIFD_FS005_FORCE_GRLLOW_L2_seed2028 5 1 \
  --batch_size \
  128 \
  --epochs \
  150 \
  --swad_start_epoch \
  45 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.0 \
  --lambda_adv \
  0.12 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.0 \
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
  0.0002 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --force_ce_grl_only

run_candidate FS005_FORCE_GRLLOW_HINGE CEN51_RIEIFD_FS005_FORCE_GRLLOW_HINGE_seed2028 5 2 \
  --batch_size \
  128 \
  --epochs \
  150 \
  --swad_start_epoch \
  45 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.0 \
  --lambda_adv \
  0.12 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.0 \
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
  --force_ce_grl_only

run_candidate FS005_DOMSIDE_GRLLOW_L2 CEN51_RIEIFD_FS005_DOMSIDE_GRLLOW_L2_seed2028 5 3 \
  --batch_size \
  128 \
  --epochs \
  150 \
  --swad_start_epoch \
  45 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
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
  0.0002 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0

run_candidate FS010_CE_FN_L2 CEN51_RIEIFD_FS010_CE_FN_L2_seed2028 10 4 \
  --batch_size \
  128 \
  --epochs \
  160 \
  --swad_start_epoch \
  55 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.0 \
  --lambda_adv \
  0.0 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.0 \
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
  0.00015 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0

run_candidate FS010_FORCE_GRLMID_L2 CEN51_RIEIFD_FS010_FORCE_GRLMID_L2_seed2028 10 5 \
  --batch_size \
  128 \
  --epochs \
  160 \
  --swad_start_epoch \
  55 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.0 \
  --lambda_adv \
  0.18 \
  --grl_lambda \
  1.0 \
  --lambda_orth \
  0.0 \
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
  0.00015 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --force_ce_grl_only

run_candidate FS010_DOMSIDE_GRLMID_L2 CEN51_RIEIFD_FS010_DOMSIDE_GRLMID_L2_seed2028 10 6 \
  --batch_size \
  128 \
  --epochs \
  160 \
  --swad_start_epoch \
  55 \
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
  --no_use_sat_consistency \
  --no_use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.0 \
  --sat_view_prob \
  0.0 \
  --sat_cons_start_epoch \
  999 \
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
  0.00015 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0

run_candidate FS010_GATEDSAT_GRLLOW_L2 CEN51_RIEIFD_FS010_GATEDSAT_GRLLOW_L2_seed2028 10 7 \
  --batch_size \
  128 \
  --epochs \
  160 \
  --swad_start_epoch \
  55 \
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
  --no_use_sat_consistency \
  --use_concat_sat_channel_aug \
  --lambda_sat_cls \
  0.0 \
  --lambda_sat_cons \
  0.0 \
  --concat_sat_ce_weight \
  0.12 \
  --sat_view_prob \
  0.12 \
  --sat_cons_start_epoch \
  999 \
  --lambda_dom \
  0.35 \
  --lambda_adv \
  0.14 \
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
  0.00015 \
  --feature_norm_guard_mode \
  l2 \
  --feature_norm_guard_target \
  0.0 \
  --sat_view_schedule \
  1@0.12:clear_leo,mixed_orbit \
  --sat_train_scenarios \
  clear_leo,mixed_orbit
