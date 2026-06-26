#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_lac_sat_rescue_20260609_022600}"
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
  echo "[CEN51-LOWSHOT-SEARCH] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
  printf '[CEN51-LOWSHOT-CMD]'
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
  --use_aug
  --use_concat_sat_channel_aug
  --concat_sat_start_epoch
  1
  --lambda_sat_cls
  0.0
  --seed
  1337
  --use_mixstyle
  --mixstyle_layers
  time_down,t1
  --mixstyle_mix
  same_tx_crossdomain
  --mixstyle_fallback
  skip
  --mixstyle_late_ramp_epochs
  40
  --domain_freq_stability_mode
  dsq
  --freq_stability_channels
  2
  --group_ce_mode
  smooth_dro_capped
  --group_ce_min_domains
  4
  --use_proto_memory
  --proto_momentum
  0.97
  --supcon_temp
  0.12
  --fishr_min_domains
  4
  --generalization_feature
  z_id
  --pa_orders
  1,3,5
  --collapse_guard
  --collapse_guard_min_epoch
  35
  --collapse_guard_best_margin
  10
  --collapse_guard_max_skipped_delta
  2
  --use_ema_ckpt
  --ema_decay
  0.999
  --use_swad_ckpt
)

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi
cd "${ROOT}"
echo "[CEN51-LOWSHOT-SEARCH] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

run_candidate CEN51_LACSR01_FS005_sat_rescue CEN51_LACSR01_FS005_sat_rescue_r010 5 4 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.45 \
  --lambda_cons \
  0.08 \
  --lambda_group_ce \
  0.1 \
  --group_ce_top_frac \
  0.35 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.65 \
  --late_adv_min_scale \
  0.7 \
  --late_cons_min_scale \
  0.45 \
  --late_group_ce_min_scale \
  0.8 \
  --late_aug_min_scale \
  0.35 \
  --primary_udu_weight \
  0.86 \
  --batch_size \
  128 \
  --epochs \
  180 \
  --concat_sat_ce_weight \
  0.6 \
  --sat_view_prob \
  0.55 \
  --sat_view_schedule \
  '1@0.30:clear_leo,mixed_orbit;120@0.55:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;150@0.65:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.001 \
  --sat_cons_start_epoch \
  140 \
  --lambda_proto \
  0.004 \
  --lambda_supcon_id \
  0.004 \
  --lambda_fishr \
  0.0 \
  --mixstyle_p \
  0.1 \
  --mixstyle_strength \
  0.38 \
  --mixstyle_late_start \
  80 \
  --mixstyle_late_min_p \
  0.02 \
  --mixstyle_late_min_strength \
  0.18 \
  --late_stable_start \
  80 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  50 \
  --swad_tolerance \
  0.7

run_candidate CEN51_LACSR02_FS005_lowreg_control CEN51_LACSR02_FS005_lowreg_control_r010 5 5 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.16 \
  --lambda_cons \
  0.03 \
  --lambda_group_ce \
  0.025 \
  --group_ce_top_frac \
  0.2 \
  --groupdro_tau \
  0.3 \
  --groupdro_cap \
  0.42 \
  --late_adv_min_scale \
  0.35 \
  --late_cons_min_scale \
  0.2 \
  --late_group_ce_min_scale \
  0.35 \
  --late_aug_min_scale \
  0.45 \
  --primary_udu_weight \
  0.84 \
  --batch_size \
  128 \
  --epochs \
  170 \
  --concat_sat_ce_weight \
  0.35 \
  --sat_view_prob \
  0.35 \
  --sat_view_schedule \
  '1@0.30:clear_leo,mixed_orbit;130@0.35:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.0 \
  --sat_cons_start_epoch \
  999 \
  --lambda_proto \
  0.004 \
  --lambda_supcon_id \
  0.004 \
  --lambda_fishr \
  0.0 \
  --mixstyle_p \
  0.1 \
  --mixstyle_strength \
  0.38 \
  --mixstyle_late_start \
  80 \
  --mixstyle_late_min_p \
  0.02 \
  --mixstyle_late_min_strength \
  0.18 \
  --late_stable_start \
  80 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  50 \
  --swad_tolerance \
  0.8

run_candidate CEN51_LACSR03_FS010_sat_rescue CEN51_LACSR03_FS010_sat_rescue_r010 10 6 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.45 \
  --lambda_cons \
  0.08 \
  --lambda_group_ce \
  0.1 \
  --group_ce_top_frac \
  0.35 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.65 \
  --late_adv_min_scale \
  0.7 \
  --late_cons_min_scale \
  0.45 \
  --late_group_ce_min_scale \
  0.8 \
  --late_aug_min_scale \
  0.35 \
  --primary_udu_weight \
  0.86 \
  --batch_size \
  128 \
  --epochs \
  185 \
  --concat_sat_ce_weight \
  0.7 \
  --sat_view_prob \
  0.62 \
  --sat_view_schedule \
  '1@0.35:clear_leo,mixed_orbit;120@0.60:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;150@0.75:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.0015 \
  --sat_cons_start_epoch \
  140 \
  --lambda_proto \
  0.006 \
  --lambda_supcon_id \
  0.006 \
  --lambda_fishr \
  0.0 \
  --mixstyle_p \
  0.12 \
  --mixstyle_strength \
  0.42 \
  --mixstyle_late_start \
  85 \
  --mixstyle_late_min_p \
  0.025 \
  --mixstyle_late_min_strength \
  0.2 \
  --late_stable_start \
  85 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  60 \
  --swad_tolerance \
  0.75

run_candidate CEN51_LACSR04_FS020_sat_rescue CEN51_LACSR04_FS020_sat_rescue_r010 20 7 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.45 \
  --lambda_cons \
  0.08 \
  --lambda_group_ce \
  0.1 \
  --group_ce_top_frac \
  0.35 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.65 \
  --late_adv_min_scale \
  0.7 \
  --late_cons_min_scale \
  0.45 \
  --late_group_ce_min_scale \
  0.8 \
  --late_aug_min_scale \
  0.35 \
  --primary_udu_weight \
  0.86 \
  --batch_size \
  256 \
  --epochs \
  200 \
  --concat_sat_ce_weight \
  0.82 \
  --sat_view_prob \
  0.7 \
  --sat_view_schedule \
  '1@0.45:clear_leo,mixed_orbit;115@0.70:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;145@0.86:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.0025 \
  --sat_cons_start_epoch \
  135 \
  --lambda_proto \
  0.008 \
  --lambda_supcon_id \
  0.01 \
  --lambda_fishr \
  0.0005 \
  --mixstyle_p \
  0.13 \
  --mixstyle_strength \
  0.5 \
  --mixstyle_late_start \
  100 \
  --mixstyle_late_min_p \
  0.03 \
  --mixstyle_late_min_strength \
  0.24 \
  --late_stable_start \
  100 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  70 \
  --swad_tolerance \
  0.75

run_candidate CEN51_LACSR05_FS050_monotonic_anchor CEN51_LACSR05_FS050_monotonic_anchor_r010 50 0 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.45 \
  --lambda_cons \
  0.08 \
  --lambda_group_ce \
  0.1 \
  --group_ce_top_frac \
  0.35 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.65 \
  --late_adv_min_scale \
  0.7 \
  --late_cons_min_scale \
  0.45 \
  --late_group_ce_min_scale \
  0.8 \
  --late_aug_min_scale \
  0.35 \
  --primary_udu_weight \
  0.86 \
  --batch_size \
  256 \
  --epochs \
  200 \
  --concat_sat_ce_weight \
  0.95 \
  --sat_view_prob \
  0.78 \
  --sat_view_schedule \
  '1@0.55:clear_leo,mixed_orbit;105@0.78:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;135@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.004 \
  --sat_cons_start_epoch \
  130 \
  --lambda_proto \
  0.014 \
  --lambda_supcon_id \
  0.018 \
  --lambda_fishr \
  0.0015 \
  --mixstyle_p \
  0.17 \
  --mixstyle_strength \
  0.62 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_min_p \
  0.045 \
  --mixstyle_late_min_strength \
  0.3 \
  --late_stable_start \
  110 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  80 \
  --swad_tolerance \
  0.65

run_candidate CEN51_LACSR06_FS100_r010_anchor CEN51_LACSR06_FS100_r010_anchor_r010 100 1 \
  --test_eval_start_epoch \
  81 \
  --test_eval_interval \
  10 \
  --seed \
  1337 \
  --collapse_guard_min_epoch \
  35 \
  --collapse_guard_best_margin \
  10 \
  --collapse_guard_max_skipped_delta \
  2 \
  --lambda_adv \
  0.45 \
  --lambda_cons \
  0.08 \
  --lambda_group_ce \
  0.1 \
  --group_ce_top_frac \
  0.35 \
  --groupdro_tau \
  0.5 \
  --groupdro_cap \
  0.65 \
  --late_adv_min_scale \
  0.7 \
  --late_cons_min_scale \
  0.45 \
  --late_group_ce_min_scale \
  0.8 \
  --late_aug_min_scale \
  0.35 \
  --primary_udu_weight \
  0.86 \
  --batch_size \
  256 \
  --epochs \
  200 \
  --concat_sat_ce_weight \
  1.05 \
  --sat_view_prob \
  0.82 \
  --sat_view_schedule \
  '1@0.65:clear_leo,mixed_orbit;100@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;130@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --use_sat_consistency \
  --lambda_sat_cons \
  0.005 \
  --sat_cons_start_epoch \
  125 \
  --lambda_proto \
  0.016 \
  --lambda_supcon_id \
  0.02 \
  --lambda_fishr \
  0.002 \
  --mixstyle_p \
  0.18 \
  --mixstyle_strength \
  0.65 \
  --mixstyle_late_start \
  110 \
  --mixstyle_late_min_p \
  0.05 \
  --mixstyle_late_min_strength \
  0.32 \
  --late_stable_start \
  110 \
  --late_stable_ramp_epochs \
  25 \
  --swad_start_epoch \
  80 \
  --swad_tolerance \
  0.6
