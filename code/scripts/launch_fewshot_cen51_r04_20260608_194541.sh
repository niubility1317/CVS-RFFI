#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-fewshot_cen51_r04_20260608_194541}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-3}"
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
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

should_skip() {
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A LAUNCHED_BY_GPU=()

run_cmd() {
  local candidate_id="$1"
  local run_name="$2"
  local shots="$3"
  local gpu="$4"
  local run_dir="$5"
  local log_path="$6"
  shift 6
  local cmd=("$@")

  if should_skip "${candidate_id}" "${run_name}"; then
    return 0
  fi

  echo "[CEN51-R04-FEWSHOT] candidate=${candidate_id} run=${run_name} shots_per_combo=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
  printf '[CEN51-R04-FEWSHOT-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

COMMON_CEN_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --wisig_split_strategy random
  --wisig_cap_strategy random
  --epochs 200
  --test_eval_policy interval_final
  --test_eval_start_epoch 1
  --test_eval_interval 10
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --arch_family cvsincnet
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_aug
  --use_concat_sat_channel_aug
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
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
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
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
  --lambda_fishr 0.005
  --fishr_min_domains 4
  --generalization_feature z_id
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 12.0
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --swad_start_epoch 90
  --swad_tolerance 0.8
)

R04_ARGS=(
  --primary_udu_weight 0.84
  --concat_sat_ce_weight 1.19
  --pa_orders 1,3,5
  --lambda_group_ce 0.088
  --group_ce_min_domains 4
  --group_ce_top_frac 0.20
  --groupdro_tau 0.37
  --groupdro_cap 0.48
  --lambda_proto 0.016
  --proto_momentum 0.970
  --lambda_supcon_id 0.022
  --lambda_fishr 0.002
  --fishr_min_domains 4
  --use_sat_consistency
  --lambda_sat_cons 0.006
  --sat_cons_start_epoch 118
  --swad_start_epoch 70
  --swad_tolerance 0.34
  --sat_view_schedule "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
)

launch_cen_train() {
  local candidate_id="$1" run_name="$2" shots="$3" gpu="$4"
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_CEN_ARGS[@]}"
    --run_name "${run_name}"
    --wisig_max_train_per_combo "${shots}"
    "${R04_ARGS[@]}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )
  run_cmd "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi

cd "${ROOT}"
echo "[CEN51-R04-FEWSHOT] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_cen_train CEN51_R04_FS005 CEN51_R04_FS005_sat_joint_guard_no_overdrive_r010 5 0
launch_cen_train CEN51_R04_FS010 CEN51_R04_FS010_sat_joint_guard_no_overdrive_r010 10 1
launch_cen_train CEN51_R04_FS020 CEN51_R04_FS020_sat_joint_guard_no_overdrive_r010 20 2
launch_cen_train CEN51_R04_FS030 CEN51_R04_FS030_sat_joint_guard_no_overdrive_r010 30 3
launch_cen_train CEN51_R04_FS050 CEN51_R04_FS050_sat_joint_guard_no_overdrive_r010 50 4
