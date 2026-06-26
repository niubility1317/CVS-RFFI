#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260608_001150_cen56_centralized_next8}"
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
  local candidate_id="$1" run_name="$2" gpu="$3" run_dir="$4" log_path="$5"
  shift 5
  local cmd=("$@")

  if should_skip "${candidate_id}" "${run_name}"; then
    return 0
  fi

  echo "[CEN56-CANDIDATE] lane=centralized candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CEN56-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "NON_LAUNCH_DIAGNOSTIC_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "DEFERRED_RETRY_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

COMMON_CEN_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
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
  --concat_sat_ce_only
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

launch_cen_train() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  shift 3
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_CEN_ARGS[@]}"
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
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi

cd "${ROOT}"
echo "[CEN56] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_cen_train CEN56_R01 CEN56_R01_a05_anchor_satcons_floor_probe_r010 0 \
  --primary_udu_weight 0.90 --concat_sat_ce_weight 1.19 --pa_orders 1,3,5 \
  --group_ce_mode dual_worst --lambda_group_ce 0.078 --group_ce_min_domains 4 --group_ce_top_frac 0.12 --groupdro_tau 0.26 --groupdro_cap 0.35 \
  --lambda_proto 0.026 --proto_momentum 0.980 --lambda_supcon_id 0.038 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.04 --use_sat_consistency --lambda_sat_cons 0.004 \
  --sat_cons_start_epoch 132 --swad_start_epoch 52 --swad_tolerance 0.24 --collapse_guard_best_margin 5.4 \
  --sat_view_schedule "1@0.96:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;130@0.88:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN56_R02 CEN56_R02_r01_strict84_satfloor_nudge_r010 1 \
  --primary_udu_weight 0.91 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 \
  --group_ce_mode dual_worst --lambda_group_ce 0.074 --group_ce_min_domains 4 --group_ce_top_frac 0.11 --groupdro_tau 0.25 --groupdro_cap 0.34 \
  --lambda_proto 0.025 --proto_momentum 0.980 --lambda_supcon_id 0.036 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.04 --use_sat_consistency --lambda_sat_cons 0.006 \
  --sat_cons_start_epoch 128 --swad_start_epoch 48 --swad_tolerance 0.22 --collapse_guard_best_margin 5.3 \
  --sat_view_schedule "1@0.96:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;125@0.89:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN56_R03 CEN56_R03_r02_receiverfloor_sat41_hold_r010 2 \
  --primary_udu_weight 0.89 --concat_sat_ce_weight 1.22 --pa_orders 1,3,5 --generalization_feature feat_joint \
  --group_ce_mode dual_worst --lambda_group_ce 0.080 --group_ce_min_domains 4 --group_ce_top_frac 0.13 --groupdro_tau 0.27 --groupdro_cap 0.37 \
  --lambda_proto 0.024 --proto_momentum 0.978 --lambda_supcon_id 0.034 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.007 --sat_cons_start_epoch 118 --swad_start_epoch 58 --swad_tolerance 0.26 \
  --collapse_guard_best_margin 5.8 --sat_view_schedule "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;118@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN56_R04 CEN56_R04_r04_anchor_strict_satbalanced_r010 3 \
  --primary_udu_weight 0.91 --concat_sat_ce_weight 1.17 --pa_orders 1,3,5 \
  --lambda_group_ce 0.070 --group_ce_min_domains 4 --group_ce_top_frac 0.16 --groupdro_tau 0.32 --groupdro_cap 0.42 \
  --lambda_proto 0.024 --proto_momentum 0.976 --lambda_supcon_id 0.030 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --mixstyle_late_min_p 0.004 --mixstyle_late_min_strength 0.08 --use_sat_consistency --lambda_sat_cons 0.004 \
  --sat_cons_start_epoch 136 --swad_start_epoch 62 --swad_tolerance 0.30 --collapse_guard_best_margin 6.0

launch_cen_train CEN56_A05 CEN56_A05_a05_anchor_featjoint_proto_stress_r010 4 \
  --primary_udu_weight 0.88 --concat_sat_ce_weight 1.23 --pa_orders 1,3,5 --generalization_feature feat_joint \
  --group_ce_mode dual_worst --lambda_group_ce 0.090 --group_ce_min_domains 4 --group_ce_top_frac 0.14 --groupdro_tau 0.28 --groupdro_cap 0.38 \
  --lambda_proto 0.034 --proto_momentum 0.982 --lambda_supcon_id 0.046 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.007 --sat_cons_start_epoch 120 --swad_start_epoch 54 --swad_tolerance 0.26 \
  --collapse_guard_best_margin 5.8 --sat_view_schedule "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN56_A06 CEN56_A06_a06_satfloor_receiver_rescue_storm_r010 5 \
  --domain_enhancer rcn_stats --domain_enhancer_strength 0.32 --primary_udu_weight 0.87 --concat_sat_ce_weight 1.16 --pa_orders 1,3,5 \
  --group_ce_mode dual_worst --lambda_group_ce 0.076 --group_ce_min_domains 4 --group_ce_top_frac 0.13 --groupdro_tau 0.27 --groupdro_cap 0.37 \
  --lambda_proto 0.020 --proto_momentum 0.974 --lambda_supcon_id 0.026 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.005 --sat_cons_start_epoch 130 --swad_start_epoch 70 --swad_tolerance 0.32 \
  --collapse_guard_best_margin 6.2 --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;120@0.90:storm_mp,mixed_orbit,rain_leo,low_elev_leo,clear_leo"

launch_cen_train CEN56_A07 CEN56_A07_a07_rxchain_floor_guarded_phystress_r010 6 \
  --primary_udu_weight 0.88 --concat_sat_ce_weight 1.16 --pa_orders 1,3,5 \
  --aug_p_rx_chain 0.006 --aug_p_awgn 0.05 --aug_snr_min_db 32 --aug_snr_max_db 48 --aug_p_multipath 0.012 \
  --lambda_group_ce 0.074 --group_ce_min_domains 4 --group_ce_top_frac 0.14 --groupdro_tau 0.28 --groupdro_cap 0.38 \
  --lambda_proto 0.018 --proto_momentum 0.972 --lambda_supcon_id 0.022 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.004 --sat_cons_start_epoch 132 --swad_start_epoch 74 --swad_tolerance 0.34 \
  --collapse_guard_best_margin 6.2

launch_cen_train CEN56_A08 CEN56_A08_a08_litee_receiver_guard_diag_r010 7 \
  --model_variant lite_e --channel_trim_scale 0.98 --primary_udu_weight 0.88 --concat_sat_ce_weight 1.14 --pa_orders 1,3,5 \
  --lambda_group_ce 0.070 --group_ce_min_domains 4 --group_ce_top_frac 0.14 --groupdro_tau 0.28 --groupdro_cap 0.38 \
  --lambda_proto 0.016 --proto_momentum 0.972 --lambda_supcon_id 0.020 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.003 --sat_cons_start_epoch 138 --swad_start_epoch 74 --swad_tolerance 0.34 \
  --collapse_guard_best_margin 5.8
