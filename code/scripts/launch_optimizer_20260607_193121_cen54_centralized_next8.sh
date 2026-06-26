#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260607_193121_cen54_centralized_next8}"
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

  echo "[CEN54-CANDIDATE] lane=centralized candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CEN54-CMD]'
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
echo "[CEN54] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_cen_train CEN54_R01 CEN54_R01_a05_risk_hold_satnudge_r010 0 \
  --primary_udu_weight 0.90 --concat_sat_ce_weight 1.17 --pa_orders 1,3,5 \
  --group_ce_mode dual_worst --lambda_group_ce 0.078 --group_ce_min_domains 4 --group_ce_top_frac 0.15 --groupdro_tau 0.29 --groupdro_cap 0.38 \
  --lambda_proto 0.032 --proto_momentum 0.980 --lambda_supcon_id 0.044 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --mixstyle_late_min_p 0.003 --mixstyle_late_min_strength 0.05 --use_sat_consistency --lambda_sat_cons 0.003 \
  --sat_cons_start_epoch 132 --swad_start_epoch 54 --swad_tolerance 0.26 --collapse_guard_best_margin 5.8 \
  --sat_view_schedule "1@0.96:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;132@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN54_R02 CEN54_R02_a05_swad_receiver_floor_r010 1 \
  --primary_udu_weight 0.88 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 \
  --lambda_group_ce 0.084 --group_ce_min_domains 4 --group_ce_top_frac 0.16 --groupdro_tau 0.30 --groupdro_cap 0.40 \
  --generalization_feature feat_joint --lambda_proto 0.028 --proto_momentum 0.981 --lambda_supcon_id 0.038 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --ema_decay 0.99940 --use_sat_consistency --lambda_sat_cons 0.006 --sat_cons_start_epoch 122 \
  --swad_start_epoch 60 --swad_tolerance 0.29 --collapse_guard_best_margin 6.0

launch_cen_train CEN54_R03 CEN54_R03_a08_clean_floor_guard_r010 2 \
  --primary_udu_weight 0.91 --concat_sat_ce_weight 1.16 --concat_sat_start_epoch 20 --pa_orders 1,3,5 \
  --lambda_group_ce 0.080 --group_ce_min_domains 4 --group_ce_top_frac 0.17 --groupdro_tau 0.32 --groupdro_cap 0.42 \
  --lambda_proto 0.024 --proto_momentum 0.978 --lambda_supcon_id 0.035 --lambda_fishr 0.0005 --fishr_min_domains 4 \
  --ema_decay 0.99925 --use_sat_consistency --lambda_sat_cons 0.004 --sat_cons_start_epoch 128 \
  --swad_start_epoch 62 --swad_tolerance 0.30 --collapse_guard_best_margin 6.0 \
  --sat_view_schedule "1@0.94:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;125@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN54_R04 CEN54_R04_r02_satmean_stabilizer_r010 3 \
  --primary_udu_weight 0.87 --concat_sat_ce_weight 1.20 --freq_feature_source sinc_energy --pa_orders 1,3 \
  --lambda_group_ce 0.090 --group_ce_min_domains 4 --group_ce_top_frac 0.17 --groupdro_tau 0.31 --groupdro_cap 0.40 \
  --lambda_proto 0.022 --proto_momentum 0.978 --lambda_supcon_id 0.030 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --ema_decay 0.99935 --use_sat_consistency --lambda_sat_cons 0.006 --sat_cons_start_epoch 118 \
  --swad_start_epoch 64 --swad_tolerance 0.30 --collapse_guard_best_margin 6.2 \
  --sat_view_schedule "1@0.96:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;135@0.84:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN54_A05 CEN54_A05_a05_clean_ceiling_guard_r010 4 \
  --primary_udu_weight 0.92 --concat_sat_ce_weight 1.13 --freq_feature_source sinc_phase_asym --pa_feature_source sinc_lowrank --pa_orders 1,3,5 \
  --lambda_group_ce 0.070 --group_ce_min_domains 4 --group_ce_top_frac 0.14 --groupdro_tau 0.28 --groupdro_cap 0.37 \
  --lambda_proto 0.036 --proto_momentum 0.982 --lambda_supcon_id 0.050 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.04 --use_sat_consistency --lambda_sat_cons 0.001 \
  --sat_cons_start_epoch 140 --swad_start_epoch 52 --swad_tolerance 0.25 --collapse_guard_best_margin 5.5

launch_cen_train CEN54_A06 CEN54_A06_litee_storm_floor_pressure_r010 5 \
  --model_variant lite_e --channel_trim_scale 0.92 --primary_udu_weight 0.84 --concat_sat_ce_weight 1.24 --pa_orders 1,3,5 \
  --lambda_group_ce 0.095 --group_ce_min_domains 4 --group_ce_top_frac 0.20 --groupdro_tau 0.34 --groupdro_cap 0.45 \
  --lambda_proto 0.020 --proto_momentum 0.974 --lambda_supcon_id 0.024 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.008 --sat_cons_start_epoch 110 \
  --swad_start_epoch 72 --swad_tolerance 0.34 --collapse_guard_best_margin 7.0 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"

launch_cen_train CEN54_A07 CEN54_A07_phase_idfreq_soft_retry_r010 6 \
  --domain_enhancer rcn_minimal_6stats --domain_enhancer_strength 0.20 --primary_udu_weight 0.84 --concat_sat_ce_weight 1.12 --pa_orders 1,3,5 \
  --id_freq_stability_mode dsq --id_time_stability_mode phase_delta --domain_time_stability_mode off --domain_freq_stability_mode off --freq_stability_channels 1 \
  --lambda_group_ce 0.072 --group_ce_min_domains 4 --group_ce_top_frac 0.18 --groupdro_tau 0.34 --groupdro_cap 0.44 \
  --lambda_proto 0.018 --proto_momentum 0.970 --lambda_supcon_id 0.020 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --use_sat_consistency --lambda_sat_cons 0.003 --sat_cons_start_epoch 132 --swad_start_epoch 78 --swad_tolerance 0.36 \
  --collapse_guard_best_margin 7.0

launch_cen_train CEN54_A08 CEN54_A08_rxchain_awgn_floor_probe_r010 7 \
  --primary_udu_weight 0.86 --concat_sat_ce_weight 1.17 --pa_orders 1,3,5 \
  --aug_p_rx_chain 0.018 --aug_p_awgn 0.12 --aug_snr_min_db 28 --aug_snr_max_db 46 --aug_p_multipath 0.035 \
  --lambda_group_ce 0.076 --group_ce_min_domains 4 --group_ce_top_frac 0.18 --groupdro_tau 0.34 --groupdro_cap 0.43 \
  --lambda_proto 0.018 --proto_momentum 0.972 --lambda_supcon_id 0.022 --lambda_fishr 0.000 --fishr_min_domains 4 \
  --sat_train_scenarios geo_clear,storm_mp,mixed_orbit,rain_leo,clear_leo,low_elev_leo \
  --use_sat_consistency --lambda_sat_cons 0.004 --sat_cons_start_epoch 132 --swad_start_epoch 80 --swad_tolerance 0.35 \
  --collapse_guard_best_margin 6.8
