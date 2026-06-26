#!/usr/bin/env bash
set -euo pipefail

# Centralized-only optimizer v4 next-16. No federated/VMB launcher or --fl_* flags.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-${ROOT}/code/train_cen31_distill.py}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/optimizer_20260530_043050_centralized_next8/CEN_A31_a22_satboost_ce1p28_stack_r010/best_primary_ood_model.pth}"
RUN_ID="${RUN_ID:-optimizer_20260605_103107_centralized_next16}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --only=*)
      ONLY_CANDIDATE="${arg#--only=}"
      ;;
    *)
      echo "[ERROR] unknown argument: ${arg}" >&2
      exit 2
      ;;
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

COMMON_TRAIN_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs 200
  --test_eval_policy every_epoch
  --test_eval_start_epoch 181
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --seed 1337
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
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
  --generalization_feature z_id
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 12.0
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --swad_tolerance 0.8
)

COMMON_KD_ARGS=(
  --dataset wisig
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --prefetch_factor 2
  --epochs 200
  --eval_interval 10
  --eval_max_batches 0
  --seed 1337
  --teacher_ckpt "${TEACHER_CKPT}"
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --groupdro_momentum 0.95
  --kd_temperature 3.0
  --kd_conf_min 0.60
  --kd_margin_min 0.05
  --kd_require_correct
  --lr 4e-4
  --lr_min 1e-6
  --wd 1e-4
  --label_smoothing 0.01
  --use_sat_view_kd
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_schedule "1:clear_leo,low_elev_leo,rain_leo;120:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
  --sat_view_loss_start_epoch 20
  --sat_view_loss_ramp_epochs 80
  --eval_sat_channel
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --eval_sat_on main
  --sat_eval_max_batches 0
  --best_select_metric clean_sat_joint
  --sat_select_eval_interval 20
  --sat_select_max_batches 0
)

run_cmd() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  local run_dir="$4"
  local log_path="$5"
  shift 5
  local cmd=("$@")

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
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
  local launched_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${launched_pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

launch_train() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_TRAIN_ARGS[@]}"
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

  echo "[CEN31-NEXT16] candidate=${candidate_id} run=${run_name} gpu=${gpu} mode=train dry_run=${DRY_RUN}"
  printf '[CEN31-NEXT16-CMD]'
  print_cmd "${cmd[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

launch_kd() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${DISTILL_SCRIPT}"
    "${COMMON_KD_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --latest_save_path "${run_dir}/latest_student.pth"
    --best_save_path "${run_dir}/best_student_primary.pth"
    --best_balanced_save_path "${run_dir}/best_student_balanced.pth"
    --latency_profile_json "${run_dir}/latency_profile.json"
    "$@"
  )

  echo "[CEN31-NEXT16] candidate=${candidate_id} run=${run_name} gpu=${gpu} mode=distill dry_run=${DRY_RUN}"
  printf '[CEN31-NEXT16-CMD]'
  print_cmd "${cmd[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

cd "${ROOT}"

launch_train CEN31_R01 \
  CEN31_R01_c04_all5_swad_satfloor_r010 \
  0 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.76 \
  --concat_sat_ce_weight 1.34 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.02 \
  --supcon_temp 0.12 \
  --lambda_fishr 0.003 \
  --fishr_min_domains 4 \
  --swad_start_epoch 90
sleep 2

launch_train CEN31_R02 \
  CEN31_R02_c04_rainstorm_lateboost_r010 \
  1 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.40 \
  --sat_view_schedule "1@0.80:clear_leo,low_elev_leo,rain_leo;120@1.00:rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --lambda_supcon_id 0.02 \
  --lambda_fishr 0.003 \
  --fishr_min_domains 4 \
  --swad_start_epoch 95
sleep 2

launch_train CEN31_R03 \
  CEN31_R03_c03_min3_groupfishr_all5_r010 \
  2 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.34 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.10 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.40 \
  --group_ce_min_domains 3 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.012 \
  --lambda_supcon_id 0.018 \
  --lambda_fishr 0.003 \
  --fishr_min_domains 3 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_R04 \
  CEN31_R04_c01_lowreg_all5_guard_r010 \
  3 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.32 \
  --sat_view_schedule "1@0.95:clear_leo,low_elev_leo,rain_leo;130@0.85:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --lambda_supcon_id 0.02 \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_R05 \
  CEN31_R05_cleanfirst_no_fishr_all5_r010 \
  4 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.80 \
  --concat_sat_ce_weight 1.20 \
  --sat_view_schedule "1@0.85:clear_leo,low_elev_leo,rain_leo;130@0.65:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.30 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.012 \
  --lambda_supcon_id 0.018 \
  --lambda_fishr 0.000 \
  --fishr_min_domains 4 \
  --swad_start_epoch 95
sleep 2

launch_train CEN31_R06 \
  CEN31_R06_satfloor_highsat_lowfishr_r010 \
  5 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.46 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;110@1.00:rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.40 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.012 \
  --lambda_supcon_id 0.018 \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --swad_start_epoch 95
sleep 2

launch_train CEN31_R07 \
  CEN31_R07_mixstop_swad_stability_r010 \
  6 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.35 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.75 \
  --concat_sat_ce_weight 1.28 \
  --sat_view_schedule "1@0.95:clear_leo,low_elev_leo,rain_leo;130@0.80:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --mixstyle_stop_epoch 150 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --lambda_supcon_id 0.02 \
  --lambda_fishr 0.003 \
  --fishr_min_domains 4 \
  --swad_start_epoch 80
sleep 2

launch_train CEN31_R08 \
  CEN31_R08_enh020_lowfishr_all5_r010 \
  7 \
  --arch_family cvsincnet \
  --model_variant lite_d \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.32 \
  --sat_view_schedule "1@0.95:clear_leo,low_elev_leo,rain_leo;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.07 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --lambda_supcon_id 0.02 \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --swad_start_epoch 95
sleep 2

launch_train CEN31_A09 \
  CEN31_A09_sinc_cvcnn_direct_satfloor_r010 \
  0 \
  --arch_family sinc_cvcnn \
  --model_variant lite_h \
  --slim_group none \
  --branch_ablation no_dac,no_pa,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.00 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.34 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 3 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --lambda_fishr 0.000 \
  --fishr_min_domains 3 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_A10 \
  CEN31_A10_cvcnn_direct_compact_guard_r010 \
  1 \
  --arch_family cvcnn \
  --model_variant lite_h \
  --slim_group none \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.00 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.26 \
  --sat_view_schedule "1@0.90:clear_leo,low_elev_leo,rain_leo;130@0.75:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 3 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --lambda_fishr 0.000 \
  --fishr_min_domains 3 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_A11 \
  CEN31_A11_resnet1d_arch_control_satfloor_r010 \
  2 \
  --arch_family resnet18_1d \
  --model_variant lite_h \
  --slim_group none \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.00 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.30 \
  --sat_view_schedule "1@0.90:clear_leo,low_elev_leo,rain_leo;130@0.75:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --lambda_group_ce 0.04 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 3 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --lambda_fishr 0.000 \
  --fishr_min_domains 3 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_A12 \
  CEN31_A12_cvs_liteb_bottleneck_r010 \
  3 \
  --arch_family cvsincnet \
  --model_variant lite_b \
  --slim_group none \
  --branch_ablation no_dac \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.30 \
  --sat_view_schedule "1@0.95:clear_leo,low_elev_leo,rain_leo;120@0.85:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.07 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.012 \
  --lambda_supcon_id 0.018 \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --swad_start_epoch 95
sleep 2

launch_train CEN31_A13 \
  CEN31_A13_cvs_liteg_phys_pruned_r010 \
  4 \
  --arch_family cvsincnet \
  --model_variant lite_g \
  --slim_group none \
  --branch_ablation no_dac,no_pa,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.15 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.32 \
  --sat_view_schedule "1@0.95:clear_leo,low_elev_leo,rain_leo;120@0.85:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.07 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 4 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --use_proto_memory \
  --lambda_proto 0.010 \
  --lambda_supcon_id 0.016 \
  --lambda_fishr 0.002 \
  --fishr_min_domains 4 \
  --swad_start_epoch 100
sleep 2

launch_train CEN31_A14 \
  CEN31_A14_cvs_liteh_timeonly_floor_r010 \
  5 \
  --arch_family cvsincnet \
  --model_variant lite_h \
  --slim_group none \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.00 \
  --exp_group s3_rxrobust_no_dac \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.36 \
  --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;120@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" \
  --lambda_group_ce 0.06 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --group_ce_min_domains 3 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65 \
  --lambda_fishr 0.000 \
  --fishr_min_domains 3 \
  --swad_start_epoch 100
sleep 2

launch_kd CEN31_A15 \
  CEN31_A15_kd_liteb_bottleneck_fullsat_r010 \
  6 \
  --arch_family cvsincnet \
  --model_variant lite_b \
  --branch_ablation no_dac,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20 \
  --lambda_kd 0.62 \
  --lambda_feature_kd 0.14 \
  --lambda_relation_kd 0.035 \
  --lambda_group_ce 0.07 \
  --group_ce_min_domains 4 \
  --sat_view_prob 0.55 \
  --lambda_sat_view_ce 0.20 \
  --lambda_sat_view_kd 0.06 \
  --lambda_sat_view_feature_kd 0.02 \
  --lambda_sat_view_relation_kd 0.005 \
  --lambda_sat_view_group_ce 0.04 \
  --best_clean_weight 0.48 \
  --best_receiver_floor_weight 0.12 \
  --best_sat_mean_weight 0.28 \
  --best_sat_floor_weight 0.12 \
  --best_clean_guard_drop 1.5
sleep 2

launch_kd CEN31_A16 \
  CEN31_A16_kd_sinccvcnn_rxguard_fullsat_r010 \
  7 \
  --arch_family sinc_cvcnn \
  --model_variant lite_h \
  --branch_ablation no_dac,no_pa,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.00 \
  --lambda_kd 0.58 \
  --lambda_feature_kd 0.10 \
  --lambda_relation_kd 0.025 \
  --lambda_group_ce 0.08 \
  --group_ce_min_domains 3 \
  --sat_view_prob 0.55 \
  --lambda_sat_view_ce 0.22 \
  --lambda_sat_view_kd 0.05 \
  --lambda_sat_view_feature_kd 0.02 \
  --lambda_sat_view_relation_kd 0.005 \
  --lambda_sat_view_group_ce 0.04 \
  --best_clean_weight 0.45 \
  --best_receiver_floor_weight 0.18 \
  --best_sat_mean_weight 0.25 \
  --best_sat_floor_weight 0.12 \
  --best_clean_guard_drop 2.0
