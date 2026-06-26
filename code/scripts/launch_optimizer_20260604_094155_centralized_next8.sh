#!/usr/bin/env bash
set -euo pipefail

# CEN31 next-8: four CEN31 fast-path recipe variants plus four structure/KD candidates.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-${ROOT}/code/train_cen31_distill.py}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/optimizer_20260530_043050_centralized_next8/CEN_A31_a22_satboost_ce1p28_stack_r010/best_primary_ood_model.pth}"
RUN_ID="${RUN_ID:-optimizer_20260604_094155_centralized_next8}"
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

COMMON_CEN31_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs 200
  --test_eval_policy every_epoch
  --test_eval_start_epoch 81
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --arch_family cvsincnet
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
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
  --swad_start_epoch 100
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
  --epochs 300
  --eval_interval 10
  --eval_max_batches 0
  --seed 1337
  --teacher_ckpt "${TEACHER_CKPT}"
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --group_ce_min_domains 4
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
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --eval_sat_on main
  --sat_eval_max_batches 0
  --best_select_metric clean_sat_joint
  --sat_select_eval_interval 20
  --sat_select_max_batches 0
)

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
    "${COMMON_CEN31_ARGS[@]}"
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

  echo "[CEN31-NEXT8] candidate=${candidate_id} run=${run_name} gpu=${gpu} mode=train dry_run=${DRY_RUN}"
  printf '[CEN31-NEXT8-CMD]'
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

  echo "[CEN31-NEXT8] candidate=${candidate_id} run=${run_name} gpu=${gpu} mode=distill dry_run=${DRY_RUN}"
  printf '[CEN31-NEXT8-CMD]'
  print_cmd "${cmd[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

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
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch_train CEN31_C01 \
  CEN31_C01_fastpath_e200_teacher_repro_r010 \
  0 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.28
sleep 2

launch_train CEN31_C02 \
  CEN31_C02_fastpath_rx8_satlight_r010 \
  1 \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.18 \
  --sat_view_schedule "1@0.80:clear_leo,low_elev_leo,rain_leo;140@0.65:clear_leo,low_elev_leo,rain_leo"
sleep 2

launch_train CEN31_C03 \
  CEN31_C03_fastpath_min3_groupfishr_r010 \
  2 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.24 \
  --group_ce_min_domains 3 \
  --fishr_min_domains 3
sleep 2

launch_train CEN31_C04 \
  CEN31_C04_fastpath_swad_worstrx_r010 \
  3 \
  --primary_udu_weight 0.76 \
  --concat_sat_ce_weight 1.22 \
  --swad_start_epoch 90
sleep 2

launch_kd CEN31_A05 \
  CEN31_A05_kd4_litef_fullsat_r010 \
  4 \
  --lambda_kd 0.65 \
  --lambda_feature_kd 0.18 \
  --lambda_relation_kd 0.04 \
  --lambda_group_ce 0.06 \
  --sat_view_prob 0.45 \
  --lambda_sat_view_ce 0.16 \
  --lambda_sat_view_kd 0.08 \
  --lambda_sat_view_feature_kd 0.02 \
  --lambda_sat_view_relation_kd 0.005 \
  --lambda_sat_view_group_ce 0.03 \
  --best_clean_weight 0.55 \
  --best_receiver_floor_weight 0.10 \
  --best_sat_mean_weight 0.25 \
  --best_sat_floor_weight 0.10 \
  --best_clean_guard_drop 1.0 \
  --arch_family cvsincnet \
  --model_variant lite_f \
  --branch_ablation no_dac,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20
sleep 2

launch_kd CEN31_A06 \
  CEN31_A06_kd4_litef_rx8_guard_r010 \
  5 \
  --arch_family cvsincnet \
  --model_variant lite_f \
  --branch_ablation no_dac,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20 \
  --lambda_kd 0.70 \
  --lambda_feature_kd 0.20 \
  --lambda_relation_kd 0.05 \
  --lambda_group_ce 0.08 \
  --sat_view_prob 0.45 \
  --best_clean_weight 0.50 \
  --best_receiver_floor_weight 0.20 \
  --best_sat_mean_weight 0.20 \
  --best_sat_floor_weight 0.10 \
  --best_clean_guard_drop 1.5 \
  --lambda_sat_view_ce 0.10 \
  --lambda_sat_view_kd 0.05 \
  --lambda_sat_view_feature_kd 0.02 \
  --lambda_sat_view_relation_kd 0.005 \
  --lambda_sat_view_group_ce 0.03
sleep 2

launch_kd CEN31_A07 \
  CEN31_A07_kd4_sinccvcnn_satdg_r010 \
  6 \
  --best_clean_weight 0.55 \
  --best_receiver_floor_weight 0.10 \
  --best_sat_mean_weight 0.25 \
  --best_sat_floor_weight 0.10 \
  --best_clean_guard_drop 1.0 \
  --arch_family sinc_cvcnn \
  --model_variant lite_h \
  --branch_ablation no_dac,no_pa,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.0 \
  --lambda_kd 0.62 \
  --lambda_feature_kd 0.12 \
  --lambda_relation_kd 0.03 \
  --sat_view_prob 0.45 \
  --lambda_sat_view_ce 0.18 \
  --lambda_sat_view_kd 0.06 \
  --lambda_sat_view_feature_kd 0.02 \
  --lambda_sat_view_relation_kd 0.005 \
  --lambda_sat_view_group_ce 0.04
sleep 2

launch_kd CEN31_A08 \
  CEN31_A08_kd4_cvcnn_compact_satfloor_r010 \
  7 \
  --best_clean_weight 0.55 \
  --best_receiver_floor_weight 0.10 \
  --best_sat_mean_weight 0.25 \
  --best_sat_floor_weight 0.10 \
  --best_clean_guard_drop 1.0 \
  --arch_family cvcnn \
  --model_variant lite_h \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.0 \
  --lambda_kd 0.60 \
  --lambda_feature_kd 0.08 \
  --lambda_relation_kd 0.02 \
  --sat_view_prob 0.30 \
  --lambda_sat_view_ce 0.10 \
  --lambda_sat_view_kd 0.04 \
  --lambda_sat_view_feature_kd 0.01 \
  --lambda_sat_view_relation_kd 0.003 \
  --lambda_sat_view_group_ce 0.02
