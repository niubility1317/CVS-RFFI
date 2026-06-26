#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-${ROOT}/code/train_cen31_distill.py}"
RUN_ID="${RUN_ID:-optimizer_20260606_094528_dual_lane_16}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/optimizer_20260604_094155_centralized_next8/CEN31_C04_fastpath_swad_worstrx_r010/best_primary_ood_model.pth}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
CPU_THREADS="${CPU_THREADS:-4}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"

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

should_skip() {
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A LAUNCHED_BY_GPU=()

run_cmd() {
  local lane="$1"
  local candidate_id="$2"
  local run_name="$3"
  local gpu="$4"
  local run_dir="$5"
  local log_path="$6"
  shift 6
  local cmd=("$@")

  if should_skip "${candidate_id}" "${run_name}"; then
    return 0
  fi

  echo "[DUAL16-CANDIDATE] lane=${lane} candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[DUAL16-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${lane}" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" \
      "${lane}" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${lane}" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
)

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

COMMON_FED_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio 0.9
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --epochs 200
  --fl_rounds 200
  --fl_client_key receiver
  --fl_clients_per_round 1.0
  --fl_test_eval_interval 10
  --fl_test_eval_last_n 5
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
  --num_workers 0
  --fl_num_workers 0
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --primary_udu_weight 0.70
  --use_aug
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
)

RA_CORE=(
  --train_mode fedprox
  --fl_local_objective receiver_agnostic_bex02
  --lambda_fishr 0.02
  --fishr_min_domains 4
  --fedprox_mu 0.01
  --fl_local_epochs 3
  --lambda_rx_adv 0.75
  --grl_lambda 1.0
)

LOW_SAT_VIEW=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 120
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.20
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

VMB_CORE=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_server_lr 0.004
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 0.50
  --grl_lambda 0.75
  --lambda_tx_adv_r 0.03
  --use_tx_adv_on_zdom
  --lambda_vmb_tx_proto 0.03
  --lambda_vmb_rx_proto 0.02
  --fl_vmb_pretrain_rounds 120
  --fl_vmb_stage1_local_steps 6
  --fl_vmb_batches_per_client 4
  --fl_vmb_stage1_objective ce
)

FED_CGRL_LOW=(
  --use_fed_cgrl
  --fed_cgrl_base_lambda 0.50
  --fed_cgrl_min_lambda 0.05
  --fed_cgrl_max_lambda 1.25
  --fed_cgrl_warmup_rounds 20
  --fed_cgrl_leak_target_acc 20.0
  --fed_cgrl_leak_stat p90
  --fed_cgrl_tx_loss_guard 0.0
  --fed_cgrl_tx_guard_release_rounds 60
  --fed_cgrl_conflict_threshold -0.10
  --fed_cgrl_conflict_source auto
  --fed_cgrl_ema 0.35
)

STYLE_PROBE=(
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy receiver_balanced
  --fl_style_replay_start_round 140
  --fl_style_phys_start_round 140
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.05
  --fl_style_transform_mix_alpha 0.20
  --fl_style_min_remote_centroids 1
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
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
  run_cmd centralized "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

launch_cen_kd() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  shift 3
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
  run_cmd centralized "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

launch_fed() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  shift 3
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_FED_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --log_dir "${LOG_ROOT}/${run_name}"
    "$@"
  )
  run_cmd federated_vmb "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
  [[ -f "${DISTILL_SCRIPT}" ]] || { echo "[ERROR] missing distill script: ${DISTILL_SCRIPT}" >&2; exit 2; }
  [[ -f "${TEACHER_CKPT}" ]] || { echo "[ERROR] missing teacher checkpoint: ${TEACHER_CKPT}" >&2; exit 2; }
fi

cd "${ROOT}"
echo "[DUAL16] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_cen_train CEN40_R01 CEN40_R01_t14_satfloor_recover_r010 0 \
  --primary_udu_weight 0.74 --concat_sat_ce_weight 1.36 --no_use_freq_band_gate --sat_view_schedule "1@1.00:clear_leo,low_elev_leo,rain_leo;120@0.85:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
launch_fed FED16_R01 FED16_R01_ra_cgrl_guarded_r010 0 \
  "${RA_CORE[@]}" "${LOW_SAT_VIEW[@]}" "${FED_CGRL_LOW[@]}"

launch_cen_train CEN40_R02 CEN40_R02_domainoff_satfloor_guard_r010 1 \
  --primary_udu_weight 0.74 --concat_sat_ce_weight 1.32 --domain_enhancer off --domain_enhancer_strength 0.0 --lambda_fishr 0.008 --swad_start_epoch 80
launch_fed FED16_R02 FED16_R02_ra_lowproto_cgrl_r010 1 \
  "${RA_CORE[@]}" "${LOW_SAT_VIEW[@]}" "${FED_CGRL_LOW[@]}" --use_fed_proto_stats --lambda_fed_proto 0.005 --fed_proto_momentum 0.90

launch_cen_train CEN40_R03 CEN40_R03_min6stats_satfloor_guard_r010 2 \
  --primary_udu_weight 0.72 --concat_sat_ce_weight 1.34 --domain_enhancer rcn_minimal_6stats --lambda_group_ce 0.08 --group_ce_min_domains 3
launch_fed FED16_R03 FED16_R03_ra_lowcoral_styleprobe_r010 2 \
  "${RA_CORE[@]}" "${LOW_SAT_VIEW[@]}" --use_fed_coral --lambda_fl_coral_zid_global 0.0003 --fl_coral_stage all --fl_coral_start_round 130 --fl_coral_feature z_id --fl_coral_cov_mode diag --fl_coral_min_count 2 --fl_coral_collect_views clean "${STYLE_PROBE[@]}"

launch_cen_train CEN40_R04 CEN40_R04_timepa_latency_sat_guard_r010 3 \
  --primary_udu_weight 0.74 --concat_sat_ce_weight 1.30 --pa_orders 1,5 --lambda_proto 0.010 --lambda_fishr 0.008
launch_fed FED16_R04 FED16_R04_fedavg_proto_floor_r010 3 \
  --train_mode fedavg --fl_local_objective receiver_agnostic_bex02 --fl_local_epochs 3 --lambda_fishr 0.01 --fishr_min_domains 4 --lambda_rx_adv 0.25 --grl_lambda 0.50 "${LOW_SAT_VIEW[@]}" --use_fed_proto_stats --lambda_fed_proto 0.003 --fed_proto_momentum 0.90

launch_cen_kd CEN40_A05 CEN40_A05_c04_litef_latency_joint_r010 4 \
  --arch_family cvsincnet --model_variant lite_f --branch_ablation no_dac,no_stats --domain_branch_ablation no_stats --domain_enhancer rcn_stats --domain_enhancer_strength 0.20 --lambda_kd 0.68 --lambda_feature_kd 0.18 --lambda_relation_kd 0.04 --lambda_group_ce 0.06 --sat_view_prob 0.45 --lambda_sat_view_ce 0.16 --lambda_sat_view_kd 0.08 --lambda_sat_view_feature_kd 0.02 --lambda_sat_view_relation_kd 0.005 --lambda_sat_view_group_ce 0.03 --best_clean_weight 0.55 --best_receiver_floor_weight 0.10 --best_sat_mean_weight 0.25 --best_sat_floor_weight 0.10 --best_clean_guard_drop 1.0
launch_fed FED16_A05 FED16_A05_vmb_pcgrad_proto120_r010 4 \
  "${VMB_CORE[@]}" "${LOW_SAT_VIEW[@]}" --fl_conflict_agg pcgrad --lambda_vmb_tx_proto 0.04 --lambda_vmb_rx_proto 0.025

launch_cen_kd CEN40_A06 CEN40_A06_c04_litef_rx8_guard_r010 5 \
  --arch_family cvsincnet --model_variant lite_f --branch_ablation no_dac,no_stats --domain_branch_ablation no_stats --domain_enhancer rcn_stats --domain_enhancer_strength 0.20 --lambda_kd 0.72 --lambda_feature_kd 0.20 --lambda_relation_kd 0.05 --lambda_group_ce 0.08 --sat_view_prob 0.45 --best_clean_weight 0.50 --best_receiver_floor_weight 0.20 --best_sat_mean_weight 0.20 --best_sat_floor_weight 0.10 --best_clean_guard_drop 1.5 --lambda_sat_view_ce 0.10 --lambda_sat_view_kd 0.05 --lambda_sat_view_feature_kd 0.02 --lambda_sat_view_relation_kd 0.005 --lambda_sat_view_group_ce 0.03
launch_fed FED16_A06 FED16_A06_vmb_cen31_profile_local4_r010 5 \
  "${VMB_CORE[@]}" "${LOW_SAT_VIEW[@]}" --fl_vmb_cen_a31_profile --fl_vmb_stage1_local_steps 4 --lambda_rx_adv 0.25 --grl_lambda 0.50 --lambda_vmb_tx_proto 0.025 --lambda_vmb_rx_proto 0.015

launch_cen_kd CEN40_A07 CEN40_A07_c04_sinccvcnn_satdg_r010 6 \
  --best_clean_weight 0.55 --best_receiver_floor_weight 0.10 --best_sat_mean_weight 0.25 --best_sat_floor_weight 0.10 --best_clean_guard_drop 1.0 --arch_family sinc_cvcnn --model_variant lite_h --branch_ablation no_dac,no_pa,no_stats --domain_branch_ablation no_stats --domain_enhancer off --domain_enhancer_strength 0.0 --lambda_kd 0.64 --lambda_feature_kd 0.12 --lambda_relation_kd 0.03 --sat_view_prob 0.45 --lambda_sat_view_ce 0.18 --lambda_sat_view_kd 0.06 --lambda_sat_view_feature_kd 0.02 --lambda_sat_view_relation_kd 0.005 --lambda_sat_view_group_ce 0.04
launch_fed FED16_A07 FED16_A07_vmb_stylebank_cgrl_r010 6 \
  "${VMB_CORE[@]}" "${LOW_SAT_VIEW[@]}" "${FED_CGRL_LOW[@]}" --fl_vmb_ra_stylebank_profile --fl_conflict_agg cosine_clip --lambda_vmb_tx_proto 0.03 --lambda_vmb_rx_proto 0.02

launch_cen_kd CEN40_A08 CEN40_A08_c04_cvcnn_compact_floor_r010 7 \
  --best_clean_weight 0.55 --best_receiver_floor_weight 0.10 --best_sat_mean_weight 0.25 --best_sat_floor_weight 0.10 --best_clean_guard_drop 1.0 --arch_family cvcnn --model_variant lite_h --branch_ablation no_dac,no_pa,no_freq,no_stats --domain_branch_ablation no_stats --domain_enhancer off --domain_enhancer_strength 0.0 --lambda_kd 0.62 --lambda_feature_kd 0.08 --lambda_relation_kd 0.02 --sat_view_prob 0.30 --lambda_sat_view_ce 0.10 --lambda_sat_view_kd 0.04 --lambda_sat_view_feature_kd 0.01 --lambda_sat_view_relation_kd 0.003 --lambda_sat_view_group_ce 0.02
launch_fed FED16_A08 FED16_A08_vmb_protoevidence_lowadv_r010 7 \
  "${VMB_CORE[@]}" "${LOW_SAT_VIEW[@]}" --use_proto_evidence_bank --proto_rho_max 0.05 --proto_top_m 4 --proto_max_per_class 8 --lambda_rx_adv 0.10 --grl_lambda 0.25 --fl_conflict_agg cosine_clip --lambda_vmb_tx_proto 0.02 --lambda_vmb_rx_proto 0.015
