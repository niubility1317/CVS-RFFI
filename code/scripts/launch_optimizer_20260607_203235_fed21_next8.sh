#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260607_203235_fed21_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
CPU_THREADS="${CPU_THREADS:-4}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"

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
  local candidate_id="$1" run_name="$2"
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

  echo "[FED21-CANDIDATE] lane=federated_vmb candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[FED21-CMD]'
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

THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
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
  --mixstyle_strength 0.60
  --mixstyle_p 0.14
  --mixstyle_late_start 120
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.04
  --mixstyle_late_min_strength 0.24
)

SAT_FLOOR_VIEW=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --fl_baseline_view_ce_only
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

VMB_BALANCED=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_local_epochs 1
  --fl_vmb_stage auto
  --fl_vmb_server_lr 0.003
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_tx_adv_r 0.015
  --use_tx_adv_on_zdom
  --fl_vmb_stage1_local_steps 4
  --fl_vmb_batches_per_client 4
  --fl_vmb_stage1_objective ce
)

STYLE_LIGHT=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy receiver_balanced
  --fl_style_replay_start_round 145
  --fl_style_phys_start_round 150
  --fl_style_dg_start_round 170
  --fl_style_dg_min_domains 2
  --style_gate_min_accept_rate 0.50
  --fl_style_min_remote_centroids 2
  --fl_style_max_views 1
  --fl_style_replay_prob 0.05
  --fl_style_transform_mix_alpha 0.05
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
)

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
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi

cd "${ROOT}"
echo "[FED21] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_fed FED21_R01 FED21_R01_r02_fedavg_cgrl_nofishr_satfloor_r010 0 \
  --train_mode fedavg --fl_local_objective receiver_agnostic_bex02 --fl_local_epochs 3 \
  --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.06 --grl_lambda 0.12 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 56 --fl_baseline_view_ce_weight 0.44 \
  --use_fed_cgrl --fed_cgrl_base_lambda 0.08 --fed_cgrl_min_lambda 0.01 --fed_cgrl_max_lambda 0.26 \
  --fed_cgrl_warmup_rounds 70 --fed_cgrl_leak_target_acc 20.0 --fed_cgrl_leak_stat p90 \
  --fed_cgrl_tx_loss_guard 0.0 --fed_cgrl_tx_guard_release_rounds 130 --fed_cgrl_conflict_threshold -0.03 \
  --fed_cgrl_conflict_source auto --fed_cgrl_ema 0.60 --fl_conflict_agg cosine_clip \
  --lambda_group_ce 0.015 --group_ce_mode smooth_dro_capped --group_ce_top_frac 0.30 --groupdro_tau 0.45 --groupdro_cap 0.58

launch_fed FED21_R02 FED21_R02_r03_fedprox_proto_latecoral_floor_r010 1 \
  --train_mode fedprox --fl_local_objective receiver_agnostic_bex02 --fedprox_mu 0.006 --fl_local_epochs 3 \
  --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.09 --grl_lambda 0.16 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 58 --fl_baseline_view_ce_weight 0.43 \
  --use_fed_proto_stats --lambda_fed_proto 0.008 --fed_proto_momentum 0.96 \
  --use_fed_coral --lambda_fl_coral_zid_global 0.00008 --fl_coral_stage all --fl_coral_start_round 180 \
  --fl_coral_feature z_id --fl_coral_cov_mode diag --fl_coral_min_count 2 --fl_coral_collect_views clean

launch_fed FED21_R03 FED21_R03_r01_fedprox_rx8_guard_satce45_r010 2 \
  --train_mode fedprox --fl_local_objective receiver_agnostic_bex02 --fedprox_mu 0.010 --fl_local_epochs 3 \
  --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.11 --grl_lambda 0.18 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 54 --fl_baseline_view_ce_weight 0.45 \
  --use_fed_proto_stats --lambda_fed_proto 0.005 --fed_proto_momentum 0.95 \
  --lambda_group_ce 0.012 --group_ce_mode smooth_dro_capped --group_ce_top_frac 0.25 --groupdro_tau 0.40 --groupdro_cap 0.52

launch_fed FED21_R04 FED21_R04_r02_fedavg_conflict_floor_hold_r010 3 \
  --train_mode fedavg --fl_local_objective receiver_agnostic_bex02 --fl_local_epochs 3 \
  --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.08 --grl_lambda 0.15 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 60 --fl_baseline_view_ce_weight 0.42 \
  --use_fed_cgrl --fed_cgrl_base_lambda 0.09 --fed_cgrl_min_lambda 0.01 --fed_cgrl_max_lambda 0.30 \
  --fed_cgrl_warmup_rounds 80 --fed_cgrl_leak_target_acc 20.0 --fed_cgrl_leak_stat p90 \
  --fed_cgrl_tx_loss_guard 0.0 --fed_cgrl_tx_guard_release_rounds 140 --fed_cgrl_conflict_threshold -0.02 \
  --fed_cgrl_conflict_source auto --fed_cgrl_ema 0.62 --fl_conflict_agg pcgrad

launch_fed FED21_A05 FED21_A05_a05_vmb_pcgrad_rx8_lift_floor_r010 4 \
  "${VMB_BALANCED[@]}" \
  --fl_vmb_pretrain_rounds 132 --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.14 --grl_lambda 0.28 \
  --lambda_vmb_tx_proto 0.034 --lambda_vmb_rx_proto 0.030 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 52 --fl_baseline_view_ce_weight 0.44 \
  --use_fed_proto_stats --lambda_fed_proto 0.006 --fed_proto_momentum 0.96 \
  --fl_conflict_agg pcgrad

launch_fed FED21_A06 FED21_A06_a05_vmb_proto_nofishr_rx8_rescue_r010 5 \
  "${VMB_BALANCED[@]}" \
  --fl_vmb_pretrain_rounds 128 --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.13 --grl_lambda 0.25 \
  --lambda_vmb_tx_proto 0.038 --lambda_vmb_rx_proto 0.030 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 58 --fl_baseline_view_ce_weight 0.42 \
  --use_fed_proto_stats --lambda_fed_proto 0.014 --fed_proto_momentum 0.97

launch_fed FED21_A07 FED21_A07_a07_latecoral_groupce_storm_floor_r010 6 \
  --train_mode fedprox --fl_local_objective receiver_agnostic_bex02 --fedprox_mu 0.007 --fl_local_epochs 3 \
  --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.09 --grl_lambda 0.18 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 62 --fl_baseline_view_ce_weight 0.50 \
  --use_fed_coral --lambda_fl_coral_zid_global 0.00018 --fl_coral_stage all --fl_coral_start_round 160 \
  --fl_coral_feature z_id --fl_coral_cov_mode diag --fl_coral_min_count 2 --fl_coral_collect_views clean \
  --lambda_group_ce 0.020 --group_ce_mode smooth_dro_capped --group_ce_top_frac 0.22 --groupdro_tau 0.36 --groupdro_cap 0.48

launch_fed FED21_A08 FED21_A08_a08_style_phys_delayed_floor_probe_r010 7 \
  "${VMB_BALANCED[@]}" "${STYLE_LIGHT[@]}" \
  --fl_style_phys_p_multipath 0.12 --fl_style_phys_min_awgn_snr_db 24 --fl_style_phys_max_multipath_taps 2 \
  --fl_vmb_pretrain_rounds 135 --lambda_fishr 0.000 --fishr_min_domains 2 --lambda_rx_adv 0.10 --grl_lambda 0.20 \
  --lambda_vmb_tx_proto 0.024 --lambda_vmb_rx_proto 0.018 \
  "${SAT_FLOOR_VIEW[@]}" --sat_cons_start_epoch 78 --fl_baseline_view_ce_weight 0.40
