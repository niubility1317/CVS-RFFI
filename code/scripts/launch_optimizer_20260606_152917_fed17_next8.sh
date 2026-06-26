#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260606_152917_fed17_next8}"
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

  echo "[FED17-CANDIDATE] lane=federated_vmb candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[FED17-CMD]'
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
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
)

SAT_FLOOR_VIEW=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 80
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.30
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

RA_CORE=(
  --train_mode fedprox
  --fl_local_objective receiver_agnostic_bex02
  --lambda_fishr 0.015
  --fishr_min_domains 4
  --fedprox_mu 0.01
  --fl_local_epochs 3
  --lambda_rx_adv 0.35
  --grl_lambda 0.60
)

FED_CGRL_GUARDED=(
  --use_fed_cgrl
  --fed_cgrl_base_lambda 0.25
  --fed_cgrl_min_lambda 0.03
  --fed_cgrl_max_lambda 0.80
  --fed_cgrl_warmup_rounds 40
  --fed_cgrl_leak_target_acc 20.0
  --fed_cgrl_leak_stat p90
  --fed_cgrl_tx_loss_guard 0.0
  --fed_cgrl_tx_guard_release_rounds 80
  --fed_cgrl_conflict_threshold -0.05
  --fed_cgrl_conflict_source auto
  --fed_cgrl_ema 0.50
)

VMB_CORE=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_server_lr 0.003
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 0.20
  --grl_lambda 0.45
  --lambda_tx_adv_r 0.015
  --use_tx_adv_on_zdom
  --lambda_vmb_tx_proto 0.025
  --lambda_vmb_rx_proto 0.015
  --fl_vmb_pretrain_rounds 100
  --fl_vmb_stage1_local_steps 4
  --fl_vmb_batches_per_client 4
  --fl_vmb_stage1_objective ce
)

STYLE_PROBE=(
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy receiver_balanced
  --fl_style_replay_start_round 120
  --fl_style_phys_start_round 120
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.04
  --fl_style_transform_mix_alpha 0.15
  --fl_style_min_remote_centroids 1
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
echo "[FED17] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_fed FED17_R01 FED17_R01_fedavg_satfloor_earlyce_r010 0 \
  --train_mode fedavg --fl_local_objective receiver_agnostic_bex02 --fl_local_epochs 3 --lambda_fishr 0.008 --fishr_min_domains 4 --lambda_rx_adv 0.15 --grl_lambda 0.25 "${SAT_FLOOR_VIEW[@]}" --use_fed_proto_stats --lambda_fed_proto 0.004 --fed_proto_momentum 0.92

launch_fed FED17_R02 FED17_R02_fedprox_proto_satfloor_r010 1 \
  "${RA_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" --use_fed_proto_stats --lambda_fed_proto 0.008 --fed_proto_momentum 0.92

launch_fed FED17_R03 FED17_R03_fedprox_coral_late_rx8_r010 2 \
  "${RA_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" --use_fed_coral --lambda_fl_coral_zid_global 0.0002 --fl_coral_stage all --fl_coral_start_round 140 --fl_coral_feature z_id --fl_coral_cov_mode diag --fl_coral_min_count 2 --fl_coral_collect_views clean

launch_fed FED17_R04 FED17_R04_fedavg_clean_anchor_sat_r010 3 \
  --train_mode fedavg --fl_local_objective receiver_agnostic_bex02 --fl_local_epochs 3 --lambda_fishr 0.006 --fishr_min_domains 4 --lambda_rx_adv 0.10 --grl_lambda 0.20 "${SAT_FLOOR_VIEW[@]}" --fl_conflict_agg cosine_clip

launch_fed FED17_A05 FED17_A05_vmb_pcgrad_satfloor_r010 4 \
  "${VMB_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" --fl_conflict_agg pcgrad --lambda_vmb_tx_proto 0.035 --lambda_vmb_rx_proto 0.020 --fl_vmb_pretrain_rounds 90

launch_fed FED17_A06 FED17_A06_vmb_cen31_profile_sat_r010 5 \
  "${VMB_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" --fl_vmb_cen_a31_profile --lambda_rx_adv 0.15 --grl_lambda 0.30 --lambda_vmb_tx_proto 0.025 --lambda_vmb_rx_proto 0.015

launch_fed FED17_A07 FED17_A07_vmb_stylebank_cgrl_guard_r010 6 \
  "${VMB_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" "${FED_CGRL_GUARDED[@]}" "${STYLE_PROBE[@]}" --fl_conflict_agg cosine_clip

launch_fed FED17_A08 FED17_A08_vmb_protoevidence_satfloor_r010 7 \
  "${VMB_CORE[@]}" "${SAT_FLOOR_VIEW[@]}" --use_proto_evidence_bank --proto_rho_max 0.05 --proto_top_m 4 --proto_max_per_class 8 --lambda_rx_adv 0.08 --grl_lambda 0.20 --fl_conflict_agg cosine_clip --lambda_vmb_tx_proto 0.020 --lambda_vmb_rx_proto 0.012
