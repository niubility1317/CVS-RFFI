#!/usr/bin/env bash
set -euo pipefail

# Federated next-8: RA_BEX02_R08 trunk, low-dose SATCE, delayed/stage-gated VMB probes.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_ID="${RUN_ID:-optimizer_20260604_094155_federated_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-2}"
CPU_THREADS="${CPU_THREADS:-${CVSRFFI_CPU_THREADS:-4}}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-${CVSRFFI_CPU_INTEROP_THREADS:-1}}"

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

THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
)

COMMON_ARGS=(
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

FSDG_RA_CORE=(
  --train_mode fedprox
  --fl_local_objective receiver_agnostic_bex02
  --lambda_fishr 0.02
  --fishr_min_domains 4
  --fedprox_mu 0.01
)

RA_DEFAULT=(
  --lambda_rx_adv 1.0
  --grl_lambda 1.0
)

SAT_CVS_ARGS=(
  --use_sat_consistency
  --fl_sat_aug_mode cvs_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
)

SAT_LATE_CE_W025_R120=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 120
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.25
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

SAT_LATE_CE_W015_R150=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 150
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.15
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

CORAL_ZID_0005_R120=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.0005
  --fl_coral_stage all
  --fl_coral_start_round 120
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

STYLE_PROBE_ONLY=(
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
  --fl_style_replay_start_round 160
  --fl_style_phys_start_round 160
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.00
  --fl_style_transform_mix_alpha 0.20
  --fl_style_min_remote_centroids 1
)

VMB_DELAYED_BASE=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_server_lr 0.004
  --fl_vmb_server_momentum 0.9
  --fl_conflict_agg cosine_clip
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 1.0
  --grl_lambda 1.0
  --lambda_tx_adv_r 0.06
  --use_tx_adv_on_zdom
  --lambda_vmb_tx_proto 0.04
  --lambda_vmb_rx_proto 0.03
  --fl_vmb_pretrain_rounds 100
  --fl_vmb_stage1_local_steps 8
  --fl_vmb_batches_per_client 4
  --fl_vmb_stage1_objective receiver_style_pretrain
  --fl_vmb_stage1_use_aux_losses
)

VMB_CLEAN_STAGE1=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_server_lr 0.004
  --fl_vmb_server_momentum 0.9
  --fl_conflict_agg cosine_clip
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 0.75
  --grl_lambda 0.75
  --lambda_tx_adv_r 0.00
  --lambda_vmb_tx_proto 0.03
  --lambda_vmb_rx_proto 0.02
  --fl_vmb_pretrain_rounds 120
  --fl_vmb_stage1_local_steps 8
  --fl_vmb_batches_per_client 4
  --fl_vmb_stage1_objective ce
)

gpu_process_count() {
  local gpu="$1"
  "${NVIDIA_SMI_BIN}" --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

launch() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

  local log_path="${LOG_ROOT}/${run_name}.out"
  local output_dir="${RUNS_ROOT}/${run_name}"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${output_dir}"
    --log_dir "${LOG_ROOT}/${run_name}"
    "$@"
  )

  echo "[FED-NEXT8] candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[FED-NEXT8-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${log_path}" || -e "${LOG_ROOT}/${run_name}" || -e "${output_dir}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${output_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_PROCS_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_PROCS_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}/${run_name}" "${output_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${output_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch FED_C01 \
  FED_C01_r08_localepoch3_nocoral_r010 \
  0 \
  "${FSDG_RA_CORE[@]}" "${RA_DEFAULT[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 3
sleep 2

launch FED_C02 \
  FED_C02_r08_localepoch3_lowcoral_late_r010 \
  1 \
  "${FSDG_RA_CORE[@]}" "${RA_DEFAULT[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 3 "${CORAL_ZID_0005_R120[@]}"
sleep 2

launch FED_C03 \
  FED_C03_r08_rxadv075_localepoch3_r010 \
  2 \
  "${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 3 --lambda_rx_adv 0.75 --grl_lambda 1.0
sleep 2

launch FED_C04 \
  FED_C04_r08_latesat025_nocoral_r010 \
  3 \
  "${FSDG_RA_CORE[@]}" "${RA_DEFAULT[@]}" "${SAT_LATE_CE_W025_R120[@]}" --fl_local_epochs 3
sleep 2

launch FED_A05 \
  FED_A05_stage1_rxstyle120_bpc4_r010 \
  4 \
  "${VMB_DELAYED_BASE[@]}" "${SAT_CVS_ARGS[@]}"
sleep 2

launch FED_A06 \
  FED_A06_stage1_clean120_bpc4_r010 \
  5 \
  "${VMB_CLEAN_STAGE1[@]}" "${SAT_CVS_ARGS[@]}"
sleep 2

launch FED_A07 \
  FED_A07_vmb_lateproto_sat025_r010 \
  6 \
  "${VMB_DELAYED_BASE[@]}" "${SAT_LATE_CE_W025_R120[@]}"
sleep 2

launch FED_A08 \
  FED_A08_ra_latesat015_styleprobe_r010 \
  7 \
  "${FSDG_RA_CORE[@]}" "${RA_DEFAULT[@]}" "${SAT_LATE_CE_W015_R150[@]}" --fl_local_epochs 3 "${STYLE_PROBE_ONLY[@]}"
