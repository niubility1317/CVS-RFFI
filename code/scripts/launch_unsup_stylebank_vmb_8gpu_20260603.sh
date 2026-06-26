#!/usr/bin/env bash
set -euo pipefail

# One-new-run-per-GPU matrix for the domain-consistency -> StyleBank -> VMB bridge.
# Hard FL contract: --wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_ID="${RUN_ID:-20260603_193500_unsup_stylebank_vmb}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_INDEX="${ONLY_INDEX:-}"
ONLY_RUN="${ONLY_RUN:-}"
ONLY_GPU="${ONLY_GPU:-}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-2}"
ALLOW_DUPLICATE_RUN="${ALLOW_DUPLICATE_RUN:-0}"
CPU_THREADS="${CPU_THREADS:-${CVSRFFI_CPU_THREADS:-4}}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-${CVSRFFI_CPU_INTEROP_THREADS:-1}}"

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
)

VMB_RA_CORE=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_pretrain_rounds 60
  --fl_vmb_stage1_local_steps 4
  --fl_vmb_stage1_use_aux_losses
  --fl_vmb_batches_per_client 4
  --fl_vmb_server_lr 0.006
  --fl_vmb_server_momentum 0.9
  --fl_conflict_agg cosine_clip
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 1.0
  --grl_lambda 1.0
  --lambda_tx_adv_r 0.10
  --use_tx_adv_on_zdom
  --lambda_vmb_tx_proto 0.10
  --lambda_vmb_rx_proto 0.10
  --use_sat_consistency
  --fl_sat_aug_mode cvs_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
)

STAGE1_RECEIVER_STYLE=(
  --fl_vmb_stage1_objective receiver_style_pretrain
)

DOMAIN_CONSISTENCY=(
  --fl_vmb_stage1_objective domain_unsup_pretrain
  --fl_domain_pretrain_train_scope domain
  --domain_unsup_pretrain_method consistency
  --lambda_domain_unsup_pretrain 0.20
  --lambda_domain_unsup_metadata_ce 0.00
  --lambda_domain_unsup_var 0.05
  --domain_unsup_noise_std 0.01
  --domain_unsup_amp_jitter 0.03
  --domain_unsup_logit_cons_weight 0.10
)

DOMAIN_METADATA=(
  --fl_vmb_stage1_objective domain_unsup_pretrain
  --fl_domain_pretrain_train_scope domain
  --domain_unsup_pretrain_method metadata_consistency
  --lambda_domain_unsup_pretrain 0.20
  --lambda_domain_unsup_metadata_ce 0.50
  --lambda_domain_unsup_var 0.05
  --domain_unsup_noise_std 0.01
  --domain_unsup_amp_jitter 0.03
  --domain_unsup_logit_cons_weight 0.10
)

STYLE_PROBE=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy receiver_balanced
  --fl_style_replay_start_round 40
  --fl_style_phys_start_round 40
  --fl_style_dg_start_round 999
  --fl_style_dg_min_domains 2
  --style_gate_min_accept_rate 0.50
  --fl_style_min_remote_centroids 2
  --fl_style_max_views 1
  --fl_style_replay_prob 0.10
  --fl_style_transform_mix_alpha 0.20
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
)

STYLE_DG=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy receiver_balanced
  --fl_style_replay_start_round 40
  --fl_style_phys_start_round 40
  --fl_style_dg_start_round 100
  --fl_style_dg_min_domains 2
  --style_gate_min_accept_rate 0.50
  --fl_style_min_remote_centroids 2
  --fl_style_max_views 1
  --fl_style_replay_prob 0.15
  --fl_style_transform_mix_alpha 0.25
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
)

FISHR_DSTYLE=(
  --lambda_fishr 0.01
  --fishr_min_domains 2
)

MIXSTYLE_LOW_LATE=(
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.55
  --mixstyle_p 0.12
  --mixstyle_late_start 120
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.04
  --mixstyle_late_min_strength 0.25
)

REAL_MIX_DIAG=(
  --fl_style_real_mix_samples 8
  --fl_style_real_mix_start_round 100
)

RUN_NAMES=(
  USBV_E0_vmb_ra_anchor_r010
  USBV_E1_domain_consistency_r010
  USBV_E2_domain_metadata_r010
  USBV_E3_stylebank_probe_r010
  USBV_E4_domain_stylebank_probe_r010
  USBV_E5_domain_stylebank_fishr_r010
  USBV_E6_domain_stylebank_fishr_mixstyle_r010
  USBV_E7_real_mix_upper_bound_diag_r010
)

RUN_GPUS=(0 1 2 3 4 5 6 7)

if [[ "${DRY_RUN}" != "1" ]]; then
  cd "${ROOT}"
  if [[ ! -f "${ROOT}/train.py" ]]; then
    echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
    exit 2
  fi
  mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
  PID_FILE="${LOG_ROOT}/launch_pids.tsv"
  if [[ ! -f "${PID_FILE}" ]]; then
    printf "run_name\tgpu\tpid\tlog_path\toutput_dir\n" > "${PID_FILE}"
  fi
else
  PID_FILE="/dev/null"
fi

declare -A GPU_ACTIVE_COUNTS
declare -A GPU_PLANNED_COUNTS
for gpu in 0 1 2 3 4 5 6 7; do
  GPU_PLANNED_COUNTS["${gpu}"]=0
  if [[ "${DRY_RUN}" == "1" ]]; then
    GPU_ACTIVE_COUNTS["${gpu}"]=0
    continue
  fi
  if ! active_pids="$("${NVIDIA_SMI_BIN}" --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
    echo "[ERROR] nvidia-smi query failed for gpu=${gpu}; refusing to launch without occupancy proof." >&2
    printf '%s\n' "${active_pids}" >&2
    exit 4
  fi
  active_count="$(printf '%s\n' "${active_pids}" | sed '/^$/d' | wc -l | tr -d ' ')"
  GPU_ACTIVE_COUNTS["${gpu}"]="${active_count:-0}"
done

for idx in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$idx]}"
  GPU="${RUN_GPUS[$idx]}"
  if [[ -n "${ONLY_GPU}" && "${ONLY_GPU}" != "${GPU}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_INDEX}" && "${ONLY_INDEX}" != "${idx}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_RUN}" && "${ONLY_RUN}" != "${RUN_NAME}" ]]; then
    continue
  fi

  EXTRA_ARGS=()
  case "${RUN_NAME}" in
    USBV_E0_vmb_ra_anchor_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${STAGE1_RECEIVER_STYLE[@]}")
      ;;
    USBV_E1_domain_consistency_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_CONSISTENCY[@]}")
      ;;
    USBV_E2_domain_metadata_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_METADATA[@]}")
      ;;
    USBV_E3_stylebank_probe_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${STAGE1_RECEIVER_STYLE[@]}" "${STYLE_PROBE[@]}")
      ;;
    USBV_E4_domain_stylebank_probe_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_METADATA[@]}" "${STYLE_PROBE[@]}")
      ;;
    USBV_E5_domain_stylebank_fishr_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_METADATA[@]}" "${STYLE_DG[@]}" "${FISHR_DSTYLE[@]}")
      ;;
    USBV_E6_domain_stylebank_fishr_mixstyle_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_METADATA[@]}" "${STYLE_DG[@]}" "${FISHR_DSTYLE[@]}" "${MIXSTYLE_LOW_LATE[@]}")
      ;;
    USBV_E7_real_mix_upper_bound_diag_r010)
      EXTRA_ARGS=("${VMB_RA_CORE[@]}" "${DOMAIN_METADATA[@]}" "${STYLE_DG[@]}" "${FISHR_DSTYLE[@]}" "${MIXSTYLE_LOW_LATE[@]}" "${REAL_MIX_DIAG[@]}")
      ;;
    *)
      echo "[ERROR] no case for run: ${RUN_NAME}" >&2
      exit 3
      ;;
  esac

  LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${ALLOW_DUPLICATE_RUN}" != "1" ]]; then
      if awk -F '\t' -v run="${RUN_NAME}" 'NR > 1 && $1 == run { found = 1 } END { exit(found ? 0 : 1) }' "${PID_FILE}"; then
        echo "[ERROR] run=${RUN_NAME} is already recorded in ${PID_FILE}; refusing duplicate launch for RUN_ID=${RUN_ID}." >&2
        exit 5
      fi
    fi
    if [[ -e "${LOG_PATH}" || -e "${LOG_ROOT}/${RUN_NAME}" || -e "${RUNS_ROOT}/${RUN_NAME}" ]]; then
      echo "[ERROR] existing log/output path for run=${RUN_NAME}; refusing duplicate launch for RUN_ID=${RUN_ID}." >&2
      echo "[ERROR] log=${LOG_PATH}" >&2
      echo "[ERROR] log_dir=${LOG_ROOT}/${RUN_NAME}" >&2
      echo "[ERROR] output_dir=${RUNS_ROOT}/${RUN_NAME}" >&2
      exit 6
    fi
    active_count="${GPU_ACTIVE_COUNTS[${GPU}]:-0}"
    planned_count="${GPU_PLANNED_COUNTS[${GPU}]:-0}"
    reserved_count=$((active_count + planned_count))
    if [[ "${reserved_count}" -ge "${MAX_PROCS_PER_GPU}" ]]; then
      echo "[SKIP] gpu=${GPU} active_compute=${active_count} planned=${planned_count} max=${MAX_PROCS_PER_GPU} run=${RUN_NAME}" >&2
      continue
    fi
  else
    active_count=0
    planned_count=0
    reserved_count=0
  fi

  CMD=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${RUN_NAME}"
    --output_dir "${RUNS_ROOT}/${RUN_NAME}"
    --log_dir "${LOG_ROOT}/${RUN_NAME}"
    "${EXTRA_ARGS[@]}"
  )

  echo "[USBV] idx=${idx} gpu=${GPU} active_compute=${active_count} planned=${planned_count} reserved=${reserved_count} run=${RUN_NAME} dry_run=${DRY_RUN}"
  printf '[USBV-CMD]'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  mkdir -p "${LOG_ROOT}/${RUN_NAME}" "${RUNS_ROOT}/${RUN_NAME}"
  nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
  PID="$!"
  GPU_PLANNED_COUNTS["${GPU}"]=$((planned_count + 1))
  printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${RUNS_ROOT}/${RUN_NAME}" | tee -a "${PID_FILE}"
done
