#!/usr/bin/env bash
set -euo pipefail

# 16-run next batch for FSDG49B + stronger receiver_agnostic_bex02 + virtual-domain probes.
# Hard FL contract: --wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_ID="${RUN_ID:-20260601_220035_fsdg49b_ra_virtual_x2gpu}"
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
)

SAT_CVS_ARGS=(
  --use_sat_consistency
  --fl_sat_aug_mode cvs_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
)

SAT_LATE_CE_W025_R100=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 100
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.25
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

SAT_LATE_CE_W050_R120=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --sat_cons_start_epoch 120
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.50
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

CORAL_NONE=()

CORAL_ZID_003_R20=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.003
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

CORAL_ZID_001_R20=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.001
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

CORAL_ZID_002_R40=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.002
  --fl_coral_stage all
  --fl_coral_start_round 40
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

CORAL_ZID_003_R60=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.003
  --fl_coral_stage all
  --fl_coral_start_round 60
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

CORAL_ZDOM_0005_R20=(
  --use_fed_coral
  --lambda_fl_coral_zdom_global 0.0005
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_dom
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

STYLE_FORCED_PROBE_R80=(
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
  --fl_style_replay_start_round 80
  --fl_style_phys_start_round 80
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.00
  --fl_style_transform_mix_alpha 0.35
  --fl_style_min_remote_centroids 1
)

STYLE_LOW_REPLAY_R100=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy target_balanced
  --fl_style_replay_start_round 100
  --fl_style_phys_start_round 100
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.05
  --fl_style_transform_mix_alpha 0.25
  --fl_style_min_remote_centroids 1
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
)

STYLE_LOW_REPLAY_R120=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy target_balanced
  --fl_style_replay_start_round 120
  --fl_style_phys_start_round 120
  --fl_style_dg_start_round 999
  --fl_style_max_views 1
  --fl_style_replay_prob 0.10
  --fl_style_transform_mix_alpha 0.20
  --fl_style_min_remote_centroids 1
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 8
  --fl_style_zdom_probe_max_examples 8
)

VMB_BASE=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
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
  --use_sat_consistency
  --fl_sat_aug_mode cvs_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
  --lambda_vmb_tx_proto 0.10
  --lambda_vmb_rx_proto 0.10
  --fl_vmb_pretrain_rounds 40
  --fl_vmb_stage1_local_steps 4
  --fl_vmb_stage1_objective receiver_style_pretrain
  --fl_vmb_stage1_use_aux_losses
)

RUN_NAMES=(
  FSDG49B_R01_repro_zidcoral003_r010
  FSDG49B_R02_zidcoral000_control_r010
  FSDG49B_R03_zidcoral001_r010
  FSDG49B_R04_zidcoral003_start60_r010
  RA_BEX02_R05_rxadv075_r010
  RA_BEX02_R06_grl075_r010
  RA_BEX02_R07_fedprox_mu005_r010
  RA_BEX02_R08_localepoch3_r010
  SATCE_R09_late_w025_R100_r010
  SATCE_R10_late_w050_R120_r010
  VIRT_R11_style_stats_forced_probe_R80_r010
  VIRT_R12_style_replay_p005_R100_r010
  VIRT_R13_style_replay_p010_R120_r010
  VIRT_R14_zdom_coral0005_forced_probe_r010
  VMB_R15_bpc4_no_cen_r010
  VMB_R16_bpc8_no_cen_r010
)

RUN_GPUS=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)

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
    FSDG49B_R01_repro_zidcoral003_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    FSDG49B_R02_zidcoral000_control_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_NONE[@]}")
      ;;
    FSDG49B_R03_zidcoral001_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_001_R20[@]}")
      ;;
    FSDG49B_R04_zidcoral003_start60_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R60[@]}")
      ;;
    RA_BEX02_R05_rxadv075_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 0.75 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    RA_BEX02_R06_grl075_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 0.75 "${CORAL_ZID_003_R20[@]}")
      ;;
    RA_BEX02_R07_fedprox_mu005_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.005 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    RA_BEX02_R08_localepoch3_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 3 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    SATCE_R09_late_w025_R100_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_LATE_CE_W025_R100[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    SATCE_R10_late_w050_R120_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_LATE_CE_W050_R120[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}")
      ;;
    VIRT_R11_style_stats_forced_probe_R80_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}" "${STYLE_FORCED_PROBE_R80[@]}")
      ;;
    VIRT_R12_style_replay_p005_R100_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}" "${STYLE_LOW_REPLAY_R100[@]}")
      ;;
    VIRT_R13_style_replay_p010_R120_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZID_003_R20[@]}" "${STYLE_LOW_REPLAY_R120[@]}")
      ;;
    VIRT_R14_zdom_coral0005_forced_probe_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" --fl_local_epochs 2 --fedprox_mu 0.01 --lambda_rx_adv 1.0 --grl_lambda 1.0 "${CORAL_ZDOM_0005_R20[@]}" "${STYLE_FORCED_PROBE_R80[@]}")
      ;;
    VMB_R15_bpc4_no_cen_r010)
      EXTRA_ARGS=("${VMB_BASE[@]}" --fl_vmb_batches_per_client 4)
      ;;
    VMB_R16_bpc8_no_cen_r010)
      EXTRA_ARGS=("${VMB_BASE[@]}" --fl_vmb_batches_per_client 8)
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
        echo "[ERROR] Use a new RUN_ID, or set ALLOW_DUPLICATE_RUN=1 only for an intentional recovery." >&2
        exit 5
      fi
    fi
    if [[ -e "${LOG_PATH}" || -e "${LOG_ROOT}/${RUN_NAME}" || -e "${RUNS_ROOT}/${RUN_NAME}" ]]; then
      echo "[ERROR] existing log/output path for run=${RUN_NAME}; refusing duplicate launch for RUN_ID=${RUN_ID}." >&2
      echo "[ERROR] log=${LOG_PATH}" >&2
      echo "[ERROR] log_dir=${LOG_ROOT}/${RUN_NAME}" >&2
      echo "[ERROR] output_dir=${RUNS_ROOT}/${RUN_NAME}" >&2
      echo "[ERROR] Use a new RUN_ID or archive the exact run outputs before any intentional recovery; ALLOW_DUPLICATE_RUN does not overwrite logs." >&2
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

  echo "[X2GPU] idx=${idx} gpu=${GPU} active_compute=${active_count} planned=${planned_count} reserved=${reserved_count} run=${RUN_NAME} dry_run=${DRY_RUN}"
  printf '[X2GPU-CMD]'
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
