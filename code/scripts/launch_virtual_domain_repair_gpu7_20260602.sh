#!/usr/bin/env bash
set -euo pipefail

# GPU7 replacement runs after stopping VMB_R15/R16.
# Goal: improve and diagnose virtual receiver domains before returning to VMB/CEN stacking.
# Hard FL contract: --wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_ID="${RUN_ID:-20260602_0145_virtual_domain_repair_gpu7}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_RUN="${ONLY_RUN:-}"
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
  --fl_local_epochs 2
  --fedprox_mu 0.01
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

CORAL_ZID_GLOBAL001_VIRTUAL001=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.001
  --lambda_fl_coral_zid_virtual 0.001
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

CORAL_ZID_VIRTUAL001_ONLY=(
  --use_fed_coral
  --lambda_fl_coral_zid_virtual 0.001
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

STYLE_ULTRASOFT_P005_R80=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy target_balanced
  --fl_style_replay_start_round 80
  --fl_style_phys_start_round 80
  --fl_style_dg_start_round 999
  --fl_style_dg_min_domains 2
  --fl_style_max_views 1
  --fl_style_replay_prob 0.05
  --fl_style_transform_mix_alpha 0.10
  --fl_style_min_remote_centroids 4
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 16
  --fl_style_zdom_probe_max_examples 8
  --fl_style_bank_max_centroids 128
  --fl_style_phys_jitter_scale 0.10
  --fl_style_phys_max_noise_std 0.0
  --fl_style_phys_p_lowpass 0.0
  --fl_style_phys_p_multipath 0.0
  --use_style_collab_eval
  --style_collab_views 1
  --style_collab_fusion conservative
  --style_collab_base_weight 2.0
  --style_collab_max_aux_weight 0.20
)

STYLE_REALMIX16_UPPER_R80=(
  --use_fed_style_bank
  --use_fl_style_bank_stats
  --fl_style_domain_label_mode target_receiver
  --fl_style_sampling_policy target_balanced
  --fl_style_replay_start_round 80
  --fl_style_phys_start_round 80
  --fl_style_dg_start_round 120
  --fl_style_dg_min_domains 2
  --fl_style_max_views 1
  --fl_style_replay_prob 0.25
  --fl_style_transform_mix_alpha 0.10
  --fl_style_min_remote_centroids 4
  --fl_style_real_mix_samples 16
  --fl_style_real_mix_start_round 80
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_force_batch
  --fl_style_zdom_probe_real_samples 16
  --fl_style_zdom_probe_max_examples 8
  --fl_style_bank_max_centroids 128
  --fl_style_phys_jitter_scale 0.10
  --fl_style_phys_max_noise_std 0.0
  --fl_style_phys_p_lowpass 0.0
  --fl_style_phys_p_multipath 0.0
  --use_style_collab_eval
  --style_collab_views 1
  --style_collab_fusion conservative
  --style_collab_base_weight 2.0
  --style_collab_max_aux_weight 0.20
)

RUN_NAMES=(
  FSDG49B_GPU7_R15_style_tbal_ultrasoft_p005_R80_r010
  FSDG49B_GPU7_R16_style_realmix16_upper_R80_r010
)

RUN_GPUS=(7 7)

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

if [[ "${DRY_RUN}" == "1" ]]; then
  active_count=0
else
  if ! active_pids="$("${NVIDIA_SMI_BIN}" --id=7 --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
    echo "[ERROR] nvidia-smi query failed for gpu=7; refusing to launch without occupancy proof." >&2
    printf '%s\n' "${active_pids}" >&2
    exit 4
  fi
  active_count="$(printf '%s\n' "${active_pids}" | sed '/^$/d' | wc -l | tr -d ' ')"
fi
planned_count=0

for idx in "${!RUN_NAMES[@]}"; do
  RUN_NAME="${RUN_NAMES[$idx]}"
  GPU="${RUN_GPUS[$idx]}"
  if [[ -n "${ONLY_RUN}" && "${ONLY_RUN}" != "${RUN_NAME}" ]]; then
    continue
  fi

  EXTRA_ARGS=()
  case "${RUN_NAME}" in
    FSDG49B_GPU7_R15_style_tbal_ultrasoft_p005_R80_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" "${CORAL_ZID_GLOBAL001_VIRTUAL001[@]}" "${STYLE_ULTRASOFT_P005_R80[@]}")
      ;;
    FSDG49B_GPU7_R16_style_realmix16_upper_R80_r010)
      EXTRA_ARGS=("${FSDG_RA_CORE[@]}" "${SAT_CVS_ARGS[@]}" "${CORAL_ZID_VIRTUAL001_ONLY[@]}" "${STYLE_REALMIX16_UPPER_R80[@]}")
      ;;
    *)
      echo "[ERROR] no case for run: ${RUN_NAME}" >&2
      exit 3
      ;;
  esac

  LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
  OUTPUT_DIR="${RUNS_ROOT}/${RUN_NAME}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${ALLOW_DUPLICATE_RUN}" != "1" ]]; then
      if awk -F '\t' -v run="${RUN_NAME}" 'NR > 1 && $1 == run { found = 1 } END { exit(found ? 0 : 1) }' "${PID_FILE}"; then
        echo "[ERROR] run=${RUN_NAME} is already recorded in ${PID_FILE}; refusing duplicate launch." >&2
        exit 5
      fi
    fi
    if [[ -e "${LOG_PATH}" || -e "${OUTPUT_DIR}" ]]; then
      echo "[ERROR] existing log/output path for run=${RUN_NAME}; use a new RUN_ID." >&2
      echo "[ERROR] log=${LOG_PATH}" >&2
      echo "[ERROR] output_dir=${OUTPUT_DIR}" >&2
      exit 6
    fi
    reserved_count=$((active_count + planned_count))
    if [[ "${reserved_count}" -ge "${MAX_PROCS_PER_GPU}" ]]; then
      echo "[SKIP] gpu=${GPU} active_compute=${active_count} planned=${planned_count} max=${MAX_PROCS_PER_GPU} run=${RUN_NAME}" >&2
      continue
    fi
  fi

  CMD=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${RUN_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${LOG_ROOT}/${RUN_NAME}"
    "${EXTRA_ARGS[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[GPU7-VDOM-CMD] gpu=%s run=%s' "${GPU}" "${RUN_NAME}"
    printf ' %q' "${CMD[@]}"
    printf '\n'
  else
    printf '[GPU7-VDOM] gpu=%s run=%s log=%s output=%s\n' "${GPU}" "${RUN_NAME}" "${LOG_PATH}" "${OUTPUT_DIR}"
    nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
    PID="$!"
    printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${OUTPUT_DIR}" >> "${PID_FILE}"
    planned_count=$((planned_count + 1))
    sleep 2
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "[GPU7-VDOM] pid_file=${PID_FILE}"
fi
