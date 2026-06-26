#!/usr/bin/env bash
set -euo pipefail

# Clean 2x2 CE/GRL experiment for centralized vs federated receiver-client training.
# Hard FL contract: --wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver
#
# Matrix:
#   CEN_CE   : centralized TX CE only
#   CEN_GRL  : centralized TX CE + receiver GRL via lambda_adv
#   FED_CE   : FedAvg receiver clients, local TX CE only
#   FED_GRL  : FedAvg receiver clients, local TX CE + receiver GRL via lambda_rx_adv

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
RUN_ID="${RUN_ID:-ce_grl_central_vs_fed_20260603}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_INDEX="${ONLY_INDEX:-}"
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

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 4 )); then
  echo "[ERROR] GPU_IDS must provide at least four comma-separated GPU ids, got: ${GPU_IDS}" >&2
  exit 2
fi

if [[ ! -f "${TRAIN_SCRIPT}" && -f "${ROOT}/train.py" ]]; then
  TRAIN_SCRIPT="${ROOT}/train.py"
fi

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

gpu_process_count() {
  local gpu="$1"
  local out
  if ! out="$("${NVIDIA_SMI_BIN}" --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "[ERROR] nvidia-smi query failed for gpu=${gpu}; refusing to launch" >&2
    return 42
  fi
  printf '%s\n' "${out}" | sed '/^$/d' | wc -l | tr -d ' '
}

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio 0.9
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --epochs 200
  --fl_rounds 200
  --fl_client_key receiver
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
  --exp_group s1_core_only
  --slim_group none
  --label_smoothing 0.0
  --force_ce_grl_only
  --no_use_aug
  --no_use_mixstyle
  --no_use_sat_consistency
  --no_use_concat_sat_channel_aug
  --no_use_proto_memory
  --no_enable_pa_aux
  --no_enable_dac_aux
  --no_use_fed_proto_stats
  --no_use_fed_coral
  --no_use_fed_style_bank
  --no_use_fl_style_bank_stats
  --no_use_tx_adv_on_zdom
  --lambda_dom 0
  --lambda_orth 0
  --lambda_cons 0
  --lambda_group_ce 0
  --lambda_fishr 0
  --lambda_proto 0
  --lambda_supcon_id 0
  --lambda_sat_cls 0
  --lambda_sat_cons 0
  --lambda_cls_pa 0
  --lambda_cls_dac 0
  --lambda_pa_joint_inv 0
  --lambda_pa_kl 0
  --lambda_dac_reg 0
  --lambda_pa_reg 0
  --lambda_fed_proto 0
  --lambda_fed_coral 0
  --lambda_fed_coral_virtual 0
  --lambda_fl_coral_zid_global 0
  --lambda_fl_coral_zid_virtual 0
  --lambda_fl_coral_zdom_global 0
  --lambda_vmb_tx_proto 0
  --lambda_vmb_rx_proto 0
  --lambda_tx_adv_r 0
  --lambda_logit_kd 0
  --concat_sat_ce_weight 0
  --fl_baseline_view_ce_weight 0
  --grl_lambda 1.0
  --primary_udu_weight 0.70
  --eval_max_batches 0
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
)

RUN_NAMES=(
  CEGRL_CEN_CE_r010
  CEGRL_CEN_GRL_r010
  CEGRL_FEDAVG_CE_receiver_r010
  CEGRL_FEDAVG_GRL_receiver_r010
)

declare -A PLANNED_PER_GPU=()

if [[ "${DRY_RUN}" != "1" ]]; then
  cd "${ROOT}"
  if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] train script not found: ${TRAIN_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${WISIG_PKL}" ]]; then
    echo "[ERROR] WiSig pickle not found: ${WISIG_PKL}" >&2
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

for idx in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$idx]}"
  gpu="${GPU_ARRAY[$idx]}"
  if [[ -n "${ONLY_INDEX}" && "${ONLY_INDEX}" != "${idx}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_RUN}" && "${ONLY_RUN}" != "${run_name}" ]]; then
    continue
  fi

  route_args=()
  case "${run_name}" in
    CEGRL_CEN_CE_r010)
      route_args=(
        --train_mode centralized
        --lambda_adv 0
        --lambda_rx_adv 0
        --rx_weight 0
        --test_eval_policy every_epoch
        --test_eval_start_epoch 171
      )
      ;;
    CEGRL_CEN_GRL_r010)
      route_args=(
        --train_mode centralized
        --lambda_adv 1.0
        --lambda_rx_adv 0
        --rx_weight 0
        --test_eval_policy every_epoch
        --test_eval_start_epoch 171
      )
      ;;
    CEGRL_FEDAVG_CE_receiver_r010)
      route_args=(
        --train_mode fedavg
        --fl_local_objective ce
        --fl_local_epochs 1
        --fl_clients_per_round 1.0
        --fl_test_eval_interval 10
        --fl_test_eval_last_n 5
        --lambda_adv 0
        --lambda_rx_adv 0
        --rx_weight 0
      )
      ;;
    CEGRL_FEDAVG_GRL_receiver_r010)
      route_args=(
        --train_mode fedavg
        --fl_local_objective receiver_agnostic_bex02
        --fl_local_epochs 1
        --fl_clients_per_round 1.0
        --fl_test_eval_interval 10
        --fl_test_eval_last_n 5
        --lambda_adv 0
        --lambda_rx_adv 1.0
        --rx_weight 1.0
      )
      ;;
    *)
      echo "[ERROR] unknown run: ${run_name}" >&2
      exit 2
      ;;
  esac

  run_dir="${RUNS_ROOT}/${run_name}"
  log_path="${LOG_ROOT}/${run_name}.out"
  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}"
    "${THREAD_ENV[@]}"
    "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_ARGS[@]}"
    "${route_args[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --log_dir "${LOG_ROOT}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )

  echo "[CE-GRL] idx=${idx} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CE-GRL-CMD] '
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  if [[ "${ALLOW_DUPLICATE_RUN}" != "1" ]]; then
    if [[ -e "${run_dir}" || -e "${log_path}" ]] || grep -q "^${run_name}[[:space:]]" "${PID_FILE}" 2>/dev/null; then
      printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
        | tee -a "${LOG_ROOT}/blocked.tsv"
      echo "[BLOCKED] run=${run_name} path collision or prior PID record; set ALLOW_DUPLICATE_RUN=1 only for deliberate relaunch" >&2
      continue
    fi
  fi

  active="$(gpu_process_count "${gpu}")"
  planned="${PLANNED_PER_GPU[$gpu]:-0}"
  if (( active + planned >= MAX_PROCS_PER_GPU )); then
    printf "%s\t%s\t%s\tactive=%s planned=%s max=%s\n" \
      "${run_name}" "${gpu}" "BLOCKED_CAPACITY" "${active}" "${planned}" "${MAX_PROCS_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    echo "[SKIP] gpu=${gpu} active_compute=${active} planned=${planned} max=${MAX_PROCS_PER_GPU}" >&2
    continue
  fi
  PLANNED_PER_GPU[$gpu]=$((planned + 1))

  mkdir -p "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${PID_FILE}"
done

