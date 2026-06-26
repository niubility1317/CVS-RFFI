#!/usr/bin/env bash
set -euo pipefail

# 8-GPU VMB pretraining optimization matrix.
# Part A: longer Stage-1 pretraining. Part B: single-source DG-inspired Stage-1 variants.
# All runs keep the formal federated contract: WiSig train ratio 0.1, 200 rounds, receiver clients.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-vmb_pretrain_sdg_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
ENFORCE_ONE_RUN_PER_GPU="${ENFORCE_ONE_RUN_PER_GPU:-1}"
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
PID_FILE="${LOG_ROOT}/launch_pids.tsv"

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx_day
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
  --fl_test_eval_last_n 10
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
  --num_workers 0
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
)

VMB_PRETRAIN_SAT_COMMON=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_stage1_objective ce
  --fl_vmb_stage1_local_steps 1
  --fl_vmb_stage1_lr_mult 1.0
  --fl_vmb_batches_per_client 1
  --fl_vmb_server_lr 0.01
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --lambda_vmb_tx_proto 0.1
  --lambda_vmb_rx_proto 0.1
  --lambda_tx_adv_r 0.1
  --lambda_rx_adv 0.1
  --lambda_orth 0.1
  --use_tx_adv_on_zdom
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.7
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
)

declare -a RUNS=(
  "0|VPT_A40_satce_w07_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 40"
  "1|VPT_A60_satce_w07_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 60"
  "2|VPT_A80_satce_w07_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 80"
  "3|VPT_A100_satce_w07_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 100"
  "4|VPT_B60_proto015_satce_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_local_steps 2 --fl_vmb_stage1_lr_mult 1.5 --lambda_vmb_tx_proto 0.15 --lambda_vmb_rx_proto 0.15 --use_fed_proto_stats --lambda_fed_proto 0.05"
  "5|VPT_B60_mixstyle_satce_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 60 --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.4 --mixstyle_alpha 0.2 --mixstyle_strength 1.0"
  "6|VPT_B60_rfdr_satce_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 60 --use_aug --aug_p_rx_chain 0.5 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq"
  "7|VPT_B60_stylebank_satce_r010|${VMB_PRETRAIN_SAT_COMMON[*]} --fl_vmb_pretrain_rounds 60 --use_fed_style_bank --fl_style_code_dim 8 --fl_style_max_views 2 --fl_style_sampling_policy target_balanced --fl_style_transform_mix_alpha 0.5 --fl_style_replay_start_round 10 --fl_style_phys_start_round 10 --fl_style_dg_start_round 20"
)

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
printf "name\tgpu\tpid\tlog\toutput_dir\n" > "${PID_FILE}"
echo "[VMB-PRETRAIN-SDG] root=${ROOT} log_root=${LOG_ROOT} runs_root=${RUNS_ROOT} runs=${#RUNS[@]} enforce_one_run_per_gpu=${ENFORCE_ONE_RUN_PER_GPU}"

declare -A SEEN_GPUS=()
for row in "${RUNS[@]}"; do
  IFS="|" read -r gpu _run_name _extra_args <<< "${row}"
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "[ERROR] duplicate GPU assignment in RUNS: gpu=${gpu}" >&2
    exit 3
  fi
  SEEN_GPUS["${gpu}"]=1
done

if [[ "${#SEEN_GPUS[@]}" -ne 8 ]]; then
  echo "[ERROR] expected 8 unique GPU assignments, got ${#SEEN_GPUS[@]}" >&2
  exit 3
fi

for row in "${RUNS[@]}"; do
  IFS="|" read -r gpu run_name extra_args <<< "${row}"
  out_dir="${RUNS_ROOT}/${run_name}"
  log_dir="${LOG_ROOT}/${run_name}"
  cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py "${COMMON_ARGS[@]}" --run_name "${run_name}" --output_dir "${out_dir}" --log_dir "${log_dir}" ${extra_args})
  echo "[RUN] gpu=${gpu} name=${run_name}"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${ROOT}/train.py" ]]; then
      echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
      exit 2
    fi
    if [[ "${ENFORCE_ONE_RUN_PER_GPU}" == "1" ]]; then
      active_pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
      if [[ -n "${active_pids}" ]]; then
        echo "[ERROR] gpu=${gpu} already has compute process(es): ${active_pids}. Set ENFORCE_ONE_RUN_PER_GPU=0 only for an explicit override." >&2
        exit 4
      fi
    fi
    (
      cd "${ROOT}"
      mkdir -p "${log_dir}" "${out_dir}"
      nohup "${cmd[@]}" > "${LOG_ROOT}/${run_name}.out" 2>&1 &
      echo $! > "${LOG_ROOT}/${run_name}.pid"
    )
    pid="$(cat "${LOG_ROOT}/${run_name}.pid")"
    printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "${pid}" "${LOG_ROOT}/${run_name}.out" "${out_dir}" | tee -a "${PID_FILE}"
    sleep 2
  fi
done
