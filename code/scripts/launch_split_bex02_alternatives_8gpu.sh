#!/usr/bin/env bash
set -euo pipefail

# 8-GPU formal validation matrix for Split-BEX02 alternatives.
# Formal runs must keep ratio/round/client hard constraints aligned with AGENTS.md.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-split_bex02_alternatives_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT:-${ROOT}/logs/${RUN_ID}}}"
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

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_train_ratio 0.1
  --epochs 200
  --fl_rounds 200
  --fl_client_key receiver
  --fl_clients_per_round 1.0
  --fl_test_eval_interval 10
  --fl_test_eval_last_n 5
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --num_workers 0
  --batch_size 128
  --eval_batch_size 256
)

VMB_COMMON=(
  --train_mode fedcvs_vmb
  --fl_vmb_stage auto
  --fl_vmb_pretrain_rounds 20
  --fl_vmb_stage1_local_steps 2
  --fl_vmb_stage1_objective ce
  --fl_vmb_stage1_lr_mult 1.5
  --fl_vmb_batches_per_client 1
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --lambda_vmb_tx_proto 0.1
  --lambda_vmb_rx_proto 0.1
  --lambda_rx_adv 0.1
  --lambda_orth 0.1
)

declare -a RUNS=(
  "0|SBX02_LVMB_r010|${VMB_COMMON[*]} --fl_local_objective local_virtual_bex02 --use_fed_style_bank --fl_style_max_views 2 --fl_style_sampling_policy target_balanced --fl_style_transform_mix_alpha 0.7"
  "1|SBX02_PROTO_r010|${VMB_COMMON[*]} --fl_local_objective receiver_agnostic_bex02 --use_fed_proto_stats --lambda_fed_proto 0.05 --lambda_vmb_tx_proto 0.15 --lambda_vmb_rx_proto 0.15"
  "2|SBX02_FISHR_r010|${VMB_COMMON[*]} --fl_local_objective receiver_agnostic_bex02 --use_fed_style_bank --fl_conflict_agg cosine_clip --lambda_fishr 0.02 --fishr_min_domains 2"
  "3|SBX02_STYLE_r010|${VMB_COMMON[*]} --fl_local_objective receiver_agnostic_bex02 --use_fed_style_bank --fl_style_code_dim 8 --fl_style_max_views 2 --fl_style_sampling_policy target_balanced --fl_style_transform_mix_alpha 0.5"
  "4|SBX02_KDLOGIT_r010|${VMB_COMMON[*]} --fl_local_objective receiver_agnostic_bex02 --use_logit_anchors --lambda_logit_kd 0.05 --kd_temperature 2.0 --kd_reliability_gate 0.7 --kd_margin_min 0.1"
  "5|SBX02_QTOKEN_r010|--train_mode split_bex02 --fl_local_objective local_virtual_bex02 --fl_vmb_stage auto --fl_vmb_pretrain_rounds 20 --fl_vmb_stage1_local_steps 2 --fl_vmb_stage1_objective ce --fl_vmb_stage1_lr_mult 1.5 --fl_vmb_batches_per_client 1 --fl_vmb_domain_balanced_sampling --fl_vmb_domain_balanced_aggregation --fl_vmb_transmitter_balanced_batch --lambda_vmb_tx_proto 0.1 --lambda_vmb_rx_proto 0.1 --lambda_rx_adv 0.1 --lambda_orth 0.1 --activation_token_route quantized --token_quant_bits 4 --fl_conflict_agg cosine_clip --use_fed_style_bank --fl_style_code_dim 8"
  "6|SBX02_SATCE_r010|${VMB_COMMON[*]} --fl_local_objective receiver_agnostic_bex02 --use_sat_consistency --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
  "7|SBX02_COMBO_r010|--train_mode split_bex02 --fl_local_objective local_virtual_bex02 --fl_vmb_stage auto --fl_vmb_pretrain_rounds 20 --fl_vmb_stage1_local_steps 2 --fl_vmb_stage1_objective ce --fl_vmb_stage1_lr_mult 1.5 --fl_vmb_batches_per_client 1 --fl_vmb_domain_balanced_sampling --fl_vmb_domain_balanced_aggregation --fl_vmb_transmitter_balanced_batch --lambda_vmb_tx_proto 0.1 --lambda_vmb_rx_proto 0.1 --lambda_rx_adv 0.1 --lambda_orth 0.1 --use_fed_style_bank --fl_style_code_dim 8 --fl_style_sampling_policy target_balanced --fl_style_transform_mix_alpha 0.5 --use_fed_proto_stats --lambda_fed_proto 0.05 --use_logit_anchors --lambda_logit_kd 0.03 --activation_token_route quantized --token_quant_bits 8 --fl_conflict_agg cosine_clip --use_sat_consistency --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only"
)

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
echo "[SPLIT-BEX02-MATRIX] root=${ROOT} log_root=${LOG_ROOT} runs_root=${RUNS_ROOT} runs=${#RUNS[@]} enforce_one_run_per_gpu=${ENFORCE_ONE_RUN_PER_GPU}"

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
  cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "${PYTHON}" train.py "${COMMON_ARGS[@]}" --run_name "${run_name}" --output_dir "${out_dir}" --log_dir "${log_dir}" ${extra_args})
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
      mkdir -p "${log_dir}"
      nohup "${cmd[@]}" > "${LOG_ROOT}/${run_name}.out" 2>&1 &
      echo $! > "${LOG_ROOT}/${run_name}.pid"
    )
  fi
done
