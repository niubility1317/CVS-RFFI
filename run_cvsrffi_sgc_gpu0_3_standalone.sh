#!/usr/bin/env bash
set -euo pipefail

cd ~/2510044040/CV-SincNet
mkdir -p logs finalist_runs

# ============================================================
# Standalone CVS-RFFI SGC queue
# - Uses root train.py / model_dual_cvsincnet.py / sgc_adapter.py
# - Does NOT use baselines/*
# - Does NOT require run_final_best_sgc_queue.sh
# - Default GPUs: 0,1,2,3
# ============================================================

if [ ! -f "train.py" ]; then
  echo "[ERROR] train.py not found. Please put this script in ~/2510044040/CV-SincNet." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GLOBAL_SEED="${GLOBAL_SEED:-1337}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAT_SCENARIO="${SAT_SCENARIO:-simple_leo}"
SAT_EVAL_ON="${SAT_EVAL_ON:-all}"
SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-simple_leo}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"
RUN_ROOT="${RUN_ROOT:-finalist_runs}"
LOG_DIR="${LOG_DIR:-logs}"
QUEUE_DIR="${QUEUE_DIR:-${RUN_ROOT}/queue_state_cvsrffi_sgc_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_NEW_SGC="${RUN_NEW_SGC:-0}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}"
QUEUE_FILE="${QUEUE_DIR}/jobs.tsv"
LOCK_FILE="${QUEUE_DIR}/jobs.lock"
: > "${QUEUE_FILE}"

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "${x}"
}

resolve_source_ckpt() {
  if [ -n "${SOURCE_CKPT:-}" ]; then
    if [ -f "${SOURCE_CKPT}" ]; then
      printf '%s\n' "${SOURCE_CKPT}"
      return 0
    fi
    echo "[ERROR] SOURCE_CKPT was set but not found: ${SOURCE_CKPT}" >&2
    return 1
  fi

  local candidates=(
    "finalist_runs/D1_domain_enhancer_off_seed1337/best_model_primary_ood.pth"
    "finalist_runs/D1_domain_enhancer_off/best_model_primary_ood.pth"
    "finalist_runs/D4_domain_no_pa_no_stats_seed1337/best_model_primary_ood.pth"
    "finalist_runs/B4_A2_ls002_seed1337/best_model_primary_ood.pth"
    "finalist_runs/B1_A1_mild_seed1337/best_model_primary_ood.pth"
    "finalist_runs/A1_fishr_sat_mild_v1/best_model_primary_ood.pth"
  )

  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  path="$(find finalist_runs -maxdepth 2 -type f -name 'best_model_primary_ood.pth' 2>/dev/null | sort | head -n 1 || true)"
  if [ -n "${path}" ] && [ -f "${path}" ]; then
    printf '%s\n' "${path}"
    return 0
  fi

  echo "[ERROR] Could not find a source checkpoint under finalist_runs." >&2
  echo "[HINT] Set SOURCE_CKPT=/path/to/best_model_primary_ood.pth and rerun." >&2
  return 1
}

SOURCE="$(resolve_source_ckpt)"

residual_only_std='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
no_res_control='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":false,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35}'
full_sgc_mild='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":true,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'

# Requires updated sgc_adapter.py. Disabled by default for old server code compatibility.
residual_bounded='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_mode":"plain","residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0,"residual_max_gamma":0.20,"residual_dropout":0.05}'
residual_multiscale='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_mode":"multiscale","residual_channels":32,"residual_kernel_sizes":[3,5,9],"residual_dilations":[1,2,4],"residual_init_gamma":0.0,"residual_max_gamma":0.20,"residual_dropout":0.05}'
residual_msgated='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_mode":"gated_multiscale","residual_channels":32,"residual_kernel_sizes":[3,5,9],"residual_dilations":[1,2,4],"residual_init_gamma":0.0,"residual_max_gamma":0.15,"residual_dropout":0.05,"residual_stat_gate":true}'
fpcr_sgc='{"adapter_mode":"fpcr","fpcr_shrinkage":0.35,"fpcr_cepstral_lifter":8,"fpcr_occupied_band_fraction":0.70,"fpcr_log_correction_clip":1.25,"fpcr_use_learned_residual":true,"fpcr_residual_channels":24,"fpcr_residual_blocks":2,"fpcr_residual_kernel_size":5,"fpcr_max_residual_ratio":0.06,"fpcr_residual_max_gamma":0.25,"fpcr_residual_init_gamma":0.0,"fpcr_residual_dropout":0.05}'

# kind | name | adapter_json | extra args
{
  printf 'cont\tE0_no_adapter_continue\t{}\t--slim_group rxrobust_lite_b_no_dac_mix015\n'
  printf 'sgc\tE1_residual_only_std\t%s\t\n' "${residual_only_std}"
  printf 'sgc\tE2_residual_only_std_res001\t%s\t--lambda_res 0.01\n' "${residual_only_std}"
  printf 'sgc\tE3_no_res_control\t%s\t\n' "${no_res_control}"
  printf 'sgc\tE4_full_sgc_mild_res001\t%s\t--lambda_res 0.01\n' "${full_sgc_mild}"
  if [ "${RUN_NEW_SGC}" = "1" ]; then
    printf 'sgc\tE5_residual_bounded_res001\t%s\t--lambda_res 0.01\n' "${residual_bounded}"
    printf 'sgc\tE6_residual_multiscale_res001\t%s\t--lambda_res 0.01\n' "${residual_multiscale}"
    printf 'sgc\tE7_residual_msgated_res001\t%s\t--lambda_res 0.01\n' "${residual_msgated}"
    printf 'sgc\tE8_fpcr_sgc_phys_budget\t%s\t--lambda_res 0.01 --lambda_fpcr_phys 0.02 --lambda_fpcr_budget 0.50\n' "${fpcr_sgc}"
  fi
} > "${QUEUE_FILE}"

claim_next_job() {
  (
    flock -x 9
    if [ ! -s "${QUEUE_FILE}" ]; then
      exit 1
    fi
    head -n 1 "${QUEUE_FILE}"
    tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
  ) 9>"${LOCK_FILE}"
}

run_job() {
  local gpu="$1"
  local kind="$2"
  local name="$3"
  local adapter_json="$4"
  local extra="${5:-}"
  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u train.py
    --dataset wisig
    --wisig_domain rx_day
    --batch_size "${BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --stage sgc_augment
    --source_ckpt "${SOURCE}"
    --train_sat_channel
    --train_sat_scenario "${SAT_SCENARIO}"
    --sat_view_source main
    --epochs "${EPOCHS}"
    --seed "${GLOBAL_SEED}"
    --lr "${LR}"
    --sat_cons_start_epoch 20
    --lambda_sat_cls 0.08
    --lambda_sat_cons 0.04
    --lambda_fishr 0.02
    --fishr_min_domains 4
    --eval_sat_channel
    --eval_sat_on "${SAT_EVAL_ON}"
    --eval_sat_scenarios "${SAT_EVAL_SCENARIOS}"
    --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_model_val.pth"
    --best_primary_save_path "${run_dir}/best_model_primary_ood.pth"
    --best_test_save_path "${run_dir}/best_model_test_overall.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_model_strict_udu.pth"
    --best_worst_rx_save_path "${run_dir}/best_model_worst_rx.pth"
  )

  if [ "${kind}" = "cont" ]; then
    cmd+=(--slim_group rxrobust_lite_b_no_dac_mix015)
  else
    cmd+=(--slim_group rxrobust_lite_b_no_dac_mix015 --sgc_adapter_kwargs "${adapter_json}")
  fi

  if [ -n "${extra}" ]; then
    # shellcheck disable=SC2206
    local extra_args=( ${extra} )
    cmd+=("${extra_args[@]}")
  fi

  echo "[QUEUE][GPU ${gpu}] start ${name} source=${SOURCE} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log_path}" 2>&1
  echo "[QUEUE][GPU ${gpu}] done ${name}"
}

worker_loop() {
  local gpu="$1"
  local line kind name adapter_json extra
  while true; do
    if ! line="$(claim_next_job)"; then
      break
    fi
    IFS=$'\t' read -r kind name adapter_json extra <<< "${line}"
    run_job "${gpu}" "${kind}" "${name}" "${adapter_json}" "${extra:-}"
  done
}

echo "[START] standalone CVS-RFFI SGC queue"
echo "[INFO] cwd=$(pwd)"
echo "[INFO] source_ckpt=${SOURCE}"
echo "[INFO] gpu_ids=${GPU_IDS}"
echo "[INFO] epochs=${EPOCHS} lr=${LR} seed=${GLOBAL_SEED}"
echo "[INFO] sat_train=${SAT_SCENARIO}"
echo "[INFO] sat_eval=${SAT_EVAL_SCENARIOS} on=${SAT_EVAL_ON} max_batches=${SAT_EVAL_MAX_BATCHES}"
echo "[INFO] run_new_sgc=${RUN_NEW_SGC}"
echo "[INFO] queue_file=${QUEUE_FILE}"

pids=()
for raw_gpu in ${GPU_IDS//,/ }; do
  gpu="$(trim "${raw_gpu}")"
  [ -n "${gpu}" ] || continue
  worker_loop "${gpu}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[DONE] standalone CVS-RFFI SGC queue finished."
