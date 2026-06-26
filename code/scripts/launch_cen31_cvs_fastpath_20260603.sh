#!/usr/bin/env bash
set -euo pipefail

# CEN31 CVS-RFFI fast-path replication.
# Same centralized CEN31 recipe as the architecture comparison, but launch only
# the optimized CVS-RFFI backbone into a separate run/log root.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen31_cvs_fastpath_20260603}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="${RUN_NAME:-CEN31FAST_cvsincnet_lited_r010}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "[ERROR] unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

COMMON_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 81
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
  --arch_family cvsincnet
)

MIXSTYLE_ARGS=(
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

CEN_A31_STACK_ARGS=(
  --primary_udu_weight 0.70
  --concat_sat_ce_weight 1.28
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --use_proto_memory
  --lambda_proto 0.015
  --proto_momentum 0.95
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --lambda_fishr 0.005
  --fishr_min_domains 4
  --generalization_feature z_id
)

cd "${ROOT}"

run_dir="${RUNS_ROOT}/${RUN_NAME}"
log_path="${LOG_ROOT}/${RUN_NAME}.out"
cmd=(
  env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
  "${COMMON_ARGS[@]}"
  --run_name "${RUN_NAME}"
  "${MIXSTYLE_ARGS[@]}"
  "${CEN_A31_STACK_ARGS[@]}"
  --latest_save_path "${run_dir}/latest_model.pth"
  --best_save_path "${run_dir}/best_val_model.pth"
  --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
  --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
  --ema_save_path "${run_dir}/ema_model.pth"
  --swa_save_path "${run_dir}/swa_model.pth"
  --swad_save_path "${run_dir}/swad_model.pth"
)

echo "[CEN31-FAST] run=${RUN_NAME} gpu=${GPU_ID} dry_run=${DRY_RUN}"
printf '[CEN31-FAST-CMD]'
print_cmd "${cmd[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] train script not found: ${TRAIN_SCRIPT}" >&2
  exit 2
fi
if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
  mkdir -p "${LOG_ROOT}"
  printf "%s\t%s\t%s\t%s\n" "${RUN_NAME}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/blocked.tsv"
  exit 0
fi

count="$(gpu_process_count "${GPU_ID}")"
if (( count >= MAX_TRAIN_PER_GPU )); then
  mkdir -p "${LOG_ROOT}"
  printf "%s\t%s\tgpu=%s active_count=%s max=%s\n" \
    "${RUN_NAME}" "BLOCKED_CAPACITY" "${GPU_ID}" "${count}" "${MAX_TRAIN_PER_GPU}" \
    | tee -a "${LOG_ROOT}/blocked.tsv"
  exit 0
fi

mkdir -p "${LOG_ROOT}" "${run_dir}"
nohup "${cmd[@]}" > "${log_path}" 2>&1 &
pid="$!"
printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU_ID}" "${pid}" "${log_path}" "${run_dir}" \
  | tee -a "${LOG_ROOT}/launch_pids.tsv"
