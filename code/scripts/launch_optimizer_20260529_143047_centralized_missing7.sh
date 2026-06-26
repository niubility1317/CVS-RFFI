#!/usr/bin/env bash
set -euo pipefail

# Seven missing centralized candidates for optimizer_20260529_143047.
# CEN_C01 / CEN_SA75 was already launched by optimizer_20260529_112412.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260529_143047_centralized_missing7}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

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
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
)

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

launch() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    "$@"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )

  echo "[CENTRALIZED-MISSING7] candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CENTRALIZED-MISSING7-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${ROOT}/train.py" ]]; then
    echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
    exit 2
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch CEN_C02 \
  CEN_C02_sa60_bridge_clear_lowrain_ce1p35_pudu72_eval81_r010 \
  0 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.35 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
sleep 2

launch CEN_C03 \
  CEN_C03_sa64_lowrain_swad_ce1p40_eval71_r010 \
  1 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.40 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_swad_ckpt \
  --swad_start_epoch 71 \
  --swad_interval 1 \
  --swad_tolerance 1.25
sleep 2

launch CEN_C04 \
  CEN_C04_sa63_balanced_view090_ce1p42_eval81_r010 \
  2 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.42 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.90 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
sleep 2

launch CEN_A05 \
  CEN_A05_sa64_fishr005_lowrain_ce1p45_eval81_r010 \
  3 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.45 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
sleep 2

launch CEN_A06 \
  CEN_A06_sa64_rxchain_aug_lowrain_ce1p35_r010 \
  4 \
  --primary_udu_weight 0.68 \
  --concat_sat_ce_weight 1.35 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 0.95 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --aug_p_rx_chain 0.20 \
  --aug_rx_chain_envs 6 \
  --aug_rx_chain_p_lowpass 0.80 \
  --aug_rx_chain_p_multipath 0.80
sleep 2

launch CEN_A07 \
  CEN_A07_sa64_all5_bridge_ce1p20_eval81_r010 \
  5 \
  --primary_udu_weight 0.68 \
  --concat_sat_ce_weight 1.20 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
sleep 2

launch CEN_A08 \
  CEN_A08_sa64_curric_lowrain_ce1p20_eval81_r010 \
  6 \
  --primary_udu_weight 0.68 \
  --concat_sat_ce_weight 1.20 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_schedule '1@1.00:clear_leo,low_elev_leo;61@0.95:low_elev_leo,rain_leo;121@0.80:rain_leo' \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
