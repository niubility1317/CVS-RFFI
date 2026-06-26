#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen31_sinc_shared_stat16_20260605}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

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
  --test_eval_policy interval_final
  --test_eval_start_epoch 1
  --test_eval_interval 10
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
  --freq_feature_source raw_fft
  --pa_feature_source raw_iq
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

CANDIDATES=(
  "S01_sinc_shared_baseline_api|0|--freq_feature_source raw_fft --pa_feature_source raw_iq"
  "S02_freq_from_sinc_energy|1|--freq_feature_source sinc_energy --pa_feature_source raw_iq"
  "S03_freq_from_sinc_phase_asym|2|--freq_feature_source sinc_phase_asym --pa_feature_source raw_iq"
  "S04_pa_from_sinc_lowrank_135|3|--freq_feature_source raw_fft --pa_feature_source sinc_lowrank --pa_orders 1,3,5"
  "S05_pa_from_sinc_lowrank_15|4|--freq_feature_source raw_fft --pa_feature_source sinc_lowrank --pa_orders 1,5"
  "S06_time_pa_no_overlap_3rd|5|--freq_feature_source raw_fft --pa_feature_source raw_iq --pa_orders 1,5"
  "S07_fftless_freq_pa_joint|6|--freq_feature_source sinc_phase_asym --pa_feature_source sinc_lowrank --pa_orders 1,5"
  "S08_sinc_shared_channel_trim|7|--model_variant lite_e --freq_feature_source sinc_energy --pa_feature_source sinc_lowrank --pa_orders 1,5"
  "T09_no_rho_circularity|0|--no_use_circularity"
  "T10_no_freq_stats_proj|1|--no_use_freq_stats"
  "T11_no_pa_stats_proj|2|--no_use_pa_stats"
  "T12_no_spectral_aux_stats_all|3|--no_use_circularity --no_use_freq_stats --no_use_pa_stats"
  "T13_no_dsq_freq_stability|4|--domain_freq_stability_mode off"
  "T14_no_freq_band_gate|5|--no_use_freq_band_gate"
  "T15_domain_enhancer_off|6|--domain_enhancer off"
  "T16_rcn_minimal_6stats|7|--domain_enhancer rcn_minimal_6stats"
)

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"

if [[ "${#CANDIDATES[@]}" -ne 16 ]]; then
  echo "[ERROR] expected exactly 16 candidates, got ${#CANDIDATES[@]}" >&2
  exit 3
fi

for item in "${CANDIDATES[@]}"; do
  IFS='|' read -r name gpu extra <<< "${item}"
  read -r -a extra_args <<< "${extra}"
  run_name="CEN31SS_${name}_r010"
  run_dir="${RUNS_ROOT}/${run_name}"
  log_path="${LOG_ROOT}/${run_name}.out"
  cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_ARGS[@]}"
    "${MIXSTYLE_ARGS[@]}"
    "${CEN_A31_STACK_ARGS[@]}"
    "${extra_args[@]}"
    --run_name "${run_name}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )
  echo "[CEN31SS] candidate=${name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CEN31SS-CMD]'
  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    printf "%s\t%s\t%s\t%s\n" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    continue
  fi
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    printf "%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    continue
  fi
  mkdir -p "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
  sleep 2
done
