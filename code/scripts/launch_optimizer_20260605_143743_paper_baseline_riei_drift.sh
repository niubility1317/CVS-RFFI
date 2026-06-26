#!/usr/bin/env bash
set -euo pipefail

# Paper-baseline optimizer for DRIFT and RIEI-FD.
# The default candidate in each family preserves the current paper reproduction
# settings; non-default candidates explicitly turn on stabilization/search knobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python" ]]; then
    PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
  else
    PYTHON="python"
  fi
fi

RUN_ID="${RUN_ID:-optimizer_20260605_143743_paper_baseline_riei_drift}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --only=*)
      ONLY_CANDIDATE="${arg#--only=}"
      ;;
    *)
      echo "[ERROR] unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" != "1" && ! -f "${WISIG_PKL}" ]]; then
  echo "[ERROR] WISIG_PKL not found: ${WISIG_PKL}" >&2
  exit 2
fi

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

run_cmd() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  local run_dir="$4"
  local log_path="$5"
  shift 5
  local cmd=("$@")

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
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
  local launched_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${launched_pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

pause_between_candidates() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    sleep 2
  fi
}

append_drift_day1_args() {
  CMD+=(
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol drift_day1
    --wisig_equalized 1
    --wisig_domain rx_day
    --wisig_out_len 256
    --wisig_train_ratio 0.1
    --wisig_val_ratio 0.9
    --wisig_guard_gap 8
    --wisig_train_days 0
    --wisig_test_days 0
    --wisig_train_rxs 1-1,14-7,7-7
    --wisig_test_rxs 1-19,19-2,2-1,2-19,20-1,7-14,8-8
    --wisig_paper_day 0
    --wisig_paper_train_samples_per_combo 800
    --wisig_paper_val_samples_per_combo 200
    --wisig_paper_test_samples_per_combo 200
    --batch_size 64
    --eval_batch_size 256
    --num_workers 0
    --prefetch_factor 2
    --seed 1337
  )
}

append_riei_original_row1_args() {
  CMD+=(
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol riei_original
    --wisig_equalized 1
    --wisig_domain rx_day
    --wisig_out_len 256
    --wisig_train_ratio 0.1
    --wisig_val_ratio 0.9
    --wisig_guard_gap 8
    --wisig_train_days 0,1,2,3
    --wisig_test_days 0,1,2,3
    --wisig_train_rxs 1-1,7-7
    --wisig_test_rxs 1-19
    --wisig_paper_day 0
    --wisig_paper_train_samples_per_combo 2400
    --wisig_paper_val_samples_per_combo 800
    --wisig_paper_test_samples_per_combo 800
    --batch_size 64
    --eval_batch_size 256
    --num_workers 0
    --prefetch_factor 2
    --seed 1337
  )
}

launch_drift_day1() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3
  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  CMD=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${ROOT}/baselines/drift/train_cvs.py"
    --epochs 200
    --lr 1e-4
    --embedding_dim 512
    --split_dim 256
    --lambda_grl 1.0
    --lambda_center 0.01
    --lambda_mse 0.02
    --grl_schedule constant
    --paper_eval_last_n 5
    --paper_eval_name test_seen_day_unseen_rx
    --test_on_val_improve
    --output_dir "${run_dir}"
  )
  append_drift_day1_args
  CMD+=("$@")
  echo "[PAPER-BASELINE-OPT] candidate=${candidate_id} family=drift_day1 run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[PAPER-BASELINE-OPT-CMD]'
  print_cmd "${CMD[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${CMD[@]}"
}

launch_riei_day1() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3
  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  CMD=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${ROOT}/baselines/riei_fd/train_cvs.py"
    --epochs 200
    --lr_all 1e-4
    --lr_fed 1e-4
    --feature_dim 512
    --lambda_mi 1.2
    --lambda_ie 1.2
    --paper_eval_last_n 10
    --paper_eval_name test_seen_day_unseen_rx
    --test_on_val_improve
    --output_dir "${run_dir}"
  )
  append_drift_day1_args
  CMD+=("$@")
  echo "[PAPER-BASELINE-OPT] candidate=${candidate_id} family=riei_drift_day1 run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[PAPER-BASELINE-OPT-CMD]'
  print_cmd "${CMD[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${CMD[@]}"
}

launch_riei_original_row1() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3
  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  CMD=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${ROOT}/baselines/riei_fd/train_cvs.py"
    --epochs 200
    --lr_all 1e-4
    --lr_fed 1e-4
    --feature_dim 512
    --lambda_mi 1.2
    --lambda_ie 1.2
    --paper_eval_last_n 10
    --paper_eval_name test_seen_day_unseen_rx
    --test_on_val_improve
    --output_dir "${run_dir}"
  )
  append_riei_original_row1_args
  CMD+=("$@")
  echo "[PAPER-BASELINE-OPT] candidate=${candidate_id} family=riei_original_row1 run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[PAPER-BASELINE-OPT-CMD]'
  print_cmd "${CMD[@]}"
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${CMD[@]}"
}

cd "${ROOT}"

launch_drift_day1 DRIFT_D01 DRIFT_D01_paper_raw_control 0
pause_between_candidates
launch_drift_day1 DRIFT_D02 DRIFT_D02_mse_mean_dim 1 --mse_reduction mean
pause_between_candidates
launch_drift_day1 DRIFT_D03 DRIFT_D03_mse_cap512 2 --mse_cap 512
pause_between_candidates
launch_drift_day1 DRIFT_D04 DRIFT_D04_mse_norm_sum 3 --normalize_features_for_mse
pause_between_candidates
launch_drift_day1 DRIFT_D05 DRIFT_D05_mse_norm_mean 4 --normalize_features_for_mse --mse_reduction mean
pause_between_candidates
launch_drift_day1 DRIFT_D06 DRIFT_D06_raw_cap2048_normreg 5 --mse_cap 2048 --lambda_feature_norm 1e-3
pause_between_candidates
launch_drift_day1 DRIFT_D07 DRIFT_D07_mse_mean_adamw_clip 6 --mse_reduction mean --optimizer adamw --weight_decay 1e-4 --grad_clip_norm 5
pause_between_candidates
launch_drift_day1 DRIFT_D08 DRIFT_D08_projection_mse_mean 7 --use_resnet_projection --mse_reduction mean
pause_between_candidates

launch_riei_day1 RIEI_D01 RIEI_D01_driftday1_paper_control 0
pause_between_candidates
launch_riei_day1 RIEI_D02 RIEI_D02_driftday1_crosscov 1 --mi_mode cross_cov
pause_between_candidates
launch_riei_day1 RIEI_D03 RIEI_D03_driftday1_cos2_temp2 2 --mi_mode cosine_square --ie_temperature 2.0
pause_between_candidates
launch_riei_day1 RIEI_D04 RIEI_D04_driftday1_dis2_clip 3 --disentangle_steps 2 --grad_clip_norm 5
pause_between_candidates
launch_riei_day1 RIEI_D05 RIEI_D05_driftday1_wd_norm 4 --weight_decay_all 1e-4 --weight_decay_fed 1e-4 --lambda_feature_norm 1e-3
pause_between_candidates
launch_riei_day1 RIEI_D06 RIEI_D06_driftday1_projection_crosscov 5 --use_resnet_projection --mi_mode cross_cov
pause_between_candidates

launch_riei_original_row1 RIEI_OR01 RIEI_OR01_table3_row1_paper_control 6
pause_between_candidates
launch_riei_original_row1 RIEI_OR02 RIEI_OR02_table3_row1_crosscov_temp2 7 --mi_mode cross_cov --ie_temperature 2.0

echo "[PAPER-BASELINE-OPT] finished run_id=${RUN_ID} dry_run=${DRY_RUN}"
