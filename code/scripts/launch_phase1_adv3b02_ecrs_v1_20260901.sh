#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
BASE_CHECKPOINT="${RUNS_ROOT}/ADV3B02_ECRS_R0/best.pth"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"
MAX_GPU_TRAIN_PROCS="${MAX_GPU_TRAIN_PROCS:-2}"
GPU_SLOT_POLL_SECONDS="${GPU_SLOT_POLL_SECONDS:-30}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

lower_pkl="$(printf '%s' "${WISIG_PKL}" | tr '[:upper:]' '[:lower:]')"
case "${lower_pkl}" in
  *manytx.pkl*|*manyrx.pkl*|*target*|*unknown*)
    echo "[ERROR] refusing non-source Phase1 WISIG_PKL: ${WISIG_PKL}" >&2
    exit 4
    ;;
esac
[[ "${lower_pkl}" == *manysig.pkl ]] || {
  echo "[ERROR] ADV3B02-ECRS-V1 requires source ManySig.pkl" >&2
  exit 4
}

candidate_enabled() {
  local candidate="$1"
  [[ -z "${ONLY_CANDIDATES}" || ",${ONLY_CANDIDATES}," == *",${candidate},"* ]]
}

wait_for_gpu_slot() {
  local gpu="$1" candidate="$2" active
  while true; do
    active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
    active="${active//[[:space:]]/}"
    if [[ "${active}" -lt "${MAX_GPU_TRAIN_PROCS}" ]]; then
      echo "[ECRS-V1-GPU-SLOT] id=${candidate} gpu=${gpu} active_before=${active} limit=${MAX_GPU_TRAIN_PROCS}"
      return 0
    fi
    echo "[ECRS-V1-GPU-WAIT] id=${candidate} gpu=${gpu} active=${active} limit=${MAX_GPU_TRAIN_PROCS} poll_s=${GPU_SLOT_POLL_SECONDS}"
    sleep "${GPU_SLOT_POLL_SECONDS}"
  done
}

build_common_cmd() {
  local gpu="$1" candidate="$2" out_dir="$3"
  common_cmd=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON}" -u "${ROOT}/code/train.py"
    --dataset wisig
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol cvs_day_rx
    --wisig_equalized 1
    --wisig_domain rx_day
    --wisig_out_len 256
    --wisig_train_days 1,2,3
    --wisig_test_days 0,1,2,3
    --wisig_train_rxs 1,3,4,6,8
    --wisig_test_rxs 0,2,5,7,9,10,11
    --wisig_split_strategy random
    --wisig_cap_strategy random
    --wisig_target_receiver_only_eval
    --wisig_max_test_per_combo 0
    --meta_ssl_max_samples_per_combo_source 0
    --seed 392005
    --use_meta_ssl_cvs
    --ssl_labeled_ratio 0.07
    --ssl_unlabeled_ratio 0.63
    --ssl_val_ratio 0.30
    --model_size M
    --model_variant lite_d
    --branch_ablation no_dac
    --domain_branch_ablation no_stats
    --epochs 200
    --test_eval_policy interval_final
    --test_eval_start_epoch 200
    --test_eval_interval 200
    --test_eval_final_window 0
    --test_eval_final_interval 0
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --concat_sat_start_epoch 1
    --concat_sat_ce_start_epoch 80
    --concat_sat_ce_weight 0.68
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
    --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --eval_sat_scenarios "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --run_name "${candidate}"
    --output_dir "${out_dir}"
    --log_dir "${LOG_ROOT}/${candidate}"
    --best_save_path "${out_dir}/best.pth"
    --latest_save_path "${out_dir}/latest.pth"
  )
}

launch_rung() {
  local rung="$1" gpu="$2" basis="$3" ridge="$4"
  local candidate="ADV3B02_ECRS_${rung}"
  candidate_enabled "${candidate}" || return 0
  local out_dir="${RUNS_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  build_common_cmd "${gpu}" "${candidate}" "${out_dir}"
  if [[ "${rung}" == "R0" ]]; then
    cmd=("${common_cmd[@]}")
    echo "[ECRS-V1-BASELINE] id=${candidate} rung=R0 mode=train_shared_baseline checkpoint=${BASE_CHECKPOINT} source_only=1 query_access=0"
    printf '[ECRS-V1-CMD]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
      return 0
    fi
    wait_for_gpu_slot "${gpu}" "${candidate}"
    if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
      echo "[ERROR] immutable baseline output already exists: ${out_dir} or ${log_path}" >&2
      exit 5
    fi
    mkdir -p "${out_dir}" "${LOG_ROOT}"
    echo "[ECRS-V1-BASELINE-START] id=${candidate} gpu=${gpu} log=${log_path}"
    "${cmd[@]}" >"${log_path}" 2>&1
    if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
      echo "[ERROR] shared R0 completed without checkpoint: ${BASE_CHECKPOINT}" >&2
      exit 6
    fi
    echo "[ECRS-V1-BASELINE-COMPLETE] id=${candidate} checkpoint=${BASE_CHECKPOINT}"
    return 0
  fi
  local ecrs_flags=(--use_ecrs --ecrs_rung "${rung}" --ecrs_basis_mode "${basis}" --ecrs_ridge_alpha "${ridge}")

  cmd=("${common_cmd[@]}"
    --init_checkpoint "${BASE_CHECKPOINT}"
    --ecrs_raw_ce_weight 0.30
    --ecrs_alpha_resp 0.15
    --lambda_ecrs_canonical 0.10
    --lambda_ecrs_content 0.10
    --lambda_ecrs_cycle 0.10
    --lambda_ecrs_split_fit 0.10
    --lambda_ecrs_pair_cross 0.10
    --lambda_ecrs_pair_surface 0.03
    --lambda_ecrs_same_tx 0.05
    --lambda_ecrs_diff_tx 0.03
    --no_ecrs_enable_learnable_basis
    --no_ecrs_enable_fasttrust
    "${ecrs_flags[@]}"
  )
  echo "[ECRS-V1-CANDIDATE] id=${candidate} rung=${rung} gpu=${gpu} basis=${basis} K=28 anchors=8 response_dim=64 rho_max=0.25 epochs=200 source_only=1 query_access=0"
  printf '[ECRS-V1-CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  wait_for_gpu_slot "${gpu}" "${candidate}"
  if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
    echo "[ERROR] immutable output already exists: ${out_dir} or ${log_path}" >&2
    exit 5
  fi
  mkdir -p "${out_dir}" "${LOG_ROOT}"
  nohup "${cmd[@]}" >"${log_path}" 2>&1 &
  echo "[ECRS-V1-LAUNCHED] id=${candidate} pid=$! log=${log_path}"
}

echo "[ECRS-V1-DATA] dataset=ManySig path=${WISIG_PKL} equalized=1 requested_split_mode=tx_rx_day_1_7_2 protocol_split=L_s/U_s/V=0.07/0.63/0.30 seed=392005 source_rxs=1,3,4,6,8 source_days=1,2,3 source_pool=90000 L_s=6300 U_s=56700 V=27000 target_rxs=0,2,5,7,9,10,11 target_days=0,1,2,3 target_tx=0,1,2,3,4,5 target_per_scenario=168000"
echo "[ECRS-V1-PROTOCOL] concat_sat_ce_only=1 lambda_sat_cls=0.68 lambda_sat_cons=0 schedule=1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
echo "[ECRS-V1-RESOURCES] max_gpu_train_processes=${MAX_GPU_TRAIN_PROCS} slot_poll_seconds=${GPU_SLOT_POLL_SECONDS} baseline_gpu=4"

launch_rung R0 4 fixed_spline 0.01
if [[ "${DRY_RUN}" != "1" && ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "[ERROR] exact source-only R0 checkpoint not found after baseline stage: ${BASE_CHECKPOINT}" >&2
  exit 6
fi
launch_rung R1 1 fixed_mp 0.01
launch_rung R2 2 fixed_spline 0.01
launch_rung R3 3 fixed_spline 0.01
launch_rung R4 4 fixed_spline 0.01
launch_rung R5 5 fixed_spline 0.01
launch_rung R6 6 fixed_spline 0.01
launch_rung R7 7 fixed_spline 0.01
launch_rung R8 0 fixed_spline 0.01
