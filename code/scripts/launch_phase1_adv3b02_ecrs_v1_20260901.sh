#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_ecrs_v1_20260901_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"

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

launch_rung() {
  local rung="$1" gpu="$2" basis="$3" ridge="$4"
  local candidate="ADV3B02_ECRS_${rung}"
  candidate_enabled "${candidate}" || return 0
  local out_dir="${RUNS_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  if [[ "${rung}" == "R0" ]]; then
    echo "[ECRS-V1-CANDIDATE] id=${candidate} rung=R0 mode=reference_checkpoint checkpoint=${BASE_CHECKPOINT} no_training=1 source_only=1 query_access=0"
    return 0
  fi
  local ecrs_flags=(--use_ecrs --ecrs_rung "${rung}" --ecrs_basis_mode "${basis}" --ecrs_ridge_alpha "${ridge}")

  cmd=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON}" -u "${ROOT}/code/train.py"
    --dataset wisig
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol cvs_day_rx
    --wisig_domain rx_day
    --wisig_out_len 256
    --use_meta_ssl_cvs
    --ssl_labeled_ratio 0.07
    --ssl_unlabeled_ratio 0.63
    --ssl_val_ratio 0.30
    --model_size M
    --model_variant lite_d
    --branch_ablation no_dac
    --domain_branch_ablation no_stats
    --init_checkpoint "${BASE_CHECKPOINT}"
    --epochs 200
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --concat_sat_start_epoch 1
    --concat_sat_ce_start_epoch 80
    --concat_sat_ce_weight 0.68
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
    --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --eval_sat_scenarios "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
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
    --run_name "${candidate}"
    --output_dir "${out_dir}"
    --log_dir "${LOG_ROOT}/${candidate}"
    --best_save_path "${out_dir}/best.pth"
    --latest_save_path "${out_dir}/latest.pth"
    "${ecrs_flags[@]}"
  )
  echo "[ECRS-V1-CANDIDATE] id=${candidate} rung=${rung} gpu=${gpu} basis=${basis} K=28 anchors=8 response_dim=64 rho_max=0.25 epochs=200 source_only=1 query_access=0"
  printf '[ECRS-V1-CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
    echo "[ERROR] immutable output already exists: ${out_dir} or ${log_path}" >&2
    exit 5
  fi
  mkdir -p "${out_dir}" "${LOG_ROOT}"
  nohup "${cmd[@]}" >"${log_path}" 2>&1 &
  echo "[ECRS-V1-LAUNCHED] id=${candidate} pid=$! log=${log_path}"
}

echo "[ECRS-V1-PROTOCOL] split=L_s/U_s/V=0.07/0.63/0.30 concat_sat_ce_only=1 lambda_sat_cls=0.68 lambda_sat_cons=0 schedule=1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak"

if [[ "${DRY_RUN}" != "1" && ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "[ERROR] converged Stage1 ADV3B02 checkpoint not found: ${BASE_CHECKPOINT}" >&2
  exit 6
fi

launch_rung R0 0 fixed_spline 0.01
launch_rung R1 1 fixed_mp 0.01
launch_rung R2 2 fixed_spline 0.01
launch_rung R3 3 fixed_spline 0.01
launch_rung R4 4 fixed_spline 0.01
launch_rung R5 5 fixed_spline 0.01
launch_rung R6 6 fixed_spline 0.01
launch_rung R7 7 fixed_spline 0.01
launch_rung R8 0 fixed_spline 0.01
