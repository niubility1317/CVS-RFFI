#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CONDA_ENV_PRIMARY="${CONDA_ENV_PRIMARY:-ssr-gpu}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-CVS-RFFI}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/bex02_ratio010_central_custom_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/bex02_ratio010_central_custom_${STAMP}}"

source /opt/miniconda3/etc/profile.d/conda.sh
if conda activate "${CONDA_ENV_PRIMARY}" 2>/dev/null; then
  ACTIVE_ENV="${CONDA_ENV_PRIMARY}"
else
  echo "[WARN] conda env '${CONDA_ENV_PRIMARY}' not found; falling back to '${CONDA_ENV_FALLBACK}'." >&2
  conda activate "${CONDA_ENV_FALLBACK}"
  ACTIVE_ENV="${CONDA_ENV_FALLBACK}"
fi

cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 3 ]; then
  echo "Need at least 3 GPUs in GPU_IDS for the three-job launch." >&2
  exit 2
fi

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl"
  --wisig_train_ratio 0.1
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --epochs 170
  --primary_udu_weight 0.65
  --test_eval_policy every_epoch
  --eval_max_batches 0
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --sat_cons_start_epoch 20
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --seed 1337
  --device cuda:0
)

launch_one() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local out_dir="${RUN_ROOT}/${tag}"
  local log_file="${LOG_ROOT}/${tag}.log"
  mkdir -p "${out_dir}"
  local cmd=(python -u train.py "${COMMON_ARGS[@]}" "$@"
    --run_name "${tag}"
    --output_dir "${out_dir}"
    --best_save_path "${out_dir}/best_model.pth"
    --latest_save_path "${out_dir}/latest_model.pth")
  {
    echo "TAG=${tag}"
    echo "GPU=${gpu}"
    echo "CONDA_ENV=${ACTIVE_ENV}"
    echo "RUN_DIR=${out_dir}"
    printf "CMD=CUDA_VISIBLE_DEVICES=%s " "${gpu}"
    printf "%q " "${cmd[@]}"
    echo
  } > "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 nohup "${cmd[@]}" >> "${log_file}" 2>&1 &
  echo "${tag}	${gpu}	$!	${out_dir}	${log_file}" | tee -a "${LOG_ROOT}/manifest.tsv"
}

launch_one "${GPU_LIST[0]}" "BEX02_fishr002_ratio010_strong_dg" \
  --train_mode centralized \
  --use_aug \
  --use_mixstyle \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --lambda_sat_cls 0.10 \
  --lambda_sat_cons 0.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4

launch_one "${GPU_LIST[1]}" "BEX02_fishr002_ratio010_baseline_satview" \
  --train_mode centralized \
  --use_aug \
  --use_mixstyle \
  --use_sat_consistency \
  --central_sat_aug_mode baseline_view \
  --sat_train_scenarios mixed_orbit \
  --sat_view_prob 1.0 \
  --sat_view_seed 2027 \
  --lambda_sat_cls 0.00 \
  --lambda_sat_cons 0.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4

launch_one "${GPU_LIST[2]}" "BEX02_ratio010_ce_grl_only" \
  --train_mode centralized \
  --wisig_domain rx \
  --force_ce_grl_only \
  --lambda_adv 1.0 \
  --lambda_dom 0

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "MANIFEST=${LOG_ROOT}/manifest.tsv"
