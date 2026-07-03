#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_multirx_newpool_k5k10_no_unknown_20260703_1115}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
TARGET_OLD_LABELS="${TARGET_OLD_LABELS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_RECEIVERS="${TARGET_RECEIVERS:-20-1,3-19,7-7,8-8}"
TARGET_NEW_TX_POOL="${TARGET_NEW_TX_POOL:-1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2,1-8,10-10,10-11,10-17,10-4,10-7,11-1,11-10,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-5,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-3,19-4,19-6,19-7,19-8,19-9,2-12,2-13,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-5,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,3-8,4-1,4-10,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-18,8-3,8-7,8-8,9-1,9-20,9-7}"
SAMPLE_RATE_HZ="${SAMPLE_RATE_HZ:-25000000}"
MAX_SAMPLES_PER_TX="${MAX_SAMPLES_PER_TX:-80}"
MAX_PAIR_CANDIDATES="${MAX_PAIR_CANDIDATES:-135}"
COMBO_SIZE="${COMBO_SIZE:-2}"
METHODS="${METHODS:-proto,knn1,knn3,knn5}"
OLD_TARGET="${OLD_TARGET:-0.85}"
SEEN_NEW_TARGET="${SEEN_NEW_TARGET:-0.85}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a receiver_array <<< "${TARGET_RECEIVERS}"
IFS=',' read -r -a target_new_pool_array <<< "${TARGET_NEW_TX_POOL}"

echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] teacher=${TEACHER_CKPT}"
echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] target_receivers=${TARGET_RECEIVERS}"
echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] target_new_pool_count=${#target_new_pool_array[@]} combo_size=${COMBO_SIZE}"
echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] unknown_policy=excluded_from_export_eval_and_success_metrics"
echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10] success=old_acc>=${OLD_TARGET} mean_new>=0.80 min_per_new_class>=${SEEN_NEW_TARGET}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

run_receiver() {
  local target_receiver="$1"
  local gpu="$2"
  local seed="$3"
  local safe_rx="${target_receiver//-/_}"
  local case_id="ADV3B02_MULTIRX_NEWPOOL_RX${safe_rx}"
  local out_dir="${RUNS_ROOT}/${case_id}"
  local log_path="${LOG_ROOT}/${case_id}.out"

  echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10-CANDIDATE] id=${case_id} gpu=${gpu} seed=${seed} target_receiver=${target_receiver}"
  local cmd=(
    env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" bash -lc
    "set -euo pipefail; \
mkdir -p '${out_dir}'; \
'${PYTHON}' -u '${ROOT}/code/export_spaceborne_features.py' \
  --ckpt '${TEACHER_CKPT}' \
  --wisig_pkl '${WISIG_PKL}' \
  --new_wisig_pkl '${NEW_WISIG_PKL}' \
  --out_npz '${out_dir}/features.npz' \
  --feature_name z_id \
  --source_tx_ids '${SOURCE_TX_IDS}' \
  --source_rxs '${CEN51_TRAIN_RXS}' \
  --target_old_tx_ids '${TARGET_OLD_TX_IDS}' \
  --target_old_rxs '${target_receiver}' \
  --target_old_channel_view satellite \
  --new_tx_ids '${TARGET_NEW_TX_POOL}' \
  --new_rxs '${target_receiver}' \
  --star_ground_channel_impl simplified_leo_residual \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --sample_rate_hz '${SAMPLE_RATE_HZ}' \
  --max_samples_per_combo 0 \
  --max_samples_per_tx '${MAX_SAMPLES_PER_TX}' \
  --batch_size 512 \
  --device cuda:0 \
  --seed '${seed}'; \
for k in 5 10; do \
  '${PYTHON}' -u '${ROOT}/code/scripts/phase2_newtx_pair_sweep.py' \
    --feature_npz '${out_dir}/features.npz' \
    --output_json '${out_dir}/pair_sweep_k'\"\${k}\"'.json' \
    --output_csv '${out_dir}/pair_sweep_k'\"\${k}\"'.csv' \
    --candidate_new_tx_ids '${TARGET_NEW_TX_POOL}' \
    --old_tx_ids '${TARGET_OLD_LABELS}' \
    --combo_size '${COMBO_SIZE}' \
    --methods '${METHODS}' \
    --k_old \"\${k}\" \
    --k_new \"\${k}\" \
    --query_per_old 40 \
    --query_per_new 40 \
    --old_target '${OLD_TARGET}' \
    --seen_new_target '${SEEN_NEW_TARGET}' \
    --max_pair_candidates '${MAX_PAIR_CANDIDATES}' \
    --seed '${seed}'; \
done"
  )

  printf "[ADV3B02-MULTIRX-NEWPOOL-K5K10-CMD] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" > "${log_path}" 2>&1 &
    echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10-LAUNCHED] id=${case_id} pid=$! gpu=${gpu} log=${log_path}"
  fi
}

gpu=0
seed=432001
for receiver in "${receiver_array[@]}"; do
  run_receiver "${receiver}" "${gpu}" "${seed}"
  gpu=$((gpu + 1))
  seed=$((seed + 1))
done

if [[ "${DRY_RUN}" != "1" ]]; then
  wait
  echo "[ADV3B02-MULTIRX-NEWPOOL-K5K10-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
fi
