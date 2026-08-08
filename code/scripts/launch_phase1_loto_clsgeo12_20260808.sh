#!/usr/bin/env bash
set -euo pipefail

# Frozen development-only cross-TX matrix.  DRY_RUN=1 prints commands without
# creating outputs or requiring the remote dataset path.
RUN_ID="${RUN_ID:-phase1_loto_clsgeo12_20260808}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
TRAIN_SCRIPT="${CODE_ROOT}/SSDG/train_ssdg.py"
DRY_RUN="${DRY_RUN:-0}"

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing dataset: ${WISIG_PKL}" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

# Fi: held primary proxy=TX_i; held secondary known validation=TX_(i+1);
# the remaining four TX are the only source-training TX.  The held proxy is
# a split role only: this launcher never feeds it to an unknown-training loss.
FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_HELD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")

# This is byte-for-byte the existing GeoSat-C common launcher configuration;
# C passes only its two arm scalars below, while G adds only the frozen key/loss.
COMMON=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --from_scratch true
  --freeze_backbone false
  --model_variant lite_d
  --split_mode tx_rx_day_1_6_3
  --labeled_ratio 0.07
  --unlabeled_ratio 0.63
  --source_val_ratio 0.30
  --epochs 120
  --label_epochs 78
  --pseudo_epochs 42
  --seed 7281105
  --sat_view_seed 9281105
  --lr 0.0002
  --weight_decay 0.0001
  --label_smoothing 0.01
  --lambda_u 0.16
  --lambda_ent 0.01
  --lambda_domain 1.0
  --lambda_adv 0.35
  --lambda_orth 0.05
  --lambda_cons 0.08
  --lambda_group_ce 0.16
  --lambda_fishr 0.04
  --use_unlabeled true
  --pseudo_domain_gate true
  --pseudo_temporal_gate true
  --use_ema_teacher true
  --use_aug true
  --use_mixstyle true
  --use_sat_consistency
  --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
  --sat_view_prob 1.0
  --lambda_sat_cls 0
  --sat_cons_start_epoch 1
  --max_grad_norm 5.0
  --lambda_proxy_unknown 0
  --lambda_soft_unknown_mixup 0
  --lambda_source_episode 0
  --lambda_direct_metric_accept 0
  --lambda_u_direct_metric_accept 0
  --lambda_u_quarantine_accept 0
  --ow_feat_domain_align_weight 0
  --ow_feat_tail_weight 0
  --ow_feat_vacuum_weight 0
  --ow_feat_radius_deg 12
  --ow_feat_inter_margin_deg 55
  --ow_feat_sample_margin_deg 5
  --checkpoint_selection final_only
  --best_metric source_val_sat_hmean
  --phase1_source_val_selection_only true
  --eval_sat_channel true
  --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
  --source_val_heavy_eval_start_epoch 120
  --source_val_heavy_eval_interval 120
  --source_val_heavy_eval_final_window 0
  --test_eval_policy interval_final
  --test_eval_start_epoch 120
  --test_eval_interval 120
  --batch_size 128
  --eval_batch_size 256
  --num_workers 4
  --prefetch_factor 2
  --device cuda:0
  --amp true
)

declare -a pids arms folds gpus outs logs train_txs known_val_txs proxy_txs

launch_arm() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local fold_index=$((fold - 1))
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local known_val_tx="${FOLD_KNOWN_VAL_TX[fold_index]}"
  local proxy_tx="${FOLD_HELD_PROXY_TX[fold_index]}"
  local candidate="F${fold}${arm}_LOTO_CLSGeo12"
  local output_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  local lambda_open_world="0"
  local -a extra_args=()

  if [[ "${arm}" == "G" ]]; then
    lambda_open_world="0.0024"
    extra_args=(--ow_feat_key id_feat_cls)
  elif [[ "${arm}" != "C" ]]; then
    echo "unsupported arm: ${arm}" >&2
    return 2
  fi

  local -a command=(
    "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON[@]}"
    --run_id "${RUN_ID}"
    --candidate_id "${candidate}"
    --run_name "${candidate}"
    --output_dir "${output_dir}"
    --phase1_source_train_tx_ids "${train_tx}"
    --phase1_source_known_validation_tx_ids "${known_val_tx}"
    --phase1_source_proxy_unknown_tx_ids "${proxy_tx}"
    --lambda_open_world_feat "${lambda_open_world}"
    --lambda_sat_cons 0.10
    "${extra_args[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${gpu}" "${CODE_ROOT}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
  pids+=("$!")
  folds+=("${fold}")
  arms+=("${arm}")
  gpus+=("${gpu}")
  outs+=("${output_dir}")
  logs+=("${log_path}")
  train_txs+=("${train_tx}")
  known_val_txs+=("${known_val_tx}")
  proxy_txs+=("${proxy_tx}")
}

# Physical-GPU schedule: no card receives more than two processes.
launch_arm 1 C 0
launch_arm 5 G 0
launch_arm 1 G 1
launch_arm 5 C 1
launch_arm 2 C 2
launch_arm 6 G 2
launch_arm 2 G 3
launch_arm 6 C 3
launch_arm 3 C 4
launch_arm 3 G 5
launch_arm 4 C 6
launch_arm 4 G 7

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

printf 'pid|fold|arm|physical_gpu|output_dir|log_path|train_tx|known_validation_tx|held_proxy_tx\n' >"${LOG_ROOT}/pids.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outs[index]}" "${logs[index]}" "${train_txs[index]}" \
    "${known_val_txs[index]}" "${proxy_txs[index]}" >>"${LOG_ROOT}/pids.tsv"
done

printf 'pid|fold|arm|physical_gpu|output_dir|log_path|train_tx|known_validation_tx|held_proxy_tx|exit_code\n' >"${LOG_ROOT}/completion.tsv"
status=0
for index in "${!pids[@]}"; do
  rc=0
  if wait "${pids[index]}"; then
    rc=0
  else
    rc=$?
    status=8
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outs[index]}" "${logs[index]}" "${train_txs[index]}" \
    "${known_val_txs[index]}" "${proxy_txs[index]}" "${rc}" >>"${LOG_ROOT}/completion.tsv"
done
exit "${status}"
