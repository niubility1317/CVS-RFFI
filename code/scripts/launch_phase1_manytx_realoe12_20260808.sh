#!/usr/bin/env bash
set -euo pipefail

# P1-ManyTx-RealOE-v2: development-only physical-RX-safe source-side OE matrix.  The launcher
# never writes an existing run/log root and never turns the static proxy,
# reserve, or authority-locked target-new identities into train batches.
RUN_ID="${RUN_ID:-phase1_manytx_realoe12_physrx_v2_20260808}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
MANYTX_PKL="${MANYTX_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManyTx.pkl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
TRAIN_SCRIPT="${CODE_ROOT}/SSDG/train_ssdg.py"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing ManySig dataset: ${WISIG_PKL}" >&2; exit 2; }
  [[ -f "${MANYTX_PKL}" ]] || { echo "missing ManyTx source-OE dataset: ${MANYTX_PKL}" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

# Authority lock: these 20 identities are fixed target_new and are never
# redefined here.  The 80/20/16 lists below are the remaining qualified extras
# in the immutable P1-ManyTx-OE-v2 partition after the common-physical-RX audit.
LOCKED_TARGET_NEW_TX="1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6"
OE_TRAIN_TX="10-4,3-1,7-8,16-20,11-17,8-14,19-1,2-13,11-1,19-19,18-1,4-1,13-19,18-4,13-3,11-10,19-11,7-20,1-11,18-11,14-8,3-19,13-20,14-9,19-4,18-17,19-7,2-17,7-10,1-10,2-7,9-1,18-14,11-4,18-15,20-18,19-2,14-12,3-20,1-12,3-2,5-1,7-13,11-20,20-4,18-5,18-2,6-1,20-7,10-17,8-1,18-16,17-10,20-1,2-19,14-20,8-8,10-7,9-20,6-6,19-20,2-6,20-5,1-15,1-14,8-13,18-20,8-18,7-11,8-7,9-7,18-12,11-7,16-16,14-14,20-14,15-19,2-8,14-13,20-8"
PROXY_TX="20-20,20-16,19-3,1-19,3-18,19-12,5-20,7-14,12-7,7-9,17-11,20-3,12-20,16-1,18-7,2-3,19-10,18-9,2-4,15-6"
RESERVE_TX="2-14,10-11,9-14,13-7,2-12,7-12,5-5,2-15,18-13,5-16,19-14,15-1,12-19,3-13,7-7,4-11"
PARTITION_ROOT="ca3ed65a533359d2abb022fa513c49101ad93235738a39b362b5cdd15879c3d1"
SOURCE_RX_LABELS="1-1,1-19,14-7,18-2,19-2,2-1"
SOURCE_DAY_LABELS="2021_03_01,2021_03_08"
TARGET_RX_LABELS="20-1,3-19,7-14,7-7,8-8"

# Six leave-one-known-TX folds.  All held TX are known-validation only; no
# batch-held known label is relabelled as unknown and the primary proxy role is
# intentionally empty for this external-real-OE protocol.
FOLD_TRAIN_TX=(
  "14-7,20-15,20-19,6-15,8-20"
  "14-10,20-15,20-19,6-15,8-20"
  "14-10,14-7,20-19,6-15,8-20"
  "14-10,14-7,20-15,6-15,8-20"
  "14-10,14-7,20-15,20-19,8-20"
  "14-10,14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")

# GeoSat-C common configuration is held constant.  C and RealOE differ only
# in whether observed source-OE batches are supplied and the RealOE loss weight.
COMMON=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --from_scratch true
  --freeze_backbone false
  --model_variant lite_d
  --split_mode tx_rx_day_1_6_3
  --wisig_train_days "${SOURCE_DAY_LABELS}"
  --wisig_train_rxs "${SOURCE_RX_LABELS}"
  --wisig_test_rxs "${TARGET_RX_LABELS}"
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
  --lambda_sat_cons 0.10
  --sat_cons_start_epoch 1
  --max_grad_norm 5.0
  --lambda_open_world_feat 0
  --lambda_proxy_unknown 0
  --lambda_soft_unknown_mixup 0
  --lambda_source_episode 0
  --lambda_direct_metric_accept 0
  --lambda_u_direct_metric_accept 0
  --lambda_u_quarantine_accept 0
  --ow_feat_domain_align_weight 0
  --ow_feat_tail_weight 0
  --ow_feat_vacuum_weight 0
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
  --phase1_allow_empty_proxy_unknown true
  --manytx_real_oe_protocol_enabled true
  --manytx_real_oe_train_tx_ids "${OE_TRAIN_TX}"
  --manytx_real_oe_proxy_tx_ids "${PROXY_TX}"
  --manytx_real_oe_reserve_tx_ids "${RESERVE_TX}"
  --manytx_locked_target_new_tx_ids "${LOCKED_TARGET_NEW_TX}"
  --manytx_real_oe_partition_root_sha256 "${PARTITION_ROOT}"
  --manytx_real_oe_days "${SOURCE_DAY_LABELS}"
  --manytx_real_oe_rxs "${SOURCE_RX_LABELS}"
  --manytx_real_oe_equalized 1
  --manytx_real_oe_tx_per_batch 16
  --manytx_real_oe_samples_per_tx 8
  --manytx_real_oe_start_epoch 61
  --manytx_real_oe_warmup_epochs 10
  --manytx_real_oe_temperature 1
  --manytx_real_oe_margin 1
  --manytx_real_oe_tau 1
)

declare -a pids folds arms gpus outs logs train_txs known_val_txs modes

launch_arm() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local fold_index=$((fold - 1))
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local known_val_tx="${FOLD_KNOWN_VAL_TX[fold_index]}"
  local candidate="F${fold}${arm}_ManyTxRealOE12"
  local output_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  local -a arm_args=()

  case "${arm}" in
    C)
      arm_args=(--manytx_real_oe_enabled false --lambda_manytx_real_oe 0)
      ;;
    G)
      arm_args=(
        --manytx_real_oe_enabled true
        --manytx_real_oe_pkl "${MANYTX_PKL}"
        --lambda_manytx_real_oe 0.02
      )
      ;;
    *)
      echo "unsupported arm: ${arm}" >&2
      return 2
      ;;
  esac

  local -a command=(
    "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON[@]}"
    --run_id "${RUN_ID}"
    --candidate_id "${candidate}"
    --run_name "${candidate}"
    --output_dir "${output_dir}"
    --phase1_source_train_tx_ids "${train_tx}"
    --phase1_source_known_validation_tx_ids "${known_val_tx}"
    --phase1_source_proxy_unknown_tx_ids ""
    "${arm_args[@]}"
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
  modes+=("${arm}")
}

# Physical-GPU schedule: every process sees its assigned physical card as cuda:0,
# and no card receives more than two training processes.
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

printf 'pid|fold|arm|physical_gpu|output_dir|log_path|train_tx|known_validation_tx|manytx_mode|partition_root\n' >"${LOG_ROOT}/pids.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outs[index]}" "${logs[index]}" "${train_txs[index]}" \
    "${known_val_txs[index]}" "${modes[index]}" "${PARTITION_ROOT}" >>"${LOG_ROOT}/pids.tsv"
done

printf 'pid|fold|arm|physical_gpu|output_dir|log_path|train_tx|known_validation_tx|manytx_mode|partition_root|exit_code\n' >"${LOG_ROOT}/completion.tsv"
status=0
for index in "${!pids[@]}"; do
  rc=0
  if wait "${pids[index]}"; then
    rc=0
  else
    rc=$?
    status=8
  fi
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outs[index]}" "${logs[index]}" "${train_txs[index]}" \
    "${known_val_txs[index]}" "${modes[index]}" "${PARTITION_ROOT}" "${rc}" >>"${LOG_ROOT}/completion.tsv"
done
exit "${status}"
