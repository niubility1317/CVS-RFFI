#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-phase1_geosat_lite_4arm_20260808_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
TRAIN_SCRIPT="${CODE_ROOT}/SSDG/train_ssdg.py"

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -f "${WISIG_PKL}" ]] || { echo "missing dataset: ${WISIG_PKL}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite run root: ${RUN_ROOT}" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite log root: ${LOG_ROOT}" >&2; exit 3; }

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

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
  --phase1_source_train_tx_ids 14-10,14-7,20-15,20-19
  --phase1_source_known_validation_tx_ids 6-15
  --phase1_source_proxy_unknown_tx_ids 8-20
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

launch_arm() {
  local arm="$1"
  local gpu="$2"
  local lambda_open="$3"
  local lambda_sat_cons="$4"
  local out="${RUN_ROOT}/${arm}"
  local log="${LOG_ROOT}/${arm}.out"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${PYTHON}" -u "${TRAIN_SCRIPT}" \
    "${COMMON[@]}" \
    --run_id "${RUN_ID}" \
    --candidate_id "${arm}" \
    --run_name "${arm}" \
    --output_dir "${out}" \
    --lambda_open_world_feat "${lambda_open}" \
    --lambda_sat_cons "${lambda_sat_cons}" \
    >"${log}" 2>&1 &
  pids+=("$!")
  arms+=("${arm}")
  gpus+=("${gpu}")
  outs+=("${out}")
  logs+=("${log}")
}

declare -a pids arms gpus outs logs
launch_arm A_ADV3B02_Z0 0 0 0
launch_arm B_ANGULAR_Z0 1 0.0024 0
launch_arm C_LEO_CONS_Z0 2 0 0.10
launch_arm D_GEOSAT_LITE_Z0 3 0.0024 0.10

for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s\n' "${pids[index]}" "${arms[index]}" "${gpus[index]}" "${outs[index]}" "${logs[index]}"
done >"${LOG_ROOT}/pids.tsv"
status=0
for index in "${!pids[@]}"; do
  row="${pids[index]}|${arms[index]}|${gpus[index]}|${outs[index]}|${logs[index]}"
  if wait "${pids[index]}"; then
    printf '%s|0\n' "${row}" >>"${LOG_ROOT}/completion.tsv"
  else
    rc=$?
    printf '%s|%s\n' "${row}" "${rc}" >>"${LOG_ROOT}/completion.tsv"
    status=10
  fi
done
exit "${status}"
