#!/usr/bin/env bash
set -euo pipefail

# Frozen six-fold P1-CB-SFCE C/G continuation matrix.  C is the exact
# GeoSat-C continuation control; G differs only by lambda_cb_sfce=0.10.
RUN_ID="${RUN_ID:-phase1_cb_sfce12_20260809_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
GEOSAT_CKPT_ROOT="${GEOSAT_CKPT_ROOT:-${PROJECT_ROOT}/runs/phase1_loto_clsgeo12_20260808_v1}"
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
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")
FOLD_GEOC_CKPT=(
  "${GEOSAT_CKPT_ROOT}/F1C_LOTO_CLSGeo12/final_ssdg.pth" "${GEOSAT_CKPT_ROOT}/F2C_LOTO_CLSGeo12/final_ssdg.pth"
  "${GEOSAT_CKPT_ROOT}/F3C_LOTO_CLSGeo12/final_ssdg.pth" "${GEOSAT_CKPT_ROOT}/F4C_LOTO_CLSGeo12/final_ssdg.pth"
  "${GEOSAT_CKPT_ROOT}/F5C_LOTO_CLSGeo12/final_ssdg.pth" "${GEOSAT_CKPT_ROOT}/F6C_LOTO_CLSGeo12/final_ssdg.pth"
)

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing ManySig dataset: ${WISIG_PKL}" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite run/log root" >&2; exit 3; }
  for ckpt in "${FOLD_GEOC_CKPT[@]}"; do
    [[ -f "${ckpt}" ]] || { echo "missing GeoSat-C checkpoint: ${ckpt}" >&2; exit 2; }
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

COMMON=(
  --dataset wisig --wisig_pkl "${WISIG_PKL}" --from_scratch false --freeze_backbone false --model_variant lite_d --id_feature_key feat_joint
  --split_mode tx_rx_day_1_6_3 --labeled_ratio 0.07 --unlabeled_ratio 0.63 --source_val_ratio 0.30
  --epochs 40 --label_epochs 40 --pseudo_epochs 0 --seed 7281105 --sat_view_seed 9281105 --lr 0.0002 --weight_decay 0.0001 --label_smoothing 0.01
  --lambda_u 0 --lambda_ent 0 --lambda_u_domain 0 --lambda_u_adv 0 --lambda_u_sat_cons 0 --lambda_u_direct_metric_accept 0 --lambda_u_quarantine_accept 0
  --lambda_domain 0 --lambda_adv 0 --lambda_orth 0 --lambda_cons 0 --lambda_group_ce 0 --lambda_fishr 0
  --lambda_zid_receiver_invariance 0 --lambda_zid_day_invariance 0 --lambda_zid_channel_invariance 0 --lambda_u_zid_receiver_invariance 0 --lambda_u_zid_day_invariance 0 --lambda_u_zid_channel_invariance 0
  --lambda_tx_proto 0 --lambda_rx_proto 0 --lambda_mask_aux 0 --lambda_tx_supcon_masked 0 --lambda_rx_supcon_masked 0 --lambda_txrx_rect 0 --lambda_proto 0
  --lambda_open_world_feat 0 --lambda_zid_compact 0 --lambda_proxy_unknown 0 --lambda_manytx_real_oe 0 --lambda_soft_unknown_mixup 0 --lambda_source_episode 0 --lambda_direct_metric_accept 0
  --manytx_real_oe_enabled false --manytx_real_oe_protocol_enabled false --use_unlabeled false --pseudo_domain_gate false --pseudo_temporal_gate false
  --use_ema_teacher false --teacher_ckpt "" --lambda_teacher_clean_kl 0 --lambda_teacher_sat_kl 0 --lambda_teacher_zid_mse 0
  --use_aug false --use_mixstyle false --mixstyle_use_domain_label false --use_sat_consistency --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --sat_view_prob 1.0
  --lambda_sat_cls 0 --lambda_sat_cons 0.10 --sat_cons_start_epoch 1 --no_use_concat_sat_channel_aug
  --use_tx_rx_balanced_sampler false --use_phase2_ground_prototypes false --use_feature_masks false --use_txrx_geometry_losses false --use_proto_memory false --reject_head false
  --phase1_ccpc_leo_frozen_mode false --phase1_ccpc_leo_enabled false --lambda_ccpc_leo 0 --phase1_pamr_frozen_mode false --phase1_pamr_enabled false --lambda_pamr 0
  --phase1_cb_sfce_frozen_mode true --cb_sfce_gamma 1 --max_grad_norm 5.0 --checkpoint_selection final_only --best_metric source_val_sat_hmean --phase1_source_val_selection_only true
  --eval_sat_channel true --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --source_val_heavy_eval_start_epoch 40 --source_val_heavy_eval_interval 40 --source_val_heavy_eval_final_window 0
  --test_eval_policy interval_final --test_eval_start_epoch 40 --test_eval_interval 40 --batch_size 128 --eval_batch_size 256 --num_workers 4 --prefetch_factor 2 --device cuda:0 --amp true
)

declare -a pids folds arms gpus outs logs
launch_arm() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local index=$((fold - 1))
  local candidate="F${fold}${arm}_CB_SFCE12"
  local output_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  local -a arm_args
  case "${arm}" in
    C) arm_args=(--phase1_cb_sfce_enabled false --lambda_cb_sfce 0) ;;
    G) arm_args=(--phase1_cb_sfce_enabled true --lambda_cb_sfce 0.10) ;;
    *) echo "unsupported arm: ${arm}" >&2; return 2 ;;
  esac
  local -a command=("${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON[@]}" --run_id "${RUN_ID}" --candidate_id "${candidate}" --run_name "${candidate}" --output_dir "${output_dir}" --baseline_ckpt "${FOLD_GEOC_CKPT[index]}" --phase1_source_train_tx_ids "${FOLD_TRAIN_TX[index]}" --phase1_source_known_validation_tx_ids "${FOLD_KNOWN_VAL_TX[index]}" --phase1_source_proxy_unknown_tx_ids "${FOLD_PROXY_TX[index]}" "${arm_args[@]}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${gpu}" "${CODE_ROOT}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); arms+=("${arm}"); gpus+=("${gpu}"); outs+=("${output_dir}"); logs+=("${log_path}")
}

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

[[ "${DRY_RUN}" == "1" ]] && exit 0
printf 'pid|fold|arm|physical_gpu|output_dir|log_path|epochs|final_only\n' >"${LOG_ROOT}/pids.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|40|true\n' "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" "${outs[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids.tsv"
done
status=0
for index in "${!pids[@]}"; do wait "${pids[index]}" || status=1; done
exit "${status}"
