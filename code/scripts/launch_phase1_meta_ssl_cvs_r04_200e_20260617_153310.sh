#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_meta_ssl_cvs_r04_200e_20260617_153310}"
CAND_ID="${CAND_ID:-PH1_MSSL_B3_PROTOGATE_200E_S1337}"
GPU_INDEX="${GPU_INDEX:-0}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/${RUN_ID}/${CAND_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_ID}/${CAND_ID}}"
STDOUT_LOG="${STDOUT_LOG:-${LOG_DIR}/stdout.log}"
EVENTS_TSV="${EVENTS_TSV:-${LOG_DIR}/scheduler_events.tsv}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
  "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
  "${PYTHON}" -u "${ROOT}/code/train.py"
  --train_mode centralized
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_protocol cvs_day_rx
  --wisig_domain rx_day
  --wisig_equalized 1
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio -1.0
  --wisig_split_strategy random
  --wisig_cap_strategy random
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --num_classes 16
  --arch_family cvsincnet
  --slim_group none
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --id_time_stability_mode off
  --id_freq_stability_mode off
  --domain_time_stability_mode off
  --domain_freq_stability_mode off
  --exp_group s3_rxrobust_no_dac
  --pa_orders 1,3,5
  --use_meta_ssl_cvs
  --use_meta_rxday_episodes
  --ssl_labeled_ratio 0.1
  --ssl_unlabeled_ratio 0.7
  --ssl_val_ratio 0.2
  --ssl_teacher_ema 0.995
  --ssl_gate_mode freematch_ups_proto
  --ssl_min_conf 0.60
  --ssl_min_margin 0.02
  --ssl_max_uncertainty 0.35
  --ssl_class_quota 64
  --ssl_receiver_quota 16
  --lambda_ssl_tx 0.5
  --lambda_ssl_proto 0.1
  --lambda_meta_ssl 0.05
  --meta_inner_scope head_proj
  --no_enable_pa_aux
  --no_enable_dac_aux
  --no_aug_enable_pa_normal
  --aug_p_pa 0.0
  --aug_p_dac 0.0
  --lambda_cls_pa 0.0
  --lambda_cls_dac 0.0
  --lambda_pa_joint_inv 0.0
  --lambda_pa_kl 0.0
  --lambda_dac_reg 0.0
  --lambda_pa_reg 0.0
  --lambda_dom 0.50
  --lambda_adv 0.20
  --grl_lambda 1.0
  --lambda_orth 0.024
  --lambda_cons 0.012
  --lambda_group_ce 0.018
  --group_ce_mode smooth_dro_capped
  --group_ce_min_domains 2
  --group_ce_top_frac 0.14
  --groupdro_tau 0.32
  --groupdro_cap 0.40
  --use_proto_memory
  --proto_momentum 0.97
  --lambda_proto 0.0025
  --lambda_supcon_id 0.0025
  --supcon_temp 0.12
  --lambda_fishr 0.0002
  --fishr_min_domains 2
  --lambda_feature_norm_guard 0.00004
  --feature_norm_guard_mode l2
  --feature_norm_guard_target 0
  --use_aug
  --aug_scale_min 0.10
  --aug_scale_max 0.32
  --late_aug_min_scale 0.16
  --use_mixstyle
  --mixstyle_p 0.025
  --mixstyle_strength 0.24
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_late_start 95
  --mixstyle_late_ramp_epochs 35
  --mixstyle_late_min_p 0.020
  --mixstyle_late_min_strength 0.156
  --no_use_concat_sat_channel_aug
  --no_use_sat_consistency
  --lambda_sat_cls 0.0
  --lambda_sat_cons 0.0
  --concat_sat_ce_weight 0.0
  --sat_cons_start_epoch 999
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --test_eval_policy interval_final
  --test_eval_start_epoch 31
  --test_eval_interval 10
  --eval_batch_size 256
  --batch_size 256
  --epochs 200
  --swad_start_epoch 75
  --swad_interval 1
  --swad_tolerance 0.85
  --primary_udu_weight 0.82
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 10.0
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --label_smoothing 0.0
  --device cuda:0
  --num_workers 4
  --prefetch_factor 2
  --seed 1337
  --run_name "${CAND_ID}"
  --log_dir "${LOG_DIR}"
  --latest_save_path "${RUN_DIR}/latest_model.pth"
  --best_save_path "${RUN_DIR}/best_val_model.pth"
  --best_test_save_path "${RUN_DIR}/best_test_overall_model.pth"
  --best_primary_save_path "${RUN_DIR}/best_primary_ood_model.pth"
  --best_unseen_day_unseen_rx_save_path "${RUN_DIR}/best_strict_udu_model.pth"
  --best_unseen_day_seen_rx_save_path "${RUN_DIR}/best_unseen_day_seen_rx_model.pth"
  --best_seen_day_unseen_rx_save_path "${RUN_DIR}/best_seen_day_unseen_rx_model.pth"
  --best_worst_rx_save_path "${RUN_DIR}/best_worst_rx_model.pth"
  --ema_save_path "${RUN_DIR}/ema_model.pth"
  --swad_save_path "${RUN_DIR}/swad_model.pth"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[PHASE1-DRY-RUN] run_id=%s cand_id=%s gpu=%s\n' "${RUN_ID}" "${CAND_ID}" "${GPU_INDEX}"
  printf '[PHASE1-CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${RUN_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "[ERROR] run/log path collision: RUN_DIR=${RUN_DIR} LOG_DIR=${LOG_DIR}" >&2
  exit 3
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
printf '%q ' "${cmd[@]}" > "${LOG_DIR}/exact_command.sh"
printf '\n' >> "${LOG_DIR}/exact_command.sh"
sha256sum "${LOG_DIR}/exact_command.sh" > "${LOG_DIR}/exact_command.sha256"
printf 'timestamp\tevent\trun_id\tcandidate_id\tgpu\tpid_or_status\n' > "${EVENTS_TSV}"
printf '%s\tSTART\t%s\t%s\t%s\t-\n' "$(date -Is)" "${RUN_ID}" "${CAND_ID}" "${GPU_INDEX}" | tee -a "${EVENTS_TSV}"

nohup "${cmd[@]}" > "${STDOUT_LOG}" 2>&1 &
pid="$!"
echo "${pid}" > "${LOG_DIR}/pid.txt"
printf '%s\tLAUNCHED\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${RUN_ID}" "${CAND_ID}" "${GPU_INDEX}" "${pid}" | tee -a "${EVENTS_TSV}"
printf '[PHASE1-LAUNCHED] run_id=%s cand_id=%s gpu=%s pid=%s stdout=%s run_dir=%s log_dir=%s\n' "${RUN_ID}" "${CAND_ID}" "${GPU_INDEX}" "${pid}" "${STDOUT_LOG}" "${RUN_DIR}" "${LOG_DIR}"
