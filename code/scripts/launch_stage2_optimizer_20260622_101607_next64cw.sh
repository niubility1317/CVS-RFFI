#!/usr/bin/env bash
# Generated locally for next64cw from next64cu after fresh dual-idle monitor gate; Phase1 seeds +1000 from next64cu; Phase2 deferred for OA-MSE post-run rollback/local repair.
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64cw_20260622_101607}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
NEW_TX_IDS="${NEW_TX_IDS:-6,7}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-8,9}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-${SOURCE_TX_IDS}}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_RXS="${TARGET_RXS:-20-1}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-3}"
PHASE1_MAX_ACTIVE_PER_GPU="${PHASE1_MAX_ACTIVE_PER_GPU:-1}"
PHASE2_MAX_ACTIVE_PER_GPU="${PHASE2_MAX_ACTIVE_PER_GPU:-3}"
COMBINED_MAX_ACTIVE_PER_GPU="${COMBINED_MAX_ACTIVE_PER_GPU:-4}"
STAGE2_MAX_SCHEDULER_SECONDS="${STAGE2_MAX_SCHEDULER_SECONDS:-5400}"
STAGE2_EXPECTED_CANDIDATE_MAX_SECONDS="${STAGE2_EXPECTED_CANDIDATE_MAX_SECONDS:-1200}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -z "${UNKNOWN_TX_IDS}" ]]; then
  echo "[ERROR] UNKNOWN_TX_IDS must not be empty for stage2 open-set validation" >&2
  exit 2
fi

SCHED_LOG="${LOG_ROOT}/scheduler.out"
EVENTS_TSV="${LOG_ROOT}/scheduler_events.tsv"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  : > "${SCHED_LOG}"
  : > "${EVENTS_TSV}"
fi

log_msg() { if [[ "${DRY_RUN}" == "1" ]]; then echo "$@"; else echo "$@" | tee -a "${SCHED_LOG}"; fi; }
event_row() { local row; row="$(printf "%s\t%s\t%s\t%s\t%s\t%s" "$(date -Is)" "$1" "$2" "$3" "$4" "$5")"; if [[ "${DRY_RUN}" == "1" ]]; then echo "${row}"; else echo "${row}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"; fi; }

run_sfe_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" shots="$8" source_proto="$9" query_per_tx="${10}" seed="${11}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol sfe --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_query_per_tx "${query_per_tx}" --shots "${shots}" --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --target_old_support_per_tx "${SFE_OLD_SUPPORT_CURRENT:-0}" --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2-SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin} max_mahal=${max_mahal} openmax_q=${openmax_q}"
  "${cmd[@]}"
}

run_source_open_set_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" source_proto="$8" query_per_tx="$9" seed="${10}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol source_open_set --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_query_per_tx "${query_per_tx}" --shots 0 --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --target_old_support_per_tx "${SFE_OLD_SUPPORT_CURRENT:-0}" --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2-STAGE2A-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin} max_mahal=${max_mahal} openmax_q=${openmax_q}"
  "${cmd[@]}"
}

run_sfe_profile() {
  local cid="$1" profile="$2" shots="$3" source_proto="$4" query_per_tx="$5" seed="$6"
  case "${profile}" in
    source_open_set)
      run_source_open_set_eval "${cid}" "stage2a_combined_t075_m005_mh6" "combined" "0.75" "0.05" "6.0" "1.0" "${source_proto}" "${query_per_tx}" "${seed}"
      run_source_open_set_eval "${cid}" "stage2a_combined_t080_m010_mh5" "combined" "0.80" "0.10" "5.0" "0.95" "${source_proto}" "${query_per_tx}" "${seed}"
      run_source_open_set_eval "${cid}" "stage2a_mahal_t080_mh5" "mahalanobis" "0.80" "" "5.0" "0.95" "${source_proto}" "${query_per_tx}" "${seed}"
      run_source_open_set_eval "${cid}" "stage2a_openmax_t080_q095" "openmax" "0.80" "" "" "0.95" "${source_proto}" "${query_per_tx}" "${seed}"
      ;;
    strict_openmax)
      run_sfe_eval "${cid}" "combined_t080_m010_mh5" "combined" "0.80" "0.10" "5.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "combined_t085_m010_mh5" "combined" "0.85" "0.10" "5.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "mahal_t080_mh5" "mahalanobis" "0.80" "" "5.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "openmax_t080_q095" "openmax" "0.80" "" "" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      ;;
    balanced_openmax)
      run_sfe_eval "${cid}" "combined_t075_m005_mh6" "combined" "0.75" "0.05" "6.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "combined_t080_m005_mh6" "combined" "0.80" "0.05" "6.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "mahal_t075_mh6" "mahalanobis" "0.75" "" "6.0" "0.97" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "openmax_t080_q097" "openmax" "0.80" "" "" "0.97" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      ;;
    score_diag)
      run_sfe_eval "${cid}" "combined_t075_m005_mh5" "combined" "0.75" "0.05" "5.0" "1.0" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "combined_t080_m010_mh5" "combined" "0.80" "0.10" "5.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "mahal_t080_mh5" "mahalanobis" "0.80" "" "5.0" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      run_sfe_eval "${cid}" "openmax_t080_q095" "openmax" "0.80" "" "" "0.95" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
      ;;
    *) echo "[ERROR] unknown SFE profile: ${profile}" >&2; return 2 ;;
  esac
}

run_sfe_bundle() {
  local cid="$1" channel_view="$2" shots="$3" seed="$4" sat_seed="$5" max_samples_per_tx="$6" source_proto="$7" query_per_tx="$8" scenarios="$9" profile="${10}"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-SFE-BEGIN] cid=${cid} channel=${channel_view} profile=${profile} shots=${shots} source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS} scenarios=${scenarios}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" --ckpt "${TEACHER_CKPT}" --wisig_pkl "${WISIG_PKL}" --new_wisig_pkl "${NEW_WISIG_PKL}" --out_npz "${out_dir}/features.npz" --feature_name z_id --source_tx_ids "${SOURCE_TX_IDS}" --source_rxs "${CEN51_TRAIN_RXS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --target_old_rxs "${TARGET_RXS}" --target_old_channel_view "${channel_view}" --target_old_sat_scenarios "${scenarios}" --target_old_sat_seed "$((sat_seed + 111))" --new_tx_ids "${NEW_TX_IDS}" --new_rxs "${TARGET_RXS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo 0 --max_samples_per_tx "${max_samples_per_tx}" --batch_size 512 --device cuda:0 --seed "${seed}" --target_new_channel_view "${channel_view}" --target_new_sat_scenarios "${scenarios}" --target_new_sat_seed "${sat_seed}"
  run_sfe_profile "${cid}" "${profile}" "${shots}" "${source_proto}" "${query_per_tx}" "${seed}"
  echo "[S2-SFE-END] cid=${cid}"
}

run_ftrc_candidate() {
  local cid="$1" adapter="$2" k="$3" lr="$4" anchor="$5" alpha="$6" rank="$7" epochs="$8" steps="$9" eval_detail="${10}" seed="${11}"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2-FTRC-BEGIN] cid=${cid} adapter=${adapter} k=${k} lr=${lr} anchor=${anchor} alpha=${alpha} rank=${rank} epochs=${epochs} steps=${steps} seed=${seed}"
  "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" --teacher_ckpt "${TEACHER_CKPT}" --output_dir "${out_dir}" --dataset wisig --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --wisig_train_ratio 0.1 --wisig_guard_gap 8 --wisig_train_days 0,1 --wisig_test_days 2,3 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_test_rxs 7,8,9,10,11 --target_loader "${TARGET_LOADER}" --target_channel_view satellite --target_label_mode labeled --target_samples_per_rx_tx "${k}" --target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --epochs "${epochs}" --adapt_steps_per_epoch "${steps}" --target_batch_size 32 --lr_adapt "${lr}" --entropy_weight 0 --consistency_weight 0 --pseudo_weight 0 --anchor_weight "${anchor}" --anchor_temperature 2.5 --eval_detail_every "${eval_detail}" --target_adapter_type "${adapter}" --adapter_rank "${rank}" --adapter_bottleneck 16 --adapter_alpha "${alpha}" --adapter_dropout 0.0 --freeze_base_stats true --update_norm false --update_classifier false --rollback_enabled true --eval_sat_channel true --eval_sat_on "${TARGET_LOADER}" --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches 0 --eval_max_batches 0 --seed "${seed}" --run_name "${cid}"
  echo "[S2-FTRC-END] cid=${cid}"
}


run_meta_ssl_protocol() {
  local cid="$1" ablation="$2" seed="$3" max_samples="$4"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[MSSL-BEGIN] cid=${cid} ablation=${ablation} split=0.1L/0.7U/0.2Val seed=${seed}"
  "${PYTHON}" -u "${ROOT}/code/train.py" \
    --dataset wisig \
    --wisig_pkl "${WISIG_PKL}" \
    --wisig_equalized 1 \
    --wisig_domain rx_day \
    --wisig_out_len 256 \
    --wisig_train_ratio 0.1 \
    --wisig_train_days 0,1 \
    --wisig_test_days 2,3 \
    --wisig_train_rxs 0,1,2,3,4,5,6 \
    --wisig_test_rxs 7,8,9,10,11 \
    --wisig_max_day123_per_combo "${max_samples}" \
    --wisig_max_train_per_combo 4 \
    --wisig_max_val_per_combo 4 \
    --wisig_max_test_per_combo 4 \
    --wisig_cap_strategy front \
    --batch_size 16 \
    --eval_batch_size 32 \
    --num_workers 0 \
    --device cpu \
    --no_eval_sat_channel \
    --use_meta_ssl_cvs \
    --meta_ssl_protocol_check_only \
    --ssl_labeled_ratio 0.1 \
    --ssl_unlabeled_ratio 0.7 \
    --ssl_val_ratio 0.2 \
    --ssl_gate_mode freematch_ups_proto \
    --ssl_min_conf 0.85 \
    --ssl_min_margin 0.05 \
    --ssl_max_uncertainty 0.08 \
    --use_meta_rxday_episodes \
    --meta_inner_scope head_proj \
    --meta_ssl_max_samples_per_combo_source "${max_samples}" \
    --meta_ssl_protocol_report "${out_dir}/meta_ssl_protocol_report.json" \
    --run_name "${cid}" \
    --seed "${seed}" \
    --epochs 1
  echo "[MSSL-END] cid=${cid} report=${out_dir}/meta_ssl_protocol_report.json"
}


run_meta_ssl_training_200e() {
  local cid="$1" ablation="$2" seed="$3"
  local out_dir="${RUNS_ROOT}/${cid}"
  local train_log_dir="${LOG_ROOT}/${cid}"
  mkdir -p "${out_dir}" "${train_log_dir}"
  echo "[PH1-MSSL-TRAIN-BEGIN] cid=${cid} ablation=${ablation} seed=${seed} split=0.1L/0.7U/0.2Val epochs=200"
  "${PYTHON}" -u "${ROOT}/code/train.py" \
    --train_mode centralized \
    --dataset wisig \
    --wisig_pkl "${WISIG_PKL}" \
    --wisig_protocol cvs_day_rx \
    --wisig_domain rx_day \
    --wisig_equalized 1 \
    --wisig_out_len 256 \
    --wisig_train_ratio 0.1 \
    --wisig_val_ratio -1.0 \
    --wisig_split_strategy random \
    --wisig_cap_strategy random \
    --wisig_train_days 0,1 \
    --wisig_test_days 2,3 \
    --wisig_train_rxs 0,1,2,3,4,5,6 \
    --wisig_test_rxs 7,8,9,10,11 \
    --num_classes 16 \
    --arch_family cvsincnet \
    --slim_group none \
    --model_variant lite_d \
    --branch_ablation no_dac \
    --domain_branch_ablation no_stats \
    --domain_enhancer rcn_stats \
    --domain_enhancer_strength 0.35 \
    --id_time_stability_mode off \
    --id_freq_stability_mode off \
    --domain_time_stability_mode off \
    --domain_freq_stability_mode off \
    --exp_group s3_rxrobust_no_dac \
    --pa_orders 1,3,5 \
    --use_meta_ssl_cvs \
    --use_meta_rxday_episodes \
    --ssl_labeled_ratio 0.1 \
    --ssl_unlabeled_ratio 0.7 \
    --ssl_val_ratio 0.2 \
    --ssl_teacher_ema 0.995 \
    --ssl_gate_mode freematch_ups_proto \
    --ssl_min_conf 0.60 \
    --ssl_min_margin 0.02 \
    --ssl_max_uncertainty 0.35 \
    --ssl_class_quota 64 \
    --ssl_receiver_quota 16 \
    --lambda_ssl_tx 0.5 \
    --lambda_ssl_proto 0.1 \
    --lambda_meta_ssl 0.05 \
    --meta_inner_scope head_proj \
    --no_enable_pa_aux \
    --no_enable_dac_aux \
    --no_aug_enable_pa_normal \
    --aug_p_pa 0.0 \
    --aug_p_dac 0.0 \
    --lambda_cls_pa 0.0 \
    --lambda_cls_dac 0.0 \
    --lambda_pa_joint_inv 0.0 \
    --lambda_pa_kl 0.0 \
    --lambda_dac_reg 0.0 \
    --lambda_pa_reg 0.0 \
    --lambda_dom 0.50 \
    --lambda_adv 0.20 \
    --grl_lambda 1.0 \
    --lambda_orth 0.024 \
    --lambda_cons 0.012 \
    --lambda_group_ce 0.018 \
    --group_ce_mode smooth_dro_capped \
    --group_ce_min_domains 2 \
    --group_ce_top_frac 0.14 \
    --groupdro_tau 0.32 \
    --groupdro_cap 0.40 \
    --use_proto_memory \
    --proto_momentum 0.97 \
    --lambda_proto 0.0025 \
    --lambda_supcon_id 0.0025 \
    --supcon_temp 0.12 \
    --lambda_fishr 0.0002 \
    --fishr_min_domains 2 \
    --lambda_feature_norm_guard 0.00004 \
    --feature_norm_guard_mode l2 \
    --feature_norm_guard_target 0 \
    --use_aug \
    --aug_scale_min 0.10 \
    --aug_scale_max 0.32 \
    --late_aug_min_scale 0.16 \
    --use_mixstyle \
    --mixstyle_p 0.025 \
    --mixstyle_strength 0.24 \
    --mixstyle_mix same_tx_crossdomain \
    --mixstyle_fallback skip \
    --mixstyle_late_start 95 \
    --mixstyle_late_ramp_epochs 35 \
    --mixstyle_late_min_p 0.020 \
    --mixstyle_late_min_strength 0.156 \
    --no_use_concat_sat_channel_aug \
    --no_use_sat_consistency \
    --lambda_sat_cls 0.0 \
    --lambda_sat_cons 0.0 \
    --concat_sat_ce_weight 0.0 \
    --sat_cons_start_epoch 999 \
    --eval_sat_channel \
    --eval_sat_on test_unseen_day_unseen_rx \
    --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
    --sat_eval_max_batches -1 \
    --test_eval_policy interval_final \
    --test_eval_start_epoch 31 \
    --test_eval_interval 10 \
    --eval_batch_size 256 \
    --batch_size 256 \
    --epochs 200 \
    --swad_start_epoch 75 \
    --swad_interval 1 \
    --swad_tolerance 0.85 \
    --primary_udu_weight 0.82 \
    --collapse_guard \
    --collapse_guard_min_epoch 35 \
    --collapse_guard_best_margin 10.0 \
    --collapse_guard_max_skipped_delta 2 \
    --use_ema_ckpt \
    --ema_decay 0.999 \
    --use_swad_ckpt \
    --label_smoothing 0.0 \
    --device cuda:0 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --seed "${seed}" \
    --run_name "${cid}" \
    --log_dir "${train_log_dir}" \
    --latest_save_path "${out_dir}/latest_model.pth" \
    --best_save_path "${out_dir}/best_val_model.pth" \
    --best_test_save_path "${out_dir}/best_test_overall_model.pth" \
    --best_primary_save_path "${out_dir}/best_primary_ood_model.pth" \
    --best_unseen_day_unseen_rx_save_path "${out_dir}/best_strict_udu_model.pth" \
    --best_unseen_day_seen_rx_save_path "${out_dir}/best_unseen_day_seen_rx_model.pth" \
    --best_seen_day_unseen_rx_save_path "${out_dir}/best_seen_day_unseen_rx_model.pth" \
    --best_worst_rx_save_path "${out_dir}/best_worst_rx_model.pth" \
    --ema_save_path "${out_dir}/ema_model.pth" \
    --swad_save_path "${out_dir}/swad_model.pth"
  echo "[PH1-MSSL-TRAIN-END] cid=${cid}"
}


run_ftrc_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" old_support="$8" source_proto="$9" query_per_tx="${10}" seed="${11}" kappa="${12}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol ftrc --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_support_per_tx "${old_support}" --target_old_query_per_tx "${query_per_tx}" --shots 0 --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --target_old_support_per_tx "${SFE_OLD_SUPPORT_CURRENT:-0}" --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --kappa "${kappa}" --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2B-FTRC-EVAL] cid=${cid} suffix=${suffix} gate=${gate} support=${old_support} source_proto=${source_proto} query=${query_per_tx} kappa=${kappa}"
  "${cmd[@]}"
}

run_ftrc_profile() {
  local cid="$1" profile="$2" old_support="$3" source_proto="$4" query_per_tx="$5" seed="$6" kappa="$7"
  case "${profile}" in
    ftrc_strict)
      run_ftrc_eval "${cid}" "stage2b_ftrc_combined_t080_m010_mh5" "combined" "0.80" "0.10" "5.0" "0.95" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_mahal_t080_mh5" "mahalanobis" "0.80" "" "5.0" "0.95" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_openmax_t080_q095" "openmax" "0.80" "" "" "0.95" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      ;;
    ftrc_balanced)
      run_ftrc_eval "${cid}" "stage2b_ftrc_combined_t075_m005_mh6" "combined" "0.75" "0.05" "6.0" "1.0" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_mahal_t075_mh6" "mahalanobis" "0.75" "" "6.0" "0.97" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_openmax_t080_q097" "openmax" "0.80" "" "" "0.97" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      ;;
    ftrc_lowfar)
      run_ftrc_eval "${cid}" "stage2b_ftrc_combined_t085_m010_mh5" "combined" "0.85" "0.10" "5.0" "0.97" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_mahal_t080_mh5" "mahalanobis" "0.80" "" "5.0" "0.97" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      run_ftrc_eval "${cid}" "stage2b_ftrc_openmax_t085_q097" "openmax" "0.85" "" "" "0.97" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
      ;;
    *) echo "[ERROR] unknown FTRC profile: ${profile}" >&2; return 2 ;;
  esac
}

run_ftrc_eval_bundle() {
  local cid="$1" channel_view="$2" old_support="$3" seed="$4" sat_seed="$5" max_samples_per_tx="$6" source_proto="$7" query_per_tx="$8" scenarios="$9" profile="${10}" kappa="${11}"
  local out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S2B-FTRC-BEGIN] cid=${cid} channel=${channel_view} profile=${profile} old_support=${old_support} source=${SOURCE_TX_IDS} target_old=${TARGET_OLD_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS} scenarios=${scenarios}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" --ckpt "${TEACHER_CKPT}" --wisig_pkl "${WISIG_PKL}" --new_wisig_pkl "${NEW_WISIG_PKL}" --out_npz "${out_dir}/features.npz" --feature_name z_id --source_tx_ids "${SOURCE_TX_IDS}" --source_rxs "${CEN51_TRAIN_RXS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --target_old_rxs "${TARGET_RXS}" --target_old_channel_view "${channel_view}" --target_old_sat_scenarios "${scenarios}" --target_old_sat_seed "$((sat_seed + 111))" --new_tx_ids "${NEW_TX_IDS}" --new_rxs "${TARGET_RXS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo 0 --max_samples_per_tx "${max_samples_per_tx}" --batch_size 512 --device cuda:0 --seed "${seed}" --target_new_channel_view "${channel_view}" --target_new_sat_scenarios "${scenarios}" --target_new_sat_seed "${sat_seed}"
  run_ftrc_profile "${cid}" "${profile}" "${old_support}" "${source_proto}" "${query_per_tx}" "${seed}" "${kappa}"
  echo "[S2B-FTRC-END] cid=${cid}"
}

declare -a CAND_ID=(S2N116_GPU0_A_MSSL_B0_CEN51_R04_refresh_A S2N116_GPU0_B_WISIG_20X1_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU0_C_WISIG_3X19_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU0_D_WISIG_7X14_OLD_PROTO_SHRINK_K1_D S2N116_GPU0_E_WISIG_7X7_OLD_ADAPTER_K2_E S2N116_GPU0_F_WISIG_8X8_OLD_DENSITY_K5_F S2N116_GPU0_G_WISIG_20X1_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU0_H_WISIG_3X19_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU1_A_MSSL_B4_MSSL_MetaProtoDG_A S2N116_GPU1_B_WISIG_7X14_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU1_C_WISIG_7X7_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU1_D_WISIG_8X8_OLD_PROTO_SHRINK_K1_D S2N116_GPU1_E_WISIG_20X1_OLD_ADAPTER_K2_E S2N116_GPU1_F_WISIG_3X19_OLD_DENSITY_K5_F S2N116_GPU1_G_WISIG_7X14_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU1_H_WISIG_7X7_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU2_A_MSSL_B2_MSSL_TeacherFree_A S2N116_GPU2_B_WISIG_8X8_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU2_C_WISIG_20X1_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU2_D_WISIG_3X19_OLD_PROTO_SHRINK_K1_D S2N116_GPU2_E_WISIG_7X14_OLD_ADAPTER_K2_E S2N116_GPU2_F_WISIG_7X7_OLD_DENSITY_K5_F S2N116_GPU2_G_WISIG_8X8_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU2_H_WISIG_20X1_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU3_A_MSSL_B3_MSSL_ProtoGate_A S2N116_GPU3_B_WISIG_3X19_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU3_C_WISIG_7X14_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU3_D_WISIG_7X7_OLD_PROTO_SHRINK_K1_D S2N116_GPU3_E_WISIG_8X8_OLD_ADAPTER_K2_E S2N116_GPU3_F_WISIG_20X1_OLD_DENSITY_K5_F S2N116_GPU3_G_WISIG_3X19_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU3_H_WISIG_7X14_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU4_A_MSSL_B1_MSSL_Udom_split_audit_A S2N116_GPU4_B_WISIG_7X7_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU4_C_WISIG_8X8_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU4_D_WISIG_20X1_OLD_PROTO_SHRINK_K1_D S2N116_GPU4_E_WISIG_3X19_OLD_ADAPTER_K2_E S2N116_GPU4_F_WISIG_7X14_OLD_DENSITY_K5_F S2N116_GPU4_G_WISIG_7X7_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU4_H_WISIG_8X8_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU5_A_MSSL_B5_MSSL_SatGate_A S2N116_GPU5_B_WISIG_20X1_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU5_C_WISIG_3X19_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU5_D_WISIG_7X14_OLD_PROTO_SHRINK_K1_D S2N116_GPU5_E_WISIG_7X7_OLD_ADAPTER_K2_E S2N116_GPU5_F_WISIG_8X8_OLD_DENSITY_K5_F S2N116_GPU5_G_WISIG_20X1_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU5_H_WISIG_3X19_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU6_A_MSSL_B0_CEN51_R04_refresh_A S2N116_GPU6_B_WISIG_7X14_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU6_C_WISIG_7X7_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU6_D_WISIG_8X8_OLD_PROTO_SHRINK_K1_D S2N116_GPU6_E_WISIG_20X1_OLD_ADAPTER_K2_E S2N116_GPU6_F_WISIG_3X19_OLD_DENSITY_K5_F S2N116_GPU6_G_WISIG_7X14_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU6_H_WISIG_7X7_OLD_NEW_K5_QUALITY_DEFER_H S2N116_GPU7_A_MSSL_B3_MSSL_ProtoGate_A S2N116_GPU7_B_WISIG_8X8_SOURCE_OPENMAX_EVAL_ONLY_B S2N116_GPU7_C_WISIG_20X1_MAHALANOBIS_EVAL_ONLY_C S2N116_GPU7_D_WISIG_3X19_OLD_PROTO_SHRINK_K1_D S2N116_GPU7_E_WISIG_7X14_OLD_ADAPTER_K2_E S2N116_GPU7_F_WISIG_7X7_OLD_DENSITY_K5_F S2N116_GPU7_G_WISIG_8X8_OLD_NEW_K2_SEEN_NEW_G S2N116_GPU7_H_WISIG_20X1_OLD_NEW_K5_QUALITY_DEFER_H)
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=(phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe)
declare -a CAND_SLOT=(GPU0/A GPU0/B GPU0/C GPU0/D GPU0/E GPU0/F GPU0/G GPU0/H GPU1/A GPU1/B GPU1/C GPU1/D GPU1/E GPU1/F GPU1/G GPU1/H GPU2/A GPU2/B GPU2/C GPU2/D GPU2/E GPU2/F GPU2/G GPU2/H GPU3/A GPU3/B GPU3/C GPU3/D GPU3/E GPU3/F GPU3/G GPU3/H GPU4/A GPU4/B GPU4/C GPU4/D GPU4/E GPU4/F GPU4/G GPU4/H GPU5/A GPU5/B GPU5/C GPU5/D GPU5/E GPU5/F GPU5/G GPU5/H GPU6/A GPU6/B GPU6/C GPU6/D GPU6/E GPU6/F GPU6/G GPU6/H GPU7/A GPU7/B GPU7/C GPU7/D GPU7/E GPU7/F GPU7/G GPU7/H)
declare -a CAND_DESC=(MSSL_B0_CEN51_R04_refresh_A WISIG_20X1_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_3X19_MAHALANOBIS_EVAL_ONLY_C WISIG_7X14_OLD_PROTO_SHRINK_K1_D WISIG_7X7_OLD_ADAPTER_K2_E WISIG_8X8_OLD_DENSITY_K5_F WISIG_20X1_OLD_NEW_K2_SEEN_NEW_G WISIG_3X19_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B4_MSSL_MetaProtoDG_A WISIG_7X14_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_7X7_MAHALANOBIS_EVAL_ONLY_C WISIG_8X8_OLD_PROTO_SHRINK_K1_D WISIG_20X1_OLD_ADAPTER_K2_E WISIG_3X19_OLD_DENSITY_K5_F WISIG_7X14_OLD_NEW_K2_SEEN_NEW_G WISIG_7X7_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B2_MSSL_TeacherFree_A WISIG_8X8_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_20X1_MAHALANOBIS_EVAL_ONLY_C WISIG_3X19_OLD_PROTO_SHRINK_K1_D WISIG_7X14_OLD_ADAPTER_K2_E WISIG_7X7_OLD_DENSITY_K5_F WISIG_8X8_OLD_NEW_K2_SEEN_NEW_G WISIG_20X1_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B3_MSSL_ProtoGate_A WISIG_3X19_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_7X14_MAHALANOBIS_EVAL_ONLY_C WISIG_7X7_OLD_PROTO_SHRINK_K1_D WISIG_8X8_OLD_ADAPTER_K2_E WISIG_20X1_OLD_DENSITY_K5_F WISIG_3X19_OLD_NEW_K2_SEEN_NEW_G WISIG_7X14_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B1_MSSL_Udom_split_audit_A WISIG_7X7_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_8X8_MAHALANOBIS_EVAL_ONLY_C WISIG_20X1_OLD_PROTO_SHRINK_K1_D WISIG_3X19_OLD_ADAPTER_K2_E WISIG_7X14_OLD_DENSITY_K5_F WISIG_7X7_OLD_NEW_K2_SEEN_NEW_G WISIG_8X8_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B5_MSSL_SatGate_A WISIG_20X1_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_3X19_MAHALANOBIS_EVAL_ONLY_C WISIG_7X14_OLD_PROTO_SHRINK_K1_D WISIG_7X7_OLD_ADAPTER_K2_E WISIG_8X8_OLD_DENSITY_K5_F WISIG_20X1_OLD_NEW_K2_SEEN_NEW_G WISIG_3X19_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B0_CEN51_R04_refresh_A WISIG_7X14_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_7X7_MAHALANOBIS_EVAL_ONLY_C WISIG_8X8_OLD_PROTO_SHRINK_K1_D WISIG_20X1_OLD_ADAPTER_K2_E WISIG_3X19_OLD_DENSITY_K5_F WISIG_7X14_OLD_NEW_K2_SEEN_NEW_G WISIG_7X7_OLD_NEW_K5_QUALITY_DEFER_H MSSL_B3_MSSL_ProtoGate_A WISIG_8X8_SOURCE_OPENMAX_EVAL_ONLY_B WISIG_20X1_MAHALANOBIS_EVAL_ONLY_C WISIG_3X19_OLD_PROTO_SHRINK_K1_D WISIG_7X14_OLD_ADAPTER_K2_E WISIG_7X7_OLD_DENSITY_K5_F WISIG_8X8_OLD_NEW_K2_SEEN_NEW_G WISIG_20X1_OLD_NEW_K5_QUALITY_DEFER_H)
declare -a CAND_LANE=(phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a ROW_SOURCE_TX_IDS=(0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5)
declare -a ROW_TARGET_OLD_TX_IDS=(0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5 0,1,2,3,4,5)
declare -a ROW_NEW_TX_IDS=(6,7 6,7,8,9 8,9,10,11 10,11,12,13 12,13,14,15 14,15,16,17 16,17,18,19 18,19,20,21 6,7 20,21,22,23 22,23,24,25 24,25,26,27 26,27,28,29 28,29,30,31 30,31,32,33 32,33,34,35 6,7 34,35,36,37 36,37,38,39 38,39,40,41 40,41,42,43 42,43,44,45 44,45,46,47 46,47,48,49 6,7 48,49,50,51 50,51,52,53 52,53,54,55 54,55,56,57 56,57,58,59 58,59,60,61 60,61,62,63 6,7 62,63,64,65 64,65,66,67 66,67,68,69 68,69,70,71 70,71,72,73 72,73,74,75 74,75,76,77 6,7 76,77,78,79 78,79,80,81 80,81,82,83 82,83,84,85 84,85,86,87 6,7,8,9 8,9,10,11 6,7 10,11,12,13 12,13,14,15 14,15,16,17 16,17,18,19 18,19,20,21 20,21,22,23 22,23,24,25 6,7 24,25,26,27 26,27,28,29 28,29,30,31 30,31,32,33 32,33,34,35 34,35,36,37 36,37,38,39)
declare -a ROW_UNKNOWN_TX_IDS=(8,9 100,101,102,103 102,103,104,105 104,105,106,107 106,107,108,109 108,109,110,111 110,111,112,113 112,113,114,115 8,9 114,115,116,117 116,117,118,119 118,119,120,121 120,121,122,123 122,123,124,125 124,125,126,127 126,127,128,129 8,9 128,129,130,131 130,131,132,133 132,133,134,135 134,135,136,137 136,137,138,139 138,139,140,141 140,141,142,143 8,9 142,143,144,145 144,145,146,147 146,147,148,149 148,149,150,151 150,151,152,153 152,153,154,155 154,155,156,157 8,9 156,157,158,159 158,159,160,161 160,161,162,163 162,163,164,165 164,165,166,167 166,167,168,169 168,169,170,171 8,9 170,171,172,173 172,173,174,175 174,175,176,177 176,177,178,179 178,179,180,181 100,101,102,103 102,103,104,105 8,9 104,105,106,107 106,107,108,109 108,109,110,111 110,111,112,113 112,113,114,115 114,115,116,117 116,117,118,119 8,9 118,119,120,121 120,121,122,123 122,123,124,125 124,125,126,127 126,127,128,129 128,129,130,131 130,131,132,133)
declare -a ROW_TARGET_RXS=(7,8,9,10,11 20-1 3-19 7-14 7-7 8-8 20-1 3-19 7,8,9,10,11 7-14 7-7 8-8 20-1 3-19 7-14 7-7 7,8,9,10,11 8-8 20-1 3-19 7-14 7-7 8-8 20-1 7,8,9,10,11 3-19 7-14 7-7 8-8 20-1 3-19 7-14 7,8,9,10,11 7-7 8-8 20-1 3-19 7-14 7-7 8-8 7,8,9,10,11 20-1 3-19 7-14 7-7 8-8 20-1 3-19 7,8,9,10,11 7-14 7-7 8-8 20-1 3-19 7-14 7-7 7,8,9,10,11 8-8 20-1 3-19 7-14 7-7 8-8 20-1)
declare -a META_ABLATION=(B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B4_MSSL_MetaProtoDG not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B2_MSSL_TeacherFree not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B1_MSSL_Udom_split_audit not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B5_MSSL_SatGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable)
declare -a META_SEED=(88000 60007 60014 60021 60028 60035 60042 60049 88013 60063 60070 60077 60084 60091 60098 60105 88026 60119 60126 60133 60140 60147 60154 60161 88039 60175 60182 60189 60196 60203 60210 60217 88052 60231 60238 60245 60252 60259 60266 60273 88065 60287 60294 60301 60308 60315 60322 60329 88078 60343 60350 60357 60364 60371 60378 60385 88091 60399 60406 60413 60420 60427 60434 60441)
declare -a META_MAX_SAMPLES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
declare -a SFE_CHANNEL=(satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite)
declare -a SFE_PROFILE=(strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax strict_openmax source_open_set source_open_set strict_openmax strict_openmax strict_openmax balanced_openmax balanced_openmax)
declare -a SFE_SHOTS=(0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5)
declare -a SFE_OLD_SUPPORT=(0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5 0 0 0 0 0 0 2 5)
declare -a SFE_SEED=(60000 60007 60014 60021 60028 60035 60042 60049 60056 60063 60070 60077 60084 60091 60098 60105 60112 60119 60126 60133 60140 60147 60154 60161 60168 60175 60182 60189 60196 60203 60210 60217 60224 60231 60238 60245 60252 60259 60266 60273 60280 60287 60294 60301 60308 60315 60322 60329 60336 60343 60350 60357 60364 60371 60378 60385 60392 60399 60406 60413 60420 60427 60434 60441)
declare -a SFE_SAT_SEED=(61000 61007 61014 61021 61028 61035 61042 61049 61056 61063 61070 61077 61084 61091 61098 61105 61112 61119 61126 61133 61140 61147 61154 61161 61168 61175 61182 61189 61196 61203 61210 61217 61224 61231 61238 61245 61252 61259 61266 61273 61280 61287 61294 61301 61308 61315 61322 61329 61336 61343 61350 61357 61364 61371 61378 61385 61392 61399 61406 61413 61420 61427 61434 61441)
declare -a SFE_MAX_SAMPLES=(260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260)
declare -a SFE_SOURCE_PROTO=(0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80 0 80 80 80 80 80 80 80)
declare -a SFE_QUERY=(0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70)
declare -a SFE_SCENARIOS=(- clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit)
declare -a FTRC_EVAL_PROFILE=(ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_balanced ftrc_strict ftrc_lowfar ftrc_balanced ftrc_balanced)
declare -a FTRC_EVAL_SUPPORT=(0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5 0 0 0 1 2 5 2 5)
declare -a FTRC_EVAL_KAPPA=(5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0 5.0 5.0 5.0 5.0 5.0 3.0 5.0 3.0)
declare -a FTRC_EVAL_SEED=(60000 60007 60014 60021 60028 60035 60042 60049 60056 60063 60070 60077 60084 60091 60098 60105 60112 60119 60126 60133 60140 60147 60154 60161 60168 60175 60182 60189 60196 60203 60210 60217 60224 60231 60238 60245 60252 60259 60266 60273 60280 60287 60294 60301 60308 60315 60322 60329 60336 60343 60350 60357 60364 60371 60378 60385 60392 60399 60406 60413 60420 60427 60434 60441)
declare -a FTRC_EVAL_SAT_SEED=(61000 61007 61014 61021 61028 61035 61042 61049 61056 61063 61070 61077 61084 61091 61098 61105 61112 61119 61126 61133 61140 61147 61154 61161 61168 61175 61182 61189 61196 61203 61210 61217 61224 61231 61238 61245 61252 61259 61266 61273 61280 61287 61294 61301 61308 61315 61322 61329 61336 61343 61350 61357 61364 61371 61378 61385 61392 61399 61406 61413 61420 61427 61434 61441)
declare -a FTRC_EVAL_MAX_SAMPLES=(260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260)
declare -a FTRC_EVAL_SOURCE_PROTO=(0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100 0 100 100 100 100 100 100 100)
declare -a FTRC_EVAL_QUERY=(0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70 0 70 70 70 70 70 70 70)
declare -a FTRC_EVAL_SCENARIOS=(- clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

PHASE1_LANE_ACTIVE="${PHASE1_LANE_ACTIVE:-0}"
if [[ "${PHASE1_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase1" ]]; then
      CAND_STATUS[$i]="deferred_phase1_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase1" "reason=phase1_lane_active_or_capacity_retry"
    fi
  done
fi

PHASE2_LANE_ACTIVE="${PHASE2_LANE_ACTIVE:-0}"
if [[ "${PHASE2_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=phase2_lane_active_or_capacity_retry"
    fi
  done
fi

PHASE2_LOCAL_PATCH_REQUIRED="${PHASE2_LOCAL_PATCH_REQUIRED:-0}"
if [[ "${PHASE2_LOCAL_PATCH_REQUIRED}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_local_verify"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_LOCAL_VERIFY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=phase2_local_patch_required"
    fi
  done
fi

launch_candidate() {
  local i="$1"
  local cid="${CAND_ID[$i]}"
  local gpu="${CAND_GPU[$i]}"
  local kind="${CAND_KIND[$i]}"
  local log_path="${LOG_ROOT}/${cid}.out"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[S2-DRY-RUN] cid=${cid} slot=${CAND_SLOT[$i]} gpu=${gpu} kind=${kind} lane=${CAND_LANE[$i]} rx=${ROW_TARGET_RXS[$i]} new=${ROW_NEW_TX_IDS[$i]} unknown=${ROW_UNKNOWN_TX_IDS[$i]} log=${log_path}"
    CAND_STATUS[$i]="dry_run"
    return 0
  fi
  if [[ "${kind}" == "phase1_train" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_meta_ssl_training_200e "${cid}" "${META_ABLATION[$i]}" "${META_SEED[$i]}" > "${log_path}" 2>&1) &
  elif [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; \
      SOURCE_TX_IDS="${ROW_SOURCE_TX_IDS[$i]}" TARGET_OLD_TX_IDS="${ROW_TARGET_OLD_TX_IDS[$i]}" NEW_TX_IDS="${ROW_NEW_TX_IDS[$i]}" UNKNOWN_TX_IDS="${ROW_UNKNOWN_TX_IDS[$i]}" TARGET_RXS="${ROW_TARGET_RXS[$i]}" SFE_OLD_SUPPORT_CURRENT="${SFE_OLD_SUPPORT[$i]}" \
      run_sfe_bundle "${cid}" "${SFE_CHANNEL[$i]}" "${SFE_SHOTS[$i]}" "${SFE_SEED[$i]}" "${SFE_SAT_SEED[$i]}" "${SFE_MAX_SAMPLES[$i]}" "${SFE_SOURCE_PROTO[$i]}" "${SFE_QUERY[$i]}" "${SFE_SCENARIOS[$i]}" "${SFE_PROFILE[$i]}" > "${log_path}" 2>&1) &
  elif [[ "${kind}" == "ftrc_eval" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; \
      SOURCE_TX_IDS="${ROW_SOURCE_TX_IDS[$i]}" TARGET_OLD_TX_IDS="${ROW_TARGET_OLD_TX_IDS[$i]}" NEW_TX_IDS="${ROW_NEW_TX_IDS[$i]}" UNKNOWN_TX_IDS="${ROW_UNKNOWN_TX_IDS[$i]}" TARGET_RXS="${ROW_TARGET_RXS[$i]}" \
      run_ftrc_eval_bundle "${cid}" "satellite" "${FTRC_EVAL_SUPPORT[$i]}" "${FTRC_EVAL_SEED[$i]}" "${FTRC_EVAL_SAT_SEED[$i]}" "${FTRC_EVAL_MAX_SAMPLES[$i]}" "${FTRC_EVAL_SOURCE_PROTO[$i]}" "${FTRC_EVAL_QUERY[$i]}" "${FTRC_EVAL_SCENARIOS[$i]}" "${FTRC_EVAL_PROFILE[$i]}" "${FTRC_EVAL_KAPPA[$i]}" > "${log_path}" 2>&1) &
  else
    echo "[ERROR] unknown candidate kind: ${kind}" >&2
    return 2
  fi
  local pid="$!"
  CAND_PID[$i]="${pid}"
  CAND_STATUS[$i]="running"
  event_row "${cid}" "LAUNCHED" "gpu=${gpu}" "pid=${pid}" "log=${log_path}"
}

log_msg "[S2-SCHEDULER] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} phase1_max=${PHASE1_MAX_ACTIVE_PER_GPU} phase2_max=${PHASE2_MAX_ACTIVE_PER_GPU} combined_max=${COMBINED_MAX_ACTIVE_PER_GPU}"
log_msg "[S2-SPLIT] default_source=${SOURCE_TX_IDS} default_target_old=${TARGET_OLD_TX_IDS} default_new=${NEW_TX_IDS} default_unknown=${UNKNOWN_TX_IDS} cen51_train_rxs=${CEN51_TRAIN_RXS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} lane=${CAND_LANE[$i]} rx=${ROW_TARGET_RXS[$i]} desc=${CAND_DESC[$i]}"
done

source "${ROOT}/tools/stage2_queue_runner_template.sh"
stage2_run_queue_scheduler
