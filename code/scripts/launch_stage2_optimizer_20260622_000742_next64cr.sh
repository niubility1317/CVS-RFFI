#!/usr/bin/env bash
# Generated locally for next64cr from next64cq after fresh dual-idle monitor gate; Phase1 seeds +1000 from next64cq; Phase2 deferred for route-duplication/local-hook repair.
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64cr_20260622_000742}"
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
TARGET_RXS="${TARGET_RXS:-7}"
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
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol sfe --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_query_per_tx "${query_per_tx}" --shots "${shots}" --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2-SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin} max_mahal=${max_mahal} openmax_q=${openmax_q}"
  "${cmd[@]}"
}

run_source_open_set_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" source_proto="$8" query_per_tx="$9" seed="${10}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol source_open_set --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_query_per_tx "${query_per_tx}" --shots 0 --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
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
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol ftrc --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --target_old_support_per_tx "${old_support}" --target_old_query_per_tx "${query_per_tx}" --shots 0 --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --kappa "${kappa}" --seed "${seed}")
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

declare -a CAND_ID=(S2N111_GPU0_A_MSSL_B0_CEN51_R04_refresh_A S2N111_GPU1_A_MSSL_B4_MSSL_MetaProtoDG_A S2N111_GPU2_A_MSSL_B2_MSSL_TeacherFree_A S2N111_GPU3_A_MSSL_B3_MSSL_ProtoGate_A S2N111_GPU4_A_MSSL_B1_MSSL_Udom_split_audit_A S2N111_GPU5_A_MSSL_B5_MSSL_SatGate_A S2N111_GPU6_A_MSSL_B0_CEN51_R04_refresh_A S2N111_GPU7_A_MSSL_B3_MSSL_ProtoGate_A S2N111_GPU0_B_STAGE2A_FRONTIER_PROTO20_Q80_B S2N111_GPU0_C_STAGE2A_FRONTIER_PROTO20_Q90_C S2N111_GPU0_D_S2B_K1_P40_Q70_LOWFAR_D S2N111_GPU0_E_S2B_K2_P40_Q70_LOWFAR_E S2N111_GPU0_F_S2B_K4_P40_Q70_STRICT_F S2N111_GPU0_G_STAGE2A_FPR95_PROTO60_Q80_G S2N111_GPU0_H_S2B_K2_P60_Q80_BAL_H S2N111_GPU1_B_STAGE2A_FRONTIER_PROTO60_Q70_B S2N111_GPU1_C_S2B_K1_P60_Q70_STRICT_C S2N111_GPU1_D_S2B_K2_P60_Q70_LOWFAR_D S2N111_GPU1_E_S2B_K4_P60_Q70_LOWFAR_E S2N111_GPU1_F_S2B_K8_P60_Q70_OVERFIT_NEG_F S2N111_GPU1_G_STAGE2A_LOWPROTO_NEG_Q60_G S2N111_GPU1_H_S2B_LOWPROTO_K1_NEG_Q60_H S2N111_GPU2_B_STAGE2A_CLEAR_FRONTIER_Q80_B S2N111_GPU2_C_STAGE2A_LOWELEV_FRONTIER_Q80_C S2N111_GPU2_D_STAGE2A_RAIN_FRONTIER_Q80_D S2N111_GPU2_E_STAGE2A_STORM_FRONTIER_Q80_E S2N111_GPU2_F_STAGE2A_MIXED_FRONTIER_Q80_F S2N111_GPU2_G_S2B_MIXED_K1_LOWFAR_Q80_G S2N111_GPU2_H_S2B_MIXED_K2_STRICT_Q80_H S2N111_GPU3_B_S2B_CLEAR_K1_LOWFAR_Q80_B S2N111_GPU3_C_S2B_LOWELEV_K1_LOWFAR_Q80_C S2N111_GPU3_D_S2B_RAIN_K1_LOWFAR_Q80_D S2N111_GPU3_E_S2B_STORM_K1_LOWFAR_Q80_E S2N111_GPU3_F_S2B_CLEARLOW_K2_BAL_Q70_F S2N111_GPU3_G_S2B_RAINSTORM_K2_BAL_Q70_G S2N111_GPU3_H_S2B_ALLSCEN_K2_STRICT_Q90_H S2N111_GPU4_B_STAGE2A_PROTO100_Q80_CONFIRM_B S2N111_GPU4_C_STAGE2A_PROTO140_Q70_AUROC_C S2N111_GPU4_D_S2B_K1_P100_Q80_LOWFAR_D S2N111_GPU4_E_S2B_K2_P100_Q80_LOWFAR_E S2N111_GPU4_F_S2B_K4_P100_Q80_BAL_F S2N111_GPU4_G_STAGE2A_K5_HARM_LOWPROTO_G S2N111_GPU4_H_STAGE2A_K20_HARM_FRONTIER_H S2N111_GPU5_B_S2B_K1_P140_Q60_FPR95_B S2N111_GPU5_C_S2B_K2_P140_Q60_FPR95_C S2N111_GPU5_D_S2B_K4_P140_Q60_LOWFAR_D S2N111_GPU5_E_STAGE2A_PROTO160_Q60_FPR95_E S2N111_GPU5_F_S2B_K1_P160_Q60_STRICT_F S2N111_GPU5_G_S2B_K2_P160_Q70_LOWFAR_G S2N111_GPU5_H_S2B_K4_P160_Q70_STRICT_H S2N111_GPU6_B_STAGE2A_TXRX_PROTO60_Q70_B S2N111_GPU6_C_S2B_TXRX_K1_P60_Q70_C S2N111_GPU6_D_S2B_TXRX_K2_P60_Q70_D S2N111_GPU6_E_S2B_TXRX_K4_P60_Q70_E S2N111_GPU6_F_STAGE2A_HARDONLY_PROTO80_Q80_F S2N111_GPU6_G_S2B_HARDONLY_K2_Q80_LOWFAR_G S2N111_GPU6_H_S2B_HARDONLY_K4_Q80_STRICT_H S2N111_GPU7_B_STAGE2A_UNKNOWN_GUARD_PROTO80_B S2N111_GPU7_C_S2B_UNKNOWN_GUARD_K1_Q70_C S2N111_GPU7_D_S2B_UNKNOWN_GUARD_K2_Q70_D S2N111_GPU7_E_S2B_UNKNOWN_GUARD_K4_Q70_E S2N111_GPU7_F_STAGE2A_HIGHFAR_REFUTE_PROTO20_Q40_F S2N111_GPU7_G_STAGE2A_K10_HARM_NEG_STRICT_G S2N111_GPU7_H_S2B_FINAL_FRONTIER_K2_P120_Q80_H)
declare -a CAND_GPU=(0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3 4 4 4 4 4 4 4 5 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7)
declare -a CAND_KIND=(phase1_train phase1_train phase1_train phase1_train phase1_train phase1_train phase1_train phase1_train sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe ftrc_eval sfe ftrc_eval ftrc_eval ftrc_eval ftrc_eval sfe ftrc_eval sfe sfe sfe sfe sfe ftrc_eval ftrc_eval ftrc_eval ftrc_eval ftrc_eval ftrc_eval ftrc_eval ftrc_eval ftrc_eval sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe ftrc_eval ftrc_eval ftrc_eval sfe ftrc_eval ftrc_eval ftrc_eval sfe ftrc_eval ftrc_eval ftrc_eval sfe ftrc_eval ftrc_eval sfe ftrc_eval ftrc_eval ftrc_eval sfe sfe ftrc_eval)
declare -a CAND_SLOT=(GPU0/A GPU1/A GPU2/A GPU3/A GPU4/A GPU5/A GPU6/A GPU7/A GPU0/B GPU0/C GPU0/D GPU0/E GPU0/F GPU0/G GPU0/H GPU1/B GPU1/C GPU1/D GPU1/E GPU1/F GPU1/G GPU1/H GPU2/B GPU2/C GPU2/D GPU2/E GPU2/F GPU2/G GPU2/H GPU3/B GPU3/C GPU3/D GPU3/E GPU3/F GPU3/G GPU3/H GPU4/B GPU4/C GPU4/D GPU4/E GPU4/F GPU4/G GPU4/H GPU5/B GPU5/C GPU5/D GPU5/E GPU5/F GPU5/G GPU5/H GPU6/B GPU6/C GPU6/D GPU6/E GPU6/F GPU6/G GPU6/H GPU7/B GPU7/C GPU7/D GPU7/E GPU7/F GPU7/G GPU7/H)
declare -a CAND_DESC=(B0_CEN51_R04_refresh_A B4_MSSL_MetaProtoDG_A B2_MSSL_TeacherFree_A B3_MSSL_ProtoGate_A B1_MSSL_Udom_split_audit_A B5_MSSL_SatGate_A B0_CEN51_R04_refresh_A B3_MSSL_ProtoGate_A FRONTIER_PROTO20_Q80_B FRONTIER_PROTO20_Q90_C K1_P40_Q70_LOWFAR_D K2_P40_Q70_LOWFAR_E K4_P40_Q70_STRICT_F FPR95_PROTO60_Q80_G K2_P60_Q80_BAL_H FRONTIER_PROTO60_Q70_B K1_P60_Q70_STRICT_C K2_P60_Q70_LOWFAR_D K4_P60_Q70_LOWFAR_E K8_P60_Q70_OVERFIT_NEG_F LOWPROTO_NEG_Q60_G LOWPROTO_K1_NEG_Q60_H CLEAR_FRONTIER_Q80_B LOWELEV_FRONTIER_Q80_C RAIN_FRONTIER_Q80_D STORM_FRONTIER_Q80_E MIXED_FRONTIER_Q80_F MIXED_K1_LOWFAR_Q80_G MIXED_K2_STRICT_Q80_H CLEAR_K1_LOWFAR_Q80_B LOWELEV_K1_LOWFAR_Q80_C RAIN_K1_LOWFAR_Q80_D STORM_K1_LOWFAR_Q80_E CLEARLOW_K2_BAL_Q70_F RAINSTORM_K2_BAL_Q70_G ALLSCEN_K2_STRICT_Q90_H PROTO100_Q80_CONFIRM_B PROTO140_Q70_AUROC_C K1_P100_Q80_LOWFAR_D K2_P100_Q80_LOWFAR_E K4_P100_Q80_BAL_F K5_HARM_LOWPROTO_G K20_HARM_FRONTIER_H K1_P140_Q60_FPR95_B K2_P140_Q60_FPR95_C K4_P140_Q60_LOWFAR_D PROTO160_Q60_FPR95_E K1_P160_Q60_STRICT_F K2_P160_Q70_LOWFAR_G K4_P160_Q70_STRICT_H TXRX_PROTO60_Q70_B TXRX_K1_P60_Q70_C TXRX_K2_P60_Q70_D TXRX_K4_P60_Q70_E HARDONLY_PROTO80_Q80_F HARDONLY_K2_Q80_LOWFAR_G HARDONLY_K4_Q80_STRICT_H UNKNOWN_GUARD_PROTO80_B UNKNOWN_GUARD_K1_Q70_C UNKNOWN_GUARD_K2_Q70_D UNKNOWN_GUARD_K4_Q70_E HIGHFAR_REFUTE_PROTO20_Q40_F K10_HARM_NEG_STRICT_G FINAL_FRONTIER_K2_P120_Q80_H)
declare -a CAND_LANE=(phase1 phase1 phase1 phase1 phase1 phase1 phase1 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a META_ABLATION=(B0_CEN51_R04_refresh B4_MSSL_MetaProtoDG B2_MSSL_TeacherFree B3_MSSL_ProtoGate B1_MSSL_Udom_split_audit B5_MSSL_SatGate B0_CEN51_R04_refresh B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable)
declare -a META_SEED=(84000 84013 84026 84039 84052 84065 84078 84091 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000)
declare -a META_MAX_SAMPLES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
declare -a SFE_CHANNEL=(satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite satellite)
declare -a SFE_PROFILE=(- - - - - - - - source_open_set source_open_set - - - source_open_set - source_open_set - - - - source_open_set - source_open_set source_open_set source_open_set source_open_set source_open_set - - - - - - - - - source_open_set source_open_set - - - source_open_set source_open_set - - - source_open_set - - - source_open_set - - - source_open_set - - source_open_set - - - source_open_set source_open_set -)
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a SFE_SEED=(60000 60007 60014 60021 60028 60035 60042 60049 60056 60063 60070 60077 60084 60091 60098 60105 60112 60119 60126 60133 60140 60147 60154 60161 60168 60175 60182 60189 60196 60203 60210 60217 60224 60231 60238 60245 60252 60259 60266 60273 60280 60287 60294 60301 60308 60315 60322 60329 60336 60343 60350 60357 60364 60371 60378 60385 60392 60399 60406 60413 60420 60427 60434 60441)
declare -a SFE_SAT_SEED=(61000 61007 61014 61021 61028 61035 61042 61049 61056 61063 61070 61077 61084 61091 61098 61105 61112 61119 61126 61133 61140 61147 61154 61161 61168 61175 61182 61189 61196 61203 61210 61217 61224 61231 61238 61245 61252 61259 61266 61273 61280 61287 61294 61301 61308 61315 61322 61329 61336 61343 61350 61357 61364 61371 61378 61385 61392 61399 61406 61413 61420 61427 61434 61441)
declare -a SFE_MAX_SAMPLES=(260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260)
declare -a SFE_SOURCE_PROTO=(0 0 0 0 0 0 0 0 20 20 40 40 40 60 60 60 60 60 60 60 10 10 60 60 60 60 60 60 60 80 80 80 80 80 80 80 100 140 100 100 100 40 60 140 140 140 160 160 160 160 60 60 60 60 80 80 80 80 80 80 80 20 40 120)
declare -a SFE_QUERY=(0 0 0 0 0 0 0 0 80 90 70 70 70 80 80 70 70 70 70 70 60 60 80 80 80 80 80 80 80 80 80 80 80 70 70 90 80 70 80 80 80 50 50 60 60 60 60 60 70 70 70 70 70 70 80 80 80 70 70 70 70 40 50 80)
declare -a SFE_SCENARIOS=(- - - - - - - - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo low_elev_leo rain_leo storm_mp mixed_orbit mixed_orbit mixed_orbit clear_leo low_elev_leo rain_leo storm_mp clear_leo,low_elev_leo rain_leo,storm_mp clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp rain_leo,storm_mp rain_leo,storm_mp rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit)
declare -a FTRC_EVAL_PROFILE=(- - - - - - - - - - ftrc_lowfar ftrc_lowfar ftrc_strict - ftrc_balanced - ftrc_strict ftrc_lowfar ftrc_lowfar ftrc_balanced - ftrc_strict - - - - - ftrc_lowfar ftrc_strict ftrc_lowfar ftrc_lowfar ftrc_lowfar ftrc_lowfar ftrc_balanced ftrc_balanced ftrc_strict - - ftrc_lowfar ftrc_lowfar ftrc_balanced - - ftrc_balanced ftrc_balanced ftrc_lowfar - ftrc_strict ftrc_lowfar ftrc_strict - ftrc_strict ftrc_lowfar ftrc_balanced - ftrc_lowfar ftrc_strict - ftrc_lowfar ftrc_lowfar ftrc_balanced - - ftrc_lowfar)
declare -a FTRC_EVAL_SUPPORT=(0 0 0 0 0 0 0 0 0 0 1 2 4 0 2 0 1 2 4 8 0 1 0 0 0 0 0 1 2 1 1 1 1 2 2 2 0 0 1 2 4 0 0 1 2 4 0 1 2 4 0 1 2 4 0 2 4 0 1 2 4 0 0 2)
declare -a FTRC_EVAL_KAPPA=(3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 8.0 8.0 5.0 3.0 5.0 3.0 5.0 8.0 8.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 8.0 5.0 8.0 8.0 8.0 8.0 5.0 5.0 5.0 3.0 3.0 8.0 8.0 5.0 3.0 3.0 5.0 5.0 8.0 3.0 5.0 8.0 5.0 3.0 5.0 8.0 5.0 3.0 8.0 5.0 3.0 8.0 8.0 5.0 3.0 3.0 8.0)
declare -a FTRC_EVAL_SEED=(60000 60007 60014 60021 60028 60035 60042 60049 60056 60063 60070 60077 60084 60091 60098 60105 60112 60119 60126 60133 60140 60147 60154 60161 60168 60175 60182 60189 60196 60203 60210 60217 60224 60231 60238 60245 60252 60259 60266 60273 60280 60287 60294 60301 60308 60315 60322 60329 60336 60343 60350 60357 60364 60371 60378 60385 60392 60399 60406 60413 60420 60427 60434 60441)
declare -a FTRC_EVAL_SAT_SEED=(61000 61007 61014 61021 61028 61035 61042 61049 61056 61063 61070 61077 61084 61091 61098 61105 61112 61119 61126 61133 61140 61147 61154 61161 61168 61175 61182 61189 61196 61203 61210 61217 61224 61231 61238 61245 61252 61259 61266 61273 61280 61287 61294 61301 61308 61315 61322 61329 61336 61343 61350 61357 61364 61371 61378 61385 61392 61399 61406 61413 61420 61427 61434 61441)
declare -a FTRC_EVAL_MAX_SAMPLES=(260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260)
declare -a FTRC_EVAL_SOURCE_PROTO=(0 0 0 0 0 0 0 0 20 20 40 40 40 60 60 60 60 60 60 60 10 10 60 60 60 60 60 60 60 80 80 80 80 80 80 80 100 140 100 100 100 40 60 140 140 140 160 160 160 160 60 60 60 60 80 80 80 80 80 80 80 20 40 120)
declare -a FTRC_EVAL_QUERY=(0 0 0 0 0 0 0 0 80 90 70 70 70 80 80 70 70 70 70 70 60 60 80 80 80 80 80 80 80 80 80 80 80 70 70 90 80 70 80 80 80 50 50 60 60 60 60 60 70 70 70 70 70 70 80 80 80 70 70 70 70 40 50 80)
declare -a FTRC_EVAL_SCENARIOS=(- - - - - - - - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo low_elev_leo rain_leo storm_mp mixed_orbit mixed_orbit mixed_orbit clear_leo low_elev_leo rain_leo storm_mp clear_leo,low_elev_leo rain_leo,storm_mp clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp rain_leo,storm_mp rain_leo,storm_mp rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit)

declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

PHASE1_LANE_ACTIVE="${PHASE1_LANE_ACTIVE:-0}"
if [[ "${PHASE1_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase1" ]]; then
      CAND_STATUS[$i]="deferred_phase1_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase1" "reason=phase1_lane_active_or_capacity_retry;exact_retry=RUN_ID=${RUN_ID}_phase1_retry PHASE1_LANE_ACTIVE=0 PHASE2_LANE_ACTIVE=1 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260622_000742_next64cr.sh"
    fi
  done
fi

PHASE2_LANE_ACTIVE="${PHASE2_LANE_ACTIVE:-0}"
if [[ "${PHASE2_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=phase2_lane_active_or_capacity_retry;exact_retry=RUN_ID=${RUN_ID}_phase2_retry PHASE1_LANE_ACTIVE=1 PHASE2_LANE_ACTIVE=0 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260622_000742_next64cr.sh"
    fi
  done
fi

PHASE2_LOCAL_PATCH_REQUIRED="${PHASE2_LOCAL_PATCH_REQUIRED:-1}"
if [[ "${PHASE2_LOCAL_PATCH_REQUIRED}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_local_verify"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_LOCAL_VERIFY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=route_duplication_repair_required;exact_retry=BLOCKED_UNTIL_PHASE2_ROUTE_REPAIR_NEW_REPAIRED_LAUNCHER_REQUIRED"
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
    echo "[S2-DRY-RUN] cid=${cid} slot=${CAND_SLOT[$i]} gpu=${gpu} kind=${kind} desc=${CAND_DESC[$i]} log=${log_path}"
    CAND_STATUS[$i]="dry_run"
    return 0
  fi
  if [[ "${kind}" == "phase1_train" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_meta_ssl_training_200e "${cid}" "${META_ABLATION[$i]}" "${META_SEED[$i]}" > "${log_path}" 2>&1) &
  elif [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "${SFE_CHANNEL[$i]}" "${SFE_SHOTS[$i]}" "${SFE_SEED[$i]}" "${SFE_SAT_SEED[$i]}" "${SFE_MAX_SAMPLES[$i]}" "${SFE_SOURCE_PROTO[$i]}" "${SFE_QUERY[$i]}" "${SFE_SCENARIOS[$i]}" "${SFE_PROFILE[$i]}" > "${log_path}" 2>&1) &
  elif [[ "${kind}" == "ftrc_eval" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_eval_bundle "${cid}" "satellite" "${FTRC_EVAL_SUPPORT[$i]}" "${FTRC_EVAL_SEED[$i]}" "${FTRC_EVAL_SAT_SEED[$i]}" "${FTRC_EVAL_MAX_SAMPLES[$i]}" "${FTRC_EVAL_SOURCE_PROTO[$i]}" "${FTRC_EVAL_QUERY[$i]}" "${FTRC_EVAL_SCENARIOS[$i]}" "${FTRC_EVAL_PROFILE[$i]}" "${FTRC_EVAL_KAPPA[$i]}" > "${log_path}" 2>&1) &
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
log_msg "[S2-SPLIT] source=${SOURCE_TX_IDS} target_old=${TARGET_OLD_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS} cen51_train_rxs=${CEN51_TRAIN_RXS} target_rxs=${TARGET_RXS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} desc=${CAND_DESC[$i]}"
done

source "${ROOT}/tools/stage2_queue_runner_template.sh"
stage2_run_queue_scheduler
