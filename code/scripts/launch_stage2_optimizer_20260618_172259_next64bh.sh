#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64bh_20260618_172259}"
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
TARGET_RXS="${TARGET_RXS:-7,8,9,10,11}"
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

declare -a CAND_ID=(S2N75_GPU0_A_MSSL_B0_CEN51_R04_refresh_A S2N75_GPU0_B_STAGE2A_OLD_FLOOR_P10_Q30_B S2N75_GPU0_C_STAGE2A_OLD_FLOOR_P40_Q30_C S2N75_GPU0_D_STAGE2A_OLD_FLOOR_P80_Q30_D S2N75_GPU0_E_STAGE2A_OLD_FLOOR_Q60_E S2N75_GPU0_F_STAGE2A_MIXED_LOWFAR_CONFIRM_F S2N75_GPU0_G_STAGE2A_CLEAR_SCENARIO_CTRL_G S2N75_GPU0_H_STAGE2A_LOWELEV_SCENARIO_CTRL_H S2N75_GPU1_A_MSSL_B4_MSSL_MetaProtoDG_A S2N75_GPU1_B_S2C_K5_SUPPORT_HARM_B S2N75_GPU1_C_S2C_K10_SUPPORT_HARM_C S2N75_GPU1_D_S2C_K20_SUPPORT_HARM_D S2N75_GPU1_E_S2C_K50_SUPPORT_HARM_E S2N75_GPU1_F_S2C_K100_SUPPORT_HARM_F S2N75_GPU1_G_S2C_K20_STRICT_MARGIN_G S2N75_GPU1_H_S2C_K50_BALANCED_CONFIRM_H S2N75_GPU2_A_MSSL_B2_MSSL_TeacherFree_A S2N75_GPU2_B_STAGE2A_RAINSTORM_P20_B S2N75_GPU2_C_STAGE2A_RAINSTORM_P80_C S2N75_GPU2_D_STAGE2A_MIXED_P20_D S2N75_GPU2_E_STAGE2A_MIXED_P80_E S2N75_GPU2_F_STAGE2A_ALLEO_Q45_F S2N75_GPU2_G_STAGE2A_ALLEO_Q20_G S2N75_GPU2_H_STAGE2A_LOWFAR_LONGQ_H S2N75_GPU3_A_MSSL_B3_MSSL_ProtoGate_A S2N75_GPU3_B_S2C_STRICT_K20_ALL_B S2N75_GPU3_C_S2C_BALANCED_K20_ALL_C S2N75_GPU3_D_S2C_SCORE_K20_ALL_D S2N75_GPU3_E_S2C_STRICT_K50_LOWELEV_E S2N75_GPU3_F_S2C_BALANCED_K50_RAINSTORM_F S2N75_GPU3_G_S2C_SCORE_K50_MIXED_G S2N75_GPU3_H_STAGE2A_SCORE_BASELINE_H S2N75_GPU4_A_MSSL_B1_MSSL_Udom_split_audit_A S2N75_GPU4_B_STAGE2A_NO_SUPPORT_NEG_B S2N75_GPU4_C_S2C_K5_MIN_SUPPORT_NEG_C S2N75_GPU4_D_S2C_K100_OVERFIT_NEG_D S2N75_GPU4_E_STAGE2A_RAIN_ONLY_NEG_E S2N75_GPU4_F_STAGE2A_STORM_ONLY_NEG_F S2N75_GPU4_G_S2C_K20_RAINSTORM_NEG_G S2N75_GPU4_H_S2C_K50_ALL_NEG_H S2N75_GPU5_A_MSSL_B5_MSSL_SatGate_A S2N75_GPU5_B_STAGE2A_MIXED_OPENMAX_CONFIRM_B S2N75_GPU5_C_STAGE2A_ALLEO_OPENMAX_CONFIRM_C S2N75_GPU5_D_STAGE2A_LOWELEV_OPENMAX_CONFIRM_D S2N75_GPU5_E_S2C_K20_OPENMAX_CONFIRM_E S2N75_GPU5_F_S2C_K50_OPENMAX_CONFIRM_F S2N75_GPU5_G_S2C_K100_OPENMAX_CONFIRM_G S2N75_GPU5_H_STAGE2A_HIGH_PROTO_CONFIRM_H S2N75_GPU6_A_MSSL_B0_CEN51_R04_refresh_A S2N75_GPU6_B_STAGE2A_P120_Q30_B S2N75_GPU6_C_STAGE2A_P120_Q60_C S2N75_GPU6_D_S2C_K20_P80_Q40_D S2N75_GPU6_E_S2C_K50_P80_Q40_E S2N75_GPU6_F_S2C_K100_P80_Q30_F S2N75_GPU6_G_STAGE2A_RAINSTORM_P120_G S2N75_GPU6_H_STAGE2A_MIXED_P120_H S2N75_GPU7_A_MSSL_B3_MSSL_ProtoGate_A S2N75_GPU7_B_STAGE2A_TAXONOMY_SAT_CONTROL_B S2N75_GPU7_C_STAGE2A_QUERY_SPLIT_STRESS_C S2N75_GPU7_D_S2C_SUPPORT_AUDIT_K20_D S2N75_GPU7_E_S2C_SUPPORT_AUDIT_K50_E S2N75_GPU7_F_STAGE2A_UNKNOWN_FAR_GUARD_F S2N75_GPU7_G_S2C_UNKNOWN_FAR_GUARD_K20_G S2N75_GPU7_H_STAGE2A_FINAL_CONFIRM_LOWFAR_H)
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=(phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe)
declare -a CAND_SLOT=(GPU0/A GPU0/B GPU0/C GPU0/D GPU0/E GPU0/F GPU0/G GPU0/H GPU1/A GPU1/B GPU1/C GPU1/D GPU1/E GPU1/F GPU1/G GPU1/H GPU2/A GPU2/B GPU2/C GPU2/D GPU2/E GPU2/F GPU2/G GPU2/H GPU3/A GPU3/B GPU3/C GPU3/D GPU3/E GPU3/F GPU3/G GPU3/H GPU4/A GPU4/B GPU4/C GPU4/D GPU4/E GPU4/F GPU4/G GPU4/H GPU5/A GPU5/B GPU5/C GPU5/D GPU5/E GPU5/F GPU5/G GPU5/H GPU6/A GPU6/B GPU6/C GPU6/D GPU6/E GPU6/F GPU6/G GPU6/H GPU7/A GPU7/B GPU7/C GPU7/D GPU7/E GPU7/F GPU7/G GPU7/H)
declare -a CAND_DESC=(B0_CEN51_R04_refresh old_floor_proto_density_p10 old_floor_proto_density_p40 old_floor_proto_density_p80 old_floor_query_density_q60 mixed_orbit_lowfar_confirm clear_leo_negative_control low_elev_leo_negative_control B4_MSSL_MetaProtoDG target_new_support_k5_harm_probe target_new_support_k10_harm_probe target_new_support_k20_harm_probe target_new_support_k50_harm_probe target_new_support_k100_overfit_probe strict_margin_k20_unknown_guard balanced_k50_confirm B2_MSSL_TeacherFree rainstorm_proto_p20 rainstorm_proto_p80 mixed_orbit_proto_p20 mixed_orbit_proto_p80 all_scenario_query_q45 all_scenario_query_q20 long_query_lowfar_guard B3_MSSL_ProtoGate stage2c_strict_k20_all stage2c_balanced_k20_all stage2c_score_k20_all stage2c_strict_k50_lowelev stage2c_balanced_k50_rainstorm stage2c_score_k50_mixed stage2a_score_baseline_all B1_MSSL_Udom_split_audit no_support_negative_control minimal_support_negative_control large_support_overfit_negative_control rain_only_control storm_only_control stage2c_rainstorm_support_control stage2c_k50_all_negative_control B5_MSSL_SatGate confirm_next64bg_mixed_openmax confirm_all_scenario_openmax confirm_low_elev_openmax confirm_stage2c_k20_openmax confirm_stage2c_k50_openmax confirm_stage2c_k100_openmax confirm_high_proto_old_floor B0_CEN51_R04_refresh high_proto_q30_stability high_proto_q60_stability stage2c_k20_high_proto_q40 stage2c_k50_high_proto_q40 stage2c_k100_high_proto_q30 rainstorm_high_proto_stability mixed_high_proto_stability B3_MSSL_ProtoGate taxonomy_sat_control query_split_stress target_new_support_audit_k20 target_new_support_audit_k50 unknown_far_guard_stage2a unknown_far_guard_stage2c_k20 final_confirm_lowfar_old_floor)
declare -a CAND_LANE=(phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a META_ABLATION=(B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B4_MSSL_MetaProtoDG not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B2_MSSL_TeacherFree not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B1_MSSL_Udom_split_audit not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B5_MSSL_SatGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable)
declare -a META_SEED=(47000 42003 42006 42009 42012 42015 42018 42021 47020 42103 42106 42109 42112 42115 42118 42121 47040 42203 42206 42209 42212 42215 42218 42221 47060 42303 42306 42309 42312 42315 42318 42321 47080 42403 42406 42409 42412 42415 42418 42421 47100 42503 42506 42509 42512 42515 42518 42521 47120 42603 42606 42609 42612 42615 42618 42621 47140 42703 42706 42709 42712 42715 42718 42721)
declare -a META_MAX_SAMPLES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
declare -a SFE_CHANNEL=(- satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite)
declare -a SFE_PROFILE=(- source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set - score_diag score_diag score_diag balanced_openmax score_diag strict_openmax balanced_openmax - source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set - strict_openmax balanced_openmax score_diag strict_openmax balanced_openmax score_diag source_open_set - source_open_set strict_openmax balanced_openmax source_open_set source_open_set score_diag score_diag - source_open_set source_open_set source_open_set balanced_openmax balanced_openmax balanced_openmax source_open_set - source_open_set source_open_set score_diag balanced_openmax score_diag source_open_set source_open_set - source_open_set source_open_set score_diag balanced_openmax source_open_set strict_openmax source_open_set)
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 0 5 10 20 50 100 20 50 0 0 0 0 0 0 0 0 0 20 20 20 50 50 50 0 0 0 5 100 0 0 20 50 0 0 0 0 20 50 100 0 0 0 0 20 50 100 0 0 0 0 0 20 50 0 20 0)
declare -a SFE_SEED=(0 42003 42006 42009 42012 42015 42018 42021 0 42103 42106 42109 42112 42115 42118 42121 0 42203 42206 42209 42212 42215 42218 42221 0 42303 42306 42309 42312 42315 42318 42321 0 42403 42406 42409 42412 42415 42418 42421 0 42503 42506 42509 42512 42515 42518 42521 0 42603 42606 42609 42612 42615 42618 42621 0 42703 42706 42709 42712 42715 42718 42721)
declare -a SFE_SAT_SEED=(0 43003 43006 43009 43012 43015 43018 43021 0 43103 43106 43109 43112 43115 43118 43121 0 43203 43206 43209 43212 43215 43218 43221 0 43303 43306 43309 43312 43315 43318 43321 0 43403 43406 43409 43412 43415 43418 43421 0 43503 43506 43509 43512 43515 43518 43521 0 43603 43606 43609 43612 43615 43618 43621 0 43703 43706 43709 43712 43715 43718 43721)
declare -a SFE_MAX_SAMPLES=(0 180 180 220 220 180 180 180 0 160 160 180 180 220 180 180 0 180 220 180 220 220 160 240 0 180 180 180 180 180 180 180 0 180 160 220 180 180 180 180 0 180 180 180 180 180 220 240 0 240 240 220 220 220 240 240 0 180 220 180 180 180 180 220)
declare -a SFE_SOURCE_PROTO=(0 10 40 80 40 20 20 20 0 30 30 40 40 50 40 30 0 20 80 20 80 40 40 60 0 40 40 40 40 40 40 40 0 30 30 50 30 30 30 30 0 20 20 20 30 30 50 120 0 120 120 80 80 80 120 120 0 40 40 40 40 40 40 60)
declare -a SFE_QUERY=(0 30 30 30 60 30 30 30 0 35 35 40 35 30 40 35 0 30 30 30 30 45 20 60 0 35 35 35 35 35 35 35 0 35 35 30 35 35 35 35 0 30 30 30 35 35 30 30 0 30 60 40 40 30 30 30 0 35 60 35 35 35 35 45)
declare -a SFE_SCENARIOS=(- clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit mixed_orbit clear_leo low_elev_leo - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - rain_leo,storm_mp rain_leo,storm_mp mixed_orbit mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit low_elev_leo rain_leo,storm_mp mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo storm_mp rain_leo,storm_mp clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit low_elev_leo clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp mixed_orbit - clear_leo,low_elev_leo clear_leo,low_elev_leo clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit)
declare -a FTRC_ADAPTER=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_K=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_LR=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_ANCHOR=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_ALPHA=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_SEED=(14600 14602 14603 14604 14605 14606 14607 14608 14601 14610 14611 14612 14613 14614 14615 14616 14602 14618 14619 14620 14621 14622 14623 14624 14603 14626 14627 14628 14629 14630 14631 14632 14604 14634 14635 14636 14637 14638 14639 14640 14605 14642 14643 14644 14645 14646 14647 14648 14606 14650 14651 14652 14653 14654 14655 14656 14607 14658 14659 14660 14661 14662 14663 14664)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

PHASE1_LANE_ACTIVE="${PHASE1_LANE_ACTIVE:-0}"
if [[ "${PHASE1_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase1" ]]; then
      CAND_STATUS[$i]="deferred_phase1_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase1" "reason=phase1_lane_active_or_capacity_retry;exact_retry=RUN_ID=${RUN_ID}_phase1_retry PHASE1_LANE_ACTIVE=0 PHASE2_LANE_ACTIVE=1 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260618_172259_next64bh.sh"
    fi
  done
fi


PHASE2_LANE_ACTIVE="${PHASE2_LANE_ACTIVE:-0}"
if [[ "${PHASE2_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=phase2_lane_active_or_capacity_retry;exact_retry=RUN_ID=${RUN_ID}_phase2_retry PHASE1_LANE_ACTIVE=1 PHASE2_LANE_ACTIVE=0 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260618_172259_next64bh.sh"
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
  elif [[ "${kind}" == "meta_ssl" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_meta_ssl_protocol "${cid}" "${META_ABLATION[$i]}" "${META_SEED[$i]}" "${META_MAX_SAMPLES[$i]}" > "${log_path}" 2>&1) &
  elif [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "${SFE_CHANNEL[$i]}" "${SFE_SHOTS[$i]}" "${SFE_SEED[$i]}" "${SFE_SAT_SEED[$i]}" "${SFE_MAX_SAMPLES[$i]}" "${SFE_SOURCE_PROTO[$i]}" "${SFE_QUERY[$i]}" "${SFE_SCENARIOS[$i]}" "${SFE_PROFILE[$i]}" > "${log_path}" 2>&1) &
  else
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "${FTRC_ADAPTER[$i]}" "${FTRC_K[$i]}" "${FTRC_LR[$i]}" "${FTRC_ANCHOR[$i]}" "${FTRC_ALPHA[$i]}" "${FTRC_RANK[$i]}" "${FTRC_EPOCHS[$i]}" "${FTRC_STEPS[$i]}" "${FTRC_EVAL_DETAIL[$i]}" "${FTRC_SEED[$i]}" > "${log_path}" 2>&1) &
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
