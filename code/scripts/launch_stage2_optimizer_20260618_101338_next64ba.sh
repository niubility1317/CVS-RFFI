#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64ba_20260618_101338}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
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
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol sfe --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --shots "${shots}" --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
  [[ -n "${margin}" ]] && cmd+=(--min_margin "${margin}")
  [[ -n "${max_mahal}" ]] && cmd+=(--max_mahalanobis "${max_mahal}")
  echo "[S2-SFE-EVAL] cid=${cid} suffix=${suffix} gate=${gate} threshold=${threshold} margin=${margin} max_mahal=${max_mahal} openmax_q=${openmax_q}"
  "${cmd[@]}"
}

run_source_open_set_eval() {
  local cid="$1" suffix="$2" gate="$3" threshold="$4" margin="$5" max_mahal="$6" openmax_q="$7" source_proto="$8" query_per_tx="$9" seed="${10}"
  local out_dir="${RUNS_ROOT}/${cid}"
  local cmd=("${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol source_open_set --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics_${suffix}.json" --manifest_json "${out_dir}/manifest_${suffix}.json" --score_table_csv "${out_dir}/score_table_${suffix}.csv" --source_tx_ids "${SOURCE_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --shots 0 --source_proto_per_tx "${source_proto}" --source_query_per_tx 20 --query_per_tx "${query_per_tx}" --unknown_threshold "${threshold}" --gate_mode "${gate}" --openmax_tail_size 20 --openmax_quantile "${openmax_q}" --openmax_min_threshold 0.02 --seed "${seed}")
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
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" --ckpt "${TEACHER_CKPT}" --wisig_pkl "${WISIG_PKL}" --new_wisig_pkl "${NEW_WISIG_PKL}" --out_npz "${out_dir}/features.npz" --feature_name z_id --source_tx_ids "${SOURCE_TX_IDS}" --new_tx_ids "${NEW_TX_IDS}" --unknown_tx_ids "${UNKNOWN_TX_IDS}" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo 0 --max_samples_per_tx "${max_samples_per_tx}" --batch_size 512 --device cuda:0 --seed "${seed}" --target_new_channel_view "${channel_view}" --target_new_sat_scenarios "${scenarios}" --target_new_sat_seed "${sat_seed}"
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

declare -a CAND_ID=(S2N68_GPU0_A_MSSL_B0_CEN51_R04_refresh_A S2N68_GPU0_B_STAGE2A_SAT_LOWFAR_CONTROL_B_B S2N68_GPU0_C_STAGE2A_CLEAN_SOURCE_CONTROL_C_C S2N68_GPU0_D_STAGE2A_LOW_ELEV_FLOOR_D_D S2N68_GPU0_E_STAGE2A_RAIN_STORM_FLOOR_F_F S2N68_GPU0_F_STAGE2A_MIXED_ORBIT_SCORE_G_G S2N68_GPU0_G_STAGE2A_SAT_LOWFAR_CONTROL_B_B_CONFIRMG S2N68_GPU0_H_STAGE2A_CLEAN_SOURCE_CONTROL_C_C_CONFIRMH S2N68_GPU1_A_MSSL_B4_MSSL_MetaProtoDG_E S2N68_GPU1_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N68_GPU1_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N68_GPU1_D_SRF_MP_PROTO_COUNT_K100_D S2N68_GPU1_E_SRF_MP_STRICT_MARGIN_K50_F S2N68_GPU1_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N68_GPU1_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N68_GPU1_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N68_GPU2_A_MSSL_B2_MSSL_TeacherFree_H S2N68_GPU2_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N68_GPU2_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N68_GPU2_D_SRF_MP_PROTO_COUNT_K100_D S2N68_GPU2_E_SRF_MP_STRICT_MARGIN_K50_F S2N68_GPU2_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N68_GPU2_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N68_GPU2_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N68_GPU3_A_MSSL_B3_MSSL_ProtoGate_A S2N68_GPU3_B_MULTICLASS_BASE_ANCHOR_CONTROL_B S2N68_GPU3_C_MULTICLASS_LOWMARGIN_BOUND_C S2N68_GPU3_D_MULTICLASS_SCORETABLE_ROLLBACK_AUDIT_D S2N68_GPU3_E_SGC_PROTOBANK_DENSITY_GATE_F S2N68_GPU3_F_SGC_SAT_EVIDENCE_ENCODER_SCORE_G S2N68_GPU3_G_MULTICLASS_BASE_ANCHOR_CONTROL_B_CONFIRMG S2N68_GPU3_H_MULTICLASS_LOWMARGIN_BOUND_C_CONFIRMH S2N68_GPU4_A_MSSL_B1_MSSL_Udom_split_audit_E S2N68_GPU4_B_SGC_DELTA_Z_PROXY_SCORE_DIAG_B S2N68_GPU4_C_SGC_LOGIT_GAP_PROXY_OPENMAX_C S2N68_GPU4_D_SGC_PROTOBANK_GUARD_K100_D S2N68_GPU4_E_SGC_OLD_FLOOR_SOURCE_OPEN_SET_F S2N68_GPU4_F_SGC_SAT_ENCODER_MIXED_ORBIT_G S2N68_GPU4_G_SGC_DELTA_Z_PROXY_SCORE_DIAG_B_CONFIRMG S2N68_GPU4_H_SGC_LOGIT_GAP_PROXY_OPENMAX_C_CONFIRMH S2N68_GPU5_A_MSSL_B5_MSSL_SatGate_H S2N68_GPU5_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N68_GPU5_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N68_GPU5_D_SRF_MP_PROTO_COUNT_K100_D S2N68_GPU5_E_SRF_MP_STRICT_MARGIN_K50_F S2N68_GPU5_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N68_GPU5_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N68_GPU5_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N68_GPU6_A_MSSL_B0_CEN51_R04_refresh_A S2N68_GPU6_B_STAGE2A_HARD_RAIN_CONTROL_B S2N68_GPU6_C_SGC_HARD_RAIN_SCORE_DIAG_C S2N68_GPU6_D_SGC_LOWELEV_DELTA_LOGIT_PROXY_D S2N68_GPU6_E_SGC_STORM_FEATURE_GUARD_F S2N68_GPU6_F_SGC_MIXED_ORBIT_STATE_PROXY_G S2N68_GPU6_G_SGC_ALL_SCENARIO_LONG_DIAG_H S2N68_GPU6_H_STAGE2A_HARD_RAIN_CONTROL_B_CONFIRMH S2N68_GPU7_A_MSSL_B3_MSSL_ProtoGate_E S2N68_GPU7_B_STAGE2A_TAXONOMY_SAT_CONTROL_B S2N68_GPU7_C_STAGE2A_TAXONOMY_CLEAN_CONTROL_C S2N68_GPU7_D_TELEMETRY_SCORETABLE_LOW_FAR_D S2N68_GPU7_E_TELEMETRY_MANIFEST_PROTOCOL_F S2N68_GPU7_F_TELEMETRY_RUNTIME_PROXY_G S2N68_GPU7_G_TELEMETRY_STATE_SIZE_LONG_DIAG_H S2N68_GPU7_H_STAGE2A_TAXONOMY_SAT_CONTROL_B_CONFIRMH)
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=(phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe phase1_train sfe sfe sfe sfe sfe sfe sfe)
declare -a CAND_SLOT=(GPU0/A GPU0/B GPU0/C GPU0/D GPU0/E GPU0/F GPU0/G GPU0/H GPU1/A GPU1/B GPU1/C GPU1/D GPU1/E GPU1/F GPU1/G GPU1/H GPU2/A GPU2/B GPU2/C GPU2/D GPU2/E GPU2/F GPU2/G GPU2/H GPU3/A GPU3/B GPU3/C GPU3/D GPU3/E GPU3/F GPU3/G GPU3/H GPU4/A GPU4/B GPU4/C GPU4/D GPU4/E GPU4/F GPU4/G GPU4/H GPU5/A GPU5/B GPU5/C GPU5/D GPU5/E GPU5/F GPU5/G GPU5/H GPU6/A GPU6/B GPU6/C GPU6/D GPU6/E GPU6/F GPU6/G GPU6/H GPU7/A GPU7/B GPU7/C GPU7/D GPU7/E GPU7/F GPU7/G GPU7/H)
declare -a CAND_DESC=(B0_CEN51_R04_refresh protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B4_MSSL_MetaProtoDG base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B2_MSSL_TeacherFree base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B3_MSSL_ProtoGate base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B1_MSSL_Udom_split_audit protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B5_MSSL_SatGate base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B0_CEN51_R04_refresh protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B3_MSSL_ProtoGate protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic)
declare -a CAND_LANE=(phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a META_ABLATION=(B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B4_MSSL_MetaProtoDG not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B2_MSSL_TeacherFree not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B1_MSSL_Udom_split_audit not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B5_MSSL_SatGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable)
declare -a META_SEED=(43600 43602 43604 43606 43609 43611 43607 43609 43620 43618 43620 43622 43625 43627 43623 43625 43639 43634 43636 43638 43641 43643 43639 43641 43648 43650 43652 43654 43657 43659 43655 43657 43668 43666 43668 43670 43673 43675 43671 43673 43687 43682 43684 43686 43689 43691 43687 43689 43696 43698 43700 43702 43705 43707 43709 43704 43716 43714 43716 43718 43721 43723 43725 43720)
declare -a META_MAX_SAMPLES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
declare -a SFE_CHANNEL=(- satellite clean satellite satellite satellite satellite clean - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite clean satellite satellite satellite satellite satellite)
declare -a SFE_PROFILE=(- source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - score_diag strict_openmax source_open_set score_diag balanced_openmax score_diag strict_openmax - score_diag strict_openmax score_diag source_open_set balanced_openmax score_diag strict_openmax - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - source_open_set score_diag balanced_openmax strict_openmax score_diag balanced_openmax source_open_set - source_open_set source_open_set score_diag source_open_set balanced_openmax score_diag source_open_set)
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 0 50 20 100 50 50 50 20 0 50 20 100 50 50 50 20 0 20 50 0 50 50 20 50 0 50 50 100 0 50 50 50 0 50 20 100 50 50 50 20 0 0 50 50 50 100 100 0 0 0 0 20 0 50 100 0)
declare -a SFE_SEED=(37600 37602 37604 37606 37609 37611 37607 37609 37620 37618 37620 37622 37625 37627 37623 37625 37639 37634 37636 37638 37641 37643 37639 37641 37648 37650 37652 37654 37657 37659 37655 37657 37668 37666 37668 37670 37673 37675 37671 37673 37687 37682 37684 37686 37689 37691 37687 37689 37696 37698 37700 37702 37705 37707 37709 37704 37716 37714 37716 37718 37721 37723 37725 37720)
declare -a SFE_SAT_SEED=(38600 38602 38604 38606 38609 38611 38607 38609 38620 38618 38620 38622 38625 38627 38623 38625 38639 38634 38636 38638 38641 38643 38639 38641 38648 38650 38652 38654 38657 38659 38655 38657 38668 38666 38668 38670 38673 38675 38671 38673 38687 38682 38684 38686 38689 38691 38687 38689 38696 38698 38700 38702 38705 38707 38709 38704 38716 38714 38716 38718 38721 38723 38725 38720)
declare -a SFE_MAX_SAMPLES=(0 180 180 180 180 180 180 180 0 180 160 180 160 180 180 160 0 180 160 180 160 180 180 160 0 180 160 180 160 180 180 160 0 180 160 180 160 180 180 160 0 180 160 180 160 180 180 160 0 180 160 180 160 180 220 180 0 180 160 180 160 180 220 180)
declare -a SFE_SOURCE_PROTO=(0 20 20 20 20 20 20 20 0 30 40 50 40 30 30 40 0 30 40 50 40 30 30 40 0 30 30 30 30 30 30 30 0 30 30 30 30 30 30 30 0 30 40 50 40 30 30 40 0 40 40 40 40 40 40 40 0 40 40 40 40 40 40 40)
declare -a SFE_QUERY=(0 30 30 30 30 30 30 30 0 35 40 30 35 35 35 40 0 35 40 30 35 35 35 40 0 35 35 35 35 35 35 35 0 35 35 35 35 35 35 35 0 35 40 30 35 35 35 40 0 35 35 35 35 35 35 35 0 35 35 35 35 35 35 35)
declare -a SFE_SCENARIOS=(- clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clean_control low_elev_leo rain_leo,storm_mp mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clean_control - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit - rain_leo,storm_mp rain_leo,storm_mp low_elev_leo storm_mp mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit rain_leo,storm_mp - clear_leo,low_elev_leo clean_control clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit mixed_orbit rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit clear_leo,low_elev_leo)
declare -a FTRC_ADAPTER=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_K=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_LR=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_ANCHOR=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_ALPHA=(0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - - 0 - - - - - - -)
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
declare -a FTRC_SEED=(11200 11201 11202 11203 11204 11205 11206 11207 11208 11209 11210 11211 11212 11213 11214 11215 11216 11217 11218 11219 11220 11221 11222 11223 11224 11225 11226 11227 11228 11229 11230 11231 11232 11233 11234 11235 11236 11237 11238 11239 11240 11241 11242 11243 11244 11245 11246 11247 11248 11249 11250 11251 11252 11253 11254 11255 11256 11257 11258 11259 11260 11261 11262 11263)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

PHASE1_LANE_ACTIVE="${PHASE1_LANE_ACTIVE:-1}"
if [[ "${PHASE1_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase1" ]]; then
      CAND_STATUS[$i]="deferred_phase1_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase1" "reason=phase1_monitor_state_0_active_lane;exact_retry=RUN_ID=${RUN_ID}_phase1_retry PHASE1_LANE_ACTIVE=0 PHASE2_LANE_ACTIVE=1 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260618_101338_next64ba.sh"
    fi
  done
fi


PHASE2_LANE_ACTIVE="${PHASE2_LANE_ACTIVE:-0}"
if [[ "${PHASE2_LANE_ACTIVE}" == "1" ]]; then
  for i in "${!CAND_ID[@]}"; do
    if [[ "${CAND_LANE[$i]:-}" == "phase2" ]]; then
      CAND_STATUS[$i]="deferred_phase2_active"
      event_row "${CAND_ID[$i]}" "DEFERRED_RETRY_CAPACITY" "gpu=${CAND_GPU[$i]}" "lane=phase2" "reason=phase2_monitor_state_0_or_phase2_already_complete;exact_retry=RUN_ID=${RUN_ID}_phase2_retry PHASE1_LANE_ACTIVE=1 PHASE2_LANE_ACTIVE=0 bash ${ROOT}/code/scripts/launch_stage2_optimizer_20260618_101338_next64ba.sh"
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
log_msg "[S2-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} desc=${CAND_DESC[$i]}"
done

source "${ROOT}/tools/stage2_queue_runner_template.sh"
stage2_run_queue_scheduler
