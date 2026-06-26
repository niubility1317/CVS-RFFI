#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64ao_20260617_201528}"
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

declare -a CAND_ID=(S2N56_GPU0_A_MSSL_B0_CEN51_R04_refresh_A S2N56_GPU0_B_STAGE2A_SAT_LOWFAR_CONTROL_B_B S2N56_GPU0_C_STAGE2A_CLEAN_SOURCE_CONTROL_C_C S2N56_GPU0_D_STAGE2A_LOW_ELEV_FLOOR_D_D S2N56_GPU0_E_STAGE2A_RAIN_STORM_FLOOR_F_F S2N56_GPU0_F_STAGE2A_MIXED_ORBIT_SCORE_G_G S2N56_GPU0_G_STAGE2A_SAT_LOWFAR_CONTROL_B_B_CONFIRMG S2N56_GPU0_H_STAGE2A_CLEAN_SOURCE_CONTROL_C_C_CONFIRMH S2N56_GPU1_A_MSSL_B4_MSSL_MetaProtoDG_E S2N56_GPU1_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N56_GPU1_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N56_GPU1_D_SRF_MP_PROTO_COUNT_K100_D S2N56_GPU1_E_SRF_MP_STRICT_MARGIN_K50_F S2N56_GPU1_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N56_GPU1_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N56_GPU1_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N56_GPU2_A_MSSL_B2_MSSL_TeacherFree_H S2N56_GPU2_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N56_GPU2_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N56_GPU2_D_SRF_MP_PROTO_COUNT_K100_D S2N56_GPU2_E_SRF_MP_STRICT_MARGIN_K50_F S2N56_GPU2_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N56_GPU2_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N56_GPU2_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N56_GPU3_A_MSSL_B3_MSSL_ProtoGate_A S2N56_GPU3_B_MULTICLASS_BASE_ANCHOR_CONTROL_B S2N56_GPU3_C_MULTICLASS_LOWMARGIN_BOUND_C S2N56_GPU3_D_MULTICLASS_SCORETABLE_ROLLBACK_AUDIT_D S2N56_GPU3_E_SGC_PROTOBANK_DENSITY_GATE_F S2N56_GPU3_F_SGC_SAT_EVIDENCE_ENCODER_SCORE_G S2N56_GPU3_G_MULTICLASS_BASE_ANCHOR_CONTROL_B_CONFIRMG S2N56_GPU3_H_MULTICLASS_LOWMARGIN_BOUND_C_CONFIRMH S2N56_GPU4_A_MSSL_B1_MSSL_Udom_split_audit_E S2N56_GPU4_B_SGC_DELTA_Z_PROXY_SCORE_DIAG_B S2N56_GPU4_C_SGC_LOGIT_GAP_PROXY_OPENMAX_C S2N56_GPU4_D_SGC_PROTOBANK_GUARD_K100_D S2N56_GPU4_E_SGC_OLD_FLOOR_SOURCE_OPEN_SET_F S2N56_GPU4_F_SGC_SAT_ENCODER_MIXED_ORBIT_G S2N56_GPU4_G_SGC_DELTA_Z_PROXY_SCORE_DIAG_B_CONFIRMG S2N56_GPU4_H_SGC_LOGIT_GAP_PROXY_OPENMAX_C_CONFIRMH S2N56_GPU5_A_MSSL_B5_MSSL_SatGate_H S2N56_GPU5_B_SRF_MP_LOWFAR_OPENMAX_Q097_B S2N56_GPU5_C_SRF_MP_SUPPORT_QUALITY_K20_C S2N56_GPU5_D_SRF_MP_PROTO_COUNT_K100_D S2N56_GPU5_E_SRF_MP_STRICT_MARGIN_K50_F S2N56_GPU5_F_SRF_MP_SCENARIO_RELIABILITY_K50_G S2N56_GPU5_G_SRF_MP_LOWFAR_OPENMAX_Q097_B_CONFIRMG S2N56_GPU5_H_SRF_MP_SUPPORT_QUALITY_K20_C_CONFIRMH S2N56_GPU6_A_MSSL_B0_CEN51_R04_refresh_A S2N56_GPU6_B_STAGE2A_HARD_RAIN_CONTROL_B S2N56_GPU6_C_SGC_HARD_RAIN_SCORE_DIAG_C S2N56_GPU6_D_SGC_LOWELEV_DELTA_LOGIT_PROXY_D S2N56_GPU6_E_SGC_STORM_FEATURE_GUARD_F S2N56_GPU6_F_SGC_MIXED_ORBIT_STATE_PROXY_G S2N56_GPU6_G_SGC_ALL_SCENARIO_LONG_DIAG_H S2N56_GPU6_H_STAGE2A_HARD_RAIN_CONTROL_B_CONFIRMH S2N56_GPU7_A_MSSL_B3_MSSL_ProtoGate_E S2N56_GPU7_B_STAGE2A_TAXONOMY_SAT_CONTROL_B S2N56_GPU7_C_STAGE2A_TAXONOMY_CLEAN_CONTROL_C S2N56_GPU7_D_TELEMETRY_SCORETABLE_LOW_FAR_D S2N56_GPU7_E_TELEMETRY_MANIFEST_PROTOCOL_F S2N56_GPU7_F_TELEMETRY_RUNTIME_PROXY_G S2N56_GPU7_G_TELEMETRY_STATE_SIZE_LONG_DIAG_H S2N56_GPU7_H_STAGE2A_TAXONOMY_SAT_CONTROL_B_CONFIRMH)
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=(meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe meta_ssl sfe sfe sfe sfe sfe sfe sfe)
declare -a CAND_SLOT=(GPU0/A GPU0/B GPU0/C GPU0/D GPU0/E GPU0/F GPU0/G GPU0/H GPU1/A GPU1/B GPU1/C GPU1/D GPU1/E GPU1/F GPU1/G GPU1/H GPU2/A GPU2/B GPU2/C GPU2/D GPU2/E GPU2/F GPU2/G GPU2/H GPU3/A GPU3/B GPU3/C GPU3/D GPU3/E GPU3/F GPU3/G GPU3/H GPU4/A GPU4/B GPU4/C GPU4/D GPU4/E GPU4/F GPU4/G GPU4/H GPU5/A GPU5/B GPU5/C GPU5/D GPU5/E GPU5/F GPU5/G GPU5/H GPU6/A GPU6/B GPU6/C GPU6/D GPU6/E GPU6/F GPU6/G GPU6/H GPU7/A GPU7/B GPU7/C GPU7/D GPU7/E GPU7/F GPU7/G GPU7/H)
declare -a CAND_DESC=(B0_CEN51_R04_refresh protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B4_MSSL_MetaProtoDG base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B2_MSSL_TeacherFree base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B3_MSSL_ProtoGate base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B1_MSSL_Udom_split_audit protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B5_MSSL_SatGate base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic base_anchored_reliability_gated_prototype_score_table_diagnostic B0_CEN51_R04_refresh protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic B3_MSSL_ProtoGate protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic protocol_safe_SFE_manifest_telemetry_diagnostic)
declare -a CAND_LANE=(phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase1 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a META_ABLATION=(B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B4_MSSL_MetaProtoDG not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B2_MSSL_TeacherFree not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B1_MSSL_Udom_split_audit not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B5_MSSL_SatGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B0_CEN51_R04_refresh not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable B3_MSSL_ProtoGate not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable not_applicable)
declare -a META_SEED=(38000 38002 38004 38006 38009 38011 38007 38009 38020 38018 38020 38022 38025 38027 38023 38025 38039 38034 38036 38038 38041 38043 38039 38041 38048 38050 38052 38054 38057 38059 38055 38057 38068 38066 38068 38070 38073 38075 38071 38073 38087 38082 38084 38086 38089 38091 38087 38089 38096 38098 38100 38102 38105 38107 38109 38104 38116 38114 38116 38118 38121 38123 38125 38120)
declare -a META_MAX_SAMPLES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
declare -a SFE_CHANNEL=(- satellite clean satellite satellite satellite satellite clean - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite satellite satellite satellite satellite satellite satellite - satellite clean satellite satellite satellite satellite satellite)
declare -a SFE_PROFILE=(- source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set source_open_set - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - score_diag strict_openmax source_open_set score_diag balanced_openmax score_diag strict_openmax - score_diag strict_openmax score_diag source_open_set balanced_openmax score_diag strict_openmax - balanced_openmax score_diag score_diag strict_openmax balanced_openmax balanced_openmax score_diag - source_open_set score_diag balanced_openmax strict_openmax score_diag balanced_openmax source_open_set - source_open_set source_open_set score_diag source_open_set balanced_openmax score_diag source_open_set)
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 0 50 20 100 50 50 50 20 0 50 20 100 50 50 50 20 0 20 50 0 50 50 20 50 0 50 50 100 0 50 50 50 0 50 20 100 50 50 50 20 0 0 50 50 50 100 100 0 0 0 0 20 0 50 100 0)
declare -a SFE_SEED=(32000 32002 32004 32006 32009 32011 32007 32009 32020 32018 32020 32022 32025 32027 32023 32025 32039 32034 32036 32038 32041 32043 32039 32041 32048 32050 32052 32054 32057 32059 32055 32057 32068 32066 32068 32070 32073 32075 32071 32073 32087 32082 32084 32086 32089 32091 32087 32089 32096 32098 32100 32102 32105 32107 32109 32104 32116 32114 32116 32118 32121 32123 32125 32120)
declare -a SFE_SAT_SEED=(33000 33002 33004 33006 33009 33011 33007 33009 33020 33018 33020 33022 33025 33027 33023 33025 33039 33034 33036 33038 33041 33043 33039 33041 33048 33050 33052 33054 33057 33059 33055 33057 33068 33066 33068 33070 33073 33075 33071 33073 33087 33082 33084 33086 33089 33091 33087 33089 33096 33098 33100 33102 33105 33107 33109 33104 33116 33114 33116 33118 33121 33123 33125 33120)
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
declare -a FTRC_SEED=(6000 6001 6002 6003 6004 6005 6006 6007 6008 6009 6010 6011 6012 6013 6014 6015 6016 6017 6018 6019 6020 6021 6022 6023 6024 6025 6026 6027 6028 6029 6030 6031 6032 6033 6034 6035 6036 6037 6038 6039 6040 6041 6042 6043 6044 6045 6046 6047 6048 6049 6050 6051 6052 6053 6054 6055 6056 6057 6058 6059 6060 6061 6062 6063)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done

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
  if [[ "${kind}" == "meta_ssl" ]]; then
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
