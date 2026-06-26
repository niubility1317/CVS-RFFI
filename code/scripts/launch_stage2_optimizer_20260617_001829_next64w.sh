#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64w_20260617_001829}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-6,7}"
TARGET_LOADER="${TARGET_LOADER:-test_unseen_day_unseen_rx}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-4}"
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

declare -a CAND_ID=("S2N38_GPU0_A_STAGE2A_LOW_FAR_SOURCE_CONTROL_A" "S2N38_GPU0_B_STAGE2A_LOW_FAR_SOURCE_CONTROL_B" "S2N38_GPU0_C_STAGE2A_LOW_FAR_SOURCE_CONTROL_C" "S2N38_GPU0_D_STAGE2A_LOW_FAR_SOURCE_CONTROL_D" "S2N38_GPU0_E_STAGE2A_LOW_FAR_SOURCE_CONTROL_E" "S2N38_GPU0_F_STAGE2A_LOW_FAR_SOURCE_CONTROL_F" "S2N38_GPU0_G_STAGE2A_LOW_FAR_SOURCE_CONTROL_G" "S2N38_GPU0_H_STAGE2A_LOW_FAR_SOURCE_CONTROL_H" "S2N38_GPU1_A_SRF_MP_LOWFAR_RHO002_TOP2_A" "S2N38_GPU1_B_SRF_MP_SUPPORTQ_RHO005_TOP2_B" "S2N38_GPU1_C_SRF_MP_TRIMLSE_BOUND_C" "S2N38_GPU1_D_SRF_MP_SCENARIO_RELIABILITY_D" "S2N38_GPU1_E_SRF_MP_DENSITY_FILTER_E" "S2N38_GPU1_F_SRF_MP_RHO002_SCORE_AUDIT_F" "S2N38_GPU1_G_SRF_MP_PROTOCOUNT_K_SWEEP_G" "S2N38_GPU1_H_SRF_MP_DEPLOY_FAR_GATE_H" "S2N38_GPU2_A_SRF_MP_LOWFAR_RHO002_TOP2_A" "S2N38_GPU2_B_SRF_MP_SUPPORTQ_RHO005_TOP2_B" "S2N38_GPU2_C_SRF_MP_TRIMLSE_BOUND_C" "S2N38_GPU2_D_SRF_MP_SCENARIO_RELIABILITY_D" "S2N38_GPU2_E_SRF_MP_DENSITY_FILTER_E" "S2N38_GPU2_F_SRF_MP_RHO002_SCORE_AUDIT_F" "S2N38_GPU2_G_SRF_MP_PROTOCOUNT_K_SWEEP_G" "S2N38_GPU2_H_SRF_MP_DEPLOY_FAR_GATE_H" "S2N38_GPU5_A_SRF_MP_LOWFAR_RHO002_TOP2_A" "S2N38_GPU5_B_SRF_MP_SUPPORTQ_RHO005_TOP2_B" "S2N38_GPU5_C_SRF_MP_TRIMLSE_BOUND_C" "S2N38_GPU5_D_SRF_MP_SCENARIO_RELIABILITY_D" "S2N38_GPU5_E_SRF_MP_DENSITY_FILTER_E" "S2N38_GPU5_F_SRF_MP_RHO002_SCORE_AUDIT_F" "S2N38_GPU5_G_SRF_MP_PROTOCOUNT_K_SWEEP_G" "S2N38_GPU5_H_SRF_MP_DEPLOY_FAR_GATE_H" "S2N38_GPU3_A_MULTICLASS_BASE_ANCHORED_CONTROL_A" "S2N38_GPU3_B_MULTICLASS_LOWMARGIN_PROTO_BOUND_B" "S2N38_GPU3_C_MULTICLASS_SAFE_RESIDUAL_DIAG_C" "S2N38_GPU3_D_MULTICLASS_SCORETABLE_ROLLBACK_AUDIT_D" "S2N38_GPU3_E_SGC_PROTOBANK_DENSITY_GATE_E" "S2N38_GPU3_F_SGC_PROTOBANK_SUPPORT_DENOISE_F" "S2N38_GPU3_G_SGC_SAT_EVIDENCE_ENCODER_SCORE_G" "S2N38_GPU3_H_SGC_PROTOBANK_STATE_PROXY_H" "S2N38_GPU4_A_SGC_STAGE2B_FEATURE_MOVEMENT_PROBE_A" "S2N38_GPU4_B_SGC_STAGE2B_LOGIT_CALIB_MOVEMENT_B" "S2N38_GPU4_C_SGC_STAGE2B_DELTA_LOGIT_GUARD_C" "S2N38_GPU4_D_SGC_STAGE2B_DELTA_Z_BOUND_D" "S2N38_GPU4_E_SGC_STAGE2B_PROTOTYPE_GUARD_E" "S2N38_GPU4_F_SGC_STAGE2B_SAT_ENCODER_PROBE_F" "S2N38_GPU4_G_SGC_STAGE2B_OLD_FLOOR_GUARD_G" "S2N38_GPU4_H_SGC_STAGE2B_BASE_ANCHOR_CHECK_H" "S2N38_GPU6_A_STAGE2A_HARD_SCENARIO_CONTROL_A" "S2N38_GPU6_B_STAGE2A_HARD_SCENARIO_CONTROL_B" "S2N38_GPU6_C_SGC_HARD_RAIN_SCORE_DIAG_C" "S2N38_GPU6_D_SGC_LOWELEV_DELTA_LOGIT_D" "S2N38_GPU6_E_SGC_STORM_FEATURE_GUARD_E" "S2N38_GPU6_F_SGC_MIXED_ORBIT_SMALLSTEP_F" "S2N38_GPU6_G_SGC_OLD_RETENTION_CHECK_G" "S2N38_GPU6_H_SGC_STATE_SIZE_PROXY_H" "S2N38_GPU7_A_SGC_DEPLOY_STATE_SIZE_A" "S2N38_GPU7_B_SGC_DEPLOY_RUNTIME_PROXY_B" "S2N38_GPU7_C_TELEMETRY_FTRC_TIMING_C" "S2N38_GPU7_D_TELEMETRY_FORGETTING_GUARD_D" "S2N38_GPU7_E_TELEMETRY_SCORETABLE_LOW_FAR_E" "S2N38_GPU7_F_TELEMETRY_MANIFEST_PROTOCOL_F" "S2N38_GPU7_G_STAGE2A_TAXONOMY_CONTROL_G" "S2N38_GPU7_H_STAGE2A_TAXONOMY_CONTROL_H")
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 5 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=("sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe")
declare -a CAND_SLOT=("GPU0/A" "GPU0/B" "GPU0/C" "GPU0/D" "GPU0/E" "GPU0/F" "GPU0/G" "GPU0/H" "GPU1/A" "GPU1/B" "GPU1/C" "GPU1/D" "GPU1/E" "GPU1/F" "GPU1/G" "GPU1/H" "GPU2/A" "GPU2/B" "GPU2/C" "GPU2/D" "GPU2/E" "GPU2/F" "GPU2/G" "GPU2/H" "GPU5/A" "GPU5/B" "GPU5/C" "GPU5/D" "GPU5/E" "GPU5/F" "GPU5/G" "GPU5/H" "GPU3/A" "GPU3/B" "GPU3/C" "GPU3/D" "GPU3/E" "GPU3/F" "GPU3/G" "GPU3/H" "GPU4/A" "GPU4/B" "GPU4/C" "GPU4/D" "GPU4/E" "GPU4/F" "GPU4/G" "GPU4/H" "GPU6/A" "GPU6/B" "GPU6/C" "GPU6/D" "GPU6/E" "GPU6/F" "GPU6/G" "GPU6/H" "GPU7/A" "GPU7/B" "GPU7/C" "GPU7/D" "GPU7/E" "GPU7/F" "GPU7/G" "GPU7/H")
declare -a CAND_DESC=("stage2a_low_far_source_control_a" "stage2a_low_far_source_control_b" "stage2a_low_far_source_control_c" "stage2a_low_far_source_control_d" "stage2a_low_far_source_control_e" "stage2a_low_far_source_control_f" "stage2a_low_far_source_control_g" "stage2a_low_far_source_control_h" "srf_mp_lowfar_rho002_top2_a" "srf_mp_supportq_rho005_top2_b" "srf_mp_trimlse_bound_c" "srf_mp_scenario_reliability_d" "srf_mp_density_filter_e" "srf_mp_rho002_score_audit_f" "srf_mp_protocount_k_sweep_g" "srf_mp_deploy_far_gate_h" "srf_mp_lowfar_rho002_top2_a" "srf_mp_supportq_rho005_top2_b" "srf_mp_trimlse_bound_c" "srf_mp_scenario_reliability_d" "srf_mp_density_filter_e" "srf_mp_rho002_score_audit_f" "srf_mp_protocount_k_sweep_g" "srf_mp_deploy_far_gate_h" "srf_mp_lowfar_rho002_top2_a" "srf_mp_supportq_rho005_top2_b" "srf_mp_trimlse_bound_c" "srf_mp_scenario_reliability_d" "srf_mp_density_filter_e" "srf_mp_rho002_score_audit_f" "srf_mp_protocount_k_sweep_g" "srf_mp_deploy_far_gate_h" "multiclass_base_anchored_control_a" "multiclass_lowmargin_proto_bound_b" "multiclass_safe_residual_diag_c" "multiclass_scoretable_rollback_audit_d" "sgc_protobank_density_gate_e" "sgc_protobank_support_denoise_f" "sgc_sat_evidence_encoder_score_g" "sgc_protobank_state_proxy_h" "sgc_stage2b_feature_movement_probe_a" "sgc_stage2b_logit_calib_movement_b" "sgc_stage2b_delta_logit_guard_c" "sgc_stage2b_delta_z_bound_d" "sgc_stage2b_prototype_guard_e" "sgc_stage2b_sat_encoder_probe_f" "sgc_stage2b_old_floor_guard_g" "sgc_stage2b_base_anchor_check_h" "stage2a_hard_scenario_control_a" "stage2a_hard_scenario_control_b" "sgc_hard_rain_score_diag_c" "sgc_lowelev_delta_logit_d" "sgc_storm_feature_guard_e" "sgc_mixed_orbit_smallstep_f" "sgc_old_retention_check_g" "sgc_state_size_proxy_h" "sgc_deploy_state_size_a" "sgc_deploy_runtime_proxy_b" "telemetry_ftrc_timing_c" "telemetry_forgetting_guard_d" "telemetry_scoretable_low_far_e" "telemetry_manifest_protocol_f" "stage2a_taxonomy_control_g" "stage2a_taxonomy_control_h")
declare -a SFE_CHANNEL=("satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "clean" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "clean")
declare -a SFE_PROFILE=("source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "source_open_set" "score_diag" "strict_openmax" "source_open_set" "score_diag" "source_open_set" "score_diag" "balanced_openmax" "strict_openmax" "source_open_set" "score_diag" "balanced_openmax" "strict_openmax" "source_open_set" "source_open_set" "source_open_set" "score_diag" "source_open_set" "score_diag" "balanced_openmax" "strict_openmax" "source_open_set" "score_diag" "balanced_openmax" "strict_openmax" "source_open_set" "score_diag" "source_open_set" "source_open_set" "source_open_set")
declare -a SFE_SHOTS=("0" "0" "0" "0" "0" "0" "0" "0" "20" "50" "20" "100" "50" "20" "100" "50" "20" "50" "20" "100" "50" "20" "100" "50" "20" "50" "20" "100" "50" "20" "100" "50" "20" "50" "20" "100" "50" "50" "50" "50" "50" "50" "50" "50" "50" "50" "50" "50" "0" "0" "50" "50" "50" "50" "50" "50" "50" "50" "20" "20" "20" "20" "0" "0")
declare -a SFE_SEED=("11500" "11501" "11502" "11503" "11504" "11505" "11506" "11507" "11508" "11509" "11510" "11511" "11512" "11513" "11514" "11515" "11516" "11517" "11518" "11519" "11520" "11521" "11522" "11523" "11524" "11525" "11526" "11527" "11528" "11529" "11530" "11531" "11532" "11533" "11534" "11535" "11536" "11537" "11538" "11539" "11540" "11541" "11542" "11543" "11544" "11545" "11546" "11547" "11548" "11549" "11550" "11551" "11552" "11553" "11554" "11555" "11556" "11557" "11558" "11559" "11560" "11561" "11562" "11563")
declare -a SFE_SAT_SEED=("12500" "12501" "12502" "12503" "12504" "12505" "12506" "12507" "12508" "12509" "12510" "12511" "12512" "12513" "12514" "12515" "12516" "12517" "12518" "12519" "12520" "12521" "12522" "12523" "12524" "12525" "12526" "12527" "12528" "12529" "12530" "12531" "12532" "12533" "12534" "12535" "12536" "12537" "12538" "12539" "12540" "12541" "12542" "12543" "12544" "12545" "12546" "12547" "12548" "12549" "12550" "12551" "12552" "12553" "12554" "12555" "12556" "12557" "12558" "12559" "12560" "12561" "12562" "12563")
declare -a SFE_MAX_SAMPLES=("160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220" "160" "180" "200" "220")
declare -a SFE_SOURCE_PROTO=("20" "20" "10" "30" "40" "10" "30" "40" "20" "30" "40" "30" "50" "20" "40" "30" "20" "30" "40" "30" "50" "20" "40" "30" "20" "30" "40" "30" "50" "20" "40" "30" "20" "30" "40" "30" "30" "30" "30" "30" "30" "40" "20" "30" "40" "20" "30" "40" "20" "20" "30" "20" "30" "40" "20" "30" "40" "20" "30" "40" "20" "20" "20" "20")
declare -a SFE_QUERY=("30" "30" "25" "35" "40" "25" "35" "30" "30" "35" "40" "30" "35" "40" "30" "35" "30" "35" "40" "30" "35" "40" "30" "35" "30" "35" "40" "30" "35" "40" "30" "35" "30" "30" "30" "30" "30" "30" "30" "30" "30" "35" "30" "35" "30" "35" "30" "35" "30" "30" "30" "35" "30" "35" "30" "35" "30" "35" "30" "35" "30" "30" "30" "30")
declare -a SFE_SCENARIOS=("clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clean_control" "clear_leo,low_elev_leo" "clean_control" "rain_leo,storm_mp,mixed_orbit" "clean_control" "low_elev_leo,rain_leo" "clean_control" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "rain_leo,storm_mp" "clean_control" "rain_leo,storm_mp,mixed_orbit" "rain_leo,storm_mp,mixed_orbit,low_elev_leo" "rain_leo,storm_mp,mixed_orbit,low_elev_leo" "rain_leo,storm_mp,mixed_orbit,low_elev_leo" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clean_control")
declare -a FTRC_ADAPTER=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "feature_residual" "logit_calibration" "logit_calibration" "feature_residual" "feature_residual" "feature_residual" "logit_calibration" "logit_calibration" "-" "-" "-" "logit_calibration" "feature_residual" "feature_residual" "logit_calibration" "logit_calibration" "feature_residual" "logit_calibration" "feature_residual" "feature_residual" "-" "-" "-" "-")
declare -a FTRC_K=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 20 50 10 50 20 100 20 0 0 0 20 50 20 20 20 20 20 10 20 0 0 0 0)
declare -a FTRC_LR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "2e-5" "3e-5" "2e-5" "8e-6" "2e-5" "8e-6" "2e-5" "2e-5" "-" "-" "-" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "8e-6" "8e-6" "-" "-" "-" "-")
declare -a FTRC_ANCHOR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "0.30" "0.20" "0.35" "0.60" "0.35" "0.55" "0.30" "0.40" "-" "-" "-" "0.35" "0.35" "0.35" "0.35" "0.35" "0.35" "0.35" "0.55" "0.55" "-" "-" "-" "-")
declare -a FTRC_ALPHA=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "0.35" "1.0" "1.0" "0.20" "0.30" "0.25" "1.0" "1.0" "-" "-" "-" "1.0" "0.30" "0.30" "1.0" "1.0" "0.30" "1.0" "0.20" "0.20" "-" "-" "-" "-")
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 0 0 0 4 4 4 4 4 4 4 4 4 0 0 0 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 6 6 8 6 8 3 0 0 0 6 6 6 3 3 3 3 3 3 0 0 0 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 14 14 10 8 12 8 12 5 0 0 0 8 8 8 5 5 5 5 5 5 0 0 0 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0)
declare -a FTRC_SEED=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 11540 11541 11542 11543 11544 11545 11546 11547 0 0 0 11551 11552 11553 11554 11555 11556 11557 11558 11559 0 0 0 0)
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
  if [[ "${kind}" == "sfe" ]]; then
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_sfe_bundle "${cid}" "${SFE_CHANNEL[$i]}" "${SFE_SHOTS[$i]}" "${SFE_SEED[$i]}" "${SFE_SAT_SEED[$i]}" "${SFE_MAX_SAMPLES[$i]}" "${SFE_SOURCE_PROTO[$i]}" "${SFE_QUERY[$i]}" "${SFE_SCENARIOS[$i]}" "${SFE_PROFILE[$i]}" > "${log_path}" 2>&1) &
  else
    (export CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"; run_ftrc_candidate "${cid}" "${FTRC_ADAPTER[$i]}" "${FTRC_K[$i]}" "${FTRC_LR[$i]}" "${FTRC_ANCHOR[$i]}" "${FTRC_ALPHA[$i]}" "${FTRC_RANK[$i]}" "${FTRC_EPOCHS[$i]}" "${FTRC_STEPS[$i]}" "${FTRC_EVAL_DETAIL[$i]}" "${FTRC_SEED[$i]}" > "${log_path}" 2>&1) &
  fi
  local pid="$!"
  CAND_PID[$i]="${pid}"
  CAND_STATUS[$i]="running"
  event_row "${cid}" "LAUNCHED" "gpu=${gpu}" "pid=${pid}" "log=${log_path}"
}

log_msg "[S2-SCHEDULER] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} max_active_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU}"
log_msg "[S2-SPLIT] source=${SOURCE_TX_IDS} new=${NEW_TX_IDS} unknown=${UNKNOWN_TX_IDS}"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} desc=${CAND_DESC[$i]}"
done

source "${ROOT}/tools/stage2_queue_runner_template.sh"
stage2_run_queue_scheduler
