#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64q_20260616_131357}"
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

declare -a CAND_ID=("S2N32_GPU0_A_STAGE2A_SOURCE_OPEN_SET_SATELLITE_A" "S2N32_GPU0_B_STAGE2A_SOURCE_OPEN_SET_CLEAN_B" "S2N32_GPU0_C_STAGE2A_SOURCE_OPEN_SET_SATELLITE_C" "S2N32_GPU0_D_STAGE2A_SOURCE_OPEN_SET_CLEAN_D" "S2N32_GPU0_E_STAGE2A_SOURCE_OPEN_SET_SATELLITE_E" "S2N32_GPU0_F_STAGE2A_SOURCE_OPEN_SET_CLEAN_F" "S2N32_GPU0_G_STAGE2A_SOURCE_OPEN_SET_SATELLITE_G" "S2N32_GPU0_H_STAGE2A_SOURCE_OPEN_SET_CLEAN_H" "S2N32_GPU1_A_SRF_MP_RHO002_TOP2_A" "S2N32_GPU1_B_SRF_MP_RHO005_TOP2_B" "S2N32_GPU1_C_SRF_MP_RHO002_TRIMLSE_C" "S2N32_GPU1_D_SRF_MP_RHO005_SUPPORTFILTER_D" "S2N32_GPU1_E_SRF_MP_RHO010_STRESS_E" "S2N32_GPU1_F_SRF_MP_RHO002_HARDFAR_F" "S2N32_GPU1_G_SRF_MP_RHO005_SCENARIO_G" "S2N32_GPU1_H_SRF_MP_RHO002_SCORETABLE_H" "S2N32_GPU2_A_SRF_MP_RHO002_TOP2_A" "S2N32_GPU2_B_SRF_MP_RHO005_TOP2_B" "S2N32_GPU2_C_SRF_MP_RHO002_TRIMLSE_C" "S2N32_GPU2_D_SRF_MP_RHO005_SUPPORTFILTER_D" "S2N32_GPU2_E_SRF_MP_RHO010_STRESS_E" "S2N32_GPU2_F_SRF_MP_RHO002_HARDFAR_F" "S2N32_GPU2_G_SRF_MP_RHO005_SCENARIO_G" "S2N32_GPU2_H_SRF_MP_RHO002_SCORETABLE_H" "S2N32_GPU5_A_SRF_MP_RHO002_TOP2_A" "S2N32_GPU5_B_SRF_MP_RHO005_TOP2_B" "S2N32_GPU5_C_SRF_MP_RHO002_TRIMLSE_C" "S2N32_GPU5_D_SRF_MP_RHO005_SUPPORTFILTER_D" "S2N32_GPU5_E_SRF_MP_RHO010_STRESS_E" "S2N32_GPU5_F_SRF_MP_RHO002_HARDFAR_F" "S2N32_GPU5_G_SRF_MP_RHO005_SCENARIO_G" "S2N32_GPU5_H_SRF_MP_RHO002_SCORETABLE_H" "S2N32_GPU3_A_MULTICLASS_HEAD_BASE_ANCHORED_RHO000_CONTROL_A" "S2N32_GPU3_B_MULTICLASS_HEAD_LOWMARGIN_PROTO_INTERVENTION_B" "S2N32_GPU3_C_MULTICLASS_HEAD_SAFE_RESIDUAL_PROTO_DIAG_C" "S2N32_GPU3_D_MULTICLASS_HEAD_MULTICLASS_SCORETABLE_BOUNDARY_D" "S2N32_GPU3_E_SGC_PROTOBANK_GATE_SAFETY_E" "S2N32_GPU3_F_SGC_PROTOBANK_GATE_SAFETY_F" "S2N32_GPU3_G_SGC_PROTOBANK_GATE_SAFETY_G" "S2N32_GPU3_H_SGC_PROTOBANK_GATE_SAFETY_H" "S2N32_GPU4_A_SGC_STAGE2B_IPFA_FEATURE_ADAPTER_A" "S2N32_GPU4_B_SGC_STAGE2B_BLRC_LOGIT_CALIBRATOR_B" "S2N32_GPU4_C_SGC_STAGE2B_SAT_EVIDENCE_LORA_C" "S2N32_GPU4_D_SGC_STAGE2B_DELTA_LOGIT_SMALLSTEP_D" "S2N32_GPU4_E_SGC_STAGE2B_PROTOBANK_GUARD_FEATURE_E" "S2N32_GPU4_F_SGC_STAGE2B_SGC_OLD_FLOOR_LORA_F" "S2N32_GPU4_G_SGC_STAGE2B_DELTA_Z_BOUND_FEATURE_G" "S2N32_GPU4_H_SGC_STAGE2B_BASE_ANCHORED_CALIB_H" "S2N32_GPU6_A_STAGE2A_HARD_SCENARIO_SATELLITE_A" "S2N32_GPU6_B_STAGE2A_HARD_SCENARIO_CLEAN_B" "S2N32_GPU6_C_SGC_HARD_SCENARIO_HARD_RAIN_GATE_C" "S2N32_GPU6_D_SGC_HARD_SCENARIO_LOWELEV_DELTA_LOGIT_D" "S2N32_GPU6_E_SGC_HARD_SCENARIO_STORM_LORA_BOUND_E" "S2N32_GPU6_F_SGC_HARD_SCENARIO_MIXED_ORBIT_SMALLSTEP_F" "S2N32_GPU6_G_SGC_HARD_SCENARIO_OLD_RETENTION_CALIB_G" "S2N32_GPU6_H_SGC_HARD_SCENARIO_SGC_STATE_SIZE_PROXY_H" "S2N32_GPU7_A_SGC_DEPLOY_STATE_A" "S2N32_GPU7_B_SGC_DEPLOY_STATE_B" "S2N32_GPU7_C_TELEMETRY_FTRC_C" "S2N32_GPU7_D_TELEMETRY_FTRC_D" "S2N32_GPU7_E_TELEMETRY_SCORETABLE_E" "S2N32_GPU7_F_TELEMETRY_SCORETABLE_F" "S2N32_GPU7_G_STAGE2A_TAXONOMY_SATELLITE_G" "S2N32_GPU7_H_STAGE2A_TAXONOMY_CLEAN_H")
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 5 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=("sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "sfe" "sfe" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "sfe" "sfe" "sfe" "sfe")
declare -a CAND_SLOT=("GPU0/A" "GPU0/B" "GPU0/C" "GPU0/D" "GPU0/E" "GPU0/F" "GPU0/G" "GPU0/H" "GPU1/A" "GPU1/B" "GPU1/C" "GPU1/D" "GPU1/E" "GPU1/F" "GPU1/G" "GPU1/H" "GPU2/A" "GPU2/B" "GPU2/C" "GPU2/D" "GPU2/E" "GPU2/F" "GPU2/G" "GPU2/H" "GPU5/A" "GPU5/B" "GPU5/C" "GPU5/D" "GPU5/E" "GPU5/F" "GPU5/G" "GPU5/H" "GPU3/A" "GPU3/B" "GPU3/C" "GPU3/D" "GPU3/E" "GPU3/F" "GPU3/G" "GPU3/H" "GPU4/A" "GPU4/B" "GPU4/C" "GPU4/D" "GPU4/E" "GPU4/F" "GPU4/G" "GPU4/H" "GPU6/A" "GPU6/B" "GPU6/C" "GPU6/D" "GPU6/E" "GPU6/F" "GPU6/G" "GPU6/H" "GPU7/A" "GPU7/B" "GPU7/C" "GPU7/D" "GPU7/E" "GPU7/F" "GPU7/G" "GPU7/H")
declare -a CAND_DESC=("stage2a-source-open-set-satellite-a" "stage2a-source-open-set-clean-b" "stage2a-source-open-set-satellite-c" "stage2a-source-open-set-clean-d" "stage2a-source-open-set-satellite-e" "stage2a-source-open-set-clean-f" "stage2a-source-open-set-satellite-g" "stage2a-source-open-set-clean-h" "srf-mp-rho002-top2-a" "srf-mp-rho005-top2-b" "srf-mp-rho002-trimlse-c" "srf-mp-rho005-supportfilter-d" "srf-mp-rho010-stress-e" "srf-mp-rho002-hardfar-f" "srf-mp-rho005-scenario-g" "srf-mp-rho002-scoretable-h" "srf-mp-rho002-top2-a" "srf-mp-rho005-top2-b" "srf-mp-rho002-trimlse-c" "srf-mp-rho005-supportfilter-d" "srf-mp-rho010-stress-e" "srf-mp-rho002-hardfar-f" "srf-mp-rho005-scenario-g" "srf-mp-rho002-scoretable-h" "srf-mp-rho002-top2-a" "srf-mp-rho005-top2-b" "srf-mp-rho002-trimlse-c" "srf-mp-rho005-supportfilter-d" "srf-mp-rho010-stress-e" "srf-mp-rho002-hardfar-f" "srf-mp-rho005-scenario-g" "srf-mp-rho002-scoretable-h" "multiclass-head-base-anchored-rho000-control-a" "multiclass-head-lowmargin-proto-intervention-b" "multiclass-head-safe-residual-proto-diag-c" "multiclass-head-multiclass-scoretable-boundary-d" "sgc-protobank-gate-safety-e" "sgc-protobank-gate-safety-f" "sgc-protobank-gate-safety-g" "sgc-protobank-gate-safety-h" "sgc-stage2b-ipfa-feature-adapter-a" "sgc-stage2b-blrc-logit-calibrator-b" "sgc-stage2b-sat-evidence-lora-c" "sgc-stage2b-delta-logit-smallstep-d" "sgc-stage2b-protobank-guard-feature-e" "sgc-stage2b-sgc-old-floor-lora-f" "sgc-stage2b-delta-z-bound-feature-g" "sgc-stage2b-base-anchored-calib-h" "stage2a-hard-scenario-satellite-a" "stage2a-hard-scenario-clean-b" "sgc-hard-scenario-hard-rain-gate-c" "sgc-hard-scenario-lowelev-delta-logit-d" "sgc-hard-scenario-storm-lora-bound-e" "sgc-hard-scenario-mixed-orbit-smallstep-f" "sgc-hard-scenario-old-retention-calib-g" "sgc-hard-scenario-sgc-state-size-proxy-h" "sgc-deploy-state-a" "sgc-deploy-state-b" "telemetry-ftrc-c" "telemetry-ftrc-d" "telemetry-scoretable-e" "telemetry-scoretable-f" "stage2a-taxonomy-satellite-g" "stage2a-taxonomy-clean-h")
declare -a SFE_CHANNEL=("satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "-" "-" "-" "-" "-" "-" "-" "-" "satellite" "clean" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "satellite" "satellite" "satellite" "clean")
declare -a SFE_PROFILE=("source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "-" "-" "-" "-" "-" "-" "-" "-" "source_open_set" "source_open_set" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "score_diag" "score_diag" "source_open_set" "source_open_set")
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 20 50 20 100 50 20 100 50 20 50 20 100 50 20 100 50 20 50 20 100 50 20 100 50 20 50 50 100 20 50 100 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 50 0 0)
declare -a SFE_SEED=(8050 8051 8052 8053 8054 8055 8056 8057 8110 8111 8112 8113 8114 8115 8116 8117 8210 8211 8212 8213 8214 8215 8216 8217 8510 8511 8512 8513 8514 8515 8516 8517 8310 8311 8312 8313 8340 8341 8342 8343 0 0 0 0 0 0 0 0 8610 8611 0 0 0 0 0 0 0 0 0 0 8742 8743 8780 8781)
declare -a SFE_SAT_SEED=(9501 9502 9503 9504 9505 9506 9507 9508 9610 9611 9612 9613 9614 9615 9616 9617 9620 9621 9622 9623 9624 9625 9626 9627 9650 9651 9652 9653 9654 9655 9656 9657 9630 9631 9632 9633 9660 9661 9662 9663 0 0 0 0 0 0 0 0 9710 9711 0 0 0 0 0 0 0 0 0 0 9742 9743 9780 9781)
declare -a SFE_MAX_SAMPLES=(180 200 220 180 200 220 180 200 140 170 200 230 140 170 200 230 140 170 200 230 140 170 200 230 140 170 200 230 140 170 200 230 140 160 180 200 160 180 200 220 0 0 0 0 0 0 0 0 180 180 0 0 0 0 0 0 0 0 0 0 160 180 160 160)
declare -a SFE_SOURCE_PROTO=(10 20 30 40 10 20 30 40 20 30 40 40 30 20 40 30 20 30 40 40 30 20 40 30 20 30 40 40 30 20 40 30 20 30 40 30 20 30 40 50 0 0 0 0 0 0 0 0 20 20 0 0 0 0 0 0 0 0 0 0 20 20 20 20)
declare -a SFE_QUERY=(30 35 40 30 35 40 30 35 30 40 30 50 40 30 40 30 30 40 30 50 40 30 40 30 30 40 30 50 40 30 40 30 30 40 30 40 30 35 40 45 0 0 0 0 0 0 0 0 30 30 0 0 0 0 0 0 0 0 0 0 30 30 30 30)
declare -a SFE_SCENARIOS=("clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo" "rain_leo" "storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clean_control" "low_elev_leo,rain_leo" "storm_mp" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "rain_leo" "storm_mp" "mixed_orbit" "low_elev_leo,rain_leo" "-" "-" "-" "-" "-" "-" "-" "-" "low_elev_leo,rain_leo,storm_mp,mixed_orbit" "low_elev_leo,rain_leo,storm_mp,mixed_orbit" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
declare -a FTRC_ADAPTER=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "feature_residual" "logit_calibration" "logit_lora" "logit_calibration" "feature_residual" "logit_lora" "feature_residual" "logit_calibration" "-" "-" "feature_residual" "logit_calibration" "logit_lora" "feature_residual" "logit_calibration" "feature_residual" "logit_calibration" "feature_residual" "logit_calibration" "feature_residual" "-" "-" "-" "-")
declare -a FTRC_K=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 20 50 10 50 100 20 100 0 0 20 50 50 10 20 20 20 50 10 20 0 0 0 0)
declare -a FTRC_LR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "3e-5" "5e-5" "8e-5" "1e-5" "5e-5" "5e-5" "1e-5" "3e-5" "-" "-" "3e-5" "3e-5" "5e-5" "1e-5" "1e-5" "3e-5" "1e-5" "3e-5" "1e-5" "3e-5" "-" "-" "-" "-")
declare -a FTRC_ANCHOR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "0.35" "0.25" "0.20" "0.45" "0.30" "0.25" "0.55" "0.35" "-" "-" "0.35" "0.35" "0.25" "0.55" "0.50" "0.45" "0.45" "0.40" "0.50" "0.45" "-" "-" "-" "-")
declare -a FTRC_ALPHA=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "0.40" "1.0" "0.60" "1.0" "0.50" "0.50" "0.25" "1.0" "-" "-" "0.40" "1.0" "0.50" "0.25" "1.0" "0.30" "1.0" "0.35" "1.0" "0.25" "-" "-" "-" "-")
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 8 4 4 8 4 4 0 0 4 4 8 4 4 4 4 4 4 4 0 0 0 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 10 6 10 10 6 10 0 0 6 8 10 6 8 3 3 3 3 3 0 0 0 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 20 10 20 20 10 20 0 0 10 16 20 10 10 5 5 5 5 5 0 0 0 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0)
declare -a FTRC_SEED=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8410 8411 8412 8413 8414 8415 8416 8417 0 0 8630 8631 8632 8633 8634 8635 8710 8711 8730 8731 0 0 0 0)
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
