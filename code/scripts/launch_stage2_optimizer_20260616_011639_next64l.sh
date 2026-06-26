#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next64l_20260616_011639}"
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

declare -a CAND_ID=("S2N27_GPU0_A_STAGE2A_SATELLITE_SOURCE_OPEN_SET_A" "S2N27_GPU0_B_STAGE2A_CLEAN_SOURCE_OPEN_SET_B" "S2N27_GPU0_C_STAGE2A_SATELLITE_SOURCE_OPEN_SET_C" "S2N27_GPU0_D_STAGE2A_CLEAN_SOURCE_OPEN_SET_D" "S2N27_GPU0_E_STAGE2A_SATELLITE_SOURCE_OPEN_SET_E" "S2N27_GPU0_F_STAGE2A_CLEAN_SOURCE_OPEN_SET_F" "S2N27_GPU0_G_STAGE2A_SATELLITE_SOURCE_OPEN_SET_G" "S2N27_GPU0_H_STAGE2A_CLEAN_SOURCE_OPEN_SET_H" "S2N27_GPU1_A_STAGE2C_GATE_CALIBRATION_A" "S2N27_GPU1_B_STAGE2C_GATE_CALIBRATION_B" "S2N27_GPU1_C_STAGE2C_GATE_CALIBRATION_C" "S2N27_GPU1_D_STAGE2C_GATE_CALIBRATION_D" "S2N27_GPU1_E_STAGE2C_GATE_CALIBRATION_E" "S2N27_GPU1_F_STAGE2C_GATE_CALIBRATION_F" "S2N27_GPU1_G_STAGE2C_GATE_CALIBRATION_G" "S2N27_GPU1_H_STAGE2C_GATE_CALIBRATION_H" "S2N27_GPU2_A_STAGE2C_PROTOTYPE_SUPPORT_A" "S2N27_GPU2_B_STAGE2C_PROTOTYPE_SUPPORT_B" "S2N27_GPU2_C_STAGE2C_PROTOTYPE_SUPPORT_C" "S2N27_GPU2_D_STAGE2C_PROTOTYPE_SUPPORT_D" "S2N27_GPU2_E_STAGE2C_PROTOTYPE_SUPPORT_E" "S2N27_GPU2_F_STAGE2C_PROTOTYPE_SUPPORT_F" "S2N27_GPU2_G_STAGE2C_PROTOTYPE_SUPPORT_G" "S2N27_GPU2_H_STAGE2C_PROTOTYPE_SUPPORT_H" "S2N27_GPU3_A_STAGE2C_SCENARIO_FLOOR_A" "S2N27_GPU3_B_STAGE2C_SCENARIO_FLOOR_B" "S2N27_GPU3_C_STAGE2C_SCENARIO_FLOOR_C" "S2N27_GPU3_D_STAGE2C_SCENARIO_FLOOR_D" "S2N27_GPU3_E_STAGE2C_SCENARIO_FLOOR_E" "S2N27_GPU3_F_STAGE2C_SCENARIO_FLOOR_F" "S2N27_GPU3_G_STAGE2C_SCENARIO_FLOOR_G" "S2N27_GPU3_H_STAGE2C_SCENARIO_FLOOR_H" "S2N27_GPU4_A_STAGE2B_LOGIT_CAL_MOVE_A" "S2N27_GPU4_B_STAGE2B_LOGIT_CAL_MOVE_B" "S2N27_GPU4_C_STAGE2B_LOGIT_CAL_MOVE_C" "S2N27_GPU4_D_STAGE2B_LOGIT_CAL_MOVE_D" "S2N27_GPU4_E_STAGE2B_LOGIT_CAL_MOVE_E" "S2N27_GPU4_F_STAGE2B_LOGIT_CAL_MOVE_F" "S2N27_GPU4_G_STAGE2B_LOGIT_CAL_MOVE_G" "S2N27_GPU4_H_STAGE2B_LOGIT_CAL_MOVE_H" "S2N27_GPU5_A_STAGE2B_FEATURE_RESIDUAL_MOVE_A" "S2N27_GPU5_B_STAGE2B_FEATURE_RESIDUAL_MOVE_B" "S2N27_GPU5_C_STAGE2B_FEATURE_RESIDUAL_MOVE_C" "S2N27_GPU5_D_STAGE2B_FEATURE_RESIDUAL_MOVE_D" "S2N27_GPU5_E_STAGE2B_FEATURE_RESIDUAL_MOVE_E" "S2N27_GPU5_F_STAGE2B_FEATURE_RESIDUAL_MOVE_F" "S2N27_GPU5_G_STAGE2B_FEATURE_RESIDUAL_MOVE_G" "S2N27_GPU5_H_STAGE2B_FEATURE_RESIDUAL_MOVE_H" "S2N27_GPU6_A_STAGE2B_LORA_MOVE_A" "S2N27_GPU6_B_STAGE2B_LORA_MOVE_B" "S2N27_GPU6_C_STAGE2B_LORA_MOVE_C" "S2N27_GPU6_D_STAGE2B_LORA_MOVE_D" "S2N27_GPU6_E_STAGE2B_LORA_MOVE_E" "S2N27_GPU6_F_STAGE2B_LORA_MOVE_F" "S2N27_GPU6_G_STAGE2B_LORA_MOVE_G" "S2N27_GPU6_H_STAGE2B_LORA_MOVE_H" "S2N27_GPU7_A_DEPLOY_TELEMETRY_A" "S2N27_GPU7_B_DEPLOY_TELEMETRY_B" "S2N27_GPU7_C_DEPLOY_TELEMETRY_C" "S2N27_GPU7_D_SCORETABLE_DIAG_D" "S2N27_GPU7_E_SCORETABLE_DIAG_E" "S2N27_GPU7_F_SCORETABLE_DIAG_F" "S2N27_GPU7_G_STAGE2A_TAXONOMY_G" "S2N27_GPU7_H_STAGE2A_TAXONOMY_H")
declare -a CAND_GPU=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7)
declare -a CAND_KIND=("sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "sfe" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "ftrc" "sfe" "sfe" "sfe" "sfe" "sfe")
declare -a CAND_SLOT=("GPU0/A" "GPU0/B" "GPU0/C" "GPU0/D" "GPU0/E" "GPU0/F" "GPU0/G" "GPU0/H" "GPU1/A" "GPU1/B" "GPU1/C" "GPU1/D" "GPU1/E" "GPU1/F" "GPU1/G" "GPU1/H" "GPU2/A" "GPU2/B" "GPU2/C" "GPU2/D" "GPU2/E" "GPU2/F" "GPU2/G" "GPU2/H" "GPU3/A" "GPU3/B" "GPU3/C" "GPU3/D" "GPU3/E" "GPU3/F" "GPU3/G" "GPU3/H" "GPU4/A" "GPU4/B" "GPU4/C" "GPU4/D" "GPU4/E" "GPU4/F" "GPU4/G" "GPU4/H" "GPU5/A" "GPU5/B" "GPU5/C" "GPU5/D" "GPU5/E" "GPU5/F" "GPU5/G" "GPU5/H" "GPU6/A" "GPU6/B" "GPU6/C" "GPU6/D" "GPU6/E" "GPU6/F" "GPU6/G" "GPU6/H" "GPU7/A" "GPU7/B" "GPU7/C" "GPU7/D" "GPU7/E" "GPU7/F" "GPU7/G" "GPU7/H")
declare -a CAND_DESC=("stage2a-satellite-source-open-set-a" "stage2a-clean-source-open-set-b" "stage2a-satellite-source-open-set-c" "stage2a-clean-source-open-set-d" "stage2a-satellite-source-open-set-e" "stage2a-clean-source-open-set-f" "stage2a-satellite-source-open-set-g" "stage2a-clean-source-open-set-h" "stage2c-gate-calibration-a" "stage2c-gate-calibration-b" "stage2c-gate-calibration-c" "stage2c-gate-calibration-d" "stage2c-gate-calibration-e" "stage2c-gate-calibration-f" "stage2c-gate-calibration-g" "stage2c-gate-calibration-h" "stage2c-prototype-support-a" "stage2c-prototype-support-b" "stage2c-prototype-support-c" "stage2c-prototype-support-d" "stage2c-prototype-support-e" "stage2c-prototype-support-f" "stage2c-prototype-support-g" "stage2c-prototype-support-h" "stage2c-scenario-floor-a" "stage2c-scenario-floor-b" "stage2c-scenario-floor-c" "stage2c-scenario-floor-d" "stage2c-scenario-floor-e" "stage2c-scenario-floor-f" "stage2c-scenario-floor-g" "stage2c-scenario-floor-h" "stage2b-logit-cal-move-a" "stage2b-logit-cal-move-b" "stage2b-logit-cal-move-c" "stage2b-logit-cal-move-d" "stage2b-logit-cal-move-e" "stage2b-logit-cal-move-f" "stage2b-logit-cal-move-g" "stage2b-logit-cal-move-h" "stage2b-feature-residual-move-a" "stage2b-feature-residual-move-b" "stage2b-feature-residual-move-c" "stage2b-feature-residual-move-d" "stage2b-feature-residual-move-e" "stage2b-feature-residual-move-f" "stage2b-feature-residual-move-g" "stage2b-feature-residual-move-h" "stage2b-lora-move-a" "stage2b-lora-move-b" "stage2b-lora-move-c" "stage2b-lora-move-d" "stage2b-lora-move-e" "stage2b-lora-move-f" "stage2b-lora-move-g" "stage2b-lora-move-h" "deploy-telemetry-a" "deploy-telemetry-b" "deploy-telemetry-c" "scoretable-diag-d" "scoretable-diag-e" "scoretable-diag-f" "stage2a-taxonomy-g" "stage2a-taxonomy-h")
declare -a SFE_CHANNEL=("satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "clean" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "satellite" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "satellite" "satellite" "satellite" "satellite" "clean")
declare -a SFE_PROFILE=("source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "source_open_set" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "strict_openmax" "score_diag" "balanced_openmax" "score_diag" "strict_openmax" "balanced_openmax" "score_diag" "balanced_openmax" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "score_diag" "score_diag" "score_diag" "source_open_set" "source_open_set")
declare -a SFE_SHOTS=(0 0 0 0 0 0 0 0 20 50 100 50 20 100 50 100 20 50 100 50 20 100 50 100 20 50 100 50 20 100 50 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 50 100 0 0)
declare -a SFE_SEED=(7050 7051 7052 7053 7054 7055 7056 7057 7060 7061 7062 7063 7064 7065 7066 7067 7070 7071 7072 7073 7074 7075 7076 7077 7080 7081 7082 7083 7084 7085 7086 7087 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7233 7234 7235 7246 7247)
declare -a SFE_SAT_SEED=(8501 8502 8503 8504 8505 8506 8507 8508 8610 8611 8612 8613 8614 8615 8616 8617 8620 8621 8622 8623 8624 8625 8626 8627 8630 8631 8632 8633 8634 8635 8636 8637 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8733 8734 8735 8746 8747)
declare -a SFE_MAX_SAMPLES=(220 240 260 220 240 260 220 240 140 180 220 260 140 180 220 260 140 180 220 260 140 180 220 260 140 180 220 260 140 180 220 260 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100 140 180 180 180)
declare -a SFE_SOURCE_PROTO=(10 20 30 40 10 20 30 40 20 30 40 20 30 40 20 30 20 30 40 20 30 40 20 30 20 30 40 20 30 40 20 30 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 20 20 20 20)
declare -a SFE_QUERY=(40 45 50 40 45 50 40 45 30 40 50 30 40 50 30 40 30 40 50 30 40 50 30 40 30 40 50 30 40 50 30 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 30 30 30 30 30)
declare -a SFE_SCENARIOS=("clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "storm_mp,mixed_orbit" "low_elev_leo,rain_leo" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo" "low_elev_leo" "rain_leo" "storm_mp" "mixed_orbit" "low_elev_leo,rain_leo" "storm_mp,mixed_orbit" "clear_leo,storm_mp,mixed_orbit" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit" "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
declare -a FTRC_ADAPTER=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "logit_calibration" "logit_calibration" "logit_calibration" "logit_calibration" "logit_calibration" "logit_calibration" "logit_calibration" "logit_calibration" "feature_residual" "feature_residual" "feature_residual" "feature_residual" "feature_residual" "feature_residual" "feature_residual" "feature_residual" "logit_lora" "logit_lora" "logit_lora" "logit_lora" "logit_lora" "logit_lora" "logit_lora" "logit_lora" "logit_calibration" "feature_residual" "logit_lora" "-" "-" "-" "-" "-")
declare -a FTRC_K=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 10 20 50 100 20 50 100 20 10 20 50 100 20 50 100 20 10 20 50 100 20 50 100 20 20 50 20 0 0 0 0 0)
declare -a FTRC_LR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "3e-5" "5e-5" "1e-4" "1e-4" "3e-5" "5e-5" "8e-5" "1e-5" "3e-5" "5e-5" "1e-4" "1e-4" "3e-5" "5e-5" "8e-5" "1e-5" "3e-5" "5e-5" "1e-4" "1e-4" "3e-5" "5e-5" "8e-5" "1e-5" "1e-5" "3e-5" "1e-5" "-" "-" "-" "-" "-")
declare -a FTRC_ANCHOR=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "0.35" "0.25" "0.15" "0.10" "0.45" "0.30" "0.20" "0.55" "0.35" "0.25" "0.15" "0.10" "0.45" "0.30" "0.20" "0.55" "0.35" "0.25" "0.15" "0.10" "0.45" "0.30" "0.20" "0.55" "0.45" "0.40" "0.50" "-" "-" "-" "-" "-")
declare -a FTRC_ALPHA=("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "0.40" "0.60" "0.80" "1.00" "0.30" "0.50" "0.75" "0.25" "0.40" "0.60" "0.80" "1.00" "0.30" "0.50" "0.75" "0.25" "1.0" "0.35" "0.25" "-" "-" "-" "-" "-")
declare -a FTRC_RANK=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 8 8 4 8 8 4 4 4 4 0 0 0 0 0)
declare -a FTRC_EPOCHS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 10 10 10 10 10 10 6 6 10 10 10 10 10 10 6 6 10 10 10 10 10 10 6 3 3 3 0 0 0 0 0)
declare -a FTRC_STEPS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 10 20 20 20 20 20 20 10 10 20 20 20 20 20 20 10 10 20 20 20 20 20 20 10 5 5 5 0 0 0 0 0)
declare -a FTRC_EVAL_DETAIL=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0)
declare -a FTRC_SEED=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7140 7141 7142 7143 7144 7145 7146 7147 7150 7151 7152 7153 7154 7155 7156 7157 7160 7161 7162 7163 7164 7165 7166 7167 7200 7201 7202 0 0 0 0 0)
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
