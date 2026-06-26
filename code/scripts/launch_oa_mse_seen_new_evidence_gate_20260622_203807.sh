#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-oa_mse_seen_new_evidence_gate_20260622_203807}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
NEW_TX_IDS="${NEW_TX_IDS:-1-16,1-18}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-}"
OA_MSE_UNKNOWN_TX_IDS="${OA_MSE_UNKNOWN_TX_IDS:-10-1,10-10}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-20-1}"
SFE_MAX_SAMPLES_PER_COMBO="${SFE_MAX_SAMPLES_PER_COMBO:-0}"
SFE_MAX_SAMPLES_PER_TX="${SFE_MAX_SAMPLES_PER_TX:-200}"
SFE_EXPORT_BATCH_SIZE="${SFE_EXPORT_BATCH_SIZE:-512}"
SFE_SOURCE_PROTO_PER_TX="${SFE_SOURCE_PROTO_PER_TX:-20}"
SFE_SOURCE_QUERY_PER_TX="${SFE_SOURCE_QUERY_PER_TX:-20}"
SFE_QUERY_PER_TX="${SFE_QUERY_PER_TX:-50}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=3"
PIDS=()
NAMES=()

echo "[SPACEBORNE-FSDA-CANDIDATE] id=OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN protocol=CVS-OA-MSE k=0 target_visibility=source_old_only_with_leo_unknown_query_eval label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --source_rxs \"${CEN51_TRAIN_RXS}\" --new_tx_ids \"${NEW_TX_IDS}\" --new_rxs \"${TARGET_RECEIVER_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --target_new_channel_view satellite --target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --target_old_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1441; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol source_open_set --feature_npz \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN/features.npz\" --output_json \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN/metrics.json\" --manifest_json \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN/manifest.json\" --score_table_csv \"${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN/score_table.csv\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --shots 0 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode oa_mse --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --oa_mse_adapter_rank 2 --oa_mse_adapter_steps 40 --old_acc_target 0.9 --seen_new_acc_target 0.75 --seed 1441")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN"
  ("${CMD[@]}" > "${LOG_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN pid=${pid} gpu=${GPU} log=${LOG_ROOT}/OA_MSE_STAGE2A_MSELITE_SOURCE_OPEN.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD protocol=CVS-OA-MSE k=2 target_visibility=target_old_leo_support_labeled_unknown_eval_only label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --source_rxs \"${CEN51_TRAIN_RXS}\" --target_old_tx_ids \"${TARGET_OLD_TX_IDS}\" --target_old_rxs \"${TARGET_RECEIVER_IDS}\" --target_old_channel_view satellite --new_tx_ids \"${NEW_TX_IDS}\" --new_rxs \"${TARGET_RECEIVER_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --target_new_channel_view satellite --target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --target_old_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1442; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol ftrc --feature_npz \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD/features.npz\" --output_json \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD/metrics.json\" --manifest_json \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD/manifest.json\" --score_table_csv \"${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD/score_table.csv\" --source_tx_ids \"${SOURCE_TX_IDS}\" --target_old_tx_ids \"${TARGET_OLD_TX_IDS}\" --target_old_support_per_tx \"2\" --target_old_query_per_tx \"50\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --shots 0 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode oa_mse --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --oa_mse_adapter_rank 2 --oa_mse_adapter_steps 40 --old_acc_target 0.9 --seen_new_acc_target 0.75 --seed 1442")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD"
  ("${CMD[@]}" > "${LOG_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD pid=${pid} gpu=${GPU} log=${LOG_ROOT}/OA_MSE_STAGE2B_SUBSPACE_TARGET_OLD.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=OA_MSE_STAGE2C_HEAD_SEEN_NEW protocol=CVS-OA-MSE k=5 target_visibility=target_old_and_seen_new_leo_support_labeled_unknown_eval_only label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --source_rxs \"${CEN51_TRAIN_RXS}\" --target_old_tx_ids \"${TARGET_OLD_TX_IDS}\" --target_old_rxs \"${TARGET_RECEIVER_IDS}\" --target_old_channel_view satellite --new_tx_ids \"${NEW_TX_IDS}\" --new_rxs \"${TARGET_RECEIVER_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --target_new_channel_view satellite --target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --target_old_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1443; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW/features.npz\" --output_json \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW/metrics.json\" --manifest_json \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW/manifest.json\" --score_table_csv \"${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW/score_table.csv\" --source_tx_ids \"${SOURCE_TX_IDS}\" --target_old_tx_ids \"${TARGET_OLD_TX_IDS}\" --target_old_support_per_tx \"2\" --target_old_query_per_tx \"50\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${OA_MSE_UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode oa_mse --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --oa_mse_adapter_rank 2 --oa_mse_adapter_steps 40 --old_acc_target 0.9 --seen_new_acc_target 0.75 --seed 1443")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW"
  ("${CMD[@]}" > "${LOG_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("OA_MSE_STAGE2C_HEAD_SEEN_NEW")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=OA_MSE_STAGE2C_HEAD_SEEN_NEW pid=${pid} gpu=${GPU} log=${LOG_ROOT}/OA_MSE_STAGE2C_HEAD_SEEN_NEW.out"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  STATUS=0
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[${idx}]}"; then
      echo "[SPACEBORNE-FSDA-COMPLETE] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=0"
    else
      rc=$?
      echo "[SPACEBORNE-FSDA-FAILED] id=${NAMES[${idx}]} pid=${PIDS[${idx}]} status=${rc}" >&2
      STATUS=${rc}
    fi
  done
  exit "${STATUS}"
fi
