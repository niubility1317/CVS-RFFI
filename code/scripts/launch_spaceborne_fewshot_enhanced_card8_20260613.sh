#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fewshot_enhanced_card8_20260613}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3}"
NEW_TX_IDS="${NEW_TX_IDS:-4,5}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-}"
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
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8"
PIDS=()
NAMES=()

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_COSINE_K5 protocol=CVS-SFE k=5 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1342; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode cosine --openmax_tail_size 20 --openmax_quantile 0.95 --openmax_min_threshold 0.02 --seed 1342")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_COSINE_K5"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_COSINE_K5.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_COSINE_K5")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_COSINE_K5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_COSINE_K5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_MARGIN_K5 protocol=CVS-SFE k=5 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="1"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1343; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode combined --openmax_tail_size 20 --openmax_quantile 0.95 --openmax_min_threshold 0.02 --min_margin 0.05 --seed 1343")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_MARGIN_K5"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_MARGIN_K5.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_MARGIN_K5")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_MARGIN_K5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_MARGIN_K5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_MAHAL_K5 protocol=CVS-SFE k=5 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="2"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1344; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode mahalanobis --openmax_tail_size 20 --openmax_quantile 0.95 --openmax_min_threshold 0.02 --max_mahalanobis 8.0 --seed 1344")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_MAHAL_K5"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_MAHAL_K5.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_MAHAL_K5")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_MAHAL_K5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_MAHAL_K5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_OPENMAX_K5 protocol=CVS-SFE k=5 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="3"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1345; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 5 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode openmax --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --seed 1345")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_OPENMAX_K5"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_OPENMAX_K5.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_OPENMAX_K5")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_OPENMAX_K5 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_OPENMAX_K5.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_COMBINED_K10 protocol=CVS-SFE k=10 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="4"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1347; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 10 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode combined --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --min_margin 0.05 --max_mahalanobis 8.0 --seed 1347")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K10"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_COMBINED_K10.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_COMBINED_K10")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_COMBINED_K10 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_COMBINED_K10.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_WISIG_GATE_COMBINED_K20 protocol=CVS-SFE k=20 target_visibility=new_class_wisig_support_labeled label_set_relation=Y_T_has_explicit_nonoverlap_tx"
GPU="5"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "set -euo pipefail; mkdir -p \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20\"; \"${PYTHON}\" -u \"${ROOT}/code/export_spaceborne_features.py\" --ckpt \"${TEACHER_CKPT}\" --wisig_pkl \"${WISIG_PKL}\" --new_wisig_pkl \"${NEW_WISIG_PKL}\" --out_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20/features.npz\" --feature_name z_id --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo \"0\" --max_samples_per_tx \"200\" --batch_size \"512\" --device cuda:0 --seed 1357; \"${PYTHON}\" -u \"${ROOT}/code/eval_spaceborne_fewshot.py\" --protocol sfe --feature_npz \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20/features.npz\" --output_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20/metrics.json\" --manifest_json \"${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20/manifest.json\" --source_tx_ids \"${SOURCE_TX_IDS}\" --new_tx_ids \"${NEW_TX_IDS}\" --unknown_tx_ids \"${UNKNOWN_TX_IDS}\" --shots 20 --source_proto_per_tx \"20\" --source_query_per_tx \"20\" --query_per_tx \"50\" --unknown_threshold 0.7 --gate_mode combined --openmax_tail_size 20 --openmax_quantile 1.0 --openmax_min_threshold 0.02 --min_margin 0.05 --max_mahalanobis 8.0 --seed 1357")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_WISIG_GATE_COMBINED_K20"
  ("${CMD[@]}" > "${LOG_ROOT}/SFE_WISIG_GATE_COMBINED_K20.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("SFE_WISIG_GATE_COMBINED_K20")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=SFE_WISIG_GATE_COMBINED_K20 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/SFE_WISIG_GATE_COMBINED_K20.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=FTRC_WISIG_FEATURE_ADAPTER_K2 protocol=CVS-FTRC k=2 target_visibility=target_receiver_satellite_support_labeled label_set_relation=Y_T_equals_Y_S"
GPU="6"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" --teacher_ckpt "${TEACHER_CKPT}" --output_dir "${RUNS_ROOT}/FTRC_WISIG_FEATURE_ADAPTER_K2" --dataset wisig --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --target_loader test_unseen_day_unseen_rx --target_channel_view satellite --target_label_mode labeled --target_samples_per_rx_tx 2 --target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --epochs 20 --adapt_steps_per_epoch 20 --target_batch_size 32 --lr_adapt 1e-4 --entropy_weight 0 --consistency_weight 0 --pseudo_weight 0 --anchor_weight 0.05 --eval_detail_every 5 --target_adapter_type feature_residual --adapter_rank 4 --adapter_bottleneck 16 --adapter_alpha 1.0 --adapter_dropout 0.0 --freeze_base_stats true --update_norm false --update_classifier false --rollback_enabled true --eval_sat_channel true --eval_sat_on test_unseen_day_unseen_rx --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches 0 --eval_max_batches 0)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/FTRC_WISIG_FEATURE_ADAPTER_K2"
  ("${CMD[@]}" > "${LOG_ROOT}/FTRC_WISIG_FEATURE_ADAPTER_K2.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("FTRC_WISIG_FEATURE_ADAPTER_K2")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=FTRC_WISIG_FEATURE_ADAPTER_K2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/FTRC_WISIG_FEATURE_ADAPTER_K2.out"
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=FTRC_WISIG_LOGIT_LORA_K2 protocol=CVS-FTRC k=2 target_visibility=target_receiver_satellite_support_labeled label_set_relation=Y_T_equals_Y_S"
GPU="7"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" --teacher_ckpt "${TEACHER_CKPT}" --output_dir "${RUNS_ROOT}/FTRC_WISIG_LOGIT_LORA_K2" --dataset wisig --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --target_loader test_unseen_day_unseen_rx --target_channel_view satellite --target_label_mode labeled --target_samples_per_rx_tx 2 --target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --epochs 20 --adapt_steps_per_epoch 20 --target_batch_size 32 --lr_adapt 1e-4 --entropy_weight 0 --consistency_weight 0 --pseudo_weight 0 --anchor_weight 0.05 --eval_detail_every 5 --target_adapter_type logit_lora --adapter_rank 4 --adapter_bottleneck 16 --adapter_alpha 1.0 --adapter_dropout 0.0 --freeze_base_stats true --update_norm false --update_classifier false --rollback_enabled true --eval_sat_channel true --eval_sat_on test_unseen_day_unseen_rx --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches 0 --eval_max_batches 0)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/FTRC_WISIG_LOGIT_LORA_K2"
  ("${CMD[@]}" > "${LOG_ROOT}/FTRC_WISIG_LOGIT_LORA_K2.out" 2>&1) &
  pid="$!"
  PIDS+=("${pid}")
  NAMES+=("FTRC_WISIG_LOGIT_LORA_K2")
  echo "[SPACEBORNE-FSDA-LAUNCHED] id=FTRC_WISIG_LOGIT_LORA_K2 pid=${pid} gpu=${GPU} log=${LOG_ROOT}/FTRC_WISIG_LOGIT_LORA_K2.out"
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
