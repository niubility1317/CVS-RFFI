#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-spaceborne_fewshot_da_satadapt_smoke_20260612}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
echo "[SPACEBORNE-FSDA] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=2"

echo "[SPACEBORNE-FSDA-CANDIDATE] id=SFE_ZID_PROTO_K5_SYNTH protocol=CVS-SFE k=5 target_visibility=new_class_satellite_support_labeled label_set_relation=Y_T_has_unknown_new_tx"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py" --protocol sfe --dry_run_synthetic --shots 5 --unknown_threshold 0.70 --output_json "${RUNS_ROOT}/SFE_ZID_PROTO_K5_SYNTH/metrics.json")
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/SFE_ZID_PROTO_K5_SYNTH"
  "${CMD[@]}" > "${LOG_ROOT}/SFE_ZID_PROTO_K5_SYNTH.out" 2>&1
fi

echo "[SPACEBORNE-FSDA-CANDIDATE] id=FTRC_SAT_RXTX_K2_LABELED_BASE protocol=CVS-FTRC k=2 target_visibility=target_receiver_satellite_support_labeled label_set_relation=Y_T_equals_Y_S"
GPU="0"
CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/code/train_target_adapt.py" --teacher_ckpt "${TEACHER_CKPT}" --output_dir "${RUNS_ROOT}/FTRC_SAT_RXTX_K2_LABELED_BASE" --dataset wisig --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --target_loader test_unseen_day_unseen_rx --target_channel_view satellite --target_label_mode labeled --target_samples_per_rx_tx 2 --target_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --epochs 1 --adapt_steps_per_epoch 2 --target_batch_size 32 --lr_adapt 1e-4 --anchor_weight 0.05 --eval_detail_every 1 --eval_sat_channel true --eval_sat_on test_unseen_day_unseen_rx --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches 1 --eval_max_batches 1)
printf "[SPACEBORNE-FSDA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}/FTRC_SAT_RXTX_K2_LABELED_BASE"
  "${CMD[@]}" > "${LOG_ROOT}/FTRC_SAT_RXTX_K2_LABELED_BASE.out" 2>&1
fi
