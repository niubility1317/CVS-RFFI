#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
GPU="${GPU:-0}"
SEED="${SEED:-392002}"
DRY_RUN=0
ONLY_ROW=""
ROWS=(R0 R1 R2 R3 R4 R5 R6 R7 R8)
FINAL_EVALUATIONS=(clean leo_clear_weak leo_low_elev_weak leo_rain_weak)
SAT_SCHEDULE='1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak'
SAT_SCENARIOS='leo_clear_weak,leo_low_elev_weak,leo_rain_weak'

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --row=R0|--row=R1|--row=R2|--row=R3|--row=R4|--row=R5|--row=R6|--row=R7|--row=R8)
      ONLY_ROW="${arg#--row=}"
      ;;
    *) echo "[FCR-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -n "${RUN_ID}" ]] || { echo "[FCR-ERROR] caller must set RUN_ID" >&2; exit 2; }
[[ -n "${OUTPUT_ROOT}" ]] || { echo "[FCR-ERROR] caller must set OUTPUT_ROOT" >&2; exit 2; }
if [[ "${DRY_RUN}" != "1" && -e "${OUTPUT_ROOT}" ]]; then
  echo "[FCR-ERROR] refusing to overwrite existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${WISIG_PKL}" ]]; then
  echo "[FCR-ERROR] source WiSig file missing: ${WISIG_PKL}" >&2
  exit 4
fi

run_row() {
  local row="$1"
  local row_root="${OUTPUT_ROOT}/${row}"
  local command=(
    env "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${CODE_ROOT}/code/train.py"
    --dataset wisig
    --wisig_pkl "${WISIG_PKL}"
    --model_variant lite_d
    --run_name "${RUN_ID}_${row}"
    --seed "${SEED}"
    --device cuda:0
    --epochs 200
    --phase1_method adv3b02_fcr
    --use_fcr
    --fcr_ablation_row "${row}"
    --use_meta_ssl_cvs
    --ssl_labeled_ratio 0.07
    --ssl_unlabeled_ratio 0.63
    --ssl_val_ratio 0.30
    --sat_view_schedule "${SAT_SCHEDULE}"
    --use_sat_consistency
    --sat_cons_start_epoch 80
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
    --eval_sat_channel
    --eval_sat_scenarios "${SAT_SCENARIOS}"
    --best_save_path "${row_root}/best_joint.pth"
    --latest_save_path "${row_root}/latest.pth"
    --best_test_save_path "${row_root}/best_overall.pth"
    --best_primary_save_path "${row_root}/best_primary.pth"
    --best_unseen_day_unseen_rx_save_path "${row_root}/best_test_model.pth"
    --best_unseen_day_seen_rx_save_path "${row_root}/best_unseen_day_seen_rx.pth"
    --best_seen_day_unseen_rx_save_path "${row_root}/best_seen_day_unseen_rx.pth"
    --best_worst_rx_save_path "${row_root}/best_worst_rx.pth"
    --ema_save_path "${row_root}/ema.pth"
    --swa_save_path "${row_root}/swa.pth"
    --swad_save_path "${row_root}/swad.pth"
    --log_dir "${row_root}/logs"
    --fcr_diagnostics_path "${row_root}/fcr_diagnostics.json"
    --fcr_predictions_path "${row_root}/fcr_predictions.json"
  )
  printf '[FCR-ROW] run_id=%s row=%s output=%s final_eval=%s\n' \
    "${RUN_ID}" "${row}" "${row_root}" "${FINAL_EVALUATIONS[*]}"
  printf '[FCR-CMD] '; printf '%q ' "${command[@]}"; printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${row_root}"
  if ! "${command[@]}" > "${row_root}/train.log" 2>&1; then
    printf 'TRAIN_FAILED\n' > "${row_root}/status.txt"
    return 5
  fi
  for artifact in best_joint.pth fcr_diagnostics.json fcr_predictions.json train.log; do
    if [[ ! -s "${row_root}/${artifact}" ]]; then
      printf 'ARTIFACT_MISSING_%s\n' "${artifact}" > "${row_root}/status.txt"
      return 6
    fi
  done
  printf 'PREDICTIONS_READY\n' > "${row_root}/status.txt"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi
for row in "${ROWS[@]}"; do
  [[ -z "${ONLY_ROW}" || "${row}" == "${ONLY_ROW}" ]] || continue
  run_row "${row}"
done
