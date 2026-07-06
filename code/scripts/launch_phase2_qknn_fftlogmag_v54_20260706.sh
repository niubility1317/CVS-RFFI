#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-phase2_qknn_fftlogmag_v54_20260706}"
REPORT_ID="phase2_qknn_hardpair_n20_20260706"
ARTIFACT_DIR="${ROOT}/automation_reports/CV-SincNet/${REPORT_ID}/artifacts/v54_fftlogmag_20260706"
LOG_DIR="${ROOT}/logs/${RUN_ID}"
FEATURE_NPZ="${ROOT}/runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08L5/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08L5_N20/features_hardpair_HP08L5_n20.npz"
AUX_NPZ="${ARTIFACT_DIR}/features_hardpair_HP08L5_n20_leo_fftlogmag96.npz"
OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2"
DEFAULT_PY="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ -x "${DEFAULT_PY}" ]]; then
  PY_CMD=("${DEFAULT_PY}")
else
  PY_CMD=(/opt/miniconda3/bin/conda run -n CVS-RFFI python)
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

mkdir -p "${ARTIFACT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

echo "run_id=${RUN_ID}"
echo "root=${ROOT}"
echo "gpu=${GPU}"
echo "python=${PY_CMD[*]}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "feature_npz=${FEATURE_NPZ}"
echo "aux_npz=${AUX_NPZ}"
date

run_cmd env CUDA_VISIBLE_DEVICES="${GPU}" "${PY_CMD[@]}" \
  code/scripts/phase2_raw_iq_sketch_export.py \
  --feature_npz "${FEATURE_NPZ}" \
  --manysig_pkl Dataset_WigSig/ManySig.pkl \
  --manytx_pkl Dataset_WigSig/ManyTx.pkl \
  --output_npz "${AUX_NPZ}" \
  --sketch_dim 96 \
  --sketch_method fft_logmag \
  --channel_view leo \
  --leo_tta_views 5 \
  --device cuda:0 \
  --batch_size 256

for spec in \
  "k5:5:421038" \
  "k10:10:421057"
do
  IFS=: read -r tag k seed <<< "${spec}"
  run_cmd "${PY_CMD[@]}" \
    code/scripts/phase2_support_metric_qknn_probe.py \
    --feature_npz "${FEATURE_NPZ}" \
    --aux_feature_npz "${AUX_NPZ}" \
    --output_json "${ARTIFACT_DIR}/n20_${tag}_v54_fftlogmag_policy_20260706.json" \
    --output_csv "${ARTIFACT_DIR}/n20_${tag}_v54_fftlogmag_policy_20260706.csv" \
    --output_predictions_csv "${ARTIFACT_DIR}/n20_${tag}_v54_fftlogmag_policy_20260706_predictions.csv" \
    --old_tx_ids "${OLD_TX}" \
    --new_tx_ids "${NEW_TX}" \
    --old_role target_old \
    --new_role target_unknown \
    --seed_start "${seed}" \
    --seed_count 1 \
    --k_old "${k}" \
    --k_new "${k}" \
    --query_per_old 70 \
    --query_per_new 70 \
    --pool_per_old "${k}" \
    --pool_per_new "${k}" \
    --transform_modes diag_whiten_fisher \
    --transform_strengths 0.1 \
    --topm_grid 4 \
    --proto_mix_grid 0.4 \
    --aux_score_weight_grid 0.22 \
    --adaptive_qknn_policy_grid stable_dualview_v54 \
    --scenario_aware \
    --balanced_assignment
done

date
echo "DONE ${RUN_ID}"
