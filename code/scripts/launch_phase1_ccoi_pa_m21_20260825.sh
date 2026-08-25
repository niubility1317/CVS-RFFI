#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
SIDECAR="${SIDECAR:-${ROOT}/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/phase1_ccoi_pa_m21_20260825}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_ccoi_pa_m21_20260825}"
OUT_ROOT="${RUN_ROOT}/${RUN_ID}"
SMOKE_ROOT="${RUN_ROOT}/${RUN_ID}_REAL_CKPT_NO_QUERY_SMOKE"

for required in "${PYTHON}" "${CHECKPOINT}" "${WISIG_PKL}" "${SIDECAR}"; do
  [[ -f "${required}" ]] || { echo "[PA-M21-PREFLIGHT] missing=${required}" >&2; exit 2; }
done
[[ ! -e "${OUT_ROOT}" ]] || { echo "[PA-M21-PREFLIGHT] output_exists=${OUT_ROOT}" >&2; exit 3; }
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "[PA-M21-PREFLIGHT] smoke_output_exists=${SMOKE_ROOT}" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}.out" ]] || { echo "[PA-M21-PREFLIGHT] log_exists=${LOG_ROOT}/${RUN_ID}.out" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}_smoke.out" ]] || { echo "[PA-M21-PREFLIGHT] smoke_log_exists=${LOG_ROOT}/${RUN_ID}_smoke.out" >&2; exit 3; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
echo "[PA-M21-PROTOCOL] train=L_s gate=V_cal fit=V_select_fit audit=V_audit_retro target_or_query_access=0 repeat_old_AB=0"

echo "[PA-M21-LAUNCH] REAL CHECKPOINT NO-QUERY SMOKE"
"${PYTHON}" -u "${ROOT}/code/audit_phase1_ccoi_pa_m21.py" \
  --output_dir "${SMOKE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --sidecar "${SIDECAR}" \
  --wisig_pkl "${WISIG_PKL}" \
  --device cuda:0 \
  --seed 20260825 \
  --sat_seed 20260824 \
  --max_eval_batches 1 \
  --legacy_migration_mode \
  --smoke_only \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}_smoke.out"
test -s "${SMOKE_ROOT}/decision_manifest.json"
echo "[PA-M21-SMOKE] PASS"

echo "[PA-M21-LAUNCH] SOURCE-ONLY TWO-STAGE AUDIT"
"${PYTHON}" -u "${ROOT}/code/audit_phase1_ccoi_pa_m21.py" \
  --output_dir "${OUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --sidecar "${SIDECAR}" \
  --wisig_pkl "${WISIG_PKL}" \
  --device cuda:0 \
  --seed 20260825 \
  --sat_seed 20260824 \
  --batch_size 128 \
  --eval_batch_size 128 \
  --head_epochs 60 \
  --factor_steps 800 \
  --probe_steps 400 \
  --loto_steps 800 \
  --gate_steps 200 \
  --bootstrap_resamples 1000 \
  --block_candidates 10,20,25 \
  --fit_ratio 0.65 \
  --major_cell_minimum 10 \
  --gate_coverage_min 0.05 \
  --legacy_migration_mode \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}.out"

for artifact in \
  split_manifest.json \
  sidecar_architecture_c1p.json \
  sidecar_architecture_c4p.json \
  sidecar_training_summary.json \
  duplicate_audit.json \
  q_conditional_probe.json \
  m0_exact_pair_retrieval.json \
  factor_matrix_c1p.json \
  factor_matrix_c4p.json \
  loto_residual_audit.json \
  gate_calibration_summary.json \
  gate_audit_summary.json \
  decision_manifest.json \
  final_report.md; do
  test -s "${OUT_ROOT}/${artifact}"
done
echo "[PA-M21-LAUNCH] ANALYZED run_id=${RUN_ID}"
