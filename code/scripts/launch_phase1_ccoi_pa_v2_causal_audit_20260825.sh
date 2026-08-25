#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
SIDECAR="${SIDECAR:-${ROOT}/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/phase1_ccoi_pa_v2_causal_audit_20260825}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_ccoi_pa_v2_causal_audit_20260825}"
OUT_ROOT="${RUN_ROOT}/${RUN_ID}"
SMOKE_ROOT="${RUN_ROOT}/${RUN_ID}_REAL_CKPT_NO_QUERY_SMOKE"

for required in "${PYTHON}" "${CHECKPOINT}" "${WISIG_PKL}" "${SIDECAR}"; do
  [[ -f "${required}" ]] || { echo "[CCOI-CAUSAL-PREFLIGHT] missing=${required}" >&2; exit 2; }
done
[[ ! -e "${OUT_ROOT}" ]] || { echo "[CCOI-CAUSAL-PREFLIGHT] output_exists=${OUT_ROOT}" >&2; exit 3; }
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "[CCOI-CAUSAL-PREFLIGHT] smoke_output_exists=${SMOKE_ROOT}" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}.out" ]] || { echo "[CCOI-CAUSAL-PREFLIGHT] log_exists=${LOG_ROOT}/${RUN_ID}.out" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}_smoke.out" ]] || { echo "[CCOI-CAUSAL-PREFLIGHT] smoke_log_exists=${LOG_ROOT}/${RUN_ID}_smoke.out" >&2; exit 3; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
echo "[CCOI-CAUSAL-PROTOCOL] fit=L_s eval=V_select target_or_query_access=0 repeat_C0_C4=0"

echo "[CCOI-CAUSAL-LAUNCH] REAL CHECKPOINT AND C4 NO-QUERY SMOKE"
"${PYTHON}" -u "${ROOT}/code/audit_phase1_ccoi_pa_v2.py" \
  --output_dir "${SMOKE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --sidecar "${SIDECAR}" \
  --wisig_pkl "${WISIG_PKL}" \
  --device cuda:0 \
  --seed 20260824 \
  --sat_seed 20260824 \
  --max_eval_batches 1 \
  --smoke_only \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}_smoke.out"
test -s "${SMOKE_ROOT}/protocol_and_smoke.json"
echo "[CCOI-CAUSAL-SMOKE] PASS"

echo "[CCOI-CAUSAL-LAUNCH] SOURCE-ONLY AUDIT"
"${PYTHON}" -u "${ROOT}/code/audit_phase1_ccoi_pa_v2.py" \
  --output_dir "${OUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --sidecar "${SIDECAR}" \
  --wisig_pkl "${WISIG_PKL}" \
  --device cuda:0 \
  --seed 20260824 \
  --sat_seed 20260824 \
  --batch_size 128 \
  --eval_batch_size 128 \
  --probe_steps 400 \
  --holdout_steps 800 \
  --bootstrap_resamples 1000 \
  --pair_thresholds 0.50,0.70,0.80,0.90,0.95,0.98,0.99 \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}.out"
for artifact in protocol_and_smoke.json feature_audit.json probe_audit.json pair_geometry.json holdout_factorization.json complementarity.json audit_manifest.json; do
  test -s "${OUT_ROOT}/${artifact}"
done
echo "[CCOI-CAUSAL-LAUNCH] ANALYZED run_id=${RUN_ID}"
