#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/phase1_jmrs01_20260826}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_jmrs01_20260826}"
RUNNER="${RUNNER:-${ROOT}/code/audit_phase1_jmrs01.py}"
SCORER="${SCORER:-${ROOT}/code/score_phase1_jmrs01.py}"
OUT_ROOT="${RUN_ROOT}/${RUN_ID}"
SMOKE_ROOT="${RUN_ROOT}/${RUN_ID}_REAL_CKPT_NO_QUERY_SMOKE"
SCORE_ROOT="${OUT_ROOT}/score"

for required in "${PYTHON}" "${CHECKPOINT}" "${WISIG_PKL}" "${RUNNER}" "${SCORER}"; do
  [[ -f "${required}" ]] || { echo "[JMRS01-PREFLIGHT] missing=${required}" >&2; exit 2; }
done
[[ ! -e "${OUT_ROOT}" ]] || { echo "[JMRS01-PREFLIGHT] output_exists=${OUT_ROOT}" >&2; exit 3; }
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "[JMRS01-PREFLIGHT] smoke_output_exists=${SMOKE_ROOT}" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}.out" ]] || { echo "[JMRS01-PREFLIGHT] log_exists=${LOG_ROOT}/${RUN_ID}.out" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}_smoke.out" ]] || { echo "[JMRS01-PREFLIGHT] smoke_log_exists=${LOG_ROOT}/${RUN_ID}_smoke.out" >&2; exit 3; }
[[ ! -e "${LOG_ROOT}/${RUN_ID}_score.out" ]] || { echo "[JMRS01-PREFLIGHT] score_log_exists=${LOG_ROOT}/${RUN_ID}_score.out" >&2; exit 3; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
echo "[JMRS01-PROTOCOL] source_only=1 train=L_s select=V_select cal=V_cal audit=held_V_select rows=M0,R1,R2,D1,P1,P2,S1 D2=REMOVED"

echo "[JMRS01-LAUNCH] REAL CHECKPOINT NO-QUERY SMOKE"
"${PYTHON}" -u "${RUNNER}" \
  --output_dir "${SMOKE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows M0,R1,R2,D1,P1,P2,S1 \
  --device cuda:0 \
  --seed 20260826 \
  --sat_seed 20260824 \
  --max_eval_batches 1 \
  --smoke_only \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}_smoke.out"
test -s "${SMOKE_ROOT}/protocol_and_smoke.json"
echo "[JMRS01-SMOKE] PASS"

echo "[JMRS01-LAUNCH] SOURCE-ONLY NESTED-LORO S0"
"${PYTHON}" -u "${RUNNER}" \
  --output_dir "${OUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows M0,R1,R2,D1,P1,P2,S1 \
  --device cuda:0 \
  --seed 20260826 \
  --sat_seed 20260824 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --epochs 200 \
  --selection_interval 10 \
  --learning_rate 3e-4 \
  --probe_samples_per_receiver 128 \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}.out"

test -s "${OUT_ROOT}/run_manifest.json"
test -s "${OUT_ROOT}/predictions.jsonl"
test -s "${OUT_ROOT}/truth.jsonl"

echo "[JMRS01-SCORE] CONNECT TRUTH AFTER PREDICTION CLOSURE"
"${PYTHON}" -u "${SCORER}" \
  --predictions "${OUT_ROOT}/predictions.jsonl" \
  --truth "${OUT_ROOT}/truth.jsonl" \
  --output_dir "${SCORE_ROOT}" \
  --bootstrap_resamples 1000 \
  --seed 20260826 \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}_score.out"

for artifact in \
  mechanism_identity_stability.json \
  mechanism_receiver_probe.json \
  mechanism_loro_metrics.json \
  mechanism_clean_sat_consistency.json \
  mechanism_complementarity.json \
  mechanism_observability.json \
  mechanism_cost.json \
  mechanism_decision.json; do
  test -s "${SCORE_ROOT}/${artifact}"
done
echo "[JMRS01-LAUNCH] ANALYZED run_id=${RUN_ID}"
