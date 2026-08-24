#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-PHASE1_CCOI_PA_V2_S20260824_20260825A}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/phase1_ccoi_pa_v2_20260825}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_ccoi_pa_v2_20260825}"
SCENARIOS="clean,leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
OUT_ROOT="${RUN_ROOT}/${RUN_ID}"
SMOKE_ROOT="${RUN_ROOT}/${RUN_ID}_REAL_CKPT_NO_QUERY_SMOKE"

for required in "${PYTHON}" "${CHECKPOINT}" "${WISIG_PKL}"; do
  [[ -f "${required}" ]] || { echo "[CCOI-V2-PREFLIGHT] missing=${required}" >&2; exit 2; }
done
[[ ! -e "${OUT_ROOT}" ]] || { echo "[CCOI-V2-PREFLIGHT] output_exists=${OUT_ROOT}" >&2; exit 3; }
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "[CCOI-V2-PREFLIGHT] smoke_output_exists=${SMOKE_ROOT}" >&2; exit 3; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
echo "[CCOI-V2-PROTOCOL] source_roles=0.07/0.63/0.15/0.15 scenarios=${SCENARIOS}"

echo "[CCOI-V2-LAUNCH] REAL CHECKPOINT NO-QUERY SMOKE"
"${PYTHON}" -u "${ROOT}/code/train_phase1_ccoi_pa.py" \
  --output_dir "${SMOKE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows C2 \
  --device cuda:0 \
  --seed 20260824 \
  --sat_seed 20260824 \
  --max_train_batches 1 \
  --max_eval_batches 1 \
  --smoke_only \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}_smoke.out"
test -s "${SMOKE_ROOT}/protocol_and_smoke.json"
echo "[CCOI-V2-SMOKE] PASS"

echo "[CCOI-V2-LAUNCH] FULL MATRIX"
"${PYTHON}" -u "${ROOT}/code/train_phase1_ccoi_pa.py" \
  --output_dir "${OUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows C0,C1,C2,C3,C4 \
  --device cuda:0 \
  --seed 20260824 \
  --sat_seed 20260824 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --q_epochs 10 \
  --head_epochs 20 \
  --fusion_alpha 0.15 \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}.out"
"${PYTHON}" -u "${ROOT}/code/score_phase1_ccoi_pa.py" --run_dir "${OUT_ROOT}" \
  2>&1 | tee -a "${LOG_ROOT}/${RUN_ID}.out"
test -s "${OUT_ROOT}/C0/metrics.json"
test -s "${OUT_ROOT}/C4/metrics.json"
echo "[CCOI-V2-LAUNCH] ANALYZED run_id=${RUN_ID}"
