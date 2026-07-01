#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

ROOT="${CVS_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/stage2_gate_eval_v2_20260701}"
declare -a CANDIDATES=(
  "FSP_VAC_R17_Q2_HARDK3_E280"
  "FSP_VAC_R20_Q2_SAT70_E280"
  "FSP_VAC_R28_Q2_SAT72_E300"
  "FSP_VAC_R22_Q2_SOURCEEPHI_E280"
  "FSP_VAC_T13_LATE60_SAT68_E260"
)

for CID in "${CANDIDATES[@]}"; do
  PKG="${ROOT}/runs/phase1_fsp_vacuum_20260701/${CID}/phase2_zid_prototypes.pt"
  CMD=("${PYTHON_BIN}" "${ROOT}/code/scripts/eval_synthetic_reject_benchmark.py" --prototype_package "${PKG}" --output "${OUT_ROOT}/${CID}/synthetic_reject_metrics.json")
  echo "[STAGE2-GATE-EVAL-V2] ${CID}"
  printf '  %q' "${CMD[@]}"
  printf '\n'
  if [[ "${MODE}" == "--execute" ]]; then
    mkdir -p "${OUT_ROOT}/${CID}"
    "${CMD[@]}"
  fi
done

