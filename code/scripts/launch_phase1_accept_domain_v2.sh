#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-phase1_accept_domain_v2_20260701}"
MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

ROOT="${CVS_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON_ARGS=(
  --dataset Dataset_WigSig/ManySig.pkl
  --split_mode tx_rx_day_1_7_2
  --labeled_ratio 0.10
  --unlabeled_ratio 0.70
  --source_val_ratio 0.20
  --phase2_export_prototypes true
  --phase2_fuse_prototypes true
  --phase2_fuse_accept_policy local_component
  --phase2_fuse_global_ball_accept false
)

declare -a MATRIX=(
  "0 OS_R17_NEGSPACE_E280 --epochs 280 --base_candidate FSP_VAC_R17_Q2_HARDK3_E280 --neg_shell_ratio 0.25 --neg_inter_ratio 0.25 --lambda_energy_out 1.0 --lambda_reject_neg 1.0"
  "1 OS_R20_TAILQ_E280 --epochs 280 --base_candidate FSP_VAC_R20_Q2_SAT70_E280 --tail_quarantine true --tail_core_quantile 0.80 --tail_accept_quantile 0.92 --lambda_tail_cvar 1.5"
  "2 OS_R28_GATEONLY_E260 --epochs 260 --base_candidate FSP_VAC_R28_Q2_SAT72_E300"
  "3 OS_T13_RISK_E260 --epochs 260 --base_candidate FSP_VAC_T13_LATE60_SAT68_E260 --unlabeled_risk_buffer true --risk_maxprob_min 0.70 --lambda_risk_energy_out 1.0"
)

for row in "${MATRIX[@]}"; do
  read -r GPU CID EXTRA <<<"${row}"
  CMD=(env "CUDA_VISIBLE_DEVICES=${GPU}" "${PYTHON_BIN}" -u "${ROOT}/code/SSDG/train_ssdg.py" --run_id "${RUN_ID}" --candidate_id "${CID}" --output_dir "${ROOT}/runs/${RUN_ID}/${CID}" "${COMMON_ARGS[@]}" ${EXTRA})
  echo "[ACCEPT-DOMAIN-V2] ${CID} gpu=${GPU}"
  printf '  %q' "${CMD[@]}"
  printf '\n'
  if [[ "${MODE}" == "--execute" ]]; then
    "${CMD[@]}" > "${ROOT}/logs/${RUN_ID}/${CID}.out" 2>&1 &
  fi
done

if [[ "${MODE}" == "--execute" ]]; then
  wait
fi
