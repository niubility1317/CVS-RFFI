#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satonly_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702 1-16,4-10"
)

declare -a HEADS=(
  "SATMARG_LIN_P2 linear 64 500 0.020 1.0 2.0"
  "SATMARG_LIN_P5 linear 64 500 0.020 1.0 5.0"
  "SATMARG_MLP_S2 mlp 64 350 0.005 2.0 1.0"
  "SATMARG_MLP_S5 mlp 64 350 0.005 5.0 1.0"
  "SATMARG_MLP_S10 mlp 64 350 0.005 10.0 1.0"
  "SATMARG_MLP_S20 mlp 64 350 0.005 20.0 1.0"
)
declare -a POLICIES=(
  "SRC9999 0.9999"
  "SRC1000 1.0000"
)

echo "[PHASE1-SATONLY-MARGIN-SWEEP] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  MERGED_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATONLY_MULTIVIEW/features_satonly_multiview.npz"
  if [[ ! -f "${MERGED_NPZ}" ]]; then
    echo "[ERROR] missing satellite-only multiview NPZ: ${MERGED_NPZ}" >&2
    exit 3
  fi
  mkdir -p "${LOG_ROOT}"
  echo "[PHASE1-SATONLY-MARGIN-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"
  for head_row in "${HEADS[@]}"; do
    read -r HEAD_NAME HEAD_TYPE HIDDEN_DIM EPOCHS LR SRC_W PROXY_W <<<"${head_row}"
    for policy_row in "${POLICIES[@]}"; do
      read -r POLICY_NAME SRC_Q <<<"${policy_row}"
      NAME="${HEAD_NAME}_${POLICY_NAME}"
      OUT_DIR="${RUNS_ROOT}/${NAME}"
      mkdir -p "${OUT_DIR}"
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_multiview_reject.py" \
        --feature_npz "${MERGED_NPZ}" \
        --source_tx_ids "${SOURCE_TX_IDS}" \
        --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
        --train_known_roles source \
        --proxy_unknown_roles proxy_unknown \
        --known_query_roles target_old \
        --unknown_query_roles target_unknown \
        --threshold_policy source_accept \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile 0.05 \
        --head_type "${HEAD_TYPE}" \
        --loss_mode margin \
        --source_loss_weight "${SRC_W}" \
        --proxy_loss_weight "${PROXY_W}" \
        --margin 1.0 \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --l2 0.0001 \
        --seed 4070211 \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 2.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        > "${LOG_ROOT}/${NAME}_satonly_margin_sweep1.out" 2>&1
    done
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satonly_margin_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/SATMARG_*_SRC*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "policy": metrics_path.parent.name,
        "head_type": training.get("head_type"),
        "loss_mode": training.get("loss_mode"),
        "source_loss_weight": training.get("source_loss_weight"),
        "proxy_loss_weight": training.get("proxy_loss_weight"),
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "source_accept_quantile": threshold.get("source_accept_quantile"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATONLY-MARGIN-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satonly_margin_sweep_summary.csv"
