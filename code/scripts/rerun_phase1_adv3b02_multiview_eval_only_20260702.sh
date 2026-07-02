#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_multiview_matrix_20260702}"
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
  "MV_LIN linear 64 500 0.020"
  "MV_MLP64 mlp 64 350 0.005"
)
declare -a POLICIES=(
  "SRC999 source_accept 0.999 0.05"
  "SRC995 source_accept 0.995 0.05"
  "SRC990 source_accept 0.990 0.05"
  "PROXY05 proxy_far 0.995 0.05"
  "PROXY02 proxy_far 0.995 0.02"
  "MIN_SRC995_PROXY05 min_source_proxy 0.995 0.05"
  "MAX_SRC995_PROXY05 max_source_proxy 0.995 0.05"
)

echo "[PHASE1-MULTIVIEW-EVAL-ONLY] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  MERGED_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/features_multiview.npz"
  if [[ ! -f "${MERGED_NPZ}" ]]; then
    echo "[ERROR] missing multiview NPZ: ${MERGED_NPZ}" >&2
    exit 3
  fi
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  echo "[PHASE1-MULTIVIEW-EVAL-ONLY-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"
  for head_row in "${HEADS[@]}"; do
    read -r HEAD_NAME HEAD_TYPE HIDDEN_DIM EPOCHS LR <<<"${head_row}"
    for policy_row in "${POLICIES[@]}"; do
      read -r POLICY_NAME POLICY SRC_Q PROXY_Q <<<"${policy_row}"
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
        --threshold_policy "${POLICY}" \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile "${PROXY_Q}" \
        --head_type "${HEAD_TYPE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --l2 0.0001 \
        --seed 4070211 \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 3.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        > "${LOG_ROOT}/${NAME}_rerun1.out" 2>&1
    done
  done
  "${PYTHON}" - <<'PY' "${RUNS_ROOT}" "${RUNS_ROOT}/summary_phase1_multiview_keepold.csv"
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    rows.append({
        "run_id": root.name,
        "policy": metrics_path.parent.name,
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "head_type": training.get("head_type"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "proxy_false_accept_rate_at_threshold": threshold.get("proxy_false_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
        "proxy_unknown_count": data.get("proxy_unknown_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(out), "rows": len(rows)}, ensure_ascii=False))
PY
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/multiview_eval_only_summary.csv"
import csv
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for summary in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/summary_phase1_multiview_keepold.csv")):
    with summary.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
with out.open("w", encoding="utf-8", newline="") as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        f.write("run_id,policy\n")
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-MULTIVIEW-EVAL-ONLY-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/multiview_eval_only_summary.csv"
