#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_anchor_rescue_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_satrepair_anchor7_rx20_1_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satrepair_anchor7_rx20_1_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satrepair_anchor7_rx3_19_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satrepair_anchor7_rx3_19_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satrepair_anchor7_rx7_14_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satrepair_anchor7_rx7_14_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satrepair_anchor7_rx7_7_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satrepair_anchor7_rx7_7_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satrepair_anchor7_rx8_8_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satrepair_anchor7_rx8_8_u1_20260702 1-16,4-10"
)

declare -a POLICIES=(
  "SATREPAIRA7_MLP_M50_PROXY02"
  "SATREPAIRA7_MLP_M50_COR_PROXY05"
  "SATREPAIRA7_LIN_PROXY05"
  "SATREPAIRA7_MLP_M50_PROXY05"
)

declare -a CONF_Q=(0.00 0.25 0.50 0.75 0.90 0.95 0.98 0.99)
declare -a MATCH_FLAGS=(0 1)

echo "[PHASE1-ANCHOR-RESCUE-SWEEP] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  FEATURE_NPZ="${ROOT}/runs/${RUN_ID}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATREPAIRA7/features_satrepair_anchor7.npz"
  for POLICY in "${POLICIES[@]}"; do
    SCORE_TABLE="${ROOT}/runs/${RUN_ID}/${POLICY}/score_table.csv"
    if [[ ! -f "${FEATURE_NPZ}" || ! -f "${SCORE_TABLE}" ]]; then
      echo "[PHASE1-ANCHOR-RESCUE-SKIP] run_id=${RUN_ID} policy=${POLICY} missing_input=1"
      continue
    fi
    for Q in "${CONF_Q[@]}"; do
      for MATCH in "${MATCH_FLAGS[@]}"; do
        SUFFIX="q${Q//./p}_match${MATCH}"
        OUT_DIR="${ROOT}/runs/${RUN_ID}/${POLICY}_ANCHOR_RESCUE_${SUFFIX}"
        mkdir -p "${OUT_DIR}"
        extra_flags=(--disable_margin_gate --disable_energy_gate)
        if [[ "${MATCH}" == "1" ]]; then
          extra_flags+=(--require_anchor_table_match)
        fi
        env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
          "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_anchor_rescue_reject.py" \
          --feature_npz "${FEATURE_NPZ}" \
          --reject_score_table_csv "${SCORE_TABLE}" \
          --source_tx_ids "${SOURCE_TX_IDS}" \
          --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
          --confidence_quantile "${Q}" \
          --unknown_far_target 0.05 \
          --max_old_drop_pp 2.0 \
          --output_json "${OUT_DIR}/metrics.json" \
          "${extra_flags[@]}" \
          > "${OUT_DIR}/anchor_rescue.out" 2>&1
      done
    done
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/anchor_rescue_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_satrepair_anchor7_*_20260702/SATREPAIRA7_*_ANCHOR_RESCUE_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    cal = data.get("calibration", {})
    policy_dir = metrics_path.parent.name
    base_policy = policy_dir.split("_ANCHOR_RESCUE_")[0]
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "base_policy": base_policy,
        "rescue_policy": policy_dir,
        "confidence_quantile": cal.get("confidence_quantile"),
        "confidence_min": cal.get("confidence_min"),
        "require_anchor_table_match": data.get("rescue_policy", {}).get("require_anchor_table_match"),
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "rescued_known_query": data.get("rescued_known_query"),
        "rescued_unknown_query": data.get("rescued_unknown_query"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
    })
fields = list(rows[0].keys()) if rows else ["run_id", "base_policy"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-ANCHOR-RESCUE-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/anchor_rescue_sweep_summary.csv"
