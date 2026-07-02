#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_oracle_reject_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${MATRIX_LOG_ROOT}"

echo "[PHASE1-ORACLE-REJECT-SWEEP] start=$(date -Is)"
mapfile -t SCORE_TABLES < <(
  find "${ROOT}/runs" -path '*/score_table.csv' \
    | grep -E 'phase1_adv3b02_(satrepair_anchor7|satrepair9|satblind15|satphysmv11|satphysmv11_constrained|sattta_rxlight|satunknown_singleview)_' \
    | grep -v '_CLASSCOND_' \
    | grep -v '_ANCHOR_RESCUE_' \
    | sort
)

for SCORE_TABLE in "${SCORE_TABLES[@]}"; do
  POLICY_DIR="$(basename "$(dirname "${SCORE_TABLE}")")"
  RUN_ID="$(basename "$(dirname "$(dirname "${SCORE_TABLE}")")")"
  OUT_DIR="${ROOT}/runs/${RUN_ID}/${POLICY_DIR}_ORACLE_DIAG"
  mkdir -p "${OUT_DIR}"
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_scoretable_oracle_reject.py" \
    --score_table_csv "${SCORE_TABLE}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --unknown_far_target 0.05 \
    --max_old_drop_pp 2.0 \
    --output_json "${OUT_DIR}/metrics.json" \
    > "${OUT_DIR}/oracle_diag.out" 2>&1
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/oracle_reject_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_*_20260702/*_ORACLE_DIAG/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_id = metrics_path.parent.parent.name
    policy = metrics_path.parent.name.replace("_ORACLE_DIAG", "")
    family = "unknown"
    for key in [
        "satrepair_anchor7",
        "satrepair9",
        "satblind15",
        "satphysmv11_constrained",
        "satphysmv11",
        "sattta_rxlight",
        "satunknown_singleview",
    ]:
        if key in run_id:
            family = key
            break
    for label, result in [
        ("global", data.get("global_oracle") or {}),
        ("global_under_far", data.get("global_oracle_best_under_far") or {}),
        ("class_conditional", data.get("class_conditional_oracle") or {}),
    ]:
        if not result:
            continue
        rows.append({
            "family": family,
            "run_id": run_id,
            "policy": policy,
            "oracle_type": label,
            "unknown_FAR": result.get("unknown_FAR"),
            "passes_unknown_far_target": result.get("passes_unknown_far_target"),
            "known_closed_accuracy_no_reject": result.get("known_closed_accuracy_no_reject"),
            "known_full_accuracy_after_reject": result.get("known_full_accuracy_after_reject"),
            "old_drop_pp_vs_closed": result.get("old_drop_pp_vs_closed"),
            "passes_old_drop_target": result.get("passes_old_drop_target"),
            "passes_dual_target": result.get("passes_dual_target"),
            "known_coverage": result.get("known_coverage"),
            "known_correct_accepted_count": result.get("known_correct_accepted_count"),
            "known_closed_correct_count": result.get("known_closed_correct_count"),
            "unknown_accepted_count": result.get("unknown_accepted_count"),
            "known_query_count": result.get("known_query_count"),
            "unknown_query_count": result.get("unknown_query_count"),
        })
fields = list(rows[0].keys()) if rows else ["family", "run_id", "policy", "oracle_type"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows), "score_tables": len(rows) // 3 if rows else 0})
PY
echo "[PHASE1-ORACLE-REJECT-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/oracle_reject_sweep_summary.csv"
