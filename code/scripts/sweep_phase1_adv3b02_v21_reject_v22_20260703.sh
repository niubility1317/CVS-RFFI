#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_v21_reject_v22_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
RUN_GLOB="${RUN_GLOB:-phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
FEATURE_RELPATH="${FEATURE_RELPATH:-features_leo_repaired.npz}"

mkdir -p "${LOG_ROOT}/parts"

PASS100C0="LEOADAPT5_MARGIN_UNK_MLP_BLEND100_CAP000"
PASS100C5="LEOADAPT5_MARGIN_UNK_MLP_BLEND100_CAP050"
PASS200C0="LEOADAPT5_MARGIN_UNK_MLP_BLEND200_CAP000"
PASS200C5="LEOADAPT5_MARGIN_UNK_MLP_BLEND200_CAP050"
LOWFAR650C0="LEOADAPT5_MARGIN_UNK_MLP_BLEND650_CAP000"
LOWFAR650C5="LEOADAPT5_MARGIN_UNK_MLP_BLEND650_CAP050"
CAP000_ALL="LEOADAPT5_MARGIN_UNK_MLP_BLEND100_CAP000,LEOADAPT5_MARGIN_UNK_MLP_BLEND200_CAP000,LEOADAPT5_MARGIN_UNK_MLP_BLEND350_CAP000,LEOADAPT5_MARGIN_UNK_MLP_BLEND500_CAP000,LEOADAPT5_MARGIN_UNK_MLP_BLEND650_CAP000,LEOADAPT5_MARGIN_UNK_MLP_BLEND800_CAP000"
CAP050_ALL="LEOADAPT5_MARGIN_UNK_MLP_BLEND100_CAP050,LEOADAPT5_MARGIN_UNK_MLP_BLEND200_CAP050,LEOADAPT5_MARGIN_UNK_MLP_BLEND350_CAP050,LEOADAPT5_MARGIN_UNK_MLP_BLEND500_CAP050,LEOADAPT5_MARGIN_UNK_MLP_BLEND650_CAP050,LEOADAPT5_MARGIN_UNK_MLP_BLEND800_CAP050"
PASS4="${PASS100C0},${PASS100C5},${PASS200C0},${PASS200C5}"
LOWFAR6="${PASS4},${LOWFAR650C0},${LOWFAR650C5}"

cat > "${LOG_ROOT}/adapter_sets.tsv" <<EOF
identity	LEOADAPT3_IDENTITY
pass100cap000	${PASS100C0}
pass100cap050	${PASS100C5}
pass200cap000	${PASS200C0}
pass200cap050	${PASS200C5}
lowfar650cap000	${LOWFAR650C0}
lowfar650cap050	${LOWFAR650C5}
pass4	${PASS4}
identity_pass4	LEOADAPT3_IDENTITY,${PASS4}
lowfar6	${LOWFAR6}
cap000_all_mlp	${CAP000_ALL}
cap050_all_mlp	${CAP050_ALL}
all_mlp_caps	${CAP000_ALL},${CAP050_ALL}
EOF

echo "[PHASE1-V21-REJECT-V22] start=$(date -Is) run_glob=${RUN_GLOB}"
while IFS=$'\t' read -r set_name adapters; do
  [[ -n "${set_name}" ]] || continue
  echo "[PHASE1-V21-REJECT-V22-SET] ${set_name} adapters=${adapters}"
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_repair_ensemble_manifold_reject_20260703.py" \
    --runs_root "${RUNS_ROOT}" \
    --run_glob "${RUN_GLOB}" \
    --adapters "${adapters}" \
    --feature_relpath "${FEATURE_RELPATH}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --out_csv "${LOG_ROOT}/parts/${set_name}.csv" \
    --metrics_json "${LOG_ROOT}/parts/${set_name}.json" \
    --run_tag "V22_v21_reject_${set_name}" \
    > "${LOG_ROOT}/parts/${set_name}.out" 2>&1
done < "${LOG_ROOT}/adapter_sets.tsv"

"${PYTHON}" - <<'PY' "${LOG_ROOT}"
import csv
import json
import math
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
rows = []
for csv_path in sorted((log_root / "parts").glob("*.csv")):
    set_name = csv_path.stem
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["adapter_set"] = set_name
            rows.append(row)

out_csv = log_root / "v21_reject_v22_summary.csv"
if rows:
    leading = ["adapter_set", "run_id", "target_rx", "mode", "score_name", "threshold_policy"]
    fields = leading + [k for k in rows[0].keys() if k not in leading]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def b(v):
    return str(v).strip().lower() in {"1", "true", "yes"}

def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return math.nan

dual = [r for r in rows if b(r.get("passes_dual_target"))]
far_ok = [r for r in rows if b(r.get("passes_unknown_far_target"))]
old_ok = [r for r in rows if b(r.get("passes_old_drop_target"))]
best_joint = None
for r in rows:
    far = f(r, "unknown_FAR")
    drop = f(r, "old_drop_pp_vs_closed")
    score = max(0.0, far - 0.05) * 100.0 + max(0.0, drop - 2.0)
    r["joint_gap_score"] = score
    key = (score, far, drop)
    if best_joint is None or key < (
        best_joint["joint_gap_score"],
        f(best_joint, "unknown_FAR"),
        f(best_joint, "old_drop_pp_vs_closed"),
    ):
        best_joint = r

best_under_far = None
for r in far_ok:
    key = (-f(r, "known_full_accuracy_after_reject"), f(r, "old_drop_pp_vs_closed"), f(r, "unknown_FAR"))
    if best_under_far is None or key < (
        -f(best_under_far, "known_full_accuracy_after_reject"),
        f(best_under_far, "old_drop_pp_vs_closed"),
        f(best_under_far, "unknown_FAR"),
    ):
        best_under_far = r

summary = {
    "phase": "phase1_v21_reject_v22",
    "rows": len(rows),
    "dual_pass": len(dual),
    "far_ok": len(far_ok),
    "old_drop_ok": len(old_ok),
    "out_csv": str(out_csv),
    "uses_target_clean": False,
    "uses_target_labels_for_threshold": False,
    "uses_unknown_query_for_threshold": False,
    "best_joint": best_joint,
    "best_under_far": best_under_far,
}
(log_root / "v21_reject_v22_best.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "dual_pass", "far_ok", "old_drop_ok", "out_csv"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-V21-REJECT-V22-DONE] end=$(date -Is) summary=${LOG_ROOT}/v21_reject_v22_summary.csv"
