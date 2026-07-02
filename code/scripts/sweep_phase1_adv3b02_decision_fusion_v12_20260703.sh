#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_decision_fusion_v12_20260703}"
OUT_CSV="${OUT_CSV:-${LOG_ROOT}/decision_fusion_v12_summary.csv}"

mkdir -p "${LOG_ROOT}"

echo "[PHASE1-DECISION-FUSION-V12] start=$(date -Is)"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_decision_fusion_reject_20260703.py" \
  --runs_root "${ROOT}/runs" \
  --out_csv "${OUT_CSV}" \
  --min_accepts all \
  --primary_branches all \
  > "${LOG_ROOT}/driver.out" 2>&1

"${PYTHON}" - <<'PY' "${OUT_CSV}" "${LOG_ROOT}/decision_fusion_v12_best.json"
import json
import sys
import pandas as pd
from pathlib import Path

csv_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
df = pd.read_csv(csv_path)
for col in ["unknown_FAR", "old_drop_pp_vs_closed", "known_closed_accuracy_no_reject", "known_full_accuracy_after_reject", "known_coverage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
dual = df[(df["unknown_FAR"] <= 0.05) & (df["old_drop_pp_vs_closed"] <= 2.0)]
far_pass = df[df["unknown_FAR"] <= 0.05]
old_pass = df[df["old_drop_pp_vs_closed"] <= 2.0]
df["joint_penalty"] = (df["unknown_FAR"] / 0.05 - 1).clip(lower=0) + (df["old_drop_pp_vs_closed"] / 2.0 - 1).clip(lower=0)

def records(frame, order, n=10):
    if frame.empty:
        return []
    return frame.sort_values(order).head(n).to_dict(orient="records")

summary = {
    "rows": int(len(df)),
    "dual_pass": int(len(dual)),
    "far_only_pass": int(len(far_pass)),
    "old_drop_only_pass": int(len(old_pass)),
    "best_far_with_old_drop_le_2": records(old_pass, ["unknown_FAR", "old_drop_pp_vs_closed"], 8),
    "best_old_drop_with_far_le_5pct": records(far_pass, ["old_drop_pp_vs_closed", "unknown_FAR"], 8),
    "nearest_joint": records(df, ["joint_penalty", "unknown_FAR", "old_drop_pp_vs_closed"], 10),
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "dual_pass", "far_only_pass", "old_drop_only_pass"]}, ensure_ascii=False))
PY

echo "[PHASE1-DECISION-FUSION-V12-DONE] end=$(date -Is) summary=${OUT_CSV}"
