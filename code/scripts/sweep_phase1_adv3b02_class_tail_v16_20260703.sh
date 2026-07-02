#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_class_tail_v16_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${LOG_ROOT}"

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TRAIN_SAT_CLEAR="${SOURCE_PAIR_DIR}/source_leo_clear_weak.npz"
TRAIN_SAT_LOW="${SOURCE_PAIR_DIR}/source_leo_low_elev_weak.npz"
TRAIN_SAT_RAIN="${SOURCE_PAIR_DIR}/source_leo_rain_weak.npz"
for required in "${CLEAN_NPZ}" "${TRAIN_SAT_CLEAR}" "${TRAIN_SAT_LOW}" "${TRAIN_SAT_RAIN}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[V16-MISSING] ${required}" >&2
    exit 2
  fi
done

SUMMARY_CSV="${LOG_ROOT}/class_tail_v16_summary.csv"
METRICS_JSON="${LOG_ROOT}/class_tail_v16_metrics.json"
BEST_JSON="${LOG_ROOT}/class_tail_v16_best.json"

echo "[PHASE1-CLASS-TAIL-V16] start=$(date -Is)"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_class_tail_reject_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --clean_npz "${CLEAN_NPZ}" \
  --train_sat_npz "${TRAIN_SAT_CLEAR}" \
  --train_sat_npz "${TRAIN_SAT_LOW}" \
  --train_sat_npz "${TRAIN_SAT_RAIN}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  > "${LOG_ROOT}/class_tail_v16.out" 2>&1

"${PYTHON}" - <<'PY' "${SUMMARY_CSV}" "${BEST_JSON}"
import csv
import json
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        for col in [
            "unknown_FAR",
            "old_drop_pp_vs_closed",
            "known_closed_accuracy_no_reject",
            "known_full_accuracy_after_reject",
            "known_coverage",
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        row["joint_penalty"] = max(0.0, row["unknown_FAR"] / 0.05 - 1.0) + max(0.0, row["old_drop_pp_vs_closed"] / 2.0 - 1.0)
        rows.append(row)

dual = [r for r in rows if r["unknown_FAR"] <= 0.05 and r["old_drop_pp_vs_closed"] <= 2.0]
far = [r for r in rows if r["unknown_FAR"] <= 0.05]
old = [r for r in rows if r["old_drop_pp_vs_closed"] <= 2.0]

def top(frame, order, n=10):
    return sorted(frame, key=lambda r: tuple(r[k] for k in order))[:n]

summary = {
    "rows": len(rows),
    "dual_pass": len(dual),
    "far_only_pass": len(far),
    "old_drop_only_pass": len(old),
    "best_far_with_old_drop_le_2": top(old, ["unknown_FAR", "old_drop_pp_vs_closed"], 10),
    "best_old_drop_with_far_le_5pct": top(far, ["old_drop_pp_vs_closed", "unknown_FAR"], 10),
    "nearest_joint": top(rows, ["joint_penalty", "unknown_FAR", "old_drop_pp_vs_closed"], 10),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "dual_pass", "far_only_pass", "old_drop_only_pass"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-CLASS-TAIL-V16-DONE] end=$(date -Is) summary=${SUMMARY_CSV} best=${BEST_JSON}"
