#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_target1_strong_v17_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${LOG_ROOT}"

SOURCE_CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
if [[ ! -f "${SOURCE_CLEAN_NPZ}" ]]; then
  echo "[V17-MISSING] ${SOURCE_CLEAN_NPZ}" >&2
  exit 2
fi

SUMMARY_CSV="${LOG_ROOT}/target1_strong_v17_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v17_metrics.json"
OUT_LOG="${LOG_ROOT}/target1_strong_v17.out"

echo "[PHASE1-TARGET1-STRONG-V17] start=$(date -Is)"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${SOURCE_CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  > "${OUT_LOG}" 2>&1

"${PYTHON}" - <<'PY' "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v17_best.json"
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
            "target_old_closed_acc",
            "target_old_delta_pp_vs_identity",
            "target_old_min_scenario_acc",
            "target_old_min_tx_acc",
            "clean_repaired_target_old_acc",
            "clean_repaired_drop_pp",
            "target_unknown_far_source05",
            "target_unknown_far_delta_vs_identity",
            "target_unknown_oldness_delta_vs_identity",
            "target_old_margin_delta_vs_identity",
            "target_old_true_dist_delta_vs_identity",
            "clean_target_old_acc",
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        row["passes_strong_target1"] = str(row.get("passes_strong_target1", "")).lower() == "true"
        rows.append(row)

candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]
passes = [r for r in candidates if r["passes_strong_target1"]]

def top(frame, key, n=10):
    return sorted(frame, key=key)[:n]

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": len(passes),
    "pass_rows": passes[:20],
    "best_by_target_old_acc": top(candidates, lambda r: (-r["target_old_closed_acc"], r["target_unknown_far_source05"])),
    "best_by_target_old_delta": top(candidates, lambda r: (-r["target_old_delta_pp_vs_identity"], r["target_unknown_far_delta_vs_identity"])),
    "best_unknown_safe": top(candidates, lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_delta_pp_vs_identity"])),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-TARGET1-STRONG-V17-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
