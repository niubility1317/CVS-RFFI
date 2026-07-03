#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_blend_target1_v20_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
BLENDS="${BLENDS:-0.10,0.20,0.35,0.50,0.65,0.80}"

mkdir -p "${LOG_ROOT}"

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
if [[ ! -f "${CLEAN_NPZ}" ]]; then
  echo "[V20-MISSING] ${CLEAN_NPZ}" >&2
  exit 2
fi

RUN_IDS="phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702,phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702,phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702,phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702,phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702,phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702,phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702,phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702,phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702,phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702"
BASE_VARIANTS="LEOADAPT5_MARGIN_UNK_LINR,LEOADAPT5_MARGIN_UNK_MLP,LEOADAPT5_ULTRAID_LINR"

echo "[PHASE1-BLEND-TARGET1-V20] start=$(date -Is) blends=${BLENDS}"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/build_phase1_adapter_blends_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --run_ids "${RUN_IDS}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --variants "${BASE_VARIANTS}" \
  --blends "${BLENDS}" \
  > "${LOG_ROOT}/build_blends.out" 2>&1

VARIANTS=""
IFS=',' read -r -a base_arr <<<"${BASE_VARIANTS}"
IFS=',' read -r -a blend_arr <<<"${BLENDS}"
for base in "${base_arr[@]}"; do
  for blend in "${blend_arr[@]}"; do
    tag=$(awk -v x="${blend}" 'BEGIN { printf("BLEND%03d", int(x * 1000 + 0.5)) }')
    item="${base}_${tag}"
    if [[ -z "${VARIANTS}" ]]; then
      VARIANTS="${item}"
    else
      VARIANTS="${VARIANTS},${item}"
    fi
  done
done

SUMMARY_CSV="${LOG_ROOT}/target1_strong_v20_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v20_metrics.json"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --variants "${VARIANTS}" \
  > "${LOG_ROOT}/target1_strong_v20_eval.out" 2>&1

"${PYTHON}" - <<'PY' "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v20_best.json"
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
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        row["passes_strong_target1"] = str(row.get("passes_strong_target1", "")).lower() == "true"
        rows.append(row)

candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]
passes = [r for r in candidates if r["passes_strong_target1"]]

def top(frame, key, n=15):
    return sorted(frame, key=key)[:n]

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": len(passes),
    "pass_rows": passes[:30],
    "best_by_target_old_acc": top(candidates, lambda r: (-r["target_old_closed_acc"], r["clean_repaired_drop_pp"], r["target_unknown_far_source05"])),
    "best_by_delta": top(candidates, lambda r: (-r["target_old_delta_pp_vs_identity"], r["clean_repaired_drop_pp"], r["target_unknown_far_delta_vs_identity"])),
    "best_unknown_safe": top(candidates, lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_closed_acc"])),
    "best_margin_safe": top(candidates, lambda r: (abs(min(r["target_old_margin_delta_vs_identity"], 0.0)), r["target_old_true_dist_delta_vs_identity"], -r["target_old_closed_acc"])),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-BLEND-TARGET1-V20-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
