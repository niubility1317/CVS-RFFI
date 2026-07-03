#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_localmem_target1_v25_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
KS="${KS:-8,32,128}"
ALPHAS="${ALPHAS:-0.25,0.50,0.75}"
THRESHOLD_QUANTILES="${THRESHOLD_QUANTILES:-0.05,0.10,0.20}"
GATE_SCALE="${GATE_SCALE:-0.03}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.05}"

mkdir -p "${LOG_ROOT}"

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TRAIN_SAT_CLEAR="${SOURCE_PAIR_DIR}/source_leo_clear_weak.npz"
TRAIN_SAT_LOW="${SOURCE_PAIR_DIR}/source_leo_low_elev_weak.npz"
TRAIN_SAT_RAIN="${SOURCE_PAIR_DIR}/source_leo_rain_weak.npz"
for required in "${CLEAN_NPZ}" "${TRAIN_SAT_CLEAR}" "${TRAIN_SAT_LOW}" "${TRAIN_SAT_RAIN}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[V25-MISSING] ${required}" >&2
    exit 2
  fi
done

RUN_IDS="phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702,phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702,phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702,phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702,phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702,phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702,phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702,phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702,phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702,phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702"

echo "[PHASE1-LOCALMEM-TARGET1-V25] start=$(date -Is) ks=${KS} alphas=${ALPHAS} threshold_quantiles=${THRESHOLD_QUANTILES}"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/build_phase1_local_residual_memory_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --train_sat_npz "${TRAIN_SAT_CLEAR}" \
  --train_sat_npz "${TRAIN_SAT_LOW}" \
  --train_sat_npz "${TRAIN_SAT_RAIN}" \
  --run_ids "${RUN_IDS}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --ks "${KS}" \
  --alphas "${ALPHAS}" \
  --threshold_quantiles "${THRESHOLD_QUANTILES}" \
  --gate_scale "${GATE_SCALE}" \
  --local_temperature "${LOCAL_TEMPERATURE}" \
  > "${LOG_ROOT}/build_localmem.out" 2>&1

VARIANTS=""
IFS=',' read -r -a k_arr <<<"${KS}"
IFS=',' read -r -a alpha_arr <<<"${ALPHAS}"
IFS=',' read -r -a q_arr <<<"${THRESHOLD_QUANTILES}"
for k in "${k_arr[@]}"; do
  for q in "${q_arr[@]}"; do
    qtag=$(awk -v x="${q}" 'BEGIN { printf("Q%02d", int(x * 100 + 0.5)) }')
    for alpha in "${alpha_arr[@]}"; do
      atag=$(awk -v x="${alpha}" 'BEGIN { printf("A%03d", int(x * 100 + 0.5)) }')
      item=$(printf "LEOMEM1_K%03d_%s_%s" "${k}" "${qtag}" "${atag}")
      if [[ -z "${VARIANTS}" ]]; then
        VARIANTS="${item}"
      else
        VARIANTS="${VARIANTS},${item}"
      fi
    done
  done
done

SUMMARY_CSV="${LOG_ROOT}/target1_strong_v25_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v25_metrics.json"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --variants "${VARIANTS}" \
  > "${LOG_ROOT}/target1_strong_v25_eval.out" 2>&1

"${PYTHON}" - "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v25_best.json" <<'PY'
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

def truth(row, key):
    return str(row.get(key, "")).lower() == "true"

def top(frame, key, n=15):
    return sorted(frame, key=key)[:n]

best_by_run = {}
for row in candidates:
    gate_count = sum(truth(row, k) for k in [
        "passes_old_recovery_gate",
        "passes_clean_fidelity_gate",
        "passes_floor_gate",
        "passes_margin_gate",
        "passes_unknown_safety_gate",
    ])
    key = row["run_id"]
    score = (gate_count, row["target_old_closed_acc"], row["target_old_min_tx_acc"], -abs(row["target_unknown_far_delta_vs_identity"]))
    if key not in best_by_run or score > best_by_run[key][0]:
        best_by_run[key] = (score, row)

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": len(passes),
    "pass_rows": passes[:30],
    "gate_counts": {
        key: sum(truth(r, key) for r in candidates)
        for key in [
            "passes_old_recovery_gate",
            "passes_clean_fidelity_gate",
            "passes_floor_gate",
            "passes_margin_gate",
            "passes_unknown_safety_gate",
            "passes_strong_target1",
        ]
    },
    "best_by_run": [item[1] for item in sorted(best_by_run.values(), key=lambda x: x[1]["run_id"])],
    "best_by_target_old_acc": top(candidates, lambda r: (-r["target_old_closed_acc"], r["clean_repaired_drop_pp"], r["target_unknown_far_source05"])),
    "best_by_tx_floor": top(candidates, lambda r: (-r["target_old_min_tx_acc"], -r["target_old_closed_acc"], r["target_unknown_far_source05"])),
    "best_unknown_safe": top(candidates, lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_delta_pp_vs_identity"])),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass", "gate_counts"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-LOCALMEM-TARGET1-V25-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
