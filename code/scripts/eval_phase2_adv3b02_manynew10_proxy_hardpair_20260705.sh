#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_manynew10_proxy_hardpair_20260705}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"

EVAL_OLD_TX_IDS="${EVAL_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_RX="${TARGET_RX:-7-14}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3}"
CASE_ID="PHASE2_MANYNEW10_RX${TARGET_RX//-/_}"

mkdir -p "${LOG_ROOT}"
echo "[MANYNEW10-PROXY-HP-EVAL] run_id=${RUN_ID} target_rx=${TARGET_RX}"

run_support_metric_eval() {
  local name="$1"
  local feature_npz="${RUNS_ROOT}/${CASE_ID}/${name}/features_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${name}"
  if [[ ! -f "${feature_npz}" ]]; then
    echo "[MANYNEW10-PROXY-HP-EVAL-MISSING] ${feature_npz}" >&2
    exit 4
  fi
  echo "[MANYNEW10-PROXY-HP-EVAL-VARIANT] ${name}"
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_support_metric_qknn_probe.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_dir}/strict_n10_k10_support_metric_qknn.json" \
    --output_csv "${out_dir}/strict_n10_k10_support_metric_qknn.csv" \
    --new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_role target_old \
    --new_role target_unknown \
    --policies stable_first,scenario_diverse \
    --seed_start 421000 \
    --seed_count 120 \
    --k_old 10 \
    --k_new 10 \
    --pool_per_old 10 \
    --pool_per_new 10 \
    --query_per_old 70 \
    --query_per_new 70 \
    --transform_modes diag_fisher \
    --transform_strengths 0.5 \
    --topm_grid 4 \
    --proto_mix_grid 0.25 \
    --radius_norm_grid 0 \
    --old_bias_grid 0.001 \
    --neg_lambda_grid 0.7 \
    --neg_threshold_grid 0.75 \
    --neg_margin_grid 0.01 \
    --mutual_only_grid true \
    --scenario_aware \
    --balanced_assignment \
    > "${LOG_ROOT}/${name}_strict_k10_support_metric_qknn.out" 2>&1

  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_support_metric_qknn_probe.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_dir}/strict_n10_k5_support_metric_qknn.json" \
    --output_csv "${out_dir}/strict_n10_k5_support_metric_qknn.csv" \
    --new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_role target_old \
    --new_role target_unknown \
    --policies stable_first,scenario_diverse \
    --seed_start 421000 \
    --seed_count 120 \
    --k_old 5 \
    --k_new 5 \
    --pool_per_old 5 \
    --pool_per_new 5 \
    --query_per_old 75 \
    --query_per_new 75 \
    --transform_modes diag_whiten_fisher \
    --transform_strengths 0.1 \
    --topm_grid 5 \
    --proto_mix_grid 0.6 \
    --radius_norm_grid 0.1 \
    --old_bias_grid 0 \
    --neg_lambda_grid 0 \
    --neg_threshold_grid 0.7 \
    --neg_margin_grid 0 \
    --mutual_only_grid true \
    --scenario_aware \
    --balanced_assignment \
    > "${LOG_ROOT}/${name}_strict_k5_support_metric_qknn.out" 2>&1
}

run_support_metric_eval MANYNEW10_PROXY_HP_NORM_SAFE
run_support_metric_eval MANYNEW10_PROXY_HP_NORM_STRONG

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/manynew10_proxy_hardpair_summary.json" <<'PY'
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/strict_n10_k*_support_metric_qknn.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best", [])
    if not best:
        continue
    row = dict(best[0])
    row["variant"] = path.parent.name
    row["eval"] = "k10" if "_k10_" in path.name else "k5"
    row["json_path"] = str(path)
    row["strict_goal_pass"] = (
        row.get("query_old_acc", 0.0) >= 0.80
        and row.get("query_min_seen_new_class_acc", 0.0) >= 0.75
    )
    rows.append(row)
rows.sort(
    key=lambda r: (
        r.get("strict_goal_pass", False),
        r.get("eval") == "k10",
        r.get("query_min_seen_new_class_acc", 0.0),
        r.get("query_seen_new_acc", 0.0),
        r.get("query_old_acc", 0.0),
    ),
    reverse=True,
)
out_path.write_text(json.dumps({"rows": rows, "best": rows[:8]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"best": rows[:4], "out": str(out_path)}, ensure_ascii=False, indent=2))
PY

echo "[MANYNEW10-PROXY-HP-EVAL-DONE] summary=${LOG_ROOT}/manynew10_proxy_hardpair_summary.json"
