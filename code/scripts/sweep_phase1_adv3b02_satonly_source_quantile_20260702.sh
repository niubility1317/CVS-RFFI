#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satonly_matrix_20260702}"
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

declare -a POLICIES=(
  "SRC9999 0.9999"
  "SRC99995 0.99995"
  "SRC1000 1.0000"
)

echo "[PHASE1-SATONLY-SOURCE-Q-SWEEP] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  SRC_FEATURE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW"
  SATONLY_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATONLY_MULTIVIEW"
  MERGED_NPZ="${SATONLY_DIR}/features_satonly_multiview.npz"
  mkdir -p "${SATONLY_DIR}" "${LOG_ROOT}"
  echo "[PHASE1-SATONLY-SOURCE-Q-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"

  "${PYTHON}" - <<'PY' "${SRC_FEATURE_DIR}" "${MERGED_NPZ}"
import json
import sys
from pathlib import Path

import numpy as np

feature_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
inputs = [feature_dir / name for name in ["sat_clear.npz", "sat_low.npz", "sat_rain.npz"]]
missing = [str(p) for p in inputs if not p.is_file()]
if missing:
    raise SystemExit(f"missing satellite feature files: {missing}")
arrays = {}
for path in inputs:
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            if key == "manifest_json":
                continue
            arrays.setdefault(key, []).append(np.asarray(data[key]))
merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
manifest = {
    "payload_source": "phase1_satonly_multiview_merge",
    "input_npz": [str(p) for p in inputs],
    "view_count": len(inputs),
    "contains_clean_view": False,
    "satellite_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
}
merged["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, **merged)
print(json.dumps({"out_npz": str(out), "rows": int(merged["features"].shape[0]), "views": len(inputs)}, ensure_ascii=False))
PY

  for policy_row in "${POLICIES[@]}"; do
    read -r POLICY_NAME SRC_Q <<<"${policy_row}"
    NAME="SATMV_LIN_${POLICY_NAME}"
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
      --threshold_policy source_accept \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile 0.05 \
      --head_type linear \
      --hidden_dim 64 \
      --epochs 500 \
      --lr 0.020 \
      --l2 0.0001 \
      --seed 4070211 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      > "${LOG_ROOT}/${NAME}_satonly_sweep1.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satonly_sourceq_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/SATMV_LIN_SRC*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
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
        "source_accept_quantile": threshold.get("source_accept_quantile"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATONLY-SOURCE-Q-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satonly_sourceq_sweep_summary.csv"
