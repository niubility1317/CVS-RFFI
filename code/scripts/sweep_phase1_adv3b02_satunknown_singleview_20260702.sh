#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satunknown_singleview_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SEED="${SEED:-4070217}"

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

declare -a HEAD_POLICIES=(
  "SATUNK_LIN_SRC9999 linear 64 500 0.020 0.9999"
  "SATUNK_LIN_SRC1000 linear 64 500 0.020 1.0000"
  "SATUNK_MLP64_SRC9999 mlp 64 350 0.005 0.9999"
  "SATUNK_MLP64_SRC1000 mlp 64 350 0.005 1.0000"
)

declare -a PROTO_POLICIES=(
  "SATUNK_PROTO_COS_SRC9999 cosine 0.0 0.0 0.0 0.9999"
  "SATUNK_PROTO_COS_SRC1000 cosine 0.0 0.0 0.0 1.0000"
  "SATUNK_PROTO_MAH_SRC9999 diag_mahalanobis 0.0 0.0 0.0 0.9999"
  "SATUNK_PROTO_MAH_SRC1000 diag_mahalanobis 0.0 0.0 0.0 1.0000"
)

echo "[PHASE1-SATUNKNOWN-SINGLEVIEW-SWEEP] start=$(date -Is) seed=${SEED}"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  SRC_FEATURE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW"
  SINGLE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW"
  SINGLE_NPZ="${SINGLE_DIR}/features_satunknown_singleview.npz"
  mkdir -p "${SINGLE_DIR}" "${LOG_ROOT}"
  echo "[PHASE1-SATUNKNOWN-SINGLEVIEW-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"

  "${PYTHON}" - <<'PY' "${SRC_FEATURE_DIR}" "${SINGLE_NPZ}" "${SEED}"
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

feature_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
seed = str(sys.argv[3])
inputs = [
    ("leo_clear_weak", feature_dir / "sat_clear.npz"),
    ("leo_low_elev_weak", feature_dir / "sat_low.npz"),
    ("leo_rain_weak", feature_dir / "sat_rain.npz"),
]
missing = [str(path) for _, path in inputs if not path.is_file()]
if missing:
    raise SystemExit(f"missing satellite feature files: {missing}")

payloads = []
for scenario, path in inputs:
    with np.load(path, allow_pickle=True) as data:
        payloads.append((scenario, {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"}))

key_fields = ["dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"]

def row_key(arrays, i):
    return tuple(str(arrays[field][i]) for field in key_fields)

maps = []
for scenario, arrays in payloads:
    mapping = {}
    for i in range(int(arrays["features"].shape[0])):
        key = row_key(arrays, i)
        if key in mapping:
            raise SystemExit(f"duplicate sample metadata in scenario={scenario}: {key}")
        mapping[key] = i
    maps.append(mapping)
all_keys = set()
for mapping in maps:
    all_keys |= set(mapping)
if not all_keys:
    raise SystemExit("no sample metadata keys across satellite scenario exports")
ordered_keys = sorted(all_keys)
selected = []
for key in ordered_keys:
    digest = hashlib.sha256(("|".join([seed, *key])).encode("utf-8")).digest()
    start = int(digest[0]) % len(payloads)
    for offset in range(len(payloads)):
        candidate = (start + offset) % len(payloads)
        if key in maps[candidate]:
            selected.append(candidate)
            break
    else:
        raise SystemExit(f"internal error: no scenario row found for key={key}")
selected = np.asarray(selected, dtype=np.int64)

ref = payloads[0][1]
merged = {}
for field in ref:
    vals = []
    for key, choice in zip(ordered_keys, selected.tolist()):
        vals.append(payloads[int(choice)][1][field][maps[int(choice)][key]])
    merged[field] = np.asarray(vals, dtype=ref[field].dtype)
n = len(ordered_keys)
merged["channel_views"] = np.asarray(["satellite_unknown_leo"] * n)
merged["sat_scenarios"] = np.asarray([payloads[int(choice)][0] for choice in selected.tolist()])
manifest = {
    "payload_source": "phase1_satunknown_singleview_stable_mix",
    "input_npz": [str(path) for _, path in inputs],
    "contains_clean_view": False,
    "per_sample_observation_count": 1,
    "scenario_assignment": "stable_hash_hidden_from_evaluator",
    "satellite_scenario_pool": [scenario for scenario, _ in inputs],
    "seed": seed,
}
merged["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, **merged)
counts = {scenario: int((merged["sat_scenarios"].astype(str) == scenario).sum()) for scenario, _ in inputs}
print(json.dumps({"out_npz": str(out), "rows": n, "contains_clean_view": False, "scenario_counts": counts}, ensure_ascii=False))
PY

  for policy_row in "${HEAD_POLICIES[@]}"; do
    read -r NAME HEAD_TYPE HIDDEN_DIM EPOCHS LR SRC_Q <<<"${policy_row}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    mkdir -p "${OUT_DIR}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_multiview_reject.py" \
      --feature_npz "${SINGLE_NPZ}" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
      --train_known_roles source \
      --proxy_unknown_roles proxy_unknown \
      --known_query_roles target_old \
      --unknown_query_roles target_unknown \
      --threshold_policy source_accept \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile 0.05 \
      --head_type "${HEAD_TYPE}" \
      --hidden_dim "${HIDDEN_DIM}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --l2 0.0001 \
      --seed 4070217 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      > "${LOG_ROOT}/${NAME}_satunknown_singleview.out" 2>&1
  done

  for policy_row in "${PROTO_POLICIES[@]}"; do
    read -r NAME METRIC CONF_W ENT_W MARGIN_W SRC_Q <<<"${policy_row}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    mkdir -p "${OUT_DIR}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_prototype_reject.py" \
      --feature_npz "${SINGLE_NPZ}" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
      --train_known_roles source \
      --proxy_unknown_roles proxy_unknown \
      --known_query_roles target_old \
      --unknown_query_roles target_unknown \
      --metric "${METRIC}" \
      --confidence_weight "${CONF_W}" \
      --entropy_weight "${ENT_W}" \
      --margin_weight "${MARGIN_W}" \
      --threshold_policy source_accept \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile 0.05 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      > "${LOG_ROOT}/${NAME}_satunknown_singleview.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satunknown_singleview_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
patterns = [
    "phase1_adv3b02_multiview_keepold_*_20260702/SATUNK_LIN_SRC*/metrics.json",
    "phase1_adv3b02_multiview_keepold_*_20260702/SATUNK_MLP64_SRC*/metrics.json",
    "phase1_adv3b02_multiview_keepold_*_20260702/SATUNK_PROTO_*_SRC*/metrics.json",
]
for pattern in patterns:
    for metrics_path in sorted(runs_root.glob(pattern)):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = data.get("threshold", {})
        training = data.get("training", {})
        scoring = data.get("scoring", {})
        manifest = data.get("manifest", {})
        rows.append({
            "run_id": metrics_path.parent.parent.name,
            "policy": metrics_path.parent.name,
            "phase": data.get("phase"),
            "head_type": training.get("head_type", ""),
            "metric": scoring.get("metric", ""),
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
            "contains_clean_view": manifest.get("contains_clean_view"),
            "per_sample_observation_count": manifest.get("per_sample_observation_count"),
        })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATUNKNOWN-SINGLEVIEW-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satunknown_singleview_sweep_summary.csv"
