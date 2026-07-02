#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_density_reject_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "rx20_1_u10 10-1,10-10"
  "rx20_1_u1 1-16,4-10"
  "rx3_19_u10 10-1,10-10"
  "rx3_19_u1 1-16,4-10"
  "rx7_14_u10 10-1,10-10"
  "rx7_14_u1 1-16,4-10"
  "rx7_7_u10 10-1,10-10"
  "rx7_7_u1 1-16,4-10"
  "rx8_8_u10 10-1,10-10"
  "rx8_8_u1 1-16,4-10"
)

declare -a FAMILIES=(
  "satblind15 phase1_adv3b02_satblind15 ADV3B02_CORE90_SOFT_E200_PHASE1_SATBLIND15 features_satblind15.npz"
  "satrepair_anchor7 phase1_adv3b02_satrepair_anchor7 ADV3B02_CORE90_SOFT_E200_PHASE1_SATREPAIRA7 features_satrepair_anchor7.npz"
)

declare -a PROTO_POLICIES=(
  "PROTO_COS_MIN05 cosine 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
  "PROTO_COS_PROXY05 cosine 0.0 0.0 0.0 proxy_far 1.0000 0.05"
  "PROTO_COS_SRC9999 cosine 0.0 0.0 0.0 source_accept 0.9999 0.05"
  "PROTO_EUC_MIN05 euclidean 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
  "PROTO_MAH_MIN05 diag_mahalanobis 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
  "PROTO_COS_CONF_MIN05 cosine 0.5 0.0 0.0 min_source_proxy 1.0000 0.05"
  "PROTO_COS_ENT_MIN05 cosine 0.0 0.5 0.0 min_source_proxy 1.0000 0.05"
  "PROTO_COS_MARG_MIN05 cosine 0.0 0.0 0.5 min_source_proxy 1.0000 0.05"
)

declare -a KNN_POLICIES=(
  "KNN_COS_K1_MIN05 mean cosine 1 min_source_proxy 1.0000 0.05 class"
  "KNN_COS_K3_MIN05 mean cosine 3 min_source_proxy 1.0000 0.05 class"
  "KNN_COS_K5_MIN05 mean cosine 5 min_source_proxy 1.0000 0.05 class"
  "KNN_EUC_K3_MIN05 mean euclidean 3 min_source_proxy 1.0000 0.05 class"
  "KNN_COS_K3_PROXY05 mean cosine 3 proxy_far 1.0000 0.05 class"
  "KNN_COS_K3_SRC9999 mean cosine 3 source_accept 0.9999 0.05 class"
  "KNN_COS_BEST_K3_MIN05 best_conf cosine 3 min_source_proxy 1.0000 0.05 class"
  "KNN_COS_MED_K3_MIN05 median cosine 3 min_source_proxy 1.0000 0.05 class"
)

echo "[PHASE1-DENSITY-REJECT-SWEEP] start=$(date -Is)"
for family in "${FAMILIES[@]}"; do
  read -r FAMILY_NAME RUN_PREFIX FEATURE_DIR FEATURE_FILE <<<"${family}"
  for cell in "${CELLS[@]}"; do
    read -r CELL_ID UNKNOWN_TX_IDS <<<"${cell}"
    RUN_ID="${RUN_PREFIX}_${CELL_ID}_20260702"
    FEATURE_NPZ="${ROOT}/runs/${RUN_ID}/${FEATURE_DIR}/${FEATURE_FILE}"
    if [[ ! -f "${FEATURE_NPZ}" ]]; then
      echo "[PHASE1-DENSITY-SKIP] family=${FAMILY_NAME} run_id=${RUN_ID} missing_feature=1"
      continue
    fi
    for row in "${PROTO_POLICIES[@]}"; do
      read -r NAME METRIC CONF_W ENT_W MARG_W TH_POLICY SRC_Q PROXY_Q <<<"${row}"
      OUT_DIR="${ROOT}/runs/${RUN_ID}/${FAMILY_NAME}_${NAME}"
      mkdir -p "${OUT_DIR}"
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_prototype_reject.py" \
        --feature_npz "${FEATURE_NPZ}" \
        --source_tx_ids "${SOURCE_TX_IDS}" \
        --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
        --metric "${METRIC}" \
        --confidence_weight "${CONF_W}" \
        --entropy_weight "${ENT_W}" \
        --margin_weight "${MARG_W}" \
        --threshold_policy "${TH_POLICY}" \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile "${PROXY_Q}" \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 2.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        > "${OUT_DIR}/density_reject.out" 2>&1
    done
    for row in "${KNN_POLICIES[@]}"; do
      read -r NAME REDUCE DIST K TH_POLICY SRC_Q PROXY_Q CLASS_FLAG <<<"${row}"
      OUT_DIR="${ROOT}/runs/${RUN_ID}/${FAMILY_NAME}_${NAME}"
      mkdir -p "${OUT_DIR}"
      extra_flags=(--exclude_self)
      if [[ "${CLASS_FLAG}" == "class" ]]; then
        extra_flags+=(--class_conditional_threshold)
      fi
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_knn_reject.py" \
        --feature_npz "${FEATURE_NPZ}" \
        --source_tx_ids "${SOURCE_TX_IDS}" \
        --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
        --feature_reduce "${REDUCE}" \
        --distance "${DIST}" \
        --knn_k "${K}" \
        --threshold_policy "${TH_POLICY}" \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile "${PROXY_Q}" \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 2.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        "${extra_flags[@]}" \
        > "${OUT_DIR}/density_reject.out" 2>&1
    done
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/density_reject_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_*_20260702/*/metrics.json")):
    name = metrics_path.parent.name
    if not (name.startswith("satblind15_PROTO_") or name.startswith("satblind15_KNN_") or name.startswith("satrepair_anchor7_PROTO_") or name.startswith("satrepair_anchor7_KNN_")):
        continue
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    score = data.get("score", {})
    rows.append({
        "family": "satrepair_anchor7" if name.startswith("satrepair_anchor7_") else "satblind15",
        "run_id": metrics_path.parent.parent.name,
        "policy": name,
        "phase": data.get("phase"),
        "score_metric": score.get("metric") or score.get("distance"),
        "threshold_policy": threshold.get("threshold_policy"),
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
    })
fields = list(rows[0].keys()) if rows else ["family", "run_id", "policy"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-DENSITY-REJECT-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/density_reject_sweep_summary.csv"
