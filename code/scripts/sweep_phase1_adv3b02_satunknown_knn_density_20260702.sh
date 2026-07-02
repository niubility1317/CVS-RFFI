#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satunknown_knn_density_matrix_20260702}"
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
  "SATKNN_COS_K1_SRC9999 cosine 1 source_accept 0.9999 0.05 global all cleanproxy"
  "SATKNN_COS_K5_SRC9999 cosine 5 source_accept 0.9999 0.05 global all cleanproxy"
  "SATKNN_COS_K10_SRC9999 cosine 10 source_accept 0.9999 0.05 global all cleanproxy"
  "SATKNN_EUC_K5_SRC9999 euclidean 5 source_accept 0.9999 0.05 global all cleanproxy"
  "SATKNN_COS_K5_PROXY05 cosine 5 proxy_far 1.0000 0.05 global all cleanproxy"
  "SATKNN_COS_K5_MIN05 cosine 5 min_source_proxy 1.0000 0.05 global all cleanproxy"
  "SATKNN_COS_K10_PROXY05 cosine 10 proxy_far 1.0000 0.05 global all cleanproxy"
  "SATKNN_EUC_K5_PROXY05 euclidean 5 proxy_far 1.0000 0.05 global all cleanproxy"
  "SATKNN_COS_K5_CLS_PROXY05 cosine 5 proxy_far 1.0000 0.05 class all cleanproxy"
  "SATKNN_COS_K5_CLS_MIN05 cosine 5 min_source_proxy 1.0000 0.05 class all cleanproxy"
  "SATKNN_COS_K5_COR_PROXY05 cosine 5 proxy_far 1.0000 0.05 global correct cleanproxy"
  "SATKNN_COS_K5_COR_SRC9999 cosine 5 source_accept 0.9999 0.05 global correct cleanproxy"
  "SATKNN_COS_K5_SRCINC_PROXY05 cosine 5 proxy_far 1.0000 0.05 global all srcincproxy"
  "SATKNN_COS_K5_SRCINC_MIN05 cosine 5 min_source_proxy 1.0000 0.05 global all srcincproxy"
)

echo "[PHASE1-SATUNKNOWN-KNN-DENSITY-SWEEP] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  FEATURE_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz"
  test -f "${FEATURE_NPZ}"
  mkdir -p "${LOG_ROOT}"
  echo "[PHASE1-SATUNKNOWN-KNN-DENSITY-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"

  for policy_row in "${POLICIES[@]}"; do
    read -r NAME DISTANCE K TH_POLICY SRC_Q PROXY_Q SCOPE TRAIN_SET PROXY_SET <<<"${policy_row}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    mkdir -p "${OUT_DIR}"
    extra_flags=()
    if [[ "${SCOPE}" == "class" ]]; then
      extra_flags+=(--class_conditional_threshold)
    fi
    if [[ "${TRAIN_SET}" == "correct" ]]; then
      extra_flags+=(--train_known_correct_only)
    fi
    if [[ "${PROXY_SET}" == "srcincproxy" ]]; then
      extra_flags+=(--source_incorrect_as_proxy)
    fi
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_knn_reject.py" \
      --feature_npz "${FEATURE_NPZ}" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
      --train_known_roles source \
      --proxy_unknown_roles proxy_unknown \
      --known_query_roles target_old \
      --unknown_query_roles target_unknown \
      --feature_reduce mean \
      --distance "${DISTANCE}" \
      --knn_k "${K}" \
      --exclude_self \
      --threshold_policy "${TH_POLICY}" \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile "${PROXY_Q}" \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      "${extra_flags[@]}" \
      > "${LOG_ROOT}/${NAME}_satunknown_knn_density.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satunknown_knn_density_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/SATKNN_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    scoring = data.get("scoring", {})
    manifest = data.get("manifest", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "policy": metrics_path.parent.name,
        "distance": scoring.get("distance", ""),
        "knn_k": scoring.get("knn_k", ""),
        "threshold_policy": threshold.get("threshold_policy", ""),
        "threshold_scope": threshold.get("threshold_scope", ""),
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
        "proxy_far_quantile": threshold.get("proxy_far_quantile"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "proxy_false_accept_rate_at_threshold": threshold.get("proxy_false_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
        "train_known_correct_only": data.get("train_known_correct_only"),
        "source_incorrect_as_proxy": data.get("source_incorrect_as_proxy"),
        "contains_clean_view": manifest.get("contains_clean_view"),
        "per_sample_observation_count": manifest.get("per_sample_observation_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATUNKNOWN-KNN-DENSITY-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satunknown_knn_density_sweep_summary.csv"
