#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_classcond_reject_matrix_20260702}"
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
  "satrepair_anchor7 SATREPAIRA7 phase1_adv3b02_satrepair_anchor7"
  "satrepair9 SATREPAIR9 phase1_adv3b02_satrepair9"
)

declare -a POLICY_SUFFIXES=(
  "MLP_M50_PROXY02"
  "MLP_M50_COR_PROXY05"
  "MLP_M50_PROXY05"
  "LIN_PROXY05"
  "LIN_SRC9999"
  "MLP_M50_SRC9999"
)

declare -a THRESHOLD_POLICIES=(
  "min_class_source_proxy"
  "source_class_accept"
  "proxy_class_far"
)
declare -a SOURCE_Q=(0.999 1.000)
declare -a PROXY_Q=(0.02 0.05 0.10)
declare -a CORRECT_FLAGS=(0 1)

echo "[PHASE1-CLASSCOND-SWEEP] start=$(date -Is)"
for family in "${FAMILIES[@]}"; do
  read -r FAMILY_NAME POLICY_PREFIX RUN_PREFIX <<<"${family}"
  for cell in "${CELLS[@]}"; do
    read -r CELL_ID UNKNOWN_TX_IDS <<<"${cell}"
    RUN_ID="${RUN_PREFIX}_${CELL_ID}_20260702"
    for SUFFIX in "${POLICY_SUFFIXES[@]}"; do
      POLICY="${POLICY_PREFIX}_${SUFFIX}"
      SCORE_TABLE="${ROOT}/runs/${RUN_ID}/${POLICY}/score_table.csv"
      if [[ ! -f "${SCORE_TABLE}" ]]; then
        echo "[PHASE1-CLASSCOND-SKIP] family=${FAMILY_NAME} run_id=${RUN_ID} policy=${POLICY} missing_score_table=1"
        continue
      fi
      for TH_POLICY in "${THRESHOLD_POLICIES[@]}"; do
        for SQ in "${SOURCE_Q[@]}"; do
          for PQ in "${PROXY_Q[@]}"; do
            for CORRECT in "${CORRECT_FLAGS[@]}"; do
              SUFFIX_ID="${TH_POLICY}_sq${SQ//./p}_pq${PQ//./p}_cor${CORRECT}"
              OUT_DIR="${ROOT}/runs/${RUN_ID}/${POLICY}_CLASSCOND_${SUFFIX_ID}"
              mkdir -p "${OUT_DIR}"
              extra_flags=()
              if [[ "${CORRECT}" == "1" ]]; then
                extra_flags+=(--correct_source_only)
              fi
              env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
                "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_scoretable_classcond_reject.py" \
                --score_table_csv "${SCORE_TABLE}" \
                --source_tx_ids "${SOURCE_TX_IDS}" \
                --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
                --threshold_policy "${TH_POLICY}" \
                --source_quantile "${SQ}" \
                --proxy_quantile "${PQ}" \
                --max_source_quantile 1.0 \
                --unknown_far_target 0.05 \
                --max_old_drop_pp 2.0 \
                --output_json "${OUT_DIR}/metrics.json" \
                "${extra_flags[@]}" \
                > "${OUT_DIR}/classcond.out" 2>&1
            done
          done
        done
      done
    done
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/classcond_reject_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_satrepair*_20260702/*_CLASSCOND_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    th = data.get("threshold", {})
    policy_dir = metrics_path.parent.name
    base_policy = policy_dir.split("_CLASSCOND_")[0]
    run_id = metrics_path.parent.parent.name
    family = "satrepair_anchor7" if "satrepair_anchor7" in run_id else "satrepair9" if "satrepair9" in run_id else "unknown"
    rows.append({
        "family": family,
        "run_id": run_id,
        "base_policy": base_policy,
        "classcond_policy": policy_dir,
        "threshold_policy": th.get("threshold_policy"),
        "source_quantile": th.get("source_quantile"),
        "proxy_quantile": th.get("proxy_quantile"),
        "correct_source_only": th.get("correct_source_only"),
        "source_calibration_count": th.get("source_calibration_count"),
        "proxy_calibration_count": th.get("proxy_calibration_count"),
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
fields = list(rows[0].keys()) if rows else ["family", "run_id", "base_policy"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-CLASSCOND-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/classcond_reject_sweep_summary.csv"
