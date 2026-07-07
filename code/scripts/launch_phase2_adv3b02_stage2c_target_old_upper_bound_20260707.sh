#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_target_old_upper_bound_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

declare -a VARIANTS=("STAGE2C_NORM_SEP" "STAGE2C_HEAD_SEP")
declare -a MODES=("proto" "linear" "mlp")

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-TARGET-OLD-UPPER] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-TARGET-OLD-UPPER] variants=${VARIANTS[*]} modes=${MODES[*]} k=5,10"
echo "[STAGE2C-TARGET-OLD-UPPER] diagnostic_only=true target_old_only=true target_old_tx_ids=${TARGET_OLD_TX_IDS}"

run_one() {
  local variant="$1"
  local mode="$2"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${mode}"
  mkdir -p "${out_dir}"
  echo "[STAGE2C-TARGET-OLD-UPPER-RUN] variant=${variant} mode=${mode}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  case "${mode}" in
    proto)
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_target_old_only_upper_bound.py" \
        --feature_npz "${feature_npz}" \
        --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
        --k_values 5,10 \
        --output_json "${out_dir}/target_old_proto_upper.json" \
        --summary_csv "${out_dir}/target_old_proto_upper.csv" \
        > "${LOG_ROOT}/${variant}_${mode}.out" 2>&1
      ;;
    linear)
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_target_old_linear_probe_upper_bound.py" \
        --feature_npz "${feature_npz}" \
        --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
        --k_values 5,10 \
        --ridge_lambdas 0.001,0.01,0.1,1.0,10.0 \
        --output_json "${out_dir}/target_old_linear_upper.json" \
        --summary_csv "${out_dir}/target_old_linear_upper.csv" \
        > "${LOG_ROOT}/${variant}_${mode}.out" 2>&1
      ;;
    mlp)
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_target_old_mlp_adapter_upper_bound.py" \
        --feature_npz "${feature_npz}" \
        --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
        --k_values 5,10 \
        --seeds 1,7,13 \
        --epochs 120 \
        --hidden_dim 64 \
        --lr 0.01 \
        --weight_decay 0.0001 \
        --dropout 0.0 \
        --device cpu \
        --output_json "${out_dir}/target_old_mlp_upper.json" \
        --summary_csv "${out_dir}/target_old_mlp_upper.csv" \
        > "${LOG_ROOT}/${variant}_${mode}.out" 2>&1
      ;;
    *) echo "[ERROR] unsupported mode: ${mode}" >&2; exit 2 ;;
  esac
}

for variant in "${VARIANTS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_one "${variant}" "${mode}"
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-TARGET-OLD-UPPER-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_target_old_upper_bound_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/target_old_*_upper.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant = path.parent.parent.name
    mode = path.parent.name
    for item in payload.get("results", []):
        row = {
            "variant": variant,
            "mode": mode,
            "k": item.get("k"),
            "support_count": item.get("support_count"),
            "query_count": item.get("query_count"),
            "old_acc": item.get("old_acc"),
            "macro_old_acc": item.get("macro_old_acc", item.get("old_acc")),
            "min_old_class_acc": item.get("min_old_class_acc"),
            "support_query_overlap_count": item.get("support_query_overlap_count"),
            "invalid_classes": ",".join(item.get("invalid_classes", [])),
            "ignored_non_target_old_rows": item.get("ignored_non_target_old_rows"),
            "json_path": str(path),
        }
        if mode == "linear":
            row["ridge_lambda"] = item.get("ridge_lambda")
        if mode == "mlp":
            row["seed"] = item.get("seed")
            row["train_acc"] = item.get("train_acc")
            row["train_loss_final"] = item.get("train_loss_final")
            row["valid_row"] = item.get("valid_row")
            row["invalid_reason"] = item.get("invalid_reason")
        rows.append(row)
rows.sort(
    key=lambda r: (
        float(r.get("old_acc") or -1.0),
        float(r.get("min_old_class_acc") or -1.0),
        float(r.get("macro_old_acc") or -1.0),
    ),
    reverse=True,
)
out = {
    "rows": rows,
    "best_rows": rows[:20],
    "diagnostic_scope": "target_old_only_upper_bound_non_deployment",
}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"best_rows": rows[:10], "out": str(out_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-TARGET-OLD-UPPER-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
