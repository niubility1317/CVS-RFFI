#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_contrast_floor_probe_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-70}"
QKNN_K="${QKNN_K:-8}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

declare -a VARIANTS=("STAGE2C_NORM_SEP" "STAGE2C_HEAD_SEP")
declare -a K_SHOTS=("5" "10")
declare -a PROFILES=(
  "FLOOR_U095_W050M02_D008_R035_S2P060Q060 0.95 0.50 0.02 0.08 0.35 0.50 0.50 2 0.60 0.60"
  "FLOOR_U095_W050M02_D008_R050_S2P060Q060 0.95 0.50 0.02 0.08 0.50 0.50 0.50 2 0.60 0.60"
  "FLOOR_U095_W050M02_D008_R035_S3P070Q070 0.95 0.50 0.02 0.08 0.35 0.50 0.50 3 0.70 0.70"
  "FLOOR_U095_W050M02_D005_R050_S3P070Q070 0.95 0.50 0.02 0.05 0.50 0.50 0.50 3 0.70 0.70"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-CONTRAST-FLOOR] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-CONTRAST-FLOOR] variants=${VARIANTS[*]} k=${K_SHOTS[*]} profiles=${#PROFILES[@]}"
echo "[STAGE2C-CONTRAST-FLOOR] diagnostic_only=true unknown_query_eval_only=true"

run_one() {
  local variant="$1"
  local profile="$2"
  local gate="$3"
  local contrast_weight="$4"
  local contrast_margin="$5"
  local delta_min="$6"
  local label_scale="$7"
  local event_scale="$8"
  local component_scale="$9"
  local min_support="${10}"
  local min_pvalue="${11}"
  local min_reliability="${12}"
  local k="${13}"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${profile}"
  local out_json="${out_dir}/stage2c_qknn_${profile}_k${k}.json"
  local out_csv="${out_dir}/stage2c_qknn_${profile}_k${k}.summary.csv"
  local out_log="${LOG_ROOT}/${variant}_${profile}_k${k}.out"
  mkdir -p "${out_dir}"
  echo "[STAGE2C-CONTRAST-FLOOR-RUN] variant=${variant} profile=${profile} k=${k} gate=${gate} delta=${delta_min} scales=${label_scale}/${event_scale}/${component_scale} floors=${min_support}/${min_pvalue}/${min_reliability}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_json}" \
    --output_summary_csv "${out_csv}" \
    --k_shot "${k}" \
    --query_per_class "${QUERY_PER_CLASS}" \
    --qknn_k "${QKNN_K}" \
    --support_selection_policy stable_first \
    --collab_counts all \
    --max_event_bytes 2048 \
    --max_event_latency_ms 25 \
    --support_calibration_mode leave_one_out \
    --score_threshold_combine min \
    --support_quantile 0.01 \
    --proxy_quantile 0.99 \
    --unknown_quantile 0.90 \
    --unknown_risk_threshold "${gate}" \
    --accept_margin_threshold -0.02 \
    --consensus_gap_threshold -0.02 \
    --consensus_score_threshold -0.02 \
    --scorer_component_vote_threshold 0.75 \
    --receiver_class_reliability_policy support_calibrated \
    --class_score_threshold_enabled \
    --fusion_policy candidate_set_cvs \
    --candidate_set_min_receivers 1 \
    --candidate_set_min_top1_receivers 1 \
    --candidate_set_min_label_receiver_class_reliability 0.0 \
    --candidate_set_max_label_unknown_risk "${gate}" \
    --candidate_set_max_event_unknown_risk "${gate}" \
    --candidate_set_max_label_risk_component_agreement 1.0 \
    --candidate_set_unknown_reject_risk "${gate}" \
    --seen_new_old_contrast_weight "${contrast_weight}" \
    --seen_new_old_contrast_margin "${contrast_margin}" \
    --seen_new_contrast_gate_enabled \
    --seen_new_contrast_gate_min_delta "${delta_min}" \
    --seen_new_contrast_gate_min_receivers 1 \
    --seen_new_contrast_risk_relief_enabled \
    --seen_new_contrast_risk_relief_min_delta "${delta_min}" \
    --seen_new_contrast_risk_relief_min_receivers 1 \
    --seen_new_contrast_risk_relief_min_support_count "${min_support}" \
    --seen_new_contrast_risk_relief_min_pvalue "${min_pvalue}" \
    --seen_new_contrast_risk_relief_min_receiver_class_reliability "${min_reliability}" \
    --seen_new_contrast_label_risk_scale "${label_scale}" \
    --seen_new_contrast_event_risk_scale "${event_scale}" \
    --seen_new_contrast_component_agreement_scale "${component_scale}" \
    > "${out_log}" 2>&1
}

for variant in "${VARIANTS[@]}"; do
  for row in "${PROFILES[@]}"; do
    read -r profile gate contrast_weight contrast_margin delta_min label_scale event_scale component_scale min_support min_pvalue min_reliability <<<"${row}"
    for k in "${K_SHOTS[@]}"; do
      run_one "${variant}" "${profile}" "${gate}" "${contrast_weight}" "${contrast_margin}" "${delta_min}" "${label_scale}" "${event_scale}" "${component_scale}" "${min_support}" "${min_pvalue}" "${min_reliability}" "${k}"
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-CONTRAST-FLOOR-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_contrast_floor_probe_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("qknn_metadata", {})
    for row in payload.get("summary_rows", []):
        item = dict(row)
        item["variant"] = path.parent.parent.name
        item["profile"] = path.parent.name
        item["json_path"] = str(path)
        item["k_shot"] = payload.get("protocol_safety", {}).get("k_shot")
        item["threshold_scope"] = payload.get("protocol_safety", {}).get("threshold_scope")
        item["unknown_query_eval_only"] = payload.get("protocol_safety", {}).get("unknown_query_eval_only")
        item["seen_new_old_contrast_weight"] = meta.get("seen_new_old_contrast_weight", 0.0)
        item["seen_new_old_contrast_margin"] = meta.get("seen_new_old_contrast_margin", 0.0)
        item["seen_new_contrast_gate_min_delta"] = payload.get("seen_new_contrast_gate_min_delta", 0.0)
        item["seen_new_contrast_risk_relief_min_delta"] = payload.get("seen_new_contrast_risk_relief_min_delta", 0.0)
        item["seen_new_contrast_risk_relief_min_support_count"] = payload.get("seen_new_contrast_risk_relief_min_support_count", 0)
        item["seen_new_contrast_risk_relief_min_pvalue"] = payload.get("seen_new_contrast_risk_relief_min_pvalue", 0.0)
        item["seen_new_contrast_risk_relief_min_receiver_class_reliability"] = payload.get("seen_new_contrast_risk_relief_min_receiver_class_reliability", 0.0)
        item["seen_new_contrast_label_risk_scale"] = payload.get("seen_new_contrast_label_risk_scale", 1.0)
        item["seen_new_contrast_event_risk_scale"] = payload.get("seen_new_contrast_event_risk_scale", 1.0)
        item["seen_new_contrast_component_agreement_scale"] = payload.get("seen_new_contrast_component_agreement_scale", 1.0)
        item["far_feasible"] = float(item.get("unknown_FAR", 1.0)) <= 0.10
        item["utility"] = (
            2.0 * float(item.get("seen_new_acc", 0.0))
            + float(item.get("min_seen_new_class_acc", 0.0))
            + float(item.get("old_acc", 0.0))
            + float(item.get("min_old_class_acc", 0.0))
            - 2.0 * float(item.get("unknown_FAR", 1.0))
        )
        rows.append(item)
rows.sort(
    key=lambda r: (
        bool(r.get("far_feasible", False)),
        float(r.get("utility", -999.0)),
        float(r.get("old_acc", 0.0)),
        float(r.get("seen_new_acc", 0.0)),
        float(r.get("min_seen_new_class_acc", 0.0)),
        -float(r.get("unknown_FAR", 1.0)),
    ),
    reverse=True,
)
feasible = [r for r in rows if r.get("far_feasible")]
out = {"rows": rows, "best_feasible": feasible[:10], "best_utility": rows[:10]}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"best_feasible": feasible[:5], "best_utility": rows[:5], "out": str(out_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-CONTRAST-FLOOR-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
