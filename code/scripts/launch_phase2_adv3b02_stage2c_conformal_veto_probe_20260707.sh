#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_conformal_veto_probe_20260707}"
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
  "CONF_VETO_M1_E090_L090_S090_C075 0.90 0.90 0.90 0.75 1"
  "CONF_VETO_M2_E090_L090_S090_C075 0.90 0.90 0.90 0.75 2"
  "CONF_VETO_M1_E085_L085_S085_C070 0.85 0.85 0.85 0.70 1"
  "CONF_VETO_M2_E085_L085_S085_C070 0.85 0.85 0.85 0.70 2"
  "CONF_VETO_M2_E080_L080_S085_C070 0.80 0.80 0.85 0.70 2"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-CONFORMAL-VETO] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-CONFORMAL-VETO] variants=${VARIANTS[*]} k=${K_SHOTS[*]} profiles=${#PROFILES[@]}"
echo "[STAGE2C-CONFORMAL-VETO] diagnostic_only=true unknown_query_eval_only=true"

run_one() {
  local variant="$1"
  local profile="$2"
  local event_veto="$3"
  local label_veto="$4"
  local shell_veto="$5"
  local component_veto="$6"
  local min_sources="$7"
  local k="$8"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${profile}"
  local out_json="${out_dir}/stage2c_qknn_${profile}_k${k}.json"
  local out_csv="${out_dir}/stage2c_qknn_${profile}_k${k}.summary.csv"
  local out_log="${LOG_ROOT}/${variant}_${profile}_k${k}.out"
  mkdir -p "${out_dir}"
  echo "[STAGE2C-CONFORMAL-VETO-RUN] variant=${variant} profile=${profile} k=${k} veto=${event_veto}/${label_veto}/${shell_veto}/${component_veto} min_sources=${min_sources}"
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
    --support_calibration_mode leave_one_out \
    --score_threshold_combine min \
    --support_quantile 0.01 \
    --proxy_quantile 0.99 \
    --unknown_quantile 0.92 \
    --unknown_risk_threshold 0.92 \
    --accept_margin_threshold -0.05 \
    --consensus_gap_threshold -0.02 \
    --consensus_score_threshold -0.02 \
    --scorer_component_vote_threshold 0.75 \
    --class_score_threshold_enabled \
    --fusion_policy scorer_cvs \
    --seen_new_rescue_enabled \
    --seen_new_rescue_risk_scale 0.25 \
    --seen_new_rescue_min_score -0.02 \
    --seen_new_rescue_min_margin -0.05 \
    --seen_new_rescue_min_agreement 0.50 \
    --conformal_rescue_enabled \
    --conformal_rescue_min_pvalue 0.0 \
    --conformal_rescue_risk_scale 0.25 \
    --conformal_rescue_min_agreement 0.50 \
    --rescue_unknown_veto_enabled \
    --rescue_unknown_veto_event_risk "${event_veto}" \
    --rescue_unknown_veto_label_risk "${label_veto}" \
    --rescue_unknown_veto_shell_risk "${shell_veto}" \
    --rescue_unknown_veto_component_agreement "${component_veto}" \
    --rescue_unknown_veto_min_sources "${min_sources}" \
    --rescue_unknown_veto_action unknown_reject \
    --old_gate_max_effective_unknown_risk 0.92 \
    --old_gate_max_component_agreement 1.0 \
    --old_gate_min_support_density 0.0 \
    --seen_new_gate_max_effective_unknown_risk 0.92 \
    --seen_new_gate_max_component_agreement 1.0 \
    --seen_new_gate_min_support_density 0.0 \
    --candidate_set_min_receivers 1 \
    --candidate_set_min_top1_receivers 1 \
    --candidate_set_max_label_unknown_risk 0.92 \
    --candidate_set_max_event_unknown_risk 0.92 \
    --candidate_set_max_label_risk_component_agreement 1.0 \
    --candidate_set_unknown_reject_risk 0.92 \
    --max_event_bytes 2048 \
    --max_event_latency_ms 25 \
    > "${out_log}" 2>&1
}

for variant in "${VARIANTS[@]}"; do
  for row in "${PROFILES[@]}"; do
    read -r profile event_veto label_veto shell_veto component_veto min_sources <<<"${row}"
    for k in "${K_SHOTS[@]}"; do
      run_one "${variant}" "${profile}" "${event_veto}" "${label_veto}" "${shell_veto}" "${component_veto}" "${min_sources}" "${k}"
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-CONFORMAL-VETO-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_conformal_veto_probe_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("summary_rows", []):
        item = dict(row)
        item["variant"] = path.parent.parent.name
        item["profile"] = path.parent.name
        item["json_path"] = str(path)
        item["k_shot"] = payload.get("protocol_safety", {}).get("k_shot")
        item["threshold_scope"] = payload.get("protocol_safety", {}).get("threshold_scope")
        item["unknown_query_eval_only"] = payload.get("protocol_safety", {}).get("unknown_query_eval_only")
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
        float(r.get("seen_new_acc", 0.0)),
        float(r.get("min_seen_new_class_acc", 0.0)),
        float(r.get("old_acc", 0.0)),
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

echo "[STAGE2C-CONFORMAL-VETO-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
