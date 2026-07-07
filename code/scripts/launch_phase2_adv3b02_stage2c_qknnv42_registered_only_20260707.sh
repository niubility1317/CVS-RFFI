#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_qknnv42_registered_only_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-70}"
QKNN_K="${QKNN_K:-8}"
INCLUDE_EVENT_RESULTS="${INCLUDE_EVENT_RESULTS:-1}"
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
  "REGISTERED_BASE"
  "REGISTERED_CENTER"
  "REGISTERED_CONTRAST"
  "REGISTERED_CENTER_CONTRAST"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-QKNNV42-REGISTERED-ONLY] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-QKNNV42-REGISTERED-ONLY] variants=${VARIANTS[*]} k=${K_SHOTS[*]} profiles=${PROFILES[*]}"
echo "[STAGE2C-QKNNV42-REGISTERED-ONLY] phase2_main=true unknown_rejection_required=false no_training=true"

profile_args() {
  local profile="$1"
  PROFILE_ARGS=(
    --support_calibration_mode leave_one_out
    --score_threshold_combine min
    --support_quantile 0.01
    --proxy_quantile 0.99
    --unknown_quantile 0.92
    --unknown_risk_threshold 1.0
    --accept_margin_threshold -0.05
    --consensus_gap_threshold -0.02
    --consensus_score_threshold -0.02
    --scorer_component_vote_threshold 1.0
    --class_score_threshold_enabled
    --fusion_policy phase2_registered_only
    --candidate_set_min_receivers 1
    --candidate_set_min_top1_receivers 1
    --candidate_set_min_conformal_pvalue 0.0
    --candidate_set_min_label_receiver_class_reliability 0.0
    --max_event_bytes 2048
    --max_event_latency_ms 25
  )
  case "${profile}" in
    REGISTERED_BASE)
      ;;
    REGISTERED_CENTER)
      PROFILE_ARGS+=(--feature_adapter_policy support_center --feature_adapter_strength 0.50)
      ;;
    REGISTERED_CONTRAST)
      PROFILE_ARGS+=(--seen_new_old_contrast_weight 0.25 --seen_new_old_contrast_margin 0.0)
      ;;
    REGISTERED_CENTER_CONTRAST)
      PROFILE_ARGS+=(
        --feature_adapter_policy support_center
        --feature_adapter_strength 0.50
        --seen_new_old_contrast_weight 0.25
        --seen_new_old_contrast_margin 0.0
      )
      ;;
    *)
      echo "[ERROR] unknown profile ${profile}" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local variant="$1"
  local profile="$2"
  local k="$3"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${profile}"
  local out_json="${out_dir}/stage2c_qknn_${profile}_k${k}.json"
  local out_csv="${out_dir}/stage2c_qknn_${profile}_k${k}.summary.csv"
  local out_log="${LOG_ROOT}/${variant}_${profile}_k${k}.out"
  mkdir -p "${out_dir}"
  profile_args "${profile}"
  echo "[STAGE2C-QKNNV42-REGISTERED-ONLY-RUN] variant=${variant} profile=${profile} k=${k} feature=${feature_npz}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  EXTRA_ARGS=()
  if [[ "${INCLUDE_EVENT_RESULTS}" == "1" ]]; then
    EXTRA_ARGS+=(--include_event_results)
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
    "${PROFILE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "${out_log}" 2>&1
}

for variant in "${VARIANTS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    for k in "${K_SHOTS[@]}"; do
      run_one "${variant}" "${profile}" "${k}"
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-QKNNV42-REGISTERED-ONLY-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_qknnv42_registered_only_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    for row in payload.get("summary_rows", []):
        count_key = str(row.get("collab_count"))
        metrics = counts.get(count_key, {})
        old_acc = float(row.get("old_acc", 0.0))
        seen_new_acc = float(row.get("seen_new_acc", 0.0))
        h_old_new = 0.0 if old_acc + seen_new_acc <= 0.0 else 2.0 * old_acc * seen_new_acc / (old_acc + seen_new_acc)
        item = dict(row)
        item["variant"] = path.parent.parent.name
        item["profile"] = path.parent.name
        item["json_path"] = str(path)
        item["k_shot"] = payload.get("protocol_safety", {}).get("k_shot")
        item["threshold_scope"] = payload.get("protocol_safety", {}).get("threshold_scope")
        item["unknown_query_eval_only"] = payload.get("protocol_safety", {}).get("unknown_query_eval_only")
        item["phase2_unknown_rejection_required"] = False
        item["H_old_new"] = h_old_new
        item["phase2_old80_ready"] = old_acc >= 0.80 and float(item.get("min_old_class_acc", 0.0)) > 0.0
        item["phase2_joint_score"] = (
            2.0 * old_acc
            + float(item.get("min_old_class_acc", 0.0))
            + seen_new_acc
            + float(item.get("min_seen_new_class_acc", 0.0))
            + h_old_new
        )
        item["phase2_registered_only_accept_count"] = int(
            metrics.get("phase2_registered_only_accept_count", 0)
        )
        rows.append(item)

rows.sort(
    key=lambda r: (
        bool(r.get("phase2_old80_ready", False)),
        float(r.get("phase2_joint_score", -999.0)),
        float(r.get("old_acc", 0.0)),
        float(r.get("min_old_class_acc", 0.0)),
        float(r.get("H_old_new", 0.0)),
        float(r.get("seen_new_acc", 0.0)),
        float(r.get("min_seen_new_class_acc", 0.0)),
    ),
    reverse=True,
)
old80 = [r for r in rows if r.get("phase2_old80_ready")]
out = {
    "rows": rows,
    "best_phase2_main": rows[:12],
    "best_old80_ready": old80[:12],
    "selection_note": "Phase2排序不使用unknown拒识指标；unknown列仅为Phase3备用诊断。",
}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"best_phase2_main": rows[:5], "best_old80_ready": old80[:5], "out": str(out_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-QKNNV42-REGISTERED-ONLY-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
