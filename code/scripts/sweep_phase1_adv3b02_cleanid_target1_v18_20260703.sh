#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_cleanid_target1_v18_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SEED="${SEED:-4070318}"
GPU="${GPU:-0}"

mkdir -p "${LOG_ROOT}"

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TRAIN_SAT_CLEAR="${SOURCE_PAIR_DIR}/source_leo_clear_weak.npz"
TRAIN_SAT_LOW="${SOURCE_PAIR_DIR}/source_leo_low_elev_weak.npz"
TRAIN_SAT_RAIN="${SOURCE_PAIR_DIR}/source_leo_rain_weak.npz"
for required in "${CLEAN_NPZ}" "${TRAIN_SAT_CLEAR}" "${TRAIN_SAT_LOW}" "${TRAIN_SAT_RAIN}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[V18-MISSING] ${required}" >&2
    exit 2
  fi
done

declare -a CELLS=(
  "phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702"
)

declare -a ADAPTERS=(
  "LEOADAPT4_CLEANID_LINR linear_residual 96 0.35 0.00 180 0.0007 0.70 2.00 1.20 0.06 1.00"
  "LEOADAPT4_CLEANID_MLP mlp_residual 128 0.25 0.05 180 0.0005 0.55 2.00 1.50 0.08 1.50"
)

echo "[PHASE1-CLEANID-TARGET1-V18] start=$(date -Is) gpu=${GPU} seed=${SEED}"
for RUN_ID in "${CELLS[@]}"; do
  RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
  SAT_NPZ="${RUN_DIR}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz"
  CLEAN_APPLY_NPZ="${RUN_DIR}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz"
  if [[ ! -f "${SAT_NPZ}" || ! -f "${CLEAN_APPLY_NPZ}" ]]; then
    echo "[V18-SKIP] run_id=${RUN_ID} missing sat_or_clean_npz"
    continue
  fi
  echo "[V18-CELL] run_id=${RUN_ID}"
  for adapter_row in "${ADAPTERS[@]}"; do
    read -r ADAPT_NAME KIND HIDDEN ALPHA DROPOUT EPOCHS LR PAIR_W COS_W CE_W RES_W CLEAN_ID_W <<<"${adapter_row}"
    ADAPT_DIR="${RUN_DIR}/${ADAPT_NAME}"
    mkdir -p "${ADAPT_DIR}" "${LOG_ROOT}/${RUN_ID}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/fit_apply_phase1_leo_feature_adapter.py" \
      --clean_npz "${CLEAN_NPZ}" \
      --sat_npz "${SAT_NPZ}" \
      --train_sat_npz "${TRAIN_SAT_CLEAR}" \
      --train_sat_npz "${TRAIN_SAT_LOW}" \
      --train_sat_npz "${TRAIN_SAT_RAIN}" \
      --out_npz "${ADAPT_DIR}/features_leo_repaired.npz" \
      --clean_apply_npz "${CLEAN_APPLY_NPZ}" \
      --clean_out_npz "${ADAPT_DIR}/features_clean_repaired.npz" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --adapter_kind "${KIND}" \
      --hidden_dim "${HIDDEN}" \
      --alpha "${ALPHA}" \
      --dropout "${DROPOUT}" \
      --epochs "${EPOCHS}" \
      --batch_size 512 \
      --lr "${LR}" \
      --weight_decay 0.0001 \
      --pair_weight "${PAIR_W}" \
      --cos_weight "${COS_W}" \
      --proto_ce_weight "${CE_W}" \
      --residual_weight "${RES_W}" \
      --clean_identity_weight "${CLEAN_ID_W}" \
      --seed "${SEED}" \
      --device "cuda:0" \
      --output_json "${ADAPT_DIR}/adapter_metrics.json" \
      --adapter_out "${ADAPT_DIR}/adapter.pt" \
      > "${LOG_ROOT}/${RUN_ID}/${ADAPT_NAME}_fit_apply.out" 2>&1
  done
done

SUMMARY_CSV="${LOG_ROOT}/target1_strong_v18_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v18_metrics.json"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --variants "LEOADAPT4_CLEANID_LINR,LEOADAPT4_CLEANID_MLP" \
  > "${LOG_ROOT}/target1_strong_v18_eval.out" 2>&1

"${PYTHON}" - <<'PY' "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v18_best.json"
import csv
import json
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        for col in [
            "target_old_closed_acc",
            "target_old_delta_pp_vs_identity",
            "target_old_min_scenario_acc",
            "target_old_min_tx_acc",
            "clean_repaired_target_old_acc",
            "clean_repaired_drop_pp",
            "target_unknown_far_source05",
            "target_unknown_far_delta_vs_identity",
            "target_unknown_oldness_delta_vs_identity",
            "target_old_margin_delta_vs_identity",
            "target_old_true_dist_delta_vs_identity",
            "clean_target_old_acc",
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        row["passes_strong_target1"] = str(row.get("passes_strong_target1", "")).lower() == "true"
        rows.append(row)

candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]
passes = [r for r in candidates if r["passes_strong_target1"]]

def top(frame, key, n=10):
    return sorted(frame, key=key)[:n]

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": len(passes),
    "pass_rows": passes[:20],
    "best_by_target_old_acc": top(candidates, lambda r: (-r["target_old_closed_acc"], r["clean_repaired_drop_pp"], r["target_unknown_far_source05"])),
    "best_by_delta": top(candidates, lambda r: (-r["target_old_delta_pp_vs_identity"], r["clean_repaired_drop_pp"], r["target_unknown_far_delta_vs_identity"])),
    "best_unknown_safe": top(candidates, lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_delta_pp_vs_identity"])),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-CLEANID-TARGET1-V18-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
