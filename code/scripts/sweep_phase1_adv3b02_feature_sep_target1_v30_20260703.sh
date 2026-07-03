#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_feature_sep_target1_v30_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
EVAL_SOURCE_TX_IDS="${EVAL_SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
PROXY_UNKNOWN_RXS="${PROXY_UNKNOWN_RXS:-1-1,1-19,14-7,18-2,19-2,2-1}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-9-1,8-3,8-18,8-13,8-1,7-11,7-10,6-6,6-1,5-5,4-11,4-1,3-8,3-18,3-13,20-8}"
SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-4070392}"

mkdir -p "${LOG_ROOT}"
if [[ ! -f "${SOURCE_CLEAN_NPZ}" ]]; then
  echo "[V30-MISSING] ${SOURCE_CLEAN_NPZ}" >&2
  exit 2
fi

declare -a CELLS=(
  "phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702:20-1:10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702:20-1:1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702:3-19:10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702:3-19:1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702:7-14:10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702:7-14:1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702:7-7:10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702:7-7:1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702:8-8:10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702:8-8:1-16,4-10"
)
CELLS_ARG="$(IFS=';'; echo "${CELLS[*]}")"

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
declare -a VARIANTS=(
  "LEOFEAT30_HEAD_PROTO_USEP id_feature_head 75 0.00020 1.00 2.00 12.00 3.00 4.00 0.005 0.20 0.20 0.15 0.00"
  "LEOFEAT30_HEAD_USEP_STRONG id_feature_head 75 0.00018 1.00 2.00 14.00 3.00 4.50 0.005 0.20 0.45 0.12 0.00"
  "LEOFEAT30_HEAD_DISTILL_USEP id_feature_head 70 0.00018 1.00 2.00 14.00 2.50 4.50 0.004 0.18 0.25 0.15 0.30"
  "LEOFEAT30_HEAD_PROTO_FLOOR id_feature_head 80 0.00016 0.90 1.80 15.00 4.00 5.00 0.003 0.35 0.15 0.18 0.00"
  "LEOFEAT30_LATE_PROTO_USEP id_late_feature 70 0.00014 0.90 1.80 14.00 3.00 5.00 0.005 0.20 0.20 0.15 0.00"
  "LEOFEAT30_LATE_USEP_STRONG id_late_feature 70 0.00012 0.90 1.80 16.00 3.50 5.50 0.004 0.25 0.35 0.12 0.00"
  "LEOFEAT30_NORM_SAFE_USEP id_norm_late_feature 60 0.00009 0.70 1.40 18.00 2.00 6.00 0.003 0.18 0.10 0.18 0.30"
  "LEOFEAT30_NORM_PROTO_USEP id_norm_late_feature 65 0.00010 0.80 1.60 16.00 3.00 5.50 0.004 0.25 0.20 0.15 0.20"
)

if (( ${#GPUS[@]} < ${#VARIANTS[@]} )); then
  echo "[V30-GPU-COUNT] need ${#VARIANTS[@]} GPUs but got ${#GPUS[@]} from ${GPUS_CSV}" >&2
  exit 2
fi

run_variant() {
  local idx="$1"
  local row="${VARIANTS[$idx]}"
  local gpu="${GPUS[$idx]}"
  read -r NAME MODE EPOCHS LR MSE_W COS_W CLEAN_W FEAT_MARGIN_W CLEAN_MARGIN_W MARGIN_TOL PROTO_CE_W PROXY_SEP_W PROXY_MAX_COS DISTILL_W <<<"${row}"
  echo "[V30-VARIANT] name=${NAME} mode=${MODE} gpu=${gpu} start=$(date -Is)"
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/train_apply_phase1_iq_preadapter_20260703.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --new_wisig_pkl "${NEW_WISIG_PKL}" \
    --runs_root "${RUNS_ROOT}" \
    --out_subdir "${NAME}" \
    --out_name "features_leo_repaired.npz" \
    --clean_out_name "features_clean_repaired.npz" \
    --identity_subdir "LEOFEAT30_IDENTITY" \
    --export_clean_control \
    --export_identity \
    --cells "${CELLS_ARG}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
    --source_rxs "${CEN51_TRAIN_RXS}" \
    --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}" \
    --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}" \
    --sat_scenarios "${SAT_SCENARIOS}" \
    --star_ground_channel_impl simplified_leo_residual \
    --no-input_adapter_enabled \
    --model_adapter_mode "${MODE}" \
    --input_repair raw \
    --clean_input_repair_mode raw \
    --epochs "${EPOCHS}" \
    --hidden_dim 32 \
    --alpha 0.00 \
    --lr "${LR}" \
    --mse_weight "${MSE_W}" \
    --cos_weight "${COS_W}" \
    --proto_ce_weight "${PROTO_CE_W}" \
    --logit_ce_weight 0.0 \
    --clean_identity_weight "${CLEAN_W}" \
    --feature_margin_weight "${FEAT_MARGIN_W}" \
    --clean_feature_margin_weight "${CLEAN_MARGIN_W}" \
    --feature_margin_tolerance "${MARGIN_TOL}" \
    --proxy_unknown_separation_weight "${PROXY_SEP_W}" \
    --proxy_unknown_max_cos "${PROXY_MAX_COS}" \
    --max_proxy_unknown_train_samples_per_tx 600 \
    --teacher_logit_distill_weight "${DISTILL_W}" \
    --distill_temperature 2.0 \
    --residual_weight 0.0 \
    --batch_size 384 \
    --max_source_samples_per_tx 1000 \
    --max_export_samples_per_tx 0 \
    --device cuda:0 \
    --seed "${SEED}" \
    > "${LOG_ROOT}/${NAME}_train_export.out" 2>&1
  echo "[V30-VARIANT-DONE] name=${NAME} mode=${MODE} gpu=${gpu} end=$(date -Is)"
}

echo "[PHASE1-FEATSEP-TARGET1-V30] start=$(date -Is) gpus=${GPUS_CSV} seed=${SEED}"
pids=()
for idx in "${!VARIANTS[@]}"; do
  run_variant "${idx}" > "${LOG_ROOT}/variant_${idx}.driver.out" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[V30-FAILED] one or more variants failed" >&2
  exit 3
fi

VARIANT_NAMES="$(printf '%s\n' "${VARIANTS[@]}" | awk '{print $1}' | paste -sd, -)"
SUMMARY_CSV="${LOG_ROOT}/target1_strong_v30_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v30_metrics.json"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${SOURCE_CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --run_glob "phase1_adv3b02_multiview_keepold_*_20260702" \
  --source_tx_ids "${EVAL_SOURCE_TX_IDS}" \
  --clean_relpath "LEOFEAT30_IDENTITY/features_clean_repaired.npz" \
  --identity_relpath "LEOFEAT30_IDENTITY/features_leo_repaired.npz" \
  --feature_relpath "{variant}/features_leo_repaired.npz" \
  --clean_repaired_relpath "{variant}/features_clean_repaired.npz" \
  --variants "${VARIANT_NAMES}" \
  > "${LOG_ROOT}/target1_strong_v30_eval.out" 2>&1

"${PYTHON}" - "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v30_best.json" <<'PY'
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
            "clean_repaired_drop_pp",
            "target_unknown_far_source05",
            "target_unknown_far_delta_vs_identity",
            "target_unknown_oldness_delta_vs_identity",
            "target_old_margin_delta_vs_identity",
            "target_old_true_dist_delta_vs_identity",
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        rows.append(row)

candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]

def truth(row, key):
    return str(row.get(key, "")).lower() == "true"

best_by_run = {}
for row in candidates:
    gate_count = sum(truth(row, k) for k in [
        "passes_old_recovery_gate",
        "passes_clean_fidelity_gate",
        "passes_floor_gate",
        "passes_margin_gate",
        "passes_unknown_safety_gate",
    ])
    score = (
        gate_count,
        row["target_old_closed_acc"],
        row["target_old_min_tx_acc"],
        -row["clean_repaired_drop_pp"],
        -row["target_unknown_far_source05"],
    )
    key = row["run_id"]
    if key not in best_by_run or score > best_by_run[key][0]:
        best_by_run[key] = (score, row)

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": sum(truth(r, "passes_strong_target1") for r in candidates),
    "gate_counts": {
        key: sum(truth(r, key) for r in candidates)
        for key in [
            "passes_old_recovery_gate",
            "passes_clean_fidelity_gate",
            "passes_floor_gate",
            "passes_margin_gate",
            "passes_unknown_safety_gate",
            "passes_strong_target1",
        ]
    },
    "best_by_run": [item[1] for item in sorted(best_by_run.values(), key=lambda x: x[1]["run_id"])],
    "best_by_target_old_acc": sorted(candidates, key=lambda r: (-r["target_old_closed_acc"], r["clean_repaired_drop_pp"], r["target_unknown_far_source05"]))[:20],
    "best_unknown_safe": sorted(candidates, key=lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_delta_pp_vs_identity"]))[:20],
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-FEATSEP-TARGET1-V30-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
