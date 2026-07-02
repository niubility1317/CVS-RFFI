#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
VARIANT="${VARIANT:-v11a}"
GPU="${GPU:-0}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_iqpre_${VARIANT}_matrix_20260703}"
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
EPOCHS="${EPOCHS:-45}"
ALPHA="${ALPHA:-0.25}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
LR="${LR:-0.0008}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_iqpre_${VARIANT}_rx20_1_u10_20260703:20-1:10-1,10-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx20_1_u1_20260703:20-1:1-16,4-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx3_19_u10_20260703:3-19:10-1,10-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx3_19_u1_20260703:3-19:1-16,4-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx7_14_u10_20260703:7-14:10-1,10-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx7_14_u1_20260703:7-14:1-16,4-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx7_7_u10_20260703:7-7:10-1,10-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx7_7_u1_20260703:7-7:1-16,4-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx8_8_u10_20260703:8-8:10-1,10-10"
  "phase1_adv3b02_iqpre_${VARIANT}_rx8_8_u1_20260703:8-8:1-16,4-10"
)

CELLS_ARG="$(IFS=';'; echo "${CELLS[*]}")"

echo "[PHASE1-IQPRE-V11] start=$(date -Is) variant=${VARIANT} gpu=${GPU}"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/train_apply_phase1_iq_preadapter_20260703.py" \
  --ckpt "${TEACHER_CKPT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --new_wisig_pkl "${NEW_WISIG_PKL}" \
  --runs_root "${ROOT}/runs" \
  --cells "${CELLS_ARG}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
  --source_rxs "${CEN51_TRAIN_RXS}" \
  --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}" \
  --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}" \
  --sat_scenarios "${SAT_SCENARIOS}" \
  --star_ground_channel_impl simplified_leo_residual \
  --epochs "${EPOCHS}" \
  --alpha "${ALPHA}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --lr "${LR}" \
  --batch_size 384 \
  --max_source_samples_per_tx 1000 \
  --max_export_samples_per_tx 200 \
  --device cuda:0 \
  --seed 4070391 \
  > "${MATRIX_LOG_ROOT}/train_export_${VARIANT}.out" 2>&1

declare -a POLICIES=(
  "IQPRE_LIN_SRC1000 linear bce 64 500 0.020 0.0001 source_accept 1.0000 0.05 1.0 1.0 1.0"
  "IQPRE_LIN_SRC9999 linear bce 64 500 0.020 0.0001 source_accept 0.9999 0.05 1.0 1.0 1.0"
  "IQPRE_LIN_PROXY05 linear bce 64 500 0.020 0.0001 proxy_far 1.0000 0.05 1.0 1.0 1.0"
  "IQPRE_LIN_MIN05 linear bce 64 500 0.020 0.0001 min_source_proxy 1.0000 0.05 1.0 1.0 1.0"
  "IQPRE_MLP64_PROXY05 mlp bce 64 350 0.005 0.0001 proxy_far 1.0000 0.05 1.0 1.0 1.0"
  "IQPRE_MLP64_MIN05 mlp bce 64 350 0.005 0.0001 min_source_proxy 1.0000 0.05 1.0 1.0 1.0"
)

for cell in "${CELLS[@]}"; do
  IFS=':' read -r RUN_ID _TARGET_RX UNKNOWN_TX_IDS <<<"${cell}"
  FEATURE_NPZ="${ROOT}/runs/${RUN_ID}/ADV3B02_CORE90_SOFT_E200_PHASE1_IQPRE_V11/features_iqpre_v11.npz"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  mkdir -p "${LOG_ROOT}"
  for policy_row in "${POLICIES[@]}"; do
    read -r NAME HEAD_TYPE LOSS_MODE HIDDEN EVAL_EPOCHS EVAL_LR L2 TH_POLICY SRC_Q PROXY_Q SRC_W PROXY_W MARGIN <<<"${policy_row}"
    OUT_DIR="${ROOT}/runs/${RUN_ID}/${NAME}"
    mkdir -p "${OUT_DIR}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_multiview_reject.py" \
      --feature_npz "${FEATURE_NPZ}" \
      --source_tx_ids "${EVAL_SOURCE_TX_IDS}" \
      --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
      --train_known_roles source \
      --proxy_unknown_roles proxy_unknown \
      --known_query_roles target_old \
      --unknown_query_roles target_unknown \
      --threshold_policy "${TH_POLICY}" \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile "${PROXY_Q}" \
      --head_type "${HEAD_TYPE}" \
      --loss_mode "${LOSS_MODE}" \
      --source_loss_weight "${SRC_W}" \
      --proxy_loss_weight "${PROXY_W}" \
      --margin "${MARGIN}" \
      --hidden_dim "${HIDDEN}" \
      --epochs "${EVAL_EPOCHS}" \
      --lr "${EVAL_LR}" \
      --l2 "${L2}" \
      --seed 4070391 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      > "${LOG_ROOT}/${NAME}_iqpre_${VARIANT}.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/iqpre_${VARIANT}_sweep_summary.csv" "${VARIANT}"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
variant = sys.argv[3]
rows = []
for metrics_path in sorted(runs_root.glob(f"phase1_adv3b02_iqpre_{variant}_*_20260703/IQPRE_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    manifest = data.get("manifest", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "policy": metrics_path.parent.name,
        "head_type": training.get("head_type", ""),
        "loss_mode": training.get("loss_mode", ""),
        "threshold_policy": threshold.get("threshold_policy", ""),
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
        "target_channel_view": manifest.get("target_channel_view"),
        "channel_views": ",".join(manifest.get("channel_views", [])) if isinstance(manifest.get("channel_views"), list) else manifest.get("channel_views", ""),
        "uses_target_clean": manifest.get("uses_target_clean"),
        "uses_target_labels_for_training": manifest.get("uses_target_labels_for_training"),
        "uses_unknown_query_for_threshold": manifest.get("uses_unknown_query_for_threshold"),
    })
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    fields = list(rows[0].keys()) if rows else ["run_id", "policy"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows), "dual_pass": sum(1 for r in rows if r.get("passes_dual_target") is True)})
PY

echo "[PHASE1-IQPRE-V11-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/iqpre_${VARIANT}_sweep_summary.csv"
