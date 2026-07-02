#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satblind15_matrix_20260702}"
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
GPU="${GPU:-0}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_satblind15_rx20_1_u10_20260702 20-1 10-1,10-10"
  "phase1_adv3b02_satblind15_rx20_1_u1_20260702 20-1 1-16,4-10"
  "phase1_adv3b02_satblind15_rx3_19_u10_20260702 3-19 10-1,10-10"
  "phase1_adv3b02_satblind15_rx3_19_u1_20260702 3-19 1-16,4-10"
  "phase1_adv3b02_satblind15_rx7_14_u10_20260702 7-14 10-1,10-10"
  "phase1_adv3b02_satblind15_rx7_14_u1_20260702 7-14 1-16,4-10"
  "phase1_adv3b02_satblind15_rx7_7_u10_20260702 7-7 10-1,10-10"
  "phase1_adv3b02_satblind15_rx7_7_u1_20260702 7-7 1-16,4-10"
  "phase1_adv3b02_satblind15_rx8_8_u10_20260702 8-8 10-1,10-10"
  "phase1_adv3b02_satblind15_rx8_8_u1_20260702 8-8 1-16,4-10"
)

declare -a POLICIES=(
  "SATBLIND15_LIN_SRC9999 linear bce 64 500 0.020 0.0001 1.0 source_accept 0.9999 0.05 all"
  "SATBLIND15_LIN_SRC999 linear bce 64 500 0.020 0.0001 1.0 source_accept 0.9990 0.05 all"
  "SATBLIND15_LIN_PROXY05 linear bce 64 500 0.020 0.0001 1.0 proxy_far 1.0000 0.05 all"
  "SATBLIND15_LIN_MIN05 linear bce 64 500 0.020 0.0001 1.0 min_source_proxy 1.0000 0.05 all"
  "SATBLIND15_MLP64_SRC9999 mlp bce 64 350 0.005 0.0001 1.0 source_accept 0.9999 0.05 all"
  "SATBLIND15_MLP64_PROXY05 mlp bce 64 350 0.005 0.0001 1.0 proxy_far 1.0000 0.05 all"
  "SATBLIND15_MLP64_MIN05 mlp bce 64 350 0.005 0.0001 1.0 min_source_proxy 1.0000 0.05 all"
  "SATBLIND15_MLP_M20_SRC9999 mlp margin 64 350 0.003 0.0001 20.0 source_accept 0.9999 0.05 all"
  "SATBLIND15_MLP_M50_SRC9999 mlp margin 64 350 0.003 0.0001 50.0 source_accept 0.9999 0.05 all"
  "SATBLIND15_MLP_M20_PROXY05 mlp margin 64 350 0.003 0.0001 20.0 proxy_far 1.0000 0.05 all"
  "SATBLIND15_MLP_M50_PROXY05 mlp margin 64 350 0.003 0.0001 50.0 proxy_far 1.0000 0.05 all"
  "SATBLIND15_MLP_M50_PROXY02 mlp margin 64 350 0.003 0.0001 50.0 proxy_far 1.0000 0.02 all"
  "SATBLIND15_MLP_M50_COR_SRC9999 mlp margin 64 350 0.003 0.0001 50.0 source_accept 0.9999 0.05 correct"
  "SATBLIND15_MLP_M50_COR_PROXY05 mlp margin 64 350 0.003 0.0001 50.0 proxy_far 1.0000 0.05 correct"
)

echo "[PHASE1-SATBLIND15-SWEEP] start=$(date -Is) gpu=${GPU}"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID TARGET_RECEIVER_IDS UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  FEATURE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATBLIND15"
  FEATURE_NPZ="${FEATURE_DIR}/features_satblind15.npz"
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${FEATURE_DIR}"
  echo "[PHASE1-SATBLIND15-CELL] run_id=${RUN_ID} target_rx=${TARGET_RECEIVER_IDS} unknown=${UNKNOWN_TX_IDS}"

  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}" \
    "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --new_wisig_pkl "${NEW_WISIG_PKL}" \
    --out_npz "${FEATURE_NPZ}" \
    --feature_name z_id \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --source_rxs "${CEN51_TRAIN_RXS}" \
    --source_channel_view satellite \
    --source_sat_scenarios "${SAT_SCENARIOS}" \
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
    --target_old_rxs "${TARGET_RECEIVER_IDS}" \
    --target_old_channel_view satellite \
    --target_old_sat_scenarios "${SAT_SCENARIOS}" \
    --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
    --target_new_channel_view satellite \
    --target_new_sat_scenarios "${SAT_SCENARIOS}" \
    --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}" \
    --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}" \
    --proxy_unknown_channel_view satellite \
    --proxy_unknown_sat_scenarios "${SAT_SCENARIOS}" \
    --satellite_tta_policy sat_rx_blind15 \
    --star_ground_channel_impl simplified_leo_residual \
    --wisig_equalized 1 \
    --wisig_domain rx_day \
    --wisig_out_len 256 \
    --max_samples_per_combo 0 \
    --max_samples_per_tx 200 \
    --batch_size 512 \
    --device "cuda:0" \
    --seed 4070237 \
    > "${LOG_ROOT}/export_satblind15.out" 2>&1

  for policy_row in "${POLICIES[@]}"; do
    read -r NAME HEAD LOSS HIDDEN EPOCHS LR L2 SRC_W TH_POLICY SRC_Q PROXY_Q TRAIN_SET <<<"${policy_row}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    mkdir -p "${OUT_DIR}"
    extra_flags=()
    if [[ "${TRAIN_SET}" == "correct" ]]; then
      extra_flags+=(--train_known_correct_only)
    fi
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
      --head_type "${HEAD}" \
      --loss_mode "${LOSS}" \
      --source_loss_weight "${SRC_W}" \
      --proxy_loss_weight 1.0 \
      --margin 1.0 \
      --hidden_dim "${HIDDEN}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --l2 "${L2}" \
      --seed 4070237 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      "${extra_flags[@]}" \
      > "${LOG_ROOT}/${NAME}_satblind15.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satblind15_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_satblind15_*_20260702/SATBLIND15_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    manifest = data.get("manifest", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "policy": metrics_path.parent.name,
        "head_type": training.get("head_type", ""),
        "loss_mode": training.get("loss_mode", ""),
        "source_loss_weight": training.get("source_loss_weight", ""),
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
        "source_accept_quantile": threshold.get("source_accept_quantile"),
        "proxy_far_quantile": threshold.get("proxy_far_quantile"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
        "train_known_correct_only": data.get("train_known_correct_only"),
        "satellite_tta_policy": manifest.get("satellite_tta_policy"),
        "satellite_tta_view_count": manifest.get("satellite_tta_view_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATBLIND15-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satblind15_sweep_summary.csv"
