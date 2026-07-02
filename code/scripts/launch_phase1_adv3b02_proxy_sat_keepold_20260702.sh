#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_proxy_sat_keepold_20260702}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
EVAL_SOURCE_TX_IDS="${EVAL_SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
PROXY_UNKNOWN_RXS="${PROXY_UNKNOWN_RXS:-1-1,1-19,14-7,18-2,19-2,2-1}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-9-1,8-3,8-18,8-13,8-1,7-11,7-10,6-6,6-1,5-5,4-11,4-1,3-8,3-18,3-13,20-8}"
TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-20-1}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-10-1,10-10}"
SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

TARGET_RECEIVER_IDS="$(printf '%s' "${TARGET_RECEIVER_IDS}" | tr -d '\r')"
UNKNOWN_TX_IDS="$(printf '%s' "${UNKNOWN_TX_IDS}" | tr -d '\r')"
EVAL_SOURCE_TX_IDS="$(printf '%s' "${EVAL_SOURCE_TX_IDS}" | tr -d '\r')"
PROXY_UNKNOWN_TX_IDS="$(printf '%s' "${PROXY_UNKNOWN_TX_IDS}" | tr -d '\r')"
PROXY_UNKNOWN_RXS="$(printf '%s' "${PROXY_UNKNOWN_RXS}" | tr -d '\r')"
SAT_SCENARIOS="$(printf '%s' "${SAT_SCENARIOS}" | tr -d '\r')"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

FEATURE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_PROXY_SAT"
FEATURE_NPZ="${FEATURE_DIR}/features.npz"

echo "[PHASE1-PROXY-SAT-OSR] run_id=${RUN_ID} dry_run=${DRY_RUN} gpu=${GPU}"
echo "[PHASE1-PROXY-SAT-OSR] objective=unknown_FAR<=0.05 and old_drop_pp_vs_closed<=3.0"
echo "[PHASE1-PROXY-SAT-OSR] source/proxy/target satellite scenarios=${SAT_SCENARIOS}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${FEATURE_DIR}"
fi

EXPORT_CMD=(
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"
  --ckpt "${TEACHER_CKPT}"
  --wisig_pkl "${WISIG_PKL}"
  --new_wisig_pkl "${NEW_WISIG_PKL}"
  --out_npz "${FEATURE_NPZ}"
  --feature_name z_id
  --source_tx_ids "${SOURCE_TX_IDS}"
  --source_rxs "${CEN51_TRAIN_RXS}"
  --source_channel_view satellite
  --source_sat_scenarios "${SAT_SCENARIOS}"
  --target_old_tx_ids "${TARGET_OLD_TX_IDS}"
  --target_old_rxs "${TARGET_RECEIVER_IDS}"
  --target_old_channel_view satellite
  --target_old_sat_scenarios "${SAT_SCENARIOS}"
  --unknown_tx_ids "${UNKNOWN_TX_IDS}"
  --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}"
  --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}"
  --proxy_unknown_channel_view satellite
  --proxy_unknown_sat_scenarios "${SAT_SCENARIOS}"
  --star_ground_channel_impl simplified_leo_residual
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --max_samples_per_combo 0
  --max_samples_per_tx 200
  --batch_size 512
  --device "cuda:0"
  --seed 4070207
)

printf "[PHASE1-PROXY-SAT-EXPORT-CMD] "; printf "%q " "${EXPORT_CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${EXPORT_CMD[@]}" > "${LOG_ROOT}/feature_export.out" 2>&1
fi

declare -a HEADS=(
  "BAL_LIN_SAMPLE linear 64 500 0.020"
  "BAL_MLP64_SAMPLE mlp 64 350 0.005"
)
declare -a POLICIES=(
  "SRC999 source_accept 0.999 0.05"
  "SRC995 source_accept 0.995 0.05"
  "SRC990 source_accept 0.990 0.05"
  "PROXY05 proxy_far 0.995 0.05"
  "PROXY02 proxy_far 0.995 0.02"
  "MIN_SRC995_PROXY05 min_source_proxy 0.995 0.05"
  "MAX_SRC995_PROXY05 max_source_proxy 0.995 0.05"
)

for head_row in "${HEADS[@]}"; do
  read -r HEAD_NAME HEAD_TYPE HIDDEN_DIM EPOCHS LR <<<"${head_row}"
  for policy_row in "${POLICIES[@]}"; do
    read -r POLICY_NAME POLICY SRC_Q PROXY_Q <<<"${policy_row}"
    NAME="${HEAD_NAME}_${POLICY_NAME}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${OUT_DIR}"
    fi
    EVAL_CMD=(
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_proxy_unknown_reject.py"
      --feature_npz "${FEATURE_NPZ}"
      --source_tx_ids "${EVAL_SOURCE_TX_IDS}"
      --unknown_tx_ids "${UNKNOWN_TX_IDS}"
      --train_known_roles source
      --proxy_unknown_roles proxy_unknown
      --known_query_roles target_old
      --unknown_query_roles target_unknown
      --threshold_policy "${POLICY}"
      --source_accept_quantile "${SRC_Q}"
      --proxy_far_quantile "${PROXY_Q}"
      --head_type "${HEAD_TYPE}"
      --hidden_dim "${HIDDEN_DIM}"
      --class_balance sample
      --epochs "${EPOCHS}"
      --lr "${LR}"
      --l2 0.0001
      --unknown_far_target 0.05
      --max_old_drop_pp 3.0
      --seed 4070207
      --output_json "${OUT_DIR}/metrics.json"
      --score_table_csv "${OUT_DIR}/score_table.csv"
    )
    printf "[PHASE1-PROXY-SAT-EVAL-CMD] policy=%s " "${NAME}"; printf "%q " "${EVAL_CMD[@]}"; printf "\n"
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${EVAL_CMD[@]}" > "${LOG_ROOT}/${NAME}.out" 2>&1
    fi
  done
done

SUMMARY_CSV="${RUNS_ROOT}/summary_phase1_proxy_sat_keepold.csv"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - <<'PY' "${RUNS_ROOT}" "${SUMMARY_CSV}"
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    rows.append({
        "policy": metrics_path.parent.name,
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "head_type": training.get("head_type"),
        "class_balance": training.get("class_balance"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "proxy_false_accept_rate_at_threshold": threshold.get("proxy_false_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
        "proxy_unknown_count": data.get("proxy_unknown_count"),
    })
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["policy"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(out), "rows": len(rows)}, ensure_ascii=False))
PY
fi

echo "[PHASE1-PROXY-SAT-OSR-DONE] run_id=${RUN_ID} summary=${SUMMARY_CSV}"
