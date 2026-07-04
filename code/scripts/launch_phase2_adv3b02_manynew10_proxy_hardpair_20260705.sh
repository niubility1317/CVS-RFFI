#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_manynew10_proxy_hardpair_20260705}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
EVAL_OLD_TX_IDS="${EVAL_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
SOURCE_RXS="${SOURCE_RXS:-0,1,2,3,4,5,6}"
TARGET_RX="${TARGET_RX:-7-14}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3}"
PROXY_UNKNOWN_RXS="${PROXY_UNKNOWN_RXS:-1-1,1-19,14-7,18-2,19-2,2-1}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2,1-8,10-11,10-17,10-4,10-7,11-1,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-4,19-6,19-7,19-8,19-9,2-12,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,4-1,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-7,8-8,9-1,9-20,9-7}"
PROXY_HARD_PAIR_IDS="${PROXY_HARD_PAIR_IDS:-15-1:20-12,20-12:15-1,4-1:7-11,7-11:4-1,1-16:8-13,8-13:1-16,6-6:8-1,8-1:6-6,4-1:7-10,7-10:4-1,1-10:8-13,8-13:1-10,15-19:9-1,9-1:15-19,1-14:6-6,6-6:1-14}"

SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
SEED="${SEED:-4070705}"
GPUS_CSV="${GPUS_CSV:-0,1}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
declare -a VARIANTS=(
  "MANYNEW10_PROXY_HP_NORM_SAFE id_norm_late_feature 60 0.00006 1.00 2.00 22.00 4.50 7.50 0.003 0.20 0.10 0.05 0.16 0.16 0.12 0.14 0.07 0.12 0.05 0.04 0.06"
  "MANYNEW10_PROXY_HP_NORM_STRONG id_norm_late_feature 60 0.00005 1.00 2.00 22.00 4.50 7.50 0.003 0.20 0.10 0.05 0.16 0.16 0.12 0.14 0.07 0.12 0.05 0.08 0.08"
)

if (( ${#GPUS[@]} < ${#VARIANTS[@]} )); then
  echo "[MANYNEW10-PROXY-HP-GPU-COUNT] need ${#VARIANTS[@]} GPUs but got ${GPUS_CSV}" >&2
  exit 2
fi

CASE_ID="PHASE2_MANYNEW10_RX${TARGET_RX//-/_}"
CELLS_ARG="${CASE_ID}:${TARGET_RX}:${TARGET_NEW_TX_IDS}"

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
echo "[MANYNEW10-PROXY-HP] run_id=${RUN_ID} dry_run=${DRY_RUN} target_rx=${TARGET_RX} gpus=${GPUS_CSV}"
echo "[MANYNEW10-PROXY-HP] target_new=${TARGET_NEW_TX_IDS}"
echo "[MANYNEW10-PROXY-HP] proxy_pool_excludes_target_new=true"
echo "[MANYNEW10-PROXY-HP] proxy_hard_pairs=${PROXY_HARD_PAIR_IDS}"
echo "[MANYNEW10-PROXY-HP] strict_success=K5 and K10 old_acc>=0.80 and every seen-new class>=0.75"

run_variant() {
  local idx="$1"
  local row="${VARIANTS[$idx]}"
  local gpu="${GPUS[$idx]}"
  read -r NAME MODE EPOCHS LR MSE_W COS_W CLEAN_W FEAT_MARGIN_W CLEAN_MARGIN_W MARGIN_TOL PROTO_CE_W PROXY_SEP_W PROXY_MAX_COS SUPCON_W DISTILL_W PROXY_PROTO_W PAIR_W PAIR_MARGIN OLD_PAIR_W OLD_MARGIN HARD_PAIR_W HARD_PAIR_MARGIN <<<"${row}"
  local log_path="${LOG_ROOT}/${NAME}_train_export.out"
  echo "[MANYNEW10-PROXY-HP-VARIANT] name=${NAME} mode=${MODE} gpu=${gpu} hard_pair_w=${HARD_PAIR_W} start=$(date -Is)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[MANYNEW10-PROXY-HP-DRY-RUN] ${NAME} -> ${log_path}"
    return 0
  fi
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/train_apply_phase1_iq_preadapter_20260703.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --new_wisig_pkl "${NEW_WISIG_PKL}" \
    --runs_root "${RUNS_ROOT}" \
    --out_subdir "${NAME}" \
    --out_name "features_leo_repaired.npz" \
    --clean_out_name "features_clean_repaired.npz" \
    --identity_subdir "MANYNEW10_IDENTITY" \
    --export_clean_control \
    --export_identity \
    --cells "${CELLS_ARG}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
    --source_rxs "${SOURCE_RXS}" \
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
    --proxy_unknown_supcon_weight "${SUPCON_W}" \
    --proxy_unknown_supcon_temperature 0.07 \
    --proxy_unknown_proto_ce_weight "${PROXY_PROTO_W}" \
    --proxy_unknown_proto_temperature 0.07 \
    --proxy_unknown_pair_margin_weight "${PAIR_W}" \
    --proxy_unknown_pair_margin "${PAIR_MARGIN}" \
    --proxy_unknown_old_margin_weight "${OLD_PAIR_W}" \
    --proxy_unknown_old_margin "${OLD_MARGIN}" \
    --proxy_unknown_hard_pair_ids "${PROXY_HARD_PAIR_IDS}" \
    --proxy_unknown_hard_pair_margin_weight "${HARD_PAIR_W}" \
    --proxy_unknown_hard_pair_margin "${HARD_PAIR_MARGIN}" \
    --max_proxy_unknown_train_samples_per_tx 500 \
    --teacher_logit_distill_weight "${DISTILL_W}" \
    --distill_temperature 2.0 \
    --residual_weight 0.0 \
    --batch_size 384 \
    --max_source_samples_per_tx 1000 \
    --max_export_samples_per_tx 80 \
    --device cuda:0 \
    --seed "${SEED}" \
    > "${log_path}" 2>&1
  echo "[MANYNEW10-PROXY-HP-VARIANT-DONE] name=${NAME} gpu=${gpu} end=$(date -Is)"
}

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
  echo "[MANYNEW10-PROXY-HP-TRAIN-FAILED] one or more variants failed" >&2
  exit 3
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[MANYNEW10-PROXY-HP-DRY-RUN-DONE]"
  exit 0
fi

run_support_metric_eval() {
  local name="$1"
  local feature_npz="${RUNS_ROOT}/${CASE_ID}/${name}/features_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${name}"
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_support_metric_qknn_probe.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_dir}/strict_n10_k10_support_metric_qknn.json" \
    --output_csv "${out_dir}/strict_n10_k10_support_metric_qknn.csv" \
    --new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_role target_old \
    --new_role target_unknown \
    --policies stable_first,scenario_diverse \
    --seed_start 421000 \
    --seed_count 120 \
    --k_old 10 \
    --k_new 10 \
    --pool_per_old 10 \
    --pool_per_new 10 \
    --query_per_old 70 \
    --query_per_new 70 \
    --transform_modes diag_fisher \
    --transform_strengths 0.5 \
    --topm_grid 4 \
    --proto_mix_grid 0.25 \
    --radius_norm_grid 0 \
    --old_bias_grid 0.001 \
    --neg_lambda_grid 0.7 \
    --neg_threshold_grid 0.75 \
    --neg_margin_grid 0.01 \
    --mutual_only_grid true \
    --scenario_aware \
    --balanced_assignment \
    > "${LOG_ROOT}/${name}_strict_k10_support_metric_qknn.out" 2>&1

  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_support_metric_qknn_probe.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_dir}/strict_n10_k5_support_metric_qknn.json" \
    --output_csv "${out_dir}/strict_n10_k5_support_metric_qknn.csv" \
    --new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_role target_old \
    --new_role target_unknown \
    --policies stable_first,scenario_diverse \
    --seed_start 421000 \
    --seed_count 120 \
    --k_old 5 \
    --k_new 5 \
    --pool_per_old 5 \
    --pool_per_new 5 \
    --query_per_old 75 \
    --query_per_new 75 \
    --transform_modes diag_whiten_fisher \
    --transform_strengths 0.1 \
    --topm_grid 5 \
    --proto_mix_grid 0.6 \
    --radius_norm_grid 0.1 \
    --old_bias_grid 0 \
    --neg_lambda_grid 0 \
    --neg_threshold_grid 0.7 \
    --neg_margin_grid 0 \
    --mutual_only_grid true \
    --scenario_aware \
    --balanced_assignment \
    > "${LOG_ROOT}/${name}_strict_k5_support_metric_qknn.out" 2>&1
}

for row in "${VARIANTS[@]}"; do
  read -r NAME _rest <<<"${row}"
  run_support_metric_eval "${NAME}"
done

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/manynew10_proxy_hardpair_summary.json" <<'PY'
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/strict_n10_k*_support_metric_qknn.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best", [])
    if not best:
        continue
    row = dict(best[0])
    row["variant"] = path.parent.name
    row["eval"] = "k10" if "_k10_" in path.name else "k5"
    row["json_path"] = str(path)
    row["strict_goal_pass"] = (
        row.get("query_old_acc", 0.0) >= 0.80
        and row.get("query_min_seen_new_class_acc", 0.0) >= 0.75
    )
    rows.append(row)
rows.sort(
    key=lambda r: (
        r.get("strict_goal_pass", False),
        r.get("eval") == "k10",
        r.get("query_min_seen_new_class_acc", 0.0),
        r.get("query_seen_new_acc", 0.0),
        r.get("query_old_acc", 0.0),
    ),
    reverse=True,
)
out_path.write_text(json.dumps({"rows": rows, "best": rows[:8]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"best": rows[:4], "out": str(out_path)}, ensure_ascii=False, indent=2))
PY

echo "[MANYNEW10-PROXY-HP-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
