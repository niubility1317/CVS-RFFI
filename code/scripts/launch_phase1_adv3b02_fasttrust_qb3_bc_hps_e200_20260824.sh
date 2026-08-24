#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-1}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-60}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ -f "${MATRIX}" && -f "${WORKER}" ]] || { echo "[QB3-ERROR] matrix/worker missing" >&2; exit 2; }

mapfile -t ROWS < <("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["schema"]=="cvs.phase1.fasttrust_qb3_bc_hps_e200_matrix.v1"
assert d["run_id"]=="phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1"
assert d["seed"]==392002 and d["epochs"]==200 and d["unlabeled_batch_size"]==256
assert d["identity_domain_objective_mode"]=="bounded_confusion"
assert d["partial_threshold_scope"]=="global" and d["decouple_partial_negative_aps"] is True
assert d["eval_batch_size"]==1024
h=d["source_val_heavy_eval"]
assert h=={"start_epoch":1,"interval":10,"final_window":20,"final_interval":1}
rows=d["rows"]
assert len(rows)==5 and [r["gpu"] for r in rows]==[0,1,2,3,4]
assert len({r["candidate"] for r in rows})==5
for r in rows:
 print("\t".join([
  str(r["gpu"]),r["candidate"],
  *[str(bool(r[k])).lower() for k in ("hard","partial","partial_set","partial_conditional")],
  str(r["feature_anchor"]),str(d["hard_effective_budget"]),str(d["partial_effective_budget"]),
  str(d["class_receiver_effective_budget"]),str(d["rc4_lambda_zdom"]),
  str(d["rc4_lambda_discriminator"]),str(d["rc4_lambda_confusion"]),
  str(d["rc4_lambda_partial_set"]),str(d["rc4_lambda_partial_conditional"]),
  str(d["identity_domain_discriminator_scale"]),str(d["identity_domain_confusion_scale"]),
  str(d["eval_batch_size"]),str(h["start_epoch"]),str(h["interval"]),
  str(h["final_window"]),str(h["final_interval"]),d["base_checkpoint"]
 ]))
' "${MATRIX}")

run_row() {
  local gpu="$1" candidate="$2" hard="$3" partial="$4" partial_set="$5" partial_conditional="$6"
  local feature_anchor="$7" hard_budget="$8" partial_budget="$9" cell_budget="${10}"
  local zdom="${11}" discriminator="${12}" confusion="${13}" pset_lambda="${14}" pcond_lambda="${15}"
  local labeled_discriminator="${16}" labeled_confusion="${17}" eval_batch="${18}"
  local heavy_start="${19}" heavy_interval="${20}" heavy_window="${21}" heavy_final_interval="${22}"
  local checkpoint="${23}"
  checkpoint="${checkpoint%$'\r'}"
  local extra=()
  [[ "${DRY_RUN}" == 1 ]] && extra+=(--dry-run)
  env ROOT="${ROOT}" CODE_ROOT="${CODE_ROOT}" PYTHON="${PYTHON}" CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" GPU="${gpu}" INIT_MODE=adv3b02_core90 \
    TOTAL_EPOCHS=200 LABEL_EPOCHS=130 PSEUDO_EPOCHS=70 \
    MUSE_S2A_START=17 MUSE_S2B_START=41 MUSE_S3A_START=69 MUSE_S3B_START=161 MUSE_S3C_START=181 \
    RC4_IDENTITY_START=11 RC4_CONSOLIDATION_START=181 RC4_CALIBRATION_EPOCHS=1,41,91,161 \
    RC4_TAIL_TRANSITION_START=91 RC4_TAIL_TRANSITION_EPOCHS=20 RC4_TAIL_TRANSITION_FLOOR=0.25 \
    BASE_CKPT="${ROOT}/${checkpoint}" CANDIDATE_ID_OVERRIDE="${candidate}" ABLATION=NONE \
    MUSE_UNLABELED_BATCH_SIZE=256 FASTTRUST_RC4=true SAT_ANCHOR_SSL=false \
    RC4_USE_ANCHOR=true RC4_USE_CALIBRATION=true RC4_ENABLE_HARD="${hard}" \
    RC4_ENABLE_PARTIAL="${partial}" RC4_ENABLE_PARTIAL_SET="${partial_set}" \
    RC4_ENABLE_PARTIAL_CONDITIONAL="${partial_conditional}" RC4_ENABLE_NEGATIVE=false \
    RC4_DECOUPLE_PARTIAL_NEGATIVE_APS=true RC4_PARTIAL_THRESHOLD_SCOPE=global \
    RC4_HARD_EFFECTIVE_BUDGET="${hard_budget}" RC4_PARTIAL_EFFECTIVE_BUDGET="${partial_budget}" \
    RC4_NEGATIVE_EFFECTIVE_BUDGET=0 RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET=0 \
    RC4_USE_CALIBRATED_PARTIAL_THRESHOLD=true RC4_CLASS_RX_CAP=true \
    RC4_CLASS_RX_EFFECTIVE_BUDGET="${cell_budget}" RC4_SATELLITE=false \
    RC4_LAMBDA_DOMAIN=0 RC4_LAMBDA_ZDOM="${zdom}" RC4_LAMBDA_DISCRIMINATOR="${discriminator}" \
    RC4_LAMBDA_CONFUSION="${confusion}" RC4_LAMBDA_PARTIAL_SET="${pset_lambda}" \
    RC4_LAMBDA_PARTIAL_CONDITIONAL="${pcond_lambda}" RC4_LAMBDA_FEATURE_ANCHOR="${feature_anchor}" \
    RC4_IDENTITY_TAIL_HARD_FINAL=0.60 RC4_IDENTITY_TAIL_PARTIAL_SET_FINAL=0.20 \
    RC4_IDENTITY_TAIL_PARTIAL_CONDITIONAL_FINAL=0 RC4_GRADIENT_TELEMETRY_EPOCHS=1,41,91,161,181,200 \
    RC4_RECOVERY_CHECKPOINT_INTERVAL=1 IDENTITY_DOMAIN_OBJECTIVE_MODE=bounded_confusion \
    IDENTITY_DOMAIN_DISCRIMINATOR_SCALE="${labeled_discriminator}" \
    IDENTITY_DOMAIN_CONFUSION_SCALE="${labeled_confusion}" \
    RC4_NONFINITE_GUARD_MIN_COUNT=8 RC4_NONFINITE_GUARD_FRACTION=0.05 \
    EVAL_BATCH_SIZE="${eval_batch}" SOURCE_VAL_HEAVY_EVAL_START_EPOCH="${heavy_start}" \
    SOURCE_VAL_HEAVY_EVAL_INTERVAL="${heavy_interval}" SOURCE_VAL_HEAVY_EVAL_FINAL_WINDOW="${heavy_window}" \
    SOURCE_VAL_HEAVY_EVAL_FINAL_INTERVAL="${heavy_final_interval}" \
    bash "${WORKER}" --only=M3 "${extra[@]}"
}

wait_for_gpu_slot() {
  local gpu="$1" uuid count
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F ', ' -v gpu="${gpu}" '$1==gpu {print $2}')"
  [[ -n "${uuid}" ]] || { echo "[QB3-ERROR] cannot resolve GPU${gpu} UUID" >&2; return 2; }
  while true; do
    count="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -v uuid="${uuid}" '$1==uuid {n++} END{print n+0}')"
    if (( count < RESOURCE_SLOT_LIMIT )); then
      echo "[QB3-SLOT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} action=launch"
      return 0
    fi
    echo "[QB3-SLOT-WAIT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} poll=${RESOURCE_POLL_SECONDS}s"
    sleep "${RESOURCE_POLL_SECONDS}"
  done
}

echo "[QB3-RUN] run_id=${RUN_ID} rows=${#ROWS[@]} dry_run=${DRY_RUN} epochs=200 U=256"
if [[ "${DRY_RUN}" == 1 ]]; then
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional feature_anchor hard_budget partial_budget cell_budget zdom discriminator confusion pset_lambda pcond_lambda labeled_discriminator labeled_confusion eval_batch heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
    echo "[QB3-ROW] gpu=${gpu} candidate=${candidate} H=${hard} Pset=${partial_set} Pcond=${partial_conditional} anchor=${feature_anchor}"
    run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$feature_anchor" "$hard_budget" "$partial_budget" "$cell_budget" "$zdom" "$discriminator" "$confusion" "$pset_lambda" "$pcond_lambda" "$labeled_discriminator" "$labeled_confusion" "$eval_batch" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint"
  done
  exit 0
fi
[[ ! -e "${RUNS_ROOT}" ]] || { echo "[QB3-ERROR] refusing overwrite: ${RUNS_ROOT}" >&2; exit 3; }
mkdir -p "${RUNS_ROOT}/dispatcher_logs"
pids=(); names=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional feature_anchor hard_budget partial_budget cell_budget zdom discriminator confusion pset_lambda pcond_lambda labeled_discriminator labeled_confusion eval_batch heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
  (wait_for_gpu_slot "$gpu" && run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$feature_anchor" "$hard_budget" "$partial_budget" "$cell_budget" "$zdom" "$discriminator" "$confusion" "$pset_lambda" "$pcond_lambda" "$labeled_discriminator" "$labeled_confusion" "$eval_batch" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint") >"${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  pids+=("$!"); names+=("${candidate}")
done
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[QB3-WORKER-FAILED] ${names[$i]}" >&2; failed=1; }
done
[[ "${failed}" == 0 ]] || exit 4
echo "[QB3-COMPLETE] ${RUN_ID}"
