#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826.json}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-1}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-60}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ -f "${MATRIX}" && -f "${WORKER}" ]] || { echo "[QB3-MATRIX-ERROR] matrix/worker missing" >&2; exit 2; }

RUN_ID="${RUN_ID:-$("${CONTROL_PYTHON}" -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["run_id"])' "${MATRIX}")}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
SCHEDULE_ROW="$("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
s=d["muse_schedule"]
values=[int(s[k]) for k in ("s2a_start","s2b_start","s3a_start","s3b_start","s3c_start")]
assert 1 < values[0] < values[1] < values[2] < values[3] < values[4] <= int(d["epochs"])
print("\t".join(map(str, values)))
' "${MATRIX}")"
IFS=$'\t' read -r MUSE_S2A_START MUSE_S2B_START MUSE_S3A_START MUSE_S3B_START MUSE_S3C_START <<<"${SCHEDULE_ROW}"

mapfile -t ROWS < <("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["schema"]=="cvs.phase1.fasttrust_qb3_matrix.v2"
assert int(d["epochs"])>0 and int(d["unlabeled_batch_size"])==256
assert d["identity_domain_objective_mode"]=="bounded_confusion"
assert d["partial_threshold_scope"]=="global" and d["decouple_partial_negative_aps"] is True
assert d["source_roles"]=={"L_s":0.07,"U_s":0.63,"V_cal":0.15,"V_select":0.15}
h=d["source_val_heavy_eval"]
rows=d["rows"]
assert rows and len({r["candidate"] for r in rows})==len(rows)
assert len({int(r["gpu"]) for r in rows})==len(rows)
for r in rows:
 assert int(r["seed"])>0 and int(r["eval_batch_size"])>0
 assert int(r["recovery_checkpoint_interval"])>0
 assert r["negative"] is False
 print("\t".join([
  str(r["gpu"]),str(r["seed"]),r["candidate"],str(d["epochs"]),
  str(r["eval_batch_size"]),str(r["recovery_checkpoint_interval"]),
  *[str(bool(r[k])).lower() for k in ("hard","partial","partial_set","partial_conditional")],
  str(r["feature_anchor"]),str(d["rc4_lambda_hard"]),
  str(d["hard_effective_budget"]),str(d["partial_effective_budget"]),
  str(d["class_receiver_effective_budget"]),str(d["rc4_lambda_zdom"]),
  str(d["rc4_lambda_discriminator"]),str(d["rc4_lambda_confusion"]),
  str(d["rc4_lambda_partial_set"]),str(d["rc4_lambda_partial_conditional"]),
  str(d["identity_domain_discriminator_scale"]),str(d["identity_domain_confusion_scale"]),
  str(h["start_epoch"]),str(h["interval"]),str(h["final_window"]),str(h["final_interval"]),
  d["base_checkpoint"]
 ]))
' "${MATRIX}")

run_row() {
  local gpu="$1" seed="$2" candidate="$3" epochs="$4" eval_batch="$5" recovery_interval="$6"
  local hard="$7" partial="$8" partial_set="$9" partial_conditional="${10}" feature_anchor="${11}"
  local hard_lambda="${12}" hard_budget="${13}" partial_budget="${14}" cell_budget="${15}" zdom="${16}"
  local discriminator="${17}" confusion="${18}" pset_lambda="${19}" pcond_lambda="${20}"
  local labeled_discriminator="${21}" labeled_confusion="${22}" heavy_start="${23}"
  local heavy_interval="${24}" heavy_window="${25}" heavy_final_interval="${26}" checkpoint="${27}"
  checkpoint="${checkpoint%$'\r'}"
  local extra=()
  [[ "${DRY_RUN}" == 1 ]] && extra+=(--dry-run)
  env ROOT="${ROOT}" CODE_ROOT="${CODE_ROOT}" PYTHON="${PYTHON}" CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" GPU="${gpu}" SEED="${seed}" INIT_MODE=adv3b02_core90 \
    TOTAL_EPOCHS="${epochs}" LABEL_EPOCHS=130 PSEUDO_EPOCHS=70 \
    MUSE_S2A_START="${MUSE_S2A_START}" MUSE_S2B_START="${MUSE_S2B_START}" \
    MUSE_S3A_START="${MUSE_S3A_START}" MUSE_S3B_START="${MUSE_S3B_START}" \
    MUSE_S3C_START="${MUSE_S3C_START}" RC4_IDENTITY_START=11 \
    RC4_CONSOLIDATION_START="${MUSE_S3C_START}" RC4_CALIBRATION_EPOCHS=1,41,91,161 \
    RC4_TAIL_TRANSITION_START=91 RC4_TAIL_TRANSITION_EPOCHS=20 RC4_TAIL_TRANSITION_FLOOR=0.25 \
    BASE_CKPT="${ROOT}/${checkpoint}" CANDIDATE_ID_OVERRIDE="${candidate}" ABLATION=NONE \
    MUSE_UNLABELED_BATCH_SIZE=256 FASTTRUST_RC4=true SAT_ANCHOR_SSL=false \
    RC4_USE_ANCHOR=true RC4_USE_CALIBRATION=true RC4_ENABLE_HARD="${hard}" \
    RC4_ENABLE_PARTIAL="${partial}" RC4_ENABLE_PARTIAL_SET="${partial_set}" \
    RC4_ENABLE_PARTIAL_CONDITIONAL="${partial_conditional}" RC4_ENABLE_NEGATIVE=false \
    RC4_DECOUPLE_PARTIAL_NEGATIVE_APS=true RC4_PARTIAL_THRESHOLD_SCOPE=global \
    RC4_LAMBDA_HARD="${hard_lambda}" RC4_HARD_EFFECTIVE_BUDGET="${hard_budget}" \
    RC4_PARTIAL_EFFECTIVE_BUDGET="${partial_budget}" \
    RC4_NEGATIVE_EFFECTIVE_BUDGET=0 RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET=0 \
    RC4_USE_CALIBRATED_PARTIAL_THRESHOLD=true RC4_CLASS_RX_CAP=true \
    RC4_CLASS_RX_EFFECTIVE_BUDGET="${cell_budget}" RC4_SATELLITE=false \
    RC4_LAMBDA_DOMAIN=0 RC4_LAMBDA_ZDOM="${zdom}" RC4_LAMBDA_DISCRIMINATOR="${discriminator}" \
    RC4_LAMBDA_CONFUSION="${confusion}" RC4_LAMBDA_PARTIAL_SET="${pset_lambda}" \
    RC4_LAMBDA_PARTIAL_CONDITIONAL="${pcond_lambda}" RC4_LAMBDA_FEATURE_ANCHOR="${feature_anchor}" \
    RC4_IDENTITY_TAIL_HARD_FINAL=0.60 RC4_IDENTITY_TAIL_PARTIAL_SET_FINAL=0.20 \
    RC4_IDENTITY_TAIL_PARTIAL_CONDITIONAL_FINAL=0 RC4_GRADIENT_TELEMETRY_EPOCHS=1,41,91,161,181,200 \
    RC4_RECOVERY_CHECKPOINT_INTERVAL="${recovery_interval}" IDENTITY_DOMAIN_OBJECTIVE_MODE=bounded_confusion \
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
  [[ -n "${uuid}" ]] || { echo "[QB3-MATRIX-ERROR] cannot resolve GPU${gpu} UUID" >&2; return 2; }
  while true; do
    count="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -v uuid="${uuid}" '$1==uuid {n++} END{print n+0}')"
    if (( count < RESOURCE_SLOT_LIMIT )); then
      echo "[QB3-MATRIX-SLOT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} action=launch"
      return 0
    fi
    echo "[QB3-MATRIX-SLOT-WAIT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} poll=${RESOURCE_POLL_SECONDS}s"
    sleep "${RESOURCE_POLL_SECONDS}"
  done
}

echo "[QB3-MATRIX-RUN] run_id=${RUN_ID} rows=${#ROWS[@]} dry_run=${DRY_RUN} U=256"
if [[ "${DRY_RUN}" == 1 ]]; then
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r gpu seed candidate epochs eval_batch recovery_interval hard partial partial_set partial_conditional feature_anchor hard_lambda hard_budget partial_budget cell_budget zdom discriminator confusion pset_lambda pcond_lambda labeled_discriminator labeled_confusion heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
    echo "[QB3-MATRIX-ROW] gpu=${gpu} seed=${seed} candidate=${candidate} epochs=${epochs} eval_batch=${eval_batch} recovery=${recovery_interval}"
    run_row "$gpu" "$seed" "$candidate" "$epochs" "$eval_batch" "$recovery_interval" "$hard" "$partial" "$partial_set" "$partial_conditional" "$feature_anchor" "$hard_lambda" "$hard_budget" "$partial_budget" "$cell_budget" "$zdom" "$discriminator" "$confusion" "$pset_lambda" "$pcond_lambda" "$labeled_discriminator" "$labeled_confusion" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint"
  done
  exit 0
fi
[[ ! -e "${RUNS_ROOT}" ]] || { echo "[QB3-MATRIX-ERROR] refusing overwrite: ${RUNS_ROOT}" >&2; exit 3; }
mkdir -p "${RUNS_ROOT}/dispatcher_logs"
pids=(); names=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r gpu seed candidate epochs eval_batch recovery_interval hard partial partial_set partial_conditional feature_anchor hard_lambda hard_budget partial_budget cell_budget zdom discriminator confusion pset_lambda pcond_lambda labeled_discriminator labeled_confusion heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
  (wait_for_gpu_slot "$gpu" && run_row "$gpu" "$seed" "$candidate" "$epochs" "$eval_batch" "$recovery_interval" "$hard" "$partial" "$partial_set" "$partial_conditional" "$feature_anchor" "$hard_lambda" "$hard_budget" "$partial_budget" "$cell_budget" "$zdom" "$discriminator" "$confusion" "$pset_lambda" "$pcond_lambda" "$labeled_discriminator" "$labeled_confusion" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint") >"${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  pids+=("$!"); names+=("${candidate}")
done
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[QB3-MATRIX-WORKER-FAILED] ${names[$i]}" >&2; failed=1; }
done
[[ "${failed}" == 0 ]] || exit 4
echo "[QB3-MATRIX-COMPLETE] ${RUN_ID}"
