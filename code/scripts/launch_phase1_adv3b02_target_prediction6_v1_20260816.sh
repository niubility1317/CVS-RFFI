#!/usr/bin/env bash
set -euo pipefail

# Frozen six-fold ADV3B02 blind target prediction entry.  The source-only
# clean authority is used only by the train-config sealer; publisher commands
# consume exactly checkpoint, completion receipt, sealed train config and the
# existing IQ-only package.
RUN_ID="phase1_adv3b02_target_prediction_20260816_v1"
ADV_TRAINING_RUN_ID="phase1_adv3b02_clic6_20260816_v2"
CLIC_TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
TARGET_PACKAGE_RUN_ID="phase1_clic_target_prediction_20260812_v1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
EVALUATOR="${CODE_ROOT}/evaluate_phase1_adv3b02_target_leo.py"
CLEAN_BINDING_MODULE="${CODE_ROOT}/build_phase1_clic_source_v_leo_iq.py"
ADV_TRAINING_ROOT="${PROJECT_ROOT}/runs/${ADV_TRAINING_RUN_ID}"
CLIC_TRAINING_ROOT="${PROJECT_ROOT}/runs/${CLIC_TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
PACKAGE_ROOT="${PROJECT_ROOT}/runs/${TARGET_PACKAGE_RUN_ID}/sealed_target/iq_only_package"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
DRY_RUN=0

FOLD_SOURCE_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)
GPU_MAP=(0 1 2 3 4 5)

for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${argument}" >&2; exit 2 ;;
  esac
done

adv_candidate_for() {
  local fold="$1"
  printf 'F%s_ADV3B02_CLIC' "${fold}"
}

clic_candidate_for() {
  local fold="$1" arm="$2"
  printf 'F%s%s_CLIC12' "${fold}" "${arm}"
}

adv_checkpoint_for() {
  local fold="$1"
  printf '%s/%s/final_ssdg.pth' "${ADV_TRAINING_ROOT}" "$(adv_candidate_for "${fold}")"
}

adv_completion_for() {
  local fold="$1"
  printf '%s/%s/phase1_training_completion_receipt.json' \
    "${ADV_TRAINING_ROOT}" "$(adv_candidate_for "${fold}")"
}

clic_checkpoint_for() {
  local fold="$1" arm="$2"
  printf '%s/%s/final_ssdg.pth' "${CLIC_TRAINING_ROOT}" "$(clic_candidate_for "${fold}" "${arm}")"
}

clic_terminal_for() {
  local fold="$1" arm="$2"
  printf '%s/%s/phase1_clic_terminal_receipt.json' \
    "${CLIC_TRAINING_ROOT}" "$(clic_candidate_for "${fold}" "${arm}")"
}

clean_for() {
  local fold="$1" arm="$2"
  printf '%s/%s/source_clean_proxy.npz' "${CLEAN_ROOT}" "$(clic_candidate_for "${fold}" "${arm}")"
}

fold_output_root_for() {
  local fold="$1"
  printf '%s/%s' "${RUN_ROOT}" "$(adv_candidate_for "${fold}")"
}

train_config_for() {
  local fold="$1"
  printf '%s/train_data_config.json' "$(fold_output_root_for "${fold}")"
}

prediction_for() {
  local fold="$1"
  printf '%s/target_prediction.json' "$(fold_output_root_for "${fold}")"
}

seal_command() {
  local fold="$1"
  SEAL_CMD=(
    "${PYTHON}" -u "${EVALUATOR}"
    --seal-train-data-config
    --checkpoint "$(adv_checkpoint_for "${fold}")"
    --completion-receipt-json "$(adv_completion_for "${fold}")"
    --clean-v4-npz "$(clean_for "${fold}" C)"
    --output "$(train_config_for "${fold}")"
  )
}

publish_command() {
  local fold="$1"
  PUBLISH_CMD=(
    "${PYTHON}" -u "${EVALUATOR}"
    --publish-target-prediction
    --checkpoint "$(adv_checkpoint_for "${fold}")"
    --completion-receipt-json "$(adv_completion_for "${fold}")"
    --train-config-manifest "$(train_config_for "${fold}")"
    --iq-only-package "${PACKAGE_ROOT}"
    --output "$(prediction_for "${fold}")"
  )
}

emit_seal_command() {
  local fold="$1" candidate canonical_clean peer_clean
  candidate="$(adv_candidate_for "${fold}")"
  canonical_clean="$(clean_for "${fold}" C)"
  peer_clean="$(clean_for "${fold}" G)"
  seal_command "${fold}"
  printf '[DRY-RUN] stage=ADV_TRAIN_CONFIG_SEAL fold=%s candidate=%s physical_gpu=CPU retry=NO canonical_clean=%s peer_clean=%s clean_binding_proof=STRICT_C_G_METADATA_EQUAL' \
    "${fold}" "${candidate}" "${canonical_clean}" "${peer_clean}"
  printf ' %q' "${SEAL_CMD[@]}"
  printf '\n'
}

emit_publish_command() {
  local fold="$1" gpu="$2" candidate
  candidate="$(adv_candidate_for "${fold}")"
  publish_command "${fold}"
  printf '[DRY-RUN] stage=ADV_TARGET_PREDICTION fold=%s candidate=%s physical_gpu=%s retry=NO' \
    "${fold}" "${candidate}" "${gpu}"
  printf ' %q' "${PUBLISH_CMD[@]}"
  printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    emit_seal_command "${fold}"
  done
  for fold in 1 2 3 4 5 6; do
    emit_publish_command "${fold}" "${GPU_MAP[fold - 1]}"
  done
  exit 0
fi

refuse_existing_roots() {
  if [[ -e "${RUN_ROOT}" || -e "${LOG_ROOT}" ]]; then
    echo "refusing to overwrite ADV target prediction run/log root" >&2
    exit 3
  fi
}

check_inputs() {
  local fold arm path
  [[ -x "${PYTHON}" ]] || { echo "missing frozen Python runtime: ${PYTHON}" >&2; return 2; }
  [[ -f "${EVALUATOR}" && -f "${CLEAN_BINDING_MODULE}" ]] || {
    echo "missing ADV evaluator or clean metadata helper" >&2
    return 2
  }
  [[ -f "${PACKAGE_ROOT}/manifest.json" && -f "${PACKAGE_ROOT}/received_iq.npz" ]] || {
    echo "missing frozen IQ-only package" >&2
    return 2
  }
  for fold in 1 2 3 4 5 6; do
    for path in "$(adv_checkpoint_for "${fold}")" "$(adv_completion_for "${fold}")"; do
      [[ -f "${path}" ]] || { echo "missing ADV fold input: ${path}" >&2; return 2; }
    done
    for arm in C G; do
      for path in \
        "$(clic_checkpoint_for "${fold}" "${arm}")" \
        "$(clic_terminal_for "${fold}" "${arm}")" \
        "$(clean_for "${fold}" "${arm}")"; do
        [[ -f "${path}" ]] || { echo "missing clean-v4 proof input: ${path}" >&2; return 2; }
      done
    done
  done
}

verify_clean_metadata_pair() {
  local fold="$1" source_tx_ids c_clean g_clean c_checkpoint c_terminal g_checkpoint g_terminal
  source_tx_ids="${FOLD_SOURCE_TX[fold - 1]}"
  c_clean="$(clean_for "${fold}" C)"
  g_clean="$(clean_for "${fold}" G)"
  c_checkpoint="$(clic_checkpoint_for "${fold}" C)"
  c_terminal="$(clic_terminal_for "${fold}" C)"
  g_checkpoint="$(clic_checkpoint_for "${fold}" G)"
  g_terminal="$(clic_terminal_for "${fold}" G)"
  PYTHONPATH="${CODE_ROOT}" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "${PYTHON}" - \
    "${fold}" "${source_tx_ids}" \
    "${c_clean}" "${g_clean}" \
    "${c_checkpoint}" "${c_terminal}" \
    "${g_checkpoint}" "${g_terminal}" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from build_phase1_clic_source_v_leo_iq import _read_clean_validation_binding


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


fold = int(sys.argv[1])
source_tx_ids = tuple(item for item in sys.argv[2].split(",") if item)
c_clean, g_clean, c_checkpoint, c_terminal, g_checkpoint, g_terminal = (
    Path(value).resolve() for value in sys.argv[3:9]
)
paths = {
    "C clean": c_clean,
    "G clean": g_clean,
    "C checkpoint": c_checkpoint,
    "C terminal": c_terminal,
    "G checkpoint": g_checkpoint,
    "G terminal": g_terminal,
}
try:
    before = {label: sha256_file(path) for label, path in paths.items()}
    c_binding = _read_clean_validation_binding(
        path=c_clean,
        arm="C",
        fold_index=fold,
        source_tx_ids=source_tx_ids,
        checkpoint_sha256=before["C checkpoint"],
        terminal_sha256=before["C terminal"],
    )
    g_binding = _read_clean_validation_binding(
        path=g_clean,
        arm="G",
        fold_index=fold,
        source_tx_ids=source_tx_ids,
        checkpoint_sha256=before["G checkpoint"],
        terminal_sha256=before["G terminal"],
    )
    compared_fields = (
        "validation_keys",
        "validation_tx_ids",
        "validation_rx_ids",
        "validation_day_ids",
        "validation_tx_rx_day_coverage",
        "validation_eq_ids",
        "validation_sig_ids",
        "validation_metadata_order_sha256",
        "validation_indices_sha256",
    )
    for field in compared_fields:
        if c_binding[field] != g_binding[field]:
            raise ValueError(f"C/G clean-v4 physical metadata differs: {field}")
    for field in ("source_receiver_ids", "source_day_ids"):
        if c_binding["manifest"].get(field) != g_binding["manifest"].get(field):
            raise ValueError(f"C/G clean-v4 manifest axis differs: {field}")
    after = {label: sha256_file(path) for label, path in paths.items()}
    if after != before:
        raise ValueError("clean-v4 metadata proof input changed during validation")
except Exception as exc:
    print(f"clean-v4 metadata proof failed for F{fold}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
PY
}

verify_all_clean_metadata_pairs() {
  local fold
  for fold in 1 2 3 4 5 6; do
    verify_clean_metadata_pair "${fold}"
  done
}

claim_exact_root() {
  local path="$1"
  if ! mkdir -- "${path}"; then
    echo "refusing to overwrite ADV target prediction run/log root" >&2
    exit 3
  fi
}

open_exclusive_fd() {
  local path="$1"
  if ! { set -o noclobber; exec {OPEN_FD}>"${path}"; }; then
    set +o noclobber
    echo "refusing to overwrite ADV target prediction log/PID evidence" >&2
    exit 3
  fi
  set +o noclobber
}

declare -A LOG_FDS=()
claim_log() {
  local path="$1"
  open_exclusive_fd "${path}"
  LOG_FDS["${path}"]="${OPEN_FD}"
}

run_claimed_log() {
  local log_path="$1" log_fd status
  shift
  log_fd="${LOG_FDS["${log_path}"]}"
  if "$@" >&"${log_fd}" 2>&1; then
    status=0
  else
    status=$?
  fi
  exec {log_fd}>&-
  unset 'LOG_FDS['"${log_path}"']'
  return "${status}"
}

launch_claimed_log() {
  local log_path="$1" log_fd
  shift
  log_fd="${LOG_FDS["${log_path}"]}"
  (
    "$@" >&"${log_fd}" 2>&1
  ) &
  LAUNCHED_PID="$!"
  exec {log_fd}>&-
  unset 'LOG_FDS['"${log_path}"']'
}

verify_prediction_closure() {
  local -a arguments
  local fold
  arguments=("${PYTHON}" - "${PACKAGE_ROOT}")
  for fold in 1 2 3 4 5 6; do
    arguments+=("$(train_config_for "${fold}")" "$(prediction_for "${fold}")")
  done
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "${arguments[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


package_root = Path(sys.argv[1]).resolve()
pairs = [(Path(sys.argv[index]), Path(sys.argv[index + 1])) for index in range(2, 14, 2)]
package_manifest_sha = sha256_file(package_root / "manifest.json")
package_data_sha = sha256_file(package_root / "received_iq.npz")
for fold, (config_path, prediction_path) in enumerate(pairs, start=1):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if config.get("schema") != "cvs.phase1.adv3b02_train_data_config.v1":
        raise SystemExit(f"F{fold} train config schema drift")
    expected = {
        "schema": "cvs.phase1.adv3b02_target_prediction.v1",
        "sealed": True,
        "truth_sidecar_opened": False,
        "row_count": 3120,
        "forward_count": 3120,
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "baseline_terminal_status": "NON_PROMOTABLE_P0_DISABLED",
        "baseline_exit_code": 8,
        "baseline_promotion_ready": False,
        "formal_performance_claim": False,
    }
    for field, value in expected.items():
        if type(prediction.get(field)) is not type(value) or prediction.get(field) != value:
            raise SystemExit(f"F{fold} prediction closure drift: {field}")
    if prediction.get("train_config_manifest_sha256") != sha256_file(config_path):
        raise SystemExit(f"F{fold} prediction/train-config SHA drift")
    if prediction.get("package_manifest_sha256") != package_manifest_sha:
        raise SystemExit(f"F{fold} package manifest SHA drift")
    if prediction.get("received_iq_data_sha256") != package_data_sha:
        raise SystemExit(f"F{fold} received-IQ SHA drift")
    rows = prediction.get("rows")
    if not isinstance(rows, list) or len(rows) != 3120:
        raise SystemExit(f"F{fold} prediction row list drift")
    tokens = [row.get("opaque_token") for row in rows if isinstance(row, dict)]
    if len(tokens) != 3120 or len(set(tokens)) != 3120:
        raise SystemExit(f"F{fold} opaque token closure drift")
print("ADV_TARGET_PREDICTION_TECHNICAL_CLOSURE=PASS folds=6 rows_per_fold=3120")
PY
}

# A mismatched clean pair or missing input must fail before creating run/log or
# any other output.  The canonical C clean is selected only after all six
# source-only C/G physical metadata comparisons pass.
refuse_existing_roots
check_inputs
verify_all_clean_metadata_pairs
refuse_existing_roots
claim_exact_root "${RUN_ROOT}"
claim_exact_root "${LOG_ROOT}"

for fold in 1 2 3 4 5 6; do
  claim_exact_root "$(fold_output_root_for "${fold}")"
  claim_log "${LOG_ROOT}/F${fold}_seal_train_config.out"
  claim_log "${LOG_ROOT}/F${fold}_target_prediction.out"
done
claim_log "${LOG_ROOT}/prediction_closure.out"
PID_FILE="${LOG_ROOT}/pids_adv_target_prediction6.tsv"
open_exclusive_fd "${PID_FILE}"
PID_FD="${OPEN_FD}"
printf 'stage|fold|physical_gpu|pid|output|log_path\n' >&"${PID_FD}"

for fold in 1 2 3 4 5 6; do
  seal_command "${fold}"
  log_path="${LOG_ROOT}/F${fold}_seal_train_config.out"
  run_claimed_log "${log_path}" env \
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_ROOT}" \
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "${SEAL_CMD[@]}"
  [[ -f "$(train_config_for "${fold}")" ]] || {
    echo "ADV train-config seal did not close: F${fold}" >&2
    exit 2
  }
done

declare -a PIDS=()
for fold in 1 2 3 4 5 6; do
  gpu="${GPU_MAP[fold - 1]}"
  publish_command "${fold}"
  log_path="${LOG_ROOT}/F${fold}_target_prediction.out"
  launch_claimed_log "${log_path}" env \
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" \
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "${PUBLISH_CMD[@]}"
  PIDS+=("${LAUNCHED_PID}")
  printf 'ADV_TARGET_PREDICTION|%s|%s|%s|%s|%s\n' \
    "${fold}" "${gpu}" "${LAUNCHED_PID}" "$(prediction_for "${fold}")" "${log_path}" \
    >&"${PID_FD}"
done
exec {PID_FD}>&-

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
[[ "${status}" == "0" ]] || exit "${status}"

closure_log="${LOG_ROOT}/prediction_closure.out"
run_claimed_log "${closure_log}" verify_prediction_closure
