#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER_LAUNCHER="${WORKER_LAUNCHER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
ONLY_CANDIDATE=""
GPU_FILTER=""
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    --gpu=*) GPU_FILTER="${arg#--gpu=}" ;;
    *) echo "[SEEDSCAN-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -n "${GPU_FILTER}" && ! "${GPU_FILTER}" =~ ^[0-7]$ ]]; then
  echo "[SEEDSCAN-ERROR] --gpu must be in 0..7" >&2
  exit 2
fi
if [[ ! -f "${MATRIX}" ]]; then
  echo "[SEEDSCAN-ERROR] matrix missing: ${MATRIX}" >&2
  exit 2
fi
if [[ ! -f "${WORKER_LAUNCHER}" ]]; then
  echo "[SEEDSCAN-ERROR] worker launcher missing: ${WORKER_LAUNCHER}" >&2
  exit 2
fi

mapfile -t MATRIX_ROWS < <(
  "${CONTROL_PYTHON}" -c '
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("schema") != "cvs.phase1.fasttrust_eff_seedscan.v1":
    raise SystemExit("unsupported matrix schema")
if data.get("run_id") != "phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1":
    raise SystemExit("run ID drift")
if int(data.get("epochs", -1)) != 200:
    raise SystemExit("epoch drift")
if int(data.get("unlabeled_batch_size", -1)) != 256 or int(data.get("eval_batch_size", -1)) != 512:
    raise SystemExit("batch-size drift")
profile = data.get("source_profile", {})
if profile.get("profile_id") != "SRC5_MAXP2":
    raise SystemExit("source profile drift")
if profile.get("train_days") != [0, 1, 2, 3] or profile.get("test_days") != [0, 1, 2, 3]:
    raise SystemExit("day-grid drift")
if profile.get("train_receiver_indices") != [1, 3, 4, 6, 8]:
    raise SystemExit("source receiver drift")
if profile.get("test_receiver_indices") != [0, 2, 5, 7, 9, 10, 11]:
    raise SystemExit("target receiver drift")
if set(profile["train_receiver_indices"]).intersection(profile["test_receiver_indices"]):
    raise SystemExit("source/target receiver overlap")
rows = data.get("rows", [])
if len(rows) != 16:
    raise SystemExit("matrix must contain exactly 16 rows")
gpu_counts = Counter()
pair = defaultdict(list)
seen = set()
for row in rows:
    gpu = int(row["gpu"])
    slot = int(row["slot"])
    seed = int(row["seed"])
    variant = str(row["variant"])
    candidate = str(row["candidate"])
    level = str(row["muse_level"])
    if gpu not in range(8) or slot not in (0, 1) or candidate in seen:
        raise SystemExit("invalid or duplicate matrix row")
    if str(row.get("init")) != "scratch":
        raise SystemExit("all rows must train from scratch")
    if (variant, level) not in {("CONTROL", "M0"), ("FASTTRUST_EFF", "M3")}:
        raise SystemExit("variant/MUSE mapping drift")
    gpu_counts[gpu] += 1
    pair[gpu].append((seed, variant))
    seen.add(candidate)
    print("\t".join((str(gpu), str(slot), str(seed), variant, candidate, level)))
if gpu_counts != Counter({gpu: 2 for gpu in range(8)}):
    raise SystemExit("matrix must assign exactly two rows per GPU")
for gpu in range(8):
    if {seed for seed, _variant in pair[gpu]} != {713101 + gpu}:
        raise SystemExit("paired seed drift")
    if {variant for _seed, variant in pair[gpu]} != {"CONTROL", "FASTTRUST_EFF"}:
        raise SystemExit("paired variant drift")
' "${MATRIX}"
)

SELECTED_ROWS=()
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r gpu slot seed variant candidate level <<< "${row}"
  if [[ -n "${ONLY_CANDIDATE}" && "${candidate}" != "${ONLY_CANDIDATE}" ]]; then
    continue
  fi
  if [[ -n "${GPU_FILTER}" && "${gpu}" != "${GPU_FILTER}" ]]; then
    continue
  fi
  SELECTED_ROWS+=("${row}")
done
if [[ "${#SELECTED_ROWS[@]}" -eq 0 ]]; then
  echo "[SEEDSCAN-ERROR] selection matched no matrix row" >&2
  exit 2
fi

run_row() {
  local gpu="$1"
  local seed="$2"
  local variant="$3"
  local candidate="$4"
  local level="$5"
  local ablation="NONE"
  if [[ "${variant}" == "FASTTRUST_EFF" ]]; then
    ablation="FASTTRUST_EFF"
  fi
  local args=("--only=${level}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  env \
    ROOT="${ROOT}" \
    CODE_ROOT="${CODE_ROOT}" \
    PYTHON="${PYTHON}" \
    CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" \
    RUNS_ROOT="${RUNS_ROOT}" \
    GPU="${gpu}" \
    SEED="${seed}" \
    INIT_MODE="scratch" \
    CANDIDATE_ID_OVERRIDE="${candidate}" \
    MUSE_UNLABELED_BATCH_SIZE="256" \
    EVAL_BATCH_SIZE="512" \
    SOURCE_VAL_HEAVY_EVAL_START_EPOCH="1" \
    SOURCE_VAL_HEAVY_EVAL_INTERVAL="10" \
    SOURCE_VAL_HEAVY_EVAL_FINAL_WINDOW="20" \
    SOURCE_VAL_HEAVY_EVAL_FINAL_INTERVAL="1" \
    TOTAL_EPOCHS="200" \
    WISIG_TRAIN_DAYS="0,1,2,3" \
    WISIG_TEST_DAYS="" \
    WISIG_TRAIN_RXS="1,3,4,6,8" \
    WISIG_TEST_RXS="" \
    WISIG_ALLOW_SHARED_DAYS_IF_RECEIVERS_DISJOINT="false" \
    PHASE1_SOURCE_ONLY_EVAL="true" \
    EVAL_ON="source_v_select" \
    EVAL_GROUP_LOADER="source_v_select" \
    ABLATION="${ablation}" \
    bash "${WORKER_LAUNCHER}" "${args[@]}"
}

echo "[SEEDSCAN-RUN] run_id=${RUN_ID} matrix=${MATRIX} selected=${#SELECTED_ROWS[@]} dry_run=${DRY_RUN} mode=paired_two_per_gpu"
if [[ "${DRY_RUN}" == "1" ]]; then
  for row in "${SELECTED_ROWS[@]}"; do
    IFS=$'\t' read -r gpu slot seed variant candidate level <<< "${row}"
    echo "[SEEDSCAN-ROW] gpu=${gpu} slot=${slot} seed=${seed} variant=${variant} candidate=${candidate} level=${level} init=scratch"
    run_row "${gpu}" "${seed}" "${variant}" "${candidate}" "${level}"
  done
  exit 0
fi

if [[ -e "${RUNS_ROOT}" ]]; then
  echo "[SEEDSCAN-ERROR] refusing to overwrite existing run root: ${RUNS_ROOT}" >&2
  exit 3
fi
mkdir -p "${RUNS_ROOT}/dispatcher_logs"

PIDS=()
CANDIDATES=()
for row in "${SELECTED_ROWS[@]}"; do
  IFS=$'\t' read -r gpu slot seed variant candidate level <<< "${row}"
  echo "[SEEDSCAN-ROW] gpu=${gpu} slot=${slot} seed=${seed} variant=${variant} candidate=${candidate} level=${level} init=scratch"
  (
    run_row "${gpu}" "${seed}" "${variant}" "${candidate}" "${level}"
  ) > "${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  PIDS+=("$!")
  CANDIDATES+=("${candidate}")
done

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[SEEDSCAN-WORKER-COMPLETE] candidate=${CANDIDATES[$index]}"
  else
    echo "[SEEDSCAN-WORKER-FAILED] candidate=${CANDIDATES[$index]} log=${RUNS_ROOT}/dispatcher_logs/${CANDIDATES[$index]}.log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 4
fi
echo "[SEEDSCAN-COMPLETE] run_id=${RUN_ID} status=ALL_ROWS_COMPLETE"
