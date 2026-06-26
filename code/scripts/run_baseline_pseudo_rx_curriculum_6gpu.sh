#!/usr/bin/env bash
set -uo pipefail

# Baseline receiver-curriculum launcher.
#
# Receiver design:
#   - Target Rx(1-19), WiSig index 1, image pairs drawn from {0,10,11,2}
#   - Target Rx(14-7), WiSig index 2, image pairs drawn from {0,1,10,11}
#   - Each image pair is expanded from K=2 to K=7 by appending receivers from
#     a fixed target-specific source universe.
#   - Evaluation reuses the existing CVS named-test module and tests every
#     receiver not included in the current training receiver set.
#
# Examples:
#   bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan SMOKE --dry-run
#   bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5
#   METHODS=cvcnn_ce,drift,riei_fd,ra_collab bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan FULL
#   PL_MODES=plain,pseudo bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan CORE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../dataset_wisig.py" ] && [ -d "${SCRIPT_DIR}/../common" ]; then
  # Server layout: CV-SincNet/scripts/<this-file>
  WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [ -f "${SCRIPT_DIR}/../dataset_wisig.py" ] && [ -d "${SCRIPT_DIR}/../baselines" ]; then
  # Alternate repo layout: <repo>/scripts/<this-file>
  WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  # Local development layout: <repo>/code/scripts/<this-file>
  WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${WORKSPACE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5}"
PLAN="${PLAN:-CORE}"
METHODS_CSV="${METHODS:-cvcnn_ce}"
PL_MODES_CSV="${PL_MODES:-plain}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/baseline_supervised_rx_curriculum}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/baseline_supervised_rx_curriculum}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"

EPOCHS="${EPOCHS:-200}"
PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-150}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.90}"
PSEUDO_MARGIN="${PSEUDO_MARGIN:-0.0}"
LAMBDA_PSEUDO="${LAMBDA_PSEUDO:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

usage() {
  sed -n '1,22p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV        GPUs to use, default 0,1,2,3,4,5
  --plan NAME          SMOKE, CORE, or FULL
  --methods CSV        cvcnn_ce,drift,riei_fd,ra_collab
  --pl-modes CSV       plain,pseudo
  --wisig-pkl PATH     Dataset_WigSig/ManySig.pkl path
  --python PATH        Python executable
  --run-root PATH      Output checkpoint root
  --log-root PATH      Log root
  --no-skip-done       Re-run even when metrics.json exists
  --stop-on-fail       Stop queue after first failure
  --stream-logs        Stream job logs to scheduler stdout
  --dry-run            Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --pl-modes) PL_MODES_CSV="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  echo "Set WISIG_PKL=/path/to/Dataset_WigSig/ManySig.pkl or pass --wisig-pkl." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

generate_queue() {
  "${PYTHON_BIN}" - "${PLAN}" "${METHODS_CSV}" "${PL_MODES_CSV}" "${QUEUE_FILE}" <<'PY'
import itertools
import sys

plan = sys.argv[1].strip().upper()
aliases = {
    "ra_collab": "ra_collab",
}
methods = [aliases.get(x.strip(), x.strip()) for x in sys.argv[2].split(",") if x.strip()]
pl_modes = [x.strip() for x in sys.argv[3].split(",") if x.strip()]
queue_file = sys.argv[4]

import os

if os.path.isdir("baselines"):
    method_modules = {
        "cvcnn_ce": "baselines.cvcnn_ce.train",
        "drift": "baselines.drift.train",
        "riei_fd": "baselines.riei_fd.train",
        "ra_collab": "baselines.ra_collab.train",
    }
else:
    method_modules = {
        "cvcnn_ce": "cvcnn_ce.train",
        "drift": "drift.train",
        "riei_fd": "riei_fd.train",
        "ra_collab": "ra_collab.train",
    }
rx_label = {
    0: "1-1",
    1: "1-19",
    2: "14-7",
    3: "18-2",
    4: "19-2",
    5: "2-1",
    6: "2-19",
    7: "20-1",
    8: "3-19",
    9: "7-14",
    10: "7-7",
    11: "8-8",
}
all_rx = list(range(12))
targets = {
    "T1": {
        "target_idx": 1,
        "target_label": "1-19",
        "universe": [0, 10, 11, 2, 5, 6, 3],
        "pairs": {
            "P01": [0, 10],
            "P02": [0, 11],
            "P03": [0, 2],
            "P04": [10, 11],
            "P05": [10, 2],
            "P06": [11, 2],
        },
    },
    "T14": {
        "target_idx": 2,
        "target_label": "14-7",
        "universe": [0, 1, 10, 11, 5, 6, 3],
        "pairs": {
            "P01": [0, 1],
            "P02": [0, 10],
            "P03": [0, 11],
            "P04": [1, 10],
            "P05": [1, 11],
            "P06": [10, 11],
        },
    },
}

if plan == "SMOKE":
    target_names = ["T1"]
    pair_names = ["P01"]
    levels = [2, 7]
    if not methods:
        methods = ["cvcnn_ce"]
    if not pl_modes:
        pl_modes = ["plain"]
elif plan == "CORE":
    target_names = ["T1", "T14"]
    pair_names = ["P01", "P02", "P03", "P04", "P05", "P06"]
    levels = [2, 3, 4, 5, 6, 7]
elif plan == "FULL":
    target_names = ["T1", "T14"]
    pair_names = ["P01", "P02", "P03", "P04", "P05", "P06"]
    levels = [2, 3, 4, 5, 6, 7]
else:
    raise SystemExit(f"unknown plan: {plan}")

rows = []
for method in methods:
    if method not in method_modules:
        raise SystemExit(f"unknown method: {method}")
for pl_mode in pl_modes:
    if pl_mode not in {"plain", "pseudo"}:
        raise SystemExit(f"unknown pseudo-label mode: {pl_mode}")

for method, pl_mode, target_name, pair_name, level in itertools.product(methods, pl_modes, target_names, pair_names, levels):
    target = targets[target_name]
    pair = list(target["pairs"][pair_name])
    train = list(pair)
    for rx in target["universe"]:
        if len(train) >= level:
            break
        if rx not in train:
            train.append(rx)
    if len(train) != level:
        raise SystemExit(f"could not build K={level} set for {target_name}_{pair_name}")
    test = [rx for rx in all_rx if rx not in set(train)]
    train_csv = ",".join(str(x) for x in train)
    test_csv = ",".join(str(x) for x in test)
    train_labels = ",".join(rx_label[x] for x in train)
    test_labels = ",".join(rx_label[x] for x in test)
    exp_id = f"{method}_{pl_mode}_{target_name}_{pair_name}_K{level}_train-{train_csv.replace(',', '-')}_test-rest"
    desc = (
        f"method={method}; mode={pl_mode}; design_target=Rx({target['target_label']}) idx={target['target_idx']}; "
        f"image_pair={pair_name}; K={level}; train_rx={train_labels}; test_all_remaining_rx={test_labels}"
    )
    rows.append([
        exp_id,
        method,
        method_modules[method],
        pl_mode,
        target_name,
        test_csv,
        train_csv,
        str(level),
        desc,
    ])

with open(queue_file, "w", encoding="utf-8") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")
PY
}

generate_queue
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

COMMON_ARGS=(
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_train_ratio 0.2
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_max_day123_per_combo 0
  --wisig_max_train_per_combo 0
  --wisig_max_val_per_combo 0
  --wisig_max_test_per_combo 0
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor 2
  --epochs "${EPOCHS}"
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --device cuda
)

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
FREE_GPUS=("${GPU_LIST[@]}")
NEXT_INDEX=0
STATUS=0

queue_line_at() {
  sed -n "$(($1 + 1))p" "${QUEUE_FILE}"
}

launch_one() {
  local gpu_id="$1" exp_id="$2" method="$3" module="$4" pl_mode="$5" target_name="$6" test_rxs="$7" train_rxs="$8" level="$9" desc="${10}"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local log="${LOG_ROOT}/${exp_id}_${STAMP}.log"
  local pseudo_args=()
  if [ "${pl_mode}" = "pseudo" ]; then
    pseudo_args=(
      --use_pseudo_labels
      --pseudo_start_epoch "${PSEUDO_START_EPOCH}"
      --pseudo_threshold "${PSEUDO_THRESHOLD}"
      --pseudo_margin "${PSEUDO_MARGIN}"
      --lambda_pseudo "${LAMBDA_PSEUDO}"
    )
  fi

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u -m "${module}" \
    "${COMMON_ARGS[@]}" \
    --wisig_train_rxs "${train_rxs}" \
    --wisig_test_rxs "${test_rxs}" \
    --output_dir "${out_dir}" \
    "${pseudo_args[@]}")"

  {
    echo "EXP_ID=${exp_id}"
    echo "METHOD=${method}"
    echo "MODULE=${module}"
    echo "PSEUDO_MODE=${pl_mode}"
    echo "TARGET=${target_name}"
    echo "TRAIN_RXS=${train_rxs}"
    echo "TEST_RXS=${test_rxs}"
    echo "LEVEL=${level}"
    echo "DESCRIPTION=${desc}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} ${desc}"
    log_msg "CMD=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  if [ "${STREAM_LOGS}" = "1" ]; then
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" 2>&1
      status="$?"
      echo "__EXIT_STATUS__=${status}"
      exit "${status}"
    ) | sed -u "s/^/[${exp_id}|GPU${gpu_id}] /" | tee -a "${log}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id method module pl_mode target_name test_rxs train_rxs level desc
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS=$'\t' read -r exp_id method module pl_mode target_name test_rxs train_rxs level desc <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${method}" "${module}" "${pl_mode}" "${target_name}" "${test_rxs}" "${train_rxs}" "${level}" "${desc}"
  done
}

reap_one() {
  wait -n
  local status=$?
  local i
  for i in "${!RUNNING_PIDS[@]}"; do
    if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      local gpu="${RUNNING_GPUS[$i]}" tag="${RUNNING_TAGS[$i]}"
      unset 'RUNNING_PIDS[i]' 'RUNNING_TAGS[i]' 'RUNNING_GPUS[i]'
      RUNNING_PIDS=("${RUNNING_PIDS[@]}")
      RUNNING_TAGS=("${RUNNING_TAGS[@]}")
      RUNNING_GPUS=("${RUNNING_GPUS[@]}")
      FREE_GPUS+=("${gpu}")
      if [ "${status}" -ne 0 ]; then
        STATUS=1
      fi
      log_msg "[FINISHED] exp=${tag} status=${status} freed_gpu=${gpu}"
      return
    fi
  done
}

log_msg "Baseline receiver-curriculum launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "METHODS=${METHODS_CSV} PL_MODES=${PL_MODES_CSV}"
log_msg "WISIG_PKL=${WISIG_PKL}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "QUEUE_FILE=${QUEUE_FILE}"
log_msg "EPOCHS=${EPOCHS} PSEUDO_START_EPOCH=${PSEUDO_START_EPOCH} PSEUDO_THRESHOLD=${PSEUDO_THRESHOLD}"
log_msg "DRY_RUN=${DRY_RUN} SKIP_DONE=${SKIP_DONE} STOP_ON_FAIL=${STOP_ON_FAIL} STREAM_LOGS=${STREAM_LOGS}"

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    reap_one
    if [ "${STATUS}" -ne 0 ] && [ "${STOP_ON_FAIL}" = "1" ]; then
      log_msg "Stopping early because a job failed and STOP_ON_FAIL=1."
      exit 1
    fi
  fi
  start_until_full
done

log_msg "Baseline receiver-curriculum queue finished status=${STATUS}"
exit "${STATUS}"
