#!/usr/bin/env bash
set -euo pipefail

# Fixed paper-core RIEI/DRIFT reproduction launcher.
# RIEI is fixed to the core16 paper-parity variant.
# DRIFT is fixed to the next10 final-70+ variant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"
cd "${WORKSPACE_ROOT}"

WISIG_PROTOCOL="${WISIG_PROTOCOL:-drift_day1}"
SEED="${SEED:-1337}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/riei_drift_fixed_paper_core_${WISIG_PROTOCOL}_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/riei_drift_fixed_paper_core_${WISIG_PROTOCOL}_seed${SEED}}"
RIEI_RUN_ROOT="${RIEI_RUN_ROOT:-${RUN_ROOT}/riei}"
DRIFT_RUN_ROOT="${DRIFT_RUN_ROOT:-${RUN_ROOT}/drift}"
RIEI_LOG_ROOT="${RIEI_LOG_ROOT:-${LOG_ROOT}/riei}"
DRIFT_LOG_ROOT="${DRIFT_LOG_ROOT:-${LOG_ROOT}/drift}"
RIEI_GPU="${RIEI_GPU:-0}"
DRIFT_GPU="${DRIFT_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_DONE_FLAG=()

usage() {
  cat <<'EOF'
Options:
  --wisig-protocol NAME  drift_day1 or riei_original; default drift_day1
  --python PATH          Python executable
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --run-root PATH        Shared fixed run root
  --log-root PATH        Shared fixed log root
  --riei-gpu ID          GPU for RIEI_C06_sum_featnorm1e4; default 0
  --drift-gpu ID         GPU for DRIFT_N02_raw_cap4000; default 1
  --no-skip-done         Re-run even if metrics.json exists
  --stop-on-fail         Stop on first child failure
  --dry-run              Create manifests and print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wisig-protocol) WISIG_PROTOCOL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --riei-gpu) RIEI_GPU="$2"; shift 2 ;;
    --drift-gpu) DRIFT_GPU="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE_FLAG+=(--no-skip-done); shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -f "${WORKSPACE_ROOT}/run_riei_drift_core16_queue.sh" ]; then
  echo "ERROR: missing run_riei_drift_core16_queue.sh" >&2
  exit 2
fi
if [ ! -f "${WORKSPACE_ROOT}/run_riei_drift_core_next10_queue.sh" ]; then
  echo "ERROR: missing run_riei_drift_core_next10_queue.sh" >&2
  exit 2
fi

mkdir -p "${RIEI_RUN_ROOT}" "${DRIFT_RUN_ROOT}" "${RIEI_LOG_ROOT}" "${DRIFT_LOG_ROOT}"

COMMON_ARGS=(
  --wisig-protocol "${WISIG_PROTOCOL}"
  --wisig-pkl "${WISIG_PKL}"
)
if [ -n "${PYTHON_BIN}" ]; then
  COMMON_ARGS+=(--python "${PYTHON_BIN}")
fi
if [ "${DRY_RUN}" = "1" ]; then
  COMMON_ARGS+=(--dry-run)
fi
COMMON_ARGS+=("${SKIP_DONE_FLAG[@]}")

echo "[FIXED] run_root=${RUN_ROOT}"
echo "[FIXED] log_root=${LOG_ROOT}"
echo "[FIXED] riei_run_root=${RIEI_RUN_ROOT}"
echo "[FIXED] drift_run_root=${DRIFT_RUN_ROOT}"
echo "[FIXED] riei_log_root=${RIEI_LOG_ROOT}"
echo "[FIXED] drift_log_root=${DRIFT_LOG_ROOT}"
echo "[FIXED] RIEI fixed variant=RIEI_C06_sum_featnorm1e4 gpu=${RIEI_GPU}"
echo "[FIXED] DRIFT fixed variant=DRIFT_N02_raw_cap4000 gpu=${DRIFT_GPU}"
echo "[FIXED] RIEI core losses=CE+lambda_mi*MI-lambda_ie*IE, lambda_feature_norm=0.0001"
echo "[FIXED] DRIFT core losses=tx_CE+rx_CE+lambda_grl*domain_CE+lambda_center*center+lambda_mse*negative_MSE, mse_cap=4000"

declare -a PIDS=()
declare -a NAMES=()

DRY_RUN="${DRY_RUN}" bash "${WORKSPACE_ROOT}/run_riei_drift_core16_queue.sh" \
  "${COMMON_ARGS[@]}" \
  --run-root "${RIEI_RUN_ROOT}" \
  --log-root "${RIEI_LOG_ROOT}" \
  --variants RIEI_C06_sum_featnorm1e4 \
  --gpu-ids "${RIEI_GPU}" &
PIDS+=("$!")
NAMES+=("RIEI_C06_sum_featnorm1e4")

DRY_RUN="${DRY_RUN}" bash "${WORKSPACE_ROOT}/run_riei_drift_core_next10_queue.sh" \
  "${COMMON_ARGS[@]}" \
  --run-root "${DRIFT_RUN_ROOT}" \
  --log-root "${DRIFT_LOG_ROOT}" \
  --variants DRIFT_N02_raw_cap4000 \
  --gpu-ids "${DRIFT_GPU}" &
PIDS+=("$!")
NAMES+=("DRIFT_N02_raw_cap4000")

status=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    echo "[FIXED][${name}] done"
  else
    rc=$?
    echo "[FIXED][${name}] failed rc=${rc}" >&2
    status="${rc}"
    if [ "${STOP_ON_FAIL}" = "1" ]; then
      break
    fi
  fi
done

echo "[FIXED] finished status=${status}"
exit "${status}"
