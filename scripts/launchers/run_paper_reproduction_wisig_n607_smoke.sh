#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_reproduction_n607_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_n607_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${ROOT}:${ROOT}/code:${PYTHONPATH:-}"

nohup "${PYTHON}" -u -m paper_reproduction.protonet_cda.train \
  --config paper_reproduction/configs/protonet_cda_wisig_n607_smoke.json \
  --device cuda:0 \
  --run-dir "${RUN_ROOT}/protonet_cda_seed1337" \
  > "${LOG_ROOT}/protonet_cda.out" 2>&1 &
PROTO_PID=$!

nohup "${PYTHON}" -u -m paper_reproduction.feature_separation_crossrx.train \
  --config paper_reproduction/configs/feature_separation_crossrx_wisig_n607_smoke.json \
  --device cuda:1 \
  --run-dir "${RUN_ROOT}/feature_separation_seed1337" \
  > "${LOG_ROOT}/feature_separation.out" 2>&1 &
FS_PID=$!

cat > "${RUN_ROOT}/launch_manifest.json" <<JSON
{
  "run_root": "${RUN_ROOT}",
  "log_root": "${LOG_ROOT}",
  "protonet_pid": ${PROTO_PID},
  "feature_separation_pid": ${FS_PID},
  "protonet_log": "${LOG_ROOT}/protonet_cda.out",
  "feature_separation_log": "${LOG_ROOT}/feature_separation.out"
}
JSON

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "PROTO_PID=${PROTO_PID}"
echo "FEATURE_SEPARATION_PID=${FS_PID}"
