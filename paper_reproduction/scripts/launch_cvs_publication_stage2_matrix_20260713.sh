#!/usr/bin/env bash
set -euo pipefail

ROOT="${CVS_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${CVS_PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
PHASE="${1:-}"
MODE="${2:---dry-run}"
GPUS=(3 4 5 6 7)

case "${PHASE}" in
  stage2b)
    CONFIG="paper_reproduction/configs/cvs_stage2b_supervised_da_publication_base_n607.json"
    ;;
  stage2c)
    CONFIG="paper_reproduction/configs/cvs_stage2c_publication_base_n607.json"
    ;;
  *)
    echo "usage: $0 <stage2b|stage2c> [--dry-run|--execute]" >&2
    exit 2
    ;;
esac
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  echo "mode must be --dry-run or --execute" >&2
  exit 2
fi

cd "${ROOT}"
RUN_ID="cvs_publication_${PHASE}_full_matrix_20260713"
OUTPUT_ROOT="paper_reproduction/runs/${RUN_ID}"
LOG_ROOT="paper_reproduction/logs/${RUN_ID}"
CVS_CONFIG="paper_reproduction/configs/cvs_proposed_stage2_publication_features_n607.json"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  cmd=(
    "${PYTHON}" -m paper_reproduction.scripts.run_cvs_publication_matrix
    --phase "${PHASE}"
    --config "${CONFIG}"
    --cvs-config "${CVS_CONFIG}"
    --output-root "${OUTPUT_ROOT}"
    --log-root "${LOG_ROOT}/rows"
    --shard-count "${#GPUS[@]}"
    --shard-index "${shard}"
    --python "${PYTHON}"
  )
  if [[ "${MODE}" == "--dry-run" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi
  pattern="run_cvs_publication_matrix.*--output-root ${OUTPUT_ROOT}.*--shard-index ${shard}"
  if pgrep -af "${pattern}" >/dev/null; then
    echo "active shard already exists: ${shard}" >&2
    exit 3
  fi
  log_file="${LOG_ROOT}/shard_${shard}_gpu_${gpu}.out"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" --execute >"${log_file}" 2>&1 &
  pid=$!
  echo "phase=${PHASE} shard=${shard} gpu=${gpu} pid=${pid} log=${log_file}"
done
