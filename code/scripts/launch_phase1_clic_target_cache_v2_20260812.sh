#!/usr/bin/env bash
set -euo pipefail

RUN_ID="phase1_clic_target_confirmation_20260812_v2"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
SPEC="${CODE_ROOT}/configs/phase1_clic_target_confirmation_20260812_v2.json"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
BUILDER="${CODE_ROOT}/scripts/build_cvs_leo_weak_iq_cache.py"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

command=("${PYTHON}" -u "${BUILDER}" --spec "${SPEC}" --device cuda:0)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] stage=CLIC_TARGET_CACHE run_id=%q physical_gpu=0' "${RUN_ID}"
  printf ' %q' "${command[@]}"; printf '\n'
  exit 0
fi

[[ -f "${SPEC}" && -f "${BUILDER}" ]] || { echo "missing target cache spec/builder" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite target cache run/log root" >&2; exit 3; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
log_path="${LOG_ROOT}/target_cache.out"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
pid=$!
printf 'pid|stage|physical_gpu|log_path\n%s|CLIC_TARGET_CACHE|0|%s\n' "${pid}" "${log_path}" >"${LOG_ROOT}/pids_target_cache.tsv"
wait "${pid}"
