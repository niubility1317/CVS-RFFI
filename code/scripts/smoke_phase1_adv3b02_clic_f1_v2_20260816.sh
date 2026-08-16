#!/usr/bin/env bash
set -euo pipefail

# Bounded v2 technical smoke for F1.  It mechanically recovers the frozen F1
# command from v2 formal dry-run and alters only the dedicated v2 control.

FORMAL_LAUNCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_phase1_adv3b02_clic6_v2_20260816.sh"
FORMAL_RUN_ID="phase1_adv3b02_clic6_20260816_v2"
SMOKE_ROOT_NAME=".smoke_phase1_adv3b02_clic6_20260816_v2_F1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
SMOKE_RUN_ROOT="${PROJECT_ROOT}/runs/${SMOKE_ROOT_NAME}"
SMOKE_LOG_ROOT="${PROJECT_ROOT}/logs/${SMOKE_ROOT_NAME}"
DRY_RUN=0

[[ ! -v RUN_ROOT && ! -v LOG_ROOT ]] || {
  echo "RUN_ROOT/LOG_ROOT overrides are forbidden for the isolated v2 smoke" >&2
  exit 2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "invalid argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -f "${FORMAL_LAUNCHER}" ]] || {
  echo "missing formal launcher: ${FORMAL_LAUNCHER}" >&2
  exit 2
}

FORMAL_F1_DRY_RUN="$({
  RUN_ID="${FORMAL_RUN_ID}" \
  RUN_ROOT="${SMOKE_RUN_ROOT}" \
  LOG_ROOT="${SMOKE_LOG_ROOT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  CODE_ROOT="${CODE_ROOT}" \
  PYTHON="${PYTHON}" \
  WISIG_PKL="${WISIG_PKL}" \
  bash "${FORMAL_LAUNCHER}" --dry-run
} | sed -n '1p')"
[[ "${FORMAL_F1_DRY_RUN}" == "[DRY-RUN] "* ]] || {
  echo "failed to mechanically recover formal v2 F1 dry-run command" >&2
  exit 2
}
F1_COMMAND="${FORMAL_F1_DRY_RUN#\[DRY-RUN\] } --phase1_adv3b02_technical_smoke_v2_max_batches 4"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] %s\n' "${F1_COMMAND}"
  exit 0
fi

[[ ! -e "${SMOKE_RUN_ROOT}" && ! -e "${SMOKE_LOG_ROOT}" ]] || {
  echo "refusing to overwrite v2 smoke run/log root" >&2
  exit 3
}
[[ -f "${WISIG_PKL}" ]] || {
  echo "missing WiSig dataset: ${WISIG_PKL}" >&2
  exit 2
}

mkdir -p "${SMOKE_RUN_ROOT}" "${SMOKE_LOG_ROOT}"
printf '%s\n' "$$" >"${SMOKE_LOG_ROOT}/F1_ADV3B02_CLIC.pid"
bash -lc "${F1_COMMAND}" >"${SMOKE_LOG_ROOT}/F1_ADV3B02_CLIC.out" 2>&1
