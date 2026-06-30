#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT}/code/scripts/launch_stage2_short195s3_multiproto_finaltest_20260630_1505.sh}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-stage2_short195s3_otherdomains_finaltest_20260630_1525}"
DRY_RUN="${DRY_RUN:-0}"

TARGET_DOMAINS=("3-19" "7-14" "7-7" "8-8")

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

echo "[STAGE2-SHORT195S3-OTHERDOMAINS] prefix=${RUN_ID_PREFIX} dry_run=${DRY_RUN}"
echo "[STAGE2-SHORT195S3-OTHERDOMAINS] domains=${TARGET_DOMAINS[*]}"
echo "[STAGE2-SHORT195S3-OTHERDOMAINS] base_launcher=${BASE_LAUNCHER}"

for domain in "${TARGET_DOMAINS[@]}"; do
  safe_domain="${domain//-/_}"
  run_id="${RUN_ID_PREFIX}_${safe_domain}"
  echo "[STAGE2-SHORT195S3-OTHERDOMAINS-DOMAIN] target=${domain} run_id=${run_id}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    env RUN_ID="${run_id}" TARGET_RECEIVER_LABEL="${domain}" RUN_NO_REJECT=1 bash "${BASE_LAUNCHER}" --dry-run
  else
    env RUN_ID="${run_id}" TARGET_RECEIVER_LABEL="${domain}" RUN_NO_REJECT=1 bash "${BASE_LAUNCHER}"
  fi
done

echo "[STAGE2-SHORT195S3-OTHERDOMAINS-DONE] prefix=${RUN_ID_PREFIX}"
