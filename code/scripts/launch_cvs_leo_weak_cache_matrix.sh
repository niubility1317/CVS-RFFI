#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SPEC_ROOT SOURCE_ROOT PYTHON LOG_ROOT GPU_CSV" >&2
  exit 2
fi

SPEC_ROOT=$1
SOURCE_ROOT=$2
PYTHON=$3
LOG_ROOT=$4
GPU_CSV=$5
BUILDER="$SOURCE_ROOT/code/scripts/build_cvs_leo_weak_iq_cache.py"

IFS=',' read -r -a GPUS <<<"$GPU_CSV"
mapfile -t SPECS < <(find "$SPEC_ROOT" -type f -name 'seed_*.json' | sort)

if [[ ${#SPECS[@]} -eq 0 || ${#GPUS[@]} -eq 0 ]]; then
  echo "no specs or GPUs" >&2
  exit 2
fi
if [[ ! -f "$BUILDER" || ! -x "$PYTHON" ]]; then
  echo "builder or Python missing" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT/cells"
printf 'spec_count=%s\ngpus=%s\n' "${#SPECS[@]}" "$GPU_CSV"

worker() {
  local worker_index=$1
  local gpu=$2
  local worker_rc=0
  local index spec cell log
  local -a outputs

  for ((index=worker_index; index<${#SPECS[@]}; index+=${#GPUS[@]})); do
    spec=${SPECS[$index]}
    cell="$(basename "$(dirname "$spec")")_$(basename "$spec" .json)"
    log="$LOG_ROOT/cells/${cell}.log"
    mapfile -t outputs < <(
      "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["out_manifest"]); [print(p["out_npz_by_scenario"][s]) for s in ("leo_clear_weak","leo_low_elev_weak","leo_rain_weak")]' \
        "$spec"
    )
    if [[ -f "${outputs[0]}" && -f "${outputs[1]}" && -f "${outputs[2]}" && -f "${outputs[3]}" ]]; then
      printf 'SKIP_COMPLETE cell=%s gpu=%s\n' "$cell" "$gpu" | tee "$log"
      continue
    fi
    if [[ -e "${outputs[0]}" || -e "${outputs[1]}" || -e "${outputs[2]}" || -e "${outputs[3]}" ]]; then
      printf 'FAIL_PARTIAL_OUTPUT cell=%s gpu=%s\n' "$cell" "$gpu" | tee "$log" >&2
      worker_rc=1
      continue
    fi
    {
      printf 'START cell=%s gpu=%s spec=%s\n' "$cell" "$gpu" "$spec"
      date --iso-8601=seconds
      if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$BUILDER" --spec "$spec" --device cuda:0; then
        printf 'COMPLETE cell=%s gpu=%s\n' "$cell" "$gpu"
      else
        rc=$?
        printf 'FAIL cell=%s gpu=%s rc=%s\n' "$cell" "$gpu" "$rc" >&2
        worker_rc=1
      fi
      date --iso-8601=seconds
    } >"$log" 2>&1
  done
  return "$worker_rc"
}

pids=()
for index in "${!GPUS[@]}"; do
  worker "$index" "${GPUS[$index]}" >"$LOG_ROOT/worker_${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

printf 'matrix_status=%s\n' "$status" | tee "$LOG_ROOT/matrix_status.txt"
exit "$status"
