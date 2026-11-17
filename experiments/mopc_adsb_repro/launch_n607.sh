#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/szu2070436088/.conda/envs/SDG-SEI/bin/python
ROOT=/home/szu2070436088/2510044040/CV-SincNet
DATA="$ROOT/datasets/mopc_adsb_20261117_r01"
RUN="$ROOT/runs/20261117-mopc-adsb-50plus25-m2-r01"
RAW="$DATA/adsb_bladerf2_10M_qt0.mat"
test -f "$RAW"
test -f "$DATA/prepared/data_manifest.json"
mkdir "$RUN"
trap 'printf "FAILED line %s\n" "$LINENO" > "$RUN/FAILED"' ERR
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.mopc_adsb_repro.smoke --data "$DATA/prepared" --output "$RUN/smoke" > "$RUN/smoke.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.mopc_adsb_repro.run train --data "$DATA/prepared" --output "$RUN/base" --version base > "$RUN/base.log" 2>&1
test -f "$RUN/base/train_complete.json"
for version in author paper; do
  if [ "$version" = author ]; then gpu=0; else gpu=1; fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" -m experiments.mopc_adsb_repro.run train --data "$DATA/prepared" --output "$RUN/$version" --version "$version" --base "$RUN/base/stage0.pt"
    "$PY" -m experiments.mopc_adsb_repro.run predict --data "$DATA/prepared" --output "$RUN/${version}_predictions" --base "$RUN/base/stage0.pt" --incremental "$RUN/$version"
    "$PY" -m experiments.mopc_adsb_repro.run score --data "$DATA/prepared" --output "$RUN/${version}_score" --predictions "$RUN/${version}_predictions"
  ) > "$RUN/$version.log" 2>&1 &
  if [ "$version" = author ]; then pa=$!; else pp=$!; fi
done
status_a=0; status_p=0
wait "$pa" || status_a=$?
wait "$pp" || status_p=$?
if [ "$status_a" -ne 0 ] || [ "$status_p" -ne 0 ]; then
  printf 'author=%s paper=%s\n' "$status_a" "$status_p" > "$RUN/FAILED"
  exit 1
fi
test -f "$RUN/author_score/metrics.json"
test -f "$RUN/paper_score/metrics.json"
printf 'complete\n' > "$RUN/COMPLETE"
