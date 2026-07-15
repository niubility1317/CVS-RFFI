#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-qknnv42_support_only_taskadapt_875_20260715_v1}"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
FEATURE_ROOT="${RUN_ROOT}/base_features"
CONFIG="${ROOT}/paper_reproduction/configs/cvs_qknnv42_support_only_taskadapt_875_stage2c_20260715_n607.json"
CKPT="${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"

OLD_TX="14-10,14-7,20-15,20-19,6-15,8-20"
NEW_TX="1-16,1-18"
TARGET_RXS="20-1,3-19,7-14,7-7,8-8"
SCENARIOS=(leo_clear_weak leo_low_elev_weak leo_rain_weak)

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "${FEATURE_ROOT}" "${LOG_ROOT}"
cd "${ROOT}"

existing=0
for scenario in "${SCENARIOS[@]}"; do
  [[ -s "${FEATURE_ROOT}/${scenario}.npz" ]] && existing=$((existing + 1))
done
if [[ "${existing}" -ne 0 && "${existing}" -ne 3 ]]; then
  echo "partial base-feature set exists (${existing}/3); refusing overwrite" >&2
  exit 3
fi

if [[ "${existing}" -eq 0 ]]; then
  for scenario in "${SCENARIOS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u code/export_spaceborne_features.py \
      --ckpt "${CKPT}" \
      --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl" \
      --new_wisig_pkl "${ROOT}/Dataset_WigSig/ManyTx.pkl" \
      --out_npz "${FEATURE_ROOT}/${scenario}.npz" \
      --feature_name z_id --aux_fft_logmag_dim 96 --include_raw_iq \
      --source_tx_ids "${OLD_TX}" --source_rxs "0" \
      --target_old_tx_ids "${OLD_TX}" --target_old_rxs "${TARGET_RXS}" \
      --new_tx_ids "${NEW_TX}" --new_rxs "${TARGET_RXS}" \
      --source_channel_view satellite --source_sat_scenarios "${scenario}" \
      --target_old_channel_view satellite --target_old_sat_scenarios "${scenario}" \
      --target_new_channel_view satellite --target_new_sat_scenarios "${scenario}" \
      --satellite_tta_policy none --star_ground_channel_impl simplified_leo_residual \
      --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 \
      --max_samples_per_combo 0 --max_samples_per_tx 400 \
      --batch_size 384 --device cuda:0 --seed 4070391 \
      >"${LOG_ROOT}/export_${scenario}.log" 2>&1
  done
fi

"${PYTHON}" - "${FEATURE_ROOT}" "${CKPT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
ckpt = Path(sys.argv[2])
scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
for scenario in scenarios:
    path = root / f"{scenario}.npz"
    with np.load(path, allow_pickle=False) as payload:
        manifest = json.loads(str(payload["manifest_json"].item()))
        roles = payload["dataset_role"].astype(str)
        mask = np.isin(roles, ["target_old", "target_new"])
        observed = set(payload["sat_scenarios"][mask].astype(str).tolist())
        views = set(payload["channel_views"][mask].astype(str).tolist())
        if payload["raw_iq"].shape[1:] != (2, 256):
            raise SystemExit(f"invalid raw IQ shape: {path}")
        if payload["fft_logmag_features"].shape[1] != 96:
            raise SystemExit(f"invalid FFT96 shape: {path}")
    if manifest.get("raw_iq_included") is not True:
        raise SystemExit(f"raw IQ provenance missing: {path}")
    if manifest.get("checkpoint_load_strict") is not True:
        raise SystemExit(f"checkpoint was not strictly loaded: {path}")
    if int(manifest.get("satellite_tta_view_count", -1)) != 1:
        raise SystemExit(f"base cache is not one-view: {path}")
    if observed != {scenario} or not views or any("clean" in value.lower() for value in views):
        raise SystemExit(f"target rows are not scenario-only LEO rows: {path}")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
PY

"${PYTHON}" -u -m paper_reproduction.scripts.run_cvs_qknnv42_support_only_taskadapt_875 \
  --config "${CONFIG}" --ckpt "${CKPT}" \
  --adapter-root "${RUN_ROOT}/adapters" --output-root "${RUN_ROOT}/results" \
  --log-root "${LOG_ROOT}" --manifest "${RUN_ROOT}/matrix_manifest.json" \
  --new-count 2 --prepare-only

echo "[QKNN-SUPPORT-ONLY-875-PREPARED] run_root=${RUN_ROOT}"
