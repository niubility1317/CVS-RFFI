#!/usr/bin/env python
"""Run NTRS B0 diagnostics from source-only clean/LEO pair exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.ntrs_b0_diagnostics import analyze_paired_shift  # noqa: E402


def _load_payload(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as payload:
        if "manifest_json" not in payload.files:
            raise ValueError(f"pair export is missing manifest_json: {path}")
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "manifest_json"}
    return arrays, manifest


def _require_exact_pair(
    clean: dict[str, np.ndarray],
    satellite: dict[str, np.ndarray],
    clean_manifest: dict,
    satellite_manifest: dict,
) -> None:
    identity_keys = ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids", "raw_labels", "domain_labels")
    for key in identity_keys:
        if key not in clean or key not in satellite:
            raise ValueError(f"pair export is missing physical identity field: {key}")
        if not np.array_equal(clean[key], satellite[key]):
            raise ValueError(f"clean/satellite physical sample mismatch: {key}")
    for key in ("checkpoint", "source", "source_tx_ids", "source_pair_role", "star_ground_channel_impl"):
        if clean_manifest.get(key) != satellite_manifest.get(key):
            raise ValueError(f"clean/satellite manifest mismatch: {key}")
    if clean_manifest.get("source_pair_role") != "V_cal":
        raise ValueError("B0 diagnostics require exact Phase1 V_cal pair exports")
    if clean_manifest.get("channel_view") != "clean":
        raise ValueError("clean B0 export must declare channel_view=clean")
    if satellite_manifest.get("channel_view") != "satellite":
        raise ValueError("satellite B0 export must declare channel_view=satellite")
    scenario = str(satellite_manifest.get("sat_scenario", ""))
    if scenario not in {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}:
        raise ValueError(f"B0 satellite export has unsupported scenario: {scenario}")


def _classifier_weight(checkpoint: Path) -> torch.Tensor:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    matches = [
        value
        for key, value in state.items()
        if str(key).removeprefix("module.").endswith("id_backbone.cls_head.head.weight")
        and torch.is_tensor(value)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one raw classifier weight, found {len(matches)}")
    return matches[0].detach().float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_npz", type=Path, required=True)
    parser.add_argument("--satellite_npz", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite B0 output: {args.output_json}")
    clean, clean_manifest = _load_payload(args.clean_npz)
    if Path(str(clean_manifest.get("checkpoint", ""))).resolve() != args.checkpoint.resolve():
        raise ValueError("B0 classifier checkpoint does not match the pair-export checkpoint")
    sat_loaded = [_load_payload(path) for path in args.satellite_npz]
    clean_features = np.asarray(clean["features"], dtype=np.float32)
    clean_labels = np.asarray(clean["raw_labels"], dtype=np.int64)
    sat_features = []
    labels = []
    tx_ids = []
    scenario_ids = []
    scenario_names = []
    for index, (payload, manifest) in enumerate(sat_loaded):
        _require_exact_pair(clean, payload, clean_manifest, manifest)
        features = np.asarray(payload["features"], dtype=np.float32)
        row_labels = np.asarray(payload["raw_labels"], dtype=np.int64)
        if features.shape != clean_features.shape or not np.array_equal(row_labels, clean_labels):
            raise ValueError("clean and satellite exports must preserve paired row order")
        sat_features.append(features)
        labels.append(row_labels)
        tx_ids.append(row_labels)
        scenario_ids.append(np.full(row_labels.shape, index, dtype=np.int64))
        scenario_names.append(str(manifest["sat_scenario"]))
    if set(scenario_names) != {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}:
        raise ValueError("B0 diagnostics require exactly the three LEO_WEAK family scenarios")
    repeats = len(sat_features)
    result = analyze_paired_shift(
        torch.from_numpy(np.tile(clean_features, (repeats, 1))),
        torch.from_numpy(np.concatenate(sat_features, axis=0)),
        torch.from_numpy(np.concatenate(labels, axis=0)),
        _classifier_weight(args.checkpoint),
        tx_ids=torch.from_numpy(np.concatenate(tx_ids, axis=0)),
        scenario_ids=torch.from_numpy(np.concatenate(scenario_ids, axis=0)),
    )
    result["protocol"] = {
        "source_only": True,
        "uses_target_clean": False,
        "uses_target_labels": False,
        "uses_unknown_query": False,
        "scenario_count": repeats,
        "source_pair_role": "V_cal",
        "physical_identity_fields_verified": ["tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"],
        "scenarios": scenario_names,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(args.output_json), "sample_count": result["sample_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
