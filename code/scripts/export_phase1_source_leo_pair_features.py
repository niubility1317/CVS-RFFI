#!/usr/bin/env python
"""Export source-only clean/LEO paired Phase1 features for adapter training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for path in (str(CODE_ROOT), str(REPO_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
for path in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, path)

from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _validate_star_ground_impl,
    extract_features_with_metadata,
)
from eval_feature_diagnosis import (  # noqa: E402
    build_model_from_ckpt,
    infer_num_domains,
    load_state_dict_safely,
    strip_module_prefix,
)
from dataset_wisig import WiSigSubsetDataset  # noqa: E402
from SSDG.train_ssdg import split_tx_rx_day_1_7_2_roles  # noqa: E402
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario  # noqa: E402


def _save_npz(path: Path, payload: dict[str, np.ndarray], manifest: dict) -> None:
    out = dict(payload)
    out["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--feature_name", default="z_id")
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--source_rxs", required=True)
    parser.add_argument("--source_days", default=None)
    parser.add_argument("--wisig_equalized", default="1")
    parser.add_argument("--wisig_domain", default="rx_day")
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--model_variant", default=None)
    parser.add_argument("--branch_ablation", default=None)
    parser.add_argument("--sample_rate_hz", type=float, default=None)
    parser.add_argument("--max_samples_per_combo", type=int, default=0)
    parser.add_argument("--max_samples_per_tx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4070311)
    parser.add_argument("--labeled_ratio", type=float, default=0.07)
    parser.add_argument("--unlabeled_ratio", type=float, default=0.63)
    parser.add_argument("--source_cal_ratio", type=float, default=0.15)
    parser.add_argument("--source_select_ratio", type=float, default=0.15)
    parser.add_argument("--sat_scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--star_ground_channel_impl", default="simplified_leo_residual", choices=["legacy_satellite", "simplified_leo_residual"])
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_ds, source_info = _build_wisig_dataset(
        pkl_path=str(args.wisig_pkl),
        tx_spec=str(args.source_tx_ids),
        role="source",
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=args.source_days,
        rxs=args.source_rxs,
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(args.max_samples_per_tx),
        seed=int(args.seed),
    )
    source_base_size = int(len(source_ds))
    _labeled_idx, _unlabeled_idx, source_cal_idx, _source_select_idx = split_tx_rx_day_1_7_2_roles(
        source_ds,
        labeled_ratio=float(args.labeled_ratio),
        unlabeled_ratio=float(args.unlabeled_ratio),
        source_cal_ratio=float(args.source_cal_ratio),
        source_select_ratio=float(args.source_select_ratio),
    )
    source_ds = WiSigSubsetDataset(
        source_ds,
        source_cal_idx,
        split_source="ssdg_source_v_cal",
    )
    source_info = {
        **source_info,
        "base_size": source_base_size,
        "size": int(len(source_ds)),
        "phase1_role": "V_cal",
        "role_ratios": {
            "L_s": float(args.labeled_ratio),
            "U_s": float(args.unlabeled_ratio),
            "V_cal": float(args.source_cal_ratio),
            "V_select": float(args.source_select_ratio),
        },
        "seed": int(args.seed),
    }
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("checkpoint must contain 'args' and 'model'")
    ckpt_args = ckpt["args"]
    state = strip_module_prefix(ckpt["model"])
    num_domains = infer_num_domains(source_ds, state=state, split_info={}, ckpt_args=ckpt_args, cli_num_domains=None)
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(ckpt_args, args, num_domains=num_domains, input_len=int(args.wisig_out_len), device=device)
    missing, unexpected, skipped_mismatch = load_state_dict_safely(model, state)

    loader = DataLoader(source_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    _validate_star_ground_impl(str(args.star_ground_channel_impl), scenarios, field="sat_scenarios")
    common_manifest = {
        "payload_source": "phase1_source_only_clean_leo_pair_features",
        "feature_name": str(args.feature_name),
        "checkpoint": str(args.ckpt),
        "source": source_info,
        "source_tx_ids": source_info["tx_labels"],
        "source_pair_role": "V_cal",
        "uses_target_clean": False,
        "uses_target_labels": False,
        "uses_unknown_query": False,
        "star_ground_channel_impl": str(args.star_ground_channel_impl),
        "sat_scenarios": scenarios,
        "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in scenarios},
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_mismatch": len(skipped_mismatch),
    }
    clean_payload = extract_features_with_metadata(
        model,
        loader,
        device=device,
        feature_name=str(args.feature_name),
        role="source",
        channel_view="clean",
    )
    _save_npz(args.out_dir / "source_clean.npz", clean_payload, {**common_manifest, "channel_view": "clean"})

    outputs = {"clean": str(args.out_dir / "source_clean.npz")}
    for i, scenario in enumerate(scenarios):
        sat_payload = extract_features_with_metadata(
            model,
            loader,
            device=device,
            feature_name=str(args.feature_name),
            role="source",
            channel_view="satellite",
            sat_scenarios=[scenario],
            sat_args=args,
            sat_seed=int(args.seed) + 1009 + i,
        )
        name = f"source_{scenario}.npz"
        _save_npz(args.out_dir / name, sat_payload, {**common_manifest, "channel_view": "satellite", "sat_scenario": scenario})
        outputs[scenario] = str(args.out_dir / name)
    summary = {
        "out_dir": str(args.out_dir),
        "source_size": int(source_info["size"]),
        "outputs": outputs,
        "protocol": {
            "source_only": True,
            "uses_target_clean": False,
            "uses_target_labels": False,
            "uses_unknown_query": False,
        },
    }
    (args.out_dir / "source_leo_pair_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
