#!/usr/bin/env python
"""Direct old-class evaluation for an exactly reconstructed ADV3B02 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for path in (str(CODE_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from SSDG import train_ssdg as ssdg_mod  # noqa: E402
from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from eval_feature_diagnosis import infer_num_domains, strip_module_prefix  # noqa: E402
from export_spaceborne_features import _build_wisig_dataset, _meta_to_list  # noqa: E402
from training_controls import parse_sat_scenarios  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_model(checkpoint: dict[str, Any], *, input_len: int, device: torch.device):
    state = strip_module_prefix(checkpoint["model"])
    checkpoint_args = dict(checkpoint.get("args") or {})
    num_domains = infer_num_domains(
        None,
        state=state,
        split_info={},
        ckpt_args=checkpoint_args,
        cli_num_domains=None,
    )
    parser = ssdg_mod.build_arg_parser()
    model_args = parser.parse_args(["--output_dir", str(PROJECT_ROOT / ".tmp_adv3b02_direct_old")])
    for key, value in checkpoint_args.items():
        setattr(model_args, key, value)
    model_args.device = str(device)
    merged = ssdg_mod.merge_checkpoint_args(
        checkpoint,
        model_args,
        input_len=int(input_len),
        num_domains=int(num_domains),
    )
    merged = ssdg_mod._apply_model_cli_args(merged, model_args)
    model = ssdg_mod.build_baseline_model(merged, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"strict checkpoint reconstruction failed: missing={list(missing)} unexpected={list(unexpected)}"
        )
    model.eval()
    return model, model_args, int(num_domains)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(score_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        correct = sum(int(row["correct"]) for row in rows)
        item: dict[str, Any] = {key: value for key, value in zip(keys, group)}
        item.update(
            {
                "correct": correct,
                "total": len(rows),
                "accuracy": correct / len(rows),
            }
        )
        output.append(item)
    return output


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset, dataset_info = _build_wisig_dataset(
        pkl_path=str(args.wisig_pkl),
        tx_spec=str(args.old_tx_ids),
        role="target_old",
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=str(args.target_days),
        rxs=str(args.target_rxs),
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(args.max_samples_per_tx),
        seed=int(args.seed),
    )
    model, model_args, num_domains = _exact_model(
        checkpoint,
        input_len=int(args.wisig_out_len),
        device=device,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    scenarios = list(parse_sat_scenarios(str(args.scenarios)))
    class_labels = list(dataset_info["tx_labels"])
    score_rows: list[dict[str, Any]] = []
    sat_args = SimpleNamespace(sat_fs_hz=float(args.sat_fs_hz), sat_fc_hz=float(args.sat_fc_hz))
    for scenario in scenarios:
        generator = make_torch_generator(device, int(args.sat_seed))
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("expected WiSig batches shaped (x, y, domain, metadata)")
            x, y, _domain, metadata = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, _ = apply_sat_channel_for_scenario(
                x,
                str(scenario),
                sat_args,
                gen=generator,
                return_meta=False,
            )
            output = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
            logits = output["tx_logits"]
            confidence, predicted = torch.softmax(logits, dim=1).max(dim=1)
            n = int(y.numel())
            tx_values = _meta_to_list(metadata, "tx", n)
            rx_values = _meta_to_list(metadata, "rx", n)
            day_values = _meta_to_list(metadata, "day", n)
            eq_values = _meta_to_list(metadata, "equalized", n)
            sig_values = _meta_to_list(metadata, "sig_i", n)
            for index in range(n):
                truth = int(y[index].cpu())
                prediction = int(predicted[index].cpu())
                predicted_label = class_labels[prediction] if 0 <= prediction < len(class_labels) else str(prediction)
                score_rows.append(
                    {
                        "sample_id": "|".join(
                            [
                                "target_old",
                                tx_values[index],
                                rx_values[index],
                                day_values[index],
                                eq_values[index],
                                sig_values[index],
                            ]
                        ),
                        "scenario": str(scenario),
                        "receiver": rx_values[index],
                        "transmitter": tx_values[index],
                        "day": day_values[index],
                        "truth_index": truth,
                        "predicted_index": prediction,
                        "predicted_transmitter": predicted_label,
                        "confidence": float(confidence[index].cpu()),
                        "correct": int(prediction == truth),
                    }
                )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "score_table.csv", score_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for level, keys in (
        ("overall", tuple()),
        ("scenario", ("scenario",)),
        ("receiver", ("receiver",)),
        ("scenario_receiver", ("scenario", "receiver")),
        ("transmitter", ("transmitter",)),
        ("scenario_receiver_transmitter", ("scenario", "receiver", "transmitter")),
    ):
        rows = _aggregate(score_rows, keys)
        for row in rows:
            aggregate_rows.append({"level": level, **row})
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    scenario_metrics = {
        row["scenario"]: {
            "accuracy": row["accuracy"],
            "correct": row["correct"],
            "total": row["total"],
        }
        for row in aggregate_rows
        if row["level"] == "scenario"
    }
    metrics = {
        "schema": "adv3b02_ground_direct_old_v1",
        "method": "ground_classifier_tx_logits",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_load_strict": True,
        "target_adaptation": False,
        "qknn": False,
        "fft_auxiliary": False,
        "tta": False,
        "support_labels_used": False,
        "score_row_count": len(score_rows),
        "scenario_metrics": scenario_metrics,
        "overall_accuracy": sum(int(row["correct"]) for row in score_rows) / len(score_rows),
    }
    manifest = {
        "schema": "adv3b02_ground_direct_old_split_v1",
        "checkpoint_load_strict": True,
        "dataset": dataset_info,
        "class_id_to_tx": class_labels,
        "target_receivers": str(args.target_rxs).split(","),
        "target_days": str(args.target_days).split(","),
        "scenarios": scenarios,
        "sat_seed_per_scenario": int(args.sat_seed),
        "num_domains_from_checkpoint": num_domains,
        "claim_boundary": "phase1_closed_set_old_direct_target_pool_diagnostic",
    }
    resolved = {
        **vars(args),
        "checkpoint_args": dict(checkpoint.get("args") or {}),
        "model_args_device": str(getattr(model_args, "device", "")),
    }
    for filename, payload in (
        ("metrics.json", metrics),
        ("split_manifest.json", manifest),
        ("resolved_config.json", resolved),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-tx-ids", default="0,1,2,3,4,5")
    parser.add_argument("--target-rxs", default="20-1,3-19,7-14,7-7,8-8")
    parser.add_argument("--target-days", default="0")
    parser.add_argument("--scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--wisig-equalized", default="1")
    parser.add_argument("--wisig-domain", default="rx_day")
    parser.add_argument("--wisig-out-len", type=int, default=256)
    parser.add_argument("--max-samples-per-combo", type=int, default=0)
    parser.add_argument("--max-samples-per-tx", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--seed",
        type=int,
        default=713130,
        help="Target-old subset seed; 713130 equals export seed 713101 + target-old offset 29.",
    )
    parser.add_argument("--sat-seed", type=int, default=713912)
    parser.add_argument("--sat-fs-hz", type=float, default=25e6)
    parser.add_argument("--sat-fc-hz", type=float, default=2.462e9)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
