from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    MAIN_SAT_EVAL_ON_NAMES,
    add_cvs_sat_eval_args,
    apply_sat_channel_for_scenario,
    make_torch_generator,
    parse_and_validate_sat_scenarios,
    resolve_sat_eval_loader_names,
)
from baselines.common.cvs_trainer import logits_from_output
from baselines.common.io import set_seed
from baselines.common.paper_protocol import train_receiver_count
from baselines.cvcnn_ce.model import BasicCVCNN
from baselines.drift.model import DRIFTModel
from baselines.riei_fd.model import RIEIModel
from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl


METHODS = {"cvcnn_ce", "riei_fd", "drift"}
GROUP_FIELDS = {
    "overall": (),
    "per_split": ("split",),
    "per_receiver": ("receiver_label",),
    "per_transmitter": ("transmitter_label",),
    "per_receiver_transmitter": ("receiver_label", "transmitter_label"),
    "per_receiver_transmitter_day": ("receiver_label", "transmitter_label", "day_label"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(values: list[Any], index: int, prefix: str) -> str:
    return str(values[index]) if 0 <= index < len(values) else f"{prefix}_{index}"


def _build_model(method: str, args: argparse.Namespace, loaders: Any) -> torch.nn.Module:
    if method == "cvcnn_ce":
        return BasicCVCNN(
            num_classes=loaders.split.num_classes,
            input_len=loaders.split.input_len,
            base_channels=int(args.cvcnn_base_channels),
            embedding_dim=int(args.cvcnn_embedding_dim),
            dropout=float(args.dropout),
        )
    num_train_receivers = train_receiver_count(
        loaders.split.split_info, loaders.split.num_receivers
    )
    if method == "riei_fd":
        return RIEIModel(
            loaders.split.num_classes,
            num_train_receivers,
            feature_dim=int(args.riei_feature_dim),
            dropout=float(args.dropout),
            encoder_use_projection=bool(args.use_resnet_projection),
        )
    return DRIFTModel(
        loaders.split.num_classes,
        num_train_receivers,
        embedding_dim=int(args.drift_embedding_dim),
        split_dim=int(args.drift_split_dim),
        dropout=float(args.dropout),
        encoder_use_projection=bool(args.use_resnet_projection),
        domain_discriminator_layers=int(args.domain_discriminator_layers),
    )


def _forward(method: str, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x, grl_lambda=0.0) if method == "drift" else model(x)
    return logits_from_output(output)


def aggregate_score_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    output: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        scenario_rows = [row for row in rows if str(row["scenario"]) == scenario]
        for level, fields in GROUP_FIELDS.items():
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in scenario_rows:
                groups[tuple(row[field] for field in fields)].append(row)
            for values, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
                correct = sum(int(row["correct"]) for row in group)
                confusion: dict[str, int] = defaultdict(int)
                for row in group:
                    key = f'{row["transmitter_label"]}->{row["predicted_transmitter_label"]}'
                    confusion[key] += 1
                item: dict[str, Any] = {
                    "scenario": scenario,
                    "group_type": level,
                    "split": "ALL",
                    "receiver_label": "ALL",
                    "transmitter_label": "ALL",
                    "day_label": "ALL",
                    "sample_count": len(group),
                    "correct_count": correct,
                    "accuracy": correct / len(group),
                    "confusion_json": json.dumps(dict(confusion), ensure_ascii=False, sort_keys=True),
                }
                for field, value in zip(fields, values):
                    item[field] = value
                output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    method = str(args.method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    scenarios = parse_and_validate_sat_scenarios(args)
    if not scenarios or any(str(value).lower() == "clean" for value in scenarios):
        raise ValueError("formal Phase1 detailed evaluation requires non-clean satellite scenarios")
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loaders = build_cvs_loaders(args, device)
    dataset = load_wisig_compact_pkl(args.wisig_pkl)
    tx_labels = [str(value) for value in dataset["tx_list"]]
    rx_labels = [str(value) for value in dataset["rx_list"]]
    day_labels = [str(value) for value in dataset["capture_date_list"]]
    model = _build_model(method, args, loaders).to(device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)
    model.eval()

    selected = resolve_sat_eval_loader_names(loaders.named_tests, args.eval_sat_on)
    expected = [name for name in MAIN_SAT_EVAL_ON_NAMES if name in loaders.named_tests]
    if selected != expected:
        raise ValueError(f"formal Phase1 detailed evaluation requires main splits {expected}, got {selected}")
    score_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for split_index, split in enumerate(selected):
            generator = make_torch_generator(
                device, int(args.sat_seed) + scenario_index * 1009 + split_index * 97
            )
            for batch_index, batch in enumerate(loaders.named_tests[split]):
                if int(args.sat_eval_max_batches) > 0 and batch_index >= int(args.sat_eval_max_batches):
                    break
                x = batch["iq"].to(device)
                x = apply_sat_channel_for_scenario(x, scenario, args, gen=generator)
                logits = _forward(method, model, x)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted = probabilities.max(dim=1)
                for i in range(int(predicted.numel())):
                    truth_i = int(batch["label"][i])
                    prediction_i = int(predicted[i].cpu())
                    receiver_i = int(batch["receiver"][i])
                    day_i = int(batch["day"][i])
                    sig_i = int(batch["sig_i"][i])
                    score_rows.append(
                        {
                            "sample_id": f"rx{receiver_i}:day{day_i}:tx{truth_i}:sig{sig_i}",
                            "scenario": str(scenario),
                            "split": str(split),
                            "receiver_index": receiver_i,
                            "receiver_label": _label(rx_labels, receiver_i, "rx"),
                            "transmitter_index": truth_i,
                            "transmitter_label": _label(tx_labels, truth_i, "tx"),
                            "day_index": day_i,
                            "day_label": _label(day_labels, day_i, "day"),
                            "predicted_transmitter_index": prediction_i,
                            "predicted_transmitter_label": _label(tx_labels, prediction_i, "tx"),
                            "confidence": float(confidence[i].cpu()),
                            "correct": int(prediction_i == truth_i),
                        }
                    )
    detailed_rows = aggregate_score_rows(score_rows)
    if not score_rows or not detailed_rows:
        raise RuntimeError("detailed evaluation produced no rows")
    if not all(math.isfinite(float(row["confidence"])) for row in score_rows):
        raise FloatingPointError("non-finite confidence in detailed score rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "score_table.csv", score_rows)
    _write_csv(output_dir / "detailed_metrics.csv", detailed_rows)
    (output_dir / "detailed_metrics.json").write_text(
        json.dumps(detailed_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scenario_metrics = {}
    for scenario in scenarios:
        rows = [row for row in score_rows if row["scenario"] == scenario]
        scenario_metrics[scenario] = {
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "correct_count": sum(int(row["correct"]) for row in rows),
            "sample_count": len(rows),
        }
    metrics = {
        "schema": "cvs_phase1_detailed_satellite_evaluation_v1",
        "method": method,
        "seed": int(args.seed),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "scenarios": scenario_metrics,
        "score_row_count": len(score_rows),
        "detailed_row_count": len(detailed_rows),
    }
    manifest = {
        "schema": "cvs_phase1_detailed_split_manifest_v1",
        "method": method,
        "formal_satellite_scenarios": list(scenarios),
        "selected_test_splits": selected,
        "receiver_labels": rx_labels,
        "transmitter_labels": tx_labels,
        "day_labels": day_labels,
        "split_info": loaders.split.split_info,
        "all_tests_satellite_augmented": True,
        "clean_control_in_formal_result": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed Phase1 CVS satellite post-hoc evaluator")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cvcnn-base-channels", type=int, default=32)
    parser.add_argument("--cvcnn-embedding-dim", type=int, default=128)
    parser.add_argument("--riei-feature-dim", type=int, default=512)
    parser.add_argument("--drift-embedding-dim", type=int, default=512)
    parser.add_argument("--drift-split-dim", type=int, default=256)
    parser.add_argument("--use-resnet-projection", action="store_true")
    parser.add_argument("--domain-discriminator-layers", type=int, default=2)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
