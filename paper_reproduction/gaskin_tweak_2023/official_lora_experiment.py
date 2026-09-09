"""Runnable closed-set configuration-portability reproduction on the public LoRa recordings."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import torch

from .calibration import (
    CalibrationState,
    MultipleDomainCalibrationState,
    aggregate_embeddings,
    calibrate,
    calibrate_multiple_domains,
    closed_set_predict,
)
from .method_config import load_method_config
from .model import TweakEncoder
from .official_lora import OfficialLoRaRecord, RecordFrameSplit, load_configuration_records, read_iq_frames, split_record_frames
from .triplet import margin_violating_triplet_loss, random_triplet_indices


@dataclass(frozen=True)
class ConfigurationPortabilityRow:
    calibration_configuration: str
    test_configuration: str


@dataclass(frozen=True)
class ConfigurationPortabilityPlan:
    source_configuration: str
    device_ids: tuple[int, ...]
    calibration_configurations: tuple[str, ...]
    test_configurations: tuple[str, ...]
    single_domain_rows: tuple[ConfigurationPortabilityRow, ...]
    group_size: int
    epochs: int


def build_configuration_portability_plan() -> ConfigurationPortabilityPlan:
    """Figure-13b plus Figure-14: Config2 source, 4x4 target matrix, and multiple calibration."""
    configurations = ("Config1", "Config2", "Config3", "Config4")
    return ConfigurationPortabilityPlan(
        source_configuration="Config2",
        device_ids=tuple(range(1, 11)),
        calibration_configurations=configurations,
        test_configurations=configurations,
        single_domain_rows=tuple(
            ConfigurationPortabilityRow(calibration, testing)
            for calibration in configurations
            for testing in configurations
        ),
        group_size=10,
        epochs=100,
    )


def _coprime_stride(size: int, seed: int) -> int:
    candidate = (seed % (size - 1)) + 1
    while gcd(candidate, size) != 1:
        candidate = candidate % (size - 1) + 1
    return candidate


def _source_training_batches(
    records: list[OfficialLoRaRecord],
    split: RecordFrameSplit,
    *,
    seed: int,
    max_batches: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Produce balanced 64-item batches with a deterministic permutation of every source frame."""
    batch_size, classes_per_batch, samples_per_class = 64, 8, 8
    if len(records) < classes_per_batch:
        raise ValueError("paper batch composition requires at least eight source devices")
    total_batches = (len(records) * len(split.training)) // batch_size
    if max_batches is not None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")
        total_batches = min(total_batches, max_batches)
    strides = [_coprime_stride(len(split.training), seed + 17 * position) for position in range(len(records))]
    offsets = [(seed * (position + 1)) % len(split.training) for position in range(len(records))]
    used = [0 for _ in records]
    for batch_index in range(total_batches):
        selected_classes = [(batch_index * classes_per_batch + item) % len(records) for item in range(classes_per_batch)]
        frames, labels = [], []
        for class_index in selected_classes:
            positions = [
                split.training.start + ((offsets[class_index] + strides[class_index] * (used[class_index] + item)) % len(split.training))
                for item in range(samples_per_class)
            ]
            used[class_index] += samples_per_class
            frames.append(read_iq_frames(records[class_index], frame_indices=split.physical_indices(positions)))
            labels.append(torch.full((samples_per_class,), records[class_index].device_id, dtype=torch.long))
        yield torch.cat(frames), torch.cat(labels)


def _copy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _train_encoder(
    records: list[OfficialLoRaRecord],
    split: RecordFrameSplit,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
    max_batches_per_epoch: int | None,
) -> tuple[TweakEncoder, dict[str, object]]:
    """Run the paper's SGD/triplet training with its configured five-rate search."""
    metadata = load_method_config().method_metadata()
    learning_rates = [float(value) for value in metadata["unpublished_defaults"]["learning_rate_grid"]["value"]]
    torch.manual_seed(seed)
    model = TweakEncoder().to(device)
    initial_state = _copy_state_dict(model.state_dict())
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch, best_lr = float("inf"), 0, 0.0
    epoch_rows: list[dict[str, float | int]] = []
    for lr_position, learning_rate in enumerate(learning_rates):
        model.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, batches = 0.0, 0
            for iq, labels in _source_training_batches(
                records,
                split,
                seed=seed + 10_000 * lr_position + epoch,
                max_batches=max_batches_per_epoch,
            ):
                iq, labels = iq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                anchors, positives, negatives = random_triplet_indices(
                    labels,
                    generator=torch.Generator(device=device).manual_seed(seed + 1_000_000 * lr_position + 10_000 * epoch + batches),
                )
                loss = margin_violating_triplet_loss(model(iq), anchors=anchors, positives=positives, negatives=negatives)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach())
                batches += 1
            mean_loss = running_loss / batches
            epoch_rows.append({"learning_rate": learning_rate, "epoch": epoch, "mean_training_loss": mean_loss, "batches": batches})
            print(json.dumps({"event": "epoch", **epoch_rows[-1]}, sort_keys=True), flush=True)
            if mean_loss < best_loss:
                best_loss, best_epoch, best_lr = mean_loss, epoch, learning_rate
                best_state = _copy_state_dict(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_learning_rate": best_lr,
        "best_mean_training_loss": best_loss,
        "epoch_rows": epoch_rows,
        "batch_size": 64,
        "batch_composition": {"classes_per_batch": 8, "samples_per_class": 8},
    }


@torch.no_grad()
def _embed_record_frames(
    model: TweakEncoder,
    record: OfficialLoRaRecord,
    frame_range: range,
    split: RecordFrameSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for start in range(frame_range.start, frame_range.stop, batch_size):
        indices = range(start, min(start + batch_size, frame_range.stop))
        features.append(model(read_iq_frames(record, frame_indices=split.physical_indices(indices)).to(device)).cpu())
    return torch.cat(features)


@torch.no_grad()
def _calibrate_configuration(
    model: TweakEncoder,
    records: list[OfficialLoRaRecord],
    split: RecordFrameSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = [_embed_record_frames(model, record, split.calibration, split, device=device, batch_size=batch_size) for record in records]
    labels = torch.cat([torch.full((feature.shape[0],), record.device_id, dtype=torch.long) for record, feature in zip(records, features)])
    return torch.cat(features), labels


@torch.no_grad()
def _evaluate_configuration(
    model: TweakEncoder,
    records: list[OfficialLoRaRecord],
    split: RecordFrameSplit,
    state: CalibrationState | MultipleDomainCalibrationState,
    *,
    device: torch.device,
    batch_size: int,
    group_size: int,
) -> dict[str, int | float]:
    model.eval()
    points, labels = [], []
    for record in records:
        embeddings = _embed_record_frames(model, record, split.testing, split, device=device, batch_size=batch_size)
        usable = (embeddings.shape[0] // group_size) * group_size
        points.append(aggregate_embeddings(embeddings[:usable], group_size=group_size))
        labels.append(torch.full((usable // group_size,), record.device_id, dtype=torch.long))
    point_tensor, label_tensor = torch.cat(points), torch.cat(labels)
    predictions = closed_set_predict(point_tensor, state)
    return {
        "accuracy": float(predictions.eq(label_tensor).float().mean()),
        "correct_decisions": int(predictions.eq(label_tensor).sum()),
        "decisions": int(label_tensor.numel()),
    }


def run_configuration_portability(
    *,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    max_train_batches_per_epoch: int | None,
    inference_batch_size: int,
) -> dict[str, object]:
    """Run Figure-13b and Figure-14 without the vanilla or ablation arms."""
    plan = build_configuration_portability_plan()
    if epochs <= 0 or inference_batch_size <= 0:
        raise ValueError("epochs and inference_batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records_by_configuration = {
        configuration: load_configuration_records(Path(data_root), configuration, device_ids=plan.device_ids)
        for configuration in plan.calibration_configurations
    }
    splits = [split_record_frames(total_samples=record.total_samples, seed=seed) for records in records_by_configuration.values() for record in records]
    if len({(split.total_frames, split.training.stop, split.calibration.stop, split.testing.start) for split in splits}) != 1:
        raise ValueError("all selected public LoRa records must share the paper's frame layout")
    split = splits[0]
    model, training = _train_encoder(
        records_by_configuration[plan.source_configuration], split, device=device, epochs=epochs, seed=seed,
        max_batches_per_epoch=max_train_batches_per_epoch,
    )
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save({"state_dict": _copy_state_dict(model.state_dict()), "training": training}, checkpoint_path)

    calibration_features: dict[str, torch.Tensor] = {}
    calibration_labels: dict[str, torch.Tensor] = {}
    calibration_states: dict[str, CalibrationState] = {}
    for configuration, records in records_by_configuration.items():
        features, labels = _calibrate_configuration(model, records, split, device=device, batch_size=inference_batch_size)
        calibration_features[configuration], calibration_labels[configuration] = features, labels
        calibration_states[configuration] = calibrate(features, labels)
    multiple_state = calibrate_multiple_domains(
        torch.cat([calibration_features[configuration] for configuration in plan.calibration_configurations]),
        torch.cat([calibration_labels[configuration] for configuration in plan.calibration_configurations]),
        [configuration for configuration in plan.calibration_configurations for _ in range(calibration_features[configuration].shape[0])],
        samples_per_class=len(split.calibration),
    )

    single_rows: list[dict[str, object]] = []
    for row in plan.single_domain_rows:
        result = _evaluate_configuration(
            model, records_by_configuration[row.test_configuration], split, calibration_states[row.calibration_configuration],
            device=device, batch_size=inference_batch_size, group_size=plan.group_size,
        )
        single_rows.append({**asdict(row), **result})
    multiple_rows = [
        {
            "test_configuration": configuration,
            **_evaluate_configuration(
                model, records_by_configuration[configuration], split, multiple_state,
                device=device, batch_size=inference_batch_size, group_size=plan.group_size,
            ),
        }
        for configuration in plan.test_configurations
    ]

    # Checkpoint smoke deliberately uses only calibration frames; no held-out test/query data is read.
    smoke_features, _ = _calibrate_configuration(
        model, records_by_configuration[plan.source_configuration][:1], split, device=device, batch_size=inference_batch_size
    )
    method_metadata = load_method_config().method_metadata()
    result: dict[str, object] = {
        "status": "PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS" if max_train_batches_per_epoch is None else "NONPAPER_SMOKE_LIMIT",
        "method_metadata": method_metadata,
        "plan": asdict(plan),
        "data": {
            "root": str(Path(data_root).resolve()),
            "source_configuration": plan.source_configuration,
            "raw_iq_datatype": "cf32",
            "frames_per_record": split.total_frames,
            "training_frames_per_record": len(split.training),
            "calibration_frames_per_record": len(split.calibration),
            "testing_frames_per_record": len(split.testing),
            "physical_frame_partition": {
                "kind": "seeded_affine_full_record_permutation",
                "seed": split.seed,
                "stride": split.permutation_stride,
                "offset": split.permutation_offset,
            },
        },
        "training": training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_no_query_smoke": {"passed": bool(smoke_features.numel()), "embedding_count": int(smoke_features.shape[0])},
        "single_domain_calibration_4x4": single_rows,
        "multiple_configuration_calibration": multiple_rows,
    }
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=20260908, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--inference-batch-size", default=1024, type=int)
    parser.add_argument("--max-train-batches-per-epoch", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_configuration_portability(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        seed=args.seed,
        epochs=args.epochs,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
        inference_batch_size=args.inference_batch_size,
    )
    print(json.dumps({"event": "complete", "status": result["status"], "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
