from __future__ import annotations

from typing import Any


PAPER_TITLE = "Class-Incremental Learning for Wireless Device Identification in IoT"


def build_stage_plan(*, total_classes: int, initial_classes: int, classes_per_increment: int) -> list[dict[str, Any]]:
    total_classes = int(total_classes)
    initial_classes = int(initial_classes)
    classes_per_increment = int(classes_per_increment)
    if total_classes <= 0 or initial_classes <= 0 or classes_per_increment <= 0:
        raise ValueError("class counts must be positive")
    if initial_classes > total_classes:
        raise ValueError("initial_classes cannot exceed total_classes")
    remaining = total_classes - initial_classes
    if remaining % classes_per_increment != 0:
        raise ValueError("remaining classes must divide evenly by classes_per_increment")

    stages: list[dict[str, Any]] = []
    known: list[int] = []
    cursor = 0
    stage_sizes = [initial_classes] + [classes_per_increment] * (remaining // classes_per_increment)
    for stage_id, size in enumerate(stage_sizes):
        train_class_ids = list(range(cursor, cursor + size))
        known.extend(train_class_ids)
        stages.append(
            {
                "stage": stage_id,
                "train_class_ids": train_class_ids,
                "old_class_ids": list(range(cursor)) if stage_id > 0 else [],
                "new_class_ids": train_class_ids,
                "known_class_ids": known.copy(),
                "uses_historical_samples": False,
            }
        )
        cursor += size
    return stages


def validate_paper_faithful_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "method_id",
        "dataset",
        "total_classes",
        "initial_classes",
        "classes_per_increment",
        "train_ratio",
        "validation_ratio",
        "batch_size",
        "incremental_epochs",
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing paper-faithful config fields: {missing}")
    if config["method_id"] != "csil_class_incremental_iot":
        raise ValueError("method_id must be csil_class_incremental_iot")
    if config["dataset"] != "ADS-B":
        raise ValueError("paper-faithful CSIL config must use ADS-B")
    if float(config["train_ratio"]) != 0.6 or float(config["validation_ratio"]) != 0.4:
        raise ValueError("paper-faithful split must be 60% train and 40% validation")
    plan = build_stage_plan(
        total_classes=int(config["total_classes"]),
        initial_classes=int(config["initial_classes"]),
        classes_per_increment=int(config["classes_per_increment"]),
    )
    if len(plan) != 5:
        raise ValueError("paper setup expects five class-incremental batches")
    checked = dict(config)
    checked["paper"] = PAPER_TITLE
    checked["stage_plan"] = plan
    checked["claim_boundary"] = "paper_faithful_adsb_class_incremental_only"
    checked["not_cvs_stage2"] = True
    checked["not_satellite_deployment_evidence"] = True
    checked["paper_reported_hyperparameters"] = {
        "incremental_batch_size": int(config["batch_size"]),
        "incremental_epochs": int(config["incremental_epochs"]),
        "optimizer": config["optimizer"],
        "learning_rate": float(config["learning_rate"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
    }
    checked["paper_unspecified_fields"] = [
        "stage0_epochs",
        "random_seed",
        "exact_100_transponder_ids",
        "kd_weight",
        "ewc_weight",
        "expanded_embedding_dimension_policy",
    ]
    return checked

