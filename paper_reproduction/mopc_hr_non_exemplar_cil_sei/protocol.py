"""Protocol checks for paper-faithful and CVS-aligned MoPC-HR runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def validate_mopc_hr_config(config: dict[str, Any]) -> dict[str, Any]:
    checked = deepcopy(config)
    required_positive = ("total_classes", "base_classes", "classes_per_increment")
    for key in required_positive:
        if int(checked.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if checked["base_classes"] >= checked["total_classes"]:
        raise ValueError("base_classes must be smaller than total_classes")
    remaining = checked["total_classes"] - checked["base_classes"]
    if remaining % checked["classes_per_increment"] != 0:
        raise ValueError("remaining classes must divide evenly into increments")
    if checked.get("replay_raw_samples", False):
        raise ValueError("MoPC-HR is non-exemplar and cannot replay historical raw samples")

    checked.setdefault("base_epochs", 20)
    checked.setdefault("incremental_epochs", 20)
    checked.setdefault("batch_size", 16)
    checked.setdefault("optimizer", "SGD")
    checked.setdefault("learning_rate", 0.01)
    checked.setdefault("momentum", 0.9)
    checked.setdefault("weight_decay", 2e-4)
    checked.setdefault("prototype_noise_std", 0.05)
    checked.setdefault("prototype_momentum", 0.97)
    checked.setdefault("similarity_mode", "paper_cosine")
    checked["replay_raw_samples"] = False
    checked["distillation_in_total_loss"] = False
    checked["claim_boundary"] = "mopc_hr_non_exemplar_cil_paper_mechanism"
    return checked
