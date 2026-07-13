from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from paper_reproduction.common.wisig_runtime import make_loader


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl


PAPER_TABLE2_TASKS = [
    "1-1->8-8",
    "8-8->1-1",
    "19-2->1-1",
    "1-1->19-2",
    "20-1->2-1",
    "2-1->20-1",
    "7-14->2-19",
    "2-19->7-14",
    "1-19->2-19",
    "2-19->1-19",
    "14-7->7-7",
    "7-7->14-7",
]

PAPER_PREPROCESSING = {
    "dataset": "WiSig subset with six transmitters, twelve receivers, four days",
    "equalized": 1,
    "normalize": True,
    "out_len": 256,
    "crop_mode": "left",
    "input_shape": "[2,256] IQ",
}


class TargetUnlabeledDataset(Dataset):
    """Target-domain training view that exposes IQ only; labels remain evaluation-only."""

    def __init__(self, base: WiSigCompactDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        iq, *_rest = self.base[index]
        return {"iq": iq}


def collate_target_unlabeled(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {"iq": torch.stack([item["iq"] for item in batch], dim=0)}


def _resolve_one(labels: list[Any], token: Any, *, name: str) -> int:
    if isinstance(token, int) and 0 <= token < len(labels):
        return int(token)
    text = str(token)
    if text in labels:
        return labels.index(text)
    if text.lstrip("-").isdigit():
        idx = int(text)
        if 0 <= idx < len(labels):
            return idx
    raise ValueError(f"cannot resolve {name} {token!r} from {labels}")


def _resolve_day_group(labels: list[Any], token: str) -> list[int]:
    text = str(token).strip()
    if not text.startswith("d"):
        return [_resolve_one(labels, text, name="day")]
    digits = text[1:]
    if not digits:
        raise ValueError(f"day task token has no day ids: {token}")
    out = []
    for char in digits:
        if not char.isdigit():
            raise ValueError(f"day task token must contain digits after d: {token}")
        idx = int(char)
        if idx >= len(labels):
            raise ValueError(f"day index out of range in {token}: {idx}")
        out.append(idx)
    return out


def _parse_task(task: str) -> tuple[str, str]:
    if "->" not in task:
        raise ValueError(f"paper task must use source->target form: {task}")
    source, target = [part.strip() for part in task.split("->", 1)]
    if not source or not target:
        raise ValueError(f"paper task must include source and target: {task}")
    return source, target


def _tx_coverage(dataset: WiSigCompactDataset) -> set[int]:
    out = set()
    for item in dataset.index:
        if hasattr(item, "tx_i"):
            out.add(int(item.tx_i))
        else:
            out.add(int(item[0]))
    return out


def _apply_domain_sample_count(dataset: WiSigCompactDataset, total_samples: int | None) -> dict[str, Any] | None:
    if total_samples is None:
        return None
    target = int(total_samples)
    if target <= 0:
        raise ValueError("paper_domain_sample_count must be positive")
    if len(dataset.index) <= target:
        return {
            "requested": target,
            "before": len(dataset.index),
            "after": len(dataset.index),
            "applied": False,
            "policy": "balanced_front_by_tx_rx_day_eq",
        }

    groups: dict[tuple[int, int, int, int], list[Any]] = defaultdict(list)
    for item in dataset.index:
        groups[(int(item.tx_i), int(item.rx_i), int(item.day_i), int(item.eq_i))].append(item)
    group_keys = sorted(groups)
    base = target // len(group_keys)
    remainder = target % len(group_keys)
    selected = []
    for offset, key in enumerate(group_keys):
        take = base + (1 if offset < remainder else 0)
        bucket = groups[key]
        if take > len(bucket):
            raise ValueError(
                "paper_domain_sample_count cannot be balanced because a tx/rx/day/eq group is too small: "
                f"group={key}, requested={take}, available={len(bucket)}"
            )
        selected.extend(bucket[:take])
    dataset.index = selected
    return {
        "requested": target,
        "before": sum(len(items) for items in groups.values()),
        "after": len(dataset.index),
        "applied": True,
        "policy": "balanced_front_by_tx_rx_day_eq",
        "groups": len(group_keys),
        "base_per_group": base,
        "remainder_groups": remainder,
    }


def build_manysig_task_datasets(
    compact: dict[str, Any],
    *,
    task: str,
    max_samples_per_combo: int | None = None,
    paper_domain_sample_count: int | None = None,
    normalize: bool | None = None,
    crop_mode: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    tx_labels = list(compact.get("tx_list", []))
    rx_labels = list(compact.get("rx_list", []))
    day_labels = list(compact.get("capture_date_list", []))
    if len(tx_labels) != 6:
        raise ValueError("DADDA paper-faithful protocol expects six transmitters")
    if len(rx_labels) != 12:
        raise ValueError("DADDA paper-faithful protocol expects twelve receivers")
    if len(day_labels) != 4:
        raise ValueError("DADDA paper-faithful protocol expects four capture days")

    source_token, target_token = _parse_task(task)
    if source_token.startswith("d") or target_token.startswith("d"):
        if not (source_token.startswith("d") and target_token.startswith("d")):
            raise ValueError(f"day-domain task must use day tokens on both sides: {task}")
        task_type = "cross_day_control"
        source_rx = list(range(len(rx_labels)))
        target_rx = list(range(len(rx_labels)))
        source_days = _resolve_day_group(day_labels, source_token)
        target_days = _resolve_day_group(day_labels, target_token)
    else:
        task_type = "cross_receiver"
        source_rx = [_resolve_one(rx_labels, source_token, name="source receiver")]
        target_rx = [_resolve_one(rx_labels, target_token, name="target receiver")]
        source_days = list(range(len(day_labels)))
        target_days = list(range(len(day_labels)))

    common = {
        "out_len": PAPER_PREPROCESSING["out_len"],
        "crop_mode": str(crop_mode or PAPER_PREPROCESSING["crop_mode"]),
        "normalize": PAPER_PREPROCESSING["normalize"] if normalize is None else bool(normalize),
        "equalized": PAPER_PREPROCESSING["equalized"],
        "domain": "rx_day",
        "max_samples_per_combo": max_samples_per_combo,
        "sample_strategy": "front",
        "seed": int(seed),
    }
    source = WiSigCompactDataset(compact, rx_keep=source_rx, day_keep=source_days, **common)
    target = WiSigCompactDataset(compact, rx_keep=target_rx, day_keep=target_days, **common)
    source_domain_cap = _apply_domain_sample_count(source, paper_domain_sample_count)
    target_domain_cap = _apply_domain_sample_count(target, paper_domain_sample_count)
    if len(source) == 0 or len(target) == 0:
        raise ValueError(f"empty source/target dataset for DADDA task {task}")
    expected_tx = set(range(len(tx_labels)))
    source_tx = _tx_coverage(source)
    target_tx = _tx_coverage(target)
    if source_tx != expected_tx or target_tx != expected_tx:
        raise ValueError(
            "DADDA closed-set protocol requires source and target datasets to cover all six TX classes; "
            f"source={sorted(source_tx)}, target={sorted(target_tx)}"
        )
    return {
        "source": source,
        "target": target,
        "meta": {
            "task": task,
            "task_type": task_type,
            "source_receiver_ids": source_rx,
            "target_receiver_ids": target_rx,
            "source_receiver_labels": [rx_labels[i] for i in source_rx],
            "target_receiver_labels": [rx_labels[i] for i in target_rx],
            "source_day_ids": source_days,
            "target_day_ids": target_days,
            "source_day_labels": [day_labels[i] for i in source_days],
            "target_day_labels": [day_labels[i] for i in target_days],
            "source_tx_ids": sorted(source_tx),
            "target_tx_ids": sorted(target_tx),
            "tx_labels": tx_labels,
            "target_label_role": "hidden_for_UDA_training_available_for_final_accuracy_only",
            "preprocessing": {
                **dict(PAPER_PREPROCESSING),
                "normalize": common["normalize"],
                "crop_mode": common["crop_mode"],
            },
            "paper_domain_sample_count": paper_domain_sample_count,
            "source_domain_sample_cap": source_domain_cap,
            "target_domain_sample_cap": target_domain_cap,
            "seed": int(seed),
        },
    }


def build_manysig_task_loaders(
    compact_or_path: dict[str, Any] | str | Path,
    *,
    task: str,
    batch_size: int,
    max_samples_per_combo: int | None = None,
    paper_domain_sample_count: int | None = None,
    normalize: bool | None = None,
    crop_mode: str | None = None,
    seed: int = 0,
    num_workers: int = 0,
) -> dict[str, Any]:
    compact = load_wisig_compact_pkl(str(compact_or_path)) if isinstance(compact_or_path, (str, Path)) else compact_or_path
    built = build_manysig_task_datasets(
        compact,
        task=task,
        max_samples_per_combo=max_samples_per_combo,
        paper_domain_sample_count=paper_domain_sample_count,
        normalize=normalize,
        crop_mode=crop_mode,
        seed=seed,
    )
    return {
        "source": make_loader(built["source"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "target_train": DataLoader(
            TargetUnlabeledDataset(built["target"]),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            collate_fn=collate_target_unlabeled,
            drop_last=False,
        ),
        "target_eval": make_loader(built["target"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "meta": built["meta"],
    }
