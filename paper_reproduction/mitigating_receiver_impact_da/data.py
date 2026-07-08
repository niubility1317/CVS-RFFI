from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from paper_reproduction.common.wisig_runtime import make_loader


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl


PAPER_PREPROCESSING = {
    "input_source": "WiSig ManySig compact pkl after packet detection, channel equalization, and normalization",
    "equalized": 1,
    "normalize": True,
    "out_len": 256,
    "crop_mode": "left",
    "representation": "[2,256] IQ tensor accepted by the 1-D ResNet feature extractor",
}


def _resolve_one(labels: list[Any], token: Any, *, name: str) -> int:
    if isinstance(token, int):
        if 0 <= token < len(labels):
            return int(token)
        raise ValueError(f"{name} index out of range: {token}")
    if token in labels:
        return labels.index(token)
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
    out: list[int] = []
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
    source, target = [part.strip() for part in str(task).split("->", 1)]
    if not source or not target:
        raise ValueError(f"paper task must include both domains: {task}")
    return source, target


def build_manysig_task_datasets(
    compact: dict[str, Any],
    *,
    task: str,
    max_samples_per_combo: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    tx_labels = list(compact.get("tx_list", []))
    rx_labels = list(compact.get("rx_list", []))
    day_labels = list(compact.get("capture_date_list", []))
    if len(tx_labels) != 6:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 6 transmitters")
    if len(rx_labels) != 12:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 12 receivers")
    if len(day_labels) != 4:
        raise ValueError("IoTJ 2024 paper-faithful ManySig protocol expects 4 capture days")

    source_token, target_token = _parse_task(task)
    if source_token.startswith("d") or target_token.startswith("d"):
        if not (source_token.startswith("d") and target_token.startswith("d")):
            raise ValueError(f"cross-day task must use day tokens on both sides: {task}")
        task_type = "cross_day"
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
        "crop_mode": PAPER_PREPROCESSING["crop_mode"],
        "normalize": PAPER_PREPROCESSING["normalize"],
        "equalized": PAPER_PREPROCESSING["equalized"],
        "domain": "rx_day",
        "max_samples_per_combo": max_samples_per_combo,
        "sample_strategy": "front",
        "seed": int(seed),
    }
    source = WiSigCompactDataset(compact, rx_keep=source_rx, day_keep=source_days, **common)
    target = WiSigCompactDataset(compact, rx_keep=target_rx, day_keep=target_days, **common)
    if len(source) == 0 or len(target) == 0:
        raise ValueError(f"empty source/target dataset for paper task {task}")
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
            "target_label_role": "hidden_for_UDA_training_available_for_final_accuracy",
            "preprocessing": dict(PAPER_PREPROCESSING),
            "seed": int(seed),
        },
    }


def build_manysig_task_loaders(
    compact_or_path: dict[str, Any] | str | Path,
    *,
    task: str,
    batch_size: int,
    max_samples_per_combo: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
) -> dict[str, Any]:
    compact = load_wisig_compact_pkl(str(compact_or_path)) if isinstance(compact_or_path, (str, Path)) else compact_or_path
    built = build_manysig_task_datasets(
        compact,
        task=task,
        max_samples_per_combo=max_samples_per_combo,
        seed=seed,
    )
    return {
        "source": make_loader(built["source"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "target_train": make_loader(built["target"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "target_eval": make_loader(built["target"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "meta": built["meta"],
    }
