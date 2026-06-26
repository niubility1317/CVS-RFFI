from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_wisig import load_wisig_compact_pkl, make_wisig_drift_day1_split, make_wisig_trainval_test_by_day_rx


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return None
        out: list[Any] = []
        for token in value.split(","):
            token = token.strip()
            if token.lstrip("-").isdigit():
                out.append(int(token))
            else:
                out.append(token)
        return out
    return [value]


def sample_to_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    x, y, d, meta = sample
    return {"iq": x, "label": int(y), "domain": int(d), "meta": meta}


def collate_wisig(batch: Iterable[Any]) -> dict[str, Any]:
    items = [sample_to_dict(item) for item in batch]
    return {
        "iq": torch.stack([item["iq"] for item in items], dim=0),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "domain": torch.tensor([item["domain"] for item in items], dtype=torch.long),
        "meta": [item.get("meta", {}) for item in items],
    }


def make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_wisig,
        drop_last=False,
    )


def tx_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
