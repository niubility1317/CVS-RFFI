from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from paper_reproduction.common.wisig_runtime import make_loader


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl


PAPER_PREPROCESSING = {
    "input_source": "WiSig ManySig compact pkl after synchronization and preamble extraction",
    "equalized": 1,
    "normalize": True,
    "normalization": "RMS/power normalization",
    "cfo_policy": "preserved",
    "out_len": 256,
    "crop_mode": "left",
    "representation": "[2,256] IQ tensor, accepted by model as [B,2,256] and converted to [B,1,256,2]",
}


def _resolve_indices(labels: list[Any], requested: list[Any] | None, *, name: str) -> list[int]:
    if requested is None:
        return list(range(len(labels)))
    out: list[int] = []
    for item in requested:
        if isinstance(item, int):
            if item < 0 or item >= len(labels):
                raise ValueError(f"{name} index out of range: {item}")
            out.append(int(item))
            continue
        if item in labels:
            out.append(labels.index(item))
            continue
        text = str(item)
        if text.lstrip("-").isdigit():
            idx = int(text)
            if 0 <= idx < len(labels):
                out.append(idx)
                continue
        raise ValueError(f"cannot resolve {name} {item!r} from {labels}")
    return out


def _labels_for(indices: list[int], labels: list[Any]) -> list[Any]:
    return [labels[i] if i < len(labels) else i for i in indices]


def validate_preprocessing_contract(config: dict[str, Any]) -> dict[str, Any]:
    equalized = int(config.get("equalized", PAPER_PREPROCESSING["equalized"]))
    out_len = int(config.get("out_len", PAPER_PREPROCESSING["out_len"]))
    crop_mode = str(config.get("crop_mode", PAPER_PREPROCESSING["crop_mode"])).lower()
    normalize = bool(config.get("normalize", PAPER_PREPROCESSING["normalize"]))
    domain = str(config.get("domain", "rx")).lower()
    if equalized != 1:
        raise ValueError("paper-faithful preprocessing requires equalized=1")
    if out_len != 256:
        raise ValueError("paper-faithful preprocessing requires first 256 IQ samples")
    if crop_mode != "left":
        raise ValueError("paper-faithful preprocessing requires crop_mode='left' for first256 IQ samples")
    if not normalize:
        raise ValueError("paper-faithful preprocessing requires power/RMS normalization")
    if domain != "rx":
        raise ValueError("paper-faithful cross-receiver UDA requires receiver domain labels")
    checked = dict(PAPER_PREPROCESSING)
    checked["domain"] = domain
    return checked


def build_manysig_receiver_uda_datasets(
    compact: dict[str, Any],
    *,
    source_receivers: list[Any],
    target_receivers: list[Any],
    seed: int = 0,
    max_samples_per_combo: int | None = None,
) -> dict[str, Any]:
    preprocessing = validate_preprocessing_contract(
        {"equalized": 1, "out_len": 256, "crop_mode": "left", "normalize": True, "domain": "rx"}
    )
    tx_labels = list(compact.get("tx_list", []))
    rx_labels = list(compact.get("rx_list", []))
    day_labels = list(compact.get("capture_date_list", []))
    if len(tx_labels) != 6:
        raise ValueError("paper-faithful ManySig expects exactly 6 transmitters")
    if len(rx_labels) != 12:
        raise ValueError("paper-faithful ManySig expects exactly 12 receivers")
    if len(day_labels) != 4:
        raise ValueError("paper-faithful ManySig expects exactly 4 capture days")
    source_rx_idx = _resolve_indices(rx_labels, source_receivers, name="source receiver")
    target_rx_idx = _resolve_indices(rx_labels, target_receivers, name="target receiver")
    overlap = sorted(set(source_rx_idx).intersection(target_rx_idx))
    if overlap:
        raise ValueError(f"source and target receivers must be disjoint, overlap={overlap}")
    common = {
        "out_len": 256,
        "crop_mode": "left",
        "normalize": True,
        "equalized": 1,
        "domain": "rx",
        "max_samples_per_combo": max_samples_per_combo,
        "sample_strategy": "front",
        "seed": int(seed),
    }
    source = WiSigCompactDataset(compact, rx_keep=source_rx_idx, **common)
    target = WiSigCompactDataset(compact, rx_keep=target_rx_idx, **common)
    return {
        "source": source,
        "target": target,
        "meta": {
            "preprocessing": preprocessing,
            "source_receiver_ids": source_rx_idx,
            "target_receiver_ids": target_rx_idx,
            "source_receiver_labels": _labels_for(source_rx_idx, rx_labels),
            "target_receiver_labels": _labels_for(target_rx_idx, rx_labels),
            "target_label_role": "hidden_for_UDA_available_only_for_eval_or_optional_finetune",
            "seed": int(seed),
        },
    }


def build_manysig_receiver_uda_loaders(
    config: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, Any]:
    pkl_path = config.get("manysig_pkl")
    if not pkl_path:
        raise ValueError("config must provide manysig_pkl for real ManySig loader construction")
    compact = load_wisig_compact_pkl(str(pkl_path))
    datasets = build_manysig_receiver_uda_datasets(
        compact,
        source_receivers=list(config.get("source_receiver_ids") or config.get("source_receiver_labels") or []),
        target_receivers=list(config.get("target_receiver_ids") or config.get("target_receiver_labels") or []),
        seed=int(config.get("seed", 0)),
        max_samples_per_combo=config.get("max_samples_per_combo"),
    )
    return {
        "source": make_loader(datasets["source"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "target": make_loader(datasets["target"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "meta": datasets["meta"],
    }
