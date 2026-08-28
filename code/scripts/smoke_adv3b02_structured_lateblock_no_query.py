"""Real-checkpoint support-only smoke for structured late-block adaptation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from cvsrffi.stage2_structured_late_block_adaptation import (
    StructuredLateBlockConfig,
    adapt_on_target_support_with_frozen_prototypes,
)
from cvsrffi.stage2_structured_late_block_runner import (
    _PROTOTYPE_PAYLOAD_ALLOWLIST,
    _SUPPORT_PAYLOAD_ALLOWLIST,
    _load_frozen_checkpoint,
    _load_npz,
    _validate_exact_keys,
)


_LEGACY_CONFIG_ALLOWLIST = frozenset(
    {
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "checkpoint_path",
        "support_path",
        "prototype_path",
        "candidate",
        "steps",
        "learning_rate",
        "seed",
        "k_shot",
    }
)
_CONFIG_ALLOWLIST = _LEGACY_CONFIG_ALLOWLIST | frozenset(
    {"min_trainable_fraction", "max_trainable_fraction"}
)


def _tensor_from_numpy_buffer(
    value: "np.ndarray",
    *,
    numpy_dtype: "np.dtype[Any]",
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    array = np.ascontiguousarray(value, dtype=numpy_dtype)
    return torch.frombuffer(
        bytearray(array.tobytes(order="C")),
        dtype=torch_dtype,
    ).clone().reshape(array.shape)


def _received_iq_tensor(value: "np.ndarray", *, label: str) -> torch.Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] != 2
        or array.shape[2] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} received_iq must be finite nonempty [N,2,L]")
    return _tensor_from_numpy_buffer(
        array,
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )


def _integer_tensor(value: "np.ndarray", *, label: str) -> torch.Tensor:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape[0] < 1
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{label} must be a nonempty integer vector")
    return _tensor_from_numpy_buffer(
        array,
        numpy_dtype=np.dtype(np.int64),
        torch_dtype=torch.int64,
    )


def _prototype_tensors(
    payload: Mapping[str, "np.ndarray"],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_exact_keys(
        payload,
        _PROTOTYPE_PAYLOAD_ALLOWLIST,
        label="prototype",
    )
    array = np.asarray(payload["prototypes"])
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] < 1
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError("prototype array must be a finite nonempty 2D matrix")
    class_ids = _integer_tensor(payload["class_ids"], label="prototype class_ids")
    if class_ids.shape[0] != array.shape[0]:
        raise ValueError("prototype matrix and class_ids must align")
    if torch.unique(class_ids).numel() != class_ids.numel():
        raise ValueError("prototype class_ids must be unique")
    prototypes = _tensor_from_numpy_buffer(
        array,
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )
    prototypes.requires_grad_(False)
    return prototypes, class_ids


def _structured_config(config: Mapping[str, Any]) -> StructuredLateBlockConfig:
    return StructuredLateBlockConfig(
        candidate=str(config["candidate"]),
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        min_trainable_fraction=float(config.get("min_trainable_fraction", 0.05)),
        max_trainable_fraction=float(config.get("max_trainable_fraction", 0.15)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or frozenset(config) not in {
        _LEGACY_CONFIG_ALLOWLIST,
        _CONFIG_ALLOWLIST,
    }:
        raise ValueError("smoke config allowlist mismatch")
    if config["protocol_schema"] != "p2_min_v1":
        raise ValueError("protocol_schema must be p2_min_v1")
    if config["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("phase2_data_status must be VALIDATED_ONCE")
    if args.output_json.exists():
        raise FileExistsError(f"smoke output already exists: {args.output_json}")

    device = torch.device(args.device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _load_frozen_checkpoint(config["checkpoint_path"], device=device)

    support_payload = _load_npz(
        config["support_path"],
        allowed=_SUPPORT_PAYLOAD_ALLOWLIST,
        label="support",
    )
    _validate_exact_keys(
        support_payload,
        _SUPPORT_PAYLOAD_ALLOWLIST,
        label="support",
    )
    support_iq = _received_iq_tensor(
        support_payload["received_iq"], label="support"
    )
    support_labels = _integer_tensor(
        support_payload["support_labels"], label="support_labels"
    )
    if support_iq.shape[0] != support_labels.shape[0]:
        raise ValueError("support IQ and labels must align")
    _support_classes, support_counts = torch.unique(
        support_labels, sorted=True, return_counts=True
    )
    if torch.any(support_counts != int(config["k_shot"])):
        raise ValueError("support payload must contain exactly K-shot rows per class")

    prototype_payload = _load_npz(
        config["prototype_path"],
        allowed=_PROTOTYPE_PAYLOAD_ALLOWLIST,
        label="prototype",
    )
    _validate_exact_keys(
        prototype_payload,
        _PROTOTYPE_PAYLOAD_ALLOWLIST,
        label="prototype",
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    prototype_before = prototypes.clone()
    audit = adapt_on_target_support_with_frozen_prototypes(
        model,
        support_iq,
        support_labels,
        frozen_prototypes=prototypes,
        prototype_class_ids=class_ids,
        context={
            "protocol_schema": config["protocol_schema"],
            "phase2_data_status": config["phase2_data_status"],
            "capsule_id": config["capsule_id"],
            "split_id": config["split_id"],
        },
        config=_structured_config(config),
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("smoke did not return a fully frozen model")
    if not torch.equal(prototypes, prototype_before) or prototypes.requires_grad:
        raise RuntimeError("smoke changed immutable class prototypes")

    receipt = asdict(audit)
    receipt.update(
        {
            "status": "REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS",
            "checkpoint_load_strict": True,
            "query_opened": False,
            "source_opened": False,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
