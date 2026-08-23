"""Real-checkpoint support-only smoke for structured late-block adaptation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from cvsrffi.stage2_structured_late_block_adaptation import (
    StructuredLateBlockConfig,
    adapt_on_target_support_with_frozen_prototypes,
)
from cvsrffi.stage2_structured_late_block_runner import (
    _PROTOTYPE_PAYLOAD_ALLOWLIST,
    _SUPPORT_PAYLOAD_ALLOWLIST,
    _integer_tensor,
    _load_frozen_checkpoint,
    _load_npz,
    _prototype_tensors,
    _received_iq_tensor,
    _validate_exact_keys,
)


_CONFIG_ALLOWLIST = frozenset(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or frozenset(config) != _CONFIG_ALLOWLIST:
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
        config=StructuredLateBlockConfig(
            candidate=str(config["candidate"]),
            steps=int(config["steps"]),
            learning_rate=float(config["learning_rate"]),
        ),
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
