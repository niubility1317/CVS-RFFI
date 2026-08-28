#!/usr/bin/env python
"""Build ADV3B02 class prototypes from one verified canonical support row.

The bridge is intentionally support-only.  It accepts the existing minimal
support NPZ plus the existing support-only target-row export audit, embeds the
received IQ with the frozen ADV3B02 identity path, and writes the exact
``{prototypes, class_ids}`` payload consumed by the existing no-query smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from cvsrffi.stage2_structured_late_block_adaptation import _identity_features
from cvsrffi.stage2_structured_late_block_runner import (
    _SUPPORT_PAYLOAD_ALLOWLIST,
    _integer_tensor,
    _load_frozen_checkpoint,
    _load_npz,
    _received_iq_tensor,
)
from cvsrffi.stage2_target_prototype_bank import encode_support_prototypes


EXPECTED_CAPSULE_ID = (
    "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2"
)
EXPECTED_SPLIT_ID = (
    "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25"
)
EXPECTED_CHECKPOINT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/"
    "best_joint_safe_ssdg.pth"
)
EXPECTED_SCENE = "leo_clear_weak"
EXPECTED_RECEIVER = "1-1"
EXPECTED_K_SHOT = 20
REGISTERED_CLASS_IDS = tuple(range(26))
TARGET_NEW_CLASS_IDS = tuple(range(6, 26))

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
_SUPPORT_AUDIT_ALLOWLIST = frozenset(
    {
        "schema",
        "mode",
        "k_shot",
        "support_input_rows",
        "support_output_rows",
        "support_class_count",
        "support_class_ids",
        "support_per_class_counts",
        "support_selected_ids",
        "support_ids_preserved",
        "query_input_opened",
        "query_input_rows",
        "query_output_rows",
        "query_ids",
        "query_ids_preserved",
        "query_truth_opened",
        "query_role_opened",
    }
)


class SupportPrototypeBuildError(ValueError):
    """Raised when the frozen canonical no-query row drifts."""


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise SupportPrototypeBuildError("config must be a string-keyed mapping")
    actual = frozenset(config)
    if actual != _CONFIG_ALLOWLIST:
        raise SupportPrototypeBuildError(
            "config allowlist mismatch: "
            f"missing={sorted(_CONFIG_ALLOWLIST - actual)}, "
            f"extra={sorted(actual - _CONFIG_ALLOWLIST)}"
        )
    resolved = dict(config)
    expected_values = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": EXPECTED_CAPSULE_ID,
        "split_id": EXPECTED_SPLIT_ID,
        "checkpoint_path": EXPECTED_CHECKPOINT,
        "k_shot": EXPECTED_K_SHOT,
    }
    for field, expected in expected_values.items():
        value = int(resolved[field]) if field == "k_shot" else str(resolved[field])
        if value != expected:
            raise SupportPrototypeBuildError(
                f"{field} mismatch: expected={expected!r}, observed={value!r}"
            )
    for field in ("support_path", "prototype_path"):
        if not str(resolved[field]).strip():
            raise SupportPrototypeBuildError(f"{field} must be nonempty")
    if str(resolved["candidate"]) not in {"freq_f3_proj", "time_t3"}:
        raise SupportPrototypeBuildError("candidate is not supported by the smoke")
    if int(resolved["steps"]) < 1:
        raise SupportPrototypeBuildError("steps must be positive")
    if float(resolved["learning_rate"]) <= 0.0:
        raise SupportPrototypeBuildError("learning_rate must be positive")
    int(resolved["seed"])
    return resolved


def _validate_row_binding(
    config: Mapping[str, Any],
    *,
    scene: str,
    receiver: str,
) -> None:
    if str(scene) != EXPECTED_SCENE:
        raise SupportPrototypeBuildError(
            f"scene mismatch: expected={EXPECTED_SCENE}, observed={scene}"
        )
    if str(receiver) != EXPECTED_RECEIVER:
        raise SupportPrototypeBuildError(
            f"receiver mismatch: expected={EXPECTED_RECEIVER}, observed={receiver}"
        )
    receiver_marker = f"rx{EXPECTED_RECEIVER}"
    for field in ("support_path", "prototype_path"):
        name = Path(str(config[field])).name
        required_markers = (EXPECTED_SCENE, receiver_marker, "k20")
        if any(marker not in name for marker in required_markers):
            raise SupportPrototypeBuildError(
                f"{field} is not bound to the frozen scene/receiver/K row"
            )


def _load_support_audit(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise SupportPrototypeBuildError(
            f"support export audit is missing or invalid: {resolved}"
        ) from exc
    if not isinstance(payload, dict) or frozenset(payload) != _SUPPORT_AUDIT_ALLOWLIST:
        raise SupportPrototypeBuildError("support export audit allowlist mismatch")
    required = {
        "schema": "cvs.stage2.target_row_export.v1",
        "mode": "support_only_no_query_smoke",
        "k_shot": EXPECTED_K_SHOT,
        "support_input_rows": EXPECTED_K_SHOT * len(REGISTERED_CLASS_IDS),
        "support_output_rows": EXPECTED_K_SHOT * len(REGISTERED_CLASS_IDS),
        "support_class_count": len(REGISTERED_CLASS_IDS),
        "support_ids_preserved": True,
        "query_input_opened": False,
        "query_input_rows": 0,
        "query_output_rows": 0,
        "query_ids": [],
        "query_ids_preserved": False,
        "query_truth_opened": False,
        "query_role_opened": False,
    }
    failed = [field for field, expected in required.items() if payload[field] != expected]
    if failed:
        raise SupportPrototypeBuildError(
            f"support-only audit contract failed: {failed}"
        )
    class_ids = payload["support_class_ids"]
    if class_ids != list(REGISTERED_CLASS_IDS):
        raise SupportPrototypeBuildError("support audit class registry mismatch")
    expected_counts = {
        str(class_id): EXPECTED_K_SHOT for class_id in REGISTERED_CLASS_IDS
    }
    if payload["support_per_class_counts"] != expected_counts:
        raise SupportPrototypeBuildError("support audit class K counts mismatch")
    physical_ids = payload["support_selected_ids"]
    expected_rows = EXPECTED_K_SHOT * len(REGISTERED_CLASS_IDS)
    if (
        not isinstance(physical_ids, list)
        or len(physical_ids) != expected_rows
        or any(not isinstance(value, str) or not value for value in physical_ids)
        or len(set(physical_ids)) != expected_rows
    ):
        raise SupportPrototypeBuildError(
            "support physical IDs must be complete, nonempty, and globally unique"
        )
    return payload


def _validate_support_payload(
    support_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        payload = _load_npz(
            support_path,
            allowed=_SUPPORT_PAYLOAD_ALLOWLIST,
            label="support",
        )
    except ValueError as exc:
        raise SupportPrototypeBuildError(str(exc)) from exc
    received_iq = _received_iq_tensor(payload["received_iq"], label="support")
    labels = _integer_tensor(payload["support_labels"], label="support_labels")
    expected_rows = EXPECTED_K_SHOT * len(REGISTERED_CLASS_IDS)
    if received_iq.shape[0] != expected_rows or labels.shape[0] != expected_rows:
        raise SupportPrototypeBuildError(
            f"support row count must be exactly {expected_rows}"
        )
    class_ids, counts = torch.unique(labels, sorted=True, return_counts=True)
    if class_ids.tolist() != list(REGISTERED_CLASS_IDS):
        raise SupportPrototypeBuildError("support class registry mismatch")
    if counts.tolist() != [EXPECTED_K_SHOT] * len(REGISTERED_CLASS_IDS):
        raise SupportPrototypeBuildError("support must be exactly K-shot per class")
    return received_iq, labels


def _write_prototypes_new(
    path: Path,
    *,
    prototypes: np.ndarray,
    class_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            prototypes=np.ascontiguousarray(prototypes, dtype=np.float32),
            class_ids=np.ascontiguousarray(class_ids, dtype=np.int64),
        )


def build_support_prototypes(
    config: Mapping[str, Any],
    *,
    support_audit_path: str | Path,
    scene: str,
    receiver: str,
    device: str | torch.device,
) -> dict[str, Any]:
    """Create one 26-class normalized support-mean prototype payload."""

    resolved = _validate_config(config)
    _validate_row_binding(resolved, scene=scene, receiver=receiver)
    output_path = Path(str(resolved["prototype_path"]))
    if output_path.exists():
        raise SupportPrototypeBuildError(
            f"prototype output already exists: {output_path}"
        )
    _load_support_audit(support_audit_path)
    support_iq, support_labels = _validate_support_payload(resolved["support_path"])

    target_device = torch.device(device)
    seed = int(resolved["seed"])
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _load_frozen_checkpoint(
        resolved["checkpoint_path"],
        device=target_device,
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise SupportPrototypeBuildError("checkpoint embedding model must be frozen")

    rows = support_iq.to(device=target_device, dtype=torch.float32)
    with torch.inference_mode():
        features = _identity_features(model, rows)
    if (
        not torch.is_tensor(features)
        or features.ndim != 2
        or features.shape[0] != rows.shape[0]
        or features.shape[1] < 1
        or not torch.isfinite(features).all()
    ):
        raise SupportPrototypeBuildError(
            "checkpoint z_id embeddings must be finite row-aligned features"
        )

    class_values = [
        "cls_"
        + hashlib.sha256(
            f"canonical-union-class-{class_id}".encode("ascii")
        ).hexdigest()
        for class_id in REGISTERED_CLASS_IDS
    ]
    class_handle_by_id = dict(zip(REGISTERED_CLASS_IDS, class_values))
    label_values = [
        class_handle_by_id[int(value)] for value in support_labels.tolist()
    ]
    bank = encode_support_prototypes(
        features.detach().cpu().numpy(),
        label_values,
        class_values,
        storage_format="fp32",
        r0=0.0,
    )
    prototypes = bank.vectors
    if prototypes is None or prototypes.shape[0] != len(REGISTERED_CLASS_IDS):
        raise SupportPrototypeBuildError("support prototype aggregation failed")
    class_ids = np.asarray(REGISTERED_CLASS_IDS, dtype=np.int64)
    _write_prototypes_new(
        output_path,
        prototypes=np.asarray(prototypes),
        class_ids=class_ids,
    )
    return {
        "status": "SUPPORT_ONLY_PROTOTYPE_BUILD_PASS",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": EXPECTED_CAPSULE_ID,
        "split_id": EXPECTED_SPLIT_ID,
        "scene": EXPECTED_SCENE,
        "receiver": EXPECTED_RECEIVER,
        "k_shot": EXPECTED_K_SHOT,
        "support_rows": int(rows.shape[0]),
        "registered_class_ids": list(REGISTERED_CLASS_IDS),
        "target_new_class_ids": list(TARGET_NEW_CLASS_IDS),
        "target_new_prototypes_from_own_support": True,
        "support_physical_ids_unique": True,
        "feature_dim": int(prototypes.shape[1]),
        "output_members": ["class_ids", "prototypes"],
        "prototype_path": str(output_path),
        "query_opened": False,
        "source_opened": False,
        "clean_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    audit = build_support_prototypes(
        config,
        support_audit_path=args.support_audit,
        scene=args.scene,
        receiver=args.receiver,
        device=args.device,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SupportPrototypeBuildError",
    "build_support_prototypes",
    "main",
]
