"""Minimal truth-free D127 S0 in-memory prediction entry.

The caller owns the already frozen 18-row matrix, decoded Phase1 asset, model,
and K-specific qKNN locks.  This module performs no data selection, truth
loading, scoring, tuning, or server orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import stage2_d127_checkpoint_hooks as hooks
from . import stage2_d127_da_candidates as da
from . import stage2_d127_joint_screen as joint
from . import stage2_zid_student_t_qknn as qknn


SCHEMA = "cvs.stage2.d127.s0.truth_free_prediction.v1"
LOCAL_WORKER_SCHEMA = "cvs.stage2.d127.s0.local_candidate_worker.v1"
ROW_COUNT = 18
ALLOWED_K = (1, 5)
ARM_IDS = (joint.M0, joint.M_DA, joint.M_L92, joint.M_JOINT)
FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {"truth", "query_truth", "role", "roles", "quota", "class_quota", "global_reassignment"}
)
_FORBIDDEN_NORMALIZED_KEYS = frozenset(
    {"truth", "querytruth", "role", "roles", "quota", "classquota", "globalreassignment"}
)
CANDIDATE_IDS = (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C)


class D127S0EntryError(ValueError):
    """Raised when the frozen S0 input or truth-free output drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127S0EntryError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    raise D127S0EntryError(f"non-serializable S0 value: {type(value).__name__}")


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key)
            normalized = "".join(character for character in text.lower() if character.isalnum())
            if normalized in _FORBIDDEN_NORMALIZED_KEYS:
                found.add(text)
            found.update(_forbidden_keys(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


@dataclass(frozen=True, slots=True)
class D127S0Row:
    """One caller-frozen target-development row without query truth."""

    row_id: str
    receiver_id: str
    k_shot: int
    scene: str
    support_iq: Tensor
    query_iq: Tensor
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]
    opaque_query_ids: tuple[str, ...]
    qknn_lock: qknn.Phase1ZIDStudentTLock

    def __post_init__(self) -> None:
        _require(bool(self.row_id) and bool(self.receiver_id) and bool(self.scene), "row identity fields must be nonempty")
        _require(self.k_shot in ALLOWED_K, "S0 allows only K1/K5")
        _require(type(self.qknn_lock) is qknn.Phase1ZIDStudentTLock, "row requires exact qKNN lock")
        _require(self.qknn_lock.active_k == self.k_shot, "row K/qKNN lock drift")
        _require(isinstance(self.support_iq, Tensor) and isinstance(self.query_iq, Tensor), "row IQ must be tensors")
        _require(self.support_iq.ndim >= 2 and self.query_iq.ndim >= 2, "row IQ layout is invalid")
        _require(self.support_iq.device == self.query_iq.device, "support/query IQ device drift")
        labels = tuple(str(value) for value in self.support_labels)
        classes = tuple(str(value) for value in self.registered_classes)
        query_ids = tuple(str(value) for value in self.opaque_query_ids)
        _require(len(classes) >= 2 and len(set(classes)) == len(classes), "registered classes must be unique")
        _require(len(labels) == len(self.support_iq), "support label count drift")
        _require(len(query_ids) == len(self.query_iq) and len(set(query_ids)) == len(query_ids), "opaque query ID count/uniqueness drift")
        _require(all(label in classes for label in labels), "support label is outside registry")
        counts = tuple(labels.count(label) for label in classes)
        _require(all(count == self.k_shot for count in counts), "support must be balanced registered-class K-shot")
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "opaque_query_ids", query_ids)


def _validate_rows(rows: Sequence[D127S0Row]) -> tuple[D127S0Row, ...]:
    frozen = tuple(rows)
    _require(len(frozen) == ROW_COUNT and all(type(row) is D127S0Row for row in frozen), "S0 requires exactly 18 typed rows")
    _require(len({row.row_id for row in frozen}) == ROW_COUNT, "S0 row IDs must be unique")
    expected_order = tuple(sorted(frozen, key=lambda row: (row.receiver_id, row.k_shot, row.scene, row.row_id)))
    _require(frozen == expected_order, "S0 rows must retain the frozen lexical order")
    receivers = tuple(sorted({row.receiver_id for row in frozen}))
    scenes = tuple(sorted({row.scene for row in frozen}))
    _require(len(receivers) == 3 and len(scenes) == 3, "S0 requires 3 receivers and 3 scenes")
    expected = {(receiver, k, scene) for receiver in receivers for k in ALLOWED_K for scene in scenes}
    actual = {(row.receiver_id, row.k_shot, row.scene) for row in frozen}
    _require(actual == expected, "S0 receiver/K/scene coverage drift")
    return frozen


def _asset_candidate(asset: da.FSRGAsset | da.RDHAAsset) -> str:
    if isinstance(asset, da.FSRGAsset):
        return asset.candidate_id
    if isinstance(asset, da.RDHAAsset):
        return da.CANDIDATE_C
    raise D127S0EntryError("S0 requires one decoded D127 asset")


def _asset_payload_bytes(asset: da.FSRGAsset | da.RDHAAsset) -> int:
    if isinstance(asset, da.FSRGAsset):
        return 4 * int(asset.dimension) + 14
    if isinstance(asset, da.RDHAAsset):
        return 1328
    raise D127S0EntryError("unsupported D127 asset")


def _arm_payload(arm: joint.D127JointArmPrediction) -> dict[str, Any]:
    return {
        "arm_id": arm.arm_id,
        "representation": arm.representation,
        "head": arm.head,
        "classes": list(arm.classes),
        "logits": np.asarray(arm.logits, dtype=np.float32).tolist(),
        "predictions": list(arm.predictions),
        "receipt": _jsonable(arm.receipt),
    }


def _support_label_ids(row: D127S0Row) -> Tensor:
    class_to_index = {label: index for index, label in enumerate(row.registered_classes)}
    return torch.tensor(
        [class_to_index[label] for label in row.support_labels],
        dtype=torch.int64,
        device=row.support_iq.device,
    )


def _run_d127_s0_candidate_worker(
    *,
    model: nn.Module,
    candidate_id: str,
    asset: da.FSRGAsset | da.RDHAAsset,
    rows: Sequence[D127S0Row],
) -> dict[str, Any]:
    """Generate one candidate's complete 18-row truth-free S0 predictions."""

    frozen_rows = _validate_rows(rows)
    _require(candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C), "unknown D127 candidate")
    _require(_asset_candidate(asset) == candidate_id, "candidate/asset binding drift")
    outputs: list[dict[str, Any]] = []
    adapter_macs_per_sample: set[int] = set()
    total_adapter_macs = 0
    total_backbone_forwards = 0
    total_query_rows = 0

    for row in frozen_rows:
        materialized = hooks.materialize_d127_candidate(
            model,
            row.support_iq,
            _support_label_ids(row),
            row.query_iq,
            asset=asset,
        )
        four = joint.run_d127_joint_four_arm(
            base_support_zid=materialized.base_cache.support_zid,
            adapted_support_zid=materialized.adapted_cache.support_zid,
            base_query_zid=materialized.base_cache.query_zid,
            adapted_query_zid=materialized.adapted_cache.query_zid,
            support_labels=row.support_labels,
            registered_classes=row.registered_classes,
            opaque_query_ids=row.opaque_query_ids,
            qknn_lock=row.qknn_lock,
        )
        _require(tuple(arm.arm_id for arm in four.arms) == ARM_IDS, "four-arm output drift")
        resource = materialized.state.receipt
        _require(resource.protocol_closed, "DA state protocol counters are not closed")
        _require(four.receipt.get("active_k") == row.k_shot, "four-arm row K drift")
        adapter_macs_per_sample.add(int(resource.adapter_macs_per_sample))
        total_adapter_macs += int(resource.adapter_macs_per_sample) * (
            len(row.support_iq) + len(row.query_iq)
        )
        total_backbone_forwards += int(materialized.hook_receipt.total_id_backbone_forwards)
        total_query_rows += len(row.query_iq)
        outputs.append(
            {
                "row_id": row.row_id,
                "receiver_id": row.receiver_id,
                "k_shot": row.k_shot,
                "scene": row.scene,
                "opaque_query_ids": list(row.opaque_query_ids),
                "arms": {arm.arm_id: _arm_payload(arm) for arm in four.arms},
                "joint_receipt": _jsonable(four.receipt),
                "hook_receipt": _jsonable(materialized.hook_receipt.as_dict()),
                "da_resource": _jsonable(resource.as_dict()),
            }
        )

    _require(len(adapter_macs_per_sample) == 1, "candidate adapter MAC changed across rows")
    payload: dict[str, Any] = {
        "schema": LOCAL_WORKER_SCHEMA,
        "candidate_id": candidate_id,
        "evaluation_scope": "LOCAL_CANDIDATE_WORKER_NON_PUBLISHABLE",
        "truth_loaded": False,
        "row_count": ROW_COUNT,
        "rows_complete": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "resource": {
            "asset_numeric_payload_bytes": _asset_payload_bytes(asset),
            "adapter_macs_per_sample": next(iter(adapter_macs_per_sample)),
            "total_adapter_macs_support_plus_query": total_adapter_macs,
            "total_id_backbone_forwards": total_backbone_forwards,
            "total_query_rows": total_query_rows,
        },
        "rows": outputs,
    }
    payload["prediction_sha256"] = _sha256(payload)
    return payload


def run_d127_s0_matrix(
    *,
    model: nn.Module,
    assets_by_candidate: Mapping[str, da.FSRGAsset | da.RDHAAsset],
    rows: Sequence[D127S0Row],
) -> dict[str, Any]:
    """Run all three candidates with M0/M_L92 computed once per S0 row."""

    frozen_rows = _validate_rows(rows)
    _require(tuple(assets_by_candidate) == CANDIDATE_IDS, "S0 candidate assets must use frozen A/B/C order")
    for candidate_id in CANDIDATE_IDS:
        _require(_asset_candidate(assets_by_candidate[candidate_id]) == candidate_id, "candidate/asset binding drift")

    output_rows: list[dict[str, Any]] = []
    candidate_resource: dict[str, dict[str, int]] = {
        candidate_id: {
            "asset_numeric_payload_bytes": _asset_payload_bytes(assets_by_candidate[candidate_id]),
            "adapter_macs_per_sample": -1,
            "total_adapter_macs_support_plus_query": 0,
            "total_id_backbone_forwards": 0,
        }
        for candidate_id in CANDIDATE_IDS
    }
    common_pair_calls = 0
    adapted_pair_calls = 0
    materialization_calls = 0

    for row in frozen_rows:
        materialized: dict[str, Any] = {}
        for candidate_id in CANDIDATE_IDS:
            item = hooks.materialize_d127_candidate(
                model,
                row.support_iq,
                _support_label_ids(row),
                row.query_iq,
                asset=assets_by_candidate[candidate_id],
            )
            materialization_calls += 1
            resource = item.state.receipt
            _require(resource.protocol_closed, "DA state protocol counters are not closed")
            expected_macs = candidate_resource[candidate_id]["adapter_macs_per_sample"]
            if expected_macs < 0:
                candidate_resource[candidate_id]["adapter_macs_per_sample"] = int(resource.adapter_macs_per_sample)
            else:
                _require(expected_macs == int(resource.adapter_macs_per_sample), "candidate adapter MAC changed across rows")
            candidate_resource[candidate_id]["total_adapter_macs_support_plus_query"] += int(
                resource.adapter_macs_per_sample
            ) * (len(row.support_iq) + len(row.query_iq))
            candidate_resource[candidate_id]["total_id_backbone_forwards"] += int(
                item.hook_receipt.total_id_backbone_forwards
            )
            materialized[candidate_id] = item

        base_owner = materialized[CANDIDATE_IDS[0]].base_cache
        for candidate_id in CANDIDATE_IDS[1:]:
            candidate_base = materialized[candidate_id].base_cache
            _require(
                np.array_equal(base_owner.support_zid, candidate_base.support_zid)
                and np.array_equal(base_owner.query_zid, candidate_base.query_zid),
                "base z160 drifted across candidates",
            )
        common = joint.run_d127_common_two_arm(
            base_support_zid=base_owner.support_zid,
            base_query_zid=base_owner.query_zid,
            support_labels=row.support_labels,
            registered_classes=row.registered_classes,
            opaque_query_ids=row.opaque_query_ids,
            qknn_lock=row.qknn_lock,
        )
        common_pair_calls += 1
        candidate_outputs: dict[str, Any] = {}
        for candidate_id in CANDIDATE_IDS:
            item = materialized[candidate_id]
            adapted = joint.run_d127_adapted_two_arm(
                adapted_support_zid=item.adapted_cache.support_zid,
                adapted_query_zid=item.adapted_cache.query_zid,
                support_labels=row.support_labels,
                registered_classes=row.registered_classes,
                opaque_query_ids=row.opaque_query_ids,
                qknn_lock=row.qknn_lock,
            )
            adapted_pair_calls += 1
            candidate_outputs[candidate_id] = {
                "arms": {arm.arm_id: _arm_payload(arm) for arm in adapted.arms},
                "adapted_pair_receipt": _jsonable(adapted.receipt),
                "hook_receipt": _jsonable(item.hook_receipt.as_dict()),
                "da_resource": _jsonable(item.state.receipt.as_dict()),
            }
        row_payload = {
            "row_id": row.row_id,
            "receiver_id": row.receiver_id,
            "k_shot": row.k_shot,
            "scene": row.scene,
            "opaque_query_ids": list(row.opaque_query_ids),
            "common_arms": {arm.arm_id: _arm_payload(arm) for arm in common.arms},
            "common_pair_receipt": _jsonable(common.receipt),
            "candidates": candidate_outputs,
        }
        row_payload["row_sha256"] = _sha256(row_payload)
        output_rows.append(row_payload)

    _require(
        materialization_calls == ROW_COUNT * len(CANDIDATE_IDS)
        and common_pair_calls == ROW_COUNT
        and adapted_pair_calls == ROW_COUNT * len(CANDIDATE_IDS),
        "S0 common/adapted call accounting drift",
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_ids": list(CANDIDATE_IDS),
        "evaluation_scope": "TARGET_DEVELOPMENT_S0_18",
        "truth_loaded": False,
        "row_count": ROW_COUNT,
        "rows_complete": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "execution_counts": {
            "candidate_materializations": materialization_calls,
            "common_two_arm_calls": common_pair_calls,
            "adapted_two_arm_calls": adapted_pair_calls,
        },
        "resource_by_candidate": candidate_resource,
        "rows": output_rows,
    }
    payload["prediction_sha256"] = _sha256(payload)
    return payload


def write_d127_s0_predictions_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one canonical truth-free prediction artifact without overwrite."""

    target = Path(path)
    _require(payload.get("schema") == SCHEMA and payload.get("truth_loaded") is False, "invalid S0 prediction payload")
    json_payload = _jsonable(payload)
    _require(not _forbidden_keys(json_payload), "forbidden truth/role/quota field")
    signed = dict(json_payload)
    digest = signed.pop("prediction_sha256", None)
    _require(isinstance(digest, str) and digest == _sha256(signed), "S0 prediction digest drift")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(json_payload) + b"\n"
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise D127S0EntryError("S0 prediction output already exists") from exc
    return target


__all__ = [
    "D127S0EntryError",
    "D127S0Row",
    "CANDIDATE_IDS",
    "FORBIDDEN_PUBLIC_FIELDS",
    "ROW_COUNT",
    "SCHEMA",
    "run_d127_s0_matrix",
    "write_d127_s0_predictions_exclusive",
]
