"""Minimal source-only release wire for the frozen D127 Phase1 assets.

This module deliberately implements only the durable half of the D127 Phase1
path.  It turns the already-validated D106 ``L_s`` joined rows into the frozen
seven receiver-held K1/K5 episode plan, parses the frozen D127 method lock,
and persists only typed INT8/FP16 A/B/C assets.  It does *not* train a model,
open a target capsule, or persist a source IQ/feature/receiver/class sidecar.

The eventual checkpoint trainer consumes :class:`D127Phase1EpisodePlan` in
memory and calls the writer below once each candidate has been quantized.
Single-candidate bundles are intentionally mergeable so A/B/C can be built on
separate GPUs without a shared mutable output directory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias
import uuid

import numpy as np
import torch
from torch import Tensor

from cvsrffi import stage2_d106_phase1_tap as d106
from cvsrffi import stage2_d127_da_candidates as da
from cvsrffi import stage2_d127_checkpoint_hooks as hooks
from cvsrffi import stage2_d127_phase1_assets as assets
from cvsrffi.stage2_d127_torch_compat import numpy_to_torch_copy
from cvsrffi import stage2_zid_student_t_qknn as qknn


METHOD_LOCK_SCHEMA = "cvs.stage2.d127.joint_s0.method_lock.v1"
EPISODE_SCHEMA = "cvs.phase1.d127.episode_manifest.v1"
ASSET_WIRE_SCHEMA = "cvs.phase1.d127.quantized_asset_wire.v1"
BUNDLE_SCHEMA = "cvs.phase1.d127.asset_bundle.v1"
COMPLETION_SCHEMA = "cvs.phase1.d127.asset_bundle_completion.v1"

EPISODE_FILE_NAME = "d127_phase1_episode_manifest.json"
MANIFEST_FILE_NAME = "d127_phase1_assets.manifest.json"
COMPLETION_FILE_NAME = "D127_PHASE1_ASSET_COMPLETE.json"

CANDIDATE_IDS = (
    da.CANDIDATE_A,
    da.CANDIDATE_B,
    da.CANDIDATE_C,
)
_CANDIDATE_TAPS = {
    da.CANDIDATE_A: da.TAP_A,
    da.CANDIDATE_B: da.TAP_B,
    da.CANDIDATE_C: da.TAP_C,
}
_CANDIDATE_FILES = {
    da.CANDIDATE_A: "d127_da_a_fsrg_time_fuse.qasset.json",
    da.CANDIDATE_B: "d127_da_b_fsrg_t2norm.qasset.json",
    da.CANDIDATE_C: "d127_da_c_rdah_joint_proj.qasset.json",
}
_HEX = frozenset("0123456789abcdef")

QuantizedD127Asset: TypeAlias = assets.QuantizedFSRGAsset | assets.QuantizedRDHAAsset


class D127Phase1ReleaseError(assets.D127Phase1AssetError):
    """Raised when a D127 Phase1 durable release contract drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127Phase1ReleaseError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256(value: Any, *, name: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and set(text).issubset(_HEX),
        f"{name} must be a lowercase SHA256",
    )
    return text


def _read_regular_bytes(path: str | Path, *, name: str) -> tuple[Path, bytes, str]:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        f"{name} must be a regular file",
    )
    try:
        payload = source.read_bytes()
    except OSError as exc:  # pragma: no cover - host filesystem faults.
        raise D127Phase1ReleaseError(f"{name} read failed") from exc
    return source, payload, _sha256_bytes(payload)


def _read_json_exact(
    path: str | Path, *, expected_sha256: str, name: str, canonical: bool
) -> tuple[Path, dict[str, Any], str]:
    source, payload, observed = _read_regular_bytes(path, name=name)
    expected = _sha256(expected_sha256, name=f"{name} expected hash")
    _require(observed == expected, f"{name} SHA256 mismatch")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127Phase1ReleaseError(f"{name} must be UTF-8 JSON") from exc
    _require(type(value) is dict, f"{name} must contain a JSON object")
    if canonical:
        _require(payload == _canonical_bytes(value), f"{name} is not canonical JSON")
    return source, value, observed


def _ordered_id_root(values: Sequence[str]) -> str:
    result = tuple(str(item) for item in values)
    _require(bool(result) and all(result), "physical-ID root requires nonempty IDs")
    return _canonical_sha256(list(result))


def _set_id_root(values: Sequence[str]) -> str:
    result = tuple(sorted({str(item) for item in values}))
    _require(bool(result) and all(result), "physical-ID set root requires nonempty IDs")
    return _canonical_sha256(list(result))


def _sample_order(receiver: str, label: str, physical_id: str) -> str:
    text = "|".join(("d127-phase1-v1", receiver, label, physical_id))
    return _sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class D127Phase1MethodLock:
    """The small, typed subset of the frozen D127 S0 method lock needed here."""

    lock_sha256: str
    checkpoint_sha256: str
    source_received_iq_sha256: str
    source_received_iq_receipt_sha256: str
    source_label_join_archive_sha256: str
    phase1_lodo_receipt_sha256: str
    quantization_margin_audit_sha256: str
    qknn_numeric: Mapping[str, float | int]

    def __post_init__(self) -> None:
        for name in (
            "lock_sha256",
            "checkpoint_sha256",
            "source_received_iq_sha256",
            "source_received_iq_receipt_sha256",
            "source_label_join_archive_sha256",
            "phase1_lodo_receipt_sha256",
            "quantization_margin_audit_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        numeric = dict(self.qknn_numeric)
        expected = {
            "student_nu",
            "kernel_effective_dim",
            "kernel_volume_gamma",
            "shared_h0",
            "scale_prior_strength",
            "scale_min_ratio",
            "scale_max_ratio",
            "temperature",
        }
        _require(set(numeric) == expected, "method-lock qKNN numeric closure drift")
        _require(type(numeric["kernel_effective_dim"]) is int, "qKNN effective dimension must be int")
        for key, value in numeric.items():
            if key != "kernel_effective_dim":
                _require(type(value) in (int, float) and float(value) > 0.0, f"qKNN {key} drift")
        _require(float(numeric["scale_min_ratio"]) <= 1.0 <= float(numeric["scale_max_ratio"]), "qKNN scale interval drift")
        object.__setattr__(self, "qknn_numeric", MappingProxyType(numeric))


def load_d127_phase1_method_lock(
    path: str | Path, *, expected_sha256: str
) -> D127Phase1MethodLock:
    """Parse and bind the frozen D127 method lock without opening any dataset."""

    _source, value, observed = _read_json_exact(
        path,
        expected_sha256=expected_sha256,
        name="D127 method lock",
        canonical=False,
    )
    required = {
        "schema",
        "candidate_id",
        "protocol_schema",
        "checkpoint",
        "phase1_asset_build",
        "domain_adaptation",
        "student_t_qknn",
    }
    _require(required.issubset(set(value)), "D127 method-lock top-level closure drift")
    _require(value.get("schema") == METHOD_LOCK_SCHEMA, "D127 method-lock schema drift")
    _require(value.get("protocol_schema") == "p2_min_v1", "D127 method-lock protocol drift")
    checkpoint = value.get("checkpoint")
    build = value.get("phase1_asset_build")
    adaptation = value.get("domain_adaptation")
    qknn_payload = value.get("student_t_qknn")
    _require(type(checkpoint) is dict and type(build) is dict, "D127 method-lock source/checkpoint drift")
    _require(type(adaptation) is dict and type(qknn_payload) is dict, "D127 method-lock DA/qKNN drift")
    _require(
        adaptation.get("query_fit_count") == 0
        and adaptation.get("query_update_count") == 0
        and adaptation.get("query_selection_count") == 0,
        "D127 method lock exposes a query fit/update/selection surface",
    )
    candidates = adaptation.get("candidates")
    _require(type(candidates) is list and len(candidates) == len(CANDIDATE_IDS), "D127 candidate closure drift")
    observed_taps = {
        item.get("candidate_id"): item.get("tap")
        for item in candidates
        if type(item) is dict
    }
    _require(observed_taps == _CANDIDATE_TAPS, "D127 candidate/tap method-lock drift")
    _require(build.get("partition_schema") == "d127-phase1-v1", "D127 partition schema drift")
    _require(build.get("receiver_held_fold_count") == 7, "D127 receiver fold count drift")
    _require(build.get("physical_samples_per_receiver_class") == 14, "D127 physical sample count drift")
    _require(build.get("support_pool_count") == 5 and build.get("outer_query_pool_count") == 9, "D127 support/query pool drift")
    _require(build.get("active_k") == [1, 5] and build.get("final_episode_count") == 14, "D127 K/episode plan drift")
    _require(build.get("k1_is_first_k5_support") is True and build.get("support_query_globally_disjoint") is True, "D127 K/disjointness contract drift")
    _require(build.get("class_loco_training_count") == 0, "D127 class-LOCO must remain disabled")
    _require(build.get("persist_source_rows_or_features") is False and build.get("persist_fp32_sidecar") is False, "D127 persistent source/FP32 prohibition drift")
    optimizer = build.get("optimizer")
    _require(
        type(optimizer) is dict
        and optimizer.get("name") == "full_batch_lbfgs"
        and optimizer.get("max_iter") == 128
        and optimizer.get("line_search_fn") == "strong_wolfe"
        and optimizer.get("initialization_count") == 1
        and optimizer.get("early_stop") is False
        and optimizer.get("parameter_scan") is False,
        "D127 Phase1 optimizer lock drift",
    )
    numeric_keys = {
        "student_nu",
        "kernel_effective_dim",
        "kernel_volume_gamma",
        "shared_h0",
        "scale_prior_strength",
        "scale_min_ratio",
        "scale_max_ratio",
        "temperature",
    }
    _require(qknn_payload.get("active_k") == [1, 5], "D127 qKNN K lock drift")
    _require(qknn_payload.get("support_storage") == "int8_fp16_scale", "D127 qKNN storage drift")
    _require(numeric_keys.issubset(qknn_payload), "D127 qKNN numeric fields missing")
    return D127Phase1MethodLock(
        lock_sha256=observed,
        checkpoint_sha256=_sha256(checkpoint.get("sha256"), name="method-lock checkpoint"),
        source_received_iq_sha256=_sha256(build.get("source_received_iq_sha256"), name="method-lock source IQ"),
        source_received_iq_receipt_sha256=_sha256(build.get("source_received_iq_receipt_sha256"), name="method-lock source IQ receipt"),
        source_label_join_archive_sha256=_sha256(build.get("source_label_join_archive_sha256"), name="method-lock source label join"),
        phase1_lodo_receipt_sha256=_sha256(qknn_payload.get("phase1_lodo_receipt_sha256"), name="method-lock phase1 LODO receipt"),
        quantization_margin_audit_sha256=_sha256(qknn_payload.get("quantization_margin_audit_sha256"), name="method-lock qKNN quantization receipt"),
        qknn_numeric={key: qknn_payload[key] for key in sorted(numeric_keys)},
    )


def build_d127_phase1_qknn_locks(
    method_lock: D127Phase1MethodLock,
) -> Mapping[int, qknn.Phase1ZIDStudentTLock]:
    """Materialize the exact existing K1/K5 typed locks from the method lock."""

    _require(type(method_lock) is D127Phase1MethodLock, "D127 qKNN build requires a typed method lock")
    values = dict(method_lock.qknn_numeric)
    locks: dict[int, qknn.Phase1ZIDStudentTLock] = {}
    for active_k in (1, 5):
        try:
            locks[active_k] = qknn.Phase1ZIDStudentTLock(
                active_k=active_k,
                student_nu=float(values["student_nu"]),
                kernel_effective_dim=int(values["kernel_effective_dim"]),
                kernel_volume_gamma=float(values["kernel_volume_gamma"]),
                shared_h0=float(values["shared_h0"]),
                scale_prior_strength=float(values["scale_prior_strength"]),
                scale_min_ratio=float(values["scale_min_ratio"]),
                scale_max_ratio=float(values["scale_max_ratio"]),
                temperature=float(values["temperature"]),
                phase1_lodo_receipt_sha256=method_lock.phase1_lodo_receipt_sha256,
                quantization_margin_audit_sha256=method_lock.quantization_margin_audit_sha256,
            )
        except qknn.ZIDStudentTQKNNError as exc:
            raise D127Phase1ReleaseError("D127 typed qKNN lock build failed") from exc
    return MappingProxyType(locks)


@dataclass(frozen=True, slots=True)
class D127Phase1Episode:
    """One in-memory source episode; raw source keys never enter its manifest."""

    fold_ordinal: int
    k_shot: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    support_labels: tuple[int, ...]
    query_labels: tuple[int, ...]
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(type(self.fold_ordinal) is int and 0 <= self.fold_ordinal < 7, "episode fold ordinal drift")
        _require(self.k_shot in (1, 5), "episode K must be frozen K1 or K5")
        _require(len(self.support_indices) == 6 * self.k_shot, "episode support count drift")
        _require(len(self.query_indices) == 6 * 9, "episode query count drift")
        _require(len(self.support_indices) == len(self.support_labels) == len(self.support_physical_ids), "episode support alignment drift")
        _require(len(self.query_indices) == len(self.query_labels) == len(self.query_physical_ids), "episode query alignment drift")
        _require(len(set(self.support_indices)) == len(self.support_indices), "episode support index duplication")
        _require(len(set(self.query_indices)) == len(self.query_indices), "episode query index duplication")
        _require(not set(self.support_indices).intersection(self.query_indices), "episode support/query index overlap")
        _require(not set(self.support_physical_ids).intersection(self.query_physical_ids), "episode support/query physical overlap")
        _require(set(self.support_labels) == set(range(6)) == set(self.query_labels), "episode class closure drift")
        for label in range(6):
            _require(self.support_labels.count(label) == self.k_shot, "episode support class count drift")
            _require(self.query_labels.count(label) == 9, "episode query class count drift")

    @property
    def episode_id(self) -> str:
        return f"d127-phase1-fold-{self.fold_ordinal:02d}-k{self.k_shot}"

    def receipt_payload(self) -> dict[str, Any]:
        """Return a source-key-free durable audit record."""

        return {
            "fold_ordinal": self.fold_ordinal,
            "k_shot": self.k_shot,
            "support_row_count": len(self.support_indices),
            "query_row_count": len(self.query_indices),
            "support_physical_root_sha256": _ordered_id_root(self.support_physical_ids),
            "query_physical_root_sha256": _ordered_id_root(self.query_physical_ids),
        }


@dataclass(frozen=True, slots=True)
class D127Phase1EpisodePlan:
    """The deterministic seven-fold K1/K5 plan and a durable closed manifest."""

    episodes: tuple[D127Phase1Episode, ...]
    manifest: Mapping[str, Any]
    contract_sha256: str

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.episodes, key=lambda item: (item.fold_ordinal, item.k_shot)))
        _require(len(ordered) == 14, "D127 episode plan must contain exactly fourteen episodes")
        _require(
            tuple((item.fold_ordinal, item.k_shot) for item in ordered)
            == tuple((fold, k) for fold in range(7) for k in (1, 5)),
            "D127 episode plan fold/K closure drift",
        )
        _require(isinstance(self.manifest, Mapping), "episode manifest must be a mapping")
        payload = dict(self.manifest)
        _require(payload.get("schema") == EPISODE_SCHEMA, "episode manifest schema drift")
        _require(_canonical_sha256({key: value for key, value in payload.items() if key != "contract_sha256"}) == self.contract_sha256, "episode manifest contract root drift")
        _require(payload.get("contract_sha256") == self.contract_sha256, "episode manifest self binding drift")
        object.__setattr__(self, "episodes", ordered)
        object.__setattr__(self, "manifest", MappingProxyType(payload))
        object.__setattr__(self, "contract_sha256", _sha256(self.contract_sha256, name="episode contract"))


def _episode_source_binding(method_lock: D127Phase1MethodLock) -> dict[str, Any]:
    return {
        "checkpoint_sha256": method_lock.checkpoint_sha256,
        "method_lock_sha256": method_lock.lock_sha256,
        "selected_received_iq_sha256": method_lock.source_received_iq_sha256,
        "selected_received_iq_receipt_sha256": method_lock.source_received_iq_receipt_sha256,
        "source_label_join_archive_sha256": method_lock.source_label_join_archive_sha256,
    }


def build_d127_phase1_episode_plan(
    joined_rows: d106.D106JoinedLSRows,
    *,
    method_lock: D127Phase1MethodLock,
) -> D127Phase1EpisodePlan:
    """Build the frozen seven receiver-held K1/K5 plan from real joined rows.

    The only persisted result is a source-key-free manifest.  ``joined_rows``
    and the index/ID tuples remain memory-only for the later checkpoint build.
    """

    _require(type(joined_rows) is d106.D106JoinedLSRows, "D127 episodes require exact D106 joined L_s rows")
    _require(type(method_lock) is D127Phase1MethodLock, "D127 episodes require a typed method lock")
    receivers = np.asarray(joined_rows.receiver_ids, dtype=np.str_)
    labels = np.asarray(joined_rows.tx_labels, dtype=np.str_)
    physical_ids = np.asarray(joined_rows.physical_ids, dtype=np.str_)
    receiver_values = tuple(sorted(set(receivers.tolist())))
    label_values = tuple(sorted(set(labels.tolist())))
    _require(len(receiver_values) == 7 and len(label_values) == 6, "D127 joined receiver/class closure drift")
    label_index = {label: index for index, label in enumerate(label_values)}
    records: list[D127Phase1Episode] = []
    for fold_ordinal, receiver in enumerate(receiver_values):
        k5_support_indices: list[int] = []
        k5_support_labels: list[int] = []
        k5_support_ids: list[str] = []
        query_indices: list[int] = []
        query_labels: list[int] = []
        query_ids: list[str] = []
        k1_support_indices: list[int] = []
        k1_support_labels: list[int] = []
        k1_support_ids: list[str] = []
        for label in label_values:
            indices = np.flatnonzero((receivers == receiver) & (labels == label)).tolist()
            _require(len(indices) == 14, "D127 receiver/class must contain exactly fourteen physical rows")
            ranked = sorted(
                indices,
                key=lambda index: (_sample_order(receiver, label, str(physical_ids[index])), str(physical_ids[index])),
            )
            _require(
                len({_sample_order(receiver, label, str(physical_ids[index])) for index in ranked}) == 14,
                "D127 physical ordering digest collision",
            )
            support5 = ranked[:5]
            outer_query = ranked[5:]
            _require(len(support5) == 5 and len(outer_query) == 9, "D127 split count drift")
            class_index = label_index[label]
            k1_support_indices.extend(support5[:1])
            k1_support_labels.append(class_index)
            k1_support_ids.append(str(physical_ids[support5[0]]))
            k5_support_indices.extend(support5)
            k5_support_labels.extend((class_index,) * 5)
            k5_support_ids.extend(str(physical_ids[index]) for index in support5)
            query_indices.extend(outer_query)
            query_labels.extend((class_index,) * 9)
            query_ids.extend(str(physical_ids[index]) for index in outer_query)
        records.extend(
            (
                D127Phase1Episode(
                    fold_ordinal=fold_ordinal,
                    k_shot=1,
                    support_indices=tuple(k1_support_indices),
                    query_indices=tuple(query_indices),
                    support_labels=tuple(k1_support_labels),
                    query_labels=tuple(query_labels),
                    support_physical_ids=tuple(k1_support_ids),
                    query_physical_ids=tuple(query_ids),
                ),
                D127Phase1Episode(
                    fold_ordinal=fold_ordinal,
                    k_shot=5,
                    support_indices=tuple(k5_support_indices),
                    query_indices=tuple(query_indices),
                    support_labels=tuple(k5_support_labels),
                    query_labels=tuple(query_labels),
                    support_physical_ids=tuple(k5_support_ids),
                    query_physical_ids=tuple(query_ids),
                ),
            )
        )
    ordered = tuple(sorted(records, key=lambda item: (item.fold_ordinal, item.k_shot)))
    all_support = tuple(identifier for episode in ordered for identifier in episode.support_physical_ids)
    all_query = tuple(identifier for episode in ordered for identifier in episode.query_physical_ids)
    _require(not set(all_support).intersection(all_query), "D127 all-episode support/query physical overlap")
    for fold in range(7):
        k1 = next(item for item in ordered if item.fold_ordinal == fold and item.k_shot == 1)
        k5 = next(item for item in ordered if item.fold_ordinal == fold and item.k_shot == 5)
        _require(set(k1.support_physical_ids).issubset(set(k5.support_physical_ids)), "D127 K1 must be a physical K5 prefix")
        _require(k1.query_physical_ids == k5.query_physical_ids, "D127 K1/K5 outer-query drift")
    source_binding = _episode_source_binding(method_lock)
    manifest_without_root: dict[str, Any] = {
        "schema": EPISODE_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "partition_schema": "d127-phase1-v1",
        "source_binding": source_binding,
        "receiver_fold_count": 7,
        "class_count": 6,
        "physical_samples_per_receiver_class": 14,
        "ordering": "sha256(d127-phase1-v1|receiver|class|physical_id)_ascending",
        "support_pool_count": 5,
        "outer_query_pool_count": 9,
        "active_k": [1, 5],
        "k1_is_first_k5_support": True,
        "support_query_globally_disjoint": True,
        "class_loco_training_count": 0,
        "episodes": [item.receipt_payload() for item in ordered],
        "all_support_physical_set_root_sha256": _set_id_root(all_support),
        "all_query_physical_set_root_sha256": _set_id_root(all_query),
        "source_rows_or_features_persisted": False,
        "receiver_or_class_keys_persisted": False,
    }
    contract = _canonical_sha256(manifest_without_root)
    manifest = {**manifest_without_root, "contract_sha256": contract}
    return D127Phase1EpisodePlan(
        episodes=ordered,
        manifest=manifest,
        contract_sha256=contract,
    )


def _buffer_wire(value: assets.FP16Buffer) -> dict[str, Any]:
    return {"width": value.width, "data_hex": value.data.hex()}


def _matrix_wire(value: assets.SymmetricInt8Matrix) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "group_axis": value.group_axis,
        "codes_hex": value.codes.hex(),
        "scales": _buffer_wire(value.scales),
    }


def _vector_wire(value: assets.SymmetricInt8Vector) -> dict[str, Any]:
    return {
        "width": value.width,
        "codes_hex": value.codes.hex(),
        "scale": _buffer_wire(value.scale),
    }


def _decode_hex(value: Any, *, name: str) -> bytes:
    _require(type(value) is str and len(value) % 2 == 0, f"{name} must be an even hexadecimal string")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise D127Phase1ReleaseError(f"{name} is not hexadecimal") from exc
    _require(result.hex() == value, f"{name} must use lowercase canonical hex")
    return result


def _buffer_from_wire(value: Any, *, name: str) -> assets.FP16Buffer:
    _require(type(value) is dict and set(value) == {"width", "data_hex"}, f"{name} wire closure drift")
    try:
        return assets.FP16Buffer(width=value["width"], data=_decode_hex(value["data_hex"], name=f"{name}.data"))
    except assets.D127Phase1AssetError as exc:
        raise D127Phase1ReleaseError(f"{name} wire invalid") from exc


def _matrix_from_wire(value: Any, *, name: str) -> assets.SymmetricInt8Matrix:
    _require(type(value) is dict and set(value) == {"shape", "group_axis", "codes_hex", "scales"}, f"{name} wire closure drift")
    shape = value.get("shape")
    _require(type(shape) is list and len(shape) == 2 and all(type(item) is int for item in shape), f"{name} shape drift")
    try:
        return assets.SymmetricInt8Matrix(
            shape=(shape[0], shape[1]),
            group_axis=value["group_axis"],
            codes=_decode_hex(value["codes_hex"], name=f"{name}.codes"),
            scales=_buffer_from_wire(value["scales"], name=f"{name}.scales"),
        )
    except assets.D127Phase1AssetError as exc:
        raise D127Phase1ReleaseError(f"{name} wire invalid") from exc


def _vector_from_wire(value: Any, *, name: str) -> assets.SymmetricInt8Vector:
    _require(type(value) is dict and set(value) == {"width", "codes_hex", "scale"}, f"{name} wire closure drift")
    try:
        return assets.SymmetricInt8Vector(
            width=value["width"],
            codes=_decode_hex(value["codes_hex"], name=f"{name}.codes"),
            scale=_buffer_from_wire(value["scale"], name=f"{name}.scale"),
        )
    except assets.D127Phase1AssetError as exc:
        raise D127Phase1ReleaseError(f"{name} wire invalid") from exc


def _asset_wire(asset: QuantizedD127Asset) -> dict[str, Any]:
    assets.assert_no_persistent_fp32_sidecar(asset)
    if isinstance(asset, assets.QuantizedFSRGAsset):
        return {
            "schema": ASSET_WIRE_SCHEMA,
            "candidate_id": asset.candidate_id,
            "tap_name": asset.tap_name,
            "kind": "fsrg",
            "payload": {
                "U": _matrix_wire(asset.U),
                "V": _matrix_wire(asset.V),
                "d_f_diag": _buffer_wire(asset.d_f_diag),
                "rho": _buffer_wire(asset.rho),
            },
        }
    if isinstance(asset, assets.QuantizedRDHAAsset):
        return {
            "schema": ASSET_WIRE_SCHEMA,
            "candidate_id": asset.candidate_id,
            "tap_name": asset.tap_name,
            "kind": "rdha",
            "payload": {
                "U": _matrix_wire(asset.U),
                "V": _matrix_wire(asset.V),
                "Q": _matrix_wire(asset.Q),
                "b": _vector_wire(asset.b),
                "mean_p1": _buffer_wire(asset.mean_p1),
                "std_p1": _buffer_wire(asset.std_p1),
                "a_max": _buffer_wire(asset.a_max),
            },
        }
    raise D127Phase1ReleaseError("D127 wire requires a typed quantized asset")


def _asset_from_wire(value: Any) -> QuantizedD127Asset:
    _require(type(value) is dict and set(value) == {"schema", "candidate_id", "tap_name", "kind", "payload"}, "D127 asset wire top-level closure drift")
    _require(value.get("schema") == ASSET_WIRE_SCHEMA, "D127 asset wire schema drift")
    candidate = value.get("candidate_id")
    tap = value.get("tap_name")
    _require(candidate in CANDIDATE_IDS and tap == _CANDIDATE_TAPS[candidate], "D127 asset wire candidate/tap drift")
    payload = value.get("payload")
    _require(type(payload) is dict, "D127 asset wire payload must be an object")
    try:
        if value.get("kind") == "fsrg":
            _require(candidate in (da.CANDIDATE_A, da.CANDIDATE_B), "D127 FSRG candidate drift")
            _require(set(payload) == {"U", "V", "d_f_diag", "rho"}, "D127 FSRG payload closure drift")
            result: QuantizedD127Asset = assets.QuantizedFSRGAsset(
                candidate_id=candidate,
                tap_name=tap,
                U=_matrix_from_wire(payload["U"], name="FSRG.U"),
                V=_matrix_from_wire(payload["V"], name="FSRG.V"),
                d_f_diag=_buffer_from_wire(payload["d_f_diag"], name="FSRG.d_f_diag"),
                rho=_buffer_from_wire(payload["rho"], name="FSRG.rho"),
            )
        elif value.get("kind") == "rdha":
            _require(candidate == da.CANDIDATE_C, "D127 RDHA candidate drift")
            _require(set(payload) == {"U", "V", "Q", "b", "mean_p1", "std_p1", "a_max"}, "D127 RDHA payload closure drift")
            result = assets.QuantizedRDHAAsset(
                U=_matrix_from_wire(payload["U"], name="RDHA.U"),
                V=_matrix_from_wire(payload["V"], name="RDHA.V"),
                Q=_matrix_from_wire(payload["Q"], name="RDHA.Q"),
                b=_vector_from_wire(payload["b"], name="RDHA.b"),
                mean_p1=_buffer_from_wire(payload["mean_p1"], name="RDHA.mean_p1"),
                std_p1=_buffer_from_wire(payload["std_p1"], name="RDHA.std_p1"),
                a_max=_buffer_from_wire(payload["a_max"], name="RDHA.a_max"),
                candidate_id=candidate,
                tap_name=tap,
            )
        else:
            raise D127Phase1ReleaseError("D127 asset wire kind drift")
    except assets.D127Phase1AssetError as exc:
        raise D127Phase1ReleaseError("D127 quantized asset reconstruction failed") from exc
    assets.assert_no_persistent_fp32_sidecar(result)
    return result


def _candidate_asset_entry(asset: QuantizedD127Asset, *, file_name: str, file_sha256: str) -> dict[str, Any]:
    return {
        "asset_file": file_name,
        "asset_sha256": _sha256(file_sha256, name="asset file"),
        "candidate_id": asset.candidate_id,
        "tap_name": _CANDIDATE_TAPS[asset.candidate_id],
        "numeric_payload_bytes": asset.numeric_payload_bytes,
        "persistent_fp32_sidecar": False,
    }


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise D127Phase1ReleaseError(f"refusing to overwrite: {path}") from exc


def _prepare_output_directory(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    _require(not output.exists() and not output.is_symlink(), f"refusing to overwrite output: {output}")
    parent = output.parent
    _require(parent.is_dir() and not parent.is_symlink(), "bundle output parent must be an existing regular directory")
    staging = parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    _require(not staging.exists() and not staging.is_symlink(), "bundle staging collision")
    staging.mkdir()
    return output, staging


def _publish_new_directory(staging: Path, output: Path, *, members: Sequence[str]) -> None:
    expected = set(members)
    observed = {item.name for item in staging.iterdir()}
    _require(observed == expected, "bundle staging member closure drift")
    try:
        staging.rename(output)
    except OSError as exc:  # pragma: no cover - host filesystem faults.
        raise D127Phase1ReleaseError("bundle publish failed") from exc


def _completion_payload(*, manifest_sha256: str, members: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "manifest_sha256": _sha256(manifest_sha256, name="completion manifest"),
        "members": list(members),
    }


def _qknn_payload(method_lock: D127Phase1MethodLock) -> dict[str, Any]:
    locks = build_d127_phase1_qknn_locks(method_lock)
    return {
        "phase1_lodo_receipt_sha256": method_lock.phase1_lodo_receipt_sha256,
        "quantization_margin_audit_sha256": method_lock.quantization_margin_audit_sha256,
        "lock_digest_by_k": {str(key): value.lock_digest for key, value in locks.items()},
    }


def _quantized_only_receipt(candidate_id: str) -> dict[str, Any]:
    """A closed receipt for unit-built wires before the real trainer runs."""

    return {
        "schema": "cvs.phase1.d127.training_receipt.v1",
        "candidate_id": candidate_id,
        "execution": "QUANTIZED_ASSET_ONLY_NO_TRAINING_RECEIPT",
        "source_rows_or_features_persisted": False,
        "target_runtime_access": False,
    }


def _validate_training_receipt(
    value: Mapping[str, Any] | None, *, candidate_id: str
) -> dict[str, Any]:
    receipt = _quantized_only_receipt(candidate_id) if value is None else dict(value)
    _require(type(receipt) is dict, "D127 training receipt must be a mapping")
    _require(receipt.get("schema") == "cvs.phase1.d127.training_receipt.v1", "D127 training receipt schema drift")
    _require(receipt.get("candidate_id") == candidate_id, "D127 training receipt candidate drift")
    _require(receipt.get("source_rows_or_features_persisted") is False and receipt.get("target_runtime_access") is False, "D127 training receipt persistence/access drift")
    try:
        encoded = _canonical_bytes(receipt)
    except (TypeError, ValueError) as exc:
        raise D127Phase1ReleaseError("D127 training receipt is not canonical JSON") from exc
    forbidden = (
        "received_iq",
        "raw_iq",
        "source_feature",
        "physical_ids",
        "receiver_ids",
        "tx_labels",
        "support_iq",
        "query_iq",
    )
    text = encoded.decode("ascii")
    _require(not any(token in text for token in forbidden), "D127 training receipt exposes a prohibited source sidecar")
    return receipt


def write_d127_phase1_single_candidate_bundle(
    *,
    output_dir: str | Path,
    candidate_id: str,
    asset: QuantizedD127Asset,
    method_lock: D127Phase1MethodLock,
    episode_plan: D127Phase1EpisodePlan,
    phase1_training_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one non-overwriting quantized candidate bundle.

    This writer accepts only already-quantized assets.  It cannot receive raw
    source IQ, source features, labels, receiver keys, or an FP32 sidecar.
    """

    _require(candidate_id in CANDIDATE_IDS, "single bundle candidate drift")
    _require(type(method_lock) is D127Phase1MethodLock, "single bundle requires typed method lock")
    _require(type(episode_plan) is D127Phase1EpisodePlan, "single bundle requires frozen episode plan")
    _require(isinstance(asset, (assets.QuantizedFSRGAsset, assets.QuantizedRDHAAsset)), "single bundle requires quantized D127 asset")
    _require(asset.candidate_id == candidate_id, "single bundle candidate/asset drift")
    assets.assert_no_persistent_fp32_sidecar(asset)
    training_receipt = _validate_training_receipt(
        phase1_training_receipt, candidate_id=candidate_id
    )
    wire = _asset_wire(asset)
    wire_bytes = _canonical_bytes(wire)
    wire_file = _CANDIDATE_FILES[candidate_id]
    episode_bytes = _canonical_bytes(dict(episode_plan.manifest))
    episode_sha = _sha256_bytes(episode_bytes)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_kind": "single_candidate",
        "protocol_schema": "p2_min_v1",
        "candidate_ids": [candidate_id],
        "method_lock_sha256": method_lock.lock_sha256,
        "checkpoint_sha256": method_lock.checkpoint_sha256,
        "source_binding": _episode_source_binding(method_lock),
        "episode_manifest_file": EPISODE_FILE_NAME,
        "episode_manifest_sha256": episode_sha,
        "episode_contract_sha256": episode_plan.contract_sha256,
        "qknn_lock_binding": _qknn_payload(method_lock),
        "phase1_training_receipt": training_receipt,
        "candidate_assets": {
            candidate_id: _candidate_asset_entry(
                asset,
                file_name=wire_file,
                file_sha256=_sha256_bytes(wire_bytes),
            )
        },
        "source_rows_or_features_persisted": False,
        "receiver_or_class_keys_persisted": False,
        "persistent_fp32_sidecar": False,
        "target_runtime_access": False,
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    output, staging = _prepare_output_directory(output_dir)
    try:
        _write_new(staging / EPISODE_FILE_NAME, episode_bytes)
        _write_new(staging / wire_file, wire_bytes)
        _write_new(staging / MANIFEST_FILE_NAME, manifest_bytes)
        marker = _completion_payload(
            manifest_sha256=manifest_sha,
            members=tuple(sorted((EPISODE_FILE_NAME, wire_file, MANIFEST_FILE_NAME))),
        )
        _write_new(staging / COMPLETION_FILE_NAME, _canonical_bytes(marker))
        _publish_new_directory(
            staging,
            output,
            members=(EPISODE_FILE_NAME, wire_file, MANIFEST_FILE_NAME, COMPLETION_FILE_NAME),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "D127_PHASE1_SINGLE_CANDIDATE_BUNDLE_WRITTEN",
        "output_dir": str(output.resolve()),
        "manifest": str((output / MANIFEST_FILE_NAME).resolve()),
        "manifest_sha256": manifest_sha,
        "candidate_id": candidate_id,
        "asset_sha256": _sha256_bytes(wire_bytes),
        "episode_contract_sha256": episode_plan.contract_sha256,
    }


def _load_bundle_directory(
    bundle_dir: str | Path, *, expected_manifest_sha256: str | None = None
) -> tuple[Path, dict[str, Any], str, dict[str, Any], dict[str, QuantizedD127Asset]]:
    root = Path(bundle_dir)
    _require(root.is_dir() and not root.is_symlink(), "D127 bundle directory must be a regular directory")
    manifest_path = root / MANIFEST_FILE_NAME
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "D127 bundle manifest missing")
    raw_manifest = manifest_path.read_bytes()
    manifest_sha = _sha256_bytes(raw_manifest)
    if expected_manifest_sha256 is not None:
        _require(manifest_sha == _sha256(expected_manifest_sha256, name="bundle expected manifest"), "D127 bundle manifest SHA mismatch")
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127Phase1ReleaseError("D127 bundle manifest JSON invalid") from exc
    _require(type(manifest) is dict and raw_manifest == _canonical_bytes(manifest), "D127 bundle manifest must be canonical")
    required = {
        "schema", "bundle_kind", "protocol_schema", "candidate_ids", "method_lock_sha256", "checkpoint_sha256", "source_binding", "episode_manifest_file", "episode_manifest_sha256", "episode_contract_sha256", "qknn_lock_binding", "phase1_training_receipt", "candidate_assets", "source_rows_or_features_persisted", "receiver_or_class_keys_persisted", "persistent_fp32_sidecar", "target_runtime_access",
    }
    _require(set(manifest) == required, "D127 bundle manifest closure drift")
    _require(manifest.get("schema") == BUNDLE_SCHEMA and manifest.get("protocol_schema") == "p2_min_v1", "D127 bundle schema/protocol drift")
    _require(manifest.get("bundle_kind") in {"single_candidate", "merged_complete"}, "D127 bundle kind drift")
    candidate_ids = manifest.get("candidate_ids")
    _require(type(candidate_ids) is list and all(item in CANDIDATE_IDS for item in candidate_ids) and len(set(candidate_ids)) == len(candidate_ids), "D127 bundle candidate list drift")
    if manifest["bundle_kind"] == "single_candidate":
        _require(len(candidate_ids) == 1, "single bundle must contain one candidate")
    else:
        _require(tuple(candidate_ids) == CANDIDATE_IDS, "merged D127 bundle must contain A/B/C in frozen order")
    for name in ("method_lock_sha256", "checkpoint_sha256", "episode_manifest_sha256", "episode_contract_sha256"):
        _sha256(manifest.get(name), name=f"bundle {name}")
    source_binding = manifest.get("source_binding")
    _require(type(source_binding) is dict and set(source_binding) == {"checkpoint_sha256", "method_lock_sha256", "selected_received_iq_sha256", "selected_received_iq_receipt_sha256", "source_label_join_archive_sha256"}, "D127 bundle source binding closure drift")
    _require(source_binding["checkpoint_sha256"] == manifest["checkpoint_sha256"] and source_binding["method_lock_sha256"] == manifest["method_lock_sha256"], "D127 bundle source checkpoint/method binding drift")
    for key, value in source_binding.items():
        _sha256(value, name=f"source binding {key}")
    _require(manifest.get("episode_manifest_file") == EPISODE_FILE_NAME, "D127 bundle episode file drift")
    episode_path = root / EPISODE_FILE_NAME
    _require(episode_path.is_file() and not episode_path.is_symlink(), "D127 episode manifest missing")
    episode_bytes = episode_path.read_bytes()
    _require(_sha256_bytes(episode_bytes) == manifest["episode_manifest_sha256"], "D127 episode manifest SHA drift")
    try:
        episode_manifest = json.loads(episode_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127Phase1ReleaseError("D127 episode manifest JSON invalid") from exc
    _require(type(episode_manifest) is dict and episode_bytes == _canonical_bytes(episode_manifest), "D127 episode manifest must be canonical")
    _require(episode_manifest.get("schema") == EPISODE_SCHEMA and episode_manifest.get("contract_sha256") == manifest["episode_contract_sha256"], "D127 episode manifest binding drift")
    _require(episode_manifest.get("source_binding") == source_binding, "D127 episode source binding drift")
    _require(_canonical_sha256({key: value for key, value in episode_manifest.items() if key != "contract_sha256"}) == manifest["episode_contract_sha256"], "D127 episode contract checksum drift")
    qknn_binding = manifest.get("qknn_lock_binding")
    _require(type(qknn_binding) is dict and set(qknn_binding) == {"phase1_lodo_receipt_sha256", "quantization_margin_audit_sha256", "lock_digest_by_k"}, "D127 qKNN bundle binding drift")
    _sha256(qknn_binding["phase1_lodo_receipt_sha256"], name="bundle LODO receipt")
    _sha256(qknn_binding["quantization_margin_audit_sha256"], name="bundle quantization receipt")
    _require(type(qknn_binding["lock_digest_by_k"]) is dict and set(qknn_binding["lock_digest_by_k"]) == {"1", "5"}, "D127 qKNN lock digest closure drift")
    for digest in qknn_binding["lock_digest_by_k"].values():
        _sha256(digest, name="bundle qKNN lock digest")
    _require(manifest.get("source_rows_or_features_persisted") is False and manifest.get("receiver_or_class_keys_persisted") is False and manifest.get("persistent_fp32_sidecar") is False and manifest.get("target_runtime_access") is False, "D127 bundle persistence/access prohibition drift")
    training = manifest.get("phase1_training_receipt")
    if manifest["bundle_kind"] == "single_candidate":
        _validate_training_receipt(training, candidate_id=candidate_ids[0])
    else:
        _require(type(training) is dict and set(training) == set(candidate_ids), "D127 merged training receipt closure drift")
        for candidate in candidate_ids:
            _validate_training_receipt(training[candidate], candidate_id=candidate)
    entries = manifest.get("candidate_assets")
    _require(type(entries) is dict and set(entries) == set(candidate_ids), "D127 candidate asset manifest closure drift")
    loaded: dict[str, QuantizedD127Asset] = {}
    expected_files = {EPISODE_FILE_NAME, MANIFEST_FILE_NAME, COMPLETION_FILE_NAME}
    for candidate in candidate_ids:
        entry = entries[candidate]
        _require(type(entry) is dict and set(entry) == {"asset_file", "asset_sha256", "candidate_id", "tap_name", "numeric_payload_bytes", "persistent_fp32_sidecar"}, "D127 candidate asset entry closure drift")
        _require(entry.get("candidate_id") == candidate and entry.get("tap_name") == _CANDIDATE_TAPS[candidate] and entry.get("persistent_fp32_sidecar") is False, "D127 candidate asset binding drift")
        file_name = entry.get("asset_file")
        _require(file_name == _CANDIDATE_FILES[candidate], "D127 candidate asset file drift")
        asset_path = root / file_name
        _require(asset_path.is_file() and not asset_path.is_symlink(), "D127 candidate asset file missing")
        asset_bytes = asset_path.read_bytes()
        _require(_sha256_bytes(asset_bytes) == _sha256(entry.get("asset_sha256"), name="candidate asset hash"), "D127 candidate asset file SHA drift")
        try:
            asset_wire = json.loads(asset_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise D127Phase1ReleaseError("D127 candidate asset wire JSON invalid") from exc
        _require(type(asset_wire) is dict and asset_bytes == _canonical_bytes(asset_wire), "D127 candidate asset wire must be canonical")
        asset = _asset_from_wire(asset_wire)
        _require(asset.candidate_id == candidate and asset.numeric_payload_bytes == entry.get("numeric_payload_bytes"), "D127 candidate asset numeric receipt drift")
        loaded[candidate] = asset
        expected_files.add(file_name)
    marker_path = root / COMPLETION_FILE_NAME
    _require(marker_path.is_file() and not marker_path.is_symlink(), "D127 completion marker missing")
    marker_bytes = marker_path.read_bytes()
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127Phase1ReleaseError("D127 completion marker JSON invalid") from exc
    _require(type(marker) is dict and marker_bytes == _canonical_bytes(marker), "D127 completion marker must be canonical")
    _require(marker == _completion_payload(manifest_sha256=manifest_sha, members=tuple(sorted(expected_files - {COMPLETION_FILE_NAME}))), "D127 completion marker binding drift")
    _require({item.name for item in root.iterdir()} == expected_files, "D127 bundle contains an unapproved sidecar")
    return root, manifest, manifest_sha, episode_manifest, loaded


def merge_d127_phase1_asset_bundles(
    bundle_dirs: Sequence[str | Path], *, output_dir: str | Path
) -> dict[str, Any]:
    """Merge exactly three same-source single-candidate bundles without overwrite."""

    _require(len(bundle_dirs) == 3, "D127 merge requires exactly three single-candidate bundles")
    loaded_parts = [_load_bundle_directory(path) for path in bundle_dirs]
    manifests = [item[1] for item in loaded_parts]
    _require(all(item["bundle_kind"] == "single_candidate" for item in manifests), "D127 merge accepts only single-candidate bundles")
    binding_keys = ("method_lock_sha256", "checkpoint_sha256", "source_binding", "episode_manifest_sha256", "episode_contract_sha256", "qknn_lock_binding")
    first = manifests[0]
    for manifest in manifests[1:]:
        _require(all(manifest[key] == first[key] for key in binding_keys), "D127 merge source/checkpoint/episode/qKNN binding drift")
    candidates = tuple(manifest["candidate_ids"][0] for manifest in manifests)
    _require(set(candidates) == set(CANDIDATE_IDS), "D127 merge candidate closure drift")
    by_candidate = {manifest["candidate_ids"][0]: (root, manifest, assets_by_id) for root, manifest, _sha, _episode, assets_by_id in loaded_parts}
    ordered_candidates = CANDIDATE_IDS
    episode_bytes = (by_candidate[ordered_candidates[0]][0] / EPISODE_FILE_NAME).read_bytes()
    _require(_sha256_bytes(episode_bytes) == first["episode_manifest_sha256"], "D127 merge episode file drift")
    candidate_asset_bytes: dict[str, bytes] = {}
    candidate_entries: dict[str, Any] = {}
    for candidate in ordered_candidates:
        root, manifest, assets_by_id = by_candidate[candidate]
        asset = assets_by_id[candidate]
        file_name = _CANDIDATE_FILES[candidate]
        bytes_value = (root / file_name).read_bytes()
        candidate_asset_bytes[candidate] = bytes_value
        candidate_entries[candidate] = _candidate_asset_entry(
            asset,
            file_name=file_name,
            file_sha256=_sha256_bytes(bytes_value),
        )
    merged_manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_kind": "merged_complete",
        "protocol_schema": "p2_min_v1",
        "candidate_ids": list(ordered_candidates),
        "method_lock_sha256": first["method_lock_sha256"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "source_binding": first["source_binding"],
        "episode_manifest_file": EPISODE_FILE_NAME,
        "episode_manifest_sha256": first["episode_manifest_sha256"],
        "episode_contract_sha256": first["episode_contract_sha256"],
        "qknn_lock_binding": first["qknn_lock_binding"],
        "phase1_training_receipt": {
            candidate: by_candidate[candidate][1]["phase1_training_receipt"]
            for candidate in ordered_candidates
        },
        "candidate_assets": candidate_entries,
        "source_rows_or_features_persisted": False,
        "receiver_or_class_keys_persisted": False,
        "persistent_fp32_sidecar": False,
        "target_runtime_access": False,
    }
    manifest_bytes = _canonical_bytes(merged_manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    output, staging = _prepare_output_directory(output_dir)
    try:
        _write_new(staging / EPISODE_FILE_NAME, episode_bytes)
        for candidate in ordered_candidates:
            _write_new(staging / _CANDIDATE_FILES[candidate], candidate_asset_bytes[candidate])
        _write_new(staging / MANIFEST_FILE_NAME, manifest_bytes)
        members = (EPISODE_FILE_NAME, *(_CANDIDATE_FILES[item] for item in ordered_candidates), MANIFEST_FILE_NAME)
        _write_new(staging / COMPLETION_FILE_NAME, _canonical_bytes(_completion_payload(manifest_sha256=manifest_sha, members=tuple(sorted(members)))))
        _publish_new_directory(staging, output, members=(*members, COMPLETION_FILE_NAME))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "D127_PHASE1_ASSET_BUNDLES_MERGED",
        "output_dir": str(output.resolve()),
        "manifest": str((output / MANIFEST_FILE_NAME).resolve()),
        "manifest_sha256": manifest_sha,
        "candidate_ids": list(ordered_candidates),
        "episode_contract_sha256": first["episode_contract_sha256"],
    }


def load_d127_phase1_asset_bundle(
    bundle_dir: str | Path, expected_manifest_sha256: str
) -> dict[str, QuantizedD127Asset]:
    """Read a complete, immutable A/B/C quantized asset bundle only."""

    _root, manifest, _sha, _episode, values = _load_bundle_directory(
        bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    _require(manifest["bundle_kind"] == "merged_complete", "D127 runtime loader requires a merged A/B/C bundle")
    _require(tuple(values) == CANDIDATE_IDS, "D127 runtime asset candidate ordering drift")
    return dict(values)


def _require_bound_file(
    path: str | Path, *, expected_sha256: str, name: str
) -> str:
    _source, _payload, observed = _read_regular_bytes(path, name=name)
    _require(
        observed == _sha256(expected_sha256, name=f"{name} expected hash"),
        f"{name} SHA256 mismatch",
    )
    return observed


def _load_real_d127_phase1_joined_rows(
    *,
    selected_iq_archive: str | Path,
    selected_iq_archive_sha256: str,
    selected_iq_receipt: str | Path,
    selected_iq_receipt_sha256: str,
    ls_label_join_archive: str | Path,
    ls_label_join_archive_sha256: str,
    method_lock: D127Phase1MethodLock,
) -> d106.D106JoinedLSRows:
    """Load the sealed D106 source inputs and perform the exact runtime join."""

    _require(
        _sha256(selected_iq_archive_sha256, name="selected L_s IQ")
        == method_lock.source_received_iq_sha256,
        "selected L_s IQ/method-lock binding drift",
    )
    _require(
        _sha256(selected_iq_receipt_sha256, name="selected L_s IQ receipt")
        == method_lock.source_received_iq_receipt_sha256,
        "selected L_s IQ receipt/method-lock binding drift",
    )
    _require(
        _sha256(ls_label_join_archive_sha256, name="L_s label join archive")
        == method_lock.source_label_join_archive_sha256,
        "L_s label join archive/method-lock binding drift",
    )
    _require_bound_file(
        selected_iq_archive,
        expected_sha256=selected_iq_archive_sha256,
        name="D127 selected L_s IQ archive",
    )
    _require_bound_file(
        selected_iq_receipt,
        expected_sha256=selected_iq_receipt_sha256,
        name="D127 selected L_s IQ receipt",
    )
    _require_bound_file(
        ls_label_join_archive,
        expected_sha256=ls_label_join_archive_sha256,
        name="D127 L_s label join archive",
    )
    try:
        selected = d106.load_d106_ls_received_iq(
            selected_iq_archive,
            selected_iq_receipt,
            expected_archive_sha256=selected_iq_archive_sha256,
            expected_receipt_sha256=selected_iq_receipt_sha256,
        )
    except d106.D106Phase1TapError as exc:
        raise D127Phase1ReleaseError("D127 sealed selected-IQ load failed") from exc
    _require(
        selected.receipt.get("input_ls_archive_sha256")
        == ls_label_join_archive_sha256,
        "D127 selected-IQ label-archive binding drift",
    )
    metadata = {
        "receiver_ids": selected.receiver_ids,
        "day_ids": selected.day_ids,
        "physical_ids": selected.physical_ids,
        "scenario_names": selected.scenario_names,
        "observation_ids": selected.observation_ids,
    }
    try:
        return d106.join_d106_ls_observations(
            metadata,
            selected.received_iq,
            ls_archive=ls_label_join_archive,
            expected_ls_archive_sha256=ls_label_join_archive_sha256,
        )
    except d106.D106Phase1TapError as exc:
        raise D127Phase1ReleaseError("D127 exact L_s label join failed") from exc


def _episode_raw_iq_by_id(
    plan: D127Phase1EpisodePlan,
    joined_rows: d106.D106JoinedLSRows,
    *,
    device: torch.device,
) -> Mapping[str, hooks.D127Phase1EpisodeIQ]:
    rows = np.asarray(joined_rows.received_iq, dtype=np.float32)
    result: dict[str, hooks.D127Phase1EpisodeIQ] = {}
    for episode in plan.episodes:
        support = np.ascontiguousarray(rows[list(episode.support_indices)], dtype=np.float32)
        query = np.ascontiguousarray(rows[list(episode.query_indices)], dtype=np.float32)
        result[episode.episode_id] = hooks.D127Phase1EpisodeIQ(
            episode_id=episode.episode_id,
            support_iq=numpy_to_torch_copy(
                np.array(support, copy=True),
                dtype=torch.float32,
                device=device,
                name="D127 Phase1 support IQ",
            ),
            query_iq=numpy_to_torch_copy(
                np.array(query, copy=True),
                dtype=torch.float32,
                device=device,
                name="D127 Phase1 query IQ",
            ),
        )
    return MappingProxyType(result)


CandidateEpisode: TypeAlias = assets.FSRGEpisode | assets.RDHAEpisode
FloatD127Asset: TypeAlias = da.FSRGAsset | da.RDHAAsset


@dataclass(frozen=True, slots=True)
class _CandidatePhase1Runtime:
    candidate_id: str
    bridge: Any
    episodes: tuple[CandidateEpisode, ...]
    qknn_locks: Mapping[int, qknn.Phase1ZIDStudentTLock]


def _candidate_episode_runtime(
    *,
    candidate_id: str,
    plan: D127Phase1EpisodePlan,
    joined_rows: d106.D106JoinedLSRows,
    model: Any,
    device: torch.device,
    bridge_factory: Any,
    method_lock: D127Phase1MethodLock,
) -> _CandidatePhase1Runtime:
    """Capture actual candidate taps/hidden rows through the frozen bridge."""

    _require(candidate_id in CANDIDATE_IDS, "D127 candidate runtime candidate drift")
    raw_by_id = _episode_raw_iq_by_id(plan, joined_rows, device=device)
    try:
        bridge = bridge_factory(
            model,
            candidate_id=candidate_id,
            episode_iq_by_id=raw_by_id,
        )
    except Exception as exc:
        raise D127Phase1ReleaseError("D127 Phase1 checkpoint bridge construction failed") from exc
    typed: list[CandidateEpisode] = []
    for source_episode in plan.episodes:
        try:
            support = bridge.capture_raw(source_episode.episode_id, split="support")
            query = bridge.capture_raw(source_episode.episode_id, split="query")
        except Exception as exc:
            raise D127Phase1ReleaseError("D127 Phase1 real checkpoint capture failed") from exc
        support_labels = numpy_to_torch_copy(
            source_episode.support_labels,
            dtype=torch.long,
            device=support.tap.device,
            name="D127 Phase1 support labels",
        )
        query_labels = numpy_to_torch_copy(
            source_episode.query_labels,
            dtype=torch.long,
            device=query.tap.device,
            name="D127 Phase1 query labels",
        )
        receiver_id = f"fold-{source_episode.fold_ordinal:02d}"
        try:
            if candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B):
                typed.append(
                    assets.FSRGEpisode(
                        episode_id=source_episode.episode_id,
                        receiver_id=receiver_id,
                        k_shot=source_episode.k_shot,
                        support_taps=support.tap.detach().clone(),
                        support_labels=support_labels,
                        query_taps=query.tap.detach().clone(),
                        query_labels=query_labels,
                        support_physical_ids=source_episode.support_physical_ids,
                        query_physical_ids=source_episode.query_physical_ids,
                    )
                )
            else:
                typed.append(
                    assets.RDHAEpisode(
                        episode_id=source_episode.episode_id,
                        receiver_id=receiver_id,
                        k_shot=source_episode.k_shot,
                        support_hidden=support.hidden.detach().clone(),
                        support_labels=support_labels,
                        query_hidden=query.hidden.detach().clone(),
                        query_labels=query_labels,
                        support_physical_ids=source_episode.support_physical_ids,
                        query_physical_ids=source_episode.query_physical_ids,
                    )
                )
        except assets.D127Phase1AssetError as exc:
            raise D127Phase1ReleaseError("D127 Phase1 captured episode contract drift") from exc
    return _CandidatePhase1Runtime(
        candidate_id=candidate_id,
        bridge=bridge,
        episodes=tuple(sorted(typed, key=lambda item: item.episode_id)),
        qknn_locks=build_d127_phase1_qknn_locks(method_lock),
    )


def _canonical_initialization(episodes: Sequence[CandidateEpisode]) -> assets.CanonicalRank2Initialization:
    """Derive the rank-two initialization from inner K5 support rows only."""

    k5 = [item for item in episodes if item.k_shot == 5]
    folds = tuple(sorted({item.receiver_id for item in k5}))
    _require(bool(folds) and len(k5) == len(folds), "D127 inner K5 receiver closure drift")
    receiver_means: list[Tensor] = []
    for receiver_id in folds:
        episode = next(item for item in k5 if item.receiver_id == receiver_id)
        rows = episode.support_taps if isinstance(episode, assets.FSRGEpisode) else episode.support_hidden
        labels = episode.support_labels
        classes = torch.unique(labels, sorted=True)
        _require(int(classes.numel()) == 6, "D127 inner class closure drift")
        class_means: list[Tensor] = []
        for class_id in classes.tolist():
            index = torch.nonzero(labels == int(class_id), as_tuple=False).reshape(-1)
            value = rows.index_select(0, index)
            class_means.append(
                value.mean(dim=(0, 2)) if value.ndim == 3 else value.mean(dim=0)
            )
        receiver_means.append(torch.stack(class_means, dim=0))
    stacked = torch.stack(receiver_means, dim=0)
    return assets.canonical_receiver_mean_svd(
        stacked,
        dimension=int(stacked.shape[-1]),
    )


def _optimizer_receipt_payload(receipt: assets.DeterministicLBFGSReceipt) -> dict[str, Any]:
    _require(receipt.max_iter == 128 and receipt.line_search_fn == "strong_wolfe" and receipt.initialization_count == 1, "D127 fixed optimizer receipt drift")
    return {
        "parameter_names": list(receipt.parameter_names),
        "max_iter": receipt.max_iter,
        "line_search_fn": receipt.line_search_fn,
        "initialization_count": receipt.initialization_count,
        "closure_calls": receipt.closure_calls,
        "internal_iterations": receipt.internal_iterations,
        "initial_gradient_nonzero": receipt.initial_gradient_norm > 0.0,
        "initial_parameter_gradients_nonzero": [value > 0.0 for value in receipt.initial_parameter_gradient_norms],
    }


def _train_candidate_on_episodes(
    runtime: _CandidatePhase1Runtime,
    episodes: Sequence[CandidateEpisode],
) -> tuple[FloatD127Asset, assets.DeterministicLBFGSReceipt, Any]:
    """Train only through the production bridge callback surfaces."""

    initialization = _canonical_initialization(episodes)
    try:
        if runtime.candidate_id in (da.CANDIDATE_A, da.CANDIDATE_B):
            typed = tuple(episodes)
            _require(all(isinstance(item, assets.FSRGEpisode) for item in typed), "D127 FSRG episode type drift")
            callbacks = runtime.bridge.fsrg_loss_callbacks(qknn_locks=runtime.qknn_locks)
            result = assets.train_fsrg_phase1_asset(
                candidate_id=runtime.candidate_id,
                episodes=assets.FrozenFSRGEpisodes(typed),
                initialization=initialization,
                callbacks=callbacks,
            )
            return result.asset, result.receipt, callbacks
        typed_c = tuple(episodes)
        _require(all(isinstance(item, assets.RDHAEpisode) for item in typed_c), "D127 RDHA episode type drift")
        callback = runtime.bridge.rdha_outer_callback(qknn_locks=runtime.qknn_locks)
        result_c = assets.train_rdah_phase1_asset(
            episodes=assets.FrozenRDHAEpisodes(typed_c),
            initialization=initialization,
            outer_query_per_sample=callback,
        )
        return result_c.asset, result_c.receipt, callback
    except (assets.D127Phase1AssetError, da.D127DACandidateError, hooks.D127CheckpointHookError) as exc:
        raise D127Phase1ReleaseError("D127 Phase1 fixed-LBFGS training failed") from exc


def _cycled_episode(episode: CandidateEpisode) -> CandidateEpisode:
    support = torch.remainder(episode.support_labels + 1, 6).to(dtype=episode.support_labels.dtype)
    query = torch.remainder(episode.query_labels + 1, 6).to(dtype=episode.query_labels.dtype)
    return replace(episode, support_labels=support, query_labels=query)


def _fsrg_audit(
    *,
    bridge: Any,
    episode: assets.FSRGEpisode,
    asset: da.FSRGAsset,
    callbacks: assets.FSRGLossCallbacks,
) -> tuple[Tensor, bool, bool]:
    state = da.fit_fsrg_support_state(
        episode.support_taps,
        episode.support_labels,
        asset,
        lambda adapted: callbacks.support_per_sample(episode, adapted),
    )
    support = da.apply_fsrg_outer(episode.support_taps, asset, state)
    query = da.apply_fsrg_outer(episode.query_taps, asset, state)
    losses = callbacks.outer_query_per_sample(episode, support, query)
    value = da.class_balanced_support_loss(losses, episode.query_labels)
    base = bridge.capture_raw(episode.episode_id, split="query").z_id
    changed = bridge.forward_with_replacement(
        episode, split="query", replacement=query
    ).z_id
    nonzero = bool(torch.any(torch.abs(state.support_gradient) > 0.0).item()) and bool(torch.any(torch.abs(state.a) > 0.0).item())
    return value, nonzero, not bool(torch.equal(base.detach(), changed.detach()))


def _rdha_audit(
    *,
    bridge: Any,
    episode: assets.RDHAEpisode,
    asset: da.RDHAAsset,
    callback: assets.RDHALossCallback,
) -> tuple[Tensor, bool, bool]:
    outer = da.apply_rdah_outer(
        episode.support_hidden,
        episode.support_labels,
        episode.query_hidden,
        asset,
    )
    state = da.fit_rdah_support_state(episode.support_hidden, episode.support_labels, asset)
    losses = callback(episode, outer.adapted_support, outer.adapted_query)
    value = da.class_balanced_support_loss(losses, episode.query_labels)
    base = bridge.capture_raw(episode.episode_id, split="query").z_id
    changed = bridge.forward_with_replacement(
        episode, split="query", replacement=outer.adapted_query
    ).z_id
    nonzero = bool(torch.any(torch.abs(state.a) > 0.0).item())
    return value, nonzero, not bool(torch.equal(base.detach(), changed.detach()))


def _audit_one_episode(
    runtime: _CandidatePhase1Runtime,
    *,
    episode: CandidateEpisode,
    asset: FloatD127Asset,
    callback: Any,
) -> tuple[Tensor, bool, bool]:
    if isinstance(episode, assets.FSRGEpisode):
        _require(isinstance(asset, da.FSRGAsset) and isinstance(callback, assets.FSRGLossCallbacks), "D127 FSRG audit binding drift")
        return _fsrg_audit(bridge=runtime.bridge, episode=episode, asset=asset, callbacks=callback)
    _require(isinstance(asset, da.RDHAAsset) and callable(callback), "D127 RDHA audit binding drift")
    return _rdha_audit(bridge=runtime.bridge, episode=episode, asset=asset, callback=callback)


def _candidate_logits(
    runtime: _CandidatePhase1Runtime,
    *,
    episode: CandidateEpisode,
    asset: FloatD127Asset,
    callback: Any,
) -> Tensor:
    """Use the same real downstream qKNN surface for the parity fixture."""

    if isinstance(episode, assets.FSRGEpisode):
        _require(isinstance(asset, da.FSRGAsset) and isinstance(callback, assets.FSRGLossCallbacks), "D127 FSRG parity binding drift")
        state = da.fit_fsrg_support_state(
            episode.support_taps,
            episode.support_labels,
            asset,
            lambda adapted: callback.support_per_sample(episode, adapted),
        )
        support = da.apply_fsrg_outer(episode.support_taps, asset, state)
        query = da.apply_fsrg_outer(episode.query_taps, asset, state)
    else:
        _require(isinstance(asset, da.RDHAAsset) and callable(callback), "D127 RDHA parity binding drift")
        outer = da.apply_rdah_outer(
            episode.support_hidden,
            episode.support_labels,
            episode.query_hidden,
            asset,
        )
        support = outer.adapted_support
        query = outer.adapted_query
    support_forward = runtime.bridge.forward_with_replacement(
        episode, split="support", replacement=support
    )
    query_forward = runtime.bridge.forward_with_replacement(
        episode, split="query", replacement=query
    )
    return runtime.bridge.deployment_qknn_logits(
        episode,
        support_zid=support_forward.z_id,
        query_zid=query_forward.z_id,
        qknn_locks=runtime.qknn_locks,
    )


def _outer_fold_audit_and_final_rebuild(
    runtime: _CandidatePhase1Runtime,
) -> tuple[QuantizedD127Asset, dict[str, Any]]:
    """Run all seven source-held audits, then rebuild once from all 14 episodes.

    The receipt intentionally records only execution/protocol booleans.  It
    contains no held accuracy, loss ranking, or release threshold.
    """

    outer_receipts: list[dict[str, Any]] = []
    for held_fold in range(7):
        inner = tuple(item for item in runtime.episodes if item.receiver_id != f"fold-{held_fold:02d}")
        held = tuple(item for item in runtime.episodes if item.receiver_id == f"fold-{held_fold:02d}")
        _require(len(inner) == 12 and len(held) == 2, "D127 outer-fold isolation closure drift")
        float_asset, optimizer_receipt, callback = _train_candidate_on_episodes(runtime, inner)
        finite = True
        nonzero = True
        changed = True
        equivariant = True
        for episode in held:
            value, state_nonzero, query_changed = _audit_one_episode(
                runtime,
                episode=episode,
                asset=float_asset,
                callback=callback,
            )
            cycled = _cycled_episode(episode)
            cycled_value, cycle_nonzero, cycle_changed = _audit_one_episode(
                runtime,
                episode=cycled,
                asset=float_asset,
                callback=callback,
            )
            finite = finite and bool(torch.isfinite(value).item()) and bool(torch.isfinite(cycled_value).item())
            nonzero = nonzero and state_nonzero and cycle_nonzero
            changed = changed and query_changed and cycle_changed
            equivariant = equivariant and bool(torch.allclose(value.detach(), cycled_value.detach(), rtol=2.0e-5, atol=2.0e-6))
        _require(finite and nonzero and changed and equivariant, "D127 outer audit failed isolation/equivariance/nonzero/query-change closure")
        outer_receipts.append(
            {
                "held_fold_ordinal": held_fold,
                "inner_fold_count": 6,
                "held_k": [1, 5],
                "inner_optimizer": _optimizer_receipt_payload(optimizer_receipt),
                "source_held_isolation": True,
                "fixed_cyclic_label_equivariant": True,
                "nonzero_state_or_gradient": True,
                "query_function_changed": True,
                "outer_query_function_finite": True,
                "performance_threshold_or_ranking_used": False,
            }
        )
    final_asset, final_optimizer, final_callback = _train_candidate_on_episodes(runtime, runtime.episodes)
    if isinstance(final_asset, da.FSRGAsset):
        quantized: QuantizedD127Asset = assets.quantize_fsrg_asset(final_asset)
    else:
        quantized = assets.quantize_rdah_asset(final_asset)
    assets.assert_no_persistent_fp32_sidecar(quantized)
    fixture = next(item for item in runtime.episodes if item.k_shot == 5)
    reference_logits = _candidate_logits(runtime, episode=fixture, asset=final_asset, callback=final_callback)
    decoded_logits = _candidate_logits(
        runtime,
        episode=fixture,
        asset=quantized.decode(device=reference_logits.device),
        callback=final_callback,
    )
    parity = assets.function_argmax_parity_receipt(
        fixture_id=f"d127-phase1-final-k5-{runtime.candidate_id}",
        reference_output=reference_logits,
        quantized_output=decoded_logits,
    )
    receipt = {
        "schema": "cvs.phase1.d127.training_receipt.v1",
        "candidate_id": runtime.candidate_id,
        "execution": "REAL_CHECKPOINT_SOURCE_ONLY_OUTER7_FINAL14",
        "outer_fold_count": 7,
        "outer_folds": outer_receipts,
        "final_rebuild": {
            "episode_count": 14,
            "k_values": [1, 5],
            "optimizer": _optimizer_receipt_payload(final_optimizer),
        },
        "quantization_parity": {
            "fixture_id": parity.fixture_id,
            "output_shape": list(parity.output_shape),
            "element_count": parity.element_count,
            "max_abs_error": parity.max_abs_error,
            "mean_abs_error": parity.mean_abs_error,
            "argmax_agreement": parity.argmax_agreement,
            "argmax_equal": parity.argmax_equal,
        },
        "source_rows_or_features_persisted": False,
        "target_runtime_access": False,
        "performance_threshold_or_ranking_used": False,
    }
    return quantized, receipt


def _build_d127_phase1_single_candidate_from_joined_rows(
    *,
    candidate_id: str,
    output_dir: str | Path,
    method_lock: D127Phase1MethodLock,
    joined_rows: d106.D106JoinedLSRows,
    model: Any,
    device: torch.device,
    bridge_factory: Any,
) -> dict[str, Any]:
    """Internal testable path; production supplies only the real hook bridge."""

    plan = build_d127_phase1_episode_plan(joined_rows, method_lock=method_lock)
    runtime = _candidate_episode_runtime(
        candidate_id=candidate_id,
        plan=plan,
        joined_rows=joined_rows,
        model=model,
        device=device,
        bridge_factory=bridge_factory,
        method_lock=method_lock,
    )
    quantized, training_receipt = _outer_fold_audit_and_final_rebuild(runtime)
    result = write_d127_phase1_single_candidate_bundle(
        output_dir=output_dir,
        candidate_id=candidate_id,
        asset=quantized,
        method_lock=method_lock,
        episode_plan=plan,
        phase1_training_receipt=training_receipt,
    )
    return {
        **result,
        "outer_fold_count": 7,
        "final_episode_count": 14,
        "training_execution": training_receipt["execution"],
    }


def build_d127_phase1_single_candidate_from_source(
    *,
    candidate_id: str,
    output_dir: str | Path,
    method_lock_path: str | Path,
    method_lock_sha256: str,
    selected_iq_archive: str | Path,
    selected_iq_archive_sha256: str,
    selected_iq_receipt: str | Path,
    selected_iq_receipt_sha256: str,
    ls_label_join_archive: str | Path,
    ls_label_join_archive_sha256: str,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Production-only real checkpoint build for one frozen D127 candidate."""

    _require(candidate_id in CANDIDATE_IDS, "D127 source build candidate drift")
    method_lock = load_d127_phase1_method_lock(
        method_lock_path,
        expected_sha256=method_lock_sha256,
    )
    _require(
        _sha256(checkpoint_sha256, name="D127 checkpoint") == method_lock.checkpoint_sha256,
        "D127 checkpoint/method-lock binding drift",
    )
    _require_bound_file(checkpoint, expected_sha256=checkpoint_sha256, name="D127 checkpoint")
    joined_rows = _load_real_d127_phase1_joined_rows(
        selected_iq_archive=selected_iq_archive,
        selected_iq_archive_sha256=selected_iq_archive_sha256,
        selected_iq_receipt=selected_iq_receipt,
        selected_iq_receipt_sha256=selected_iq_receipt_sha256,
        ls_label_join_archive=ls_label_join_archive,
        ls_label_join_archive_sha256=ls_label_join_archive_sha256,
        method_lock=method_lock,
    )
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise D127Phase1ReleaseError("D127 device is invalid") from exc
    try:
        model, checkpoint_receipt = hooks.load_d127_frozen_checkpoint(
            checkpoint,
            device=torch_device,
        )
    except hooks.D127CheckpointHookError as exc:
        raise D127Phase1ReleaseError("D127 real checkpoint reconstruction failed") from exc
    _require(
        checkpoint_receipt.get("checkpoint_sha256") == method_lock.checkpoint_sha256
        and checkpoint_receipt.get("all_checkpoint_parameters_frozen") is True
        and checkpoint_receipt.get("eval_mode") is True,
        "D127 checkpoint receipt closure drift",
    )
    return _build_d127_phase1_single_candidate_from_joined_rows(
        candidate_id=candidate_id,
        output_dir=output_dir,
        method_lock=method_lock,
        joined_rows=joined_rows,
        model=model,
        device=torch_device,
        bridge_factory=hooks.D127Phase1CheckpointBridge,
    )


__all__ = [
    "ASSET_WIRE_SCHEMA",
    "BUNDLE_SCHEMA",
    "CANDIDATE_IDS",
    "D127Phase1Episode",
    "D127Phase1EpisodePlan",
    "D127Phase1MethodLock",
    "D127Phase1ReleaseError",
    "QuantizedD127Asset",
    "build_d127_phase1_episode_plan",
    "build_d127_phase1_qknn_locks",
    "build_d127_phase1_single_candidate_from_source",
    "load_d127_phase1_asset_bundle",
    "load_d127_phase1_method_lock",
    "merge_d127_phase1_asset_bundles",
    "write_d127_phase1_single_candidate_bundle",
]
