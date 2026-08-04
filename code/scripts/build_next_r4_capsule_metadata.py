#!/usr/bin/env python3
"""Build the immutable NEXT-R4 Phase2 capsule/split metadata.

This entry point is intentionally mechanical.  It reads the sealed D106
strict feature tap and the received-IQ archive, checks that the two archives
describe the same ordered physical observations, and emits only canonical
JSON metadata.  No feature, IQ, truth, metric, or performance value is
opened for a decision.  The only choices made here are the frozen six-class
registry, the two held receivers, and the deterministic opaque-ID split.

The prepare-side metadata keeps class-grouped query maps.  The NEXT-R4
``prepare`` CLI consumes those maps and strips the grouping before the
predictor package is sealed; this builder therefore must not be reused as a
predictor-side package writer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from cvsrffi import stage2_next_r4_matrix as matrix  # noqa: E402
from cvsrffi import stage2_zid_student_t_qknn as qknn  # noqa: E402


CAPSULE_METADATA_SCHEMA = (
    "cvs.stage2.next_r4.fa_rdce3_cer_plr160.capsule_metadata.v1"
)
CAPSULE_IDENTITY_SCHEMA = "cvs.stage2.next_r4.capsule_identity.v1"
SPLIT_IDENTITY_SCHEMA = "cvs.stage2.next_r4.split_identity.v1"
BUILD_STATUS = "NEXT_R4_CAPSULE_METADATA_COMPLETE"
PROTOCOL_SCHEMA = matrix.PROTOCOL_SCHEMA

# The R4 metadata split has no free seed/split knob.  This value is retained
# only because the existing runtime binding receipt carries a nonnegative seed
# field; it is not used to select data or method parameters.
RUNTIME_RECEIPT_SEED = 0

# This is the single frozen opaque-ID ordering salt for this metadata
# revision.  It is deliberately a public constant so an independent checker
# can reproduce the selection without seeing any label or feature data.
PHYSICAL_ID_SORT_SALT = "cvs.stage2.next_r4.proxy24.opaque_physical_id_order.v1"
PHYSICAL_ID_SORT_POLICY = {
    "algorithm": "sha256",
    "salt": PHYSICAL_ID_SORT_SALT,
    "input": "utf8(salt + '|' + opaque_physical_id)",
    "order": "ascending_digest_then_opaque_id",
}

FIXED_CLASSES = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
HELD_RECEIVERS = matrix.HELD_RECEIVERS
EXPECTED_ROWS = 588
EXPECTED_RECEIVER_COUNT = 7
EXPECTED_CLASS_COUNT = 6
EXPECTED_PER_RECEIVER_CLASS = 14
EXPECTED_PHASE1_FIT_COUNT = 420
K1_SUPPORT_COUNT = 1
K5_SUPPORT_COUNT = 5
QUERY_PER_CLASS = 9

STRICT_TAP_REQUIRED_MEMBERS = (
    "pre_relu",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
STRICT_TAP_OPTIONAL_MEMBERS = ("z_dom",)
RECEIVED_REQUIRED_MEMBERS = (
    "received_iq",
    "receiver_ids",
    "physical_ids",
    "observation_ids",
)
RECEIVED_OPTIONAL_MEMBERS = ("day_ids", "scenario_names")

LOCK_FIELDS = frozenset(
    {
        "active_k",
        "student_nu",
        "kernel_effective_dim",
        "kernel_volume_gamma",
        "shared_h0",
        "scale_prior_strength",
        "scale_min_ratio",
        "scale_max_ratio",
        "temperature",
        "phase1_lodo_receipt_sha256",
        "quantization_margin_audit_sha256",
        "schema",
    }
)


class NextR4CapsuleMetadataError(ValueError):
    """Raised when the frozen NEXT-R4 metadata boundary does not close."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _require_sha(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise NextR4CapsuleMetadataError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR4CapsuleMetadataError(f"{name} must be a lowercase SHA256") from error
    return value


def _regular_bytes(path: Path, *, name: str, expected_sha256: str | None = None) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NextR4CapsuleMetadataError(
            f"{name} must be an absolute regular non-symlink file"
        )
    payload = path.read_bytes()
    observed = _sha_bytes(payload)
    if expected_sha256 is not None and observed != _require_sha(
        expected_sha256, name=f"{name} expected SHA256"
    ):
        raise NextR4CapsuleMetadataError(f"{name} SHA256 mismatch")
    return payload


def _strings(value: Any, *, name: str, count: int, unique: bool = False) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"} or len(array) != count:
        raise NextR4CapsuleMetadataError(f"{name} must be a string vector of length {count}")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise NextR4CapsuleMetadataError(f"{name} contains a blank value")
    if unique and len(set(result)) != len(result):
        raise NextR4CapsuleMetadataError(f"{name} contains duplicate values")
    return result


def _load_npz(payload: bytes, *, name: str) -> Mapping[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except Exception as error:
        raise NextR4CapsuleMetadataError(f"{name} must be a no-pickle NPZ") from error


def _validate_strict_tap(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    payload = _regular_bytes(path, name="D106 strict tap", expected_sha256=expected_sha256)
    arrays = _load_npz(payload, name="D106 strict tap")
    keys = set(arrays)
    required = set(STRICT_TAP_REQUIRED_MEMBERS)
    allowed = required | set(STRICT_TAP_OPTIONAL_MEMBERS)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise NextR4CapsuleMetadataError("D106 strict tap member closure drift")
    pre_relu = np.asarray(arrays["pre_relu"])
    if (
        pre_relu.dtype != np.dtype("<f4")
        or pre_relu.shape != (EXPECTED_ROWS, 160)
        or not np.isfinite(pre_relu).all()
    ):
        raise NextR4CapsuleMetadataError("D106 strict tap pre_relu dtype/shape/finite drift")
    # z_dom is intentionally not consumed; if present it is checked only for
    # a finite row-aligned closure, never used to construct this split.
    if "z_dom" in arrays:
        z_dom = np.asarray(arrays["z_dom"])
        if (
            z_dom.dtype != np.dtype("<f4")
            or z_dom.shape != (EXPECTED_ROWS, 160)
            or not np.isfinite(z_dom).all()
        ):
            raise NextR4CapsuleMetadataError("D106 strict tap z_dom dtype/shape/finite drift")
    values: dict[str, Any] = {"pre_relu": pre_relu}
    for key in STRICT_TAP_REQUIRED_MEMBERS[1:]:
        values[key] = _strings(
            arrays[key],
            name=f"strict tap {key}",
            count=EXPECTED_ROWS,
            unique=key in {"physical_ids", "observation_ids"},
        )
    receivers = tuple(sorted(set(values["receiver_ids"])))
    classes = tuple(sorted(set(values["tx_labels"])))
    if len(receivers) != EXPECTED_RECEIVER_COUNT or len(classes) != EXPECTED_CLASS_COUNT:
        raise NextR4CapsuleMetadataError("D106 strict tap receiver/class registry drift")
    if classes != FIXED_CLASSES or any(item not in receivers for item in HELD_RECEIVERS):
        raise NextR4CapsuleMetadataError("D106 strict tap fixed class/held receiver registry drift")
    receiver_class_counts: dict[tuple[str, str], int] = {}
    for receiver, class_id in zip(values["receiver_ids"], values["tx_labels"], strict=True):
        key = (receiver, class_id)
        receiver_class_counts[key] = receiver_class_counts.get(key, 0) + 1
    if set(receiver_class_counts) != {
        (receiver, class_id) for receiver in receivers for class_id in FIXED_CLASSES
    } or any(
        receiver_class_counts[key] != EXPECTED_PER_RECEIVER_CLASS
        for key in receiver_class_counts
    ):
        raise NextR4CapsuleMetadataError(
            "D106 strict tap must close 7x6x14 receiver/class rows"
        )
    return values, _sha_bytes(payload)


def _validate_received_iq(
    path: Path,
    *,
    strict: Mapping[str, Any],
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    payload = _regular_bytes(path, name="received-IQ archive", expected_sha256=expected_sha256)
    arrays = _load_npz(payload, name="received-IQ archive")
    required = set(RECEIVED_REQUIRED_MEMBERS)
    allowed = required | set(RECEIVED_OPTIONAL_MEMBERS)
    if not required.issubset(arrays) or not set(arrays).issubset(allowed):
        raise NextR4CapsuleMetadataError("received-IQ member closure drift")
    iq = np.asarray(arrays["received_iq"])
    if (
        iq.dtype != np.dtype("<f4")
        or iq.ndim != 3
        or iq.shape[0] != EXPECTED_ROWS
        or iq.shape[1] != 2
        or iq.shape[2] < 1
        or not np.isfinite(iq).all()
    ):
        raise NextR4CapsuleMetadataError(
            "received-IQ must be finite little-endian float32 [588,2,T]"
        )
    values: dict[str, Any] = {"received_iq": np.ascontiguousarray(iq, dtype=np.float32)}
    for key in ("receiver_ids", "physical_ids", "observation_ids"):
        values[key] = _strings(
            arrays[key],
            name=f"received {key}",
            count=EXPECTED_ROWS,
            unique=key in {"physical_ids", "observation_ids"},
        )
    for key in RECEIVED_OPTIONAL_MEMBERS:
        if key in arrays:
            values[key] = _strings(
                arrays[key], name=f"received {key}", count=EXPECTED_ROWS
            )
    for key in ("receiver_ids", "physical_ids", "observation_ids"):
        if values[key] != strict[key]:
            raise NextR4CapsuleMetadataError(
                f"strict tap/received {key} itemwise order mismatch"
            )
    for key in RECEIVED_OPTIONAL_MEMBERS:
        if key in values and values[key] != strict[key]:
            raise NextR4CapsuleMetadataError(
                f"strict tap/received {key} itemwise order mismatch"
            )
    return values, _sha_bytes(payload)


def _lock_mapping(value: Any, *, name: str, expected_k: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != LOCK_FIELDS:
        raise NextR4CapsuleMetadataError(
            f"{name} must contain exactly the frozen qKNN lock fields"
        )
    if value.get("active_k") != expected_k or value.get("schema") != qknn.LOCK_SCHEMA:
        raise NextR4CapsuleMetadataError(f"{name} schema/active K drift")
    # Constructing the existing typed lock validates positivity and hash
    # fields, but the original mapping is returned untouched below.
    try:
        qknn.Phase1ZIDStudentTLock(**value)
    except Exception as error:
        raise NextR4CapsuleMetadataError(f"{name} typed qKNN lock validation failed") from error
    return dict(value)


def _load_qknn_locks(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = _regular_bytes(path, name="qKNN locks JSON")
    try:
        source = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR4CapsuleMetadataError("qKNN locks must be UTF-8 JSON") from error
    if type(source) is not dict:
        raise NextR4CapsuleMetadataError("qKNN locks must be a JSON object")
    # Accept the two external spellings already used by the R4 tooling while
    # requiring exactly K1 and K5 lock objects in either case.
    if set(source) == {"K1", "K5"}:
        raw = {"1": source["K1"], "5": source["K5"]}
    elif set(source) == {"1", "5"}:
        raw = {"1": source["1"], "5": source["5"]}
    elif set(source) == {"schema", "K1", "K5"}:
        if source["schema"] != qknn.LOCK_SCHEMA:
            raise NextR4CapsuleMetadataError("qKNN lock container schema drift")
        raw = {"1": source["K1"], "5": source["K5"]}
    elif set(source) == {"schema", "qknn_lock_by_k"}:
        if source["schema"] != qknn.LOCK_SCHEMA or type(source["qknn_lock_by_k"]) is not dict:
            raise NextR4CapsuleMetadataError("qKNN lock container schema drift")
        if set(source["qknn_lock_by_k"]) != {"1", "5"}:
            raise NextR4CapsuleMetadataError("qKNN locks must contain exactly K1/K5")
        raw = dict(source["qknn_lock_by_k"])
    else:
        raise NextR4CapsuleMetadataError(
            "qKNN locks must contain exactly K1/K5 (with optional schema container)"
        )
    lock1 = _lock_mapping(raw["1"], name="qKNN K1 lock", expected_k=1)
    lock5 = _lock_mapping(raw["5"], name="qKNN K5 lock", expected_k=5)
    # Values are untouched; only the outer key spelling is made explicit for
    # the predictor's existing qknn_lock_by_k contract.
    return {"1": lock1, "5": lock5}, source, _sha_bytes(payload)


def _opaque_sort_key(physical_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{PHYSICAL_ID_SORT_SALT}|{physical_id}".encode("utf-8")
    ).hexdigest()
    return digest, physical_id


def _ordered_root(values: Sequence[str]) -> str:
    if not values or any(type(item) is not str or not item for item in values):
        raise NextR4CapsuleMetadataError("ordered ID root input contains a blank/non-string")
    if len(set(values)) != len(values):
        raise NextR4CapsuleMetadataError("ordered ID root input contains duplicates")
    return _sha_bytes("\n".join(values).encode("utf-8"))


def _row_split(
    *,
    strict: Mapping[str, Any],
    held_receiver: str,
    held_class: str,
) -> dict[str, Any]:
    physical = strict["physical_ids"]
    observations = strict["observation_ids"]
    by_group: dict[str, list[tuple[str, str]]] = {class_id: [] for class_id in FIXED_CLASSES}
    for receiver, class_id, physical_id, observation_id in zip(
        strict["receiver_ids"],
        strict["tx_labels"],
        physical,
        observations,
        strict=True,
    ):
        if receiver == held_receiver:
            by_group[class_id].append((physical_id, observation_id))
    if any(len(by_group[class_id]) != EXPECTED_PER_RECEIVER_CLASS for class_id in FIXED_CLASSES):
        raise NextR4CapsuleMetadataError("held receiver/class physical group count drift")
    ordered: dict[str, tuple[tuple[str, str], ...]] = {}
    for class_id in FIXED_CLASSES:
        entries = tuple(sorted(by_group[class_id], key=lambda pair: _opaque_sort_key(pair[0])))
        if len({pair[0] for pair in entries}) != EXPECTED_PER_RECEIVER_CLASS:
            raise NextR4CapsuleMetadataError("opaque physical-ID group is not unique")
        ordered[class_id] = entries
    k5 = {class_id: tuple(pair[0] for pair in ordered[class_id][:K5_SUPPORT_COUNT]) for class_id in FIXED_CLASSES}
    k1 = {class_id: values[:K1_SUPPORT_COUNT] for class_id, values in k5.items()}
    query = {
        class_id: tuple(pair[0] for pair in ordered[class_id][K5_SUPPORT_COUNT:])
        for class_id in FIXED_CLASSES
    }
    query_observation = {
        class_id: tuple(pair[1] for pair in ordered[class_id][K5_SUPPORT_COUNT:])
        for class_id in FIXED_CLASSES
    }
    phase1 = tuple(
        physical_id
        for receiver, class_id, physical_id in zip(
            strict["receiver_ids"], strict["tx_labels"], physical, strict=True
        )
        if receiver != held_receiver and class_id != held_class
    )
    if len(phase1) != EXPECTED_PHASE1_FIT_COUNT or len(set(phase1)) != len(phase1):
        raise NextR4CapsuleMetadataError("Phase1 fit exclusion must produce exactly 420 IDs")
    support_union = {item for values in k5.values() for item in values}
    query_union = {item for values in query.values() for item in values}
    observation_union = {item for values in query_observation.values() for item in values}
    if len(support_union) != EXPECTED_CLASS_COUNT * K5_SUPPORT_COUNT:
        raise NextR4CapsuleMetadataError("K5 support physical IDs overlap")
    if len(query_union) != EXPECTED_CLASS_COUNT * QUERY_PER_CLASS:
        raise NextR4CapsuleMetadataError("query physical IDs do not close over 9/class")
    if len(observation_union) != EXPECTED_CLASS_COUNT * QUERY_PER_CLASS:
        raise NextR4CapsuleMetadataError("query observation IDs do not close over 9/class")
    if support_union & query_union or support_union & observation_union or set(phase1) & (support_union | query_union | observation_union):
        raise NextR4CapsuleMetadataError("support/query/Phase1 physical-ID disjointness drift")
    if any(k1[class_id] != k5[class_id][:1] for class_id in FIXED_CLASSES):
        raise NextR4CapsuleMetadataError("K1 support is not the exact K5 prefix")
    return {
        "k1_support_ids_by_class": {class_id: list(k1[class_id]) for class_id in FIXED_CLASSES},
        "k5_support_ids_by_class": {class_id: list(k5[class_id]) for class_id in FIXED_CLASSES},
        "query_ids_by_class": {class_id: list(query[class_id]) for class_id in FIXED_CLASSES},
        "query_observation_ids_by_class": {
            class_id: list(query_observation[class_id]) for class_id in FIXED_CLASSES
        },
        "phase1_fit_ids": list(phase1),
        "phase1_fit_count": len(phase1),
        "phase1_fit_physical_root_sha256": _ordered_root(phase1),
        "support_k1_physical_root_sha256": _ordered_root(
            tuple(item for class_id in FIXED_CLASSES for item in k1[class_id])
        ),
        "support_k5_physical_root_sha256": _ordered_root(
            tuple(item for class_id in FIXED_CLASSES for item in k5[class_id])
        ),
        "query_physical_root_sha256": _ordered_root(
            tuple(item for class_id in FIXED_CLASSES for item in query[class_id])
        ),
        "query_observation_root_sha256": _ordered_root(
            tuple(item for class_id in FIXED_CLASSES for item in query_observation[class_id])
        ),
        "k1_is_exact_k5_prefix": True,
        "support_query_physical_ids_disjoint": True,
        "support_query_observation_ids_disjoint": True,
        "query_count_by_class": {class_id: QUERY_PER_CLASS for class_id in FIXED_CLASSES},
    }


def _build_metadata(
    *,
    strict: Mapping[str, Any],
    strict_tap_sha256: str,
    received: Mapping[str, Any],
    received_iq_sha256: str,
    qknn_lock_by_k: Mapping[str, Mapping[str, Any]],
    qknn_locks_source: Mapping[str, Any],
    qknn_locks_sha256: str,
    validator_receipt_sha256: str,
) -> dict[str, Any]:
    plan = matrix.build_next_r4_proxy24_plan(FIXED_CLASSES)
    matrix.validate_next_r4_proxy24_plan(plan)
    rows: list[dict[str, Any]] = []
    by_outer: dict[tuple[str, str], dict[str, Any]] = {}
    for planned in plan["rows"]:
        key = (str(planned["held_receiver"]), str(planned["held_class"]))
        if key not in by_outer:
            by_outer[key] = _row_split(
                strict=strict, held_receiver=key[0], held_class=key[1]
            )
        row = dict(planned)
        row.update(by_outer[key])
        # Re-run the frozen matrix binding as an independent mechanical
        # legality check.  Its flattened receipt is intentionally not emitted
        # here because this is the prepare-side, class-grouped metadata.
        k1_planned = next(
            item
            for item in plan["rows"]
            if item["held_receiver"] == key[0]
            and item["held_class"] == key[1]
            and int(item["active_k"]) == 1
        )
        k5_planned = next(
            item
            for item in plan["rows"]
            if item["held_receiver"] == key[0]
            and item["held_class"] == key[1]
            and int(item["active_k"]) == 5
        )
        try:
            matrix.bind_next_r4_physical_ids(
                row_k1=matrix.outer_key_from_mapping(k1_planned),
                row_k5=matrix.outer_key_from_mapping(k5_planned),
                phase1_fit_ids=row["phase1_fit_ids"],
                k1_support_ids_by_class=row["k1_support_ids_by_class"],
                k5_support_ids_by_class=row["k5_support_ids_by_class"],
                query_ids_by_class=row["query_ids_by_class"],
                query_observation_ids_by_class=row["query_observation_ids_by_class"],
                query_ids_by_view={view: row["query_ids_by_class"] for view in matrix.QUERY_VIEW_IDS},
                query_observation_ids_by_view={
                    view: row["query_observation_ids_by_class"] for view in matrix.QUERY_VIEW_IDS
                },
            )
        except Exception as error:
            raise NextR4CapsuleMetadataError(
                f"NEXT-R4 row physical binding failed for {key[0]}|{key[1]}"
            ) from error
        rows.append(row)
    if len(rows) != matrix.ROW_COUNT or len({row["row_id"] for row in rows}) != matrix.ROW_COUNT:
        raise NextR4CapsuleMetadataError("NEXT-R4 metadata must close over 24 unique rows")

    physical_root = _ordered_root(tuple(strict["physical_ids"]))
    observation_root = _ordered_root(tuple(strict["observation_ids"]))
    capsule_identity = {
        "schema": CAPSULE_IDENTITY_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "received_iq_sha256": received_iq_sha256,
        "strict_tap_sha256": strict_tap_sha256,
        "row_count": EXPECTED_ROWS,
        "physical_id_root_sha256": physical_root,
        "observation_id_root_sha256": observation_root,
        "receiver_registry": sorted(set(strict["receiver_ids"])),
        "class_registry": list(FIXED_CLASSES),
        "held_receivers": list(HELD_RECEIVERS),
        "single_leo_observation": True,
    }
    capsule_id = _canonical_sha(capsule_identity)
    row_id_order = [str(row["row_id"]) for row in rows]
    row_id_set = sorted(row_id_order)
    row_identity = [
        {
            "row_id": row["row_id"],
            "held_receiver": row["held_receiver"],
            "held_class": row["held_class"],
            "active_k": row["active_k"],
            "phase1_fit_physical_root_sha256": row["phase1_fit_physical_root_sha256"],
            "support_k1_physical_root_sha256": row["support_k1_physical_root_sha256"],
            "support_k5_physical_root_sha256": row["support_k5_physical_root_sha256"],
            "query_physical_root_sha256": row["query_physical_root_sha256"],
            "query_observation_root_sha256": row["query_observation_root_sha256"],
        }
        for row in rows
    ]
    split_identity = {
        "schema": SPLIT_IDENTITY_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "capsule_id": capsule_id,
        "class_registry": list(FIXED_CLASSES),
        "held_receivers": list(HELD_RECEIVERS),
        "row_count": len(rows),
        "row_id_order": row_id_order,
        "row_id_set_sorted": row_id_set,
        "row_identity": row_identity,
        "physical_id_sort_policy": PHYSICAL_ID_SORT_POLICY,
        "k1_support_count": K1_SUPPORT_COUNT,
        "k5_support_count": K5_SUPPORT_COUNT,
        "query_per_class": QUERY_PER_CLASS,
        "phase1_fit_count": EXPECTED_PHASE1_FIT_COUNT,
    }
    split_id = _canonical_sha(split_identity)
    metadata: dict[str, Any] = {
        "schema": CAPSULE_METADATA_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "validator_receipt_sha256": validator_receipt_sha256,
        "received_iq_sha256": received_iq_sha256,
        "strict_tap_sha256": strict_tap_sha256,
        "qknn_locks_sha256": qknn_locks_sha256,
        "class_registry": list(FIXED_CLASSES),
        "held_receivers": list(HELD_RECEIVERS),
        "rows": rows,
        "seed": RUNTIME_RECEIPT_SEED,
        "qknn_lock_by_k": {key: dict(value) for key, value in qknn_lock_by_k.items()},
        "qknn_locks": dict(qknn_locks_source),
        "qknn_lock_values_unchanged": True,
        "physical_id_sort_policy": PHYSICAL_ID_SORT_POLICY,
        "capsule_identity": capsule_identity,
        "split_identity": split_identity,
        "matrix_sha256": plan["matrix_sha256"],
        "row_count": len(rows),
        "physical_row_count": EXPECTED_ROWS,
        "physical_id_count": len(set(strict["physical_ids"])),
        "observation_id_count": len(set(strict["observation_ids"])),
        "receiver_count": len(set(strict["receiver_ids"])),
        "class_count": len(FIXED_CLASSES),
        "per_receiver_class_count": EXPECTED_PER_RECEIVER_CLASS,
        "phase1_fit_count_per_row": EXPECTED_PHASE1_FIT_COUNT,
        "support_k1_count_per_class": K1_SUPPORT_COUNT,
        "support_k5_count_per_class": K5_SUPPORT_COUNT,
        "query_count_per_class": QUERY_PER_CLASS,
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "truth_free": True,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "true_batch_class_count_access": False,
        "global_reassignment": False,
        "output_overwrite_allowed": False,
    }
    # Canonical hash is calculated by the caller after writing; identity
    # hashes above deliberately do not include file paths or output location.
    return metadata


def prepare_next_r4_capsule_metadata(
    *,
    strict_tap: Path,
    received_iq: Path,
    qknn_locks: Path,
    validator_receipt_sha256: str,
    strict_tap_sha256: str | None = None,
    received_iq_sha256: str | None = None,
) -> dict[str, Any]:
    """Build metadata in memory without creating an output file."""

    validator = _require_sha(validator_receipt_sha256, name="validator receipt SHA256")
    strict, strict_sha = _validate_strict_tap(
        strict_tap, expected_sha256=strict_tap_sha256
    )
    received, received_sha = _validate_received_iq(
        received_iq, strict=strict, expected_sha256=received_iq_sha256
    )
    locks_by_k, locks_source, locks_sha = _load_qknn_locks(qknn_locks)
    return _build_metadata(
        strict=strict,
        strict_tap_sha256=strict_sha,
        received=received,
        received_iq_sha256=received_sha,
        qknn_lock_by_k=locks_by_k,
        qknn_locks_source=locks_source,
        qknn_locks_sha256=locks_sha,
        validator_receipt_sha256=validator,
    )


def _new_output(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if (
        not path.is_absolute()
        or path != resolved
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise NextR4CapsuleMetadataError(
            "output metadata path must be a new absolute child of an existing directory"
        )
    return path


def build_next_r4_capsule_metadata(
    *,
    strict_tap: Path,
    received_iq: Path,
    qknn_locks: Path,
    validator_receipt_sha256: str,
    output_path: Path | None = None,
    strict_tap_sha256: str | None = None,
    received_iq_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate inputs and optionally write one canonical, non-overwriting JSON."""

    metadata = prepare_next_r4_capsule_metadata(
        strict_tap=strict_tap,
        received_iq=received_iq,
        qknn_locks=qknn_locks,
        validator_receipt_sha256=validator_receipt_sha256,
        strict_tap_sha256=strict_tap_sha256,
        received_iq_sha256=received_iq_sha256,
    )
    raw = _canonical_bytes(metadata)
    result: dict[str, Any] = {
        "status": BUILD_STATUS,
        "capsule_id": metadata["capsule_id"],
        "split_id": metadata["split_id"],
        "received_iq_sha256": metadata["received_iq_sha256"],
        "strict_tap_sha256": metadata["strict_tap_sha256"],
        "metadata_sha256": _sha_bytes(raw),
        "physical_id_count": metadata["physical_id_count"],
        "receiver_count": metadata["receiver_count"],
        "class_count": metadata["class_count"],
        "row_count": metadata["row_count"],
        "phase1_fit_count_per_row": metadata["phase1_fit_count_per_row"],
        "output_path": None,
        "metadata": metadata,
    }
    if output_path is not None:
        destination = _new_output(output_path)
        with destination.open("xb") as handle:
            handle.write(raw)
        result["output_path"] = str(destination)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build NEXT-R4 capsule/split metadata from sealed D106 inputs"
    )
    parser.add_argument("--strict-tap", required=True, type=Path)
    parser.add_argument("--received-iq", required=True, type=Path)
    parser.add_argument("--qknn-locks", required=True, type=Path)
    parser.add_argument("--validator-receipt-sha256", required=True)
    parser.add_argument("--output", "--output-path", dest="output_path", required=True, type=Path)
    parser.add_argument("--strict-tap-sha256")
    parser.add_argument("--received-iq-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_next_r4_capsule_metadata(
        strict_tap=args.strict_tap,
        received_iq=args.received_iq,
        qknn_locks=args.qknn_locks,
        validator_receipt_sha256=args.validator_receipt_sha256,
        output_path=args.output_path,
        strict_tap_sha256=args.strict_tap_sha256,
        received_iq_sha256=args.received_iq_sha256,
    )
    summary = {key: value for key, value in result.items() if key != "metadata"}
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())


__all__ = [
    "BUILD_STATUS",
    "CAPSULE_IDENTITY_SCHEMA",
    "CAPSULE_METADATA_SCHEMA",
    "EXPECTED_PHASE1_FIT_COUNT",
    "FIXED_CLASSES",
    "HELD_RECEIVERS",
    "NextR4CapsuleMetadataError",
    "PHYSICAL_ID_SORT_POLICY",
    "PHYSICAL_ID_SORT_SALT",
    "build_next_r4_capsule_metadata",
    "main",
    "prepare_next_r4_capsule_metadata",
]
