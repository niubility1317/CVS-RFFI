#!/usr/bin/env python3
"""Run the smallest D106 G0 mechanical check from one pinned tap archive.

This is deliberately independent of the failed production manifest/receipt
closure.  It reads one immutable, externally SHA256-pinned archive, rejects
performance-oriented fields, derives ``z_id`` from ``pre_relu``, and directly
executes the frozen leave-cell-out G0 fold mechanics for K=1/5/10.  Its JSON
result is non-formal functional evidence only; it contains no labels,
predictions, or performance metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d106_rcmr_g0 as g0  # noqa: E402
from cvsrffi.stage2_d106_phase1_tap import (  # noqa: E402
    D106Phase1TapRows,
    PROTOCOL_SCHEMA,
    TAP_RECEIPT_SCHEMA,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock  # noqa: E402


ONE_SHOT_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.one_shot.v2"
ONE_SHOT_STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
ARCHIVE_BYTE_CAP = 64 * 1024 * 1024
TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
_FORBIDDEN_FIELD_TOKENS = ("truth", "accuracy", "floor")
_FORBIDDEN_FIELD_NAMES = {"acc", "h"}
_D105_LODO_LOCK_SHA256 = (
    "7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08"
)
_RCMR_METHOD_LOCK_SHA256 = (
    "be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c"
)


class OneShotG0Error(ValueError):
    """Raised when the constrained one-shot input or output drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OneShotG0Error(f"{name} must be a lowercase SHA256")
    return value


def _require_absolute_file(path: Path, *, name: str) -> Path:
    if not path.is_absolute():
        raise OneShotG0Error(f"{name} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise OneShotG0Error(f"cannot stat {name}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise OneShotG0Error(f"{name} must be a regular non-symlink file")
    return path


def _read_pinned_archive(path: Path, *, expected_sha256: str) -> bytes:
    archive = _require_absolute_file(path, name="archive")
    expected = _require_sha256(expected_sha256, name="archive SHA256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(archive, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise OneShotG0Error("archive must be a non-empty regular file")
        if before.st_size > ARCHIVE_BYTE_CAP:
            raise OneShotG0Error("archive exceeds the fixed byte cap")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OneShotG0Error("archive changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise OneShotG0Error("archive identity changed during read")
    observed = _sha256_bytes(payload)
    if observed != expected:
        raise OneShotG0Error("archive SHA256 does not match the fixed input")
    return payload


def _reject_forbidden_field_names(names: Sequence[str]) -> None:
    for name in names:
        folded = name.lower()
        if (
            folded in _FORBIDDEN_FIELD_NAMES
            or any(token in folded for token in _FORBIDDEN_FIELD_TOKENS)
        ):
            raise OneShotG0Error("archive contains a forbidden performance field")


def _freeze_array(value: np.ndarray, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise OneShotG0Error(f"archive {name} dtype/shape drift")
    if dtype.kind == "f" and not np.isfinite(array).all():
        raise OneShotG0Error(f"archive {name} contains non-finite values")
    frozen = np.ascontiguousarray(array.copy())
    frozen.setflags(write=False)
    return frozen


def _freeze_text_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"} or array.shape != (g0.EXPECTED_ROWS,):
        raise OneShotG0Error(f"archive {name} dtype/shape drift")
    text = array.astype(str)
    if any(not item or not item.strip() for item in text.tolist()):
        raise OneShotG0Error(f"archive {name} contains blank identifiers")
    frozen = np.ascontiguousarray(array.copy())
    frozen.setflags(write=False)
    return frozen


def _load_rows(archive_bytes: bytes, *, archive_sha256: str) -> D106Phase1TapRows:
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as loaded:
            names = tuple(loaded.files)
            _reject_forbidden_field_names(names)
            if names != TAP_MEMBERS:
                raise OneShotG0Error("archive member set/order drift")
            arrays = {name: np.asarray(loaded[name]) for name in TAP_MEMBERS}
    except OneShotG0Error:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise OneShotG0Error("archive is not a valid no-pickle tap NPZ") from exc

    pre_relu = _freeze_array(
        arrays["pre_relu"],
        dtype=np.dtype(np.float32),
        shape=(g0.EXPECTED_ROWS, g0.Z_DIM),
        name="pre_relu",
    )
    z_dom = _freeze_array(
        arrays["z_dom"],
        dtype=np.dtype(np.float32),
        shape=(g0.EXPECTED_ROWS, g0.Z_DIM),
        name="z_dom",
    )
    text = {
        name: _freeze_text_array(arrays[name], name=name)
        for name in TAP_MEMBERS[2:]
    }
    if len(set(text["physical_ids"].astype(str).tolist())) != g0.EXPECTED_ROWS:
        raise OneShotG0Error("archive physical IDs must be unique")
    if len(set(text["observation_ids"].astype(str).tolist())) != g0.EXPECTED_ROWS:
        raise OneShotG0Error("archive observation IDs must be unique")
    z_id = np.ascontiguousarray(np.maximum(pre_relu, np.float32(0.0)))
    z_id.setflags(write=False)
    receipt = MappingProxyType(
        {
            "schema": TAP_RECEIPT_SCHEMA,
            "protocol_schema": PROTOCOL_SCHEMA,
            "row_count": g0.EXPECTED_ROWS,
            "exact_inner_join": True,
            "same_received_iq_for_zid_zdom": True,
            "z_id_storage_policy": "derive_relu_pre_relu",
            "feature_stage_source_pool_access": False,
            "clean_iq_access": False,
            "target_access": False,
            "formal_query_access": False,
            "one_shot_archive_sha256": archive_sha256,
            "one_shot_member_root_sha256": _sha256_bytes(
                _canonical_bytes(list(TAP_MEMBERS))
            ),
        }
    )
    return D106Phase1TapRows(
        pre_relu=pre_relu,
        z_dom=z_dom,
        tx_labels=text["tx_labels"],
        receiver_ids=text["receiver_ids"],
        day_ids=text["day_ids"],
        physical_ids=text["physical_ids"],
        scenario_names=text["scenario_names"],
        observation_ids=text["observation_ids"],
        z_id=z_id,
        receipt=receipt,
    )


def _predecessor_locks(rows: D106Phase1TapRows) -> tuple[Phase1ZIDStudentTLock, ...]:
    tap_receipt_sha256 = _sha256_bytes(g0._canonical_bytes(rows.receipt))
    values = dict(g0.PREDECESSOR_NUMERIC_LOCK)
    return tuple(
        Phase1ZIDStudentTLock(
            active_k=active_k,
            phase1_lodo_receipt_sha256=_D105_LODO_LOCK_SHA256,
            quantization_margin_audit_sha256=tap_receipt_sha256,
            **values,
        )
        for active_k in g0.K_VALUES
    )


def _require_registered_classes(value: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(value)
    if len(classes) != g0.EXPECTED_CLASSES or len(set(classes)) != len(classes):
        raise OneShotG0Error("exactly six distinct externally supplied classes are required")
    if any(type(item) is not str or not item for item in classes):
        raise OneShotG0Error("registered classes must be non-empty strings")
    return classes


def _assert_no_performance_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        _reject_forbidden_field_names(tuple(str(key) for key in value))
        for member in value.values():
            _assert_no_performance_fields(member)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _assert_no_performance_fields(member)


def _validate_one_shot_locks(
    locks: Sequence[Phase1ZIDStudentTLock], *, tap_receipt_sha256: str
) -> tuple[Phase1ZIDStudentTLock, ...]:
    result = tuple(locks)
    if len(result) != len(g0.K_VALUES):
        raise OneShotG0Error("one-shot requires exactly K1/K5/K10 locks")
    for active_k, lock in zip(g0.K_VALUES, result, strict=True):
        if type(lock) is not Phase1ZIDStudentTLock or lock.active_k != active_k:
            raise OneShotG0Error("one-shot predecessor lock/K binding drift")
        for name, expected in g0.PREDECESSOR_NUMERIC_LOCK.items():
            if getattr(lock, name) != expected:
                raise OneShotG0Error("one-shot predecessor numeric lock drift")
        if (
            lock.phase1_lodo_receipt_sha256 != _D105_LODO_LOCK_SHA256
            or lock.quantization_margin_audit_sha256 != tap_receipt_sha256
        ):
            raise OneShotG0Error("one-shot predecessor lock receipt binding drift")
    if tuple(lock.active_k for lock in result) != g0.K_VALUES or len(
        {lock.lock_digest for lock in result}
    ) != len(result):
        raise OneShotG0Error("one-shot predecessor lock set drift")
    return result


def _fold_inputs(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    registry: tuple[str, ...],
) -> dict[str, Any]:
    """Rebuild only the frozen fold's support/query arrays; labels stay support-only."""

    rows = g0._revalidate_snapshot(snapshot)
    physical = g0._typed_tokens(rows.physical_ids, "physical IDs", count=g0.EXPECTED_ROWS)
    receiver_ids = g0._typed_tokens(
        rows.receiver_ids, "receiver IDs", count=g0.EXPECTED_ROWS
    )
    day_ids = g0._typed_tokens(rows.day_ids, "day IDs", count=g0.EXPECTED_ROWS)
    index_by_id = {value: index for index, value in enumerate(physical)}
    query_indices = np.asarray(
        [index_by_id[value] for value in fold.query_ids], dtype=np.int64
    )
    query_mask = np.zeros(g0.EXPECTED_ROWS, dtype=bool)
    query_mask[query_indices] = True
    if len(query_indices) > g0.MAX_QUERY_ROWS_PER_FOLD or np.any(
        (np.asarray(receiver_ids) == fold.receiver_id)
        & (np.asarray(day_ids) == fold.day_id)
        & ~query_mask
    ):
        raise OneShotG0Error("one-shot fold query closure drift")
    support_pool = np.flatnonzero(~query_mask).astype(np.int64)
    if np.any(
        (np.asarray(receiver_ids)[support_pool] == fold.receiver_id)
        & (np.asarray(day_ids)[support_pool] == fold.day_id)
    ):
        raise OneShotG0Error("one-shot held cell reached support")
    pool_labels = g0._typed_tokens(
        rows.tx_labels[support_pool], "support-pool labels", count=len(support_pool)
    )
    by_class: dict[str, list[int]] = {class_id: [] for class_id in registry}
    for index, label in zip(support_pool.tolist(), pool_labels, strict=True):
        if label not in by_class:
            raise OneShotG0Error("one-shot support label outside registry")
        by_class[label].append(index)
    selected: list[int] = []
    for class_id in registry:
        ordered = sorted(
            by_class[class_id], key=lambda index: physical[index].encode("utf-8")
        )
        if len(ordered) < active_k:
            raise OneShotG0Error("one-shot fold lacks the frozen K support")
        selected.extend(ordered[:active_k])
    support_indices = np.asarray(selected, dtype=np.int64)
    support_ids = tuple(physical[index] for index in selected)
    support_labels = g0._typed_tokens(
        rows.tx_labels[support_indices], "support labels", count=len(selected)
    )
    return {
        "support_ids": support_ids,
        "support_labels": support_labels,
        "support_plus": np.ascontiguousarray(rows.z_id[support_indices], dtype=np.float32),
        "support_signed": np.ascontiguousarray(
            rows.pre_relu[support_indices], dtype=np.float32
        ),
        "query_plus": np.ascontiguousarray(rows.z_id[query_indices], dtype=np.float32),
        "query_signed": np.ascontiguousarray(
            rows.pre_relu[query_indices], dtype=np.float32
        ),
    }


def _baseline_trace(
    inputs: Mapping[str, Any],
    *,
    registry: tuple[str, ...],
    predecessor_lock: Phase1ZIDStudentTLock,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Return the frozen baseline logits and its actual per-support kernel terms."""

    qknn = g0._qknn_module
    support_plus = np.asarray(inputs["support_plus"], dtype=np.float32)
    support_labels = tuple(inputs["support_labels"])
    support_ids = tuple(inputs["support_ids"])
    query_plus = np.asarray(inputs["query_plus"], dtype=np.float32)
    metric = g0.identity_shared_psd_metric(config=predecessor_lock)
    bank = g0.build_typed_zid_support_bank(
        support_plus, support_labels, registry, config=predecessor_lock
    )
    logits = g0.score_zid_student_t_logits(bank, query_plus, metric=metric)
    normalized_support = qknn.normalize_zid_rows(support_plus)
    class_map = {label: index for index, label in enumerate(registry)}
    original_class_indices = np.asarray(
        [class_map[label] for label in support_labels], dtype=np.int16
    )
    codes, scales, _decoded = qknn._quantize_rows(normalized_support)
    order = qknn._canonical_order(codes, scales, original_class_indices)
    if not (
        np.array_equal(bank.codes_qint8, codes[order])
        and np.array_equal(bank.scales_fp16, scales[order])
        and np.array_equal(bank.class_indices_int16, original_class_indices[order])
    ):
        raise OneShotG0Error("one-shot baseline bank order drift")
    support_ids_in_bank_order = tuple(support_ids[int(index)] for index in order)
    query_features = qknn.normalize_zid_rows(query_plus).astype(np.float64)
    decoded_support = qknn.decode_zid_support_bank(bank).astype(np.float64)
    cosine = qknn._precision_cosine(query_features, decoded_support, metric)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    kernels = np.empty_like(distance, dtype=np.float64)
    for class_index, expected in enumerate(bank.support_counts):
        mask = bank.class_indices_int16 == class_index
        if int(np.count_nonzero(mask)) != expected:
            raise OneShotG0Error("one-shot baseline support class count drift")
        h = float(bank.class_scales_fp16[class_index])
        kernels[:, mask] = (
            -predecessor_lock.kernel_volume_gamma
            * predecessor_lock.kernel_effective_dim
            * math.log(h)
            - 0.5
            * (predecessor_lock.student_nu + predecessor_lock.kernel_effective_dim)
            * np.log1p(distance[:, mask] / (predecessor_lock.student_nu * h * h))
        )
    rechecked_logits = qknn._score_with_support(
        support=decoded_support,
        class_indices=bank.class_indices_int16,
        support_counts=bank.support_counts,
        class_scales=bank.class_scales_fp16,
        query=query_features,
        config=predecessor_lock,
        metric=metric,
    )
    if not np.array_equal(logits, rechecked_logits):
        raise OneShotG0Error("one-shot baseline score path drift")
    return logits, kernels, support_ids_in_bank_order, query_features


def _candidate_trace(
    inputs: Mapping[str, Any],
    *,
    registry: tuple[str, ...],
    active_k: int,
    rcmr_method_lock_sha256: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Return frozen RCMR class scores and its actual per-support evidence terms."""

    support_ids = tuple(inputs["support_ids"])
    support_labels = tuple(inputs["support_labels"])
    support_plus = np.asarray(inputs["support_plus"], dtype=np.float32)
    support_signed = np.asarray(inputs["support_signed"], dtype=np.float32)
    support_root = g0._support_root(support_ids)
    paired_receipt = g0._paired_view_receipt(
        support_ids, support_plus, support_signed
    )
    built_state = g0._nonformal_state_from_support(
        support_plus,
        support_signed,
        support_labels,
        support_ids,
        registry,
        active_k=active_k,
        support_root_sha256=support_root,
        paired_view_receipt_sha256=paired_receipt,
        rcmr_method_lock_sha256=rcmr_method_lock_sha256,
    )
    wire = g0._serialize_nonformal_state(built_state)
    state = g0._deserialize_nonformal_state(
        wire, expected_sha256=g0._sha256(wire)
    )
    if state.state_receipt_sha256 != built_state.state_receipt_sha256:
        raise OneShotG0Error("one-shot candidate state wire drift")
    context = g0._prepare_nonformal_context(state)
    order = tuple(sorted(range(len(support_ids)), key=lambda index: support_ids[index]))
    support_ids_in_state_order = tuple(support_ids[index] for index in order)
    class_map = {label: index for index, label in enumerate(registry)}
    expected_indices = np.asarray(
        [class_map[support_labels[index]] for index in order], dtype=np.uint8
    )
    if not np.array_equal(state.class_indices, expected_indices):
        raise OneShotG0Error("one-shot candidate state/support binding drift")

    candidate_scores: list[np.ndarray] = []
    evidence_rows: list[np.ndarray] = []
    plus_rows: list[np.ndarray] = []
    signed_rows: list[np.ndarray] = []
    for query_plus, query_signed in zip(
        np.asarray(inputs["query_plus"]), np.asarray(inputs["query_signed"]), strict=True
    ):
        plus_feature = g0._rcmr_module._finite_l2_normalized_vector(
            query_plus, "one-shot query_plus"
        )
        signed_feature = g0._rcmr_module._finite_l2_normalized_vector(
            query_signed, "one-shot query_signed"
        )
        count = len(state.class_indices)
        distances_plus = np.asarray(
            [
                g0._rcmr_module._dot_distance(plus_feature, context.decoded_plus[slot])
                for slot in range(count)
            ],
            dtype=np.float64,
        )
        distances_signed = np.asarray(
            [
                g0._rcmr_module._dot_distance(
                    signed_feature, context.decoded_signed[slot]
                )
                for slot in range(count)
            ],
            dtype=np.float64,
        )
        alpha_plus = g0._rcmr_module._midranks(distances_plus)
        alpha_signed = g0._rcmr_module._midranks(distances_signed)
        beta_plus = np.asarray(
            [
                g0._rcmr_module._midrank_from_profile(
                    context.profiles_plus[slot], float(distances_plus[slot])
                )
                for slot in range(count)
            ],
            dtype=np.float64,
        )
        beta_signed = np.asarray(
            [
                g0._rcmr_module._midrank_from_profile(
                    context.profiles_signed[slot], float(distances_signed[slot])
                )
                for slot in range(count)
            ],
            dtype=np.float64,
        )
        query_reliability = math.exp(
            -float(np.mean(np.abs(alpha_plus - alpha_signed)))
        )
        weights = query_reliability * state.reliabilities.astype(np.float64, copy=False)
        evidence = (
            (1.0 - alpha_plus) * (1.0 - beta_plus)
            + weights * (1.0 - alpha_signed) * (1.0 - beta_signed)
        ) / (1.0 + weights)
        scores = np.zeros(len(registry), dtype=np.float64)
        for slot in range(count):
            scores[int(state.class_indices[slot])] += float(evidence[slot])
        scores /= float(state.active_k)
        if not (
            np.isfinite(evidence).all()
            and np.isfinite(scores).all()
            and 0.0 < query_reliability <= 1.0
        ):
            raise OneShotG0Error("one-shot candidate numeric path drift")
        candidate_scores.append(scores)
        evidence_rows.append(evidence)
        plus_rows.append(plus_feature)
        signed_rows.append(signed_feature)
    return (
        np.ascontiguousarray(np.stack(candidate_scores), dtype=np.float64),
        np.ascontiguousarray(np.stack(evidence_rows), dtype=np.float64),
        support_ids_in_state_order,
        np.ascontiguousarray(np.stack(plus_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(signed_rows), dtype=np.float64),
    )


def _argmax_labels(scores: np.ndarray, registry: tuple[str, ...]) -> tuple[str, ...]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(registry) or not np.isfinite(values).all():
        raise OneShotG0Error("one-shot score layout drift")
    labels: list[str] = []
    for row in values:
        maximum = max(float(value) for value in row)
        winners = [
            index
            for index, value in enumerate(row)
            if g0._rcmr_module._same_binary64(float(value), maximum)
        ]
        if len(winners) != 1:
            raise OneShotG0Error("one-shot cross-class score tie")
        labels.append(registry[winners[0]])
    return tuple(labels)


def _top_two_margins(scores: np.ndarray) -> np.ndarray:
    margins: list[float] = []
    for row in np.asarray(scores, dtype=np.float64):
        maximum = max(float(value) for value in row)
        winners = [
            index
            for index, value in enumerate(row)
            if g0._rcmr_module._same_binary64(float(value), maximum)
        ]
        if len(winners) != 1:
            raise OneShotG0Error("one-shot margin cross-class tie")
        runner_up = max(float(value) for index, value in enumerate(row) if index != winners[0])
        margins.append(maximum - runner_up)
    return np.ascontiguousarray(np.asarray(margins, dtype=np.float64))


def _dominant_support_signatures(
    contributions: np.ndarray, support_ids: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    signatures: list[tuple[str, ...]] = []
    for row in np.asarray(contributions, dtype=np.float64):
        maximum = max(float(value) for value in row)
        tied = tuple(
            sorted(
                (
                    support_ids[index]
                    for index, value in enumerate(row)
                    if g0._rcmr_module._same_binary64(float(value), maximum)
                ),
                key=lambda value: value.encode("utf-8"),
            )
        )
        if not tied:
            raise OneShotG0Error("one-shot dominant support is empty")
        signatures.append(tied)
    return tuple(signatures)


def _bitmap_root(
    *, metric: str, query_root_sha256: str, bits: str
) -> str:
    return g0._sha256(
        {
            "schema": ONE_SHOT_SCHEMA + ".mechanical_bitmap.v1",
            "metric": metric,
            "query_root_sha256": query_root_sha256,
            "encoding": "ascii01_query_order",
            "bits": bits,
        }
    )


def _changed_metric(
    *, metric: str, query_root_sha256: str, baseline: Sequence[Any], candidate: Sequence[Any]
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise OneShotG0Error("one-shot metric query length drift")
    bits = "".join(
        "1" if candidate_value != baseline_value else "0"
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    )
    return {
        "changed_count": bits.count("1"),
        "changed_bitmap_root_sha256": _bitmap_root(
            metric=metric, query_root_sha256=query_root_sha256, bits=bits
        ),
    }


def _array_changed_metric(
    *, metric: str, query_root_sha256: str, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    left = np.asarray(baseline)
    right = np.asarray(candidate)
    if left.shape != right.shape or left.ndim < 1:
        raise OneShotG0Error("one-shot array metric shape drift")
    bits = "".join(
        "1" if not np.array_equal(left[index], right[index]) else "0"
        for index in range(len(left))
    )
    return {
        "changed_count": bits.count("1"),
        "changed_bitmap_root_sha256": _bitmap_root(
            metric=metric, query_root_sha256=query_root_sha256, bits=bits
        ),
    }


def _fold_audit(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    predecessor_lock: Phase1ZIDStudentTLock,
    registry: tuple[str, ...],
    common_query_order_root_sha256: str,
) -> dict[str, Any]:
    """Run the frozen fold core, then verify non-performance mechanics in-place."""

    query_ids, core_candidate, core_baseline, core_receipt = g0._execute_fold(
        snapshot,
        fold,
        active_k=active_k,
        predecessor_lock=predecessor_lock,
        rcmr_method_lock_sha256=_RCMR_METHOD_LOCK_SHA256,
        registry=registry,
        common_query_order_root_sha256=common_query_order_root_sha256,
    )
    inputs = _fold_inputs(
        snapshot, fold, active_k=active_k, registry=registry
    )
    baseline_logits, baseline_kernels, baseline_support_ids, baseline_features = _baseline_trace(
        inputs, registry=registry, predecessor_lock=predecessor_lock
    )
    (
        candidate_scores,
        candidate_evidence,
        candidate_support_ids,
        _candidate_plus_features,
        candidate_signed_features,
    ) = _candidate_trace(
        inputs,
        registry=registry,
        active_k=active_k,
        rcmr_method_lock_sha256=_RCMR_METHOD_LOCK_SHA256,
    )
    audit_baseline = g0._unique_argmax(baseline_logits, registry)
    audit_candidate = _argmax_labels(candidate_scores, registry)
    if audit_baseline != core_baseline or audit_candidate != core_candidate:
        raise OneShotG0Error("one-shot audit/frozen fold argmax drift")
    query_root = fold.query_root_sha256
    baseline_neighbors = _dominant_support_signatures(
        baseline_kernels, baseline_support_ids
    )
    candidate_neighbors = _dominant_support_signatures(
        candidate_evidence, candidate_support_ids
    )
    baseline_margin = _top_two_margins(baseline_logits.astype(np.float64))
    candidate_margin = _top_two_margins(candidate_scores)
    feature = _array_changed_metric(
        metric="signed_view_feature_vs_zid_feature",
        query_root_sha256=query_root,
        baseline=baseline_features,
        candidate=candidate_signed_features,
    )
    neighbor = _changed_metric(
        metric="dominant_support_contribution",
        query_root_sha256=query_root,
        baseline=baseline_neighbors,
        candidate=candidate_neighbors,
    )
    margin = _array_changed_metric(
        metric="top1_minus_top2_margin_binary64",
        query_root_sha256=query_root,
        baseline=baseline_margin.view(np.uint64),
        candidate=candidate_margin.view(np.uint64),
    )
    argmax = _changed_metric(
        metric="argmax_class",
        query_root_sha256=query_root,
        baseline=core_baseline,
        candidate=core_candidate,
    )
    if argmax["changed_count"] != core_receipt["argmax_changed_count"]:
        raise OneShotG0Error("one-shot frozen fold argmax receipt drift")
    payload = {
        "query_root_sha256": query_root,
        "fold_execution_receipt_sha256": core_receipt["execution_receipt_sha256"],
        "baseline_feature_root_sha256": g0._array_root(baseline_features),
        "candidate_feature_root_sha256": g0._array_root(candidate_signed_features),
        "feature_changed_count": feature["changed_count"],
        "feature_changed_bitmap_root_sha256": feature["changed_bitmap_root_sha256"],
        "baseline_neighbor_root_sha256": g0._sha256(
            [list(item) for item in baseline_neighbors]
        ),
        "candidate_neighbor_root_sha256": g0._sha256(
            [list(item) for item in candidate_neighbors]
        ),
        "neighbor_changed_count": neighbor["changed_count"],
        "neighbor_changed_bitmap_root_sha256": neighbor["changed_bitmap_root_sha256"],
        "baseline_margin_root_sha256": g0._array_root(baseline_margin),
        "candidate_margin_root_sha256": g0._array_root(candidate_margin),
        "margin_changed_count": margin["changed_count"],
        "margin_changed_bitmap_root_sha256": margin["changed_bitmap_root_sha256"],
        "baseline_argmax_root_sha256": g0._sha256(list(core_baseline)),
        "candidate_argmax_root_sha256": g0._sha256(list(core_candidate)),
        "argmax_changed_count": argmax["changed_count"],
        "argmax_changed_bitmap_root_sha256": core_receipt[
            "argmax_changed_bitmap_root_sha256"
        ],
    }
    payload["fold_mechanical_audit_root_sha256"] = g0._sha256(payload)
    return payload


def _aggregate_per_k(
    *, active_k: int, fold_audits: Sequence[Mapping[str, Any]], query_root_sha256: str
) -> dict[str, Any]:
    if len(fold_audits) != g0.EXPECTED_FOLDS:
        raise OneShotG0Error("one-shot fold audit count drift")
    result: dict[str, Any] = {
        "K": active_k,
        "query_count": g0.EXPECTED_ROWS,
        "query_ids_root_sha256": query_root_sha256,
        "fold_count": g0.EXPECTED_FOLDS,
        "fold_execution_receipts_root_sha256": g0._sha256(
            [item["fold_execution_receipt_sha256"] for item in fold_audits]
        ),
        "fold_mechanical_audits_root_sha256": g0._sha256(
            [item["fold_mechanical_audit_root_sha256"] for item in fold_audits]
        ),
    }
    for metric in ("feature", "neighbor", "margin", "argmax"):
        result[f"{metric}_changed_count"] = sum(
            int(item[f"{metric}_changed_count"]) for item in fold_audits
        )
        for side in ("baseline", "candidate"):
            result[f"{side}_{metric}_root_sha256"] = g0._sha256(
                [item[f"{side}_{metric}_root_sha256"] for item in fold_audits]
            )
        result[f"{metric}_changed_bitmap_roots_root_sha256"] = g0._sha256(
            [item[f"{metric}_changed_bitmap_root_sha256"] for item in fold_audits]
        )
    result["per_k_execution_root_sha256"] = g0._sha256(result)
    return result


def _resource_summary(snapshot: Any) -> dict[str, Any]:
    resources = g0._resource_analysis(snapshot)
    summary = {
        name: resources[name]
        for name in (
            "analysis_numeric_array_budget_bytes",
            "incremental_numeric_array_peak_analysis_estimate_bytes",
            "parameter_scan_count",
            "query_state_updates",
            "analysis_budget_is_process_rss_cap",
            "process_rss_measured",
        )
    }
    summary["resource_summary_root_sha256"] = g0._sha256(summary)
    return summary


def _real_archive_execution(
    rows: D106Phase1TapRows, *, registered_classes: Sequence[str]
) -> dict[str, Any]:
    registry = g0._canonical_registry(_require_registered_classes(registered_classes))
    tap_receipt_sha256 = _sha256_bytes(g0._canonical_bytes(rows.receipt))
    snapshot = g0._snapshot_from_rows(
        rows, tap_receipt_sha256=tap_receipt_sha256
    )
    locks = _validate_one_shot_locks(
        _predecessor_locks(rows), tap_receipt_sha256=snapshot.tap_receipt_sha256
    )
    plan = g0._build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    if len(query_order) != g0.EXPECTED_ROWS or len(set(query_order)) != g0.EXPECTED_ROWS:
        raise OneShotG0Error("one-shot common query closure drift")
    query_root = g0._sha256(list(query_order))
    per_k: list[dict[str, Any]] = []
    for active_k, predecessor_lock in zip(g0.K_VALUES, locks, strict=True):
        audits = [
            _fold_audit(
                snapshot,
                fold,
                active_k=active_k,
                predecessor_lock=predecessor_lock,
                registry=registry,
                common_query_order_root_sha256=query_root,
            )
            for fold in plan
        ]
        per_k.append(
            _aggregate_per_k(
                active_k=active_k,
                fold_audits=audits,
                query_root_sha256=query_root,
            )
        )
    argmax_changed = {str(item["K"]): int(item["argmax_changed_count"]) for item in per_k}
    zero_changed = [
        active_k for active_k in g0.K_VALUES if argmax_changed[str(active_k)] == 0
    ]
    functional_gate_pass = not zero_changed
    execution = {
        "K_values": list(g0.K_VALUES),
        "fold_count": g0.EXPECTED_FOLDS,
        "query_count_per_k": g0.EXPECTED_ROWS,
        "common_query_order_root_sha256": query_root,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "argmax_changed_count_by_k": argmax_changed,
        "argmax_changed_count": sum(argmax_changed.values()),
        "zero_changed_k_values": zero_changed,
        "functional_gate_status": (
            "G0_PASS_PROCEED_G1"
            if functional_gate_pass
            else "REJECT_REVISION_NO_FUNCTION"
        ),
        "functional_gate_pass": functional_gate_pass,
        "per_k": per_k,
        "resource_summary": _resource_summary(snapshot),
    }
    execution["core_execution_root_sha256"] = g0._sha256(execution)
    return execution


def _write_new_output(path: Path, payload: Mapping[str, Any]) -> bytes:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise OneShotG0Error("output must be a new file in an existing absolute directory")
    document = dict(payload)
    document["output_receipt_sha256"] = _sha256_bytes(_canonical_bytes(document))
    encoded = _canonical_bytes(document)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        total = 0
        while total < len(encoded):
            total += os.write(descriptor, encoded[total:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return encoded


def run_one_shot(
    *,
    archive_path: Path,
    expected_archive_sha256: str,
    registered_classes: Sequence[str],
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Execute the constrained data path; the command-line entry is POSIX-only."""

    if type(run_id) is not str or not run_id or len(run_id.encode("utf-8")) > 160:
        raise OneShotG0Error("run ID must be a short non-empty string")
    archive_bytes = _read_pinned_archive(
        archive_path, expected_sha256=expected_archive_sha256
    )
    archive_sha256 = _sha256_bytes(archive_bytes)
    rows = _load_rows(archive_bytes, archive_sha256=archive_sha256)
    execution = _real_archive_execution(rows, registered_classes=registered_classes)
    result = {
        "schema": ONE_SHOT_SCHEMA,
        "status": ONE_SHOT_STATUS,
        "run_id": run_id,
        "archive_sha256": archive_sha256,
        "archive_member_names": list(TAP_MEMBERS),
        "row_count": g0.EXPECTED_ROWS,
        "real_archive_g0_executed": True,
        "g0_decision_consumption_allowed": True,
        "g1_entry_allowed": execution["functional_gate_pass"],
        "formal_performance_claim": False,
        "performance_metrics_emitted": False,
        "query_label_read_for_scoring": False,
        "metric_definitions": {
            "feature": "candidate signed-view normalized feature versus baseline z_id normalized feature",
            "neighbor": "dominant actual per-support contribution signature with full exact-tie set",
            "margin": "actual class-score top1-minus-top2 compared by binary64 bits",
            "argmax": "same-query candidate versus baseline class decision",
        },
        **execution,
    }
    _assert_no_performance_fields(result)
    _write_new_output(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--registered-class", required=True, action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.name != "posix":
        raise OneShotG0Error("the one-shot command-line entry is POSIX-only")
    result = run_one_shot(
        archive_path=args.archive,
        expected_archive_sha256=args.archive_sha256,
        registered_classes=args.registered_class,
        run_id=args.run_id,
        output_path=args.output,
    )
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
