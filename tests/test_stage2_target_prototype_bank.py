from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi.stage2_target_prototype_bank import (
    SCHEMA,
    TargetPrototypeBank,
    TargetPrototypeBankError,
    append_normalized_vectors,
    append_support_prototypes,
    encode_normalized_vectors,
    encode_support_prototypes,
    score_mixed_fp32,
)


FEATURE_DIM = 160


def _class(index: int) -> str:
    return "cls_" + hashlib.sha256(f"class-{index}".encode()).hexdigest()


def _direction(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _support(classes: tuple[str, ...], k_shot: int) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_handle in enumerate(classes):
        for rank in range(k_shot):
            row = _direction(class_index)
            if k_shot > 1:
                row = row + np.float32(0.01 * (rank + 1)) * _direction(
                    20 + class_index
                )
            rows.append(row)
            labels.append(class_handle)
    return np.stack(rows), labels


@pytest.mark.parametrize("storage_format", ("fp32", "fp16", "int8"))
def test_k1_uses_locked_r0_and_common_record(storage_format: str) -> None:
    classes = (_class(0), _class(1))
    features, labels = _support(classes, 1)
    bank = encode_support_prototypes(
        features,
        labels,
        classes,
        storage_format=storage_format,
        r0=0.27,
    )

    assert bank.schema == SCHEMA
    assert bank.storage_format == storage_format
    assert bank.classes == classes
    assert bank.old_class_count == len(classes)
    assert bank.radius.dtype == np.float16
    np.testing.assert_array_equal(
        bank.radius, np.full(len(classes), np.float16(0.27), dtype=np.float16)
    )
    assert bank.count.dtype == np.uint16
    np.testing.assert_array_equal(bank.count, 1)
    assert len(bank.old_prefix_sha256) == 64
    assert not bank.radius.flags.writeable
    assert not bank.count.flags.writeable
    if storage_format == "int8":
        assert bank.vectors is None
        assert bank.q is not None and bank.q.dtype == np.int8
        assert bank.scale is not None and bank.scale.dtype == np.float16
        assert not np.any(bank.q == -128)
    else:
        assert bank.q is None and bank.scale is None
        assert bank.vectors is not None
        expected = np.float32 if storage_format == "fp32" else np.float16
        assert bank.vectors.dtype == expected


def test_support_radius_and_count_use_same_semantics_for_all_formats() -> None:
    classes = (_class(0), _class(1), _class(2))
    features, labels = _support(classes, 5)
    banks = {
        storage: encode_support_prototypes(
            features,
            labels,
            classes,
            storage_format=storage,
            r0=0.20,
        )
        for storage in ("fp32", "fp16", "int8")
    }
    reference = banks["fp32"]
    for bank in banks.values():
        np.testing.assert_array_equal(bank.radius, reference.radius)
        np.testing.assert_array_equal(bank.count, 5)
        assert np.all(bank.radius > 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(bank.decoded_vectors(), axis=1), 1.0, atol=0.01
        )


def test_normalized_vector_codec_and_score_api_are_paired() -> None:
    rng = np.random.default_rng(713101)
    vectors = rng.normal(size=(4, FEATURE_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=FEATURE_DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    classes = tuple(_class(index) for index in range(4))
    radius = np.asarray([0.10, 0.11, 0.12, 0.13], dtype=np.float32)
    count = np.full(4, 10, dtype=np.uint16)

    scores: dict[str, np.ndarray] = {}
    for storage in ("fp32", "fp16", "int8"):
        bank = encode_normalized_vectors(
            vectors,
            classes,
            radius=radius,
            count=count,
            storage_format=storage,
        )
        scores[storage] = score_mixed_fp32(bank, query)
        np.testing.assert_allclose(
            scores[storage], query @ bank.decoded_vectors().T, atol=1.0e-6
        )
        assert scores[storage].shape == (4,)
        assert not scores[storage].flags.writeable
    np.testing.assert_allclose(scores["fp16"], scores["fp32"], atol=5.0e-4)
    np.testing.assert_allclose(scores["int8"], scores["fp32"], atol=0.01)


def test_storage_audit_compares_actual_and_equivalent_formats() -> None:
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(5, FEATURE_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    classes = tuple(_class(index) for index in range(5))
    banks = {
        storage: encode_normalized_vectors(
            vectors,
            classes,
            radius=np.full(5, 0.2),
            count=np.full(5, 10, dtype=np.uint16),
            storage_format=storage,
        )
        for storage in ("fp32", "fp16", "int8")
    }

    audits = {
        storage: bank.storage_audit(vectors)
        for storage, bank in banks.items()
    }
    expected_order = (
        audits["int8"]["state_bytes_by_format"]["int8"]
        < audits["fp16"]["state_bytes_by_format"]["fp16"]
        < audits["fp32"]["state_bytes_by_format"]["fp32"]
    )
    assert expected_order
    for storage, bank in banks.items():
        audit = audits[storage]
        assert audit["logical_state_bytes"] == bank.logical_state_bytes
        assert audit["actual_matches_format_bytes"] is True
        assert audit["quantization_error_available"] is True
        assert set(audit["error_by_format"]) == {"fp32", "fp16", "int8"}
        assert audit["error_by_format"]["fp32"]["max_abs_error"] == 0.0
        assert audit["mixed_fp32_prototype_macs_per_scored_row"] == 5 * FEATURE_DIM
    assert audits["int8"]["actual_error"]["max_abs_error"] < 0.002
    assert audits["fp16"]["actual_error"]["max_abs_error"] < 0.001


@pytest.mark.parametrize("storage_format", ("fp32", "fp16", "int8"))
def test_append_support_is_old_prefix_byte_frozen(storage_format: str) -> None:
    old_classes = (_class(0), _class(1))
    new_classes = (_class(2), _class(3))
    old_features, old_labels = _support(old_classes, 5)
    new_features, new_labels = _support(new_classes, 5)
    before = encode_support_prototypes(
        old_features,
        old_labels,
        old_classes,
        storage_format=storage_format,
        r0=0.20,
    )
    old_payload = (
        before.q.tobytes() + before.scale.tobytes()
        if storage_format == "int8"
        else before.vectors.tobytes()
    )
    old_metadata = before.radius.tobytes() + before.count.tobytes()

    after = append_support_prototypes(
        before,
        new_features,
        new_labels,
        new_classes,
        r0=0.20,
    )

    assert after.classes == old_classes + new_classes
    assert after.old_class_count == len(old_classes)
    assert after.old_prefix_sha256 == before.old_prefix_sha256
    if storage_format == "int8":
        assert after.q is not None and after.scale is not None
        prefix_payload = (
            after.q[: len(old_classes)].tobytes()
            + after.scale[: len(old_classes)].tobytes()
        )
    else:
        assert after.vectors is not None
        prefix_payload = after.vectors[: len(old_classes)].tobytes()
    prefix_metadata = (
        after.radius[: len(old_classes)].tobytes()
        + after.count[: len(old_classes)].tobytes()
    )
    assert prefix_payload == old_payload
    assert prefix_metadata == old_metadata


def test_append_normalized_vectors_uses_same_freeze_rule() -> None:
    old_classes = (_class(0), _class(1))
    before = encode_normalized_vectors(
        np.stack([_direction(0), _direction(1)]),
        old_classes,
        radius=[0.2, 0.2],
        count=[10, 10],
        storage_format="int8",
    )
    after = append_normalized_vectors(
        before,
        np.stack([_direction(2)]),
        (_class(2),),
        radius=[0.15],
        count=[10],
    )
    assert after.old_prefix_sha256 == before.old_prefix_sha256
    assert after.classes == old_classes + (_class(2),)


def test_append_rejects_overlap_or_k_shot_drift() -> None:
    classes = (_class(0), _class(1))
    features, labels = _support(classes, 5)
    bank = encode_support_prototypes(
        features, labels, classes, storage_format="int8", r0=0.20
    )
    with pytest.raises(TargetPrototypeBankError, match="overlap"):
        append_normalized_vectors(
            bank,
            np.stack([_direction(2)]),
            (_class(0),),
            radius=[0.2],
            count=[5],
        )
    with pytest.raises(TargetPrototypeBankError, match="K-shot"):
        append_normalized_vectors(
            bank,
            np.stack([_direction(2)]),
            (_class(2),),
            radius=[0.2],
            count=[1],
        )


def test_k1_append_rejects_a_different_locked_r0() -> None:
    old_classes = (_class(0), _class(1))
    before = encode_normalized_vectors(
        np.stack([_direction(0), _direction(1)]),
        old_classes,
        radius=[0.2, 0.2],
        count=[1, 1],
        storage_format="int8",
    )
    with pytest.raises(TargetPrototypeBankError, match="r0"):
        append_normalized_vectors(
            before,
            np.stack([_direction(2)]),
            (_class(2),),
            radius=[0.3],
            count=[1],
        )


def test_invalid_int8_payload_and_nonopaque_class_fail_closed() -> None:
    classes = (_class(0),)
    q = np.zeros((1, FEATURE_DIM), dtype=np.int8)
    q[0, 0] = -128
    scale = np.ones(1, dtype=np.float16)
    radius = np.ones(1, dtype=np.float16)
    count = np.ones(1, dtype=np.uint16)
    with pytest.raises(TargetPrototypeBankError, match="INT8"):
        TargetPrototypeBank(
            schema=SCHEMA,
            storage_format="int8",
            classes=classes,
            old_class_count=1,
            radius=radius,
            count=count,
            old_prefix_sha256="0" * 64,
            q=q,
            scale=scale,
        )
    with pytest.raises(TargetPrototypeBankError, match="opaque"):
        encode_normalized_vectors(
            np.stack([_direction(0)]),
            ("old-a",),
            radius=[0.2],
            count=[1],
            storage_format="fp16",
        )


def test_support_encoding_rejects_unbalanced_k_and_bad_r0() -> None:
    classes = (_class(0), _class(1))
    features = np.stack([_direction(0), _direction(1), _direction(1)])
    labels = [classes[0], classes[1], classes[1]]
    with pytest.raises(TargetPrototypeBankError, match="K-shot"):
        encode_support_prototypes(
            features, labels, classes, storage_format="int8", r0=0.2
        )
    one, one_labels = _support(classes, 1)
    with pytest.raises(TargetPrototypeBankError, match="r0"):
        encode_support_prototypes(
            one, one_labels, classes, storage_format="int8", r0=float("nan")
        )
