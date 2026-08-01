from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import cvsrffi.stage2_d106_rcmr_2v_qknn as rcmr
from cvsrffi.stage2_d106_matrix_protocol import (
    LEO_SCENARIOS,
    STATES,
    freeze_d106_matrix_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD_LOCK_PATH = ROOT / "configs" / "d106_rcmr_2v_method_lock_20260801.json"


def _sha(character: str) -> str:
    return character * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zero_first_code_row(wire: bytes, *, count: int, signed: bool) -> bytes:
    header_offset = len(rcmr.WIRE_MAGIC) + 4
    header_size = struct.unpack(">I", wire[len(rcmr.WIRE_MAGIC) : header_offset])[0]
    body_offset = header_offset + header_size
    tampered = bytearray(wire)
    view_offset = count * rcmr.Z_DIM if signed else 0
    start = body_offset + view_offset
    tampered[start : start + rcmr.Z_DIM] = b"\0" * rcmr.Z_DIM
    return bytes(tampered)


def _lock():
    return rcmr.load_d106_rcmr_2v_method_lock(
        METHOD_LOCK_PATH, expected_sha256=_file_sha(METHOD_LOCK_PATH)
    )


def _support_rows(
    *, k: int = 2, classes: tuple[str, ...] = ("class_a", "class_b", "class_c")
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    plus_rows: list[np.ndarray] = []
    signed_rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, class_name in enumerate(classes):
        for shot in range(k):
            plus = np.zeros(rcmr.Z_DIM, dtype=np.float32)
            plus[class_index] = np.float32(1.8 + 0.03 * shot)
            plus[20 + class_index] = np.float32(0.25 + 0.02 * shot)
            plus[40 + shot] = np.float32(0.04 * (class_index + 1))
            signed = plus.copy()
            signed[80 + class_index] = np.float32(-0.16 - 0.01 * shot)
            signed[100 + shot] = np.float32(0.02 * (class_index + 1))
            plus_rows.append(plus)
            signed_rows.append(signed)
            labels.append(class_name)
            physical_ids.append(f"physical-{class_index}-{shot}")
    # Deliberately return a noncanonical source order; sealing must canonicalize
    # global slots by physical ID instead of relying on caller order.
    order = list(reversed(range(len(plus_rows))))
    return (
        np.stack([plus_rows[index] for index in order]).astype(np.float32),
        np.stack([signed_rows[index] for index in order]).astype(np.float32),
        [labels[index] for index in order],
        [physical_ids[index] for index in order],
    )


def _binding(physical_ids: list[str], *, k: int, row: str = "row-d106"):
    return rcmr.D106RCMR2VBinding(
        capsule_id=_sha("1"),
        split_id=_sha("2"),
        validator_receipt_sha256=_sha("3"),
        support_physical_root_sha256=rcmr._support_physical_root(tuple(physical_ids)),
        row_id=row,
        seed=617,
        active_k=k,
        da_receipt_sha256=_sha("4"),
        paired_view_receipt_sha256=_sha("5"),
    )


def _state(
    *,
    k: int = 2,
    classes: tuple[str, ...] = ("class_a", "class_b", "class_c"),
    row: str = "row-d106",
):
    plus, signed, labels, physical_ids = _support_rows(k=k, classes=classes)
    binding = _binding(physical_ids, k=k, row=row)
    state = rcmr.build_d106_rcmr_2v_state(
        plus,
        signed,
        labels,
        physical_ids,
        classes,
        binding=binding,
        method_lock=_lock(),
    )
    return state, binding, plus, signed, labels, physical_ids


def _query(plus: np.ndarray, signed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_plus = plus[-1].copy()
    query_plus[125] += np.float32(0.011)
    query_signed = signed[-1].copy()
    query_signed[126] -= np.float32(0.007)
    return query_plus, query_signed


def test_formal_state_wire_strict_loader_and_compact_state(tmp_path: Path) -> None:
    state, binding, _plus, _signed, _labels, _ids = _state()
    assert state.is_formal
    assert not hasattr(state, "support_plus")
    assert not hasattr(state, "support_signed")
    assert state.codes_plus.dtype == np.dtype(np.int8)
    assert state.codes_signed.dtype == np.dtype(np.int8)
    assert state.scales_plus.dtype == np.dtype("<f2")
    assert state.reliabilities.dtype == np.dtype("<f2")
    assert state.class_index.dtype == np.dtype(np.uint8)
    assert not state.codes_plus.flags.writeable

    wire = rcmr.serialize_d106_rcmr_2v_state(state)
    wire_sha = hashlib.sha256(wire).hexdigest()
    restored = rcmr.deserialize_d106_rcmr_2v_state(
        wire,
        expected_wire_sha256=wire_sha,
        expected_binding=binding,
        method_lock=_lock(),
    )
    assert restored.is_formal
    assert rcmr.serialize_d106_rcmr_2v_state(restored) == wire

    wire_path = tmp_path / "state.rcmr"
    wire_path.write_bytes(wire)
    loaded = rcmr.load_d106_rcmr_2v_state(
        wire_path,
        expected_wire_sha256=wire_sha,
        expected_binding=binding,
        method_lock=_lock(),
    )
    assert loaded.state_receipt_sha256 == state.state_receipt_sha256
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.serialize_d106_rcmr_2v_state(replace(state, _formal_token=None))
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.deserialize_d106_rcmr_2v_state(
            wire + b"trailing",
            expected_wire_sha256=hashlib.sha256(wire + b"trailing").hexdigest(),
            expected_binding=binding,
            method_lock=_lock(),
        )
    for signed_view in (False, True):
        zero_row_wire = _zero_first_code_row(
            wire, count=len(state.class_index), signed=signed_view
        )
        with pytest.raises(rcmr.D106RCMR2VError, match="quantized state"):
            rcmr.deserialize_d106_rcmr_2v_state(
                zero_row_wire,
                expected_wire_sha256=hashlib.sha256(zero_row_wire).hexdigest(),
                expected_binding=binding,
                method_lock=_lock(),
            )


def test_two_view_midrank_quantization_and_k1_nonidentity() -> None:
    ranks = rcmr._midranks(np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_array_equal(ranks, np.asarray([0.375, 0.375, 0.75]))

    state, binding, plus, signed, _labels, _ids = _state(k=1)
    context = rcmr.prepare_d106_rcmr_2v_scoring_context(state)
    query_plus, query_signed = _query(plus, signed)
    original = rcmr.score_d106_rcmr_2v_query(
        state,
        query_plus,
        query_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=context,
    )
    alternate_signed = signed[0].copy()
    changed = rcmr.score_d106_rcmr_2v_query(
        state,
        query_plus,
        alternate_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=context,
    )
    assert state.binding.active_k == 1
    assert not np.array_equal(original.scores, changed.scores)
    assert 0.0 < original.query_reliability <= 1.0
    assert np.all(state.codes_plus >= -127)
    assert np.all(state.codes_plus <= 127)


def test_label_permutation_preserves_named_class_scores() -> None:
    state, binding, plus, signed, labels, physical_ids = _state()
    permuted_classes = ("class_c", "class_a", "class_b")
    permuted = rcmr.build_d106_rcmr_2v_state(
        plus,
        signed,
        labels,
        physical_ids,
        permuted_classes,
        binding=binding,
        method_lock=_lock(),
    )
    query_plus, query_signed = _query(plus, signed)
    left = rcmr.score_d106_rcmr_2v_query(
        state,
        query_plus,
        query_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=rcmr.prepare_d106_rcmr_2v_scoring_context(state),
    )
    right = rcmr.score_d106_rcmr_2v_query(
        permuted,
        query_plus,
        query_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=rcmr.prepare_d106_rcmr_2v_scoring_context(permuted),
    )
    left_scores = dict(zip(left.registry, left.scores.tolist(), strict=True))
    right_scores = dict(zip(right.registry, right.scores.tolist(), strict=True))
    assert left.predicted_class == right.predicted_class
    for class_name in state.registry:
        assert left_scores[class_name] == right_scores[class_name]


def test_zero_nonfinite_class_count_and_cross_class_tie_fail_closed() -> None:
    state, binding, plus, signed, labels, physical_ids = _state()
    zero = plus.copy()
    zero[0] = 0.0
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.build_d106_rcmr_2v_state(
            zero,
            signed,
            labels,
            physical_ids,
            state.registry,
            binding=binding,
            method_lock=_lock(),
        )
    wrong_labels = labels.copy()
    wrong_labels[0] = "class_a"
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.build_d106_rcmr_2v_state(
            plus,
            signed,
            wrong_labels,
            physical_ids,
            state.registry,
            binding=binding,
            method_lock=_lock(),
        )
    context = rcmr.prepare_d106_rcmr_2v_scoring_context(state)
    bad_query = plus[-1].copy()
    bad_query[0] = np.nan
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.score_d106_rcmr_2v_query(
            state,
            bad_query,
            signed[-1],
            da_receipt_sha256=binding.da_receipt_sha256,
            context=context,
        )

    tied_plus = np.zeros((2, rcmr.Z_DIM), dtype=np.float32)
    tied_plus[:, 0] = 1.0
    tied_signed = tied_plus.copy()
    tied_labels = ["alpha", "beta"]
    tied_ids = ["tie-0", "tie-1"]
    tied_binding = _binding(tied_ids, k=1, row="tie-row")
    tied_state = rcmr.build_d106_rcmr_2v_state(
        tied_plus,
        tied_signed,
        tied_labels,
        tied_ids,
        ("alpha", "beta"),
        binding=tied_binding,
        method_lock=_lock(),
    )
    with pytest.raises(rcmr.D106RCMRCrossClassTieError, match="CROSS_CLASS_SCORE_TIE"):
        rcmr.score_d106_rcmr_2v_query(
            tied_state,
            tied_plus[0],
            tied_signed[0],
            da_receipt_sha256=tied_binding.da_receipt_sha256,
            context=rcmr.prepare_d106_rcmr_2v_scoring_context(tied_state),
        )


def test_context_state_drift_and_no_query_update() -> None:
    state, binding, plus, signed, _labels, _ids = _state()
    context = rcmr.prepare_d106_rcmr_2v_scoring_context(state)
    query_plus, query_signed = _query(plus, signed)
    before_wire = rcmr.serialize_d106_rcmr_2v_state(state)
    before_context_receipt = context.context_receipt_sha256
    prediction = rcmr.score_d106_rcmr_2v_query(
        state,
        query_plus,
        query_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=context,
    )
    assert prediction.state_receipt_sha256 == state.state_receipt_sha256
    assert rcmr.serialize_d106_rcmr_2v_state(state) == before_wire
    assert context.context_receipt_sha256 == before_context_receipt
    assert not hasattr(context, "query_plus")
    assert not hasattr(context, "prediction")

    other_state, _other_binding, _p2, _s2, _l2, _i2 = _state(row="other-row")
    with pytest.raises(rcmr.D106RCMR2VError):
        rcmr.score_d106_rcmr_2v_query(
            other_state,
            query_plus,
            query_signed,
            da_receipt_sha256=binding.da_receipt_sha256,
            context=context,
        )
    for value in (
        state.codes_plus,
        state.codes_signed,
        state.scales_plus,
        state.scales_signed,
        state.reliabilities,
        state.class_index,
        context.decoded_plus,
        context.decoded_signed,
        context.profiles_plus,
        context.profiles_signed,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_resource_receipt_has_no_persistent_nxn_state() -> None:
    state, _binding_value, _plus, _signed, _labels, _ids = _state()
    receipt = rcmr.audit_d106_rcmr_2v_resources(state)
    assert receipt["persistent_dense_nxn_bytes"] == 0
    assert receipt["query_state_updates"] == 0
    assert receipt["current_numeric_state_bytes"] == 2 * 6 * rcmr.Z_DIM + 7 * 6
    assert receipt["prepare_support_support_mac"] == 6 * 5 * rcmr.Z_DIM
    for value in (
        state.codes_plus,
        state.codes_signed,
        state.scales_plus,
        state.scales_signed,
        state.reliabilities,
        state.class_index,
    ):
        assert value.shape != (6, 6)


def test_score_hot_path_uses_only_the_prepared_immutable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, binding, plus, signed, _labels, _ids = _state()
    context = rcmr.prepare_d106_rcmr_2v_scoring_context(state)
    query_plus, query_signed = _query(plus, signed)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("score hot path recomputed sealed support geometry")

    monkeypatch.setattr(rcmr, "_array_receipt", _forbidden)
    monkeypatch.setattr(rcmr, "_profiles", _forbidden)
    monkeypatch.setattr(rcmr, "_pairwise_distance_matrix", _forbidden)
    prediction = rcmr.score_d106_rcmr_2v_query(
        state,
        query_plus,
        query_signed,
        da_receipt_sha256=binding.da_receipt_sha256,
        context=context,
    )
    assert prediction.predicted_class in state.registry
    with pytest.raises(ValueError):
        prediction.scores.setflags(write=True)


def test_quantization_ties_even_and_fp64_distance_oracle() -> None:
    rows = np.zeros((1, rcmr.Z_DIM), dtype=np.float64)
    rows[0, :5] = np.asarray([127.0, 2.5, 3.5, -2.5, -3.5], dtype=np.float64)
    codes, scales = rcmr._quantize_rows(rows, "ties-even")
    assert scales[0] == np.float16(1.0)
    np.testing.assert_array_equal(
        codes[0, :5], np.asarray([127, 2, 4, -2, -4], dtype=np.int8)
    )

    left = np.zeros(rcmr.Z_DIM, dtype=np.float64)
    right = np.zeros(rcmr.Z_DIM, dtype=np.float64)
    left[:3] = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    right[:3] = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
    dot = 0.0
    for coordinate in range(rcmr.Z_DIM):
        dot += float(left[coordinate]) * float(right[coordinate])
    expected = min(2.0, max(0.0, 1.0 - dot))
    assert rcmr._same_binary64(rcmr._dot_distance(left, right), expected)


def _max_scale_state():
    classes = tuple(f"class_{index:02d}" for index in range(rcmr.MAX_REGISTERED_CLASSES))
    count = rcmr.MAX_SUPPORT_ROWS
    plus = np.zeros((count, rcmr.Z_DIM), dtype=np.float32)
    signed = np.zeros_like(plus)
    labels: list[str] = []
    physical_ids: list[str] = []
    position = 0
    for class_index, class_name in enumerate(classes):
        for shot in range(rcmr.MAX_SUPPORT_PER_CLASS):
            plus[position, class_index] = 1.0
            plus[position, 40 + shot] = np.float32(0.01 * (class_index + 1))
            plus[position, 80 + (position % 40)] = np.float32(0.001 * (shot + 1))
            signed[position] = plus[position]
            signed[position, 120 + class_index] = np.float32(-0.02 - 0.001 * shot)
            labels.append(class_name)
            physical_ids.append(f"max-{position:03d}")
            position += 1
    binding = _binding(physical_ids, k=rcmr.MAX_SUPPORT_PER_CLASS, row="max-row")
    state = rcmr.build_d106_rcmr_2v_state(
        plus,
        signed,
        labels,
        physical_ids,
        classes,
        binding=binding,
        method_lock=_lock(),
    )
    return state


def test_max_n_resource_and_actual_wire_accounting() -> None:
    state = _max_scale_state()
    context = rcmr.prepare_d106_rcmr_2v_scoring_context(state)
    wire = rcmr.serialize_d106_rcmr_2v_state(state)
    receipt = rcmr.audit_d106_rcmr_2v_resources(state)
    assert context.profiles_plus.shape == (260, 259)
    assert context.profiles_signed.shape == (260, 259)
    assert receipt["current_numeric_state_bytes"] == 85020
    assert receipt["temporary_prepare_peak_bytes"] == 2285920
    assert receipt["design_fixed_binary_payload_bytes_at_max_n"] == 86060
    assert receipt["design_fixed_total_bytes_at_max_n"] == 2371980
    assert receipt["actual_canonical_wire_bytes"] == len(wire)
    assert 86060 < len(wire) <= rcmr.MAX_CANONICAL_WIRE_BYTES
    assert receipt["actual_prepare_plus_canonical_wire_bytes"] == 2285920 + len(wire)
    assert "not measured" in receipt["unaccounted_overhead"]


def test_g0_synthetic_reject_and_token_cap() -> None:
    rejected = rcmr.judge_d106_rcmr_2v_g0(
        ["opaque-1", "opaque-2"],
        ["class_a", "class_b"],
        ["class_a", "class_b"],
        formal_tap_receipt_sha256=_sha("f"),
    )
    assert rejected["g0_status"] == "REJECT_NO_FUNCTION"
    assert rejected["argmax_changed_count"] == 0
    assert rejected["query_truth_access"] is False
    observed = rcmr.judge_d106_rcmr_2v_g0(
        ["opaque-1", "opaque-2"],
        ["class_b", "class_b"],
        ["class_a", "class_b"],
        formal_tap_receipt_sha256=_sha("f"),
    )
    assert observed["g0_status"] == "G0_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
    ids = ["short-a", "short-b"]
    with pytest.raises(rcmr.D106RCMR2VError, match="wire-token limit"):
        _binding(ids, k=1, row="x" * 128)
    with pytest.raises(rcmr.D106RCMR2VError, match="wire-token limit"):
        rcmr._registry_tokens(("x" * 128, "class_b"))


def test_exact_registry_and_target25_row_wire_boundaries() -> None:
    legal_class = "cls_" + "a" * 64
    assert len(legal_class.encode("utf-8")) == 68
    assert len(json.dumps(legal_class).encode("utf-8")) == 70
    assert rcmr._registry_tokens((legal_class, "class_b"))[0] == legal_class
    oversized_class = legal_class + "0"
    assert len(oversized_class.encode("utf-8")) == 69
    assert len(json.dumps(oversized_class).encode("utf-8")) == 71
    with pytest.raises(rcmr.D106RCMR2VError, match="wire-token limit"):
        rcmr._registry_tokens((oversized_class, "class_b"))

    matrix = freeze_d106_matrix_protocol()
    row_ids = [
        f"{job.job_id}::{scenario}::{state}"
        for job in matrix.jobs
        for scenario in LEO_SCENARIOS
        for state in STATES
    ]
    longest = max(row_ids, key=lambda value: len(value.encode("utf-8")))
    assert len(longest.encode("utf-8")) == 66
    assert len(json.dumps(longest).encode("utf-8")) == 68
    assert (
        rcmr._require_text(
            longest,
            "row_id",
            max_wire_bytes=rcmr.MAX_ROW_ID_WIRE_BYTES,
        )
        == longest
    )
    with pytest.raises(rcmr.D106RCMR2VError, match="wire-token limit"):
        rcmr._require_text(
            longest + "x",
            "row_id",
            max_wire_bytes=rcmr.MAX_ROW_ID_WIRE_BYTES,
        )


def test_method_lock_closes_exact_wire_caps() -> None:
    document = json.loads(METHOD_LOCK_PATH.read_text(encoding="utf-8"))
    assert document["max_registry_token_wire_bytes"] == 70
    assert document["max_row_id_wire_bytes"] == 68
    assert document["max_canonical_wire_bytes"] == 90000
    assert rcmr.MAX_REGISTRY_TOKEN_WIRE_BYTES == 70
    assert rcmr.MAX_ROW_ID_WIRE_BYTES == 68
    assert rcmr.MAX_CANONICAL_WIRE_BYTES == 90000
    assert _lock().is_loader_authorized
