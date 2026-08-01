from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json

import numpy as np
import pytest

import cvsrffi.stage2_d107_scmkrr as scm


CLASSES = ("tx_a", "tx_b", "tx_c", "tx_d", "tx_e", "tx_f")
TAU = np.asarray([0.18, 0.24, 0.31], dtype=np.float64)
SPECTRUM = np.asarray([0.52, 0.63, 0.77], dtype=np.float64)


def _rows(
    *, k: int = 2, classes: tuple[str, ...] = CLASSES
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_name in enumerate(classes):
        for shot in range(k):
            row = np.zeros(scm.Z_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.4 + 0.03 * shot)
            row[20 + class_index] = np.float32(0.31 + 0.02 * shot)
            row[60 + shot] = np.float32(-0.08 * (class_index + 1))
            row[100 + ((class_index + shot) % 20)] = np.float32(
                0.025 * (shot + 1)
            )
            rows.append(row)
            labels.append(class_name)
    order = list(reversed(range(len(rows))))
    return (
        np.stack([rows[index] for index in order]).astype(np.float32),
        [labels[index] for index in order],
    )


def _query(support: np.ndarray, labels: list[str], class_name: str = "tx_c") -> np.ndarray:
    row = support[labels.index(class_name)].copy()
    row[140] += np.float32(0.037)
    row[141] -= np.float32(0.019)
    return row[None, :]


def _state(
    arm: str,
    *,
    k: int = 2,
    classes: tuple[str, ...] = CLASSES,
    anchor: np.ndarray | None = None,
) -> tuple[scm.SCMKRRState, np.ndarray, list[str]]:
    support, labels = _rows(k=k, classes=classes)
    if anchor is None:
        anchor, _ = _rows(k=k, classes=CLASSES)
    state = scm.build_scmkrr_state(
        support,
        labels,
        classes,
        anchor,
        TAU,
        SPECTRUM,
        arm,
    )
    return state, support, labels


def _decode(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    return rows / np.linalg.norm(rows, axis=1)[:, None]


def _rbf(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.exp(-np.maximum(distances, 0.0) / bandwidth)


@pytest.mark.parametrize("arm", scm.ARMS)
def test_formula_matches_independent_reference_for_all_arms(arm: str) -> None:
    state, support, labels = _state(arm)
    query = _query(support, labels)
    q = query.astype(np.float64)
    q /= np.linalg.norm(q, axis=1)[:, None]
    anchors = _decode(state.anchor_codes_qint8, state.anchor_scales_fp16)
    prototypes = _decode(
        state.prototype_codes_qint8, state.prototype_scales_fp16
    )
    expected = _rbf(q, prototypes, state.bandwidth)
    if arm in ("M_DA", "M_JOINT"):
        expected = (
            expected
            - np.mean(_rbf(q, anchors, state.bandwidth), axis=1)[:, None]
            - state.prototype_anchor_mean_fp16.astype(np.float64)[None, :]
            + state.anchor_grand_mean
        )
    if arm in ("M_HEAD", "M_JOINT"):
        expected = expected @ state.head_coefficients_fp16.astype(np.float64)
    np.testing.assert_array_equal(
        scm.score_scmkrr_query(state, query), expected.astype(np.float32)
    )


def test_four_arms_are_bound_and_mechanistically_distinct() -> None:
    support, labels = _rows(k=2)
    query = _query(support, labels)
    states = {
        arm: scm.build_scmkrr_state(
            support, labels, CLASSES, support, TAU, SPECTRUM, arm
        )
        for arm in scm.ARMS
    }
    scores = {arm: scm.score_scmkrr_query(state, query) for arm, state in states.items()}
    assert states["M0"].state_receipt_sha256 != states["M_DA"].state_receipt_sha256
    assert states["M_HEAD"].head_coefficients_fp16.shape == (6, 6)
    assert states["M0"].head_coefficients_fp16.shape == (0, 0)
    assert not np.array_equal(scores["M0"], scores["M_DA"])
    assert not np.array_equal(scores["M0"], scores["M_HEAD"])
    assert not np.array_equal(scores["M_DA"], scores["M_JOINT"])


@pytest.mark.parametrize("arm", scm.ARMS)
def test_k1_is_a_real_nonfallback_all_class_state(arm: str) -> None:
    state, support, labels = _state(arm, k=1)
    scores = scm.score_scmkrr_query(state, _query(support, labels, "tx_f"))
    assert state.prototype_codes_qint8.shape == (len(CLASSES), scm.Z_DIM)
    assert state.anchor_codes_qint8.shape == (len(CLASSES), scm.Z_DIM)
    assert scores.shape == (1, len(CLASSES))
    assert state.bandwidth > 0.0
    assert state.regularization > 0.0


@pytest.mark.parametrize("arm", scm.ARMS)
def test_registered_class_permutation_only_permutes_score_columns(arm: str) -> None:
    support, labels = _rows(k=2)
    query = _query(support, labels)
    original = scm.build_scmkrr_state(
        support, labels, CLASSES, support, TAU, SPECTRUM, arm
    )
    permutation = ("tx_d", "tx_a", "tx_f", "tx_c", "tx_b", "tx_e")
    permuted = scm.build_scmkrr_state(
        support, labels, permutation, support, TAU, SPECTRUM, arm
    )
    original_scores = scm.score_scmkrr_query(original, query)
    permuted_scores = scm.score_scmkrr_query(permuted, query)
    reorder = [permutation.index(class_name) for class_name in CLASSES]
    np.testing.assert_allclose(
        original_scores, permuted_scores[:, reorder], rtol=0.0, atol=2e-3
    )


@pytest.mark.parametrize("arm", scm.ARMS)
def test_common_positive_scale_and_signed_permutation_orthogonal_invariance(
    arm: str,
) -> None:
    support, labels = _rows(k=2)
    query = _query(support, labels)
    baseline = scm.build_scmkrr_state(
        support, labels, CLASSES, support, TAU, SPECTRUM, arm
    )
    baseline_scores = scm.score_scmkrr_query(baseline, query)

    scaled = scm.build_scmkrr_state(
        support * np.float32(4.0),
        labels,
        CLASSES,
        support * np.float32(4.0),
        TAU,
        SPECTRUM,
        arm,
    )
    np.testing.assert_array_equal(
        baseline_scores,
        scm.score_scmkrr_query(scaled, query * np.float32(4.0)),
    )

    order = np.arange(scm.Z_DIM)[::-1]
    signs = np.where(np.arange(scm.Z_DIM) % 2 == 0, 1.0, -1.0).astype(np.float32)
    transformed_support = support[:, order] * signs
    transformed_query = query[:, order] * signs
    transformed = scm.build_scmkrr_state(
        transformed_support,
        labels,
        CLASSES,
        transformed_support,
        TAU,
        SPECTRUM,
        arm,
    )
    np.testing.assert_allclose(
        baseline_scores,
        scm.score_scmkrr_query(transformed, transformed_query),
        rtol=0.0,
        atol=2e-3,
    )


def test_common_translation_is_not_silently_an_isometric_noop() -> None:
    support, labels = _rows(k=2)
    query = _query(support, labels)
    baseline = scm.build_scmkrr_state(
        support, labels, CLASSES, support, TAU, SPECTRUM, "M_JOINT"
    )
    shift = np.zeros((1, scm.Z_DIM), dtype=np.float32)
    shift[0, 150] = np.float32(0.35)
    translated = scm.build_scmkrr_state(
        support + shift,
        labels,
        CLASSES,
        support + shift,
        TAU,
        SPECTRUM,
        "M_JOINT",
    )
    assert not np.array_equal(
        scm.score_scmkrr_query(baseline, query),
        scm.score_scmkrr_query(translated, query + shift),
    )


def test_after_state_reuses_exact_frozen_before_anchor() -> None:
    before_support, before_labels = _rows(k=2)
    before = scm.build_scmkrr_state(
        before_support,
        before_labels,
        CLASSES,
        before_support,
        TAU,
        SPECTRUM,
        "M_JOINT",
    )
    all_classes = CLASSES + ("tx_g", "tx_h")
    after_support, after_labels = _rows(k=2, classes=all_classes)
    after = scm.build_scmkrr_state(
        after_support,
        after_labels,
        all_classes,
        before_support,
        TAU,
        SPECTRUM,
        "M_JOINT",
    )
    np.testing.assert_array_equal(before.anchor_codes_qint8, after.anchor_codes_qint8)
    np.testing.assert_array_equal(before.anchor_scales_fp16, after.anchor_scales_fp16)
    assert after.prototype_codes_qint8.shape[0] == len(all_classes)


def test_public_surface_has_no_truth_role_quota_fit_or_update() -> None:
    build_parameters = set(inspect.signature(scm.build_scmkrr_state).parameters)
    score_parameters = set(inspect.signature(scm.score_scmkrr_query).parameters)
    forbidden = {
        "truth",
        "query_truth",
        "role",
        "old_role",
        "new_role",
        "quota",
        "batch_class_count",
        "query_labels",
    }
    assert not build_parameters.intersection(forbidden)
    assert score_parameters == {"state", "query_signed"}
    state_names = {field.name for field in fields(scm.SCMKRRState)}
    assert not any(token in name for name in state_names for token in ("truth", "role", "quota"))


@pytest.mark.parametrize("arm", scm.ARMS)
def test_query_batch_equals_independent_single_rows_and_does_not_update_state(
    arm: str,
) -> None:
    state, support, labels = _state(arm)
    queries = np.concatenate(
        [_query(support, labels, "tx_b"), _query(support, labels, "tx_e")], axis=0
    )
    receipt_before = state.state_receipt_sha256
    batch = scm.score_scmkrr_query(state, queries)
    separate = np.concatenate(
        [scm.score_scmkrr_query(state, queries[index : index + 1]) for index in range(2)],
        axis=0,
    )
    np.testing.assert_array_equal(batch, separate)
    assert state.state_receipt_sha256 == receipt_before


def test_rho_lambda_and_simplex_head_match_frozen_equations() -> None:
    state, _, _ = _state("M_HEAD")
    expected_rho = float(np.median(TAU / (TAU + SPECTRUM)))
    assert state.rho == expected_rho
    prototypes = _decode(
        state.prototype_codes_qint8, state.prototype_scales_fp16
    )
    kernel = _rbf(prototypes, prototypes, state.bandwidth)
    expected_lambda = expected_rho / (1.0 - expected_rho) * np.trace(kernel) / len(CLASSES)
    assert state.regularization == expected_lambda
    target = np.eye(len(CLASSES)) - np.ones((len(CLASSES), len(CLASSES))) / len(CLASSES)
    expected_head = np.linalg.solve(
        kernel + expected_lambda * np.eye(len(CLASSES)), target
    ).astype(np.float16)
    np.testing.assert_array_equal(state.head_coefficients_fp16, expected_head)


def test_wire_roundtrip_is_canonical_immutable_and_tamper_evident() -> None:
    state, support, labels = _state("M_JOINT")
    query = _query(support, labels)
    wire = scm.serialize_scmkrr_state(state)
    restored = scm.deserialize_scmkrr_state(
        wire, expected_wire_sha256=hashlib.sha256(wire).hexdigest()
    )
    assert scm.serialize_scmkrr_state(restored) == wire
    assert not restored.anchor_codes_qint8.flags.writeable
    assert not restored.head_coefficients_fp16.flags.writeable
    np.testing.assert_array_equal(
        scm.score_scmkrr_query(state, query),
        scm.score_scmkrr_query(restored, query),
    )

    tampered = bytearray(wire)
    tampered[-1] ^= 1
    with pytest.raises(scm.SCMKRRError):
        scm.deserialize_scmkrr_state(bytes(tampered))
    with pytest.raises(scm.SCMKRRError):
        scm.deserialize_scmkrr_state(wire + b"x")
    with pytest.raises(scm.SCMKRRError):
        scm.deserialize_scmkrr_state(wire[:-1])
    with pytest.raises(scm.SCMKRRError):
        scm.deserialize_scmkrr_state(
            wire, expected_wire_sha256="0" * 64
        )


def test_dataclass_tamper_and_invalid_inputs_fail_closed() -> None:
    state, support, labels = _state("M0")
    with pytest.raises(scm.SCMKRRError):
        replace(state, bandwidth=state.bandwidth * 2.0)
    bad = support.copy()
    bad[0, 0] = np.nan
    with pytest.raises(scm.SCMKRRError):
        scm.build_scmkrr_state(
            bad, labels, CLASSES, support, TAU, SPECTRUM, "M0"
        )
    zero = support.copy()
    zero[0] = 0.0
    with pytest.raises(scm.SCMKRRError):
        scm.build_scmkrr_state(
            zero, labels, CLASSES, support, TAU, SPECTRUM, "M0"
        )
    with pytest.raises(scm.SCMKRRError):
        scm.build_scmkrr_state(
            support, labels, CLASSES, support, np.asarray([0.0]), SPECTRUM, "M0"
        )
    with pytest.raises(scm.SCMKRRError):
        scm.build_scmkrr_state(
            support,
            labels,
            CLASSES,
            support,
            np.asarray([0.1, 0.2], dtype=np.float64),
            np.asarray([0.3, 0.4], dtype=np.float64),
            "M0",
        )
    with pytest.raises(scm.SCMKRRError):
        scm.score_scmkrr_query(state, np.zeros((1, scm.Z_DIM), dtype=np.float32))


def test_degenerate_bandwidth_uses_only_epsilon_and_exact_tie_fails_closed() -> None:
    row = np.zeros(scm.Z_DIM, dtype=np.float32)
    row[0] = 1.0
    support = np.stack([row, row]).astype(np.float32)
    state = scm.build_scmkrr_state(
        support,
        ["tx_a", "tx_b"],
        ("tx_a", "tx_b"),
        support,
        TAU,
        SPECTRUM,
        "M0",
    )
    assert state.bandwidth == np.finfo(np.float64).eps
    with pytest.raises(scm.SCMKRRTieError, match=scm.SCMKRRTieError.code):
        scm.score_scmkrr_query(state, row[None, :])


def test_resource_summary_closes_maximum_state_and_wire_bound() -> None:
    classes = tuple(f"tx_{index:02d}" for index in range(26))
    support, labels = _rows(k=10, classes=classes)
    anchor = support[:60].copy()
    state = scm.build_scmkrr_state(
        support, labels, classes, anchor, TAU, SPECTRUM, "M_JOINT"
    )
    summary = scm.scmkrr_resource_summary(state)
    assert summary["anchor_rows"] == 60
    assert summary["registered_class_count"] == 26
    assert summary["numeric_state_bytes"] < 16 * 1024
    assert summary["canonical_wire_bytes"] <= scm.MAX_CANONICAL_WIRE_BYTES
    assert summary["query_mac_upper_bound"] == (60 + 26) * 160 + 26 * 26


def test_method_lock_is_canonical_and_records_traceability() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "stage2_d107_scmkrr_r1.json"
    )
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    assert lock["candidate_id"] == scm.CANDIDATE_ID
    assert tuple(lock["arms"]) == scm.ARMS
    assert lock["forbidden_inputs"]
    assert {item["id"] for item in lock["traceability"]} == {
        "D107-01",
        "D107-02",
        "D107-03",
        "D107-04",
        "D107-05",
        "D107-06",
    }
    assert {item["status"] for item in lock["traceability"]} == {"verified"}
