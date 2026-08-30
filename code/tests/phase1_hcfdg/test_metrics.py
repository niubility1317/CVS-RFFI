from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi.phase1_hcfdg.metrics import (
    SameRowMetrics,
    conditional_receiver_leakage,
    counterfactual_effectiveness,
    domain_drift_ratio,
    environment_tx_leakage,
    rank_source_rows,
    specific_gap,
)


def _source_probe_fixture() -> tuple[np.ndarray, ...]:
    train_embeddings = np.asarray(
        [
            [3.0, 0.0],
            [4.0, 0.0],
            [-3.0, 0.0],
            [-4.0, 0.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, -3.0],
            [0.0, -4.0],
        ]
    )
    train_tx = np.asarray([10, 10, 10, 10, 20, 20, 20, 20])
    train_receiver = np.asarray([1, 1, 2, 2, 1, 1, 2, 2])
    validation_embeddings = np.asarray(
        [
            [3.2, 0.0],
            [3.8, 0.0],
            [-3.2, 0.0],
            [-3.8, 0.0],
            [0.0, 3.2],
            [0.0, 3.8],
            [0.0, -3.2],
            [0.0, -3.8],
        ]
    )
    validation_tx = train_tx.copy()
    validation_receiver = train_receiver.copy()
    return (
        train_embeddings,
        train_tx,
        train_receiver,
        validation_embeddings,
        validation_tx,
        validation_receiver,
    )


def test_conditional_receiver_leakage_returns_macro_and_per_tx_accuracy():
    (
        train_embeddings,
        train_tx,
        train_receiver,
        validation_embeddings,
        validation_tx,
        validation_receiver,
    ) = _source_probe_fixture()

    result = conditional_receiver_leakage(
        train_embeddings,
        train_tx,
        train_receiver,
        validation_embeddings,
        validation_tx,
        validation_receiver,
        ridge=1.0,
    )

    assert result.macro_accuracy == pytest.approx(1.0)
    assert result.per_tx == {10: pytest.approx(1.0), 20: pytest.approx(1.0)}
    assert float(result) == pytest.approx(1.0)


def test_environment_tx_leakage_uses_validation_only_for_scoring():
    train_environment = np.asarray(
        [[1.0, 0.0], [1.2, 0.0], [-1.0, 0.0], [-1.2, 0.0]]
    )
    train_tx = np.asarray([10, 10, 20, 20])
    validation_environment = np.asarray(
        [[0.9, 0.0], [1.1, 0.0], [-0.9, 0.0], [-1.1, 0.0]]
    )
    validation_tx = train_tx.copy()

    result = environment_tx_leakage(
        train_environment,
        train_tx,
        validation_environment,
        validation_tx,
        ridge=1.0,
    )

    assert result.macro_accuracy == pytest.approx(1.0)
    assert result.per_tx == {10: pytest.approx(1.0), 20: pytest.approx(1.0)}


def test_environment_tx_leakage_rejects_validation_tx_missing_from_train():
    with pytest.raises(ValueError, match="no train rows"):
        environment_tx_leakage(
            np.asarray([[1.0, 0.0], [1.1, 0.0]]),
            np.asarray([10, 10]),
            np.asarray([[-1.0, 0.0]]),
            np.asarray([20]),
        )


def test_specific_gap_is_specific_head_accuracy_minus_common_head_accuracy():
    labels = np.asarray([0, 1, 1, 0])
    common_logits = np.asarray([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    specific_logits = np.asarray([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]])

    assert specific_gap(common_logits, specific_logits, labels) == pytest.approx(0.5)
    assert specific_gap(0.5, 0.75) == pytest.approx(0.25)


def test_counterfactual_effectiveness_reports_identity_and_environment_terms():
    labels = np.asarray([0, 1, 0, 1])
    original_logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    counterfactual_logits = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]]
    )
    target_environment = np.asarray([0, 1, 0, 1])
    counterfactual_environment_logits = np.asarray(
        [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 0.0]]
    )

    result = counterfactual_effectiveness(
        original_logits,
        counterfactual_logits,
        labels=labels,
        counterfactual_environment_logits=counterfactual_environment_logits,
        target_environment_labels=target_environment,
    )

    assert result.identity_retention == pytest.approx(0.75)
    assert result.environment_switch == pytest.approx(0.75)
    assert result.to_dict() == {
        "identity_retention": pytest.approx(0.75),
        "environment_switch": pytest.approx(0.75),
    }


def test_domain_drift_ratio_matches_closed_form_example():
    class_domain_centers = np.asarray(
        [[[-1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [3.0, 0.0]]]
    )
    class_centers = np.asarray([[0.0, 0.0], [2.0, 0.0]])

    value = domain_drift_ratio(class_domain_centers, class_centers)

    assert value == pytest.approx(0.25)


def test_domain_drift_ratio_accepts_flattened_class_domain_centers():
    class_domain_centers = np.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    )
    class_centers = np.asarray([[0.0, 0.0], [2.0, 0.0]])

    assert domain_drift_ratio(class_domain_centers, class_centers) == pytest.approx(0.25)


def test_source_ranking_never_combines_different_rows_and_round_trips():
    rows = (
        SameRowMetrics(
            row_id="A5-F8-S392001",
            candidate_id="A5",
            heldout_receiver=8,
            seed=392001,
            clean=0.95,
            leo_mean=0.85,
            leo_floor=0.70,
            min_class_margin=0.12,
        ),
        SameRowMetrics(
            row_id="A5-F8-S392002",
            candidate_id="A5",
            heldout_receiver=8,
            seed=392002,
            clean=0.92,
            leo_mean=0.88,
            leo_floor=0.80,
            min_class_margin=0.16,
        ),
        SameRowMetrics(
            row_id="A4-F8-S392002",
            candidate_id="A4",
            heldout_receiver=8,
            seed=392002,
            clean=0.94,
            leo_mean=0.83,
            leo_floor=0.75,
            min_class_margin=0.15,
        ),
    )
    rows_by_id = {row.row_id: row for row in rows}

    ranked = rank_source_rows(rows)

    assert ranked[0].row_id == "A5-F8-S392002"
    source_row = rows_by_id[ranked[0].row_id]
    assert (
        ranked[0].candidate_id,
        ranked[0].heldout_receiver,
        ranked[0].seed,
        ranked[0].clean,
        ranked[0].leo_mean,
        ranked[0].leo_floor,
    ) == (
        source_row.candidate_id,
        source_row.heldout_receiver,
        source_row.seed,
        source_row.clean,
        source_row.leo_mean,
        source_row.leo_floor,
    )
    assert ranked[0].harmonic_score == pytest.approx(
        2.0 * ranked[0].clean * ranked[0].leo_mean / (ranked[0].clean + ranked[0].leo_mean)
    )

    payload = json.loads(json.dumps(ranked[0].to_dict(), sort_keys=True))
    assert SameRowMetrics.from_dict(payload) == ranked[0]

    duplicate_payload = rows[0].to_dict()
    duplicate_payload["candidate_id"] = "A6"
    with pytest.raises(ValueError, match="unique row_id"):
        rank_source_rows((rows[0], duplicate_payload))


def test_same_row_structured_identity_survives_serialization_and_coercion():
    row = SameRowMetrics(
        row_id="run-20260830-row-17",
        candidate_id="HCF-DG-A5",
        heldout_receiver=8,
        seed=392002,
        clean=0.92,
        leo_mean=0.88,
        leo_floor=0.80,
        lodo_mean=0.84,
        resources={"gpu_hours": 1.25},
    )
    expected_identity = {
        "row_id": "run-20260830-row-17",
        "candidate_id": "HCF-DG-A5",
        "heldout_receiver": 8,
        "seed": 392002,
    }

    payload = row.to_dict()
    assert {key: payload[key] for key in expected_identity} == expected_identity
    assert SameRowMetrics.from_dict(payload) == row
    assert SameRowMetrics.from_json(row.to_json()) == row
    assert rank_source_rows([payload]) == [row]
    assert rank_source_rows([SimpleNamespace(**payload)]) == [row]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_id", ""),
        ("candidate_id", "   "),
        ("heldout_receiver", 0),
        ("heldout_receiver", 8.0),
        ("heldout_receiver", True),
        ("seed", -1),
        ("seed", 392002.0),
        ("seed", False),
    ),
)
def test_same_row_rejects_invalid_structured_identity(field, value):
    payload = {
        "row_id": "run-20260830-row-17",
        "candidate_id": "HCF-DG-A5",
        "heldout_receiver": 8,
        "seed": 392002,
        "clean": 0.92,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        SameRowMetrics(**payload)


@pytest.mark.parametrize("field", ("candidate_id", "heldout_receiver", "seed"))
def test_same_row_from_dict_rejects_missing_structured_identity(field):
    payload = {
        "row_id": "run-20260830-row-17",
        "candidate_id": "HCF-DG-A5",
        "heldout_receiver": 8,
        "seed": 392002,
        "clean": 0.92,
    }
    del payload[field]

    with pytest.raises(ValueError, match=field):
        SameRowMetrics.from_dict(payload)
