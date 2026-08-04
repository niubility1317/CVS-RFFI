from __future__ import annotations

import hashlib

import pytest

from cvsrffi import stage2_next_r4_matrix as matrix


CLASSES = ("tx-z", "tx-a", "tx-f", "tx-c", "tx-e", "tx-b")


def _maps(classes: tuple[str, ...]):
    ordered = tuple(sorted(classes))
    support5 = {
        cls: [f"support-{cls}-{index}" for index in range(5)] for cls in ordered
    }
    support1 = {cls: values[:1] for cls, values in support5.items()}
    query = {cls: [f"query-{cls}-{index}" for index in range(9)] for cls in ordered}
    observations = {
        cls: [f"observation-{cls}-{index}" for index in range(9)] for cls in ordered
    }
    phase1 = [f"phase1-{index}" for index in range(matrix.PHASE1_FIT_COUNT)]
    return support1, support5, query, observations, phase1


def test_frozen_four_state_counts_and_sorted_registry() -> None:
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)
    assert tuple(plan["held_classes"]) == tuple(sorted(CLASSES))
    assert plan["candidate_id"] == "NEXT-R4-FA-RDCE3-CER-PLR160"
    assert tuple(plan["state_ids"]) == matrix.STATE_IDS
    assert plan["state_names_zh"] == matrix.STATE_NAMES_ZH
    assert plan["row_count"] == 24
    assert plan["k1_unique_prediction_count"] == 48
    assert plan["k1_artifact_count"] == 96
    assert plan["k5_unique_prediction_count"] == 96
    assert plan["k5_artifact_count"] == 96
    assert plan["unique_prediction_count"] == 144
    assert plan["artifact_arm_count"] == 192
    assert plan["k1_h_semantics"] == "per_logit_alias_receipt"
    assert plan["k5_h_semantics"] == "unique_prediction"
    assert all(row["formal_new_registration_claim"] is False for row in plan["rows"])
    matrix.validate_next_r4_proxy24_plan(plan)


def test_registry_and_receivers_fail_closed() -> None:
    with pytest.raises(matrix.NextR4MatrixError):
        matrix.build_next_r4_proxy24_plan(CLASSES[:-1])
    with pytest.raises(matrix.NextR4MatrixError):
        matrix.build_next_r4_proxy24_plan(CLASSES, held_receivers=("18-2", "1-1"))
    with pytest.raises(matrix.NextR4MatrixError):
        matrix.build_next_r4_proxy24_plan(("tx-a", "tx-a", "tx-b", "tx-c", "tx-d", "tx-e"))


def test_k_prefix_query_reuse_and_fa_state_receipt() -> None:
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)
    rows = [matrix.outer_key_from_mapping(value) for value in plan["rows"][:2]]
    support1, support5, query, observations, phase1 = _maps(CLASSES)
    view_ids = {"K1": query, "K5": query, **{state: query for state in matrix.STATE_IDS}}
    view_observations = {
        "K1": observations,
        "K5": observations,
        **{state: observations for state in matrix.STATE_IDS},
    }
    state_sha = "a" * 64
    receipt = matrix.bind_next_r4_physical_ids(
        row_k1=rows[0],
        row_k5=rows[1],
        phase1_fit_ids=phase1,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
        query_observation_ids_by_class=observations,
        query_ids_by_view=view_ids,
        query_observation_ids_by_view=view_observations,
        fa_state_sha256_by_state={"DA1_REG0": state_sha, "DA1_REG1": state_sha},
    )
    assert receipt["k1_is_exact_k5_prefix"] is True
    assert receipt["common_query_physical_ids_across_k_states"] is True
    assert receipt["common_query_observation_ids_across_k_states"] is True
    assert receipt["fa_state_reuse_verified"] is True
    assert matrix.validate_next_r4_binding(receipt)["binding_sha256"] == receipt[
        "binding_sha256"
    ]


def test_query_or_fa_reuse_drift_is_rejected() -> None:
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)
    rows = [matrix.outer_key_from_mapping(value) for value in plan["rows"][:2]]
    support1, support5, query, observations, phase1 = _maps(CLASSES)
    drift_query = {key: list(value) for key, value in query.items()}
    drift_query[sorted(CLASSES)[0]] = ["query-drift"] + drift_query[sorted(CLASSES)[0]][1:]
    with pytest.raises(matrix.NextR4MatrixError, match="query_ids_by_view"):
        matrix.bind_next_r4_physical_ids(
            row_k1=rows[0],
            row_k5=rows[1],
            phase1_fit_ids=phase1,
            k1_support_ids_by_class=support1,
            k5_support_ids_by_class=support5,
            query_ids_by_class=query,
            query_observation_ids_by_class=observations,
            query_ids_by_view={"K1": query, "K5": drift_query},
        )
    with pytest.raises(matrix.NextR4MatrixError):
        matrix.validate_fa_state_reuse({"DA1_REG0": "a" * 64, "DA1_REG1": "b" * 64})


def test_plan_digest_is_immutable() -> None:
    plan = dict(matrix.build_next_r4_proxy24_plan(CLASSES))
    plan["unique_prediction_count"] = 143
    with pytest.raises(matrix.NextR4MatrixError):
        matrix.validate_next_r4_proxy24_plan(plan)
