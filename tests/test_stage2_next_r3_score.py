from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi import stage2_next_r3_matrix as matrix
from cvsrffi import stage2_next_r3_score as scorer


CLASSES = tuple(f"tx-{index}" for index in range(6))


def _fixture() -> tuple[dict, dict, dict]:
    plan = dict(matrix.build_next_r3_proxy24_plan(CLASSES))
    truth: dict[str, str] = {}
    rows: list[dict] = []
    for planned in plan["rows"]:
        registrations: dict[str, dict] = {}
        for registration_id in matrix.REGISTRATION_IDS:
            registered = tuple(
                planned["retained_classes"]
                if registration_id == "REG0"
                else planned["all_registered_classes"]
            )
            query_ids = []
            for cls in registered:
                for index in range(matrix.QUERY_PER_CLASS):
                    query_id = f"{planned['held_receiver']}|{planned['held_class']}|{cls}|q{index}"
                    query_ids.append(query_id)
                    truth[query_id] = cls
            states: dict[str, dict] = {}
            for state_id in matrix.STATE_IDS:
                arms: dict[str, list[str]] = {}
                for arm_id in matrix.ARM_IDS:
                    values = [truth[qid] for qid in query_ids]
                    # One same-row K5 DA1 L improvement exercises Q/L DID.
                    if (
                        planned["active_k"] == 5
                        and state_id == "DA1_REG1"
                        and arm_id == "R1L"
                        and values
                    ):
                        values = list(values)
                        values[0] = registered[0]
                    arms[arm_id] = values
                states[state_id] = {
                    "query_physical_ids": query_ids,
                    "arms": arms,
                }
            registrations[registration_id] = {
                "registered_classes": registered,
                "query_physical_ids": query_ids,
                "states": states,
            }
        rows.append(
            {
                "row_id": planned["row_id"],
                "held_receiver": planned["held_receiver"],
                "held_class": planned["held_class"],
                "active_k": planned["active_k"],
                "retained_classes": planned["retained_classes"],
                "all_registered_classes": planned["all_registered_classes"],
                "evaluation_semantics": matrix.PROXY_SEMANTICS,
                "formal_new_registration_claim": False,
                "registrations": registrations,
            }
        )
    prediction = {
        "schema": scorer.PREDICTION_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "matrix_sha256": plan["matrix_sha256"],
        "evaluation_semantics": matrix.PROXY_SEMANTICS,
        "formal_new_registration_claim": False,
        "truth_loaded": False,
        "rows_complete": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "rows": rows,
    }
    return plan, prediction, truth


def test_complete_matrix_scores_same_row_context_and_na_reg0() -> None:
    plan, prediction, truth = _fixture()
    result = scorer.score_next_r3_proxy24(
        prediction=prediction, plan=plan, truth_by_query_id=truth
    )
    assert result["row_count"] == 24
    assert result["state_prediction_count"] == 96
    assert result["formal_new_registration_claim"] is False
    assert result["cross_row_best_selection_used"] is False
    assert result["decision"] == "SOURCE_HELD_PROXY_SCORED_ONLY"
    reg0 = next(
        item
        for item in result["state_scores"]
        if item["registration_id"] == "REG0"
    )
    assert reg0["N_seen_new"] is None
    assert reg0["H_old_new"] is None
    assert reg0["registration_metric_status"] == "NA_BEFORE_REGISTRATION"
    assert "by_registration" in result["causal_comparisons_by_k"]["5"]


def test_forbidden_truth_and_incomplete_rows_fail_before_scoring() -> None:
    plan, prediction, truth = _fixture()
    bad = deepcopy(prediction)
    bad["query_truth"] = "must-not-be-here"
    with pytest.raises(scorer.NextR3ScoreError, match="forbidden"):
        scorer.score_next_r3_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)
    incomplete = deepcopy(prediction)
    incomplete["rows"] = incomplete["rows"][:-1]
    with pytest.raises(scorer.NextR3ScoreError, match="24"):
        scorer.score_next_r3_proxy24(
            prediction=incomplete, plan=plan, truth_by_query_id=truth
        )


def test_truth_join_requires_complete_prediction_closure() -> None:
    plan, prediction, truth = _fixture()
    truth.pop(next(iter(truth)))
    with pytest.raises(scorer.NextR3ScoreError, match="coverage"):
        scorer.score_next_r3_proxy24(
            prediction=prediction, plan=plan, truth_by_query_id=truth
        )
