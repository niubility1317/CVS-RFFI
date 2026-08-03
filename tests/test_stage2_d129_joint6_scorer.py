from __future__ import annotations

import copy
import hashlib

import pytest

from cvsrffi import stage2_d129_joint6_matrix as matrix
from cvsrffi import stage2_d129_joint6_scorer as scorer


def _fixture():
    plan = matrix.build_joint6_loco_plan(
        [f"rx{i}" for i in range(7)], [f"tx{i}" for i in range(6)]
    )
    truth = {}
    rows = []
    for candidate_id in matrix.CANDIDATE_IDS:
        for row in plan["rows"]:
            classes = row["registered_classes"]
            query_ids = [
                f"{row['row_id']}|q={class_id}|n={index}"
                for class_id in classes
                for index in range(9)
            ]
            for class_id in classes:
                for index in range(9):
                    truth[f"{row['row_id']}|q={class_id}|n={index}"] = class_id
            baseline = [class_id for class_id in classes for _ in range(9)]
            if row["active_k"] == 5:
                # Synthetic closure: every treatment repairs one distinct error
                # without creating a loss, so all three frozen contrasts pass.
                r0q = baseline.copy()
                r0f = baseline.copy()
                r0l = baseline.copy()
                r1q = baseline.copy()
                r1f = baseline.copy()
                r1l = baseline.copy()
                r0q[0] = classes[1]
                r0f[1] = classes[2]
                r1f[2] = classes[3]
            else:
                r0q = baseline.copy()
                r0f = r0q
                r0l = r0q
                r1q = baseline.copy()
                r1f = r1q
                r1l = r1q
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "row_id": row["row_id"],
                    "held_receiver": row["held_receiver"],
                    "held_class": row["held_class"],
                    "active_k": row["active_k"],
                    "registered_classes": row["registered_classes"],
                    "opaque_query_ids": query_ids,
                    "binding_sha256": hashlib.sha256(
                        f"binding|{row['row_id']}".encode()
                    ).hexdigest(),
                    "phase1_seal_sha256": hashlib.sha256(
                        f"seal|{row['row_id']}".encode()
                    ).hexdigest(),
                    "query_physical_root_sha256": hashlib.sha256(
                        "\n".join(query_ids).encode()
                    ).hexdigest(),
                    "checkpoint_sha256": "a" * 64,
                    "asset_sha256": hashlib.sha256(
                        f"asset|{candidate_id}|{row['row_id']}".encode()
                    ).hexdigest(),
                    "common_r0_sha256": hashlib.sha256(
                        f"common|{row['row_id']}".encode()
                    ).hexdigest(),
                    "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
                    "formal_new_registration_claim": False,
                    "arms": {
                        "R0Q": r0q,
                        "R0F": r0f,
                        "R0L": r0l,
                        "R1Q": r1q,
                        "R1F": r1f,
                        "R1L": r1l,
                    },
                }
            )
    prediction = {
        "schema": scorer.PREDICTION_SCHEMA,
        "matrix_sha256": plan["matrix_sha256"],
        "protocol_schema": "p2_min_v1",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_sha256": "a" * 64,
        "archive_sha256": "b" * 64,
        "method_lock_sha256": "c" * 64,
        "query_catalog_root_sha256": "d" * 64,
        "candidate_ids": list(matrix.CANDIDATE_IDS),
        "arm_ids": list(matrix.ARM_IDS),
        "rows_complete": True,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "rows": rows,
    }
    return plan, prediction, truth


def test_complete_matrix_scores_three_frozen_k5_comparisons() -> None:
    plan, prediction, truth = _fixture()
    result = scorer.score_joint6_screen(
        prediction=prediction, plan=plan, truth_by_query_id=truth
    )
    assert result["truth_opened_after_complete_prediction"] is True
    assert result["partial_performance_selection_used"] is False
    assert result["formal_new_registration_claim"] is False
    assert result["formal_H_old_new_emitted"] is False
    for candidate_id in matrix.CANDIDATE_IDS:
        candidate = result["candidate_scores"][candidate_id]
        assert candidate["candidate_pass"] is True
        assert candidate["k1_head_gain_claim_allowed"] is False
        assert set(candidate["k5_primary_comparisons"]) == {
            "DA_EFFECT",
            "LITE_BASE",
            "JOINT_REPLACE",
        }
        assert all(
            value["pass"] for value in candidate["k5_primary_comparisons"].values()
        )


def test_k1_non_alias_and_forbidden_truth_field_fail_closed() -> None:
    plan, prediction, truth = _fixture()
    broken = copy.deepcopy(prediction)
    broken["rows"][0]["arms"]["R0L"] = list(
        broken["rows"][0]["arms"]["R0L"]
    )
    broken["rows"][0]["arms"]["R0L"][0] = "tx5"
    with pytest.raises(scorer.D129Joint6ScorerError, match="K1 F/L"):
        scorer.score_joint6_screen(
            prediction=broken, plan=plan, truth_by_query_id=truth
        )
    leaked = copy.deepcopy(prediction)
    leaked["rows"][0]["query_role"] = "new"
    with pytest.raises(scorer.D129Joint6ScorerError, match="forbidden"):
        scorer.score_joint6_screen(
            prediction=leaked, plan=plan, truth_by_query_id=truth
        )


def test_incomplete_rows_and_common_arm_drift_fail_closed() -> None:
    plan, prediction, truth = _fixture()
    incomplete = copy.deepcopy(prediction)
    incomplete["rows"].pop()
    with pytest.raises(scorer.D129Joint6ScorerError, match="row coverage"):
        scorer.score_joint6_screen(
            prediction=incomplete, plan=plan, truth_by_query_id=truth
        )
    drift = copy.deepcopy(prediction)
    second_candidate = matrix.ROW_COUNT_PER_CANDIDATE
    drift["rows"][second_candidate]["arms"]["R0Q"][0] = "tx5"
    with pytest.raises(scorer.D129Joint6ScorerError, match="common R0"):
        scorer.score_joint6_screen(
            prediction=drift, plan=plan, truth_by_query_id=truth
        )


def test_any_main_effect_loss_rejects_candidate_without_micro_gate() -> None:
    plan, prediction, truth = _fixture()
    broken = copy.deepcopy(prediction)
    for row in broken["rows"][: matrix.ROW_COUNT_PER_CANDIDATE]:
        if row["active_k"] == 5:
            row["arms"]["R1Q"] = list(row["arms"]["R0Q"])
    result = scorer.score_joint6_screen(
        prediction=broken, plan=plan, truth_by_query_id=truth
    )
    candidate = result["candidate_scores"][matrix.CANDIDATE_IDS[0]]
    assert candidate["candidate_pass"] is False
    assert candidate["k5_primary_comparisons"]["DA_EFFECT"]["pass"] is False
    assert (
        candidate["k5_primary_comparisons"]["DA_EFFECT"][
            "delta_H_retained_held_proxy"
        ]
        == 0.0
    )
