from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterator, Mapping

import pytest

from cvsrffi import stage2_next_r4_matrix as matrix
from cvsrffi import stage2_next_r4_score as scorer


CLASSES = ("tx-z", "tx-a", "tx-f", "tx-c", "tx-e", "tx-b")


def _binding(plan: Mapping[str, object], row_k1: Mapping[str, object], row_k5: Mapping[str, object]):
    classes = tuple(sorted(CLASSES))
    support5 = {
        cls: [f"support-{row_k1['held_receiver']}-{row_k1['held_class']}-{cls}-{i}" for i in range(5)]
        for cls in classes
    }
    support1 = {cls: values[:1] for cls, values in support5.items()}
    query = {
        cls: [f"query-{row_k1['held_receiver']}-{row_k1['held_class']}-{cls}-{i}" for i in range(2)]
        for cls in classes
    }
    observation = {
        cls: [f"observation-{row_k1['held_receiver']}-{row_k1['held_class']}-{cls}-{i}" for i in range(2)]
        for cls in classes
    }
    phase1 = [f"phase1-{row_k1['held_receiver']}-{row_k1['held_class']}-{i}" for i in range(3)]
    view_ids = {"K1": query, "K5": query, **{state: query for state in matrix.STATE_IDS}}
    view_obs = {"K1": observation, "K5": observation, **{state: observation for state in matrix.STATE_IDS}}
    receipt = matrix.bind_next_r4_physical_ids(
        row_k1=matrix.outer_key_from_mapping(row_k1),
        row_k5=matrix.outer_key_from_mapping(row_k5),
        phase1_fit_ids=phase1,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
        query_observation_ids_by_class=observation,
        query_ids_by_view=view_ids,
        query_observation_ids_by_view=view_obs,
    )
    return dict(receipt)


def _prediction_fixture() -> tuple[dict, dict, dict]:
    plan = dict(matrix.build_next_r4_proxy24_plan(CLASSES))
    rows_by_key = {(r["held_receiver"], r["held_class"], r["active_k"]): r for r in plan["rows"]}
    bindings = {}
    for receiver in matrix.HELD_RECEIVERS:
        for held in sorted(CLASSES):
            bindings[(receiver, held)] = _binding(
                plan,
                rows_by_key[(receiver, held, 1)],
                rows_by_key[(receiver, held, 5)],
            )

    truth: dict[str, str] = {}
    rows: list[dict] = []
    for planned in plan["rows"]:
        receiver, held, active_k = planned["held_receiver"], planned["held_class"], planned["active_k"]
        binding = bindings[(receiver, held)]
        classes = tuple(planned["all_registered_classes"])
        query_ids = tuple(qid for cls in classes for qid in binding["query_ids_by_class"][cls])
        query_obs = tuple(oid for cls in classes for oid in binding["query_observation_ids_by_class"][cls])
        for cls in classes:
            for qid in binding["query_ids_by_class"][cls]:
                truth[qid] = cls

        def predictions(registered: tuple[str, ...], *, perturb: bool = False) -> list[str]:
            values = [truth[qid] if truth[qid] in registered else registered[0] for qid in query_ids]
            if perturb and values:
                values[0] = registered[-1]
            return values

        registrations: dict[str, dict] = {}
        for registration_id in matrix.REGISTRATION_IDS:
            registered = tuple(planned["retained_classes"] if registration_id == "REG0" else planned["all_registered_classes"])
            states: dict[str, dict] = {}
            for state_id in matrix.STATE_IDS:
                if matrix.registration_for_state(state_id) != registration_id:
                    continue
                q_values = predictions(registered, perturb=state_id == "DA1_REG0")
                h_values = list(q_values)
                if active_k == 5 and state_id == "DA1_REG1":
                    h_values[0] = registered[0]
                states[state_id] = {
                    "state_id": state_id,
                    "state_name_zh": matrix.STATE_NAMES_ZH[state_id],
                    "registered_classes": list(registered),
                    "query_physical_ids": list(query_ids),
                    "query_observation_ids": list(query_obs),
                    "arms": {
                        "Q": {"predictions": q_values, "receipt": {"exact_qknn_alias": False, "unique_prediction": True}},
                            "H": {
                                "predictions": h_values,
                                "receipt": {
                                    "exact_qknn_alias": active_k == 1,
                                    "alias_target_arm": "Q" if active_k == 1 else None,
                                    "unique_prediction": active_k == 5,
                                    "head_status": "K1_EXACT_QKNN_ALIAS" if active_k == 1 else "FUNCTIONAL",
                                },
                            },
                    },
                }
            registrations[registration_id] = {"registered_classes": list(registered), "states": states}
        rows.append({
            "row_id": planned["row_id"], "held_receiver": receiver, "held_class": held, "active_k": active_k,
            "retained_classes": list(planned["retained_classes"]), "all_registered_classes": list(planned["all_registered_classes"]),
            "evaluation_semantics": matrix.PROXY_SEMANTICS, "formal_new_registration_claim": False,
            "binding_receipt": binding,
            "fa_state_reuse_receipt": dict(
                matrix.validate_fa_state_reuse(
                    {
                        "DA1_REG0": ("a" if int(planned["active_k"]) == 1 else "b") * 64,
                        "DA1_REG1": ("a" if int(planned["active_k"]) == 1 else "b") * 64,
                    }
                )
            ),
            "query_isolation_receipt": {
                "query_rows_used_for_fit": 0,
                "query_state_updates": 0,
                "query_selection_count": 0,
                "global_reassignment_calls": 0,
                "query_truth_access": False,
                "query_role_access": False,
                "class_quota_access": False,
                "true_batch_class_count_access": False,
                "query_batch_dependency": False,
            },
            "registrations": registrations,
        })
    prediction = {
        "schema": scorer.PREDICTION_SCHEMA, "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA, "matrix_sha256": plan["matrix_sha256"],
        "evaluation_semantics": matrix.PROXY_SEMANTICS, "formal_new_registration_claim": False,
        "truth_loaded": False, "rows_complete": True,
        "query_rows_used_for_fit": 0, "query_state_updates": 0, "query_selection_count": 0,
        "rows": rows,
    }
    return plan, prediction, truth


def test_complete_24_rows_four_states_and_explicit_na_metrics() -> None:
    plan, prediction, truth = _prediction_fixture()
    result = scorer.score_next_r4_proxy24(prediction=prediction, plan=plan, truth_by_query_id=truth)
    assert result["row_count"] == 24
    assert result["unique_prediction_count"] == 144
    assert result["artifact_arm_count"] == 192
    assert result["cross_row_best_selection_used"] is False
    states = [state for row in result["row_scores"] for reg in row["registrations"].values() for state in reg["states"].values()]
    assert {state["state_name_zh"] for state in states} == set(matrix.STATE_NAMES_ZH.values())
    reg0_state = next(state for state in states if state["state_id"] == "DA0_REG0")
    metric = reg0_state["arms"]["Q"]
    assert metric["seen_new_acc"] == "N/A"
    assert metric["H_old_new"] == "N/A"
    assert metric["registration_metric_status"] == "NA_BEFORE_REGISTRATION"
    reg1_state = next(state for state in states if state["state_id"] == "DA1_REG1")
    assert isinstance(reg1_state["arms"]["Q"]["H_old_new"], float)
    assert "aggregates_by_receiver_k_state_and_arm" in result
    assert "(DA1_H-DA1_Q)-(DA0_H-DA0_Q)" in result["comparisons_by_k"]["5"]


def test_k1_alias_tamper_is_rejected() -> None:
    plan, prediction, truth = _prediction_fixture()
    bad = deepcopy(prediction)
    state = bad["rows"][0]["registrations"]["REG0"]["states"]["DA0_REG0"]
    state["arms"]["H"]["predictions"][0] = state["registered_classes"][0]
    state["arms"]["H"]["receipt"]["exact_qknn_alias"] = True
    state["arms"]["Q"]["predictions"][0] = state["registered_classes"][-1]
    with pytest.raises(scorer.NextR4ScoreError, match="exact Q alias"):
        scorer.score_next_r4_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)


def test_k5_no_head_function_exact_alias_is_accepted() -> None:
    plan, prediction, truth = _prediction_fixture()
    row = next(item for item in prediction["rows"] if int(item["active_k"]) == 5)
    state = row["registrations"]["REG1"]["states"]["DA1_REG1"]
    state["arms"]["H"] = {
        "predictions": list(state["arms"]["Q"]["predictions"]),
        "receipt": {
            "exact_qknn_alias": True,
            "alias_target_arm": "Q",
            "unique_prediction": False,
            "head_status": "NO_HEAD_FUNCTION",
            "no_head_function_reason": "Sr_ZERO",
        },
    }
    result = scorer.score_next_r4_proxy24(
        prediction=prediction, plan=plan, truth_by_query_id=truth
    )
    assert result["rows_complete"] is True


def test_truth_is_not_read_before_prediction_closure() -> None:
    plan, prediction, _ = _prediction_fixture()
    incomplete = deepcopy(prediction)
    incomplete["rows"] = incomplete["rows"][:-1]

    class Sentinel(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("truth was opened before prediction closure")

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("truth was opened before prediction closure")

        def __len__(self) -> int:
            raise AssertionError("truth was opened before prediction closure")

    with pytest.raises(scorer.NextR4ScoreError, match="24 rows"):
        scorer.score_next_r4_proxy24(prediction=incomplete, plan=plan, truth_by_query_id=Sentinel())


def test_fa_state_reuse_tamper_and_binding_tamper_fail_closed() -> None:
    plan, prediction, truth = _prediction_fixture()
    bad_fa = deepcopy(prediction)
    bad_fa["rows"][0]["fa_state_reuse_receipt"] = dict(bad_fa["rows"][0]["fa_state_reuse_receipt"])
    bad_fa["rows"][0]["fa_state_reuse_receipt"]["target_sha256"] = "b" * 64
    with pytest.raises(scorer.NextR4ScoreError, match="FA reuse"):
        scorer.score_next_r4_proxy24(prediction=bad_fa, plan=plan, truth_by_query_id=truth)
    bad_binding = deepcopy(prediction)
    bad_binding["rows"][0]["binding_receipt"]["query_ids_by_class"][sorted(CLASSES)[0]][0] = "query-tampered"
    with pytest.raises(scorer.NextR4ScoreError, match="binding"):
        scorer.score_next_r4_proxy24(prediction=bad_binding, plan=plan, truth_by_query_id=truth)


def test_scorer_revalidates_row_query_isolation_receipt() -> None:
    plan, prediction, truth = _prediction_fixture()
    bad = deepcopy(prediction)
    bad["rows"][0]["query_isolation_receipt"]["query_truth_access"] = True
    with pytest.raises(scorer.NextR4ScoreError, match="query_truth_access"):
        scorer.score_next_r4_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)
    bad = deepcopy(prediction)
    del bad["rows"][0]["query_isolation_receipt"]["query_role_access"]
    with pytest.raises(scorer.NextR4ScoreError, match="query_role_access"):
        scorer.score_next_r4_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)


def test_incomplete_rows_are_rejected_even_with_complete_truth() -> None:
    plan, prediction, truth = _prediction_fixture()
    bad = deepcopy(prediction)
    bad["rows"] = bad["rows"][:-1]
    with pytest.raises(scorer.NextR4ScoreError, match="24 rows"):
        scorer.score_next_r4_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)


def test_row_old_ba_is_macro_balanced_for_unequal_query_counts() -> None:
    row = matrix.build_next_r4_proxy24_plan(CLASSES)["rows"][0]
    old_classes = tuple(row["retained_classes"])
    query_ids = ("a0", "a1", "b0", "c0", "d0", "e0")
    truth = {
        "a0": old_classes[0],
        "a1": old_classes[0],
        "b0": old_classes[1],
        "c0": old_classes[2],
        "d0": old_classes[3],
        "e0": old_classes[4],
    }
    predictions = (
        old_classes[0],
        old_classes[0],
        old_classes[0],
        old_classes[2],
        old_classes[3],
        old_classes[4],
    )
    result = scorer._metric_row(
        row=row,
        registration_id="REG0",
        state_id="DA0_REG0",
        arm_id="Q",
        query_ids=query_ids,
        predictions=predictions,
        truth=truth,
    )
    assert result["old_ba"] == pytest.approx(0.8)
    assert result["retained_correct_count"] / result["retained_query_count"] == pytest.approx(5 / 6)


def test_aggregate_seen_new_is_held_class_macro_balanced() -> None:
    classes = tuple(sorted(CLASSES))
    metrics = []
    for index, held_class in enumerate(classes):
        retained = tuple(cls for cls in classes if cls != held_class)
        new_count = 10 if index == 0 else 1
        new_correct = new_count if index == 0 else 0
        per_class = {
            cls: {
                "correct": new_correct if cls == held_class else 1,
                "count": new_count if cls == held_class else 1,
                "accuracy": (new_correct / new_count) if cls == held_class else 1.0,
            }
            for cls in classes
        }
        metrics.append({
            "row_id": f"row-{index}", "held_receiver": "1-1", "held_class": held_class,
            "active_k": 1, "state_id": "DA0_REG1", "arm_id": "Q",
            "registration_id": "REG1", "all_registered_classes": classes,
            "registered_classes": classes, "retained_classes": retained,
            "retained_correct_count": len(retained), "retained_query_count": len(retained),
            "total_correct_count": len(retained) + new_correct,
            "total_query_count": len(retained) + new_count,
            "new_correct_count": new_correct, "new_query_count": new_count,
            "per_class": per_class,
        })
    result = scorer._aggregate(metrics, active_k=1, state_id="DA0_REG1", arm_id="Q", receiver="1-1")
    assert result["seen_new_acc"] == pytest.approx(1 / len(classes))
    assert result["new_correct_count"] / result["new_query_count"] == pytest.approx(10 / 15)


def test_forgetting_pairs_registration_before_after_at_fixed_da() -> None:
    metrics = [
        {"row_id": "r", "arm_id": "Q", "state_id": "DA0_REG0", "registration_id": "REG0", "old_ba": 0.90, "forgetting": "N/A"},
        {"row_id": "r", "arm_id": "Q", "state_id": "DA0_REG1", "registration_id": "REG1", "old_ba": 0.70, "forgetting": "N/A"},
        {"row_id": "r", "arm_id": "Q", "state_id": "DA1_REG0", "registration_id": "REG0", "old_ba": 0.85, "forgetting": "N/A"},
        {"row_id": "r", "arm_id": "Q", "state_id": "DA1_REG1", "registration_id": "REG1", "old_ba": 0.70, "forgetting": "N/A"},
    ]
    by_state = {item["state_id"]: item for item in scorer._attach_forgetting(metrics)}
    assert by_state["DA0_REG1"]["forgetting"] == pytest.approx(0.20)
    assert by_state["DA1_REG1"]["forgetting"] == pytest.approx(0.15)

    aggregate_block = {
        state_id: {
            arm_id: {"old_ba": value, "forgetting": "N/A"}
            for arm_id in ("Q", "H")
        }
        for state_id, value in {
            "DA0_REG0": 0.90, "DA0_REG1": 0.70,
            "DA1_REG0": 0.85, "DA1_REG1": 0.70,
        }.items()
    }
    aggregate = scorer._attach_aggregate_forgetting(aggregate_block)
    assert aggregate["DA0_REG1"]["Q"]["forgetting"] == pytest.approx(0.20)
    assert aggregate["DA1_REG1"]["H"]["forgetting"] == pytest.approx(0.15)
