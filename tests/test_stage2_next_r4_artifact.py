from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi import stage2_next_r4_artifact as artifact
from cvsrffi import stage2_next_r4_matrix as matrix
from cvsrffi import stage2_next_r4_score as scorer


CLASSES = ("tx-z", "tx-a", "tx-f", "tx-c", "tx-e", "tx-b")


def _isolation() -> dict[str, object]:
    return {
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "global_reassignment_calls": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "true_batch_class_count_access": False,
        "query_batch_dependency": False,
    }


def _binding(row_k1: dict, row_k5: dict) -> tuple[dict, dict[str, str]]:
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
    views = {"K1": query, "K5": query, **{state: query for state in matrix.STATE_IDS}}
    observation_views = {
        "K1": observation,
        "K5": observation,
        **{state: observation for state in matrix.STATE_IDS},
    }
    receipt = dict(
        matrix.bind_next_r4_physical_ids(
            row_k1=matrix.outer_key_from_mapping(row_k1),
            row_k5=matrix.outer_key_from_mapping(row_k5),
            phase1_fit_ids=[
                f"phase1-{row_k1['held_receiver']}-{row_k1['held_class']}-{i}" for i in range(3)
            ],
            k1_support_ids_by_class=support1,
            k5_support_ids_by_class=support5,
            query_ids_by_class=query,
            query_observation_ids_by_class=observation,
            query_ids_by_view=views,
            query_observation_ids_by_view=observation_views,
        )
    )
    truth = {
        query_id: class_id
        for class_id, query_ids in query.items()
        for query_id in query_ids
    }
    return receipt, truth


def _row_results() -> tuple[dict, dict, dict]:
    plan = dict(matrix.build_next_r4_proxy24_plan(CLASSES))
    rows_by_key = {(row["held_receiver"], row["held_class"], row["active_k"]): row for row in plan["rows"]}
    bindings = {
        (receiver, held): _binding(
            rows_by_key[(receiver, held, 1)], rows_by_key[(receiver, held, 5)]
        )
        for receiver in matrix.HELD_RECEIVERS
        for held in sorted(CLASSES)
    }
    truth: dict[str, str] = {}
    rows: list[dict] = []
    for planned in plan["rows"]:
        receiver, held, active_k = planned["held_receiver"], planned["held_class"], int(planned["active_k"])
        binding, binding_truth = bindings[(receiver, held)]
        classes = tuple(planned["all_registered_classes"])
        query_ids = tuple(binding["query_physical_ids"])
        query_obs = tuple(binding["query_observation_ids"])
        truth.update(binding_truth)

        def prediction(registered: tuple[str, ...]) -> list[str]:
            return [truth[qid] if truth[qid] in registered else registered[0] for qid in query_ids]

        registrations: dict[str, dict] = {}
        for registration_id in matrix.REGISTRATION_IDS:
            registered = tuple(
                planned["retained_classes"] if registration_id == "REG0" else planned["all_registered_classes"]
            )
            states: dict[str, dict] = {}
            state_ids = ("DA0_REG0", "DA1_REG0") if registration_id == "REG0" else ("DA0_REG1", "DA1_REG1")
            for state_id in state_ids:
                q_values = prediction(registered)
                if active_k == 5:
                    h_values = list(q_values)
                    h_values[0] = registered[-1]
                    h_receipt = {
                        "exact_qknn_alias": False,
                        "unique_prediction": True,
                        "head_status": "FUNCTIONAL",
                    }
                else:
                    h_values = list(q_values)
                    h_receipt = {
                        "exact_qknn_alias": True,
                        "alias_target_arm": "Q",
                        "unique_prediction": False,
                    }
                states[state_id] = {
                    "state_id": state_id,
                    "state_name_zh": matrix.STATE_NAMES_ZH[state_id],
                    "registered_classes": list(registered),
                    "query_physical_ids": list(query_ids),
                    "query_observation_ids": list(query_obs),
                    "arms": {
                        "Q": {
                            "predictions": q_values,
                            "receipt": {"exact_qknn_alias": False, "unique_prediction": True},
                        },
                        "H": {"predictions": h_values, "receipt": h_receipt},
                    },
                }
            registrations[registration_id] = {"registered_classes": list(registered), "states": states}
        rows.append(
            {
                "row_id": planned["row_id"],
                "held_receiver": receiver,
                "held_class": held,
                "active_k": active_k,
                "binding_receipt": binding,
                "fa_state_reuse_receipt": dict(
                    matrix.validate_fa_state_reuse(
                        {
                            "DA1_REG0": ("a" if active_k == 1 else "b") * 64,
                            "DA1_REG1": ("a" if active_k == 1 else "b") * 64,
                        }
                    )
                ),
                "registrations": registrations,
                "resource_receipt": {"schema": "test.resource.v1", "state_bytes": 6},
                "query_isolation_receipt": _isolation(),
            }
        )
    return plan, rows, truth


def test_builder_emits_complete_truth_free_artifact_and_scores() -> None:
    plan, rows, truth = _row_results()
    prediction = artifact.build_next_r4_prediction_artifact(plan=plan, row_results=rows)
    assert prediction["schema"] == artifact.PREDICTION_SCHEMA
    assert prediction["truth_loaded"] is False
    assert prediction["row_count"] == 24
    assert prediction["unique_prediction_count"] == 144
    assert prediction["artifact_arm_count"] == 192
    assert all(
        row["query_isolation_receipt"]["query_truth_access"] is False
        and "truth" not in row
        and "query_truth" not in row
        for row in prediction["rows"]
    )
    forbidden = {
        "query_ids_by_class", "query_observation_ids_by_class", "query_count_by_class"
    }
    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)
    assert forbidden.isdisjoint(keys(prediction))
    scored = scorer.score_next_r4_proxy24(prediction=prediction, plan=plan, truth_by_query_id=truth)
    assert scored["row_count"] == 24
    assert scored["unique_prediction_count"] == 144
    assert scored["artifact_arm_count"] == 192


def test_builder_rejects_forbidden_logits_and_truth_before_projection() -> None:
    plan, rows, _ = _row_results()
    bad = deepcopy(rows)
    bad[0]["resource_receipt"] = {"logits": [1.0]}
    with pytest.raises(artifact.NextR4ArtifactError, match="forbidden"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)
    bad = deepcopy(rows)
    bad[0]["registrations"]["REG0"]["states"]["DA0_REG0"]["arms"]["Q"]["receipt"]["truth"] = "tx-a"
    with pytest.raises(artifact.NextR4ArtifactError, match="forbidden"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)


def test_builder_rejects_old_class_grouped_binding_recursively() -> None:
    plan, rows, _ = _row_results()
    bad = deepcopy(rows)
    binding = bad[0]["binding_receipt"]
    binding["schema"] = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.row_binding.v1"
    binding["query_ids_by_class"] = {"leaked-class": binding["query_physical_ids"]}
    binding["query_count_by_class"] = {"leaked-class": binding["query_count"]}
    binding.pop("binding_sha256")
    binding["binding_sha256"] = matrix.canonical_sha256(binding)
    with pytest.raises(artifact.NextR4ArtifactError, match="forbidden"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)


def test_builder_rejects_incomplete_rows_and_query_updates() -> None:
    plan, rows, _ = _row_results()
    with pytest.raises(artifact.NextR4ArtifactError, match="24 rows"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=rows[:-1])
    bad = deepcopy(rows)
    bad[0]["query_isolation_receipt"]["query_state_updates"] = 1
    with pytest.raises(artifact.NextR4ArtifactError, match="must be zero"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)
    bad = deepcopy(rows)
    bad[0]["query_isolation_receipt"]["global_reassignment_calls"] = 1
    with pytest.raises(artifact.NextR4ArtifactError, match="must be zero"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)
    bad = deepcopy(rows)
    del bad[0]["query_isolation_receipt"]["query_truth_access"]
    with pytest.raises(artifact.NextR4ArtifactError, match="is required"):
        artifact.build_next_r4_prediction_artifact(plan=plan, row_results=bad)


def test_builder_accepts_k5_no_head_function_alias_receipt() -> None:
    plan, rows, _ = _row_results()
    row = next(item for item in rows if int(item["active_k"]) == 5)
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
    prediction = artifact.build_next_r4_prediction_artifact(plan=plan, row_results=rows)
    assert prediction["rows"][rows.index(row)]["active_k"] == 5
