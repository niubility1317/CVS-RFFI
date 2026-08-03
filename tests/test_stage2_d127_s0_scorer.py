from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi import stage2_d127_s0_scorer as scorer


def _sign(document: dict, field: str) -> dict:
    unsigned = dict(document)
    unsigned.pop(field, None)
    document[field] = scorer.canonical_sha256(unsigned)
    return document


def _arm(arm_id: str, classes: list[str], predictions: list[str]) -> dict:
    return {"arm_id": arm_id, "classes": classes, "predictions": predictions}


def _row(index: int, *, state: str, receiver: str, k_shot: int, scene: str) -> dict:
    old = ["old0", "old1"]
    registered = old if state == "before" else [*old, "new0"]
    after_ids = [f"q-{index}-old0", f"q-{index}-old1", f"q-{index}-new0"]
    query_ids = after_ids[:2] if state == "before" else after_ids
    if state == "before":
        common_predictions = ["old0", "old1"]
        adapted_predictions = ["old0", "old1"]
        joint_predictions = ["old0", "old1"]
    else:
        common_predictions = ["old0", "old0", "old0"]
        adapted_predictions = ["old0", "old1", "new0"] if k_shot == 1 else ["old0", "old1", "old0"]
        joint_predictions = ["old0", "old1", "new0"]
    row = {
        "row_id": f"{receiver}.k{k_shot}.{scene}", "receiver_id": receiver, "k_shot": k_shot, "scene": scene,
        "opaque_query_ids": query_ids,
        "common_arms": {
            "M0": _arm("M0", registered, common_predictions),
            "M_L92": _arm("M_L92", registered, common_predictions),
        },
        "candidates": {
            candidate: {
                "arms": {
                    "M_DA": _arm("M_DA", registered, adapted_predictions),
                    "M_JOINT": _arm("M_JOINT", registered, joint_predictions),
                }
            }
            for candidate in scorer.CANDIDATE_IDS
        },
    }
    return _sign(row, "row_sha256")


def _payload(state: str, rows: list[dict]) -> dict:
    payload = {
        "schema": scorer.PREDICTION_SCHEMA, "candidate_ids": list(scorer.CANDIDATE_IDS),
        "truth_loaded": False, "row_count": 18, "rows_complete": True,
        "query_rows_used_for_fit": 0, "query_state_updates": 0, "query_selection_count": 0,
        "rows": rows,
    }
    return _sign(payload, "prediction_sha256")


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    before_rows: list[dict] = []
    after_rows: list[dict] = []
    manifest_rows: list[dict] = []
    truth_queries: list[dict] = []
    index = 0
    for receiver in ("r0", "r1", "r2"):
        for k_shot in (1, 5):
            for scene in ("s0", "s1", "s2"):
                before = _row(index, state="before", receiver=receiver, k_shot=k_shot, scene=scene)
                after = _row(index, state="after", receiver=receiver, k_shot=k_shot, scene=scene)
                before_rows.append(before)
                after_rows.append(after)
                formal_hash = f"{index + 1:064x}"
                manifest_rows.append(
                    {
                        "row_id": before["row_id"], "receiver_id": receiver, "k_shot": k_shot, "scene": scene,
                        "old_classes": ["old0", "old1"], "new_classes": ["new0"],
                        "before_query_ids_sha256": scorer.canonical_sha256(before["opaque_query_ids"]),
                        "after_query_ids_sha256": scorer.canonical_sha256(after["opaque_query_ids"]),
                        "formal_d92_row_key": f"formal-{before['row_id']}", "formal_d92_score_row_sha256": formal_hash,
                    }
                )
                truth_queries.extend(
                    [
                        {"opaque_query_id": after["opaque_query_ids"][0], "label": "old0", "role": "old"},
                        {"opaque_query_id": after["opaque_query_ids"][1], "label": "old1", "role": "old"},
                        {"opaque_query_id": after["opaque_query_ids"][2], "label": "new0", "role": "new"},
                    ]
                )
                index += 1
    before_payload = _payload("before", before_rows)
    after_payload = _payload("after", after_rows)
    pair = {
        "schema": scorer.PAIR_MANIFEST_SCHEMA, "pair_id": "d127-s0-fixture", "protocol_schema": scorer.PROTOCOL_SCHEMA,
        "truth_open": False, "method_lock_sha256": "a" * 64, "capsule_id": "capsule-fixture", "split_id": "split-fixture",
        "query_id_root_sha256": scorer.canonical_sha256(sorted(query["opaque_query_id"] for query in truth_queries)),
        "candidate_ids": list(scorer.CANDIDATE_IDS), "arm_ids": list(scorer.ARM_IDS), "row_count": 18,
        "before_prediction_sha256": before_payload["prediction_sha256"], "after_prediction_sha256": after_payload["prediction_sha256"],
        "rows": manifest_rows,
    }
    _sign(pair, "pair_manifest_sha256")
    truth = {
        "schema": scorer.TRUTH_CATALOG_SCHEMA, "truth_open": True, "pair_manifest_sha256": pair["pair_manifest_sha256"],
        "query_count": len(truth_queries), "queries": truth_queries,
    }
    _sign(truth, "truth_catalog_sha256")
    formal = {
        "schema": scorer.FORMAL_D92_REFERENCE_SCHEMA, "pair_manifest_sha256": pair["pair_manifest_sha256"],
        "pipeline_receipt_sha256": "b" * 64, "row_count": 18,
        "rows": [
            {key: row[key] for key in ("row_id", "receiver_id", "k_shot", "scene", "formal_d92_row_key", "formal_d92_score_row_sha256")}
            for row in manifest_rows
        ],
    }
    _sign(formal, "formal_d92_reference_sha256")
    return before_payload, after_payload, pair, truth, formal


def test_complete_before_after_s0_closure_scores_all_same_rows() -> None:
    before, after, pair, truth, formal = _fixture()
    validated = scorer.validate_d127_s0_prediction_pairs(
        before_prediction=before, after_prediction=after, pair_manifest=pair, expected_method_lock_sha256="a" * 64
    )
    assert validated["pair_manifest_sha256"] == pair["pair_manifest_sha256"]
    result = scorer.score_d127_s0(
        before_prediction=before, after_prediction=after, pair_manifest=pair,
        truth_catalog=truth, formal_d92_reference=formal, expected_method_lock_sha256="a" * 64,
    )
    assert result["row_count"] == 18 and result["metric_row_count"] == 18 * 3 * 4
    assert len(result["same_row_results"]) == 18
    assert {row["group_value"] for row in result["aggregates"]["k_shot"]} == {1, 5}
    assert all(item["all_three_direction_pass"] for item in result["s0_direction_decisions"])
    metric = result["same_row_results"][0]["candidate_arm_metrics"][0]
    assert {"B_old", "A_old", "seen_new", "H_old_new", "old_per_class_floor", "forgetting", "total_correct_count"}.issubset(metric)


def test_truth_cannot_be_present_before_prediction_closure() -> None:
    before, after, pair, _, _ = _fixture()
    before["truth_loaded"] = True
    _sign(before, "prediction_sha256")
    pair["before_prediction_sha256"] = before["prediction_sha256"]
    _sign(pair, "pair_manifest_sha256")
    with pytest.raises(scorer.D127S0ScorerError, match="truth-closed"):
        scorer.validate_d127_s0_prediction_pairs(before_prediction=before, after_prediction=after, pair_manifest=pair)


def test_missing_after_query_fails_pair_receipt() -> None:
    before, after, pair, _, _ = _fixture()
    after["rows"][0]["opaque_query_ids"].pop()
    _sign(after["rows"][0], "row_sha256")
    _sign(after, "prediction_sha256")
    pair["after_prediction_sha256"] = after["prediction_sha256"]
    _sign(pair, "pair_manifest_sha256")
    with pytest.raises(scorer.D127S0ScorerError, match="prediction/query length|after query receipt"):
        scorer.validate_d127_s0_prediction_pairs(before_prediction=before, after_prediction=after, pair_manifest=pair)


def test_truth_role_drift_is_rejected_without_after_only_forgetting() -> None:
    before, after, pair, truth, formal = _fixture()
    truth["queries"][0]["role"] = "new"
    _sign(truth, "truth_catalog_sha256")
    with pytest.raises(scorer.D127S0ScorerError, match="before/after old query IDs|label/role"):
        scorer.score_d127_s0(before_prediction=before, after_prediction=after, pair_manifest=pair, truth_catalog=truth, formal_d92_reference=formal)


def test_formal_d92_same_row_key_or_hash_drift_is_rejected() -> None:
    before, after, pair, truth, formal = _fixture()
    formal["rows"][0]["formal_d92_row_key"] = "wrong-formal-key"
    _sign(formal, "formal_d92_reference_sha256")
    with pytest.raises(scorer.D127S0ScorerError, match="formal D92 same-row key/hash drift"):
        scorer.score_d127_s0(before_prediction=before, after_prediction=after, pair_manifest=pair, truth_catalog=truth, formal_d92_reference=formal)
