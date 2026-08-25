from __future__ import annotations

import pytest

from scripts import score_m29_tasr48_matrix as scorer
from scripts import score_m24_safe_residual_suite as shared_scorer


def test_m29_scorer_behavior_normalizes_only_known_legacy_weight_typo() -> None:
    behavior = {
        "schema": "behavior",
        "fallback_counts": {},
        "full_block_weights": {"full": 1.0, "block3": 1.0},
        "fisher_gate_accept_counts": {"attempted": 0, "accepted": 0},
        "atomic_rollback_counts": {"attempted": 0, "rolled_back": 0},
        "failure_closure_count": 0,
    }
    normalized = scorer._m29_scorer_behavior(behavior)
    assert normalized["full_block_weights"] == {"full": 1.0, "block3": 0.0}
    assert behavior["full_block_weights"] == {"full": 1.0, "block3": 1.0}

    changed = {**behavior, "full_block_weights": {"full": 0.25, "block3": 0.25}}
    with pytest.raises(ValueError, match="weight receipt drift"):
        scorer._m29_scorer_behavior(changed)


def test_m29_scorer_resource_projects_extended_auxiliary_scope() -> None:
    resource = {key: 0 for key in shared_scorer._SCORER_RESOURCE_KEYS}
    resource.update(
        {
            "schema": "resource",
            "candidate_peak_memory_isolated": False,
            "end_to_end_query_latency_available": False,
            "end_to_end_query_latency_ms": None,
            "batch1_head_resource": None,
            "auxiliary_state_cost_in_candidate_resource": True,
            "auxiliary_prediction_cost_in_candidate_latency": True,
        }
    )
    _quantization, projected = scorer._m29_scorer_receipts(
        {
            "schema": "quantization",
            "max_logit_abs_error": 0.0,
            "mean_logit_abs_error": 0.0,
            "argmax_flip_rate": 0.0,
            "prediction_agreement_rate": 1.0,
        },
        resource,
    )
    assert projected["auxiliary_state_cost_in_candidate_resource"] is False
    assert projected["auxiliary_prediction_cost_in_candidate_latency"] is False
    assert resource["auxiliary_state_cost_in_candidate_resource"] is True

    changed = dict(resource)
    changed["auxiliary_prediction_cost_in_candidate_latency"] = False
    with pytest.raises(ValueError, match="auxiliary resource scope drift"):
        scorer._m29_scorer_receipts({}, changed)
