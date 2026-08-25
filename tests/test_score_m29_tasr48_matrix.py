from __future__ import annotations

import pytest

from scripts import score_m29_tasr48_matrix as scorer


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
