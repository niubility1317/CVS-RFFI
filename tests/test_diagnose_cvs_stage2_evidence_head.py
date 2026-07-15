from __future__ import annotations

import numpy as np

from paper_reproduction.scripts.diagnose_cvs_stage2_evidence_head import (
    build_evidence_head_config,
    summarize_predictions,
)


def _base_head() -> dict[str, object]:
    return {
        "schema": "cvs.phase2.symmetric_locked_head.v1",
        "mode": "three_leo_support_symmetric_locked",
        "selected": {
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
        },
        "source_feature_mean": [0.0, 0.0],
        "source_feature_std": [1.0, 1.0],
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }


def test_build_evidence_head_config_is_role_free_and_locked() -> None:
    result = build_evidence_head_config(
        _base_head(),
        negative_quantile=0.95,
        prior_physical_shots=8.0,
        scale_floor=0.05,
        inverse_scale_cap=10.0,
    )
    assert result["schema"] == "cvs.phase2.symmetric_evidence_head.v2"
    encoded = repr(result).lower()
    assert "query" not in encoded
    assert "old_class" not in encoded
    assert "new_class" not in encoded
    assert "quota" not in encoded


def test_summary_keeps_before_after_new_floor_and_forgetting_on_same_rows() -> None:
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    tokens = []
    roles = []
    truths = []
    names = []
    for scenario_index, scenario in enumerate(scenarios):
        for local, (role, truth, name) in enumerate(
            (("target_old", 0, "old0"), ("target_old", 1, "old1"), ("target_new", 2, "new2"))
        ):
            tokens.append(f"qid_{scenario_index}_{local}")
            roles.append(role)
            truths.append(truth)
            names.append(name)
    candidate_before = np.asarray([0, 0, 0] * 3, dtype=np.int64)
    candidate_after = np.asarray([0, 1, 2, 0, 2, 2, 0, 1, 1], dtype=np.int64)
    arrays = {
        "query_tokens": np.asarray(tokens),
        "scenarios": np.repeat(np.asarray(scenarios), 3),
        "candidate_after": candidate_after,
        "candidate_before": candidate_before,
        "identity_after": candidate_after,
        "identity_before": candidate_before,
        "direct": np.asarray([0, 1, 0] * 3, dtype=np.int64),
        "shared_view_counts": np.asarray([1, 3, 5] * 3, dtype=np.int64),
    }
    truth_rows = [
        {
            "query_token": token,
            "evaluation_role": role,
            "true_class_index": truth,
            "transmitter_label": name,
        }
        for token, role, truth, name in zip(tokens, roles, truths, names)
    ]
    result = summarize_predictions(arrays, truth_rows, old_class_count=2)

    assert len(result["scenario_rows"]) == 3
    assert len(result["per_old_class"]) == 6
    assert result["aggregate"]["seen_new_acc"] == 2.0 / 3.0
    assert result["aggregate"]["old_acc_before_increment"] == 0.5
    assert result["aggregate"]["old_acc_after_increment"] == 5.0 / 6.0
    assert result["aggregate"]["average_forgetting"] == -1.0 / 3.0
    assert result["aggregate"]["global_min_old_class_acc"] == 0.0
