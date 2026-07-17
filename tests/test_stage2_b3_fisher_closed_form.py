from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_b3_fisher_closed_form import (
    B3FisherClosedFormError,
    MAX_ACTIVE_SCALARS,
    fit_b3_fisher_closed_form,
    score_b3_fisher_closed_form,
)


def _support(class_count: int, k_shot: int, seed: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    classes = tuple(f"old_{index}" for index in range(class_count))
    centers = rng.normal(size=(class_count, 288)).astype(np.float32)
    centers[:, 32:] *= np.float32(0.08)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, class_name in enumerate(classes):
        for _ in range(k_shot):
            row = centers[index] + rng.normal(scale=0.12, size=288).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row)
            labels.append(class_name)
    return np.stack(rows), np.asarray(labels), classes


def test_six_class_k10_is_deterministic_bounded_and_lower_mac_than_adam15() -> None:
    x, y, classes = _support(6, 10, 1)
    first = fit_b3_fisher_closed_form(x, y, classes)
    second = fit_b3_fisher_closed_form(x, y, classes)

    assert first.state.active_scalars == MAX_ACTIVE_SCALARS == 2016
    assert first.state.optimizer_steps == 0
    assert first.state.log_diag.tobytes() == second.state.log_diag.tobytes()
    assert first.state.weights.tobytes() == second.state.weights.tobytes()
    assert not first.state.log_diag.flags.writeable
    assert not first.state.weights.flags.writeable
    assert sum(bool(row["selected"]) for row in first.solver_trace) == 1
    audit = first.resource_audit
    assert audit["adaptation_mode"] == "EVAL_ONLY_CLOSED_FORM_ADAPTATION"
    assert audit["active_scalar_cap_pass"] is True
    assert audit["estimated_adaptation_macs"] < audit["estimated_adam15_reference_macs"]
    assert audit["estimated_macs_reduction_fraction_vs_adam15"] > 0.80
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_features_used_for_fit"] is False
    assert audit["query_role_oracle_access"] is False
    assert audit["query_class_quota_access"] is False
    assert audit["clean_sample_access"] is False
    assert audit["source_sample_access"] is False
    assert audit["dense_query_graph_bytes"] == 0


def test_k1_has_identity_diagonal_and_no_pseudo_loso() -> None:
    x, y, classes = _support(4, 1, 2)
    result = fit_b3_fisher_closed_form(x, y, classes)

    np.testing.assert_array_equal(result.state.log_diag, np.zeros(288, dtype=np.float32))
    assert result.state.selected_strength == 0.0
    assert len(result.solver_trace) == 1
    assert result.solver_trace[0]["solver"] == "k1_identity_no_loso"
    assert result.solver_trace[0]["query_rows_used"] == 0


def test_diagonal_is_block_centered_capped_and_scores_are_readonly() -> None:
    x, y, classes = _support(3, 5, 3)
    result = fit_b3_fisher_closed_form(x, y, classes)
    diag = result.state.log_diag

    assert abs(float(np.mean(diag[:160]))) < 2.0e-6
    assert abs(float(np.mean(diag[160:256]))) < 2.0e-6
    assert abs(float(np.mean(diag[256:]))) < 2.0e-6
    assert float(np.max(np.abs(diag))) <= np.log(1.5) + 1.0e-6
    scores = score_b3_fisher_closed_form(result.state, x)
    assert scores.shape == (15, 3)
    assert scores.dtype == np.float32
    assert not scores.flags.writeable


def test_rejects_more_than_six_classes_and_unbalanced_support() -> None:
    x, y, classes = _support(7, 2, 4)
    with pytest.raises(B3FisherClosedFormError, match=">6"):
        fit_b3_fisher_closed_form(x, y, classes)

    x2, y2, classes2 = _support(3, 2, 5)
    with pytest.raises(B3FisherClosedFormError, match="class-symmetric"):
        fit_b3_fisher_closed_form(x2[:-1], y2[:-1], classes2)
