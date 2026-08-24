from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_capta.prototype_transport import (
    A1_SUPPORT_SHRINK,
    A2_SHARED_SHIFT,
    A3_R4_SUPPORT_SHIFT,
    CaptaPrototypeError,
    fit_capta_prototypes,
)


def test_a1_uses_effective_support_count_for_spherical_shrinkage() -> None:
    source = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = np.asarray(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    state = fit_capta_prototypes(
        source,
        support,
        labels,
        candidate_id=A1_SUPPORT_SHRINK,
        rank=4,
        prior_strength=2.0,
    )

    np.testing.assert_allclose(
        state.target_prototypes[0],
        np.asarray([0.70710677, 0.70710677], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        state.target_prototypes[1],
        np.asarray([0.0, 1.0], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(state.effective_samples, [2.0, 2.0])
    assert state.audit["backward_count"] == 0
    assert state.audit["trainable_parameter_count"] == 0


def test_a2_estimates_one_class_balanced_shared_translation() -> None:
    source = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    support = np.asarray(
        [[0.8, 0.0, 0.6], [0.8, 0.0, 0.6], [0.0, 0.8, 0.6], [0.0, 0.8, 0.6]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    state = fit_capta_prototypes(
        source,
        support,
        labels,
        candidate_id=A2_SHARED_SHIFT,
        rank=4,
        prior_strength=2.0,
    )

    np.testing.assert_allclose(
        state.shared_shift,
        np.asarray([-0.1, -0.1, 0.6], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        state.transported_prototypes[0],
        np.asarray([0.82851714, -0.09205746, 0.55234477], dtype=np.float32),
        atol=1.0e-6,
    )
    assert state.audit["class_balanced_shift"] is True


def test_a3_rank_four_state_is_deterministic_and_inputs_stay_frozen() -> None:
    source = np.eye(6, 8, dtype=np.float32)
    residual = np.asarray(
        [
            [0.10, 0.00, 0.04, 0.00, 0.02, 0.00, 0.00, 0.00],
            [0.08, 0.02, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00],
            [0.09, 0.00, 0.02, 0.00, 0.00, 0.04, 0.00, 0.00],
            [0.11, 0.01, 0.00, 0.02, 0.00, 0.00, 0.03, 0.00],
            [0.10, 0.00, 0.03, 0.00, 0.01, 0.00, 0.00, 0.02],
            [0.07, 0.03, 0.00, 0.01, 0.00, 0.02, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    support = source + residual
    support /= np.linalg.norm(support, axis=1, keepdims=True)
    labels = np.arange(6, dtype=np.int64)
    source_before = source.copy()
    support_before = support.copy()

    first = fit_capta_prototypes(
        source,
        support,
        labels,
        candidate_id=A3_R4_SUPPORT_SHIFT,
        rank=4,
        prior_strength=3.0,
    )
    second = fit_capta_prototypes(
        source,
        support,
        labels,
        candidate_id=A3_R4_SUPPORT_SHIFT,
        rank=4,
        prior_strength=3.0,
    )

    assert first.domain_basis.shape == (8, 4)
    assert first.domain_code.shape == (4,)
    assert first.audit["effective_rank"] == 4
    assert first.audit["basis_source"] == "target_support_class_residuals"
    np.testing.assert_array_equal(first.domain_basis, second.domain_basis)
    np.testing.assert_array_equal(first.domain_code, second.domain_code)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(support, support_before)
    assert first.source_prototypes.flags.writeable is False
    assert first.target_prototypes.flags.writeable is False


@pytest.mark.parametrize(
    ("labels", "candidate_id", "rank", "prior_strength"),
    [
        (np.asarray([0, 0], dtype=np.int64), A1_SUPPORT_SHRINK, 4, 2.0),
        (np.asarray([0, 2], dtype=np.int64), A2_SHARED_SHIFT, 4, 2.0),
        (np.asarray([0, 1], dtype=np.int64), "CAPTA_UNKNOWN", 4, 2.0),
        (np.asarray([0, 1], dtype=np.int64), A3_R4_SUPPORT_SHIFT, 5, 2.0),
        (np.asarray([0, 1], dtype=np.int64), A3_R4_SUPPORT_SHIFT, 4, 0.0),
    ],
)
def test_invalid_or_incomplete_support_state_fails_closed(
    labels: np.ndarray,
    candidate_id: str,
    rank: int,
    prior_strength: float,
) -> None:
    source = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = source.copy()

    with pytest.raises(CaptaPrototypeError):
        fit_capta_prototypes(
            source,
            support,
            labels,
            candidate_id=candidate_id,
            rank=rank,
            prior_strength=prior_strength,
        )
