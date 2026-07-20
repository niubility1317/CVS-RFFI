from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d93_paired_ground_transport import (
    D93PairedGroundTransportError,
    canonical_ground_geometry,
    fit_paired_ground_transport,
    transform_registered_features,
    transform_z160,
)


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _fixture(seed: int = 9):
    rng = np.random.default_rng(seed)
    classes = _normalize(rng.normal(size=(6, 160)))
    nuisance = np.linalg.qr(rng.normal(size=(160, 5)))[0]
    prototypes = []
    for _ in range(14):
        coefficient = rng.normal(scale=0.08, size=(6, 5))
        prototypes.append(_normalize(classes + coefficient @ nuisance.T))
    ground = np.stack(prototypes)
    mask = np.ones((14, 6), dtype=np.uint8)
    canonical, identity, nuisance_basis, _ = canonical_ground_geometry(ground, mask)
    interaction = rng.normal(scale=0.12, size=(nuisance_basis.shape[1], identity.shape[1]))
    operator = np.eye(160) + nuisance_basis @ interaction @ identity.T
    translation = rng.normal(scale=0.01, size=160)
    target_centers = _normalize(canonical @ operator.T + translation[None, :])
    return rng, ground, mask, canonical, target_centers


@pytest.mark.parametrize("include_scale", [False, True])
def test_k1_paired_transport_reduces_ground_target_error(include_scale: bool) -> None:
    _, ground, mask, canonical, target = _fixture()
    labels = np.arange(6, dtype=np.int64)
    transport = fit_paired_ground_transport(
        ground,
        mask,
        target,
        labels,
        include_nuisance_scale=include_scale,
    )
    restored = transform_z160(target, transport)
    before = np.mean(np.linalg.norm(target - canonical, axis=1))
    after = np.mean(np.linalg.norm(restored - canonical, axis=1))
    assert after < before
    assert transport.audit["k_shot"] == 1
    assert transport.audit["ground_to_target_identity_pairing_used"] is True
    assert transport.audit["query_rows_used"] == 0
    assert transport.audit["ground_aggregate_prototypes_only"] is True
    assert transport.audit["target_clean_iq_access"] is False
    assert transport.audit["target_new_clean_iq_access"] is False
    assert transport.audit["same_physical_iq_multi_channel_views"] is False
    assert transport.audit["phase2_channel_simulator_calls"] == 0
    assert transport.parameter_count < 500
    assert transport.incremental_state_bytes < 2048


def test_registered_transform_preserves_auxiliary_and_unit_norm() -> None:
    rng, ground, mask, _, target = _fixture()
    support = np.repeat(target, 2, axis=0)
    labels = np.repeat(np.arange(6), 2)
    transport = fit_paired_ground_transport(
        ground, mask, support, labels, include_nuisance_scale=True
    )
    primary = _normalize(rng.normal(size=(11, 160)))
    auxiliary = _normalize(rng.normal(size=(11, 128)))
    registered = _normalize(np.concatenate([primary, 4.0 * auxiliary], axis=1))
    transformed = transform_registered_features(registered, transport)
    assert transformed.shape == (11, 288)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1e-6)
    original_aux = _normalize(registered[:, 160:])
    transformed_aux = _normalize(transformed[:, 160:])
    np.testing.assert_allclose(original_aux, transformed_aux, atol=1e-6)


def test_ground_domain_and_class_permutation_preserve_transformed_geometry() -> None:
    _, ground, mask, _, target = _fixture()
    labels = np.arange(6, dtype=np.int64)
    first = fit_paired_ground_transport(
        ground, mask, target, labels, include_nuisance_scale=False
    )
    domain_order = np.array([5, 1, 11, 0, 9, 3, 12, 4, 8, 2, 13, 7, 6, 10])
    class_order = np.array([3, 0, 5, 1, 4, 2])
    permuted_labels = np.arange(6, dtype=np.int64)
    second = fit_paired_ground_transport(
        ground[domain_order][:, class_order],
        mask[domain_order][:, class_order],
        target[class_order],
        permuted_labels,
        include_nuisance_scale=False,
    )
    probe = _normalize(np.random.default_rng(17).normal(size=(20, 160)))
    np.testing.assert_allclose(
        transform_z160(probe, first), transform_z160(probe, second), atol=2e-5
    )


def test_rejects_unbalanced_target_support() -> None:
    _, ground, mask, _, target = _fixture()
    rows = np.concatenate([target, target[:1]], axis=0)
    labels = np.concatenate([np.arange(6), np.array([0])])
    with pytest.raises(D93PairedGroundTransportError, match="K-shot balance"):
        fit_paired_ground_transport(
            ground, mask, rows, labels, include_nuisance_scale=False
        )
