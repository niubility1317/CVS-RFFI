from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m24_invariance_breaking import (
    G1,
    G2,
    G3,
    G4,
    balanced_if256,
    fit_m24_invariance_breaking,
)


CLASSES = ("c0", "c1", "c2")


def _blocks(k_shot: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2404)
    identity_centres = np.zeros((3, 160), dtype=np.float64)
    fft_centres = np.zeros((3, 96), dtype=np.float64)
    identity_centres[np.arange(3), np.arange(3)] = 1.0
    fft_centres[np.arange(3), np.arange(3)] = 1.0
    nuisance = np.zeros(256, dtype=np.float64)
    nuisance[12] = 1.0
    nuisance[173] = -0.7
    nuisance /= np.linalg.norm(nuisance)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_name in enumerate(CLASSES):
        for sample_index in range(k_shot):
            joined = np.concatenate(
                [identity_centres[class_index], fft_centres[class_index]]
            )
            amplitude = (sample_index - (k_shot - 1) / 2.0) * 0.18
            noise = 0.01 * rng.normal(size=256)
            if class_index == 2:
                noise *= 5.0
            rows.append(np.concatenate([joined + amplitude * nuisance + noise, np.ones(10)]))
            labels.append(class_name)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def test_balanced_feature_removes_fixed_fft_energy_dominance() -> None:
    rows, _ = _blocks()
    legacy = physical_if256(rows)
    balanced = balanced_if256(rows)
    legacy_fft_fraction = np.mean(np.sum(legacy[:, 160:] ** 2, axis=1))
    balanced_fft_fraction = np.mean(np.sum(balanced[:, 160:] ** 2, axis=1))
    assert legacy_fft_fraction == pytest.approx(16.0 / 17.0, abs=1e-6)
    assert balanced_fft_fraction == pytest.approx(0.5, abs=1e-6)
    assert not np.allclose(legacy, balanced)


def test_fit_contract_is_support_only_and_has_no_query_or_truth_argument() -> None:
    parameters = set(inspect.signature(fit_m24_invariance_breaking).parameters)
    assert "query" not in " ".join(parameters).lower()
    assert "truth" not in " ".join(parameters).lower()


def test_g2_is_noninvertible_and_query_batch_composition_invariant() -> None:
    rows, labels = _blocks()
    state, audit = fit_m24_invariance_breaking(
        arm=G2,
        support_blocks=rows,
        support_labels=labels,
        classes=CLASSES,
        k_shot=5,
        domain_digest="support-only",
    )
    assert audit["projection"]["projection_removed_rank"] == 1
    assert audit["projection"]["projection_active"] is True
    query = rows[[0, 6, 12]]
    together = state.score(query)
    separately = np.concatenate([state.score(item[None, :]) for item in query], axis=0)
    assert np.allclose(together, separately, atol=1e-7)


def test_g3_penalizes_the_more_uncertain_class_symmetrically() -> None:
    rows, labels = _blocks()
    _, audit = fit_m24_invariance_breaking(
        arm=G3,
        support_blocks=rows,
        support_labels=labels,
        classes=CLASSES,
        k_shot=5,
        domain_digest="support-only",
    )
    penalties = np.asarray(audit["uncertainty_penalty"], dtype=np.float64)
    assert penalties[2] > penalties[0]
    permutation = np.asarray([2, 0, 1])
    permuted_classes = tuple(CLASSES[index] for index in permutation)
    _, permuted_audit = fit_m24_invariance_breaking(
        arm=G3,
        support_blocks=rows,
        support_labels=labels,
        classes=permuted_classes,
        k_shot=5,
        domain_digest="support-only",
    )
    assert np.allclose(
        np.asarray(permuted_audit["uncertainty_penalty"]), penalties[permutation]
    )


def test_g4_uses_two_count_normalized_prototypes_only_when_k_is_sufficient() -> None:
    rows, labels = _blocks(k_shot=5)
    state, audit = fit_m24_invariance_breaking(
        arm=G4,
        support_blocks=rows,
        support_labels=labels,
        classes=CLASSES,
        k_shot=5,
        domain_digest="support-only",
    )
    assert audit["prototype_count_by_class"] == [2, 2, 2]
    assert audit["prototype_pooling"] == "class_count_normalized_logmeanexp"
    assert np.isfinite(state.score(rows)).all()

    small_rows, small_labels = _blocks(k_shot=2)
    _, small_audit = fit_m24_invariance_breaking(
        arm=G4,
        support_blocks=small_rows,
        support_labels=small_labels,
        classes=CLASSES,
        k_shot=2,
        domain_digest="support-only",
    )
    assert small_audit["prototype_count_by_class"] == [1, 1, 1]
    assert small_audit["k_specialization"] == "K2_PROJECTED_SINGLE_PROTOTYPE"


def test_g1_is_a_real_k1_head_instead_of_forced_historical_f1() -> None:
    rows, labels = _blocks(k_shot=1)
    state, audit = fit_m24_invariance_breaking(
        arm=G1,
        support_blocks=rows,
        support_labels=labels,
        classes=CLASSES,
        k_shot=1,
        domain_digest="support-only",
    )
    assert audit["k_specialization"] == "K1_FROZEN_BALANCED_PROTOTYPE"
    assert audit["historical_f1_fallback"] is False
    assert state.classes == CLASSES

