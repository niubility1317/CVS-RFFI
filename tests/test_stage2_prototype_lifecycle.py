from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_prototype_lifecycle import (
    FEATURE_DIM,
    LifecycleConfig,
    LifecycleError,
    fit_old_snapshot,
    predict_one,
    register_new_classes,
    score_batch,
    score_one,
)


OLD = ("old-a", "old-b", "old-c")
CAPSULE = "a" * 64
RECEIPT = "c" * 64
AFTER_CAPSULE = "d" * 64
AFTER_RECEIPT = "e" * 64


def _direction(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _support(
    classes: tuple[str, ...], k: int, offset: int, *, spread: float = 0.03
) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    for class_index, label in enumerate(classes):
        for rank in range(k):
            row = _direction(offset + class_index)
            row[60 + rank] = spread * ((rank % 3) - 1)
            rows.append(row)
            labels.append(label)
    return np.stack(rows), labels


def _old(k: int, config: LifecycleConfig | None = None):
    support, labels = _support(OLD, k, 0)
    return fit_old_snapshot(
        support,
        labels,
        OLD,
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        config=config,
    )


def _register(
    state,
    new_support: np.ndarray,
    new_labels: list[str],
    new_classes: tuple[str, ...],
):
    old_support, old_labels = _support(OLD, state.k_shot, 0)
    return register_new_classes(
        state,
        old_support,
        old_labels,
        new_support,
        new_labels,
        new_classes,
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        after_registration_capsule_root_sha256=AFTER_CAPSULE,
        after_registration_receipt_sha256=AFTER_RECEIPT,
    )


def test_k1_uses_one_center_and_disables_unverifiable_radius() -> None:
    state = _old(1, LifecycleConfig(radius_prior=0.27))
    assert state.k_shot == 1
    assert state.center_policy == "mean"
    assert state.radius_policy == "fixed_preregistered_prior_k1"
    np.testing.assert_array_equal(state.radii, np.full(3, 0.27, dtype=np.float32))
    np.testing.assert_array_equal(state.radius_active, np.zeros(3, dtype=np.bool_))
    assert state.boundaries == ()
    assert state.support_audit["stage2b_radius_support_guard"]["status"] == (
        "OFF_K1_SELF_EXCLUSION_NOT_ESTIMABLE"
    )
    assert state.support_audit["boundary_policy"] == (
        "forced_off_k1_self_exclusion_not_estimable"
    )
    assert state.resource_audit()["boundary_k1_forced_off"] is True
    assert predict_one(state, _direction(1))[0] == "old-b"


def test_k5_uses_mean_or_medoid_and_loo_radius_shrink() -> None:
    config = LifecycleConfig(radius_prior=0.40, radius_shrink_offset=4.0)
    state = _old(5, config)
    assert state.center_policy in ("mean", "medoid")
    assert state.radius_policy == "loo_q80_shrunk"
    assert state.support_audit["radius_shrink"] == pytest.approx(0.5)
    assert state.support_audit["radius_shrink_space"] == "squared_radius_rms"
    assert np.all(state.radii < config.radius_prior)
    assert set(state.support_audit["candidate_metrics"]) == {"mean", "medoid"}


def test_radius_shrink_uses_squared_radius_space() -> None:
    prior = 0.40
    rows = np.stack(
        [_direction(class_index) for class_index in range(3) for _ in range(5)]
    )
    labels = [label for label in OLD for _ in range(5)]
    state = fit_old_snapshot(
        rows,
        labels,
        OLD,
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        config=LifecycleConfig(radius_prior=prior, radius_shrink_offset=4.0),
    )
    expected = prior * np.sqrt(1.0 - 0.5)
    np.testing.assert_allclose(state.radii, expected, rtol=1.0e-6, atol=1.0e-7)


@pytest.mark.parametrize("k", [10, 20])
def test_k10_k20_allow_fixed_robust_candidates_and_lto_shrink(k: int) -> None:
    state = _old(k, LifecycleConfig(radius_prior=0.30))
    assert set(state.support_audit["candidate_metrics"]) == {
        "mean",
        "medoid",
        "robust_trim",
    }
    assert state.center_policy in ("mean", "medoid", "robust_trim")
    assert state.radius_policy == "lto_q80_shrunk"
    expected = (k - 2) / ((k - 2) + 4.0)
    assert state.support_audit["radius_shrink"] == pytest.approx(expected)


def test_stage2c_is_append_only_and_rejects_duplicate_registration() -> None:
    before = _old(5)
    new_support, new_labels = _support(("new-x", "new-y"), 5, 10)
    after = _register(before, new_support, new_labels, ("new-x", "new-y"))
    assert after.classes == OLD + ("new-x", "new-y")
    assert after.stage == "stage2c_append_only"
    assert after.support_count_by_class.tolist() == [5, 5, 5, 5, 5]
    with pytest.raises(LifecycleError, match="append-only"):
        register_new_classes(
            after,
            new_support,
            new_labels,
            new_support[:5],
            new_labels[:5],
            ("new-x",),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )


def test_old_prototype_radius_and_score_path_are_bitwise_locked() -> None:
    before = _old(5)
    new_support, new_labels = _support(("new-x",), 5, 10)
    after = _register(before, new_support, new_labels, ("new-x",))
    np.testing.assert_array_equal(after.old_prototype_snapshot, before.prototypes)
    np.testing.assert_array_equal(after.old_radius_snapshot, before.radii)
    np.testing.assert_array_equal(after.prototypes[:3], before.prototypes)
    np.testing.assert_array_equal(after.radii[:3], before.radii)
    np.testing.assert_array_equal(after.radius_active[:3], before.radius_active)
    assert not after.prototypes.flags.writeable
    assert not after.radii.flags.writeable
    for probe in (_direction(0), _direction(1), _direction(10)):
        np.testing.assert_array_equal(score_one(after, probe)[:3], score_one(before, probe))


def test_batch_wrapper_is_exactly_sample_local() -> None:
    before = _old(5)
    new_support, new_labels = _support(("new-x",), 5, 10)
    state = _register(before, new_support, new_labels, ("new-x",))
    batch = np.stack([_direction(0), _direction(1), _direction(10)])
    scores = score_batch(state, batch)
    singles = np.stack([score_one(state, row) for row in batch])
    np.testing.assert_array_equal(scores, singles)
    changed = batch.copy()
    changed[1] = _direction(2)
    changed_scores = score_batch(state, changed)
    np.testing.assert_array_equal(changed_scores[0], scores[0])
    np.testing.assert_array_equal(changed_scores[2], scores[2])


def test_collision_boundary_is_sparse_support_only_and_at_most_one_per_new() -> None:
    before = _old(
        5,
        LifecycleConfig(
            boundary_enabled=True,
            boundary_min_collision_cosine=-0.5,
            boundary_topk=4,
        ),
    )
    # Force a sparse candidate by threshold while keeping the base prototype
    # non-invasive to the Stage2-B old snapshot.
    rows = []
    labels = []
    for rank in range(5):
        row = _direction(10)
        row[20] = 0.04 + rank * 0.001
        rows.append(row)
        labels.append("new-close")
    state = _register(before, np.stack(rows), labels, ("new-close",))
    assert len(state.boundaries) <= 1
    assert len(state.boundaries) == 1
    boundary = state.boundaries[0]
    assert boundary.new_class_index == 3
    assert len(boundary.feature_indices) <= 4
    assert state.resource_audit()["sparse_collision_boundaries_per_new_class_max"] == 1


def test_radius_and_boundary_off_are_exact_base_cosine_fallbacks() -> None:
    config = LifecycleConfig(radius_enabled=False, boundary_enabled=False)
    before = _old(5, config)
    new_support, new_labels = _support(("new-x",), 5, 10)
    state = _register(before, new_support, new_labels, ("new-x",))
    probe = _direction(1)
    expected = probe @ state.prototypes.T
    np.testing.assert_array_equal(score_one(state, probe), expected.astype(np.float32))
    assert state.boundaries == ()


def test_radius_bounds_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(LifecycleError, match="finite"):
        LifecycleConfig(radius_prior=float("inf")).validate()
    with pytest.raises(LifecycleError, match="out of range"):
        LifecycleConfig(radius_prior=2.1, radius_max=2.1).validate()
    state = _old(
        5,
        LifecycleConfig(
            radius_prior=0.0,
            radius_min=0.0,
            radius_max=2.0,
            radius_penalty_weight=1.0,
            radius_penalty_clip=0.03,
        ),
    )
    scores = score_one(state, -_direction(0))
    assert state.radius_active.all()
    assert np.isfinite(scores).all()
    base = -_direction(0) @ state.prototypes.T
    assert np.max(base - scores) <= 0.030001
    bad = _direction(0)
    bad[4] = np.nan
    with pytest.raises(LifecycleError, match="finite"):
        score_one(state, bad)


def test_capsule_root_is_required_and_registration_must_match_exactly() -> None:
    before = _old(5)
    assert before.old_support_capsule_root_sha256 == CAPSULE
    old_support, old_labels = _support(OLD, 5, 0)
    new_support, new_labels = _support(("new-x",), 5, 10)
    with pytest.raises(LifecycleError, match="capsule root"):
        register_new_classes(
            before,
            old_support,
            old_labels,
            new_support,
            new_labels,
            ("new-x",),
            old_support_capsule_root_sha256="b" * 64,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )


def test_old_support_content_and_receipt_are_recomputed_not_label_k_only() -> None:
    before = _old(5)
    old_support, old_labels = _support(OLD, 5, 0)
    new_support, new_labels = _support(("new-x",), 5, 10)
    changed = old_support.copy()
    changed[0, 40] = 0.25
    with pytest.raises(LifecycleError, match="content SHA256"):
        register_new_classes(
            before,
            changed,
            old_labels,
            new_support,
            new_labels,
            ("new-x",),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )
    with pytest.raises(LifecycleError, match="receipt SHA256"):
        register_new_classes(
            before,
            old_support,
            old_labels,
            new_support,
            new_labels,
            ("new-x",),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256="f" * 64,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )


def test_adversarial_new_prototype_intrusion_fails_before_radius_boundary() -> None:
    before = _old(5)
    old_support, old_labels = _support(OLD, 5, 0)
    adversarial = []
    labels = []
    for rank in range(5):
        row = _direction(0)
        row[90 + rank] = 0.001
        adversarial.append(row)
        labels.append("new-copy-old-a")
    with pytest.raises(LifecycleError, match="prototype intrusion guard"):
        register_new_classes(
            before,
            old_support,
            old_labels,
            np.stack(adversarial),
            labels,
            ("new-copy-old-a",),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )


def test_radius_guards_expose_per_class_mask_and_numeric_metadata_accounting() -> None:
    before = _old(5)
    old_mask = before.radius_active.copy()
    new_support, new_labels = _support(("new-x", "new-y"), 5, 10)
    after = _register(before, new_support, new_labels, ("new-x", "new-y"))
    assert after.current_support_capsule_root_sha256 == AFTER_CAPSULE
    assert after.current_support_receipt_sha256 == AFTER_RECEIPT
    np.testing.assert_array_equal(after.radius_active[: len(OLD)], old_mask)
    assert after.radius_active.dtype == np.bool_
    guard = after.support_audit["new_radius_support_guard"]
    assert guard["data"] == "all_registered_leo_weak_support_only"
    assert guard["accepted_radius_count"] == int(
        np.sum(after.radius_active[len(OLD) :])
    )
    intrusion = after.support_audit["new_prototype_intrusion_guard"]
    assert intrusion["evaluated_directions"] == (
        "old_to_new",
        "new_to_old",
        "new_to_new",
    )
    resource = after.resource_audit()
    assert resource["radius_active_class_count"] == int(np.sum(after.radius_active))
    assert resource["persistent_numeric_state_bytes"] > 0
    assert resource["serialized_metadata_estimate_bytes"] > 0
    assert resource["support_audit_artifact_bytes"] > 0
    assert resource["persistent_state_excludes_support_audit_artifact"] is True
    assert resource["persistent_state_bytes"] == (
        resource["persistent_numeric_state_bytes"]
        + resource["serialized_metadata_estimate_bytes"]
    )


def test_registration_validates_old_registry_k_and_guards_all_support() -> None:
    config = LifecycleConfig(
        boundary_enabled=True,
        boundary_min_collision_cosine=-0.5,
        boundary_topk=4,
    )
    before = _old(5, config)
    old_support, old_labels = _support(OLD, 5, 0)
    new_support, new_labels = _support(("new-x", "new-y"), 5, 10)
    with pytest.raises(LifecycleError, match="exact K"):
        register_new_classes(
            before,
            old_support[:-1],
            old_labels[:-1],
            new_support,
            new_labels,
            ("new-x", "new-y"),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )
    wrong_labels = list(old_labels)
    wrong_labels[0] = "not-registered"
    with pytest.raises(LifecycleError, match="registry"):
        register_new_classes(
            before,
            old_support,
            wrong_labels,
            new_support,
            new_labels,
            ("new-x", "new-y"),
            old_support_capsule_root_sha256=CAPSULE,
            old_support_receipt_sha256=RECEIPT,
            after_registration_capsule_root_sha256=AFTER_CAPSULE,
            after_registration_receipt_sha256=AFTER_RECEIPT,
        )

    state = register_new_classes(
        before,
        old_support,
        old_labels,
        new_support,
        new_labels,
        ("new-x", "new-y"),
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        after_registration_capsule_root_sha256=AFTER_CAPSULE,
        after_registration_receipt_sha256=AFTER_RECEIPT,
    )
    guard = state.support_audit["boundary_support_guard"]
    assert guard["data"] == "all_registered_leo_weak_support_only"
    assert guard["guard_directions"] == ("old_to_new", "new_to_old")
    assert guard["support_rows"] == 5 * len(state.classes)
    assert guard["class_count"] == len(state.classes)
    assert guard["combined_guard_passed"] is True
    assert guard["criteria"] == (
        "every_class_accuracy_non_decreasing",
        "worst_true_margin_non_decreasing",
    )
    baseline_accuracy = np.asarray(guard["baseline_per_class_accuracy"])
    baseline_margin = guard["baseline_worst_true_margin"]
    assert np.all(
        np.asarray(guard["final_per_class_accuracy"]) + 1.0e-7
        >= baseline_accuracy
    )
    assert guard["final_worst_true_margin"] + 1.0e-7 >= baseline_margin
    for decision in guard["candidate_decisions"].values():
        if decision["accepted"]:
            assert np.all(
                np.asarray(decision["per_class_accuracy"]) + 1.0e-7
                >= baseline_accuracy
            )
            assert decision["worst_true_margin"] + 1.0e-7 >= baseline_margin


def test_multi_new_boundary_decisions_are_registration_order_invariant() -> None:
    config = LifecycleConfig(
        boundary_enabled=True,
        boundary_min_collision_cosine=-0.5,
        boundary_topk=4,
    )
    before = _old(5, config)
    old_support, old_labels = _support(OLD, 5, 0)
    new_support, new_labels = _support(("new-x", "new-y"), 5, 10)
    forward = register_new_classes(
        before,
        old_support,
        old_labels,
        new_support,
        new_labels,
        ("new-x", "new-y"),
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        after_registration_capsule_root_sha256=AFTER_CAPSULE,
        after_registration_receipt_sha256=AFTER_RECEIPT,
    )
    reverse_order = np.asarray(
        [index for index, label in enumerate(new_labels) if label == "new-y"]
        + [index for index, label in enumerate(new_labels) if label == "new-x"]
    )
    reverse = register_new_classes(
        before,
        old_support,
        old_labels,
        new_support[reverse_order],
        [new_labels[index] for index in reverse_order],
        ("new-y", "new-x"),
        old_support_capsule_root_sha256=CAPSULE,
        old_support_receipt_sha256=RECEIPT,
        after_registration_capsule_root_sha256=AFTER_CAPSULE,
        after_registration_receipt_sha256=AFTER_RECEIPT,
    )

    def semantic_decisions(state):
        decisions = state.support_audit["boundary_support_guard"][
            "candidate_decisions"
        ]
        return {
            name: (value["rival_class"], value["accepted"])
            for name, value in decisions.items()
        }

    assert semantic_decisions(forward) == semantic_decisions(reverse)
    assert {
        forward.classes[item.new_class_index]: forward.classes[item.rival_class_index]
        for item in forward.boundaries
    } == {
        reverse.classes[item.new_class_index]: reverse.classes[item.rival_class_index]
        for item in reverse.boundaries
    }


def test_resource_and_public_api_expose_no_query_fit_or_oracle_surface() -> None:
    before = _old(10)
    new_support, new_labels = _support(("new-x", "new-y"), 10, 10)
    state = _register(before, new_support, new_labels, ("new-x", "new-y"))
    resource = state.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["adaptation_epochs"] == 0
    assert resource["optimizer_steps"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_updates"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["prototype_count_per_class_max"] == 1
    assert resource["phase2_query_decision_policy"] == "per_sample_all_registered_classes"
    assert resource["query_role_oracle_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["query_batch_global_assignment"] is False
    assert resource["persistent_state_bytes"] <= 256 * 1024
    for function in (score_one, score_batch, predict_one):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {
            "labels",
            "roles",
            "old_new_role",
            "quota",
            "class_counts",
            "query_truth",
        }
