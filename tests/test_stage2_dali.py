from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from cvsrffi.stage2_ciaf import CiafError, FEATURE_DIM, Int8DomainClassComponent
from cvsrffi.stage2_dali import (
    DaliConfig,
    fit_old_dali,
    predict_one_dali,
    register_new_dali,
    rerank_old_scores_dali,
    score_one_dali,
)


OLD = ("old-a", "old-b", "old-c")


def _direction(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _component() -> Int8DomainClassComponent:
    q = np.zeros((4, 3, FEATURE_DIM), dtype=np.int8)
    scale = np.full((4, 3), 1.0 / 127.0, dtype=np.float16)
    mask = np.ones((4, 3), dtype=np.uint8)
    for domain in range(4):
        for class_index in range(3):
            q[domain, class_index, class_index] = 127
            q[domain, class_index, 20 + domain] = domain + 1
    return Int8DomainClassComponent(q, scale, mask, OLD)


def _component_with_inactive_domain() -> Int8DomainClassComponent:
    component = _component()
    mask = np.array(component.domain_class_mask, copy=True)
    q = np.array(component.domain_class_q, copy=True)
    mask[-1, 0] = 0
    q[-1, 0] = 0
    return Int8DomainClassComponent(q, component.domain_class_scale, mask, OLD)


def _support(
    classes: tuple[str, ...], k: int, offset: int
) -> tuple[np.ndarray, list[str]]:
    rows = []
    labels = []
    for class_index, label in enumerate(classes):
        for rank in range(k):
            row = _direction(offset + class_index)
            row[80 + rank] = 0.01
            rows.append(row)
            labels.append(label)
    return np.stack(rows), labels


def _registered_state(*, k: int = 5, config: DaliConfig | None = None):
    support, labels = _support(OLD, k, 0)
    logits = support[:, :3] * 4.0
    before = fit_old_dali(
        _component(), support, labels, logits, config=config or DaliConfig()
    )
    new_classes = ("new-y", "new-x")
    new_support, new_labels = _support(new_classes, k, 10)
    after = register_new_dali(
        before, new_support, new_labels, registered_classes=new_classes
    )
    return before, after


def test_dali_is_zero_train_and_never_persists_dequantized_ground_anchors() -> None:
    before, state = _registered_state(config=DaliConfig(ground_weight=0.05))
    resource = state.resource_audit()
    assert resource["trainable_parameters"] == 0
    assert resource["adaptation_epochs"] == 0
    assert resource["persistent_full_precision_ground_anchor_count"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["selected_int8_domain_anchors_per_old_class"] == 1
    assert resource["support_domain_selection_access"] is False
    assert resource["query_domain_selection_access"] is False
    assert resource["new_score_policy"] == "bitwise_unchanged"
    assert state.persistent_state_bytes <= 256 * 1024
    assert not hasattr(state, "ground_anchor_prototypes")
    assert before.component is state.component


def test_component_only_fixed_medoid_is_support_independent_and_preregisterable() -> None:
    component = _component()
    support_a, labels_a = _support(OLD, 5, 0)
    support_b, labels_b = _support(OLD, 5, 30)
    state_a = fit_old_dali(
        component, support_a, labels_a, None, config=DaliConfig(direct_weight=0.0)
    )
    state_b = fit_old_dali(
        component, support_b, labels_b, None, config=DaliConfig(direct_weight=0.0)
    )
    assert state_a.medoid_domain_index == state_b.medoid_domain_index
    assert state_a.fixed_medoid_domain_index == state_a.medoid_domain_index
    assert (
        state_a.support_audit["selection_data"]
        == "immutable_int8_component_only_no_target_support"
    )

    explicit = fit_old_dali(
        component,
        support_a,
        labels_a,
        None,
        config=DaliConfig(medoid_domain_index=2),
    )
    assert explicit.medoid_domain_index == 2
    assert (
        explicit.support_audit["medoid_policy"]
        == "explicit_preregistered_component_domain_index"
    )
    wrong = (state_a.medoid_domain_index + 1) % 4
    with pytest.raises(CiafError, match="state drift"):
        replace(state_a, fixed_medoid_domain_index=wrong)


def test_registration_preserves_old_state_and_old_scores_bitwise() -> None:
    before, after = _registered_state(
        config=DaliConfig(ground_weight=0.05, direct_weight=0.25)
    )
    assert after.classes == OLD + ("new-y", "new-x")
    for name in (
        "old_support_prototypes",
        "ground_weight_by_old_class",
        "support_margin_q25_by_old_class",
    ):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    np.testing.assert_array_equal(
        after.target_prototypes[: before.old_class_count], before.target_prototypes
    )
    probe = _direction(0)
    logits = np.asarray([1.0, -0.2, 0.1], dtype=np.float32)
    np.testing.assert_array_equal(
        score_one_dali(after, probe, logits)[:3],
        score_one_dali(before, probe, logits),
    )


def test_public_reranker_strictly_preserves_max_old_and_new_score_bits() -> None:
    _, state = _registered_state(
        config=DaliConfig(
            ground_weight=0.10,
            direct_weight=0.25,
            evidence_clip=0.04,
            k_shrink_offset=2.0,
        )
    )
    rng = np.random.default_rng(713101)
    for _ in range(50):
        base = rng.normal(size=len(state.classes)).astype(np.float32)
        z_id = rng.normal(size=FEATURE_DIM).astype(np.float32)
        direct = rng.normal(size=3).astype(np.float32)
        scores = rerank_old_scores_dali(state, base, z_id, direct)
        np.testing.assert_array_equal(scores[3:], base[3:])
        assert np.max(scores[:3]) == np.max(base[:3])
        base_group_is_old = int(np.argmax(base)) < 3
        reranked_group_is_old = int(np.argmax(scores)) < 3
        assert reranked_group_is_old == base_group_is_old
        # A shared max-restoring shift cannot enlarge pairwise delta spread
        # beyond twice the configured per-class clip.
        assert np.ptp(scores[:3] - base[:3]) <= 0.080001


def test_new_prediction_and_identity_are_unchanged_when_base_group_is_new() -> None:
    _, state = _registered_state(
        config=DaliConfig(ground_weight=0.10, direct_weight=0.50)
    )
    base = np.asarray([0.1, 0.2, -0.1, 0.7, 0.6], dtype=np.float32)
    scores = rerank_old_scores_dali(
        state, base, _direction(0), np.asarray([9.0, -9.0, 0.0], dtype=np.float32)
    )
    assert int(np.argmax(base)) == int(np.argmax(scores)) == 3
    np.testing.assert_array_equal(scores[3:], base[3:])


def test_zero_optional_weights_matches_arbitrary_base_scores_bitwise() -> None:
    _, state = _registered_state(
        config=DaliConfig(ground_weight=0.0, direct_weight=0.0)
    )
    base = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    scores = rerank_old_scores_dali(state, base, _direction(1), None)
    np.testing.assert_array_equal(scores, base)

    probe_scores = score_one_dali(state, _direction(1), None)
    expected = _direction(1) @ state.target_prototypes.T
    np.testing.assert_allclose(probe_scores, expected, atol=1.0e-7)
    assert predict_one_dali(state, _direction(1), None)[0] == "old-b"


def test_ground_and_direct_ablation_surfaces_are_independent() -> None:
    support, labels = _support(OLD, 5, 0)
    ground_only = fit_old_dali(
        _component(),
        support,
        labels,
        None,
        config=DaliConfig(ground_weight=0.05, direct_weight=0.0),
    )
    assert np.isfinite(score_one_dali(ground_only, _direction(0), None)).all()

    with pytest.raises(CiafError, match="support direct logits"):
        fit_old_dali(
            _component(),
            support,
            labels,
            None,
            config=DaliConfig(ground_weight=0.0, direct_weight=0.25),
        )
    direct_only = fit_old_dali(
        _component(),
        support,
        labels,
        support[:, :3],
        config=DaliConfig(ground_weight=0.0, direct_weight=0.25),
    )
    with pytest.raises(CiafError, match="required by direct evidence"):
        score_one_dali(direct_only, _direction(0), None)
    assert np.isfinite(
        score_one_dali(direct_only, _direction(0), np.asarray([1.0, 0.0, 0.0]))
    ).all()


def test_k1_uses_strong_class_symmetric_shrink_and_registers_repeatedly() -> None:
    support, labels = _support(OLD, 1, 0)
    state = fit_old_dali(
        _component(), support, labels, None, config=DaliConfig(k_shrink_offset=2.0)
    )
    assert state.k_shot == 1
    assert state.evidence_k_shrink == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(
        state.ground_weight_by_old_class, np.full(3, 0.05 / 3.0), atol=1.0e-7
    )
    assert np.isfinite(state.support_margin_q25_by_old_class).all()

    first_z, first_y = _support(("new-a",), 1, 10)
    first = register_new_dali(state, first_z, first_y, registered_classes=("new-a",))
    second_z, second_y = _support(("new-b",), 1, 11)
    second = register_new_dali(first, second_z, second_y, registered_classes=("new-b",))
    assert second.classes == OLD + ("new-a", "new-b")
    assert second.support_count_by_class.tolist() == [1, 1, 1, 1, 1]
    np.testing.assert_array_equal(second.target_prototypes[:4], first.target_prototypes)


def test_top_m_surface_is_removed_and_pair_or_invalid_medoid_fail_closed() -> None:
    assert "top_m" not in DaliConfig.__dataclass_fields__
    with pytest.raises(CiafError, match="fixed-medoid max-old"):
        DaliConfig(pair_weight=0.01).validate()
    with pytest.raises(CiafError, match="fixed-medoid max-old"):
        DaliConfig(medoid_domain_index=True).validate()  # type: ignore[arg-type]

    support, labels = _support(OLD, 5, 0)
    with pytest.raises(CiafError, match="cover every old class"):
        fit_old_dali(
            _component_with_inactive_domain(),
            support,
            labels,
            None,
            config=DaliConfig(medoid_domain_index=3),
        )


def test_only_int8_component_persists_and_dequantization_is_call_local() -> None:
    _, state = _registered_state()
    assert state.component.domain_class_q.dtype == np.int8
    assert not state.component.domain_class_q.flags.writeable
    assert not any(
        value.dtype.kind == "f" and value.ndim == 3
        for value in vars(state).values()
        if isinstance(value, np.ndarray)
    )
    first = score_one_dali(state, _direction(0), np.asarray([1.0, 0.0, 0.0]))
    second = score_one_dali(state, _direction(0), np.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(first, second)
    with pytest.raises(TypeError):
        state.support_audit["selection_data"] = "mutated"  # type: ignore[index]


def test_nan_shape_registry_and_direct_logit_guards() -> None:
    support, labels = _support(OLD, 5, 0)
    state = fit_old_dali(
        _component(),
        support,
        labels,
        support[:, :3],
        config=DaliConfig(direct_weight=0.1),
    )
    with pytest.raises(CiafError, match="exactly one"):
        score_one_dali(state, np.stack([_direction(0), _direction(1)]), np.zeros(3))
    with pytest.raises(CiafError, match="finite"):
        score_one_dali(
            state,
            np.full(FEATURE_DIM, np.nan, dtype=np.float32),
            np.zeros(3),
        )
    with pytest.raises(CiafError, match="exactly one sample"):
        score_one_dali(state, _direction(0), np.zeros((1, 3)))
    with pytest.raises(CiafError, match="base scores"):
        rerank_old_scores_dali(
            state, np.zeros(2, dtype=np.float32), _direction(0), np.zeros(3)
        )
    bad_logits = support[:, :3].copy()
    bad_logits[0, 0] = np.nan
    with pytest.raises(CiafError, match="direct-logit"):
        fit_old_dali(_component(), support, labels, bad_logits, config=DaliConfig())
    with pytest.raises(CiafError, match="new registry"):
        register_new_dali(state, support[:5], ["old-a"] * 5, registered_classes=("old-a",))


def test_public_surfaces_have_no_truth_role_quota_assignment_inputs() -> None:
    forbidden = ("query", "truth", "role", "quota", "assignment", "clean", "source")
    functions = (
        fit_old_dali,
        register_new_dali,
        rerank_old_scores_dali,
        score_one_dali,
        predict_one_dali,
    )
    for function in functions:
        parameters = inspect.signature(function).parameters
        assert not any(
            token in name.lower() for name in parameters for token in forbidden
        )
