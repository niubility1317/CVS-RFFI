from __future__ import annotations

import numpy as np
import pytest
import torch
import cvsrffi.stage2_trainable_lowrank_support_adapter as d11

from cvsrffi.stage2_trainable_lowrank_support_adapter import (
    AdapterHyperparameters,
    TrainableLowRankAdapterState,
    TrainableLowRankAdapterError,
    _build_validated_feature_artifact_internal,
    evaluate_joint_registration_leave_two_out,
    fit_locked,
    predict_all_registered,
    register_new_classes,
    select_and_fit_k10,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _support(
    classes: tuple[str, ...], k: int, *, dim: int = 32, seed: int = 7
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    base = np.concatenate(
        [
            centers[index][None, :] + 0.03 * rng.normal(size=(k, dim))
            for index in range(len(classes))
        ],
        axis=0,
    ).astype(np.float32)
    second = (base + 0.01 * rng.normal(size=base.shape)).astype(np.float32)
    return {"base": base, "fixed_rx_view": second}, labels, ranks


def _hp() -> AdapterHyperparameters:
    return AdapterHyperparameters(
        candidate_id="d11_rank8_test",
        rank=8,
        epochs=2,
        learning_rate=0.01,
        seed=19,
    )


def _artifact(
    views: dict[str, np.ndarray],
    *,
    seed: int = 101,
    runtime_sha256: str = HASH_A,
):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(next(iter(views.values()))), 2, 16)).astype(np.float32)
    hashes = [
        __import__("hashlib").sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    ]
    counters = {name: 0 for name in views}

    def extract(_: np.ndarray, view_id: str) -> np.ndarray:
        index = counters[view_id]
        counters[view_id] += 1
        return np.asarray(views[view_id][index : index + 1], dtype=np.float32)

    return _build_validated_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"sid_{seed}_{index}" for index in range(len(iq))],
        parent_received_iq_sha256=hashes,
        sealed_runtime_sha256=runtime_sha256,
        feature_code_sha256=HASH_B,
        sealed_phase1_checkpoint_sha256=HASH_C,
        view_seed_by_id={name: 0 for name in views},
        extract_single_received_iq_view=extract,
    )


def test_k10_selection_and_resource_bounds() -> None:
    views, labels, ranks = _support(("a", "b", "c"), 10)
    result = select_and_fit_k10(
        _artifact(views), labels, ranks, candidates=(_hp(),), device="cpu"
    )
    assert result.state.k_shot == 10
    assert result.state.resource["trainable_parameters"] == 520
    assert result.state.resource["trainable_parameters"] <= 12_000
    assert result.state.resource["adapt_epochs"] <= 20
    assert result.validation["selected_candidate_id"] == "d11_rank8_test"
    assert len(result.trace) == 12


def test_training_bridge_supports_numpy2_torch21_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def incompatible_from_numpy(_value):
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    def incompatible_tensor_numpy(_value):
        raise TypeError("Numpy is not available")

    monkeypatch.setattr(torch, "from_numpy", incompatible_from_numpy)
    monkeypatch.setattr(torch.Tensor, "numpy", incompatible_tensor_numpy)
    views, labels, ranks = _support(("a", "b"), 5)
    state = fit_locked(
        _artifact(views),
        labels,
        ranks,
        k_shot=5,
        hyperparameters=_hp(),
    ).state
    assert state.classes == ("a", "b")
    assert state.low_rank_u.dtype == np.float32


def test_after_registration_freezes_adapter_and_old_prototypes() -> None:
    old_views, old_labels, old_ranks = _support(("old0", "old1"), 5)
    before = fit_locked(
        _artifact(old_views),
        old_labels,
        old_ranks,
        k_shot=5,
        hyperparameters=_hp(),
    ).state
    new_views, new_labels, new_ranks = _support(("new0", "new1"), 5, seed=13)
    after = register_new_classes(
        before,
        _artifact(new_views, seed=202),
        new_labels,
        new_ranks,
        k_shot=5,
        expected_old_support_feature_artifact_sha256=(
            before.support_feature_artifact_sha256
        ),
    )
    np.testing.assert_array_equal(after.low_rank_u, before.low_rank_u)
    np.testing.assert_array_equal(after.low_rank_v, before.low_rank_v)
    np.testing.assert_array_equal(after.gate, before.gate)
    np.testing.assert_array_equal(
        after.prototypes[: len(before.classes)], before.prototypes
    )
    assert after.classes == ("old0", "old1", "new0", "new1")
    assert after.old_class_count == 2
    assert after.registration_generation == 1


def test_joint_registration_l2o_evaluates_old_and_new_held_rows() -> None:
    old_views, old_labels, old_ranks = _support(("old0", "old1"), 10)
    new_views, new_labels, new_ranks = _support(
        ("new0", "new1"), 10, seed=13
    )
    audit, trace = evaluate_joint_registration_leave_two_out(
        _artifact(old_views),
        old_labels,
        old_ranks,
        _artifact(new_views, seed=203),
        new_labels,
        new_ranks,
        hyperparameters=_hp(),
    )
    assert len(audit["folds"]) == 5
    assert len(trace) == 15
    assert set(audit["after_old"]["per_class_accuracy"]) == {"old0", "old1"}
    assert set(audit["after_new"]["per_class_accuracy"]) == {"new0", "new1"}
    assert 0.0 <= audit["h_old_new"] <= 1.0
    assert "old_forgetting_vs_before_adapter" in audit


def test_predictor_has_only_all_registered_per_sample_interface() -> None:
    views, labels, ranks = _support(("a", "b"), 5)
    state = fit_locked(
        _artifact(views), labels, ranks, k_shot=5, hyperparameters=_hp()
    ).state
    single = {name: rows[:1] for name, rows in views.items()}
    predictions, scores = predict_all_registered(
        state, _artifact(single, seed=204)
    )
    assert predictions.shape == (1,)
    assert scores.shape == (1, 2)
    with pytest.raises(TrainableLowRankAdapterError, match="exactly one"):
        predict_all_registered(
            state,
            _artifact({name: rows[:2] for name, rows in views.items()}, seed=205),
        )


def test_fixed_iq_views_do_not_increase_physical_k() -> None:
    views, labels, ranks = _support(("a", "b"), 5)
    duplicated_labels = np.repeat(labels, 2)
    duplicated_ranks = np.repeat(ranks, 2)
    with pytest.raises(TrainableLowRankAdapterError, match="alignment drift"):
        fit_locked(
            _artifact(views),
            duplicated_labels,
            duplicated_ranks,
            k_shot=5,
            hyperparameters=_hp(),
        )


def test_k5_cannot_reach_k10_surplus_rows() -> None:
    views, labels, ranks = _support(("a", "b"), 10)
    with pytest.raises(TrainableLowRankAdapterError, match="strict physical K-shot"):
        fit_locked(
            _artifact(views),
            labels,
            ranks,
            k_shot=5,
            hyperparameters=_hp(),
        )


def test_more_than_three_views_fail_closed() -> None:
    views, labels, ranks = _support(("a", "b"), 5)
    views.update({"v2": views["base"], "v3": views["base"]})
    with pytest.raises(TrainableLowRankAdapterError, match="artifact drift"):
        _artifact(views)


def test_ordinary_mapping_and_wrong_bindings_fail_closed() -> None:
    assert not hasattr(d11, "build_validated_feature_artifact")
    views, labels, ranks = _support(("a", "b"), 5)
    with pytest.raises(TrainableLowRankAdapterError, match="validated artifact"):
        fit_locked(views, labels, ranks, k_shot=5, hyperparameters=_hp())
    before = fit_locked(
        _artifact(views), labels, ranks, k_shot=5, hyperparameters=_hp()
    ).state
    new_views, new_labels, new_ranks = _support(("n0", "n1"), 5, seed=88)
    with pytest.raises(TrainableLowRankAdapterError, match="fingerprint mismatch"):
        register_new_classes(
            before,
            _artifact(new_views, seed=300),
            new_labels,
            new_ranks,
            k_shot=5,
            expected_old_support_feature_artifact_sha256="0" * 64,
        )
    with pytest.raises(TrainableLowRankAdapterError, match="binding mismatch"):
        predict_all_registered(
            before,
            _artifact(
                {name: rows[:1] for name, rows in views.items()},
                seed=301,
                runtime_sha256="d" * 64,
            ),
        )


def test_oversized_state_fails_at_construction() -> None:
    feature_dim = 32
    hp = _hp()
    u = np.zeros((feature_dim, hp.rank), dtype=np.float32)
    v = np.zeros_like(u)
    gate = np.zeros(hp.rank, dtype=np.float32)
    prototypes = np.zeros((3000, feature_dim), dtype=np.float32)
    actual = u.nbytes + v.nbytes + gate.nbytes + prototypes.nbytes
    with pytest.raises(TrainableLowRankAdapterError, match="resource audit drift"):
        TrainableLowRankAdapterState(
            schema="x",
            candidate_id="x",
            classes=tuple(f"c{i}" for i in range(len(prototypes))),
            prototypes=prototypes,
            low_rank_u=u,
            low_rank_v=v,
            gate=gate,
            hyperparameters=hp,
            feature_dim=feature_dim,
            k_shot=5,
            view_ids=("base",),
            old_class_count=len(prototypes),
            registration_generation=0,
            resource={"persistent_state_bytes": actual},
            support_feature_artifact_sha256=HASH_A,
            sealed_runtime_sha256=HASH_A,
            feature_code_sha256=HASH_B,
            sealed_phase1_checkpoint_sha256=HASH_C,
        )


def test_state_arrays_are_bytes_backed_readonly_and_content_hashed() -> None:
    views, labels, ranks = _support(("a", "b"), 5)
    state = fit_locked(
        _artifact(views), labels, ranks, k_shot=5, hyperparameters=_hp()
    ).state
    assert len(state.state_content_sha256) == 64
    with pytest.raises(ValueError):
        state.prototypes[0, 0] = state.prototypes[0, 0] + 1.0
    with pytest.raises(ValueError):
        state.prototypes.setflags(write=True)
    with pytest.raises(TrainableLowRankAdapterError, match="content SHA mismatch"):
        TrainableLowRankAdapterState(
            schema=state.schema,
            candidate_id=state.candidate_id,
            classes=state.classes,
            prototypes=state.prototypes,
            low_rank_u=state.low_rank_u,
            low_rank_v=state.low_rank_v,
            gate=state.gate,
            hyperparameters=state.hyperparameters,
            feature_dim=state.feature_dim,
            k_shot=state.k_shot,
            view_ids=state.view_ids,
            old_class_count=state.old_class_count,
            registration_generation=state.registration_generation,
            resource=state.resource,
            support_feature_artifact_sha256=(
                state.support_feature_artifact_sha256
            ),
            sealed_runtime_sha256=state.sealed_runtime_sha256,
            feature_code_sha256=state.feature_code_sha256,
            sealed_phase1_checkpoint_sha256=(
                state.sealed_phase1_checkpoint_sha256
            ),
            state_content_sha256="0" * 64,
        )
