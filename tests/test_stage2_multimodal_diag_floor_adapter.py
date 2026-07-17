from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_multimodal_diag_floor_adapter import (
    BLOCK_DIMS,
    DEFAULT_BLOCK_ENERGY,
    FEATURE_DIM,
    D25C3Config,
    D25C3Error,
    D25C3LossWeights,
    append_stage2c_new_suffix,
    fit_stage2b_diag_floor,
    predict_one,
    project_block_centered_gamma,
    score_one,
    transform_concat288,
)


OLD = ("old-c", "old-a", "old-b")
NEW = ("new-y", "new-x")


def _weights() -> D25C3LossWeights:
    # These are explicit C3 loss coefficients, not the Phase1
    # 0.07/0.63/0.30 labeled/unlabeled/validation data split.
    return D25C3LossWeights(
        equal_class_ce=1.0,
        tail_cvar=0.5,
        hard_negative_margin=0.25,
        proximity=0.05,
    )


def _config(*, before: int = 2, after: int = 0) -> D25C3Config:
    return D25C3Config(
        loss_weights=_weights(),
        stage2b_steps=before,
        stage2c_steps=after,
        learning_rate=0.02,
    )


def _concat_support(
    classes: tuple[str, ...], k: int, *, seed: int
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    offsets = np.cumsum((0,) + BLOCK_DIMS)
    for class_index, label in enumerate(classes):
        centers = [rng.normal(size=dimension) for dimension in BLOCK_DIMS]
        centers = [value / np.linalg.norm(value) for value in centers]
        for _ in range(k):
            row = np.zeros(FEATURE_DIM, dtype=np.float32)
            for block_index, energy in enumerate(DEFAULT_BLOCK_ENERGY):
                start, stop = offsets[block_index], offsets[block_index + 1]
                value = centers[block_index] + 0.08 * rng.normal(
                    size=BLOCK_DIMS[block_index]
                )
                value /= np.linalg.norm(value)
                row[start:stop] = value * np.sqrt(energy)
            rows.append(row)
            labels.append(label)
        # Make class centers deterministic but distinct without relying on
        # lexical registry order.
        rng = np.random.default_rng(seed + 101 * (class_index + 1))
    return np.stack(rows), labels


def _fit(*, k: int = 3, before: int = 2, after: int = 0):
    rows, labels = _concat_support(OLD, k, seed=71)
    return fit_stage2b_diag_floor(
        rows, labels, OLD, config=_config(before=before, after=after)
    )


def test_loss_weights_are_explicit_and_not_phase1_split_defaults() -> None:
    with pytest.raises(TypeError):
        D25C3Config()  # type: ignore[call-arg]
    config = _config()
    payload = config.lock_payload()["loss_weights"]
    assert payload["phase1_split_semantics"] is False
    assert payload == {
        "equal_class_ce": 1.0,
        "tail_cvar": 0.5,
        "hard_negative_margin": 0.25,
        "proximity": 0.05,
        "phase1_split_semantics": False,
    }


def test_gamma_projection_is_block_centered_and_clipped() -> None:
    raw = np.linspace(-4.0, 3.0, FEATURE_DIM, dtype=np.float32)
    projected = project_block_centered_gamma(raw)
    assert projected.shape == (FEATURE_DIM,)
    assert not projected.flags.writeable
    assert np.max(np.abs(projected)) <= 0.35 + 1.0e-7
    start = 0
    for dimension in BLOCK_DIMS:
        assert abs(float(np.sum(projected[start : start + dimension]))) < 3.0e-6
        start += dimension


def test_transform_preserves_all_three_fixed_block_energies() -> None:
    rows, _ = _concat_support(OLD, 2, seed=72)
    gamma = project_block_centered_gamma(
        np.random.default_rng(73).normal(size=FEATURE_DIM).astype(np.float32)
    )
    transformed = transform_concat288(rows, gamma)
    assert transformed.dtype == np.float32
    start = 0
    for dimension, energy in zip(BLOCK_DIMS, DEFAULT_BLOCK_ENERGY):
        block = transformed[:, start : start + dimension]
        np.testing.assert_allclose(
            np.sum(block.astype(np.float64) ** 2, axis=1), energy, atol=2.0e-6
        )
        start += dimension
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=2e-6)


def test_stage2b_trains_exactly_288_shared_parameters_and_logs_margin() -> None:
    result = _fit(before=3)
    state = result.state
    assert state.gamma.shape == (288,)
    assert state.stage2b_optimizer_steps == 3
    assert len(result.training_trace) == 3
    assert all(row["full_batch"] is True for row in result.training_trace)
    assert all(row["query_rows_used"] == 0 for row in result.training_trace)
    assert all("hard_negative_margin" in row for row in result.training_trace)
    assert all(len(row["shared_gamma_sha256"]) == 64 for row in result.training_trace)
    assert all(len(row["per_class_ce"]) == len(OLD) for row in result.training_trace)
    audit = state.resource_audit()
    assert audit["shared_adapter_trainable_parameters"] == 288
    assert audit["stage2b_trainable_parameters"] == 288
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["compute_dtype"] == "fp32"
    assert audit["temporary_bytes_upper_bound"] == 3 * 288 * 4 + len(OLD) * 4


def test_two_step_fit_persists_projected_gamma_without_bitwise_reprojection() -> None:
    state = _fit(before=2).state
    assert state.stage2b_optimizer_steps == 2
    assert np.max(np.abs(state.gamma)) <= 0.35 + 1.0e-7
    start = 0
    for dimension in BLOCK_DIMS:
        assert abs(float(np.sum(state.gamma[start : start + dimension]))) < 3.0e-6
        start += dimension


def test_k1_falls_back_to_zero_step_identity_gamma() -> None:
    result = _fit(k=1, before=20)
    assert result.state.stage2b_optimizer_steps == 0
    np.testing.assert_array_equal(result.state.gamma, np.zeros(288, dtype=np.float32))
    assert result.training_trace[0]["status"] == "K1_IDENTITY_FALLBACK"


def test_stage2c_default_append_freezes_shared_and_old_prefix_bitwise() -> None:
    before = _fit(before=2, after=0).state
    probe, _ = _concat_support((OLD[0],), 1, seed=74)
    score_before = score_one(before, probe[0])
    new_rows, new_labels = _concat_support(NEW, before.k_shot, seed=75)
    result = append_stage2c_new_suffix(before, new_rows, new_labels, NEW)
    after = result.state
    assert after.stage2c_optimizer_steps == 0
    assert result.training_trace[0]["status"] == "ZERO_STEP_TARGET_ONLY_APPEND"
    assert after.shared_sha256 == before.shared_sha256
    assert after.old_prefix_sha256 == before.old_prefix_sha256
    assert after.gamma.tobytes() == before.gamma.tobytes()
    assert (
        after.prototypes[: before.old_class_count].tobytes()
        == before.prototypes.tobytes()
    )
    np.testing.assert_array_equal(score_one(after, probe[0])[: len(OLD)], score_before)


def test_optional_stage2c_updates_only_new_suffix_and_counts_it_separately() -> None:
    before = _fit(before=2, after=2).state
    new_rows, new_labels = _concat_support(NEW, before.k_shot, seed=76)
    result = append_stage2c_new_suffix(before, new_rows, new_labels, NEW)
    after = result.state
    assert after.stage2c_optimizer_steps == 2
    assert len(result.training_trace) == 2
    assert all(row["updated_state"] == "new_prototype_suffix_only" for row in result.training_trace)
    assert all("hard_negative_margin" in row for row in result.training_trace)
    assert all(len(row["new_suffix_sha256"]) == 64 for row in result.training_trace)
    assert after.gamma.tobytes() == before.gamma.tobytes()
    assert after.prototypes[: len(OLD)].tobytes() == before.prototypes.tobytes()
    audit = after.resource_audit()
    assert audit["shared_adapter_trainable_parameters"] == 288
    assert audit["stage2c_optional_new_suffix_parameters"] == len(NEW) * 288
    assert audit["total_optimizer_steps"] == 4
    assert audit["total_adaptation_epochs"] == 4
    assert audit["resource_tier"] == "FORMAL_DEPLOYMENT"


def test_repeated_stage2c_append_fails_closed() -> None:
    before = _fit(before=2, after=0).state
    new_rows, new_labels = _concat_support(NEW, before.k_shot, seed=761)
    after = append_stage2c_new_suffix(before, new_rows, new_labels, NEW).state
    more_rows, more_labels = _concat_support(("new-z",), before.k_shot, seed=762)
    with pytest.raises(D25C3Error, match="repeated append"):
        append_stage2c_new_suffix(after, more_rows, more_labels, ("new-z",))


def test_resource_tier_marks_20_plus_20_as_exploration_and_rejects_over_45() -> None:
    result = _fit(before=20, after=20)
    new_rows, new_labels = _concat_support(NEW, result.state.k_shot, seed=763)
    after = append_stage2c_new_suffix(
        result.state, new_rows, new_labels, NEW
    ).state
    audit = after.resource_audit()
    assert audit["stage2b_adaptation_epochs"] == 20
    assert audit["stage2c_adaptation_epochs"] == 20
    assert audit["total_adaptation_epochs"] == 40
    assert audit["formal_adaptation_epoch_limit"] == 30
    assert audit["formal_adaptation_epoch_limit_pass"] is False
    assert audit["exploration_150pct_adaptation_epoch_limit"] == 45
    assert audit["exploration_150pct_adaptation_epoch_limit_pass"] is True
    assert audit["resource_tier"] == "PERFORMANCE_EXPLORATION_150PCT"
    with pytest.raises(D25C3Error, match="45 adaptation-epoch"):
        D25C3Config(
            loss_weights=_weights(), stage2b_steps=20, stage2c_steps=26
        )


def test_old_score_freeze_does_not_claim_old_prediction_non_forgetting() -> None:
    old_rows, old_labels = _concat_support(OLD, 3, seed=764)
    before = fit_stage2b_diag_floor(
        old_rows, old_labels, OLD, config=_config(before=0, after=0)
    ).state
    probe = old_rows[0]
    before_label, before_scores = predict_one(before, probe)
    assert before_label == OLD[0]
    # A registered new class may legitimately occupy the probe more closely
    # than the old class mean.  Frozen old raw scores therefore do not imply a
    # frozen all-class argmax after registration.
    new_rows = np.repeat(probe[None, :], before.k_shot, axis=0)
    new_labels = ["new-collision"] * before.k_shot
    after = append_stage2c_new_suffix(
        before, new_rows, new_labels, ("new-collision",)
    ).state
    after_label, after_scores = predict_one(after, probe)
    np.testing.assert_array_equal(after_scores[: len(OLD)], before_scores)
    assert after_label == "new-collision"
    audit = after.resource_audit()
    assert audit["old_raw_score_prefix_frozen"] is True
    assert audit["old_prediction_non_forgetting_guaranteed"] is False
    assert audit["requires_runner_old_support_non_degradation_gate"] is True


def test_step_and_clip_hard_limits_fail_closed() -> None:
    for kwargs, match in (
        ({"stage2b_steps": 21}, "Stage2-B"),
        ({"stage2c_steps": 31}, "Stage2-C"),
        ({"gamma_clip": 0.36}, "0.35"),
    ):
        with pytest.raises(D25C3Error, match=match):
            D25C3Config(loss_weights=_weights(), **kwargs)


def test_public_fit_signatures_have_no_query_or_oracle_inputs() -> None:
    forbidden = ("query", "truth", "role", "quota", "assignment", "source", "clean")
    for function in (fit_stage2b_diag_floor, append_stage2c_new_suffix):
        names = inspect.signature(function).parameters
        assert not any(token in name.lower() for name in names for token in forbidden)


def test_scoring_is_one_sample_all_registered_and_input_energy_is_checked() -> None:
    before = _fit(before=0).state
    new_rows, new_labels = _concat_support(NEW, before.k_shot, seed=77)
    after = append_stage2c_new_suffix(before, new_rows, new_labels, NEW).state
    label, scores = predict_one(after, new_rows[0])
    assert label in after.classes
    assert scores.shape == (len(OLD) + len(NEW),)
    assert scores.dtype == np.float32
    assert not scores.flags.writeable
    with pytest.raises(D25C3Error, match="exactly one"):
        score_one(after, new_rows[:2])
    broken = new_rows.copy()
    broken[:, :160] *= 0.5
    with pytest.raises(D25C3Error, match="block energy"):
        transform_concat288(broken, after.gamma)


def test_query_path_has_no_explicit_float64_casts() -> None:
    transform_source = inspect.getsource(transform_concat288)
    score_source = inspect.getsource(score_one)
    validator = inspect.getsource(
        __import__(
            "cvsrffi.stage2_multimodal_diag_floor_adapter", fromlist=["_validate_concat_rows"]
        )._validate_concat_rows
    )
    for source in (transform_source, score_source, validator):
        assert "float64" not in source
    assert "astype(np.float64)" not in score_source
