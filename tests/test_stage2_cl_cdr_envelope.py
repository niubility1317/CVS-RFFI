from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_cl_cdr_envelope as clcdr
from cvsrffi.stage2_cl_cdr_envelope import (
    ClCdrEnvelopeError,
    ClCdrHyperparameters,
    _normalize,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
    predict_all_registered,
)
from cvsrffi.stage2_joint_residual_logit_head import (
    _build_runtime_authorized_feature_artifact_internal,
)


def _support(classes, k, dim=24, seed=7, noise=0.12):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.asarray(classes), k)
    ranks = np.tile(np.arange(k, dtype=np.int64), len(classes))
    rows = np.concatenate(
        [centers[i][None, :] + noise * rng.normal(size=(k, dim)) for i in range(len(classes))]
    ).astype(np.float32)
    return rows, labels, ranks


def _artifact(rows, seed=101, runtime="a" * 64):
    rng = np.random.default_rng(seed)
    iq = rng.normal(size=(len(rows), 2, 12)).astype(np.float32)
    parents = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in iq]
    cursor = 0

    def extract(_):
        nonlocal cursor
        value = rows[cursor:cursor + 1]
        cursor += 1
        return value

    return _build_runtime_authorized_feature_artifact_internal(
        iq,
        physical_sample_ids=[f"sid_{seed}_{i}" for i in range(len(rows))],
        parent_received_iq_sha256=parents,
        sealed_runtime_sha256=runtime,
        feature_code_sha256="b" * 64,
        sealed_phase1_checkpoint_sha256="c" * 64,
        extract_single_received_iq=extract,
        operator_id="base",
        view_seed=0,
    )


def _before_after(k=10, noise=0.12):
    old = _support(("old0", "old1", "old2", "old3"), k, noise=noise)
    new = _support(("new0", "new1"), k, seed=13, noise=noise)
    joint_rows = np.concatenate([old[0], new[0]])
    return (
        _artifact(old[0]), old[1], old[2],
        _artifact(joint_rows), np.concatenate([old[1], new[1]]),
        np.concatenate([old[2], new[2]])
    )


def _hp(rank=8, stability=0.5):
    return ClCdrHyperparameters(
        candidate_id=f"clcdr_r{rank}",
        rank=rank,
        shrink=0.5,
        ridge=0.01,
        gamma=0.05,
        min_stability=stability,
    )


def _zero():
    return ClCdrHyperparameters(
        candidate_id="clcdr_z0", rank=0, shrink=1.0, ridge=0.01,
        gamma=0.0, min_stability=1.0, force_zero=True
    )


def _locked_base(rows, state):
    z = _normalize(rows)
    old = z @ state.prototypes[: state.old_class_count].T
    if len(state.classes) == state.old_class_count:
        return old
    new = z @ state.prototypes[state.old_class_count :].T
    return np.concatenate([old, new], axis=1)


def test_rank_is_bounded_and_inner_loo_reselects_dimensions() -> None:
    fitted = fit_before_after_locked(*_before_after(), k_shot=10, hyperparameters=_hp())
    assert fitted.after_state.selected_dims.shape == (6, 8)
    after_trace = fitted.trace[1]["diagnostics"]
    assert all(len(row["inner_loo_reselected_dims"]) == 60 for row in after_trace)
    assert all(len(dims) == 8 for row in after_trace for dims in row["inner_loo_reselected_dims"])
    for row in after_trace:
        assert (
            row["stability_policy"]
            == "minimum_nested_l2o_pairwise_consensus"
        )
        if row["enabled"]:
            expected = min(
                value["stability"]
                for value in row["class_safety"]["nested_l2o_calibration"]
            )
            assert row["stability"] == pytest.approx(expected)


def test_nested_l2o_calibration_excludes_evaluated_row_from_every_model() -> None:
    rows, labels, _ = _support(
        ("c0", "c1", "c2"), 5, dim=18, seed=919, noise=0.08
    )
    classes = tuple(sorted(np.unique(labels).tolist()))
    calibration = clcdr._nested_l2o_calibration(
        rows, labels, classes, 0, 0, _hp(stability=0.0)
    )
    assert calibration is not None
    changed = np.array(rows, copy=True)
    changed[0] *= -1000.0
    attacked = clcdr._nested_l2o_calibration(
        changed, labels, classes, 0, 0, _hp(stability=0.0)
    )
    assert attacked is not None
    assert calibration["calibration_excludes_evaluated_record"] is True
    assert calibration["every_calibration_model_excludes_evaluated_record"] is True
    for key in (
        "q_pos", "q_neg", "gap", "mid", "half_gap", "stability",
        "nested_reselected_dims",
    ):
        assert calibration[key] == attacked[key]


def test_unstable_classes_fall_back_to_identity() -> None:
    fitted = fit_before_after_locked(
        *_before_after(noise=0.9), k_shot=10, hyperparameters=_hp(rank=16, stability=1.0)
    )
    assert np.any(~fitted.after_state.enabled)
    disabled = np.flatnonzero(~fitted.after_state.enabled)
    assert np.all(fitted.after_state.selected_dims[disabled] == -1)


def test_cross_fitted_safety_requires_non_degrade_zero_capture_and_gap() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    )
    for row in fitted.trace[0]["diagnostics"] + fitted.trace[1]["diagnostics"]:
        if row["enabled"]:
            safety = row["class_safety"]
            assert safety["own_correct_candidate"] >= safety["own_correct_base"]
            assert safety["zero_added_capture"] is True
            assert all(
                value == 0
                for value in safety[
                    "added_capture_by_other_truth_class"
                ].values()
            )
            assert safety["llr_gap"] >= safety["min_llr_gap"]
    blocked = fit_before_after_locked(
        *_before_after(),
        k_shot=10,
        hyperparameters=ClCdrHyperparameters(
            candidate_id="gap_block",
            rank=8,
            shrink=0.5,
            ridge=0.01,
            gamma=0.05,
            min_stability=0.0,
            min_llr_gap=100.0,
        ),
    )
    assert not np.any(blocked.after_state.enabled)


def test_conformal_mid_half_gap_are_persisted_and_h_is_clipped() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    )
    state_by_phase = (
        (fitted.before_state, fitted.trace[0]["diagnostics"]),
        (fitted.after_state, fitted.trace[1]["diagnostics"]),
    )
    for state, diagnostics in state_by_phase:
        for row in diagnostics:
            index = state.classes.index(row["class_handle"])
            if row["enabled"]:
                safety = row["class_safety"]
                assert state.llr_mid[index] == pytest.approx(
                    0.5
                    * (
                        safety["own_llr_low_quantile"]
                        + safety["rest_llr_high_quantile"]
                    ),
                    abs=1.0e-6,
                )
                assert state.llr_half_gap[index] == pytest.approx(
                    0.5 * safety["llr_gap"], abs=1.0e-6
                )
                assert state.llr_half_gap[index] > 0.0
            else:
                assert state.llr_mid[index] == 0.0
                assert state.llr_half_gap[index] == 0.0

    state = fitted.after_state
    probes = np.random.default_rng(81).normal(
        size=(300, state.feature_dim)
    ).astype(np.float32)
    base = _locked_base(probes, state)
    correction = _score_numpy(probes, state) - base
    assert np.max(np.abs(correction)) <= state.hyperparameters.gamma + 1.0e-6
    index = int(np.flatnonzero(state.enabled)[0])
    z = _normalize(probes)
    dims = state.selected_dims[index]
    current = z[:, dims]
    llr = -0.5 * np.mean(
        np.square(current - state.class_mean[index]) / state.class_var[index]
        - np.square(current - state.rest_mean[index]) / state.rest_var[index]
        + np.log(state.class_var[index] / state.rest_var[index]),
        axis=1,
    )
    rival_pool = (
        base[:, : state.old_class_count]
        if index < state.old_class_count
        else base
    )
    rival = np.max(
        np.concatenate(
            [rival_pool[:, :index], rival_pool[:, index + 1 :]], axis=1
        ),
        axis=1,
    )
    active = base[:, index] - rival >= -state.hyperparameters.margin_band
    expected = np.zeros(len(probes), dtype=np.float32)
    expected[active] = (
        np.float32(state.hyperparameters.gamma)
        * np.clip(
            (llr[active] - state.llr_mid[index])
            / (state.llr_half_gap[index] + clcdr.EPS),
            -1.0,
            1.0,
        ).astype(np.float32)
    )
    np.testing.assert_allclose(correction[:, index], expected, atol=1.0e-7)


def test_conformal_arrays_are_state_hashed_and_resource_accounted() -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    assert state.resource["persistent_array_state_bytes"] == clcdr._array_bytes(
        state
    )
    changed = np.array(state.llr_mid, copy=True)
    changed[np.flatnonzero(state.enabled)[0]] += np.float32(0.01)
    with pytest.raises(ClCdrEnvelopeError, match="state content SHA mismatch"):
        replace(state, llr_mid=changed)


def test_margin_band_uses_uncorrected_base_and_skips_far_losers() -> None:
    state = fit_before_after_locked(
        *_before_after(),
        k_shot=10,
        hyperparameters=ClCdrHyperparameters(
            candidate_id="band0",
            rank=8,
            shrink=0.5,
            ridge=0.01,
            gamma=0.05,
            min_stability=0.0,
            margin_band=0.0,
        ),
    ).after_state
    probes = np.random.default_rng(123).normal(
        size=(200, state.feature_dim)
    ).astype(np.float32)
    base = _locked_base(probes, state)
    scored = _score_numpy(probes, state)
    for index in np.flatnonzero(state.enabled)[
        np.flatnonzero(state.enabled) >= state.old_class_count
    ]:
        rival = np.max(
            np.concatenate([base[:, :index], base[:, index + 1 :]], axis=1),
            axis=1,
        )
        inactive = base[:, index] < rival
        np.testing.assert_array_equal(scored[inactive, index], base[inactive, index])


def test_after_old_envelope_and_scores_are_bitwise_locked() -> None:
    fitted = fit_before_after_locked(*_before_after(), k_shot=10, hyperparameters=_hp())
    old = fitted.after_state.old_class_count
    assert np.any(fitted.before_state.enabled)
    assert np.any(fitted.after_state.enabled)
    for name in (
        "selected_dims", "class_mean", "class_var", "rest_mean", "rest_var",
        "enabled", "stability", "llr_mid", "llr_half_gap",
    ):
        np.testing.assert_array_equal(getattr(fitted.before_state, name), getattr(fitted.after_state, name)[:old])
    rng = np.random.default_rng(10)
    probes = rng.normal(size=(12, fitted.after_state.feature_dim)).astype(np.float32)
    before = _score_numpy(probes, fitted.before_state)
    after = _score_numpy(probes, fitted.after_state)
    np.testing.assert_array_equal(after[:, :old], before)


def test_z0_after_old_base_scores_are_bitwise_locked_on_random_probes() -> None:
    fitted = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_zero()
    )
    assert not np.any(fitted.before_state.enabled)
    assert not np.any(fitted.after_state.enabled)
    probes = np.random.default_rng(1010).normal(
        size=(513, fitted.after_state.feature_dim)
    ).astype(np.float32)
    before = _score_numpy(probes, fitted.before_state)
    after = _score_numpy(probes, fitted.after_state)
    np.testing.assert_array_equal(
        after[:, : fitted.after_state.old_class_count], before
    )


def test_joint_l2o_and_held_mutation_do_not_feed_train_state() -> None:
    args = list(_before_after())
    original, trace = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    assert len(original["folds"]) == 5
    assert original["old_score_bitwise_locked"] is True
    assert any(row["phase"] == "after_cl_cdr_fit" for row in trace)
    changed_old = np.array(args[0].features, copy=True)
    changed_joint = np.array(args[3].features, copy=True)
    old_held = np.isin(np.asarray(args[2]), (0, 1))
    joint_held = np.isin(np.asarray(args[5]), (0, 1))
    changed_old[old_held] *= -30.0
    changed_joint[joint_held] *= -30.0
    args[0], args[3] = _artifact(changed_old), _artifact(changed_joint)
    changed, _ = evaluate_joint_leave_two_out(*args, hyperparameters=_hp())
    for key in ("before_selection_sha", "after_selection_sha", "calibration_sha", "enabled", "stability"):
        assert original["folds"][0][key] == changed["folds"][0][key]


def test_zero_is_alpha0_and_state_under_80kib() -> None:
    state = fit_before_after_locked(*_before_after(), k_shot=10, hyperparameters=_zero()).after_state
    assert not np.any(state.enabled)
    assert state.selected_dims.shape == (6, 0)
    probes = np.random.default_rng(4).normal(size=(7, state.feature_dim)).astype(np.float32)
    expected = _locked_base(probes, state)
    np.testing.assert_array_equal(_score_numpy(probes, state), expected.astype(np.float32))
    assert state.resource["persistent_array_state_bytes"] < 80 * 1024
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0


def test_k1_uses_only_its_single_physical_support_and_falls_back_to_identity() -> None:
    args = _before_after(k=1)
    fitted = fit_before_after_locked(
        *args, k_shot=1, hyperparameters=_hp()
    )
    assert fitted.before_state.k_shot == 1
    assert fitted.after_state.k_shot == 1
    assert fitted.before_state.hyperparameters.force_zero is True
    assert fitted.after_state.hyperparameters.force_zero is True
    assert fitted.before_state.hyperparameters.rank == 0
    assert fitted.after_state.hyperparameters.rank == 0
    assert fitted.before_state.hyperparameters.gamma == 0.0
    assert fitted.after_state.hyperparameters.gamma == 0.0
    assert fitted.before_state.selected_dims.shape == (4, 0)
    assert fitted.after_state.selected_dims.shape == (6, 0)
    assert not np.any(fitted.before_state.enabled)
    assert not np.any(fitted.after_state.enabled)
    assert all(not phase["diagnostics"] for phase in fitted.trace)
    probes = np.random.default_rng(501).normal(
        size=(9, fitted.after_state.feature_dim)
    ).astype(np.float32)
    expected = _locked_base(probes, fitted.after_state)
    np.testing.assert_array_equal(
        _score_numpy(probes, fitted.after_state), expected.astype(np.float32)
    )
    expected_after_ids = {
        (str(args[4][i]), int(args[5][i]), args[3].physical_sample_ids[i])
        for i in range(len(args[4]))
    }
    assert len(expected_after_ids) == len(fitted.after_state.classes)
    assert fitted.after_state.support_selection_sha256 == clcdr._selection_sha(
        args[3],
        np.asarray(args[4]).astype(str),
        np.asarray(args[5], dtype=np.int64),
        np.ones(len(args[4]), dtype=bool),
    )


def test_state_npz_json_roundtrip_is_hash_pinned_and_score_exact(
    tmp_path,
) -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    npz_path = tmp_path / "clcdr_state.npz"
    metadata_path = tmp_path / "clcdr_state.json"
    hashes = clcdr.save_cl_cdr_state(state, npz_path, metadata_path)
    with pytest.raises(ClCdrEnvelopeError, match="already exists"):
        clcdr.save_cl_cdr_state(state, npz_path, metadata_path)
    loaded = clcdr.load_cl_cdr_state(
        npz_path,
        metadata_path,
        expected_npz_sha256=hashes["npz_sha256"],
        expected_metadata_sha256=hashes["metadata_sha256"],
    )
    assert loaded.state_content_sha256 == state.state_content_sha256
    for name in (
        "prototypes", "selected_dims", "class_mean", "class_var",
        "rest_mean", "rest_var", "enabled", "stability", "llr_mid",
        "llr_half_gap",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(state, name))
    probes = np.random.default_rng(888).normal(
        size=(47, state.feature_dim)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        _score_numpy(probes, loaded), _score_numpy(probes, state)
    )
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ClCdrEnvelopeError, match="external hash"):
        clcdr.load_cl_cdr_state(
            npz_path,
            metadata_path,
            expected_npz_sha256=hashes["npz_sha256"],
            expected_metadata_sha256=hashes["metadata_sha256"],
        )
    missing = tmp_path / "missing.npz"
    with pytest.raises(ClCdrEnvelopeError, match="file unavailable"):
        clcdr.load_cl_cdr_state(
            missing,
            metadata_path,
            expected_npz_sha256="0" * 64,
            expected_metadata_sha256=clcdr._sha256_path(metadata_path),
        )


def test_state_loader_rejects_extra_metadata_and_hp_keys(tmp_path) -> None:
    state = fit_before_after_locked(
        *_before_after(), k_shot=10, hyperparameters=_hp()
    ).after_state
    for target in ("top", "hp"):
        npz_path = tmp_path / f"{target}.npz"
        metadata_path = tmp_path / f"{target}.json"
        clcdr.save_cl_cdr_state(state, npz_path, metadata_path)
        import json
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if target == "top":
            metadata["unexpected"] = 1
            message = "metadata key drift"
        else:
            metadata["hyperparameters"]["unexpected"] = 1
            message = "hyperparameter key drift"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with pytest.raises(ClCdrEnvelopeError, match=message):
            clcdr.load_cl_cdr_state(
                npz_path,
                metadata_path,
                expected_npz_sha256=clcdr._sha256_path(npz_path),
                expected_metadata_sha256=clcdr._sha256_path(metadata_path),
            )


def test_mapping_exact_k_and_runtime_binding_fail_closed() -> None:
    args = list(_before_after())
    args[0] = {"base": np.zeros((40, 24), dtype=np.float32)}
    with pytest.raises(ClCdrEnvelopeError, match="authorized"):
        fit_before_after_locked(*args, k_shot=10, hyperparameters=_hp())
    with pytest.raises(ClCdrEnvelopeError, match="strict physical"):
        fit_before_after_locked(*_before_after(), k_shot=5, hyperparameters=_hp())
    state = fit_before_after_locked(*_before_after(), k_shot=10, hyperparameters=_hp()).after_state
    rows = _support(("probe", "unused"), 1)[0][:1]
    with pytest.raises(ClCdrEnvelopeError, match="binding"):
        predict_all_registered(state, _artifact(rows, runtime="d" * 64))


def test_prediction_is_one_sample_all_registered_and_readonly() -> None:
    state = fit_before_after_locked(*_before_after(), k_shot=10, hyperparameters=_hp()).after_state
    rows = _support(("probe", "unused"), 1)[0]
    prediction, scores = predict_all_registered(state, _artifact(rows[:1]))
    assert prediction.shape == (1,)
    assert scores.shape == (1, 6)
    with pytest.raises(ClCdrEnvelopeError, match="exactly one"):
        predict_all_registered(state, _artifact(rows))
    with pytest.raises(ValueError):
        state.class_mean[0, 0] += 1.0


def test_module_has_no_query_oracle_or_public_artifact_factory() -> None:
    source = inspect.getsource(clcdr)
    assert "_ARTIFACT_TOKEN" not in source
    assert "_select_artifact" not in source
    assert "query_role" not in source
    assert "query_quota" not in source
    assert not hasattr(clcdr, "build_runtime_authorized_feature_artifact")
