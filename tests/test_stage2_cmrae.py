from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

import cvsrffi.stage2_cmrae as d18
from cvsrffi.stage2_cmrae import (
    CmraeError,
    CmraeSceneSupport,
    _apply_equalizer_iq,
    _bounded_zero_mean_log_gain,
    _build_runtime_authorized_received_iq_artifact_internal,
    _fit_common_coefficients,
    _seal_runtime_authorized_backbone_internal,
    evaluate_k10_outer_l2o,
    fit_before_after_locked,
    load_state_bytes,
    predict_scores,
    preregistered_candidates,
    select_k10_candidate_three_scene,
    serialize_state_bytes,
)


RUNTIME = "a" * 64
FEATURE_CODE = "b" * 64
CHECKPOINT = "c" * 64
AUTHORITY_ANCHOR = "f" * 64


def _extract(iq: np.ndarray) -> np.ndarray:
    z = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    spectrum = np.fft.fft(z, axis=1)
    return np.concatenate(
        [np.abs(spectrum[:, :12]), z.real[:, :4], z.imag[:, :4]], axis=1
    ).astype(np.float32)


def _backbone():
    return _seal_runtime_authorized_backbone_internal(
        _extract,
        feature_code_sha256=FEATURE_CODE,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
    )


def _artifact(iq: np.ndarray, scene: str, *, prefix: str, purpose: str = "support"):
    iq = np.ascontiguousarray(iq, dtype=np.float32)
    parents = [hashlib.sha256(row.tobytes()).hexdigest() for row in iq]
    seed_base = int(hashlib.sha256(prefix.encode()).hexdigest()[:8], 16)
    return _build_runtime_authorized_received_iq_artifact_internal(
        iq,
        physical_sample_ids=[f"{prefix}_physical_{i}" for i in range(len(iq))],
        parent_received_iq_sha256=parents,
        overlay_tokens=[f"{prefix}::overlay-token::{i}" for i in range(len(iq))],
        source_leo_provenance_sha256=[
            hashlib.sha256(f"{prefix}::source-provenance::{i}".encode()).hexdigest()
            for i in range(len(iq))
        ],
        source_leo_cache_sha256=[
            hashlib.sha256(f"{prefix}::source-cache".encode()).hexdigest()
            for _ in range(len(iq))
        ],
        target_channel_views=[scene] * len(iq),
        satellite_seeds=[seed_base + i for i in range(len(iq))],
        overlay_provenance_sha256=[
            hashlib.sha256(f"{prefix}_overlay_{i}".encode()).hexdigest()
            for i in range(len(iq))
        ],
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose=purpose,
    )


def _iq_for_classes(classes, k, *, seed=4, length=32):
    rng = np.random.default_rng(seed)
    n = np.arange(length, dtype=np.float64)
    rows = []
    for class_index, _ in enumerate(classes):
        tone = class_index + 2
        for rank in range(k):
            phase = rng.uniform(-np.pi, np.pi)
            carrier = np.exp(1j * (2 * np.pi * tone * n / length + phase))
            envelope = 1.0 + 0.22 * np.cos(2 * np.pi * n / length + 0.3 * class_index)
            noise = 0.015 * (rng.normal(size=length) + 1j * rng.normal(size=length))
            rows.append(envelope * carrier + noise)
    z = np.asarray(rows)
    return np.ascontiguousarray(np.stack([z.real, z.imag], axis=1), dtype=np.float32)


def _before_after(k=5, *, scene="leo_clear_weak", seed=4):
    old_classes = ("oa", "ob", "oc")
    new_classes = ("nx", "ny")
    old_iq = _iq_for_classes(old_classes, k, seed=seed)
    new_iq = _iq_for_classes(new_classes, k, seed=seed + 31)
    after_iq = np.concatenate([old_iq, new_iq])
    old_labels = np.repeat(np.asarray(old_classes), k)
    new_labels = np.repeat(np.asarray(new_classes), k)
    old_ranks = np.tile(np.arange(k), len(old_classes))
    new_ranks = np.tile(np.arange(k), len(new_classes))
    before = _artifact(old_iq, scene, prefix=f"{scene}_{seed}_old")
    # Preserve the exact old identifiers, hashes, seeds, and overlay provenance.
    temp_new = _artifact(new_iq, scene, prefix=f"{scene}_{seed}_new")
    after = _build_runtime_authorized_received_iq_artifact_internal(
        after_iq,
        physical_sample_ids=before.physical_sample_ids + temp_new.physical_sample_ids,
        parent_received_iq_sha256=(
            before.parent_received_iq_sha256 + temp_new.parent_received_iq_sha256
        ),
        overlay_tokens=before.overlay_tokens + temp_new.overlay_tokens,
        source_leo_provenance_sha256=(
            before.source_leo_provenance_sha256
            + temp_new.source_leo_provenance_sha256
        ),
        source_leo_cache_sha256=(
            before.source_leo_cache_sha256 + temp_new.source_leo_cache_sha256
        ),
        target_channel_views=[scene] * len(after_iq),
        satellite_seeds=before.satellite_seeds + temp_new.satellite_seeds,
        overlay_provenance_sha256=(
            before.overlay_provenance_sha256 + temp_new.overlay_provenance_sha256
        ),
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    return (
        before, old_labels, old_ranks,
        after, np.concatenate([old_labels, new_labels]),
        np.concatenate([old_ranks, new_ranks]),
    )


def _hp(value=0.125):
    return next(h for h in preregistered_candidates() if h.lambda_equalizer == value)


def _slice_artifact(artifact, mask, *, purpose="support"):
    indices = np.flatnonzero(mask)
    return _build_runtime_authorized_received_iq_artifact_internal(
        artifact.received_iq[indices],
        physical_sample_ids=[artifact.physical_sample_ids[i] for i in indices],
        parent_received_iq_sha256=[artifact.parent_received_iq_sha256[i] for i in indices],
        overlay_tokens=[artifact.overlay_tokens[i] for i in indices],
        source_leo_provenance_sha256=[
            artifact.source_leo_provenance_sha256[i] for i in indices
        ],
        source_leo_cache_sha256=[
            artifact.source_leo_cache_sha256[i] for i in indices
        ],
        target_channel_views=[artifact.target_channel_views[i] for i in indices],
        satellite_seeds=[artifact.satellite_seeds[i] for i in indices],
        overlay_provenance_sha256=[artifact.overlay_provenance_sha256[i] for i in indices],
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        purpose=purpose,
    )


@lru_cache(maxsize=1)
def _locked_k10_fixture():
    scenes = tuple(
        CmraeSceneSupport(
            scene, *_before_after(k=10, scene=scene, seed=1200 + index)
        )
        for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS)
    )
    return scenes, select_k10_candidate_three_scene(
        scenes, backbone=_backbone(),
        selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )


def _locked_prefix(k: int):
    scenes, selection = _locked_k10_fixture()
    scene = scenes[0]
    before_ranks = np.asarray(scene.before_ranks)
    after_ranks = np.asarray(scene.after_ranks)
    before_mask = before_ranks < k
    after_mask = after_ranks < k
    return (
        (
            _slice_artifact(scene.before_artifact, before_mask),
            np.asarray(scene.before_labels)[before_mask], before_ranks[before_mask],
            _slice_artifact(scene.after_artifact, after_mask),
            np.asarray(scene.after_labels)[after_mask], after_ranks[after_mask],
        ),
        selection,
    )


def test_received_iq_artifact_is_internal_sha_bound_and_ordinary_array_fails() -> None:
    iq = _iq_for_classes(("a", "b"), 1)
    operator_artifact = _artifact(iq, "leo_clear_weak", prefix="operator")
    assert operator_artifact.operator_id == d18.OPERATOR_ID
    assert "::" in operator_artifact.overlay_tokens[0]
    parents = [hashlib.sha256(row.tobytes()).hexdigest() for row in iq]
    parents[0] = "0" * 64
    with pytest.raises(CmraeError, match="actual received-IQ SHA"):
        _build_runtime_authorized_received_iq_artifact_internal(
            iq,
            physical_sample_ids=["p0", "p1"],
            parent_received_iq_sha256=parents,
            overlay_tokens=["token-0", "token-1"],
            source_leo_provenance_sha256=["1" * 64, "2" * 64],
            source_leo_cache_sha256=["3" * 64, "3" * 64],
            target_channel_views=["leo_clear_weak"] * 2,
            satellite_seeds=[1, 2],
            overlay_provenance_sha256=["d" * 64, "e" * 64],
            sealed_runtime_sha256=RUNTIME,
            sealed_phase1_checkpoint_sha256=CHECKPOINT,
            feature_code_sha256=FEATURE_CODE,
            purpose="support",
        )
    args = list(_before_after())
    args[0] = np.zeros((15, 2, 32), dtype=np.float32)
    with pytest.raises(CmraeError, match="support artifact"):
        fit_before_after_locked(
            *args, k_shot=5, hyperparameters=_hp(), backbone=_backbone()
        )


def test_dct8_is_class_balanced_scale_equivariant_and_zero_safe() -> None:
    a = _iq_for_classes(("a",), 2, seed=1)
    b = _iq_for_classes(("b",), 6, seed=2)
    iq = np.concatenate([a, b])
    labels = np.asarray(["a"] * 2 + ["b"] * 6)
    coeff = _fit_common_coefficients(iq, labels)
    expected = 0.5 * (
        _fit_common_coefficients(a, np.asarray(["a"] * 2))
        + _fit_common_coefficients(b, np.asarray(["b"] * 6))
    )
    np.testing.assert_allclose(coeff, expected, atol=1e-6)
    np.testing.assert_allclose(
        coeff, _fit_common_coefficients(iq * 3.7, labels), atol=4e-6
    )
    zeros = np.zeros((4, 2, 32), dtype=np.float32)
    np.testing.assert_array_equal(
        _fit_common_coefficients(zeros, np.asarray(["a", "a", "b", "b"])),
        np.zeros(8, dtype=np.float32),
    )


def test_equalizer_preserves_fft_bin_phase_and_gain_is_really_bounded() -> None:
    args = _before_after(k=10)
    fitted = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(0.25), backbone=_backbone()
    )
    state = fitted.before_state
    output = _apply_equalizer_iq(args[0], state.common_dct_coefficients, state.hyperparameters)
    source_z = args[0].received_iq[:, 0] + 1j * args[0].received_iq[:, 1]
    output_z = output[:, 0] + 1j * output[:, 1]
    source_fft = np.fft.fftshift(np.fft.fft(source_z, axis=1), axes=1)
    output_fft = np.fft.fftshift(np.fft.fft(output_z, axis=1), axes=1)
    mask = np.abs(source_fft) > 1e-4
    phase_delta = np.angle(output_fft[mask] / source_fft[mask])
    assert np.max(np.abs(phase_delta)) < 3e-5
    source_rms = np.sqrt(np.mean(np.abs(source_z) ** 2, axis=1))
    output_rms = np.sqrt(np.mean(np.abs(output_z) ** 2, axis=1))
    np.testing.assert_allclose(output_rms, source_rms, atol=2e-6, rtol=2e-6)
    gain = np.abs(output_fft[mask]) / np.abs(source_fft[mask])
    registered_bound = np.exp(4 * 0.25 * np.log(1.10))
    assert gain.min() >= 1 / registered_bound - 2e-5
    assert gain.max() <= registered_bound + 2e-5
    probe = np.linspace(-state.hyperparameters.tau, state.hyperparameters.tau, 32)
    pre_rms_log_gain = _bounded_zero_mean_log_gain(probe, state.hyperparameters)
    assert np.exp(np.mean(pre_rms_log_gain)) == pytest.approx(1.0, abs=1e-12)
    assert state.resource["cfo_estimation"] is False
    assert state.resource["cfo_derotation"] is False
    assert state.resource["fft_bin_phase_unrotated"] is True
    assert len(state.resource["dct_basis_sha256"]) == 64


def test_after_freezes_equalizer_and_old_prototypes_new_only_changes_new() -> None:
    args = list(_before_after(k=10))
    first = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(), backbone=_backbone()
    )
    after_iq = np.array(args[3].received_iq, copy=True)
    old_count = len(args[0].received_iq)
    after_iq[old_count:] *= -2.5
    new_part = _artifact(
        after_iq[old_count:], "leo_clear_weak", prefix="mutated_new"
    )
    args[3] = _build_runtime_authorized_received_iq_artifact_internal(
        after_iq,
        physical_sample_ids=args[0].physical_sample_ids + new_part.physical_sample_ids,
        parent_received_iq_sha256=args[0].parent_received_iq_sha256 + new_part.parent_received_iq_sha256,
        overlay_tokens=args[0].overlay_tokens + new_part.overlay_tokens,
        source_leo_provenance_sha256=(
            args[0].source_leo_provenance_sha256
            + new_part.source_leo_provenance_sha256
        ),
        source_leo_cache_sha256=(
            args[0].source_leo_cache_sha256 + new_part.source_leo_cache_sha256
        ),
        target_channel_views=["leo_clear_weak"] * len(after_iq),
        satellite_seeds=args[0].satellite_seeds + new_part.satellite_seeds,
        overlay_provenance_sha256=args[0].overlay_provenance_sha256 + new_part.overlay_provenance_sha256,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    second = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(), backbone=_backbone()
    )
    np.testing.assert_array_equal(
        first.before_state.common_dct_coefficients,
        second.before_state.common_dct_coefficients,
    )
    np.testing.assert_array_equal(
        first.after_state.prototypes[:3], second.after_state.prototypes[:3]
    )
    assert not np.array_equal(
        first.after_state.prototypes[3:], second.after_state.prototypes[3:]
    )
    probe = _artifact(
        _iq_for_classes(("probe",), 1, seed=777),
        "leo_clear_weak", prefix="old_score_lock", purpose="inference",
    )
    _, before_scores = predict_scores(
        first.before_state, probe, backbone=_backbone()
    )
    _, after_scores = predict_scores(
        first.after_state, probe, backbone=_backbone()
    )
    np.testing.assert_array_equal(after_scores[:, :3], before_scores)


def test_enrollment_forwards_each_unique_physical_support_exactly_once() -> None:
    calls = {"rows": 0}

    def counted_extract(iq):
        assert iq.shape[0] == 1
        calls["rows"] += 1
        return _extract(iq)

    backbone = _seal_runtime_authorized_backbone_internal(
        counted_extract,
        feature_code_sha256=FEATURE_CODE,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
    )
    args = _before_after(k=10, seed=4321)
    fitted = fit_before_after_locked(
        *args, k_shot=10, hyperparameters=_hp(), backbone=backbone
    )
    unique_support = len(args[3].received_iq)
    assert calls["rows"] == unique_support
    assert fitted.after_state.resource["enrollment_unique_physical_support_count"] == unique_support
    assert fitted.after_state.resource["enrollment_backbone_forwards"] == unique_support
    assert fitted.after_state.resource["enrollment_repeated_old_support_backbone_forwards"] == 0
    assert fitted.trace[0]["after_old_support_backbone_recomputed"] is False
    assert fitted.trace[0]["after_new_only_backbone_forwards"] == 20
    assert fitted.after_state.support_artifact_sha256 == args[3].artifact_sha256
    assert fitted.after_state.support_selection_sha256 == d18._selection_sha(
        args[3], np.asarray(args[4]), np.asarray(args[5]), fitted.after_state.classes
    )


def test_k1_is_bitwise_z0_k2_to_k4_closed_k5_exact() -> None:
    k1_args, selection = _locked_prefix(1)
    k1 = fit_before_after_locked(
        *k1_args, k_shot=1,
        hyperparameters=selection.selected_hyperparameters, backbone=_backbone(),
        k10_lock_certificate=selection.k10_lock_certificate,
        expected_selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    assert k1.after_state.hyperparameters.force_zero is True
    assert k1.after_state.hyperparameters.candidate_id == "D18_Z0"
    assert k1.after_state.selection_authority_anchor_sha256 == AUTHORITY_ANCHOR
    assert (
        k1.after_state.locked_k10_candidate_id
        == selection.selected_hyperparameters.candidate_id
    )
    assert k1.trace[0]["candidate_id"] == "D18_Z0"
    assert (
        k1.trace[0]["locked_k10_candidate_id"]
        == selection.selected_hyperparameters.candidate_id
    )
    assert k1.after_state.resource["fft_forward_transforms_per_sample"] == 0
    assert k1.after_state.resource["ifft_inverse_transforms_per_sample"] == 0
    np.testing.assert_array_equal(
        _apply_equalizer_iq(
            k1_args[0],
            k1.before_state.common_dct_coefficients,
            k1.before_state.hyperparameters,
        ),
        k1_args[0].received_iq,
    )
    for k in (2, 3, 4):
        with pytest.raises(CmraeError, match="K2-K4"):
            fit_before_after_locked(
                *_before_after(k=k), k_shot=k,
                hyperparameters=_hp(), backbone=_backbone(),
            )
    k5_args, selection = _locked_prefix(5)
    k5 = fit_before_after_locked(
        *k5_args, k_shot=5,
        hyperparameters=selection.selected_hyperparameters, backbone=_backbone(),
        k10_lock_certificate=selection.k10_lock_certificate,
        expected_selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    assert k5.after_state.k_shot == 5
    other = next(
        hp for hp in preregistered_candidates()
        if hp.candidate_id != selection.selected_hyperparameters.candidate_id
    )
    with pytest.raises(CmraeError, match="selected candidate lock"):
        fit_before_after_locked(
            *k5_args, k_shot=5, hyperparameters=other, backbone=_backbone(),
            k10_lock_certificate=selection.k10_lock_certificate,
            expected_selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
        )
    with pytest.raises(CmraeError, match="expected selection authority anchor"):
        fit_before_after_locked(
            *k5_args, k_shot=5,
            hyperparameters=selection.selected_hyperparameters,
            backbone=_backbone(),
            k10_lock_certificate=selection.k10_lock_certificate,
        )
    with pytest.raises(CmraeError, match="authority anchor mismatch"):
        fit_before_after_locked(
            *k5_args, k_shot=5,
            hyperparameters=selection.selected_hyperparameters,
            backbone=_backbone(),
            k10_lock_certificate=selection.k10_lock_certificate,
            expected_selection_authority_anchor_sha256="e" * 64,
        )
    with pytest.raises(CmraeError, match="nested support prefix"):
        fit_before_after_locked(
            *_before_after(k=5, scene=d18.ALLOWED_CHANNEL_VIEWS[0], seed=9999),
            k_shot=5, hyperparameters=selection.selected_hyperparameters,
            backbone=_backbone(),
            k10_lock_certificate=selection.k10_lock_certificate,
            expected_selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
        )


def test_k1_positive_selector_lock_preserves_lock_identity_but_executes_z0(
    monkeypatch,
) -> None:
    scenes, original = _locked_k10_fixture()
    positive_hp = _hp(0.125)

    def forced_evidence(_scenes, *, backbone):
        assert tuple(_scenes) == scenes
        return (
            scenes,
            [dict(row) for row in original.evaluations],
            list(original.trace[:-1]),
            positive_hp,
        )

    monkeypatch.setattr(
        d18, "_select_k10_candidate_three_scene_evidence", forced_evidence
    )
    forced = select_k10_candidate_three_scene(
        scenes, backbone=_backbone(),
        selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    k1_args, _ = _locked_prefix(1)
    fitted = fit_before_after_locked(
        *k1_args, k_shot=1, hyperparameters=positive_hp, backbone=_backbone(),
        k10_lock_certificate=forced.k10_lock_certificate,
        expected_selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    assert fitted.after_state.locked_k10_candidate_id == positive_hp.candidate_id
    assert fitted.after_state.hyperparameters.candidate_id == "D18_Z0"
    assert fitted.trace[0]["candidate_id"] == "D18_Z0"


def test_k10_outer_l2o_has_fold_class_floor_h_forgetting_and_train_only_sha() -> None:
    args = list(_before_after(k=10))
    result, trace = evaluate_k10_outer_l2o(
        *args, hyperparameters=_hp(), backbone=_backbone(),
        scene_id="leo_clear_weak",
    )
    assert len(result["folds"]) == 5
    assert len(trace) == 5
    assert all(row["train_rows_per_class"] == 8 for row in result["folds"])
    assert all(row["development_selection_repeats_backbone_forwards"] for row in result["folds"])
    assert all(row["deployment_resource_evidence"] is False for row in result["folds"])
    for row in result["folds"]:
        for field in (
            "before_old_per_class", "after_old_per_class", "seen_new_per_class",
            "before_old_floor", "after_old_floor", "seen_new_floor",
            "H_old_new", "forgetting", "outer_train_state_sha256",
        ):
            assert field in row
    original_sha = result["folds"][0]["outer_train_state_sha256"]
    original_selection_sha = result["folds"][0]["outer_train_support_selection_sha256"]
    old_iq = np.array(args[0].received_iq, copy=True)
    after_iq = np.array(args[3].received_iq, copy=True)
    old_held = np.isin(args[2], (0, 1))
    after_held = np.isin(args[5], (0, 1))
    old_iq[old_held] *= 9.0
    after_iq[after_held] *= 9.0
    attacked_before = _build_runtime_authorized_received_iq_artifact_internal(
        old_iq,
        physical_sample_ids=args[0].physical_sample_ids,
        parent_received_iq_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in old_iq],
        overlay_tokens=args[0].overlay_tokens,
        source_leo_provenance_sha256=args[0].source_leo_provenance_sha256,
        source_leo_cache_sha256=args[0].source_leo_cache_sha256,
        target_channel_views=["leo_clear_weak"] * len(old_iq),
        satellite_seeds=args[0].satellite_seeds,
        overlay_provenance_sha256=args[0].overlay_provenance_sha256,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    attacked_after = _build_runtime_authorized_received_iq_artifact_internal(
        after_iq,
        physical_sample_ids=args[3].physical_sample_ids,
        parent_received_iq_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in after_iq],
        overlay_tokens=args[3].overlay_tokens,
        source_leo_provenance_sha256=args[3].source_leo_provenance_sha256,
        source_leo_cache_sha256=args[3].source_leo_cache_sha256,
        target_channel_views=["leo_clear_weak"] * len(after_iq),
        satellite_seeds=args[3].satellite_seeds,
        overlay_provenance_sha256=args[3].overlay_provenance_sha256,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    attacked, _ = evaluate_k10_outer_l2o(
        attacked_before, args[1], args[2], attacked_after, args[4], args[5],
        hyperparameters=_hp(), backbone=_backbone(),
        scene_id="leo_clear_weak",
    )
    assert attacked["folds"][0]["outer_train_state_sha256"] == original_sha
    assert (
        attacked["folds"][0]["outer_train_support_selection_sha256"]
        == original_selection_sha
    )


def test_three_scene_atomic_selector_uses_one_lambda_and_any_missing_scene_fails() -> None:
    scenes = []
    for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS):
        args = _before_after(k=10, scene=scene, seed=100 + index)
        scenes.append(CmraeSceneSupport(scene, *args))
    result = select_k10_candidate_three_scene(
        scenes, backbone=_backbone(),
        selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    assert len(result.evaluations) == 3
    assert len(result.fitted_scenes) == 3
    assert all(
        fit.after_state.hyperparameters == result.selected_hyperparameters
        for fit in result.fitted_scenes
    )
    assert all(len(row["scene_results"]) == 3 for row in result.evaluations)
    assert all(len(scene["folds"]) == 5 for row in result.evaluations for scene in row["scene_results"])
    assert result.trace[-1]["development_authority_only"] is True
    assert (
        result.k10_lock_certificate.selection_authority_anchor_sha256
        == AUTHORITY_ANCHOR
    )
    assert result.trace[-1]["selection_authority_anchor_sha256"] == AUTHORITY_ANCHOR
    certificate_text = repr(result.k10_lock_certificate.scene_prefix_locks)
    assert scenes[0].before_artifact.physical_sample_ids[0] not in certificate_text
    assert scenes[0].before_artifact.parent_received_iq_sha256[0] not in certificate_text
    assert scenes[0].before_artifact.overlay_provenance_sha256[0] not in certificate_text
    assert len(result.k10_lock_certificate.k10_selection_authority_sha256) == 64
    with pytest.raises(CmraeError, match="three-scene atomic"):
        select_k10_candidate_three_scene(
            scenes[:2], backbone=_backbone(),
            selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
        )
    with pytest.raises(CmraeError, match="selection authority anchor"):
        select_k10_candidate_three_scene(
            scenes, backbone=_backbone(),
            selection_authority_anchor_sha256="not-a-sha",
        )
    with pytest.raises(TypeError, match="selection_authority_anchor_sha256"):
        select_k10_candidate_three_scene(scenes, backbone=_backbone())


def test_state_self_seals_roundtrips_and_malicious_resource_cannot_reseal() -> None:
    state = fit_before_after_locked(
        *_before_after(k=10), k_shot=10,
        hyperparameters=_hp(), backbone=_backbone(),
    ).after_state
    assert state.resource["trainable_parameters"] == 0
    assert state.resource["adapt_epochs"] == 0
    assert state.resource["dense_query_graph"] is False
    assert state.resource["cmrae_adapter_estimated_serialized_state_bytes"] < 16 * 1024
    assert state.resource["persistent_array_state_bytes"] == (
        state.resource["cmrae_adapter_state_bytes"]
        + state.resource["registered_prototype_state_bytes"]
    )
    assert state.resource["estimated_full_state_bytes_uncompressed"] > (
        state.resource["cmrae_adapter_estimated_serialized_state_bytes"]
    )
    assert state.resource["post_reception_views_per_physical_sample"] == 1
    assert state.authority_scope == "development_diagnostic_only"
    with pytest.raises(ValueError):
        state.common_dct_coefficients[0] += 1
    payload, digest = serialize_state_bytes(state)
    assert len(payload) <= d18.MAX_FULL_SERIALIZED_STATE_BYTES
    loaded = load_state_bytes(payload, expected_sha256=digest)
    assert loaded.state_content_sha256 == state.state_content_sha256
    np.testing.assert_array_equal(loaded.prototypes, state.prototypes)
    assert loaded.operator_id == d18.OPERATOR_ID
    with pytest.raises(CmraeError, match="serialized state SHA"):
        load_state_bytes(payload + b"x", expected_sha256=digest)
    bad = dict(state.resource)
    bad["fft_forward_transforms_per_sample"] = 0
    with pytest.raises(CmraeError, match="state drift"):
        replace(state, resource=bad, state_content_sha256="")
    with pytest.raises(CmraeError, match="hyperparameter drift"):
        replace(
            state,
            hyperparameters=replace(state.hyperparameters, tau=np.log(1.2)),
            state_content_sha256="",
        )
    with pytest.raises(CmraeError, match="state drift"):
        replace(state, operator_id="unregistered", state_content_sha256="")
    with pytest.raises(CmraeError, match="state drift"):
        replace(state, feature_code_sha256="g" * 64, state_content_sha256="")
    tampered = replace(state)
    object.__setattr__(tampered, "state_content_sha256", "0" * 64)
    with pytest.raises(CmraeError, match="state drift"):
        d18._validate_state(tampered)


def test_single_sample_prediction_scores_all_registered_without_truth_surface() -> None:
    state = fit_before_after_locked(
        *_before_after(k=10), k_shot=10,
        hyperparameters=_hp(), backbone=_backbone(),
    ).after_state
    iq = _iq_for_classes(("probe",), 1, seed=900)
    artifact = _artifact(
        iq, "leo_clear_weak", prefix="inference", purpose="inference"
    )
    predicted, scores = predict_scores(state, artifact, backbone=_backbone())
    assert predicted in state.classes
    assert scores.shape == (1, len(state.classes))
    two = _artifact(
        np.concatenate([iq, iq * 1.01]), "leo_clear_weak",
        prefix="inference_two", purpose="inference",
    )
    with pytest.raises(CmraeError, match="single-query"):
        predict_scores(state, two, backbone=_backbone())
    assert set(inspect.signature(predict_scores).parameters) == {
        "state", "artifact", "backbone"
    }


def test_preregistered_surface_is_exact() -> None:
    candidates = preregistered_candidates()
    assert [(v.candidate_id, v.lambda_equalizer) for v in candidates] == [
        ("D18_Z0", 0.0),
        ("D18_CMRAE_L0125", 0.125),
        ("D18_CMRAE_L0250", 0.25),
    ]
    assert all(v.dct_rank == 8 for v in candidates)
    assert all(v.tau == pytest.approx(np.log(1.10), abs=1e-15) for v in candidates)


def test_redteam_cross_scene_reused_physical_id_is_rejected() -> None:
    scene_rows = []
    for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS):
        scene_rows.append(CmraeSceneSupport(scene, *_before_after(k=10, scene=scene, seed=500 + index)))
    reused = scene_rows[0].before_artifact.physical_sample_ids[0]
    target = scene_rows[1]
    before_ids = list(target.before_artifact.physical_sample_ids)
    before_ids[0] = reused
    attacked_before = _build_runtime_authorized_received_iq_artifact_internal(
        target.before_artifact.received_iq,
        physical_sample_ids=before_ids,
        parent_received_iq_sha256=target.before_artifact.parent_received_iq_sha256,
        overlay_tokens=target.before_artifact.overlay_tokens,
        source_leo_provenance_sha256=target.before_artifact.source_leo_provenance_sha256,
        source_leo_cache_sha256=target.before_artifact.source_leo_cache_sha256,
        target_channel_views=[target.scene_id] * len(before_ids),
        satellite_seeds=target.before_artifact.satellite_seeds,
        overlay_provenance_sha256=target.before_artifact.overlay_provenance_sha256,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    after_ids = list(target.after_artifact.physical_sample_ids)
    after_ids[0] = reused
    attacked_after = _build_runtime_authorized_received_iq_artifact_internal(
        target.after_artifact.received_iq,
        physical_sample_ids=after_ids,
        parent_received_iq_sha256=target.after_artifact.parent_received_iq_sha256,
        overlay_tokens=target.after_artifact.overlay_tokens,
        source_leo_provenance_sha256=target.after_artifact.source_leo_provenance_sha256,
        source_leo_cache_sha256=target.after_artifact.source_leo_cache_sha256,
        target_channel_views=[target.scene_id] * len(after_ids),
        satellite_seeds=target.after_artifact.satellite_seeds,
        overlay_provenance_sha256=target.after_artifact.overlay_provenance_sha256,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    scene_rows[1] = CmraeSceneSupport(
        target.scene_id,
        attacked_before, target.before_labels, target.before_ranks,
        attacked_after, target.after_labels, target.after_ranks,
    )
    with pytest.raises(CmraeError, match="cross-scene"):
        select_k10_candidate_three_scene(
            scene_rows, backbone=_backbone(),
            selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
        )


def test_redteam_independent_k5_without_k10_lock_is_rejected() -> None:
    with pytest.raises(CmraeError, match="K10 lock certificate"):
        fit_before_after_locked(
            *_before_after(k=5), k_shot=5,
            hyperparameters=_hp(), backbone=_backbone(),
        )


@pytest.mark.parametrize("reuse_kind", ("parent", "overlay", "overlay_token"))
def test_redteam_cross_scene_parent_or_overlay_reuse_is_rejected(reuse_kind) -> None:
    rows = [
        CmraeSceneSupport(scene, *_before_after(k=10, scene=scene, seed=700 + index))
        for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS)
    ]
    source = rows[0]
    target = rows[1]
    before_iq = np.array(target.before_artifact.received_iq, copy=True)
    after_iq = np.array(target.after_artifact.received_iq, copy=True)
    before_overlay = list(target.before_artifact.overlay_provenance_sha256)
    after_overlay = list(target.after_artifact.overlay_provenance_sha256)
    before_tokens = list(target.before_artifact.overlay_tokens)
    after_tokens = list(target.after_artifact.overlay_tokens)
    if reuse_kind == "parent":
        before_iq[0] = source.before_artifact.received_iq[0]
        after_iq[0] = source.before_artifact.received_iq[0]
    elif reuse_kind == "overlay":
        before_overlay[0] = source.before_artifact.overlay_provenance_sha256[0]
        after_overlay[0] = source.before_artifact.overlay_provenance_sha256[0]
    else:
        before_tokens[0] = source.before_artifact.overlay_tokens[0]
        after_tokens[0] = source.before_artifact.overlay_tokens[0]
    attacked_before = _build_runtime_authorized_received_iq_artifact_internal(
        before_iq,
        physical_sample_ids=target.before_artifact.physical_sample_ids,
        parent_received_iq_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in before_iq],
        overlay_tokens=before_tokens,
        source_leo_provenance_sha256=target.before_artifact.source_leo_provenance_sha256,
        source_leo_cache_sha256=target.before_artifact.source_leo_cache_sha256,
        target_channel_views=[target.scene_id] * len(before_iq),
        satellite_seeds=target.before_artifact.satellite_seeds,
        overlay_provenance_sha256=before_overlay,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    attacked_after = _build_runtime_authorized_received_iq_artifact_internal(
        after_iq,
        physical_sample_ids=target.after_artifact.physical_sample_ids,
        parent_received_iq_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in after_iq],
        overlay_tokens=after_tokens,
        source_leo_provenance_sha256=target.after_artifact.source_leo_provenance_sha256,
        source_leo_cache_sha256=target.after_artifact.source_leo_cache_sha256,
        target_channel_views=[target.scene_id] * len(after_iq),
        satellite_seeds=target.after_artifact.satellite_seeds,
        overlay_provenance_sha256=after_overlay,
        sealed_runtime_sha256=RUNTIME,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
        feature_code_sha256=FEATURE_CODE,
        purpose="support",
    )
    rows[1] = CmraeSceneSupport(
        target.scene_id,
        attacked_before, target.before_labels, target.before_ranks,
        attacked_after, target.after_labels, target.after_ranks,
    )
    with pytest.raises(CmraeError, match="cross-scene"):
        select_k10_candidate_three_scene(
            rows, backbone=_backbone(),
            selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
        )


def test_shared_source_authority_cache_and_satellite_seed_bind_but_do_not_define_physical_reuse() -> None:
    rows = [
        CmraeSceneSupport(scene, *_before_after(k=10, scene=scene, seed=8100 + index))
        for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS)
    ]
    source = rows[0]
    target = rows[1]

    def rebind(artifact, source_artifact):
        return _build_runtime_authorized_received_iq_artifact_internal(
            artifact.received_iq,
            physical_sample_ids=artifact.physical_sample_ids,
            parent_received_iq_sha256=artifact.parent_received_iq_sha256,
            overlay_tokens=artifact.overlay_tokens,
            source_leo_provenance_sha256=(
                source_artifact.source_leo_provenance_sha256
            ),
            source_leo_cache_sha256=source_artifact.source_leo_cache_sha256,
            target_channel_views=artifact.target_channel_views,
            satellite_seeds=source_artifact.satellite_seeds,
            overlay_provenance_sha256=artifact.overlay_provenance_sha256,
            sealed_runtime_sha256=artifact.sealed_runtime_sha256,
            sealed_phase1_checkpoint_sha256=(
                artifact.sealed_phase1_checkpoint_sha256
            ),
            feature_code_sha256=artifact.feature_code_sha256,
            purpose=artifact.purpose,
        )

    rebound_before = rebind(target.before_artifact, source.before_artifact)
    rebound_after = rebind(target.after_artifact, source.after_artifact)
    original_sha = d18._selection_sha(
        target.after_artifact,
        np.asarray(target.after_labels), np.asarray(target.after_ranks),
        tuple(sorted(np.unique(np.asarray(target.after_labels)).tolist())),
    )
    rebound_sha = d18._selection_sha(
        rebound_after,
        np.asarray(target.after_labels), np.asarray(target.after_ranks),
        tuple(sorted(np.unique(np.asarray(target.after_labels)).tolist())),
    )
    assert rebound_sha != original_sha
    rows[1] = CmraeSceneSupport(
        target.scene_id,
        rebound_before, target.before_labels, target.before_ranks,
        rebound_after, target.after_labels, target.after_ranks,
    )
    selected = select_k10_candidate_three_scene(
        rows, backbone=_backbone(),
        selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    )
    assert selected.k10_lock_certificate.selection_authority_anchor_sha256 == AUTHORITY_ANCHOR


def test_full_serialized_state_over_256k_is_rejected_on_write_and_load() -> None:
    def huge_extract(iq):
        seed = int(abs(float(iq[0, 0, 0])) * 1_000_000) % (2**32)
        return np.random.default_rng(seed).normal(size=(1, 20_000)).astype(np.float32)

    huge_backbone = _seal_runtime_authorized_backbone_internal(
        huge_extract,
        feature_code_sha256=FEATURE_CODE,
        sealed_phase1_checkpoint_sha256=CHECKPOINT,
    )
    state = fit_before_after_locked(
        *_before_after(k=10, seed=8888), k_shot=10,
        hyperparameters=_hp(), backbone=huge_backbone,
    ).after_state
    assert state.resource["persistent_array_state_bytes"] > 256 * 1024
    with pytest.raises(CmraeError, match="exceeds 256KiB"):
        serialize_state_bytes(state)
    oversized = b"x" * (d18.MAX_FULL_SERIALIZED_STATE_BYTES + 1)
    with pytest.raises(CmraeError, match="exceeds 256KiB"):
        load_state_bytes(
            oversized, expected_sha256=hashlib.sha256(oversized).hexdigest()
        )


def test_k10_lock_certificate_direct_sign_copy_replace_and_tamper_are_rejected() -> None:
    scenes, selection = _locked_k10_fixture()
    certificate = selection.k10_lock_certificate
    kwargs = {
        "schema": certificate.schema,
        "selected_candidate_id": certificate.selected_candidate_id,
        "scene_prefix_locks": certificate.scene_prefix_locks,
        "selection_authority_anchor_sha256": (
            certificate.selection_authority_anchor_sha256
        ),
        "k10_selection_authority_sha256": certificate.k10_selection_authority_sha256,
        "authority_scope": certificate.authority_scope,
        "certificate_sha256": certificate.certificate_sha256,
    }
    with pytest.raises(CmraeError, match="selector-issued"):
        d18.CmraeK10LockCertificate(**kwargs)
    with pytest.raises(CmraeError, match="selector-issued"):
        d18.CmraeK10LockCertificate(**kwargs, _token=object())
    with pytest.raises(CmraeError, match="cannot be copied"):
        copy.copy(certificate)
    with pytest.raises(CmraeError, match="cannot be copied"):
        copy.deepcopy(certificate)
    with pytest.raises(TypeError, match="dataclass"):
        replace(certificate, selected_candidate_id="D18_Z0")
    assert not hasattr(d18, "_K10_LOCK_TOKEN")
    assert not hasattr(d18, "_issue_k10_lock_certificate")
    assert not hasattr(d18, "_create_k10_lock_surface")
    fresh_scenes = tuple(
        CmraeSceneSupport(
            scene, *_before_after(k=10, scene=scene, seed=9100 + index)
        )
        for index, scene in enumerate(d18.ALLOWED_CHANNEL_VIEWS)
    )
    tampered = select_k10_candidate_three_scene(
        fresh_scenes, backbone=_backbone(),
        selection_authority_anchor_sha256=AUTHORITY_ANCHOR,
    ).k10_lock_certificate
    object.__setattr__(tampered, "selected_candidate_id", "D18_CMRAE_L0250")
    with pytest.raises(CmraeError, match="certificate drift"):
        d18._validate_k10_certificate(tampered)


def test_candidate_ranking_is_invariant_to_swapping_old_and_new_floors() -> None:
    result = {
        "candidate_id": "D18_CMRAE_L0125",
        "worst_after_old_floor": 0.80,
        "worst_seen_new_floor": 0.60,
        "worst_scene_H_old_new": 0.66,
        "mean_H_old_new": 0.70,
        "mean_joint": 0.72,
        "max_forgetting": 0.03,
    }
    swapped = dict(result)
    swapped["worst_after_old_floor"] = result["worst_seen_new_floor"]
    swapped["worst_seen_new_floor"] = result["worst_after_old_floor"]
    original_key = d18._symmetric_candidate_rank_key(result)
    swapped_key = d18._symmetric_candidate_rank_key(swapped)
    assert original_key == swapped_key
    assert original_key[0] == min(
        result["worst_after_old_floor"], result["worst_seen_new_floor"]
    )
    # Better balanced floor wins even if its old floor alone is lower.
    balanced = dict(result)
    balanced["worst_after_old_floor"] = 0.65
    balanced["worst_seen_new_floor"] = 0.65
    assert d18._symmetric_candidate_rank_key(balanced) > original_key
