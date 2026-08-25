from __future__ import annotations

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_m29_d92 import (
    FFT_ALPHA1,
    IDENTITY_ONLY,
    TASR_ALPHA1,
    d92_feature_geometry,
    fit_m29_d92,
    make_m29_features,
)
from cvsrffi.stage2_m29_tasr import (
    build_phase1_tasr_bundle,
    estimate_target_spectral_calibration,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    rng = np.random.default_rng(29)
    labels = np.repeat(np.asarray(["c0", "c1", "c2"]), 9)
    receivers = np.tile(np.asarray([f"r{i}" for i in range(9)]), 3)
    identity = _unit(rng.normal(size=(27, 160)))
    fft = rng.normal(size=(27, 96))
    bundle, _audit = build_phase1_tasr_bundle(
        fft,
        labels,
        receivers,
        dataset_roles=np.repeat("source", len(labels)),
        checkpoint_sha256="a" * 64,
        class_registry=("c0", "c1", "c2"),
        rank=8,
    )
    calibration = estimate_target_spectral_calibration(
        fft,
        labels,
        bundle,
    )
    return identity, fft, labels, receivers, bundle, calibration


def test_m29_feature_geometries_are_real_and_normalized() -> None:
    identity, fft, _labels, _receivers, bundle, calibration = _fixture()
    f160 = make_m29_features(identity, fft, arm=IDENTITY_ONLY)
    f256 = make_m29_features(identity, fft, arm=FFT_ALPHA1)
    f208 = make_m29_features(
        identity,
        fft,
        arm=TASR_ALPHA1,
        tasr_bundle=bundle,
        calibration=calibration,
    )
    assert f160.shape == (27, 160)
    assert f256.shape == (27, 256)
    assert f208.shape == (27, 208)
    assert np.allclose(np.linalg.norm(f160, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(f256, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(f208, axis=1), 1.0)
    assert not np.any(np.all(f208[:, 160:] == 0.0, axis=1))


def test_d92_geometry_restores_globals_on_success_and_failure() -> None:
    original = (d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS)
    with d92_feature_geometry((160, 48)):
        assert d42.FEATURE_DIM == 208
        assert d42.BLOCK_DIMS == (160, 48)
        assert d42.BLOCK_SLICES == (slice(0, 160), slice(160, 208))
    assert (d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS) == original

    with pytest.raises(RuntimeError, match="boom"):
        with d92_feature_geometry((160,)):
            assert d42.FEATURE_DIM == 160
            raise RuntimeError("boom")
    assert (d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS) == original


def test_tasr_requires_frozen_support_calibration() -> None:
    identity, fft, *_rest = _fixture()
    with pytest.raises(ValueError, match="bundle and calibration"):
        make_m29_features(identity, fft, arm=TASR_ALPHA1)


def test_identity160_runs_the_real_d92_fit_without_padding() -> None:
    rng = np.random.default_rng(2901)
    old_classes = tuple(f"old{i}" for i in range(6))
    new_classes = tuple(f"new{i}" for i in range(5))
    state = fit_m29_d92(
        arm=IDENTITY_ONLY,
        old_identity160=_unit(rng.normal(size=(6, 160))),
        old_fft96=rng.normal(size=(6, 96)),
        old_labels=np.asarray(old_classes),
        old_classes=old_classes,
        new_identity160=_unit(rng.normal(size=(5, 160))),
        new_fft96=rng.normal(size=(5, 96)),
        new_labels=np.asarray(new_classes),
        new_classes=new_classes,
        seed=2901,
        device="cpu",
    )
    assert state.inference.compiled_affine_state.feature_dim == 160
    assert state.inference.compiled_affine_state.block_offsets == (0, 160)
    assert state.audit["zero_padding_used"] is False
    assert state.audit["query_rows_used"] == 0
    scores = state.score(_unit(rng.normal(size=(3, 160))), rng.normal(size=(3, 96)))
    assert scores.shape == (3, 11)
    assert np.isfinite(scores).all()
