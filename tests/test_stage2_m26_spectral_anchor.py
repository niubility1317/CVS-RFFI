from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_m26_spectral_anchor import (
    CHECKPOINT_SHA256_PATTERN,
    build_phase1_spectral_anchor,
    fft_envelope_ripple,
    fft_magnitude_geometry,
    load_m26_spectral_anchor,
    publish_m26_spectral_anchor,
)


def _source_rows() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(82601)
    classes = tuple(f"old-{index}" for index in range(6))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, name in enumerate(classes):
        for _shot in range(8):
            identity = 0.02 * rng.normal(size=160)
            fft = 0.02 * rng.normal(size=96)
            identity[index * 11] += 1.0
            fft[index * 7] += 1.0
            rows.append(np.concatenate([identity, fft]))
            labels.append(name)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), classes


def test_fft96_is_split_into_finite_orthogonal_32_and_64_blocks() -> None:
    rng = np.random.default_rng(82602)
    fft = rng.normal(size=(7, 96)).astype(np.float32)
    envelope, ripple = fft_envelope_ripple(fft)
    assert envelope.shape == (7, 32)
    assert ripple.shape == (7, 64)
    assert np.isfinite(envelope).all() and np.isfinite(ripple).all()
    np.testing.assert_allclose(np.linalg.norm(envelope, axis=1), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(ripple, axis=1), 1.0, atol=1.0e-6)


def test_fft96_magnitude_geometry_is_finite_unit_and_not_raw_fft() -> None:
    frequency = np.linspace(-1.0, 1.0, 96)
    rows = np.stack(
        [
            0.8 * frequency + 0.2 * np.sin((index + 2) * np.pi * frequency)
            + 0.1 * np.roll(frequency**2, index)
            for index in range(5)
        ]
    ).astype(np.float32)
    geometry = fft_magnitude_geometry(rows)
    assert geometry.shape == (5, 96)
    assert np.isfinite(geometry).all()
    np.testing.assert_allclose(np.linalg.norm(geometry, axis=1), 1.0, atol=1.0e-6)
    assert not np.allclose(geometry, rows / np.linalg.norm(rows, axis=1, keepdims=True))


def test_fft96_magnitude_geometry_detects_local_and_mirror_changes() -> None:
    frequency = np.linspace(-1.0, 1.0, 96)
    base = 0.7 * frequency + 0.15 * frequency**2
    changed = base.copy()
    changed[14:20] += np.asarray([0.0, 0.3, -0.2, 0.4, -0.1, 0.2])
    changed[70:76] -= 0.25
    geometry = fft_magnitude_geometry(np.stack([base, changed]).astype(np.float32))
    assert float(np.linalg.norm(geometry[0] - geometry[1])) > 0.10


def test_phase1_anchor_persists_only_six_int8_aggregate_centres(tmp_path: Path) -> None:
    rows, labels, classes = _source_rows()
    checkpoint = "a" * 64
    assert CHECKPOINT_SHA256_PATTERN.fullmatch(checkpoint)
    component, audit = build_phase1_spectral_anchor(
        rows,
        labels,
        class_registry=classes,
        checkpoint_sha256=checkpoint,
    )
    assert component.class_registry == classes
    assert component.identity_q.shape == (6, 160)
    assert component.fft_q.shape == (6, 96)
    assert component.identity_q.dtype == np.int8
    assert component.fft_q.dtype == np.int8
    assert component.centres().shape == (6, 256)
    assert len(component.component_id) == 64
    assert audit["source_row_count"] == 48
    assert audit["persisted_member_or_sample_count"] == 0

    path = tmp_path / "m26_phase1_anchor.npz"
    publish_m26_spectral_anchor(path, component)
    loaded = load_m26_spectral_anchor(path, expected_checkpoint_sha256=checkpoint)
    np.testing.assert_allclose(loaded.centres(), component.centres())
    assert loaded.component_id == component.component_id
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "schema",
            "feature_schema",
            "checkpoint_sha256",
            "class_registry",
            "identity_q",
            "identity_scale",
            "fft_q",
            "fft_scale",
        }


def test_anchor_component_identity_changes_with_quantized_content() -> None:
    rows, labels, classes = _source_rows()
    first, _audit = build_phase1_spectral_anchor(
        rows,
        labels,
        class_registry=classes,
        checkpoint_sha256="f" * 64,
    )
    changed_rows = rows.copy()
    changed_rows[labels == classes[0], :160] = np.roll(
        changed_rows[labels == classes[0], :160], 1, axis=1
    )
    second, _audit = build_phase1_spectral_anchor(
        changed_rows,
        labels,
        class_registry=classes,
        checkpoint_sha256="f" * 64,
    )
    assert second.component_id != first.component_id


def test_anchor_rejects_target_rows_registry_drift_and_overwrite(tmp_path: Path) -> None:
    rows, labels, classes = _source_rows()
    with pytest.raises(ValueError, match="source-only"):
        build_phase1_spectral_anchor(
            rows,
            labels,
            class_registry=classes,
            checkpoint_sha256="b" * 64,
            dataset_roles=np.asarray(["source"] * 47 + ["target_old"]),
        )
    with pytest.raises(ValueError, match="class-symmetric"):
        build_phase1_spectral_anchor(
            rows[:-1],
            labels[:-1],
            class_registry=classes,
            checkpoint_sha256="b" * 64,
        )

    component, _audit = build_phase1_spectral_anchor(
        rows,
        labels,
        class_registry=classes,
        checkpoint_sha256="b" * 64,
    )
    path = tmp_path / "anchor.npz"
    publish_m26_spectral_anchor(path, component)
    with pytest.raises(FileExistsError):
        publish_m26_spectral_anchor(path, component)
