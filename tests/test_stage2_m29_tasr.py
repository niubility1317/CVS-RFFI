from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_m29_tasr import (
    FFT_DIM,
    TASR_DIM,
    build_phase1_tasr_bundle,
    estimate_target_spectral_calibration,
    load_phase1_tasr_bundle,
    publish_phase1_tasr_bundle,
    tasr48_raw,
    transform_tasr48,
)


def _source_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(82901)
    classes = ("old-0", "old-1", "old-2")
    receivers = tuple(f"rx-{index}" for index in range(9))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    domains: list[str] = []
    frequency = np.linspace(-1.0, 1.0, FFT_DIM)
    receiver_shape = {
        receiver: (
            0.04 * (index - 4) * frequency
            + 0.02 * np.sin((index + 1) * np.pi * frequency)
            + 0.01 * (index % 3 - 1) * frequency**2
        )
        for index, receiver in enumerate(receivers)
    }
    for class_index, name in enumerate(classes):
        class_shape = 0.18 * np.sin((class_index + 2) * np.pi * frequency)
        for receiver in receivers:
            for _ in range(5):
                rows.append(
                    class_shape
                    + receiver_shape[receiver]
                    + 0.005 * rng.normal(size=FFT_DIM)
                )
                labels.append(name)
                domains.append(receiver)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(domains),
        classes,
    )


def test_phase1_bundle_builds_receiver_centered_rank8_state_without_samples() -> None:
    rows, labels, receivers, classes = _source_fixture()
    bundle, audit = build_phase1_tasr_bundle(
        rows,
        labels,
        receivers,
        class_registry=classes,
        checkpoint_sha256="a" * 64,
        dataset_roles=np.asarray(["source"] * len(rows)),
        rank=8,
    )

    assert bundle.global_mean().shape == (FFT_DIM,)
    assert bundle.basis().shape == (FFT_DIM, 8)
    assert bundle.eigenvalues().shape == (8,)
    assert bundle.tasr_location().shape == (TASR_DIM,)
    assert bundle.tasr_scale().shape == (TASR_DIM,)
    np.testing.assert_allclose(bundle.basis().T @ bundle.basis(), np.eye(8), atol=0.04)
    assert np.all(bundle.eigenvalues() > 0.0)
    assert bundle.eigenvalues_q.dtype == np.int8
    assert np.asarray(bundle.eigenvalues_scale).shape == ()
    assert len(set(bundle.eigenvalues_q.tolist())) > 1
    assert audit["source_row_count"] == len(rows)
    assert audit["receiver_count"] == 9
    assert audit["persisted_member_or_sample_count"] == 0
    assert audit["query_rows_used"] == 0

    permutation = np.random.default_rng(82902).permutation(len(rows))
    repeated, _ = build_phase1_tasr_bundle(
        rows[permutation],
        labels[permutation],
        receivers[permutation],
        class_registry=classes,
        checkpoint_sha256="a" * 64,
        dataset_roles=np.asarray(["source"] * len(rows)),
        rank=8,
    )
    assert repeated.component_id == bundle.component_id


def test_phase1_bundle_rejects_target_rows_and_missing_receiver_class_cells() -> None:
    rows, labels, receivers, classes = _source_fixture()
    roles = np.asarray(["source"] * len(rows))
    roles[-1] = "target_old"
    with pytest.raises(ValueError, match="source-only"):
        build_phase1_tasr_bundle(
            rows,
            labels,
            receivers,
            class_registry=classes,
            checkpoint_sha256="b" * 64,
            dataset_roles=roles,
        )

    keep = ~((labels == classes[-1]) & (receivers == "rx-8"))
    with pytest.raises(ValueError, match="receiver-class"):
        build_phase1_tasr_bundle(
            rows[keep],
            labels[keep],
            receivers[keep],
            class_registry=classes,
            checkpoint_sha256="b" * 64,
            dataset_roles=np.asarray(["source"] * int(np.sum(keep))),
        )


def test_tasr48_raw_matches_fixed_smooth_difference_and_pooling_contract() -> None:
    row = np.linspace(-0.4, 0.8, FFT_DIM, dtype=np.float64)
    row[47:50] += np.asarray([0.3, -0.2, 0.1])
    actual = tasr48_raw(row[None, :])[0]

    padded = np.pad(row, (4, 4), mode="reflect")
    smooth = np.convolve(padded, np.ones(9) / 9.0, mode="valid")
    residual = row - smooth
    d1 = np.diff(row)
    d2 = np.diff(d1)
    expected = np.concatenate(
        [
            np.asarray([part.mean() for part in np.array_split(residual, 16)]),
            np.asarray([np.sqrt(np.mean(part**2)) for part in np.array_split(residual, 16)]),
            np.asarray([np.sqrt(np.mean(part**2)) for part in np.array_split(d1, 8)]),
            np.asarray([np.sqrt(np.mean(part**2)) for part in np.array_split(d2, 8)]),
        ]
    )
    assert actual.shape == (TASR_DIM,)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_target_calibration_is_class_equal_support_only_and_query_transform_is_frozen() -> None:
    rows, labels, receivers, classes = _source_fixture()
    bundle, _ = build_phase1_tasr_bundle(
        rows,
        labels,
        receivers,
        class_registry=classes,
        checkpoint_sha256="c" * 64,
        dataset_roles=np.asarray(["source"] * len(rows)),
        rank=8,
    )
    support = np.stack(
        [
            rows[(labels == name) & (receivers == "rx-1")][:2].mean(axis=0)
            for name in classes
        ]
    ).astype(np.float32)
    support_labels = np.asarray(classes)
    first = estimate_target_spectral_calibration(support, support_labels, bundle)
    duplicated = estimate_target_spectral_calibration(
        np.repeat(support, 4, axis=0),
        np.repeat(support_labels, 4),
        bundle,
    )
    np.testing.assert_allclose(first.delta, duplicated.delta, atol=1.0e-7)
    assert first.query_rows_used == 0
    assert first.frozen is True

    before = first.delta.copy()
    transformed = transform_tasr48(rows[:7], first, bundle)
    assert transformed.shape == (7, TASR_DIM)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1.0e-6)
    np.testing.assert_array_equal(first.delta, before)


def test_phase1_tasr_bundle_roundtrip_is_strict_and_non_overwriting(tmp_path: Path) -> None:
    rows, labels, receivers, classes = _source_fixture()
    bundle, _ = build_phase1_tasr_bundle(
        rows,
        labels,
        receivers,
        class_registry=classes,
        checkpoint_sha256="d" * 64,
        dataset_roles=np.asarray(["source"] * len(rows)),
        rank=8,
    )
    path = tmp_path / "tasr_bundle.npz"
    publish_phase1_tasr_bundle(path, bundle)
    loaded = load_phase1_tasr_bundle(path, expected_checkpoint_sha256="d" * 64)
    assert loaded.component_id == bundle.component_id
    np.testing.assert_allclose(loaded.basis(), bundle.basis())
    with np.load(path, allow_pickle=False) as payload:
        assert "tau" not in payload.files
        assert payload["eigenvalues_q"].dtype == np.int8
        assert payload["eigenvalues_scale"].shape == ()
    with pytest.raises(FileExistsError):
        publish_phase1_tasr_bundle(path, bundle)
    with pytest.raises(ValueError, match="checkpoint"):
        load_phase1_tasr_bundle(path, expected_checkpoint_sha256="e" * 64)
