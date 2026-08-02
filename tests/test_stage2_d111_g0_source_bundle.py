from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_d111_g0_source_bundle import (
    COMPONENT_STATE,
    D111G0SourceBundleError,
    EXPECTED_CELLS,
    EXPECTED_ROWS,
    MANIFEST_NAME,
    NPZ_NAME,
    PAYLOAD_MEMBERS,
    TAP_MEMBERS,
    build_d111_g0_source_bundle,
    load_d111_g0_source_bundle,
)
from cvsrffi.stage2_d111_loo_gat_bundle import FEATURE_DIM


CLASSES = tuple(f"tx-{index}" for index in range(6))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def _strict_tap_arrays(*, sample_amplitude: float = 0.15) -> dict[str, np.ndarray]:
    """A deterministic 7x4x6 source tap with exactly 588 physical rows."""

    rows: dict[str, list[object]] = {name: [] for name in TAP_MEMBERS}
    for receiver_index in range(7):
        for day_index in range(4):
            cell_index = receiver_index * 4 + day_index
            # 14 cells x 4 classes x 6 plus 14 cells x 3 classes x 6 = 588.
            per_class = 4 if cell_index % 2 == 0 else 3
            domain_shift = np.asarray(
                [
                    0.020 * (receiver_index - 3),
                    0.025 * (day_index - 1.5),
                    0.013 * (((2 * receiver_index + day_index) % 5) - 2),
                ],
                dtype=np.float32,
            )
            for class_index, class_id in enumerate(CLASSES):
                for sample_index in range(per_class):
                    pre_relu = np.zeros(FEATURE_DIM, dtype=np.float32)
                    pre_relu[:3] = 0.85 + domain_shift
                    pre_relu[10 + class_index] = 1.75
                    # Keep the strictly positive per-cell RMS radius above the
                    # persisted FP16 quantization scale floor.
                    pre_relu[40 + sample_index] = sample_amplitude * (sample_index + 1)
                    token = (
                        f"PHYS-SECRET-r{receiver_index}-d{day_index}-"
                        f"c{class_index}-s{sample_index}"
                    )
                    rows["pre_relu"].append(pre_relu)
                    rows["z_dom"].append(pre_relu * np.float32(0.5))
                    rows["tx_labels"].append(class_id)
                    rows["receiver_ids"].append(f"rx-{receiver_index}")
                    rows["day_ids"].append(f"day-{day_index}")
                    rows["physical_ids"].append(token)
                    rows["scenario_names"].append("phase1-source")
                    rows["observation_ids"].append(f"OBS-{token}")
    arrays = {
        "pre_relu": np.asarray(rows["pre_relu"], dtype=np.float32),
        "z_dom": np.asarray(rows["z_dom"], dtype=np.float32),
        "tx_labels": np.asarray(rows["tx_labels"], dtype=np.str_),
        "receiver_ids": np.asarray(rows["receiver_ids"], dtype=np.str_),
        "day_ids": np.asarray(rows["day_ids"], dtype=np.str_),
        "physical_ids": np.asarray(rows["physical_ids"], dtype=np.str_),
        "scenario_names": np.asarray(rows["scenario_names"], dtype=np.str_),
        "observation_ids": np.asarray(rows["observation_ids"], dtype=np.str_),
    }
    assert tuple(arrays) == TAP_MEMBERS
    assert arrays["pre_relu"].shape == (EXPECTED_ROWS, FEATURE_DIM)
    return arrays


def _write_tap(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    # np.savez preserves this insertion order, which is also part of the tap contract.
    np.savez_compressed(path, **arrays)
    return path


def _reference_geometry(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(arrays["pre_relu"], 0.0)
    x = _normalise(x)
    labels = arrays["tx_labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    days = arrays["day_ids"].astype(str)
    cells = sorted(set(zip(receivers, days, strict=True)))
    assert len(cells) == EXPECTED_CELLS
    anchors: list[np.ndarray] = []
    variances: list[float] = []
    for class_id in CLASSES:
        centres: list[np.ndarray] = []
        radii: list[float] = []
        for receiver, day in cells:
            mask = (labels == class_id) & (receivers == receiver) & (days == day)
            centre = _normalise(np.mean(x[mask], axis=0, keepdims=True))[0]
            centres.append(centre)
            radii.append(
                float(np.sqrt(np.sum(np.square(x[mask] - centre)) / (np.sum(mask) * FEATURE_DIM)))
            )
        anchors.append(_normalise(np.mean(np.asarray(centres), axis=0, keepdims=True))[0])
        variances.append(float(np.mean(np.square(radii))))
    return np.asarray(anchors), np.asarray(variances)


def test_builds_exact_588_geometry_and_redacts_source_rows_and_ids(tmp_path: Path) -> None:
    arrays = _strict_tap_arrays()
    expected_anchors, expected_v_g = _reference_geometry(arrays)
    tap = _write_tap(tmp_path / "d106_ls_strict_tap.npz", arrays)
    root = tmp_path / "g0_bundle"

    result = build_d111_g0_source_bundle(
        tap, root, expected_tap_sha256=_sha256(tap)
    )
    assert Path(result["root"]) == root
    assert result["class_registry"] == CLASSES
    assert result["component_state"] == COMPONENT_STATE
    bundle = load_d111_g0_source_bundle(root)
    np.testing.assert_allclose(
        _normalise(bundle.anchors), expected_anchors, rtol=0.0, atol=1.5e-2
    )
    np.testing.assert_allclose(bundle.v_g, expected_v_g, rtol=3.0e-2, atol=1.0e-8)

    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["component_state"] == "NONFORMAL_G0_FUNCTIONAL_ONLY"
    assert manifest["formal_phase2_eligible"] is False
    assert manifest["performance_claim_allowed"] is False
    assert manifest["performance_metrics_allowed"] is False
    assert manifest["target_access"] is False
    assert manifest["target_access_allowed"] is False
    assert manifest["resource_receipt"] == {
        **manifest["resource_receipt"],
        "persistent_source_rows": 0,
        "persistent_source_ids": 0,
        "persistent_query_state_bytes": 0,
    }
    assert {item.name for item in root.iterdir()} == {NPZ_NAME, MANIFEST_NAME}
    with np.load(root / NPZ_NAME, allow_pickle=False) as archive:
        assert tuple(archive.files) == PAYLOAD_MEMBERS
        assert "physical_ids" not in archive.files
        assert "observation_ids" not in archive.files
        assert "pre_relu" not in archive.files

    persisted = b"".join(item.read_bytes() for item in root.iterdir())
    assert b"PHYS-SECRET" not in persisted
    assert b"OBS-PHYS-SECRET" not in persisted


@pytest.mark.parametrize("fault", ("duplicate_physical", "missing_cell", "nonfinite"))
def test_rejects_strict_tap_identity_grid_and_finite_failures(
    tmp_path: Path, fault: str
) -> None:
    arrays = _strict_tap_arrays()
    if fault == "duplicate_physical":
        arrays["physical_ids"][1] = arrays["physical_ids"][0]
    elif fault == "missing_cell":
        missing = (arrays["receiver_ids"] == "rx-0") & (arrays["day_ids"] == "day-0")
        arrays["receiver_ids"][missing] = "rx-1"
    else:
        arrays["pre_relu"][0, 0] = np.nan
    tap = _write_tap(tmp_path / f"{fault}.npz", arrays)
    with pytest.raises(D111G0SourceBundleError):
        build_d111_g0_source_bundle(tap, tmp_path / f"{fault}_output")


def test_refuses_to_overwrite_an_immutable_g0_output(tmp_path: Path) -> None:
    arrays = _strict_tap_arrays()
    tap = _write_tap(tmp_path / "strict.npz", arrays)
    root = tmp_path / "g0_bundle"
    build_d111_g0_source_bundle(tap, root)
    with pytest.raises(FileExistsError):
        build_d111_g0_source_bundle(tap, root)


def test_strictly_positive_small_radii_do_not_trigger_a_fake_degeneracy_reject(
    tmp_path: Path,
) -> None:
    # Unit-sphere chord scatter can be much smaller than an FP16 scale while
    # still being mathematically positive and therefore a valid source cell.
    arrays = _strict_tap_arrays(sample_amplitude=0.01)
    tap = _write_tap(tmp_path / "small_positive_radius.npz", arrays)
    result = build_d111_g0_source_bundle(tap, tmp_path / "small_radius_bundle")
    assert result["component_state"] == COMPONENT_STATE
