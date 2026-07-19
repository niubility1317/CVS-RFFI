from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d68_crossfitted_signed_frozen_registry_calibration.py"
SPEC = importlib.util.spec_from_file_location("test_d68_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d68 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d68)


class _FakeD42:
    FEATURE_DIM = 4


def _support(class_count: int, k_shot: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for class_index in range(class_count):
        for rank in range(k_shot):
            vector = np.zeros(4, dtype=np.float64)
            vector[class_index % 3] = 1.0 + 0.03 * rank
            vector[3] = 0.2 * class_index + 0.01 * rank
            rows.append(vector)
            labels.append(class_index)
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def _rows(
    rows: np.ndarray, labels: np.ndarray, class_count: int, invert: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([x[y == index].mean(axis=0) for index in range(class_count)])
    coefficient = means.copy()
    if invert and class_count > 1:
        coefficient[1] *= -1.0
    intercept = -0.5 * np.sum(coefficient * means, axis=1)
    return coefficient.astype(np.float32), intercept.astype(np.float32)


def _fake_d65_builder(_: Any):
    records: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {"pending": None, "completed_pairs": 0}

    def fit(rows, labels, class_count, k_shot):
        if lifecycle["pending"] is None:
            old_count = int(class_count)
            phase = "stage2b_fit_and_freeze"
            lifecycle["pending"] = old_count
        else:
            old_count = int(lifecycle["pending"])
            phase = "stage2c_append_only"
            lifecycle["pending"] = None
            lifecycle["completed_pairs"] += 1
        coefficient, intercept = _rows(rows, labels, class_count)
        records.append({"phase": phase})
        return coefficient, intercept, {
            "d65_old_class_count": old_count,
            "d65_phase": phase,
            "d65_actual_k": int(k_shot),
        }

    return fit, records, lifecycle


def _fake_d62_builder(_: Any):
    records: list[dict[str, Any]] = []

    def fit(rows, labels, class_count, k_shot):
        coefficient, intercept = _rows(rows, labels, class_count, invert=False)
        return coefficient, intercept, {"d62_boundary_status": "synthetic_d62"}

    return fit, records


def _fake_d65_expert(
    _d42, rows, labels, class_count, k_shot, old_class_count
):
    coefficient, intercept = _rows(rows, labels, class_count)
    return coefficient, intercept, {
        "covariance_sha256": "0" * 64,
        "old_class_count": int(old_class_count),
        "class_count": int(class_count),
        "k_shot": int(k_shot),
    }


def test_build_d68_fit_flips_inverted_rows_and_closes_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(d68.d65, "build_d65_fit", _fake_d65_builder)
    monkeypatch.setattr(d68.d62, "build_d62_fit", _fake_d62_builder)
    monkeypatch.setattr(d68.d67, "_d65_expert", _fake_d65_expert)
    fit, records, state = d68.build_d68_fit(_FakeD42())
    before_x, before_y = _support(3, 8)
    final_x, final_y = _support(4, 8)
    before_coefficient, _, before = fit(before_x, before_y, 3, 8)
    final_coefficient, _, final = fit(final_x, final_y, 4, 8)
    assert before_coefficient.shape == (3, 4)
    assert final_coefficient.shape == (4, 4)
    assert before["d68_phase"] == "stage2b_before"
    assert final["d68_phase"] == "stage2c_final"
    assert before["d68_crossfit_fold_count"] == 8
    assert final["d68_crossfit_fold_count"] == 8
    assert before["d68_orientation_by_class"][1] == -1.0
    assert np.array_equal(before_coefficient, final_coefficient[:3])
    assert final["d68_old_row_fp32_bitwise_unchanged"] is True
    assert all(
        item["held_train_intersection_count"] == 0
        for item in final["d68_crossfit_partition_audit"]
    )
    assert len(records) == 2
    assert state["d65_lifecycle"]["completed_pairs"] == 1
    assert state["d65_lifecycle"]["pending"] is None
    assert state["calibration_lifecycle"]["completed_pairs"] == 1
    assert state["calibration_lifecycle"]["pending"] is None


def test_k1_is_prediction_exact_d62_fallback(monkeypatch) -> None:
    monkeypatch.setattr(d68.d65, "build_d65_fit", _fake_d65_builder)
    monkeypatch.setattr(d68.d62, "build_d62_fit", _fake_d62_builder)
    fit, _, _ = d68.build_d68_fit(_FakeD42())
    rows, labels = _support(3, 1)
    coefficient, intercept, audit = fit(rows, labels, 3, 1)
    expected_coefficient, expected_intercept, _ = _fake_d62_builder(None)[0](
        rows, labels, 3, 1
    )
    assert np.array_equal(coefficient, expected_coefficient)
    assert np.array_equal(intercept, expected_intercept)
    assert audit["d68_boundary_status"] == "k1_exact_d62_fallback"
    assert audit["d68_crossfit_fold_count"] == 0


def test_probe_source_has_no_role_scene_threshold_or_ground_branch() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "true_role ==",
        "scenario ==",
        "receiver ==",
        "class_handle ==",
        "orientation_threshold",
        "temperature_grid",
        "query_labels",
        "query_class_quota",
        "component_dir",
    )
    assert not any(token in source for token in forbidden)


def test_helper_hashes_are_bound_to_probe_root() -> None:
    hashes = d68._helper_hashes_for_probe_root(ROOT)
    assert set(hashes) == {
        "d68_d67_helper_sha256",
        "d68_d62_helper_sha256",
        "d68_d65_helper_sha256",
        "d68_d67_core_sha256",
        "d68_core_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())
