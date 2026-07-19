from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d67_crossfitted_registry_consistent_row_stacking.py"
SPEC = importlib.util.spec_from_file_location("test_d67_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d67 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d67)


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


def _fake_d62_builder(_: Any):
    records: list[dict[str, Any]] = []

    def fit(rows, labels, class_count, k_shot):
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        means = np.stack([x[y == index].mean(axis=0) for index in range(class_count)])
        coefficient = means.astype(np.float32)
        intercept = (-0.5 * np.sum(means * means, axis=1)).astype(np.float32)
        records.extend({"synthetic": True} for _ in range(2 * (int(k_shot) + 1)))
        return coefficient, intercept, {"d62_boundary_status": "synthetic_d62"}

    return fit, records


def _fake_d65_expert(
    _d42, rows, labels, class_count, k_shot, old_class_count
):
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([x[y == index].mean(axis=0) for index in range(class_count)])
    coefficient = means.copy()
    coefficient[:, 3] += np.linspace(-0.25, 0.25, class_count)
    intercept = -0.5 * np.sum(coefficient * means, axis=1)
    return coefficient.astype(np.float32), intercept.astype(np.float32), {
        "covariance_sha256": "0" * 64,
        "old_class_count": int(old_class_count),
        "class_count": int(class_count),
        "k_shot": int(k_shot),
    }


def test_build_d67_fit_has_four_fold_exact_once_and_closes_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(d67.d62, "build_d62_fit", _fake_d62_builder)
    monkeypatch.setattr(d67, "_d65_expert", _fake_d65_expert)
    fit, records, state = d67.build_d67_fit(_FakeD42())
    before_x, before_y = _support(3, 8)
    final_x, final_y = _support(4, 8)
    before_coef, before_intercept, before = fit(before_x, before_y, 3, 8)
    final_coef, final_intercept, final = fit(final_x, final_y, 4, 8)
    assert before_coef.shape == (3, 4)
    assert before_intercept.shape == (3,)
    assert final_coef.shape == (4, 4)
    assert final_intercept.shape == (4,)
    assert before["d67_phase"] == "stage2b_before"
    assert final["d67_phase"] == "stage2c_final"
    assert before["d67_crossfit_fold_count"] == 4
    assert final["d67_crossfit_fold_count"] == 4
    assert all(
        item["held_train_intersection_count"] == 0
        for item in before["d67_crossfit_partition_audit"]
    )
    assert np.all((np.asarray(before["d67_alpha_by_class"]) >= 0.0))
    assert np.all((np.asarray(before["d67_alpha_by_class"]) <= 1.0))
    assert len(records) == 2
    assert state["lifecycle"]["completed_pairs"] == 1
    assert state["lifecycle"]["pending_old_class_count"] is None


def test_k4_is_prediction_exact_d62_fallback(monkeypatch) -> None:
    monkeypatch.setattr(d67.d62, "build_d62_fit", _fake_d62_builder)
    monkeypatch.setattr(d67, "_d65_expert", _fake_d65_expert)
    fit, _, _ = d67.build_d67_fit(_FakeD42())
    rows, labels = _support(3, 4)
    coefficient, intercept, audit = fit(rows, labels, 3, 4)
    expected_coefficient, expected_intercept, _ = _fake_d62_builder(None)[0](
        rows, labels, 3, 4
    )
    assert np.array_equal(coefficient, expected_coefficient)
    assert np.array_equal(intercept, expected_intercept)
    assert audit["d67_boundary_status"] == "k_le_4_exact_d62_fallback"
    assert audit["d67_crossfit_fold_count"] == 0
    assert audit["d67_alpha_by_class"] == [0.0, 0.0, 0.0]


def test_probe_source_has_no_role_scene_or_threshold_weight_branch() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "true_role ==",
        "scenario ==",
        "receiver ==",
        "class_handle ==",
        "alpha_grid",
        "temperature_grid",
        "query_labels",
        "query_class_quota",
    )
    assert not any(token in source for token in forbidden)
