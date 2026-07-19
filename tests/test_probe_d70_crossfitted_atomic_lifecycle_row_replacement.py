from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "code"
    / "scripts"
    / "probe_d70_crossfitted_atomic_lifecycle_row_replacement.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d70_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(class_count: int, k: int, seed: int = 700) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index in range(class_count):
        rows.append(rng.standard_normal((k, 9)) + class_index)
        labels.extend([class_index] * k)
    return np.concatenate(rows), np.asarray(labels, dtype=np.int64)


def test_build_wraps_d62_and_preserves_exact_before(monkeypatch) -> None:
    component_records = []

    def build(_d42):
        def fit(rows, labels, class_count, k_shot):
            coefficient = np.stack(
                [rows[labels == index].mean(axis=0) for index in range(class_count)]
            ).astype(np.float32)
            intercept = np.arange(class_count, dtype=np.float32)
            return coefficient, intercept, {
                "d43_class_common_affine_omitted": True,
                "d62_probe_arm": probe.d62.ARM,
            }

        return fit, component_records

    monkeypatch.setattr(probe.d62, "build_d62_fit", build)
    fit, lifecycle, records = probe.build_d70_fit(object())
    rows, labels = _support(3, 4)
    expected = np.stack(
        [rows[labels == index].mean(axis=0) for index in range(3)]
    ).astype(np.float32)
    coefficient, intercept, audit = fit(rows, labels, 3, 4)
    assert np.array_equal(coefficient, expected)
    assert np.array_equal(intercept, np.arange(3, dtype=np.float32))
    assert audit["d43_probe_arm"] == probe.ARM
    assert audit["d70_probe_arm"] == probe.ARM
    assert audit["d70_formula"] == probe.FORMULA
    assert audit["d70_exact_d62_fallback"] is True
    assert lifecycle.pending and records is component_records


def test_source_closes_counts_and_atomic_fallback_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "lifecycle.completed_pairs != 30" in source
    assert "len(lifecycle.records) != 60" in source
    assert "lifecycle.inner_fit_count != 120" in source
    assert "len(component_records) != 2280" in source
    assert '"d70_ground_component_input_count": 0' in source
    assert '"d70_query_extra_macs": 0' in source


def test_probe_has_no_parameter_scan_or_query_role_formula() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "temperature" not in source
    assert "threshold" not in source
    assert "role_specific_query_branch\": false" in source
    assert "class_id_specific_formula\": false" in source

