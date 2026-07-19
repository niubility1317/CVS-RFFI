from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d69_frozen_d62_old_append_d62_new.py"
SPEC = importlib.util.spec_from_file_location("probe_d69_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _support(class_count: int, k: int, seed: int = 690) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index in range(class_count):
        rows.append(rng.standard_normal((k, 9)) + class_index)
        labels.extend([class_index] * k)
    return np.concatenate(rows), np.asarray(labels, dtype=np.int64)


def test_build_wraps_d62_without_changing_before(monkeypatch) -> None:
    calls = []

    def build(_d42):
        def fit(rows, labels, class_count, k_shot):
            calls.append((class_count, k_shot))
            coef = np.stack(
                [rows[labels == index].mean(axis=0) for index in range(class_count)]
            ).astype(np.float32)
            bias = np.arange(class_count, dtype=np.float32)
            return coef, bias, {
                "d43_class_common_affine_omitted": True,
                "d62_probe_arm": probe.d62.ARM,
            }

        return fit, []

    monkeypatch.setattr(probe.d62, "build_d62_fit", build)
    fit, lifecycle, records = probe.build_d69_fit(object())
    rows, labels = _support(3, 4)
    expected_coef = np.stack(
        [rows[labels == index].mean(axis=0) for index in range(3)]
    ).astype(np.float32)
    coef, bias, audit = fit(rows, labels, 3, 4)
    assert np.array_equal(coef, expected_coef)
    assert np.array_equal(bias, np.arange(3, dtype=np.float32))
    assert audit["d43_probe_arm"] == probe.ARM
    assert audit["d69_probe_arm"] == probe.ARM
    assert audit["d69_formula"] == probe.FORMULA
    assert calls == [(3, 4)] and records == [] and lifecycle.pending


@dataclass
class _State:
    classes: tuple[str, ...]
    coef1_qint8: np.ndarray
    coef2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    intercept_fp16: np.ndarray
    coef_fp32: np.ndarray
    intercept_fp32: np.ndarray


def _state(class_count: int) -> _State:
    return _State(
        classes=tuple(f"c{index}" for index in range(class_count)),
        coef1_qint8=np.arange(class_count * 2).reshape(class_count, 2),
        coef2_qint8=np.arange(class_count * 3).reshape(class_count, 3),
        scale1_fp16=np.arange(class_count, dtype=np.float16),
        scale2_fp16=np.arange(class_count, dtype=np.float16),
        intercept_fp16=np.arange(class_count, dtype=np.float16),
        coef_fp32=np.arange(class_count * 5, dtype=np.float32).reshape(class_count, 5),
        intercept_fp32=np.arange(class_count, dtype=np.float32),
    )


def test_compiled_state_old_row_identity_checks_all_fields() -> None:
    before = _state(3)
    final = _state(5)
    assert probe._state_old_rows_equal(before, final)
    final.coef2_qint8[1, 1] += 1
    assert not probe._state_old_rows_equal(before, final)


def test_probe_source_has_no_ground_query_or_parameter_scan() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert '"d69_ground_component_input_count": 0' in source
    assert '"d69_query_extra_macs": 0' in source
    assert "temperature" not in source
    assert "offset" not in source
    assert "threshold" not in source
    assert "lifecycle.completed_pairs != 30" in source
    assert "len(lifecycle.records) != 60" in source
