from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d110_scpm_g0_one_shot.py"


def _module():
    spec = importlib.util.spec_from_file_location("d110_scpm_g0_one_shot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tap_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(110)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    observations: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            cell = receiver_index * 4 + day_index
            for class_index in range(6):
                count = 4 if (cell + class_index) % 2 == 0 else 3
                for sample_index in range(count):
                    row = rng.normal(0.0, 0.012, size=160).astype(np.float32)
                    row[0] = np.float32(1.0 + 0.02 * receiver_index)
                    row[10 + class_index] += np.float32(0.42)
                    row[40 + day_index] += np.float32(0.08)
                    row[80 + receiver_index] += np.float32(0.06)
                    row[120 + sample_index] += np.float32(0.015)
                    rows.append(row)
                    labels.append(f"tx-{class_index}")
                    receivers.append(f"rx-{receiver_index}")
                    days.append(f"day-{day_index}")
                    physical.append(
                        f"p-{receiver_index:02d}-{day_index:02d}-"
                        f"{class_index:02d}-{sample_index:02d}"
                    )
                    observations.append(f"obs-{len(observations):04d}")
    pre_relu = np.stack(rows).astype(np.float32)
    assert pre_relu.shape == (588, 160)
    return {
        "pre_relu": pre_relu,
        "z_dom": np.zeros_like(pre_relu),
        "tx_labels": np.asarray(labels, dtype=np.str_),
        "receiver_ids": np.asarray(receivers, dtype=np.str_),
        "day_ids": np.asarray(days, dtype=np.str_),
        "physical_ids": np.asarray(physical, dtype=np.str_),
        "scenario_names": np.full(588, "leo_weak", dtype=np.str_),
        "observation_ids": np.asarray(observations, dtype=np.str_),
    }


def _write_npz(path: Path) -> str:
    np.savez(path, **_tap_arrays())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d110_real_archive_g0_is_thin_truth_free_and_non_overwriting(
    tmp_path: Path,
) -> None:
    module = _module()
    archive = tmp_path / "d106_ls_strict_tap.npz"
    archive_sha = _write_npz(archive)
    output = tmp_path / "d110_g0.json"
    classes = tuple(f"tx-{index}" for index in range(6))

    result = module.run_one_shot(
        archive_path=archive.resolve(),
        expected_archive_sha256=archive_sha,
        registered_classes=classes,
        run_id="d110-local-g0",
        output_path=output.resolve(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == result | {
        "output_receipt_sha256": persisted["output_receipt_sha256"]
    }
    assert result["candidate_id"] == module.CANDIDATE_ID
    assert result["real_archive_g0_executed"] is True
    assert result["performance_metrics_emitted"] is False
    assert result["query_label_read_for_scoring"] is False
    assert set(result["argmax_changed_count_by_k"]) == {"1", "5", "10"}
    assert result["g1_entry_allowed"] is result["functional_gate_pass"]
    assert result["functional_gate_status"] in {
        "G0_PASS_PROCEED_G1",
        "REJECT_REVISION_NO_FUNCTION",
    }
    assert result["geometry"]["prior_quantized_bytes"] == 12
    assert result["resource_summary"]["parameter_scan_count"] == 0
    assert result["resource_summary"]["query_state_updates"] == 0
    assert result["resource_summary"]["resource_budget_exceeded"] is False
    assert (
        result["resource_summary"]["incremental_numeric_array_peak_estimate_bytes"]
        <= result["resource_summary"]["numeric_array_budget_bytes"]
    )
    assert len(result["per_k"]) == 3
    for row in result["per_k"]:
        assert row["query_count"] == 588
        assert row["fold_count"] == 28
        assert row["runtime_state_numeric_bytes_max"] > 0
        for metric in ("feature", "neighbor", "margin", "argmax"):
            assert 0 <= row[f"{metric}_changed_count"] <= 588
            assert row[f"{metric}_changed_bitmap_roots_root_sha256"]

    serialized = json.dumps(persisted, ensure_ascii=True, sort_keys=True).lower()
    for forbidden in ('"truth', '"accuracy', '"floor', '"acc"', '"h"'):
        assert forbidden not in serialized
    with pytest.raises(module.base.OneShotG0Error, match="output must be a new file"):
        module.run_one_shot(
            archive_path=archive.resolve(),
            expected_archive_sha256=archive_sha,
            registered_classes=classes,
            run_id="d110-local-g0-repeat",
            output_path=output.resolve(),
        )
    with pytest.raises(module.base.OneShotG0Error, match="SHA256"):
        module.run_one_shot(
            archive_path=archive.resolve(),
            expected_archive_sha256="0" * 64,
            registered_classes=classes,
            run_id="d110-local-g0-wrong-sha",
            output_path=(tmp_path / "wrong.json").resolve(),
        )
