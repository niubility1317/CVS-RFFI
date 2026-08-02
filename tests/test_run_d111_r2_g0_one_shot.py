from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cvsrffi.stage2_d111_g0_source_bundle import build_d111_g0_source_bundle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d111_r2_g0_one_shot.py"
CLASSES = tuple(f"tx-{index}" for index in range(6))


def _module():
    spec = importlib.util.spec_from_file_location("d111_r2_g0_one_shot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tap_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1112)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    observations: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            cell = receiver_index * 4 + day_index
            per_class = 4 if cell % 2 == 0 else 3
            for class_index, class_id in enumerate(CLASSES):
                for sample_index in range(per_class):
                    row = rng.normal(0.0, 0.008, size=160).astype(np.float32)
                    row[:3] = np.asarray(
                        [
                            0.88 + 0.020 * (receiver_index - 3),
                            0.76 + 0.030 * (day_index - 1.5),
                            0.69 + 0.014 * (((2 * receiver_index + day_index) % 5) - 2),
                        ],
                        dtype=np.float32,
                    )
                    row[10 + class_index] = 1.75
                    row[40 + sample_index] = 0.15 * (sample_index + 1)
                    rows.append(row)
                    labels.append(class_id)
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
        "scenario_names": np.full(588, "phase1-source", dtype=np.str_),
        "observation_ids": np.asarray(observations, dtype=np.str_),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_synthetic_g0_one_shot_is_truth_free_nonperformance_and_immutable(
    tmp_path: Path,
) -> None:
    module = _module()
    archive = tmp_path / "d106_ls_strict_tap.npz"
    archive_sha = _write_npz(archive, _tap_arrays())
    bundle_dir = tmp_path / "d111_g0_bundle"
    build_d111_g0_source_bundle(
        archive, bundle_dir, expected_tap_sha256=archive_sha
    )
    output = tmp_path / "d111_g0.json"
    result = module.run_one_shot(
        archive_path=archive.resolve(),
        expected_archive_sha256=archive_sha,
        bundle_dir=bundle_dir.resolve(),
        registered_classes=CLASSES,
        run_id="d111-r2-local-g0",
        output_path=output.resolve(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == result | {"output_receipt_sha256": persisted["output_receipt_sha256"]}
    assert result["performance_metrics_emitted"] is False
    assert result["formal_performance_claim"] is False
    assert result["query_label_read_for_scoring"] is False
    assert result["resource_summary"]["query_rows_used_for_fit"] == 0
    assert result["resource_summary"]["query_state_updates"] == 0
    assert result["K_values"] == [1, 5, 10]
    assert result["fold_count"] == 28
    assert result["query_count_per_k"] == 588
    assert set(result["argmax_changed_count_by_k"]) == {"1", "5", "10"}
    assert result["g1_entry_allowed"] is result["functional_gate_pass"]
    assert result["functional_gate_status"] in {
        "G0_PASS_PROCEED_G1",
        "REJECT_REVISION_NO_FUNCTION",
    }
    assert len(result["per_k"]) == 3
    for per_k in result["per_k"]:
        assert per_k["K"] in {1, 5, 10}
        assert per_k["fold_count"] == 28
        assert per_k["query_count"] == 588
        assert per_k["feature_changed_count"] == 0
        assert (
            per_k["baseline_feature_root_sha256"]
            == per_k["candidate_feature_root_sha256"]
        )
        # D111 reweights only the old-class density mix.  It must never alter
        # the frozen M0 support-neighbor identity used by a query.
        assert per_k["neighbor_changed_count"] == 0
        assert (
            per_k["baseline_neighbor_root_sha256"]
            == per_k["candidate_neighbor_root_sha256"]
        )
        assert per_k["per_k_execution_root_sha256"]
        for metric in ("anchor", "score", "margin", "argmax"):
            assert isinstance(per_k[f"{metric}_changed_count"], int)
            assert per_k[f"{metric}_changed_count"] >= 0

    serialized = json.dumps(persisted, ensure_ascii=True, sort_keys=True).lower()
    for forbidden in ('"truth', '"accuracy', '"floor', '"h"'):
        assert forbidden not in serialized
    parameters = set(inspect.signature(module.run_one_shot).parameters)
    assert parameters == {
        "archive_path",
        "expected_archive_sha256",
        "bundle_dir",
        "registered_classes",
        "run_id",
        "output_path",
    }
    with pytest.raises(ValueError, match="output must be a new file"):
        module.run_one_shot(
            archive_path=archive.resolve(),
            expected_archive_sha256=archive_sha,
            bundle_dir=bundle_dir.resolve(),
            registered_classes=CLASSES,
            run_id="d111-r2-local-g0-repeat",
            output_path=output.resolve(),
        )


def test_any_zero_k_argmax_change_closes_revision_before_g1() -> None:
    module = _module()
    gate = module.functional_gate_from_argmax_counts({1: 17, 5: 0, 10: 4})
    assert gate == {
        "argmax_changed_count_by_k": {"1": 17, "5": 0, "10": 4},
        "zero_changed_k_values": [5],
        "functional_gate_status": "REJECT_REVISION_NO_FUNCTION",
        "functional_gate_pass": False,
    }
    with pytest.raises(module.D111R2G0Error, match="K1/K5/K10"):
        module.functional_gate_from_argmax_counts({1: 1, 5: 1})
