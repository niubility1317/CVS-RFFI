from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d121_lbr_g0_one_shot.py"


def _one_shot_module():
    spec = importlib.util.spec_from_file_location("d121_one_shot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tap_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(106)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    scenarios: list[str] = []
    observations: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            cell_index = receiver_index * 4 + day_index
            for class_index in range(6):
                count = 4 if (cell_index + class_index) % 2 == 0 else 3
                for sample_index in range(count):
                    pre_relu = rng.normal(0.0, 0.002, size=160).astype(np.float32)
                    pre_relu[0] = np.float32(1.0)
                    pre_relu[10 + class_index] = np.float32(0.24 + 0.002 * sample_index)
                    signed_class = (class_index + 1) % 6 if cell_index == 0 else class_index
                    if cell_index == 0:
                        pre_relu[10 + signed_class] = np.float32(0.235)
                    pre_relu[30 + signed_class] = np.float32(-6.0)
                    rows.append(pre_relu)
                    labels.append(f"tx-{class_index}")
                    receivers.append(f"rx-{receiver_index}")
                    days.append(f"day-{day_index}")
                    physical.append(
                        f"p-{receiver_index:02d}-{day_index:02d}-"
                        f"{class_index:02d}-{sample_index:02d}"
                    )
                    scenarios.append("leo_weak")
                    observations.append(f"obs-{len(observations):04d}")
    pre_relu = np.stack(rows).astype(np.float32)
    assert pre_relu.shape == (588, 160)
    return {
        "pre_relu": pre_relu,
        "z_dom": np.zeros_like(pre_relu, dtype=np.float32),
        "tx_labels": np.asarray(labels, dtype=np.str_),
        "receiver_ids": np.asarray(receivers, dtype=np.str_),
        "day_ids": np.asarray(days, dtype=np.str_),
        "physical_ids": np.asarray(physical, dtype=np.str_),
        "scenario_names": np.asarray(scenarios, dtype=np.str_),
        "observation_ids": np.asarray(observations, dtype=np.str_),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_archive_g0_emits_only_mechanical_lbr_evidence(tmp_path: Path) -> None:
    module = _one_shot_module()
    archive = tmp_path / "d121_ls_strict_tap.npz"
    archive_sha256 = _write_npz(archive, _tap_arrays())
    output = tmp_path / "d121-g0.json"

    result = module.run_one_shot(
        archive_path=archive.resolve(),
        expected_archive_sha256=archive_sha256,
        registered_classes=tuple(f"tx-{index}" for index in range(6)),
        run_id="local-d121-no-performance-smoke",
        output_path=output.resolve(),
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result["archive_sha256"] == archive_sha256
    assert result["candidate_id"] == "D121_LBR_QKNN_M_HEAD"
    assert result["real_archive_g0_executed"] is True
    assert result["performance_metrics_emitted"] is False
    assert result["formal_performance_claim"] is False
    assert set(result["argmax_changed_count_by_k"]) == {"1", "5", "10"}
    assert len(result["per_k"]) == 3
    for per_k in result["per_k"]:
        assert per_k["query_count"] == 588
        assert per_k["lbr_rival_index_roots_root_sha256"]
        assert per_k["rival_index_count"] > 0
        for metric in ("support_kernel", "score", "margin", "argmax"):
            assert isinstance(per_k[f"{metric}_changed_count"], int)
            assert per_k[f"{metric}_changed_bitmap_roots_root_sha256"]
            assert per_k[f"baseline_{metric}_root_sha256"]
            assert per_k[f"candidate_{metric}_root_sha256"]
    assert result["resource_summary"]["query_rows_used_for_fit"] == 0
    assert result["resource_summary"]["query_state_updates"] == 0
    assert result["resource_summary"]["query_selection_count"] == 0
    serialized = json.dumps(persisted, sort_keys=True).lower()
    assert all(token not in serialized for token in ("truth", "accuracy", "floor", "\"h\""))


def test_g0_argmax_rule_and_forbidden_archive_field(tmp_path: Path) -> None:
    module = _one_shot_module()
    assert module.functional_gate_from_argmax_counts({1: 1, 5: 2, 10: 3})[
        "functional_gate_pass"
    ]
    assert not module.functional_gate_from_argmax_counts({1: 1, 5: 0, 10: 3})[
        "functional_gate_pass"
    ]

    arrays = _tap_arrays()
    arrays["truth"] = np.asarray(["not-used"], dtype=np.str_)
    archive = tmp_path / "d121-forbidden.npz"
    archive_sha256 = _write_npz(archive, arrays)
    with pytest.raises(module.base.OneShotG0Error, match="forbidden performance field"):
        module.run_one_shot(
            archive_path=archive.resolve(),
            expected_archive_sha256=archive_sha256,
            registered_classes=tuple(f"tx-{index}" for index in range(6)),
            run_id="forbidden-field-smoke",
            output_path=(tmp_path / "forbidden.json").resolve(),
        )
