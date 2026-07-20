from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d99_d100_phase1_lodo import current_code_sha256
from scripts import run_d99_d100_phase1_lodo as runner


def _grid() -> dict[str, list[float]]:
    return {
        "eta": [0.25],
        "student_nu": [3.0],
        "kernel_volume_gamma": [1.0],
        "shared_h0": [0.5],
        "scale_prior_strength": [2.0],
        "scale_min_ratio": [0.5],
        "scale_max_ratio": [2.0],
        "d99_temperature": [1.0],
        "lambda0": [0.1],
        "ridge_temperature": [1.0],
        "alpha": [0.35],
    }


def _config(tmp_path: Path) -> dict:
    return {
        "schema": runner.CONFIG_SCHEMA,
        "run_id": "d99_d100_lodo_unit_v1",
        "seed": 991,
        "feature_archive_path": str(tmp_path / "archive.npz"),
        "feature_archive_sha256": "1" * 64,
        "feature_archive_manifest_path": str(tmp_path / "archive.manifest.json"),
        "feature_archive_manifest_sha256": "2" * 64,
        "ground_bundle_npz_path": str(tmp_path / "ground.npz"),
        "ground_bundle_npz_sha256": "3" * 64,
        "ground_release_manifest_path": str(tmp_path / "ground.manifest.json"),
        "ground_release_manifest_sha256": "4" * 64,
        "base_d99_lock_path": str(tmp_path / "d99.json"),
        "base_d99_lock_sha256": "5" * 64,
        "d81_ground_component_dir": str(tmp_path / "d81"),
        "d81_ground_manifest_sha256": "6" * 64,
        "d81_device": "cpu",
        "d81_metric_seed": 713101,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "candidate_grid": _grid(),
        "expected_module_sha256": current_code_sha256(),
        "execution_mode": "development_diagnostic",
        "output_dir": str(tmp_path / "output"),
    }


def _base_d99_values() -> dict:
    return {
        "density_tau": 0.2,
        "max_ground_rank": 2,
        "max_target_rank": 2,
        "coverage_floor": 0.01,
        "ground_energy_scale": 0.01,
        "target_energy_scale": 0.01,
        "shrinkage_prior_strength": 2.0,
        "ground_weight_max": 0.8,
        "target_weight_max": 0.6,
        "student_nu": 3.0,
        "kernel_effective_dim": 12,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.5,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "z_weight": 0.7,
        "fft_weight": 0.2,
        "rf_weight": 0.1,
        "eta_k1": 0.1,
        "eta_k5": 0.2,
        "eta_k10": 0.3,
        "eta_k20": 0.4,
        "eta_k20_lodo_artifact_sha256": None,
        "phase1_receipt_sha256": "1" * 64,
        "ground_aggregation_receipt_sha256": "2" * 64,
        "ground_bundle_receipt_sha256": "3" * 64,
        "quantization_margin_audit_sha256": "4" * 64,
        "validation_method_lock_sha256": "5" * 64,
        "d81_phase1_lock_sha256": "6" * 64,
        "ground_old_registry": ["old-a", "old-b"],
    }


def test_release_config_is_exact_code_bound_and_nonoverwriting(tmp_path: Path) -> None:
    value = _config(tmp_path)
    checked = runner.validate_release_config(value)
    assert checked["run_id"] == "d99_d100_lodo_unit_v1"
    value["extra"] = True
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="exact schema"):
        runner.validate_release_config(value)
    value = _config(tmp_path)
    value["expected_module_sha256"] = dict(value["expected_module_sha256"])
    value["expected_module_sha256"]["stage2_d99_d100_phase1_lodo"] = "0" * 64
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="source SHA"):
        runner.validate_release_config(value)
    output = Path(_config(tmp_path)["output_dir"])
    output.mkdir()
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="already exists"):
        runner.validate_release_config(_config(tmp_path))


def test_bound_json_and_exclusive_writer_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    raw = json.dumps({"a": 1}, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    import hashlib

    assert runner._read_bound_json(path, hashlib.sha256(raw).hexdigest(), "config") == {"a": 1}
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="path/SHA256"):
        runner._read_bound_json(path, "0" * 64, "config")
    output = tmp_path / "immutable.json"
    runner._exclusive_write(output, b"first")
    with pytest.raises(FileExistsError):
        runner._exclusive_write(output, b"second")
    assert output.read_bytes() == b"first"


def test_nonformal_diagnostic_requires_explicit_execution_mode(tmp_path: Path) -> None:
    value = _config(tmp_path)
    value["execution_mode"] = "formal_lock"
    checked = runner.validate_release_config(value)
    assert checked["execution_mode"] == "formal_lock"
    value["execution_mode"] = "auto"
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="execution_mode"):
        runner.validate_release_config(value)

    diagnostic = {"status": runner.STATUS_DIAGNOSTIC}
    assert runner._validate_execution_outcome(diagnostic, "development_diagnostic") is False
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="refuses nonformal"):
        runner._validate_execution_outcome(diagnostic, "formal_lock")
    formal = {"status": runner.STATUS_FORMAL}
    assert runner._validate_execution_outcome(formal, "formal_lock") is True
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="cannot emit a formal"):
        runner._validate_execution_outcome(formal, "development_diagnostic")


def test_development_d99_prior_wrapper_is_required_and_formal_refuses_it() -> None:
    values = _base_d99_values()
    wrapper = {
        "schema": runner.DEVELOPMENT_D99_PRIOR_SCHEMA,
        "status": runner.DEVELOPMENT_D99_PRIOR_STATUS,
        "values": values,
        "placeholder_evidence_fields": list(
            runner.DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS
        ),
    }
    parsed = runner._parse_base_d99_lock(wrapper, "development_diagnostic")
    assert type(parsed) is d99.Phase1D99Lock
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="requires exact"):
        runner._parse_base_d99_lock(values, "development_diagnostic")
    with pytest.raises(runner.D99D100ReleaseRunnerError, match="formal_lock refuses"):
        runner._parse_base_d99_lock(wrapper, "formal_lock")
    parsed_formal = runner._parse_base_d99_lock(values, "formal_lock")
    assert type(parsed_formal) is d99.Phase1D99Lock
