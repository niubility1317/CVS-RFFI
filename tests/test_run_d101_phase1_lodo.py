from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi import stage2_d101_phase1_lodo as d101_lodo
from scripts import run_d101_phase1_lodo as runner


def _d99_d100_grid() -> dict[str, list[float]]:
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


def _d101_grid() -> dict[str, list[float]]:
    return {
        "block_variance_z160": [0.8],
        "block_variance_fft96": [1.1],
        "block_variance_rf32": [0.7],
        "prior_dof": [8.0],
        "target_rank_k5plus": [2.0],
        "lambda_relative": [0.08],
        "rda_temperature": [0.9],
        "d101_alpha": [0.35],
    }


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _config(tmp_path: Path) -> dict:
    archive = tmp_path / "archive.npz"
    archive_sha = _write(archive, b"immutable-feature-archive")
    base_lock = tmp_path / "d99.json"
    base_lock_sha = _write(base_lock, b"{}")
    return {
        "schema": runner.CONFIG_SCHEMA,
        "run_id": "d101_lodo_unit_v1",
        "seed": 991,
        "feature_archive_path": str(archive),
        "feature_archive_sha256": archive_sha,
        "feature_archive_manifest_path": str(tmp_path / "archive.manifest.json"),
        "feature_archive_manifest_sha256": "2" * 64,
        "ground_bundle_npz_path": str(tmp_path / "ground.npz"),
        "ground_bundle_npz_sha256": "3" * 64,
        "ground_release_manifest_path": str(tmp_path / "ground.manifest.json"),
        "ground_release_manifest_sha256": "4" * 64,
        "base_d99_lock_path": str(base_lock),
        "base_d99_lock_sha256": base_lock_sha,
        "d81_ground_component_dir": str(tmp_path / "d81"),
        "d81_ground_manifest_sha256": "6" * 64,
        "d81_device": "cpu",
        "d81_metric_seed": 713101,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "d99_d100_grid": _d99_d100_grid(),
        "d101_grid": _d101_grid(),
        "gate_lock": asdict(d101_lodo.D101LODOGateLock()),
        "expected_module_sha256": d101_lodo.current_code_sha256(),
        "execution_mode": runner.EXECUTION_MODE,
        "output_dir": str(tmp_path / "output"),
    }


def _receipt(status: str) -> dict:
    return {
        "schema": d101_lodo.SCHEMA,
        "status": status,
        "scientific_phase1_hard_gate_passed": status == d101_lodo.STATUS_ADMITTED,
        "formal_phase1_lock": False,
        "formal_phase2_eligible": False,
        "target_authority": False,
        "n607_authority": False,
        "canonical_lock_artifact_write_allowed": False,
        "receipt_sha256": "a" * 64,
    }


def _write_config(tmp_path: Path, value: dict) -> tuple[Path, str]:
    path = tmp_path / "config.json"
    raw = json.dumps(value, sort_keys=True).encode("utf-8")
    return path, _write(path, raw)


def _patch_run_dependencies(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> tuple[object, object, object]:
    bundle = object()
    authority = object()
    base_lock = object()
    monkeypatch.setattr(
        runner.d99_runner, "_load_ground_bundle", lambda _config: (bundle, b"manifest")
    )
    monkeypatch.setattr(
        runner.d99_runner,
        "load_ground_release_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        runner.d99_runner,
        "_parse_base_d99_lock",
        lambda _payload, _mode: base_lock,
    )

    class FakeScorer:
        scorer_id = "b" * 64

        @classmethod
        def from_component(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(runner.d99_runner, "D81Phase1EpisodeScorer", FakeScorer)
    monkeypatch.setattr(
        runner.d101_lodo,
        "run_phase1_d101_nested_lodo",
        lambda *_args, **_kwargs: _receipt(status),
    )
    monkeypatch.setattr(runner.d101_lodo, "verify_receipt", lambda _value: True)
    return bundle, authority, base_lock


def test_config_is_exact_sha_bound_development_only_and_closes_both_grids_and_gate(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    checked = runner.validate_release_config(value)
    assert checked["run_id"] == "d101_lodo_unit_v1"
    assert checked["execution_mode"] == runner.EXECUTION_MODE
    assert type(checked["gate_lock"]) is d101_lodo.D101LODOGateLock
    assert d101_lodo.base.candidate_grid(checked["d99_d100_grid"])
    assert d101_lodo.d101_candidate_grid(checked["d101_grid"])

    extra = dict(value)
    extra["extra"] = True
    with pytest.raises(runner.D101ReleaseRunnerError, match="exact schema"):
        runner.validate_release_config(extra)
    formal = dict(value)
    formal["execution_mode"] = "formal_lock"
    with pytest.raises(runner.D101ReleaseRunnerError, match="development_diagnostic-only"):
        runner.validate_release_config(formal)
    bad_gate = dict(value)
    bad_gate["gate_lock"] = dict(value["gate_lock"])
    bad_gate["gate_lock"].pop("oracle_union_gain_floor")
    with pytest.raises(runner.D101ReleaseRunnerError, match="gate_lock field"):
        runner.validate_release_config(bad_gate)
    drift = dict(value)
    drift["expected_module_sha256"] = dict(value["expected_module_sha256"])
    drift["expected_module_sha256"]["stage2_d101_phase1_lodo"] = "0" * 64
    with pytest.raises(runner.D101ReleaseRunnerError, match="source SHA"):
        runner.validate_release_config(drift)
    Path(value["output_dir"]).mkdir()
    with pytest.raises(runner.D101ReleaseRunnerError, match="already exists"):
        runner.validate_release_config(value)


@pytest.mark.parametrize(
    "status", [d101_lodo.STATUS_ADMITTED, d101_lodo.STATUS_REJECTED]
)
def test_admitted_and_legal_reject_receipts_publish_as_atomic_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    value = _config(tmp_path)
    config_path, config_sha = _write_config(tmp_path, value)
    captured = {}
    _patch_run_dependencies(monkeypatch, status)

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _receipt(status)

    monkeypatch.setattr(runner.d101_lodo, "run_phase1_d101_nested_lodo", fake_run)
    result = runner.run_from_config(config_path, config_sha)
    output = Path(value["output_dir"])
    assert result["status"] == status
    assert result["development_diagnostic_only"] is True
    assert result["canonical_lock_artifact"] is False
    assert result["formal_phase1_lock"] is False
    assert result["formal_phase2_eligible"] is False
    assert result["target_authority"] is False
    assert result["n607_authority"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        runner.RECEIPT_FILENAME,
        "result.json",
    ]
    assert json.loads((output / runner.RECEIPT_FILENAME).read_text())["status"] == status
    assert json.loads((output / "result.json").read_text()) == result
    assert captured["kwargs"]["d99_d100_grid"] == value["d99_d100_grid"]
    assert captured["kwargs"]["d101_grid"] == value["d101_grid"]
    assert asdict(captured["kwargs"]["gate_lock"]) == value["gate_lock"]
    assert captured["kwargs"]["code_sha256"] == value["expected_module_sha256"]
    assert captured["kwargs"]["seed"] == value["seed"]
    with pytest.raises(runner.D101ReleaseRunnerError, match="already exists"):
        runner.run_from_config(config_path, config_sha)


def test_verify_failure_never_creates_output_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _config(tmp_path)
    config_path, config_sha = _write_config(tmp_path, value)
    _patch_run_dependencies(monkeypatch, d101_lodo.STATUS_ADMITTED)
    monkeypatch.setattr(runner.d101_lodo, "verify_receipt", lambda _value: False)
    with pytest.raises(runner.D101ReleaseRunnerError, match="fixed point"):
        runner.run_from_config(config_path, config_sha)
    checked = runner.validate_release_config(value)
    assert not Path(checked["output_dir"]).exists()
    assert not Path(checked["staging_dir"]).exists()


def test_config_file_sha_drift_fails_before_any_output(tmp_path: Path) -> None:
    value = _config(tmp_path)
    config_path, _config_sha = _write_config(tmp_path, value)
    with pytest.raises(
        runner.d99_runner.D99D100ReleaseRunnerError, match="path/SHA256 drift"
    ):
        runner.run_from_config(config_path, "0" * 64)
    checked = runner.validate_release_config(value)
    assert not Path(checked["output_dir"]).exists()
    assert not Path(checked["staging_dir"]).exists()


def test_second_exclusive_write_failure_leaves_no_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _config(tmp_path)
    config_path, config_sha = _write_config(tmp_path, value)
    _patch_run_dependencies(monkeypatch, d101_lodo.STATUS_REJECTED)
    original = runner.d99_runner._exclusive_write
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected result write failure")
        original(path, payload)

    monkeypatch.setattr(runner.d99_runner, "_exclusive_write", fail_second)
    with pytest.raises(OSError, match="injected"):
        runner.run_from_config(config_path, config_sha)
    checked = runner.validate_release_config(value)
    assert not Path(checked["output_dir"]).exists()
    assert not Path(checked["staging_dir"]).exists()
