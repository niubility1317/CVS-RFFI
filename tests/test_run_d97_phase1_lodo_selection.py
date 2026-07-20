from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_d97_phase1_lodo_selection as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> tuple[Path, dict]:
    value = {
        "schema": runner.CONFIG_SCHEMA,
        "run_id": "d97_unit_v1",
        "seed": 97,
        "archive_path": str(tmp_path / "archive.npz"),
        "archive_manifest_path": str(tmp_path / "archive.manifest.json"),
        "archive_manifest_sha256": "1" * 64,
        "ground_component_dir": str(tmp_path / "ground"),
        "ground_manifest_sha256": "2" * 64,
        "phase1_checkpoint_sha256": runner.BASE_CHECKPOINT_SHA256,
        "device": "cpu",
        "candidate_grid": {
            "beta": [4.0],
            "temp_base": [1.0],
            "temp_qk": [1.0],
            "eta_max": [0.25],
            "k1_eta_prior": [0.1],
        },
        "expected_module_sha256": {
            name: _sha(path) for name, path in runner.REQUIRED_MODULES.items()
        },
        "output_dir": str(tmp_path / "output"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path, value


def test_config_is_exact_hash_bound_and_nonoverwriting(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    checked = runner.validate_release_config(
        runner._read_bound_json(path, _sha(path))
    )
    assert checked["run_id"] == "d97_unit_v1"
    with pytest.raises(runner.D97ReleaseRunnerError, match="path/SHA256"):
        runner._read_bound_json(path, "0" * 64)
    value["extra"] = True
    with pytest.raises(runner.D97ReleaseRunnerError, match="exact schema"):
        runner.validate_release_config(value)
    Path(checked["output_dir"]).mkdir()
    with pytest.raises(runner.D97ReleaseRunnerError, match="already exists"):
        runner.validate_release_config(json.loads(path.read_text(encoding="utf-8")))


def test_module_hash_and_checkpoint_drift_fail_closed(tmp_path: Path) -> None:
    path, value = _config(tmp_path)
    del path
    value["expected_module_sha256"]["stage2_d81_phase1_episode_scorer"] = "0" * 64
    with pytest.raises(runner.D97ReleaseRunnerError, match="module source"):
        runner.validate_release_config(value)
    value["expected_module_sha256"] = {
        name: _sha(module_path)
        for name, module_path in runner.REQUIRED_MODULES.items()
    }
    value["phase1_checkpoint_sha256"] = "0" * 64
    with pytest.raises(runner.D97ReleaseRunnerError, match="checkpoint identity"):
        runner.validate_release_config(value)


def test_run_writes_one_verified_immutable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _config(tmp_path)

    class FakeScorer:
        scorer_id = "3" * 64

    monkeypatch.setattr(
        runner.D81Phase1EpisodeScorer,
        "from_component",
        lambda *args, **kwargs: FakeScorer(),
    )
    receipt = {
        "receipt_sha256": "4" * 64,
        "development_lock_frozen": True,
        "full_phase1_lock": False,
        "formal_target_claim_allowed": False,
        "selected_parameters": {"beta": 4.0},
        "outer_lodo_summary": {"balanced_accuracy": 0.5},
        "final_lock_evaluation_summary": {"balanced_accuracy": 0.5},
        "int8_margin_audit": {"aggregate": {"final_top1_flip_rate": 0.0}},
    }
    monkeypatch.setattr(runner, "run_phase1_lodo_selection", lambda *a, **k: receipt)
    monkeypatch.setattr(runner, "verify_receipt", lambda value: value is receipt)
    result = runner.run_from_config(config_path, _sha(config_path))
    assert result["development_lock_frozen"] is True
    assert Path(result["result_path"]).is_file()
    assert (Path(result["result_path"]).parent / "d97_phase1_lodo_receipt.json").is_file()
    with pytest.raises(runner.D97ReleaseRunnerError, match="already exists"):
        runner.run_from_config(config_path, _sha(config_path))

