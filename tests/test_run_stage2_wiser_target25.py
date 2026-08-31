from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def _module():
    path = Path(__file__).parents[1] / "code" / "scripts" / "run_stage2_wiser_target25.py"
    spec = importlib.util.spec_from_file_location("run_stage2_wiser_target25", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target25_cli_exposes_only_the_four_lifecycle_commands() -> None:
    script = Path(__file__).parents[1] / "code" / "scripts" / "run_stage2_wiser_target25.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    for command in ("prepare", "run-shard", "score-shard", "analyze"):
        assert command in result.stdout
    assert "smoke" not in result.stdout


def test_target25_prediction_validator_does_not_open_truth(tmp_path: Path) -> None:
    module = _module()
    assert module._validate_prediction_registry is not None


def test_target25_champion_file_hashes_reject_same_id_wrong_bytes(tmp_path: Path) -> None:
    module = _module()
    paths = {name: tmp_path / name for name in ("config.json", "checkpoint.pt", "summary.npz", "binding.json")}
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    job = {
        "champion_p3_config_sha256": module._sha256(paths["config.json"]),
        "champion_checkpoint_sha256": module._sha256(paths["checkpoint.pt"]),
        "champion_source_summary_sha256": module._sha256(paths["summary.npz"]),
        "champion_source_binding_sha256": module._sha256(paths["binding.json"]),
    }
    module._validate_champion_files(job=job, p3_config=paths["config.json"], checkpoint=paths["checkpoint.pt"], source_summary=paths["summary.npz"], source_binding=paths["binding.json"])
    paths["summary.npz"].write_bytes(b"different-but-same-semantic-id")
    with pytest.raises(ValueError, match="source_summary"):
        module._validate_champion_files(job=job, p3_config=paths["config.json"], checkpoint=paths["checkpoint.pt"], source_summary=paths["summary.npz"], source_binding=paths["binding.json"])
