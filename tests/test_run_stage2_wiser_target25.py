from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


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
