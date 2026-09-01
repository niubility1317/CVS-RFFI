from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_marc_ot_phase1_bundle.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_marc_ot_phase1_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_exposes_only_frozen_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--device" in completed.stdout
    assert "--target" not in completed.stdout
    assert "--query" not in completed.stdout


def test_cli_parser_requires_exact_three_arguments() -> None:
    module = _load_script_module()
    parser = module.build_parser()
    parsed = parser.parse_args(
        ["--config", "config.json", "--output-root", "out", "--device", "cpu"]
    )
    assert vars(parsed) == {
        "config": "config.json",
        "output_root": "out",
        "device": "cpu",
    }
