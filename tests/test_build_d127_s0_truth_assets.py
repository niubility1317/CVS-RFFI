from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_truth_asset_builder_cli_exposes_only_pinned_post_open_inputs() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "build_d127_s0_truth_assets.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--truth-open-event" in result.stdout
    assert "--d92-retry2-manifest-sha256" in result.stdout
    assert "--truth-catalog-output" in result.stdout
    assert "--formal-d92-reference-output" in result.stdout
    assert "--build-receipt-output" in result.stdout
    assert "prediction" in result.stdout and "truth" in result.stdout
