import json
from pathlib import Path
import subprocess
import sys


def test_finalize_runner_resources_cli_help() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "finalize_d103_r2_runner_resources.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--matrix-status" in result.stdout
    assert "--run-root" in result.stdout


def test_finalize_runner_resources_charges_post_analysis_reserve(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "finalize_d103_r2_runner_resources.py"
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "data.bin").write_bytes(b"abc")
    status = tmp_path / "matrix_status.json"
    status.write_text(
        json.dumps(
            {
                "status": "ARTIFACTS_COMPLETE",
                "completed_fit_count": 246,
                "completed_meta_steps": 98_400,
                "total_gpu_hours": 1.0,
                "peak_memory_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )
    output = run_root / "analysis" / "resources.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix-status",
            str(status),
            "--run-root",
            str(run_root),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["run_root_bytes"] == 3 + 16 * 1024**2
