from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_jmrs01_20260826.sh"


def _bash() -> str:
    if os.name == "nt":
        path = Path(r"C:\Program Files\Git\bin\bash.exe")
        if path.is_file():
            return str(path)
    found = shutil.which("bash")
    if not found:
        pytest.skip("bash is unavailable")
    return found


def _shell_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":", 1)[1]
        return f"/{drive}{tail}"
    return str(resolved)


def _base_env(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "root"
    (root / "code").mkdir(parents=True)
    checkpoint = root / "checkpoint.pth"
    wisig = root / "ManySig.pkl"
    runner = root / "code" / "audit_phase1_jmrs01.py"
    scorer = root / "code" / "score_phase1_jmrs01.py"
    checkpoint.write_bytes(b"checkpoint")
    wisig.write_bytes(b"wisig")
    runner.write_text("# runner placeholder\n", encoding="utf-8")
    scorer.write_text("# scorer placeholder\n", encoding="utf-8")
    return {
        **os.environ,
        "ROOT": str(root).replace("\\", "/"),
        "CHECKPOINT": str(checkpoint).replace("\\", "/"),
        "WISIG_PKL": str(wisig).replace("\\", "/"),
        "RUNNER": str(runner).replace("\\", "/"),
        "SCORER": str(scorer).replace("\\", "/"),
        "RUN_ROOT": str(tmp_path / "runs").replace("\\", "/"),
        "LOG_ROOT": str(tmp_path / "logs").replace("\\", "/"),
        "RUN_ID": "JMRS01_TEST_RUN",
    }


def test_launcher_refuses_existing_formal_output_before_python_runs(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    output = Path(env["RUN_ROOT"]) / env["RUN_ID"]
    output.mkdir(parents=True)
    marker = tmp_path / "python_was_called"
    fake_python = tmp_path / "fake-python.sh"
    fake_python.write_text(f"#!/usr/bin/env bash\ntouch '{marker.as_posix()}'\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env["PYTHON"] = fake_python.as_posix()

    completed = subprocess.run([_bash(), _shell_path(LAUNCHER)], env=env, text=True, capture_output=True)

    assert completed.returncode == 3
    assert "output_exists" in completed.stderr
    assert not marker.exists()


def test_launcher_executes_smoke_then_formal_then_independent_scorer(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    call_log = tmp_path / "calls.txt"
    fake_program = tmp_path / "fake_program.py"
    artifact_lines = " ".join(
        (
            "mechanism_identity_stability.json",
            "mechanism_receiver_probe.json",
            "mechanism_loro_metrics.json",
            "mechanism_clean_sat_consistency.json",
            "mechanism_complementarity.json",
            "mechanism_observability.json",
            "mechanism_cost.json",
            "mechanism_decision.json",
        )
    )
    fake_program.write_text(
        "import pathlib, sys\n"
        f"log = pathlib.Path({str(call_log)!r})\n"
        "with log.open('a', encoding='utf-8') as handle: handle.write(' '.join(sys.argv) + '\\n')\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output_dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "if '--predictions' not in args:\n"
        "    if '--smoke_only' in args: (out / 'protocol_and_smoke.json').write_text('{}')\n"
        "    else:\n"
        "        (out / 'run_manifest.json').write_text('{}')\n"
        "        (out / 'predictions.jsonl').write_text('{}')\n"
        "        (out / 'truth.jsonl').write_text('{}')\n"
        "else:\n"
        f"    names = {artifact_lines.split()!r}\n"
        "    for name in names: (out / name).write_text('{}')\n",
        encoding="utf-8",
    )
    env["PYTHON"] = sys.executable.replace("\\", "/")
    env["RUNNER"] = fake_program.as_posix()
    env["SCORER"] = fake_program.as_posix()

    completed = subprocess.run([_bash(), _shell_path(LAUNCHER)], env=env, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "--smoke_only" in calls[0]
    assert "--smoke_only" not in calls[1] and "--predictions" not in calls[1]
    assert "--predictions" in calls[2] and "--truth" in calls[2]
    assert "--rows M0,R1,R2,D1,P1,P2,S1" in calls[1]
    assert "D2" not in calls[1]
