import asyncio
from collections import Counter
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys

import pytest


def _load_matrix_runner():
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d103_r2_fit_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "d103_fit_matrix_runner_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_matrix_cli_help_and_two_worker_default() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d103_r2_fit_matrix.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--workers-per-gpu" in result.stdout
    assert "--output-root" in result.stdout


def test_two_rows_with_same_fingerprint_trigger_systemic_stop() -> None:
    module = _load_matrix_runner()
    counts: Counter[str] = Counter()
    assert module._record_failure_fingerprint(counts, "same") is None
    assert module._record_failure_fingerprint(counts, "other") is None
    assert module._record_failure_fingerprint(counts, "same") == "same"
    assert counts == Counter({"same": 2, "other": 1})


def test_stop_signals_only_prebound_run_owned_pids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_matrix_runner()
    expected_cwd = tmp_path / "repo"
    fit_root = tmp_path / "run" / "fits" / "fit-001"
    expected_cwd.mkdir()
    fit_root.mkdir(parents=True)
    state = {
        100: {
            "pid": 100,
            "alive": True,
            "cwd": str(expected_cwd.resolve()),
            "cmdline": ["python", "--output-dir", str(fit_root.resolve())],
        },
        101: {
            "pid": 101,
            "alive": True,
            "cwd": str(expected_cwd.resolve()),
            "cmdline": ["worker", str(fit_root.resolve())],
        },
        102: {
            "pid": 102,
            "alive": True,
            "cwd": str(tmp_path / "unrelated"),
            "cmdline": ["unrelated-worker"],
        },
    }
    signals = []

    class FakeProcess:
        pid = 100
        returncode = None

        async def wait(self):
            return 0

    def observe(pid: int):
        return dict(state[pid])

    def kill(pid: int, sig: int):
        signals.append((pid, sig))
        state[pid]["alive"] = False

    async def no_sleep(_seconds: float):
        return None

    monkeypatch.setattr(module, "_descendant_pids", lambda pid: [101, 102])
    monkeypatch.setattr(module, "_process_observation", observe)
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: {
            "available": True,
            "rows": [{"pid": 102, "used_memory_mib": "256"}],
        },
    )
    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    record = asyncio.run(
        module._stop_bound_run_process_tree(
            FakeProcess(),
            fit_root=fit_root,
            expected_cwd=expected_cwd,
            gpu_lane="3",
        )
    )
    assert signals == [(101, signal.SIGTERM), (100, signal.SIGTERM)]
    assert all(pid != 102 for pid, _ in signals)
    assert record["gpu_lane"] == "3"
    assert record["bound_run_owned_pids"] == [100, 101]
    assert record["unbound_live_pids"] == [102]
    assert record["run_owned_binding_pass"] is False
    assert record["kill_escalation_used"] is False
    assert record["all_run_owned_pids_stopped"] is True
    assert record["stopped_pids_still_on_gpu"] == []
    assert record["post_stop_process_tree"][-1]["alive"] is True
