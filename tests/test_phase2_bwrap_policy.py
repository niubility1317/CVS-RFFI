from __future__ import annotations

from pathlib import Path

import pytest

from cvsrffi.phase2_bwrap_policy import BwrapPolicyError, build_phase2_bwrap_command


def _tree(tmp_path: Path):
    paths = {}
    for name in ("runtime", "package", "output", "system", "scorer"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    for name in ("seal", "request"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        paths[name] = path
    python = paths["system"] / "python"
    python.write_bytes(b"python")
    paths["python"] = python
    return paths


def test_bwrap_policy_has_one_project_write_root_and_no_scorer_mount(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    command = build_phase2_bwrap_command(
        bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
        detached_seal=paths["seal"], request_json=paths["request"],
        output_root=paths["output"], python_executable=paths["python"],
        predictor_argv=["/runtime/code/scripts/run.py"],
        system_read_roots=[paths["system"]], trusted_system_read_roots=[paths["system"]],
        forbidden_roots=[paths["scorer"]],
    )
    assert command.count("--bind") == 1
    assert command[command.index("--bind") + 2] == "/output"
    assert str(paths["scorer"]) not in command
    assert "--unshare-all" in command
    assert "--share-net" not in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--clearenv" in command
    assert command.count("--ro-bind") == 5
    runtime_bind = command.index(str(paths["runtime"].resolve()))
    assert command[runtime_bind - 1] == "--ro-bind"
    assert command[runtime_bind + 1] == "/runtime/code"
    assert command[command.index("--chdir") + 1] == "/runtime/code"


def test_bwrap_policy_rejects_truth_root_inside_package(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    truth = paths["package"] / "truth"
    truth.mkdir()
    with pytest.raises(BwrapPolicyError, match="overlaps"):
        build_phase2_bwrap_command(
            bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
            detached_seal=paths["seal"], request_json=paths["request"],
            output_root=paths["output"], python_executable=paths["python"],
            predictor_argv=["run.py"], system_read_roots=[paths["system"]],
            trusted_system_read_roots=[paths["system"]],
            forbidden_roots=[truth],
        )


def test_bwrap_policy_rejects_python_outside_readonly_runtime(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(BwrapPolicyError, match="python executable"):
        build_phase2_bwrap_command(
            bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
            detached_seal=paths["seal"], request_json=paths["request"],
            output_root=paths["output"], python_executable=paths["python"],
            predictor_argv=["run.py"], system_read_roots=[other],
            trusted_system_read_roots=[other],
            forbidden_roots=[paths["scorer"]],
        )


def test_bwrap_policy_contains_no_inner_tracer_or_inherited_trace_fd(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    command = build_phase2_bwrap_command(
        bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
        detached_seal=paths["seal"], request_json=paths["request"],
        output_root=paths["output"], python_executable=paths["python"],
        predictor_argv=["/runtime/code/scripts/run.py"],
        system_read_roots=[paths["system"]], trusted_system_read_roots=[paths["system"]],
        forbidden_roots=[paths["scorer"]],
    )
    assert command[0] == "bwrap"
    assert str(paths["python"].resolve()) in command
    assert "-o" not in command
    assert not any("strace" in value or "/proc/self/fd/" in value for value in command)


def test_bwrap_policy_rejects_system_root_that_exposes_project_parent(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    with pytest.raises(BwrapPolicyError, match="overlap"):
        build_phase2_bwrap_command(
            bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
            detached_seal=paths["seal"], request_json=paths["request"],
            output_root=paths["output"], python_executable=paths["python"],
            predictor_argv=["run.py"], system_read_roots=[tmp_path],
            trusted_system_read_roots=[tmp_path],
            forbidden_roots=[paths["scorer"]],
        )


def test_bwrap_policy_rejects_caller_nominated_untrusted_system_root(tmp_path: Path) -> None:
    paths = _tree(tmp_path)
    other = tmp_path / "untrusted_data"
    other.mkdir()
    with pytest.raises(BwrapPolicyError, match="fixed trusted allowlist"):
        build_phase2_bwrap_command(
            bwrap="bwrap", runtime_root=paths["runtime"], package_root=paths["package"],
            detached_seal=paths["seal"], request_json=paths["request"],
            output_root=paths["output"], python_executable=paths["python"],
            predictor_argv=["run.py"], system_read_roots=[other],
            trusted_system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
        )
