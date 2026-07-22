from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts import build_qknnv42_fft96_125_plan as module
from paper_reproduction.scripts import run_qknnv42_fft96_125_plan as runner


def _source_plan(path: Path) -> None:
    packages = []
    old_labels = [f"old_{index}" for index in range(10)]
    nested = {
        5: [f"new_{index}" for index in range(5)],
        10: [f"new_{index}" for index in range(10)],
        20: [f"new_{index}" for index in range(20)],
    }
    for receiver in module.RECEIVERS:
        for seed in module.SEEDS:
            for new_count in module.NEW_COUNTS:
                packages.append(
                    {
                        "receiver": receiver,
                        "seed": seed,
                        "new_class_count": new_count,
                        "old_class_labels": old_labels,
                        "new_class_labels": nested[new_count],
                        "target_cache_set": (
                            f"/sealed/target/rx_{receiver}/seed_{seed}/cache_set.json"
                        ),
                    }
                )
    path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.adv3b02_ci_strict_plan.v1",
                "packages": packages,
            }
        ),
        encoding="utf-8",
    )


def test_plan_keeps_local_hash_inputs_separate_from_remote_posix_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_plan = tmp_path / "source.json"
    _source_plan(source_plan)
    artifacts = {}
    for name in ("base", "lock", "adapter", "head", "tta"):
        artifacts[name] = tmp_path / name
        artifacts[name].write_bytes(name.encode("ascii"))
    expected_base_sha = (
        "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9"
    )
    real_sha256 = module._sha256

    def fake_sha256(path: Path) -> str:
        if path == artifacts["base"].resolve():
            return expected_base_sha
        return real_sha256(path)

    monkeypatch.setattr(module, "_sha256", fake_sha256)
    output = tmp_path / "plan.json"
    args = argparse.Namespace(
        experiment_id="test",
        run_root="/home/user/project/runs/test",
        source_strict_plan=source_plan,
        remote_source_strict_plan="/home/user/project/source_plan.json",
        base_runtime=artifacts["base"],
        remote_base_runtime="/home/user/project/base.ts",
        candidate_lock=artifacts["lock"],
        remote_candidate_lock="/home/user/project/lock.json",
        adapter=artifacts["adapter"],
        remote_adapter="/home/user/project/adapter.json",
        head_artifact=artifacts["head"],
        remote_head_artifact="/home/user/project/head.json",
        tta_policy=artifacts["tta"],
        remote_tta_policy="/home/user/project/tta.json",
        output=output,
    )
    plan = module.build(args)
    assert plan["counts"] == {
        "packages": 100,
        "bundles": 125,
        "state_cells": 500,
        "scenario_rows": 1500,
    }
    assert plan["artifacts"]["base_runtime"]["path"] == "/home/user/project/base.ts"
    assert plan["packages"][0]["predictor_package_root"].startswith(
        "/home/user/project/runs/test/"
    )
    assert plan["packages"][0]["reference_new_class_labels"] == [
        f"new_{index}" for index in range(20)
    ]
    assert plan["state_cells"][-1]["output_root"].startswith(
        "/home/user/project/runs/test/"
    )
    assert "\\" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("value", [r"\home\user\run", "relative/run"])
def test_plan_rejects_non_posix_or_relative_remote_paths(value: str) -> None:
    with pytest.raises(ValueError):
        module._remote_path(value, name="test path")


def test_stage2b_package_uses_unregistered_new20_reference_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run_json(command, *, cwd):
        captured.extend(command)
        assert cwd == tmp_path
        return {
            "predictor_package_root_sha256": "1" * 64,
            "predictor_package_seal_sha256": "2" * 64,
            "scoring_manifest_sha256": "3" * 64,
        }

    monkeypatch.setattr(runner, "_run_json", fake_run_json)
    package_parent = tmp_path / "package"
    package = {
        "package_id": "before",
        "stage": "stage2b",
        "registration_state": "before_registration",
        "target_cache_set": "/sealed/cache_set.json",
        "predictor_package_root": str(package_parent / "predictor"),
        "scorer_root": str(package_parent / "scorer"),
        "detached_seal": str(package_parent / "predictor.seal.json"),
        "build_receipt": str(package_parent / "package_build_receipt.json"),
        "receiver": "20-1",
        "seed": 713101,
        "old_class_labels": ["old-a", "old-b"],
        "new_class_count": 0,
        "new_class_labels": [],
        "reference_new_class_labels": [f"new-{index}" for index in range(20)],
    }
    plan = {
        "nested_new_class_labels": {
            "20": [f"new-{index}" for index in range(20)]
        },
        "artifacts": {
            "candidate_lock": {"path": "/sealed/lock.json"},
            "base_runtime": {"path": "/sealed/base.ts"},
            "adapter": {"path": "/sealed/adapter.json"},
            "head_artifact": {"path": "/sealed/head.json"},
            "tta_policy": {"path": "/sealed/tta.json"},
        },
    }
    receipt = runner._build_package(plan, package, project_root=tmp_path)
    option_index = captured.index("--stage2b-reference-new-class-labels")
    assert captured[option_index + 1] == ",".join(
        package["reference_new_class_labels"]
    )
    assert "--stage2b-mixed-cache-old-query-only" not in captured
    assert receipt["status"] == "PASS"
    assert receipt["reference_new_class_count"] == 20


def test_qknn_plan_uses_landlock_pre_run_evidence_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run_json(command, *, cwd):
        commands.append(list(command))
        assert cwd == tmp_path
        return {"status": "PASS"}

    monkeypatch.setattr(runner, "_run_json", fake_run_json)
    package = {
        "pre_run_evidence_root": str(tmp_path / "evidence"),
        "predictor_package_root": str(tmp_path / "package"),
        "detached_seal": str(tmp_path / "package.seal.json"),
        "scorer_root": str(tmp_path / "scorer"),
    }
    receipt = {"predictor_package_seal_sha256": "a" * 64}
    result = runner._ensure_pre_run_evidence(
        package,
        receipt,
        project_root=tmp_path,
        runtime_closure_root=tmp_path / "closure",
        landlock_launcher=tmp_path / "code/scripts/landlock_entry.py",
        landlock_policy_module=tmp_path / "code/cvsrffi/landlock_policy.py",
        strace=tmp_path / "system/strace",
        python_executable=tmp_path / "system/python",
        system_read_roots=[tmp_path / "system"],
    )
    command = commands[0]
    assert command[1].replace("\\", "/").endswith(
        "code/scripts/build_cvs_stage2_landlock_pre_run_evidence.py"
    )
    assert "--landlock-launcher" in command
    assert "--landlock-policy-module" in command
    assert not any("bwrap" in value for value in command)
    assert result == tmp_path / "evidence/runtime_isolation_evidence.json"
