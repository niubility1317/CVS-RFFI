from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts import build_qknnv42_fft96_125_plan as module


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
    assert plan["state_cells"][-1]["output_root"].startswith(
        "/home/user/project/runs/test/"
    )
    assert "\\" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("value", [r"\home\user\run", "relative/run"])
def test_plan_rejects_non_posix_or_relative_remote_paths(value: str) -> None:
    with pytest.raises(ValueError):
        module._remote_path(value, name="test path")
