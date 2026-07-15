from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest

from paper_reproduction.scripts.build_cvs_stage2c_effective8_formal_plan import (
    generate_plan,
)
from paper_reproduction.scripts.run_cvs_stage2c_effective8_formal_plan import (
    build_stage_steps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    REPO_ROOT
    / "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"
)


def test_source_pipeline_has_strict_gate_order(tmp_path: Path) -> None:
    manifest = generate_plan(
        PLAN,
        out_dir=tmp_path / "generated",
        runtime_project_root="/srv/CV-SincNet",
    )
    steps = build_stage_steps(manifest, stage="source_pipeline")
    assert [step["phase"] for step in steps] == [
        "source_cache_build",
        "source_cache_build",
        "train",
        "source_validation",
        "candidate_lock",
    ]


def test_eight_matrix_shards_cover_each_cache_and_benchmark_once(
    tmp_path: Path,
) -> None:
    manifest = generate_plan(
        PLAN,
        out_dir=tmp_path / "generated",
        runtime_project_root="/srv/CV-SincNet",
    )
    all_steps = [
        step
        for shard_index in range(8)
        for step in build_stage_steps(
            manifest,
            stage="matrix_shard",
            shard_index=shard_index,
            shard_count=8,
        )
    ]
    assert sum(step["phase"] == "target_cache_build" for step in all_steps) == 25
    assert sum(step["phase"] == "benchmark" for step in all_steps) == 300
    commands = [tuple(step["command"]) for step in all_steps]
    assert len(commands) == len(set(commands))


def test_finalize_collects_then_summarizes(tmp_path: Path) -> None:
    manifest = generate_plan(
        PLAN,
        out_dir=tmp_path / "generated",
        runtime_project_root="/srv/CV-SincNet",
    )
    steps = build_stage_steps(manifest, stage="finalize")
    assert [step["phase"] for step in steps] == ["collect", "summarize"]


def test_runner_rejects_declared_counts_without_complete_commands(tmp_path: Path) -> None:
    manifest = generate_plan(
        PLAN,
        out_dir=tmp_path / "generated",
        runtime_project_root="/srv/CV-SincNet",
    )
    tampered = deepcopy(manifest)
    tampered["commands"]["benchmark"] = tampered["commands"]["benchmark"][:-1]
    with pytest.raises(ValueError, match="command count drift"):
        build_stage_steps(tampered, stage="matrix_shard")
