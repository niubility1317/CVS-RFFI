#!/usr/bin/env python
"""Execute a generated effective8 formal plan with fail-closed stage gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


TARGET_KEY_RE = re.compile(r"(rx_[^/]+/seed_\d+)")
CELL_RE = re.compile(r"/new_(5|10|20)_k_(1|5|10|20)\.json(?:/|$)")
FORMAL_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
FORMAL_SEEDS = (713101, 713102, 713103, 713104, 713105)
FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FORMAL_NEW_COUNTS = (5, 10, 20)
FORMAL_K = (1, 5, 10, 20)
EXPECTED_COUNTS = {
    "source_cache_sets": 2,
    "target_cache_sets": 25,
    "benchmark_invocations": 300,
    "formal_scenario_rows": 900,
    "collection_invocations": 1,
    "summary_invocations": 1,
}
EXPECTED_COMMAND_LENGTHS = {
    "source_cache_build": 2,
    "train": 1,
    "source_validation": 1,
    "candidate_lock": 1,
    "target_cache_build": 25,
    "benchmark": 300,
    "collect": 1,
    "summarize": 1,
}
EXPECTED_SCRIPTS = {
    "source_cache_build": "code/scripts/build_cvs_leo_weak_iq_cache.py",
    "train": "code/scripts/train_apply_phase1_iq_preadapter_20260703.py",
    "source_validation": "paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py",
    "candidate_lock": "paper_reproduction/scripts/build_cvs_stage2c_candidate_lock.py",
    "target_cache_build": "code/scripts/build_cvs_leo_weak_iq_cache.py",
    "benchmark": "paper_reproduction/scripts/benchmark_cvs_adaptive_rxlight_tta.py",
    "collect": "paper_reproduction/scripts/collect_cvs_stage2c_formal_outputs.py",
    "summarize": "paper_reproduction/scripts/summarize_cvs_stage2c_locked_matrix.py",
}


def _target_key(command: Sequence[str]) -> str:
    normalized = "/".join(str(value).replace("\\", "/") for value in command)
    match = TARGET_KEY_RE.search(normalized)
    if match is None:
        raise ValueError(f"matrix command lacks receiver/seed key: {command}")
    return str(match.group(1))


def _command_option(command: Sequence[Any], option: str) -> str:
    values = [str(value) for value in command]
    if option not in values or values.index(option) + 1 >= len(values):
        raise ValueError(f"formal command lacks {option}: {values}")
    return values[values.index(option) + 1]


def validate_execution_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the formal matrix shape instead of trusting declared counts."""

    if (
        manifest.get("schema")
        != "cvs_stage2c_effective8_generated_execution_plan_v1"
        or manifest.get("phase2_sample_view_policy")
        != "leo_weak_only_no_clean_access"
        or manifest.get("clean_sample_access") is not False
        or manifest.get("clean_derived_signal_access") is not False
        or dict(manifest.get("expected_counts", {})) != EXPECTED_COUNTS
    ):
        raise ValueError("formal execution manifest contract drift")
    isolation_status = manifest.get("phase2_runtime_isolation_status")
    launch_authority = manifest.get("launch_authority")
    if (isolation_status, launch_authority) not in {
        ("LOCAL_PROTOCOL_REPAIR_REQUIRED", False),
        ("PHASE2_RUNTIME_ISOLATION_VERIFIED", True),
    }:
        raise ValueError("formal runtime isolation status/authority drift")
    contract = dict(manifest.get("formal_matrix_contract", {}))
    expected_contract = {
        "target_receivers": list(FORMAL_RECEIVERS),
        "confirmation_seeds": list(FORMAL_SEEDS),
        "leo_weak_scenarios": list(FORMAL_SCENARIOS),
        "new_class_counts": list(FORMAL_NEW_COUNTS),
        "k_values": list(FORMAL_K),
        "query_per_tx": 20,
        "support_pool_max_k": 20,
    }
    if contract != expected_contract:
        raise ValueError("formal matrix contract drift")
    if list(manifest.get("stage_order", [])) != [
        "source_cache_build",
        "train",
        "source_validation",
        "candidate_lock",
        "target_cache_build",
        "benchmark",
        "collect",
        "summarize",
    ]:
        raise ValueError("formal stage order drift")
    if dict(manifest.get("fail_closed_dependencies", {})) != {
        "train_requires_source_train_cache": True,
        "source_validation_requires_training_complete": True,
        "candidate_lock_requires_source_validation_pass": True,
        "target_cache_build_before_phase2": True,
        "benchmark_requires_candidate_lock": True,
        "collect_requires_all_benchmarks": True,
        "summarize_requires_collection": True,
    }:
        raise ValueError("formal dependency graph drift")
    commands = dict(manifest.get("commands", {}))
    if set(commands) != set(EXPECTED_COMMAND_LENGTHS):
        raise ValueError("formal command phase set drift")
    for phase, expected_length in EXPECTED_COMMAND_LENGTHS.items():
        phase_commands = list(commands.get(phase, []))
        if len(phase_commands) != expected_length:
            raise ValueError(f"formal command count drift: {phase}")
        expected_script = EXPECTED_SCRIPTS[phase]
        if any(
            len(command) < 2
            or str(command[0]) != "python"
            or str(command[1]) != expected_script
            for command in phase_commands
        ):
            raise ValueError(f"formal command script drift: {phase}")

    expected_target_keys = {
        f"rx_{receiver.replace('-', '_')}/seed_{seed}"
        for receiver in FORMAL_RECEIVERS
        for seed in FORMAL_SEEDS
    }
    target_keys = [_target_key(command) for command in commands["target_cache_build"]]
    if len(set(target_keys)) != 25 or set(target_keys) != expected_target_keys:
        raise ValueError("formal target-cache receiver/seed coverage drift")
    benchmark_cells: dict[str, set[tuple[int, int]]] = {}
    benchmark_command_by_cell: dict[tuple[str, int, int], Sequence[Any]] = {}
    for command in commands["benchmark"]:
        key = _target_key(command)
        normalized = "/" + "/".join(
            str(value).replace("\\", "/").strip("/") for value in command
        )
        match = CELL_RE.search(normalized)
        if match is None:
            raise ValueError(f"formal benchmark cell identity is missing: {command}")
        cell = (int(match.group(1)), int(match.group(2)))
        if cell in benchmark_cells.setdefault(key, set()):
            raise ValueError(f"duplicate formal benchmark cell: {key}/{cell}")
        benchmark_cells[key].add(cell)
        benchmark_command_by_cell[(key, cell[0], cell[1])] = command
    expected_cells = {
        (new_count, k_shot)
        for new_count in FORMAL_NEW_COUNTS
        for k_shot in FORMAL_K
    }
    if set(benchmark_cells) != expected_target_keys or any(
        cells != expected_cells for cells in benchmark_cells.values()
    ):
        raise ValueError("formal benchmark matrix identity coverage drift")

    cache_contracts = list(manifest.get("target_cache_contracts", []))
    if len(cache_contracts) != 25:
        raise ValueError("formal target-cache contract count drift")
    seen_contracts: set[tuple[str, int]] = set()
    for raw in cache_contracts:
        item = dict(raw)
        receiver = str(item.get("receiver", ""))
        seed = int(item.get("seed", -1))
        identity = (receiver, seed)
        if identity in seen_contracts:
            raise ValueError(f"duplicate target-cache contract: {identity}")
        seen_contracts.add(identity)
        expected_satellite_seeds = {
            scenario: seed * 10 + index
            for index, scenario in enumerate(FORMAL_SCENARIOS)
        }
        key = f"rx_{receiver.replace('-', '_')}/seed_{seed}"
        if (
            key not in expected_target_keys
            or not str(item.get("cache_set_manifest", "")).replace("\\", "/").endswith(
                f"/{key.replace('/seed_', '/seed_')}/cache_set.json"
            )
            or dict(item.get("satellite_seed_by_scenario", {}))
            != expected_satellite_seeds
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(item.get("cache_build_spec_sha256", ""))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("cache_build_spec_content_sha256", "")),
            )
        ):
            raise ValueError(f"target-cache contract drift: {identity}")
    if seen_contracts != {
        (receiver, seed) for receiver in FORMAL_RECEIVERS for seed in FORMAL_SEEDS
    }:
        raise ValueError("formal target-cache contract identity coverage drift")
    cache_contract_by_key = {
        f"rx_{str(value['receiver']).replace('-', '_')}/seed_{int(value['seed'])}": dict(value)
        for value in cache_contracts
    }
    for command in commands["target_cache_build"]:
        key = _target_key(command)
        if _command_option(command, "--spec") != str(
            cache_contract_by_key[key]["cache_build_spec"]
        ):
            raise ValueError(f"target-cache command/spec binding drift: {key}")

    config_contracts = list(manifest.get("stage2_config_contracts", []))
    if len(config_contracts) != 300:
        raise ValueError("formal Stage2 config-contract count drift")
    seen_config_cells: set[tuple[str, int, int]] = set()
    for raw in config_contracts:
        item = dict(raw)
        receiver = str(item.get("receiver", ""))
        seed = int(item.get("seed", -1))
        new_count = int(item.get("new_class_count", -1))
        k_shot = int(item.get("k_shot", -1))
        key = f"rx_{receiver.replace('-', '_')}/seed_{seed}"
        identity = (key, new_count, k_shot)
        if identity in seen_config_cells or identity not in benchmark_command_by_cell:
            raise ValueError(f"Stage2 config-contract identity drift: {identity}")
        seen_config_cells.add(identity)
        if (
            str(item.get("cache_set_manifest", ""))
            != str(cache_contract_by_key[key]["cache_set_manifest"])
            or _command_option(benchmark_command_by_cell[identity], "--config")
            != str(item.get("config", ""))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(item.get("config_file_sha256", ""))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(item.get("config_content_sha256", ""))
            )
        ):
            raise ValueError(f"Stage2 config-contract binding drift: {identity}")
    if seen_config_cells != set(benchmark_command_by_cell):
        raise ValueError("Stage2 config-contract coverage drift")
    return dict(manifest)


def build_stage_steps(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict[str, Any]]:
    manifest = validate_execution_manifest(manifest)
    commands = dict(manifest.get("commands", {}))
    if stage == "source_pipeline":
        ordered = []
        for phase in (
            "source_cache_build",
            "train",
            "source_validation",
            "candidate_lock",
        ):
            for command in commands.get(phase, []):
                ordered.append({"phase": phase, "command": list(command)})
        return ordered
    if stage == "finalize":
        ordered = []
        for phase in ("collect", "summarize"):
            phase_commands = list(commands.get(phase, []))
            if len(phase_commands) != 1:
                raise ValueError(f"finalize phase={phase} must have exactly one command")
            ordered.append({"phase": phase, "command": list(phase_commands[0])})
        return ordered
    if stage != "matrix_shard":
        raise ValueError(f"unknown formal execution stage={stage!r}")
    if int(shard_count) < 1 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("invalid shard_index/shard_count")
    target_commands = list(commands.get("target_cache_build", []))
    benchmark_commands = list(commands.get("benchmark", []))
    benchmarks_by_key: dict[str, list[list[str]]] = {}
    for command in benchmark_commands:
        benchmarks_by_key.setdefault(_target_key(command), []).append(list(command))
    steps: list[dict[str, Any]] = []
    for index, cache_command in enumerate(target_commands):
        if index % int(shard_count) != int(shard_index):
            continue
        key = _target_key(cache_command)
        matched = benchmarks_by_key.get(key, [])
        if len(matched) != 12:
            raise ValueError(f"target key={key} must have exactly 12 benchmark commands")
        steps.append(
            {"phase": "target_cache_build", "target_key": key, "command": list(cache_command)}
        )
        steps.extend(
            {
                "phase": "benchmark",
                "target_key": key,
                "command": command,
            }
            for command in matched
        )
    return steps


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    log_dir: Path,
    state_path: Path,
) -> dict[str, Any]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    else:
        state = {"schema": "cvs_effective8_formal_runner_state_v1", "steps": {}}
    records = dict(state.get("steps", {}))
    environment = dict(os.environ)
    pythonpath = [str(project_root / "code"), str(project_root)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(str(environment["PYTHONPATH"]))
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, raw_step in enumerate(steps):
        command = [str(value) for value in raw_step["command"]]
        if command and command[0] == "python":
            command[0] = sys.executable
        step_id = f"{index:04d}_{raw_step['phase']}"
        previous = dict(records.get(step_id, {}))
        if previous.get("status") == "complete" and previous.get("command") == command:
            continue
        log_path = log_dir / f"{step_id}.log"
        started = time.time()
        record = {
            **dict(raw_step),
            "command": command,
            "status": "running",
            "started_unix": started,
            "log_path": str(log_path),
        }
        records[step_id] = record
        state["steps"] = records
        state["status"] = "running"
        _write_state(state_path, state)
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record.update(
            {
                "returncode": int(completed.returncode),
                "finished_unix": time.time(),
                "elapsed_seconds": float(time.time() - started),
                "status": "complete" if completed.returncode == 0 else "failed",
            }
        )
        records[step_id] = record
        state["steps"] = records
        state["status"] = record["status"]
        _write_state(state_path, state)
        if completed.returncode != 0:
            raise RuntimeError(
                f"formal plan failed closed at {step_id}; inspect {log_path}"
            )
    state["status"] = "complete"
    state["finished_unix"] = time.time()
    _write_state(state_path, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_manifest", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("source_pipeline", "matrix_shard", "finalize"), required=True
    )
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--state_json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.plan_manifest.read_text(encoding="utf-8-sig"))
    manifest = validate_execution_manifest(manifest)
    if manifest.get("launch_authority") is not True:
        blockers = ",".join(str(value) for value in manifest.get("protocol_blockers", []))
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: formal runner is fail-closed until "
            f"sealed runtime isolation and predict/score separation land; blockers={blockers}"
        )
    steps = build_stage_steps(
        manifest,
        stage=str(args.stage),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    state = execute_steps(
        steps,
        project_root=args.project_root.resolve(),
        log_dir=args.log_dir,
        state_path=args.state_json,
    )
    print(
        json.dumps(
            {"status": state["status"], "step_count": len(steps)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
