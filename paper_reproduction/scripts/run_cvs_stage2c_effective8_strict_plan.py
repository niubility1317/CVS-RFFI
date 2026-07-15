#!/usr/bin/env python3
"""Run the strict effective8 smoke or an authorized eight-way matrix shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import validate_strict_plan
from paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_package import run_package


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value))
    temporary.replace(path)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical(value))
    if os.name == "posix":
        path.chmod(0o444)


def _device_command(command: Sequence[str], device: str) -> list[str]:
    resolved = [str(value) for value in command]
    if resolved and resolved[0] == "python":
        resolved[0] = sys.executable
    if "--device" in resolved:
        resolved[resolved.index("--device") + 1] = device
    return resolved


def _run_logged(command: Sequence[str], *, project_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            [str(value) for value in command], cwd=project_root,
            stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"strict cache build failed rc={completed.returncode}; inspect {log_path}")


def _cache_by_id(plan: Mapping[str, Any], cache_id: str) -> dict[str, Any]:
    matched = [item for item in plan["cache_steps"] if item["cache_id"] == cache_id]
    if len(matched) != 1:
        raise ValueError(f"strict cache id is absent or duplicated: {cache_id}")
    return dict(matched[0])


def _ensure_cache(
    cache: Mapping[str, Any], *, project_root: Path, device: str, log_path: Path
) -> None:
    manifest = Path(cache["cache_set_manifest"])
    if manifest.is_file() and not manifest.is_symlink():
        return
    if manifest.exists() or manifest.parent.exists():
        raise RuntimeError("partial target cache exists; refusing destructive resume")
    _run_logged(_device_command(cache["build_command"], device), project_root=project_root, log_path=log_path)
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError("target cache command completed without a regular cache manifest")


def run_smoke(
    plan: Mapping[str, Any], *, project_root: Path, device: str, log_dir: Path,
    smoke_receipt_path: Path,
) -> dict[str, Any]:
    if plan.get("smoke_authority") is not True:
        raise RuntimeError("strict smoke authority is false")
    package_id = str(plan["smoke_package_id"])
    package = next(item for item in plan["package_steps"] if item["package_id"] == package_id)
    cache = _cache_by_id(plan, str(package["cache_id"]))
    _ensure_cache(cache, project_root=project_root, device=device, log_path=log_dir / "smoke_cache.log")
    receipts = run_package(
        plan, package_id=package_id, project_root=project_root,
        device=device, execution_mode="smoke",
        k_values=[int(plan["smoke_k_shot"])],
    )
    if len(receipts) != 1 or receipts[0].get("status") != "PROTOCOL_VALID":
        raise RuntimeError("strict N607 smoke did not produce one PROTOCOL_VALID receipt")
    receipt = {
        "schema": "cvs.stage2c.effective8.n607_landlock_smoke.v1",
        "status": "PASS",
        "candidate_capsule_sha256": plan["candidate_capsule_sha256"],
        "package_id": package_id,
        "k_shot": int(plan["smoke_k_shot"]),
        "device": device,
        "cell_receipt": receipts[0],
        "cell_receipt_sha256": hashlib.sha256(_canonical(receipts[0])).hexdigest(),
        "matrix_launch_authority_recommended": True,
    }
    _write_new(smoke_receipt_path, receipt)
    return receipt


def run_matrix_shard(
    plan: Mapping[str, Any], *, project_root: Path, device: str,
    shard_index: int, shard_count: int, log_dir: Path, state_path: Path,
) -> dict[str, Any]:
    if plan.get("launch_authority") is not True or plan.get("authority_state") != "N607_LANDLOCK_SMOKE_PASS":
        raise RuntimeError("formal 300-cell matrix remains fail-closed without authorized N607 smoke")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid matrix shard index/count")
    state = _read(state_path) if state_path.exists() else {
        "schema": "cvs.stage2c.effective8.strict_shard_state.v1",
        "shard_index": shard_index, "shard_count": shard_count, "steps": {},
    }
    steps = dict(state.get("steps", {}))
    selected_caches = [
        cache for index, cache in enumerate(plan["cache_steps"])
        if index % shard_count == shard_index
    ]
    packages_by_cache: dict[str, list[dict[str, Any]]] = {}
    for package in plan["package_steps"]:
        packages_by_cache.setdefault(str(package["cache_id"]), []).append(dict(package))
    for cache in selected_caches:
        cache_id = str(cache["cache_id"])
        _ensure_cache(
            cache, project_root=project_root, device=device,
            log_path=log_dir / f"{cache_id}__cache.log",
        )
        for package in packages_by_cache[cache_id]:
            package_id = str(package["package_id"])
            prior = dict(steps.get(package_id, {}))
            if prior.get("status") == "complete":
                continue
            steps[package_id] = {"status": "running", "started_unix": time.time()}
            state.update({"status": "running", "steps": steps})
            _write_atomic(state_path, state)
            receipts = run_package(
                plan, package_id=package_id, project_root=project_root, device=device,
                execution_mode="formal",
            )
            if len(receipts) != 4 or any(item.get("status") != "PROTOCOL_VALID" for item in receipts):
                raise RuntimeError(f"strict package did not complete four K cells: {package_id}")
            steps[package_id] = {
                "status": "complete", "finished_unix": time.time(),
                "cell_ids": [item["cell_id"] for item in receipts],
            }
            state["steps"] = steps
            _write_atomic(state_path, state)
    state.update({"status": "complete", "finished_unix": time.time(), "steps": steps})
    _write_atomic(state_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "matrix_shard"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    args = parser.parse_args(argv)
    plan = validate_strict_plan(_read(args.plan_manifest))
    project_root = args.project_root.resolve(strict=True)
    if args.stage == "smoke":
        if args.smoke_receipt is None:
            raise ValueError("--smoke-receipt is required for smoke")
        result = run_smoke(
            plan, project_root=project_root, device=args.device,
            log_dir=args.log_dir, smoke_receipt_path=args.smoke_receipt,
        )
    else:
        if args.state_json is None:
            raise ValueError("--state-json is required for matrix_shard")
        result = run_matrix_shard(
            plan, project_root=project_root, device=args.device,
            shard_index=args.shard_index, shard_count=args.shard_count,
            log_dir=args.log_dir, state_path=args.state_json,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
