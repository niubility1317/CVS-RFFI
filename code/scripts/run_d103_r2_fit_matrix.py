#!/usr/bin/env python3
"""Execute the immutable 246-fit D103-R2 matrix with bounded GPU lanes."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_falsifier import build_complete_fit_plan  # noqa: E402
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    sha256_file,
    validate_teacher_fit_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _process_observation(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        return {"pid": pid, "alive": False, "cwd": None, "cmdline": []}
    try:
        cwd = str((proc / "cwd").resolve(strict=True))
    except OSError:
        cwd = None
    try:
        cmdline = [
            token.decode("utf-8", errors="replace")
            for token in (proc / "cmdline").read_bytes().split(b"\0")
            if token
        ]
    except OSError:
        cmdline = []
    return {"pid": pid, "alive": True, "cwd": cwd, "cmdline": cmdline}


def _descendant_pids(root_pid: int) -> list[int]:
    parent_by_pid: dict[int, int] = {}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parent_by_pid[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants: list[int] = []
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = sorted(
            pid for pid, ppid in parent_by_pid.items() if ppid == parent
        )
        descendants.extend(children)
        frontier.extend(children)
    return descendants


def _gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}:{error}"}
    rows = []
    for line in result.stdout.splitlines():
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) == 2 and tokens[0].isdigit():
            rows.append({"pid": int(tokens[0]), "used_memory_mib": tokens[1]})
    return {"available": result.returncode == 0, "rows": rows}


def _record_failure_fingerprint(
    fingerprints: Counter[str], fingerprint: str | None
) -> str | None:
    if not fingerprint:
        return None
    normalized = str(fingerprint)
    fingerprints[normalized] += 1
    return normalized if fingerprints[normalized] >= 2 else None


async def _stop_bound_run_process_tree(
    process: asyncio.subprocess.Process,
    *,
    fit_root: Path,
    expected_cwd: Path,
    gpu_lane: str,
) -> dict[str, Any]:
    """Stop only the exact observed process tree bound to one immutable fit."""

    escalated_pids: list[int] = []
    descendant_pids = _descendant_pids(process.pid)
    observations = [
        _process_observation(pid)
        for pid in [process.pid, *descendant_pids]
    ]
    bound_pids = {
        int(observation["pid"])
        for observation in observations
        if (
            observation["alive"]
            and observation["cwd"] == str(expected_cwd.resolve())
            and str(fit_root.resolve())
            in "\0".join(observation["cmdline"])
        )
    }
    alive_pids = {
        int(observation["pid"])
        for observation in observations
        if observation["alive"]
    }
    binding_pass = alive_pids == bound_pids
    signaled_pids: list[int] = []
    ordered_pids = [*reversed(descendant_pids), process.pid]
    for owned_pid in ordered_pids:
        if owned_pid in bound_pids:
            try:
                os.kill(owned_pid, signal.SIGTERM)
                signaled_pids.append(owned_pid)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        _process_observation(pid)["alive"] for pid in bound_pids
    ):
        await asyncio.sleep(0.2)
    for owned_pid in ordered_pids:
        if (
            owned_pid in bound_pids
            and _process_observation(owned_pid)["alive"]
        ):
            try:
                os.kill(owned_pid, signal.SIGKILL)
                escalated_pids.append(owned_pid)
            except ProcessLookupError:
                pass
    if process.pid in bound_pids:
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
    await asyncio.sleep(1.0)
    post = [
        _process_observation(pid)
        for pid in [process.pid, *descendant_pids]
    ]
    gpu_snapshot = _gpu_snapshot()
    gpu_pids = {
        int(row["pid"]) for row in gpu_snapshot.get("rows", [])
    }
    return {
        "gpu_lane": str(gpu_lane),
        "gpu_snapshot": gpu_snapshot,
        "pre_stop_process_tree": observations,
        "run_owned_binding_pass": binding_pass,
        "bound_run_owned_pids": sorted(bound_pids),
        "unbound_live_pids": sorted(alive_pids - bound_pids),
        "graceful_termination_sent": bool(signaled_pids),
        "graceful_termination_pids": signaled_pids,
        "kill_escalation_used": bool(escalated_pids),
        "kill_escalation_pids": escalated_pids,
        "post_stop_process_tree": post,
        "all_run_owned_pids_stopped": not any(
            row["alive"] for row in post if row["pid"] in bound_pids
        ),
        "stopped_pids_still_on_gpu": sorted(bound_pids & gpu_pids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-archive", type=Path, required=True)
    parser.add_argument("--labeled-manifest", type=Path, required=True)
    parser.add_argument("--unlabeled-archive", type=Path, required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--source-val-seal", type=Path, required=True)
    parser.add_argument("--source-val-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable fit matrix root exists: {output}")
    if args.workers_per_gpu not in (1, 2):
        raise ValueError("workers-per-gpu must be 1 or 2")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU lane list must be unique and non-empty")
    with np.load(args.labeled_archive, allow_pickle=False) as archive:
        receivers = tuple(sorted(set(archive["receiver_ids"].astype(str).tolist())))
        classes = tuple(sorted(set(archive["tx_labels"].astype(str).tolist())))
        days = tuple(sorted(set(archive["day_ids"].astype(str).tolist())))
    plan = build_complete_fit_plan(receivers, classes, days)
    output.mkdir(parents=True, exist_ok=False)
    fits_root = output / "fits"
    logs_root = output / "logs"
    fits_root.mkdir()
    logs_root.mkdir()
    _write(
        output / "matrix_plan.json",
        {
            "schema": "cvs.d103_r2.rxid_crossreceiver.fit_matrix.v1",
            "fit_count": len(plan),
            "gpus": list(gpus),
            "workers_per_gpu": args.workers_per_gpu,
            "fits": [asdict(spec) for spec in plan],
            "performance_dispatch_control": False,
            "systemic_stop_rule": (
                "two_distinct_rows_same_normalized_exception_fingerprint"
            ),
        },
    )
    lanes = [
        gpu for gpu in gpus for _ in range(args.workers_per_gpu)
    ]
    queue = list(plan)
    active: dict[asyncio.Task[tuple[Any, ...]], tuple[Any, str]] = {}
    fingerprints: Counter[str] = Counter()
    results = []
    stop_records: list[dict[str, Any]] = []
    start = time.monotonic()

    async def launch(spec, gpu):
        command = [
            str(args.python.resolve()),
            str((ROOT / "scripts" / "run_d103_r1_phase1_fit.py").resolve()),
            "--labeled-archive",
            str(args.labeled_archive.resolve()),
            "--labeled-manifest",
            str(args.labeled_manifest.resolve()),
            "--unlabeled-archive",
            str(args.unlabeled_archive.resolve()),
            "--unlabeled-manifest",
            str(args.unlabeled_manifest.resolve()),
            "--source-val-seal",
            str(args.source_val_seal.resolve()),
            "--source-val-manifest",
            str(args.source_val_manifest.resolve()),
            "--output-dir",
            str(fits_root / spec.fit_id),
            "--device",
            "cuda:0",
        ]
        for flag, value in (
            ("--held-receiver", spec.held_receiver),
            ("--held-day", spec.held_day),
            ("--held-class", spec.held_class),
        ):
            if value is not None:
                command.extend((flag, value))
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        stdout_path = logs_root / f"{spec.fit_id}.stdout.log"
        stderr_path = logs_root / f"{spec.fit_id}.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(ROOT.parent),
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                returncode = await process.wait()
            except asyncio.CancelledError:
                if process.returncode is None:
                    record = await _stop_bound_run_process_tree(
                        process,
                        fit_root=(fits_root / spec.fit_id).resolve(),
                        expected_cwd=ROOT.parent.resolve(),
                        gpu_lane=gpu,
                    )
                    stop_records.append(
                        {"fit_id": spec.fit_id, **record}
                    )
                raise
        fingerprint = None
        failed_path = fits_root / spec.fit_id / "fit_failed.json"
        if failed_path.is_file():
            fingerprint = _read(failed_path).get(
                "normalized_exception_fingerprint"
            )
        return spec, gpu, returncode, fingerprint, process.pid

    for gpu in lanes:
        if not queue:
            break
        spec = queue.pop(0)
        task = asyncio.create_task(launch(spec, gpu))
        active[task] = (spec, gpu)

    systemic_fingerprint = None
    while active:
        done, _ = await asyncio.wait(
            active, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            spec, gpu = active.pop(task)
            completed_spec, _, returncode, fingerprint, pid = await task
            results.append(
                {
                    "fit_id": completed_spec.fit_id,
                    "gpu": gpu,
                    "pid": pid,
                    "returncode": returncode,
                    "normalized_exception_fingerprint": fingerprint,
                }
            )
            if returncode != 0 and fingerprint:
                repeated = _record_failure_fingerprint(
                    fingerprints, str(fingerprint)
                )
                if repeated is not None:
                    systemic_fingerprint = repeated
            if systemic_fingerprint is None and queue:
                next_spec = queue.pop(0)
                next_task = asyncio.create_task(launch(next_spec, gpu))
                active[next_task] = (next_spec, gpu)
        if systemic_fingerprint is not None:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            active.clear()
            break

    elapsed = time.monotonic() - start
    success = sum(row["returncode"] == 0 for row in results)
    failed = sum(row["returncode"] != 0 for row in results)
    launched = len(plan) - len(queue)
    cancelled = launched - len(results)
    completed_manifests = [
        _read(fits_root / row["fit_id"] / "fit_complete.json")
        for row in results
        if row["returncode"] == 0
    ]
    spec_by_id = {spec.fit_id: spec for spec in plan}
    expected_input_sha = {
        "labeled_archive": sha256_file(args.labeled_archive),
        "unlabeled_archive": sha256_file(args.unlabeled_archive),
        "source_val_seal": sha256_file(args.source_val_seal),
    }
    manifest_errors: list[dict[str, str]] = []
    access_receipts: list[str] = []
    for result in results:
        if result["returncode"] != 0:
            continue
        fit_id = result["fit_id"]
        fit_root = fits_root / fit_id
        manifest = _read(fit_root / "fit_complete.json")
        teacher_path = fit_root / "teacher_arrays_fp32_ground_only.npz"
        try:
            with np.load(teacher_path, allow_pickle=False) as archive:
                teacher = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            spec = spec_by_id[fit_id]
            validated = validate_teacher_fit_manifest(
                manifest,
                teacher,
                expected_outer_spec={
                    "held_receiver": spec.held_receiver,
                    "held_day": spec.held_day,
                    "held_class": spec.held_class,
                },
                checkpoint_sha256=manifest["checkpoint_sha256"],
                runtime_sha256=manifest["runtime_sha256"],
                teacher_archive_sha256=sha256_file(teacher_path),
            )
            if validated["input_sha256"] != expected_input_sha:
                raise ValueError("fit input SHA differs from matrix inputs")
            access_receipts.append(validated["access_receipt_sha256"])
        except Exception as error:
            manifest_errors.append(
                {
                    "fit_id": fit_id,
                    "error": f"{type(error).__name__}:{error}",
                }
            )
    fit_manifest_validation_pass = bool(
        success == 246
        and len(completed_manifests) == 246
        and not manifest_errors
        and len(access_receipts) == 246
    )
    status = (
        "ARTIFACTS_COMPLETE"
        if (
            success == 246
            and failed == 0
            and not queue
            and fit_manifest_validation_pass
        )
        else (
            "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT"
            if systemic_fingerprint is not None
            else "INCOMPLETE_NO_PERFORMANCE_RESULT"
        )
    )
    total_gpu_hours = float(
        sum(float(row["fit_elapsed_seconds"]) for row in completed_manifests)
        / 3600.0
    )
    peak_memory_bytes = int(
        max(
            (int(row["peak_cuda_memory_bytes"]) for row in completed_manifests),
            default=0,
        )
    )
    run_root_bytes = int(
        sum(
            path.stat().st_size
            for path in output.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    )
    _write(
        output / "matrix_status.json",
        {
            "schema": "cvs.d103_r2.rxid_crossreceiver.fit_matrix_status.v1",
            "status": status,
            "planned_fit_count": 246,
            "launched_fit_count": launched,
            "completed_fit_count": success,
            "failed_fit_count": failed,
            "cancelled_run_owned_fit_count": cancelled,
            "undispatched_fit_count": len(queue),
            "completed_meta_steps": success * 400,
            "elapsed_seconds": elapsed,
            "total_gpu_hours": total_gpu_hours,
            "peak_memory_bytes": peak_memory_bytes,
            "run_root_bytes_at_matrix_close": run_root_bytes,
            "systemic_exception_fingerprint": systemic_fingerprint,
            "fit_manifest_validation_pass": fit_manifest_validation_pass,
            "fit_access_receipt_count": len(access_receipts),
            "fit_input_sha256": expected_input_sha,
            "fit_manifest_errors": manifest_errors,
            "systemic_stop_records": stop_records,
            "performance_result": False,
            "results": results,
        },
    )
    print(status)
    return 0 if status == "ARTIFACTS_COMPLETE" else 1


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
