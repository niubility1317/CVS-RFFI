#!/usr/bin/env python3
"""Run the locked D1 125-job Stage2-B/C independent stability tranche."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ROW_PIPELINE = REPO_ROOT / "code/scripts/run_cvs_somph_diag_row_pipeline.py"

SCHEMA = "cvs.phase2.somph_diag_125_stability.v1"
EVENT_SCHEMA = "cvs.phase2.somph_diag_125_event.v1"
SUMMARY_SCHEMA = "cvs.phase2.somph_diag_125_summary.v1"
CANDIDATE = "d1_historical_diag_fftrf"
CLAIM_SCOPE = "development_only_not_formal_confirmation"
FORMAL_LAUNCH_AUTHORITY = False
LOCKED_SHARD_COUNT = 8
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
CONFIRMATION_SEEDS = (713102, 713103, 713104, 713105, 713106)
SLICES = (
    (10, 5),
    (10, 10),
    (10, 20),
    (5, 20),
    (1, 20),
)
SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)

PHASE2_CONTRACT = {
    "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
    "clean_sample_access": False,
    "clean_derived_signal_access": False,
    "phase2_clean_dataset_reachable": False,
    "phase2_clean_cache_reachable": False,
    "phase2_clean_control_flow_reachable": False,
    "phase2_pretrained_artifact_policy": (
        "sealed_phase1_checkpoint_only"
    ),
    "phase2_query_decision_policy": (
        "per_sample_all_registered_classes"
    ),
    "phase2_query_role_oracle_access": False,
    "phase2_query_true_batch_class_count_access": False,
    "phase2_query_class_quota_access": False,
    "phase2_query_batch_global_assignment": False,
}


class StabilityLauncherError(ValueError):
    """Raised when the fixed tranche or no-overwrite boundary drifts."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_receiver(receiver: str) -> str:
    return str(receiver).replace("-", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                dict(payload),
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _locate_receiver_seed_inputs(
    *, cache_root: Path, authority_root: Path, receiver: str, seed: int
) -> dict[str, str]:
    receiver_leaf = f"rx_{_safe_receiver(receiver)}"
    cache_manifest = cache_root / receiver_leaf / f"seed_{seed}" / "cache_set.json"
    authority_bundle = (
        authority_root / f"authority_bundle_{receiver_leaf}_seed_{seed}"
    )
    authority_commit = authority_bundle / "COMMIT.json"
    for name, path in (
        ("cache manifest", cache_manifest),
        ("authority bundle", authority_bundle),
        ("authority COMMIT", authority_commit),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} missing: {path}")
    if not cache_manifest.is_file() or cache_manifest.is_symlink():
        raise StabilityLauncherError(
            f"cache manifest must be a regular file: {cache_manifest}"
        )
    if not authority_bundle.is_dir() or authority_bundle.is_symlink():
        raise StabilityLauncherError(
            f"authority bundle must be a regular directory: {authority_bundle}"
        )
    if not authority_commit.is_file() or authority_commit.is_symlink():
        raise StabilityLauncherError(
            f"authority COMMIT must be a regular file: {authority_commit}"
        )
    return {
        "cache_manifest": str(cache_manifest.resolve()),
        "authority_bundle": str(authority_bundle.resolve()),
        "authority_commit_path": str(authority_commit.resolve()),
        "authority_commit_sha256": _sha256(authority_commit),
    }


def build_manifest(
    *,
    cache_root: str | Path,
    authority_root: str | Path,
    phase1_checkpoint: str | Path,
    sealed_runtime: str | Path,
    method_lock: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build the exact locked 125-job tranche from receiver/seed inputs."""

    cache = Path(cache_root).resolve(strict=True)
    authority = Path(authority_root).resolve(strict=True)
    checkpoint = Path(phase1_checkpoint).resolve(strict=True)
    runtime = Path(sealed_runtime).resolve(strict=True)
    lock = Path(method_lock).resolve(strict=True)
    output = Path(output_root).resolve()
    jobs: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in CONFIRMATION_SEEDS:
            inputs = _locate_receiver_seed_inputs(
                cache_root=cache,
                authority_root=authority,
                receiver=receiver,
                seed=seed,
            )
            for k_shot, new_count in SLICES:
                job_id = (
                    f"rx_{_safe_receiver(receiver)}__seed_{seed}"
                    f"__k_{k_shot}__new_{new_count}"
                )
                jobs.append(
                    {
                        "index": len(jobs),
                        "planned_shard_index": len(jobs) % LOCKED_SHARD_COUNT,
                        "job_id": job_id,
                        "receiver": receiver,
                        "seed": seed,
                        "seed_role": (
                            "independent_stability_not_formal_confirmation"
                        ),
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "candidate": CANDIDATE,
                        "row_pair": {
                            "before_registration": "stage2b",
                            "after_registration": "stage2c",
                        },
                        "scenarios": list(SCENARIOS),
                        "support_nesting": {
                            "reference_k": 10,
                            "policy": (
                                "existing_row_builder_physical_prefix"
                            ),
                            "k1_uses_first_k10_physical_support": (
                                k_shot == 1
                            ),
                            "k5_uses_first_five_k10_physical_support": (
                                k_shot == 5
                            ),
                        },
                        **inputs,
                        "output_root": str(output / "jobs" / job_id),
                    }
                )
    expected = len(RECEIVERS) * len(CONFIRMATION_SEEDS) * len(SLICES)
    if len(jobs) != 125 or len(jobs) != expected:
        raise StabilityLauncherError(
            f"fixed tranche must contain exactly 125 jobs, got {len(jobs)}"
        )
    return {
        "schema": SCHEMA,
        "status": "LOCKED_INDEPENDENT_STABILITY_TRANCHE",
        "claim_scope": CLAIM_SCOPE,
        "formal_launch_authority": FORMAL_LAUNCH_AUTHORITY,
        "candidate": CANDIDATE,
        "locked_shard_count": LOCKED_SHARD_COUNT,
        "planned_shard_job_counts": [
            sum(
                int(job["planned_shard_index"]) == shard_index
                for job in jobs
            )
            for shard_index in range(LOCKED_SHARD_COUNT)
        ],
        "receivers": list(RECEIVERS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "development_seed_excluded": 713101,
        "slices": [
            {"k_shot": k_shot, "new_class_count": new_count}
            for k_shot, new_count in SLICES
        ],
        "job_count": len(jobs),
        "row_pair_count": len(jobs),
        "scenario_pair_count": len(jobs) * len(SCENARIOS),
        "scenario_state_metric_count": len(jobs) * len(SCENARIOS) * 2,
        "phase2_contract": dict(PHASE2_CONTRACT),
        "protocol_note": (
            "Manifest declarations do not replace the existing sealed row "
            "pipeline pre-open provenance and runtime access verification."
        ),
        "stage2_balance": (
            "Every job evaluates Stage2-B target-old adaptation before "
            "registration and Stage2-C target-old plus seen-new registration "
            "after registration with equal importance."
        ),
        "phase1_checkpoint": str(checkpoint),
        "phase1_checkpoint_sha256": _sha256(checkpoint),
        "sealed_runtime": str(runtime),
        "sealed_runtime_sha256": _sha256(runtime),
        "method_lock": str(lock),
        "method_lock_sha256": _sha256(lock),
        "row_pipeline": str(ROW_PIPELINE),
        "output_root": str(output),
        "jobs": jobs,
    }


def _job_command(
    job: Mapping[str, Any],
    *,
    phase1_checkpoint: str,
    sealed_runtime: str,
    method_lock: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROW_PIPELINE),
        "--cache-manifest",
        str(job["cache_manifest"]),
        "--authority-bundle",
        str(job["authority_bundle"]),
        "--authority-commit-sha256",
        str(job["authority_commit_sha256"]),
        "--phase1-checkpoint",
        phase1_checkpoint,
        "--sealed-runtime",
        sealed_runtime,
        "--method-lock",
        method_lock,
        "--output-root",
        str(job["output_root"]),
        "--receiver",
        str(job["receiver"]),
        "--seed",
        str(job["seed"]),
        "--k-shot",
        str(job["k_shot"]),
        "--new-count",
        str(job["new_class_count"]),
        "--device",
        device,
        "--candidate",
        CANDIDATE,
    ]


def _selected(index: int, shard_index: int, shard_count: int) -> bool:
    return int(index) % int(shard_count) == int(shard_index)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.shard_count) != LOCKED_SHARD_COUNT:
        raise StabilityLauncherError(
            f"shard count is locked to {LOCKED_SHARD_COUNT}"
        )
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise StabilityLauncherError("invalid shard index/count")
    manifest = build_manifest(
        cache_root=args.cache_root,
        authority_root=args.authority_root,
        phase1_checkpoint=args.phase1_checkpoint,
        sealed_runtime=args.sealed_runtime,
        method_lock=args.method_lock,
        output_root=args.output_root,
    )
    output = Path(manifest["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "matrix_manifest.json"
    manifest_bytes = _canonical_bytes(manifest)
    try:
        if not manifest_path.exists():
            _write_new(manifest_path, manifest)
    except FileExistsError:
        # Concurrent shards may race only on the identical immutable manifest.
        pass
    if manifest_path.exists():
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.read_bytes() != manifest_bytes
        ):
            raise StabilityLauncherError(
                "existing matrix manifest differs; refusing overwrite"
            )
    else:
        raise StabilityLauncherError("matrix manifest was not published")

    selected = [
        job
        for job in manifest["jobs"]
        if int(job["planned_shard_index"]) == int(args.shard_index)
    ]
    if not selected:
        raise StabilityLauncherError("locked shard unexpectedly has no jobs")
    if args.manifest_only:
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "MANIFEST_ONLY",
            "claim_scope": CLAIM_SCOPE,
            "formal_launch_authority": FORMAL_LAUNCH_AUTHORITY,
            "manifest": str(manifest_path),
            "total_job_count": manifest["job_count"],
            "selected_job_count": len(selected),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
        }

    shard_leaf = f"shard_{int(args.shard_index):03d}_of_{int(args.shard_count):03d}"
    events_path = output / "events" / f"{shard_leaf}.jsonl"
    summary_path = output / "summaries" / f"{shard_leaf}.json"
    if events_path.exists() or summary_path.exists():
        raise StabilityLauncherError(
            "shard events or summary already exists; refusing overwrite"
        )

    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    for job in selected:
        job_id = str(job["job_id"])
        stdout_path = output / "logs" / f"{job_id}.stdout.log"
        stderr_path = output / "logs" / f"{job_id}.stderr.log"
        if (
            Path(job["output_root"]).exists()
            or stdout_path.exists()
            or stderr_path.exists()
        ):
            failure = {
                "job_id": job_id,
                "returncode": None,
                "error": "job output or log already exists; no overwrite",
            }
            failed.append(failure)
            _append_event(
                events_path,
                {
                    "schema": EVENT_SCHEMA,
                    "timestamp": _now(),
                    "event": "JOB_REFUSED_EXISTING_OUTPUT",
                    **failure,
                },
            )
            if args.fail_fast:
                break
            continue

        command = _job_command(
            job,
            phase1_checkpoint=manifest["phase1_checkpoint"],
            sealed_runtime=manifest["sealed_runtime"],
            method_lock=manifest["method_lock"],
            device=args.device,
        )
        _append_event(
            events_path,
            {
                "schema": EVENT_SCHEMA,
                "timestamp": _now(),
                "event": "JOB_START",
                "job_id": job_id,
                "index": job["index"],
                "receiver": job["receiver"],
                "seed": job["seed"],
                "k_shot": job["k_shot"],
                "new_class_count": job["new_class_count"],
                "candidate": CANDIDATE,
                "device": args.device,
                "command": command,
            },
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout:
            with stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
        event = {
            "schema": EVENT_SCHEMA,
            "timestamp": _now(),
            "event": (
                "JOB_COMPLETE" if result.returncode == 0 else "JOB_FAILED"
            ),
            "job_id": job_id,
            "returncode": result.returncode,
            "output_root": job["output_root"],
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        _append_event(events_path, event)
        if result.returncode == 0:
            completed.append(job_id)
        else:
            failed.append(
                {
                    "job_id": job_id,
                    "returncode": result.returncode,
                    "error": "row pipeline technical failure",
                }
            )
            if args.fail_fast:
                break

    status = "PASS" if not failed and len(completed) == len(selected) else "PARTIAL_FAILURE"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "claim_scope": CLAIM_SCOPE,
        "formal_launch_authority": FORMAL_LAUNCH_AUTHORITY,
        "manifest": str(manifest_path),
        "events": str(events_path),
        "candidate": CANDIDATE,
        "device": args.device,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "total_job_count": manifest["job_count"],
        "selected_job_count": len(selected),
        "completed_job_count": len(completed),
        "failed_job_count": len(failed),
        "completed_job_ids": completed,
        "failures": failed,
        "fail_fast": bool(args.fail_fast),
    }
    _write_new(summary_path, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-root", required=True)
    result.add_argument("--authority-root", required=True)
    result.add_argument("--phase1-checkpoint", required=True)
    result.add_argument("--sealed-runtime", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", required=True)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument(
        "--shard-count",
        type=int,
        choices=(LOCKED_SHARD_COUNT,),
        default=LOCKED_SHARD_COUNT,
    )
    result.add_argument("--fail-fast", action="store_true")
    result.add_argument("--manifest-only", action="store_true")
    return result


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {"PASS", "MANIFEST_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
