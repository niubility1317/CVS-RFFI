#!/usr/bin/env python3
"""Bounded CLI for the frozen D92 E0 continuous-session screen.

The runner owns orchestration only.  It builds the five-job matrix manifest,
invokes the truth-free prediction entry, and reports technical artifact
counts.  Truth and scoring are intentionally implemented by the separate
analyzer CLI.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_continuous_session_matrix as matrix  # noqa: E402


DEFAULT_CONFIG = CODE_ROOT.parent / "configs" / "stage2_d92_e0_continuous_session_v1.json"


class D92ContinuousSessionRunnerError(RuntimeError):
    """Raised when the small continuous-session runner cannot proceed."""


PredictionEntry = Callable[..., Mapping[str, Any]]
DeltaEntry = Callable[..., Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable JSON file and refuse to replace an old receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError:
        # Read-only output is useful on POSIX, but not a reason to turn a
        # completed experiment into a runner failure on Windows.
        pass


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise D92ContinuousSessionRunnerError(f"{label} is not a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92ContinuousSessionRunnerError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise D92ContinuousSessionRunnerError(f"{label} must be an object")
    return payload


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _read_json(path, label="matrix manifest")
    try:
        matrix.validate_continuous_session_manifest(manifest)
    except (TypeError, ValueError, KeyError) as error:
        raise D92ContinuousSessionRunnerError(
            f"continuous-session manifest is not frozen: {error}"
        ) from error
    return manifest


def prepare(
    *,
    method_lock_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build and persist the five-job manifest without opening experiment data."""

    output = Path(output_root).resolve()
    manifest = matrix.build_continuous_session_manifest(
        method_lock_path=method_lock_path,
        output_root=output,
        require_package_files=False,
    )
    manifest_path = output / "matrix_manifest.json"
    _write_json_new(manifest_path, manifest)
    return {
        "status": "PREPARED",
        "manifest_path": str(manifest_path),
        "job_count": len(manifest["jobs"]),
        "session_fit_count": manifest["session_fit_count"],
    }


def _job(manifest: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    matches = [row for row in manifest["jobs"] if row.get("job_id") == job_id]
    if len(matches) != 1:
        raise D92ContinuousSessionRunnerError(f"unknown job_id: {job_id}")
    return matches[0]


def _schedule_payload(schedule_names: Sequence[str]) -> dict[str, dict[str, list[int]]]:
    unknown = [name for name in schedule_names if name not in matrix.SCHEDULES]
    if unknown:
        raise D92ContinuousSessionRunnerError(f"unknown schedule: {unknown[0]}")
    return {
        name: {
            "increments": list(matrix.SCHEDULES[name]),
            "arrival_order": list(matrix.ARRIVAL_ORDERS[name]),
        }
        for name in schedule_names
    }


def _prediction_kwargs(
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    schedule_names: Sequence[str],
    prepared_delta_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    packages = job["packages"]

    def package_value(name: str, key: str) -> Any:
        return packages[name][key]

    # The core prediction function owns package loading.  This dictionary
    # contains only truth-free package identities and schedule metadata; the
    # dispatch adapter below filters it to the concrete public signature.
    return {
        "manifest": manifest,
        "job": job,
        "job_record": job,
        "schedules": _schedule_payload(schedule_names),
        "schedule_names": list(schedule_names),
        "prepared_delta_root": prepared_delta_root,
        "output_root": output_root,
        "device": device,
        "before_enrollment_package_root": package_value(
            "before_enrollment", "package_root"
        ),
        "before_enrollment_seal_path": package_value(
            "before_enrollment", "detached_seal_path"
        ),
        "before_enrollment_seal_sha256": package_value(
            "before_enrollment", "expected_seal_sha256"
        ),
        "before_apply_package_root": package_value("before_apply", "package_root"),
        "before_apply_seal_path": package_value("before_apply", "detached_seal_path"),
        "before_apply_seal_sha256": package_value(
            "before_apply", "expected_seal_sha256"
        ),
        "after_apply_package_root": package_value("after_apply", "package_root"),
        "after_apply_seal_path": package_value("after_apply", "detached_seal_path"),
        "after_apply_seal_sha256": package_value(
            "after_apply", "expected_seal_sha256"
        ),
        "ground_component_dir": manifest["ground_component_dir"],
        "ground_manifest_path": manifest["ground_manifest_path"],
        "ground_manifest_sha256": manifest["ground_manifest_sha256"],
    }


def _load_prediction_entry() -> PredictionEntry:
    # Delayed import keeps ``--help`` and matrix preparation independent of
    # torch/checkpoint availability on the local workstation.
    from cvsrffi.stage2_d92_continuous_session_prediction import (  # noqa: PLC0415
        run_continuous_session_prediction,
    )

    return run_continuous_session_prediction


def _load_delta_entry() -> DeltaEntry:
    # Delta preparation is a separate truth-free stage.  Import it lazily so
    # matrix preparation and CLI help do not require the prediction runtime.
    from cvsrffi.stage2_d92_continuous_session_prediction import (  # noqa: PLC0415
        prepare_continuous_session_support_deltas,
    )

    return prepare_continuous_session_support_deltas


def _invoke_entry(
    entry: Callable[..., Mapping[str, Any]],
    kwargs: dict[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    try:
        signature = inspect.signature(entry)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
    try:
        result = entry(**kwargs)
    except TypeError as error:
        raise D92ContinuousSessionRunnerError(
            f"{label} entry signature/API mismatch: {error}"
        ) from error
    if not isinstance(result, Mapping):
        raise D92ContinuousSessionRunnerError(f"{label} entry must return a mapping")
    return result


def _require_completed_entry(
    result: Mapping[str, Any],
    *,
    expected_status: str,
    required_paths: Sequence[Path],
    label: str,
) -> None:
    if result.get("status") != expected_status:
        raise D92ContinuousSessionRunnerError(f"{label} entry did not complete")
    if any(path.is_symlink() or not path.is_file() for path in required_paths):
        raise D92ContinuousSessionRunnerError(f"{label} completion manifest is missing")


def _invoke_prediction(
    entry: PredictionEntry,
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    schedule_names: Sequence[str],
    prepared_delta_root: Path,
    output_root: Path,
    device: str,
) -> Mapping[str, Any] | None:
    kwargs = _prediction_kwargs(
        manifest=manifest,
        job=job,
        schedule_names=schedule_names,
        prepared_delta_root=prepared_delta_root,
        output_root=output_root,
        device=device,
    )
    return _invoke_entry(entry, kwargs, label="prediction")


def _delta_kwargs(
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    packages = job["packages"]

    def package_value(name: str, key: str) -> Any:
        return packages[name][key]

    return {
        "manifest": manifest,
        "job": job,
        "job_record": job,
        "output_root": output_root,
        "prepared_delta_root": output_root,
        "before_enrollment_package_root": package_value(
            "before_enrollment", "package_root"
        ),
        "before_enrollment_seal_path": package_value(
            "before_enrollment", "detached_seal_path"
        ),
        "before_enrollment_seal_sha256": package_value(
            "before_enrollment", "expected_seal_sha256"
        ),
        "after_enrollment_package_root": package_value(
            "after_enrollment", "package_root"
        ),
        "after_enrollment_seal_path": package_value(
            "after_enrollment", "detached_seal_path"
        ),
        "after_enrollment_seal_sha256": package_value(
            "after_enrollment", "expected_seal_sha256"
        ),
        "k_shot": int(job["k_shot"]),
        "new_class_count": int(job["new_class_count"]),
    }


def _prepared_delta_root(manifest: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    return Path(str(manifest["output_root"])).resolve() / "deltas" / str(job["job_id"])


def _require_prepared_delta(path: Path) -> None:
    if not path.is_dir() or not any(path.iterdir()):
        raise D92ContinuousSessionRunnerError(
            f"prepared delta root is missing or empty: {path}"
        )


def prepare_deltas(
    *,
    manifest_path: str | Path,
    job_id: str | None = None,
    delta_entry: DeltaEntry | None = None,
) -> dict[str, Any]:
    """Prepare one or all immutable support-delta roots without query access."""

    manifest = _load_manifest(manifest_path)
    jobs = [
        _job(manifest, job_id)
    ] if job_id is not None else list(manifest["jobs"])
    entry = delta_entry or _load_delta_entry()
    prepared: list[str] = []
    for job in jobs:
        destination = _prepared_delta_root(manifest, job)
        _ensure_new_directory(destination, label="delta")
        result = _invoke_entry(
            entry,
            _delta_kwargs(manifest=manifest, job=job, output_root=destination),
            label="delta",
        )
        _require_completed_entry(
            result,
            expected_status="PREPARED_DELTA_SUPPORT_COMPLETE",
            required_paths=(
                destination / "prepared_manifest.json",
                destination / "COMMIT.json",
            ),
            label="delta",
        )
        _write_json_new(
            destination / "delta_receipt.json",
            {
                "schema": "cvs.phase2.d92_e0_continuous_session.delta_receipt.v1",
                "status": "DELTAS_PREPARED_TRUTH_FREE",
                "timestamp": _now(),
                "job_id": job["job_id"],
                "output_root": str(destination),
                "new_class_count": int(job["new_class_count"]),
                "k_shot": int(job["k_shot"]),
            },
        )
        prepared.append(str(destination))
    return {
        "status": "DELTAS_PREPARED",
        "job_count": len(prepared),
        "delta_roots": prepared,
    }


def _ensure_new_directory(path: Path, *, label: str) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"{label} output already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def smoke(
    *,
    manifest_path: str | Path,
    device: str = "cpu",
    prediction_entry: PredictionEntry | None = None,
) -> dict[str, Any]:
    """Run only the first job's two truth-free liveness schedules."""

    manifest = _load_manifest(manifest_path)
    job = manifest["jobs"][0]
    schedule_names = ("batch_5", "singleton_forward")
    job_root = Path(str(job["output_root"]))
    prepared_delta_root = _prepared_delta_root(manifest, job)
    _require_prepared_delta(prepared_delta_root)
    output = job_root / "smoke"
    _ensure_new_directory(output, label="smoke")
    entry = prediction_entry or _load_prediction_entry()
    result = _invoke_prediction(
        entry,
        manifest=manifest,
        job=job,
        schedule_names=schedule_names,
        prepared_delta_root=prepared_delta_root,
        output_root=output,
        device=device,
    )
    _require_completed_entry(
        result,
        expected_status=(
            "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTIONS_COMPLETE"
        ),
        required_paths=(output / "prediction_manifest.json",),
        label="prediction",
    )
    receipt = {
        "schema": "cvs.phase2.d92_e0_continuous_session.smoke_receipt.v1",
        "status": "PREDICTION_COMPLETE_TRUTH_FREE",
        "timestamp": _now(),
        "job_id": job["job_id"],
        "schedules": list(schedule_names),
        "output_root": str(output),
    }
    _write_json_new(job_root / "smoke_receipt.json", receipt)
    return receipt


def run_job(
    *,
    manifest_path: str | Path,
    job_id: str,
    device: str = "cpu",
    prediction_entry: PredictionEntry | None = None,
) -> dict[str, Any]:
    """Run one job serially over all four frozen schedules."""

    manifest = _load_manifest(manifest_path)
    job = _job(manifest, job_id)
    schedule_names = tuple(matrix.SCHEDULE_NAMES)
    job_root = Path(str(job["output_root"]))
    receipt_path = job_root / "job_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"job output already exists: {job_root}")
    output = job_root / "full"
    prepared_delta_root = _prepared_delta_root(manifest, job)
    _require_prepared_delta(prepared_delta_root)
    _ensure_new_directory(output, label="job")
    entry = prediction_entry or _load_prediction_entry()
    result = _invoke_prediction(
        entry,
        manifest=manifest,
        job=job,
        schedule_names=schedule_names,
        prepared_delta_root=prepared_delta_root,
        output_root=output,
        device=device,
    )
    _require_completed_entry(
        result,
        expected_status=(
            "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTIONS_COMPLETE"
        ),
        required_paths=(output / "prediction_manifest.json",),
        label="prediction",
    )
    receipt = {
        "schema": matrix.JOB_RECEIPT_SCHEMA,
        "status": "PREDICTION_COMPLETE_TRUTH_FREE",
        "timestamp": _now(),
        "job_id": job["job_id"],
        "schedules": list(schedule_names),
        "output_root": str(output),
    }
    _write_json_new(receipt_path, receipt)
    return receipt


def status(*, manifest_path: str | Path) -> dict[str, Any]:
    """Return technical file counts without reading labels, scores, or metrics."""

    manifest = _load_manifest(manifest_path)
    completed = 0
    prediction_count = 0
    execution_receipt_count = 0
    job_directories = 0
    for job in manifest["jobs"]:
        root = Path(str(job["output_root"]))
        if root.is_dir():
            job_directories += 1
        if (root / "job_receipt.json").is_file():
            completed += 1
        if root.is_dir():
            prediction_count += sum(
                1 for path in root.rglob("prediction_artifact.npz") if path.is_file()
            )
            execution_receipt_count += sum(
                1 for path in root.rglob("execution_receipt.json") if path.is_file()
            )
    return {
        "status": "TECHNICAL_COUNTS_ONLY",
        "manifest_path": str(Path(manifest_path).resolve()),
        "job_count": int(manifest["job_count"]),
        "job_directory_count": job_directories,
        "completed_job_count": completed,
        "prediction_artifact_count": prediction_count,
        "execution_receipt_count": execution_receipt_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D92 E0 continuous-session truth-free runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="write the five-job manifest")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--output-root", type=Path, required=True)

    delta_parser = subparsers.add_parser(
        "prepare-deltas", help="prepare immutable support deltas for each job"
    )
    delta_parser.add_argument("--manifest", type=Path, required=True)
    delta_parser.add_argument("--job-id")

    smoke_parser = subparsers.add_parser("smoke", help="run first-job liveness schedules")
    smoke_parser.add_argument("--manifest", type=Path, required=True)
    smoke_parser.add_argument("--device", default="cpu")

    job_parser = subparsers.add_parser("run-job", help="run one job over all schedules")
    job_parser.add_argument("--manifest", type=Path, required=True)
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--device", default="cpu")

    status_parser = subparsers.add_parser("status", help="show technical artifact counts")
    status_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(method_lock_path=args.config, output_root=args.output_root)
    elif args.command == "prepare-deltas":
        result = prepare_deltas(manifest_path=args.manifest, job_id=args.job_id)
    elif args.command == "smoke":
        result = smoke(manifest_path=args.manifest, device=args.device)
    elif args.command == "run-job":
        result = run_job(
            manifest_path=args.manifest,
            job_id=args.job_id,
            device=args.device,
        )
    else:
        result = status(manifest_path=args.manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke
    raise SystemExit(main())


__all__ = [
    "D92ContinuousSessionRunnerError",
    "build_parser",
    "main",
    "prepare",
    "prepare_deltas",
    "run_job",
    "smoke",
    "status",
]
