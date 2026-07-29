"""Publish 25 canonical Stage2-A scoring sidecars for fresh v11."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, TypeVar


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7283101, 7283102, 7283103, 7283104, 7283105)
EXPECTED_TASKS = 25
MAX_WORKERS = 8
REUSED_RECEIVER = "20-1"
REUSED_METHOD_SEED = 7283101
SOURCE_TRUTH_SCHEMAS = {
    "cvs.phase2.query_truth_sidecar.v2",
    "cvs.phase2.query_truth_sidecar.v3",
}
SOURCE_TRUTH_KEYS = {"schema", "stage", "receiver", "seed", "rows"}
_TASK = TypeVar("_TASK")


def _normalize_exception_fingerprint(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not re.match(
            r"^(?:[A-Za-z_][\w.]*(?:Error|Exception)|"
            r"AssertionError|KeyboardInterrupt|SystemExit):",
            line,
        ):
            continue
        line = re.sub(r"/(?:[^/\s:]+/)+[^/\s:]+", "<PATH>", line)
        line = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", line)
        line = re.sub(r"\b\d+\b", "<N>", line)
        return " ".join(line.split())
    return ""


def _run_in_waves(
    tasks: list[_TASK],
    *,
    runner: Callable[[_TASK], dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], bool]:
    results: list[dict[str, object]] = []
    fingerprints: Counter[str] = Counter()
    systemic_stop = False
    for offset in range(0, len(tasks), MAX_WORKERS):
        wave = tasks[offset : offset + MAX_WORKERS]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(wave)
        ) as executor:
            wave_results = list(executor.map(runner, wave))
        results.extend(wave_results)
        for result in wave_results:
            fingerprint = str(result.get("exception_fingerprint", ""))
            if (
                int(result["returncode"]) != 0
                or result.get("artifact_validated") is False
            ) and fingerprint:
                fingerprints[fingerprint] += 1
        systemic_stop = any(count >= 2 for count in fingerprints.values())
        if systemic_stop:
            break
    return results, dict(sorted(fingerprints.items())), systemic_stop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _package_base(args: argparse.Namespace, receiver: str, seed: int) -> Path:
    if receiver == REUSED_RECEIVER and seed == REUSED_METHOD_SEED:
        return args.reuse_root
    return (
        args.package_root
        / "artifacts"
        / "packages"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{seed}"
    )


def _load_unique_receipt(log_path: Path) -> dict[str, object]:
    receipts: list[dict[str, object]] = []
    for line in log_path.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("sidecar receipt must be a JSON object")
        receipts.append(value)
    if len(receipts) != 1:
        raise ValueError("sidecar log must contain exactly one JSON receipt")
    return receipts[0]


def _validate_published_artifact(
    *,
    task: dict[str, object],
    receipt: dict[str, object],
    formal_loader: Callable[..., tuple[dict, dict, dict]],
    published_truth_schema: str,
) -> None:
    output = Path(task["output"]).absolute()
    truth_path = Path(str(receipt["truth_sidecar_path"])).absolute()
    manifest_path = Path(str(receipt["scoring_manifest_path"])).absolute()
    if truth_path.parent != output or manifest_path.parent != output:
        raise ValueError("sidecar receipt output path drift")
    if (
        receipt.get("source_truth_schema") != task["source_truth_schema"]
        or receipt.get("published_truth_schema") != published_truth_schema
    ):
        raise ValueError("sidecar receipt schema transition drift")
    if (
        not truth_path.is_file()
        or truth_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise FileNotFoundError("published sidecar files are incomplete")
    truth, manifest, _audit = formal_loader(
        manifest_path,
        expected_scoring_manifest_sha256=str(
            receipt["scoring_manifest_sha256"]
        ),
    )
    if (
        truth.get("schema") != published_truth_schema
        or truth.get("stage") != "stage2a"
        or truth.get("receiver") != task["receiver"]
        or int(truth.get("seed")) != int(task["method_seed"])
        or truth.get("rows") != task["source_truth_rows"]
    ):
        raise ValueError("published Stage2-A truth identity drift")
    if (
        manifest.get("predictor_package_root_sha256")
        != task["predictor_package_root_sha256"]
        or manifest.get("predictor_package_seal_sha256")
        != task["predictor_package_seal_sha256"]
        or receipt.get("source_stage2b_truth_sha256")
        != task["source_truth_sha256"]
    ):
        raise ValueError("published scoring manifest binding drift")


def _artifact_counts(
    results: list[dict[str, object]],
) -> dict[str, int | bool]:
    process_succeeded = sum(
        row["process_returncode"] == 0 for row in results
    )
    validated = sum(row["artifact_validated"] is True for row in results)
    failed = sum(row["returncode"] != 0 for row in results)
    validation_failed = sum(
        row["process_returncode"] == 0
        and row["artifact_validated"] is not True
        for row in results
    )
    if validated + failed != len(results):
        raise AssertionError(
            "validated/failed accounting does not cover completed tasks"
        )
    return {
        "succeeded": validated,
        "failed": failed,
        "process_succeeded": process_succeeded,
        "execution_failed": len(results) - process_succeeded,
        "validated": validated,
        "validation_failed": validation_failed,
        "published_sidecar_files": validated * 2,
        "registry_and_seal_authorized": (
            validated == EXPECTED_TASKS and failed == 0
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError("fresh sidecar output root must be absent")
    release_code = str(args.release_root / "code")
    if release_code not in sys.path:
        sys.path.insert(0, release_code)
    from cvsrffi.stage2_metric_scorer import (  # noqa: PLC0415
        TRUTH_SIDECAR_SCHEMA,
        _validate_truth_rows,
        load_verified_scoring_sidecar as load_formal_sidecar,
    )
    from cvsrffi.stage2_scoring_sidecar import (  # noqa: PLC0415
        load_verified_scoring_sidecar as load_source_sidecar,
    )

    log_root = args.output_root / "logs" / "sidecar_completion"
    sidecar_root = args.output_root / "artifacts" / "sidecars" / "stage2a"

    tasks: list[dict[str, object]] = []
    package_source_counts = {"package_root": 0, "reuse_root": 0}
    for receiver in RECEIVERS:
        for seed in METHOD_SEEDS:
            package = _package_base(args, receiver, seed)
            if package == args.reuse_root:
                package_source_counts["reuse_root"] += 1
            else:
                package_source_counts["package_root"] += 1
            scorer = package / "before" / "scorer"
            manifest_path = scorer / "scoring_manifest.json"
            truth, manifest, source_audit = load_source_sidecar(manifest_path)
            if (
                set(truth) != SOURCE_TRUTH_KEYS
                or truth.get("schema") not in SOURCE_TRUTH_SCHEMAS
                or truth.get("stage") != "stage2b"
                or truth.get("receiver") != receiver
                or int(truth.get("seed")) != seed
                or not isinstance(truth.get("rows"), list)
                or not truth["rows"]
            ):
                raise ValueError("source Stage2-B sidecar identity drift")
            _validate_truth_rows(truth)
            output = (
                sidecar_root
                / f"rx_{receiver.replace('-', '_')}"
                / f"method_{seed}"
            )
            command = [
                "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
                str(
                    args.release_root
                    / "code"
                    / "scripts"
                    / "build_full_ablation_stage2a_scoring_sidecar.py"
                ),
                "--source-stage2b-truth",
                str(scorer / "truth_sidecar.json"),
                "--expected-source-truth-sha256",
                str(manifest["truth_sidecar_sha256"]),
                "--predictor-package-root-sha256",
                str(manifest["predictor_package_root_sha256"]),
                "--predictor-package-seal-sha256",
                str(manifest["predictor_package_seal_sha256"]),
                "--output-root",
                str(output),
            ]
            tasks.append(
                {
                    "receiver": receiver,
                    "method_seed": seed,
                    "command": command,
                    "output": output,
                    "source_truth_schema": truth["schema"],
                    "source_truth_rows": truth["rows"],
                    "source_truth_sha256": source_audit[
                        "truth_sidecar_sha256"
                    ],
                    "predictor_package_root_sha256": manifest[
                        "predictor_package_root_sha256"
                    ],
                    "predictor_package_seal_sha256": manifest[
                        "predictor_package_seal_sha256"
                    ],
                }
            )
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")
    if package_source_counts != {"package_root": 24, "reuse_root": 1}:
        raise ValueError("sidecar package source count drift")

    log_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = release_code + (
        os.pathsep + inherited if inherited else ""
    )
    started = time.time()

    def run_one(task: dict[str, object]) -> dict[str, object]:
        key = (
            f"rx_{str(task['receiver']).replace('-', '_')}"
            f"_m{task['method_seed']}"
        )
        log_path = log_root / f"{key}.log"
        with log_path.open("xb") as stream:
            process = subprocess.run(
                list(task["command"]),
                cwd=args.release_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        artifact_validated = False
        validation_error = ""
        returncode = process.returncode
        if process.returncode == 0:
            try:
                receipt = _load_unique_receipt(log_path)
                _validate_published_artifact(
                    task=task,
                    receipt=receipt,
                    formal_loader=load_formal_sidecar,
                    published_truth_schema=TRUTH_SIDECAR_SCHEMA,
                )
                artifact_validated = True
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                returncode = 3
        fingerprint = (
            _normalize_exception_fingerprint(log_path)
            if process.returncode != 0
            else validation_error
        )
        return {
            "key": key,
            "receiver": task["receiver"],
            "method_seed": task["method_seed"],
            "process_returncode": process.returncode,
            "returncode": returncode,
            "exception_fingerprint": fingerprint,
            "artifact_validated": artifact_validated,
            "validation_error": validation_error,
            "output": str(task["output"]),
            "log": str(log_path),
        }

    results, exception_fingerprints, systemic_stop = _run_in_waves(
        tasks,
        runner=run_one,
    )
    artifact_counts = _artifact_counts(results)
    summary = {
        "schema": "cvs.stage2.v11.sidecar_completion.v1",
        "expected": EXPECTED_TASKS,
        "launched": len(results),
        "completed": len(results),
        "not_launched": EXPECTED_TASKS - len(results),
        "exception_fingerprints": exception_fingerprints,
        "systemic_stop": systemic_stop,
        "package_source_counts": package_source_counts,
        "elapsed_sec": time.time() - started,
        "results": results,
        **artifact_counts,
    }
    summary_path = args.output_root / "source" / "sidecar_completion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
