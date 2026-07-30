"""Publish and formally reload 30 Stage2-C v3 scoring sidecars."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, TypeVar


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
VARIANTS = {"new20": 20, "new5": 5}
EXPECTED_TASKS = 30
MAX_WORKERS = 8
SOURCE_TRUTH_SCHEMAS = {
    "cvs.phase2.query_truth_sidecar.v2",
    "cvs.phase2.query_truth_sidecar.v3",
}
_TASK = TypeVar("_TASK")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _exclusive_readonly_json(
    path: Path,
    payload: Mapping[str, Any],
    canonical_json_bytes: Callable[[Mapping[str, Any]], bytes],
) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing Stage2-C sidecar")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return hashlib.sha256(data).hexdigest()


def _package_base(
    root: Path,
    receiver: str,
    method_seed: int,
    variant: str,
) -> Path:
    return (
        root
        / "artifacts"
        / "packages"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
        / variant
    )


def _output_base(
    root: Path,
    receiver: str,
    method_seed: int,
    variant: str,
) -> Path:
    return (
        root
        / "artifacts"
        / "sidecars"
        / "stage2c"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
        / variant
    )


def _normalize_exception_fingerprint(exc: BaseException) -> str:
    line = f"{type(exc).__name__}: {exc}"
    line = re.sub(r"/(?:[^/\s:]+/)+[^/\s:]+", "<PATH>", line)
    line = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", line)
    line = re.sub(r"\b\d+\b", "<N>", line)
    return " ".join(line.split())


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
            if int(result["returncode"]) != 0 and fingerprint:
                fingerprints[fingerprint] += 1
        systemic_stop = any(count >= 2 for count in fingerprints.values())
        if systemic_stop:
            break
    return results, dict(sorted(fingerprints.items())), systemic_stop


def main() -> int:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError("fresh Stage2-C sidecar root must be absent")
    release_code = str(args.release_root / "code")
    if release_code not in sys.path:
        sys.path.insert(0, release_code)
    from cvsrffi.stage2_metric_scorer import (  # noqa: PLC0415
        SCORING_MANIFEST_SCHEMA,
        TRUTH_SIDECAR_SCHEMA,
        TRUTH_TOP_LEVEL_KEYS,
        _validate_truth_rows,
        canonical_json_bytes,
        load_verified_scoring_sidecar as load_formal_sidecar,
    )
    from cvsrffi.stage2_scoring_sidecar import (  # noqa: PLC0415
        load_verified_scoring_sidecar as load_source_sidecar,
    )

    tasks: list[dict[str, object]] = []
    for receiver in RECEIVERS:
        for method_seed in METHOD_SEEDS:
            for variant, new_count in VARIANTS.items():
                package = _package_base(
                    args.package_root,
                    receiver,
                    method_seed,
                    variant,
                )
                package_manifest_path = package / "predictor" / "package_manifest.json"
                package_manifest = json.loads(
                    package_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    package_manifest.get("stage") != "stage2c"
                    or package_manifest.get("receiver") != receiver
                    or int(package_manifest.get("seed", -1)) != method_seed
                    or int(package_manifest.get("new_class_count", -1))
                    != new_count
                ):
                    raise ValueError("Stage2-C package manifest identity drift")
                source_manifest_path = package / "scorer" / "scoring_manifest.json"
                truth, source_manifest, source_audit = load_source_sidecar(
                    source_manifest_path
                )
                if (
                    set(truth) != TRUTH_TOP_LEVEL_KEYS
                    or truth.get("schema") not in SOURCE_TRUTH_SCHEMAS
                    or truth.get("stage") != "stage2c"
                    or truth.get("receiver") != receiver
                    or int(truth.get("seed", -1)) != method_seed
                    or not list(truth.get("rows") or [])
                ):
                    raise ValueError("source Stage2-C truth identity drift")
                _validate_truth_rows(truth)
                tasks.append(
                    {
                        "receiver": receiver,
                        "method_seed": method_seed,
                        "variant": variant,
                        "new_class_count": new_count,
                        "source_truth": truth,
                        "source_truth_schema": truth["schema"],
                        "source_truth_sha256": source_audit[
                            "truth_sidecar_sha256"
                        ],
                        "predictor_package_root_sha256": source_manifest[
                            "predictor_package_root_sha256"
                        ],
                        "predictor_package_seal_sha256": source_manifest[
                            "predictor_package_seal_sha256"
                        ],
                        "output": _output_base(
                            args.output_root,
                            receiver,
                            method_seed,
                            variant,
                        ),
                    }
                )
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")

    log_root = args.output_root / "logs" / "sidecar_completion"
    log_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def run_one(task: dict[str, object]) -> dict[str, object]:
        key = (
            f"rx_{str(task['receiver']).replace('-', '_')}"
            f"_m{task['method_seed']}_{task['variant']}"
        )
        log_path = log_root / f"{key}.json"
        try:
            output = Path(task["output"])
            published_truth = dict(task["source_truth"])
            published_truth["schema"] = TRUTH_SIDECAR_SCHEMA
            _validate_truth_rows(published_truth)
            truth_path = output / "truth_sidecar.json"
            manifest_path = output / "scoring_manifest.json"
            truth_sha256 = _exclusive_readonly_json(
                truth_path,
                published_truth,
                canonical_json_bytes,
            )
            manifest = {
                "schema": SCORING_MANIFEST_SCHEMA,
                "predictor_package_root_sha256": task[
                    "predictor_package_root_sha256"
                ],
                "predictor_package_seal_sha256": task[
                    "predictor_package_seal_sha256"
                ],
                "truth_sidecar_json": truth_path.name,
                "truth_sidecar_sha256": truth_sha256,
                "scorer_output_must_not_feed_predictor": True,
            }
            manifest_sha256 = _exclusive_readonly_json(
                manifest_path,
                manifest,
                canonical_json_bytes,
            )
            loaded_truth, loaded_manifest, _audit = load_formal_sidecar(
                manifest_path,
                expected_scoring_manifest_sha256=manifest_sha256,
            )
            if (
                loaded_truth.get("schema") != TRUTH_SIDECAR_SCHEMA
                or loaded_truth.get("stage") != "stage2c"
                or loaded_truth.get("receiver") != task["receiver"]
                or int(loaded_truth.get("seed", -1))
                != int(task["method_seed"])
                or loaded_truth.get("rows") != task["source_truth"]["rows"]
                or loaded_manifest.get("predictor_package_root_sha256")
                != task["predictor_package_root_sha256"]
                or loaded_manifest.get("predictor_package_seal_sha256")
                != task["predictor_package_seal_sha256"]
            ):
                raise ValueError("published Stage2-C sidecar binding drift")
            receipt = {
                "schema": "cvs.full_ablation.stage2c.sidecar_receipt.v1",
                "receiver": task["receiver"],
                "method_seed": task["method_seed"],
                "variant": task["variant"],
                "new_class_count": task["new_class_count"],
                "source_truth_schema": task["source_truth_schema"],
                "source_truth_sha256": task["source_truth_sha256"],
                "published_truth_schema": TRUTH_SIDECAR_SCHEMA,
                "truth_sidecar_path": str(truth_path),
                "truth_sidecar_sha256": truth_sha256,
                "scoring_manifest_path": str(manifest_path),
                "scoring_manifest_sha256": manifest_sha256,
                "artifact_validated": True,
            }
            log_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            returncode = 0
            fingerprint = ""
            validation_error = ""
        except Exception as exc:
            returncode = 3
            fingerprint = _normalize_exception_fingerprint(exc)
            validation_error = f"{type(exc).__name__}: {exc}"
            log_path.write_text(validation_error + "\n", encoding="utf-8")
        return {
            key: task[key]
            for key in (
                "receiver",
                "method_seed",
                "variant",
                "new_class_count",
                "output",
            )
        } | {
            "key": key,
            "returncode": returncode,
            "artifact_validated": returncode == 0,
            "validation_error": validation_error,
            "exception_fingerprint": fingerprint,
            "log": str(log_path),
        }

    results, fingerprints, systemic_stop = _run_in_waves(
        tasks,
        runner=run_one,
    )
    succeeded = sum(row["returncode"] == 0 for row in results)
    summary = {
        "schema": "cvs.full_ablation.stage2c.sidecar_completion.v1",
        "expected": EXPECTED_TASKS,
        "launched": len(results),
        "completed": len(results),
        "succeeded": succeeded,
        "failed": sum(row["returncode"] != 0 for row in results),
        "not_launched": EXPECTED_TASKS - len(results),
        "validated": sum(row["artifact_validated"] is True for row in results),
        "published_sidecar_files": succeeded * 2,
        "registry_and_seal_authorized": (
            succeeded == EXPECTED_TASKS and not systemic_stop
        ),
        "exception_fingerprints": fingerprints,
        "systemic_stop": systemic_stop,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    summary_path = args.output_root / "source" / "sidecar_completion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if succeeded == EXPECTED_TASKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
