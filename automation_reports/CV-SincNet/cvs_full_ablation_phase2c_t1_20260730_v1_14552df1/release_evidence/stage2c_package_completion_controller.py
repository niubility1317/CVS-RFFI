"""Build the 45 exact screening-seed Stage2-B/C predictor packages."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, TypeVar

import numpy as np


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
NEW_CLASS_DRAW_SEED = 7282401
STAGES = ("before", "new20", "new5")
EXPECTED_TASKS = 45
SOURCE_SCHEMA = "cvs.stage2.package_build_summary.v1"
SOURCE_METHOD_SEEDS = (7283101, 7283102, 7283103, 7283104, 7283105)
OUTPUT_FLAGS = (
    "--predictor-out-root",
    "--scorer-out-root",
    "--detached-seal-path",
)
IMMUTABLE_INPUT_FLAGS = (
    "--target-cache-set",
    "--candidate-lock",
    "--checkpoint",
    "--adapter",
    "--head-artifact",
    "--tta-policy-json",
    "--phase1-deployment-binding",
    "--phase1-class-label-binding",
)
_TASK = TypeVar("_TASK")


def _json_ready_fields(
    task: dict[str, object],
    fields: tuple[str, ...],
) -> dict[str, object]:
    return {
        field: str(task[field]) if isinstance(task[field], Path) else task[field]
        for field in fields
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arg(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _set_arg(command: list[str], flag: str, value: str | Path | int) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def _labels(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("new-class label list contains duplicates")
    return values


def _output_base(
    root: Path,
    receiver: str,
    method_seed: int,
    stage: str,
) -> Path:
    return (
        root
        / "artifacts"
        / "packages"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
        / stage
    )


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
    max_workers: int,
    runner: Callable[[_TASK], dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], bool]:
    results: list[dict[str, object]] = []
    fingerprints: Counter[str] = Counter()
    systemic_stop = False
    for offset in range(0, len(tasks), max_workers):
        wave = tasks[offset : offset + max_workers]
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


def _templates(
    source: dict[str, object],
) -> dict[tuple[str, str], list[str]]:
    if (
        source.get("schema") != SOURCE_SCHEMA
        or int(source.get("expected", -1)) != 50
        or int(source.get("completed", -1)) != 50
        or int(source.get("succeeded", -1)) != 0
        or int(source.get("failed", -1)) != 50
    ):
        raise ValueError("source package summary header drift")
    rows = list(source.get("results") or [])
    if len(rows) != 50:
        raise ValueError("source package summary must contain 50 rows")
    templates: dict[tuple[str, str], list[str]] = {}
    identities: set[tuple[str, int, str]] = set()
    normalized: dict[tuple[str, str], list[list[str]]] = {}
    for row in rows:
        receiver = str(row["receiver"])
        method_seed = int(row["method_seed"])
        cache_seed = int(row["cache_seed"])
        stage = str(row["stage"])
        if (
            receiver not in RECEIVERS
            or method_seed not in SOURCE_METHOD_SEEDS
            or cache_seed != 713101 + (method_seed - SOURCE_METHOD_SEEDS[0])
            or stage not in {"before", "new20"}
            or int(row["support_seed"]) != method_seed + 100
            or int(row["query_seed"]) != method_seed + 200
            or int(row["returncode"]) != 1
        ):
            raise ValueError("source package identity drift")
        identity = (receiver, method_seed, stage)
        if identity in identities:
            raise ValueError("source package identity is duplicated")
        identities.add(identity)
        command = list(row["command"])
        if len(command) < 2:
            raise ValueError("source package command is malformed")
        target_cache_set = _arg(command, "--target-cache-set")
        if (
            f"/rx_{receiver.replace('-', '_')}/" not in target_cache_set
            or f"/seed_{cache_seed}/" not in target_cache_set
        ):
            raise ValueError("source target-cache identity drift")
        normalized_command = list(command)
        normalized_command[1] = "<SCRIPT>"
        for flag in (*OUTPUT_FLAGS, "--target-cache-set"):
            _set_arg(normalized_command, flag, f"<{flag}>")
        for flag in ("--seed", "--support-seed", "--query-seed"):
            _set_arg(normalized_command, flag, f"<{flag}>")
        normalized.setdefault((receiver, stage), []).append(normalized_command)
        templates.setdefault((receiver, stage), command)
    expected = {
        (receiver, stage)
        for receiver in RECEIVERS
        for stage in ("before", "new20")
    }
    if set(templates) != expected:
        raise ValueError("source package summary lacks receiver/stage coverage")
    expected_identities = {
        (receiver, method_seed, stage)
        for receiver in RECEIVERS
        for method_seed in SOURCE_METHOD_SEEDS
        for stage in ("before", "new20")
    }
    if identities != expected_identities:
        raise ValueError("source package summary identity coverage drift")
    for key, commands in normalized.items():
        if len(commands) != len(SOURCE_METHOD_SEEDS):
            raise ValueError(f"source package template coverage drift: {key}")
        baseline = commands[0]
        if any(command != baseline for command in commands[1:]):
            raise ValueError(f"source package command template drift: {key}")
    return templates


def _build_command(
    *,
    template: list[str],
    release_root: Path,
    output_root: Path,
    receiver: str,
    method_seed: int,
    stage: str,
    new_pool: list[str],
    new_labels: list[str],
) -> list[str]:
    command = list(template)
    command[1] = str(
        release_root / "code" / "scripts" / "build_cvs_stage2_predictor_bundle.py"
    )
    support_seed = method_seed + 100
    query_seed = method_seed + 200
    output = _output_base(output_root, receiver, method_seed, stage)
    for flag, value in (
        ("--receiver", receiver),
        ("--seed", method_seed),
        ("--support-seed", support_seed),
        ("--query-seed", query_seed),
        ("--predictor-out-root", output / "predictor"),
        ("--scorer-out-root", output / "scorer"),
        ("--detached-seal-path", output / "predictor.seal.json"),
    ):
        _set_arg(command, flag, value)
    if stage == "before":
        _set_arg(command, "--stage", "stage2b")
        _set_arg(command, "--new-class-draw-seed", 0)
        _set_arg(command, "--new-class-count", 0)
        _set_arg(
            command,
            "--stage2b-reference-new-class-labels",
            ",".join(new_pool),
        )
    else:
        _set_arg(command, "--stage", "stage2c")
        _set_arg(command, "--new-class-draw-seed", NEW_CLASS_DRAW_SEED)
        _set_arg(command, "--new-class-pool-labels", ",".join(new_pool))
        _set_arg(command, "--new-class-labels", ",".join(new_labels))
        _set_arg(command, "--new-class-count", len(new_labels))
    return command


def _validate_output(task: dict[str, object]) -> None:
    output = Path(task["output"])
    seal = output / "predictor.seal.json"
    manifest = output / "predictor" / "package_manifest.json"
    scoring = output / "scorer" / "scoring_manifest.json"
    truth = output / "scorer" / "truth_sidecar.json"
    for path in (seal, manifest, scoring, truth):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"package artifact is missing: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    truth_payload = json.loads(truth.read_text(encoding="utf-8"))
    expected_stage = "stage2b" if task["stage"] == "before" else "stage2c"
    if (
        truth_payload.get("stage") != expected_stage
        or truth_payload.get("receiver") != task["receiver"]
        or int(truth_payload.get("seed", -1)) != int(task["method_seed"])
        or not list(truth_payload.get("rows") or [])
    ):
        raise ValueError("package truth-sidecar identity drift")


def main() -> int:
    args = _parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    if args.output_root.exists():
        raise FileExistsError("fresh package output root must be absent")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256):
        raise ValueError("expected source SHA-256 must be lowercase hexadecimal")
    if not args.source_summary.is_file() or args.source_summary.is_symlink():
        raise FileNotFoundError("source package summary is not a regular file")
    if _sha256(args.source_summary) != args.expected_source_sha256:
        raise ValueError("source package summary SHA-256 drift")
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    templates = _templates(source)

    new20_template = templates[(RECEIVERS[0], "new20")]
    new_pool = _labels(_arg(new20_template, "--new-class-pool-labels"))
    if len(new_pool) != 20:
        raise ValueError("canonical new-class pool must contain 20 labels")
    order = np.random.default_rng(NEW_CLASS_DRAW_SEED).permutation(len(new_pool))
    new20 = [new_pool[int(index)] for index in order[:20]]
    new5 = new20[:5]

    tasks: list[dict[str, object]] = []
    for receiver in RECEIVERS:
        receiver_template = templates[(receiver, "new20")]
        if _labels(_arg(receiver_template, "--new-class-pool-labels")) != new_pool:
            raise ValueError("receiver new-class pool drift")
        for method_seed in METHOD_SEEDS:
            for stage in STAGES:
                template_stage = "before" if stage == "before" else "new20"
                labels = [] if stage == "before" else (new20 if stage == "new20" else new5)
                command = _build_command(
                    template=templates[(receiver, template_stage)],
                    release_root=args.release_root,
                    output_root=args.output_root,
                    receiver=receiver,
                    method_seed=method_seed,
                    stage=stage,
                    new_pool=new_pool,
                    new_labels=labels,
                )
                for flag in IMMUTABLE_INPUT_FLAGS:
                    path = Path(_arg(command, flag))
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"immutable package input is unreadable: {path}"
                        )
                tasks.append(
                    {
                        "receiver": receiver,
                        "method_seed": method_seed,
                        "support_seed": method_seed + 100,
                        "query_seed": method_seed + 200,
                        "stage": stage,
                        "new_class_count": len(labels),
                        "command": command,
                        "output": _output_base(
                            args.output_root,
                            receiver,
                            method_seed,
                            stage,
                        ),
                    }
                )
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")

    log_root = args.output_root / "logs" / "package_completion"
    log_root.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    release_code = str(args.release_root / "code")
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = release_code + (
        os.pathsep + inherited if inherited else ""
    )
    started = time.time()

    def run_one(task: dict[str, object]) -> dict[str, object]:
        key = (
            f"rx_{str(task['receiver']).replace('-', '_')}"
            f"_m{task['method_seed']}_{task['stage']}"
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
        returncode = process.returncode
        validation_error = ""
        if returncode == 0:
            try:
                _validate_output(task)
            except Exception as exc:
                returncode = 3
                validation_error = f"{type(exc).__name__}: {exc}"
        fingerprint = (
            _normalize_exception_fingerprint(log_path)
            if process.returncode != 0
            else validation_error
        )
        return _json_ready_fields(
            task,
            (
                "receiver",
                "method_seed",
                "support_seed",
                "query_seed",
                "stage",
                "new_class_count",
                "output",
            ),
        ) | {
            "key": key,
            "process_returncode": process.returncode,
            "returncode": returncode,
            "artifact_validated": returncode == 0,
            "validation_error": validation_error,
            "exception_fingerprint": fingerprint,
            "log": str(log_path),
        }

    results, fingerprints, systemic_stop = _run_in_waves(
        tasks,
        max_workers=args.max_workers,
        runner=run_one,
    )
    succeeded = sum(row["returncode"] == 0 for row in results)
    summary = {
        "schema": "cvs.full_ablation.stage2c.package_completion.v1",
        "expected": EXPECTED_TASKS,
        "launched": len(results),
        "completed": len(results),
        "succeeded": succeeded,
        "failed": sum(row["returncode"] != 0 for row in results),
        "not_launched": EXPECTED_TASKS - len(results),
        "validated": sum(row["artifact_validated"] is True for row in results),
        "exception_fingerprints": fingerprints,
        "systemic_stop": systemic_stop,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    summary_path = args.output_root / "source" / "package_completion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if succeeded == EXPECTED_TASKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
