"""Build only the 48 missing Stage2 predictor packages for the fresh v9 input root."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, TypeVar


EXPECTED_ROWS = 50
EXPECTED_BUILD_ROWS = 48
REUSED_RECEIVER = "20-1"
REUSED_METHOD_SEED = 7283101
OUTPUT_FLAGS = (
    "--predictor-out-root",
    "--scorer-out-root",
    "--detached-seal-path",
)
_TASK = TypeVar("_TASK")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def _output_base(input_root: Path, row: dict[str, object]) -> Path:
    receiver = str(row["receiver"]).replace("-", "_")
    return (
        input_root
        / "artifacts"
        / "packages"
        / f"rx_{receiver}"
        / f"method_{int(row['method_seed'])}"
        / str(row["stage"])
    )


def _build_command(
    *,
    original: list[str],
    release_root: Path,
    input_root: Path,
    row: dict[str, object],
) -> list[str]:
    command = list(original)
    command[1] = str(
        release_root / "code" / "scripts" / "build_cvs_stage2_predictor_bundle.py"
    )
    output_base = _output_base(input_root, row)
    replacements = {
        "--predictor-out-root": output_base / "predictor",
        "--scorer-out-root": output_base / "scorer",
        "--detached-seal-path": output_base / "predictor.seal.json",
    }
    for flag in OUTPUT_FLAGS:
        index = command.index(flag)
        command[index + 1] = str(replacements[flag])
    return command


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
    fingerprint_counts: Counter[str] = Counter()
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
                fingerprint_counts[fingerprint] += 1
        systemic_stop = any(count >= 2 for count in fingerprint_counts.values())
        if systemic_stop:
            break
    return results, dict(sorted(fingerprint_counts.items())), systemic_stop


def main() -> int:
    args = _parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    if args.input_root.exists():
        raise FileExistsError(f"fresh input root already exists: {args.input_root}")
    for stage in ("before", "new20"):
        seal = args.reuse_root / stage / "predictor.seal.json"
        if not seal.is_file():
            raise FileNotFoundError(f"reused package seal is missing: {seal}")

    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    rows = source["results"]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} source rows, got {len(rows)}")

    tasks: list[tuple[dict[str, object], list[str]]] = []
    for row in rows:
        if (
            row["receiver"] == REUSED_RECEIVER
            and int(row["method_seed"]) == REUSED_METHOD_SEED
        ):
            continue
        original = list(row["command"])
        binding_index = original.index("--phase1-class-label-binding")
        binding_path = Path(original[binding_index + 1])
        if not binding_path.is_file():
            raise FileNotFoundError(
                f"existing immutable class-label binding is unreadable: {binding_path}"
            )
        command = _build_command(
            original=original,
            release_root=args.release_root,
            input_root=args.input_root,
            row=row,
        )
        if command[command.index("--phase1-class-label-binding") + 1] != str(
            binding_path
        ):
            raise AssertionError("class-label binding path must remain unchanged")
        tasks.append((row, command))
    if len(tasks) != EXPECTED_BUILD_ROWS:
        raise ValueError(
            f"expected {EXPECTED_BUILD_ROWS} build rows, got {len(tasks)}"
        )

    log_root = args.input_root / "logs" / "package_completion"
    source_root = args.input_root / "source"
    log_root.mkdir(parents=True, exist_ok=False)
    source_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def run_one(item: tuple[dict[str, object], list[str]]) -> dict[str, object]:
        row, command = item
        key = (
            f"rx_{str(row['receiver']).replace('-', '_')}"
            f"_m{int(row['method_seed'])}_{row['stage']}"
        )
        log_path = log_root / f"{key}.log"
        with log_path.open("xb") as stream:
            process = subprocess.run(
                command,
                cwd=args.release_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        fingerprint = (
            _normalize_exception_fingerprint(log_path)
            if process.returncode != 0
            else ""
        )
        return {
            "key": key,
            "receiver": row["receiver"],
            "method_seed": row["method_seed"],
            "stage": row["stage"],
            "returncode": process.returncode,
            "exception_fingerprint": fingerprint,
            "log": str(log_path),
            "output": str(_output_base(args.input_root, row)),
        }

    results, exception_fingerprints, systemic_stop = _run_in_waves(
        tasks,
        max_workers=args.max_workers,
        runner=run_one,
    )
    for result in results:
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema": "cvs.stage2.v9.package_completion.v1",
        "expected_build": EXPECTED_BUILD_ROWS,
        "reused": 2,
        "launched": len(results),
        "completed": len(results),
        "succeeded": sum(row["returncode"] == 0 for row in results),
        "failed": sum(row["returncode"] != 0 for row in results),
        "not_launched": EXPECTED_BUILD_ROWS - len(results),
        "exception_fingerprints": exception_fingerprints,
        "systemic_stop": systemic_stop,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    summary_path = source_root / "package_completion_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "expected_build",
                    "reused",
                    "launched",
                    "completed",
                    "succeeded",
                    "failed",
                    "not_launched",
                    "exception_fingerprints",
                    "systemic_stop",
                    "elapsed_sec",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
