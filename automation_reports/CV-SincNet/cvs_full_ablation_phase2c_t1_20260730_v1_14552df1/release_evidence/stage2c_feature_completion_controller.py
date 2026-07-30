"""Build and formally reload 75 exact Stage2-C screening feature identities."""

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
import sys
import time
from typing import Callable, TypeVar


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
VARIANT_K = {
    "new20": (1, 2, 5, 10),
    "new5": (10,),
}
NEW_CLASS_DRAW_SEED = 7282401
GPU_COUNT = 8
SLOTS_PER_GPU = 2
OCCUPANCY_POLL_SECONDS = 30
EXPECTED_TASKS = 75
EXPECTED_SCOPE_CACHES = 225
_TASK = TypeVar("_TASK")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    return parser.parse_args()


def _package_base(
    root: Path,
    receiver: str,
    method_seed: int,
) -> Path:
    return (
        root
        / "artifacts"
        / "packages"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
    )


def _feature_root(
    root: Path,
    receiver: str,
    method_seed: int,
    variant: str,
    k_shot: int,
) -> Path:
    return (
        root
        / "artifacts"
        / "features"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
        / variant
        / f"k_{k_shot}"
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
    runner: Callable[[_TASK], dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    bool,
    list[dict[str, object]],
]:
    results: list[dict[str, object]] = []
    fingerprints: Counter[str] = Counter()
    systemic_stop = False
    occupancy_snapshots: list[dict[str, object]] = []
    pending = list(tasks)
    wave_index = 0
    while pending:
        occupancy = _gpu_compute_occupancy()
        gpu_slots = _available_gpu_slots(occupancy)
        occupancy_snapshots.append(
            {
                "wave_index": wave_index,
                "external_compute_processes_by_gpu": {
                    str(gpu): occupancy[gpu] for gpu in range(GPU_COUNT)
                },
                "available_slots": list(gpu_slots),
            }
        )
        if not gpu_slots:
            time.sleep(OCCUPANCY_POLL_SECONDS)
            continue
        raw_wave = pending[: len(gpu_slots)]
        pending = pending[len(raw_wave) :]
        wave = [
            {**task, "gpu": gpu_slots[index]}
            for index, task in enumerate(raw_wave)
        ]
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
        wave_index += 1
    return (
        results,
        dict(sorted(fingerprints.items())),
        systemic_stop,
        occupancy_snapshots,
    )


def _gpu_compute_occupancy() -> dict[int, int]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid = [value.strip() for value in line.split(",", 1)]
        uuid_to_index[uuid] = int(index_text)
    if set(uuid_to_index.values()) != set(range(GPU_COUNT)):
        raise RuntimeError("N607 GPU inventory must contain indices 0-7")
    occupancy = {gpu: 0 for gpu in range(GPU_COUNT)}
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in process_result.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        uuid, pid_text = [value.strip() for value in line.split(",", 1)]
        if uuid not in uuid_to_index or not pid_text.isdigit():
            raise RuntimeError("N607 compute-process inventory is malformed")
        occupancy[uuid_to_index[uuid]] += 1
    return occupancy


def _available_gpu_slots(occupancy: dict[int, int]) -> tuple[int, ...]:
    if set(occupancy) != set(range(GPU_COUNT)):
        raise ValueError("GPU occupancy must cover indices 0-7")
    return tuple(
        gpu
        for gpu in range(GPU_COUNT)
        for _ in range(max(0, SLOTS_PER_GPU - int(occupancy[gpu])))
    )


def _build_command(
    args: argparse.Namespace,
    *,
    receiver: str,
    method_seed: int,
    variant: str,
    k_shot: int,
) -> list[str]:
    package = _package_base(args.package_root, receiver, method_seed)
    before = package / "before"
    after = package / variant
    before_seal = before / "predictor.seal.json"
    after_seal = after / "predictor.seal.json"
    support_seed = method_seed + 100
    query_seed = method_seed + 200
    new_count = 20 if variant == "new20" else 5
    return [
        "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
        str(
            args.release_root
            / "code"
            / "scripts"
            / "build_full_ablation_stage2_feature_cache.py"
        ),
        "--before-package-root",
        str(before / "predictor"),
        "--before-seal-path",
        str(before_seal),
        "--before-seal-sha256",
        _sha256(before_seal),
        "--after-package-root",
        str(after / "predictor"),
        "--after-seal-path",
        str(after_seal),
        "--after-seal-sha256",
        _sha256(after_seal),
        "--phase1-deployment-binding-path",
        str(args.phase1_root / "artifacts/phase1_final/deployment_binding.json"),
        "--ground-component-dir",
        str(args.phase1_root / "artifacts/phase1_unsigned/package/component"),
        "--ground-manifest-sha256",
        "03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a",
        "--phase1-prototype-path",
        str(
            args.phase1_root
            / "artifacts/phase1_normalized/deployment_prototype"
            / "phase2_zid_prototypes.pt"
        ),
        "--phase1-prototype-manifest-path",
        str(
            args.phase1_root
            / "artifacts/phase1_normalized/deployment_prototype"
            / "phase2_zid_prototypes.json"
        ),
        "--expected-phase1-prototype-sha256",
        "e0e10b671dec5088bcb6e59b475dc3a99060b0ccbc03581e345a5e953b6088f0",
        "--expected-phase1-prototype-manifest-sha256",
        "89c1f21a5476e8d6b6a27264af6505d9ac4ab6eb66b32ea1e54c1d21405fc527",
        "--expected-phase1-bundle-sha256",
        "1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7",
        "--cache-output-root",
        str(
            _feature_root(
                args.output_root,
                receiver,
                method_seed,
                variant,
                k_shot,
            )
        ),
        "--phase2-data-status",
        "VALIDATED_ONCE",
        "--capsule-id",
        (
            f"d18-reuse-validated-once-rx{receiver}-seed713101"
            f"-m{method_seed}-k{k_shot}-new{new_count}"
        ),
        "--split-id",
        (
            f"p2_min_v1-rx{receiver}-m{method_seed}-s{support_seed}"
            f"-q{query_seed}-d{NEW_CLASS_DRAW_SEED}-k{k_shot}"
            f"-new{new_count}"
        ),
        "--k-shot",
        str(k_shot),
        "--method-seed",
        str(method_seed),
        "--support-seed",
        str(support_seed),
        "--query-seed",
        str(query_seed),
        "--new-class-draw-seed",
        str(NEW_CLASS_DRAW_SEED),
        "--device",
        "cuda:0",
    ]


def _load_unique_receipt(log_path: Path) -> dict[str, object]:
    values = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("feature log must contain exactly one JSON receipt")
    return values[0]


def _validate_receipt(
    *,
    task: dict[str, object],
    receipt: dict[str, object],
    loader: Callable[..., dict[str, object]],
) -> None:
    if (
        receipt.get("cache_output_root") != str(task["output"])
        or receipt.get("receiver") != task["receiver"]
        or int(receipt.get("method_seed", -1)) != int(task["method_seed"])
        or int(receipt.get("k_shot", -1)) != int(task["k_shot"])
        or int(receipt.get("new_class_count", -1))
        != int(task["new_class_count"])
        or receipt.get("query_truth_opened") is not False
        or receipt.get("raw_dataset_opened") is not False
        or receipt.get("cross_launch_data_identity_required") is not False
        or set(receipt.get("caches") or {}) != {
            "stage2a",
            "stage2b",
            "stage2c",
        }
    ):
        raise ValueError("feature completion receipt identity drift")
    loaded: dict[str, dict[str, object]] = {}
    for scope, item in receipt["caches"].items():
        loaded[scope] = loader(
            item["payload_path"],
            item["manifest_path"],
            expected_payload_sha256=item["payload_sha256"],
            expected_manifest_sha256=item["manifest_sha256"],
        )
    manifest = loaded["stage2c"]["manifest"]
    if (
        manifest.get("stage_scope") != "stage2c"
        or manifest.get("receiver") != task["receiver"]
        or int(manifest.get("method_seed", -1)) != int(task["method_seed"])
        or int(manifest.get("support_seed", -1))
        != int(task["support_seed"])
        or int(manifest.get("query_seed", -1)) != int(task["query_seed"])
        or int(manifest.get("new_class_draw_seed", -1))
        != NEW_CLASS_DRAW_SEED
        or int(manifest.get("k_shot", -1)) != int(task["k_shot"])
        or len(list(manifest.get("new_classes") or []))
        != int(task["new_class_count"])
        or manifest.get("query_truth_present") is not False
        or manifest.get("clean_source_samples_present") is not False
    ):
        raise ValueError("Stage2-C feature manifest identity drift")


def main() -> int:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError("fresh feature output root must be absent")
    release_code = str(args.release_root / "code")
    if release_code not in sys.path:
        sys.path.insert(0, release_code)
    from cvsrffi.stage2_ablation_feature_cache import (  # noqa: PLC0415
        load_feature_cache,
    )

    tasks: list[dict[str, object]] = []
    for receiver in RECEIVERS:
        for method_seed in METHOD_SEEDS:
            for variant, k_values in VARIANT_K.items():
                for k_shot in k_values:
                    output = _feature_root(
                        args.output_root,
                        receiver,
                        method_seed,
                        variant,
                        k_shot,
                    )
                    tasks.append(
                        {
                            "receiver": receiver,
                            "method_seed": method_seed,
                            "support_seed": method_seed + 100,
                            "query_seed": method_seed + 200,
                            "variant": variant,
                            "new_class_count": 20 if variant == "new20" else 5,
                            "k_shot": k_shot,
                            "output": output,
                            "command": _build_command(
                                args,
                                receiver=receiver,
                                method_seed=method_seed,
                                variant=variant,
                                k_shot=k_shot,
                            ),
                        }
                    )
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")

    log_root = args.output_root / "logs" / "feature_completion"
    log_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def run_one(task: dict[str, object]) -> dict[str, object]:
        key = (
            f"rx_{str(task['receiver']).replace('-', '_')}"
            f"_m{task['method_seed']}_{task['variant']}_k{task['k_shot']}"
        )
        log_path = log_root / f"{key}.log"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = release_code + (
            os.pathsep + inherited if inherited else ""
        )
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
                receipt = _load_unique_receipt(log_path)
                _validate_receipt(
                    task=task,
                    receipt=receipt,
                    loader=load_feature_cache,
                )
            except Exception as exc:
                returncode = 3
                validation_error = f"{type(exc).__name__}: {exc}"
        fingerprint = (
            _normalize_exception_fingerprint(log_path)
            if process.returncode != 0
            else validation_error
        )
        return {
            key: task[key]
            for key in (
                "receiver",
                "method_seed",
                "support_seed",
                "query_seed",
                "variant",
                "new_class_count",
                "k_shot",
                "gpu",
                "output",
            )
        } | {
            "key": key,
            "process_returncode": process.returncode,
            "returncode": returncode,
            "artifact_validated": returncode == 0,
            "validation_error": validation_error,
            "exception_fingerprint": fingerprint,
            "log": str(log_path),
        }

    results, fingerprints, systemic_stop, occupancy_snapshots = _run_in_waves(
        tasks,
        runner=run_one,
    )
    succeeded = sum(row["returncode"] == 0 for row in results)
    summary = {
        "schema": "cvs.full_ablation.stage2c.feature_completion.v1",
        "expected": EXPECTED_TASKS,
        "launched": len(results),
        "completed": len(results),
        "succeeded": succeeded,
        "failed": sum(row["returncode"] != 0 for row in results),
        "not_launched": EXPECTED_TASKS - len(results),
        "validated": sum(row["artifact_validated"] is True for row in results),
        "generated_scope_caches": succeeded * 3,
        "expected_scope_caches": EXPECTED_SCOPE_CACHES,
        "stage2c_identity_count": succeeded,
        "exception_fingerprints": fingerprints,
        "systemic_stop": systemic_stop,
        "gpu_count": GPU_COUNT,
        "slots_per_gpu": SLOTS_PER_GPU,
        "gpu_occupancy_snapshots": occupancy_snapshots,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    summary_path = args.output_root / "source" / "feature_completion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if succeeded == EXPECTED_TASKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
