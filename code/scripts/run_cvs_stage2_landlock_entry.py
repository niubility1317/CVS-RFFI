#!/usr/bin/env python3
"""Apply Landlock v4, clear the environment, and exec the Phase2 predictor."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_landlock_policy import (
    apply_landlock_v4,
    apply_network_seccomp_deny,
    close_inherited_fds,
)


def _regular(path: Path, *, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def _directory(path: Path, *, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{name} must be a non-symlink directory")
    return resolved


def _device(path: Path, *, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not stat.S_ISCHR(resolved.stat().st_mode):
        raise ValueError(f"{name} must be a character device")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--predictor-entry", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--system-read-root", type=Path, action="append", required=True)
    parser.add_argument("--gpu-device", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    runtime = _directory(args.runtime_root, name="runtime root")
    package = _directory(args.package_root, name="package root")
    output = _directory(args.output_root, name="output root")
    if any(output.iterdir()):
        raise ValueError("Landlock predictor output root must start empty")
    seal = _regular(args.detached_seal, name="detached seal")
    request = _regular(args.request_json, name="request JSON")
    python = _regular(args.python_executable, name="Python executable")
    predictor = _regular(args.predictor_entry, name="predictor entry")
    if runtime not in predictor.parents:
        raise ValueError("predictor entry must be inside the runtime closure")
    system_roots = [
        _directory(path, name="system read root") for path in args.system_read_root
    ]
    gpu_devices = [
        _device(path, name="GPU device") for path in args.gpu_device
    ]
    read_only = [
        runtime,
        package,
        seal,
        request,
        python,
        *system_roots,
    ]
    for optional in (Path("/proc"), Path("/sys")):
        if optional.is_dir():
            read_only.append(optional.resolve(strict=True))
    read_write = [output, *gpu_devices]
    for device_path in (
        Path("/dev/null"),
        Path("/dev/zero"),
        Path("/dev/random"),
        Path("/dev/urandom"),
    ):
        if device_path.exists():
            read_write.append(_device(device_path, name="system device"))

    close_inherited_fds()
    apply_landlock_v4(
        read_only_paths=read_only,
        read_write_paths=read_write,
    )
    apply_network_seccomp_deny()
    home = output / ".home"
    temp = output / ".tmp"
    cache = home / ".cache"
    for path in (home, temp, cache):
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(home),
        "TMPDIR": str(temp),
        "XDG_CACHE_HOME": str(cache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "CUDA_CACHE_DISABLE": "1",
        "PYTHONPATH": str(runtime),
    }
    argv = [
        str(python),
        str(predictor),
        "--request-json",
        str(request),
        "--predictor-package-root",
        str(package),
        "--detached-seal-path",
        str(seal),
        "--expected-seal-sha256",
        str(args.expected_seal_sha256),
        "--output-root",
        str(output),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
    ]
    os.chdir(runtime)
    os.execve(str(python), argv, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
