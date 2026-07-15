"""Construct the minimal Linux bubblewrap policy for a Phase2 predictor."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Iterable, Sequence


class BwrapPolicyError(ValueError):
    pass


def _resolved_existing(path: str | Path, *, directory: bool) -> Path:
    value = Path(path)
    resolved = value.resolve(strict=True)
    if value.is_symlink() or (not resolved.is_dir() if directory else not resolved.is_file()):
        kind = "directory" if directory else "file"
        raise BwrapPolicyError(f"bwrap mount source must be a regular {kind}: {value}")
    return resolved


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def build_phase2_bwrap_command(
    *,
    bwrap: str,
    runtime_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    request_json: str | Path,
    output_root: str | Path,
    python_executable: str | Path,
    predictor_argv: Sequence[str],
    system_read_roots: Iterable[str | Path],
    trusted_system_read_roots: Iterable[str | Path],
    gpu_devices: Iterable[str | Path] = (),
    forbidden_roots: Iterable[str | Path] = (),
    strace_executable: str | Path | None = None,
    strace_output_fd: int | None = None,
) -> list[str]:
    """Return a no-network, one-write-root bwrap command.

    ``forbidden_roots`` is an external scorer/truth-root assertion.  It is not
    mounted and may not overlap any predictor-visible project artifact root.
    """

    runtime = _resolved_existing(runtime_root, directory=True)
    package = _resolved_existing(package_root, directory=True)
    seal = _resolved_existing(detached_seal, directory=False)
    request = _resolved_existing(request_json, directory=False)
    output = _resolved_existing(output_root, directory=True)
    python = _resolved_existing(python_executable, directory=False)
    project_visible = (runtime, package, seal, request, output)
    forbidden = [Path(path).resolve(strict=True) for path in forbidden_roots]
    for root in forbidden:
        if any(_overlaps(root, visible) for visible in project_visible):
            raise BwrapPolicyError("scorer/truth root overlaps predictor-visible roots")
    for first_index, first in enumerate(project_visible):
        for second in project_visible[first_index + 1 :]:
            if _overlaps(first, second):
                raise BwrapPolicyError("predictor mount sources must be physically disjoint")

    command = [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--dir",
        "/runtime",
        "--ro-bind",
        str(runtime),
        "/runtime/code",
        "--ro-bind",
        str(package),
        "/sealed/package",
        "--ro-bind",
        str(seal),
        "/sealed/package.seal.json",
        "--ro-bind",
        str(request),
        "/sealed/request.json",
        "--bind",
        str(output),
        "/output",
    ]
    trusted_system = {
        _resolved_existing(raw, directory=Path(raw).is_dir())
        for raw in trusted_system_read_roots
    }
    if not trusted_system:
        raise BwrapPolicyError("trusted system read-root allowlist is empty")
    seen_system: set[Path] = set()
    for raw in system_read_roots:
        path = _resolved_existing(raw, directory=Path(raw).is_dir())
        if path not in trusted_system:
            raise BwrapPolicyError("system read root is outside the fixed trusted allowlist")
        if path in seen_system:
            continue
        if str(path) in {"/proc", "/dev", "/tmp"} or path == Path(path.anchor) or any(
            _overlaps(path, visible) for visible in (*project_visible, *forbidden)
        ):
            raise BwrapPolicyError(
                "system read root may not be filesystem root or overlap project/truth roots"
            )
        seen_system.add(path)
        command.extend(["--ro-bind", str(path), str(path)])
    python_parent = python.parent
    if python_parent not in seen_system and not any(
        python == root or root in python.parents for root in seen_system
    ):
        raise BwrapPolicyError("python executable is outside declared read-only system roots")
    trace: Path | None = None
    if (strace_executable is None) != (strace_output_fd is None):
        raise BwrapPolicyError("strace executable and inherited output FD must be provided together")
    if strace_executable is not None:
        trace = _resolved_existing(strace_executable, directory=False)
        if not _inside(trace, seen_system):
            raise BwrapPolicyError("strace executable is outside declared read-only system roots")
        if not isinstance(strace_output_fd, int) or isinstance(strace_output_fd, bool) or strace_output_fd < 3:
            raise BwrapPolicyError("strace output FD must be an inherited non-stdio descriptor")
    for raw in gpu_devices:
        device = Path(raw).resolve(strict=True)
        if not stat.S_ISCHR(device.stat().st_mode):
            raise BwrapPolicyError("GPU device mount source must be a character device")
        command.extend(["--dev-bind", str(device), str(device)])
    command.extend(
        [
            "--clearenv",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "HOME",
            "/tmp/home",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "XDG_CACHE_HOME",
            "/tmp/home/.cache",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--setenv",
            "CUDA_CACHE_DISABLE",
            "1",
            "--setenv",
            "PYTHONPATH",
            "/runtime/code",
            "--chdir",
            "/runtime/code",
        ]
    )
    if trace is not None:
        command.extend(
            [
                str(trace),
                "-f",
                "-qq",
                "-yy",
                "-s",
                "4096",
                "-e",
                "trace=open,openat,openat2",
                "-o",
                f"/proc/self/fd/{strace_output_fd}",
            ]
        )
    command.extend([str(python), *[str(value) for value in predictor_argv]])
    return command
