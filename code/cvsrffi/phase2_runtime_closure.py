"""Build and verify the minimal Phase2 predictor Python runtime closure.

The produced ``runtime`` directory is intended to be mounted read-only at
``/runtime/code``.  It contains only the reviewed predictor modules and the
single production entry script; the manifest is deliberately kept outside
that mount so it cannot become an undeclared importable runtime member.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


RUNTIME_CLOSURE_SCHEMA = "cvs_phase2_predictor_runtime_closure_v1"
RUNTIME_MOUNT_PATH = "/runtime/code"
RUNTIME_ENTRYPOINT = "/runtime/code/scripts/run_cvs_stage2_predictor.py"
RUNTIME_MANIFEST_NAME = "runtime_closure_manifest.json"

RUNTIME_MEMBER_ALLOWLIST = (
    "cvsrffi/__init__.py",
    "cvsrffi/phase2_memfd_snapshot.py",
    "cvsrffi/phase2_runtime_contract.py",
    "cvsrffi/phase2_symmetric_head.py",
    "cvsrffi/stage2_predictor_bundle.py",
    "cvsrffi/stage2_predictor_runtime.py",
    "cvsrffi/stage2_predictor_entry.py",
    "cvsrffi/stage2_prediction_artifact.py",
    "scripts/run_cvs_stage2_predictor.py",
)

_ALLOWED_INTERNAL_IMPORTS = {
    "cvsrffi.phase2_runtime_contract",
    "cvsrffi.phase2_symmetric_head",
    "cvsrffi.phase2_memfd_snapshot",
    "cvsrffi.stage2_predictor_bundle",
    "cvsrffi.stage2_predictor_runtime",
    "cvsrffi.stage2_predictor_entry",
    "cvsrffi.stage2_prediction_artifact",
}

# Keep this intentionally exact.  A new dependency must receive an explicit
# review instead of silently widening the predictor runtime.
_ALLOWED_EXTERNAL_IMPORTS = {
    "__future__",
    "argparse",
    "collections.abc",
    "contextlib",
    "ctypes",
    "dataclasses",
    "datetime",
    "errno",
    "fcntl",
    "hashlib",
    "io",
    "json",
    "numpy",
    "os",
    "pathlib",
    "platform",
    "re",
    "secrets",
    "stat",
    "sys",
    "time",
    "torch",
    "typing",
    "zipfile",
}

_FORBIDDEN_IMPORT_PARTS = {
    "dataset",
    "datasets",
    "dataloader",
    "legacy",
    "loader",
    "paper_reproduction",
    "ssdg",
    "train",
    "trainer",
    "training",
    "wisig_runtime",
}

_EXPECTED_IMPORTS_BY_MEMBER = {
    "cvsrffi/__init__.py": set(),
    "cvsrffi/phase2_memfd_snapshot.py": {
        "__future__",
        "contextlib",
        "ctypes",
        "dataclasses",
        "fcntl",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "stat",
        "typing",
    },
    "cvsrffi/phase2_runtime_contract.py": {
        "__future__",
        "collections.abc",
        "pathlib",
        "re",
        "typing",
    },
    "cvsrffi/phase2_symmetric_head.py": {
        "__future__",
        "numpy",
        "typing",
    },
    "cvsrffi/stage2_predictor_bundle.py": {
        "__future__",
        "contextlib",
        "cvsrffi.phase2_memfd_snapshot",
        "cvsrffi.phase2_runtime_contract",
        "hashlib",
        "json",
        "numpy",
        "os",
        "pathlib",
        "re",
        "stat",
        "typing",
        "zipfile",
    },
    "cvsrffi/stage2_predictor_runtime.py": {
        "__future__",
        "cvsrffi.phase2_symmetric_head",
        "cvsrffi.stage2_predictor_bundle",
        "hashlib",
        "json",
        "numpy",
        "pathlib",
        "time",
        "torch",
        "typing",
    },
    "cvsrffi/stage2_predictor_entry.py": {
        "__future__",
        "cvsrffi.phase2_runtime_contract",
        "cvsrffi.stage2_predictor_bundle",
        "cvsrffi.stage2_predictor_runtime",
        "numpy",
        "pathlib",
        "torch",
        "typing",
    },
    "cvsrffi/stage2_prediction_artifact.py": {
        "__future__",
        "ctypes",
        "datetime",
        "errno",
        "hashlib",
        "io",
        "json",
        "numpy",
        "os",
        "pathlib",
        "re",
        "secrets",
        "stat",
        "sys",
        "typing",
        "zipfile",
    },
    "scripts/run_cvs_stage2_predictor.py": {
        "__future__",
        "argparse",
        "cvsrffi.phase2_memfd_snapshot",
        "cvsrffi.phase2_runtime_contract",
        "cvsrffi.stage2_prediction_artifact",
        "cvsrffi.stage2_predictor_bundle",
        "cvsrffi.stage2_predictor_entry",
        "json",
        "numpy",
        "os",
        "pathlib",
        "stat",
        "sys",
        "torch",
        "typing",
    },
}

_MANIFEST_KEYS = {
    "schema",
    "runtime_mount_path",
    "python_path",
    "entrypoint",
    "member_allowlist",
    "member_count",
    "members",
    "root_sha256",
    "copy_policy",
    "import_closure_policy",
}

_MEMBER_KEYS = {
    "relative_path",
    "sha256",
    "size_bytes",
    "imports",
}


class Phase2RuntimeClosureError(ValueError):
    """Raised when a runtime closure is wider than the reviewed allowlist."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _root_sha256(entries: list[Mapping[str, Any]]) -> str:
    canonical = "\n".join(
        f"{entry['relative_path']}\0{entry['sha256']}\0{entry['size_bytes']}"
        for entry in sorted(entries, key=lambda value: str(value["relative_path"]))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_relative_member(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise Phase2RuntimeClosureError(f"invalid runtime member path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise Phase2RuntimeClosureError(f"invalid runtime member path: {value!r}")
    return parsed


def _assert_no_symlink_chain(root: Path, path: Path) -> None:
    root_resolved = root.resolve(strict=True)
    path_resolved = path.resolve(strict=True)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise Phase2RuntimeClosureError(f"runtime member escapes source root: {path}") from exc

    if root.is_symlink():
        raise Phase2RuntimeClosureError(f"source root must not be a symlink: {root}")
    cursor = root
    relative = path.relative_to(root)
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Phase2RuntimeClosureError(f"runtime member symlink is forbidden: {cursor}")


def _read_regular_source(root: Path, relative_path: str) -> bytes:
    parsed = _validate_relative_member(relative_path)
    path = root.joinpath(*parsed.parts)
    if not path.exists():
        raise FileNotFoundError(path)
    _assert_no_symlink_chain(root, path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise Phase2RuntimeClosureError(f"runtime member is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _import_parts(module: str) -> set[str]:
    normalized = module.replace("-", "_").lower()
    return {part for part in normalized.split(".") if part}


def _validate_import(module: str, *, relative_path: str) -> None:
    parts = _import_parts(module)
    forbidden = sorted(parts & _FORBIDDEN_IMPORT_PARTS)
    if forbidden:
        raise Phase2RuntimeClosureError(
            f"forbidden training/dataset/legacy import in {relative_path}: {module}"
        )
    if module.startswith("cvsrffi."):
        if module not in _ALLOWED_INTERNAL_IMPORTS:
            raise Phase2RuntimeClosureError(
                f"internal import is outside the runtime closure in {relative_path}: {module}"
            )
        return
    if module not in _ALLOWED_EXTERNAL_IMPORTS:
        raise Phase2RuntimeClosureError(
            f"unreviewed external import in {relative_path}: {module}"
        )


def _audit_imports(payload: bytes, *, relative_path: str) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Phase2RuntimeClosureError(
            f"runtime Python source is not UTF-8: {relative_path}"
        ) from exc
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        raise Phase2RuntimeClosureError(
            f"runtime Python source does not parse: {relative_path}"
        ) from exc

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise Phase2RuntimeClosureError(
                    f"relative import is forbidden in runtime closure: {relative_path}"
                )
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
            }:
                raise Phase2RuntimeClosureError(
                    f"dynamic code/import is forbidden in runtime closure: {relative_path}"
                )
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "import_module"
                and isinstance(target.value, ast.Name)
                and target.value.id == "importlib"
            ):
                raise Phase2RuntimeClosureError(
                    f"dynamic importlib import is forbidden in runtime closure: {relative_path}"
                )
    for module in sorted(imports):
        _validate_import(module, relative_path=relative_path)
    expected = _EXPECTED_IMPORTS_BY_MEMBER.get(relative_path)
    if expected is None:
        raise Phase2RuntimeClosureError(
            f"runtime member has no reviewed import closure: {relative_path}"
        )
    if imports != expected:
        raise Phase2RuntimeClosureError(
            f"runtime member import allowlist mismatch in {relative_path}: "
            f"missing={sorted(expected - imports)}, extra={sorted(imports - expected)}"
        )
    return sorted(imports)


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write while creating runtime closure member: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, mode)


def _expected_output_files() -> set[str]:
    return {
        *(f"runtime/{value}" for value in RUNTIME_MEMBER_ALLOWLIST),
        RUNTIME_MANIFEST_NAME,
    }


def _expected_output_dirs() -> set[str]:
    return {"runtime", "runtime/cvsrffi", "runtime/scripts"}


def _audit_output_tree(output_root: Path) -> None:
    if output_root.is_symlink() or not output_root.is_dir():
        raise Phase2RuntimeClosureError(
            f"runtime closure root must be a regular directory: {output_root}"
        )
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in output_root.rglob("*"):
        relative = path.relative_to(output_root).as_posix()
        if path.is_symlink():
            raise Phase2RuntimeClosureError(f"runtime closure contains symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_dirs.add(relative)
        else:
            raise Phase2RuntimeClosureError(
                f"runtime closure contains non-file member: {relative}"
            )
    expected_files = _expected_output_files()
    expected_dirs = _expected_output_dirs()
    if actual_files != expected_files:
        raise Phase2RuntimeClosureError(
            "runtime closure file allowlist mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    if actual_dirs != expected_dirs:
        raise Phase2RuntimeClosureError(
            "runtime closure directory allowlist mismatch: "
            f"missing={sorted(expected_dirs - actual_dirs)}, "
            f"extra={sorted(actual_dirs - expected_dirs)}"
        )


def verify_phase2_runtime_closure(output_root: str | Path) -> dict[str, Any]:
    """Verify exact members, import closure, permissions, and all digests."""

    root = Path(output_root)
    _audit_output_tree(root)
    manifest_path = root / RUNTIME_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise Phase2RuntimeClosureError("runtime closure manifest schema keys drift")
    if manifest.get("schema") != RUNTIME_CLOSURE_SCHEMA:
        raise Phase2RuntimeClosureError("runtime closure manifest schema drift")
    if manifest.get("runtime_mount_path") != RUNTIME_MOUNT_PATH:
        raise Phase2RuntimeClosureError("runtime closure mount path drift")
    if manifest.get("python_path") != RUNTIME_MOUNT_PATH:
        raise Phase2RuntimeClosureError("runtime closure PYTHONPATH drift")
    if manifest.get("entrypoint") != RUNTIME_ENTRYPOINT:
        raise Phase2RuntimeClosureError("runtime closure entrypoint drift")
    if tuple(manifest.get("member_allowlist", [])) != RUNTIME_MEMBER_ALLOWLIST:
        raise Phase2RuntimeClosureError("runtime closure member allowlist drift")
    if manifest.get("member_count") != len(RUNTIME_MEMBER_ALLOWLIST):
        raise Phase2RuntimeClosureError("runtime closure member count drift")
    if manifest.get("copy_policy") != "O_EXCL_NO_OVERWRITE_READ_ONLY":
        raise Phase2RuntimeClosureError("runtime closure copy policy drift")
    if manifest.get("import_closure_policy") != "EXACT_NO_TRAIN_DATASET_LEGACY":
        raise Phase2RuntimeClosureError("runtime closure import policy drift")

    raw_members = manifest.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != len(RUNTIME_MEMBER_ALLOWLIST):
        raise Phase2RuntimeClosureError("runtime closure manifest members drift")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != _MEMBER_KEYS:
            raise Phase2RuntimeClosureError("runtime closure member schema drift")
        relative = str(raw["relative_path"])
        if relative in by_path:
            raise Phase2RuntimeClosureError(f"duplicate runtime closure member: {relative}")
        by_path[relative] = raw
    if tuple(by_path) != RUNTIME_MEMBER_ALLOWLIST:
        raise Phase2RuntimeClosureError("runtime closure manifest member order drift")

    verified_entries: list[dict[str, Any]] = []
    runtime_root = root / "runtime"
    for relative in RUNTIME_MEMBER_ALLOWLIST:
        payload = _read_regular_source(runtime_root, relative)
        path = runtime_root.joinpath(*PurePosixPath(relative).parts)
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise Phase2RuntimeClosureError(f"runtime closure member is writable: {relative}")
        imports = _audit_imports(payload, relative_path=relative)
        entry = {
            "relative_path": relative,
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
            "imports": imports,
        }
        if dict(by_path[relative]) != entry:
            raise Phase2RuntimeClosureError(f"runtime closure member digest drift: {relative}")
        verified_entries.append(entry)
    root_sha = _root_sha256(verified_entries)
    if manifest.get("root_sha256") != root_sha:
        raise Phase2RuntimeClosureError("runtime closure root SHA256 drift")
    return {
        "schema": RUNTIME_CLOSURE_SCHEMA,
        "manifest": str(manifest_path),
        "runtime_root": str(runtime_root),
        "runtime_mount_path": RUNTIME_MOUNT_PATH,
        "entrypoint": RUNTIME_ENTRYPOINT,
        "member_count": len(verified_entries),
        "root_sha256": root_sha,
        "verified": True,
    }


def build_phase2_runtime_closure(
    source_code_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Copy only the exact reviewed predictor runtime, without overwrite."""

    source_root = Path(source_code_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise Phase2RuntimeClosureError(
            f"source code root must be a regular directory: {source_root}"
        )

    # Read and audit every source before creating output, so a rejected import
    # cannot leave a misleading partial closure behind.
    source_payloads: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    for relative in RUNTIME_MEMBER_ALLOWLIST:
        payload = _read_regular_source(source_root, relative)
        imports = _audit_imports(payload, relative_path=relative)
        source_payloads[relative] = payload
        entries.append(
            {
                "relative_path": relative,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "imports": imports,
            }
        )

    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime closure: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    runtime_root = destination / "runtime"
    (runtime_root / "cvsrffi").mkdir(parents=True, exist_ok=False)
    (runtime_root / "scripts").mkdir(parents=True, exist_ok=False)
    for relative in RUNTIME_MEMBER_ALLOWLIST:
        target = runtime_root.joinpath(*PurePosixPath(relative).parts)
        _write_exclusive(target, source_payloads[relative])

    manifest = {
        "schema": RUNTIME_CLOSURE_SCHEMA,
        "runtime_mount_path": RUNTIME_MOUNT_PATH,
        "python_path": RUNTIME_MOUNT_PATH,
        "entrypoint": RUNTIME_ENTRYPOINT,
        "member_allowlist": list(RUNTIME_MEMBER_ALLOWLIST),
        "member_count": len(entries),
        "members": entries,
        "root_sha256": _root_sha256(entries),
        "copy_policy": "O_EXCL_NO_OVERWRITE_READ_ONLY",
        "import_closure_policy": "EXACT_NO_TRAIN_DATASET_LEGACY",
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(destination / RUNTIME_MANIFEST_NAME, manifest_payload)
    return verify_phase2_runtime_closure(destination)


__all__ = [
    "Phase2RuntimeClosureError",
    "RUNTIME_CLOSURE_SCHEMA",
    "RUNTIME_ENTRYPOINT",
    "RUNTIME_MANIFEST_NAME",
    "RUNTIME_MEMBER_ALLOWLIST",
    "RUNTIME_MOUNT_PATH",
    "build_phase2_runtime_closure",
    "verify_phase2_runtime_closure",
]
