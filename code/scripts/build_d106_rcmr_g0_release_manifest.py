#!/usr/bin/env python3
"""Build a sealed source-closure manifest for the D106 RCMR-G0 clean runner.

This is a local preregistration tool, not a production runner.  It deliberately
does not enumerate the whole ``cvsrffi`` package: it begins with the five
direct modules declared by the clean child, follows only statically visible
``cvsrffi`` imports with an AST fixed point, and hashes precisely that closure.
The G0 expected-code map is obtained only through its reviewed public packaging
interface; a private helper is never an acceptable fallback.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


ROOT_PACKAGE = "cvsrffi"
RUNNER_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.production_runner.v2"
RELEASE_MANIFEST_SCHEMA = RUNNER_SCHEMA + ".release_manifest.v1"
RUNNER_RELATIVE_PATH = "scripts/run_d106_rcmr_g0_production.py"
CHILD_RELATIVE_PATH = "scripts/d106_rcmr_g0_clean_child.py"
PACKAGE_INIT_RELATIVE_PATH = "cvsrffi/__init__.py"
G0_MODULE = "cvsrffi.stage2_d106_rcmr_g0"
G0_PUBLIC_CODE_INTERFACE = "get_d106_rcmr_g0_release_expected_code_sha256"
SOURCE_BYTES_CAP = 4 * 1024 * 1024
MANIFEST_BYTES_CAP = 256 * 1024
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MODULE_NAME = re.compile(r"^cvsrffi(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


class ReleaseManifestBuildError(RuntimeError):
    """Raised when a local D106 release closure cannot be proven."""


@dataclass(frozen=True, slots=True)
class ReleaseManifestBuildResult:
    output_path: Path
    manifest_sha256: str
    manifest_bytes: bytes
    production_module_closure: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ReleaseManifestBuildError(f"{name} must be a lowercase SHA256")
    return value


def _require_commit(value: Any) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        raise ReleaseManifestBuildError("expected release commit must be a lowercase SHA1")
    return value


def _relative_parts(value: str, *, name: str) -> tuple[str, ...]:
    if type(value) is not str or not value or "\\" in value:
        raise ReleaseManifestBuildError(f"{name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ReleaseManifestBuildError(f"{name} escapes the code root")
    return path.parts


def _resolve_code_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise ReleaseManifestBuildError("code root must be absolute")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise ReleaseManifestBuildError("code root does not exist") from error
    if not root.is_dir():
        raise ReleaseManifestBuildError("code root must be a directory")
    package = root / ROOT_PACKAGE
    if not package.is_dir() or package.is_symlink():
        raise ReleaseManifestBuildError("code root lacks a real cvsrffi package directory")
    return root


def _assert_beneath(root: Path, path: Path, *, name: str) -> Path:
    """Resolve a code file while rejecting path traversal and symlinks."""

    try:
        logical = path.relative_to(root)
    except ValueError as error:
        raise ReleaseManifestBuildError(f"{name} escapes the code root") from error
    cursor = root
    for part in logical.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReleaseManifestBuildError(f"{name} uses a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReleaseManifestBuildError(f"{name} is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ReleaseManifestBuildError(f"{name} escapes the code root") from error
    return resolved


def _read_regular(path: Path, *, root: Path, name: str) -> bytes:
    resolved = _assert_beneath(root, path, name=name)
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise ReleaseManifestBuildError(f"cannot stat {name}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseManifestBuildError(f"{name} must be a regular file")
    if metadata.st_size > SOURCE_BYTES_CAP:
        raise ReleaseManifestBuildError(f"{name} exceeds source byte cap")
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ReleaseManifestBuildError(f"cannot read {name}") from error
    after = resolved.stat()
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReleaseManifestBuildError(f"{name} changed during read")
    if len(payload) != metadata.st_size:
        raise ReleaseManifestBuildError(f"{name} size changed during read")
    return payload


def _read_relative_source(root: Path, relative_path: str) -> bytes:
    parts = _relative_parts(relative_path, name="release source path")
    return _read_regular(root.joinpath(*parts), root=root, name=f"release source {relative_path}")


def _module_file(root: Path, module_name: str, *, required: bool) -> Path | None:
    if _MODULE_NAME.fullmatch(module_name) is None:
        raise ReleaseManifestBuildError(f"invalid cvsrffi module name: {module_name!r}")
    pieces = module_name.split(".")[1:]
    candidate = root / ROOT_PACKAGE / Path(*pieces).with_suffix(".py")
    if not candidate.exists() and not candidate.is_symlink():
        if required:
            raise ReleaseManifestBuildError(f"missing imported cvsrffi module: {module_name}")
        return None
    return _assert_beneath(root, candidate, name=f"cvsrffi module {module_name}")


def _module_relative_path(module_name: str) -> str:
    if _MODULE_NAME.fullmatch(module_name) is None:
        raise ReleaseManifestBuildError(f"invalid cvsrffi module name: {module_name!r}")
    return module_name.replace(".", "/") + ".py"


def _is_package_module(root: Path, module_name: str) -> bool:
    if module_name == ROOT_PACKAGE:
        return True
    if _MODULE_NAME.fullmatch(module_name) is None:
        return False
    pieces = module_name.split(".")[1:]
    init = root / ROOT_PACKAGE / Path(*pieces) / "__init__.py"
    if not init.exists() and not init.is_symlink():
        return False
    _assert_beneath(root, init, name=f"cvsrffi package {module_name}")
    return True


def _base_package_for_module(root: Path, module_name: str) -> str:
    return module_name if _is_package_module(root, module_name) else module_name.rpartition(".")[0]


def _resolve_from_base(root: Path, module_name: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = _base_package_for_module(root, module_name)
    relative = "." * node.level + (node.module or "")
    try:
        resolved = importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError) as error:
        raise ReleaseManifestBuildError(
            f"relative import escapes cvsrffi package in {module_name}"
        ) from error
    if not (resolved == ROOT_PACKAGE or resolved.startswith(ROOT_PACKAGE + ".")):
        raise ReleaseManifestBuildError(
            f"relative import escapes cvsrffi package in {module_name}"
        )
    return resolved


def _candidate_submodule(root: Path, base: str, name: str) -> str | None:
    if name == "*" or not _is_package_module(root, base):
        return None
    candidate = f"{base}.{name}"
    if _MODULE_NAME.fullmatch(candidate) is None:
        return None
    return candidate if _module_file(root, candidate, required=False) is not None else None


def _static_cvsrffi_imports(root: Path, module_name: str, source: bytes) -> set[str]:
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text, filename=_module_relative_path(module_name))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ReleaseManifestBuildError(f"cannot parse {module_name} with AST") from error
    discovered: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == ROOT_PACKAGE:
                    continue
                if name.startswith(ROOT_PACKAGE + "."):
                    if _MODULE_NAME.fullmatch(name) is None:
                        raise ReleaseManifestBuildError(
                            f"invalid absolute cvsrffi import in {module_name}: {name!r}"
                        )
                    discovered.add(name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(root, module_name, node)
            if base is None:
                continue
            if base == ROOT_PACKAGE:
                pass
            elif base.startswith(ROOT_PACKAGE + "."):
                if _MODULE_NAME.fullmatch(base) is None:
                    raise ReleaseManifestBuildError(
                        f"invalid cvsrffi import in {module_name}: {base!r}"
                    )
                discovered.add(base)
            else:
                continue
            for alias in node.names:
                candidate = _candidate_submodule(root, base, alias.name)
                if candidate is not None:
                    discovered.add(candidate)
    return discovered


def _direct_modules_from_child(root: Path) -> tuple[str, ...]:
    child_source = _read_relative_source(root, CHILD_RELATIVE_PATH)
    try:
        tree = ast.parse(child_source.decode("utf-8"), filename=CHILD_RELATIVE_PATH)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ReleaseManifestBuildError("cannot parse clean child direct-module declaration") from error
    values: list[Any] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == "_REQUIRED_DIRECT_MODULES"
            for target in targets
        ):
            continue
        try:
            values.append(ast.literal_eval(value))
        except (ValueError, TypeError) as error:
            raise ReleaseManifestBuildError("clean child direct-module declaration is not literal") from error
    if len(values) != 1 or type(values[0]) not in {set, frozenset, tuple, list}:
        raise ReleaseManifestBuildError("clean child direct-module declaration is ambiguous")
    modules = tuple(sorted(values[0]))
    if (
        len(modules) != 5
        or len(set(modules)) != 5
        or any(type(module) is not str or _MODULE_NAME.fullmatch(module) is None for module in modules)
    ):
        raise ReleaseManifestBuildError("clean child must declare exactly five direct G0 modules")
    return modules


def discover_production_module_closure(code_root: str | Path) -> tuple[str, ...]:
    """Return the AST-derived, fixed-point project-module closure only."""

    root = _resolve_code_root(code_root)
    pending = list(_direct_modules_from_child(root))
    discovered: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in discovered:
            continue
        source_path = _module_file(root, module_name, required=True)
        assert source_path is not None
        source = _read_regular(source_path, root=root, name=f"cvsrffi module {module_name}")
        imports = _static_cvsrffi_imports(root, module_name, source)
        discovered.add(module_name)
        pending.extend(sorted(imports - discovered, reverse=True))
    return tuple(sorted(discovered))


def _run_git(code_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git failure"
        raise ReleaseManifestBuildError(f"Git checkout inspection failed: {detail}")
    return completed.stdout


def verify_clean_release_checkout(
    *, code_root: str | Path, expected_release_commit: str
) -> str:
    """Require the explicit commit and a fully clean local Git checkout."""

    root = _resolve_code_root(code_root)
    expected = _require_commit(expected_release_commit)
    observed = _run_git(root, "rev-parse", "HEAD").strip()
    if observed != expected:
        raise ReleaseManifestBuildError("local release Git HEAD mismatch")
    dirty = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ReleaseManifestBuildError("local release checkout is not clean")
    return expected


@contextmanager
def _g0_module_from_code_root(root: Path) -> Iterator[Any]:
    """Import the audited public G0 packaging interface from this exact tree."""

    saved_path = list(sys.path)
    saved_dont_write_bytecode = sys.dont_write_bytecode
    saved_modules = {
        name: module for name, module in sys.modules.items()
        if name == ROOT_PACKAGE or name.startswith(ROOT_PACKAGE + ".")
    }
    for name in tuple(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(G0_MODULE)
        expected_path = _module_file(root, G0_MODULE, required=True)
        assert expected_path is not None
        actual_file = getattr(module, "__file__", None)
        if type(actual_file) is not str:
            raise ReleaseManifestBuildError("G0 public interface module lacks a source file")
        try:
            actual_path = Path(actual_file).resolve(strict=True)
        except OSError as error:
            raise ReleaseManifestBuildError("G0 public interface source file is unavailable") from error
        if actual_path != expected_path:
            raise ReleaseManifestBuildError("G0 public interface resolved outside the release tree")
        yield module
    except ReleaseManifestBuildError:
        raise
    except Exception as error:
        raise ReleaseManifestBuildError("cannot import G0 public packaging interface") from error
    finally:
        for name in tuple(sys.modules):
            if name == ROOT_PACKAGE or name.startswith(ROOT_PACKAGE + "."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dont_write_bytecode


def _canonical_g0_expected_code_map(value: Any) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ReleaseManifestBuildError("G0 public packaging interface must return a non-empty dict")
    canonical: dict[str, str] = {}
    for name, digest in value.items():
        if type(name) is not str or not name:
            raise ReleaseManifestBuildError("G0 public expected-code map has an invalid key")
        canonical[name] = _require_sha256(digest, f"G0 expected-code {name}")
    return dict(sorted(canonical.items()))


def read_g0_public_expected_code_sha256(code_root: str | Path) -> dict[str, str]:
    """Read only the public, reviewed G0 expected-code packaging surface."""

    root = _resolve_code_root(code_root)
    with _g0_module_from_code_root(root) as module:
        exports = getattr(module, "__all__", None)
        if type(exports) not in {list, tuple} or G0_PUBLIC_CODE_INTERFACE not in exports:
            raise ReleaseManifestBuildError(
                "G0 public packaging interface unavailable; "
                f"expected public export {G0_PUBLIC_CODE_INTERFACE}()"
            )
        interface = getattr(module, G0_PUBLIC_CODE_INTERFACE, None)
        if not callable(interface) or getattr(interface, "__module__", None) != G0_MODULE:
            raise ReleaseManifestBuildError("G0 public packaging interface binding is invalid")
        source_file = inspect.getsourcefile(interface)
        expected_path = _module_file(root, G0_MODULE, required=True)
        assert expected_path is not None
        try:
            interface_path = Path(source_file or "").resolve(strict=True)
        except OSError as error:
            raise ReleaseManifestBuildError("G0 public packaging interface source is unavailable") from error
        if interface_path != expected_path:
            raise ReleaseManifestBuildError("G0 public packaging interface source drift")
        first = interface()
        second = interface()
        if first is second:
            raise ReleaseManifestBuildError("G0 public packaging interface must return a fresh dict")
        first_map = _canonical_g0_expected_code_map(first)
        second_map = _canonical_g0_expected_code_map(second)
        if first_map != second_map:
            raise ReleaseManifestBuildError("G0 public packaging interface is not deterministic")
        return first_map


def _canonical_registry(values: Sequence[str]) -> list[str]:
    if len(values) != 6 or any(type(value) is not str or not value for value in values):
        raise ReleaseManifestBuildError("registered classes must contain exactly six non-empty strings")
    ordered = sorted(values, key=lambda value: value.encode("utf-8"))
    if len(set(ordered)) != 6:
        raise ReleaseManifestBuildError("registered classes must be unique")
    return ordered


def _code_files_sha256(root: Path, modules: Sequence[str]) -> dict[str, str]:
    paths = {
        RUNNER_RELATIVE_PATH,
        CHILD_RELATIVE_PATH,
        PACKAGE_INIT_RELATIVE_PATH,
        *(_module_relative_path(module) for module in modules),
    }
    result: dict[str, str] = {}
    for relative_path in sorted(paths):
        payload = _read_relative_source(root, relative_path)
        result[relative_path] = _sha256(payload)
    return result


def _write_new_manifest(output_path: str | Path, payload: bytes) -> Path:
    target = Path(output_path)
    if not target.is_absolute():
        raise ReleaseManifestBuildError("manifest output path must be explicit and absolute")
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ReleaseManifestBuildError("manifest output parent must already exist")
    if target.exists() or target.is_symlink():
        raise ReleaseManifestBuildError("manifest output already exists; refusing to overwrite")
    try:
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ReleaseManifestBuildError("manifest output already exists; refusing to overwrite") from error
    except OSError as error:
        raise ReleaseManifestBuildError("cannot reserve new manifest output") from error
    try:
        total = 0
        while total < len(payload):
            written = os.write(descriptor, payload[total:])
            if written <= 0:
                raise ReleaseManifestBuildError("cannot write manifest output")
            total += written
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            target.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    return target


def build_release_manifest(
    *,
    code_root: str | Path,
    expected_release_commit: str,
    registered_classes: Sequence[str],
    output_path: str | Path,
    expected_d105_lock_authority_sha256: str | None = None,
) -> ReleaseManifestBuildResult:
    """Build one new canonical manifest from a clean, explicitly pinned tree."""

    root = _resolve_code_root(code_root)
    expected_commit = verify_clean_release_checkout(
        code_root=root, expected_release_commit=expected_release_commit
    )
    registry = _canonical_registry(registered_classes)
    authority = expected_d105_lock_authority_sha256
    if authority is not None:
        authority = _require_sha256(authority, "expected D105 lock authority")
    closure = discover_production_module_closure(root)
    expected_code = read_g0_public_expected_code_sha256(root)
    source_map = _code_files_sha256(root, closure)
    manifest = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "release_commit": expected_commit,
        "registered_classes": registry,
        "expected_d105_lock_authority_sha256": authority,
        "runner_path": RUNNER_RELATIVE_PATH,
        "child_entry_path": CHILD_RELATIVE_PATH,
        "production_module_closure": list(closure),
        "code_files_sha256": source_map,
        "g0_expected_code_sha256": expected_code,
    }
    payload = _canonical_bytes(manifest)
    if len(payload) > MANIFEST_BYTES_CAP:
        raise ReleaseManifestBuildError("release manifest exceeds byte cap")
    # Recheck after every source and public-interface read.  This makes a
    # concurrent source edit fail closed rather than minting a mixed snapshot.
    verify_clean_release_checkout(code_root=root, expected_release_commit=expected_commit)
    output = _write_new_manifest(output_path, payload)
    return ReleaseManifestBuildResult(
        output_path=output,
        manifest_sha256=_sha256(payload),
        manifest_bytes=payload,
        production_module_closure=closure,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, help="absolute local code directory")
    parser.add_argument(
        "--expected-release-commit", required=True, help="explicit 40-character Git HEAD"
    )
    parser.add_argument(
        "--registered-class",
        action="append",
        dest="registered_classes",
        default=[],
        help="repeat exactly six times; values are canonicalized bytewise",
    )
    parser.add_argument(
        "--expected-d105-lock-authority-sha256",
        default=None,
        help="optional, externally preregistered D105 authority digest",
    )
    parser.add_argument("--output", required=True, help="new absolute manifest JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_release_manifest(
            code_root=arguments.code_root,
            expected_release_commit=arguments.expected_release_commit,
            registered_classes=arguments.registered_classes,
            output_path=arguments.output,
            expected_d105_lock_authority_sha256=arguments.expected_d105_lock_authority_sha256,
        )
    except ReleaseManifestBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"manifest_path={result.output_path}")
    print(f"manifest_sha256={result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
