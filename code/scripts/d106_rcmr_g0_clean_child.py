"""Byte-loaded clean-child entry for D106 RCMR-G0 production mechanics.

This file is executed only by the parent's ``python -I`` bootstrap after its
own bytes have been checked from an inherited O_NOFOLLOW release-root FD.
It accepts no rows, locks, request object, or executable callback from callers.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import linecache
import os
from pathlib import PurePosixPath
import re
import stat
import sys
from types import ModuleType
from typing import Any


RUNNER_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.production_runner.v2"
RELEASE_MANIFEST_SCHEMA = RUNNER_SCHEMA + ".release_manifest.v1"
EXECUTE_REQUEST_SCHEMA = RUNNER_SCHEMA + ".clean_execute_request.v1"
VERIFY_REQUEST_SCHEMA = RUNNER_SCHEMA + ".clean_verify_request.v1"
VERIFY_RECEIPT_SCHEMA = RUNNER_SCHEMA + ".clean_verify_receipt.v1"
CHILD_RELATIVE_PATH = "scripts/d106_rcmr_g0_clean_child.py"
RUNNER_RELATIVE_PATH = "scripts/run_d106_rcmr_g0_production.py"
SOURCE_CAP = 4 * 1024 * 1024
RESULT_CAP = 2 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_MODULE_NAME = re.compile(r"^cvsrffi(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_REQUIRED_DIRECT_MODULES = {
    "cvsrffi.stage2_d106_rcmr_g0",
    "cvsrffi.stage2_d106_phase1_tap",
    "cvsrffi.stage2_d106_train_only_predecessor_lock",
    "cvsrffi.stage2_d106_rcmr_2v_qknn",
    "cvsrffi.stage2_zid_student_t_qknn",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{name} SHA256 closure")
    return value


def _relative_parts(value: Any) -> tuple[str, ...]:
    if type(value) is not str or not value or "\\" in value:
        raise RuntimeError("release path type")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RuntimeError("release path escape")
    return path.parts


def _read_fd(fd: int, cap: int, name: str) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{name} regular-file closure")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1 << 20, cap + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise RuntimeError(f"{name} cap")
        chunks.append(chunk)
    after = os.fstat(fd)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"{name} changed")
    return b"".join(chunks)


def _read_root_source(root_fd: int, relative_path: str) -> bytes:
    parts = _relative_parts(relative_path)
    directory_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        return _read_fd(file_fd, SOURCE_CAP, f"release source {relative_path}")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _manifest_and_sources() -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    raw = RELEASE_MANIFEST_BYTES
    expected = _require_sha(EXPECTED_RELEASE_MANIFEST_SHA256, "external manifest")
    if type(raw) is not bytes or _sha256(raw) != expected:
        raise RuntimeError("external release manifest SHA")
    document = json.loads(raw.decode("utf-8"))
    expected_keys = {
        "schema", "release_commit", "registered_classes", "expected_d105_lock_authority_sha256",
        "runner_path", "child_entry_path", "production_module_closure",
        "code_files_sha256", "g0_expected_code_sha256",
    }
    if (
        type(document) is not dict
        or raw != _canonical_bytes(document)
        or set(document) != expected_keys
        or document.get("schema") != RELEASE_MANIFEST_SCHEMA
        or type(document.get("release_commit")) is not str
        or _HEX40.fullmatch(document["release_commit"]) is None
        or document.get("runner_path") != RUNNER_RELATIVE_PATH
        or document.get("child_entry_path") != CHILD_RELATIVE_PATH
    ):
        raise RuntimeError("release manifest closure")
    classes = document.get("registered_classes")
    if (
        type(classes) is not list or len(classes) != 6
        or any(type(value) is not str or not value for value in classes)
        or len(set(classes)) != 6
        or classes != sorted(classes, key=lambda value: value.encode("utf-8"))
    ):
        raise RuntimeError("release class registry")
    if document["expected_d105_lock_authority_sha256"] is not None:
        _require_sha(document["expected_d105_lock_authority_sha256"], "D105 authority")
    modules = document.get("production_module_closure")
    if (
        type(modules) is not list or not modules or modules != sorted(modules)
        or len(set(modules)) != len(modules)
        or any(type(value) is not str or _MODULE_NAME.fullmatch(value) is None for value in modules)
        or not _REQUIRED_DIRECT_MODULES.issubset(set(modules))
    ):
        raise RuntimeError("release module closure")
    code_map = document.get("code_files_sha256")
    expected_paths = {
        RUNNER_RELATIVE_PATH, CHILD_RELATIVE_PATH, "cvsrffi/__init__.py",
        *{module.replace(".", "/") + ".py" for module in modules},
    }
    if type(code_map) is not dict or set(code_map) != expected_paths:
        raise RuntimeError("release source path closure")
    sources: dict[str, tuple[str, bytes]] = {}
    for relative_path, digest in sorted(code_map.items()):
        _relative_parts(relative_path)
        payload = _read_root_source(CODE_ROOT_FD, relative_path)
        if _sha256(payload) != _require_sha(digest, relative_path):
            raise RuntimeError(f"release source SHA mismatch: {relative_path}")
        if relative_path == "cvsrffi/__init__.py":
            module_name = "cvsrffi"
        elif relative_path.startswith("cvsrffi/"):
            module_name = relative_path[:-3].replace("/", ".")
        else:
            continue
        sources[module_name] = (relative_path, payload)
    expected_module_names = {"cvsrffi", *modules}
    if set(sources) != expected_module_names:
        raise RuntimeError("release import source closure")
    g0_map = document.get("g0_expected_code_sha256")
    if type(g0_map) is not dict or not g0_map:
        raise RuntimeError("G0 expected code map")
    for key, digest in g0_map.items():
        if type(key) is not str or not key:
            raise RuntimeError("G0 expected code key")
        _require_sha(digest, f"G0 expected code {key}")
    return document, sources


class _VerifiedProjectFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Executes only pre-read, manifest-hashed project source bytes."""

    def __init__(self, sources: dict[str, tuple[str, bytes]]) -> None:
        self._sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cvsrffi" or fullname.startswith("cvsrffi."):
            if fullname not in self._sources:
                raise ImportError(f"unlisted project module: {fullname}")
            relative_path, _payload = self._sources[fullname]
            return importlib.util.spec_from_loader(
                fullname, self, is_package=relative_path.endswith("/__init__.py")
            )
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        relative_path, payload = self._sources[module.__name__]
        origin = os.path.join(CODE_ROOT_PATH, *relative_path.split("/"))
        source = payload.decode("utf-8")
        linecache.cache[origin] = (len(payload), None, source.splitlines(True), origin)
        module.__file__ = origin
        module.__loader__ = self
        if relative_path.endswith("/__init__.py"):
            module.__path__ = [os.path.dirname(origin)]
        exec(compile(payload, origin, "exec"), module.__dict__)


@contextmanager
def _verified_project_import_lifecycle(sources: dict[str, tuple[str, bytes]]):
    """Keep the rejecting finder installed for the complete G0 operation."""

    finder = _VerifiedProjectFinder(sources)
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module("cvsrffi.stage2_d106_rcmr_g0")
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError as error:
            raise RuntimeError("verified project finder lifecycle drift") from error


def _parse_request() -> dict[str, Any]:
    if type(REQUEST_BYTES) is not bytes:
        raise RuntimeError("child request type")
    document = json.loads(REQUEST_BYTES.decode("utf-8"))
    if type(document) is not dict or REQUEST_BYTES != _canonical_bytes(document):
        raise RuntimeError("child request canonical closure")
    return document


def _execute(manifest: dict[str, Any], sources: dict[str, tuple[str, bytes]]) -> None:
    request = _parse_request()
    if set(request) != {"schema", "archive_path", "archive_sha256", "receipt_path", "receipt_sha256"}:
        raise RuntimeError("execute request key closure")
    if request.get("schema") != EXECUTE_REQUEST_SCHEMA:
        raise RuntimeError("execute request schema")
    for name in ("archive_path", "receipt_path"):
        if type(request.get(name)) is not str or not request[name]:
            raise RuntimeError(f"execute {name}")
    archive_sha = _require_sha(request.get("archive_sha256"), "archive")
    receipt_sha = _require_sha(request.get("receipt_sha256"), "receipt")
    with _verified_project_import_lifecycle(sources) as g0:
        production_request = g0.D106RCMRG0ProductionRequest(
            registered_classes=tuple(manifest["registered_classes"]),
            expected_release_commit=manifest["release_commit"],
            expected_code_sha256=tuple(sorted(manifest["g0_expected_code_sha256"].items())),
            expected_d105_lock_authority_sha256=manifest["expected_d105_lock_authority_sha256"],
        )
        result = g0.run_d106_rcmr_g0_from_formal_tap(
            request["archive_path"], request["receipt_path"],
            expected_archive_sha256=archive_sha, expected_receipt_sha256=receipt_sha,
            request=production_request,
        )
    if type(result) is not bytes or len(result) > RESULT_CAP:
        raise RuntimeError("execute result cap/type")
    sys.stdout.buffer.write(result)


def _verify(manifest: dict[str, Any], sources: dict[str, tuple[str, bytes]]) -> None:
    request = _parse_request()
    if set(request) != {"schema", "result_base64", "expected_result_sha256"}:
        raise RuntimeError("verify request key closure")
    if request.get("schema") != VERIFY_REQUEST_SCHEMA:
        raise RuntimeError("verify request schema")
    expected = _require_sha(request.get("expected_result_sha256"), "verify result")
    if type(request.get("result_base64")) is not str:
        raise RuntimeError("verify result payload")
    try:
        result = base64.b64decode(request["result_base64"], validate=True)
    except Exception as error:
        raise RuntimeError("verify result base64") from error
    if len(result) > RESULT_CAP or _sha256(result) != expected:
        raise RuntimeError("verify result external SHA")
    with _verified_project_import_lifecycle(sources) as g0:
        verified = g0.verify_d106_rcmr_g0_production_result_bytes(
            result, expected_sha256=expected
        )
        if verified is not result:
            raise RuntimeError("verify result identity")
    receipt = _canonical_bytes(
        {
            "schema": VERIFY_RECEIPT_SCHEMA,
            "status": "VERIFIED",
            "result_sha256": expected,
            "release_manifest_sha256": EXPECTED_RELEASE_MANIFEST_SHA256,
        }
    )
    sys.stdout.buffer.write(receipt)


def _main() -> None:
    if CHILD_MODE == "execute":
        manifest, sources = _manifest_and_sources()
        _execute(manifest, sources)
    elif CHILD_MODE == "verify":
        manifest, sources = _manifest_and_sources()
        _verify(manifest, sources)
    else:
        raise RuntimeError("unsupported clean child mode")


if __name__ == "__main__":
    _main()
