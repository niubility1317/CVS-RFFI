#!/usr/bin/env python3
"""POSIX-only, clean-child publisher for D106 RCMR-G0 production bytes.

This parent intentionally never imports ``cvsrffi``.  The externally pinned
release manifest is the trust root; it names every production source file.
Fresh ``python -I`` children execute and independently verify the result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import selectors
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
RUNNER_RELATIVE_PATH = "scripts/run_d106_rcmr_g0_production.py"
CHILD_RELATIVE_PATH = "scripts/d106_rcmr_g0_clean_child.py"
RUNNER_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.production_runner.v2"
RELEASE_MANIFEST_SCHEMA = RUNNER_SCHEMA + ".release_manifest.v1"
EXECUTE_REQUEST_SCHEMA = RUNNER_SCHEMA + ".clean_execute_request.v1"
VERIFY_REQUEST_SCHEMA = RUNNER_SCHEMA + ".clean_verify_request.v1"
VERIFY_RECEIPT_SCHEMA = RUNNER_SCHEMA + ".clean_verify_receipt.v1"
COMPLETION_SCHEMA = RUNNER_SCHEMA + ".completion.v1"
RESULT_NAME = "d106_rcmr_g0_production_result.json"
MANIFEST_NAME = "d106_rcmr_g0_production_execution_manifest.json"
COMPLETION_NAME = "COMPLETED.json"
PRODUCTION_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.production.v3"
PRODUCTION_EXECUTED_STATUS = "REAL_G0_EXECUTED_NON_FORMAL_TRAIN_ONLY_MECHANICAL"
RESULT_BYTES_CAP = 2 * 1024 * 1024
MANIFEST_BYTES_CAP = 2 * 1024 * 1024
RELEASE_MANIFEST_BYTES_CAP = 256 * 1024
CHILD_SOURCE_BYTES_CAP = 4 * 1024 * 1024
CHILD_STDERR_BYTES_CAP = 64 * 1024
CHILD_VERIFY_BYTES_CAP = 16 * 1024
CHILD_TIMEOUT_SECONDS = 300
COMPLETION_BYTES_CAP = 32 * 1024
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


class D106RCMRG0ProductionRunnerError(RuntimeError):
    """Raised when the production release closure cannot be proven."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise D106RCMRG0ProductionRunnerError(f"{name} must be a lowercase SHA256")
    return value


def _require_posix_production() -> None:
    """Hard stop before children, paths, or output state on non-POSIX hosts."""

    if os.name != "posix":
        raise D106RCMRG0ProductionRunnerError(
            "PRODUCTION_POSIX_ONLY_WINDOWS_TEST_ONLY_NO_OUTPUT"
        )


def _relative_parts(value: Any, name: str) -> tuple[str, ...]:
    if type(value) is not str or not value or "\\" in value:
        raise D106RCMRG0ProductionRunnerError(f"{name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise D106RCMRG0ProductionRunnerError(f"{name} escapes the release root")
    if path.as_posix() != value:
        raise D106RCMRG0ProductionRunnerError(f"{name} is not normalized")
    return path.parts


def _read_open_regular(fd: int, *, cap: int, name: str) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise D106RCMRG0ProductionRunnerError(f"{name} must be a regular file")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1 << 20, cap + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise D106RCMRG0ProductionRunnerError(f"{name} resource cap exceeded")
        chunks.append(chunk)
    after = os.fstat(fd)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise D106RCMRG0ProductionRunnerError(f"{name} changed during read")
    return b"".join(chunks)


def _open_absolute_dirfd(path: str | Path, name: str) -> int:
    _require_posix_production()
    source = Path(path)
    if not source.is_absolute():
        raise D106RCMRG0ProductionRunnerError(f"{name} must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        for part in source.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise D106RCMRG0ProductionRunnerError(f"{name} is not a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_at(root_fd: int, relative_path: str, *, cap: int, name: str) -> bytes:
    parts = _relative_parts(relative_path, name)
    cursor = os.dup(root_fd)
    file_fd: int | None = None
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cursor
            )
            os.close(cursor)
            cursor = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cursor)
        return _read_open_regular(file_fd, cap=cap, name=name)
    except OSError as error:
        raise D106RCMRG0ProductionRunnerError(f"cannot read {name}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(cursor)


def _open_regular_absolute(path: str | Path, *, cap: int, name: str) -> tuple[int, bytes]:
    source = Path(path)
    if not source.is_absolute() or source.name in {"", ".", ".."}:
        raise D106RCMRG0ProductionRunnerError(f"{name} must be an absolute file path")
    parent_fd = _open_absolute_dirfd(source.parent, f"{name} parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        payload = _read_open_regular(descriptor, cap=cap, name=name)
        return descriptor, payload
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise D106RCMRG0ProductionRunnerError(f"cannot read {name}") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


def _parse_release_manifest_bytes(
    payload: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, "externally preregistered release manifest")
    if _sha256(payload) != expected:
        raise D106RCMRG0ProductionRunnerError("release manifest external SHA256 mismatch")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0ProductionRunnerError("release manifest must be canonical UTF-8 JSON") from error
    expected_keys = {
        "schema",
        "release_commit",
        "registered_classes",
        "expected_d105_lock_authority_sha256",
        "runner_path",
        "child_entry_path",
        "production_module_closure",
        "code_files_sha256",
        "g0_expected_code_sha256",
    }
    if (
        type(document) is not dict
        or payload != _canonical_bytes(document)
        or set(document) != expected_keys
        or document.get("schema") != RELEASE_MANIFEST_SCHEMA
        or type(document.get("release_commit")) is not str
        or _HEX40.fullmatch(document["release_commit"]) is None
        or document.get("runner_path") != RUNNER_RELATIVE_PATH
        or document.get("child_entry_path") != CHILD_RELATIVE_PATH
    ):
        raise D106RCMRG0ProductionRunnerError("release manifest semantic closure drift")
    classes = document.get("registered_classes")
    if (
        type(classes) is not list
        or len(classes) != 6
        or any(type(value) is not str or not value for value in classes)
        or len(set(classes)) != 6
        or classes != sorted(classes, key=lambda value: value.encode("utf-8"))
    ):
        raise D106RCMRG0ProductionRunnerError("release manifest class registry drift")
    authority = document.get("expected_d105_lock_authority_sha256")
    if authority is not None:
        _require_sha256(authority, "release manifest D105 authority")
    modules = document.get("production_module_closure")
    if (
        type(modules) is not list
        or not modules
        or any(type(value) is not str or _MODULE_NAME.fullmatch(value) is None for value in modules)
        or modules != sorted(modules)
        or len(set(modules)) != len(modules)
        or not _REQUIRED_DIRECT_MODULES.issubset(set(modules))
    ):
        raise D106RCMRG0ProductionRunnerError("release manifest module closure drift")
    code_map = document.get("code_files_sha256")
    if type(code_map) is not dict or not 3 <= len(code_map) <= 128:
        raise D106RCMRG0ProductionRunnerError("release manifest code map type/size drift")
    expected_paths = {
        RUNNER_RELATIVE_PATH,
        CHILD_RELATIVE_PATH,
        "cvsrffi/__init__.py",
        *{value.replace(".", "/") + ".py" for value in modules},
    }
    if set(code_map) != expected_paths:
        raise D106RCMRG0ProductionRunnerError("release manifest code path closure drift")
    for relative_path, digest in code_map.items():
        _relative_parts(relative_path, "release manifest code path")
        _require_sha256(digest, f"release source {relative_path}")
    g0_map = document.get("g0_expected_code_sha256")
    if type(g0_map) is not dict or not g0_map:
        raise D106RCMRG0ProductionRunnerError("release manifest G0 source map drift")
    for key, digest in g0_map.items():
        if type(key) is not str or not key:
            raise D106RCMRG0ProductionRunnerError("release manifest G0 source key drift")
        _require_sha256(digest, f"release G0 source {key}")
    return document


def validate_release_manifest_test_only(
    manifest_bytes: bytes, *, expected_manifest_sha256: str
) -> Mapping[str, Any]:
    """Parse an externally anchored manifest without spawning or publishing.

    This is the only supported non-POSIX inspection surface.  It never emits
    a production result, a completion marker, or any output directory.
    """

    if type(manifest_bytes) is not bytes or len(manifest_bytes) > RELEASE_MANIFEST_BYTES_CAP:
        raise D106RCMRG0ProductionRunnerError("TEST_ONLY manifest byte resource cap")
    document = _parse_release_manifest_bytes(
        manifest_bytes, expected_sha256=expected_manifest_sha256
    )
    return {
        "schema": document["schema"],
        "release_commit": document["release_commit"],
        "source_file_count": len(document["code_files_sha256"]),
        "test_only": True,
    }


def verify_d106_rcmr_g0_release_checkout_for_packaging(
    *, code_root: str | Path, expected_release_commit: str
) -> Mapping[str, str]:
    """Local pre-registration check; intentionally not required on Git archives."""

    expected = str(expected_release_commit)
    if _HEX40.fullmatch(expected) is None:
        raise D106RCMRG0ProductionRunnerError("expected release commit must be SHA1")
    root = Path(code_root).resolve()
    command = ["git", "-C", str(root), "rev-parse", "HEAD"]
    head = subprocess.run(command, capture_output=True, text=True, check=False)
    if head.returncode != 0 or head.stdout.strip() != expected:
        raise D106RCMRG0ProductionRunnerError("local release Git HEAD mismatch")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout:
        raise D106RCMRG0ProductionRunnerError("local release checkout is not clean")
    return {"release_commit": expected, "checkout": str(root)}


def _verify_optional_git_head(code_root: Path, release_commit: str) -> None:
    """Compare HEAD if available; Git absence is valid for N607 release archives."""

    git_marker = code_root.parent / ".git"
    if not git_marker.exists():
        return
    result = subprocess.run(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip() != release_commit:
        raise D106RCMRG0ProductionRunnerError("optional local Git HEAD differs from release manifest")


def _verify_release_sources(root_fd: int, manifest: Mapping[str, Any]) -> None:
    for relative_path, expected in sorted(manifest["code_files_sha256"].items()):
        payload = _read_regular_at(
            root_fd,
            relative_path,
            cap=CHILD_SOURCE_BYTES_CAP,
            name=f"release source {relative_path}",
        )
        if _sha256(payload) != expected:
            raise D106RCMRG0ProductionRunnerError(
                f"release source SHA256 mismatch: {relative_path}"
            )


# The parent runner is source-pinned by the manifest.  This small bootstrap
# reads only inherited, already-open FDs, validates the child bytes before
# executing them, and launches with ``python -I``.  It imports no project code.
_BOOTSTRAP = r'''
import argparse, hashlib, json, os, stat, sys
def canon(v): return json.dumps(v, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
def readfd(fd, cap, name):
    b=os.fstat(fd)
    if not stat.S_ISREG(b.st_mode): raise RuntimeError(name+" is not regular")
    out=[]; n=0
    while True:
        x=os.read(fd, min(1048576, cap+1-n))
        if not x: break
        n += len(x)
        if n > cap: raise RuntimeError(name+" cap")
        out.append(x)
    a=os.fstat(fd)
    if (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)!=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns): raise RuntimeError(name+" changed")
    return b"".join(out)
def readrel(root, rel, cap):
    parts=rel.split("/")
    if not parts or any((not p or p in (".","..")) for p in parts): raise RuntimeError("bad child path")
    d=os.dup(root); f=None
    try:
        for p in parts[:-1]:
            q=os.open(p, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=d); os.close(d); d=q
        f=os.open(parts[-1], os.O_RDONLY|os.O_NOFOLLOW, dir_fd=d)
        return readfd(f, cap, "child source")
    finally:
        if f is not None: os.close(f)
        os.close(d)
p=argparse.ArgumentParser()
p.add_argument("--mode", required=True); p.add_argument("--root-fd", type=int, required=True)
p.add_argument("--manifest-fd", type=int, required=True); p.add_argument("--manifest-sha256", required=True)
p.add_argument("--child-path", required=True); p.add_argument("--code-root", required=True); p.add_argument("--request-cap", type=int, required=True)
a=p.parse_args()
os.lseek(a.manifest_fd, 0, os.SEEK_SET); raw=readfd(a.manifest_fd, 262144, "release manifest")
if hashlib.sha256(raw).hexdigest()!=a.manifest_sha256: raise RuntimeError("manifest external SHA mismatch")
d=json.loads(raw.decode("utf-8"))
if raw!=canon(d) or d.get("child_entry_path")!=a.child_path: raise RuntimeError("manifest bootstrap closure")
h=d.get("code_files_sha256",{}).get(a.child_path)
if not isinstance(h,str): raise RuntimeError("child hash missing")
src=readrel(a.root_fd, a.child_path, 4194304)
if hashlib.sha256(src).hexdigest()!=h: raise RuntimeError("child source SHA mismatch")
request=sys.stdin.buffer.read(a.request_cap+1)
if len(request)>a.request_cap: raise RuntimeError("child request cap")
ns={"__name__":"__main__","__file__":os.path.join(a.code_root,*a.child_path.split("/")),"CHILD_MODE":a.mode,"CODE_ROOT_FD":a.root_fd,"CODE_ROOT_PATH":a.code_root,"RELEASE_MANIFEST_BYTES":raw,"EXPECTED_RELEASE_MANIFEST_SHA256":a.manifest_sha256,"REQUEST_BYTES":request}
exec(compile(src, ns["__file__"], "exec"), ns)
'''


def _run_clean_child(
    *,
    mode: str,
    root_fd: int,
    manifest_fd: int,
    code_root: Path,
    manifest_sha256: str,
    request_bytes: bytes,
    stdout_cap: int,
) -> bytes:
    if type(request_bytes) is not bytes or len(request_bytes) > 3 * RESULT_BYTES_CAP + 16_384:
        raise D106RCMRG0ProductionRunnerError("clean child request resource cap exceeded")
    command = [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        "--mode",
        mode,
        "--root-fd",
        str(root_fd),
        "--manifest-fd",
        str(manifest_fd),
        "--manifest-sha256",
        manifest_sha256,
        "--child-path",
        CHILD_RELATIVE_PATH,
        "--code-root",
        str(code_root),
        "--request-cap",
        str(3 * RESULT_BYTES_CAP + 16_384),
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
    }
    process = subprocess.Popen(
        command,
        cwd=str(code_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        pass_fds=(root_fd, manifest_fd),
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    try:
        process.stdin.write(request_bytes)
        process.stdin.close()
    except BrokenPipeError:
        pass
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {
        process.stdout.fileno(): (captured["stdout"], stdout_cap, "stdout"),
        process.stderr.fileno(): (captured["stderr"], CHILD_STDERR_BYTES_CAP, "stderr"),
    }
    selector = selectors.DefaultSelector()
    for fd in streams:
        os.set_blocking(fd, False)
        selector.register(fd, selectors.EVENT_READ)
    started = time.monotonic()
    try:
        while streams:
            if time.monotonic() - started > CHILD_TIMEOUT_SECONDS:
                process.kill()
                raise D106RCMRG0ProductionRunnerError("clean child timed out")
            for event, _mask in selector.select(timeout=0.2):
                fd = event.fd
                buffer, cap, label = streams[fd]
                chunk = os.read(fd, 65_536)
                if not chunk:
                    selector.unregister(fd)
                    del streams[fd]
                    continue
                buffer.extend(chunk)
                if len(buffer) > cap:
                    process.kill()
                    raise D106RCMRG0ProductionRunnerError(
                        f"clean child {label} resource cap exceeded"
                    )
        return_code = process.wait(timeout=5)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        raise
    finally:
        selector.close()
    if return_code != 0:
        detail = bytes(captured["stderr"]).decode("utf-8", errors="replace").strip()
        raise D106RCMRG0ProductionRunnerError(
            "clean child failed" + (f": {detail}" if detail else "")
        )
    return bytes(captured["stdout"])


def _extract_public_result(result_bytes: bytes) -> tuple[dict[str, Any], bytes, str]:
    """Structural parent check; nested proof is performed by verifier child."""

    if type(result_bytes) is not bytes or not result_bytes or len(result_bytes) > RESULT_BYTES_CAP:
        raise D106RCMRG0ProductionRunnerError("production result byte closure drift")
    try:
        document = json.loads(result_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0ProductionRunnerError("production result must be canonical JSON") from error
    if type(document) is not dict or result_bytes != _canonical_bytes(document):
        raise D106RCMRG0ProductionRunnerError("production result canonical closure drift")
    expected_keys = {
        "schema", "status", "candidate_id", "protocol_schema", "algorithm_execution_scope",
        "real_g0_executed", "runner_authority", "deployable", "formal_performance_claim",
        "authority_flags", "held_label_audit_status", "same_process_held_label_capability_absence_claimed",
        "rows_or_labels_returned", "expected_release_commit", "request_receipt_sha256",
        "code_sha256", "tap_archive_sha256", "tap_receipt_sha256",
        "predecessor_lock_bundle_root_sha256", "strict_loader_evidence", "execution_manifest",
        "execution_manifest_root_sha256", "result_receipt_sha256",
    }
    if (
        set(document) != expected_keys
        or document.get("schema") != PRODUCTION_SCHEMA
        or document.get("status") != PRODUCTION_EXECUTED_STATUS
        or document.get("real_g0_executed") is not True
        or document.get("runner_authority") is not False
        or document.get("deployable") is not False
        or document.get("formal_performance_claim") is not False
        or document.get("rows_or_labels_returned") is not False
        or document.get("same_process_held_label_capability_absence_claimed") is not False
        or type(document.get("execution_manifest")) is not dict
        or type(document.get("authority_flags")) is not dict
        or any(value is not False for value in document["authority_flags"].values())
    ):
        raise D106RCMRG0ProductionRunnerError("public production result closure drift")
    receipt = _require_sha256(document.get("result_receipt_sha256"), "result receipt")
    receipt_payload = dict(document)
    receipt_payload.pop("result_receipt_sha256")
    if _sha256(_canonical_bytes(receipt_payload)) != receipt:
        raise D106RCMRG0ProductionRunnerError("production result receipt mismatch")
    manifest_bytes = _canonical_bytes(document["execution_manifest"])
    if (
        len(manifest_bytes) > MANIFEST_BYTES_CAP
        or _sha256(manifest_bytes)
        != _require_sha256(document.get("execution_manifest_root_sha256"), "execution manifest root")
    ):
        raise D106RCMRG0ProductionRunnerError("production manifest root closure drift")
    return document, manifest_bytes, _sha256(result_bytes)


def _verify_child_receipt(
    payload: bytes, *, expected_result_sha256: str, expected_manifest_sha256: str
) -> None:
    if len(payload) > CHILD_VERIFY_BYTES_CAP:
        raise D106RCMRG0ProductionRunnerError("clean verifier receipt resource cap exceeded")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0ProductionRunnerError("clean verifier receipt is not JSON") from error
    expected_keys = {"schema", "status", "result_sha256", "release_manifest_sha256"}
    if (
        type(document) is not dict
        or payload != _canonical_bytes(document)
        or set(document) != expected_keys
        or document.get("schema") != VERIFY_RECEIPT_SCHEMA
        or document.get("status") != "VERIFIED"
        or document.get("result_sha256") != expected_result_sha256
        or document.get("release_manifest_sha256") != expected_manifest_sha256
    ):
        raise D106RCMRG0ProductionRunnerError("clean verifier receipt closure drift")


def _fstat_identity(fd: int) -> tuple[int, int]:
    metadata = os.fstat(fd)
    return int(metadata.st_dev), int(metadata.st_ino)


def _create_new_output_dir(output_dir: str | Path) -> int:
    target = Path(output_dir)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise D106RCMRG0ProductionRunnerError("output directory must be an absolute leaf")
    parent_fd = _open_absolute_dirfd(target.parent, "output parent")
    output_fd: int | None = None
    try:
        before = _fstat_identity(parent_fd)
        os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
        if _fstat_identity(parent_fd) != before:
            raise D106RCMRG0ProductionRunnerError("output parent changed during mkdir")
        output_fd = os.open(
            target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        return output_fd
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite output directory: {target}") from error
    except Exception:
        if output_fd is not None:
            os.close(output_fd)
        raise
    finally:
        os.close(parent_fd)


def _write_new_dirfd(directory_fd: int, name: str, payload: bytes) -> None:
    if type(payload) is not bytes or not payload or "/" in name or name in {"", ".", ".."}:
        raise D106RCMRG0ProductionRunnerError("invalid production artifact payload/name")
    directory_identity = _fstat_identity(directory_fd)
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    temporary_fd: int | None = None
    try:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite artifact: {name}")
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
        os.fsync(temporary_fd)
        metadata = os.fstat(temporary_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(payload):
            raise D106RCMRG0ProductionRunnerError("temporary artifact identity drift")
        os.close(temporary_fd)
        temporary_fd = None
        # linkat-style publication never overwrites a raced destination.
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if _fstat_identity(directory_fd) != directory_identity:
            raise D106RCMRG0ProductionRunnerError("artifact directory identity drift")
    except Exception:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def _completion_bytes(
    *, result_sha256: str, manifest_sha256: str, manifest_root_sha256: str,
    release_manifest_sha256: str, release_commit: str
) -> bytes:
    payload = {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "marker_written_last": True,
        "result_name": RESULT_NAME,
        "result_sha256": result_sha256,
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
        "execution_manifest_root_sha256": manifest_root_sha256,
        "release_manifest_sha256": release_manifest_sha256,
        "release_commit": release_commit,
    }
    payload["completion_receipt_sha256"] = _sha256(_canonical_bytes(payload))
    result = _canonical_bytes(payload)
    if len(result) > COMPLETION_BYTES_CAP:
        raise D106RCMRG0ProductionRunnerError("completion marker resource cap exceeded")
    return result


def run_d106_rcmr_g0_production(
    *, archive_path: str | Path, archive_sha256: str, receipt_path: str | Path,
    receipt_sha256: str, release_manifest_path: str | Path,
    expected_release_manifest_sha256: str, output_dir: str | Path,
) -> Mapping[str, Any]:
    """Execute and verify only from actual files in two fresh clean children."""

    _require_posix_production()
    archive_digest = _require_sha256(archive_sha256, "tap archive")
    receipt_digest = _require_sha256(receipt_sha256, "tap receipt")
    manifest_digest = _require_sha256(
        expected_release_manifest_sha256, "externally preregistered release manifest"
    )
    root_fd = _open_absolute_dirfd(CODE_ROOT, "release code root")
    manifest_fd: int | None = None
    output_fd: int | None = None
    try:
        manifest_fd, manifest_bytes = _open_regular_absolute(
            release_manifest_path, cap=RELEASE_MANIFEST_BYTES_CAP, name="release manifest"
        )
        manifest = _parse_release_manifest_bytes(
            manifest_bytes, expected_sha256=manifest_digest
        )
        _verify_optional_git_head(CODE_ROOT, manifest["release_commit"])
        _verify_release_sources(root_fd, manifest)
        execute_request = _canonical_bytes(
            {
                "schema": EXECUTE_REQUEST_SCHEMA,
                "archive_path": str(Path(archive_path)),
                "archive_sha256": archive_digest,
                "receipt_path": str(Path(receipt_path)),
                "receipt_sha256": receipt_digest,
            }
        )
        result_bytes = _run_clean_child(
            mode="execute", root_fd=root_fd, manifest_fd=manifest_fd, code_root=CODE_ROOT,
            manifest_sha256=manifest_digest, request_bytes=execute_request,
            stdout_cap=RESULT_BYTES_CAP,
        )
        result, execution_manifest_bytes, result_digest = _extract_public_result(result_bytes)
        verify_request = _canonical_bytes(
            {
                "schema": VERIFY_REQUEST_SCHEMA,
                "result_base64": base64.b64encode(result_bytes).decode("ascii"),
                "expected_result_sha256": result_digest,
            }
        )
        verify_bytes = _run_clean_child(
            mode="verify", root_fd=root_fd, manifest_fd=manifest_fd, code_root=CODE_ROOT,
            manifest_sha256=manifest_digest, request_bytes=verify_request,
            stdout_cap=CHILD_VERIFY_BYTES_CAP,
        )
        _verify_child_receipt(
            verify_bytes, expected_result_sha256=result_digest,
            expected_manifest_sha256=manifest_digest,
        )
        # No output state exists until both the actual-file execution and the
        # independent fresh verifier have succeeded.
        output_fd = _create_new_output_dir(output_dir)
        _write_new_dirfd(output_fd, RESULT_NAME, result_bytes)
        _write_new_dirfd(output_fd, MANIFEST_NAME, execution_manifest_bytes)
        completion = _completion_bytes(
            result_sha256=result_digest,
            manifest_sha256=_sha256(execution_manifest_bytes),
            manifest_root_sha256=result["execution_manifest_root_sha256"],
            release_manifest_sha256=manifest_digest,
            release_commit=manifest["release_commit"],
        )
        _write_new_dirfd(output_fd, COMPLETION_NAME, completion)
        return {
            "status": result["status"],
            "output_dir": str(Path(output_dir)),
            "result_sha256": result_digest,
            "release_manifest_sha256": manifest_digest,
        }
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        os.close(root_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="POSIX clean-child D106 RCMR-G0 publisher")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-release-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_d106_rcmr_g0_production(
        archive_path=args.archive, archive_sha256=args.archive_sha256,
        receipt_path=args.receipt, receipt_sha256=args.receipt_sha256,
        release_manifest_path=args.release_manifest,
        expected_release_manifest_sha256=args.expected_release_manifest_sha256,
        output_dir=args.output_dir,
    )
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
