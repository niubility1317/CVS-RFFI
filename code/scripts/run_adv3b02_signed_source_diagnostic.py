#!/usr/bin/env python
"""Run the ADV3B02 numerical diagnostic from a signed no-Git source root.

This development-only runner verifies the external signed source receipt and
exact ZIP before extraction.  It never signs, selects a runtime, grants formal
authority, reads Phase2 data, or changes N607 state.  Missing source authority
retains the consumer's ``BLOCKED_MISSING_SIGNED_SOURCE_RECEIPT`` exit-2
boundary and creates no output directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import somph_runtime_trust as runtime_trust  # noqa: E402


SOURCE_SCHEMA = "cvs.development.source_archive_commit_receipt.v1"
RUNNER_SCHEMA = "cvs.development.adv3b02_signed_source_runner.v1"
BLOCKED_STATUS = "BLOCKED_MISSING_SIGNED_SOURCE_RECEIPT"
ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
KEY_ID = "somph-authority-ed25519-20260716"
PUBLIC_KEY_HEX = "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
PUBLIC_KEY_SHA256 = "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
# SHA256 of the canonical LF Git blob, not a Windows CRLF worktree rendering.
TRUST_HELPER_GIT_BLOB_SHA256 = (
    "4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc"
)
BASE_CHECKPOINT_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
BASE_CHECKPOINT_PATH = "/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
ARMS: dict[str, dict[str, str]] = {
    "b202": {
        "runtime_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/adv3b02_base_runtime.ts",
        "runtime_sha256": "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9",
        "lineage_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/runtime_parity_receipt.json",
        "lineage_sha256": "db8635b986bcaea6cbe6f954e90e5ed37b9fb6042876628392db96fe82be42f4",
    },
    "f119": {
        "runtime_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt",
        "runtime_sha256": "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a",
        "lineage_path": "/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json",
        "lineage_sha256": "0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523",
    },
}


class SignedSourceRunnerError(RuntimeError):
    """Raised when signed-source isolation cannot be proven."""


class MissingSignedSourceReceiptError(SignedSourceRunnerError):
    """Raised before any output when the archive or receipt is absent."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _safe_member(value: Any) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or "\\" in text
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in text)
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".py"
        or str(pure) != text
    ):
        raise SignedSourceRunnerError("signed source member path is unsafe")
    return text


def _read_regular_snapshot(path: str | Path, name: str) -> tuple[Path, bytes]:
    requested = Path(path)
    try:
        before = requested.lstat()
    except FileNotFoundError as exc:
        raise MissingSignedSourceReceiptError(f"{name} is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SignedSourceRunnerError(f"{name} must be a regular non-symlink file")
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise SignedSourceRunnerError(f"{name} changed during snapshot")
    return resolved, b"".join(chunks)


def _open_bound_asset_snapshot(path: str | Path, name: str) -> dict[str, Any]:
    requested = Path(path)
    try:
        before = requested.lstat()
    except FileNotFoundError as exc:
        raise MissingSignedSourceReceiptError(f"{name} is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SignedSourceRunnerError(f"{name} must be a regular non-symlink file")
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or len(payload) != opened.st_size
        ):
            raise SignedSourceRunnerError(f"{name} changed during bound snapshot")
    except BaseException:
        os.close(descriptor)
        raise
    return {
        "name": name,
        "path": resolved,
        "bytes": payload,
        "sha256": _sha256(payload),
        "size": len(payload),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "mtime_ns": int(opened.st_mtime_ns),
        "ctime_ns": int(opened.st_ctime_ns),
        "descriptor": descriptor,
    }


def _close_bound_assets(snapshots: Sequence[Mapping[str, Any]]) -> None:
    for snapshot in snapshots:
        descriptor = snapshot.get("descriptor")
        if isinstance(descriptor, int):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _postflight_bound_asset(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = int(snapshot["descriptor"])
    held = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    held_bytes = b"".join(chunks)
    try:
        current_path, current_bytes = _read_regular_snapshot(
            snapshot["path"], str(snapshot["name"])
        )
        current = current_path.lstat()
    except MissingSignedSourceReceiptError as exc:
        raise SignedSourceRunnerError(
            f"{snapshot['name']} path disappeared after initial validation"
        ) from exc
    if (
        held.st_dev != snapshot["device"]
        or held.st_ino != snapshot["inode"]
        or held.st_size != snapshot["size"]
        or held.st_mtime_ns != snapshot["mtime_ns"]
        or held.st_ctime_ns != snapshot["ctime_ns"]
        or held_bytes != snapshot["bytes"]
        or current.st_dev != snapshot["device"]
        or current.st_ino != snapshot["inode"]
        or current.st_size != snapshot["size"]
        or current.st_mtime_ns != snapshot["mtime_ns"]
        or current.st_ctime_ns != snapshot["ctime_ns"]
        or current_bytes != snapshot["bytes"]
    ):
        raise SignedSourceRunnerError(
            f"{snapshot['name']} changed or was replaced after initial validation"
        )
    return {
        "name": snapshot["name"],
        "path": str(snapshot["path"]),
        "bytes": snapshot["size"],
        "sha256": snapshot["sha256"],
        "device": snapshot["device"],
        "inode": snapshot["inode"],
        "mtime_ns": snapshot["mtime_ns"],
        "ctime_ns": snapshot["ctime_ns"],
        "fd_held_through_child": True,
        "postflight_identity_and_bytes_match": True,
        "child_path_mode": "signed_or_canonical_original_path_with_fd_held_and_postflight_reject",
    }


def _validate_sha(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise SignedSourceRunnerError(f"{name} is not lowercase SHA256")
    return text


def _manifest_root(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json({"source_members": list(rows)}))


def _fixed_verify(message: bytes, signature: bytes) -> None:
    helper_path, helper_bytes = _read_regular_snapshot(
        CODE_ROOT / "cvsrffi/somph_runtime_trust.py", "fixed source trust helper"
    )
    module_path = Path(str(runtime_trust.__file__)).resolve(strict=True)
    if (
        helper_path != module_path
        or _sha256(helper_bytes) != TRUST_HELPER_GIT_BLOB_SHA256
        or runtime_trust.PINNED_AUTHORITY_ISSUER != ISSUER
        or runtime_trust.PINNED_AUTHORITY_KEY_ID != KEY_ID
        or runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX != PUBLIC_KEY_HEX
        or runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256 != PUBLIC_KEY_SHA256
        or _sha256(bytes.fromhex(PUBLIC_KEY_HEX)) != PUBLIC_KEY_SHA256
    ):
        raise SignedSourceRunnerError("fixed source trust identity drift")
    try:
        runtime_trust.verify_ed25519(bytes.fromhex(PUBLIC_KEY_HEX), message, signature)
    except (TypeError, ValueError) as exc:
        raise SignedSourceRunnerError("source release signature is invalid") from exc


def _production_platform_guard() -> dict[str, Any]:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise SignedSourceRunnerError(
            "production signed-source isolation is Linux-only; Windows is test-only"
        )
    if not Path("/proc/self/fd").is_dir():
        raise SignedSourceRunnerError("Linux /proc/self/fd is required for FD audit")
    return {
        "platform": sys.platform,
        "os_name": os.name,
        "linux_only_production": True,
        "trust_helper_identity": "canonical_lf_git_blob_sha256",
        "trust_helper_sha256": TRUST_HELPER_GIT_BLOB_SHA256,
    }


def _validate_release_snapshots(
    *,
    archive_path: Path,
    archive_bytes: bytes,
    receipt_path: Path,
    receipt_bytes: bytes,
    verifier: Callable[[bytes, bytes], None] = _fixed_verify,
) -> dict[str, Any]:
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignedSourceRunnerError("source release receipt is not UTF-8 JSON") from exc
    exact = {
        "schema",
        "issuer",
        "key_id",
        "public_key_sha256",
        "source_archive_path",
        "source_archive_sha256",
        "source_git_commit",
        "source_members",
        "source_manifest_root_sha256",
        "git_policy",
        "signature_hex",
    }
    if not isinstance(receipt, dict) or set(receipt) != exact:
        raise SignedSourceRunnerError("source release receipt exact schema drift")
    if (
        receipt.get("schema") != SOURCE_SCHEMA
        or receipt.get("issuer") != ISSUER
        or receipt.get("key_id") != KEY_ID
        or receipt.get("public_key_sha256") != PUBLIC_KEY_SHA256
        or receipt.get("source_archive_path") != str(archive_path)
        or receipt.get("source_archive_sha256") != _sha256(archive_bytes)
        or receipt.get("git_policy") != {"mode": "signed_manifest_only_no_git"}
    ):
        raise SignedSourceRunnerError("source release fixed binding drift")
    commit = str(receipt.get("source_git_commit", ""))
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise SignedSourceRunnerError("source release Git commit is invalid")
    raw_rows = receipt.get("source_members")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SignedSourceRunnerError("source release member manifest is empty")
    rows = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise SignedSourceRunnerError("source member exact schema drift")
        member = _safe_member(raw["path"])
        size = raw["bytes"]
        if member in seen or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SignedSourceRunnerError("source member duplicate or size drift")
        rows.append(
            {"path": member, "bytes": size, "sha256": _validate_sha(raw["sha256"], "member SHA256")}
        )
        seen.add(member)
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise SignedSourceRunnerError("source member manifest is not sorted")
    if receipt.get("source_manifest_root_sha256") != _manifest_root(rows):
        raise SignedSourceRunnerError("source member manifest root drift")
    try:
        signature = bytes.fromhex(str(receipt.get("signature_hex", "")))
    except ValueError as exc:
        raise SignedSourceRunnerError("source signature is not hex") from exc
    if len(signature) != 64:
        raise SignedSourceRunnerError("source signature length drift")
    body = {key: value for key, value in receipt.items() if key != "signature_hex"}
    verifier(_canonical_json(body), signature)
    archive_rows: list[dict[str, Any]] = []
    member_payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as bundle:
            if bundle.comment:
                raise SignedSourceRunnerError("source archive comment is forbidden")
            names: set[str] = set()
            for info in bundle.infolist():
                member = _safe_member(info.filename)
                unix_type = (int(info.external_attr) >> 16) & 0o170000
                if (
                    member in names
                    or info.is_dir()
                    or unix_type != stat.S_IFREG
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 0x1
                    or info.extra
                    or info.comment
                ):
                    raise SignedSourceRunnerError(
                        "source archive has duplicate, directory, symlink, compressed, encrypted, or extra metadata member"
                    )
                payload = bundle.read(info)
                member_payloads[member] = payload
                archive_rows.append(
                    {"path": member, "bytes": len(payload), "sha256": _sha256(payload)}
                )
                names.add(member)
    except zipfile.BadZipFile as exc:
        raise SignedSourceRunnerError("source archive is not a valid ZIP") from exc
    if archive_rows != rows:
        raise SignedSourceRunnerError("source archive has missing, reordered, or extra members")
    diagnostic = "code/scripts/diagnose_adv3b02_runtime_numerics.py"
    trust = "code/cvsrffi/somph_runtime_trust.py"
    if diagnostic not in member_payloads or trust not in member_payloads:
        raise SignedSourceRunnerError("source archive lacks consumer or fixed trust helper")
    return {
        "receipt": receipt,
        "receipt_sha256": _sha256(receipt_bytes),
        "archive_sha256": _sha256(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "members": rows,
        "member_payloads": member_payloads,
    }


def _no_git_ancestor(path: Path) -> None:
    current = path.resolve(strict=True)
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.exists() or marker.is_symlink():
            raise SignedSourceRunnerError(
                f"isolation root would inherit a Git parent: {candidate}"
            )


def _write_file_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _extract_exact(root: Path, members: Mapping[str, bytes]) -> None:
    if any(root.iterdir()):
        raise SignedSourceRunnerError("isolation root is not empty")
    for member in sorted(members):
        destination = root.joinpath(*PurePosixPath(member).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = destination.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise SignedSourceRunnerError("extraction parent escapes isolation root") from exc
        for ancestor in (resolved_parent, *resolved_parent.parents):
            if ancestor == root.parent:
                break
            if ancestor.is_symlink():
                raise SignedSourceRunnerError("symlink extraction ancestor is forbidden")
        _write_file_new(destination, members[member])


def _git_unavailable_audit(root: Path, env: Mapping[str, str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            cwd=root,
            env=dict(env),
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        return {
            "git_binary_available": False,
            "repository_discovered": False,
            "returncode": None,
            "stderr_sha256": _sha256(str(exc).encode()),
        }
    if result.returncode == 0:
        raise SignedSourceRunnerError("Git repository remains discoverable in isolation root")
    return {
        "git_binary_available": True,
        "repository_discovered": False,
        "returncode": int(result.returncode),
        "stdout_sha256": _sha256(result.stdout),
        "stderr_sha256": _sha256(result.stderr),
    }


def _assert_bound_asset(
    snapshot: Mapping[str, Any], expected_path: str, expected_sha: str, name: str
) -> Path:
    resolved = Path(snapshot["path"])
    if str(resolved) != str(Path(expected_path)) or snapshot["sha256"] != expected_sha:
        raise SignedSourceRunnerError(f"{name} is not the preregistered canonical asset")
    return resolved


def _publish_directory(staging: Path, output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("refusing to overwrite runner output root")
    if os.name == "nt":
        try:
            os.rename(staging, output_root)
        except OSError as exc:
            raise SignedSourceRunnerError("runner output publication failed") from exc
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise SignedSourceRunnerError("Linux renameat2 is required for no-replace output")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(output_root), 1)
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError("refusing to overwrite runner output root")
            raise SignedSourceRunnerError(
                f"runner output publication failed: {os.strerror(error)}"
            )
        return
    raise SignedSourceRunnerError("atomic no-replace output is unsupported")


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def make_writable_and_retry(function, value, _error) -> None:
        os.chmod(value, 0o700)
        function(value)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def run_signed_source_diagnostic(
    *,
    arm_id: str,
    checkpoint: str | Path,
    runtime: str | Path,
    lineage_evidence: str | Path,
    source_archive: str | Path,
    source_release_receipt: str | Path,
    isolation_parent: str | Path,
    output_root: str | Path,
    device: str,
    include_cpu_control: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    started = time.perf_counter()
    platform_audit = _production_platform_guard()
    if arm_id not in ARMS:
        raise SignedSourceRunnerError("arm-id must be preregistered b202 or f119")
    if not str(device).startswith("cuda:") or not str(device)[5:].isdigit():
        raise SignedSourceRunnerError("device must be explicit cuda:<index>")
    snapshots: list[dict[str, Any]] = []
    try:
        archive_snapshot = _open_bound_asset_snapshot(source_archive, "source archive")
        snapshots.append(archive_snapshot)
        receipt_snapshot = _open_bound_asset_snapshot(
            source_release_receipt, "source release receipt"
        )
        snapshots.append(receipt_snapshot)
        checkpoint_snapshot = _open_bound_asset_snapshot(checkpoint, "checkpoint")
        snapshots.append(checkpoint_snapshot)
        runtime_snapshot = _open_bound_asset_snapshot(runtime, "runtime")
        snapshots.append(runtime_snapshot)
        lineage_snapshot = _open_bound_asset_snapshot(
            lineage_evidence, "lineage evidence"
        )
        snapshots.append(lineage_snapshot)
    except BaseException:
        _close_bound_assets(snapshots)
        raise
    try:
        archive_path = Path(archive_snapshot["path"])
        archive_bytes = bytes(archive_snapshot["bytes"])
        receipt_path = Path(receipt_snapshot["path"])
        receipt_bytes = bytes(receipt_snapshot["bytes"])
        release = _validate_release_snapshots(
            archive_path=archive_path,
            archive_bytes=archive_bytes,
            receipt_path=receipt_path,
            receipt_bytes=receipt_bytes,
        )
        checkpoint_path = _assert_bound_asset(
            checkpoint_snapshot,
            BASE_CHECKPOINT_PATH,
            BASE_CHECKPOINT_SHA256,
            "checkpoint",
        )
        arm = ARMS[arm_id]
        runtime_path = _assert_bound_asset(
            runtime_snapshot, arm["runtime_path"], arm["runtime_sha256"], "runtime"
        )
        lineage_path = _assert_bound_asset(
            lineage_snapshot,
            arm["lineage_path"],
            arm["lineage_sha256"],
            "lineage evidence",
        )
    except BaseException:
        _close_bound_assets(snapshots)
        raise
    staging: Path | None = None
    isolation: Path | None = None
    try:
        parent = Path(isolation_parent).resolve(strict=True)
        if not parent.is_dir() or parent.is_symlink():
            raise SignedSourceRunnerError("isolation parent must be a real directory")
        _no_git_ancestor(parent)
        output = Path(output_root).resolve(strict=False)
        if output.exists() or output.is_symlink():
            raise FileExistsError("refusing to overwrite runner output root")
        output_parent = output.parent.resolve(strict=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output_parent)
        )
        isolation = Path(
            tempfile.mkdtemp(prefix="adv3b02_signed_source_", dir=parent)
        )
    except BaseException:
        if isolation is not None and isolation.exists():
            _remove_tree(isolation)
        if staging is not None and staging.exists():
            _remove_tree(staging)
        _close_bound_assets(snapshots)
        raise
    assert staging is not None and isolation is not None
    published = False
    try:
        _no_git_ancestor(isolation)
        _extract_exact(isolation, release["member_payloads"])
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [str(isolation / "code"), str(isolation)]
                ),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_CEILING_DIRECTORIES": str(isolation),
                "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
            }
        )
        git_audit = _git_unavailable_audit(isolation, env)
        diagnostic_path = isolation / "code/scripts/diagnose_adv3b02_runtime_numerics.py"
        artifact = staging / "diagnostic.json"
        command = [
            sys.executable,
            str(diagnostic_path),
            "--checkpoint",
            str(checkpoint_path),
            "--runtime",
            str(runtime_path),
            "--lineage-evidence",
            str(lineage_path),
            "--arm-id",
            arm_id,
            "--source-archive",
            str(archive_path),
            "--source-release-receipt",
            str(receipt_path),
            "--artifact-out",
            str(artifact),
            "--device",
            str(device),
        ]
        if include_cpu_control:
            command.append("--include-cpu-control")
        try:
            child = subprocess.run(
                command,
                cwd=isolation,
                env=env,
                capture_output=True,
                timeout=int(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = bytes(exc.stdout or b"")
            stderr = bytes(exc.stderr or b"")
            child_returncode = None
            child_status = "CHILD_TIMEOUT_NO_AUTHORITY"
        else:
            stdout = bytes(child.stdout)
            stderr = bytes(child.stderr)
            child_returncode = int(child.returncode)
            child_status = (
                "DEVELOPMENT_DIAGNOSTIC_CHILD_COMPLETE_NO_AUTHORITY"
                if child.returncode == 0 and artifact.is_file()
                else "CHILD_FAILED_NO_AUTHORITY"
            )
        _write_file_new(staging / "child_stdout.txt", stdout)
        _write_file_new(staging / "child_stderr.txt", stderr)
        artifact_record = None
        if artifact.exists():
            if artifact.is_symlink() or not artifact.is_file():
                raise SignedSourceRunnerError("child diagnostic artifact is not regular")
            artifact_bytes = artifact.read_bytes()
            artifact_record = {
                "path": "diagnostic.json",
                "bytes": len(artifact_bytes),
                "sha256": _sha256(artifact_bytes),
            }
            os.chmod(artifact, 0o444)
        asset_postflight = [
            _postflight_bound_asset(snapshot) for snapshot in snapshots
        ]
        _remove_tree(isolation)
        if isolation.exists():
            raise SignedSourceRunnerError("isolation source root was not removed")
        audit = {
            "schema": RUNNER_SCHEMA,
            "status": child_status,
            "formal_authority": False,
            "formal_metric_claim_allowed": False,
            "target_access": False,
            "source_cache_access": False,
            "runtime_selection_performed": False,
            "production_platform": platform_audit,
            "asset_snapshot_binding": {
                "status": "ALL_ASSETS_FD_HELD_AND_POSTFLIGHT_MATCHED",
                "child_reads_original_signed_or_canonical_paths": True,
                "original_path_replacement_policy": "fail_before_output_publication",
                "assets": asset_postflight,
            },
            "arm_id": arm_id,
            "checkpoint": {"path": str(checkpoint_path), "sha256": BASE_CHECKPOINT_SHA256},
            "runtime": {"path": str(runtime_path), "sha256": arm["runtime_sha256"]},
            "lineage": {"path": str(lineage_path), "sha256": arm["lineage_sha256"]},
            "source_release": {
                "archive_path": str(archive_path),
                "archive_sha256": release["archive_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": release["receipt_sha256"],
                "source_git_commit": release["receipt"]["source_git_commit"],
                "member_count": len(release["members"]),
            },
            "isolation": {
                "root_removed_after_run": True,
                "root_parent": str(parent),
                "no_git_ancestor": True,
                "git_ceiling_directories": str(isolation),
                "git_probe": git_audit,
                "pythonpath_only_signed_roots": True,
            },
            "child": {
                "returncode": child_returncode,
                "timeout_seconds": int(timeout_seconds),
                "stdout_bytes": len(stdout),
                "stdout_sha256": _sha256(stdout),
                "stderr_bytes": len(stderr),
                "stderr_sha256": _sha256(stderr),
                "diagnostic_artifact": artifact_record,
            },
            "wall_time_seconds": float(time.perf_counter() - started),
        }
        audit_bytes = _canonical_json(audit)
        _write_file_new(staging / "runner_audit.json", audit_bytes)
        _publish_directory(staging, output)
        published = True
        return {
            "status": child_status,
            "output_root": str(output),
            "runner_audit_sha256": _sha256(audit_bytes),
            "child_returncode": child_returncode,
            "diagnostic_artifact_emitted": artifact_record is not None,
            "formal_authority": False,
        }
    finally:
        try:
            if isolation.exists():
                _remove_tree(isolation)
            if not published:
                _remove_tree(staging)
        finally:
            _close_bound_assets(snapshots)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", choices=sorted(ARMS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--lineage-evidence", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-release-receipt", type=Path)
    parser.add_argument("--isolation-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--include-cpu-control", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.source_archive is None or args.source_release_receipt is None:
        print(
            json.dumps(
                {
                    "status": BLOCKED_STATUS,
                    "artifact_emitted": False,
                    "formal_authority": False,
                    "reason": "external signed source archive and receipt are required",
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 2
    try:
        result = run_signed_source_diagnostic(
            arm_id=args.arm_id,
            checkpoint=args.checkpoint,
            runtime=args.runtime,
            lineage_evidence=args.lineage_evidence,
            source_archive=args.source_archive,
            source_release_receipt=args.source_release_receipt,
            isolation_parent=args.isolation_parent,
            output_root=args.output_root,
            device=args.device,
            include_cpu_control=bool(args.include_cpu_control),
            timeout_seconds=args.timeout_seconds,
        )
    except MissingSignedSourceReceiptError:
        print(
            json.dumps(
                {
                    "status": BLOCKED_STATUS,
                    "artifact_emitted": False,
                    "formal_authority": False,
                    "reason": "external signed source archive or receipt is missing",
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0 if result["child_returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
