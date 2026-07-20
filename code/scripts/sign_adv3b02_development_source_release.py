#!/usr/bin/env python
"""Build and sign one code-only ADV3B02 numerical-diagnostic release.

This is an offline producer for
``cvs.development.source_archive_commit_receipt.v1``.  It does not authorize
runtime selection, parity, target access, or a formal metric claim.  The
production entry point fixes the existing SOMP-H Ed25519 trust identity and
accepts the externally held key only as an explicit input.

The emitted ZIP is intentionally an exact-member archive, not a full Git
archive.  The consumer requires the archive manifest to equal the project
modules actually imported by the diagnostic worker.  A versioned member lock
therefore supplies the reviewed, sorted list of Python source paths.
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
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import somph_runtime_trust as runtime_trust  # noqa: E402
from scripts import sign_cvs_somph_authority_lock as lock_signer  # noqa: E402


SOURCE_RELEASE_SCHEMA = "cvs.development.source_archive_commit_receipt.v1"
SOURCE_MEMBER_LOCK_SCHEMA = "cvs.development.adv3b02_source_member_lock.v1"
SOURCE_RELEASE_ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
SOURCE_RELEASE_KEY_ID = "somph-authority-ed25519-20260716"
SOURCE_RELEASE_PUBLIC_KEY_HEX = (
    "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
)
SOURCE_RELEASE_PUBLIC_KEY_SHA256 = (
    "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
)
SOURCE_ARCHIVE_NAME = "adv3b02_numerical_source.zip"
SOURCE_RECEIPT_NAME = "source_release_receipt.json"
_SHA256_LENGTH = 64


class ADV3B02SourceReleaseSigningError(RuntimeError):
    """Raised before publication when the signed source release is invalid."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
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


def _source_manifest_root(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_json_bytes({"source_members": list(rows)}))


def _run_git(repo_root: Path, arguments: Sequence[str], *, timeout: int = 30) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ADV3B02SourceReleaseSigningError(
            "signed source release Git preflight failed"
        ) from exc
    return bytes(completed.stdout)


def _validate_commit(value: str) -> str:
    commit = str(value).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ADV3B02SourceReleaseSigningError(
            "source commit must be one lowercase 40-hex Git object ID"
        )
    return commit


def _clean_git_audit(repo_root: str | Path, source_commit: str) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ADV3B02SourceReleaseSigningError(
            "source release repository must be a real directory"
        )
    top = Path(
        _run_git(root, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    ).resolve(strict=True)
    if os.path.normcase(str(top)) != os.path.normcase(str(root)):
        raise ADV3B02SourceReleaseSigningError(
            "source release repository root is not the Git top level"
        )
    commit = _validate_commit(source_commit)
    head = _run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if head != commit:
        raise ADV3B02SourceReleaseSigningError(
            "source release HEAD does not equal the requested commit"
        )
    if _run_git(root, ["cat-file", "-t", commit]).strip() != b"commit":
        raise ADV3B02SourceReleaseSigningError(
            "source release object is not a Git commit"
        )
    status = _run_git(root, ["status", "--porcelain=v1", "-z"], timeout=60)
    if status:
        raise ADV3B02SourceReleaseSigningError(
            "source release requires a clean Git worktree including no untracked files"
        )
    return {
        "repository_root": str(root),
        "source_git_commit": commit,
        "dirty": False,
        "status_root_sha256": _sha256(status),
    }


def _safe_member_path(value: Any) -> str:
    member = str(value)
    pure = PurePosixPath(member)
    if (
        not member
        or pure.is_absolute()
        or "\\" in member
        or any(ord(character) < 32 or ord(character) == 127 for character in member)
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".py"
        or str(pure) != member
    ):
        raise ADV3B02SourceReleaseSigningError(
            "source member path must be a normalized relative Python path"
        )
    return member


def _git_blob(repo_root: Path, commit: str, member: str) -> bytes:
    tree_row = _run_git(
        repo_root,
        ["ls-tree", "-z", commit, "--", member],
    )
    rows = [row for row in tree_row.split(b"\0") if row]
    if len(rows) != 1:
        raise ADV3B02SourceReleaseSigningError(
            "source member is not one tracked Git object"
        )
    try:
        descriptor, encoded_path = rows[0].split(b"\t", 1)
        mode, kind, object_id = descriptor.split(b" ", 2)
        actual_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ADV3B02SourceReleaseSigningError(
            "source member Git descriptor is invalid"
        ) from exc
    if (
        actual_path != member
        or kind != b"blob"
        or mode not in {b"100644", b"100755"}
        or len(object_id) != 40
    ):
        raise ADV3B02SourceReleaseSigningError(
            "source member is missing, a symlink, or not a regular Git blob"
        )
    return _run_git(repo_root, ["cat-file", "blob", object_id.decode("ascii")], timeout=60)


def _member_lock(
    repo_root: Path,
    source_commit: str,
    member_lock_path: str | Path,
) -> tuple[list[str], dict[str, Any]]:
    requested_lock_path = Path(member_lock_path)
    if requested_lock_path.is_symlink():
        raise ADV3B02SourceReleaseSigningError(
            "source member lock must be a regular tracked file"
        )
    lock_path = requested_lock_path.resolve(strict=True)
    if not lock_path.is_file():
        raise ADV3B02SourceReleaseSigningError(
            "source member lock must be a regular tracked file"
        )
    try:
        relative = lock_path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ADV3B02SourceReleaseSigningError(
            "source member lock must be inside the source repository"
        ) from exc
    relative = _safe_member_path(relative) if relative.endswith(".py") else str(PurePosixPath(relative))
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
    ):
        raise ADV3B02SourceReleaseSigningError("source member lock path is unsafe")
    raw = _git_blob(repo_root, source_commit, relative)
    try:
        lock = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ADV3B02SourceReleaseSigningError(
            "source member lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(lock, dict) or set(lock) != {"schema", "members"}:
        raise ADV3B02SourceReleaseSigningError(
            "source member lock exact schema drift"
        )
    if lock.get("schema") != SOURCE_MEMBER_LOCK_SCHEMA:
        raise ADV3B02SourceReleaseSigningError(
            "source member lock schema drift"
        )
    raw_members = lock.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ADV3B02SourceReleaseSigningError(
            "source member lock must contain source paths"
        )
    members = [_safe_member_path(item) for item in raw_members]
    if members != sorted(members) or len(set(members)) != len(members):
        raise ADV3B02SourceReleaseSigningError(
            "source member lock paths must be unique and sorted"
        )
    if relative in set(members):
        raise ADV3B02SourceReleaseSigningError(
            "source member lock is review metadata, not an execution member"
        )
    return members, {
        "path": relative,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _deterministic_archive(
    repo_root: Path,
    source_commit: str,
    members: Sequence[str],
) -> tuple[bytes, list[dict[str, Any]]]:
    payloads: list[tuple[str, bytes]] = [
        (member, _git_blob(repo_root, source_commit, member)) for member in members
    ]
    rows = [
        {"path": member, "bytes": len(payload), "sha256": _sha256(payload)}
        for member, payload in payloads
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for member, payload in payloads:
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, payload)
    archive_bytes = buffer.getvalue()
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            observed = []
            for info in archive.infolist():
                unix_mode = (int(info.external_attr) >> 16) & 0o170000
                if info.is_dir() or unix_mode != stat.S_IFREG:
                    raise ADV3B02SourceReleaseSigningError(
                        "source archive contains a directory or non-regular member"
                    )
                raw = archive.read(info)
                observed.append(
                    {
                        "path": info.filename,
                        "bytes": len(raw),
                        "sha256": _sha256(raw),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ADV3B02SourceReleaseSigningError(
            "source archive deterministic serialization failed"
        ) from exc
    if observed != rows:
        raise ADV3B02SourceReleaseSigningError(
            "source archive does not exactly reproduce the reviewed member rows"
        )
    return archive_bytes, rows


def _native_test_execution_path(value: str) -> str:
    return str(Path(value).resolve(strict=False))


def _production_execution_path(value: str, archive_name: str) -> str:
    candidate = str(value)
    pure = PurePosixPath(candidate)
    if (
        not pure.is_absolute()
        or "\\" in candidate
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != candidate
        or pure.name != archive_name
    ):
        raise ADV3B02SourceReleaseSigningError(
            "signed execution archive path must be one normalized absolute POSIX path"
        )
    return candidate


def _outside_repository(path: Path, repo_root: Path) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError:
        return
    raise ADV3B02SourceReleaseSigningError(
        "source release outputs must be outside the source repository"
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_readonly(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("refusing to overwrite source release output root")
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ADV3B02SourceReleaseSigningError(
                "Linux renameat2 is required for no-replace publication"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(
                    "refusing to overwrite source release output root"
                )
            raise OSError(error, os.strerror(error))
        return
    raise ADV3B02SourceReleaseSigningError(
        "atomic no-replace publication is unsupported on this platform"
    )


def _publish_transaction(
    *,
    output_root: Path,
    archive_name: str,
    archive_bytes: bytes,
    receipt_name: str,
    receipt_bytes: bytes,
) -> None:
    if output_root.exists():
        raise FileExistsError("refusing to overwrite source release output root")
    parent = output_root.parent.resolve(strict=True)
    _fsync_directory(parent)
    staging = parent / f".{output_root.name}.staging-{secrets.token_hex(8)}"
    staging.mkdir()
    published = False

    def remove_created_tree(path: Path) -> None:
        def make_writable_and_retry(function, value, _error) -> None:
            os.chmod(value, 0o700)
            function(value)

        shutil.rmtree(path, onerror=make_writable_and_retry)

    try:
        _write_new_readonly(staging / archive_name, archive_bytes)
        _write_new_readonly(staging / receipt_name, receipt_bytes)
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, output_root)
        published = True
        _fsync_directory(parent)
    except BaseException:
        cleanup = output_root if published else staging
        if cleanup.exists():
            remove_created_tree(cleanup)
        try:
            _fsync_directory(parent)
        except OSError:
            pass
        raise


def _sign_source_archive_release_impl(
    *,
    repo_root: str | Path,
    source_commit: str,
    member_lock_path: str | Path,
    archive_output: str | Path,
    signed_archive_execution_path: str,
    receipt_output: str | Path,
    issuer: str,
    key_id: str,
    public_key_sha256: str,
    sign_message: Callable[[bytes], bytes],
    verify_signature: Callable[[bytes, bytes], None],
    execution_path_normalizer: Callable[[str], str],
) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    commit = _validate_commit(source_commit)
    preflight = _clean_git_audit(root, commit)
    archive_path = Path(archive_output).resolve(strict=False)
    receipt_path = Path(receipt_output).resolve(strict=False)
    if archive_path == receipt_path or archive_path.parent != receipt_path.parent:
        raise ADV3B02SourceReleaseSigningError(
            "archive and receipt must be distinct children of one new output root"
        )
    output_root = archive_path.parent
    if output_root.exists():
        raise FileExistsError("refusing to overwrite source release output root")
    if archive_path.name != SOURCE_ARCHIVE_NAME or receipt_path.name != SOURCE_RECEIPT_NAME:
        raise ADV3B02SourceReleaseSigningError(
            "source release output filenames are fixed"
        )
    _outside_repository(output_root, root)
    signed_path = execution_path_normalizer(str(signed_archive_execution_path))
    members, lock_audit = _member_lock(root, commit, member_lock_path)
    archive_bytes, source_members = _deterministic_archive(root, commit, members)
    body = {
        "schema": SOURCE_RELEASE_SCHEMA,
        "issuer": str(issuer),
        "key_id": str(key_id),
        "public_key_sha256": str(public_key_sha256),
        "source_archive_path": signed_path,
        "source_archive_sha256": _sha256(archive_bytes),
        "source_git_commit": commit,
        "source_members": source_members,
        "source_manifest_root_sha256": _source_manifest_root(source_members),
        "git_policy": {"mode": "signed_manifest_only_no_git"},
    }
    message = _canonical_json_bytes(body)
    signature = bytes(sign_message(message))
    if len(signature) != 64:
        raise ADV3B02SourceReleaseSigningError(
            "Ed25519 source release signature length drift"
        )
    try:
        verify_signature(message, signature)
    except (TypeError, ValueError) as exc:
        raise ADV3B02SourceReleaseSigningError(
            "Ed25519 source release signature is not valid for the pinned identity"
        ) from exc
    postflight = _clean_git_audit(root, commit)
    if postflight != preflight:
        raise ADV3B02SourceReleaseSigningError(
            "source repository changed during release production"
        )
    receipt = {**body, "signature_hex": signature.hex()}
    receipt_bytes = _canonical_json_bytes(receipt)
    _publish_transaction(
        output_root=output_root,
        archive_name=archive_path.name,
        archive_bytes=archive_bytes,
        receipt_name=receipt_path.name,
        receipt_bytes=receipt_bytes,
    )
    return {
        "schema": SOURCE_RELEASE_SCHEMA,
        "status": "SIGNED_DEVELOPMENT_SOURCE_RELEASE_CREATED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "target_access": False,
        "source_cache_access": False,
        "runtime_selection_authorized": False,
        "source_git_commit": commit,
        "member_lock": lock_audit,
        "source_member_count": len(source_members),
        "source_manifest_root_sha256": body["source_manifest_root_sha256"],
        "source_archive": str(archive_path),
        "signed_source_archive_execution_path": signed_path,
        "source_archive_sha256": body["source_archive_sha256"],
        "source_archive_bytes": len(archive_bytes),
        "source_release_receipt": str(receipt_path),
        "source_release_receipt_sha256": _sha256(receipt_bytes),
        "source_release_receipt_bytes": len(receipt_bytes),
        "signature_verified": True,
        "git_policy": body["git_policy"],
    }


def _make_production_signer() -> Callable[..., dict[str, Any]]:
    issuer = SOURCE_RELEASE_ISSUER
    key_id = SOURCE_RELEASE_KEY_ID
    public_key_hex = SOURCE_RELEASE_PUBLIC_KEY_HEX
    public_key_sha256 = SOURCE_RELEASE_PUBLIC_KEY_SHA256
    public_key = bytes.fromhex(public_key_hex)
    verifier = runtime_trust.verify_ed25519
    if (
        runtime_trust.PINNED_AUTHORITY_ISSUER != issuer
        or runtime_trust.PINNED_AUTHORITY_KEY_ID != key_id
        or runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX != public_key_hex
        or runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256 != public_key_sha256
        or _sha256(public_key) != public_key_sha256
    ):
        raise ADV3B02SourceReleaseSigningError(
            "source release producer and SOMP-H runtime trust identity drift"
        )

    def sign_source_archive_release(
        *,
        repo_root: str | Path,
        source_commit: str,
        member_lock_path: str | Path,
        archive_output: str | Path,
        signed_archive_execution_path: str,
        receipt_output: str | Path,
        private_key_path: str | Path,
        openssl_bin: str | Path = lock_signer.PINNED_OPENSSL_BINARY_PATH,
    ) -> dict[str, Any]:
        _, openssl_bytes, openssl_sha, runtime_files = (
            lock_signer._pinned_openssl_binary(openssl_bin)
        )
        private_key = lock_signer._resolved_regular_file(
            private_key_path, context="Ed25519 private key"
        )
        with lock_signer._private_openssl_executable(
            verified_bytes=openssl_bytes,
            expected_sha256=openssl_sha,
            runtime_files=runtime_files,
        ) as private_openssl:
            return _sign_source_archive_release_impl(
                repo_root=repo_root,
                source_commit=source_commit,
                member_lock_path=member_lock_path,
                archive_output=archive_output,
                signed_archive_execution_path=signed_archive_execution_path,
                receipt_output=receipt_output,
                issuer=issuer,
                key_id=key_id,
                public_key_sha256=public_key_sha256,
                sign_message=lambda message: lock_signer._sign_with_openssl(
                    openssl_binary=private_openssl,
                    private_key=private_key,
                    message=message,
                ),
                verify_signature=lambda message, signature: verifier(
                    public_key, message, signature
                ),
                execution_path_normalizer=lambda value: _production_execution_path(
                    value, SOURCE_ARCHIVE_NAME
                ),
            )

    return sign_source_archive_release


sign_source_archive_release = _make_production_signer()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--member-lock", type=Path, required=True)
    parser.add_argument("--archive-output", type=Path, required=True)
    parser.add_argument("--signed-archive-execution-path", required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument(
        "--openssl-bin",
        type=Path,
        default=Path(lock_signer.PINNED_OPENSSL_BINARY_PATH),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = sign_source_archive_release(
        repo_root=args.repo_root,
        source_commit=args.source_commit,
        member_lock_path=args.member_lock,
        archive_output=args.archive_output,
        signed_archive_execution_path=args.signed_archive_execution_path,
        receipt_output=args.receipt_output,
        private_key_path=args.private_key,
        openssl_bin=args.openssl_bin,
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
