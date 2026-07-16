#!/usr/bin/env python
"""Sign one SOMP-H authority lock with the externally held Ed25519 key."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import somph_lineage_authority as authority  # noqa: E402
from cvsrffi import somph_authority_lock_builder as lock_builder  # noqa: E402


SIGNING_RECEIPT_SCHEMA = "cvs.phase2.somph_authority_signing_receipt.v2"
PINNED_OPENSSL_BINARY_PATH = (
    r"F:\App\miniconda3\Library\bin\openssl.exe"
)
PINNED_OPENSSL_BINARY_SHA256 = (
    "2e081772a6cf076e43cd9a1fda5f26cfe5dd55d3e11dd1019081da4878c8ea5b"
)
_PINNED_OPENSSL_RUNTIME_SHA256 = {
    "libcrypto-3-x64.dll": (
        "2124998938de36d25dfa32adc8ffcdb324eac0cc98552a160a08e41f17aade9a"
    ),
    "libssl-3-x64.dll": (
        "a0d26c33b88d139f85a98ccd5dd583cbdf55392d0f6ed2bd6d6425bdc7382c8d"
    ),
}
_SIGNING_RECEIPT_KEYS = {
    "schema",
    "formal_launch_authority",
    "lock_file_sha256",
    "lock_canonical_sha256",
    "signed_authority_envelope_sha256",
    "pinned_authority_public_key_hex",
    "pinned_authority_public_key_sha256",
    "authority_key_id",
    "openssl_binary_sha256",
    "authority_lock_build_receipt_sha256",
    "cache_spec_manifest_sha256",
    "cache_spec_cell_id",
}


class SomphAuthoritySigningError(RuntimeError):
    """Raised when the external OpenSSL signing operation fails closed."""


def _json_line(payload: Mapping[str, Any]) -> bytes:
    return authority.canonical_json_bytes(dict(payload)) + b"\n"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_created(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_new_readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.chmod(path, 0o444)
        _fsync_directory(path.parent)
    except BaseException:
        if created:
            _remove_created(path)
        raise


def _resolved_regular_file(path: str | Path, *, context: str) -> Path:
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file():
        raise SomphAuthoritySigningError(f"{context} must be a regular file")
    return candidate


def _openssl_sha256(path: Path) -> str:
    _raw, digest, _size = authority._read_external_bytes(
        path,
        context="OpenSSL binary",
    )
    return digest


def _pinned_openssl_binary(
    requested: str | Path | None,
) -> tuple[Path, bytes, str, dict[str, tuple[bytes, str]]]:
    expected = _resolved_regular_file(
        PINNED_OPENSSL_BINARY_PATH,
        context="pinned OpenSSL binary",
    )
    candidate = _resolved_regular_file(
        (
            requested
            if requested is not None
            else PINNED_OPENSSL_BINARY_PATH
        ),
        context="OpenSSL binary",
    )
    if os.path.normcase(str(candidate)) != os.path.normcase(str(expected)):
        raise SomphAuthoritySigningError(
            "OpenSSL binary path is not pinned for this release"
        )
    raw, digest, _size = authority._read_external_bytes(
        candidate,
        context="OpenSSL binary",
    )
    if digest != PINNED_OPENSSL_BINARY_SHA256:
        raise SomphAuthoritySigningError(
            "pinned OpenSSL binary SHA256 mismatch"
        )
    runtime_files: dict[str, tuple[bytes, str]] = {}
    for name, expected_digest in _PINNED_OPENSSL_RUNTIME_SHA256.items():
        runtime_path = _resolved_regular_file(
            candidate.parent / name,
            context=f"pinned OpenSSL runtime {name}",
        )
        runtime_raw, runtime_digest, _runtime_size = (
            authority._read_external_bytes(
                runtime_path,
                context=f"pinned OpenSSL runtime {name}",
            )
        )
        if runtime_digest != expected_digest:
            raise SomphAuthoritySigningError(
                f"pinned OpenSSL runtime SHA256 mismatch: {name}"
            )
        runtime_files[name] = (runtime_raw, runtime_digest)
    return candidate, raw, digest, runtime_files


def _windows_lock_private_openssl_paths(
    directory: Path,
    files: tuple[Path, ...],
) -> list[int]:
    if os.name != "nt":
        return []

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    file_flag_backup_semantics = 0x02000000
    invalid_handle = ctypes.c_void_p(-1).value
    handles: list[int] = []
    try:
        paths_and_flags = [(directory, file_flag_backup_semantics)]
        paths_and_flags.extend(
            (path, file_attribute_normal) for path in files
        )
        for path, flags in paths_and_flags:
            handle = create_file(
                str(path),
                generic_read,
                file_share_read,
                None,
                open_existing,
                flags,
                None,
            )
            if handle in (None, invalid_handle):
                error = ctypes.get_last_error()
                raise SomphAuthoritySigningError(
                    "failed to lock private OpenSSL execution path"
                ) from OSError(error, os.strerror(error))
            handles.append(int(handle))
    except BaseException:
        for handle in reversed(handles):
            close_handle(ctypes.c_void_p(handle))
        raise
    return handles


def _close_windows_handles(handles: list[int]) -> None:
    if not handles:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    for handle in reversed(handles):
        close_handle(ctypes.c_void_p(handle))


@contextlib.contextmanager
def _private_openssl_executable(
    *,
    verified_bytes: bytes,
    expected_sha256: str,
    runtime_files: Mapping[str, tuple[bytes, str]],
) -> Iterator[Path]:
    directory = Path(tempfile.mkdtemp(prefix="somph-authority-openssl-"))
    executable = directory / "openssl.exe"
    created_files: list[Path] = []
    handles: list[int] = []
    ready_for_execution = False
    payloads = {"openssl.exe": (verified_bytes, expected_sha256)}
    payloads.update(dict(runtime_files))
    try:
        os.chmod(directory, 0o700)
        for name, (payload, _digest) in payloads.items():
            path = directory / name
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
            os.chmod(path, 0o500 if path == executable else 0o400)
            created_files.append(path)
        handles = _windows_lock_private_openssl_paths(
            directory,
            tuple(created_files),
        )
        for name, (_payload, digest) in payloads.items():
            if _openssl_sha256(directory / name) != digest:
                raise SomphAuthoritySigningError(
                    f"private OpenSSL runtime SHA256 mismatch: {name}"
                )
        ready_for_execution = True
        yield executable
    finally:
        integrity_error: BaseException | None = None
        if ready_for_execution:
            try:
                for name, (_payload, digest) in payloads.items():
                    if _openssl_sha256(directory / name) != digest:
                        raise SomphAuthoritySigningError(
                            "private OpenSSL runtime changed during "
                            f"signing: {name}"
                        )
            except BaseException as exc:
                integrity_error = exc
        _close_windows_handles(handles)
        for path in reversed(created_files):
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        if integrity_error is not None:
            raise integrity_error


def _clean_openssl_environment() -> dict[str, str]:
    allowed = (
        "HOME",
        "USERPROFILE",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
    )
    environment = {
        key: os.environ[key]
        for key in allowed
        if key in os.environ
    }
    environment["LC_ALL"] = "C"
    return environment


def _sign_with_openssl(
    *,
    openssl_binary: Path,
    private_key: Path,
    message: bytes,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="somph-authority-sign-") as temporary:
        message_path = Path(temporary) / "authority_message.bin"
        descriptor = os.open(
            message_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(message)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        try:
            completed = subprocess.run(
                [
                    str(openssl_binary),
                    "pkeyutl",
                    "-config",
                    os.devnull,
                    "-sign",
                    "-rawin",
                    "-inkey",
                    str(private_key),
                    "-in",
                    str(message_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_clean_openssl_environment(),
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SomphAuthoritySigningError(
                "OpenSSL Ed25519 signing failed"
            ) from exc
    if completed.returncode != 0 or len(completed.stdout) != 64:
        raise SomphAuthoritySigningError("OpenSSL Ed25519 signing failed")
    return bytes(completed.stdout)


def _validate_file_descriptor(
    value: Any,
    *,
    field: str,
    build_spec: bool = False,
) -> None:
    keys = (
        authority._BUILD_SPEC_DESCRIPTOR_KEYS
        if build_spec
        else authority._FILE_DESCRIPTOR_KEYS
    )
    descriptor = authority._require_exact_dict(value, keys, field=field)
    path = descriptor.get("path")
    if not isinstance(path, str) or not path:
        raise authority.SomphLineageAuthorityError(
            f"{field}.path must be nonempty"
        )
    authority._require_int(
        descriptor.get("size_bytes"),
        field=f"{field}.size_bytes",
    )
    if build_spec:
        authority._require_sha256(
            descriptor.get("file_sha256"),
            field=f"{field}.file_sha256",
        )
        authority._require_sha256(
            descriptor.get("canonical_sha256"),
            field=f"{field}.canonical_sha256",
        )
    else:
        authority._require_sha256(
            descriptor.get("sha256"),
            field=f"{field}.sha256",
        )


def _validate_lock_for_signing(lock: dict[str, Any]) -> None:
    if set(lock) != authority._LOCK_KEYS:
        raise authority.SomphLineageAuthorityError(
            "authority lock exact schema drift"
        )
    if lock.get("schema") != authority.AUTHORITY_LOCK_SCHEMA:
        raise authority.SomphLineageAuthorityError("authority lock schema drift")
    authority._validate_single_observation_contract(
        lock,
        field="authority lock for signing",
        require_audit_evidence=True,
    )
    _receiver, _seed, roles = authority._validate_lock_formal_identity(lock)
    _validate_file_descriptor(
        lock.get("cache_set_manifest"),
        field="cache_set_manifest",
    )
    _validate_file_descriptor(lock.get("exporter"), field="exporter")
    _validate_file_descriptor(
        lock.get("build_spec"),
        field="build_spec",
        build_spec=True,
    )
    for field in (
        "cache_sha256_by_scenario",
        "channel_config_sha256_by_scenario",
        "physical_sample_ids_sha256_by_scenario",
        "post_channel_iq_sha256_root_by_scenario",
        "overlay_ids_sha256_by_scenario",
    ):
        authority._scenario_sha_map(lock.get(field), field=field)
    for field in (
        "physical_sample_scenario_assignment_sha256",
        "cache_role_inputs_root_sha256",
    ):
        authority._require_sha256(lock.get(field), field=field)

    closure = authority._require_exact_dict(
        lock.get("channel_code_closure"),
        authority._CHANNEL_CLOSURE_KEYS,
        field="channel_code_closure",
    )
    authority._require_sha256(
        closure.get("closure_sha256"),
        field="channel_code_closure.closure_sha256",
    )
    members = closure.get("members")
    if (
        not isinstance(members, list)
        or len(members) != len(authority.CHANNEL_CODE_LOGICAL_MEMBERS)
    ):
        raise authority.SomphLineageAuthorityError(
            "channel_code_closure.members count drift"
        )
    for index, (member, logical_name) in enumerate(
        zip(members, authority.CHANNEL_CODE_LOGICAL_MEMBERS)
    ):
        descriptor = authority._require_exact_dict(
            member,
            authority._CHANNEL_MEMBER_KEYS,
            field=f"channel_code_closure.members[{index}]",
        )
        if descriptor.get("logical_name") != logical_name:
            raise authority.SomphLineageAuthorityError(
                "channel code logical member order drift"
            )
        _validate_file_descriptor(
            {
                "path": descriptor.get("path"),
                "sha256": descriptor.get("sha256"),
                "size_bytes": descriptor.get("size_bytes"),
            },
            field=f"channel_code_closure.members[{index}]",
        )

    datasets = lock.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != len(roles):
        raise authority.SomphLineageAuthorityError(
            "authority lock dataset count drift"
        )
    expected_tx_by_role = {
        "target_old": lock["old_tx_ids"],
        "target_new": lock["new_tx_ids"],
    }
    for index, (dataset, role) in enumerate(zip(datasets, roles)):
        descriptor = authority._require_exact_dict(
            dataset,
            authority._DATASET_KEYS,
            field=f"datasets[{index}]",
        )
        if (
            descriptor.get("role") != role
            or descriptor.get("tx_ids") != expected_tx_by_role[role]
        ):
            raise authority.SomphLineageAuthorityError(
                f"datasets[{index}] formal role/TX binding drift"
            )
        _validate_file_descriptor(
            {
                "path": descriptor.get("path"),
                "sha256": descriptor.get("sha256"),
                "size_bytes": descriptor.get("size_bytes"),
            },
            field=f"datasets[{index}]",
        )


def _validate_production_build_authority(
    lock: dict[str, Any],
    *,
    lock_file_sha256: str,
    lock_canonical_sha256: str,
    lock_build_receipt_path: str | Path,
    cache_spec_manifest_path: str | Path,
) -> dict[str, str]:
    try:
        receipt, _raw, receipt_sha, _size = authority._read_external_json(
            lock_build_receipt_path,
            context="SOMP-H authority lock build receipt",
        )
        manifest, _manifest_raw, manifest_sha, _manifest_size = (
            authority._read_external_json(
                cache_spec_manifest_path,
                context="SOMP-H locked cache-spec manifest for signing",
            )
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthoritySigningError(str(exc)) from exc
    if set(receipt) != lock_builder.AUTHORITY_LOCK_BUILD_RECEIPT_KEYS:
        raise SomphAuthoritySigningError(
            "authority lock build receipt exact schema drift"
        )
    expected_manifest_sha = lock_builder.FORMAL_CACHE_SPEC_MANIFEST_SHA256
    if (
        receipt.get("schema")
        != lock_builder.AUTHORITY_LOCK_BUILD_RECEIPT_SCHEMA
        or receipt.get("status") != lock_builder.AUTHORITY_LOCK_BUILD_STATUS
        or receipt.get("external_authority_lock_verified") is not False
        or receipt.get("formal_launch_authority") is not False
        or receipt.get("cache_spec_manifest_sha256")
        != expected_manifest_sha
        or manifest_sha != expected_manifest_sha
        or receipt.get("authority_lock_sha256") != lock_file_sha256
        or receipt.get("authority_lock_canonical_sha256")
        != lock_canonical_sha256
        or receipt.get("receiver") != lock.get("receiver")
        or receipt.get("seed") != lock.get("seed")
        or receipt.get("cache_scope") != "stage2_registered"
        or lock.get("cache_scope") != "stage2_registered"
        or receipt.get("required_samples_per_tx") != 40
        or receipt.get("cache_set_manifest_sha256")
        != lock.get("cache_set_manifest", {}).get("sha256")
        or receipt.get("build_spec_file_sha256")
        != lock.get("build_spec", {}).get("file_sha256")
        or receipt.get("build_spec_canonical_sha256")
        != lock.get("build_spec", {}).get("canonical_sha256")
        or receipt.get("exporter_sha256")
        != lock.get("exporter", {}).get("sha256")
        or receipt.get("channel_code_closure_sha256")
        != lock.get("channel_code_closure", {}).get("closure_sha256")
        or receipt.get("cache_role_inputs_root_sha256")
        != lock.get("cache_role_inputs_root_sha256")
        or receipt.get("physical_sample_ids_sha256_by_scenario")
        != lock.get("physical_sample_ids_sha256_by_scenario")
        or receipt.get("physical_sample_scenario_assignment_sha256")
        != lock.get("physical_sample_scenario_assignment_sha256")
        or receipt.get("cross_scenario_physical_disjointness_audit") != "PASS"
        or receipt.get("single_observation_contract_audit") != "PASS"
    ):
        raise SomphAuthoritySigningError(
            "authority lock build receipt/lock binding drift"
        )
    try:
        authority._validate_single_observation_contract(
            receipt,
            field="authority lock build receipt for signing",
            require_audit_evidence=True,
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthoritySigningError(str(exc)) from exc
    for field in (
        "cache_sha256_by_scenario",
        "channel_config_sha256_by_scenario",
        "post_channel_iq_sha256_root_by_scenario",
        "overlay_ids_sha256_by_scenario",
        "physical_sample_ids_sha256_by_scenario",
    ):
        if receipt.get(field) != lock.get(field):
            raise SomphAuthoritySigningError(
                f"authority lock build receipt root drift: {field}"
            )
    if (
        manifest.get("schema")
        != "cvs.phase2.somph_registered_cache_build_matrix.v2"
        or manifest.get("formal_launch_authority") is not False
        or manifest.get("required_samples_per_tx") != 40
    ):
        raise SomphAuthoritySigningError(
            "locked cache-spec manifest signing contract drift"
        )
    cell_id = receipt.get("cache_spec_cell_id")
    expected_cell_id = (
        f"rx_{str(lock.get('receiver')).replace('-', '_')}_seed_{lock.get('seed')}"
    )
    cells = manifest.get("cells")
    matches = [
        item
        for item in cells
        if isinstance(item, dict) and item.get("cell_id") == cell_id
    ] if isinstance(cells, list) else []
    if (
        not isinstance(cell_id, str)
        or cell_id != expected_cell_id
        or len(matches) != 1
    ):
        raise SomphAuthoritySigningError(
            "locked cache-spec cell signing identity drift"
        )
    cell = matches[0]
    if (
        cell.get("receiver") != lock.get("receiver")
        or cell.get("seed") != lock.get("seed")
        or cell.get("cache_scope") != "stage2_registered"
        or cell.get("required_samples_per_tx") != 40
        or cell.get("spec_file_sha256")
        != lock.get("build_spec", {}).get("file_sha256")
        or cell.get("spec_canonical_sha256")
        != lock.get("build_spec", {}).get("canonical_sha256")
    ):
        raise SomphAuthoritySigningError(
            "locked cache-spec cell/build-spec signing drift"
        )
    return {
        "authority_lock_build_receipt_sha256": receipt_sha,
        "cache_spec_manifest_sha256": manifest_sha,
        "cache_spec_cell_id": cell_id,
    }
def _sign_authority_lock_impl(
    authority_lock_path: str | Path,
    *,
    private_key_path: str | Path,
    openssl_bin: str | Path,
    envelope_output: str | Path,
    receipt_output: str | Path,
    verify_signed_envelope: Callable[..., None],
    receipt_public_key_hex: str,
    receipt_public_key_sha256: str,
    build_authority_binding: Mapping[str, str],
    expected_lock_file_sha256: str,
    expected_lock_canonical_sha256: str,
) -> dict[str, Any]:
    """Sign and publish an exact pinned-authority envelope plus detached receipt."""

    envelope_path = Path(envelope_output)
    receipt_path = Path(receipt_output)
    if envelope_path.resolve(strict=False) == receipt_path.resolve(strict=False):
        raise SomphAuthoritySigningError(
            "envelope and signing receipt outputs must be distinct"
        )
    if envelope_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "refusing to overwrite signed authority envelope or signing receipt"
        )

    lock, _lock_bytes, lock_file_sha, _lock_size = authority._read_external_json(
        authority_lock_path,
        context="SOMP-H authority lock for signing",
    )
    _validate_lock_for_signing(lock)
    lock_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(lock)
    )
    if (
        lock_file_sha != expected_lock_file_sha256
        or lock_canonical_sha != expected_lock_canonical_sha256
    ):
        raise SomphAuthoritySigningError(
            "authority lock changed after production preflight"
        )
    binding_keys = {
        "authority_lock_build_receipt_sha256",
        "cache_spec_manifest_sha256",
        "cache_spec_cell_id",
    }
    if set(build_authority_binding) != binding_keys:
        raise SomphAuthoritySigningError(
            "signing build-authority binding exact schema drift"
        )
    authority._require_sha256(
        build_authority_binding["authority_lock_build_receipt_sha256"],
        field="authority_lock_build_receipt_sha256",
    )
    authority._require_sha256(
        build_authority_binding["cache_spec_manifest_sha256"],
        field="cache_spec_manifest_sha256",
    )
    if not build_authority_binding["cache_spec_cell_id"]:
        raise SomphAuthoritySigningError(
            "cache_spec_cell_id binding missing"
        )

    (
        _,
        openssl_verified_bytes,
        openssl_sha_before,
        openssl_runtime_files,
    ) = _pinned_openssl_binary(openssl_bin)
    private_key = _resolved_regular_file(
        private_key_path,
        context="Ed25519 private key",
    )

    envelope: dict[str, Any] = {
        "schema": authority.AUTHORITY_ENVELOPE_SCHEMA,
        "domain": authority.AUTHORITY_SIGNATURE_DOMAIN,
        "issuer": authority.PINNED_AUTHORITY_ISSUER,
        "key_id": authority.PINNED_AUTHORITY_KEY_ID,
        "lock_canonical_sha256": lock_canonical_sha,
        "authority_lock_build_receipt_sha256": (
            build_authority_binding[
                "authority_lock_build_receipt_sha256"
            ]
        ),
        "cache_spec_manifest_sha256": build_authority_binding[
            "cache_spec_manifest_sha256"
        ],
        "cache_spec_cell_id": build_authority_binding[
            "cache_spec_cell_id"
        ],
        "signature_ed25519_hex": "",
    }
    with _private_openssl_executable(
        verified_bytes=openssl_verified_bytes,
        expected_sha256=openssl_sha_before,
        runtime_files=openssl_runtime_files,
    ) as private_openssl:
        signature = _sign_with_openssl(
            openssl_binary=private_openssl,
            private_key=private_key,
            message=authority._authority_signature_message(envelope),
        )
    openssl_sha_after = openssl_sha_before
    envelope["signature_ed25519_hex"] = signature.hex()
    verify_signed_envelope(
        envelope,
        lock_canonical_sha256=lock_canonical_sha,
        expected_cache_spec_cell_id=build_authority_binding[
            "cache_spec_cell_id"
        ],
        expected_build_receipt_sha256=build_authority_binding[
            "authority_lock_build_receipt_sha256"
        ],
    )
    try:
        receipt_public_key = bytes.fromhex(receipt_public_key_hex)
        signature_bytes = bytes.fromhex(envelope["signature_ed25519_hex"])
    except ValueError as exc:
        raise SomphAuthoritySigningError(
            "receipt public-key identity is not valid hex"
        ) from exc
    if (
        hashlib.sha256(receipt_public_key).hexdigest()
        != receipt_public_key_sha256
    ):
        raise SomphAuthoritySigningError(
            "receipt public-key identity SHA256 mismatch"
        )
    authority._verify_ed25519(
        receipt_public_key,
        authority._authority_signature_message(envelope),
        signature_bytes,
    )

    envelope_bytes = _json_line(envelope)
    envelope_sha = hashlib.sha256(envelope_bytes).hexdigest()
    receipt = {
        "schema": SIGNING_RECEIPT_SCHEMA,
        "formal_launch_authority": False,
        "lock_file_sha256": lock_file_sha,
        "lock_canonical_sha256": lock_canonical_sha,
        "signed_authority_envelope_sha256": envelope_sha,
        "pinned_authority_public_key_hex": (
            receipt_public_key_hex
        ),
        "pinned_authority_public_key_sha256": (
            receipt_public_key_sha256
        ),
        "authority_key_id": authority.PINNED_AUTHORITY_KEY_ID,
        "openssl_binary_sha256": openssl_sha_after,
        **dict(build_authority_binding),
    }
    if set(receipt) != _SIGNING_RECEIPT_KEYS:
        raise AssertionError("signing receipt exact schema drift")
    receipt_bytes = _json_line(receipt)

    created: list[Path] = []
    try:
        _write_new_readonly(envelope_path, envelope_bytes)
        created.append(envelope_path)
        _write_new_readonly(receipt_path, receipt_bytes)
        created.append(receipt_path)
    except BaseException:
        for path in reversed(created):
            _remove_created(path)
        raise

    return {
        "signed_authority_envelope": str(envelope_path),
        "signed_authority_envelope_sha256": envelope_sha,
        "signing_receipt": str(receipt_path),
        "signing_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "lock_file_sha256": lock_file_sha,
        "lock_canonical_sha256": lock_canonical_sha,
    }


def sign_authority_lock(
    authority_lock_path: str | Path,
    *,
    private_key_path: str | Path,
    openssl_bin: str | Path,
    envelope_output: str | Path,
    receipt_output: str | Path,
    lock_build_receipt_path: str | Path,
    cache_spec_manifest_path: str | Path,
) -> dict[str, Any]:
    """Production signer; identity and verifier are never caller-selectable."""

    lock, _lock_raw, lock_file_sha, _lock_size = authority._read_external_json(
        authority_lock_path,
        context="SOMP-H authority lock for production signing preflight",
    )
    _validate_lock_for_signing(lock)
    lock_canonical_sha = authority.sha256_bytes(
        authority.canonical_json_bytes(lock)
    )
    build_binding = _validate_production_build_authority(
        lock,
        lock_file_sha256=lock_file_sha,
        lock_canonical_sha256=lock_canonical_sha,
        lock_build_receipt_path=lock_build_receipt_path,
        cache_spec_manifest_path=cache_spec_manifest_path,
    )
    return _sign_authority_lock_impl(
        authority_lock_path,
        private_key_path=private_key_path,
        openssl_bin=openssl_bin,
        envelope_output=envelope_output,
        receipt_output=receipt_output,
        verify_signed_envelope=authority._verify_signed_envelope,
        receipt_public_key_hex=authority.PINNED_AUTHORITY_PUBLIC_KEY_HEX,
        receipt_public_key_sha256=authority.PINNED_AUTHORITY_PUBLIC_KEY_SHA256,
        build_authority_binding=build_binding,
        expected_lock_file_sha256=lock_file_sha,
        expected_lock_canonical_sha256=lock_canonical_sha,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-lock", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument(
        "--openssl-bin",
        type=Path,
        default=Path(PINNED_OPENSSL_BINARY_PATH),
    )
    parser.add_argument("--envelope-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--lock-build-receipt", type=Path, required=True)
    parser.add_argument("--cache-spec-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = sign_authority_lock(
        args.authority_lock,
        private_key_path=args.private_key,
        openssl_bin=args.openssl_bin,
        envelope_output=args.envelope_output,
        receipt_output=args.receipt_output,
        lock_build_receipt_path=args.lock_build_receipt,
        cache_spec_manifest_path=args.cache_spec_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
