"""Externally locked authority wrapper for SOMP-H LEO_weak lineage.

This module remains outside the Phase2 predictor boundary.  It accepts one
pre-locked authority document plus its externally supplied byte SHA, validates
the actual Phase1 inputs, asks the structural lineage producer to recompute the
sample lineage, and publishes one committed directory atomically.
"""

from __future__ import annotations

import json
import ctypes
import errno
import hashlib
import os
import secrets
import shutil
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cvsrffi import somph_leo_weak_lineage_seal as structural
from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS, canonical_json_sha256
from cvsrffi.somph_formal_matrix import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEED,
    FORMAL_NEW_CLASS_COUNTS,
    FORMAL_RECEIVERS,
    NEW_TX_IDS,
    OLD_TX_IDS,
)
from cvsrffi.stage2_predictor_bundle import (
    PredictorPackageError,
    _hash_handle,
    _json_from_handle,
    _validate_package_root_exact_allowlist,
    _zip_members_from_handle,
    canonical_json_bytes,
    open_regular_member_same_fd,
    sha256_bytes,
)
from training_controls import sat_channel_config_for_scenario


AUTHORITY_LOCK_SCHEMA = "cvs.phase2.somph_leo_weak_authority_lock.v1"
AUTHORITY_ATTESTATION_SCHEMA = (
    "cvs.phase2.somph_leo_weak_authority_attestation.v1"
)
AUTHORITY_COMMIT_SCHEMA = "cvs.phase2.somph_leo_weak_authority_commit.v1"
AUTHORITY_ENVELOPE_SCHEMA = "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1"
AUTHORITY_STATUS = "EXTERNAL_AUTHORITY_LOCK_VERIFIED"
AUTHORITY_SIGNATURE_DOMAIN = "cvs.somph.leo_weak.authority_lock.ed25519.v1"
PINNED_AUTHORITY_ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
PINNED_AUTHORITY_KEY_ID = "somph-authority-ed25519-20260716"
PINNED_AUTHORITY_PUBLIC_KEY_HEX = (
    "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
)
PINNED_AUTHORITY_PUBLIC_KEY_SHA256 = (
    "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
)

AUTHORITY_LOCK_NAME = "authority_lock.json"
AUTHORITY_ENVELOPE_NAME = "signed_authority_envelope.json"
AUTHORITY_LOCK_BUILD_RECEIPT_NAME = "authority_lock_build_receipt.json"
CACHE_SPEC_MANIFEST_NAME = "cache_spec_manifest.json"
STRUCTURAL_RECEIPT_NAME = "structural_receipt.json"
STRUCTURAL_SEAL_NAME = "structural_receipt.seal.json"
AUTHORITY_ATTESTATION_NAME = "authority_attestation.json"
AUTHORITY_COMMIT_NAME = "COMMIT.json"

CHANNEL_CODE_LOGICAL_MEMBERS = (
    "cvsrffi_eval.py",
    "cvsrffi_tensors.py",
    "sat_channel.py",
    "training_controls.py",
)

_LOCK_KEYS = {
    "schema",
    "receiver",
    "seed",
    "cache_scope",
    "old_tx_ids",
    "new_tx_ids",
    "cache_set_manifest",
    "cache_sha256_by_scenario",
    "exporter",
    "build_spec",
    "channel_code_closure",
    "channel_config_sha256_by_scenario",
    "physical_sample_ids_sha256",
    "post_channel_iq_sha256_root_by_scenario",
    "overlay_ids_sha256_by_scenario",
    "cache_role_inputs_root_sha256",
    "datasets",
}
_FILE_DESCRIPTOR_KEYS = {"path", "sha256", "size_bytes"}
_BUILD_SPEC_DESCRIPTOR_KEYS = {
    "path",
    "file_sha256",
    "canonical_sha256",
    "size_bytes",
}
_CHANNEL_CLOSURE_KEYS = {"closure_sha256", "members"}
_CHANNEL_MEMBER_KEYS = {"logical_name", "path", "sha256", "size_bytes"}
_DATASET_KEYS = {"role", "path", "sha256", "size_bytes", "tx_ids"}
_BUILD_SPEC_REQUIRED_KEYS = {
    "schema",
    "cache_set_id",
    "cache_scope",
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "star_ground_channel_impl",
    "role_specs",
    "dataset_seed",
    "satellite_seed_by_scenario",
    "out_npz_by_scenario",
    "out_manifest",
    "batch_size",
    "wisig_out_len",
    "wisig_equalized",
    "wisig_domain",
}
_BUILD_SPEC_OPTIONAL_KEYS = {"sat_fs_hz", "sat_fc_hz"}
_ROLE_SPEC_REQUIRED_KEYS = {"role", "pkl", "tx_ids", "rxs"}
_ROLE_SPEC_OPTIONAL_KEYS = {
    "days",
    "max_samples_per_combo",
    "max_samples_per_tx",
}
_ROLE_INPUT_KEYS = {
    "role",
    "dataset_sha256",
    "dataset_size_bytes",
    "requested_tx_ids",
    "requested_rxs",
    "requested_days",
    "dataset_seed",
    "resolved_info",
    "physical_sample_count",
}
_BUNDLE_MEMBERS = {
    AUTHORITY_LOCK_NAME,
    AUTHORITY_ENVELOPE_NAME,
    AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
    CACHE_SPEC_MANIFEST_NAME,
    STRUCTURAL_RECEIPT_NAME,
    STRUCTURAL_SEAL_NAME,
    AUTHORITY_ATTESTATION_NAME,
    AUTHORITY_COMMIT_NAME,
}
_ENVELOPE_KEYS = {
    "schema",
    "domain",
    "issuer",
    "key_id",
    "lock_canonical_sha256",
    "authority_lock_build_receipt_sha256",
    "cache_spec_manifest_sha256",
    "cache_spec_cell_id",
    "signature_ed25519_hex",
}
_AUTHORITY_LOCK_BUILD_RECEIPT_KEYS = {
    "schema",
    "status",
    "cache_spec_manifest_sha256",
    "cache_spec_manifest_size_bytes",
    "cache_spec_cell_id",
    "required_samples_per_tx",
    "receiver",
    "seed",
    "cache_scope",
    "cache_set_manifest_sha256",
    "build_spec_file_sha256",
    "build_spec_canonical_sha256",
    "exporter_sha256",
    "channel_code_closure_sha256",
    "dataset_authority_root_sha256",
    "cache_role_inputs_root_sha256",
    "physical_sample_ids_sha256",
    "cache_sha256_by_scenario",
    "channel_config_sha256_by_scenario",
    "post_channel_iq_sha256_root_by_scenario",
    "overlay_ids_sha256_by_scenario",
    "cache_recompute_audits",
    "authority_lock_sha256",
    "authority_lock_canonical_sha256",
    "external_authority_lock_verified",
    "formal_launch_authority",
}
_FORMAL_CACHE_SPEC_MANIFEST_SHA256 = (
    "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
)
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

_ED_Q = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_Q - 2, _ED_Q)) % _ED_Q
_ED_I = pow(2, (_ED_Q - 1) // 4, _ED_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_Q - 2, _ED_Q) % _ED_Q
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q:
        x = x * _ED_I % _ED_Q
    if x & 1:
        x = _ED_Q - x
    return x


_ED_BY = 4 * pow(5, _ED_Q - 2, _ED_Q) % _ED_Q
_ED_BX = _ed_xrecover(_ED_BY)
_ED_B = (_ED_BX, _ED_BY)
_ED_IDENTITY = (0, 1)


def _ed_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    common = _ED_D * x1 * x2 * y1 * y2 % _ED_Q
    return (
        (x1 * y2 + x2 * y1) * pow(1 + common, _ED_Q - 2, _ED_Q) % _ED_Q,
        (y1 * y2 + x1 * x2) * pow(1 - common, _ED_Q - 2, _ED_Q) % _ED_Q,
    )


def _ed_scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _ED_IDENTITY
    addend = point
    value = scalar
    while value:
        if value & 1:
            result = _ed_add(result, addend)
        addend = _ed_add(addend, addend)
        value >>= 1
    return result


def _ed_encode(point: tuple[int, int]) -> bytes:
    x, y = point
    value = y | ((x & 1) << 255)
    return value.to_bytes(32, "little")


def _ed_decode(value: bytes) -> tuple[int, int]:
    if len(value) != 32:
        raise SomphLineageAuthorityError("Ed25519 point length drift")
    raw = int.from_bytes(value, "little")
    y = raw & ((1 << 255) - 1)
    if y >= _ED_Q:
        raise SomphLineageAuthorityError("Ed25519 point is non-canonical")
    x = _ed_xrecover(y)
    if (x & 1) != (raw >> 255):
        x = _ED_Q - x
    point = (x, y)
    if _ed_encode(point) != value:
        raise SomphLineageAuthorityError("Ed25519 point encoding drift")
    return point


def _verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        raise SomphLineageAuthorityError("Ed25519 key/signature length drift")
    public_point = _ed_decode(public_key)
    r_point = _ed_decode(signature[:32])
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED_L:
        raise SomphLineageAuthorityError("Ed25519 signature scalar drift")
    if (
        public_point == _ED_IDENTITY
        or _ed_scalar_mult(public_point, _ED_L) != _ED_IDENTITY
    ):
        raise SomphLineageAuthorityError("Ed25519 public key subgroup drift")
    challenge = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
    ) % _ED_L
    left = _ed_scalar_mult(_ED_B, (8 * scalar) % _ED_L)
    right = _ed_add(
        _ed_scalar_mult(r_point, 8),
        _ed_scalar_mult(public_point, (8 * challenge) % _ED_L),
    )
    if _ed_encode(left) != _ed_encode(right):
        raise SomphLineageAuthorityError("Ed25519 authority signature invalid")


def _authority_signature_message(envelope: Mapping[str, Any]) -> bytes:
    signed = {
        "schema": envelope["schema"],
        "domain": envelope["domain"],
        "issuer": envelope["issuer"],
        "key_id": envelope["key_id"],
        "lock_canonical_sha256": envelope["lock_canonical_sha256"],
        "authority_lock_build_receipt_sha256": envelope[
            "authority_lock_build_receipt_sha256"
        ],
        "cache_spec_manifest_sha256": envelope[
            "cache_spec_manifest_sha256"
        ],
        "cache_spec_cell_id": envelope["cache_spec_cell_id"],
    }
    return (
        b"cvs.somph.leo_weak.authority_lock.ed25519.v1"
        + b"\x00"
        + canonical_json_bytes(signed)
    )


def _verify_signed_envelope(
    envelope: dict[str, Any],
    *,
    lock_canonical_sha256: str,
    expected_cache_spec_cell_id: str,
    expected_build_receipt_sha256: str | None = None,
) -> None:
    if set(envelope) != _ENVELOPE_KEYS:
        raise SomphLineageAuthorityError("signed authority envelope exact schema drift")
    expected = {
        "schema": "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1",
        "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
        "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
        "key_id": "somph-authority-ed25519-20260716",
        "lock_canonical_sha256": lock_canonical_sha256,
        "cache_spec_manifest_sha256": (
            "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
        ),
        "cache_spec_cell_id": expected_cache_spec_cell_id,
    }
    if any(envelope.get(key) != value for key, value in expected.items()):
        raise SomphLineageAuthorityError(
            "signed authority envelope pinned identity/binding drift"
        )
    receipt_sha = _require_sha256(
        envelope.get("authority_lock_build_receipt_sha256"),
        field="signed envelope authority_lock_build_receipt_sha256",
    )
    if (
        expected_build_receipt_sha256 is not None
        and receipt_sha != expected_build_receipt_sha256
    ):
        raise SomphLineageAuthorityError(
            "signed authority envelope build receipt binding drift"
        )
    try:
        public_key = bytes.fromhex(
            "ec301433b5a625f8e34f887f5aeea664"
            "e809236d1b871fcc0ffeb47cb540bdc1"
        )
        signature = bytes.fromhex(str(envelope.get("signature_ed25519_hex", "")))
    except ValueError as exc:
        raise SomphLineageAuthorityError(
            "signed authority envelope hex invalid"
        ) from exc
    if hashlib.sha256(public_key).hexdigest() != (
        "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
    ):
        raise SomphLineageAuthorityError("pinned authority public key SHA drift")
    _verify_ed25519(
        public_key,
        _authority_signature_message(envelope),
        signature,
    )


class SomphLineageAuthorityError(PredictorPackageError):
    """Raised when an externally locked authority bundle fails closed."""


def _require_sha256(value: Any, *, field: str) -> str:
    try:
        return structural._require_sha256(value, field=field)
    except structural.SomphLineageError as exc:
        raise SomphLineageAuthorityError(str(exc)) from exc


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SomphLineageAuthorityError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_exact_dict(value: Any, keys: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SomphLineageAuthorityError(f"{field} exact schema drift")
    return dict(value)


def _scenario_sha_map(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or tuple(value) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphLineageAuthorityError(
            f"{field} must use the exact formal scenario order"
        )
    return {
        scenario: _require_sha256(value[scenario], field=f"{field}.{scenario}")
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }


def _read_external_bytes(
    path: str | Path, *, context: str
) -> tuple[bytes, str, int]:
    try:
        with structural._open_external_same_fd(path) as handle:
            digest, size = _hash_handle(handle)
            raw = handle.read()
    except structural.SomphLineageError as exc:
        raise SomphLineageAuthorityError(str(exc)) from exc
    if len(raw) != size:
        raise SomphLineageAuthorityError(f"{context} changed during same-FD read")
    return raw, digest, size


def _read_external_json(
    path: str | Path, *, context: str
) -> tuple[dict[str, Any], bytes, str, int]:
    raw, digest, size = _read_external_bytes(path, context=context)
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphLineageAuthorityError(f"invalid JSON for {context}") from exc
    if not isinstance(value, dict):
        raise SomphLineageAuthorityError(f"{context} root must be an object")
    return value, raw, digest, size


def _verify_file_descriptor(
    value: Any,
    *,
    field: str,
    build_spec: bool = False,
) -> tuple[dict[str, Any], bytes]:
    keys = _BUILD_SPEC_DESCRIPTOR_KEYS if build_spec else _FILE_DESCRIPTOR_KEYS
    descriptor = _require_exact_dict(value, keys, field=field)
    path = descriptor.get("path")
    if not isinstance(path, str) or not path:
        raise SomphLineageAuthorityError(f"{field}.path must be nonempty")
    expected_size = _require_int(
        descriptor.get("size_bytes"), field=f"{field}.size_bytes"
    )
    raw, digest, size = _read_external_bytes(path, context=field)
    expected_file_sha = _require_sha256(
        descriptor.get("file_sha256" if build_spec else "sha256"),
        field=f"{field}.file_sha256" if build_spec else f"{field}.sha256",
    )
    if digest != expected_file_sha or size != expected_size:
        raise SomphLineageAuthorityError(f"{field} external byte binding mismatch")
    return descriptor, raw


def _split_tx_ids(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, str):
        raise SomphLineageAuthorityError(f"{field} must be a comma-separated string")
    result = [item.strip() for item in value.split(",")]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise SomphLineageAuthorityError(f"{field} contains empty or duplicate TX IDs")
    return result


def _expected_roles(scope: str) -> tuple[str, ...]:
    if scope == "stage2_target_old":
        return ("target_old",)
    if scope == "stage2_registered":
        return ("target_old", "target_new")
    raise SomphLineageAuthorityError("authority lock cache_scope is not formal")


def _validate_lock_formal_identity(lock: dict[str, Any]) -> tuple[str, int, tuple[str, ...]]:
    receiver = lock.get("receiver")
    if receiver not in FORMAL_RECEIVERS:
        raise SomphLineageAuthorityError("authority lock receiver is not formal")
    seed = lock.get("seed")
    if seed not in (DEVELOPMENT_SEED, *CONFIRMATION_SEEDS):
        raise SomphLineageAuthorityError("authority lock seed is not formal")
    old_tx = lock.get("old_tx_ids")
    new_tx = lock.get("new_tx_ids")
    if old_tx != list(OLD_TX_IDS):
        raise SomphLineageAuthorityError("authority lock old TX registry drift")
    if not isinstance(new_tx, list) or any(not isinstance(item, str) for item in new_tx):
        raise SomphLineageAuthorityError("authority lock new TX registry invalid")
    scope = lock.get("cache_scope")
    roles = _expected_roles(str(scope))
    if scope == "stage2_target_old":
        if new_tx:
            raise SomphLineageAuthorityError(
                "target-old authority lock must not include new TXs"
            )
    elif (
        len(new_tx) not in FORMAL_NEW_CLASS_COUNTS
        or new_tx != list(NEW_TX_IDS[: len(new_tx)])
    ):
        raise SomphLineageAuthorityError(
            "registered authority lock must use a formal nested new-TX prefix"
        )
    return str(receiver), int(seed), roles


def _validate_build_spec(
    payload: dict[str, Any],
    *,
    lock: dict[str, Any],
    receiver: str,
    seed: int,
    roles: tuple[str, ...],
    build_spec_dir: Path | None = None,
) -> list[dict[str, Any]]:
    keys = set(payload)
    if (
        not _BUILD_SPEC_REQUIRED_KEYS.issubset(keys)
        or keys - (_BUILD_SPEC_REQUIRED_KEYS | _BUILD_SPEC_OPTIONAL_KEYS)
    ):
        raise SomphLineageAuthorityError("real build spec exact schema drift")
    expected = {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v1",
        "cache_scope": lock["cache_scope"],
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        "dataset_seed": seed,
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
    }
    failed = [key for key, expected_value in expected.items() if payload.get(key) != expected_value]
    if failed:
        raise SomphLineageAuthorityError(f"real build spec contract failed: {failed}")
    if not isinstance(payload.get("cache_set_id"), str) or not payload["cache_set_id"]:
        raise SomphLineageAuthorityError("real build spec cache_set_id missing")
    _require_int(payload.get("batch_size"), field="build_spec.batch_size", minimum=1)
    if int(payload["batch_size"]) > 4096:
        raise SomphLineageAuthorityError("real build spec batch_size exceeds 4096")
    _require_int(payload.get("wisig_out_len"), field="build_spec.wisig_out_len", minimum=1)
    for optional_float in ("sat_fs_hz", "sat_fc_hz"):
        if optional_float in payload and (
            isinstance(payload[optional_float], bool)
            or not isinstance(payload[optional_float], (int, float))
            or float(payload[optional_float]) <= 0
        ):
            raise SomphLineageAuthorityError(
                f"real build spec {optional_float} must be positive"
            )
    for field in ("satellite_seed_by_scenario", "out_npz_by_scenario"):
        mapping = payload.get(field)
        if not isinstance(mapping, dict) or tuple(mapping) != FORMAL_LEO_WEAK_SCENARIOS:
            raise SomphLineageAuthorityError(
                f"real build spec {field} scenario order drift"
            )
    if any(
        _require_int(
            payload["satellite_seed_by_scenario"][scenario],
            field=f"build_spec.satellite_seed_by_scenario.{scenario}",
        )
        < 0
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphLineageAuthorityError("real build spec satellite seed invalid")
    if any(
        not isinstance(payload["out_npz_by_scenario"][scenario], str)
        or not payload["out_npz_by_scenario"][scenario]
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphLineageAuthorityError("real build spec output path missing")
    if not isinstance(payload.get("out_manifest"), str) or not payload["out_manifest"]:
        raise SomphLineageAuthorityError("real build spec out_manifest missing")

    role_specs = payload.get("role_specs")
    if not isinstance(role_specs, list) or len(role_specs) != len(roles):
        raise SomphLineageAuthorityError("real build spec role_specs count drift")
    checked: list[dict[str, Any]] = []
    expected_tx_by_role = {
        "target_old": list(lock["old_tx_ids"]),
        "target_new": list(lock["new_tx_ids"]),
    }
    for index, (raw, expected_role) in enumerate(zip(role_specs, roles)):
        if not isinstance(raw, dict):
            raise SomphLineageAuthorityError("real build spec role spec must be an object")
        role_keys = set(raw)
        if (
            not _ROLE_SPEC_REQUIRED_KEYS.issubset(role_keys)
            or role_keys - (_ROLE_SPEC_REQUIRED_KEYS | _ROLE_SPEC_OPTIONAL_KEYS)
        ):
            raise SomphLineageAuthorityError(
                f"real build spec role exact schema drift: {expected_role}"
            )
        role = dict(raw)
        if role.get("role") != expected_role:
            raise SomphLineageAuthorityError("real build spec role ordering drift")
        if _split_tx_ids(
            role.get("tx_ids"), field=f"build_spec.role_specs[{index}].tx_ids"
        ) != expected_tx_by_role[expected_role]:
            raise SomphLineageAuthorityError(
                f"real build spec TX registry drift for {expected_role}"
            )
        if role.get("rxs") != receiver:
            raise SomphLineageAuthorityError(
                f"real build spec receiver drift for {expected_role}"
            )
        if not isinstance(role.get("pkl"), str) or not role["pkl"]:
            raise SomphLineageAuthorityError(
                f"real build spec dataset path missing for {expected_role}"
            )
        for count_field in ("max_samples_per_combo", "max_samples_per_tx"):
            if count_field in role:
                _require_int(
                    role[count_field],
                    field=f"build_spec.role_specs[{index}].{count_field}",
                )
        raw_pkl = Path(role["pkl"])
        if not raw_pkl.is_absolute():
            raw_pkl = (build_spec_dir or Path.cwd()) / raw_pkl
        role["pkl"] = str(raw_pkl.absolute())
        checked.append(role)
    return checked


def _validate_datasets(
    value: Any,
    *,
    roles: tuple[str, ...],
    role_specs: list[dict[str, Any]],
    receiver: str,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list) or len(value) != len(roles):
        raise SomphLineageAuthorityError("authority lock dataset count drift")
    checked: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    for index, (raw, role, role_spec) in enumerate(zip(value, roles, role_specs)):
        dataset = _require_exact_dict(raw, _DATASET_KEYS, field=f"datasets[{index}]")
        if dataset.get("role") != role:
            raise SomphLineageAuthorityError("authority lock dataset role ordering drift")
        if dataset.get("tx_ids") != _split_tx_ids(
            role_spec["tx_ids"], field=f"datasets[{index}].tx_ids"
        ):
            raise SomphLineageAuthorityError(
                f"authority lock dataset TX binding drift for {role}"
            )
        descriptor = {
            "path": dataset.get("path"),
            "sha256": dataset.get("sha256"),
            "size_bytes": dataset.get("size_bytes"),
        }
        checked_descriptor, _raw = _verify_file_descriptor(
            descriptor, field=f"datasets[{index}]"
        )
        if Path(str(checked_descriptor["path"])).resolve() != Path(
            str(role_spec["pkl"])
        ).resolve():
            raise SomphLineageAuthorityError(
                f"build spec/authority dataset path drift for {role}"
            )
        checked.append(dataset)
        root_rows.append(
            {
                "role": role,
                "sha256": dataset["sha256"],
                "size_bytes": dataset["size_bytes"],
                "tx_ids": dataset["tx_ids"],
                "receiver": receiver,
                "dataset_seed": seed + index * 10_007,
            }
        )
    return checked, sha256_bytes(canonical_json_bytes(root_rows))


def _dataset_root_from_lock(
    lock: Mapping[str, Any],
    *,
    receiver: str,
    seed: int,
    roles: tuple[str, ...],
) -> str:
    values = lock.get("datasets")
    if not isinstance(values, list) or len(values) != len(roles):
        raise SomphLineageAuthorityError("committed authority dataset count drift")
    rows: list[dict[str, Any]] = []
    expected_tx = {
        "target_old": list(lock["old_tx_ids"]),
        "target_new": list(lock["new_tx_ids"]),
    }
    for index, (raw, role) in enumerate(zip(values, roles)):
        dataset = _require_exact_dict(
            raw, _DATASET_KEYS, field=f"committed datasets[{index}]"
        )
        if dataset.get("role") != role or dataset.get("tx_ids") != expected_tx[role]:
            raise SomphLineageAuthorityError(
                "committed authority dataset role/TX binding drift"
            )
        if not isinstance(dataset.get("path"), str) or not dataset["path"]:
            raise SomphLineageAuthorityError(
                "committed authority dataset path missing"
            )
        _require_sha256(
            dataset.get("sha256"),
            field=f"committed datasets[{index}].sha256",
        )
        _require_int(
            dataset.get("size_bytes"),
            field=f"committed datasets[{index}].size_bytes",
        )
        rows.append(
            {
                "role": role,
                "sha256": dataset["sha256"],
                "size_bytes": dataset["size_bytes"],
                "tx_ids": dataset["tx_ids"],
                "receiver": receiver,
                "dataset_seed": seed + index * 10_007,
            }
        )
    return sha256_bytes(canonical_json_bytes(rows))


def _validate_channel_closure(value: Any) -> tuple[dict[str, Path], str]:
    closure = _require_exact_dict(
        value, _CHANNEL_CLOSURE_KEYS, field="channel_code_closure"
    )
    expected_closure = _require_sha256(
        closure.get("closure_sha256"),
        field="channel_code_closure.closure_sha256",
    )
    members = closure.get("members")
    if not isinstance(members, list) or len(members) != len(
        CHANNEL_CODE_LOGICAL_MEMBERS
    ):
        raise SomphLineageAuthorityError("channel code closure member count drift")
    paths: dict[str, Path] = {}
    descriptors: list[dict[str, Any]] = []
    for index, (raw, logical_name) in enumerate(
        zip(members, CHANNEL_CODE_LOGICAL_MEMBERS)
    ):
        member = _require_exact_dict(
            raw, _CHANNEL_MEMBER_KEYS, field=f"channel_code_closure.members[{index}]"
        )
        if member.get("logical_name") != logical_name:
            raise SomphLineageAuthorityError(
                "channel code logical member allowlist/order drift"
            )
        descriptor = {
            "path": member.get("path"),
            "sha256": member.get("sha256"),
            "size_bytes": member.get("size_bytes"),
        }
        checked, _raw_bytes = _verify_file_descriptor(
            descriptor, field=f"channel_code_closure.members[{index}]"
        )
        paths[logical_name] = Path(str(checked["path"]))
        descriptors.append(
            {
                "logical_name": logical_name,
                "sha256": checked["sha256"],
                "size_bytes": checked["size_bytes"],
            }
        )
    observed_closure = sha256_bytes(
        canonical_json_bytes(
            {
                "schema": structural.CHANNEL_CODE_CLOSURE_SCHEMA,
                "members": descriptors,
            }
        )
    )
    if observed_closure != expected_closure:
        raise SomphLineageAuthorityError("channel code closure root mismatch")
    return paths, observed_closure


def _cache_role_inputs(
    cache_set_manifest: dict[str, Any],
    *,
    manifest_path: Path,
    expected_cache_hashes: Mapping[str, str],
    expected_tx_by_role: Mapping[str, list[str]],
    receiver: str,
    build_spec: Mapping[str, Any] | None = None,
    required_samples_per_tx: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    cache_paths = cache_set_manifest.get("cache_npz_by_scenario")
    if not isinstance(cache_paths, dict) or tuple(cache_paths) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphLineageAuthorityError("cache-set cache path scenario order drift")
    result: dict[str, list[dict[str, Any]]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        path = structural._resolve_cache(manifest_path, cache_paths[scenario])
        try:
            with structural._open_external_same_fd(path) as handle:
                digest, _size = _hash_handle(handle)
                if digest != expected_cache_hashes[scenario]:
                    raise SomphLineageAuthorityError(
                        f"authority cache SHA mismatch for {scenario}"
                    )
                _zip_members_from_handle(handle, context=f"authority cache:{scenario}")
                handle.seek(0)
                with np.load(handle, allow_pickle=False) as archive:
                    manifest = structural._embedded_manifest(
                        np.array(archive["manifest_json"], copy=True),
                        scenario=scenario,
                    )
                    roles = np.asarray(archive["dataset_role"]).astype(str)
                    tx_ids = np.asarray(archive["tx_ids"]).astype(str)
                    rx_ids = np.asarray(archive["rx_ids"]).astype(str)
                    satellite_seeds = np.asarray(
                        archive["satellite_seeds"]
                    )
        except structural.SomphLineageError as exc:
            raise SomphLineageAuthorityError(str(exc)) from exc
        if roles.shape != tx_ids.shape or roles.shape != rx_ids.shape:
            raise SomphLineageAuthorityError(
                f"cache identity row shape drift for {scenario}"
            )
        if not bool(np.all(rx_ids == receiver)):
            raise SomphLineageAuthorityError(
                f"cache receiver authority drift for {scenario}"
            )
        if set(roles.tolist()) != set(expected_tx_by_role):
            raise SomphLineageAuthorityError(
                f"cache role authority drift for {scenario}"
            )
        for role, expected_tx in expected_tx_by_role.items():
            if set(tx_ids[roles == role].tolist()) != set(expected_tx):
                raise SomphLineageAuthorityError(
                    f"cache TX coverage authority drift for {scenario}:{role}"
                )
        if build_spec is not None or required_samples_per_tx is not None:
            if (
                build_spec is None
                or required_samples_per_tx is None
                or isinstance(required_samples_per_tx, bool)
                or required_samples_per_tx < 1
                or satellite_seeds.dtype != np.int64
                or satellite_seeds.shape != roles.shape
            ):
                raise SomphLineageAuthorityError(
                    "strict cache seed/coverage inputs invalid"
                )
            expected_channel_config = dict(
                sat_channel_config_for_scenario(scenario)
            )
            expected_channel_config.update(
                {
                    "fs_hz": float(build_spec.get("sat_fs_hz", 25e6)),
                    "fc_hz": float(build_spec.get("sat_fc_hz", 2.462e9)),
                    "star_ground_channel_impl": "simplified_leo_residual",
                }
            )
            expected_channel_sha = canonical_json_sha256(
                expected_channel_config
            )
            observed_channel = manifest.get("channel_config")
            if (
                not isinstance(observed_channel, dict)
                or canonical_json_sha256(observed_channel)
                != expected_channel_sha
                or manifest.get("channel_config_sha256")
                != expected_channel_sha
            ):
                raise SomphLineageAuthorityError(
                    f"cache channel config/fixed code drift for {scenario}"
                )
            base_seed = build_spec.get(
                "satellite_seed_by_scenario", {}
            ).get(scenario)
            if isinstance(base_seed, bool) or not isinstance(base_seed, int):
                raise SomphLineageAuthorityError(
                    f"cache build-spec satellite seed invalid for {scenario}"
                )
            role_order = tuple(expected_tx_by_role)
            expected_role_seeds = {
                role: base_seed + index * 1_000_003
                for index, role in enumerate(role_order)
            }
            if manifest.get("role_satellite_seeds") != expected_role_seeds:
                raise SomphLineageAuthorityError(
                    f"cache role satellite seed/build-spec drift for {scenario}"
                )
            observed_counts = Counter(
                zip(roles.tolist(), tx_ids.tolist(), rx_ids.tolist())
            )
            expected_cells = {
                (role, tx_id, receiver)
                for role, expected_tx in expected_tx_by_role.items()
                for tx_id in expected_tx
            }
            if set(observed_counts) != expected_cells or any(
                observed_counts[cell] != required_samples_per_tx
                for cell in expected_cells
            ):
                raise SomphLineageAuthorityError(
                    f"cache exact per-role/TX/receiver coverage drift for {scenario}"
                )
            for role, expected_seed in expected_role_seeds.items():
                role_seeds = set(
                    int(value)
                    for value in satellite_seeds[roles == role].tolist()
                )
                if role_seeds != {expected_seed}:
                    raise SomphLineageAuthorityError(
                        f"cache row satellite seed/build-spec drift for {scenario}:{role}"
                    )
        role_inputs = manifest.get("role_inputs")
        if not isinstance(role_inputs, list):
            raise SomphLineageAuthorityError(
                f"cache role_inputs missing for {scenario}"
            )
        result[scenario] = [dict(item) for item in role_inputs if isinstance(item, dict)]
        if len(result[scenario]) != len(role_inputs):
            raise SomphLineageAuthorityError(
                f"cache role_inputs object drift for {scenario}"
            )
        if required_samples_per_tx is not None:
            expected_role_counts = {
                role: required_samples_per_tx * len(tx_ids_for_role)
                for role, tx_ids_for_role in expected_tx_by_role.items()
            }
            for row in result[scenario]:
                role = row.get("role")
                if (
                    role not in expected_role_counts
                    or row.get("physical_sample_count")
                    != expected_role_counts[role]
                ):
                    raise SomphLineageAuthorityError(
                        f"cache role_inputs physical_sample_count drift for {scenario}"
                    )
    return result


def _verify_cache_role_inputs(
    values: Mapping[str, list[dict[str, Any]]],
    *,
    datasets: list[dict[str, Any]],
    role_specs: list[dict[str, Any]],
    seed: int,
) -> str:
    expected_rows: list[dict[str, Any]] = []
    for index, (dataset, role_spec) in enumerate(zip(datasets, role_specs)):
        expected_rows.append(
            {
                "role": dataset["role"],
                "dataset_sha256": dataset["sha256"],
                "dataset_size_bytes": dataset["size_bytes"],
                "requested_tx_ids": role_spec["tx_ids"],
                "requested_rxs": role_spec["rxs"],
                "requested_days": role_spec.get("days"),
                "dataset_seed": seed + index * 10_007,
            }
        )
    reference: list[dict[str, Any]] | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        observed = values[scenario]
        if len(observed) != len(expected_rows):
            raise SomphLineageAuthorityError(
                f"cache role_inputs count drift for {scenario}"
            )
        normalized: list[dict[str, Any]] = []
        for index, (row, expected) in enumerate(zip(observed, expected_rows)):
            if set(row) != _ROLE_INPUT_KEYS:
                raise SomphLineageAuthorityError(
                    f"cache role_inputs exact schema drift for {scenario}:{index}"
                )
            if any(row.get(key) != expected_value for key, expected_value in expected.items()):
                raise SomphLineageAuthorityError(
                    f"cache role_inputs authority binding drift for {scenario}:{index}"
                )
            if not isinstance(row.get("resolved_info"), dict):
                raise SomphLineageAuthorityError(
                    f"cache role_inputs resolved_info drift for {scenario}:{index}"
                )
            _require_int(
                row.get("physical_sample_count"),
                field=f"cache role_inputs physical_sample_count {scenario}:{index}",
                minimum=1,
            )
            normalized.append(row)
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise SomphLineageAuthorityError(
                "cache role_inputs drift across LEO_weak scenarios"
            )
    return sha256_bytes(canonical_json_bytes(reference or []))


def _write_new_readonly(path: Path, payload: bytes) -> tuple[str, int]:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
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
    if stat.S_IMODE(path.lstat().st_mode) & _WRITE_BITS:
        raise SomphLineageAuthorityError(f"authority member is not read-only: {path}")
    return sha256_bytes(payload), len(payload)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    def _repair(function, target, _exc):
        try:
            os.chmod(target, 0o700)
            function(target)
        except OSError:
            pass

    if path.exists():
        shutil.rmtree(path, onerror=_repair)


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename one same-parent directory without replacement."""

    if source.parent != target.parent:
        raise SomphLineageAuthorityError(
            "authority staging and destination must share one parent"
        )
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(target), 0x00000008):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(
                    errno.EEXIST, "authority bundle already exists", str(target)
                )
            raise OSError(error, os.strerror(error), str(target))
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SomphLineageAuthorityError(
            "atomic no-replace directory rename is unavailable"
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
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            error, "authority bundle already exists", str(target)
        )
    raise OSError(error, os.strerror(error), str(target))


def _member_descriptor(path: Path, *, name: str) -> dict[str, Any]:
    with open_regular_member_same_fd(path.parent, name) as handle:
        digest, size = _hash_handle(handle)
    return {"name": name, "sha256": digest, "size_bytes": size}


def _validate_commit_members(value: Any) -> list[dict[str, Any]]:
    expected_names = (
        AUTHORITY_LOCK_NAME,
        AUTHORITY_ENVELOPE_NAME,
        AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
        CACHE_SPEC_MANIFEST_NAME,
        STRUCTURAL_RECEIPT_NAME,
        STRUCTURAL_SEAL_NAME,
        AUTHORITY_ATTESTATION_NAME,
    )
    if not isinstance(value, list) or len(value) != len(expected_names):
        raise SomphLineageAuthorityError("authority commit member list drift")
    result: list[dict[str, Any]] = []
    for raw, name in zip(value, expected_names):
        item = _require_exact_dict(
            raw, {"name", "sha256", "size_bytes"}, field=f"commit member:{name}"
        )
        if item.get("name") != name:
            raise SomphLineageAuthorityError("authority commit member order drift")
        _require_sha256(item.get("sha256"), field=f"commit member:{name}.sha256")
        _require_int(item.get("size_bytes"), field=f"commit member:{name}.size_bytes")
        result.append(item)
    return result


def _verify_lock_receipt_binding(
    lock: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    direct = {
        "cache_set_manifest_sha256": lock["cache_set_manifest"]["sha256"],
        "exporter_sha256": lock["exporter"]["sha256"],
        "build_spec_sha256": lock["build_spec"]["canonical_sha256"],
        "channel_code_closure_sha256": lock["channel_code_closure"][
            "closure_sha256"
        ],
        "physical_sample_ids_sha256": lock["physical_sample_ids_sha256"],
    }
    if any(receipt.get(key) != value for key, value in direct.items()):
        raise SomphLineageAuthorityError(
            "authority lock/structural receipt direct root binding drift"
        )
    scenario_receipts = receipt.get("scenario_receipts")
    if (
        not isinstance(scenario_receipts, dict)
        or tuple(scenario_receipts) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphLineageAuthorityError(
            "structural receipt scenario registry drift"
        )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        item = scenario_receipts[scenario]
        expected = {
            "cache_sha256": lock["cache_sha256_by_scenario"][scenario],
            "channel_config_sha256": lock[
                "channel_config_sha256_by_scenario"
            ][scenario],
            "physical_sample_ids_sha256": lock[
                "physical_sample_ids_sha256"
            ],
            "post_channel_iq_sha256_root": lock[
                "post_channel_iq_sha256_root_by_scenario"
            ][scenario],
            "overlay_ids_sha256": lock[
                "overlay_ids_sha256_by_scenario"
            ][scenario],
        }
        if not isinstance(item, dict) or any(
            item.get(key) != value for key, value in expected.items()
        ):
            raise SomphLineageAuthorityError(
                f"authority lock/structural scenario root drift: {scenario}"
            )


def _verify_build_authority_binding(
    lock: Mapping[str, Any],
    *,
    lock_file_sha256: str,
    lock_canonical_sha256: str,
    build_receipt: Mapping[str, Any],
    build_receipt_sha256: str,
    cache_spec_manifest: Mapping[str, Any],
    cache_spec_manifest_sha256: str,
    cache_spec_manifest_size_bytes: int,
) -> tuple[str, str]:
    _require_exact_dict(
        build_receipt,
        _AUTHORITY_LOCK_BUILD_RECEIPT_KEYS,
        field="authority lock build receipt",
    )
    receipt_sha = _require_sha256(
        build_receipt_sha256,
        field="authority lock build receipt SHA",
    )
    manifest_sha = _require_sha256(
        cache_spec_manifest_sha256,
        field="cache-spec manifest SHA",
    )
    expected_cell_id = (
        f"rx_{str(lock.get('receiver')).replace('-', '_')}_seed_{lock.get('seed')}"
    )
    direct = {
        "schema": "cvs.phase1.somph_authority_lock_build_receipt.v1",
        "status": "UNSIGNED_OFFLINE_AUTHORITY_LOCK_BUILT",
        "cache_spec_manifest_sha256": _FORMAL_CACHE_SPEC_MANIFEST_SHA256,
        "cache_spec_manifest_size_bytes": cache_spec_manifest_size_bytes,
        "cache_spec_cell_id": expected_cell_id,
        "required_samples_per_tx": 40,
        "receiver": lock.get("receiver"),
        "seed": lock.get("seed"),
        "cache_scope": "stage2_registered",
        "cache_set_manifest_sha256": lock.get(
            "cache_set_manifest", {}
        ).get("sha256"),
        "build_spec_file_sha256": lock.get("build_spec", {}).get(
            "file_sha256"
        ),
        "build_spec_canonical_sha256": lock.get("build_spec", {}).get(
            "canonical_sha256"
        ),
        "exporter_sha256": lock.get("exporter", {}).get("sha256"),
        "channel_code_closure_sha256": lock.get(
            "channel_code_closure", {}
        ).get("closure_sha256"),
        "cache_role_inputs_root_sha256": lock.get(
            "cache_role_inputs_root_sha256"
        ),
        "physical_sample_ids_sha256": lock.get(
            "physical_sample_ids_sha256"
        ),
        "authority_lock_sha256": lock_file_sha256,
        "authority_lock_canonical_sha256": lock_canonical_sha256,
        "external_authority_lock_verified": False,
        "formal_launch_authority": False,
    }
    if (
        lock.get("cache_scope") != "stage2_registered"
        or manifest_sha != _FORMAL_CACHE_SPEC_MANIFEST_SHA256
        or any(build_receipt.get(key) != value for key, value in direct.items())
    ):
        raise SomphLineageAuthorityError(
            "authority build receipt/lock/official manifest binding drift"
        )
    dataset_root_sha = _require_sha256(
        build_receipt.get("dataset_authority_root_sha256"),
        field="authority build receipt dataset authority root",
    )
    for field in (
        "cache_sha256_by_scenario",
        "channel_config_sha256_by_scenario",
        "post_channel_iq_sha256_root_by_scenario",
        "overlay_ids_sha256_by_scenario",
    ):
        if build_receipt.get(field) != lock.get(field):
            raise SomphLineageAuthorityError(
                f"authority build receipt root drift: {field}"
            )
    audits = build_receipt.get("cache_recompute_audits")
    if not isinstance(audits, dict) or tuple(audits) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphLineageAuthorityError(
            "authority build receipt cache audit registry drift"
        )
    if (
        cache_spec_manifest.get("schema")
        != "cvs.phase2.somph_registered_cache_build_matrix.v1"
        or cache_spec_manifest.get("formal_launch_authority") is not False
        or cache_spec_manifest.get("required_samples_per_tx") != 40
    ):
        raise SomphLineageAuthorityError(
            "official cache-spec manifest contract drift"
        )
    cells = cache_spec_manifest.get("cells")
    matches = [
        item
        for item in cells
        if isinstance(item, dict) and item.get("cell_id") == expected_cell_id
    ] if isinstance(cells, list) else []
    if len(matches) != 1:
        raise SomphLineageAuthorityError(
            "official cache-spec manifest cell identity drift"
        )
    cell = matches[0]
    expected_cell = {
        "receiver": lock.get("receiver"),
        "seed": lock.get("seed"),
        "cache_scope": "stage2_registered",
        "required_samples_per_tx": 40,
        "spec_file_sha256": lock.get("build_spec", {}).get("file_sha256"),
        "spec_canonical_sha256": lock.get("build_spec", {}).get(
            "canonical_sha256"
        ),
    }
    if any(cell.get(key) != value for key, value in expected_cell.items()):
        raise SomphLineageAuthorityError(
            "official cache-spec cell/build-spec binding drift"
        )
    return receipt_sha, dataset_root_sha


def write_somph_lineage_authority_bundle(
    authority_lock_path: str | Path,
    *,
    signed_authority_envelope_path: str | Path,
    expected_signed_authority_envelope_sha256: str,
    authority_lock_build_receipt_path: str | Path,
    cache_spec_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate one pre-locked authority document and atomically publish it."""

    lock, lock_bytes, lock_sha, lock_size = _read_external_json(
        authority_lock_path, context="SOMP-H lineage authority lock"
    )
    _require_exact_dict(lock, _LOCK_KEYS, field="authority lock")
    if lock.get("schema") != AUTHORITY_LOCK_SCHEMA:
        raise SomphLineageAuthorityError("authority lock schema drift")
    lock_canonical_sha = sha256_bytes(canonical_json_bytes(lock))
    expected_envelope_sha = _require_sha256(
        expected_signed_authority_envelope_sha256,
        field="expected_signed_authority_envelope_sha256",
    )
    envelope, envelope_bytes, envelope_sha, _envelope_size = _read_external_json(
        signed_authority_envelope_path,
        context="SOMP-H signed authority envelope",
    )
    if envelope_sha != expected_envelope_sha:
        raise SomphLineageAuthorityError(
            "external signed authority envelope SHA mismatch"
        )
    (
        build_receipt,
        build_receipt_bytes,
        build_receipt_sha,
        _build_receipt_size,
    ) = _read_external_json(
        authority_lock_build_receipt_path,
        context="SOMP-H authority lock build receipt",
    )
    (
        cache_spec_manifest,
        cache_spec_manifest_bytes,
        cache_spec_manifest_sha,
        cache_spec_manifest_size,
    ) = _read_external_json(
        cache_spec_manifest_path,
        context="SOMP-H official locked cache-spec manifest",
    )
    receiver, seed, roles = _validate_lock_formal_identity(lock)
    _verified_build_receipt_sha, build_receipt_dataset_root = (
        _verify_build_authority_binding(
        lock,
        lock_file_sha256=lock_sha,
        lock_canonical_sha256=lock_canonical_sha,
        build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha,
        cache_spec_manifest=cache_spec_manifest,
        cache_spec_manifest_sha256=cache_spec_manifest_sha,
        cache_spec_manifest_size_bytes=cache_spec_manifest_size,
        )
    )
    _verify_signed_envelope(
        envelope,
        lock_canonical_sha256=lock_canonical_sha,
        expected_cache_spec_cell_id=(
            f"rx_{receiver.replace('-', '_')}_seed_{seed}"
        ),
        expected_build_receipt_sha256=build_receipt_sha,
    )

    cache_set_descriptor, _cache_set_bytes = _verify_file_descriptor(
        lock["cache_set_manifest"], field="cache_set_manifest"
    )
    cache_set, _raw_set, set_sha, _set_size = _read_external_json(
        cache_set_descriptor["path"], context="authority cache-set manifest"
    )
    if set_sha != cache_set_descriptor["sha256"]:
        raise SomphLineageAuthorityError("authority cache-set manifest changed")
    expected_caches = _scenario_sha_map(
        lock["cache_sha256_by_scenario"],
        field="cache_sha256_by_scenario",
    )
    exporter_descriptor, _exporter_bytes = _verify_file_descriptor(
        lock["exporter"], field="exporter"
    )
    build_descriptor, build_bytes = _verify_file_descriptor(
        lock["build_spec"], field="build_spec", build_spec=True
    )
    try:
        build_spec = json.loads(build_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphLineageAuthorityError("invalid authority build spec JSON") from exc
    if not isinstance(build_spec, dict):
        raise SomphLineageAuthorityError("authority build spec root must be an object")
    canonical_build_sha = canonical_json_sha256(build_spec)
    if canonical_build_sha != _require_sha256(
        build_descriptor["canonical_sha256"],
        field="build_spec.canonical_sha256",
    ):
        raise SomphLineageAuthorityError("authority build spec canonical SHA mismatch")
    build_spec_path = Path(str(build_descriptor["path"]))
    build_spec_dir = build_spec_path.absolute().parent
    role_specs = _validate_build_spec(
        build_spec,
        lock=lock,
        receiver=receiver,
        seed=seed,
        roles=roles,
        build_spec_dir=build_spec_dir,
    )
    raw_out_manifest = Path(str(build_spec["out_manifest"]))
    if not raw_out_manifest.is_absolute():
        raw_out_manifest = build_spec_dir / raw_out_manifest
    if (
        raw_out_manifest.resolve()
        != Path(str(cache_set_descriptor["path"])).resolve()
        or build_spec["cache_set_id"] != cache_set.get("cache_set_id")
    ):
        raise SomphLineageAuthorityError(
            "real build spec/cache-set manifest identity drift"
        )
    cache_paths = cache_set.get("cache_npz_by_scenario")
    if not isinstance(cache_paths, dict):
        raise SomphLineageAuthorityError("authority cache-set path map missing")
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        raw_out_npz = Path(str(build_spec["out_npz_by_scenario"][scenario]))
        if not raw_out_npz.is_absolute():
            raw_out_npz = build_spec_dir / raw_out_npz
        actual_cache_path = structural._resolve_cache(
            Path(str(cache_set_descriptor["path"])),
            cache_paths.get(scenario),
        )
        if raw_out_npz.resolve() != (
            actual_cache_path.resolve()
        ):
            raise SomphLineageAuthorityError(
                f"real build spec/cache-set output path drift for {scenario}"
            )
    datasets, dataset_root = _validate_datasets(
        lock["datasets"],
        roles=roles,
        role_specs=role_specs,
        receiver=receiver,
        seed=seed,
    )
    if dataset_root != build_receipt_dataset_root:
        raise SomphLineageAuthorityError(
            "authority build receipt dataset authority root drift"
        )
    channel_paths, channel_closure_sha = _validate_channel_closure(
        lock["channel_code_closure"]
    )
    channel_config_roots = _scenario_sha_map(
        lock["channel_config_sha256_by_scenario"],
        field="channel_config_sha256_by_scenario",
    )
    physical_root = _require_sha256(
        lock["physical_sample_ids_sha256"],
        field="physical_sample_ids_sha256",
    )
    iq_roots = _scenario_sha_map(
        lock["post_channel_iq_sha256_root_by_scenario"],
        field="post_channel_iq_sha256_root_by_scenario",
    )
    overlay_roots = _scenario_sha_map(
        lock["overlay_ids_sha256_by_scenario"],
        field="overlay_ids_sha256_by_scenario",
    )
    strict_required_samples: int | None = None
    if lock["cache_scope"] == "stage2_registered":
        declared_counts = {
            role_spec.get("max_samples_per_tx")
            for role_spec in role_specs
        }
        if declared_counts != {40}:
            raise SomphLineageAuthorityError(
                "formal registered authority requires exact maxK20+Q20 coverage"
            )
        strict_required_samples = 40
    role_inputs = _cache_role_inputs(
        cache_set,
        manifest_path=Path(str(cache_set_descriptor["path"])),
        expected_cache_hashes=expected_caches,
        expected_tx_by_role={
            "target_old": list(lock["old_tx_ids"]),
            **(
                {"target_new": list(lock["new_tx_ids"])}
                if lock["cache_scope"] == "stage2_registered"
                else {}
            ),
        },
        receiver=receiver,
        build_spec=(
            build_spec if strict_required_samples is not None else None
        ),
        required_samples_per_tx=strict_required_samples,
    )
    role_inputs_root = _verify_cache_role_inputs(
        role_inputs,
        datasets=datasets,
        role_specs=role_specs,
        seed=seed,
    )
    if role_inputs_root != _require_sha256(
        lock["cache_role_inputs_root_sha256"],
        field="cache_role_inputs_root_sha256",
    ):
        raise SomphLineageAuthorityError(
            "authority lock/cache role-input root mismatch"
        )

    destination = Path(output_root).absolute()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_stat = parent.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise SomphLineageAuthorityError(
            "authority output parent must be a non-symlink directory"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite authority bundle: {destination}")
    staging = parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.staging"
    )
    staging.mkdir(mode=0o700)
    published = False
    try:
        structural_receipt, _structural_seal = (
            structural.write_somph_leo_weak_lineage_seal(
                cache_set_descriptor["path"],
                expected_scope=lock["cache_scope"],
                expected_cache_set_manifest_sha256=cache_set_descriptor["sha256"],
                expected_cache_sha256_by_scenario=expected_caches,
                exporter_path=exporter_descriptor["path"],
                expected_exporter_sha256=exporter_descriptor["sha256"],
                build_spec_path=build_descriptor["path"],
                expected_build_spec_sha256=build_descriptor["canonical_sha256"],
                channel_code_members=channel_paths,
                expected_channel_code_closure_sha256=channel_closure_sha,
                expected_channel_config_sha256_by_scenario=channel_config_roots,
                expected_physical_sample_ids_sha256=physical_root,
                expected_post_channel_iq_sha256_root_by_scenario=iq_roots,
                expected_overlay_ids_sha256_by_scenario=overlay_roots,
                receipt_path=staging / STRUCTURAL_RECEIPT_NAME,
                detached_seal_path=staging / STRUCTURAL_SEAL_NAME,
            )
        )
        if (
            structural_receipt.get("external_authority_lock_verified") is not False
            or structural_receipt.get("formal_launch_authority") is not False
        ):
            raise SomphLineageAuthorityError(
                "structural producer exceeded its authority boundary"
            )
        _verify_lock_receipt_binding(lock, structural_receipt)
        _write_new_readonly(staging / AUTHORITY_LOCK_NAME, lock_bytes)
        _write_new_readonly(staging / AUTHORITY_ENVELOPE_NAME, envelope_bytes)
        _write_new_readonly(
            staging / AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
            build_receipt_bytes,
        )
        _write_new_readonly(
            staging / CACHE_SPEC_MANIFEST_NAME,
            cache_spec_manifest_bytes,
        )
        structural_receipt_descriptor = _member_descriptor(
            staging / STRUCTURAL_RECEIPT_NAME, name=STRUCTURAL_RECEIPT_NAME
        )
        structural_seal_descriptor = _member_descriptor(
            staging / STRUCTURAL_SEAL_NAME, name=STRUCTURAL_SEAL_NAME
        )
        attestation = {
            "schema": AUTHORITY_ATTESTATION_SCHEMA,
            "status": AUTHORITY_STATUS,
            "external_authority_lock_verified": True,
            "authority_lock_sha256": lock_sha,
            "authority_lock_canonical_sha256": lock_canonical_sha,
            "signed_authority_envelope_sha256": envelope_sha,
            "authority_issuer": (
                "qknnv42_stage2bc_extreme_light_route_20260716"
            ),
            "authority_key_id": "somph-authority-ed25519-20260716",
            "receiver": receiver,
            "seed": seed,
            "cache_scope": lock["cache_scope"],
            "old_tx_ids_sha256": sha256_bytes(
                canonical_json_bytes(lock["old_tx_ids"])
            ),
            "new_tx_ids_sha256": sha256_bytes(
                canonical_json_bytes(lock["new_tx_ids"])
            ),
            "dataset_authority_root_sha256": dataset_root,
            "cache_role_inputs_root_sha256": role_inputs_root,
            "structural_receipt_sha256": structural_receipt_descriptor["sha256"],
            "structural_detached_seal_sha256": structural_seal_descriptor["sha256"],
            "formal_launch_authority": False,
        }
        attestation_bytes = canonical_json_bytes(attestation) + b"\n"
        _write_new_readonly(
            staging / AUTHORITY_ATTESTATION_NAME, attestation_bytes
        )
        members = [
            _member_descriptor(staging / name, name=name)
            for name in (
                AUTHORITY_LOCK_NAME,
                AUTHORITY_ENVELOPE_NAME,
                AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
                CACHE_SPEC_MANIFEST_NAME,
                STRUCTURAL_RECEIPT_NAME,
                STRUCTURAL_SEAL_NAME,
                AUTHORITY_ATTESTATION_NAME,
            )
        ]
        commit = {
            "schema": AUTHORITY_COMMIT_SCHEMA,
            "status": AUTHORITY_STATUS,
            "external_authority_lock_verified": True,
            "authority_lock_sha256": lock_sha,
            "signed_authority_envelope_sha256": envelope_sha,
            "authority_issuer": (
                "qknnv42_stage2bc_extreme_light_route_20260716"
            ),
            "authority_key_id": "somph-authority-ed25519-20260716",
            "members": members,
            "bundle_root_sha256": sha256_bytes(canonical_json_bytes(members)),
            "formal_launch_authority": False,
        }
        commit_bytes = canonical_json_bytes(commit) + b"\n"
        commit_sha, _commit_size = _write_new_readonly(
            staging / AUTHORITY_COMMIT_NAME, commit_bytes
        )
        _fsync_directory(staging)
        verify_somph_lineage_authority_bundle(
            staging, expected_commit_sha256=commit_sha
        )
        os.chmod(staging, 0o555)
        if stat.S_IMODE(staging.lstat().st_mode) & _WRITE_BITS:
            raise SomphLineageAuthorityError(
                "authority staging directory is not read-only"
            )
        _fsync_directory(parent)
        _publish_directory_noreplace(staging, destination)
        published = True
        _fsync_directory(parent)
    finally:
        if not published:
            _remove_tree(staging)
    return {
        "authority_bundle_root": str(destination),
        "authority_commit_sha256": commit_sha,
        "authority_lock_sha256": lock_sha,
        "signed_authority_envelope_sha256": envelope_sha,
        "external_authority_lock_verified": True,
        "formal_launch_authority": False,
    }


def verify_somph_lineage_authority_bundle(
    bundle_root: str | Path,
    *,
    expected_commit_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Consume an authority bundle from the external expected commit SHA."""

    expected_commit = _require_sha256(
        expected_commit_sha256, field="expected_commit_sha256"
    )
    root = Path(bundle_root)
    _validate_package_root_exact_allowlist(root, allowed_files=_BUNDLE_MEMBERS)
    with open_regular_member_same_fd(root, AUTHORITY_COMMIT_NAME) as handle:
        commit_sha, _commit_size = _hash_handle(handle)
        commit = _json_from_handle(handle, context="SOMP-H authority commit")
    if commit_sha != expected_commit:
        raise SomphLineageAuthorityError("external authority commit SHA mismatch")
    expected_commit_keys = {
        "schema",
        "status",
        "external_authority_lock_verified",
        "authority_lock_sha256",
        "signed_authority_envelope_sha256",
        "authority_issuer",
        "authority_key_id",
        "members",
        "bundle_root_sha256",
        "formal_launch_authority",
    }
    if set(commit) != expected_commit_keys:
        raise SomphLineageAuthorityError("authority commit exact schema drift")
    expected_values = {
        "schema": AUTHORITY_COMMIT_SCHEMA,
        "status": AUTHORITY_STATUS,
        "external_authority_lock_verified": True,
        "authority_issuer": (
            "qknnv42_stage2bc_extreme_light_route_20260716"
        ),
        "authority_key_id": "somph-authority-ed25519-20260716",
        "formal_launch_authority": False,
    }
    if any(commit.get(key) != value for key, value in expected_values.items()):
        raise SomphLineageAuthorityError("authority commit contract drift")
    lock_sha = _require_sha256(
        commit.get("authority_lock_sha256"),
        field="authority commit lock SHA",
    )
    members = _validate_commit_members(commit.get("members"))
    if commit.get("bundle_root_sha256") != sha256_bytes(
        canonical_json_bytes(members)
    ):
        raise SomphLineageAuthorityError("authority commit bundle root drift")

    payloads: dict[str, dict[str, Any]] = {}
    for descriptor in members:
        with open_regular_member_same_fd(root, descriptor["name"]) as handle:
            digest, size = _hash_handle(handle)
            payload = _json_from_handle(
                handle, context=f"authority member:{descriptor['name']}"
            )
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise SomphLineageAuthorityError(
                f"authority committed member mismatch: {descriptor['name']}"
            )
        payloads[descriptor["name"]] = payload
    if members[0]["sha256"] != lock_sha:
        raise SomphLineageAuthorityError("authority commit/lock digest drift")
    lock = payloads[AUTHORITY_LOCK_NAME]
    if set(lock) != _LOCK_KEYS or lock.get("schema") != AUTHORITY_LOCK_SCHEMA:
        raise SomphLineageAuthorityError("committed authority lock schema drift")
    receiver, seed, roles = _validate_lock_formal_identity(lock)
    envelope = payloads[AUTHORITY_ENVELOPE_NAME]
    envelope_sha = next(
        item["sha256"]
        for item in members
        if item["name"] == AUTHORITY_ENVELOPE_NAME
    )
    if commit.get("signed_authority_envelope_sha256") != envelope_sha:
        raise SomphLineageAuthorityError(
            "authority commit/signed envelope digest drift"
        )
    lock_canonical_sha = sha256_bytes(canonical_json_bytes(lock))
    build_receipt = payloads[AUTHORITY_LOCK_BUILD_RECEIPT_NAME]
    build_receipt_sha = next(
        item["sha256"]
        for item in members
        if item["name"] == AUTHORITY_LOCK_BUILD_RECEIPT_NAME
    )
    cache_spec_manifest = payloads[CACHE_SPEC_MANIFEST_NAME]
    cache_spec_manifest_descriptor = next(
        item for item in members if item["name"] == CACHE_SPEC_MANIFEST_NAME
    )
    _verified_build_receipt_sha, build_receipt_dataset_root = (
        _verify_build_authority_binding(
        lock,
        lock_file_sha256=lock_sha,
        lock_canonical_sha256=lock_canonical_sha,
        build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha,
        cache_spec_manifest=cache_spec_manifest,
        cache_spec_manifest_sha256=cache_spec_manifest_descriptor["sha256"],
        cache_spec_manifest_size_bytes=cache_spec_manifest_descriptor[
            "size_bytes"
        ],
        )
    )
    _verify_signed_envelope(
        envelope,
        lock_canonical_sha256=lock_canonical_sha,
        expected_cache_spec_cell_id=(
            f"rx_{receiver.replace('-', '_')}_seed_{seed}"
        ),
        expected_build_receipt_sha256=build_receipt_sha,
    )
    committed_dataset_root = _dataset_root_from_lock(
        lock,
        receiver=receiver,
        seed=seed,
        roles=roles,
    )
    if (
        committed_dataset_root
        != build_receipt_dataset_root
    ):
        raise SomphLineageAuthorityError(
            "committed authority build receipt dataset root drift"
        )

    receipt, _seal = structural.verify_somph_leo_weak_lineage_seal(
        root / STRUCTURAL_RECEIPT_NAME,
        root / STRUCTURAL_SEAL_NAME,
        expected_detached_seal_sha256=next(
            item["sha256"] for item in members if item["name"] == STRUCTURAL_SEAL_NAME
        ),
    )
    _verify_lock_receipt_binding(lock, receipt)
    attestation = payloads[AUTHORITY_ATTESTATION_NAME]
    expected_attestation_keys = {
        "schema",
        "status",
        "external_authority_lock_verified",
        "authority_lock_sha256",
        "authority_lock_canonical_sha256",
        "signed_authority_envelope_sha256",
        "authority_issuer",
        "authority_key_id",
        "receiver",
        "seed",
        "cache_scope",
        "old_tx_ids_sha256",
        "new_tx_ids_sha256",
        "dataset_authority_root_sha256",
        "cache_role_inputs_root_sha256",
        "structural_receipt_sha256",
        "structural_detached_seal_sha256",
        "formal_launch_authority",
    }
    if set(attestation) != expected_attestation_keys:
        raise SomphLineageAuthorityError("authority attestation exact schema drift")
    expected_attestation = {
        "schema": AUTHORITY_ATTESTATION_SCHEMA,
        "status": AUTHORITY_STATUS,
        "external_authority_lock_verified": True,
        "authority_lock_sha256": lock_sha,
        "authority_lock_canonical_sha256": lock_canonical_sha,
        "signed_authority_envelope_sha256": envelope_sha,
        "authority_issuer": (
            "qknnv42_stage2bc_extreme_light_route_20260716"
        ),
        "authority_key_id": "somph-authority-ed25519-20260716",
        "receiver": receiver,
        "seed": seed,
        "cache_scope": lock["cache_scope"],
        "old_tx_ids_sha256": sha256_bytes(canonical_json_bytes(lock["old_tx_ids"])),
        "new_tx_ids_sha256": sha256_bytes(canonical_json_bytes(lock["new_tx_ids"])),
        "structural_receipt_sha256": next(
            item["sha256"]
            for item in members
            if item["name"] == STRUCTURAL_RECEIPT_NAME
        ),
        "structural_detached_seal_sha256": next(
            item["sha256"] for item in members if item["name"] == STRUCTURAL_SEAL_NAME
        ),
        "dataset_authority_root_sha256": committed_dataset_root,
        "cache_role_inputs_root_sha256": lock[
            "cache_role_inputs_root_sha256"
        ],
        "formal_launch_authority": False,
    }
    if any(attestation.get(key) != value for key, value in expected_attestation.items()):
        raise SomphLineageAuthorityError("authority attestation binding drift")
    for root_field in ("dataset_authority_root_sha256", "cache_role_inputs_root_sha256"):
        _require_sha256(attestation.get(root_field), field=root_field)
    if receipt.get("formal_launch_authority") is not False:
        raise SomphLineageAuthorityError("structural receipt authorized launch")
    return lock, attestation, commit
