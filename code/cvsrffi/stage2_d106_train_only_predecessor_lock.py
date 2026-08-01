"""Fixed non-formal D106 predecessor-lock bytes for train-only mechanics.

The official surface publishes canonical immutable bytes only.  Typed
``Phase1ZIDStudentTLock`` objects are reconstructed afresh in a private,
strictly revalidated consumption path; they are never a persistent public
bundle capability.  Supplied tap hashes are provenance labels only and are
explicitly insufficient for a real-G0 or promotion claim until a later caller
binds them through the strict D106 tap loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping

from . import stage2_d105_phase1_bundle as _d105_module
from . import stage2_d106_rcmr_2v_qknn as _rcmr_module
from . import stage2_zid_student_t_qknn as _qknn_module
from .stage2_d105_phase1_bundle import (
    D105Phase1BundleError,
    load_d105_candidate_method_lock,
    load_d105_candidate_runtime_manifest,
)
from .stage2_d106_rcmr_2v_qknn import (
    CANDIDATE_ID,
    D106RCMR2VError,
    load_d106_rcmr_2v_method_lock,
)
from .stage2_zid_student_t_qknn import LOCK_SCHEMA as _QKNN_LOCK_SCHEMA
from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


_SCHEMA = "cvs.phase1.d106.nonformal_train_only_predecessor_lock_bundle.v2"
_RECEIPT_SCHEMA = _SCHEMA + ".receipt.v1"
_LOCK_WIRE_SCHEMA = _SCHEMA + ".lock_wire.v1"
_RESOURCE_SCHEMA = _SCHEMA + ".resources.v1"
_BUNDLE_KIND = "NON_FORMAL_TRAIN_ONLY_MECHANICAL_PREDECESSOR_LOCK_BUNDLE"
_BUNDLE_STATUS = "NON_FORMAL_TRAIN_ONLY_MECHANICAL_PREDECESSOR_LOCKS_LOADED"
_PROTOCOL_SCHEMA = "p2_min_v1"
_D105_CANDIDATE_ID = "D105-CBRC+LPO-RC"
_K_VALUES = (1, 5, 10)
_TAP_HASH_PROVENANCE = "CALLER_SUPPLIED_UNVERIFIED"
_PYTHON_OBJECT_RSS_ACCOUNTING = "NOT_MEASURED_EXCLUDED_FROM_ACCOUNTING"

_CODE_ROOT = Path(__file__).resolve().parents[1]
_WORKTREE_ROOT = _CODE_ROOT.parent
_CONFIG_ROOT = _WORKTREE_ROOT / "configs"
_MODULE_PATH = Path(__file__).resolve()

# These names intentionally remain module-private and are not exported.  The
# official APIs capture import-time copies and reject any later global drift.
_BUNDLE_PATH = (_CONFIG_ROOT / "d106_train_only_predecessor_lock_bundle_20260801.json").resolve()
_D105_METHOD_LOCK_PATH = (_CONFIG_ROOT / "d105_candidate_method_lock_20260731.json").resolve()
_D105_RUNTIME_MANIFEST_PATH = (
    _CONFIG_ROOT / "d105_candidate_runtime_manifest_20260731.json"
).resolve()
_D106_RCMR_METHOD_LOCK_PATH = (
    _CONFIG_ROOT / "d106_rcmr_2v_method_lock_20260801.json"
).resolve()
_D106_RCMR_SOURCE_PATH = (_CODE_ROOT / "cvsrffi" / "stage2_d106_rcmr_2v_qknn.py").resolve()

_BUNDLE_FILE_SHA256 = "a1006909a60620479d6b64f7bd35d91b8c0e09c8738a226ef34c6da1a88668de"
_D105_METHOD_LOCK_SHA256 = "7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08"
_D105_RUNTIME_MANIFEST_SHA256 = "9b1887e64851851be8a81118a3b3728cd94517de6c9ae275f8574764cb30c38e"
_D105_CHECKPOINT_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
_D106_RCMR_METHOD_LOCK_SHA256 = "be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c"

_EXPECTED_STUDENT_T_QKNN = {
    "student_nu": 3.0,
    "kernel_effective_dim": 12,
    "kernel_volume_gamma": 1.0,
    "shared_h0": 0.35,
    "scale_prior_strength": 2.0,
    "scale_min_ratio": 0.5,
    "scale_max_ratio": 2.0,
    "temperature": 0.85,
    "support_storage": "int8_fp16_scale",
}
_EXPECTED_TYPED_LOCK_BINDINGS = {
    "phase1_lodo_receipt_sha256": "d105_candidate_method_lock_sha256",
    "quantization_margin_audit_sha256": "d106_tap_receipt_sha256",
}
_EXPECTED_DEPENDENCY_CODE_SHA256 = {
    "cvsrffi/stage2_d105_phase1_bundle.py": (
        "91931cb3893cb902a7eef1e509d209b232d2769225b012c9a0027c978a3ced39"
    ),
    "cvsrffi/stage2_d106_rcmr_2v_qknn.py": (
        "ca641737f4ba26093c9d70f6ad8048e4cad9c6a0fa737920149d7ffe0fef73fa"
    ),
    "cvsrffi/stage2_zid_student_t_qknn.py": (
        "f7bc2ab7e6f9457085973099431db934edfa840ba37e904288ff4720726101e2"
    ),
}
_AUTHORITY_FLAG_NAMES = (
    "d105_formal_authority",
    "held_truth_access",
    "performance_metrics_computed",
    "phase2_promotion_authority",
    "runner_authority",
    "target_access",
)
_EXPECTED_AUTHORITY_FLAGS = {name: False for name in _AUTHORITY_FLAG_NAMES}
_EXPECTED_TAP_BINDING_POLICY = {
    "external_strict_tap_loader_binding_required": True,
    "tap_hash_provenance": _TAP_HASH_PROVENANCE,
    "this_bundle_is_not_sufficient_for_real_g0_promotion": True,
}

_CANONICAL_BUNDLE_BYTES_CAP = 8192
_STATIC_BUNDLE_JSON_BYTES_CAP = 4096
_LOADER_MODULE_SOURCE_BYTES_CAP = 131072
_LOCK_WIRE_BYTES_CAP_PER_K = 2048


class D106TrainOnlyPredecessorLockError(ValueError):
    """Raised when the non-formal predecessor-lock closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        return item

    return json.dumps(
        plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106TrainOnlyPredecessorLockError(f"{name} must be a lowercase SHA256")
    return value


def _require_exact_str(value: Any, expected: str, name: str) -> str:
    if type(value) is not str or value != expected:
        raise D106TrainOnlyPredecessorLockError(f"{name} exact string drift")
    return value


def _require_exact_int(value: Any, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise D106TrainOnlyPredecessorLockError(f"{name} exact integer drift")
    return value


def _require_exact_float(value: Any, expected: float, name: str) -> float:
    if type(value) is not float or value != expected:
        raise D106TrainOnlyPredecessorLockError(f"{name} exact float drift")
    return value


def _require_exact_bool(value: Any, expected: bool, name: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise D106TrainOnlyPredecessorLockError(f"{name} exact bool drift")
    return value


def _read_regular_bytes(path: Path, *, name: str) -> tuple[bytes, str]:
    """Read and hash one stable regular file from the same opened descriptor."""

    try:
        before = path.lstat()
    except OSError as error:
        raise D106TrainOnlyPredecessorLockError(f"cannot read {name}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise D106TrainOnlyPredecessorLockError(f"{name} must be a regular non-symlink file")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise D106TrainOnlyPredecessorLockError(f"{name} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise D106TrainOnlyPredecessorLockError(f"{name} changed during read")
        payload = b"".join(chunks)
    except OSError as error:
        raise D106TrainOnlyPredecessorLockError(f"cannot read {name}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload, _sha256_bytes(payload)


def _load_canonical_json(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, Any], str, bytes]:
    payload, actual_sha256 = _read_regular_bytes(path, name=name)
    if actual_sha256 != _require_sha256(expected_sha256, f"{name} SHA256"):
        raise D106TrainOnlyPredecessorLockError(f"{name} SHA256 drift")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106TrainOnlyPredecessorLockError(
            f"{name} must be canonical UTF-8 JSON"
        ) from error
    canonical = _canonical_bytes(document)
    if type(document) is not dict or payload not in {canonical, canonical + b"\n"}:
        raise D106TrainOnlyPredecessorLockError(f"{name} canonical JSON drift")
    return document, actual_sha256, payload


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise D106TrainOnlyPredecessorLockError(f"{name} key closure drift")
    return value


def _validate_exact_sha_map(
    value: Any, expected: Mapping[str, str], name: str
) -> None:
    mapping = _require_exact_keys(value, set(expected), name)
    for key, expected_sha256 in expected.items():
        _require_exact_str(mapping[key], expected_sha256, f"{name}.{key}")
        _require_sha256(mapping[key], f"{name}.{key}")


def _validate_exact_false_flags(value: Any, name: str) -> None:
    flags = _require_exact_keys(value, set(_AUTHORITY_FLAG_NAMES), name)
    for flag in _AUTHORITY_FLAG_NAMES:
        _require_exact_bool(flags[flag], False, f"{name}.{flag}")


def _validate_file_binding(
    value: Any, *, file_name: str, sha256: str, name: str
) -> None:
    binding = _require_exact_keys(value, {"file", "sha256"}, name)
    _require_exact_str(binding["file"], file_name, f"{name}.file")
    _require_exact_str(binding["sha256"], sha256, f"{name}.sha256")
    _require_sha256(binding["sha256"], f"{name}.sha256")


def _validate_student_t_qknn(value: Any) -> None:
    student = _require_exact_keys(value, set(_EXPECTED_STUDENT_T_QKNN), "student_t_qknn")
    for name, expected in _EXPECTED_STUDENT_T_QKNN.items():
        if type(expected) is int:
            _require_exact_int(student[name], expected, f"student_t_qknn.{name}")
        elif type(expected) is float:
            _require_exact_float(student[name], expected, f"student_t_qknn.{name}")
        else:
            _require_exact_str(student[name], expected, f"student_t_qknn.{name}")


def _validate_tap_binding_policy(value: Any) -> None:
    policy = _require_exact_keys(
        value, set(_EXPECTED_TAP_BINDING_POLICY), "tap_binding_policy"
    )
    _require_exact_str(
        policy["tap_hash_provenance"],
        _TAP_HASH_PROVENANCE,
        "tap_binding_policy.tap_hash_provenance",
    )
    _require_exact_bool(
        policy["external_strict_tap_loader_binding_required"],
        True,
        "tap_binding_policy.external_strict_tap_loader_binding_required",
    )
    _require_exact_bool(
        policy["this_bundle_is_not_sufficient_for_real_g0_promotion"],
        True,
        "tap_binding_policy.this_bundle_is_not_sufficient_for_real_g0_promotion",
    )


def _validate_bundle_document(document: Any) -> None:
    expected_keys = {
        "authority_flags",
        "bundle_kind",
        "candidate_id",
        "checkpoint_sha256",
        "d105_candidate_id",
        "d105_candidate_method_lock",
        "d105_candidate_runtime_manifest",
        "d106_rcmr_method_lock",
        "dependency_code_sha256",
        "k_values",
        "protocol_schema",
        "schema",
        "student_t_qknn",
        "tap_binding_policy",
        "typed_lock_bindings",
    }
    bundle = _require_exact_keys(document, expected_keys, "predecessor-lock bundle")
    _require_exact_str(bundle["schema"], _SCHEMA, "bundle.schema")
    _require_exact_str(bundle["bundle_kind"], _BUNDLE_KIND, "bundle.bundle_kind")
    _require_exact_str(bundle["candidate_id"], CANDIDATE_ID, "bundle.candidate_id")
    _require_exact_str(
        bundle["protocol_schema"], _PROTOCOL_SCHEMA, "bundle.protocol_schema"
    )
    _require_exact_str(
        bundle["d105_candidate_id"], _D105_CANDIDATE_ID, "bundle.d105_candidate_id"
    )
    _require_exact_str(
        bundle["checkpoint_sha256"], _D105_CHECKPOINT_SHA256, "bundle.checkpoint_sha256"
    )
    _require_sha256(bundle["checkpoint_sha256"], "bundle.checkpoint_sha256")
    _validate_file_binding(
        bundle["d105_candidate_method_lock"],
        file_name=_D105_METHOD_LOCK_PATH.name,
        sha256=_D105_METHOD_LOCK_SHA256,
        name="bundle.d105_candidate_method_lock",
    )
    _validate_file_binding(
        bundle["d105_candidate_runtime_manifest"],
        file_name=_D105_RUNTIME_MANIFEST_PATH.name,
        sha256=_D105_RUNTIME_MANIFEST_SHA256,
        name="bundle.d105_candidate_runtime_manifest",
    )
    _validate_file_binding(
        bundle["d106_rcmr_method_lock"],
        file_name=_D106_RCMR_METHOD_LOCK_PATH.name,
        sha256=_D106_RCMR_METHOD_LOCK_SHA256,
        name="bundle.d106_rcmr_method_lock",
    )
    _validate_exact_sha_map(
        bundle["dependency_code_sha256"],
        _EXPECTED_DEPENDENCY_CODE_SHA256,
        "bundle.dependency_code_sha256",
    )
    k_values = bundle["k_values"]
    if type(k_values) is not list or len(k_values) != len(_K_VALUES):
        raise D106TrainOnlyPredecessorLockError("bundle K-value closure drift")
    for index, active_k in enumerate(_K_VALUES):
        _require_exact_int(k_values[index], active_k, f"bundle.k_values[{index}]")
    _validate_student_t_qknn(bundle["student_t_qknn"])
    _validate_tap_binding_policy(bundle["tap_binding_policy"])
    bindings = _require_exact_keys(
        bundle["typed_lock_bindings"],
        set(_EXPECTED_TYPED_LOCK_BINDINGS),
        "bundle.typed_lock_bindings",
    )
    for name, expected in _EXPECTED_TYPED_LOCK_BINDINGS.items():
        _require_exact_str(bindings[name], expected, f"bundle.typed_lock_bindings.{name}")
    _validate_exact_false_flags(bundle["authority_flags"], "bundle.authority_flags")


def _validate_runtime_manifest(document: Any) -> None:
    manifest = _require_exact_keys(
        document,
        {
            "candidate_id",
            "checkpoint_sha256",
            "core_file_sha256",
            "entrypoints",
            "protocol_schema",
            "schema",
        },
        "D105 runtime manifest",
    )
    _require_exact_str(
        manifest["schema"],
        "cvs.stage2.d105.candidate_runtime_manifest.v1",
        "D105 runtime manifest.schema",
    )
    _require_exact_str(
        manifest["candidate_id"], _D105_CANDIDATE_ID, "D105 runtime manifest.candidate_id"
    )
    _require_exact_str(
        manifest["checkpoint_sha256"],
        _D105_CHECKPOINT_SHA256,
        "D105 runtime manifest.checkpoint_sha256",
    )
    _require_sha256(manifest["checkpoint_sha256"], "D105 runtime manifest.checkpoint_sha256")
    _require_exact_str(
        manifest["protocol_schema"],
        _PROTOCOL_SCHEMA,
        "D105 runtime manifest.protocol_schema",
    )
    for field in ("core_file_sha256", "entrypoints"):
        mapping = manifest[field]
        if type(mapping) is not dict or not mapping:
            raise D106TrainOnlyPredecessorLockError(
                f"D105 runtime manifest.{field} closure drift"
            )
        for key, value in mapping.items():
            if type(key) is not str or type(value) is not str:
                raise D106TrainOnlyPredecessorLockError(
                    f"D105 runtime manifest.{field} type drift"
                )
            if field == "core_file_sha256":
                _require_sha256(value, f"D105 runtime manifest.{field}.{key}")


def _verify_dependency_code(document: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    expected = document["dependency_code_sha256"]
    observed: list[tuple[str, str]] = []
    for relative_path, expected_sha256 in sorted(expected.items()):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise D106TrainOnlyPredecessorLockError("dependency code path drift")
        _payload, actual = _read_regular_bytes(
            _CODE_ROOT / relative, name="predecessor-lock dependency code"
        )
        if actual != _require_sha256(expected_sha256, "dependency code SHA256"):
            raise D106TrainOnlyPredecessorLockError(
                "predecessor-lock dependency code SHA256 drift"
            )
        observed.append((relative.as_posix(), actual))
    return tuple(observed)


def _function_source_sha256(function: Callable[..., Any], *, name: str) -> str:
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError) as error:
        raise D106TrainOnlyPredecessorLockError(
            f"{name} source is unavailable"
        ) from error
    return _sha256_bytes(source)


@dataclass(frozen=True, slots=True)
class _CallablePin:
    global_name: str
    callable_value: Callable[..., Any]
    owner_module: Any
    callable_module: Any
    module_name: str
    qualname: str
    source_sha256: str
    code_object_id: int
    module_path: Path
    module_file_sha256: str


def _capture_callable_pin(
    global_name: str, callable_value: Callable[..., Any]
) -> _CallablePin:
    callable_module = inspect.getmodule(callable_value)
    if callable_module is None or type(getattr(callable_module, "__name__", None)) is not str:
        raise D106TrainOnlyPredecessorLockError(f"cannot pin {global_name} module")
    module_file = getattr(callable_module, "__file__", None)
    if type(module_file) is not str:
        raise D106TrainOnlyPredecessorLockError(f"cannot pin {global_name} module file")
    module_path = Path(module_file).resolve()
    _payload, module_file_sha256 = _read_regular_bytes(
        module_path, name=f"{global_name} module"
    )
    qualname = getattr(callable_value, "__qualname__", None)
    module_name = getattr(callable_value, "__module__", None)
    code = getattr(callable_value, "__code__", None)
    if type(qualname) is not str or type(module_name) is not str or code is None:
        raise D106TrainOnlyPredecessorLockError(f"cannot pin {global_name} callable")
    return _CallablePin(
        global_name=global_name,
        callable_value=callable_value,
        owner_module=sys.modules[__name__],
        callable_module=callable_module,
        module_name=module_name,
        qualname=qualname,
        source_sha256=_function_source_sha256(callable_value, name=global_name),
        code_object_id=id(code),
        module_path=module_path,
        module_file_sha256=module_file_sha256,
    )


@dataclass(frozen=True, slots=True)
class _StaticPin:
    module: Any
    module_path: Path
    module_source_sha256: str
    bundle_path: Path
    bundle_file_sha256: str
    d105_method_lock_path: Path
    d105_method_lock_sha256: str
    d105_runtime_manifest_path: Path
    d105_runtime_manifest_sha256: str
    d106_rcmr_method_lock_path: Path
    d106_rcmr_method_lock_sha256: str
    d106_rcmr_source_path: Path
    checkpoint_sha256: str
    lock_class: type[Phase1ZIDStudentTLock]
    d105_module: Any
    rcmr_module: Any
    qknn_module: Any


def _verify_callable_pin(pin: _CallablePin) -> None:
    current_global = pin.owner_module.__dict__.get(pin.global_name)
    if current_global is not pin.callable_value:
        raise D106TrainOnlyPredecessorLockError(
            f"imported loader callable drift: {pin.global_name}"
        )
    if (
        inspect.getmodule(pin.callable_value) is not pin.callable_module
        or getattr(pin.callable_value, "__module__", None) != pin.module_name
        or getattr(pin.callable_value, "__qualname__", None) != pin.qualname
        or id(getattr(pin.callable_value, "__code__", None)) != pin.code_object_id
    ):
        raise D106TrainOnlyPredecessorLockError(
            f"imported loader identity drift: {pin.global_name}"
        )
    if _function_source_sha256(pin.callable_value, name=pin.global_name) != pin.source_sha256:
        raise D106TrainOnlyPredecessorLockError(
            f"imported loader source drift: {pin.global_name}"
        )
    module_file = getattr(pin.callable_module, "__file__", None)
    if type(module_file) is not str or Path(module_file).resolve() != pin.module_path:
        raise D106TrainOnlyPredecessorLockError(
            f"imported loader module path drift: {pin.global_name}"
        )
    _payload, observed_sha256 = _read_regular_bytes(
        pin.module_path, name=f"{pin.global_name} module"
    )
    if observed_sha256 != pin.module_file_sha256:
        raise D106TrainOnlyPredecessorLockError(
            f"imported loader module source drift: {pin.global_name}"
        )


_MODULE_SOURCE_BYTES, _MODULE_SOURCE_SHA256 = _read_regular_bytes(
    _MODULE_PATH, name="D106 predecessor-lock module"
)
_D105_METHOD_LOCK_LOADER_PIN = _capture_callable_pin(
    "load_d105_candidate_method_lock", load_d105_candidate_method_lock
)
_D105_RUNTIME_LOADER_PIN = _capture_callable_pin(
    "load_d105_candidate_runtime_manifest", load_d105_candidate_runtime_manifest
)
_D106_RCMR_LOADER_PIN = _capture_callable_pin(
    "load_d106_rcmr_2v_method_lock", load_d106_rcmr_2v_method_lock
)
_STATIC_PIN = _StaticPin(
    module=sys.modules[__name__],
    module_path=_MODULE_PATH,
    module_source_sha256=_MODULE_SOURCE_SHA256,
    bundle_path=_BUNDLE_PATH,
    bundle_file_sha256=_BUNDLE_FILE_SHA256,
    d105_method_lock_path=_D105_METHOD_LOCK_PATH,
    d105_method_lock_sha256=_D105_METHOD_LOCK_SHA256,
    d105_runtime_manifest_path=_D105_RUNTIME_MANIFEST_PATH,
    d105_runtime_manifest_sha256=_D105_RUNTIME_MANIFEST_SHA256,
    d106_rcmr_method_lock_path=_D106_RCMR_METHOD_LOCK_PATH,
    d106_rcmr_method_lock_sha256=_D106_RCMR_METHOD_LOCK_SHA256,
    d106_rcmr_source_path=_D106_RCMR_SOURCE_PATH,
    checkpoint_sha256=_D105_CHECKPOINT_SHA256,
    lock_class=Phase1ZIDStudentTLock,
    d105_module=_d105_module,
    rcmr_module=_rcmr_module,
    qknn_module=_qknn_module,
)


def _make_runtime_verifier(
    pin: _StaticPin,
    callable_pins: tuple[_CallablePin, ...],
    *,
    read_regular: Callable[..., tuple[bytes, str]] = _read_regular_bytes,
    verify_callable: Callable[[_CallablePin], None] = _verify_callable_pin,
) -> Callable[[], None]:
    def verify() -> None:
        module = pin.module
        expected_globals = {
            "_BUNDLE_PATH": pin.bundle_path,
            "_BUNDLE_FILE_SHA256": pin.bundle_file_sha256,
            "_D105_METHOD_LOCK_PATH": pin.d105_method_lock_path,
            "_D105_METHOD_LOCK_SHA256": pin.d105_method_lock_sha256,
            "_D105_RUNTIME_MANIFEST_PATH": pin.d105_runtime_manifest_path,
            "_D105_RUNTIME_MANIFEST_SHA256": pin.d105_runtime_manifest_sha256,
            "_D106_RCMR_METHOD_LOCK_PATH": pin.d106_rcmr_method_lock_path,
            "_D106_RCMR_METHOD_LOCK_SHA256": pin.d106_rcmr_method_lock_sha256,
            "_D106_RCMR_SOURCE_PATH": pin.d106_rcmr_source_path,
            "_D105_CHECKPOINT_SHA256": pin.checkpoint_sha256,
            "_MODULE_PATH": pin.module_path,
            "_MODULE_SOURCE_SHA256": pin.module_source_sha256,
            "_d105_module": pin.d105_module,
            "_rcmr_module": pin.rcmr_module,
            "_qknn_module": pin.qknn_module,
            "Phase1ZIDStudentTLock": pin.lock_class,
        }
        for name, expected in expected_globals.items():
            if module.__dict__.get(name) is not expected and module.__dict__.get(name) != expected:
                raise D106TrainOnlyPredecessorLockError(
                    f"import-time closure global drift: {name}"
                )
        _payload, self_sha256 = read_regular(pin.module_path, name="D106 predecessor-lock module")
        if self_sha256 != pin.module_source_sha256:
            raise D106TrainOnlyPredecessorLockError("loader module source SHA256 drift")
        for callable_pin in callable_pins:
            verify_callable(callable_pin)
        if getattr(pin.qknn_module, "Phase1ZIDStudentTLock", None) is not pin.lock_class:
            raise D106TrainOnlyPredecessorLockError("typed lock class identity drift")
        rcmr_file = getattr(pin.rcmr_module, "__file__", None)
        if type(rcmr_file) is not str or Path(rcmr_file).resolve() != pin.d106_rcmr_source_path:
            raise D106TrainOnlyPredecessorLockError("D106 RCMR module path drift")
        _payload, rcmr_source_sha256 = read_regular(
            pin.d106_rcmr_source_path, name="D106 RCMR raw source"
        )
        if rcmr_source_sha256 != _EXPECTED_DEPENDENCY_CODE_SHA256[
            "cvsrffi/stage2_d106_rcmr_2v_qknn.py"
        ]:
            raise D106TrainOnlyPredecessorLockError("D106 RCMR raw source SHA256 drift")

    return verify


_VERIFY_RUNTIME_CLOSURE = _make_runtime_verifier(
    _STATIC_PIN,
    (
        _D105_METHOD_LOCK_LOADER_PIN,
        _D105_RUNTIME_LOADER_PIN,
        _D106_RCMR_LOADER_PIN,
    ),
)


@dataclass(frozen=True, slots=True)
class _StaticClosure:
    bundle_document: Mapping[str, Any]
    bundle_file_sha256: str
    d105_method_lock_sha256: str
    d105_runtime_manifest_sha256: str
    checkpoint_sha256: str
    d106_rcmr_method_lock_sha256: str
    dependency_code_sha256: tuple[tuple[str, str], ...]
    loader_module_source_bytes: int
    static_bundle_json_bytes: int


def _make_static_loader(
    pin: _StaticPin,
    runtime_verifier: Callable[[], None],
    *,
    load_json: Callable[..., tuple[dict[str, Any], str, bytes]] = _load_canonical_json,
    validate_bundle: Callable[[Any], None] = _validate_bundle_document,
    validate_runtime: Callable[[Any], None] = _validate_runtime_manifest,
    verify_dependencies: Callable[[Mapping[str, Any]], tuple[tuple[str, str], ...]] = _verify_dependency_code,
    read_regular: Callable[..., tuple[bytes, str]] = _read_regular_bytes,
    d105_method_loader: Callable[..., dict[str, Any]] = load_d105_candidate_method_lock,
    d105_runtime_loader: Callable[..., dict[str, Any]] = load_d105_candidate_runtime_manifest,
    rcmr_loader: Callable[..., Any] = load_d106_rcmr_2v_method_lock,
) -> Callable[[], _StaticClosure]:
    def load_static() -> _StaticClosure:
        runtime_verifier()
        bundle, bundle_file_sha256, bundle_raw = load_json(
            pin.bundle_path,
            expected_sha256=pin.bundle_file_sha256,
            name="D106 train-only predecessor-lock bundle",
        )
        validate_bundle(bundle)
        dependencies = verify_dependencies(bundle)
        runtime_document, runtime_sha256, _runtime_raw = load_json(
            pin.d105_runtime_manifest_path,
            expected_sha256=pin.d105_runtime_manifest_sha256,
            name="D105 candidate runtime manifest",
        )
        validate_runtime(runtime_document)
        try:
            d105_runtime = d105_runtime_loader(
                pin.d105_runtime_manifest_path,
                expected_checkpoint_sha256=pin.checkpoint_sha256,
            )
        except (D105Phase1BundleError, ValueError) as error:
            raise D106TrainOnlyPredecessorLockError(
                "D105 candidate runtime-manifest validation failed"
            ) from error
        if (
            type(d105_runtime) is not dict
            or _require_sha256(
                d105_runtime.get("d105_candidate_runtime_manifest_sha256"),
                "D105 runtime loader manifest SHA256",
            )
            != pin.d105_runtime_manifest_sha256
            or _require_sha256(
                d105_runtime.get("checkpoint_sha256"), "D105 runtime loader checkpoint SHA256"
            )
            != pin.checkpoint_sha256
        ):
            raise D106TrainOnlyPredecessorLockError(
                "D105 candidate runtime-manifest binding drift"
            )
        _method_document, method_file_sha256, _method_raw = load_json(
            pin.d105_method_lock_path,
            expected_sha256=pin.d105_method_lock_sha256,
            name="D105 candidate method lock",
        )
        if method_file_sha256 != pin.d105_method_lock_sha256:
            raise D106TrainOnlyPredecessorLockError("D105 candidate method-lock binding drift")
        try:
            d105_lock = d105_method_loader(
                pin.d105_method_lock_path,
                expected_checkpoint_sha256=pin.checkpoint_sha256,
                expected_runtime_sha256=runtime_sha256,
            )
        except (D105Phase1BundleError, ValueError) as error:
            raise D106TrainOnlyPredecessorLockError(
                "D105 candidate method-lock validation failed"
            ) from error
        if (
            type(d105_lock) is not dict
            or _require_sha256(
                d105_lock.get("d105_candidate_method_lock_sha256"),
                "D105 method loader lock SHA256",
            )
            != pin.d105_method_lock_sha256
            or _require_sha256(
                d105_lock.get("runtime_sha256"), "D105 method loader runtime SHA256"
            )
            != pin.d105_runtime_manifest_sha256
            or _require_sha256(
                d105_lock.get("checkpoint_sha256"), "D105 method loader checkpoint SHA256"
            )
            != pin.checkpoint_sha256
        ):
            raise D106TrainOnlyPredecessorLockError("D105 candidate method-lock binding drift")
        _rcmr_payload, rcmr_source_sha256 = read_regular(
            pin.d106_rcmr_source_path, name="D106 RCMR raw source"
        )
        if rcmr_source_sha256 != _EXPECTED_DEPENDENCY_CODE_SHA256[
            "cvsrffi/stage2_d106_rcmr_2v_qknn.py"
        ]:
            raise D106TrainOnlyPredecessorLockError("D106 RCMR raw source SHA256 drift")
        try:
            d106_rcmr_lock = rcmr_loader(
                pin.d106_rcmr_method_lock_path,
                expected_sha256=pin.d106_rcmr_method_lock_sha256,
            )
        except (D106RCMR2VError, ValueError) as error:
            raise D106TrainOnlyPredecessorLockError(
                "D106 RCMR method-lock validation failed"
            ) from error
        if (
            _require_sha256(
                getattr(d106_rcmr_lock, "document_sha256", None),
                "D106 RCMR method loader SHA256",
            )
            != pin.d106_rcmr_method_lock_sha256
        ):
            raise D106TrainOnlyPredecessorLockError("D106 RCMR method-lock binding drift")
        if len(bundle_raw) > _STATIC_BUNDLE_JSON_BYTES_CAP:
            raise D106TrainOnlyPredecessorLockError("static bundle JSON resource cap exceeded")
        _module_raw, loader_module_sha256 = read_regular(
            pin.module_path, name="D106 predecessor-lock module"
        )
        if loader_module_sha256 != pin.module_source_sha256:
            raise D106TrainOnlyPredecessorLockError("loader module source SHA256 drift")
        if len(_module_raw) > _LOADER_MODULE_SOURCE_BYTES_CAP:
            raise D106TrainOnlyPredecessorLockError("loader module source resource cap exceeded")
        return _StaticClosure(
            bundle_document=bundle,
            bundle_file_sha256=bundle_file_sha256,
            d105_method_lock_sha256=pin.d105_method_lock_sha256,
            d105_runtime_manifest_sha256=runtime_sha256,
            checkpoint_sha256=pin.checkpoint_sha256,
            d106_rcmr_method_lock_sha256=pin.d106_rcmr_method_lock_sha256,
            dependency_code_sha256=dependencies,
            loader_module_source_bytes=len(_module_raw),
            static_bundle_json_bytes=len(bundle_raw),
        )

    return load_static


_OFFICIAL_STATIC_LOADER = _make_static_loader(_STATIC_PIN, _VERIFY_RUNTIME_CLOSURE)


def _validate_typed_locks(
    locks: tuple[Phase1ZIDStudentTLock, ...], *, tap_receipt_sha256: str
) -> None:
    if type(locks) is not tuple or len(locks) != len(_K_VALUES):
        raise D106TrainOnlyPredecessorLockError("three-K typed-lock count drift")
    for active_k, lock in zip(_K_VALUES, locks, strict=True):
        if type(lock) is not _STATIC_PIN.lock_class:
            raise D106TrainOnlyPredecessorLockError("typed-lock class drift")
        _require_exact_int(lock.active_k, active_k, "typed lock active_k")
        _require_exact_float(lock.student_nu, 3.0, "typed lock student_nu")
        _require_exact_int(
            lock.kernel_effective_dim, 12, "typed lock kernel_effective_dim"
        )
        _require_exact_float(
            lock.kernel_volume_gamma, 1.0, "typed lock kernel_volume_gamma"
        )
        _require_exact_float(lock.shared_h0, 0.35, "typed lock shared_h0")
        _require_exact_float(
            lock.scale_prior_strength, 2.0, "typed lock scale_prior_strength"
        )
        _require_exact_float(lock.scale_min_ratio, 0.5, "typed lock scale_min_ratio")
        _require_exact_float(lock.scale_max_ratio, 2.0, "typed lock scale_max_ratio")
        _require_exact_float(lock.temperature, 0.85, "typed lock temperature")
        _require_exact_str(lock.schema, _QKNN_LOCK_SCHEMA, "typed lock schema")
        _require_exact_str(
            lock.phase1_lodo_receipt_sha256,
            _D105_METHOD_LOCK_SHA256,
            "typed lock D105 method binding",
        )
        _require_exact_str(
            lock.quantization_margin_audit_sha256,
            tap_receipt_sha256,
            "typed lock tap-receipt binding",
        )
        _require_sha256(lock.phase1_lodo_receipt_sha256, "typed lock D105 method binding")
        _require_sha256(lock.quantization_margin_audit_sha256, "typed lock tap binding")
        _require_sha256(lock.lock_digest, "typed lock digest")


def _construct_typed_locks(
    static: _StaticClosure, *, tap_receipt_sha256: str
) -> tuple[Phase1ZIDStudentTLock, ...]:
    tap_receipt_sha = _require_sha256(tap_receipt_sha256, "D106 tap receipt")
    student = static.bundle_document["student_t_qknn"]
    locks = tuple(
        _STATIC_PIN.lock_class(
            active_k=active_k,
            student_nu=student["student_nu"],
            kernel_effective_dim=student["kernel_effective_dim"],
            kernel_volume_gamma=student["kernel_volume_gamma"],
            shared_h0=student["shared_h0"],
            scale_prior_strength=student["scale_prior_strength"],
            scale_min_ratio=student["scale_min_ratio"],
            scale_max_ratio=student["scale_max_ratio"],
            temperature=student["temperature"],
            phase1_lodo_receipt_sha256=static.d105_method_lock_sha256,
            quantization_margin_audit_sha256=tap_receipt_sha,
        )
        for active_k in _K_VALUES
    )
    _validate_typed_locks(locks, tap_receipt_sha256=tap_receipt_sha)
    return locks


def _wire_for_lock(lock: Phase1ZIDStudentTLock) -> bytes:
    return _canonical_bytes(
        {
            "schema": _LOCK_WIRE_SCHEMA,
            "K": lock.active_k,
            "lock_digest": lock.lock_digest,
            "lock": {
                "schema": lock.schema,
                "active_k": lock.active_k,
                "student_nu": lock.student_nu,
                "kernel_effective_dim": lock.kernel_effective_dim,
                "kernel_volume_gamma": lock.kernel_volume_gamma,
                "shared_h0": lock.shared_h0,
                "scale_prior_strength": lock.scale_prior_strength,
                "scale_min_ratio": lock.scale_min_ratio,
                "scale_max_ratio": lock.scale_max_ratio,
                "temperature": lock.temperature,
                "phase1_lodo_receipt_sha256": lock.phase1_lodo_receipt_sha256,
                "quantization_margin_audit_sha256": lock.quantization_margin_audit_sha256,
            },
        }
    )


def _lock_wires(
    locks: tuple[Phase1ZIDStudentTLock, ...], *, tap_receipt_sha256: str
) -> tuple[bytes, ...]:
    _validate_typed_locks(locks, tap_receipt_sha256=tap_receipt_sha256)
    wires = tuple(_wire_for_lock(lock) for lock in locks)
    if any(len(wire) > _LOCK_WIRE_BYTES_CAP_PER_K for wire in wires):
        raise D106TrainOnlyPredecessorLockError("typed-lock wire resource cap exceeded")
    return wires


def _resource_receipt(
    static: _StaticClosure,
    wires: tuple[bytes, ...],
    *,
    canonical_bundle_bytes_exact: int,
) -> dict[str, Any]:
    wire_bytes_by_k = {str(active_k): len(wire) for active_k, wire in zip(_K_VALUES, wires, strict=True)}
    return {
        "schema": _RESOURCE_SCHEMA,
        "canonical_bundle_bytes_exact": canonical_bundle_bytes_exact,
        "canonical_bundle_bytes_cap": _CANONICAL_BUNDLE_BYTES_CAP,
        "lock_wire_bytes_by_k": wire_bytes_by_k,
        "lock_wire_bytes_total": sum(wire_bytes_by_k.values()),
        "lock_wire_bytes_cap_per_k": _LOCK_WIRE_BYTES_CAP_PER_K,
        "loader_module_source_bytes": static.loader_module_source_bytes,
        "loader_module_source_bytes_cap": _LOADER_MODULE_SOURCE_BYTES_CAP,
        "static_bundle_json_bytes": static.static_bundle_json_bytes,
        "static_bundle_json_bytes_cap": _STATIC_BUNDLE_JSON_BYTES_CAP,
        "python_object_and_rss_bytes": _PYTHON_OBJECT_RSS_ACCOUNTING,
    }


def _build_bundle_bytes(
    static: _StaticClosure,
    *,
    tap_archive_sha256: str,
    tap_receipt_sha256: str,
    locks: tuple[Phase1ZIDStudentTLock, ...],
) -> bytes:
    tap_archive_sha = _require_sha256(tap_archive_sha256, "D106 tap archive")
    tap_receipt_sha = _require_sha256(tap_receipt_sha256, "D106 tap receipt")
    wires = _lock_wires(locks, tap_receipt_sha256=tap_receipt_sha)
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "status": _BUNDLE_STATUS,
        "bundle_kind": _BUNDLE_KIND,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": _PROTOCOL_SCHEMA,
        "bundle_file_sha256": static.bundle_file_sha256,
        "d105_candidate_method_lock_sha256": static.d105_method_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": static.d105_runtime_manifest_sha256,
        "checkpoint_sha256": static.checkpoint_sha256,
        "d106_rcmr_method_lock_sha256": static.d106_rcmr_method_lock_sha256,
        "dependency_code_sha256": dict(static.dependency_code_sha256),
        "loader_module_sha256": _MODULE_SOURCE_SHA256,
        "d106_tap_archive_sha256": tap_archive_sha,
        "d106_tap_receipt_sha256": tap_receipt_sha,
        "tap_hash_provenance": _TAP_HASH_PROVENANCE,
        "external_strict_tap_loader_binding_required": True,
        "real_g0_promotion_authority": False,
        "typed_lock_bindings": dict(_EXPECTED_TYPED_LOCK_BINDINGS),
        "K_values": list(_K_VALUES),
        "lock_digest_by_k": {
            str(lock.active_k): lock.lock_digest for lock in locks
        },
        "lock_wire_sha256_by_k": {
            str(lock.active_k): _sha256_bytes(wire)
            for lock, wire in zip(locks, wires, strict=True)
        },
        "authority_flags": dict(_EXPECTED_AUTHORITY_FLAGS),
        "resource_receipt": _resource_receipt(
            static, wires, canonical_bundle_bytes_exact=0
        ),
    }
    for _ in range(8):
        provisional = _canonical_bytes(receipt)
        receipt["resource_receipt"]["canonical_bundle_bytes_exact"] = len(provisional)
        result = _canonical_bytes(receipt)
        if len(result) == receipt["resource_receipt"]["canonical_bundle_bytes_exact"]:
            if len(result) > _CANONICAL_BUNDLE_BYTES_CAP:
                raise D106TrainOnlyPredecessorLockError(
                    "canonical bundle resource cap exceeded"
                )
            return result
    raise D106TrainOnlyPredecessorLockError("canonical bundle byte-size closure drift")


@dataclass(frozen=True, slots=True, init=False)
class D106TrainOnlyPredecessorLockResourceSummary:
    """Immutable primitive-only resource view reconstructed from canonical bytes."""

    canonical_bundle_bytes: int
    canonical_bundle_bytes_cap: int
    lock_wire_bytes_by_k: tuple[tuple[int, int], ...]
    lock_wire_bytes_total: int
    lock_wire_bytes_cap_per_k: int
    loader_module_source_bytes: int
    loader_module_source_bytes_cap: int
    static_bundle_json_bytes: int
    static_bundle_json_bytes_cap: int
    python_object_and_rss_bytes: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("resource summaries are reconstructed from canonical bundle bytes")

    @classmethod
    def _create(
        cls,
        *,
        canonical_bundle_bytes: int,
        canonical_bundle_bytes_cap: int,
        lock_wire_bytes_by_k: tuple[tuple[int, int], ...],
        lock_wire_bytes_total: int,
        lock_wire_bytes_cap_per_k: int,
        loader_module_source_bytes: int,
        loader_module_source_bytes_cap: int,
        static_bundle_json_bytes: int,
        static_bundle_json_bytes_cap: int,
        python_object_and_rss_bytes: str,
    ) -> D106TrainOnlyPredecessorLockResourceSummary:
        instance = object.__new__(cls)
        for name, value in (
            ("canonical_bundle_bytes", canonical_bundle_bytes),
            ("canonical_bundle_bytes_cap", canonical_bundle_bytes_cap),
            ("lock_wire_bytes_by_k", lock_wire_bytes_by_k),
            ("lock_wire_bytes_total", lock_wire_bytes_total),
            ("lock_wire_bytes_cap_per_k", lock_wire_bytes_cap_per_k),
            ("loader_module_source_bytes", loader_module_source_bytes),
            ("loader_module_source_bytes_cap", loader_module_source_bytes_cap),
            ("static_bundle_json_bytes", static_bundle_json_bytes),
            ("static_bundle_json_bytes_cap", static_bundle_json_bytes_cap),
            ("python_object_and_rss_bytes", python_object_and_rss_bytes),
        ):
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class D106TrainOnlyPredecessorLockSummary:
    """Immutable primitive-only summary; it is not a lock capability."""

    receipt_sha256: str
    schema: str
    status: str
    candidate_id: str
    protocol_schema: str
    bundle_file_sha256: str
    d105_candidate_method_lock_sha256: str
    d105_candidate_runtime_manifest_sha256: str
    checkpoint_sha256: str
    d106_rcmr_method_lock_sha256: str
    d106_tap_archive_sha256: str
    d106_tap_receipt_sha256: str
    tap_hash_provenance: str
    external_strict_tap_loader_binding_required: bool
    real_g0_promotion_authority: bool
    k_values: tuple[int, ...]
    lock_digest_by_k: tuple[tuple[int, str], ...]
    lock_wire_sha256_by_k: tuple[tuple[int, str], ...]
    authority_flags: tuple[tuple[str, bool], ...]
    resource: D106TrainOnlyPredecessorLockResourceSummary

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("lock summaries are reconstructed from canonical bundle bytes")

    @classmethod
    def _create(
        cls,
        *,
        receipt_sha256: str,
        schema: str,
        status: str,
        candidate_id: str,
        protocol_schema: str,
        bundle_file_sha256: str,
        d105_candidate_method_lock_sha256: str,
        d105_candidate_runtime_manifest_sha256: str,
        checkpoint_sha256: str,
        d106_rcmr_method_lock_sha256: str,
        d106_tap_archive_sha256: str,
        d106_tap_receipt_sha256: str,
        tap_hash_provenance: str,
        external_strict_tap_loader_binding_required: bool,
        real_g0_promotion_authority: bool,
        k_values: tuple[int, ...],
        lock_digest_by_k: tuple[tuple[int, str], ...],
        lock_wire_sha256_by_k: tuple[tuple[int, str], ...],
        authority_flags: tuple[tuple[str, bool], ...],
        resource: D106TrainOnlyPredecessorLockResourceSummary,
    ) -> D106TrainOnlyPredecessorLockSummary:
        instance = object.__new__(cls)
        for name, value in (
            ("receipt_sha256", receipt_sha256),
            ("schema", schema),
            ("status", status),
            ("candidate_id", candidate_id),
            ("protocol_schema", protocol_schema),
            ("bundle_file_sha256", bundle_file_sha256),
            ("d105_candidate_method_lock_sha256", d105_candidate_method_lock_sha256),
            (
                "d105_candidate_runtime_manifest_sha256",
                d105_candidate_runtime_manifest_sha256,
            ),
            ("checkpoint_sha256", checkpoint_sha256),
            ("d106_rcmr_method_lock_sha256", d106_rcmr_method_lock_sha256),
            ("d106_tap_archive_sha256", d106_tap_archive_sha256),
            ("d106_tap_receipt_sha256", d106_tap_receipt_sha256),
            ("tap_hash_provenance", tap_hash_provenance),
            (
                "external_strict_tap_loader_binding_required",
                external_strict_tap_loader_binding_required,
            ),
            ("real_g0_promotion_authority", real_g0_promotion_authority),
            ("k_values", k_values),
            ("lock_digest_by_k", lock_digest_by_k),
            ("lock_wire_sha256_by_k", lock_wire_sha256_by_k),
            ("authority_flags", authority_flags),
            ("resource", resource),
        ):
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    receipt_bytes: bytes
    summary: D106TrainOnlyPredecessorLockSummary
    locks: tuple[Phase1ZIDStudentTLock, ...]


def _strict_int_map(
    value: Any, *, expected: Mapping[str, int], name: str
) -> None:
    mapping = _require_exact_keys(value, set(expected), name)
    for key, expected_value in expected.items():
        _require_exact_int(mapping[key], expected_value, f"{name}.{key}")


def _validate_resource_receipt(
    value: Any,
    *,
    static: _StaticClosure,
    wires: tuple[bytes, ...],
    bundle_bytes: bytes,
) -> D106TrainOnlyPredecessorLockResourceSummary:
    expected_keys = {
        "schema",
        "canonical_bundle_bytes_exact",
        "canonical_bundle_bytes_cap",
        "lock_wire_bytes_by_k",
        "lock_wire_bytes_total",
        "lock_wire_bytes_cap_per_k",
        "loader_module_source_bytes",
        "loader_module_source_bytes_cap",
        "static_bundle_json_bytes",
        "static_bundle_json_bytes_cap",
        "python_object_and_rss_bytes",
    }
    resource = _require_exact_keys(value, expected_keys, "resource receipt")
    _require_exact_str(resource["schema"], _RESOURCE_SCHEMA, "resource receipt.schema")
    _require_exact_int(
        resource["canonical_bundle_bytes_exact"],
        len(bundle_bytes),
        "resource receipt.canonical_bundle_bytes_exact",
    )
    _require_exact_int(
        resource["canonical_bundle_bytes_cap"],
        _CANONICAL_BUNDLE_BYTES_CAP,
        "resource receipt.canonical_bundle_bytes_cap",
    )
    expected_wire_bytes = {
        str(active_k): len(wire) for active_k, wire in zip(_K_VALUES, wires, strict=True)
    }
    _strict_int_map(
        resource["lock_wire_bytes_by_k"],
        expected=expected_wire_bytes,
        name="resource receipt.lock_wire_bytes_by_k",
    )
    _require_exact_int(
        resource["lock_wire_bytes_total"],
        sum(expected_wire_bytes.values()),
        "resource receipt.lock_wire_bytes_total",
    )
    _require_exact_int(
        resource["lock_wire_bytes_cap_per_k"],
        _LOCK_WIRE_BYTES_CAP_PER_K,
        "resource receipt.lock_wire_bytes_cap_per_k",
    )
    _require_exact_int(
        resource["loader_module_source_bytes"],
        static.loader_module_source_bytes,
        "resource receipt.loader_module_source_bytes",
    )
    _require_exact_int(
        resource["loader_module_source_bytes_cap"],
        _LOADER_MODULE_SOURCE_BYTES_CAP,
        "resource receipt.loader_module_source_bytes_cap",
    )
    _require_exact_int(
        resource["static_bundle_json_bytes"],
        static.static_bundle_json_bytes,
        "resource receipt.static_bundle_json_bytes",
    )
    _require_exact_int(
        resource["static_bundle_json_bytes_cap"],
        _STATIC_BUNDLE_JSON_BYTES_CAP,
        "resource receipt.static_bundle_json_bytes_cap",
    )
    _require_exact_str(
        resource["python_object_and_rss_bytes"],
        _PYTHON_OBJECT_RSS_ACCOUNTING,
        "resource receipt.python_object_and_rss_bytes",
    )
    return D106TrainOnlyPredecessorLockResourceSummary._create(
        canonical_bundle_bytes=resource["canonical_bundle_bytes_exact"],
        canonical_bundle_bytes_cap=resource["canonical_bundle_bytes_cap"],
        lock_wire_bytes_by_k=tuple(
            (active_k, expected_wire_bytes[str(active_k)]) for active_k in _K_VALUES
        ),
        lock_wire_bytes_total=resource["lock_wire_bytes_total"],
        lock_wire_bytes_cap_per_k=resource["lock_wire_bytes_cap_per_k"],
        loader_module_source_bytes=resource["loader_module_source_bytes"],
        loader_module_source_bytes_cap=resource["loader_module_source_bytes_cap"],
        static_bundle_json_bytes=resource["static_bundle_json_bytes"],
        static_bundle_json_bytes_cap=resource["static_bundle_json_bytes_cap"],
        python_object_and_rss_bytes=resource["python_object_and_rss_bytes"],
    )


def _validate_published_bundle_bytes(
    bundle_receipt_bytes: Any, *, static: _StaticClosure
) -> _VerifiedBundle:
    if type(bundle_receipt_bytes) is not bytes:
        raise D106TrainOnlyPredecessorLockError("published bundle must be immutable bytes")
    if len(bundle_receipt_bytes) > _CANONICAL_BUNDLE_BYTES_CAP:
        raise D106TrainOnlyPredecessorLockError("published bundle resource cap exceeded")
    try:
        receipt = json.loads(bundle_receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106TrainOnlyPredecessorLockError(
            "published bundle must be canonical UTF-8 JSON bytes"
        ) from error
    if type(receipt) is not dict or _canonical_bytes(receipt) != bundle_receipt_bytes:
        raise D106TrainOnlyPredecessorLockError("published bundle canonical-byte drift")
    expected_keys = {
        "K_values",
        "authority_flags",
        "bundle_file_sha256",
        "bundle_kind",
        "candidate_id",
        "checkpoint_sha256",
        "d105_candidate_method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d106_rcmr_method_lock_sha256",
        "d106_tap_archive_sha256",
        "d106_tap_receipt_sha256",
        "dependency_code_sha256",
        "external_strict_tap_loader_binding_required",
        "loader_module_sha256",
        "lock_digest_by_k",
        "lock_wire_sha256_by_k",
        "protocol_schema",
        "real_g0_promotion_authority",
        "resource_receipt",
        "schema",
        "status",
        "tap_hash_provenance",
        "typed_lock_bindings",
    }
    document = _require_exact_keys(receipt, expected_keys, "published bundle")
    _require_exact_str(document["schema"], _RECEIPT_SCHEMA, "published bundle.schema")
    _require_exact_str(document["status"], _BUNDLE_STATUS, "published bundle.status")
    _require_exact_str(
        document["bundle_kind"], _BUNDLE_KIND, "published bundle.bundle_kind"
    )
    _require_exact_str(document["candidate_id"], CANDIDATE_ID, "published bundle.candidate_id")
    _require_exact_str(
        document["protocol_schema"], _PROTOCOL_SCHEMA, "published bundle.protocol_schema"
    )
    for name, expected_sha256 in (
        ("bundle_file_sha256", static.bundle_file_sha256),
        ("d105_candidate_method_lock_sha256", static.d105_method_lock_sha256),
        (
            "d105_candidate_runtime_manifest_sha256",
            static.d105_runtime_manifest_sha256,
        ),
        ("checkpoint_sha256", static.checkpoint_sha256),
        ("d106_rcmr_method_lock_sha256", static.d106_rcmr_method_lock_sha256),
        ("loader_module_sha256", _STATIC_PIN.module_source_sha256),
    ):
        _require_exact_str(document[name], expected_sha256, f"published bundle.{name}")
        _require_sha256(document[name], f"published bundle.{name}")
    _validate_exact_sha_map(
        document["dependency_code_sha256"],
        dict(static.dependency_code_sha256),
        "published bundle.dependency_code_sha256",
    )
    tap_archive_sha256 = _require_sha256(
        document["d106_tap_archive_sha256"], "published bundle D106 tap archive"
    )
    tap_receipt_sha256 = _require_sha256(
        document["d106_tap_receipt_sha256"], "published bundle D106 tap receipt"
    )
    _require_exact_str(
        document["tap_hash_provenance"],
        _TAP_HASH_PROVENANCE,
        "published bundle.tap_hash_provenance",
    )
    _require_exact_bool(
        document["external_strict_tap_loader_binding_required"],
        True,
        "published bundle.external_strict_tap_loader_binding_required",
    )
    _require_exact_bool(
        document["real_g0_promotion_authority"],
        False,
        "published bundle.real_g0_promotion_authority",
    )
    bindings = _require_exact_keys(
        document["typed_lock_bindings"],
        set(_EXPECTED_TYPED_LOCK_BINDINGS),
        "published bundle.typed_lock_bindings",
    )
    for name, expected in _EXPECTED_TYPED_LOCK_BINDINGS.items():
        _require_exact_str(bindings[name], expected, f"published bundle.typed_lock_bindings.{name}")
    k_values = document["K_values"]
    if type(k_values) is not list or len(k_values) != len(_K_VALUES):
        raise D106TrainOnlyPredecessorLockError("published bundle K-value closure drift")
    for index, active_k in enumerate(_K_VALUES):
        _require_exact_int(k_values[index], active_k, f"published bundle.K_values[{index}]")
    _validate_exact_false_flags(document["authority_flags"], "published bundle.authority_flags")
    locks = _construct_typed_locks(static, tap_receipt_sha256=tap_receipt_sha256)
    wires = _lock_wires(locks, tap_receipt_sha256=tap_receipt_sha256)
    expected_digests = {str(lock.active_k): lock.lock_digest for lock in locks}
    expected_wire_sha256 = {
        str(lock.active_k): _sha256_bytes(wire)
        for lock, wire in zip(locks, wires, strict=True)
    }
    _validate_exact_sha_map(
        document["lock_digest_by_k"],
        expected_digests,
        "published bundle.lock_digest_by_k",
    )
    _validate_exact_sha_map(
        document["lock_wire_sha256_by_k"],
        expected_wire_sha256,
        "published bundle.lock_wire_sha256_by_k",
    )
    resource = _validate_resource_receipt(
        document["resource_receipt"],
        static=static,
        wires=wires,
        bundle_bytes=bundle_receipt_bytes,
    )
    summary = D106TrainOnlyPredecessorLockSummary._create(
        receipt_sha256=_sha256_bytes(bundle_receipt_bytes),
        schema=document["schema"],
        status=document["status"],
        candidate_id=document["candidate_id"],
        protocol_schema=document["protocol_schema"],
        bundle_file_sha256=document["bundle_file_sha256"],
        d105_candidate_method_lock_sha256=document["d105_candidate_method_lock_sha256"],
        d105_candidate_runtime_manifest_sha256=document[
            "d105_candidate_runtime_manifest_sha256"
        ],
        checkpoint_sha256=document["checkpoint_sha256"],
        d106_rcmr_method_lock_sha256=document["d106_rcmr_method_lock_sha256"],
        d106_tap_archive_sha256=tap_archive_sha256,
        d106_tap_receipt_sha256=tap_receipt_sha256,
        tap_hash_provenance=document["tap_hash_provenance"],
        external_strict_tap_loader_binding_required=document[
            "external_strict_tap_loader_binding_required"
        ],
        real_g0_promotion_authority=document["real_g0_promotion_authority"],
        k_values=tuple(_K_VALUES),
        lock_digest_by_k=tuple(
            (active_k, expected_digests[str(active_k)]) for active_k in _K_VALUES
        ),
        lock_wire_sha256_by_k=tuple(
            (active_k, expected_wire_sha256[str(active_k)]) for active_k in _K_VALUES
        ),
        authority_flags=tuple(
            (name, _EXPECTED_AUTHORITY_FLAGS[name]) for name in _AUTHORITY_FLAG_NAMES
        ),
        resource=resource,
    )
    return _VerifiedBundle(
        receipt_bytes=bytes(bundle_receipt_bytes), summary=summary, locks=locks
    )


def _make_official_apis(
    static_loader: Callable[[], _StaticClosure],
    *,
    construct_locks: Callable[..., tuple[Phase1ZIDStudentTLock, ...]] = _construct_typed_locks,
    build_bundle: Callable[..., bytes] = _build_bundle_bytes,
    validate_published: Callable[..., _VerifiedBundle] = _validate_published_bundle_bytes,
) -> tuple[
    Callable[..., bytes],
    Callable[[bytes], D106TrainOnlyPredecessorLockSummary],
    Callable[[bytes], tuple[Phase1ZIDStudentTLock, ...]],
]:
    def load_bundle(*, tap_archive_sha256: str, tap_receipt_sha256: str) -> bytes:
        tap_archive_sha = _require_sha256(tap_archive_sha256, "D106 tap archive")
        tap_receipt_sha = _require_sha256(tap_receipt_sha256, "D106 tap receipt")
        static = static_loader()
        locks = construct_locks(static, tap_receipt_sha256=tap_receipt_sha)
        published = build_bundle(
            static,
            tap_archive_sha256=tap_archive_sha,
            tap_receipt_sha256=tap_receipt_sha,
            locks=locks,
        )
        # Verify what is published through the same strict consumption path;
        # return only immutable bytes, never the transient typed locks.
        validate_published(published, static=static)
        return bytes(published)

    def summarize_bundle(
        bundle_receipt_bytes: bytes,
    ) -> D106TrainOnlyPredecessorLockSummary:
        static = static_loader()
        return validate_published(bundle_receipt_bytes, static=static).summary

    def reconstruct_for_internal_consumer(
        bundle_receipt_bytes: bytes,
    ) -> tuple[Phase1ZIDStudentTLock, ...]:
        """Private G0 integration hook; it revalidates bytes and rebuilds locks."""

        static = static_loader()
        verified = validate_published(bundle_receipt_bytes, static=static)
        return tuple(verified.locks)

    return load_bundle, summarize_bundle, reconstruct_for_internal_consumer


(
    load_d106_train_only_predecessor_lock_bundle,
    summarize_d106_train_only_predecessor_lock_bundle,
    _strict_reconstruct_d106_train_only_predecessor_locks,
) = _make_official_apis(_OFFICIAL_STATIC_LOADER)


__all__ = [
    "D106TrainOnlyPredecessorLockError",
    "D106TrainOnlyPredecessorLockResourceSummary",
    "D106TrainOnlyPredecessorLockSummary",
    "load_d106_train_only_predecessor_lock_bundle",
    "summarize_d106_train_only_predecessor_lock_bundle",
]
