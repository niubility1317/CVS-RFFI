"""Fail-closed D106 RCMR-2V G0 orchestration.

The production surface is deliberately one call from SHA-pinned tap paths to
canonical result bytes.  It never returns a tap handle, row object, label
array, fold plan, query identifier, prediction, score, or authority token. It
strictly loads the sealed tap, derives fresh train-only predecessor locks from
the actual archive/receipt bytes, and publishes only non-formal mechanical
root/bitmap/count evidence.

The separate synthetic entry point exercises the mechanical algorithm only.
Its schema and status are permanently non-formal and cannot be promoted to a
runner or deployment result.  Because Python cannot provide same-process
capability isolation, the honest audit statement is
``AUDITED_PATH_DID_NOT_READ_HELD_LABELS``; this module does not claim that held
label capability is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d106_phase1_tap as _tap_module
from . import stage2_d106_rcmr_2v_qknn as _rcmr_module
from . import stage2_d106_train_only_predecessor_lock as _predecessor_module
from . import stage2_zid_student_t_qknn as _qknn_module
from .stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    PROTOCOL_SCHEMA,
    TAP_ARCHIVE_NAME,
    TAP_RECEIPT_SCHEMA,
    load_d106_phase1_ls_tap as _load_d106_phase1_ls_tap,
)
from .stage2_d106_rcmr_2v_qknn import CANDIDATE_ID
from .stage2_d106_train_only_predecessor_lock import (
    _strict_reconstruct_d106_train_only_predecessor_locks as _strict_reconstruct_predecessor_locks,
    load_d106_train_only_predecessor_lock_bundle as _load_predecessor_lock_bundle,
)
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


PRODUCTION_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.production.v3"
INTERNAL_ROWS_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.internal_rows_test.v1"
PRODUCTION_EXECUTION_MANIFEST_SCHEMA = (
    "cvs.phase1.d106.rcmr_2v_g0.production_execution_manifest.v1"
)
PRODUCTION_FOLD_EXECUTION_SCHEMA = PRODUCTION_EXECUTION_MANIFEST_SCHEMA + ".fold.v1"
PRODUCTION_PER_K_SCHEMA = PRODUCTION_EXECUTION_MANIFEST_SCHEMA + ".per_k.v1"
SYNTHETIC_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.synthetic_test.v2"
FOLD_EXECUTION_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.fold_execution.v2"
PER_K_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.per_k.v2"
RESOURCE_SCHEMA = "cvs.phase1.d106.rcmr_2v_g0.resources.v2"
PRODUCTION_BLOCKED_STATUS = "REAL_G0_BLOCKED_MISSING_D105_LOCK_AUTHORITY"
PRODUCTION_EXECUTED_STATUS = "REAL_G0_EXECUTED_NON_FORMAL_TRAIN_ONLY_MECHANICAL"
INTERNAL_ROWS_STATUS = "INTERNAL_ROWS_TEST_ONLY_NOT_A_PRODUCTION_ARTIFACT"
STRICT_LOADER_EVIDENCE_SCHEMA = PRODUCTION_SCHEMA + ".strict_loader_evidence.v1"
PRODUCTION_MECHANICAL_STATUS = (
    "NON_DEPLOYABLE_TRAIN_ONLY_MECHANICAL_AWAITING_RUNNER_AUTHORITY"
)
SYNTHETIC_STATUS = "SYNTHETIC_TEST_ONLY_NON_FORMAL"
HELD_LABEL_AUDIT_STATUS = "AUDITED_PATH_DID_NOT_READ_HELD_LABELS"
ALGORITHM_SCOPE = "NON_FORMAL_TRAIN_ONLY_MECHANICAL"
FOLD_POLICY = "all_7x4_receiver_day_leave_cell_out"
SUPPORT_POLICY = "per_class_utf8_physical_id_first_k_from_nonheld_cells"
K_VALUES = (1, 5, 10)
EXPECTED_ROWS = 588
EXPECTED_CLASSES = 6
EXPECTED_RECEIVERS = 7
EXPECTED_DAYS = 4
EXPECTED_FOLDS = EXPECTED_RECEIVERS * EXPECTED_DAYS
MAX_QUERY_ROWS_PER_FOLD = 24
Z_DIM = 160
MAX_TOKEN_UTF8_BYTES = 256
ANALYSIS_NUMERIC_ARRAY_BUDGET_BYTES = 1 << 20

# A repository search on 2026-08-01 found no canonical D105 three-K authority
# artifact or known digest set.  An arbitrary common SHA is not an authority.
D105_CANONICAL_THREE_K_LOCK_AUTHORITY_SHA256: str | None = None

PREDECESSOR_NUMERIC_LOCK = MappingProxyType(
    {
        "student_nu": 3.0,
        "kernel_effective_dim": 12,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.35,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 0.85,
    }
)

DESIGN_TRACEABILITY = (
    ("G0-R2-01", "production is one archive/receipt-path call", "implemented"),
    ("G0-R2-02", "no public handle, rows, labels, or token", "implemented"),
    ("G0-R2-03", "synthetic and production schemas are disjoint", "implemented"),
    ("G0-R2-04", "strict actual-tap bytes bind fresh train-only three-K locks", "implemented"),
    ("G0-R2-05", "every K must change at least one argmax", "implemented"),
    ("G0-R2-06", "fold receipts bind all required roots", "implemented"),
    ("G0-R2-07", "RCMR state wire round-trip is mandatory", "implemented"),
    ("G0-R2-08", "tap arrays use immutable bytes backing and revalidation", "implemented"),
    ("G0-R2-09", "results are canonical immutable bytes", "implemented"),
    ("G0-R2-10", "resource accounting is an analysis budget, not RSS", "implemented"),
)


class D106RCMRG0Error(ValueError):
    """Raised when the G0 evidence or execution closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if type(value) is bytes else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106RCMRG0Error(f"{name} must be a lowercase SHA256")
    return value


def _require_commit(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106RCMRG0Error("expected release commit must be a lowercase Git SHA1")
    return value


def _token(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D106RCMRG0Error(f"{name} must be a non-empty exact string")
    if len(value.encode("utf-8")) > MAX_TOKEN_UTF8_BYTES:
        raise D106RCMRG0Error(f"{name} exceeds the UTF-8 byte limit")
    return value


def _typed_tokens(value: Any, name: str, *, count: int | None = None) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
            raise D106RCMRG0Error(f"{name} must be a one-dimensional string array")
        raw = value.astype(str).tolist()
    elif isinstance(value, (tuple, list)):
        raw = list(value)
    else:
        raise D106RCMRG0Error(f"{name} must be an exact token sequence")
    if count is not None and len(raw) != count:
        raise D106RCMRG0Error(f"{name} row count drift")
    return tuple(_token(item, name) for item in raw)


def _array_root(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256(
        _canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
        + b"\0"
        + array.tobytes(order="C")
    )


def _support_root(physical_ids: Sequence[str]) -> str:
    return _sha256(sorted(str(value) for value in physical_ids))


def _canonical_registry(values: Sequence[str]) -> tuple[str, ...]:
    registry = tuple(
        sorted(
            (_token(value, "registered class") for value in values),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(registry) != EXPECTED_CLASSES or len(set(registry)) != EXPECTED_CLASSES:
        raise D106RCMRG0Error("D106 G0 requires exactly six registered classes")
    return registry


def _regular_file_sha256(path: str | Path, name: str) -> tuple[Path, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D106RCMRG0Error(f"{name} must be an ordinary file")
    resolved = source.resolve()
    return resolved, hashlib.sha256(resolved.read_bytes()).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_sha256(function: Any) -> str:
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError) as error:
        raise D106RCMRG0Error("production callable source is unavailable") from error
    return hashlib.sha256(source).hexdigest()


_PRODUCTION_DEPENDENCY_MODULES = (
    __name__,
    _tap_module.__name__,
    _predecessor_module.__name__,
    _qknn_module.__name__,
    _rcmr_module.__name__,
)
_PRODUCTION_IMPORTED_CALLABLES = (
    "_load_d106_phase1_ls_tap",
    "_load_predecessor_lock_bundle",
    "_strict_reconstruct_predecessor_locks",
    "build_typed_zid_support_bank",
    "identity_shared_psd_metric",
    "score_zid_student_t_logits",
)
_PRODUCTION_STATIC_BINDINGS = (
    "_tap_module",
    "_predecessor_module",
    "_qknn_module",
    "_rcmr_module",
    "D106Phase1TapRows",
    "Phase1ZIDStudentTLock",
    "CANDIDATE_ID",
    "TAP_ARCHIVE_NAME",
    "TAP_RECEIPT_SCHEMA",
    "_STRICT_LOADER_CAPABILITY_ISSUER",
)


def _code_integrity_payload(code: CodeType) -> dict[str, Any]:
    """Return a source-independent, recursive code-object description.

    ``inspect.getsource`` is useful as a review receipt but is not an
    execution-integrity primitive: a replaced code object can spoof source
    coordinates.  This payload deliberately binds executable bytes, nested
    code, exception tables, arguments, names, and closure layout instead.
    """

    def constant(value: Any) -> Any:
        if isinstance(value, CodeType):
            return {"code": _code_integrity_payload(value)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [constant(item) for item in value]}
        if isinstance(value, frozenset):
            return {"frozenset": sorted((constant(item) for item in value), key=repr)}
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float:
            return {"float_hex": value.hex()}
        if type(value) is complex:
            return {"complex": [value.real.hex(), value.imag.hex()]}
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "constants": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "linetable": getattr(code, "co_linetable", b"").hex(),
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _runtime_value_payload(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Fingerprint defaults/closure cells without executing user code."""

    seen = set() if _seen is None else _seen
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    identity = id(value)
    if identity in seen:
        return {"cycle": identity}
    seen.add(identity)
    try:
        if isinstance(value, tuple):
            return {"tuple": [_runtime_value_payload(item, _seen=seen) for item in value]}
        if isinstance(value, list):
            return {"list": [_runtime_value_payload(item, _seen=seen) for item in value]}
        if isinstance(value, Mapping):
            entries = [
                [
                    _runtime_value_payload(key, _seen=seen),
                    _runtime_value_payload(member, _seen=seen),
                ]
                for key, member in value.items()
            ]
            return {"mapping": sorted(entries, key=lambda item: _canonical_bytes(item).hex())}
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "ndarray": {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                }
            }
        if isinstance(value, CodeType):
            return {"code": _code_integrity_payload(value)}
        if isinstance(value, FunctionType):
            return {
                "function": {
                    "module": value.__module__,
                    "qualname": value.__qualname__,
                    "identity": id(value),
                    "code_identity": id(value.__code__),
                }
            }
        return {
            "opaque": {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "identity": identity,
            }
        }
    finally:
        seen.discard(identity)


def _callable_source_sha256(function: FunctionType) -> str | None:
    try:
        return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        # Dataclass-generated methods can have no recoverable source.  Their
        # object/code/default/closure pins remain mandatory.
        return None


@dataclass(frozen=True, slots=True)
class _ProductionCallablePin:
    label: str
    owner: Any
    owner_name: str
    callable_value: FunctionType
    callable_module: Any
    module_name: str
    qualname: str
    code_object: CodeType
    code_sha256: str
    defaults_sha256: str
    kwdefaults_sha256: str
    closure_sha256: str
    source_sha256: str | None
    module_path: Path
    module_file_sha256: str


def _owner_callable(owner: Any, name: str) -> FunctionType | None:
    raw = vars(owner).get(name)
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    return raw if isinstance(raw, FunctionType) else None


def _capture_production_callable_pin(
    label: str, owner: Any, owner_name: str, function: FunctionType
) -> _ProductionCallablePin:
    callable_module = inspect.getmodule(function)
    module_file = getattr(callable_module, "__file__", None)
    code = getattr(function, "__code__", None)
    if (
        callable_module is None
        or type(module_file) is not str
        or not isinstance(code, CodeType)
        or type(function.__module__) is not str
        or type(function.__qualname__) is not str
    ):
        raise D106RCMRG0Error(f"cannot pin production callable: {label}")
    module_path = Path(module_file).resolve(strict=True)
    return _ProductionCallablePin(
        label=label,
        owner=owner,
        owner_name=owner_name,
        callable_value=function,
        callable_module=callable_module,
        module_name=function.__module__,
        qualname=function.__qualname__,
        code_object=code,
        code_sha256=_sha256(_code_integrity_payload(code)),
        defaults_sha256=_sha256(_runtime_value_payload(function.__defaults__)),
        kwdefaults_sha256=_sha256(_runtime_value_payload(function.__kwdefaults__)),
        closure_sha256=_sha256(
            _runtime_value_payload(
                tuple(cell.cell_contents for cell in (function.__closure__ or ()))
            )
        ),
        source_sha256=_callable_source_sha256(function),
        module_path=module_path,
        module_file_sha256=_file_sha256(module_path),
    )


def _capture_production_runtime_closure_minimal() -> tuple[
    Mapping[str, _ProductionCallablePin], Mapping[str, Any]
]:
    """Freeze the explicit five-module execution surface used by G0.

    The release manifest/clean child owns complete file-closure enforcement.
    This in-process guard deliberately remains narrow and catches direct
    callable/object/code replacement without becoming a dynamic module graph.
    """

    pins: dict[str, _ProductionCallablePin] = {}
    for module_name in _PRODUCTION_DEPENDENCY_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            raise D106RCMRG0Error(f"production dependency module missing: {module_name}")
        for name, value in sorted(vars(module).items()):
            if isinstance(value, FunctionType) and value.__module__ == module_name:
                label = f"{module_name}:{name}"
                pins[label] = _capture_production_callable_pin(label, module, name, value)
            elif isinstance(value, type) and value.__module__ == module_name:
                for method_name in sorted(vars(value)):
                    method = _owner_callable(value, method_name)
                    if method is not None:
                        label = f"{module_name}:{value.__qualname__}.{method_name}"
                        pins[label] = _capture_production_callable_pin(
                            label, value, method_name, method
                        )
    owner = sys.modules[__name__]
    for name in _PRODUCTION_IMPORTED_CALLABLES:
        function = _owner_callable(owner, name)
        if function is None:
            raise D106RCMRG0Error(f"production imported callable missing: {name}")
        label = f"{__name__}:import:{name}"
        pins[label] = _capture_production_callable_pin(label, owner, name, function)
    static = {name: vars(owner).get(name) for name in _PRODUCTION_STATIC_BINDINGS}
    if any(value is None for value in static.values()):
        raise D106RCMRG0Error("production static binding missing")
    return MappingProxyType(pins), MappingProxyType(static)


def _verify_production_callable_pin(pin: _ProductionCallablePin) -> None:
    current = _owner_callable(pin.owner, pin.owner_name)
    if current is not pin.callable_value:
        raise D106RCMRG0Error(f"production callable binding drift: {pin.label}")
    if (
        inspect.getmodule(current) is not pin.callable_module
        or current.__module__ != pin.module_name
        or current.__qualname__ != pin.qualname
        or current.__code__ is not pin.code_object
    ):
        raise D106RCMRG0Error(f"production callable code identity drift: {pin.label}")
    if (
        _sha256(_code_integrity_payload(current.__code__)) != pin.code_sha256
        or _sha256(_runtime_value_payload(current.__defaults__)) != pin.defaults_sha256
        or _sha256(_runtime_value_payload(current.__kwdefaults__))
        != pin.kwdefaults_sha256
        or _sha256(
            _runtime_value_payload(
                tuple(cell.cell_contents for cell in (current.__closure__ or ()))
            )
        )
        != pin.closure_sha256
        or _callable_source_sha256(current) != pin.source_sha256
    ):
        raise D106RCMRG0Error(f"production callable fingerprint drift: {pin.label}")
    module_file = getattr(pin.callable_module, "__file__", None)
    if type(module_file) is not str or Path(module_file).resolve(strict=True) != pin.module_path:
        raise D106RCMRG0Error(f"production callable module-path drift: {pin.label}")
    if _file_sha256(pin.module_path) != pin.module_file_sha256:
        raise D106RCMRG0Error(f"production callable module-file drift: {pin.label}")


_PRODUCTION_RUNTIME_PINS: Mapping[str, _ProductionCallablePin] = MappingProxyType({})
_PRODUCTION_RUNTIME_STATIC: Mapping[str, Any] = MappingProxyType({})


def _current_code_sha256() -> dict[str, str]:
    return {
        "g0_executor_module": _file_sha256(__file__),
        "tap_loader_module": _file_sha256(_tap_module.__file__),
        "tap_loader_callable": _source_sha256(_load_d106_phase1_ls_tap),
        "predecessor_lock_module": _file_sha256(_predecessor_module.__file__),
        "predecessor_lock_bundle_callable": _source_sha256(_load_predecessor_lock_bundle),
        "predecessor_lock_reconstruct_callable": _source_sha256(
            _strict_reconstruct_predecessor_locks
        ),
        "rcmr_module": _file_sha256(_rcmr_module.__file__),
        "qknn_module": _file_sha256(_qknn_module.__file__),
    }


def get_d106_rcmr_g0_release_expected_code_sha256() -> dict[str, str]:
    """Return a fresh release-manifest code map without granting authority.

    This local packaging helper reports the current closed source digest map.
    It neither executes production nor authorizes a production, promotion, or
    performance claim.  A new ordinary ``dict`` is returned on every call so a
    caller cannot mutate retained G0 state.
    """

    return dict(_current_code_sha256())


@dataclass(frozen=True, slots=True)
class D106RCMRG0ProductionRequest:
    """External expectations only; this object carries no authority capability."""

    registered_classes: tuple[str, ...]
    expected_release_commit: str
    expected_code_sha256: tuple[tuple[str, str], ...]
    expected_d105_lock_authority_sha256: str | None = None
    request_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        registry = _canonical_registry(self.registered_classes)
        commit = _require_commit(self.expected_release_commit)
        pairs = tuple(self.expected_code_sha256)
        required = tuple(sorted(_current_code_sha256()))
        if (
            len(pairs) != len(required)
            or tuple(sorted(name for name, _value in pairs)) != required
            or len({name for name, _value in pairs}) != len(required)
        ):
            raise D106RCMRG0Error("production expected-code key closure drift")
        canonical_pairs = tuple(sorted((str(name), _require_sha256(value, name)) for name, value in pairs))
        authority = self.expected_d105_lock_authority_sha256
        if authority is not None:
            authority = _require_sha256(authority, "D105 three-K lock authority")
        payload = {
            "schema": PRODUCTION_SCHEMA + ".request.v1",
            "registered_classes": list(registry),
            "expected_release_commit": commit,
            "expected_code_sha256": dict(canonical_pairs),
            "expected_d105_lock_authority_sha256": authority,
        }
        object.__setattr__(self, "registered_classes", registry)
        object.__setattr__(self, "expected_release_commit", commit)
        object.__setattr__(self, "expected_code_sha256", canonical_pairs)
        object.__setattr__(self, "expected_d105_lock_authority_sha256", authority)
        object.__setattr__(self, "request_receipt_sha256", _sha256(payload))


@dataclass(frozen=True, slots=True)
class _StrictLoaderCapability:
    """One-use misuse guard, not a same-process security authority.

    The actual authority boundary is the clean child interpreter and its
    externally pinned release manifest.  This object only prevents accidental
    direct rows/locks calls from reaching the public production serializer.
    """

    issuer: object = field(repr=False)
    rows_identity: int
    tap_archive_sha256: str
    tap_receipt_sha256: str
    predecessor_lock_bundle_root_sha256: str
    request_receipt_sha256: str
    code_closure_root_sha256: str
    strict_loader_evidence: Mapping[str, Any]


_STRICT_LOADER_CAPABILITY_ISSUER = object()
_ISSUED_STRICT_LOADER_CAPABILITIES: dict[int, _StrictLoaderCapability] = {}


def _code_closure_root(code_sha256: Mapping[str, str]) -> str:
    if type(code_sha256) is not dict or not code_sha256:
        raise D106RCMRG0Error("production code closure must be a non-empty exact map")
    canonical: dict[str, str] = {}
    for name, value in code_sha256.items():
        if type(name) is not str:
            raise D106RCMRG0Error("production code closure key type drift")
        canonical[name] = _require_sha256(value, f"production code {name}")
    return _sha256(canonical)


def _mint_strict_loader_capability(
    *,
    rows: D106Phase1TapRows,
    tap_archive_sha256: str,
    tap_receipt_sha256: str,
    predecessor_lock_bundle_root_sha256: str,
    request: D106RCMRG0ProductionRequest,
    code_sha256: Mapping[str, str],
) -> _StrictLoaderCapability:
    """Bind the only production serializer call to verified actual-file input."""

    archive_sha = _require_sha256(tap_archive_sha256, "strict tap archive")
    receipt_sha = _require_sha256(tap_receipt_sha256, "strict tap receipt")
    bundle_root = _require_sha256(
        predecessor_lock_bundle_root_sha256, "strict predecessor bundle"
    )
    request_receipt = _require_sha256(
        request.request_receipt_sha256, "strict production request receipt"
    )
    code_root = _code_closure_root(dict(code_sha256))
    evidence = {
        "schema": STRICT_LOADER_EVIDENCE_SCHEMA,
        "issuer": "clean_child_public_actual_file_entry",
        "tap_archive_name": TAP_ARCHIVE_NAME,
        "tap_archive_sha256": archive_sha,
        "tap_receipt_sha256": receipt_sha,
        "predecessor_lock_bundle_root_sha256": bundle_root,
        "request_receipt_sha256": request_receipt,
        "code_closure_root_sha256": code_root,
    }
    evidence["strict_loader_evidence_receipt_sha256"] = _sha256(evidence)
    capability = _StrictLoaderCapability(
        issuer=_STRICT_LOADER_CAPABILITY_ISSUER,
        rows_identity=id(rows),
        tap_archive_sha256=archive_sha,
        tap_receipt_sha256=receipt_sha,
        predecessor_lock_bundle_root_sha256=bundle_root,
        request_receipt_sha256=request_receipt,
        code_closure_root_sha256=code_root,
        strict_loader_evidence=MappingProxyType(dict(evidence)),
    )
    _ISSUED_STRICT_LOADER_CAPABILITIES[id(capability)] = capability
    return capability


def _consume_strict_loader_capability(
    capability: _StrictLoaderCapability,
    *,
    rows: D106Phase1TapRows,
    tap_archive_sha256: str,
    tap_receipt_sha256: str,
    predecessor_lock_bundle_root_sha256: str,
    request: D106RCMRG0ProductionRequest,
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Consume the one-use guard and rebind every final serializer input."""

    if type(capability) is not _StrictLoaderCapability:
        raise D106RCMRG0Error("public production serialization requires strict-loader capability")
    issued = _ISSUED_STRICT_LOADER_CAPABILITIES.pop(id(capability), None)
    if issued is not capability or capability.issuer is not _STRICT_LOADER_CAPABILITY_ISSUER:
        raise D106RCMRG0Error("strict-loader capability was not issued by actual-file entry")
    if (
        capability.rows_identity != id(rows)
        or capability.tap_archive_sha256 != _require_sha256(
            tap_archive_sha256, "production tap archive"
        )
        or capability.tap_receipt_sha256 != _require_sha256(
            tap_receipt_sha256, "production tap receipt"
        )
        or capability.predecessor_lock_bundle_root_sha256
        != _require_sha256(predecessor_lock_bundle_root_sha256, "production predecessor bundle")
        or capability.request_receipt_sha256
        != _require_sha256(request.request_receipt_sha256, "production request receipt")
        or capability.code_closure_root_sha256 != _code_closure_root(dict(code_sha256))
    ):
        raise D106RCMRG0Error("strict-loader capability evidence binding drift")
    evidence = dict(capability.strict_loader_evidence)
    receipt = evidence.pop("strict_loader_evidence_receipt_sha256", None)
    if (
        _require_sha256(receipt, "strict-loader evidence receipt") != _sha256(evidence)
        or evidence.get("schema") != STRICT_LOADER_EVIDENCE_SCHEMA
        or evidence.get("issuer") != "clean_child_public_actual_file_entry"
        or evidence.get("tap_archive_name") != TAP_ARCHIVE_NAME
    ):
        raise D106RCMRG0Error("strict-loader evidence receipt drift")
    evidence["strict_loader_evidence_receipt_sha256"] = receipt
    return evidence


def _rebuild_production_request_payload(
    request: D106RCMRG0ProductionRequest,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Revalidate every frozen request field against its canonical receipt.

    ``frozen=True`` does not protect against a same-process
    ``object.__setattr__``.  Production therefore reconstructs the receipt
    payload immediately before using any request field.
    """

    if type(request) is not D106RCMRG0ProductionRequest:
        raise D106RCMRG0Error("production requires the exact request type")
    if type(request.registered_classes) is not tuple:
        raise D106RCMRG0Error("production request registered-class type drift")
    registry = _canonical_registry(request.registered_classes)
    if request.registered_classes != registry:
        raise D106RCMRG0Error("production request registered-class value drift")
    commit = _require_commit(request.expected_release_commit)
    pairs = request.expected_code_sha256
    if type(pairs) is not tuple:
        raise D106RCMRG0Error("production request expected-code type drift")
    required = tuple(sorted(_current_code_sha256()))
    if len(pairs) != len(required):
        raise D106RCMRG0Error("production request expected-code length drift")
    canonical_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str:
            raise D106RCMRG0Error("production request expected-code pair type drift")
        name, value = pair
        canonical_pairs.append((name, _require_sha256(value, f"request code {name}")))
    canonical = tuple(sorted(canonical_pairs))
    if (
        tuple(name for name, _value in canonical) != required
        or len({name for name, _value in canonical}) != len(required)
        or pairs != canonical
    ):
        raise D106RCMRG0Error("production request expected-code value drift")
    authority = request.expected_d105_lock_authority_sha256
    if authority is not None:
        authority = _require_sha256(authority, "request D105 lock authority")
    payload = {
        "schema": PRODUCTION_SCHEMA + ".request.v1",
        "registered_classes": list(registry),
        "expected_release_commit": commit,
        "expected_code_sha256": dict(canonical),
        "expected_d105_lock_authority_sha256": authority,
    }
    if (
        type(request.request_receipt_sha256) is not str
        or _require_sha256(request.request_receipt_sha256, "production request receipt")
        != _sha256(payload)
    ):
        raise D106RCMRG0Error("production request receipt/value closure drift")
    return dict(canonical), payload


def _verify_production_code_closure(request: D106RCMRG0ProductionRequest) -> dict[str, str]:
    expected_code, _request_payload = _rebuild_production_request_payload(request)
    if not _PRODUCTION_RUNTIME_PINS or not _PRODUCTION_RUNTIME_STATIC:
        raise D106RCMRG0Error("production runtime closure is not frozen")
    owner = sys.modules[__name__]
    for name, expected in _PRODUCTION_RUNTIME_STATIC.items():
        if vars(owner).get(name) is not expected:
            raise D106RCMRG0Error(f"production static binding drift: {name}")
    for pin in _PRODUCTION_RUNTIME_PINS.values():
        _verify_production_callable_pin(pin)
    actual = _current_code_sha256()
    if actual != expected_code:
        raise D106RCMRG0Error("production source/code SHA256 drift detected")
    return actual


def _seal_result(payload: Mapping[str, Any]) -> bytes:
    if "result_receipt_sha256" in payload:
        raise D106RCMRG0Error("result payload already contains a receipt")
    result = dict(payload)
    result["result_receipt_sha256"] = _sha256(payload)
    return _canonical_bytes(result)


def _verify_synthetic_nested_receipts(document: dict[str, Any]) -> None:
    if (
        document.get("status") != SYNTHETIC_STATUS
        or document.get("algorithm_execution_scope") != ALGORITHM_SCOPE
        or document.get("runner_authority") is not False
        or document.get("deployable") is not False
        or document.get("formal_performance_claim") is not False
        or document.get("real_g0_executed") is not False
        or document.get("external_execution_manifest_root_sha256") is not None
        or document.get("opaque_claim_not_independently_verified") is not True
        or document.get("promotion_or_runner_consumption_allowed") is not False
        or document.get("K_values") != list(K_VALUES)
        or document.get("fold_count") != EXPECTED_FOLDS
        or document.get("query_count_per_k") != EXPECTED_ROWS
    ):
        raise D106RCMRG0Error("synthetic result lifecycle closure drift")
    common_root = _require_sha256(
        document.get("common_query_order_root_sha256"), "common query order root"
    )
    snapshot_root = _require_sha256(
        document.get("tap_snapshot_root_sha256"), "tap snapshot root"
    )
    tap_receipt = _require_sha256(document.get("tap_receipt_sha256"), "tap receipt")
    registry_root = _require_sha256(document.get("registry_root_sha256"), "registry root")
    per_k = document.get("per_k")
    if type(per_k) is not list or len(per_k) != len(K_VALUES):
        raise D106RCMRG0Error("synthetic per-K closure drift")
    identity_sequences: list[tuple[str, ...]] = []
    changed_by_k: dict[str, int] = {}
    per_k_receipts: list[str] = []
    for active_k, item in zip(K_VALUES, per_k, strict=True):
        if type(item) is not dict or item.get("schema") != PER_K_SCHEMA or item.get("K") != active_k:
            raise D106RCMRG0Error("synthetic per-K schema/K drift")
        per_k_receipt = _require_sha256(
            item.get("per_k_receipt_sha256"), "per-K receipt"
        )
        without_per_k_receipt = dict(item)
        without_per_k_receipt.pop("per_k_receipt_sha256")
        if _sha256(without_per_k_receipt) != per_k_receipt:
            raise D106RCMRG0Error("synthetic per-K receipt mismatch")
        if (
            item.get("common_query_order_root_sha256") != common_root
            or item.get("query_ids_root_sha256") != common_root
            or item.get("tap_snapshot_root_sha256") != snapshot_root
            or item.get("tap_receipt_sha256") != tap_receipt
            or item.get("registry_root_sha256") != registry_root
        ):
            raise D106RCMRG0Error("synthetic per-K/top binding drift")
        folds = item.get("fold_execution_receipts")
        if type(folds) is not list or len(folds) != EXPECTED_FOLDS:
            raise D106RCMRG0Error("synthetic fold receipt count drift")
        fold_receipts: list[str] = []
        fold_identities: list[str] = []
        fold_bitmap_roots: list[str] = []
        derived_changed = 0
        for expected_index, fold in enumerate(folds):
            if (
                type(fold) is not dict
                or fold.get("schema") != FOLD_EXECUTION_SCHEMA
                or fold.get("fold_index") != expected_index
                or fold.get("K") != active_k
                or fold.get("common_query_order_root_sha256") != common_root
                or fold.get("tap_snapshot_root_sha256") != snapshot_root
                or fold.get("tap_receipt_sha256") != tap_receipt
                or fold.get("registry_root_sha256") != registry_root
                or fold.get("algorithm_execution_scope") != ALGORITHM_SCOPE
                or fold.get("runner_authority") is not False
                or fold.get("p2_validated_or_deployable_claimed") is not False
                or fold.get("external_execution_manifest_root_sha256") is not None
                or fold.get("opaque_claim_not_independently_verified") is not True
                or fold.get("promotion_or_runner_consumption_allowed") is not False
            ):
                raise D106RCMRG0Error("synthetic fold lifecycle/top binding drift")
            for name in (
                "query_root_sha256",
                "query_plus_root_sha256",
                "query_signed_root_sha256",
                "query_indices_root_sha256",
                "support_pool_indices_root_sha256",
                "support_indices_root_sha256",
                "support_root_sha256",
                "baseline_bank_receipt_sha256",
                "paired_view_receipt_sha256",
                "rcmr_state_receipt_sha256",
                "rcmr_wire_sha256",
                "rcmr_method_lock_sha256",
                "candidate_argmax_root_sha256",
                "baseline_argmax_root_sha256",
            ):
                _require_sha256(fold.get(name), f"fold {name}")
            bitmap = fold.get("argmax_changed_bitmap")
            if (
                type(bitmap) is not str
                or not 1 <= len(bitmap) <= MAX_QUERY_ROWS_PER_FOLD
                or any(bit not in "01" for bit in bitmap)
            ):
                raise D106RCMRG0Error("synthetic fold changed bitmap encoding drift")
            bitmap_root = _sha256(
                {
                    "encoding": "ascii01_query_order",
                    "query_count": len(bitmap),
                    "bits": bitmap,
                }
            )
            fold_changed = bitmap.count("1")
            if (
                fold.get("argmax_changed_bitmap_root_sha256") != bitmap_root
                or fold.get("argmax_changed_count") != fold_changed
            ):
                raise D106RCMRG0Error("synthetic fold changed bitmap/count drift")
            expected_identity = _sha256(
                {
                    "schema": FOLD_EXECUTION_SCHEMA + ".full_identity.v1",
                    "fold_index": expected_index,
                    "fold_id": _token(fold.get("fold_id"), "fold ID"),
                    "receiver_id": _token(fold.get("receiver_id"), "receiver ID"),
                    "day_id": _token(fold.get("day_id"), "day ID"),
                    "query_root_sha256": fold["query_root_sha256"],
                    "tap_snapshot_root_sha256": snapshot_root,
                }
            )
            if fold.get("fold_identity_root_sha256") != expected_identity:
                raise D106RCMRG0Error("synthetic full fold identity mismatch")
            execution_receipt = _require_sha256(
                fold.get("execution_receipt_sha256"), "fold execution receipt"
            )
            without_execution_receipt = dict(fold)
            without_execution_receipt.pop("execution_receipt_sha256")
            if _sha256(without_execution_receipt) != execution_receipt:
                raise D106RCMRG0Error("synthetic fold execution receipt mismatch")
            fold_receipts.append(execution_receipt)
            fold_identities.append(expected_identity)
            fold_bitmap_roots.append(bitmap_root)
            derived_changed += fold_changed
        if item.get("fold_execution_receipts_root_sha256") != _sha256(fold_receipts):
            raise D106RCMRG0Error("synthetic fold-receipt aggregate root mismatch")
        if item.get("fold_changed_bitmap_roots_root_sha256") != _sha256(
            fold_bitmap_roots
        ):
            raise D106RCMRG0Error("synthetic fold bitmap aggregate root mismatch")
        identity_sequences.append(tuple(fold_identities))
        if item.get("argmax_changed_count") != derived_changed:
            raise D106RCMRG0Error("synthetic per-K changed count/bitmap drift")
        changed_by_k[str(active_k)] = derived_changed
        per_k_receipts.append(per_k_receipt)
    if len(set(identity_sequences)) != 1:
        raise D106RCMRG0Error("synthetic fold identity sequence differs across K")
    zero_changed = [active_k for active_k in K_VALUES if changed_by_k[str(active_k)] == 0]
    expected_status = (
        "G0_EVERY_K_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
        if not zero_changed
        else "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    )
    if (
        document.get("argmax_changed_count_by_k") != changed_by_k
        or document.get("argmax_changed_count") != sum(changed_by_k.values())
        or document.get("zero_changed_k_values") != zero_changed
        or document.get("functional_gate_pass") is not (not zero_changed)
        or document.get("functional_gate_status") != expected_status
        or document.get("canonical_execution_root_sha256") != _sha256(per_k_receipts)
    ):
        raise D106RCMRG0Error("synthetic aggregate functional/root closure drift")


_PRODUCTION_AUTHORITY_FLAGS = MappingProxyType(
    {
        "d105_formal_authority": False,
        "held_truth_access": False,
        "performance_metrics_computed": False,
        "phase2_promotion_authority": False,
        "runner_authority": False,
        "target_access": False,
    }
)


_PRODUCTION_FOLD_ROOT_FIELDS = (
    "fold_identity_root_sha256",
    "query_root_sha256",
    "query_plus_root_sha256",
    "query_signed_root_sha256",
    "query_indices_root_sha256",
    "support_pool_indices_root_sha256",
    "support_indices_root_sha256",
    "support_root_sha256",
    "baseline_bank_receipt_sha256",
    "paired_view_receipt_sha256",
    "rcmr_state_receipt_sha256",
    "rcmr_wire_sha256",
    "rcmr_method_lock_sha256",
    "registry_root_sha256",
    "common_query_order_root_sha256",
    "tap_receipt_sha256",
    "tap_snapshot_root_sha256",
    "candidate_argmax_root_sha256",
    "baseline_argmax_root_sha256",
)


def _verify_production_nested_receipts(document: dict[str, Any]) -> None:
    """Verify the public root/bitmap/count-only production receipt tree."""

    expected_top_keys = {
        "schema",
        "status",
        "candidate_id",
        "protocol_schema",
        "algorithm_execution_scope",
        "real_g0_executed",
        "runner_authority",
        "deployable",
        "formal_performance_claim",
        "authority_flags",
        "held_label_audit_status",
        "same_process_held_label_capability_absence_claimed",
        "rows_or_labels_returned",
        "expected_release_commit",
        "request_receipt_sha256",
        "code_sha256",
        "tap_archive_sha256",
        "tap_receipt_sha256",
        "predecessor_lock_bundle_root_sha256",
        "strict_loader_evidence",
        "execution_manifest",
        "execution_manifest_root_sha256",
    }
    if (
        set(document) != expected_top_keys
        or document.get("status") != PRODUCTION_EXECUTED_STATUS
        or document.get("candidate_id") != CANDIDATE_ID
        or document.get("protocol_schema") != PROTOCOL_SCHEMA
        or document.get("algorithm_execution_scope") != ALGORITHM_SCOPE
        or document.get("real_g0_executed") is not True
        or document.get("runner_authority") is not False
        or document.get("deployable") is not False
        or document.get("formal_performance_claim") is not False
        or document.get("authority_flags") != dict(_PRODUCTION_AUTHORITY_FLAGS)
        or document.get("held_label_audit_status") != HELD_LABEL_AUDIT_STATUS
        or document.get("same_process_held_label_capability_absence_claimed") is not False
        or document.get("rows_or_labels_returned") is not False
    ):
        raise D106RCMRG0Error("production lifecycle/authority closure drift")
    _require_commit(document.get("expected_release_commit"))
    _require_sha256(document.get("request_receipt_sha256"), "production request receipt")
    code = document.get("code_sha256")
    if type(code) is not dict or set(code) != set(_current_code_sha256()):
        raise D106RCMRG0Error("production code closure receipt drift")
    for name, value in code.items():
        _require_sha256(value, f"production code {name}")
    archive_sha = _require_sha256(document.get("tap_archive_sha256"), "tap archive")
    receipt_sha = _require_sha256(document.get("tap_receipt_sha256"), "tap receipt")
    _require_sha256(
        document.get("predecessor_lock_bundle_root_sha256"), "predecessor lock bundle"
    )
    strict_evidence = document.get("strict_loader_evidence")
    expected_evidence_keys = {
        "schema",
        "issuer",
        "tap_archive_name",
        "tap_archive_sha256",
        "tap_receipt_sha256",
        "predecessor_lock_bundle_root_sha256",
        "request_receipt_sha256",
        "code_closure_root_sha256",
        "strict_loader_evidence_receipt_sha256",
    }
    if (
        type(strict_evidence) is not dict
        or set(strict_evidence) != expected_evidence_keys
        or strict_evidence.get("schema") != STRICT_LOADER_EVIDENCE_SCHEMA
        or strict_evidence.get("issuer") != "clean_child_public_actual_file_entry"
        or strict_evidence.get("tap_archive_name") != TAP_ARCHIVE_NAME
        or strict_evidence.get("tap_archive_sha256") != archive_sha
        or strict_evidence.get("tap_receipt_sha256") != receipt_sha
        or strict_evidence.get("predecessor_lock_bundle_root_sha256")
        != document.get("predecessor_lock_bundle_root_sha256")
        or strict_evidence.get("request_receipt_sha256")
        != document.get("request_receipt_sha256")
        or strict_evidence.get("code_closure_root_sha256") != _code_closure_root(code)
    ):
        raise D106RCMRG0Error("production strict-loader evidence closure drift")
    strict_evidence_payload = dict(strict_evidence)
    strict_evidence_receipt = _require_sha256(
        strict_evidence_payload.pop("strict_loader_evidence_receipt_sha256", None),
        "strict-loader evidence receipt",
    )
    if _sha256(strict_evidence_payload) != strict_evidence_receipt:
        raise D106RCMRG0Error("production strict-loader evidence receipt mismatch")
    manifest = document.get("execution_manifest")
    if type(manifest) is not dict:
        raise D106RCMRG0Error("production execution manifest is not an object")
    manifest_bytes = _canonical_bytes(manifest)
    if document.get("execution_manifest_root_sha256") != _sha256(manifest_bytes):
        raise D106RCMRG0Error("production execution manifest root mismatch")
    expected_manifest_keys = {
        "schema",
        "candidate_id",
        "protocol_schema",
        "algorithm_execution_scope",
        "K_values",
        "fold_count",
        "query_count_per_k",
        "common_query_order_root_sha256",
        "tap_archive_sha256",
        "tap_receipt_sha256",
        "tap_snapshot_root_sha256",
        "registry_root_sha256",
        "predecessor_lock_bundle_root_sha256",
        "predecessor_lock_set_root_sha256",
        "per_k",
        "per_k_execution_receipts_root_sha256",
        "canonical_execution_root_sha256",
        "argmax_changed_count_by_k",
        "argmax_changed_count",
        "zero_changed_k_values",
        "functional_gate_pass",
        "functional_gate_status",
        "authority_flags",
        "strict_loader_evidence_receipt_sha256",
        "execution_manifest_receipt_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema") != PRODUCTION_EXECUTION_MANIFEST_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
        or manifest.get("algorithm_execution_scope") != ALGORITHM_SCOPE
        or manifest.get("K_values") != list(K_VALUES)
        or manifest.get("fold_count") != EXPECTED_FOLDS
        or manifest.get("query_count_per_k") != EXPECTED_ROWS
        or manifest.get("tap_archive_sha256") != archive_sha
        or manifest.get("tap_receipt_sha256") != receipt_sha
        or manifest.get("predecessor_lock_bundle_root_sha256")
        != document.get("predecessor_lock_bundle_root_sha256")
        or manifest.get("strict_loader_evidence_receipt_sha256")
        != strict_evidence_receipt
        or manifest.get("authority_flags") != dict(_PRODUCTION_AUTHORITY_FLAGS)
    ):
        raise D106RCMRG0Error("production execution manifest closure drift")
    for name in (
        "common_query_order_root_sha256",
        "tap_snapshot_root_sha256",
        "registry_root_sha256",
        "predecessor_lock_set_root_sha256",
    ):
        _require_sha256(manifest.get(name), f"production manifest {name}")
    receipt = _require_sha256(
        manifest.get("execution_manifest_receipt_sha256"), "production manifest receipt"
    )
    receipt_payload = dict(manifest)
    receipt_payload.pop("execution_manifest_receipt_sha256")
    if _sha256(receipt_payload) != receipt:
        raise D106RCMRG0Error("production execution manifest receipt mismatch")
    per_k = manifest.get("per_k")
    if type(per_k) is not list or len(per_k) != len(K_VALUES):
        raise D106RCMRG0Error("production per-K closure drift")
    per_k_receipts: list[str] = []
    changed_by_k: dict[str, int] = {}
    identity_sequences: list[tuple[str, ...]] = []
    for active_k, item in zip(K_VALUES, per_k, strict=True):
        expected_per_k_keys = {
            "schema",
            "K",
            "fold_count",
            "query_count",
            "fold_execution_receipts",
            "fold_execution_receipts_root_sha256",
            "fold_changed_bitmap_roots_root_sha256",
            "argmax_changed_count",
            "per_k_receipt_sha256",
        }
        if (
            type(item) is not dict
            or set(item) != expected_per_k_keys
            or item.get("schema") != PRODUCTION_PER_K_SCHEMA
            or item.get("K") != active_k
            or item.get("fold_count") != EXPECTED_FOLDS
            or item.get("query_count") != EXPECTED_ROWS
        ):
            raise D106RCMRG0Error("production per-K schema/count drift")
        folds = item.get("fold_execution_receipts")
        if type(folds) is not list or len(folds) != EXPECTED_FOLDS:
            raise D106RCMRG0Error("production fold receipt count drift")
        fold_receipts: list[str] = []
        bitmap_roots: list[str] = []
        identities: list[str] = []
        changed = 0
        query_rows = 0
        for index, fold in enumerate(folds):
            expected_fold_keys = {
                "schema",
                "K",
                "fold_index",
                *_PRODUCTION_FOLD_ROOT_FIELDS,
                "argmax_changed_bitmap",
                "argmax_changed_bitmap_root_sha256",
                "argmax_changed_count",
                "execution_receipt_sha256",
            }
            if (
                type(fold) is not dict
                or set(fold) != expected_fold_keys
                or fold.get("schema") != PRODUCTION_FOLD_EXECUTION_SCHEMA
                or fold.get("K") != active_k
                or fold.get("fold_index") != index
            ):
                raise D106RCMRG0Error("production fold schema/key closure drift")
            for name in _PRODUCTION_FOLD_ROOT_FIELDS:
                _require_sha256(fold.get(name), f"production fold {name}")
            bitmap = fold.get("argmax_changed_bitmap")
            if (
                type(bitmap) is not str
                or not 1 <= len(bitmap) <= MAX_QUERY_ROWS_PER_FOLD
                or any(bit not in "01" for bit in bitmap)
            ):
                raise D106RCMRG0Error("production fold bitmap encoding drift")
            bitmap_root = _sha256(
                {
                    "encoding": "ascii01_query_order",
                    "query_count": len(bitmap),
                    "bits": bitmap,
                }
            )
            if (
                fold.get("argmax_changed_bitmap_root_sha256") != bitmap_root
                or fold.get("argmax_changed_count") != bitmap.count("1")
            ):
                raise D106RCMRG0Error("production fold bitmap/count drift")
            fold_receipt = _require_sha256(
                fold.get("execution_receipt_sha256"), "production fold receipt"
            )
            receipt_without = dict(fold)
            receipt_without.pop("execution_receipt_sha256")
            if _sha256(receipt_without) != fold_receipt:
                raise D106RCMRG0Error("production fold receipt mismatch")
            fold_receipts.append(fold_receipt)
            bitmap_roots.append(bitmap_root)
            identities.append(fold["fold_identity_root_sha256"])
            changed += bitmap.count("1")
            query_rows += len(bitmap)
        if (
            item.get("fold_execution_receipts_root_sha256") != _sha256(fold_receipts)
            or item.get("fold_changed_bitmap_roots_root_sha256") != _sha256(bitmap_roots)
            or item.get("argmax_changed_count") != changed
            or query_rows != item.get("query_count")
        ):
            raise D106RCMRG0Error("production per-K aggregate closure drift")
        per_k_receipt = _require_sha256(item.get("per_k_receipt_sha256"), "production per-K receipt")
        per_k_without = dict(item)
        per_k_without.pop("per_k_receipt_sha256")
        if _sha256(per_k_without) != per_k_receipt:
            raise D106RCMRG0Error("production per-K receipt mismatch")
        per_k_receipts.append(per_k_receipt)
        changed_by_k[str(active_k)] = changed
        identity_sequences.append(tuple(identities))
    if len(set(identity_sequences)) != 1:
        raise D106RCMRG0Error("production fold identity order differs across K")
    zero_changed = [active_k for active_k in K_VALUES if changed_by_k[str(active_k)] == 0]
    expected_status = (
        "G0_EVERY_K_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
        if not zero_changed
        else "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    )
    if (
        manifest.get("per_k_execution_receipts_root_sha256") != _sha256(per_k_receipts)
        or manifest.get("canonical_execution_root_sha256") != _sha256(per_k_receipts)
        or manifest.get("argmax_changed_count_by_k") != changed_by_k
        or manifest.get("argmax_changed_count") != sum(changed_by_k.values())
        or manifest.get("zero_changed_k_values") != zero_changed
        or manifest.get("functional_gate_pass") is not (not zero_changed)
        or manifest.get("functional_gate_status") != expected_status
    ):
        raise D106RCMRG0Error("production functional-gate/root closure drift")


def _verify_d106_rcmr_g0_result_bytes_for_schema(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_schema: str,
) -> bytes:
    """Verify one exact result schema under a required external SHA256."""

    if type(payload) is not bytes:
        raise D106RCMRG0Error("G0 result must be immutable bytes")
    if _sha256(payload) != _require_sha256(expected_sha256, "externally anchored result bytes"):
        raise D106RCMRG0Error("G0 result external SHA256 mismatch")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0Error("G0 result is not canonical JSON") from error
    if type(document) is not dict or payload != _canonical_bytes(document):
        raise D106RCMRG0Error("G0 result canonical byte closure drift")
    if document.get("schema") != expected_schema:
        raise D106RCMRG0Error("G0 result schema is not accepted by this verifier")
    receipt = _require_sha256(document.pop("result_receipt_sha256", None), "result receipt")
    if expected_schema == PRODUCTION_SCHEMA:
        _verify_production_nested_receipts(document)
    elif expected_schema == SYNTHETIC_SCHEMA:
        _verify_synthetic_nested_receipts(document)
    else:
        raise D106RCMRG0Error("G0 verifier schema configuration drift")
    if _sha256(document) != receipt:
        raise D106RCMRG0Error("G0 nested result receipt mismatch")
    return payload


def verify_d106_rcmr_g0_production_result_bytes(
    payload: bytes, *, expected_sha256: str
) -> bytes:
    """Verify only a public actual-file production result.

    This function is not an authority issuer and cannot promote a result. The
    supplied SHA256 is external to this verifier; recomputing it from the bytes
    being checked defeats that boundary.
    """

    return _verify_d106_rcmr_g0_result_bytes_for_schema(
        payload,
        expected_sha256=expected_sha256,
        expected_schema=PRODUCTION_SCHEMA,
    )


def verify_d106_rcmr_g0_synthetic_test_result_bytes(
    payload: bytes, *, expected_sha256: str
) -> bytes:
    """Verify only synthetic test mechanics; it is not an authority issuer.

    This test-only entry accepts no production schema and cannot authorize a
    production, promotion, or performance claim.
    """

    return _verify_d106_rcmr_g0_result_bytes_for_schema(
        payload,
        expected_sha256=expected_sha256,
        expected_schema=SYNTHETIC_SCHEMA,
    )


def run_d106_rcmr_g0_from_formal_tap(
    archive_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_receipt_sha256: str,
    request: D106RCMRG0ProductionRequest,
) -> bytes:
    """Execute the non-formal mechanical G0 closure from a strict tap path.

    The public call accepts only ordinary archive/receipt paths plus external
    digests.  It strictly loads the sealed 588-row tap before it constructs the
    train-only predecessor lock bytes and privately rebuilds fresh locks.
    """

    actual_code = _verify_production_code_closure(request)
    # The strict loader hashes and parses each actual input from the same
    # O_NOFOLLOW file descriptor.  Do not pre-hash then reopen paths here.
    actual_archive = _require_sha256(expected_archive_sha256, "formal tap archive")
    actual_receipt = _require_sha256(expected_receipt_sha256, "formal tap receipt")
    try:
        rows = _load_d106_phase1_ls_tap(
            archive_path,
            receipt_path,
            expected_archive_sha256=actual_archive,
            expected_receipt_sha256=actual_receipt,
        )
    except Exception as error:
        raise D106RCMRG0Error("strict D106 tap loader rejected production inputs") from error
    try:
        bundle_bytes = _load_predecessor_lock_bundle(
            tap_archive_sha256=actual_archive,
            tap_receipt_sha256=actual_receipt,
        )
        if type(bundle_bytes) is not bytes:
            raise D106RCMRG0Error("predecessor bundle must publish immutable bytes")
        bundle_document = json.loads(bundle_bytes.decode("utf-8"))
        if type(bundle_document) is not dict or _canonical_bytes(bundle_document) != bundle_bytes:
            raise D106RCMRG0Error("predecessor bundle canonical-byte closure drift")
        if (
            bundle_document.get("d106_tap_archive_sha256") != actual_archive
            or bundle_document.get("d106_tap_receipt_sha256") != actual_receipt
            or bundle_document.get("external_strict_tap_loader_binding_required") is not True
        ):
            raise D106RCMRG0Error("predecessor bundle actual-tap binding drift")
        initial_locks = _strict_reconstruct_predecessor_locks(bundle_bytes)
        predecessor_locks = _strict_reconstruct_predecessor_locks(bundle_bytes)
    except Exception as error:
        if isinstance(error, D106RCMRG0Error):
            raise
        raise D106RCMRG0Error("strict predecessor-lock reconstruction failed") from error
    if (
        predecessor_locks is initial_locks
        or len(predecessor_locks) != len(initial_locks)
        or any(left is right for left, right in zip(initial_locks, predecessor_locks, strict=True))
    ):
        raise D106RCMRG0Error("predecessor locks were not rebuilt fresh")
    locks = _validate_synthetic_locks(
        predecessor_locks, tap_receipt_sha256=actual_receipt
    )
    method_lock_sha = _require_sha256(
        bundle_document.get("d106_rcmr_method_lock_sha256"), "RCMR method lock"
    )
    bundle_root = _sha256(bundle_bytes)
    capability = _mint_strict_loader_capability(
        rows=rows,
        tap_archive_sha256=actual_archive,
        tap_receipt_sha256=actual_receipt,
        predecessor_lock_bundle_root_sha256=bundle_root,
        request=request,
        code_sha256=actual_code,
    )
    try:
        internal = _execute_rows_internal_test(
            rows,
            registered_classes=request.registered_classes,
            predecessor_locks=locks,
            rcmr_method_lock_sha256=method_lock_sha,
            tap_archive_sha256=actual_archive,
            tap_receipt_sha256=actual_receipt,
            predecessor_lock_bundle_root_sha256=bundle_root,
            code_sha256=actual_code,
            request=request,
        )
        return _finalize_strict_production_result(
            internal,
            strict_loader_capability=capability,
            rows=rows,
            tap_archive_sha256=actual_archive,
            tap_receipt_sha256=actual_receipt,
            predecessor_lock_bundle_root_sha256=bundle_root,
            code_sha256=actual_code,
            request=request,
        )
    except Exception:
        _ISSUED_STRICT_LOADER_CAPABILITIES.pop(id(capability), None)
        raise


def _execute_rows_internal_test(
    rows: D106Phase1TapRows,
    *,
    registered_classes: tuple[str, ...],
    predecessor_locks: tuple[Phase1ZIDStudentTLock, ...],
    rcmr_method_lock_sha256: str,
    tap_archive_sha256: str,
    tap_receipt_sha256: str,
    predecessor_lock_bundle_root_sha256: str,
    code_sha256: Mapping[str, str],
    request: D106RCMRG0ProductionRequest,
) -> bytes:
    """Execute rows mechanics only; this can never mint a production artifact."""

    archive_sha = _require_sha256(tap_archive_sha256, "production tap archive")
    receipt_sha = _require_sha256(tap_receipt_sha256, "production tap receipt")
    bundle_root = _require_sha256(
        predecessor_lock_bundle_root_sha256, "production predecessor lock bundle"
    )
    registry = _canonical_registry(registered_classes)
    locks = _validate_synthetic_locks(predecessor_locks, tap_receipt_sha256=receipt_sha)
    method_lock_sha = _require_sha256(rcmr_method_lock_sha256, "production RCMR method lock")
    snapshot = _snapshot_from_rows(
        rows,
        tap_receipt_sha256=receipt_sha,
        snapshot_schema=INTERNAL_ROWS_SCHEMA,
    )
    plan = _build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    common_query_root = _sha256(list(query_order))
    registry_root = _sha256(list(registry))
    per_k: list[dict[str, Any]] = []
    for active_k, predecessor_lock in zip(K_VALUES, locks, strict=True):
        compact_folds: list[dict[str, Any]] = []
        all_query: list[str] = []
        for fold in plan:
            query_ids, _candidate, _baseline, full_receipt = _execute_fold(
                snapshot,
                fold,
                active_k=active_k,
                predecessor_lock=predecessor_lock,
                rcmr_method_lock_sha256=method_lock_sha,
                registry=registry,
                common_query_order_root_sha256=common_query_root,
            )
            all_query.extend(query_ids)
            compact = {
                "schema": PRODUCTION_FOLD_EXECUTION_SCHEMA,
                "K": active_k,
                "fold_index": fold.index,
                **{
                    name: _require_sha256(full_receipt.get(name), f"production fold {name}")
                    for name in _PRODUCTION_FOLD_ROOT_FIELDS
                },
                "argmax_changed_bitmap": full_receipt["argmax_changed_bitmap"],
                "argmax_changed_bitmap_root_sha256": _require_sha256(
                    full_receipt.get("argmax_changed_bitmap_root_sha256"),
                    "production fold bitmap root",
                ),
                "argmax_changed_count": full_receipt["argmax_changed_count"],
            }
            bitmap = compact["argmax_changed_bitmap"]
            if (
                type(bitmap) is not str
                or not 1 <= len(bitmap) <= MAX_QUERY_ROWS_PER_FOLD
                or any(bit not in "01" for bit in bitmap)
                or compact["argmax_changed_count"] != bitmap.count("1")
            ):
                raise D106RCMRG0Error("production compact fold bitmap/count drift")
            if compact["argmax_changed_bitmap_root_sha256"] != _sha256(
                {
                    "encoding": "ascii01_query_order",
                    "query_count": len(bitmap),
                    "bits": bitmap,
                }
            ):
                raise D106RCMRG0Error("production compact fold bitmap root drift")
            compact["execution_receipt_sha256"] = _sha256(compact)
            compact_folds.append(compact)
        if tuple(all_query) != query_order:
            raise D106RCMRG0Error("production query order differs across K")
        changed = sum(int(item["argmax_changed_count"]) for item in compact_folds)
        item = {
            "schema": PRODUCTION_PER_K_SCHEMA,
            "K": active_k,
            "fold_count": EXPECTED_FOLDS,
            "query_count": EXPECTED_ROWS,
            "fold_execution_receipts": compact_folds,
            "fold_execution_receipts_root_sha256": _sha256(
                [fold["execution_receipt_sha256"] for fold in compact_folds]
            ),
            "fold_changed_bitmap_roots_root_sha256": _sha256(
                [fold["argmax_changed_bitmap_root_sha256"] for fold in compact_folds]
            ),
            "argmax_changed_count": changed,
        }
        item["per_k_receipt_sha256"] = _sha256(item)
        per_k.append(item)
    changed_by_k = {str(item["K"]): int(item["argmax_changed_count"]) for item in per_k}
    zero_changed = [active_k for active_k in K_VALUES if changed_by_k[str(active_k)] == 0]
    functional_status = (
        "G0_EVERY_K_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
        if not zero_changed
        else "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    )
    lock_set_root = _sha256(
        [
            {"K": lock.active_k, "lock_digest_sha256": lock.lock_digest}
            for lock in locks
        ]
    )
    manifest = {
        "schema": PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "algorithm_execution_scope": ALGORITHM_SCOPE,
        "K_values": list(K_VALUES),
        "fold_count": EXPECTED_FOLDS,
        "query_count_per_k": EXPECTED_ROWS,
        "common_query_order_root_sha256": common_query_root,
        "tap_archive_sha256": archive_sha,
        "tap_receipt_sha256": receipt_sha,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "registry_root_sha256": registry_root,
        "predecessor_lock_bundle_root_sha256": bundle_root,
        "predecessor_lock_set_root_sha256": lock_set_root,
        "per_k": per_k,
        "per_k_execution_receipts_root_sha256": _sha256(
            [item["per_k_receipt_sha256"] for item in per_k]
        ),
        "canonical_execution_root_sha256": _sha256(
            [item["per_k_receipt_sha256"] for item in per_k]
        ),
        "argmax_changed_count_by_k": changed_by_k,
        "argmax_changed_count": sum(changed_by_k.values()),
        "zero_changed_k_values": zero_changed,
        "functional_gate_pass": not zero_changed,
        "functional_gate_status": functional_status,
        "authority_flags": dict(_PRODUCTION_AUTHORITY_FLAGS),
    }
    manifest["execution_manifest_receipt_sha256"] = _sha256(manifest)
    manifest_bytes = _canonical_bytes(manifest)
    payload = {
        "schema": INTERNAL_ROWS_SCHEMA,
        "status": INTERNAL_ROWS_STATUS,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "algorithm_execution_scope": ALGORITHM_SCOPE,
        "real_g0_executed": False,
        "runner_authority": False,
        "deployable": False,
        "formal_performance_claim": False,
        "authority_flags": dict(_PRODUCTION_AUTHORITY_FLAGS),
        "held_label_audit_status": HELD_LABEL_AUDIT_STATUS,
        "same_process_held_label_capability_absence_claimed": False,
        "rows_or_labels_returned": False,
        "expected_release_commit": request.expected_release_commit,
        "request_receipt_sha256": request.request_receipt_sha256,
        "code_sha256": dict(code_sha256),
        "tap_archive_sha256": archive_sha,
        "tap_receipt_sha256": receipt_sha,
        "predecessor_lock_bundle_root_sha256": bundle_root,
        "execution_manifest": json.loads(manifest_bytes.decode("utf-8")),
        "execution_manifest_root_sha256": _sha256(manifest_bytes),
        "internal_test_only": True,
    }
    result = _seal_result(payload)
    _revalidate_snapshot(snapshot)
    return result


def _finalize_strict_production_result(
    internal_result: bytes,
    *,
    strict_loader_capability: _StrictLoaderCapability,
    rows: D106Phase1TapRows,
    tap_archive_sha256: str,
    tap_receipt_sha256: str,
    predecessor_lock_bundle_root_sha256: str,
    code_sha256: Mapping[str, str],
    request: D106RCMRG0ProductionRequest,
) -> bytes:
    """Mint public production bytes only after the actual-file evidence chain.

    A direct rows executor yields ``INTERNAL_ROWS_SCHEMA`` and is permanently
    rejected by the public verifier.  This finalizer consumes the one-use
    misuse guard, then revalidates the request and runtime closure immediately
    before it changes the schema/status to public production values.
    """

    actual_code = _verify_production_code_closure(request)
    if dict(code_sha256) != actual_code:
        raise D106RCMRG0Error("final production code-closure binding drift")
    evidence = _consume_strict_loader_capability(
        strict_loader_capability,
        rows=rows,
        tap_archive_sha256=tap_archive_sha256,
        tap_receipt_sha256=tap_receipt_sha256,
        predecessor_lock_bundle_root_sha256=predecessor_lock_bundle_root_sha256,
        request=request,
        code_sha256=actual_code,
    )
    if type(internal_result) is not bytes:
        raise D106RCMRG0Error("internal rows result must be immutable bytes")
    try:
        internal = json.loads(internal_result.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0Error("internal rows result is not canonical JSON") from error
    if type(internal) is not dict or internal_result != _canonical_bytes(internal):
        raise D106RCMRG0Error("internal rows result canonical-byte closure drift")
    internal_receipt = _require_sha256(
        internal.pop("result_receipt_sha256", None), "internal rows receipt"
    )
    if (
        internal.get("schema") != INTERNAL_ROWS_SCHEMA
        or internal.get("status") != INTERNAL_ROWS_STATUS
        or internal.get("internal_test_only") is not True
        or internal.get("real_g0_executed") is not False
        or internal.get("tap_archive_sha256") != evidence["tap_archive_sha256"]
        or internal.get("tap_receipt_sha256") != evidence["tap_receipt_sha256"]
        or internal.get("predecessor_lock_bundle_root_sha256")
        != evidence["predecessor_lock_bundle_root_sha256"]
        or internal.get("request_receipt_sha256") != evidence["request_receipt_sha256"]
        or internal.get("code_sha256") != actual_code
        or _sha256(internal) != internal_receipt
    ):
        raise D106RCMRG0Error("internal rows evidence chain drift")
    manifest = internal.get("execution_manifest")
    if type(manifest) is not dict:
        raise D106RCMRG0Error("internal rows manifest type drift")
    internal_manifest_bytes = _canonical_bytes(manifest)
    if internal.get("execution_manifest_root_sha256") != _sha256(internal_manifest_bytes):
        raise D106RCMRG0Error("internal rows manifest root drift")
    public_manifest = dict(manifest)
    public_manifest["strict_loader_evidence_receipt_sha256"] = evidence[
        "strict_loader_evidence_receipt_sha256"
    ]
    public_manifest.pop("execution_manifest_receipt_sha256", None)
    public_manifest["execution_manifest_receipt_sha256"] = _sha256(public_manifest)
    public_manifest_bytes = _canonical_bytes(public_manifest)
    payload = dict(internal)
    payload.pop("internal_test_only")
    payload["schema"] = PRODUCTION_SCHEMA
    payload["status"] = PRODUCTION_EXECUTED_STATUS
    payload["real_g0_executed"] = True
    payload["strict_loader_evidence"] = evidence
    payload["execution_manifest"] = json.loads(public_manifest_bytes.decode("utf-8"))
    payload["execution_manifest_root_sha256"] = _sha256(public_manifest_bytes)
    return _seal_result(payload)


def _freeze_bytes_array(value: np.ndarray, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str) -> np.ndarray:
    candidate = np.asarray(value)
    if candidate.dtype != dtype or candidate.shape != shape:
        raise D106RCMRG0Error(f"tap {name} layout drift")
    contiguous = np.ascontiguousarray(candidate)
    if dtype.kind == "f" and not np.isfinite(contiguous).all():
        raise D106RCMRG0Error(f"tap {name} contains non-finite values")
    backing = bytes(contiguous.tobytes(order="C"))
    result = np.frombuffer(backing, dtype=dtype).reshape(shape)
    if result.flags.writeable or not isinstance(result.base, np.ndarray):
        raise D106RCMRG0Error(f"tap {name} immutable bytes backing drift")
    return result


@dataclass(frozen=True, slots=True)
class _TapSnapshot:
    pre_relu: np.ndarray = field(repr=False)
    z_dom: np.ndarray = field(repr=False)
    tx_labels: np.ndarray = field(repr=False)
    receiver_ids: np.ndarray = field(repr=False)
    day_ids: np.ndarray = field(repr=False)
    physical_ids: np.ndarray = field(repr=False)
    scenario_names: np.ndarray = field(repr=False)
    observation_ids: np.ndarray = field(repr=False)
    z_id: np.ndarray = field(repr=False)
    receipt_bytes: bytes = field(repr=False)
    array_roots: tuple[tuple[str, str], ...]
    snapshot_schema: str
    tap_receipt_sha256: str
    tap_snapshot_root_sha256: str


_SNAPSHOT_ARRAY_NAMES = (
    "pre_relu",
    "z_dom",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
    "z_id",
)


def _snapshot_from_rows(
    rows: Any,
    *,
    tap_receipt_sha256: str,
    snapshot_schema: str = SYNTHETIC_SCHEMA,
) -> _TapSnapshot:
    if type(rows) is not D106Phase1TapRows:
        raise D106RCMRG0Error("synthetic G0 requires the exact tap row type")
    receipt = rows.receipt
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != TAP_RECEIPT_SCHEMA
        or receipt.get("protocol_schema") != PROTOCOL_SCHEMA
        or receipt.get("row_count") != EXPECTED_ROWS
        or receipt.get("exact_inner_join") is not True
        or receipt.get("same_received_iq_for_zid_zdom") is not True
        or receipt.get("z_id_storage_policy") != "derive_relu_pre_relu"
        or receipt.get("feature_stage_source_pool_access") is not False
        or receipt.get("clean_iq_access") is not False
        or receipt.get("target_access") is not False
        or receipt.get("formal_query_access") is not False
    ):
        raise D106RCMRG0Error("tap receipt semantic drift")
    frozen: dict[str, np.ndarray] = {}
    frozen["pre_relu"] = _freeze_bytes_array(
        rows.pre_relu, dtype=np.dtype(np.float32), shape=(EXPECTED_ROWS, Z_DIM), name="pre_relu"
    )
    frozen["z_dom"] = _freeze_bytes_array(
        rows.z_dom, dtype=np.dtype(np.float32), shape=(EXPECTED_ROWS, Z_DIM), name="z_dom"
    )
    frozen["z_id"] = _freeze_bytes_array(
        rows.z_id, dtype=np.dtype(np.float32), shape=(EXPECTED_ROWS, Z_DIM), name="z_id"
    )
    for name in _SNAPSHOT_ARRAY_NAMES[2:8]:
        source = np.asarray(getattr(rows, name))
        if source.dtype.kind not in {"U", "S"}:
            raise D106RCMRG0Error(f"tap {name} dtype drift")
        frozen[name] = _freeze_bytes_array(
            source, dtype=source.dtype, shape=(EXPECTED_ROWS,), name=name
        )
    if not np.array_equal(frozen["z_id"], np.maximum(frozen["pre_relu"], np.float32(0.0))):
        raise D106RCMRG0Error("tap same-IQ ReLU view drift")
    physical_ids = _typed_tokens(frozen["physical_ids"], "physical IDs", count=EXPECTED_ROWS)
    if len(set(physical_ids)) != EXPECTED_ROWS:
        raise D106RCMRG0Error("tap physical IDs must be unique")
    if type(snapshot_schema) is not str or not snapshot_schema:
        raise D106RCMRG0Error("tap snapshot schema must be an exact non-empty string")
    roots = tuple((name, _array_root(frozen[name])) for name in _SNAPSHOT_ARRAY_NAMES)
    tap_receipt = _require_sha256(tap_receipt_sha256, "tap receipt")
    snapshot_root = _sha256(
        {
            "schema": snapshot_schema + ".tap_snapshot.v1",
            "tap_receipt_sha256": tap_receipt,
            "array_roots": dict(roots),
        }
    )
    snapshot = _TapSnapshot(
        **frozen,
        receipt_bytes=_canonical_bytes(receipt),
        array_roots=roots,
        snapshot_schema=snapshot_schema,
        tap_receipt_sha256=tap_receipt,
        tap_snapshot_root_sha256=snapshot_root,
    )
    _revalidate_snapshot(snapshot)
    return snapshot


def _revalidate_snapshot(snapshot: Any) -> _TapSnapshot:
    if type(snapshot) is not _TapSnapshot:
        raise D106RCMRG0Error("tap snapshot type drift")
    expected = dict(snapshot.array_roots)
    if type(snapshot.snapshot_schema) is not str or not snapshot.snapshot_schema:
        raise D106RCMRG0Error("tap snapshot schema drift")
    if tuple(expected) != _SNAPSHOT_ARRAY_NAMES:
        raise D106RCMRG0Error("tap snapshot array-root closure drift")
    for name in _SNAPSHOT_ARRAY_NAMES:
        value = getattr(snapshot, name)
        if not isinstance(value, np.ndarray) or value.flags.writeable:
            raise D106RCMRG0Error("tap snapshot immutable array drift")
        if _array_root(value) != expected[name]:
            raise D106RCMRG0Error(f"tap snapshot {name} root drift")
    if not np.array_equal(snapshot.z_id, np.maximum(snapshot.pre_relu, np.float32(0.0))):
        raise D106RCMRG0Error("tap snapshot ReLU relation drift")
    if _sha256(snapshot.receipt_bytes) != snapshot.tap_receipt_sha256:
        # Synthetic tests bind the exact canonical receipt bytes as their tap
        # receipt.  The real strict loader would bind the on-disk receipt SHA.
        raise D106RCMRG0Error("tap snapshot receipt root drift")
    expected_snapshot_root = _sha256(
        {
            "schema": snapshot.snapshot_schema + ".tap_snapshot.v1",
            "tap_receipt_sha256": snapshot.tap_receipt_sha256,
            "array_roots": expected,
        }
    )
    if snapshot.tap_snapshot_root_sha256 != expected_snapshot_root:
        raise D106RCMRG0Error("tap snapshot aggregate root drift")
    return snapshot


@dataclass(frozen=True, slots=True)
class _Fold:
    index: int
    fold_id: str
    receiver_id: str
    day_id: str
    query_ids: tuple[str, ...]
    query_root_sha256: str


def _build_fold_plan(snapshot: _TapSnapshot) -> tuple[_Fold, ...]:
    rows = _revalidate_snapshot(snapshot)
    receivers = _typed_tokens(rows.receiver_ids, "receiver IDs", count=EXPECTED_ROWS)
    days = _typed_tokens(rows.day_ids, "day IDs", count=EXPECTED_ROWS)
    physical = _typed_tokens(rows.physical_ids, "physical IDs", count=EXPECTED_ROWS)
    receiver_registry = tuple(sorted(set(receivers), key=lambda value: value.encode("utf-8")))
    day_registry = tuple(sorted(set(days), key=lambda value: value.encode("utf-8")))
    if len(receiver_registry) != EXPECTED_RECEIVERS or len(day_registry) != EXPECTED_DAYS:
        raise D106RCMRG0Error("tap receiver/day cardinality is not 7x4")
    folds: list[_Fold] = []
    for index, (receiver, day) in enumerate(
        (item for r in receiver_registry for item in ((r, d) for d in day_registry))
    ):
        query_ids = tuple(
            sorted(
                (physical[i] for i in range(EXPECTED_ROWS) if receivers[i] == receiver and days[i] == day),
                key=lambda value: value.encode("utf-8"),
            )
        )
        if not 1 <= len(query_ids) <= MAX_QUERY_ROWS_PER_FOLD:
            raise D106RCMRG0Error("fold query count must be within 1..24")
        query_root = _sha256(list(query_ids))
        identity = {
            "schema": FOLD_EXECUTION_SCHEMA + ".identity.v1",
            "tap_receipt_sha256": rows.tap_receipt_sha256,
            "receiver_id": receiver,
            "day_id": day,
            "query_root_sha256": query_root,
        }
        folds.append(
            _Fold(
                index=index,
                fold_id=f"d106-g0-{index:02d}-{_sha256(identity)[:16]}",
                receiver_id=receiver,
                day_id=day,
                query_ids=query_ids,
                query_root_sha256=query_root,
            )
        )
    all_ids = tuple(query for fold in folds for query in fold.query_ids)
    if len(folds) != EXPECTED_FOLDS or len(all_ids) != EXPECTED_ROWS or len(set(all_ids)) != EXPECTED_ROWS:
        raise D106RCMRG0Error("fold plan does not close all 588 rows")
    return tuple(folds)


def _validate_synthetic_locks(
    locks: Sequence[Phase1ZIDStudentTLock], *, tap_receipt_sha256: str
) -> tuple[Phase1ZIDStudentTLock, ...]:
    if not isinstance(locks, (tuple, list)) or len(locks) != len(K_VALUES):
        raise D106RCMRG0Error("synthetic test requires K1/K5/K10 locks")
    result = tuple(locks)
    for active_k, lock in zip(K_VALUES, result, strict=True):
        if type(lock) is not Phase1ZIDStudentTLock or lock.active_k != active_k:
            raise D106RCMRG0Error("synthetic predecessor lock/K binding drift")
        for name, expected in PREDECESSOR_NUMERIC_LOCK.items():
            observed = getattr(lock, name)
            if type(expected) is int:
                valid = type(observed) is int and observed == expected
            else:
                valid = math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=0.0)
            if not valid:
                raise D106RCMRG0Error("synthetic predecessor numeric lock drift")
        if lock.quantization_margin_audit_sha256 != tap_receipt_sha256:
            raise D106RCMRG0Error("synthetic lock/tap receipt binding drift")
    if tuple(lock.active_k for lock in result) != K_VALUES or len({lock.lock_digest for lock in result}) != 3:
        raise D106RCMRG0Error("synthetic K lock set is missing, duplicated, or reordered")
    return result


def _unique_argmax(logits: np.ndarray, classes: tuple[str, ...]) -> tuple[str, ...]:
    values = np.asarray(logits)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != len(classes) or not np.isfinite(values).all():
        raise D106RCMRG0Error("predecessor logits layout drift")
    result: list[str] = []
    for row in values:
        winners = np.flatnonzero(row == np.float32(np.max(row)))
        if len(winners) != 1:
            raise D106RCMRG0Error("predecessor cross-class tie is fail closed")
        result.append(classes[int(winners[0])])
    return tuple(result)


def _paired_view_receipt(
    support_ids: tuple[str, ...], support_plus: np.ndarray, support_signed: np.ndarray
) -> str:
    return _sha256(
        {
            "schema": FOLD_EXECUTION_SCHEMA + ".paired_views.v1",
            "support_ids_root_sha256": _sha256(list(support_ids)),
            "support_plus_root_sha256": _array_root(support_plus),
            "support_signed_root_sha256": _array_root(support_signed),
        }
    )


_IDENTITY_DA_RECEIPT_SHA256 = _sha256(
    {"schema": FOLD_EXECUTION_SCHEMA + ".identity_da.v1", "mapping": "identity", "query_updates": 0}
)


@dataclass(frozen=True, slots=True)
class _NonFormalRCMRState:
    """Synthetic-only mechanics; it has no Phase2 lifecycle or formal token."""

    registry: tuple[str, ...]
    active_k: int
    codes_plus: np.ndarray = field(repr=False)
    codes_signed: np.ndarray = field(repr=False)
    scales_plus: np.ndarray = field(repr=False)
    scales_signed: np.ndarray = field(repr=False)
    reliabilities: np.ndarray = field(repr=False)
    class_indices: np.ndarray = field(repr=False)
    support_root_sha256: str
    paired_view_receipt_sha256: str
    rcmr_method_lock_sha256: str
    state_receipt_sha256: str
    lifecycle_status: str = ALGORITHM_SCOPE

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SYNTHETIC_SCHEMA + ".nonformal_rcmr_state.v1",
            "lifecycle_status": self.lifecycle_status,
            "formal_authority": False,
            "phase2_validated_once": False,
            "registry": list(self.registry),
            "active_k": self.active_k,
            "support_root_sha256": self.support_root_sha256,
            "paired_view_receipt_sha256": self.paired_view_receipt_sha256,
            "rcmr_method_lock_sha256": self.rcmr_method_lock_sha256,
            "arrays": {
                "codes_plus": _array_root(self.codes_plus),
                "codes_signed": _array_root(self.codes_signed),
                "scales_plus": _array_root(self.scales_plus),
                "scales_signed": _array_root(self.scales_signed),
                "reliabilities": _array_root(self.reliabilities),
                "class_indices": _array_root(self.class_indices),
            },
        }


@dataclass(frozen=True, slots=True)
class _NonFormalRCMRContext:
    decoded_plus: np.ndarray = field(repr=False)
    decoded_signed: np.ndarray = field(repr=False)
    profiles_plus: np.ndarray = field(repr=False)
    profiles_signed: np.ndarray = field(repr=False)
    state_receipt_sha256: str
    lifecycle_status: str = ALGORITHM_SCOPE


def _freeze_dynamic(value: np.ndarray, dtype: np.dtype[Any], name: str) -> np.ndarray:
    candidate = np.ascontiguousarray(value, dtype=dtype)
    return _freeze_bytes_array(
        candidate, dtype=dtype, shape=tuple(candidate.shape), name=name
    )


def _nonformal_state_from_support(
    support_plus: np.ndarray,
    support_signed: np.ndarray,
    support_labels: tuple[str, ...],
    support_ids: tuple[str, ...],
    registry: tuple[str, ...],
    *,
    active_k: int,
    support_root_sha256: str,
    paired_view_receipt_sha256: str,
    rcmr_method_lock_sha256: str,
) -> _NonFormalRCMRState:
    """Build synthetic mechanics without constructing any formal RCMR type."""

    plus = _rcmr_module._finite_l2_normalized_rows(support_plus, "nonformal support_plus")
    signed = _rcmr_module._finite_l2_normalized_rows(
        support_signed, "nonformal support_signed"
    )
    if plus.shape != signed.shape or len(plus) != len(registry) * active_k:
        raise D106RCMRG0Error("nonformal RCMR support C/K/N closure drift")
    class_map = {name: index for index, name in enumerate(registry)}
    try:
        unordered_class = np.asarray(
            [class_map[label] for label in support_labels], dtype=np.uint8
        )
    except KeyError as error:
        raise D106RCMRG0Error("nonformal RCMR support label drift") from error
    counts = np.bincount(unordered_class.astype(np.int64), minlength=len(registry))
    if not np.all(counts == active_k):
        raise D106RCMRG0Error("nonformal RCMR support is not balanced K-shot")
    order = np.asarray(
        sorted(range(len(support_ids)), key=lambda index: support_ids[index]),
        dtype=np.int64,
    )
    plus = np.ascontiguousarray(plus[order], dtype=np.float64)
    signed = np.ascontiguousarray(signed[order], dtype=np.float64)
    class_indices = np.ascontiguousarray(unordered_class[order], dtype=np.uint8)
    codes_plus, scales_plus = _rcmr_module._quantize_rows(plus, "nonformal support_plus")
    codes_signed, scales_signed = _rcmr_module._quantize_rows(
        signed, "nonformal support_signed"
    )
    decoded_plus = _rcmr_module._decode_rows(
        codes_plus, scales_plus, "nonformal support_plus"
    )
    decoded_signed = _rcmr_module._decode_rows(
        codes_signed, scales_signed, "nonformal support_signed"
    )
    reliabilities = _rcmr_module._support_reliability(decoded_plus, decoded_signed)
    fields = {
        "registry": registry,
        "active_k": active_k,
        "codes_plus": _freeze_dynamic(codes_plus, np.dtype(np.int8), "nonformal codes_plus"),
        "codes_signed": _freeze_dynamic(codes_signed, np.dtype(np.int8), "nonformal codes_signed"),
        "scales_plus": _freeze_dynamic(scales_plus, np.dtype("<f2"), "nonformal scales_plus"),
        "scales_signed": _freeze_dynamic(scales_signed, np.dtype("<f2"), "nonformal scales_signed"),
        "reliabilities": _freeze_dynamic(reliabilities, np.dtype("<f2"), "nonformal reliabilities"),
        "class_indices": _freeze_dynamic(class_indices, np.dtype(np.uint8), "nonformal class indices"),
        "support_root_sha256": _require_sha256(support_root_sha256, "support root"),
        "paired_view_receipt_sha256": _require_sha256(
            paired_view_receipt_sha256, "paired-view receipt"
        ),
        "rcmr_method_lock_sha256": _require_sha256(
            rcmr_method_lock_sha256, "RCMR method lock"
        ),
    }
    provisional = _NonFormalRCMRState(**fields, state_receipt_sha256="0" * 64)
    state = replace(provisional, state_receipt_sha256=_sha256(provisional.payload()))
    if state.lifecycle_status != ALGORITHM_SCOPE or state.payload()["formal_authority"] is not False:
        raise D106RCMRG0Error("nonformal RCMR lifecycle drift")
    return state


def _serialize_nonformal_state(state: _NonFormalRCMRState) -> bytes:
    if type(state) is not _NonFormalRCMRState or state.lifecycle_status != ALGORITHM_SCOPE:
        raise D106RCMRG0Error("nonformal wire requires the exact mechanical state")
    header = _canonical_bytes(
        {
            "schema": SYNTHETIC_SCHEMA + ".nonformal_rcmr_wire.v1",
            "state": state.payload(),
            "state_receipt_sha256": state.state_receipt_sha256,
        }
    )
    body = b"".join(
        value.tobytes(order="C")
        for value in (
            state.codes_plus,
            state.codes_signed,
            state.scales_plus,
            state.scales_signed,
            state.reliabilities,
            state.class_indices,
        )
    )
    return struct.pack(">I", len(header)) + header + body


def _deserialize_nonformal_state(
    payload: bytes, *, expected_sha256: str
) -> _NonFormalRCMRState:
    if type(payload) is not bytes or _sha256(payload) != _require_sha256(
        expected_sha256, "nonformal wire"
    ):
        raise D106RCMRG0Error("nonformal RCMR wire SHA256 mismatch")
    if len(payload) < 5:
        raise D106RCMRG0Error("nonformal RCMR wire truncated")
    header_size = struct.unpack(">I", payload[:4])[0]
    header_end = 4 + header_size
    try:
        header = json.loads(payload[4:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106RCMRG0Error("nonformal RCMR wire header drift") from error
    if (
        type(header) is not dict
        or payload[4:header_end] != _canonical_bytes(header)
        or header.get("schema") != SYNTHETIC_SCHEMA + ".nonformal_rcmr_wire.v1"
        or type(header.get("state")) is not dict
    ):
        raise D106RCMRG0Error("nonformal RCMR wire canonical closure drift")
    document = header["state"]
    if (
        document.get("schema") != SYNTHETIC_SCHEMA + ".nonformal_rcmr_state.v1"
        or document.get("lifecycle_status") != ALGORITHM_SCOPE
        or document.get("formal_authority") is not False
        or document.get("phase2_validated_once") is not False
    ):
        raise D106RCMRG0Error("nonformal RCMR wire lifecycle drift")
    registry = _canonical_registry(document.get("registry", ()))
    active_k = document.get("active_k")
    if type(active_k) is not int or active_k not in K_VALUES:
        raise D106RCMRG0Error("nonformal RCMR wire K drift")
    count = len(registry) * active_k
    layouts = (
        ("codes_plus", np.dtype(np.int8), (count, Z_DIM)),
        ("codes_signed", np.dtype(np.int8), (count, Z_DIM)),
        ("scales_plus", np.dtype("<f2"), (count,)),
        ("scales_signed", np.dtype("<f2"), (count,)),
        ("reliabilities", np.dtype("<f2"), (count,)),
        ("class_indices", np.dtype(np.uint8), (count,)),
    )
    position = header_end
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape in layouts:
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if position + size > len(payload):
            raise D106RCMRG0Error("nonformal RCMR wire numeric payload truncated")
        raw = np.frombuffer(payload[position : position + size], dtype=dtype).reshape(shape)
        arrays[name] = _freeze_dynamic(raw, dtype, f"wire {name}")
        position += size
    if position != len(payload):
        raise D106RCMRG0Error("nonformal RCMR wire trailing bytes")
    state = _NonFormalRCMRState(
        registry=registry,
        active_k=active_k,
        **arrays,
        support_root_sha256=_require_sha256(document.get("support_root_sha256"), "wire support root"),
        paired_view_receipt_sha256=_require_sha256(
            document.get("paired_view_receipt_sha256"), "wire paired-view receipt"
        ),
        rcmr_method_lock_sha256=_require_sha256(
            document.get("rcmr_method_lock_sha256"), "wire RCMR method lock"
        ),
        state_receipt_sha256=_require_sha256(
            header.get("state_receipt_sha256"), "wire state receipt"
        ),
    )
    if state.payload() != document or _sha256(document) != state.state_receipt_sha256:
        raise D106RCMRG0Error("nonformal RCMR wire state receipt drift")
    return state


def _prepare_nonformal_context(state: _NonFormalRCMRState) -> _NonFormalRCMRContext:
    decoded_plus = _rcmr_module._decode_rows(
        state.codes_plus, state.scales_plus, "nonformal context plus"
    )
    decoded_signed = _rcmr_module._decode_rows(
        state.codes_signed, state.scales_signed, "nonformal context signed"
    )
    profiles_plus = _rcmr_module._profiles(
        _rcmr_module._pairwise_distance_matrix(decoded_plus)
    )
    profiles_signed = _rcmr_module._profiles(
        _rcmr_module._pairwise_distance_matrix(decoded_signed)
    )
    return _NonFormalRCMRContext(
        decoded_plus=_freeze_dynamic(decoded_plus, np.dtype(np.float64), "nonformal decoded plus"),
        decoded_signed=_freeze_dynamic(decoded_signed, np.dtype(np.float64), "nonformal decoded signed"),
        profiles_plus=_freeze_dynamic(profiles_plus, np.dtype(np.float64), "nonformal profiles plus"),
        profiles_signed=_freeze_dynamic(profiles_signed, np.dtype(np.float64), "nonformal profiles signed"),
        state_receipt_sha256=state.state_receipt_sha256,
    )


def _score_nonformal_query(
    state: _NonFormalRCMRState,
    context: _NonFormalRCMRContext,
    query_plus: np.ndarray,
    query_signed: np.ndarray,
) -> str:
    if (
        type(state) is not _NonFormalRCMRState
        or type(context) is not _NonFormalRCMRContext
        or context.state_receipt_sha256 != state.state_receipt_sha256
        or state.lifecycle_status != ALGORITHM_SCOPE
        or context.lifecycle_status != ALGORITHM_SCOPE
    ):
        raise D106RCMRG0Error("nonformal scorer lifecycle binding drift")
    plus_query = _rcmr_module._finite_l2_normalized_vector(query_plus, "nonformal query_plus")
    signed_query = _rcmr_module._finite_l2_normalized_vector(
        query_signed, "nonformal query_signed"
    )
    count = len(state.class_indices)
    distances_plus = np.asarray(
        [_rcmr_module._dot_distance(plus_query, context.decoded_plus[slot]) for slot in range(count)],
        dtype=np.float64,
    )
    distances_signed = np.asarray(
        [_rcmr_module._dot_distance(signed_query, context.decoded_signed[slot]) for slot in range(count)],
        dtype=np.float64,
    )
    alpha_plus = _rcmr_module._midranks(distances_plus)
    alpha_signed = _rcmr_module._midranks(distances_signed)
    beta_plus = np.asarray(
        [
            _rcmr_module._midrank_from_profile(
                context.profiles_plus[slot], float(distances_plus[slot])
            )
            for slot in range(count)
        ],
        dtype=np.float64,
    )
    beta_signed = np.asarray(
        [
            _rcmr_module._midrank_from_profile(
                context.profiles_signed[slot], float(distances_signed[slot])
            )
            for slot in range(count)
        ],
        dtype=np.float64,
    )
    query_reliability = math.exp(-float(np.mean(np.abs(alpha_plus - alpha_signed))))
    weights = query_reliability * state.reliabilities.astype(np.float64, copy=False)
    evidence = (
        (1.0 - alpha_plus) * (1.0 - beta_plus)
        + weights * (1.0 - alpha_signed) * (1.0 - beta_signed)
    ) / (1.0 + weights)
    scores = np.zeros(len(state.registry), dtype=np.float64)
    for slot in range(count):
        scores[int(state.class_indices[slot])] += float(evidence[slot])
    scores /= float(state.active_k)
    maximum = max(float(score) for score in scores)
    winners = [
        index
        for index, score in enumerate(scores)
        if _rcmr_module._same_binary64(float(score), maximum)
    ]
    if len(winners) != 1:
        raise D106RCMRG0Error("nonformal RCMR exact cross-class tie")
    return state.registry[winners[0]]


def _execute_fold(
    snapshot: _TapSnapshot,
    fold: _Fold,
    *,
    active_k: int,
    predecessor_lock: Phase1ZIDStudentTLock,
    rcmr_method_lock_sha256: str,
    registry: tuple[str, ...],
    common_query_order_root_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    rows = _revalidate_snapshot(snapshot)
    physical = _typed_tokens(rows.physical_ids, "physical IDs", count=EXPECTED_ROWS)
    receiver_ids = _typed_tokens(rows.receiver_ids, "receiver IDs", count=EXPECTED_ROWS)
    day_ids = _typed_tokens(rows.day_ids, "day IDs", count=EXPECTED_ROWS)
    index_by_id = {value: index for index, value in enumerate(physical)}
    query_indices = np.asarray([index_by_id[value] for value in fold.query_ids], dtype=np.int64)
    if len(query_indices) > MAX_QUERY_ROWS_PER_FOLD:
        raise D106RCMRG0Error("fold query hard bound exceeded")
    query_mask = np.zeros(EXPECTED_ROWS, dtype=bool)
    query_mask[query_indices] = True
    if np.any(
        (np.asarray(receiver_ids) == fold.receiver_id)
        & (np.asarray(day_ids) == fold.day_id)
        & ~query_mask
    ):
        raise D106RCMRG0Error("held cell is not completely assigned to query")
    support_pool = np.flatnonzero(~query_mask).astype(np.int64)
    if np.any(
        (np.asarray(receiver_ids)[support_pool] == fold.receiver_id)
        & (np.asarray(day_ids)[support_pool] == fold.day_id)
    ):
        raise D106RCMRG0Error("held-cell label path reached support")
    # This is the only fold-semantic label read.  The disjointness proof above
    # executes immediately before it.  We claim audited path behavior, not
    # same-process label capability isolation.
    pool_labels = _typed_tokens(rows.tx_labels[support_pool], "support-pool labels", count=len(support_pool))
    by_class: dict[str, list[int]] = {class_id: [] for class_id in registry}
    for index, label in zip(support_pool.tolist(), pool_labels, strict=True):
        if label not in by_class:
            raise D106RCMRG0Error("support label outside registry")
        by_class[label].append(index)
    selected: list[int] = []
    for class_id in registry:
        ordered = sorted(by_class[class_id], key=lambda index: physical[index].encode("utf-8"))
        if len(ordered) < active_k:
            raise D106RCMRG0Error("fold lacks requested K")
        selected.extend(ordered[:active_k])
    support_indices = np.asarray(selected, dtype=np.int64)
    support_ids = tuple(physical[index] for index in selected)
    support_labels = _typed_tokens(rows.tx_labels[support_indices], "support labels", count=len(selected))
    support_plus = np.ascontiguousarray(rows.z_id[support_indices], dtype=np.float32)
    support_signed = np.ascontiguousarray(rows.pre_relu[support_indices], dtype=np.float32)
    query_plus = np.ascontiguousarray(rows.z_id[query_indices], dtype=np.float32)
    query_signed = np.ascontiguousarray(rows.pre_relu[query_indices], dtype=np.float32)
    support_root = _support_root(support_ids)
    paired_receipt = _paired_view_receipt(support_ids, support_plus, support_signed)
    fold_identity_root = _sha256(
        {
            "schema": FOLD_EXECUTION_SCHEMA + ".full_identity.v1",
            "fold_index": fold.index,
            "fold_id": fold.fold_id,
            "receiver_id": fold.receiver_id,
            "day_id": fold.day_id,
            "query_root_sha256": fold.query_root_sha256,
            "tap_snapshot_root_sha256": rows.tap_snapshot_root_sha256,
        }
    )
    query_plus_root = _array_root(query_plus)
    query_signed_root = _array_root(query_signed)
    _revalidate_snapshot(snapshot)

    baseline_bank = build_typed_zid_support_bank(
        support_plus, support_labels, registry, config=predecessor_lock
    )
    baseline_argmax = _unique_argmax(
        score_zid_student_t_logits(
            baseline_bank,
            query_plus,
            metric=identity_shared_psd_metric(config=predecessor_lock),
        ),
        registry,
    )
    try:
        built_state = _nonformal_state_from_support(
            support_plus,
            support_signed,
            support_labels,
            support_ids,
            registry,
            active_k=active_k,
            support_root_sha256=support_root,
            paired_view_receipt_sha256=paired_receipt,
            rcmr_method_lock_sha256=rcmr_method_lock_sha256,
        )
        wire = _serialize_nonformal_state(built_state)
        wire_sha = _sha256(wire)
        state = _deserialize_nonformal_state(wire, expected_sha256=wire_sha)
        if state.state_receipt_sha256 != built_state.state_receipt_sha256:
            raise D106RCMRG0Error("nonformal RCMR wire state receipt round-trip drift")
        context = _prepare_nonformal_context(state)
        candidate_argmax = tuple(
            _score_nonformal_query(
                state,
                context,
                query_plus[index],
                query_signed[index],
            )
            for index in range(len(query_indices))
        )
    except (TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, D106RCMRG0Error):
            raise
        raise D106RCMRG0Error("nonformal RCMR fold execution failed closed") from error
    if len(candidate_argmax) != len(fold.query_ids) or len(baseline_argmax) != len(fold.query_ids):
        raise D106RCMRG0Error("fold query-order length drift")
    changed_bitmap = "".join(
        "1" if candidate != baseline else "0"
        for candidate, baseline in zip(candidate_argmax, baseline_argmax, strict=True)
    )
    changed_bitmap_root = _sha256(
        {
            "encoding": "ascii01_query_order",
            "query_count": len(fold.query_ids),
            "bits": changed_bitmap,
        }
    )
    registry_root = _sha256(list(registry))
    receipt_payload = {
        "schema": FOLD_EXECUTION_SCHEMA,
        "fold_index": fold.index,
        "fold_id": fold.fold_id,
        "receiver_id": fold.receiver_id,
        "day_id": fold.day_id,
        "fold_identity_root_sha256": fold_identity_root,
        "K": active_k,
        "query_root_sha256": fold.query_root_sha256,
        "query_plus_root_sha256": query_plus_root,
        "query_signed_root_sha256": query_signed_root,
        "query_indices_root_sha256": _array_root(query_indices),
        "support_pool_indices_root_sha256": _array_root(support_pool),
        "support_indices_root_sha256": _array_root(support_indices),
        "support_root_sha256": support_root,
        "baseline_bank_receipt_sha256": baseline_bank.bank_receipt_sha256,
        "paired_view_receipt_sha256": paired_receipt,
        "rcmr_state_receipt_sha256": state.state_receipt_sha256,
        "rcmr_wire_sha256": wire_sha,
        "rcmr_method_lock_sha256": rcmr_method_lock_sha256,
        "registry_root_sha256": registry_root,
        "common_query_order_root_sha256": common_query_order_root_sha256,
        "tap_receipt_sha256": rows.tap_receipt_sha256,
        "tap_snapshot_root_sha256": rows.tap_snapshot_root_sha256,
        "candidate_argmax_root_sha256": _sha256(list(candidate_argmax)),
        "baseline_argmax_root_sha256": _sha256(list(baseline_argmax)),
        "argmax_changed_bitmap": changed_bitmap,
        "argmax_changed_bitmap_root_sha256": changed_bitmap_root,
        "argmax_changed_count": changed_bitmap.count("1"),
        "algorithm_execution_scope": ALGORITHM_SCOPE,
        "runner_authority": False,
        "p2_validated_or_deployable_claimed": False,
        "external_execution_manifest_root_sha256": None,
        "opaque_claim_not_independently_verified": True,
        "promotion_or_runner_consumption_allowed": False,
    }
    receipt_payload["execution_receipt_sha256"] = _sha256(receipt_payload)
    _revalidate_snapshot(snapshot)
    return fold.query_ids, candidate_argmax, baseline_argmax, receipt_payload


def _execute_synthetic(
    snapshot: _TapSnapshot,
    *,
    registered_classes: tuple[str, ...],
    predecessor_locks: tuple[Phase1ZIDStudentTLock, ...],
    rcmr_method_lock_sha256: str,
    synthetic_test_id: str,
) -> bytes:
    plan = _build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    common_query_root = _sha256(list(query_order))
    registry_root = _sha256(list(registered_classes))
    per_k: list[dict[str, Any]] = []
    for active_k, predecessor_lock in zip(K_VALUES, predecessor_locks, strict=True):
        all_candidate: list[str] = []
        all_baseline: list[str] = []
        all_query: list[str] = []
        fold_receipts: list[dict[str, Any]] = []
        for fold in plan:
            query_ids, candidate, baseline, fold_receipt = _execute_fold(
                snapshot,
                fold,
                active_k=active_k,
                predecessor_lock=predecessor_lock,
                rcmr_method_lock_sha256=rcmr_method_lock_sha256,
                registry=registered_classes,
                common_query_order_root_sha256=common_query_root,
            )
            all_query.extend(query_ids)
            all_candidate.extend(candidate)
            all_baseline.extend(baseline)
            fold_receipts.append(fold_receipt)
        if tuple(all_query) != query_order:
            raise D106RCMRG0Error("query order differs across K")
        changed = sum(int(receipt["argmax_changed_count"]) for receipt in fold_receipts)
        per_k_payload = {
            "schema": PER_K_SCHEMA,
            "K": active_k,
            "argmax_changed_count": int(changed),
            "query_ids_root_sha256": _sha256(all_query),
            "candidate_argmax_root_sha256": _sha256(all_candidate),
            "baseline_argmax_root_sha256": _sha256(all_baseline),
            "fold_execution_receipts": fold_receipts,
            "fold_execution_receipts_root_sha256": _sha256(
                [receipt["execution_receipt_sha256"] for receipt in fold_receipts]
            ),
            "fold_changed_bitmap_roots_root_sha256": _sha256(
                [
                    receipt["argmax_changed_bitmap_root_sha256"]
                    for receipt in fold_receipts
                ]
            ),
            "common_query_order_root_sha256": common_query_root,
            "tap_receipt_sha256": snapshot.tap_receipt_sha256,
            "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
            "registry_root_sha256": registry_root,
        }
        per_k_payload["per_k_receipt_sha256"] = _sha256(per_k_payload)
        per_k.append(per_k_payload)
    changed_by_k = {str(item["K"]): int(item["argmax_changed_count"]) for item in per_k}
    zero_changed = [active_k for active_k in K_VALUES if changed_by_k[str(active_k)] == 0]
    functional_status = (
        "G0_EVERY_K_ARGMAX_CHANGED_NO_PERFORMANCE_CLAIM"
        if not zero_changed
        else "REJECT_NO_FUNCTION_K_ZERO_CHANGED"
    )
    payload = {
        "schema": SYNTHETIC_SCHEMA,
        "status": SYNTHETIC_STATUS,
        "synthetic_test_id": _token(synthetic_test_id, "synthetic test ID"),
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "algorithm_execution_scope": ALGORITHM_SCOPE,
        "functional_gate_status": functional_status,
        "functional_gate_pass": not zero_changed,
        "zero_changed_k_values": zero_changed,
        "K_values": list(K_VALUES),
        "argmax_changed_count_by_k": changed_by_k,
        "argmax_changed_count": sum(changed_by_k.values()),
        "fold_count": EXPECTED_FOLDS,
        "query_count_per_k": EXPECTED_ROWS,
        "common_query_order_root_sha256": common_query_root,
        "tap_receipt_sha256": snapshot.tap_receipt_sha256,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "registry_root_sha256": registry_root,
        "per_k": per_k,
        "canonical_execution_root_sha256": _sha256(
            [item["per_k_receipt_sha256"] for item in per_k]
        ),
        "external_execution_manifest_root_sha256": None,
        "opaque_claim_not_independently_verified": True,
        "promotion_or_runner_consumption_allowed": False,
        "held_label_audit_status": HELD_LABEL_AUDIT_STATUS,
        "same_process_held_label_capability_absence_claimed": False,
        "runner_authority": False,
        "deployable": False,
        "formal_performance_claim": False,
        "real_g0_executed": False,
        "source_held_access": False,
        "target_access": False,
        "performance_value_field_count": 0,
        "resources": _resource_analysis(snapshot),
    }
    return _seal_result(payload)


def run_d106_rcmr_g0_synthetic_test(
    rows: D106Phase1TapRows,
    *,
    registered_classes: Sequence[str],
    predecessor_locks: Sequence[Phase1ZIDStudentTLock],
    rcmr_method_lock_sha256: str,
    synthetic_test_id: str,
) -> bytes:
    """Run only the non-formal synthetic mechanical test surface."""

    method_lock_sha = _require_sha256(
        rcmr_method_lock_sha256, "synthetic RCMR method lock"
    )
    receipt_bytes = _canonical_bytes(rows.receipt)
    snapshot = _snapshot_from_rows(rows, tap_receipt_sha256=_sha256(receipt_bytes))
    registry = _canonical_registry(registered_classes)
    locks = _validate_synthetic_locks(
        predecessor_locks, tap_receipt_sha256=snapshot.tap_receipt_sha256
    )
    return _execute_synthetic(
        snapshot,
        registered_classes=registry,
        predecessor_locks=locks,
        rcmr_method_lock_sha256=method_lock_sha,
        synthetic_test_id=synthetic_test_id,
    )


def _resource_analysis(snapshot: _TapSnapshot | None) -> dict[str, Any]:
    """List known arrays as a dimensional analysis estimate, never as RSS."""

    max_support = EXPECTED_CLASSES * max(K_VALUES)
    max_query = MAX_QUERY_ROWS_PER_FOLD
    components = {
        "query_indices_int64": max_query * 8,
        "support_pool_indices_int64": EXPECTED_ROWS * 8,
        "selected_support_indices_int64": max_support * 8,
        "query_mask_bool": EXPECTED_ROWS,
        "support_plus_float32_copy": max_support * Z_DIM * 4,
        "support_signed_float32_copy": max_support * Z_DIM * 4,
        "query_plus_float32_copy": max_query * Z_DIM * 4,
        "query_signed_float32_copy": max_query * Z_DIM * 4,
        "baseline_support_normalized_float32": max_support * Z_DIM * 4,
        "baseline_codes_int8": max_support * Z_DIM,
        "baseline_scales_float16": max_support * 2,
        "baseline_class_indices_int16": max_support * 2,
        "baseline_class_scales_float16": EXPECTED_CLASSES * 2,
        "baseline_decoded_support_float32": max_support * Z_DIM * 4,
        "baseline_query_normalized_float32": max_query * Z_DIM * 4,
        "baseline_identity_metric_float32": Z_DIM * Z_DIM * 4,
        "baseline_logits_float32": max_query * EXPECTED_CLASSES * 4,
        "nonformal_normalized_support_two_views_float64": 2 * max_support * Z_DIM * 8,
        "nonformal_quantizer_input_two_views_float64": 2 * max_support * Z_DIM * 8,
        "nonformal_codes_two_views_int8": 2 * max_support * Z_DIM,
        "nonformal_scales_two_views_float16": 2 * max_support * 2,
        "nonformal_decoded_support_two_views_float64": 2 * max_support * Z_DIM * 8,
        "nonformal_reliability_float16": max_support * 2,
        "nonformal_class_indices_uint8": max_support,
        "nonformal_canonical_order_int64": max_support * 8,
        "nonformal_reliability_row_scratch_float64": 4 * (max_support - 1) * 8,
        "nonformal_wire_numeric_payload": 2 * max_support * Z_DIM + 7 * max_support,
        "context_decoded_support_two_views_float64": 2 * max_support * Z_DIM * 8,
        "context_profiles_two_views_float64": 2 * max_support * (max_support - 1) * 8,
        "context_single_pairwise_matrix_float64": max_support * max_support * 8,
        "query_two_normalized_vectors_float64": 2 * Z_DIM * 8,
        "query_two_distance_vectors_float64": 2 * max_support * 8,
        "query_two_alpha_vectors_float64": 2 * max_support * 8,
        "query_two_beta_vectors_float64": 2 * max_support * 8,
        "query_weights_float64": max_support * 8,
        "query_two_match_vectors_float64": 2 * max_support * 8,
        "query_evidence_float64": max_support * 8,
        "query_class_scores_float64": EXPECTED_CLASSES * 8,
    }
    fold_common_names = (
        "query_indices_int64",
        "support_pool_indices_int64",
        "selected_support_indices_int64",
        "query_mask_bool",
        "support_plus_float32_copy",
        "support_signed_float32_copy",
        "query_plus_float32_copy",
        "query_signed_float32_copy",
        "baseline_support_normalized_float32",
        "baseline_codes_int8",
        "baseline_scales_float16",
        "baseline_class_indices_int16",
        "baseline_class_scales_float16",
        "baseline_decoded_support_float32",
        "baseline_query_normalized_float32",
        "baseline_identity_metric_float32",
        "baseline_logits_float32",
    )
    builder_names = tuple(name for name in components if name.startswith("nonformal_"))
    context_names = tuple(name for name in components if name.startswith("context_"))
    query_names = tuple(name for name in components if name.startswith("query_") and name not in fold_common_names)
    common = sum(components[name] for name in fold_common_names)
    builder_peak = common + sum(components[name] for name in builder_names)
    context_peak = common + sum(components[name] for name in context_names + query_names)
    peak = max(context_peak, builder_peak)
    if peak > ANALYSIS_NUMERIC_ARRAY_BUDGET_BYTES:
        raise D106RCMRG0Error("accounted arrays exceed the analysis budget")
    snapshot_arrays = (
        {name: int(getattr(snapshot, name).nbytes) for name in _SNAPSHOT_ARRAY_NAMES}
        if snapshot is not None
        else {name: "MEASURE_AT_EXECUTION" for name in _SNAPSHOT_ARRAY_NAMES}
    )
    return {
        "schema": RESOURCE_SCHEMA,
        "K_values": list(K_VALUES),
        "fold_count": EXPECTED_FOLDS,
        "query_rows_per_k": EXPECTED_ROWS,
        "fold_query_rows_hard_max": MAX_QUERY_ROWS_PER_FOLD,
        "max_support_rows": max_support,
        "tap_snapshot_array_bytes": snapshot_arrays,
        "known_incremental_numeric_array_analysis_estimate_bytes": components,
        "builder_phase_numeric_array_analysis_estimate_bytes": builder_peak,
        "context_query_phase_numeric_array_analysis_estimate_bytes": context_peak,
        "incremental_numeric_array_peak_analysis_estimate_bytes": peak,
        "analysis_numeric_array_budget_bytes": ANALYSIS_NUMERIC_ARRAY_BUDGET_BYTES,
        "analysis_budget_is_process_rss_cap": False,
        "process_rss_measured": False,
        "analysis_estimate_is_measured_peak": False,
        "parameter_scan_count": 0,
        "query_state_updates": 0,
        "unaccounted_overhead": (
            "Python containers, allocator behavior, interpreter overhead, hidden library "
            "temporaries, and process RSS are not measured; tap ndarray nbytes are listed "
            "separately and are outside the incremental 1 MiB analysis budget"
        ),
    }


def audit_d106_rcmr_g0_resources() -> dict[str, Any]:
    """Return the frozen-dimension analysis estimate without claiming RSS."""

    return _resource_analysis(None)


_PRODUCTION_RUNTIME_PINS, _PRODUCTION_RUNTIME_STATIC = (
    _capture_production_runtime_closure_minimal()
)


__all__ = [
    "ALGORITHM_SCOPE",
    "ANALYSIS_NUMERIC_ARRAY_BUDGET_BYTES",
    "D105_CANONICAL_THREE_K_LOCK_AUTHORITY_SHA256",
    "DESIGN_TRACEABILITY",
    "D106RCMRG0Error",
    "D106RCMRG0ProductionRequest",
    "EXPECTED_FOLDS",
    "EXPECTED_ROWS",
    "FOLD_POLICY",
    "HELD_LABEL_AUDIT_STATUS",
    "K_VALUES",
    "MAX_QUERY_ROWS_PER_FOLD",
    "PREDECESSOR_NUMERIC_LOCK",
    "PRODUCTION_BLOCKED_STATUS",
    "PRODUCTION_EXECUTED_STATUS",
    "PRODUCTION_EXECUTION_MANIFEST_SCHEMA",
    "PRODUCTION_MECHANICAL_STATUS",
    "PRODUCTION_SCHEMA",
    "SUPPORT_POLICY",
    "SYNTHETIC_SCHEMA",
    "SYNTHETIC_STATUS",
    "audit_d106_rcmr_g0_resources",
    "get_d106_rcmr_g0_release_expected_code_sha256",
    "run_d106_rcmr_g0_from_formal_tap",
    "run_d106_rcmr_g0_synthetic_test",
    "verify_d106_rcmr_g0_production_result_bytes",
    "verify_d106_rcmr_g0_synthetic_test_result_bytes",
]
