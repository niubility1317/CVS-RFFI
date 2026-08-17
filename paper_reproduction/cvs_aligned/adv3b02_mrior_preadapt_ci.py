"""Frozen, query-free MRIOR-SDA preadaptation artifacts for CI enrollment.

Only source rows and sealed target-old support enter the adaptation call.  A
formal artifact binds that call to the frozen MRIOR lock and to canonical,
verified lineage digests before any downstream component can open query.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (
    ADV3B02MethodModel,
    _adapt,
    set_seed,
)


ARTIFACT_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_artifact.v2"
STATE_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_state.v2"
INPUT_BINDING_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_input_binding.v1"
METHOD_LOCK_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_method_lock.v1"
STATE_FILENAME = "mrior_preadapt_state.pt"
MANIFEST_FILENAME = "manifest.json"
METHOD_ID = "mrior_sda"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUERY_UNOPENED_RECEIPT = {
    "query_opened_before_model_lock": False,
    "query_rows_used_for_training": 0,
    "query_truth_access": False,
    "query_role_access": False,
    "query_class_quota_access": False,
    "query_global_reassignment_access": False,
}
_BINDING_KEYS = {
    "schema",
    "checkpoint_sha256",
    "source_cache_sha256",
    "support_token_sha256",
    "target_package_seal_sha256",
    "receiver",
    "seed",
    "k_shot",
    "scene",
}
_METHOD_LOCK_KEYS = {
    "schema",
    "method_id",
    "adapt_steps",
    "mrior_adapt_learning_rate",
    "mrior_estimate_steps",
    "target_ce_weight",
    "dvkl_weight",
    "mrior_mu",
}
_LOSS_KEY_ORDER = (
    "loss",
    "source_ce",
    "target_support_ce",
    "weighted_ce",
    "dvkl",
    "target_ce_weight",
    "dvkl_weight",
    "mu",
    "estimate_loss",
    "estimate_zeta",
    "estimate_steps",
)
_LOSS_KEYS = set(_LOSS_KEY_ORDER)
_TRACE_KEYS = {
    "method",
    "scenario",
    "phase",
    "step",
    "total_steps",
    *_LOSS_KEYS,
}
_RESOURCE_KEYS = {
    "adapt_steps",
    "final_adaptation_losses",
    "optimizer",
    "learning_rate",
    "adv3b02_gradient_updates",
}
_STATE_KEYS = {
    "schema",
    "model_state",
    "loss_trace",
    "resource",
    "input_binding",
    "input_binding_sha256",
    "method_lock",
    "method_lock_sha256",
    "query_unopened_receipt",
}
_MANIFEST_KEYS = {
    "schema",
    "artifact_id",
    "method_id",
    "input_binding",
    "input_binding_sha256",
    "method_lock",
    "method_lock_sha256",
    "state_filename",
    "state_sha256",
    "query_unopened_receipt",
}
_FROZEN_METHOD_LOCK = {
    "schema": METHOD_LOCK_SCHEMA,
    "method_id": METHOD_ID,
    "adapt_steps": 200,
    "mrior_adapt_learning_rate": 0.0006,
    "mrior_estimate_steps": 7,
    "target_ce_weight": 1.0,
    "dvkl_weight": 0.005,
    "mrior_mu": 0.5,
}


def _safe_receiver(receiver: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(receiver)).strip("_")
    if not value:
        raise ValueError("receiver must contain at least one alphanumeric character")
    return value


def preadapt_key(receiver: str, seed: int, k_shot: int, scene: str) -> str:
    """Return the shared MRIOR preadaptation identity for one sealed scenario."""
    return (
        f"rx_{_safe_receiver(receiver)}__seed_{int(seed)}__k_{int(k_shot)}"
        f"__scene_{str(scene)}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _nonempty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a nonempty, trimmed string")
    return value


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite scalar")
    return normalized


def _matches_locked_scalar(observed: float, expected: Any) -> bool:
    """Accept only the float32 serialization noise from the locked scalar."""
    return math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1.0e-8)


def _exact_mapping(
    value: Any, *, expected_keys: set[str], surface: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MRIOR preadaptation {surface} schema drift")
    normalized = dict(value)
    if set(normalized) != expected_keys:
        raise ValueError(f"MRIOR preadaptation {surface} schema drift")
    return normalized


@dataclass(frozen=True)
class MRIORPreadaptInputBinding:
    """Canonical verified lineage and target-package binding for one artifact."""

    checkpoint_sha256: str
    source_cache_sha256: str
    support_token_sha256: str
    target_package_seal_sha256: str
    receiver: str
    seed: int
    k_shot: int
    scene: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _digest(self.checkpoint_sha256, field_name="checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "source_cache_sha256",
            _digest(self.source_cache_sha256, field_name="source_cache_sha256"),
        )
        object.__setattr__(
            self,
            "support_token_sha256",
            _digest(self.support_token_sha256, field_name="support_token_sha256"),
        )
        object.__setattr__(
            self,
            "target_package_seal_sha256",
            _digest(
                self.target_package_seal_sha256,
                field_name="target_package_seal_sha256",
            ),
        )
        object.__setattr__(
            self, "receiver", _nonempty_text(self.receiver, field_name="receiver")
        )
        object.__setattr__(self, "seed", _positive_int(self.seed, field_name="seed"))
        object.__setattr__(
            self, "k_shot", _positive_int(self.k_shot, field_name="k_shot")
        )
        object.__setattr__(self, "scene", _nonempty_text(self.scene, field_name="scene"))

    @classmethod
    def from_verified_values(
        cls,
        *,
        checkpoint_sha256: str,
        source_cache_sha256: str,
        support_token_sha256: str,
        target_package_seal_sha256: str,
        receiver: str,
        seed: int,
        k_shot: int,
        scene: str,
    ) -> "MRIORPreadaptInputBinding":
        return cls(
            checkpoint_sha256=checkpoint_sha256,
            source_cache_sha256=source_cache_sha256,
            support_token_sha256=support_token_sha256,
            target_package_seal_sha256=target_package_seal_sha256,
            receiver=receiver,
            seed=seed,
            k_shot=k_shot,
            scene=scene,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "MRIORPreadaptInputBinding":
        values = _exact_mapping(
            payload, expected_keys=_BINDING_KEYS, surface="input binding"
        )
        if values["schema"] != INPUT_BINDING_SCHEMA:
            raise ValueError("MRIOR preadaptation input binding schema drift")
        return cls.from_verified_values(
            checkpoint_sha256=values["checkpoint_sha256"],
            source_cache_sha256=values["source_cache_sha256"],
            support_token_sha256=values["support_token_sha256"],
            target_package_seal_sha256=values["target_package_seal_sha256"],
            receiver=values["receiver"],
            seed=values["seed"],
            k_shot=values["k_shot"],
            scene=values["scene"],
        )

    def canonical_payload(self) -> dict[str, str | int]:
        return {
            "schema": INPUT_BINDING_SCHEMA,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_cache_sha256": self.source_cache_sha256,
            "support_token_sha256": self.support_token_sha256,
            "target_package_seal_sha256": self.target_package_seal_sha256,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "scene": self.scene,
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


def _validated_input_binding(value: Any) -> MRIORPreadaptInputBinding:
    if not isinstance(value, MRIORPreadaptInputBinding):
        raise ValueError("MRIOR preadaptation requires a canonical input binding")
    return MRIORPreadaptInputBinding.from_payload(value.canonical_payload())


def expected_mrior_preadapt_method_lock() -> dict[str, Any]:
    """Return a fresh copy of the only formal MRIOR preadaptation lock."""
    return dict(_FROZEN_METHOD_LOCK)


def _normalize_method_lock(value: Any, *, require_frozen: bool) -> dict[str, Any]:
    lock = _exact_mapping(value, expected_keys=_METHOD_LOCK_KEYS, surface="method lock")
    if lock["schema"] != METHOD_LOCK_SCHEMA or lock["method_id"] != METHOD_ID:
        raise ValueError("MRIOR preadaptation method lock schema drift")
    normalized = {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": METHOD_ID,
        "adapt_steps": _positive_int(lock["adapt_steps"], field_name="adapt_steps"),
        "mrior_adapt_learning_rate": _finite_float(
            lock["mrior_adapt_learning_rate"], field_name="mrior_adapt_learning_rate"
        ),
        "mrior_estimate_steps": _positive_int(
            lock["mrior_estimate_steps"], field_name="mrior_estimate_steps"
        ),
        "target_ce_weight": _finite_float(
            lock["target_ce_weight"], field_name="target_ce_weight"
        ),
        "dvkl_weight": _finite_float(lock["dvkl_weight"], field_name="dvkl_weight"),
        "mrior_mu": _finite_float(lock["mrior_mu"], field_name="mrior_mu"),
    }
    if require_frozen and normalized != _FROZEN_METHOD_LOCK:
        raise ValueError("MRIOR preadaptation formal method lock drift")
    return normalized


def _method_lock_from_parameters(
    *,
    adapt_steps: int,
    learning_rate: float,
    estimate_steps: int,
    target_ce_weight: float,
    dvkl_weight: float,
    mu: float,
) -> dict[str, Any]:
    return _normalize_method_lock(
        {
            "schema": METHOD_LOCK_SCHEMA,
            "method_id": METHOD_ID,
            "adapt_steps": adapt_steps,
            "mrior_adapt_learning_rate": learning_rate,
            "mrior_estimate_steps": estimate_steps,
            "target_ce_weight": target_ce_weight,
            "dvkl_weight": dvkl_weight,
            "mrior_mu": mu,
        },
        require_frozen=False,
    )


def _method_lock_sha256(lock: Mapping[str, Any]) -> str:
    normalized = _normalize_method_lock(lock, require_frozen=False)
    return _canonical_sha256(normalized)


def _copy_model_state(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("MRIOR preadapted model state must be nonempty")
    copied: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("MRIOR preadapted model state must contain named tensors only")
        copied[name] = value.detach().cpu().clone()
    return copied


def _validate_query_unopened_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or dict(receipt) != _QUERY_UNOPENED_RECEIPT:
        raise ValueError("MRIOR preadaptation query-unopened receipt drift")
    return dict(_QUERY_UNOPENED_RECEIPT)


def _expected_trace_steps(adapt_steps: int) -> list[int]:
    steps = [1]
    steps.extend(step for step in range(20, adapt_steps + 1, 20) if step != 1)
    if adapt_steps not in steps:
        steps.append(adapt_steps)
    return steps


def _validate_loss_scalars(value: Any, *, surface: str) -> dict[str, float]:
    losses = _exact_mapping(value, expected_keys=_LOSS_KEYS, surface=surface)
    return {
        key: _finite_float(losses[key], field_name=f"{surface}.{key}")
        for key in _LOSS_KEY_ORDER
    }


def _validate_loss_trace(value: Any, *, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("MRIOR preadaptation loss trace schema drift")
    expected_steps = _expected_trace_steps(int(lock["adapt_steps"]))
    if len(value) != len(expected_steps):
        raise ValueError("MRIOR preadaptation loss trace schema drift")
    normalized_trace: list[dict[str, Any]] = []
    for row, expected_step in zip(value, expected_steps):
        trace_row = _exact_mapping(row, expected_keys=_TRACE_KEYS, surface="loss trace")
        if (
            trace_row["method"] != METHOD_ID
            or trace_row["scenario"] != "sealed_by_caller"
            or trace_row["phase"] != "target_support_adaptation"
        ):
            raise ValueError("MRIOR preadaptation loss trace schema drift")
        if _positive_int(trace_row["step"], field_name="loss trace step") != expected_step:
            raise ValueError("MRIOR preadaptation loss trace schema drift")
        if _positive_int(
            trace_row["total_steps"], field_name="loss trace total_steps"
        ) != int(lock["adapt_steps"]):
            raise ValueError("MRIOR preadaptation loss trace schema drift")
        losses = _validate_loss_scalars(
            {key: trace_row[key] for key in _LOSS_KEY_ORDER},
            surface="loss trace",
        )
        for trace_key, lock_key in (
            ("target_ce_weight", "target_ce_weight"),
            ("dvkl_weight", "dvkl_weight"),
            ("mu", "mrior_mu"),
            ("estimate_steps", "mrior_estimate_steps"),
        ):
            if not _matches_locked_scalar(losses[trace_key], lock[lock_key]):
                raise ValueError("MRIOR preadaptation loss trace method lock drift")
        normalized_trace.append(
            {
                "method": METHOD_ID,
                "scenario": "sealed_by_caller",
                "phase": "target_support_adaptation",
                "step": expected_step,
                "total_steps": int(lock["adapt_steps"]),
                **losses,
            }
        )
    return normalized_trace


def _validate_resource(value: Any, *, lock: Mapping[str, Any]) -> dict[str, Any]:
    resource = _exact_mapping(value, expected_keys=_RESOURCE_KEYS, surface="resource")
    if _positive_int(resource["adapt_steps"], field_name="resource adapt_steps") != int(
        lock["adapt_steps"]
    ):
        raise ValueError("MRIOR preadaptation resource method lock drift")
    if resource["optimizer"] != "Adam_minimax":
        raise ValueError("MRIOR preadaptation resource schema drift")
    if not _matches_locked_scalar(
        _finite_float(resource["learning_rate"], field_name="resource learning_rate"),
        lock["mrior_adapt_learning_rate"],
    ):
        raise ValueError("MRIOR preadaptation resource method lock drift")
    if _positive_int(
        resource["adv3b02_gradient_updates"],
        field_name="resource adv3b02_gradient_updates",
    ) != int(lock["adapt_steps"]):
        raise ValueError("MRIOR preadaptation resource method lock drift")
    losses = _validate_loss_scalars(
        resource["final_adaptation_losses"], surface="resource final_adaptation_losses"
    )
    for resource_key, lock_key in (
        ("target_ce_weight", "target_ce_weight"),
        ("dvkl_weight", "dvkl_weight"),
        ("mu", "mrior_mu"),
        ("estimate_steps", "mrior_estimate_steps"),
    ):
        if not _matches_locked_scalar(losses[resource_key], lock[lock_key]):
            raise ValueError("MRIOR preadaptation resource method lock drift")
    return {
        "adapt_steps": int(lock["adapt_steps"]),
        "final_adaptation_losses": losses,
        "optimizer": "Adam_minimax",
        "learning_rate": float(lock["mrior_adapt_learning_rate"]),
        "adv3b02_gradient_updates": int(lock["adapt_steps"]),
    }


@dataclass
class MRIORPreadaptResult:
    """The sealed target-old MRIOR output, with no query input or payload."""

    model_state: dict[str, torch.Tensor]
    loss_trace: list[dict[str, Any]]
    resource: dict[str, Any]
    input_binding: MRIORPreadaptInputBinding
    method_lock: dict[str, Any]
    is_formal: bool
    query_unopened_receipt: dict[str, Any] = field(
        default_factory=lambda: dict(_QUERY_UNOPENED_RECEIPT)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.is_formal, bool):
            raise ValueError("MRIOR preadaptation result formal flag must be boolean")
        self.input_binding = _validated_input_binding(self.input_binding)
        self.method_lock = _normalize_method_lock(
            self.method_lock, require_frozen=self.is_formal
        )
        self.model_state = _copy_model_state(self.model_state)
        self.loss_trace = _validate_loss_trace(self.loss_trace, lock=self.method_lock)
        self.resource = _validate_resource(self.resource, lock=self.method_lock)
        self.query_unopened_receipt = _validate_query_unopened_receipt(
            self.query_unopened_receipt
        )

    @property
    def input_binding_sha256(self) -> str:
        return self.input_binding.canonical_sha256

    @property
    def method_lock_sha256(self) -> str:
        return _method_lock_sha256(self.method_lock)

    @classmethod
    def from_model(
        cls,
        model: ADV3B02MethodModel,
        *,
        trace: list[dict[str, Any]],
        resource: dict[str, Any],
        input_binding: MRIORPreadaptInputBinding,
        method_lock: Mapping[str, Any],
        is_formal: bool,
    ) -> "MRIORPreadaptResult":
        if str(model.method) != METHOD_ID:
            raise ValueError("MRIOR preadaptation requires an mrior_sda method model")
        return cls(
            model_state=_copy_model_state(model.state_dict()),
            loss_trace=copy.deepcopy(trace),
            resource=copy.deepcopy(resource),
            input_binding=input_binding,
            method_lock=dict(method_lock),
            is_formal=is_formal,
        )


def _validate_target_old_support(
    target_old_x: Any,
    target_old_y: Any,
    *,
    binding: MRIORPreadaptInputBinding,
) -> None:
    if not torch.is_tensor(target_old_x) or not torch.is_tensor(target_old_y):
        raise ValueError("target-old support tensors must be tensors")
    if (
        target_old_x.ndim < 1
        or target_old_y.ndim != 1
        or target_old_x.shape[0] != target_old_y.numel()
        or target_old_y.numel() == 0
    ):
        raise ValueError("target-old support tensors must be aligned and nonempty")
    if target_old_y.dtype != torch.long:
        raise ValueError("target-old support labels must be torch.long")
    _labels, counts = torch.unique(target_old_y.detach().cpu(), sorted=True, return_counts=True)
    if counts.numel() == 0 or not bool(torch.all(counts == binding.k_shot)):
        raise ValueError("target-old support K-shot counts do not match the input binding")


def fit_mrior_preadapted_backbone(
    backbone: nn.Module,
    source_loader: DataLoader,
    target_old_x: torch.Tensor,
    target_old_y: torch.Tensor,
    *,
    binding: MRIORPreadaptInputBinding,
    seed: int,
    adapt_steps: int = 200,
    learning_rate: float = 6.0e-4,
    estimate_steps: int = 7,
    target_ce_weight: float = 1.0,
    dvkl_weight: float = 0.005,
    mu: float = 0.5,
    _test_only_allow_nonfrozen_params: bool = False,
) -> MRIORPreadaptResult:
    """Adapt a copied MRIOR backbone from source and verified old support only.

    Formal calls can use only the frozen 200/0.0006/7/1.0/0.005/0.5 lock.
    The explicit test-only escape exercises the real minimax path at a tiny
    budget, and yields a result that the formal artifact writer rejects.
    """
    canonical_binding = _validated_input_binding(binding)
    actual_seed = _positive_int(seed, field_name="seed")
    if actual_seed != canonical_binding.seed:
        raise ValueError("MRIOR preadaptation seed does not match the input binding")
    if not isinstance(_test_only_allow_nonfrozen_params, bool):
        raise ValueError("MRIOR preadaptation test-only flag must be boolean")
    method_lock = _method_lock_from_parameters(
        adapt_steps=adapt_steps,
        learning_rate=learning_rate,
        estimate_steps=estimate_steps,
        target_ce_weight=target_ce_weight,
        dvkl_weight=dvkl_weight,
        mu=mu,
    )
    is_formal = not _test_only_allow_nonfrozen_params
    if is_formal and method_lock != _FROZEN_METHOD_LOCK:
        raise ValueError("MRIOR preadaptation fit parameters must equal the frozen method lock")
    _validate_target_old_support(target_old_x, target_old_y, binding=canonical_binding)
    set_seed(actual_seed)
    config = {
        "method_id": METHOD_ID,
        "seed": actual_seed,
        "adapt_steps": method_lock["adapt_steps"],
        "mrior_adapt_learning_rate": method_lock["mrior_adapt_learning_rate"],
        "mrior_estimate_steps": method_lock["mrior_estimate_steps"],
        "target_ce_weight": method_lock["target_ce_weight"],
        "dvkl_weight": method_lock["dvkl_weight"],
        "mrior_mu": method_lock["mrior_mu"],
    }
    exact_model = copy.deepcopy(backbone)
    if not hasattr(exact_model, "id_backbone"):
        exact_model = SimpleNamespace(
            id_backbone=exact_model,
            id_feature_key=str(getattr(backbone, "id_feature_key", "feat_joint")),
        )
    model = ADV3B02MethodModel(
        exact_model,
        method=METHOD_ID,
        feature_dim=int(backbone.emb_dim),
    ).to(target_old_x.device)
    trace, resource = _adapt(
        config,
        model,
        source_loader,
        target_old_x,
        target_old_y,
        scenario="sealed_by_caller",
        device=target_old_x.device,
    )
    return MRIORPreadaptResult.from_model(
        model,
        trace=trace,
        resource=resource,
        input_binding=canonical_binding,
        method_lock=method_lock,
        is_formal=is_formal,
    )


def _validated_result(
    result: Any, *, require_formal: bool
) -> tuple[
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    dict[str, Any],
    MRIORPreadaptInputBinding,
    dict[str, Any],
    dict[str, Any],
]:
    if not isinstance(result, MRIORPreadaptResult):
        raise ValueError("MRIOR preadaptation result type drift")
    if not isinstance(result.is_formal, bool):
        raise ValueError("MRIOR preadaptation result formal flag must be boolean")
    if require_formal and result.is_formal is not True:
        raise ValueError("MRIOR preadaptation formal writer rejects test-only results")
    binding = _validated_input_binding(result.input_binding)
    lock = _normalize_method_lock(result.method_lock, require_frozen=require_formal)
    state = _copy_model_state(result.model_state)
    trace = _validate_loss_trace(result.loss_trace, lock=lock)
    resource = _validate_resource(result.resource, lock=lock)
    receipt = _validate_query_unopened_receipt(result.query_unopened_receipt)
    return state, trace, resource, binding, lock, receipt


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_mrior_preadapt_artifact(
    artifact_root: Path | str,
    result: MRIORPreadaptResult,
) -> dict[str, Any]:
    """Write a new immutable formal artifact from its own canonical binding."""
    state, trace, resource, binding, lock, receipt = _validated_result(
        result, require_formal=True
    )
    binding_payload = binding.canonical_payload()
    binding_sha256 = binding.canonical_sha256
    method_lock_sha256 = _method_lock_sha256(lock)
    payload = {
        "schema": STATE_SCHEMA,
        "model_state": state,
        "loss_trace": trace,
        "resource": resource,
        "input_binding": binding_payload,
        "input_binding_sha256": binding_sha256,
        "method_lock": lock,
        "method_lock_sha256": method_lock_sha256,
        "query_unopened_receipt": receipt,
    }
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=False)
    state_path = root / STATE_FILENAME
    with state_path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": preadapt_key(
            binding.receiver,
            binding.seed,
            binding.k_shot,
            binding.scene,
        ),
        "method_id": METHOD_ID,
        "input_binding": binding_payload,
        "input_binding_sha256": binding_sha256,
        "method_lock": lock,
        "method_lock_sha256": method_lock_sha256,
        "state_filename": STATE_FILENAME,
        "state_sha256": _sha256_file(state_path),
        "query_unopened_receipt": receipt,
    }
    _write_json_new(root / MANIFEST_FILENAME, manifest)
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("MRIOR preadaptation manifest is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MRIOR preadaptation manifest is unreadable") from exc
    manifest = _exact_mapping(payload, expected_keys=_MANIFEST_KEYS, surface="manifest")
    if manifest["schema"] != ARTIFACT_SCHEMA or manifest["method_id"] != METHOD_ID:
        raise ValueError("MRIOR preadaptation manifest method/schema drift")
    if manifest["state_filename"] != STATE_FILENAME:
        raise ValueError("MRIOR preadaptation state filename drift")
    binding = MRIORPreadaptInputBinding.from_payload(manifest["input_binding"])
    if _digest(
        manifest["input_binding_sha256"], field_name="input_binding_sha256"
    ) != binding.canonical_sha256:
        raise ValueError("MRIOR preadaptation manifest input binding drift")
    lock = _normalize_method_lock(manifest["method_lock"], require_frozen=True)
    if _digest(
        manifest["method_lock_sha256"], field_name="method_lock_sha256"
    ) != _method_lock_sha256(lock):
        raise ValueError("MRIOR preadaptation manifest method lock drift")
    if manifest["artifact_id"] != preadapt_key(
        binding.receiver, binding.seed, binding.k_shot, binding.scene
    ):
        raise ValueError("MRIOR preadaptation artifact identity drift")
    _digest(manifest["state_sha256"], field_name="state_sha256")
    _validate_query_unopened_receipt(manifest["query_unopened_receipt"])
    return manifest


def _load_state(path: Path, *, expected_state_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("MRIOR preadaptation state is unavailable")
    if _sha256_file(path) != _digest(expected_state_sha256, field_name="state_sha256"):
        raise ValueError("MRIOR preadaptation state digest drift")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError("MRIOR preadaptation state is unreadable") from exc
    state = _exact_mapping(payload, expected_keys=_STATE_KEYS, surface="state")
    if state["schema"] != STATE_SCHEMA:
        raise ValueError("MRIOR preadaptation state schema version drift")
    binding = MRIORPreadaptInputBinding.from_payload(state["input_binding"])
    if _digest(state["input_binding_sha256"], field_name="input_binding_sha256") != binding.canonical_sha256:
        raise ValueError("MRIOR preadaptation state input binding drift")
    lock = _normalize_method_lock(state["method_lock"], require_frozen=True)
    if _digest(state["method_lock_sha256"], field_name="method_lock_sha256") != _method_lock_sha256(lock):
        raise ValueError("MRIOR preadaptation state method lock drift")
    state["model_state"] = _copy_model_state(state["model_state"])
    state["loss_trace"] = _validate_loss_trace(state["loss_trace"], lock=lock)
    state["resource"] = _validate_resource(state["resource"], lock=lock)
    state["input_binding"] = binding.canonical_payload()
    state["input_binding_sha256"] = binding.canonical_sha256
    state["method_lock"] = lock
    state["method_lock_sha256"] = _method_lock_sha256(lock)
    state["query_unopened_receipt"] = _validate_query_unopened_receipt(
        state["query_unopened_receipt"]
    )
    return state


def load_verified_mrior_preadapt_artifact(
    artifact_root: Path | str,
    *,
    expected_input_binding_sha256: str,
    expected_method_lock_sha256: str,
) -> MRIORPreadaptResult:
    """Load only a formal artifact matching both canonical binding hashes."""
    expected_binding_sha256 = _digest(
        expected_input_binding_sha256, field_name="expected_input_binding_sha256"
    )
    expected_lock_sha256 = _digest(
        expected_method_lock_sha256, field_name="expected_method_lock_sha256"
    )
    root = Path(artifact_root)
    manifest = _read_manifest(root / MANIFEST_FILENAME)
    if manifest["input_binding_sha256"] != expected_binding_sha256:
        raise ValueError("MRIOR preadaptation input binding drift")
    if manifest["method_lock_sha256"] != expected_lock_sha256:
        raise ValueError("MRIOR preadaptation method lock drift")
    state = _load_state(
        root / STATE_FILENAME,
        expected_state_sha256=manifest["state_sha256"],
    )
    if state["input_binding"] != manifest["input_binding"]:
        raise ValueError("MRIOR preadaptation state input binding drift")
    if state["input_binding_sha256"] != manifest["input_binding_sha256"]:
        raise ValueError("MRIOR preadaptation state input binding digest drift")
    if state["method_lock"] != manifest["method_lock"]:
        raise ValueError("MRIOR preadaptation state method lock drift")
    if state["method_lock_sha256"] != manifest["method_lock_sha256"]:
        raise ValueError("MRIOR preadaptation state method lock digest drift")
    if state["query_unopened_receipt"] != manifest["query_unopened_receipt"]:
        raise ValueError("MRIOR preadaptation query-unopened receipt mismatch")
    return MRIORPreadaptResult(
        model_state=state["model_state"],
        loss_trace=state["loss_trace"],
        resource=state["resource"],
        input_binding=MRIORPreadaptInputBinding.from_payload(state["input_binding"]),
        method_lock=state["method_lock"],
        is_formal=True,
        query_unopened_receipt=state["query_unopened_receipt"],
    )
