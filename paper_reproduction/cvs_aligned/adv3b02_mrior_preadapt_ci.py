"""Frozen MRIOR-SDA preadaptation artifacts for paper-full CI enrollment.

The module deliberately has no query input surface.  It adapts a copied
ADV3B02 identity backbone from source rows and target-old support only, then
serializes that locked state before any downstream enrollment can open query.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (
    ADV3B02MethodModel,
    _adapt,
    set_seed,
)


ARTIFACT_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_artifact.v1"
STATE_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_state.v1"
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
_STATE_KEYS = {
    "schema",
    "model_state",
    "loss_trace",
    "resource",
    "input_digests",
    "query_unopened_receipt",
}
_MANIFEST_KEYS = {
    "schema",
    "artifact_id",
    "method_id",
    "receiver",
    "seed",
    "k_shot",
    "scene",
    "checkpoint_sha256",
    "source_cache_sha256",
    "support_token_sha256",
    "method_lock_sha256",
    "state_filename",
    "state_sha256",
    "query_unopened_receipt",
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


def _digest(value: str, *, field_name: str) -> str:
    normalized = str(value)
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


def _positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _binding(
    *,
    checkpoint_sha256: str,
    source_cache_sha256: str,
    support_token_sha256: str,
    receiver: str,
    seed: int,
    k_shot: int,
    scene: str,
    method_lock_sha256: str,
) -> dict[str, str | int]:
    receiver_value = str(receiver)
    scene_value = str(scene)
    if not receiver_value or not scene_value:
        raise ValueError("receiver and scene must be nonempty")
    return {
        "checkpoint_sha256": _digest(checkpoint_sha256, field_name="checkpoint_sha256"),
        "source_cache_sha256": _digest(source_cache_sha256, field_name="source_cache_sha256"),
        "support_token_sha256": _digest(support_token_sha256, field_name="support_token_sha256"),
        "receiver": receiver_value,
        "seed": _positive_int(seed, field_name="seed"),
        "k_shot": _positive_int(k_shot, field_name="k_shot"),
        "scene": scene_value,
        "method_lock_sha256": _digest(method_lock_sha256, field_name="method_lock_sha256"),
    }


def _copy_model_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("MRIOR preadapted model state must be nonempty")
    copied: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("MRIOR preadapted model state must contain named tensors only")
        copied[name] = value.detach().cpu().clone()
    return copied


def _validate_query_unopened_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or dict(receipt) != _QUERY_UNOPENED_RECEIPT:
        raise ValueError("MRIOR preadaptation query-unopened receipt drift")
    return dict(_QUERY_UNOPENED_RECEIPT)


def _reject_query_data(value: Any, *, surface: str) -> None:
    """Keep query rows, truth, and roles out of trace/resource payloads."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if "query" in key_text or "truth" in key_text or "role" in key_text:
                raise ValueError(f"MRIOR preadaptation {surface} exposes forbidden query data")
            _reject_query_data(nested, surface=surface)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_query_data(nested, surface=surface)


@dataclass
class MRIORPreadaptResult:
    """The query-free, serializable output of MRIOR target-old preadaptation."""

    model_state: dict[str, torch.Tensor]
    loss_trace: list[dict[str, Any]]
    resource: dict[str, Any]
    input_digests: dict[str, str] = field(default_factory=dict)
    query_unopened_receipt: dict[str, Any] = field(
        default_factory=lambda: dict(_QUERY_UNOPENED_RECEIPT)
    )

    @classmethod
    def from_model(
        cls,
        model: ADV3B02MethodModel,
        *,
        trace: list[dict[str, Any]],
        resource: dict[str, Any],
    ) -> "MRIORPreadaptResult":
        if str(model.method) != METHOD_ID:
            raise ValueError("MRIOR preadaptation requires an mrior_sda method model")
        return cls(
            model_state=_copy_model_state(model.state_dict()),
            loss_trace=copy.deepcopy(trace),
            resource=copy.deepcopy(resource),
        )


def fit_mrior_preadapted_backbone(
    backbone: nn.Module,
    source_loader: DataLoader,
    target_old_x: torch.Tensor,
    target_old_y: torch.Tensor,
    *,
    seed: int,
    adapt_steps: int = 200,
    learning_rate: float = 6.0e-4,
    estimate_steps: int = 7,
    target_ce_weight: float = 1.0,
    dvkl_weight: float = 0.005,
    mu: float = 0.5,
) -> MRIORPreadaptResult:
    """Adapt a copied ADV3B02 identity backbone from source and old support.

    The caller supplies a verified source loader and sealed target-old support.
    There is intentionally no query argument or query-derived state.
    """
    if target_old_x.shape[0] != target_old_y.numel() or target_old_y.numel() == 0:
        raise ValueError("target-old support tensors must be aligned and nonempty")
    set_seed(int(seed))
    config = {
        "method_id": METHOD_ID,
        "seed": int(seed),
        "adapt_steps": int(adapt_steps),
        "mrior_adapt_learning_rate": float(learning_rate),
        "mrior_estimate_steps": int(estimate_steps),
        "target_ce_weight": float(target_ce_weight),
        "dvkl_weight": float(dvkl_weight),
        "mrior_mu": float(mu),
    }
    model = ADV3B02MethodModel(
        copy.deepcopy(backbone),
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
    return MRIORPreadaptResult.from_model(model, trace=trace, resource=resource)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_mrior_preadapt_artifact(
    artifact_root: Path | str,
    result: MRIORPreadaptResult,
    *,
    checkpoint_sha256: str,
    source_cache_sha256: str,
    support_token_sha256: str,
    receiver: str,
    seed: int,
    k_shot: int,
    scene: str,
    method_lock_sha256: str,
) -> dict[str, Any]:
    """Write a new immutable, query-free MRIOR preadaptation artifact."""
    binding = _binding(
        checkpoint_sha256=checkpoint_sha256,
        source_cache_sha256=source_cache_sha256,
        support_token_sha256=support_token_sha256,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        scene=scene,
        method_lock_sha256=method_lock_sha256,
    )
    input_digests = {
        field_name: str(binding[field_name])
        for field_name in (
            "checkpoint_sha256",
            "source_cache_sha256",
            "support_token_sha256",
            "method_lock_sha256",
        )
    }
    if result.input_digests and dict(result.input_digests) != input_digests:
        raise ValueError("MRIOR preadaptation result input digest drift")
    receipt = _validate_query_unopened_receipt(result.query_unopened_receipt)
    _reject_query_data(result.loss_trace, surface="loss trace")
    _reject_query_data(result.resource, surface="resource")
    payload = {
        "schema": STATE_SCHEMA,
        "model_state": _copy_model_state(result.model_state),
        "loss_trace": copy.deepcopy(result.loss_trace),
        "resource": copy.deepcopy(result.resource),
        "input_digests": input_digests,
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
            str(binding["receiver"]),
            int(binding["seed"]),
            int(binding["k_shot"]),
            str(binding["scene"]),
        ),
        "method_id": METHOD_ID,
        **binding,
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
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ValueError("MRIOR preadaptation manifest schema drift")
    if payload["schema"] != ARTIFACT_SCHEMA or payload["method_id"] != METHOD_ID:
        raise ValueError("MRIOR preadaptation manifest method/schema drift")
    if payload["state_filename"] != STATE_FILENAME:
        raise ValueError("MRIOR preadaptation state filename drift")
    _validate_query_unopened_receipt(payload["query_unopened_receipt"])
    return payload


def _load_state(path: Path, *, expected_state_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("MRIOR preadaptation state is unavailable")
    if _sha256_file(path) != _digest(expected_state_sha256, field_name="state_sha256"):
        raise ValueError("MRIOR preadaptation state digest drift")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError("MRIOR preadaptation state is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != _STATE_KEYS:
        raise ValueError("MRIOR preadaptation state schema drift")
    if payload["schema"] != STATE_SCHEMA:
        raise ValueError("MRIOR preadaptation state schema version drift")
    payload["model_state"] = _copy_model_state(payload["model_state"])
    if not isinstance(payload["loss_trace"], list) or not isinstance(payload["resource"], dict):
        raise ValueError("MRIOR preadaptation state trace/resource drift")
    _reject_query_data(payload["loss_trace"], surface="loss trace")
    _reject_query_data(payload["resource"], surface="resource")
    payload["input_digests"] = {
        str(key): _digest(str(value), field_name=str(key))
        for key, value in dict(payload["input_digests"]).items()
    }
    payload["query_unopened_receipt"] = _validate_query_unopened_receipt(
        payload["query_unopened_receipt"]
    )
    return payload


def load_verified_mrior_preadapt_artifact(
    artifact_root: Path | str,
    *,
    expected_checkpoint_sha256: str,
    expected_source_cache_sha256: str,
    expected_support_token_sha256: str,
    expected_receiver: str,
    expected_seed: int,
    expected_k_shot: int,
    expected_scene: str,
    expected_method_lock_sha256: str,
) -> MRIORPreadaptResult:
    """Load only an artifact whose complete sealed input binding matches."""
    expected = _binding(
        checkpoint_sha256=expected_checkpoint_sha256,
        source_cache_sha256=expected_source_cache_sha256,
        support_token_sha256=expected_support_token_sha256,
        receiver=expected_receiver,
        seed=expected_seed,
        k_shot=expected_k_shot,
        scene=expected_scene,
        method_lock_sha256=expected_method_lock_sha256,
    )
    root = Path(artifact_root)
    manifest = _read_manifest(root / MANIFEST_FILENAME)
    for field_name, expected_value in expected.items():
        if manifest[field_name] != expected_value:
            raise ValueError(f"MRIOR preadaptation {field_name} binding drift")
    expected_id = preadapt_key(
        str(expected["receiver"]),
        int(expected["seed"]),
        int(expected["k_shot"]),
        str(expected["scene"]),
    )
    if manifest["artifact_id"] != expected_id:
        raise ValueError("MRIOR preadaptation artifact identity drift")
    state = _load_state(
        root / STATE_FILENAME,
        expected_state_sha256=str(manifest["state_sha256"]),
    )
    expected_digests = {
        field_name: str(expected[field_name])
        for field_name in (
            "checkpoint_sha256",
            "source_cache_sha256",
            "support_token_sha256",
            "method_lock_sha256",
        )
    }
    if state["input_digests"] != expected_digests:
        raise ValueError("MRIOR preadaptation state input digest drift")
    if state["query_unopened_receipt"] != manifest["query_unopened_receipt"]:
        raise ValueError("MRIOR preadaptation query-unopened receipt mismatch")
    return MRIORPreadaptResult(
        model_state=state["model_state"],
        loss_trace=copy.deepcopy(state["loss_trace"]),
        resource=copy.deepcopy(state["resource"]),
        input_digests=expected_digests,
        query_unopened_receipt=state["query_unopened_receipt"],
    )
