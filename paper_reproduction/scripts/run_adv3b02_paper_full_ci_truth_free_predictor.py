#!/usr/bin/env python3
"""Run paper-mechanism CSIL/MoPC-HR with a trainable sealed ADV3B02."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact  # noqa: E402
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PREDICTOR_PACKAGE_SEAL_SCHEMA,
    SEAL_REQUIRED_KEYS,
    _ensure_root,
    _hash_handle,
    _json_from_handle,
    _materialize_npz,
    _validate_manifest,
    _validate_query_arrays,
    _validate_support_arrays,
    open_regular_member_same_fd,
    preflight_stage2_predictor_package,
    sha256_file,
    validate_relative_member_path,
)
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_ci_heads import prototype_baseline  # noqa: E402
from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import (  # noqa: E402
    METHODS as LEGACY_METHODS,
    fit_paper_full,
    predict_after as predict_after_legacy,
    predict_before as predict_before_legacy,
)
from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (  # noqa: E402
    load_verified_mrior_preadapt_artifact,
)
from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import (  # noqa: E402
    METHODS as OFFICIAL_METHODS,
    fit_official_repo,
    predict_after as predict_after_official,
    predict_before as predict_before_official,
)


MRIOR_PREADAPT_METHOD_TO_LEGACY = {
    "mrior_sda_then_csil_paper_full": "csil_paper_full",
    "mrior_sda_then_mopc_hr_paper_full": "mopc_hr_paper_full",
}
MRIOR_PREADAPT_BINDINGS_SCHEMA = (
    "cvs.phase2.adv3b02_mrior_preadapt_predictor_bindings.v1"
)
MRIOR_PREADAPT_CHECKPOINT_SCHEMA = "adv3b02.torchscript_identity_runtime.v1"
METHODS = (
    LEGACY_METHODS
    + tuple(MRIOR_PREADAPT_METHOD_TO_LEGACY)
    + OFFICIAL_METHODS
)


def _method_receipt_semantics(method: str) -> tuple[str, str, str]:
    if method in MRIOR_PREADAPT_METHOD_TO_LEGACY:
        return (
            "cvs.phase2.adv3b02_mrior_preadapt_ci_predictor_receipt.v1",
            "FORMAL_COMPARISON_MRIOR_PREADAPT",
            "formal_paper_method_comparison_mrior_preadapt",
        )
    if "sequential5" in method:
        return (
            "cvs.phase2.adv3b02_official_corefix_adapter_predictor_receipt.v2",
            "ORDERED_ARRIVAL_DIAGNOSTIC",
            "ORDERED_ARRIVAL_DIAGNOSTIC/SEQUENTIAL_CVS_ADAPTER",
        )
    if method.endswith("_cvs_adapter"):
        return (
            "cvs.phase2.adv3b02_official_corefix_adapter_predictor_receipt.v2",
            "FORMAL_COMPARISON_INTERFACE_ADAPTER",
            "OFFICIAL_CORE_CVS_INTERFACE_ADAPTER",
        )
    return (
        "cvs.phase2.adv3b02_paper_full_ci_predictor_receipt.v1",
        "FORMAL_COMPARISON_BASELINE",
        "formal_paper_method_comparison_baseline",
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return normalized


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _read_mrior_preadapt_bindings(value: Path | str | None) -> dict[str, dict[str, Any]]:
    if value is None:
        raise ValueError("MRIOR preadapted methods require --mrior-preadapt-bindings")
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("MRIOR preadapt bindings file is unavailable") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MRIOR preadapt bindings file is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "bindings"}:
        raise ValueError("MRIOR preadapt bindings schema drift")
    if payload["schema"] != MRIOR_PREADAPT_BINDINGS_SCHEMA:
        raise ValueError("MRIOR preadapt bindings schema drift")
    raw_bindings = payload["bindings"]
    if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != set(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise ValueError("MRIOR preadapt bindings scenario drift")
    bindings: dict[str, dict[str, Any]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        raw = raw_bindings[scenario]
        if not isinstance(raw, Mapping) or set(raw) != {
            "artifact_root",
            "expected_input_binding_sha256",
            "expected_method_lock_sha256",
        }:
            raise ValueError("MRIOR preadapt binding entry drift")
        artifact_root = raw["artifact_root"]
        if not isinstance(artifact_root, str) or not artifact_root.strip():
            raise ValueError("MRIOR preadapt artifact root is missing")
        try:
            resolved_root = Path(artifact_root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("MRIOR preadapt artifact root is unavailable") from exc
        bindings[scenario] = {
            "artifact_root": resolved_root,
            "expected_input_binding_sha256": _require_sha256(
                raw["expected_input_binding_sha256"],
                field="MRIOR preadapt expected input binding SHA",
            ),
            "expected_method_lock_sha256": _require_sha256(
                raw["expected_method_lock_sha256"],
                field="MRIOR preadapt expected method lock SHA",
            ),
        }
    return bindings


def _preflight_mrior_preadapt_metadata(
    package_root: Path,
    *,
    detached_seal_path: Path | str,
    expected_seal_sha256: str,
) -> dict[str, Any]:
    """Validate sealed package metadata without opening support/query members."""

    root = _ensure_root(package_root)
    seal_path = Path(detached_seal_path)
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("detached seal must be a regular non-symlink file")
    expected_digest = _require_sha256(
        expected_seal_sha256, field="detached seal expected SHA"
    )
    if sha256_file(seal_path) != expected_digest:
        raise ValueError("detached seal digest mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    if not isinstance(seal, dict) or set(seal) != SEAL_REQUIRED_KEYS:
        raise ValueError("detached seal exact schema mismatch")
    if seal.get("schema") != PREDICTOR_PACKAGE_SEAL_SCHEMA:
        raise ValueError("detached seal schema drift")
    manifest_relative = validate_relative_member_path(seal["manifest_relative_path"])
    if manifest_relative != "package_manifest.json":
        raise ValueError("package manifest path drift")
    with open_regular_member_same_fd(root, manifest_relative) as handle:
        manifest_digest, manifest_size = _hash_handle(handle)
        if (
            manifest_digest != seal["manifest_sha256"]
            or manifest_size != seal["manifest_size_bytes"]
        ):
            raise ValueError("package manifest detached digest mismatch")
        manifest = _json_from_handle(handle, context="package manifest")
    _validate_manifest(manifest)
    if manifest["package_root_sha256"] != seal["package_root_sha256"]:
        raise ValueError("manifest/seal package root mismatch")
    if seal["artifact_member_allowlist_sha256"] != seal["package_root_sha256"]:
        raise ValueError("artifact member allowlist digest mismatch")
    return manifest


def _restore_mrior_preadapted_backbone(
    backbone: torch.nn.Module, model_state: Any
) -> None:
    if not isinstance(model_state, Mapping):
        raise ValueError("MRIOR preadapted backbone state is invalid")
    state: dict[str, torch.Tensor] = {}
    for name, value in model_state.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("MRIOR preadapted backbone state is invalid")
        if name.startswith("id_backbone."):
            state[name.removeprefix("id_backbone.")] = value
        elif not name.startswith("estimate_network."):
            raise ValueError("MRIOR preadapted backbone state has an unknown surface")
    if not state:
        raise ValueError("MRIOR preadapted backbone state misses id_backbone")
    try:
        backbone.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("MRIOR preadapted backbone state is incompatible") from exc


def _validated_mrior_preadapt_lineage(
    result: Any,
    *,
    artifact_root: Path,
    checkpoint_sha256: str,
    target_package_seal_sha256: str,
    receiver: str,
    seed: int,
    k_shot: int,
    scenario: str,
) -> dict[str, Any]:
    binding = getattr(result, "input_binding", None)
    if binding is None:
        raise ValueError("MRIOR preadaptation input binding drift")
    _require_sha256(
        target_package_seal_sha256,
        field="MRIOR preadaptation current package seal SHA",
    )
    if (
        getattr(binding, "checkpoint_sha256", None) != checkpoint_sha256
        or getattr(binding, "receiver", None) != receiver
        or getattr(binding, "seed", None) != seed
        or getattr(binding, "k_shot", None) != k_shot
        or getattr(binding, "scene", None) != scenario
    ):
        raise ValueError("MRIOR preadaptation row binding drift")
    # The input-binding SHA protects the Task2 anchor-package seal. That anchor
    # deliberately differs from the current package seal when the same old
    # support is reused across new-class counts, so equality here would break
    # the required cross-new-count artifact reuse.
    if not isinstance(artifact_root, Path):
        raise ValueError("MRIOR preadapt artifact root drift")
    return {
        "state": "DA1_REG0",
        "artifact_root": str(artifact_root),
        "artifact_manifest_sha256": sha256_file(artifact_root / "manifest.json"),
        "artifact_state_sha256": sha256_file(
            artifact_root / "mrior_preadapt_state.pt"
        ),
        "input_binding_sha256": binding.canonical_sha256,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "source_cache_sha256": binding.source_cache_sha256,
        "target_package_seal_sha256": binding.target_package_seal_sha256,
        "support_token_sha256": binding.support_token_sha256,
        "receiver": binding.receiver,
        "seed": binding.seed,
        "k_shot": binding.k_shot,
        "scenario": binding.scene,
    }


def _verify_mrior_preadapt_artifacts(
    *,
    manifest: Mapping[str, Any],
    roles: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if int(manifest["seed"]) != int(args.seed):
        raise ValueError("package seed does not match predictor seed")
    bindings = _read_mrior_preadapt_bindings(
        getattr(args, "mrior_preadapt_bindings", None)
    )
    checkpoint = roles.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MRIOR preadaptation checkpoint descriptor is missing")
    checkpoint_sha256 = _require_sha256(
        checkpoint.get("sha256"), field="MRIOR preadaptation checkpoint SHA"
    )
    package_seal_sha256 = _require_sha256(
        args.expected_seal_sha256, field="MRIOR preadaptation package seal SHA"
    )
    receiver = str(manifest["receiver"])
    results: dict[str, Any] = {}
    lineage_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        entry = bindings[scenario]
        result = load_verified_mrior_preadapt_artifact(
            entry["artifact_root"],
            expected_input_binding_sha256=entry["expected_input_binding_sha256"],
            expected_method_lock_sha256=entry["expected_method_lock_sha256"],
        )
        lineage_by_scenario[scenario] = _validated_mrior_preadapt_lineage(
            result,
            artifact_root=entry["artifact_root"],
            checkpoint_sha256=checkpoint_sha256,
            target_package_seal_sha256=package_seal_sha256,
            receiver=receiver,
            seed=int(args.seed),
            k_shot=int(args.k_shot),
            scenario=scenario,
        )
        results[scenario] = result
    return results, lineage_by_scenario


def _load_mrior_preadapted_backbones(
    *,
    package_root: Path,
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    checkpoint_identity = _checkpoint_identity(roles.get("checkpoint"))
    base_backbone, feature_fn, audit = _load_exact_backbone(
        package_root,
        manifest,
        device=device,
        verify_checkpoint_member=True,
    )
    prepared: dict[str, tuple[torch.nn.Module, Any, dict[str, Any]]] = {}
    audits: list[dict[str, Any]] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        backbone = copy.deepcopy(base_backbone)
        _restore_mrior_preadapted_backbone(backbone, results[scenario].model_state)
        prepared[scenario] = (backbone, feature_fn, audit)
        audits.append({"scenario": scenario, **audit})
    return prepared, audits, checkpoint_identity


def _old_support_token_sha256(
    arrays: Mapping[str, np.ndarray],
    *,
    old_class_count: int,
    k_shot: int,
    scenario: str,
) -> str:
    labels = np.asarray(arrays["support_pool_class_indices"], dtype=np.int64)
    ranks = np.asarray(arrays["support_pool_rank_within_class"], dtype=np.int64)
    tokens = np.asarray(arrays["support_pool_tokens"]).astype(str)
    selected = sorted(
        (
            (int(label), int(rank), str(token))
            for label, rank, token in zip(labels.tolist(), ranks.tolist(), tokens.tolist())
            if int(label) < int(old_class_count) and int(rank) < int(k_shot)
        ),
        key=lambda value: (value[0], value[1]),
    )
    expected_pairs = [
        (class_index, rank)
        for class_index in range(int(old_class_count))
        for rank in range(int(k_shot))
    ]
    if [(label, rank) for label, rank, _token in selected] != expected_pairs:
        raise ValueError("MRIOR preadaptation target-old support identity drift")
    ordered_tokens = [token for _label, _rank, token in selected]
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError("MRIOR preadaptation target-old support token collision")
    return _canonical_sha256(
        {
            "old_class_count": int(old_class_count),
            "k_shot": int(k_shot),
            "scenario": scenario,
            "ordered_support_tokens": ordered_tokens,
        }
    )


def _mrior_preadapt_receipt_fields(
    lineage_by_scenario: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "mrior_preadapt_da1_reg0_lineage_by_scenario": lineage_by_scenario,
        "mrior_preadapt_artifact_manifest_sha256_by_scenario": {
            scenario: lineage_by_scenario[scenario]["artifact_manifest_sha256"]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "mrior_preadapt_source_access_declaration": {
            "phase": "DA1_REG0",
            "adaptation_source_access": "sealed_mrior_source_leo_cache_provenance",
            "predictor_runtime_source_cache_opened": False,
            "source_cache_sha256_by_scenario": {
                scenario: lineage_by_scenario[scenario]["source_cache_sha256"]
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
        },
        "query_opened_after_model_lock": True,
    }


def _write_json_new(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tensor(value: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(value)
    if dtype == torch.float32:
        array = array.astype(np.float32, copy=False)
    elif dtype == torch.int64:
        array = array.astype(np.int64, copy=False)
    else:
        raise TypeError(dtype)
    # N607 currently pairs NumPy 2.2.5 with Torch 2.1.0; its NumPy C bridge
    # rejects genuine ndarrays. The buffer protocol avoids that ABI boundary.
    # clone() owns the storage before the local NumPy array leaves scope.
    tensor = torch.frombuffer(memoryview(array), dtype=dtype)
    return tensor.reshape(array.shape).clone().to(device=device, dtype=dtype)


def _selected_support(arrays: Mapping[str, np.ndarray], *, k_shot: int):
    labels = np.asarray(arrays["support_pool_class_indices"], dtype=np.int64)
    ranks = np.asarray(arrays["support_pool_rank_within_class"], dtype=np.int64)
    selected = np.flatnonzero(ranks < int(k_shot))
    expected = int(np.unique(labels).size) * int(k_shot)
    if len(selected) != expected:
        raise ValueError("nested support selection count drift")
    return (
        np.asarray(arrays["support_pool_leo_weak_iq"], dtype=np.float32)[selected],
        labels[selected],
        np.asarray(arrays["support_pool_tokens"]).astype(str)[selected],
    )


def _handles(indices: torch.Tensor, class_handles: list[str]) -> np.ndarray:
    values = indices.detach().cpu().tolist()
    if any(int(value) < 0 or int(value) >= len(class_handles) for value in values):
        raise ValueError("predicted class index is outside the registry")
    return np.asarray([class_handles[int(value)] for value in values])


def _checkpoint_identity(descriptor: Any) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping) or descriptor.get("artifact_role") != "checkpoint":
        raise ValueError("MRIOR preadaptation checkpoint descriptor is missing")
    relative_path = validate_relative_member_path(descriptor.get("relative_path"))
    sha256 = _require_sha256(
        descriptor.get("sha256"), field="MRIOR preadaptation checkpoint SHA"
    )
    size_bytes = descriptor.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("MRIOR preadaptation checkpoint size is invalid")
    if descriptor.get("schema") != MRIOR_PREADAPT_CHECKPOINT_SCHEMA:
        raise ValueError("MRIOR preadaptation checkpoint schema drift")
    if descriptor.get("scenario") is not None or descriptor.get("npz_members") != []:
        raise ValueError("MRIOR preadaptation checkpoint descriptor drift")
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "schema": descriptor["schema"],
    }


def _load_exact_backbone(
    package_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    verify_checkpoint_member: bool = False,
):
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    descriptor = roles["checkpoint"]
    checkpoint_identity = (
        _checkpoint_identity(descriptor) if verify_checkpoint_member else None
    )
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        if checkpoint_identity is not None:
            digest, size = _hash_handle(handle)
            if (
                digest != checkpoint_identity["sha256"]
                or size != checkpoint_identity["size_bytes"]
            ):
                raise ValueError("MRIOR preadaptation checkpoint member digest drift")
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    exact, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=256,
        device=device,
    )
    if not hasattr(exact, "id_backbone"):
        raise ValueError("ADV3B02 checkpoint misses id_backbone")
    feature_key = str(getattr(exact, "id_feature_key", "feat_joint"))

    def feature_fn(backbone: torch.nn.Module, rows: torch.Tensor):
        auxiliary = backbone_forward_compat(
            backbone,
            rows,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        feature = auxiliary.get(feature_key)
        if not torch.is_tensor(feature):
            feature = auxiliary.get("feat_joint")
        logits = auxiliary.get("logits")
        if not torch.is_tensor(feature) or not torch.is_tensor(logits):
            raise ValueError("ADV3B02 identity backbone output drift")
        return feature.float(), logits.float()

    if checkpoint_identity is not None:
        audit = {**audit, "checkpoint_member_identity": checkpoint_identity}
    return exact.id_backbone.to(device), feature_fn, audit


def _load_base_state(
    package_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
    old_count: int,
    expected_total_capacity: int,
) -> dict[str, Any]:
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    descriptor = roles["head"]
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    schema = state.get("schema")
    if schema == "cvs.adv3b02.official_repo_base_state.v2":
        if int(state.get("total_capacity", 0)) != int(
            expected_total_capacity
        ):
            raise ValueError(
                "official-repo base state total capacity drift"
            )
        if int(state.get("base_sample_count", 0)) != 8400:
            raise ValueError("official-repo base state requires exactly 8400 rows")
        if (
            int(state.get("csil_base_train_sample_count", 0)) != 5879
            or int(state.get("fisher_sample_count", 0)) != 2521
            or state.get("source_train_fisher_disjoint") is not True
        ):
            raise ValueError("official CSIL 70/30 train/Fisher split drift")
        if not isinstance(state.get("csil"), dict) or not isinstance(
            state.get("mopc_hr"), dict
        ):
            raise ValueError("official-repo method base states are missing")
        classifier_weight = state["mopc_hr"].get("classifier_weight")
        classifier_bias = state["mopc_hr"].get("classifier_bias")
        if (
            not torch.is_tensor(classifier_weight)
            or not torch.is_tensor(classifier_bias)
            or int(classifier_weight.shape[0]) != int(expected_total_capacity)
            or int(classifier_bias.shape[0]) != int(expected_total_capacity)
        ):
            raise ValueError(
                "official-repo MoPC classifier capacity drift"
            )
        return {
            "csil": state["csil"],
            "mopc_hr": state["mopc_hr"],
            "receipt": {
                "schema": schema,
                "checkpoint_sha256": state.get("checkpoint_sha256"),
                "base_sample_count": int(state["base_sample_count"]),
                "total_capacity": int(state["total_capacity"]),
                "base_class_counts": list(state.get("base_class_counts", [])),
                "csil_base_train_sample_count": int(
                    state["csil_base_train_sample_count"]
                ),
                "fisher_sample_count": int(state["fisher_sample_count"]),
                "fisher_class_counts": list(state.get("fisher_class_counts", [])),
                "source_train_fisher_disjoint": True,
                "source_receiver_labels": list(
                    state.get("source_receiver_labels", [])
                ),
                "official_repo_commits": dict(
                    state.get("official_repo_commits", {})
                ),
                "raw_exemplars_stored": bool(
                    state.get("raw_exemplars_stored", True)
                ),
            },
        }
    if schema != "cvs.adv3b02.paper_full_base_state.v1":
        raise ValueError("paper-full base state schema drift")
    if len(state.get("old_class_labels", [])) != int(old_count):
        raise ValueError("paper-full base state old-class count drift")
    fingerprints = state.get("old_fingerprints")
    prototypes = state.get("old_prototypes")
    fisher = state.get("fisher")
    if (
        not torch.is_tensor(fingerprints)
        or not torch.is_tensor(prototypes)
        or not isinstance(fisher, dict)
    ):
        raise ValueError("paper-full base state tensor surface drift")
    return {
        "old_fingerprints": fingerprints.to(device),
        "old_prototypes": prototypes.to(device),
        "fisher": {name: value.to(device) for name, value in fisher.items()},
        "receipt": {
            "schema": state["schema"],
            "checkpoint_sha256": state.get("checkpoint_sha256"),
            "base_sample_count": int(state.get("base_sample_count", 0)),
            "source_receiver_labels": list(state.get("source_receiver_labels", [])),
            "raw_exemplars_stored": bool(state.get("raw_exemplars_stored", True)),
        },
    }


@torch.no_grad()
def _forward_direct(
    backbone: torch.nn.Module,
    feature_fn,
    rows: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    logits = []
    for start in range(0, len(rows), int(batch_size)):
        feature, logit = feature_fn(backbone, rows[start : start + int(batch_size)])
        features.append(feature)
        logits.append(logit)
    return torch.cat(features), torch.cat(logits)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    method = str(args.method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    is_mrior_preadapt = method in MRIOR_PREADAPT_METHOD_TO_LEGACY
    if not is_mrior_preadapt and getattr(args, "mrior_preadapt_bindings", None) is not None:
        raise ValueError(
            "only MRIOR preadapted methods may receive --mrior-preadapt-bindings"
        )
    if is_mrior_preadapt and getattr(args, "mrior_preadapt_bindings", None) is None:
        raise ValueError("MRIOR preadapted methods require --mrior-preadapt-bindings")
    effective_method = MRIOR_PREADAPT_METHOD_TO_LEGACY.get(method, method)
    package_root = Path(args.package_root).resolve(strict=True)
    mrior_preadapt_results: dict[str, Any] = {}
    mrior_preadapt_lineage_by_scenario: dict[str, dict[str, Any]] = {}
    preadapted_backbones_by_scenario = {}
    preadapt_checkpoint_audits: list[dict[str, Any]] = []
    mrior_preadapt_checkpoint_identity: dict[str, Any] | None = None
    mrior_preadapt_device: torch.device | None = None
    if is_mrior_preadapt:
        mrior_preadapt_device = torch.device(
            str(args.device) if torch.cuda.is_available() else "cpu"
        )
        if mrior_preadapt_device.type == "cuda":
            torch.empty(0, device=mrior_preadapt_device)
            torch.cuda.reset_peak_memory_stats(mrior_preadapt_device)
        metadata_manifest = _preflight_mrior_preadapt_metadata(
            package_root,
            detached_seal_path=args.detached_seal,
            expected_seal_sha256=args.expected_seal_sha256,
        )
        if metadata_manifest["stage"] != "stage2c":
            raise ValueError("paper-full CI predictor requires Stage2-C")
        if int(metadata_manifest["seed"]) != int(args.seed):
            raise ValueError("package seed does not match predictor seed")
        metadata_roles = {
            item["artifact_role"]: item for item in metadata_manifest["members"]
        }
        (
            mrior_preadapt_results,
            mrior_preadapt_lineage_by_scenario,
        ) = _verify_mrior_preadapt_artifacts(
            manifest=metadata_manifest,
            roles=metadata_roles,
            args=args,
        )
        (
            preadapted_backbones_by_scenario,
            preadapt_checkpoint_audits,
            mrior_preadapt_checkpoint_identity,
        ) = _load_mrior_preadapted_backbones(
            package_root=package_root,
            manifest=metadata_manifest,
            results=mrior_preadapt_results,
            device=mrior_preadapt_device,
        )
    manifest, _seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=args.detached_seal,
        expected_seal_sha256=args.expected_seal_sha256,
    )
    if manifest["stage"] != "stage2c":
        raise ValueError("paper-full CI predictor requires Stage2-C")
    if is_mrior_preadapt and int(manifest["seed"]) != int(args.seed):
        raise ValueError("package seed does not match predictor seed")
    old_count = int(args.old_class_count)
    class_handles = [item["class_handle"] for item in manifest["registered_classes"]]
    if not 0 < old_count < len(class_handles):
        raise ValueError("old class count is inconsistent with the class registry")
    if int(args.k_shot) > int(manifest["support_pool_max_k"]):
        raise ValueError("K exceeds sealed support pool")
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    if is_mrior_preadapt:
        if mrior_preadapt_device is None or mrior_preadapt_checkpoint_identity is None:
            raise ValueError("MRIOR preadaptation checkpoint preparation is missing")
        if _checkpoint_identity(roles.get("checkpoint")) != mrior_preadapt_checkpoint_identity:
            raise ValueError("MRIOR preadaptation checkpoint identity drift after preflight")
        device = mrior_preadapt_device
    else:
        device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.empty(0, device=device)
            torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    base_state = _load_base_state(
        package_root,
        manifest,
        device=device,
        old_count=old_count,
        expected_total_capacity=int(args.expected_total_capacity),
    )
    opened_roles = ["checkpoint", "head"]
    fitted_by_scenario = {}
    support_by_scenario = {}
    resources = []
    loss_trace = []
    reference_support_tokens = None
    checkpoint_audits = list(preadapt_checkpoint_audits)
    training_started = time.perf_counter()

    # Enrollment: only checkpoint and support members are open.
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        arrays, support_manifest = _materialize_npz(
            package_root, roles[f"support:{scenario}"]
        )
        _validate_support_arrays(
            arrays,
            support_manifest,
            scenario=scenario,
            class_count=len(class_handles),
            max_k=int(manifest["support_pool_max_k"]),
        )
        opened_roles.append(f"support:{scenario}")
        if is_mrior_preadapt:
            if _old_support_token_sha256(
                arrays,
                old_class_count=old_count,
                k_shot=int(args.k_shot),
                scenario=scenario,
            ) != mrior_preadapt_lineage_by_scenario[scenario][
                "support_token_sha256"
            ]:
                raise ValueError("MRIOR preadaptation target-old support token binding drift")
        iq_np, labels_np, tokens = _selected_support(arrays, k_shot=int(args.k_shot))
        if reference_support_tokens is None:
            reference_support_tokens = tokens
        elif not np.array_equal(reference_support_tokens, tokens):
            raise ValueError("physical support ordering drift across scenarios")
        support_x = _tensor(iq_np, dtype=torch.float32, device=device)
        support_y = _tensor(labels_np, dtype=torch.int64, device=device).long()
        if is_mrior_preadapt:
            backbone, feature_fn, audit = preadapted_backbones_by_scenario[scenario]
        else:
            backbone, feature_fn, audit = _load_exact_backbone(
                package_root, manifest, device=device
            )
            checkpoint_audits.append({"scenario": scenario, **audit})
        started = time.perf_counter()
        if method in OFFICIAL_METHODS:
            fitted = fit_official_repo(
                method,
                backbone,
                support_x,
                support_y,
                feature_fn=feature_fn,
                old_count=old_count,
                seed=int(args.seed) + scenario_index * 1009,
                base_state=base_state,
            )
        else:
            fitted = fit_paper_full(
                effective_method,
                backbone,
                support_x,
                support_y,
                feature_fn=feature_fn,
                old_count=old_count,
                seed=int(args.seed) + scenario_index * 1009,
                base_state=base_state,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resource = {
            "scenario": scenario,
            **fitted.resource,
            "adaptation_seconds": time.perf_counter() - started,
        }
        if is_mrior_preadapt:
            resource["mrior_preadapt_da1_reg0"] = (
                mrior_preadapt_lineage_by_scenario[scenario]
            )
        fitted_by_scenario[scenario] = fitted
        support_by_scenario[scenario] = (support_x, support_y, feature_fn)
        resources.append(resource)
        loss_trace.extend({"scenario": scenario, **row} for row in fitted.loss_trace)

    # Serialize/hash-lock all trainable model and prototype state before query opens.
    state_path = output_dir / "enrolled_models.pt"
    with state_path.open("xb") as handle:
        torch.save(
            {
                scenario: fitted_by_scenario[scenario].serializable_state()
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    model_state_sha256 = sha256_file(state_path)
    enrollment_receipt_path = output_dir / "enrollment_receipt.json"
    enrollment_receipt = {
            "schema": "cvs.phase2.adv3b02_paper_full_ci_enrollment_receipt.v1",
            "status": "PASS",
            "method": method,
            "receiver": manifest["receiver"],
            "seed": int(args.seed),
            "k_shot": int(args.k_shot),
            "new_class_count": int(manifest["new_class_count"]),
            "query_members_opened_before_model_lock": False,
            "opened_roles_before_model_lock": list(opened_roles),
            "enrolled_model_state_sha256": model_state_sha256,
            "query_rows_used_for_training": 0,
            "checkpoint_load_audits": checkpoint_audits,
            "base_state_receipt": base_state["receipt"],
    }
    if is_mrior_preadapt:
        enrollment_receipt.update(
            _mrior_preadapt_receipt_fields(mrior_preadapt_lineage_by_scenario)
        )
    _write_json_new(enrollment_receipt_path, enrollment_receipt)

    streams = {
        name: []
        for name in (
            "candidate_after",
            "candidate_before",
            "identity_after",
            "identity_before",
            "direct",
        )
    }
    token_rows = []
    scenario_rows = []
    reference_query_tokens = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        query, query_manifest = _materialize_npz(
            package_root, roles[f"query:{scenario}"]
        )
        _validate_query_arrays(query, query_manifest, scenario=scenario)
        opened_roles.append(f"query:{scenario}")
        query_iq = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
        query_tokens = np.asarray(query["query_tokens"]).astype(str)
        if reference_query_tokens is None:
            reference_query_tokens = query_tokens
        elif not np.array_equal(reference_query_tokens, query_tokens):
            raise ValueError("physical query ordering drift across scenarios")
        if reference_support_tokens is not None and set(reference_support_tokens) & set(query_tokens):
            raise ValueError("support/query opaque token overlap")
        query_x = _tensor(query_iq, dtype=torch.float32, device=device)
        fitted = fitted_by_scenario[scenario]
        support_x, support_y, feature_fn = support_by_scenario[scenario]
        if method in OFFICIAL_METHODS:
            candidate_before = predict_before_official(fitted, query_x)
            candidate_after = predict_after_official(fitted, query_x)
        else:
            candidate_before = predict_before_legacy(fitted, query_x)
            candidate_after = predict_after_legacy(fitted, query_x)
        support_features, _ = _forward_direct(
            fitted.teacher_backbone,
            feature_fn,
            support_x,
            batch_size=int(args.batch_size),
        )
        query_features, direct_logits = _forward_direct(
            fitted.teacher_backbone,
            feature_fn,
            query_x,
            batch_size=int(args.batch_size),
        )
        old_mask = support_y < old_count
        identity_before = prototype_baseline(
            support_features[old_mask],
            support_y[old_mask],
            query_features,
            class_count=old_count,
        )
        identity_after = prototype_baseline(
            support_features,
            support_y,
            query_features,
            class_count=len(class_handles),
        )
        if direct_logits.ndim != 2 or int(direct_logits.shape[1]) != old_count:
            raise ValueError("direct ADV3B02 class order drift")
        streams["candidate_after"].append(_handles(candidate_after, class_handles))
        streams["candidate_before"].append(_handles(candidate_before, class_handles[:old_count]))
        streams["identity_after"].append(_handles(identity_after, class_handles))
        streams["identity_before"].append(_handles(identity_before, class_handles[:old_count]))
        streams["direct"].append(_handles(direct_logits.argmax(1), class_handles[:old_count]))
        token_rows.append(query_tokens)
        scenario_rows.append(np.asarray([scenario] * len(query_tokens)))

    prediction_path = output_dir / "prediction_artifact.cvspred"
    publication = publish_prediction_artifact(
        prediction_path,
        stage="Stage2-C",
        row_id=str(args.row_id),
        receiver=str(manifest["receiver"]),
        k_shot=int(args.k_shot),
        candidate_lock_sha256=str(manifest["candidate_lock_sha256"]),
        package_root_sha256=str(manifest["package_root_sha256"]),
        package_seal_sha256=str(args.expected_seal_sha256),
        query_tokens=np.concatenate(token_rows),
        scenarios=np.concatenate(scenario_rows),
        candidate_after=np.concatenate(streams["candidate_after"]),
        candidate_before=np.concatenate(streams["candidate_before"]),
        identity_after=np.concatenate(streams["identity_after"]),
        identity_before=np.concatenate(streams["identity_before"]),
        direct=np.concatenate(streams["direct"]),
        shared_view_counts=np.ones(sum(len(v) for v in token_rows), dtype=np.uint8),
    )
    _write_json_new(output_dir / "loss_trace.json", loss_trace)
    receipt_schema, receipt_status, claim_boundary = _method_receipt_semantics(method)
    receipt = {
        "schema": receipt_schema,
        "status": receipt_status,
        "method": method,
        "method_claim_boundary": claim_boundary,
        "row_id": str(args.row_id),
        "receiver": manifest["receiver"],
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "new_class_count": int(manifest["new_class_count"]),
        "old_class_count": old_count,
        "registered_class_count": len(class_handles),
        "backbone": "ADV3B02",
        "base_backbone_fully_trained": True,
        "incremental_backbone_frozen": all(
            bool(item.get("backbone_frozen_incremental", False))
            for item in resources
        ),
        "backbone_frozen": all(
            bool(item.get("backbone_frozen_incremental", False))
            for item in resources
        ),
        "candidate_resources_by_scenario": resources,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "total_enrollment_seconds": time.perf_counter() - training_started,
        "comparison_method_protocol_scope": (
            "stage2_main_method_protocol_exempt_new_class_leo_required"
        ),
        "new_class_support_channel_policy": "leo_satellite_required",
        "new_class_query_channel_policy": "leo_satellite_required",
        "base_source_reference_access_allowed": True,
        "base_state_receipt": base_state["receipt"],
        "fixed_received_iq_reused_across_epochs": True,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "query_rows_used_for_training": 0,
        "query_labels_available_to_predictor": False,
        "dense_query_graph_used": False,
        "query_members_opened_before_model_lock": False,
        "runtime_open_role_ledger": opened_roles,
        "package_preopen_audit": package_audit,
        "enrolled_model_state_sha256": model_state_sha256,
        "enrollment_receipt_sha256": sha256_file(enrollment_receipt_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "prediction_immutable_state": publication["immutable_state"],
    }
    if is_mrior_preadapt:
        receipt.update(
            _mrior_preadapt_receipt_fields(mrior_preadapt_lineage_by_scenario)
        )
    receipt_path = output_dir / "predictor_receipt.json"
    _write_json_new(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "predictor_receipt": str(receipt_path),
        "predictor_receipt_sha256": sha256_file(receipt_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--old-class-count", type=int, default=6)
    parser.add_argument("--expected-total-capacity", type=int, required=True)
    parser.add_argument("--k-shot", type=int, choices=(1, 5, 10, 20), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mrior-preadapt-bindings", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(predict(parse_args()), ensure_ascii=False, sort_keys=True))
