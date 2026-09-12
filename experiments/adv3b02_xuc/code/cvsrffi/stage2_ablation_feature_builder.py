"""Build stage-scoped truth-free caches from sealed production packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.phase2_prototypes import verify_endpoint_accept_v1_manifest
from cvsrffi.phase1_adv3b02_deployment_bundle import (
    load_formal_adv3b02_deployment_bundle,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    MANIFEST_NAME as PHASE1_COMPONENT_MANIFEST_NAME,
)
from cvsrffi.stage2_ablation_feature_cache import publish_feature_cache
from cvsrffi.stage2_d89_v2_radius_cauchy_center import (
    radius_reliability_ground_spectrum,
)
from cvsrffi.stage2_diag_cosine_exploration import (
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_stage2_predictor_bundle,
)


class Stage2AblationFeatureBuilderError(ValueError):
    """Raised when sealed production inputs cannot produce reusable caches."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_package(
    root: str | Path,
    seal: str | Path,
    seal_sha256: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    support, query, manifest, audit = (
        load_verified_stage2_predictor_bundle(
            root,
            detached_seal_path=seal,
            expected_seal_sha256=str(seal_sha256).lower(),
        )
    )
    return (
        {key: dict(value) for key, value in support.items()},
        {key: dict(value) for key, value in query.items()},
        dict(manifest),
        dict(audit),
    )


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _deployment_prototypes(
    path: str | Path,
    *,
    expected_old_class_count: int,
) -> np.ndarray:
    package = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(package, Mapping):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype package must be a mapping"
        )
    prototypes = package.get("prototypes")
    if not torch.is_tensor(prototypes):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype package lacks prototypes"
        )
    identity = np.asarray(
        prototypes.detach().float().cpu().tolist(),
        dtype=np.float32,
    )
    if (
        identity.shape != (expected_old_class_count, 160)
        or not np.isfinite(identity).all()
    ):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 deployment prototype shape drift"
        )
    norms = np.linalg.norm(identity, axis=1, keepdims=True)
    if bool(np.any(norms <= 1e-8)):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 deployment prototype contains a zero row"
        )
    identity = identity / norms
    padded = np.concatenate(
        [
            identity,
            np.zeros(
                (expected_old_class_count, 128),
                dtype=np.float32,
            ),
        ],
        axis=1,
    )
    return padded / np.maximum(
        np.linalg.norm(padded, axis=1, keepdims=True),
        1e-12,
    )


def _verified_deployment_prototypes(
    pt_path: str | Path,
    json_path: str | Path,
    *,
    expected_pt_sha256: str,
    expected_json_sha256: str,
    expected_phase1_bundle_sha256: str,
    expected_old_class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if _sha256_file(pt_path) != str(expected_pt_sha256).lower():
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype PT detached hash mismatch"
        )
    if _sha256_file(json_path) != str(expected_json_sha256).lower():
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype JSON detached hash mismatch"
        )
    package = torch.load(
        Path(pt_path),
        map_location="cpu",
        weights_only=False,
    )
    try:
        sidecar = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype JSON is unreadable"
        ) from exc
    if not isinstance(package, Mapping) or not isinstance(sidecar, Mapping):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype PT/JSON roots must be mappings"
        )
    if _jsonable(package) != sidecar:
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype PT/JSON content drift"
        )
    try:
        endpoint = verify_endpoint_accept_v1_manifest(package)
    except (TypeError, ValueError) as exc:
        raise Stage2AblationFeatureBuilderError(
            "Phase1 endpoint manifest verification failed"
        ) from exc
    identity = endpoint.get("inference_identity", {})
    if (
        not isinstance(identity, Mapping)
        or int(identity.get("known_class_count", -1))
        != expected_old_class_count
        or list(identity.get("logit_class_order", ()))
        != list(range(expected_old_class_count))
        or str(identity.get("source_checkpoint_sha256", "")).lower()
        != str(expected_phase1_bundle_sha256).lower()
        or not str(identity.get("run_id", "")).strip()
        or not str(identity.get("candidate_id", "")).strip()
    ):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 prototype inference identity drift"
        )
    return (
        _deployment_prototypes(
            pt_path,
            expected_old_class_count=expected_old_class_count,
        ),
        dict(identity),
    )


def _registered_handles(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    rows = manifest.get("registered_classes")
    if not isinstance(rows, list):
        raise Stage2AblationFeatureBuilderError(
            "registered class manifest is missing"
        )
    handles: list[str] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or int(row.get("class_index", -1)) != index
            or not str(row.get("class_handle", "")).strip()
        ):
            raise Stage2AblationFeatureBuilderError(
                "registered class order is not explicit"
            )
        handles.append(str(row["class_handle"]))
    if len(handles) != len(set(handles)):
        raise Stage2AblationFeatureBuilderError(
            "registered class handles are not unique"
        )
    return tuple(handles)


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.startswith("cuda:") or not torch.cuda.is_available():
        raise Stage2AblationFeatureBuilderError(
            f"requested device is unavailable: {value}"
        )
    result = torch.device(value)
    if int(value.split(":", 1)[1]) >= torch.cuda.device_count():
        raise Stage2AblationFeatureBuilderError(
            f"requested device is unavailable: {value}"
        )
    return result


_DEPLOYMENT_BINDING_KEYS = {
    "schema",
    "package_root",
    "detached_seal_path",
    "detached_seal_sha256",
    "signature_envelope_path",
    "signature_envelope_sha256",
    "checkpoint_lineage_sha256",
    "runtime_sha256",
    "component_pre_sign_content_root_sha256",
    "class_handle_binding_sha256",
    "parity_receipt_sha256",
    "generation_lock_sha256",
    "method_lock_sha256",
    "generation_config_sha256",
    "generation_code_sha256",
    "outer_content_root_sha256",
    "phase1_completion_receipt_path",
    "phase1_completion_receipt_sha256",
    "generation_config_path",
    "prototype_pt_path",
    "prototype_pt_sha256",
    "prototype_json_path",
    "prototype_json_sha256",
}
_GENERATION_CONFIG_KEYS = {
    "schema",
    "row_key",
    "run_id",
    "checkpoint_lineage_sha256",
    "completion_receipt_sha256",
    "original_prototype_pt_sha256",
    "original_prototype_json_sha256",
    "normalized_prototype_pt_sha256",
    "normalized_prototype_json_sha256",
    "prototype_normalization_status",
    "class_handle_binding_sha256",
    "component_export",
}


def _load_formal_runtime(
    binding_path: str | Path,
    *,
    expected_phase1_bundle_sha256: str,
    phase1_prototype_path: str | Path,
    phase1_prototype_manifest_path: str | Path,
    expected_phase1_prototype_sha256: str,
    expected_phase1_prototype_manifest_sha256: str,
) -> tuple[
    Any,
    tuple[str, ...],
    str,
    dict[str, Any],
    Any,
    Path,
]:
    try:
        binding = json.loads(
            Path(binding_path).read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2AblationFeatureBuilderError(
            "formal Phase1 deployment binding is unreadable"
        ) from exc
    if (
        not isinstance(binding, Mapping)
        or set(binding) != _DEPLOYMENT_BINDING_KEYS
        or binding.get("schema")
        != "cvs.full_ablation.phase1.deployment_binding.v1"
        or str(binding.get("checkpoint_lineage_sha256", "")).lower()
        != str(expected_phase1_bundle_sha256).lower()
    ):
        raise Stage2AblationFeatureBuilderError(
            "formal Phase1 deployment binding drift"
        )
    locked_paths = {
        "prototype_pt_path": Path(binding["prototype_pt_path"]).resolve(),
        "prototype_json_path": Path(binding["prototype_json_path"]).resolve(),
        "generation_config_path": Path(
            binding["generation_config_path"]
        ).resolve(),
        "phase1_completion_receipt_path": Path(
            binding["phase1_completion_receipt_path"]
        ).resolve(),
    }
    if (
        locked_paths["prototype_pt_path"]
        != Path(phase1_prototype_path).resolve()
        or locked_paths["prototype_json_path"]
        != Path(phase1_prototype_manifest_path).resolve()
        or _sha256_file(locked_paths["prototype_pt_path"])
        != str(expected_phase1_prototype_sha256).lower()
        or _sha256_file(locked_paths["prototype_json_path"])
        != str(expected_phase1_prototype_manifest_sha256).lower()
        or binding["prototype_pt_sha256"]
        != str(expected_phase1_prototype_sha256).lower()
        or binding["prototype_json_sha256"]
        != str(expected_phase1_prototype_manifest_sha256).lower()
        or _sha256_file(locked_paths["generation_config_path"])
        != binding["generation_config_sha256"]
        or _sha256_file(locked_paths["phase1_completion_receipt_path"])
        != binding["phase1_completion_receipt_sha256"]
    ):
        raise Stage2AblationFeatureBuilderError(
            "signed Phase1 prototype lock path/hash drift"
        )
    try:
        generation_config = json.loads(
            locked_paths["generation_config_path"].read_text(
                encoding="utf-8-sig"
            )
        )
        completion = json.loads(
            locked_paths["phase1_completion_receipt_path"].read_text(
                encoding="utf-8-sig"
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2AblationFeatureBuilderError(
            "signed Phase1 prototype lock is unreadable"
        ) from exc
    if (
        not isinstance(generation_config, Mapping)
        or set(generation_config) != _GENERATION_CONFIG_KEYS
        or generation_config.get("schema")
        != "cvs.full_ablation.phase1.deployment_generation_config.v1"
        or generation_config.get("checkpoint_lineage_sha256")
        != binding["checkpoint_lineage_sha256"]
        or generation_config.get("completion_receipt_sha256")
        != binding["phase1_completion_receipt_sha256"]
        or generation_config.get("normalized_prototype_pt_sha256")
        != binding["prototype_pt_sha256"]
        or generation_config.get("normalized_prototype_json_sha256")
        != binding["prototype_json_sha256"]
        or generation_config.get("class_handle_binding_sha256")
        != binding["class_handle_binding_sha256"]
        or not isinstance(completion, Mapping)
        or completion.get("phase1_training_complete") is not True
        or completion.get("terminal_status") != "COMPLETE"
        or int(completion.get("exit_code", -1)) != 0
        or str(completion.get("selected_checkpoint_sha256", "")).lower()
        != binding["checkpoint_lineage_sha256"]
    ):
        raise Stage2AblationFeatureBuilderError(
            "signed Phase1 prototype lock content drift"
        )
    verified = load_formal_adv3b02_deployment_bundle(
        binding["package_root"],
        detached_seal_path=binding["detached_seal_path"],
        expected_detached_seal_sha256=(
            binding["detached_seal_sha256"]
        ),
        signature_envelope_path=binding["signature_envelope_path"],
        expected_signature_envelope_sha256=(
            binding["signature_envelope_sha256"]
        ),
        expected_checkpoint_lineage_sha256=(
            binding["checkpoint_lineage_sha256"]
        ),
        expected_runtime_sha256=binding["runtime_sha256"],
        expected_component_pre_sign_content_root_sha256=(
            binding["component_pre_sign_content_root_sha256"]
        ),
        expected_class_handle_binding_sha256=(
            binding["class_handle_binding_sha256"]
        ),
        expected_parity_receipt_sha256=(
            binding["parity_receipt_sha256"]
        ),
        expected_generation_lock_sha256=(
            binding["generation_lock_sha256"]
        ),
        expected_method_lock_sha256=binding["method_lock_sha256"],
        expected_generation_config_sha256=(
            binding["generation_config_sha256"]
        ),
        expected_generation_code_sha256=(
            binding["generation_code_sha256"]
        ),
        expected_outer_content_root_sha256=(
            binding["outer_content_root_sha256"]
        ),
    )
    formal = dict(verified.formal_phase2_context)
    if (
        formal.get("formal_phase2_eligible") is not True
        or formal.get("runtime_checkpoint_parity_verified") is not True
        or formal.get("outer_signature_verified") is not True
    ):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 deployment bundle lacks formal authority"
        )
    rows = verified.class_binding.get("class_id_to_handle")
    if not isinstance(rows, list):
        raise Stage2AblationFeatureBuilderError(
            "Phase1 deployment class binding is missing"
        )
    handles = tuple(str(row.get("class_handle", "")) for row in rows)
    return (
        verified.runtime,
        handles,
        str(binding["runtime_sha256"]).lower(),
        formal,
        verified.component,
        Path(binding["package_root"]).resolve() / "component",
    )


def _ground_spectrum_from_formal_v2_component(
    component: Any,
    *,
    component_dir: str | Path,
    sealed_component_dir: str | Path,
    expected_manifest_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive the D81 spectrum from the outer-sealed Phase1 v2 component."""

    supplied_root = Path(component_dir).resolve()
    sealed_root = Path(sealed_component_dir).resolve()
    manifest_path = supplied_root / PHASE1_COMPONENT_MANIFEST_NAME
    if supplied_root != sealed_root:
        raise Stage2AblationFeatureBuilderError(
            "ground component is not the outer-sealed Phase1 component"
        )
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha256_file(manifest_path)
        != str(expected_manifest_sha256).lower()
    ):
        raise Stage2AblationFeatureBuilderError(
            "outer-sealed ground component manifest hash drift"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2AblationFeatureBuilderError(
            "outer-sealed ground component manifest is unreadable"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or _jsonable(component.manifest) != manifest
        or manifest.get("schema")
        != "int8_domain_class_center_lowrank_residual_radius_v2"
        or int(manifest.get("feature_dim", -1)) != 160
    ):
        raise Stage2AblationFeatureBuilderError(
            "outer-sealed ground component content drift"
        )

    prototypes = np.stack(
        [
            component.reconstruct_domain(domain)
            for domain in component.domain_registry
        ]
    )
    radius = np.stack(
        [
            component.radius_for_domain(domain)
            for domain in component.domain_registry
        ]
    )
    resource = component.resource_audit()
    reconstruction_rmse = float(resource["reconstruction_rmse"])
    basis, weights, spectrum_audit = (
        radius_reliability_ground_spectrum(
            prototypes,
            radius,
            reconstruction_rmse,
        )
    )
    statistics_macs = int(
        resource["all_residual_domain_enrollment_reconstruction_macs"]
        + radius.size * (10 * 160 + 8)
        + 160 * 160 * 8
    )
    ground_audit = dict(spectrum_audit)
    ground_audit.update(
        {
            **{
                f"d81_{key}": value
                for key, value in spectrum_audit.items()
            },
            "ground_component_input_count": int(radius.size),
            "ground_int8_component_logical_state_bytes": int(
                resource["logical_deployment_state_bytes"]
            ),
            "ground_covariance_statistics_mac_upper_bound": (
                statistics_macs
            ),
            "ground_statistic_semantics": (
                "v2_cell_radius_reliability_ground_spectrum_for_"
                "d81_cauchy_center"
            ),
            "ground_bundle_contains_sample_radius": False,
            "ground_bundle_contains_aggregated_p90_radius": True,
            "ground_bundle_contains_sample_count": False,
            "ground_aggregated_center_access": True,
            "ground_aggregated_p90_radius_access": True,
            "ground_sample_radius_access": False,
            "ground_sample_feature_access": False,
            "ground_target_identity_mapping_access": False,
            "ground_class_score_access": False,
            "ground_component_update_access": False,
            "dense_ground_bank_persisted": False,
            "quantization_noise_floor_policy": (
                "manifest_reconstruction_rmse_squared"
            ),
            "ground_component_state": str(
                component.manifest["component_state"]
            ),
            "ground_component_manifest_sha256": str(
                expected_manifest_sha256
            ).lower(),
            "ground_component_outer_joint_seal_verified": True,
            "d81_basis_transient_fp64_bytes": int(
                basis.nbytes
                + weights.nbytes
                + prototypes.nbytes
                + radius.nbytes
            ),
        }
    )
    return basis, weights, ground_audit


def _support_features(
    payload: Mapping[str, np.ndarray],
    *,
    model: Any,
    runtime_device: torch.device,
    class_handles: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ranks = np.asarray(
        payload["support_pool_rank_within_class"], dtype=np.int64
    )
    class_indices = np.asarray(
        payload["support_pool_class_indices"], dtype=np.int64
    )
    mask = ranks < int(k_shot)
    if (
        ranks.shape != class_indices.shape
        or ranks.ndim != 1
        or int(np.sum(mask)) != int(k_shot) * len(class_handles)
        or int(class_indices[mask].min()) != 0
        or int(class_indices[mask].max()) != len(class_handles) - 1
    ):
        raise Stage2AblationFeatureBuilderError(
            "support assignment drift"
        )
    iq = np.asarray(
        payload["support_pool_leo_weak_iq"], dtype=np.float32
    )[mask]
    zid = forward_zid160(
        model, iq, device=runtime_device, batch_size=64
    )
    features = registered_feature(iq, zid)
    labels = np.asarray(class_handles, dtype=str)[class_indices[mask]]
    return features, labels, class_indices[mask], int(len(iq))


def _query_features(
    payload: Mapping[str, np.ndarray],
    *,
    model: Any,
    runtime_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    iq = np.asarray(payload["query_leo_weak_iq"], dtype=np.float32)
    tokens = np.asarray(payload["query_tokens"]).astype(str)
    if (
        iq.ndim != 3
        or tokens.ndim != 1
        or len(iq) != len(tokens)
        or len(iq) == 0
    ):
        raise Stage2AblationFeatureBuilderError("query payload drift")
    zid = forward_zid160(
        model, iq, device=runtime_device, batch_size=1
    )
    return registered_feature(iq, zid), tokens, int(len(iq))


def build_feature_cache_from_sealed_row_pair(
    *,
    before_package_root: str | Path,
    before_seal_path: str | Path,
    before_seal_sha256: str,
    after_package_root: str | Path,
    after_seal_path: str | Path,
    after_seal_sha256: str,
    phase1_deployment_binding_path: str | Path,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    phase1_prototype_path: str | Path,
    phase1_prototype_manifest_path: str | Path,
    expected_phase1_prototype_sha256: str,
    expected_phase1_prototype_manifest_sha256: str,
    expected_phase1_bundle_sha256: str,
    cache_output_root: str | Path,
    phase2_data_status: str,
    capsule_id: str,
    split_id: str,
    k_shot: int,
    method_seed: int,
    support_seed: int,
    query_seed: int,
    new_class_draw_seed: int,
    device: str,
) -> dict[str, Any]:
    """Extract features once per package without opening truth or datasets."""

    before_support, before_query, before_manifest, _ = _load_package(
        before_package_root,
        before_seal_path,
        before_seal_sha256,
    )
    after_support, after_query, after_manifest, _ = _load_package(
        after_package_root,
        after_seal_path,
        after_seal_sha256,
    )
    if (
        before_manifest.get("stage") != "stage2b"
        or int(before_manifest.get("new_class_count", -1)) != 0
        or after_manifest.get("stage") != "stage2c"
        or str(before_manifest.get("receiver", ""))
        != str(after_manifest.get("receiver", ""))
        or str(before_manifest.get("candidate_lock_sha256", ""))
        != str(after_manifest.get("candidate_lock_sha256", ""))
    ):
        raise Stage2AblationFeatureBuilderError(
            "production before/after package role or lock drift"
        )
    old_classes = _registered_handles(before_manifest)
    all_classes = _registered_handles(after_manifest)
    new_classes = all_classes[len(old_classes) :]
    if (
        len(old_classes) != 6
        or all_classes[: len(old_classes)] != old_classes
        or len(new_classes) not in {5, 10, 20}
        or int(after_manifest.get("new_class_count", -1))
        != len(new_classes)
        or int(k_shot) not in {1, 2, 5, 10}
        or int(k_shot)
        > min(
            int(before_manifest.get("support_pool_max_k", 0)),
            int(after_manifest.get("support_pool_max_k", 0)),
        )
    ):
        raise Stage2AblationFeatureBuilderError(
            "production class registry or K-shot drift"
        )

    runtime_device = _device(device)
    (
        model,
        deployment_handles,
        runtime_sha256,
        formal_context,
        formal_component,
        sealed_component_dir,
    ) = (
        _load_formal_runtime(
            phase1_deployment_binding_path,
            expected_phase1_bundle_sha256=(
                expected_phase1_bundle_sha256
            ),
            phase1_prototype_path=phase1_prototype_path,
            phase1_prototype_manifest_path=(
                phase1_prototype_manifest_path
            ),
            expected_phase1_prototype_sha256=(
                expected_phase1_prototype_sha256
            ),
            expected_phase1_prototype_manifest_sha256=(
                expected_phase1_prototype_manifest_sha256
            ),
        )
    )
    model = model.to(runtime_device)
    model.eval()
    if deployment_handles != old_classes:
        raise Stage2AblationFeatureBuilderError(
            "Phase1 deployment and predictor old-class order drift"
        )

    ground_basis, spectral_weights, ground_audit = (
        _ground_spectrum_from_formal_v2_component(
            formal_component,
            component_dir=ground_component_dir,
            sealed_component_dir=sealed_component_dir,
            expected_manifest_sha256=ground_manifest_sha256,
        )
    )
    deployment, prototype_identity = _verified_deployment_prototypes(
        phase1_prototype_path,
        phase1_prototype_manifest_path,
        expected_pt_sha256=expected_phase1_prototype_sha256,
        expected_json_sha256=(
            expected_phase1_prototype_manifest_sha256
        ),
        expected_phase1_bundle_sha256=expected_phase1_bundle_sha256,
        expected_old_class_count=len(old_classes),
    )

    payloads_by_scope: dict[
        str, dict[str, dict[str, np.ndarray]]
    ] = {scope: {} for scope in ("stage2a", "stage2b", "stage2c")}
    support_forward_count = 0
    query_forward_count = 0
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_support_x, before_support_y, before_indices, before_count = (
            _support_features(
                before_support[scenario],
                model=model,
                runtime_device=runtime_device,
                class_handles=old_classes,
                k_shot=int(k_shot),
            )
        )
        after_support_x, after_support_y, after_indices, after_count = (
            _support_features(
                after_support[scenario],
                model=model,
                runtime_device=runtime_device,
                class_handles=all_classes,
                k_shot=int(k_shot),
            )
        )
        before_x, before_tokens, before_query_count = _query_features(
            before_query[scenario],
            model=model,
            runtime_device=runtime_device,
        )
        after_x, after_tokens, after_query_count = _query_features(
            after_query[scenario],
            model=model,
            runtime_device=runtime_device,
        )
        old_mask = after_indices < len(old_classes)
        new_mask = ~old_mask
        if set(before_indices.tolist()) != set(range(len(old_classes))):
            raise Stage2AblationFeatureBuilderError(
                "before old-support registry drift"
            )
        payloads_by_scope["stage2a"][scenario] = {
            "query_features": before_x,
            "query_tokens": before_tokens,
        }
        payloads_by_scope["stage2b"][scenario] = {
            "old_support_features": before_support_x,
            "old_support_labels": before_support_y,
            "query_features": before_x,
            "query_tokens": before_tokens,
        }
        payloads_by_scope["stage2c"][scenario] = {
            "old_support_features": after_support_x[old_mask],
            "old_support_labels": after_support_y[old_mask],
            "new_support_features": after_support_x[new_mask],
            "new_support_labels": after_support_y[new_mask],
            "query_features": after_x,
            "query_tokens": after_tokens,
        }
        support_forward_count += int(before_count + after_count)
        query_forward_count += int(
            before_query_count + after_query_count
        )

    output_root = Path(cache_output_root).absolute()
    common = {
        "receiver": str(after_manifest["receiver"]),
        "method_seed": int(method_seed),
        "phase2_data_status": str(phase2_data_status),
        "capsule_id": str(capsule_id),
        "split_id": str(split_id),
        "phase1_bundle_sha256": str(
            expected_phase1_bundle_sha256
        ).lower(),
        "phase1_prototype_sha256": str(
            expected_phase1_prototype_sha256
        ).lower(),
        "old_classes": old_classes,
        "deployment_prototypes": deployment,
    }
    receipts: dict[str, dict[str, Any]] = {}
    for scope in ("stage2a", "stage2b", "stage2c"):
        source_manifest = (
            after_manifest if scope == "stage2c" else before_manifest
        )
        source_seal = (
            after_seal_sha256 if scope == "stage2c" else before_seal_sha256
        )
        scope_root = output_root / scope
        receipts[scope] = publish_feature_cache(
            scope_root / "features.npz",
            scope_root / "features.manifest.json",
            stage_scope=scope,
            query_seed=int(query_seed),
            package_root_sha256=str(
                source_manifest["package_root_sha256"]
            ),
            package_seal_sha256=str(source_seal).lower(),
            support_seed=(0 if scope == "stage2a" else int(support_seed)),
            new_class_draw_seed=(
                int(new_class_draw_seed) if scope == "stage2c" else 0
            ),
            new_classes=new_classes if scope == "stage2c" else (),
            scenario_payloads=payloads_by_scope[scope],
            ground_basis=(ground_basis if scope != "stage2a" else None),
            ground_spectral_weights=(
                spectral_weights if scope != "stage2a" else None
            ),
            ground_audit=(ground_audit if scope != "stage2a" else {}),
            **common,
        )
    return {
        "caches": receipts,
        "cache_output_root": str(output_root),
        "receiver": str(after_manifest["receiver"]),
        "method_seed": int(method_seed),
        "k_shot": int(k_shot),
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "support_backbone_forward_count": support_forward_count,
        "query_backbone_forward_count": query_forward_count,
        "query_truth_opened": False,
        "raw_dataset_opened": False,
        "cross_launch_data_identity_required": False,
        "feature_extraction_reuse_policy": (
            "one_extraction_per_sealed_package_pair_three_stage_caches"
        ),
        "feature_runtime_sha256": runtime_sha256,
        "formal_phase1_deployment_context": formal_context,
        "phase1_prototype_sha256": str(
            expected_phase1_prototype_sha256
        ).lower(),
        "phase1_prototype_manifest_sha256": str(
            expected_phase1_prototype_manifest_sha256
        ).lower(),
        "phase1_prototype_identity": prototype_identity,
    }


__all__ = [
    "Stage2AblationFeatureBuilderError",
    "build_feature_cache_from_sealed_row_pair",
]
