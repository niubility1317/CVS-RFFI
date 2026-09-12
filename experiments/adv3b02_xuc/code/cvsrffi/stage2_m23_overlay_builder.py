"""Build an M2.3 overlay from a validated v2 cache and sealed received IQ."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    CenterLowRankPrototypeComponent,
    MANIFEST_NAME,
    NPZ_NAME,
    validate_center_lowrank_component,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_ablation_feature_builder import (
    _load_package,
    _registered_handles,
)
from cvsrffi.stage2_ablation_feature_cache import load_feature_cache
from cvsrffi.stage2_m23_overlay_cache import publish_m23_overlay_cache
from cvsrffi.stage2_m23_rfguard import build_rfguard_blocks, extract_rf_lite_quality


class M23OverlayBuilderError(ValueError):
    """Raised when sealed rows cannot be joined to their validated cache."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_component_registry_binding(
    component: CenterLowRankPrototypeComponent,
    base_cache: Mapping[str, Any],
) -> None:
    """Check the consumed class registry without reviving historical seal gates."""

    if tuple(str(value) for value in component.class_registry) != tuple(
        str(value) for value in base_cache["old_classes"]
    ):
        raise M23OverlayBuilderError("Phase1 component/base-cache binding drift")


def compose_m23_overlay_payloads(
    base_cache: Mapping[str, Any],
    predictor_support: Mapping[str, Mapping[str, Any]],
    predictor_query: Mapping[str, Mapping[str, Any]],
    *,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    k_shot: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Join same-row received IQ to frozen legacy features without truth access."""

    scenarios = tuple(FORMAL_LEO_WEAK_SCENARIOS)
    if (
        set(base_cache.get("scenario_payloads", {})) != set(scenarios)
        or set(predictor_support) != set(scenarios)
        or set(predictor_query) != set(scenarios)
    ):
        raise M23OverlayBuilderError("formal scenario registry drift")
    old_registry = tuple(str(value) for value in old_classes)
    new_registry = tuple(str(value) for value in new_classes)
    all_registry = old_registry + new_registry
    if (
        len(old_registry) != 6
        or len(new_registry) not in {5, 10, 20}
        or len(set(all_registry)) != len(all_registry)
        or int(k_shot) not in {1, 2, 5, 10}
    ):
        raise M23OverlayBuilderError("class/K-shot registry drift")

    result: dict[str, dict[str, np.ndarray]] = {}
    for scenario in scenarios:
        cached = base_cache["scenario_payloads"][scenario]
        support = predictor_support[scenario]
        query = predictor_query[scenario]
        try:
            ranks = np.asarray(support["support_pool_rank_within_class"], dtype=np.int64)
            indices = np.asarray(support["support_pool_class_indices"], dtype=np.int64)
            pool_iq = np.asarray(support["support_pool_leo_weak_iq"], dtype=np.float32)
            query_iq = np.asarray(query["query_leo_weak_iq"], dtype=np.float32)
            package_tokens = np.asarray(query["query_tokens"]).astype(str)
            cached_tokens = np.asarray(cached["query_tokens"]).astype(str)
        except (KeyError, TypeError, ValueError) as exc:
            raise M23OverlayBuilderError(f"{scenario} predictor payload schema drift") from exc
        mask = ranks < int(k_shot)
        if (
            ranks.ndim != 1
            or ranks.shape != indices.shape
            or len(pool_iq) != len(ranks)
            or int(np.sum(mask)) != int(k_shot) * len(all_registry)
            or np.any(indices[mask] < 0)
            or np.any(indices[mask] >= len(all_registry))
        ):
            raise M23OverlayBuilderError(f"{scenario} support assignment drift")
        selected_indices = indices[mask]
        selected_iq = pool_iq[mask]
        expected_labels = np.asarray(all_registry)[selected_indices]
        old_mask = selected_indices < len(old_registry)
        new_mask = ~old_mask
        cached_old_labels = np.asarray(cached["old_support_labels"]).astype(str)
        cached_new_labels = np.asarray(cached["new_support_labels"]).astype(str)
        if (
            not np.array_equal(cached_old_labels, expected_labels[old_mask])
            or not np.array_equal(cached_new_labels, expected_labels[new_mask])
        ):
            raise M23OverlayBuilderError(f"{scenario} support label/order binding drift")
        if (
            package_tokens.shape != cached_tokens.shape
            or not np.array_equal(package_tokens, cached_tokens)
            or len(query_iq) != len(cached_tokens)
        ):
            raise M23OverlayBuilderError(f"{scenario} query token binding drift")

        old_lite, old_quality = extract_rf_lite_quality(selected_iq[old_mask])
        new_lite, new_quality = extract_rf_lite_quality(selected_iq[new_mask])
        query_lite, _unused_query_quality = extract_rf_lite_quality(query_iq)
        try:
            old_blocks = build_rfguard_blocks(cached["old_support_features"], old_lite)
            new_blocks = build_rfguard_blocks(cached["new_support_features"], new_lite)
            query_blocks = build_rfguard_blocks(cached["query_features"], query_lite)
        except (KeyError, ValueError) as exc:
            raise M23OverlayBuilderError(f"{scenario} feature-cache join drift") from exc
        result[scenario] = {
            "old_support_blocks": old_blocks,
            "old_support_labels": cached_old_labels,
            "old_support_quality": old_quality,
            "new_support_blocks": new_blocks,
            "new_support_labels": cached_new_labels,
            "new_support_quality": new_quality,
            "query_blocks": query_blocks,
            "query_tokens": cached_tokens,
        }
    return result


def _component_arrays(component: Any) -> dict[str, np.ndarray]:
    names = (
        "core_q",
        "core_scale",
        "residual_basis_q",
        "residual_basis_scale",
        "residual_coeff_q",
        "residual_coeff_scale",
        "domain_registry",
        "residual_domain_registry",
        "class_registry",
        "center_domain_handle",
    )
    result: dict[str, np.ndarray] = {}
    for name in names:
        if not hasattr(component, name):
            raise M23OverlayBuilderError("formal Phase1 component is incomplete")
        result[name] = np.asarray(getattr(component, name))
    return result


def _load_bound_component(
    component_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> CenterLowRankPrototypeComponent:
    root = Path(component_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    if _sha256_file(manifest_path) != str(expected_manifest_sha256).lower():
        raise M23OverlayBuilderError("Phase1 component manifest SHA256 drift")
    try:
        manifest = validate_center_lowrank_component(
            root,
            expected_checkpoint_sha256=str(expected_checkpoint_sha256).lower(),
        )
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest != loaded_manifest:
            raise M23OverlayBuilderError("Phase1 component manifest readback drift")
        with np.load(root / NPZ_NAME, allow_pickle=False) as arrays:
            component = CenterLowRankPrototypeComponent(
                core_q=np.array(arrays["core_q"], copy=True),
                core_scale=np.array(arrays["core_scale"], copy=True),
                residual_basis_q=np.array(arrays["residual_basis_q"], copy=True),
                residual_basis_scale=np.array(
                    arrays["residual_basis_scale"], copy=True
                ),
                residual_coeff_q=np.array(arrays["residual_coeff_q"], copy=True),
                residual_coeff_scale=np.array(
                    arrays["residual_coeff_scale"], copy=True
                ),
                radius_q=np.array(arrays["radius_q"], copy=True),
                radius_scale=np.array(arrays["radius_scale"], copy=True),
                domain_registry=tuple(arrays["domain_registry"].astype(str).tolist()),
                residual_domain_registry=tuple(
                    arrays["residual_domain_registry"].astype(str).tolist()
                ),
                class_registry=tuple(arrays["class_registry"].astype(str).tolist()),
                center_domain_handle=str(
                    np.asarray(arrays["center_domain_handle"]).item()
                ),
                manifest=manifest,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, M23OverlayBuilderError):
            raise
        raise M23OverlayBuilderError("Phase1 aggregate component validation failed") from exc
    return component


def build_m23_overlay_from_sealed_inputs(
    *,
    base_feature_cache_payload: str | Path,
    base_feature_cache_manifest: str | Path,
    base_feature_cache_payload_sha256: str,
    base_feature_cache_manifest_sha256: str,
    predictor_package_root: str | Path,
    predictor_seal_path: str | Path,
    predictor_seal_sha256: str,
    phase1_component_dir: str | Path,
    expected_phase1_component_manifest_sha256: str,
    overlay_payload_path: str | Path,
    overlay_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify existing artifacts, derive RF state, and publish one overlay."""

    base = load_feature_cache(
        base_feature_cache_payload,
        base_feature_cache_manifest,
        expected_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        expected_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
    )
    base_manifest = base["manifest"]
    if base_manifest.get("stage_scope") != "stage2c":
        raise M23OverlayBuilderError("M2.3 overlay requires a Stage2-C base cache")
    support, query, package_manifest, _package_audit = _load_package(
        predictor_package_root,
        predictor_seal_path,
        str(predictor_seal_sha256).lower(),
    )
    registered = _registered_handles(package_manifest)
    expected_registry = base["old_classes"] + base["new_classes"]
    if (
        package_manifest.get("stage") != "stage2c"
        or registered != expected_registry
        or str(package_manifest.get("receiver")) != str(base_manifest["receiver"])
        or str(package_manifest.get("package_root_sha256", "")).lower()
        != str(base_manifest["package_root_sha256"]).lower()
        or str(predictor_seal_sha256).lower()
        != str(base_manifest["package_seal_sha256"]).lower()
        or _sha256_file(predictor_seal_path) != str(predictor_seal_sha256).lower()
    ):
        raise M23OverlayBuilderError("predictor package/base-cache binding drift")

    bound_manifest_sha = str(
        base_manifest.get("ground_audit", {}).get(
            "ground_component_manifest_sha256", ""
        )
    ).lower()
    if bound_manifest_sha != str(expected_phase1_component_manifest_sha256).lower():
        raise M23OverlayBuilderError("base cache/component manifest binding drift")
    component = _load_bound_component(
        phase1_component_dir,
        expected_manifest_sha256=expected_phase1_component_manifest_sha256,
        expected_checkpoint_sha256=str(base_manifest["phase1_bundle_sha256"]),
    )
    _validate_component_registry_binding(component, base)

    payloads = compose_m23_overlay_payloads(
        base,
        support,
        query,
        old_classes=base["old_classes"],
        new_classes=base["new_classes"],
        k_shot=int(base_manifest["k_shot"]),
    )
    published = publish_m23_overlay_cache(
        overlay_payload_path,
        overlay_manifest_path,
        receiver=str(base_manifest["receiver"]),
        k_shot=int(base_manifest["k_shot"]),
        method_seed=int(base_manifest["method_seed"]),
        support_seed=int(base_manifest["support_seed"]),
        query_seed=int(base_manifest["query_seed"]),
        new_class_draw_seed=int(base_manifest["new_class_draw_seed"]),
        phase2_data_status=str(base_manifest["phase2_data_status"]),
        capsule_id=str(base_manifest["capsule_id"]),
        split_id=str(base_manifest["split_id"]),
        base_feature_cache_payload_sha256=str(base_feature_cache_payload_sha256).lower(),
        base_feature_cache_manifest_sha256=str(base_feature_cache_manifest_sha256).lower(),
        predictor_package_root_sha256=str(base_manifest["package_root_sha256"]).lower(),
        predictor_package_seal_sha256=str(predictor_seal_sha256).lower(),
        phase1_bundle_sha256=str(base_manifest["phase1_bundle_sha256"]).lower(),
        phase1_component_manifest_sha256=str(
            expected_phase1_component_manifest_sha256
        ).lower(),
        old_classes=base["old_classes"],
        new_classes=base["new_classes"],
        scenario_payloads=payloads,
        ground_component=_component_arrays(component),
    )
    return {
        **published,
        "overlay_payload_path": str(Path(overlay_payload_path).absolute()),
        "overlay_manifest_path": str(Path(overlay_manifest_path).absolute()),
        "query_truth_opened": False,
        "raw_dataset_opened": False,
        "base_data_revalidated": False,
        "historical_base_cache_component_binding_reused": True,
        "new_signature_or_authority_gate_added": False,
    }


__all__ = [
    "M23OverlayBuilderError",
    "build_m23_overlay_from_sealed_inputs",
    "compose_m23_overlay_payloads",
]
