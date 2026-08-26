"""Immutable Phase1 deployment binding for SF-TAPFT R0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .phase1_adv3b02_deployment_bundle import (
    VerifiedADV3B02DeploymentBundle,
    load_formal_adv3b02_deployment_bundle,
)


_PHASE1_BUNDLE_KEYS = frozenset(
    {
        "package_root",
        "detached_seal_path",
        "expected_detached_seal_sha256",
        "signature_envelope_path",
        "expected_signature_envelope_sha256",
        "expected_checkpoint_lineage_sha256",
        "expected_runtime_sha256",
        "expected_component_pre_sign_content_root_sha256",
        "expected_class_handle_binding_sha256",
        "expected_parity_receipt_sha256",
        "expected_generation_lock_sha256",
        "expected_method_lock_sha256",
        "expected_generation_config_sha256",
        "expected_generation_code_sha256",
        "expected_outer_content_root_sha256",
    }
)


@dataclass(frozen=True)
class SFTAPFTPhase1Binding:
    outer_content_root_sha256: str
    checkpoint_lineage_sha256: str
    runtime_sha256: str
    class_handle_binding_sha256: str
    class_handles: tuple[str, ...]
    component_pre_sign_content_root_sha256: str


def _formal_loader_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    phase1_bundle = config.get("phase1_bundle") if isinstance(config, Mapping) else None
    if not isinstance(phase1_bundle, Mapping) or set(phase1_bundle) != _PHASE1_BUNDLE_KEYS:
        raise ValueError("phase1_bundle must contain the complete formal deployment mapping")
    return dict(phase1_bundle)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"training checkpoint is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_class_handles(verified: VerifiedADV3B02DeploymentBundle) -> tuple[str, ...]:
    rows = verified.class_binding.get("class_id_to_handle")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Phase1 class registry is invalid")
    handles: list[str] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"class_index", "class_handle"}
            or row["class_index"] != index
            or not isinstance(row["class_handle"], str)
            or not row["class_handle"]
        ):
            raise ValueError("Phase1 class registry must use contiguous ordered class_index values")
        handles.append(row["class_handle"])
    if len(handles) != len(set(handles)):
        raise ValueError("Phase1 class registry handles must be unique")
    return tuple(handles)


def load_sf_tapft_phase1_binding(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    formal_loader: Callable[..., VerifiedADV3B02DeploymentBundle] = load_formal_adv3b02_deployment_bundle,
) -> SFTAPFTPhase1Binding:
    """Load only R0's immutable Phase1 identifiers and ordered class handles."""

    verified = formal_loader(**_formal_loader_kwargs(config))
    context = verified.formal_phase2_context
    if context.get("formal_phase2_eligible") is not True:
        raise ValueError("formal Phase1 bundle is not Phase2 eligible")
    expected_checkpoint_sha = str(context["checkpoint_lineage_sha256"]).lower()
    actual_checkpoint_sha = _sha256_file(Path(checkpoint_path))
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("SF-TAPFT checkpoint lineage does not match Phase1 bundle")
    handles = _ordered_class_handles(verified)
    return SFTAPFTPhase1Binding(
        outer_content_root_sha256=str(context["outer_content_root_sha256"]),
        checkpoint_lineage_sha256=expected_checkpoint_sha,
        runtime_sha256=str(context["runtime_sha256"]),
        class_handle_binding_sha256=str(context["class_handle_binding_sha256"]),
        class_handles=handles,
        component_pre_sign_content_root_sha256=str(
            context["component_pre_sign_content_root_sha256"]
        ),
    )


__all__ = ["SFTAPFTPhase1Binding", "load_sf_tapft_phase1_binding"]
