from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from cvsrffi.stage2_ablation_feature_builder import (
    _deployment_prototypes,
    _ground_spectrum_from_formal_v2_component,
    Stage2AblationFeatureBuilderError,
    build_feature_cache_from_sealed_row_pair,
)
from cvsrffi.stage2_d81_query_evaluation import (
    _require_cross_state_lock,
)


def _handle(value: str) -> str:
    return "cls_" + hashlib.sha256(value.encode()).hexdigest()


def test_feature_builder_has_no_truth_or_dataset_input_surface() -> None:
    parameters = inspect.signature(
        build_feature_cache_from_sealed_row_pair
    ).parameters
    assert not any(
        token in name
        for name in parameters
        for token in (
            "truth",
            "dataset",
            "clean",
            "source_sample",
        )
    )


def test_phase1_identity_prototypes_are_padded_to_288(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase2_zid_prototypes.pt"
    prototypes = torch.arange(
        6 * 160, dtype=torch.float32
    ).reshape(6, 160) + 1.0
    torch.save({"prototypes": prototypes}, path)
    result = _deployment_prototypes(
        path,
        expected_old_class_count=6,
    )
    assert result.shape == (6, 288)
    assert np.allclose(
        np.linalg.norm(result, axis=1),
        np.ones(6),
    )
    assert np.count_nonzero(result[:, 160:]) == 0


def test_d81_cross_state_lock_accepts_registered_k2() -> None:
    old = [
        {
            "class_index": index,
            "class_handle": _handle(f"old-{index}"),
        }
        for index in range(6)
    ]
    new = [
        {
            "class_index": index + 6,
            "class_handle": _handle(f"new-{index}"),
        }
        for index in range(5)
    ]
    common = {
        "receiver": "20-1",
        "seed": 840001,
        "k_shot": 2,
        "phase1_checkpoint_sha256": "a" * 64,
        "feature_runtime_sha256": "b" * 64,
        "method_lock_sha256": "c" * 64,
        "row_handle": "row-1",
        "row_manifest_sha256": "d" * 64,
    }
    before_enrollment = {
        **common,
        "registration_state": "before",
        "registered_classes": old,
    }
    before_apply = {
        **common,
        "registration_state": "before",
    }
    after_enrollment = {
        **common,
        "registration_state": "after",
        "registered_classes": old + new,
    }
    after_apply = {
        **common,
        "registration_state": "after",
    }
    old_handles, all_handles, k_shot = _require_cross_state_lock(
        before_enrollment,
        before_apply,
        after_enrollment,
        after_apply,
    )
    assert k_shot == 2
    assert len(old_handles) == 6
    assert len(all_handles) == 11


class _FormalV2Component:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest
        self.domain_registry = tuple(
            f"domain-{index}" for index in range(14)
        )
        rng = np.random.default_rng(7283101)
        base = rng.normal(size=(6, 160))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        self._prototypes = []
        self._radius = []
        for index in range(14):
            rows = base + (index - 6.5) * 0.0005 * rng.normal(
                size=(6, 160)
            )
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            self._prototypes.append(rows.astype(np.float32))
            self._radius.append(
                np.full(6, 0.02 + index * 0.0001, dtype=np.float32)
            )

    def reconstruct_domain(self, domain: str) -> np.ndarray:
        return self._prototypes[self.domain_registry.index(domain)]

    def radius_for_domain(self, domain: str) -> np.ndarray:
        return self._radius[self.domain_registry.index(domain)]

    def resource_audit(self) -> dict[str, object]:
        return {
            "reconstruction_rmse": 0.001,
            "all_residual_domain_enrollment_reconstruction_macs": 37440,
            "logical_deployment_state_bytes": 7328,
        }


def _formal_component_fixture(
    tmp_path: Path,
) -> tuple[_FormalV2Component, Path, str]:
    root = tmp_path / "package" / "component"
    root.mkdir(parents=True)
    manifest = {
        "schema": "int8_domain_class_center_lowrank_residual_radius_v2",
        "feature_dim": 160,
        "component_state": "PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL",
    }
    path = root / "manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _FormalV2Component(manifest), root, digest


def test_feature_builder_uses_outer_sealed_v2_ground_component(
    tmp_path: Path,
) -> None:
    component, root, digest = _formal_component_fixture(tmp_path)
    basis, weights, audit = (
        _ground_spectrum_from_formal_v2_component(
            component,
            component_dir=root,
            sealed_component_dir=root,
            expected_manifest_sha256=digest,
        )
    )
    assert basis.shape[0] == 160
    assert weights.shape == (basis.shape[1],)
    assert np.isclose(np.sum(weights), 1.0)
    assert audit["ground_component_input_count"] == 84
    assert audit["d81_basis_sha256"] == audit["basis_sha256"]
    assert (
        audit["ground_statistic_semantics"]
        == "v2_cell_radius_reliability_ground_spectrum_for_d81_cauchy_center"
    )
    assert audit["ground_component_outer_joint_seal_verified"] is True


def test_feature_builder_rejects_unsealed_ground_component_path(
    tmp_path: Path,
) -> None:
    component, root, digest = _formal_component_fixture(tmp_path)
    other = tmp_path / "other" / "component"
    other.mkdir(parents=True)
    (other / "manifest.json").write_bytes(
        (root / "manifest.json").read_bytes()
    )
    with np.testing.assert_raises_regex(
        Stage2AblationFeatureBuilderError,
        "not the outer-sealed",
    ):
        _ground_spectrum_from_formal_v2_component(
            component,
            component_dir=other,
            sealed_component_dir=root,
            expected_manifest_sha256=digest,
        )


def test_feature_builder_rejects_v2_ground_manifest_hash_drift(
    tmp_path: Path,
) -> None:
    component, root, _ = _formal_component_fixture(tmp_path)
    with np.testing.assert_raises_regex(
        Stage2AblationFeatureBuilderError,
        "manifest hash drift",
    ):
        _ground_spectrum_from_formal_v2_component(
            component,
            component_dir=root,
            sealed_component_dir=root,
            expected_manifest_sha256="0" * 64,
        )
