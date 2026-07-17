from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_d21_support_only_lifecycle as runner


OLD = ("old-a", "old-b", "old-c")
NEW = ("new-x", "new-y")
CAPSULE = "e" * 64


class FakeComponent:
    def __init__(
        self,
        radius_rows: np.ndarray,
        *,
        formal: bool = True,
        radius_provenance: str = "phase1_offline_aggregate_p90_cosine_distance_v1",
    ) -> None:
        self._rows = np.asarray(radius_rows, dtype=np.float32)
        self.domain_registry = tuple(f"domain-{index}" for index in range(len(self._rows)))
        self.class_registry = tuple(f"phase1-tx-{index}" for index in range(self._rows.shape[1]))
        self.manifest = {
            "schema": "int8_domain_class_center_lowrank_residual_radius_v2",
            "formal_phase2_eligible": formal,
            "radius_provenance": radius_provenance,
            "radius_definition": "p90_cosine_distance_to_phase1_domain_class_centroid",
            "checkpoint_sha256": "a" * 64,
            "class_handle_binding_sha256": "b" * 64,
            "deployment_bundle_root_sha256": "c" * 64,
        }

    def radius_for_domain(self, domain_handle: str) -> np.ndarray:
        return self._rows[self.domain_registry.index(domain_handle)].copy()

    def resource_audit(self):
        return {"compressed_numeric_payload_bytes": 1234}


def _direction(index: int) -> np.ndarray:
    value = np.zeros(runner.FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _support(classes: tuple[str, ...], k: int, offset: int):
    rows = []
    labels = []
    for class_index, label in enumerate(classes):
        for rank in range(k):
            row = _direction(offset + class_index)
            if k > 1:
                row[80 + rank] = 0.01 * ((rank % 3) - 1)
            rows.append(row)
            labels.append(label)
    return np.stack(rows), labels


def test_cli_and_runner_have_no_query_or_scorer_surface() -> None:
    destinations = {action.dest.lower() for action in runner.build_parser()._actions}
    forbidden = ("query", "scorer", "truth", "role", "quota", "assignment")
    assert not any(token in name for name in destinations for token in forbidden)
    for function in (runner.run, runner.evaluate_support_lifecycle):
        parameters = {name.lower() for name in inspect.signature(function).parameters}
        assert not any(token in name for name in parameters for token in forbidden)
    assert not hasattr(runner, "_receipt_token")
    assert "sealed" not in inspect.signature(
        runner._synthetic_receipt_token
    ).parameters
    assert {"component_dir", "deployment_bundle_root_sha256"}.isdisjoint(destinations)
    assert {
        "joint_package_root",
        "joint_detached_seal",
        "joint_signature_envelope",
        "outer_content_root_sha256",
    } <= destinations
    run_parameters = set(inspect.signature(runner.run).parameters)
    assert {"component_dir", "expected_deployment_bundle_root_sha256"}.isdisjoint(
        run_parameters
    )
    source = inspect.getsource(runner.run)
    assert source.index("load_formal_adv3b02_deployment_bundle") < source.index(
        "materialize_somph_enrollment_with_signed_authority"
    )


def test_target_capsule_without_joint_binding_requires_rebuild() -> None:
    with pytest.raises(runner.D21RunnerError, match="rebuild target capsule"):
        runner._require_target_joint_binding(
            {"feature_runtime_sha256": "d" * 64},
            surface_name="legacy target manifest",
            expected_outer_content_root_sha256="c" * 64,
            expected_detached_seal_sha256="8" * 64,
            expected_signature_envelope_sha256="9" * 64,
            expected_checkpoint_lineage_sha256="a" * 64,
            expected_runtime_sha256="d" * 64,
        )
    complete = {
        "phase1_adv3b02_outer_content_root_sha256": "c" * 64,
        "phase1_adv3b02_detached_seal_sha256": "8" * 64,
        "phase1_adv3b02_signature_envelope_sha256": "9" * 64,
        "phase1_checkpoint_lineage_sha256": "a" * 64,
        "feature_runtime_sha256": "0" * 64,
    }
    with pytest.raises(runner.D21RunnerError, match="binding drift"):
        runner._require_target_joint_binding(
            complete,
            surface_name="wrong runtime target manifest",
            expected_outer_content_root_sha256="c" * 64,
            expected_detached_seal_sha256="8" * 64,
            expected_signature_envelope_sha256="9" * 64,
            expected_checkpoint_lineage_sha256="a" * 64,
            expected_runtime_sha256="d" * 64,
        )


def test_v2_component_binding_uses_all_radius_rows_and_rejects_substitutes() -> None:
    component = FakeComponent(
        np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    )
    prior, audit = runner.component_global_median_radius_prior(
        component, expected_old_class_count=3
    )
    assert prior == pytest.approx(0.35)
    assert audit["radius_value_count"] == 6
    assert audit["synthetic_or_substitute_radius_used"] is False
    with pytest.raises(runner.D21RunnerError, match="binding drift"):
        runner.component_global_median_radius_prior(
            component, expected_old_class_count=4
        )
    missing = FakeComponent(
        np.zeros((2, 3), dtype=np.float32),
        formal=False,
        radius_provenance="radius_provenance_missing_direction_only_development",
    )
    with pytest.raises(runner.D21RunnerError, match="binding drift"):
        runner.component_global_median_radius_prior(
            missing, expected_old_class_count=3
        )


def test_fixed_candidates_and_k1_old_lock_with_safe_l0_fallback() -> None:
    component = FakeComponent(np.full((2, 3), 2.0, dtype=np.float32))
    old_z, old_labels = _support(OLD, 1, 0)
    new_z, new_labels = _support(NEW, 1, 10)
    result = runner.evaluate_support_lifecycle(
        old_support_z_id=old_z,
        old_support_labels=old_labels,
        new_support_z_id=new_z,
        new_support_labels=new_labels,
        old_classes=OLD,
        new_classes=NEW,
        component=component,
    )
    assert result["status"] == "SYNTHETIC_LOCAL_COMPLETE"
    assert "evidence_mode" not in inspect.signature(
        runner.evaluate_support_lifecycle
    ).parameters
    assert result["k_shot"] == 1
    assert tuple(result["candidates"]) == runner.CANDIDATE_ORDER
    assert result["candidates"][runner.L0]["config"] == {
        "radius_enabled": False,
        "boundary_enabled": False,
        "radius_prior": 2.0,
    }
    assert result["candidates"][runner.L1]["config"]["radius_enabled"] is True
    assert result["candidates"][runner.L1]["config"]["boundary_enabled"] is False
    assert result["candidates"][runner.L2]["config"]["boundary_enabled"] is True
    assert result["selection"]["selected_candidate_id"] == runner.L0
    assert result["selection"]["fallback_to_l0"] is True
    for row in result["candidates"].values():
        assert row["internal_target_score_lock"] == {
            "lock_kind": "internal-target-score-lock",
            "dali_integrated": False,
            "dali_lock_claimed": False,
            "old_prototype_bitwise_locked": True,
            "old_radius_bitwise_locked": True,
            "old_radius_active_mask_bitwise_locked": True,
            "old_score_columns_bitwise_locked": True,
            "probe_rows": 5,
        }
        assert row["before_state"]["radius_policy"] == "fixed_preregistered_prior_k1"
        assert not any(row["after_state"]["radius_active"])
        assert row["after_state"]["boundaries"] == []
        assert set(row["after_support_result"]["per_class"]) == set(OLD + NEW)
        assert row["after_resource"]["trainable_parameters"] == 0
        assert row["after_resource"]["adaptation_epochs"] == 0


def test_dlpack_extraction_reuses_compatibility_closure(monkeypatch) -> None:
    class FakeRuntime(torch.nn.Module):
        def forward(self, batch):
            feature = torch.zeros((len(batch), runner.FEATURE_DIM), dtype=torch.float32)
            feature[:, 0] = batch[:, 0, 0]
            return {"features": feature, "logits": torch.zeros((len(batch), 3))}

    def blocked(*args, **kwargs):
        raise AssertionError("torch.from_numpy must not be used")

    monkeypatch.setattr(torch, "from_numpy", blocked)
    iq = np.zeros((2, 2, 16), dtype=np.float32)
    iq[0, 0, 0] = 1.0
    iq[1, 0, 0] = 2.0
    features, audit = runner._extract_z_id(
        FakeRuntime(), torch.device("cpu"), iq
    )
    assert features[:, 0].tolist() == [1.0, 2.0]
    assert audit["dlpack_numpy_torch_bridge"] is True
    assert audit["derived_views_per_support"] == 0


def test_mocked_sealed_materialization_writes_bound_commit(tmp_path, monkeypatch) -> None:
    joint_binding = {
        "phase1_adv3b02_outer_content_root_sha256": "c" * 64,
        "phase1_adv3b02_detached_seal_sha256": "8" * 64,
        "phase1_adv3b02_signature_envelope_sha256": "9" * 64,
        "phase1_checkpoint_lineage_sha256": "a" * 64,
        "feature_runtime_sha256": "d" * 64,
    }
    before_manifest = {
        "kind": "before",
        "receiver": "rx0",
        "seed": 7,
        "k_shot": 1,
        **joint_binding,
        "registered_classes": [
            {"class_index": index, "class_handle": value}
            for index, value in enumerate(OLD)
        ],
        "package_root_sha256": "e" * 64,
    }
    after_manifest = {
        **before_manifest,
        "kind": "after",
        "registered_classes": [
            {"class_index": index, "class_handle": value}
            for index, value in enumerate(OLD + NEW)
        ],
        "package_root_sha256": "f" * 64,
    }
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    component = FakeComponent(np.full((2, 3), 2.0, dtype=np.float32))
    component.class_registry = OLD
    binding_calls = []

    class FakeRuntime(torch.nn.Module):
        def forward(self, batch):
            return {"features": torch.zeros((len(batch), runner.FEATURE_DIM))}

    formal_context = {
        "schema": runner.FORMAL_CONTEXT_SCHEMA,
        "formal_phase2_eligible": True,
        "standalone_component_formal_phase2_eligible": False,
        "outer_signature_verified": True,
        "detached_seal_verified": True,
        "runtime_checkpoint_parity_verified": True,
        "outer_content_root_sha256": "c" * 64,
    }

    def preopen(root, seal, *, expected_seal_sha256):
        return before_manifest if root == before_root else after_manifest

    def load_joint(path, **kwargs):
        binding_calls.append(kwargs)
        return runner.VerifiedADV3B02DeploymentBundle(
            runtime=FakeRuntime(),
            component=component,
            class_binding={},
            parity_receipt={},
            generation_lock={},
            method_lock={},
            formal_phase2_context=formal_context,
            audit={"status": "PASS"},
        )

    def materialize(root, **kwargs):
        manifest = before_manifest if root == before_root else after_manifest
        return SimpleNamespace(
            manifest=manifest,
            materialized_payloads={scene: {} for scene in runner.FORMAL_LEO_WEAK_SCENARIOS},
        )

    def payload_rows(payload, manifest, overlay, *, scenario):
        classes = OLD if manifest["kind"] == "before" else OLD + NEW
        iq = np.zeros((len(classes), 2, 8), dtype=np.float32)
        for index in range(len(classes)):
            iq[index, 0, 0] = index if index < len(OLD) else 10 + index - len(OLD)
        return {
            "iq": iq,
            "labels": np.asarray(classes),
            "ranks": np.zeros(len(classes), dtype=np.int64),
            "tokens": np.asarray([f"token-{value}" for value in classes]),
            "hashes": np.asarray([f"hash-{value}" for value in classes]),
            "overlay_tokens": np.asarray([f"overlay-{value}" for value in classes]),
            "satellite_seeds": np.arange(len(classes), dtype=np.int64),
        }

    def extract(model, device, iq):
        rows = []
        for value in iq[:, 0, 0].astype(int):
            rows.append(_direction(value))
        return np.stack(rows), {"support_rows": len(rows), "mock": True}

    monkeypatch.setattr(runner, "_preopen_manifest", preopen)
    monkeypatch.setattr(runner, "load_formal_adv3b02_deployment_bundle", load_joint)
    monkeypatch.setattr(runner, "materialize_somph_enrollment_with_signed_authority", materialize)
    monkeypatch.setattr(
        runner,
        "finalize_somph_enrollment_authority_after_materialization",
        lambda evidence: {
            **joint_binding,
            "package_root_sha256": evidence.manifest["package_root_sha256"],
            "post_materialization_audit_sha256": (
                "6" if evidence.manifest["kind"] == "before" else "7"
            )
            * 64,
        },
    )
    monkeypatch.setattr(runner, "_require_post_materialization_authority", lambda *a: None)
    monkeypatch.setattr(runner, "_overlay_index", lambda *a: ({}, {}))
    monkeypatch.setattr(runner, "_old_reuse", lambda *a: None)
    monkeypatch.setattr(runner, "_payload_rows", payload_rows)
    monkeypatch.setattr(runner, "_extract_z_id", extract)

    output = tmp_path / "result"
    receipt = runner.run(
        before_root=before_root,
        before_seal=tmp_path / "before.seal",
        expected_before_seal_sha256="1" * 64,
        before_formal_policy=tmp_path / "before.policy",
        before_formal_policy_authorization=tmp_path / "before.authorization",
        before_signed_policy_authorization_envelope=tmp_path / "before.envelope",
        expected_before_signed_policy_authorization_envelope_sha256="2" * 64,
        after_root=after_root,
        after_seal=tmp_path / "after.seal",
        expected_after_seal_sha256="3" * 64,
        after_formal_policy=tmp_path / "after.policy",
        after_formal_policy_authorization=tmp_path / "after.authorization",
        after_signed_policy_authorization_envelope=tmp_path / "after.envelope",
        expected_after_signed_policy_authorization_envelope_sha256="4" * 64,
        joint_package_root=tmp_path / "joint",
        joint_detached_seal=tmp_path / "joint.seal",
        expected_joint_detached_seal_sha256="8" * 64,
        joint_signature_envelope=tmp_path / "joint.envelope",
        expected_joint_signature_envelope_sha256="9" * 64,
        expected_checkpoint_lineage_sha256="a" * 64,
        expected_runtime_sha256="d" * 64,
        expected_component_pre_sign_content_root_sha256="1" * 64,
        expected_class_handle_binding_sha256="b" * 64,
        expected_parity_receipt_sha256="2" * 64,
        expected_generation_lock_sha256="3" * 64,
        expected_method_lock_sha256="4" * 64,
        expected_generation_config_sha256="5" * 64,
        expected_generation_code_sha256="6" * 64,
        expected_outer_content_root_sha256="c" * 64,
        output=output,
        device_name="cpu",
    )
    assert binding_calls == [
        {
            "detached_seal_path": tmp_path / "joint.seal",
            "expected_detached_seal_sha256": "8" * 64,
            "signature_envelope_path": tmp_path / "joint.envelope",
            "expected_signature_envelope_sha256": "9" * 64,
            "expected_checkpoint_lineage_sha256": "a" * 64,
            "expected_runtime_sha256": "d" * 64,
            "expected_component_pre_sign_content_root_sha256": "1" * 64,
            "expected_class_handle_binding_sha256": "b" * 64,
            "expected_parity_receipt_sha256": "2" * 64,
            "expected_generation_lock_sha256": "3" * 64,
            "expected_method_lock_sha256": "4" * 64,
            "expected_generation_config_sha256": "5" * 64,
            "expected_generation_code_sha256": "6" * 64,
            "expected_outer_content_root_sha256": "c" * 64,
        }
    ]
    commit = json.loads((output / "COMMIT.json").read_text(encoding="utf-8"))
    assert receipt["k_shot"] == 1
    assert commit["query_opened"] is False and commit["scorer_opened"] is False
    assert set(commit["artifacts"]) == {
        "lifecycle_results.json",
        "support_audit.json",
        "resource_audit.json",
    }
    for name, digest in commit["artifacts"].items():
        assert hashlib_sha256(output / name) == digest
    assert commit["actual_artifact_bytes"] == {
        name: (output / name).stat().st_size
        for name in (
            "lifecycle_results.json",
            "support_audit.json",
            "resource_audit.json",
            "COMMIT.json",
        )
    }
    assert commit["total_delivery_footprint_bytes"] == sum(
        commit["actual_artifact_bytes"].values()
    )
    resource = json.loads((output / "resource_audit.json").read_text(encoding="utf-8"))
    assert resource["actual_artifact_bytes"] == commit["actual_artifact_bytes"]
    assert resource["total_delivery_footprint_bytes"] == commit[
        "total_delivery_footprint_bytes"
    ]


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
