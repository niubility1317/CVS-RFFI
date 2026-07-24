from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi import somph_runtime_trust as runtime_trust
from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle_module
from cvsrffi import stage2_grb_jp4_adv_drqknn_bcrr as grb_module
from cvsrffi.phase1_adv3b02_deployment_bundle import (
    ADV3B02DeploymentBundleError,
    COMPONENT_PROFILE_GRB_JP4_Q4,
    COMPONENT_PROFILE_SCHEMA,
    ROLE_TO_PATH,
    SIGNATURE_DOMAIN,
    SIGNATURE_ENVELOPE_SCHEMA,
    build_unsigned_adv3b02_deployment_bundle,
    class_handle_binding_sha256,
    component_profile_for_schema,
    load_formal_adv3b02_deployment_bundle,
    reverify_formal_adv3b02_deployment_bundle,
    runtime_structure_receipt,
    VerifiedADV3B02DeploymentBundle,
)
from cvsrffi.phase1_grb_jp4_bundle import (
    ALLOWED_NPZ_MEMBERS,
    CLASS_COUNT,
    COMPONENT_PROFILE,
    FEATURE_DIM,
    HIDDEN_DIM,
    NPZ_NAME,
    RANK,
    SCHEMA,
    build_grb_jp4_component,
    load_grb_jp4_component,
    save_grb_jp4_component,
    validate_grb_jp4_component,
)
from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (
    FormalGRBJP4State,
    GRBJP4SpikeError,
    GroundReceiverBasis,
    _joint_proj_linear,
    _fit_stage2_b_from_precomputed_jacobian_development_only,
    append_formal_stage2_c,
    deserialize_jp4_fit_state,
    fit_stage2_b_from_support_iq,
    predict_five_arms,
    resource_receipt,
    serialize_jp4_fit_state,
)
from cvsrffi.stage2_predictor_bundle import canonical_json_bytes, sha256_file


CHECKPOINT_SHA = "a" * 64
CODE_SHA = "b" * 64
CONFIG_SHA = "c" * 64
CLASSES = tuple(f"tx-{index}" for index in range(CLASS_COUNT))


def _aggregate() -> dict:
    rng = np.random.default_rng(2407)
    prototypes = rng.normal(size=(CLASS_COUNT, FEATURE_DIM)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    shifts = rng.normal(size=(9, FEATURE_DIM)).astype(np.float32)
    # The final two rows are present but unobserved.  Their values must not
    # affect the deterministic factor construction or generation digest.
    shifts[-2:] = rng.normal(size=(2, FEATURE_DIM)).astype(np.float32)
    return {
        "feature_key": "z_id",
        "prototypes": prototypes,
        "domain_shifts": {
            "domain_shift": shifts,
            "domain_counts": np.asarray([6, 4, 5, 3, 2, 7, 3, 0, 0]),
        },
    }


def _weight() -> np.ndarray:
    return np.random.default_rng(2408).normal(size=(FEATURE_DIM, HIDDEN_DIM)).astype(np.float32)


def _build() -> tuple[dict[str, np.ndarray], dict]:
    return build_grb_jp4_component(
        _aggregate(),
        class_registry=CLASSES,
        checkpoint_joint_proj_weight=_weight(),
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=class_handle_binding_sha256(CLASSES),
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
        provenance_status="UNVERIFIED_COMPONENT_FIXTURE_ONLY",
        formal_phase2_eligible=False,
    )


class _TinyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.joint_proj = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_DIM, FEATURE_DIM, bias=False),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            self.joint_proj[0].weight.copy_(torch.from_numpy(_weight()))


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_head = _TinyHead()


class _TinyRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _TinyBackbone()

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.grb_jp4_forward(iq)[0]

    @torch.jit.export
    def grb_jp4_forward(
        self, iq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = iq.reshape(iq.shape[0], -1)
        pre_relu = self.id_backbone.cls_head.joint_proj[0](hidden)
        z_id = self.id_backbone.cls_head.joint_proj[1](pre_relu)
        z_dom = iq[:, 0, :]
        return z_id, z_dom, hidden, pre_relu


class _SameWeightDifferentForwardRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _TinyBackbone()

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.grb_jp4_forward(iq)[0]

    @torch.jit.export
    def grb_jp4_forward(
        self, iq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = iq.reshape(iq.shape[0], -1)
        pre_relu = self.id_backbone.cls_head.joint_proj[0](hidden)
        z_id = -self.id_backbone.cls_head.joint_proj[1](pre_relu)
        z_dom = iq[:, 0, :]
        return z_id, z_dom, hidden, pre_relu


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _grb_outer_fixture(tmp_path: Path) -> tuple[Path, dict]:
    runtime_path = tmp_path / "runtime.pt"
    torch.jit.save(torch.jit.script(_TinyRuntime().eval()), str(runtime_path))
    runtime_sha = sha256_file(runtime_path)
    binding_sha = class_handle_binding_sha256(CLASSES)
    binding_path = tmp_path / "inputs" / "class.json"
    _write_json(
        binding_path,
        {
            "schema": "phase1_tx_class_handle_binding_v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "class_id_to_handle": [
                {"class_index": index, "class_handle": handle}
                for index, handle in enumerate(CLASSES)
            ],
            "class_handle_binding_sha256": binding_sha,
        },
    )
    payload, manifest = _build()
    component_dir = tmp_path / "component"
    saved = save_grb_jp4_component(component_dir, payload, manifest)
    parity_path = tmp_path / "inputs" / "parity.json"
    _write_json(
        parity_path,
        {
            "schema": "cvs.phase1.runtime_checkpoint_parity_receipt.v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "runtime_sha256": runtime_sha,
            "parity_status": "PASS",
            "max_abs_output_delta": 0.0,
            "parity_vector_root_sha256": "d" * 64,
            **runtime_structure_receipt(runtime_path),
        },
    )
    generation_path = tmp_path / "inputs" / "generation.json"
    _write_json(
        generation_path,
        {
            "schema": "cvs.phase1.prototype_generation_lock.v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "component_pre_sign_content_root_sha256": saved["pre_sign_content_root_sha256"],
            "class_handle_binding_sha256": binding_sha,
            "generation_config_sha256": CONFIG_SHA,
            "generation_code_sha256": CODE_SHA,
            "phase1_stream_sha256": "e" * 64,
            "radius_generation_proof_sha256": "f" * 64,
        },
    )
    method_path = tmp_path / "inputs" / "method.json"
    _write_json(
        method_path,
        {
            "schema": "cvs.phase1.adv3b02_method_lock.v1",
            "method_id": "ADV3B02-GRB-JP4",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "runtime_sha256": runtime_sha,
            "component_pre_sign_content_root_sha256": saved["pre_sign_content_root_sha256"],
            "class_handle_binding_sha256": binding_sha,
            "parity_receipt_sha256": sha256_file(parity_path),
            "generation_lock_sha256": sha256_file(generation_path),
            "generation_config_sha256": CONFIG_SHA,
            "generation_code_sha256": CODE_SHA,
        },
    )
    bundle = tmp_path / "bundle"
    seal = tmp_path / "external" / "seal.json"
    request = tmp_path / "external" / "request.json"
    result = build_unsigned_adv3b02_deployment_bundle(
        bundle,
        torchscript_runtime_path=runtime_path,
        component_dir=component_dir,
        class_binding_path=binding_path,
        parity_receipt_path=parity_path,
        generation_lock_path=generation_path,
        method_lock_path=method_path,
        detached_seal_path=seal,
        signing_request_path=request,
    )
    envelope = tmp_path / "external" / "signature.json"
    _write_json(
        envelope,
        {
            "schema": SIGNATURE_ENVELOPE_SCHEMA,
            "domain": SIGNATURE_DOMAIN,
            "issuer": runtime_trust.PINNED_AUTHORITY_ISSUER,
            "key_id": runtime_trust.PINNED_AUTHORITY_KEY_ID,
            "detached_seal_sha256": result["detached_seal_sha256"],
            "signature_ed25519_hex": "01" * 64,
        },
    )
    return bundle, {
        "detached_seal_path": seal,
        "expected_detached_seal_sha256": result["detached_seal_sha256"],
        "signature_envelope_path": envelope,
        "expected_signature_envelope_sha256": sha256_file(envelope),
        "expected_checkpoint_lineage_sha256": CHECKPOINT_SHA,
        "expected_runtime_sha256": runtime_sha,
        "expected_component_pre_sign_content_root_sha256": saved["pre_sign_content_root_sha256"],
        "expected_class_handle_binding_sha256": binding_sha,
        "expected_parity_receipt_sha256": sha256_file(parity_path),
        "expected_generation_lock_sha256": sha256_file(generation_path),
        "expected_method_lock_sha256": sha256_file(method_path),
        "expected_generation_config_sha256": CONFIG_SHA,
        "expected_generation_code_sha256": CODE_SHA,
        "expected_outer_content_root_sha256": result["outer_content_root_sha256"],
    }


def test_compact_q4_builder_is_deterministic_and_uses_fixed_shapes() -> None:
    first, first_manifest = _build()
    second, second_manifest = _build()
    assert first_manifest == second_manifest
    for key in ALLOWED_NPZ_MEMBERS:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["p_g_q"].shape == (6, 160)
    assert first["l_g_q"].shape == (4, 160)
    assert first["r_q"].shape == (4, 320)
    assert first_manifest["kappa_g"] >= 1.0
    assert first_manifest["schema"] == SCHEMA
    assert first_manifest["component_profile"] == COMPONENT_PROFILE
    assert first_manifest["resource_audit"]["p_g_numeric_payload_bytes"] == 972
    assert first_manifest["resource_audit"]["l_g_numeric_payload_bytes"] == 648
    assert first_manifest["resource_audit"]["r_numeric_payload_bytes"] == 1288
    assert first_manifest["resource_audit"]["stage2_theta_numeric_payload_bytes"] == 6
    assert first_manifest["resource_audit"]["component_numeric_payload_bytes"] == 2908
    assert first_manifest["resource_audit"]["component_plus_theta_numeric_payload_bytes"] == 2914
    for codes, scales in (
        (first["p_g_q"], first["p_g_scale"]),
        (first["l_g_q"], first["l_g_scale"]),
        (first["r_q"], first["r_scale"]),
    ):
        assert codes.dtype == np.int8 and not np.any(codes == -128)
        assert scales.dtype == np.float16 and np.all(scales > 0)


def test_unobserved_domain_rows_do_not_change_canonical_ground_factors() -> None:
    aggregate = _aggregate()
    changed = _aggregate()
    changed["domain_shifts"]["domain_shift"][-2:] *= -1000.0
    common = {
        "class_registry": CLASSES,
        "checkpoint_joint_proj_weight": _weight(),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "class_handle_binding_sha256": class_handle_binding_sha256(CLASSES),
        "generation_code_sha256": CODE_SHA,
        "generation_config_sha256": CONFIG_SHA,
        "provenance_status": "UNVERIFIED_COMPONENT_FIXTURE_ONLY",
        "formal_phase2_eligible": False,
    }
    first, manifest = build_grb_jp4_component(aggregate, **common)
    second, other_manifest = build_grb_jp4_component(changed, **common)
    np.testing.assert_array_equal(first["l_g_q"], second["l_g_q"])
    np.testing.assert_array_equal(first["l_g_scale"], second["l_g_scale"])
    assert manifest["source_aggregate_generation_digest_sha256"] == other_manifest["source_aggregate_generation_digest_sha256"]


def test_ambiguous_q4_singular_subspace_is_rejected_before_sealing() -> None:
    from cvsrffi.phase1_grb_jp4_bundle import _canonical_svd_rows

    ambiguous = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    ambiguous[np.arange(5), np.arange(5)] = np.asarray([7.0, 5.0, 3.0, 3.0, 1.0])
    with pytest.raises(ValueError, match="ambiguous q4 singular subspace"):
        _canonical_svd_rows(ambiguous, rank=RANK, field="test aggregate")


def test_component_roundtrip_exposes_only_readonly_compact_factors(tmp_path: Path) -> None:
    payload, manifest = _build()
    result = save_grb_jp4_component(tmp_path, payload, manifest)
    validated = validate_grb_jp4_component(
        tmp_path,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=class_handle_binding_sha256(CLASSES),
        expected_pre_sign_content_root_sha256=result["pre_sign_content_root_sha256"],
    )
    component = load_grb_jp4_component(
        tmp_path,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=class_handle_binding_sha256(CLASSES),
        expected_pre_sign_content_root_sha256=result["pre_sign_content_root_sha256"],
        allow_pending_outer_joint_seal_development=True,
    )
    assert validated["formal_phase2_eligible"] is False
    assert component.ground_prototypes().shape == (6, 160)
    assert component.ground_left_factors().shape == (4, 160)
    assert component.checkpoint_right_factors().shape == (4, 320)
    assert not component.ground_prototypes().flags.writeable
    assert component.class_registry == CLASSES
    with np.load(tmp_path / NPZ_NAME, allow_pickle=False) as archive:
        assert set(archive.files) == ALLOWED_NPZ_MEMBERS
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8").lower()
    assert not any(token in text for token in ("source_path", "sample_id", "member_id", "sample_count"))


def test_tamper_wrong_binding_and_formal_load_fail_closed(tmp_path: Path) -> None:
    payload, manifest = _build()
    result = save_grb_jp4_component(tmp_path, payload, manifest)
    with pytest.raises(ValueError, match="formal GRB component load requires"):
        load_grb_jp4_component(
            tmp_path,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            expected_class_handle_binding_sha256=class_handle_binding_sha256(CLASSES),
            expected_pre_sign_content_root_sha256=result["pre_sign_content_root_sha256"],
        )
    parsed = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    parsed["class_handle_binding_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        validate_grb_jp4_component(tmp_path)


def test_inner_filename_is_bound_by_grb_pre_sign_root(tmp_path: Path) -> None:
    payload, manifest = _build()
    saved = save_grb_jp4_component(tmp_path, payload, manifest)
    manifest_path = tmp_path / "manifest.json"
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["member_allowlist"] = ["renamed_component.npz"]
    from cvsrffi.phase1_grb_jp4_bundle import _pre_sign_content_root
    assert _pre_sign_content_root(changed, saved["component_npz_sha256"]) != saved[
        "pre_sign_content_root_sha256"
    ]
    manifest_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    (tmp_path / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="allowlist drift"):
        validate_grb_jp4_component(tmp_path)


def test_existing_joint_seal_profile_extension_is_not_a_new_container_or_authority() -> None:
    profile = component_profile_for_schema(SCHEMA)
    assert profile == {
        "schema": COMPONENT_PROFILE_SCHEMA,
        "component_schema": SCHEMA,
        "component_profile": COMPONENT_PROFILE_GRB_JP4_Q4,
        "container_member_count": 8,
        "signature_domain": "cvs.phase1.adv3b02_deployment_bundle.ed25519.v1",
        "method_lock_schema": "cvs.phase1.adv3b02_method_lock.v1",
    }
    assert len(ROLE_TO_PATH) == 8
    assert COMPONENT_PROFILE == COMPONENT_PROFILE_GRB_JP4_Q4
    with pytest.raises(Exception, match="unrecognized"):
        component_profile_for_schema("unknown")


def test_existing_eight_member_joint_seal_dispatches_grb_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, kwargs = _grb_outer_fixture(tmp_path)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    loaded = load_formal_adv3b02_deployment_bundle(bundle, **kwargs)
    assert loaded.formal_phase2_context["component_profile"] == COMPONENT_PROFILE_GRB_JP4_Q4
    assert loaded.formal_phase2_context["checkpoint_lineage_sha256"] == CHECKPOINT_SHA
    assert loaded.formal_phase2_context["checkpoint_sha256"] == CHECKPOINT_SHA
    assert loaded.formal_phase2_context["method_lock_sha256"] == kwargs["expected_method_lock_sha256"]
    assert loaded.formal_phase2_context["component_inner_filename"] == NPZ_NAME
    assert loaded.formal_phase2_context["component_outer_slot_relative_path"] == ROLE_TO_PATH["v2_component_npz"]
    assert loaded.audit["component_inner_filename_bound_by_pre_sign_root"] is True
    assert loaded.audit["component_outer_slot_bound_by_outer_content_root"] is True
    assert loaded.component.ground_left_factors().shape == (RANK, FEATURE_DIM)
    assert len(ROLE_TO_PATH) == 8
    manifest_path = bundle / "component" / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"tamper")
    with pytest.raises(ADV3B02DeploymentBundleError, match="deployment member digest drift"):
        load_formal_adv3b02_deployment_bundle(bundle, **kwargs)


def test_verified_bundle_closes_formal_fit_merge_append_and_five_arm_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root, kwargs = _grb_outer_fixture(tmp_path)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    rng = np.random.default_rng(2410)
    old_iq = torch.from_numpy(
        rng.normal(size=(CLASS_COUNT, 2, FEATURE_DIM)).astype(np.float32)
    )
    old_tokens = tuple(f"old-physical-{index}" for index in range(CLASS_COUNT))
    stage_b = fit_stage2_b_from_support_iq(
        bundle=bundle,
        support_iq=old_iq,
        support_labels=CLASSES,
        support_physical_tokens=old_tokens,
    )
    assert stage_b.stage == "S_B"
    assert stage_b.jp4.k_shot == 1
    assert stage_b.lifecycle_receipt["same_iq_two_model_states"] is True
    assert stage_b.resources["support_forward_calls"] == 2
    assert stage_b.resources["closed_form_solve_mac"] == 0
    assert stage_b.resources["second_full_model_weight_copy"] is False
    assert stage_b.resources["live_model_weight_instances_max"] == 1
    assert bundle.runtime is None
    ownership = stage_b.resources["runtime_ownership_receipt"]
    assert ownership["source_bundle_runtime_consumed_before_reverification"] is True
    assert ownership["formal_materialization_count"] == 3
    assert ownership["verified_runtime_reload_count"] == 2
    assert ownership["release_count"] == 2
    int8_audit = stage_b.resources["int8_theta_support_audit"]
    assert int8_audit["top1_agreement"] >= 0.995
    assert int8_audit["large_margin_flip_count"] == 0
    assert int8_audit["query_rows_used_for_fit"] == 0
    assert "_FORMAL_STATE_ISSUER" not in vars(grb_module)
    assert "_issue_formal_state_after_validation" not in vars(grb_module)
    assert "_issuer" not in inspect.signature(FormalGRBJP4State).parameters
    with pytest.raises(GRBJP4SpikeError, match="production-factory"):
        FormalGRBJP4State(
            stage_b.runtime,
            stage_b.runtime_member_path,
            stage_b.runtime_sha256,
            stage_b.runtime_phase,
            stage_b.ground,
            stage_b.jp4,
            stage_b.no_ground_state,
            stage_b.adapted_state,
            stage_b.stage,
            stage_b.old_support_labels,
            stage_b.old_support_tokens,
            stage_b.old_support_iq_sha256,
            stage_b.all_support_tokens,
            stage_b.registered_classes,
            stage_b.bundle_receipt_sha256,
            stage_b.lifecycle_receipt,
            stage_b.resources,
        )
    rogue_fit, _rogue_append, _rogue_registry = (
        grb_module._formal_state_orchestrator_factory()
    )
    rogue_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    rogue_state = rogue_fit(
        bundle=rogue_bundle,
        support_iq=old_iq,
        support_labels=CLASSES,
        support_physical_tokens=tuple(
            f"rogue-old-physical-{index}" for index in range(CLASS_COUNT)
        ),
    )
    with pytest.raises(GRBJP4SpikeError, match="issuance/runtime ownership"):
        predict_five_arms(
            rogue_state,
            query_iq=old_iq[:1],
            query_physical_tokens=("rogue-query-physical-0",),
        )
    wire = serialize_jp4_fit_state(stage_b.jp4)
    replay = deserialize_jp4_fit_state(
        wire,
        expected_ground_digest=stage_b.ground.digest,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_joint_weight_sha256_before=stage_b.ground.joint_weight_sha256,
        expected_formal_bundle_receipt_sha256=stage_b.bundle_receipt_sha256,
    )
    assert replay.wire_bytes() == stage_b.jp4.wire_bytes()
    with pytest.raises(GRBJP4SpikeError, match="wire digest"):
        deserialize_jp4_fit_state(
            wire[:-1] + bytes([wire[-1] ^ 1]),
            expected_ground_digest=stage_b.ground.digest,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            expected_joint_weight_sha256_before=stage_b.ground.joint_weight_sha256,
            expected_formal_bundle_receipt_sha256=stage_b.bundle_receipt_sha256,
        )

    new_iq = torch.from_numpy(
        rng.normal(size=(1, 2, FEATURE_DIM)).astype(np.float32)
    )
    stage_c = append_formal_stage2_c(
        stage_b,
        old_support_iq=old_iq,
        old_support_labels=CLASSES,
        old_support_physical_tokens=old_tokens,
        new_support_iq=new_iq,
        new_support_labels=("new-a",),
        new_registered_classes=("new-a",),
        new_support_physical_tokens=("new-physical-0",),
    )
    assert stage_c.stage == "S_C"
    assert stage_c.jp4.wire_bytes() == stage_b.jp4.wire_bytes()
    assert stage_c.lifecycle_receipt["jp4_state_byte_identical_to_s_b"] is True
    assert stage_c.resources["runtime_ownership_receipt"][
        "formal_materialization_count"
    ] == 4
    assert stage_c.resources["runtime_ownership_receipt"][
        "verified_runtime_reload_count"
    ] == 3
    query_iq = torch.from_numpy(
        rng.normal(size=(3, 2, FEATURE_DIM)).astype(np.float32)
    )
    logits, predictions, closure = predict_five_arms(
        stage_c,
        query_iq=query_iq,
        query_physical_tokens=("query-0", "query-1", "query-2"),
    )
    assert tuple(logits) == ("M0", "M_DA_NG", "M_DA", "M_OTHER", "M_JOINT")
    assert tuple(predictions) == tuple(logits)
    assert closure["row_class_token_lifecycle_matched"] is True
    assert closure["query_rows_used_for_fit"] == 0
    assert closure["resource_receipt"]["query_forward_calls"] == 2
    np.testing.assert_array_equal(logits["M0"], logits["M_DA_NG"])
    np.testing.assert_array_equal(logits["M0"], logits["M_DA"])
    np.testing.assert_array_equal(logits["M_OTHER"], logits["M_JOINT"])

    k5_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    k5_labels = tuple(label for label in CLASSES for _ in range(5))
    k5_tokens = tuple(f"k5-physical-{index}" for index in range(len(k5_labels)))
    k5_iq = torch.from_numpy(
        rng.normal(size=(len(k5_labels), 2, FEATURE_DIM)).astype(np.float32)
    )
    k5 = fit_stage2_b_from_support_iq(
        bundle=k5_bundle,
        support_iq=k5_iq,
        support_labels=k5_labels,
        support_physical_tokens=k5_tokens,
    )
    assert k5.jp4.k_shot == 5
    assert k5.jp4.fit_receipt["fallback"] == "none"
    assert k5.resources["analytic_jacobian_rows"] == len(k5_labels)
    assert k5.resources["closed_form_solve_mac"] == RANK ** 3
    assert k5.resources["int8_theta_support_audit"]["top1_agreement"] >= 0.995
    assert (
        k5.resources["int8_theta_support_audit"]["large_margin_flip_count"]
        == 0
    )
    assert (
        k5.jp4.joint_weight_semantic_sha256
        != k5.jp4.joint_weight_sha256_before
    )
    k5_logits, _, k5_closure = predict_five_arms(
        k5,
        query_iq=query_iq,
        query_physical_tokens=("k5-query-0", "k5-query-1", "k5-query-2"),
    )
    assert all(value.shape == (3, CLASS_COUNT) for value in k5_logits.values())
    assert k5_closure["row_class_token_lifecycle_matched"] is True

    state_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    forged_state = fit_stage2_b_from_support_iq(
        bundle=state_bundle,
        support_iq=old_iq,
        support_labels=CLASSES,
        support_physical_tokens=old_tokens,
    )
    semantic_runtime = torch.jit.script(
        _SameWeightDifferentForwardRuntime().eval()
    )
    torch.testing.assert_close(
        _joint_proj_linear(semantic_runtime).weight,
        _joint_proj_linear(forged_state.runtime).weight,
    )
    assert (
        str(semantic_runtime._c._get_method("grb_jp4_forward").graph)
        != str(forged_state.runtime._c._get_method("grb_jp4_forward").graph)
    )
    object.__setattr__(forged_state, "runtime", semantic_runtime)
    with pytest.raises(GRBJP4SpikeError, match="issuance/runtime ownership"):
        append_formal_stage2_c(
            forged_state,
            old_support_iq=old_iq,
            old_support_labels=CLASSES,
            old_support_physical_tokens=old_tokens,
            new_support_iq=new_iq,
            new_support_labels=("new-a",),
            new_registered_classes=("new-a",),
            new_support_physical_tokens=("forged-new-physical-0",),
        )
    with pytest.raises(GRBJP4SpikeError, match="issuance/runtime ownership"):
        predict_five_arms(
            forged_state,
            query_iq=query_iq,
            query_physical_tokens=(
                "forged-query-0",
                "forged-query-1",
                "forged-query-2",
            ),
        )

    forged_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    forged_context = dict(forged_bundle.formal_phase2_context)
    forged_context["component_profile"] = "center_lowrank_int8_radius_v2"
    object.__setattr__(forged_bundle, "formal_phase2_context", forged_context)
    with pytest.raises(GRBJP4SpikeError, match="external re-verification"):
        fit_stage2_b_from_support_iq(
            bundle=forged_bundle,
            support_iq=old_iq,
            support_labels=CLASSES,
            support_physical_tokens=old_tokens,
        )
    forged_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    forged_method = dict(forged_bundle.method_lock)
    forged_method["generation_code_sha256"] = "0" * 64
    object.__setattr__(forged_bundle, "method_lock", forged_method)
    with pytest.raises(GRBJP4SpikeError, match="external re-verification"):
        fit_stage2_b_from_support_iq(
            bundle=forged_bundle,
            support_iq=old_iq,
            support_labels=CLASSES,
            support_physical_tokens=old_tokens,
        )
    forged_bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    object.__setattr__(forged_bundle, "runtime", object())
    with pytest.raises(GRBJP4SpikeError, match="external re-verification"):
        fit_stage2_b_from_support_iq(
            bundle=forged_bundle,
            support_iq=old_iq,
            support_labels=CLASSES,
            support_physical_tokens=old_tokens,
        )
    with pytest.raises(ADV3B02DeploymentBundleError, match="production-factory"):
        VerifiedADV3B02DeploymentBundle()
    assert set(inspect.signature(fit_stage2_b_from_support_iq).parameters) == {
        "bundle",
        "support_iq",
        "support_labels",
        "support_physical_tokens",
    }


def test_verified_bundle_factory_has_no_issuer_bypass_and_rejects_retained_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root, kwargs = _grb_outer_fixture(tmp_path)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    bundle = load_formal_adv3b02_deployment_bundle(bundle_root, **kwargs)
    assert "_VERIFIED_BUNDLE_LOADER_ISSUER" not in vars(deployment_bundle_module)
    assert "_loader_issuer" not in inspect.signature(
        VerifiedADV3B02DeploymentBundle
    ).parameters
    retained_runtime = bundle.runtime
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="weakref remained live"
    ):
        reverify_formal_adv3b02_deployment_bundle(bundle)
    assert retained_runtime is not None


def test_formal_outer_component_closes_deployment_wire_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, kwargs = _grb_outer_fixture(tmp_path)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    loaded = load_formal_adv3b02_deployment_bundle(bundle, **kwargs)
    ground = GroundReceiverBasis.from_verified_joint_component(
        loaded.component,
        formal_phase2_context=loaded.formal_phase2_context,
        checkpoint_weight=_weight(),
        method_lock_sha256=kwargs["expected_method_lock_sha256"],
    )
    labels = tuple(handle for handle in CLASSES for _ in range(5))
    support_zid = np.repeat(ground.prototypes(), repeats=5, axis=0)
    jacobian = np.zeros((len(labels), RANK, FEATURE_DIM), dtype=np.float32)
    for direction in range(RANK):
        jacobian[:, direction, direction] = np.float32(0.125)
    state = _fit_stage2_b_from_precomputed_jacobian_development_only(
        support_zid=support_zid,
        support_jacobian=jacobian,
        support_labels=labels,
        support_physical_tokens=tuple(f"physical-{index}" for index in range(len(labels))),
        ground=ground,
        checkpoint_weight=_weight(),
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    receipt = resource_receipt(state, ground)
    assert receipt["numeric_payload_bytes"] == 2914
    assert receipt["jp4_metadata_and_payload_bytes"] <= 4096
    assert receipt["jp4_wire_limit_bytes"] == 4096


def test_public_component_api_has_no_runtime_source_or_target_access_surface() -> None:
    public = (
        build_grb_jp4_component,
        save_grb_jp4_component,
        validate_grb_jp4_component,
        load_grb_jp4_component,
    )
    forbidden = ("query", "target", "dataset", "path", "member", "sample")
    for function in public:
        assert not any(
            token in parameter.lower()
            for parameter in inspect.signature(function).parameters
            for token in forbidden
        )
