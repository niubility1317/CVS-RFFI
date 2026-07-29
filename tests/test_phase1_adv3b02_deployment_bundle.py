from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle
from cvsrffi import somph_runtime_trust as runtime_trust

from cvsrffi.phase1_adv3b02_deployment_bundle import (
    ADV3B02DeploymentBundleError,
    SIGNATURE_DOMAIN,
    SIGNATURE_ENVELOPE_SCHEMA,
    build_unsigned_adv3b02_deployment_bundle,
    class_handle_binding_sha256,
    load_formal_adv3b02_deployment_bundle,
    runtime_structure_receipt,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    FEATURE_DIM,
    PENDING_OUTER_JOINT_SEAL,
    build_center_lowrank_component,
    radius_generation_proof_sha256,
    save_center_lowrank_component,
)
from cvsrffi.stage2_predictor_bundle import canonical_json_bytes, sha256_file


CHECKPOINT_SHA = "a" * 64
CODE_SHA = "c" * 64
CONFIG_SHA = "d" * 64
STREAM_SHA = "e" * 64


class _TinyRuntime(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1.0


class _RuntimeWithUnusedBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("unused_full_precision", torch.ones(2048))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1.0


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _v1_payload(
    classes: list[str], domain_registry: list[str] | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(2107)
    domains = 4
    dense = rng.normal(size=(domains, len(classes), FEATURE_DIM)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=-1, keepdims=True)
    maximum = np.max(np.abs(dense), axis=-1)
    scale = np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(dense / scale[..., None]), -127, 127).astype(np.int8)
    payload = {
        "domain_class_q": q,
        "domain_class_scale": scale.astype(np.float16),
        "domain_class_mask": np.ones((domains, len(classes)), dtype=np.uint8),
        "domain_registry": np.asarray(
            domain_registry
            if domain_registry is not None
            else [f"rx_day:{20 + index}" for index in range(domains)],
            dtype=np.str_,
        ),
        "class_registry": np.asarray(classes, dtype=np.str_),
        "feature_schema": np.asarray("ADV3B02:z_id:unit_l2:160:v1", dtype=np.str_),
    }
    radius = rng.uniform(0.01, 0.10, size=(domains, len(classes))).astype(np.float32)
    return payload, radius


def _fixture(
    tmp_path: Path, domain_registry: list[str] | None = None
) -> tuple[Path, Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    classes = ["tx-0", "tx-1", "tx-2"]
    runtime_path = tmp_path / "adv3b02.torchscript.pt"
    traced = torch.jit.trace(_TinyRuntime(), torch.zeros(1, 2))
    torch.jit.save(traced, str(runtime_path))
    runtime_sha = sha256_file(runtime_path)
    runtime_structure = runtime_structure_receipt(runtime_path)

    class_binding_path = tmp_path / "inputs" / "class_binding.json"
    binding_sha = class_handle_binding_sha256(classes)
    _write_json(
        class_binding_path,
        {
            "schema": "phase1_tx_class_handle_binding_v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "class_id_to_handle": [
                {"class_index": index, "class_handle": handle}
                for index, handle in enumerate(classes)
            ],
            "class_handle_binding_sha256": binding_sha,
        },
    )
    v1, radius = _v1_payload(classes, domain_registry=domain_registry)
    proof = radius_generation_proof_sha256(
        v1,
        radius,
        phase1_stream_sha256=STREAM_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=binding_sha,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
    )
    payload, manifest = build_center_lowrank_component(
        v1,
        radius_p90_cosine_distance=radius,
        phase1_stream_sha256=STREAM_SHA,
        radius_generation_proof_sha256_value=proof,
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=binding_sha,
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
        provenance_status=PENDING_OUTER_JOINT_SEAL,
        formal_phase2_eligible=False,
    )
    component_dir = tmp_path / "component_input"
    component_result = save_center_lowrank_component(component_dir, payload, manifest)

    parity_path = tmp_path / "inputs" / "parity.json"
    _write_json(
        parity_path,
        {
            "schema": "cvs.phase1.runtime_checkpoint_parity_receipt.v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "runtime_sha256": runtime_sha,
            "parity_status": "PASS",
            "max_abs_output_delta": 0.0,
            "max_abs_output_delta_tolerance": 0.001,
            "decision_equivalence_verified": True,
            "numeric_policy": "fp32_cuda_tf32_disabled_cudnn_deterministic_v1",
            "parity_device_type": "cuda",
            "cuda_device_index": 0,
            "cuda_device_capability": [8, 6],
            "torch_version": "2.1.0+cu121",
            "cuda_runtime_version": "12.1",
            "cudnn_version": 8902,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms_enabled": True,
            "parity_vector_root_sha256": "f" * 64,
            "validated_batch_sizes": [1, 8, 64, 256],
            "feature_dim": 160,
            "logit_dim": 6,
            "finite_outputs_verified": True,
            **runtime_structure,
        },
    )
    generation_path = tmp_path / "inputs" / "generation.json"
    _write_json(
        generation_path,
        {
            "schema": "cvs.phase1.prototype_generation_lock.v1",
            "checkpoint_lineage_sha256": CHECKPOINT_SHA,
            "component_pre_sign_content_root_sha256": component_result[
                "pre_sign_content_root_sha256"
            ],
            "class_handle_binding_sha256": binding_sha,
            "generation_config_sha256": CONFIG_SHA,
            "generation_code_sha256": CODE_SHA,
            "phase1_stream_sha256": STREAM_SHA,
            "radius_generation_proof_sha256": proof,
        },
    )
    method_path = tmp_path / "inputs" / "method.json"
    method = {
        "schema": "cvs.phase1.adv3b02_method_lock.v1",
        "method_id": "ADV3B02-D21-R3",
        "checkpoint_lineage_sha256": CHECKPOINT_SHA,
        "runtime_sha256": runtime_sha,
        "component_pre_sign_content_root_sha256": component_result[
            "pre_sign_content_root_sha256"
        ],
        "class_handle_binding_sha256": binding_sha,
        "parity_receipt_sha256": sha256_file(parity_path),
        "generation_lock_sha256": sha256_file(generation_path),
        "generation_config_sha256": CONFIG_SHA,
        "generation_code_sha256": CODE_SHA,
    }
    _write_json(method_path, method)
    bundle = tmp_path / "bundle"
    seal = tmp_path / "external" / "bundle.seal.json"
    request = tmp_path / "external" / "signing_request.json"
    result = build_unsigned_adv3b02_deployment_bundle(
        bundle,
        torchscript_runtime_path=runtime_path,
        component_dir=component_dir,
        class_binding_path=class_binding_path,
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
    kwargs = {
        "detached_seal_path": seal,
        "expected_detached_seal_sha256": result["detached_seal_sha256"],
        "signature_envelope_path": envelope,
        "expected_signature_envelope_sha256": sha256_file(envelope),
        "expected_checkpoint_lineage_sha256": result["checkpoint_lineage_sha256"],
        "expected_runtime_sha256": result["runtime_sha256"],
        "expected_component_pre_sign_content_root_sha256": result[
            "component_pre_sign_content_root_sha256"
        ],
        "expected_class_handle_binding_sha256": result[
            "class_handle_binding_sha256"
        ],
        "expected_parity_receipt_sha256": result["parity_receipt_sha256"],
        "expected_generation_lock_sha256": result["generation_lock_sha256"],
        "expected_method_lock_sha256": result["method_lock_sha256"],
        "expected_generation_config_sha256": result["generation_config_sha256"],
        "expected_generation_code_sha256": result["generation_code_sha256"],
        "expected_outer_content_root_sha256": result["outer_content_root_sha256"],
    }
    return bundle, seal, request, kwargs


def test_unsigned_request_and_formal_load_are_jointly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, request, kwargs = _fixture(tmp_path)
    called = []

    def verifier(public_key: bytes, message: bytes, signature: bytes) -> None:
        called.append((public_key, message, signature))
        assert message.startswith(SIGNATURE_DOMAIN.encode("ascii") + b"\x00")

    monkeypatch.setattr(runtime_trust, "verify_ed25519", verifier)

    loaded = load_formal_adv3b02_deployment_bundle(bundle, **kwargs)
    assert len(called) == 1
    assert loaded.formal_phase2_context["formal_phase2_eligible"] is True
    assert loaded.formal_phase2_context[
        "standalone_component_formal_phase2_eligible"
    ] is False
    assert loaded.component.manifest["formal_phase2_eligible"] is False
    assert all(
        handle.startswith("rx_day:") for handle in loaded.component.domain_registry
    )
    torch.testing.assert_close(
        loaded.runtime(torch.zeros(1, 2)), torch.ones(1, 2)
    )
    assert loaded.audit["hash_and_materialization_same_file_descriptor"] is True
    signing_request = json.loads(request.read_text(encoding="utf-8"))
    assert signing_request["unsigned_signature_envelope"]["issuer"] == (
        runtime_trust.PINNED_AUTHORITY_ISSUER
    )
    assert signing_request["unsigned_signature_envelope"]["key_id"] == (
        runtime_trust.PINNED_AUTHORITY_KEY_ID
    )
    assert not (bundle / "signature.json").exists()
    assert kwargs["expected_class_handle_binding_sha256"] == (
        class_handle_binding_sha256(["tx-0", "tx-1", "tx-2"])
    )


def test_formal_load_rejects_root_tamper_and_unexpected_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _, kwargs = _fixture(tmp_path)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    (bundle / "unexpected.cache").write_bytes(b"forbidden")
    with pytest.raises(ADV3B02DeploymentBundleError, match="unexpected package file"):
        load_formal_adv3b02_deployment_bundle(bundle, **kwargs)


def test_wrong_signature_is_fail_closed_and_no_verifier_injection_exists(
    tmp_path: Path,
) -> None:
    bundle, _, _, kwargs = _fixture(tmp_path)
    parameters = inspect.signature(load_formal_adv3b02_deployment_bundle).parameters
    assert "signature_verifier" not in parameters
    assert "expected_signature_key_id" not in parameters
    with pytest.raises(ADV3B02DeploymentBundleError, match="authority signature invalid"):
        load_formal_adv3b02_deployment_bundle(bundle, **kwargs)


def test_detached_seal_replacement_is_rejected_before_parsing(tmp_path: Path) -> None:
    bundle, seal, _, kwargs = _fixture(tmp_path)
    seal.write_bytes(seal.read_bytes() + b"replacement")
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="detached seal external trust-root mismatch"
    ):
        load_formal_adv3b02_deployment_bundle(bundle, **kwargs)


def test_builder_rejects_raw_checkpoint_and_forbidden_lock_key(tmp_path: Path) -> None:
    bundle, _, _, kwargs = _fixture(tmp_path / "good")
    raw_checkpoint = tmp_path / "model.pth"
    raw_checkpoint.write_bytes(b"raw")
    with pytest.raises(ADV3B02DeploymentBundleError, match="raw training checkpoint"):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "bad-pth",
            torchscript_runtime_path=raw_checkpoint,
            component_dir=tmp_path / "good" / "component_input",
            class_binding_path=tmp_path / "good" / "inputs" / "class_binding.json",
            parity_receipt_path=tmp_path / "good" / "inputs" / "parity.json",
            generation_lock_path=tmp_path / "good" / "inputs" / "generation.json",
            method_lock_path=tmp_path / "good" / "inputs" / "method.json",
            detached_seal_path=tmp_path / "bad.seal",
            signing_request_path=tmp_path / "bad.request",
        )

    method_path = tmp_path / "good" / "inputs" / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["dataset_path"] = "forbidden"
    _write_json(method_path, method)
    with pytest.raises(ADV3B02DeploymentBundleError, match="exact schema mismatch"):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "bad-key",
            torchscript_runtime_path=tmp_path / "good" / "adv3b02.torchscript.pt",
            component_dir=tmp_path / "good" / "component_input",
            class_binding_path=tmp_path / "good" / "inputs" / "class_binding.json",
            parity_receipt_path=tmp_path / "good" / "inputs" / "parity.json",
            generation_lock_path=tmp_path / "good" / "inputs" / "generation.json",
            method_lock_path=method_path,
            detached_seal_path=tmp_path / "bad2.seal",
            signing_request_path=tmp_path / "bad2.request",
        )
    assert bundle.is_dir() and kwargs["expected_runtime_sha256"] == sha256_file(
        tmp_path / "good" / "adv3b02.torchscript.pt"
    )


def test_builder_requires_stage2_batch_shape_and_finite_parity_closure(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path / "good")
    parity_path = tmp_path / "good" / "inputs" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["validated_batch_sizes"] = [1, 8, 256]
    _write_json(parity_path, parity)
    method_path = tmp_path / "good" / "inputs" / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["parity_receipt_sha256"] = sha256_file(parity_path)
    _write_json(method_path, method)
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="batch/shape/finite"
    ):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "missing-batch",
            torchscript_runtime_path=(
                tmp_path / "good" / "adv3b02.torchscript.pt"
            ),
            component_dir=tmp_path / "good" / "component_input",
            class_binding_path=(
                tmp_path / "good" / "inputs" / "class_binding.json"
            ),
            parity_receipt_path=parity_path,
            generation_lock_path=(
                tmp_path / "good" / "inputs" / "generation.json"
            ),
            method_lock_path=method_path,
            detached_seal_path=tmp_path / "missing-batch.seal",
            signing_request_path=tmp_path / "missing-batch.request",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_abs_output_delta", 0.0011),
        ("decision_equivalence_verified", False),
        ("max_abs_output_delta_tolerance", 0.002),
        ("parity_device_type", "cpu"),
        ("cudnn_allow_tf32", True),
        ("cudnn_deterministic", False),
        ("deterministic_algorithms_enabled", False),
    ),
)
def test_builder_rejects_runtime_outside_fixed_fp32_parity_policy(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / field
    _fixture(root)
    parity_path = root / "inputs" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity[field] = value
    _write_json(parity_path, parity)
    method_path = root / "inputs" / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["parity_receipt_sha256"] = sha256_file(parity_path)
    _write_json(method_path, method)
    with pytest.raises(
        ADV3B02DeploymentBundleError,
        match="fixed tolerance",
    ):
        build_unsigned_adv3b02_deployment_bundle(
            root / "bad-policy",
            torchscript_runtime_path=root / "adv3b02.torchscript.pt",
            component_dir=root / "component_input",
            class_binding_path=root / "inputs" / "class_binding.json",
            parity_receipt_path=parity_path,
            generation_lock_path=root / "inputs" / "generation.json",
            method_lock_path=method_path,
            detached_seal_path=root / "bad-policy.seal",
            signing_request_path=root / "bad-policy.request",
        )


def test_class_binding_is_semantic_not_file_format_and_order_drift_rejected(
    tmp_path: Path,
) -> None:
    _, _, _, kwargs = _fixture(tmp_path / "good")
    binding_path = tmp_path / "good" / "inputs" / "class_binding.json"
    original_file_sha = sha256_file(binding_path)
    document = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    assert sha256_file(binding_path) != original_file_sha
    assert document["class_handle_binding_sha256"] == kwargs[
        "expected_class_handle_binding_sha256"
    ]
    result = build_unsigned_adv3b02_deployment_bundle(
        tmp_path / "reformatted-bundle",
        torchscript_runtime_path=tmp_path / "good" / "adv3b02.torchscript.pt",
        component_dir=tmp_path / "good" / "component_input",
        class_binding_path=binding_path,
        parity_receipt_path=tmp_path / "good" / "inputs" / "parity.json",
        generation_lock_path=tmp_path / "good" / "inputs" / "generation.json",
        method_lock_path=tmp_path / "good" / "inputs" / "method.json",
        detached_seal_path=tmp_path / "reformatted.seal",
        signing_request_path=tmp_path / "reformatted.request",
    )
    assert result["class_handle_binding_sha256"] == document[
        "class_handle_binding_sha256"
    ]

    rows = document["class_id_to_handle"]
    reversed_handles = [row["class_handle"] for row in reversed(rows)]
    document["class_id_to_handle"] = [
        {"class_index": index, "class_handle": handle}
        for index, handle in enumerate(reversed_handles)
    ]
    document["class_handle_binding_sha256"] = class_handle_binding_sha256(
        reversed_handles
    )
    _write_json(binding_path, document)
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="component/class binding digest drift"
    ):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "reordered-bundle",
            torchscript_runtime_path=tmp_path / "good" / "adv3b02.torchscript.pt",
            component_dir=tmp_path / "good" / "component_input",
            class_binding_path=binding_path,
            parity_receipt_path=tmp_path / "good" / "inputs" / "parity.json",
            generation_lock_path=tmp_path / "good" / "inputs" / "generation.json",
            method_lock_path=tmp_path / "good" / "inputs" / "method.json",
            detached_seal_path=tmp_path / "reordered.seal",
            signing_request_path=tmp_path / "reordered.request",
        )


def test_runtime_structure_rejects_extra_file_and_unsigned_unused_buffer(
    tmp_path: Path,
) -> None:
    traced = torch.jit.trace(_TinyRuntime(), torch.zeros(1, 2))
    extra_runtime = tmp_path / "extra.torchscript.pt"
    torch.jit.save(traced, str(extra_runtime), _extra_files={"secret.bin": b"x"})
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="extra-file member"
    ):
        runtime_structure_receipt(extra_runtime)

    _fixture(tmp_path / "buffered")
    buffered_runtime = tmp_path / "buffered" / "buffered.torchscript.pt"
    buffered = torch.jit.trace(_RuntimeWithUnusedBuffer(), torch.zeros(1, 2))
    torch.jit.save(buffered, str(buffered_runtime))
    buffered_sha = sha256_file(buffered_runtime)
    parity_path = tmp_path / "buffered" / "inputs" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["runtime_sha256"] = buffered_sha
    _write_json(parity_path, parity)
    method_path = tmp_path / "buffered" / "inputs" / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["runtime_sha256"] = buffered_sha
    method["parity_receipt_sha256"] = sha256_file(parity_path)
    _write_json(method_path, method)
    with pytest.raises(
        ADV3B02DeploymentBundleError, match="runtime structure receipt drift"
    ):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "buffered-bad-bundle",
            torchscript_runtime_path=buffered_runtime,
            component_dir=tmp_path / "buffered" / "component_input",
            class_binding_path=tmp_path / "buffered" / "inputs" / "class_binding.json",
            parity_receipt_path=parity_path,
            generation_lock_path=tmp_path / "buffered" / "inputs" / "generation.json",
            method_lock_path=method_path,
            detached_seal_path=tmp_path / "buffered-bad.seal",
            signing_request_path=tmp_path / "buffered-bad.request",
        )


def test_deployment_identifiers_reject_path_and_artifact_tokens(tmp_path: Path) -> None:
    with pytest.raises(ADV3B02DeploymentBundleError, match="non-opaque"):
        class_handle_binding_sha256(["tx-0", "folder/model.pth"])

    _fixture(tmp_path / "method")
    method_path = tmp_path / "method" / "inputs" / "method.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["method_id"] = "folder/model.pt"
    _write_json(method_path, method)
    with pytest.raises(ADV3B02DeploymentBundleError, match="non-opaque"):
        build_unsigned_adv3b02_deployment_bundle(
            tmp_path / "bad-method-bundle",
            torchscript_runtime_path=tmp_path / "method" / "adv3b02.torchscript.pt",
            component_dir=tmp_path / "method" / "component_input",
            class_binding_path=tmp_path / "method" / "inputs" / "class_binding.json",
            parity_receipt_path=tmp_path / "method" / "inputs" / "parity.json",
            generation_lock_path=tmp_path / "method" / "inputs" / "generation.json",
            method_lock_path=method_path,
            detached_seal_path=tmp_path / "bad-method.seal",
            signing_request_path=tmp_path / "bad-method.request",
        )


@pytest.mark.parametrize("bad_handle", [r"C:\domain", "C:/domain"])
def test_domain_handle_rejects_windows_path_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_handle: str,
) -> None:
    domains = [bad_handle, "rx_day:21", "rx_day:22", "rx_day:23"]
    bundle, _, _, kwargs = _fixture(tmp_path, domain_registry=domains)
    monkeypatch.setattr(runtime_trust, "verify_ed25519", lambda *args: None)
    with pytest.raises(ADV3B02DeploymentBundleError, match="non-opaque"):
        load_formal_adv3b02_deployment_bundle(bundle, **kwargs)
