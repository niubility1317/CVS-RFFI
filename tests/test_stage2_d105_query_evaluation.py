from __future__ import annotations

"""Release identity-split traceability record.

R1 data package identity uses data_feature_runtime/data_materialization_lock.
R2 candidate asset identity uses D105 runtime-manifest/candidate-method-lock.
R3 canonical candidate manifest and exact nested lock are read and validated.
R4 both identity planes are bound into prediction/state receipts.
R5 data, implementation entrypoint, DA/HEAD, and matrix tamper fail closed.
R6 one public canonical helper binds plan and evaluator prediction contexts.
R7 signed path-free D92 authority precedes same-root IQ materialization.
R8 one D92 authority commit binds all four packages and all six split receipts.

Each item is pending until its named test passes in the final focused run.
The record is embedded here because this work package may edit only this test
and its evaluator implementation.
"""

import ast
import builtins
from dataclasses import fields, replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi.rxid_metabias4_bundle import (
    build_rxid_metabias4_bundle,
    serialize_rxid_metabias4_bundle,
)
from cvsrffi.somph_predictor_bundle import (
    APPLY_ONLY,
    ENROLLMENT_ONLY,
    FORMAL_LEO_WEAK_SCENARIOS,
)
from cvsrffi.stage2_d105_cbrc import (
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
    make_d105_cbrc_bundle_handle,
)
from cvsrffi.stage2_d105_four_arm import ARMS
from cvsrffi.stage2_d105_query_evaluation import (
    D105Phase1BundleAuthority,
    D105QueryEvaluationContext,
    D105QueryEvaluationError,
    D105SealedPackageRef,
    D105SplitAuthority,
    build_d105_prediction_context,
    evaluate_d105_query_row,
)
import cvsrffi.stage2_d105_query_evaluation as evaluation
import cvsrffi.stage2_d105_phase1_bundle as phase1
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


OLD = tuple(f"old-{index}" for index in range(6))
NEW = tuple(f"new-{index}" for index in range(5))
ALL = OLD + NEW
HASHES = tuple(f"{index:x}" * 64 for index in range(1, 16))


def _canonical_sha(value) -> str:
    import json

    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _bundle(
    checkpoint_sha256: str = HASHES[0],
    candidate_runtime_sha256: str = HASHES[1],
    candidate_lock_sha256: str = HASHES[2],
):
    rng = np.random.default_rng(105)
    basis = np.zeros((32, 160), dtype=np.float32)
    basis[:, :32] = np.eye(32, dtype=np.float32)
    coefficient = rng.normal(0, 0.02, (160, 4)).astype(np.float32)
    codebook = np.zeros((6, 32), dtype=np.float32)
    codebook[:, :6] = np.eye(6, dtype=np.float32)
    targets = rng.normal(0, 0.1, (6, 4)).astype(np.float32)
    return build_rxid_metabias4_bundle(
        basis,
        coefficient,
        codebook,
        targets,
        np.full((6, 4), 4.0, dtype=np.float32),
        np.full(6, 1.5, dtype=np.float32),
        cell_min_physical_count=np.full(6, 2, dtype=np.int16),
        cell_class_count=np.full(6, 3, dtype=np.int16),
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=candidate_runtime_sha256,
        method_lock_sha256=candidate_lock_sha256,
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def _candidate_identity(tmp_path: Path, checkpoint_sha256: str):
    code_root = Path(evaluation.__file__).resolve().parents[1]
    core_files = phase1.D105_CANDIDATE_RUNTIME_FILES
    runtime = {
        "schema": "cvs.stage2.d105.candidate_runtime_manifest.v1",
        "candidate_id": "D105-CBRC+LPO-RC",
        "protocol_schema": "p2_min_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "entrypoints": dict(phase1.D105_CANDIDATE_RUNTIME_ENTRYPOINTS),
        "core_file_sha256": {
            name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
            for name in core_files
        },
    }
    runtime_path = tmp_path / "candidate_runtime_manifest.json"
    runtime_path.write_bytes(evaluation._canonical_bytes(runtime))
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    lock = {
        "schema": "cvs.stage2.d105.candidate_method_lock.v1",
        "candidate_id": "D105-CBRC+LPO-RC",
        "protocol_schema": "p2_min_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "d105_candidate_runtime_manifest_sha256": runtime_sha,
        "d105_cbrc": {
            "semantic_revision": "cbrc_mb4_task_balanced_huber_irls4_fp16_v1",
            "code_dim": 4,
            "domain_dim": 32,
            "allowed_k": [1, 5, 10],
            "irls_steps": 4,
            "old_new_task_mass": [0.5, 0.5],
            "k1_zero_coefficient": True,
            "ground_old_multiprototype_enabled": False,
            "deployment_coefficient_dtype": "float16",
            "query_transform": "relu_l2norm_pre_relu_plus_mb4",
            "query_state_updates": 0,
        },
        "student_t_qknn": {
            "student_nu": 3.0,
            "kernel_effective_dim": 12,
            "kernel_volume_gamma": 1.0,
            "shared_h0": 0.35,
            "scale_prior_strength": 2.0,
            "scale_min_ratio": 0.5,
            "scale_max_ratio": 2.0,
            "temperature": 0.85,
            "support_storage": "int8_fp16_scale",
        },
        "four_arm": {
            "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
            "same_da_state_for_da_and_joint": True,
            "same_head_code_config_for_head_and_joint": True,
            "query_truth_surface": False,
        },
        "source_held": {
            "receiver_held_k": [1, 5, 10],
            "class_loco_k": 1,
            "tx_probe_algorithm": "receiver_held_ridge_l2_0.01",
            "tx_probe_max_balanced_accuracy": 0.25,
            "int8_min_top1_agreement": 0.995,
            "large_margin_minimum": 0.10,
            "large_margin_flip_max": 0,
            "truth_open_after_prediction": True,
        },
        "target25": {
            "seed": 713102,
            "claim_scope": "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE",
            "formal_launch_authority": False,
            "slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
            "leo_scenarios": [
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            ],
            "outer_row_count": 25,
            "scenario_arm_pair_count": 300,
            "state_prediction_surface_count": 600,
        },
    }
    lock_path = tmp_path / "candidate_method_lock.json"
    lock_path.write_bytes(evaluation._canonical_bytes(lock))
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return runtime_path, runtime_sha, runtime, lock_path, lock_sha, lock


def _lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )


def _support_payload(classes: tuple[str, ...], scenario_index: int):
    rows = len(classes)
    iq = np.zeros((rows, 2, 8), dtype=np.float32)
    for index in range(rows):
        iq[index, 0, :] = np.float32(0.03 * (index + 1))
        iq[index, 1, :] = np.float32(0.01 * (scenario_index + 1))
    tokens = np.asarray(
        [f"sid_{scenario_index}_{label}_0" for label in classes], dtype="<U64"
    )
    return {
        "support_leo_weak_iq": iq,
        "support_class_indices": np.arange(rows, dtype=np.int64),
        "support_rank_within_class": np.zeros(rows, dtype=np.int64),
        "support_tokens": tokens,
        "support_overlay_tokens": np.asarray(
            [f"overlay_{value}" for value in tokens], dtype="<U96"
        ),
        "support_satellite_seeds": np.arange(rows, dtype=np.int64),
        "support_post_channel_iq_sha256": np.asarray(
            [hashlib.sha256(row.tobytes()).hexdigest() for row in iq], dtype="<U64"
        ),
        "manifest_json": np.asarray(["{}"], dtype="<U2"),
    }


def _query_payload(
    classes: tuple[str, ...],
    scenario_index: int,
    *,
    prefix_iq: np.ndarray | None = None,
    prefix_tokens: tuple[str, ...] = (),
):
    new_count = len(classes)
    iq = np.zeros((len(prefix_tokens) + new_count, 2, 8), dtype=np.float32)
    if prefix_iq is not None:
        iq[: len(prefix_tokens)] = prefix_iq
    for index in range(new_count):
        row = len(prefix_tokens) + index
        iq[row, 0, :] = np.float32(0.04 * (index + 1))
        iq[row, 1, :] = np.float32(0.02 * (scenario_index + 1))
    tokens = prefix_tokens + tuple(
        f"qid_{scenario_index}_{label}" for label in classes
    )
    return {
        "query_leo_weak_iq": iq,
        "query_tokens": np.asarray(tokens, dtype="<U64"),
        "query_overlay_tokens": np.asarray(
            [f"overlay_{value}" for value in tokens], dtype="<U96"
        ),
        "query_satellite_seeds": np.arange(len(tokens), dtype=np.int64),
        "query_post_channel_iq_sha256": np.asarray(
            [hashlib.sha256(row.tobytes()).hexdigest() for row in iq], dtype="<U64"
        ),
        "manifest_json": np.asarray(["{}"], dtype="<U2"),
    }


def _manifest(
    *,
    profile: str,
    state: str,
    stage: str,
    classes: tuple[str, ...],
    checkpoint_sha256: str,
):
    return {
        "profile": profile,
        "registration_state": state,
        "stage": stage,
        "receiver": "20-1",
        "seed": 392002,
        "k_shot": 1,
        "phase1_checkpoint_sha256": checkpoint_sha256,
        "feature_runtime_sha256": HASHES[1],
        "method_lock_sha256": HASHES[2],
        "registered_classes": [{"class_handle": value} for value in classes],
        "row_handle": None if profile == ENROLLMENT_ONLY else "row-1",
        "row_manifest_sha256": None if profile == ENROLLMENT_ONLY else HASHES[8],
        "package_root_sha256": _canonical_sha([profile, state]),
    }


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"exact-checkpoint-test-bytes")
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (
        runtime_path,
        candidate_runtime_sha,
        candidate_runtime,
        lock_path,
        candidate_lock_sha,
        candidate_lock,
    ) = _candidate_identity(tmp_path, checkpoint_sha)
    bundle = _bundle(
        checkpoint_sha,
        candidate_runtime_sha,
        candidate_lock_sha,
    )
    wire = serialize_rxid_metabias4_bundle(bundle)
    receipt_root = compute_d105_bundle_receipt_root(bundle)
    validated_id = HASHES[9]
    validator_receipt = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=validated_id,
        expected_content_root_sha256=bundle.content_root_sha256,
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=candidate_runtime_sha,
        method_lock_sha256=candidate_lock_sha,
        receipt_root_sha256=receipt_root,
    )
    before_support = {}
    before_query = {}
    after_support = {}
    after_query = {}
    split_authorities = []
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        before_support[scenario] = _support_payload(OLD, scenario_index)
        before_query[scenario] = _query_payload(OLD, scenario_index)
        old_iq = before_query[scenario]["query_leo_weak_iq"]
        old_tokens = tuple(before_query[scenario]["query_tokens"].tolist())
        after_support[scenario] = _support_payload(ALL, scenario_index + 10)
        after_query[scenario] = _query_payload(
            NEW,
            scenario_index,
            prefix_iq=old_iq,
            prefix_tokens=old_tokens,
        )
        for state, support, query in (
            ("BEFORE_REGISTRATION", before_support[scenario], before_query[scenario]),
            ("AFTER_REGISTRATION", after_support[scenario], after_query[scenario]),
        ):
            support_tokens = tuple(support["support_tokens"].tolist())
            query_tokens = tuple(query["query_tokens"].tolist())
            split_authorities.append(
                D105SplitAuthority(
                    registration_state=state,
                    scenario=scenario,
                    capsule_id=_canonical_sha([state, scenario, "capsule"]),
                    split_id=_canonical_sha([state, scenario, "split"]),
                    validator_receipt_sha256=_canonical_sha(
                        [state, scenario, "validator"]
                    ),
                    support_token_root_sha256=_canonical_sha(
                        sorted(support_tokens)
                    ),
                    query_token_root_sha256=_canonical_sha(sorted(query_tokens)),
                )
            )
    manifests = (
        _manifest(
            profile=ENROLLMENT_ONLY,
            state="before",
            stage="stage2b",
            classes=OLD,
            checkpoint_sha256=checkpoint_sha,
        ),
        _manifest(
            profile=APPLY_ONLY,
            state="before",
            stage="stage2b",
            classes=OLD,
            checkpoint_sha256=checkpoint_sha,
        ),
        _manifest(
            profile=ENROLLMENT_ONLY,
            state="after",
            stage="stage2c",
            classes=ALL,
            checkpoint_sha256=checkpoint_sha,
        ),
        _manifest(
            profile=APPLY_ONLY,
            state="after",
            stage="stage2c",
            classes=ALL,
            checkpoint_sha256=checkpoint_sha,
        ),
    )
    packages = (before_support, before_query, after_support, after_query)
    split_authority_commit = _canonical_sha(["D92", "split-authority"])
    split_authorities = [
        replace(
            authority,
            validator_receipt_sha256=split_authority_commit,
        )
        for authority in split_authorities
    ]
    refs = tuple(
        D105SealedPackageRef(
            package_root=f"package-{index}",
            detached_seal_path=f"seal-{index}",
            expected_seal_sha256=f"{index + 1:x}" * 64,
            formal_policy_path=f"policy-{index}.json",
            formal_policy_authorization_path=f"authorization-{index}.json",
            signed_policy_authorization_envelope_path=f"envelope-{index}.json",
            expected_signed_policy_authorization_envelope_sha256=(
                f"{index + 5:x}" * 64
            ),
        )
        for index in range(4)
    )
    by_root = {}
    for index, (ref, payload, manifest) in enumerate(
        zip(refs, packages, manifests, strict=True)
    ):
        by_root[str(ref.package_root)] = (
            payload,
            manifest,
            {
                "schema": "test.authorized_materialization.v1",
                "authority_commit_sha256": split_authority_commit,
                "package_root_sha256": manifest["package_root_sha256"],
                "package_detached_seal_sha256": ref.expected_seal_sha256,
                "signed_policy_authorization_envelope_sha256": (
                    ref.expected_signed_policy_authorization_envelope_sha256
                ),
                "receiver": manifest["receiver"],
                "seed": manifest["seed"],
                "stage": manifest["stage"],
                "registration_state": manifest["registration_state"],
                "k_shot": manifest["k_shot"],
            },
        )

    def package_loader(ref):
        return by_root[str(ref.package_root)]

    model_calls = []

    class DummyModel(torch.nn.Module):
        def forward(self, value):
            return value

    def model_loader(raw, input_len, device):
        model_calls.append((raw, input_len, str(device)))
        return DummyModel().eval(), {
            "loader": "fake_exact_loader",
            "checkpoint_load_strict": True,
            "input_len": input_len,
        }

    tap_calls = []

    def feature_extractor(model, received_iq):
        values = received_iq.detach().cpu().numpy()
        tap_calls.append(values.copy())
        rows = len(values)
        pre = np.zeros((rows, 160), dtype=np.float32)
        domain = np.zeros((rows, 160), dtype=np.float32)
        signal = values.mean(axis=(1, 2))
        for index in range(rows):
            pre[index, :] = np.float32(0.03)
            pre[index, index % 16] += np.float32(0.8 + signal[index])
            domain[index, index % 6] = np.float32(1.0)
            domain[index, 10] = signal[index]
        receipt = hashlib.sha256(values.tobytes()).hexdigest()
        return SimpleNamespace(pre_relu=pre, z_dom=domain, receipt_sha256=receipt)

    authority = D105Phase1BundleAuthority(
        bundle_dir=tmp_path / "formal-phase1-asset",
        manifest_sha256=HASHES[10],
        bundle_wire_sha256=hashlib.sha256(wire).hexdigest(),
        validated_bundle_id_sha256=validated_id,
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=bundle.content_root_sha256,
        checkpoint_sha256=checkpoint_sha,
        candidate_runtime_manifest_path=runtime_path,
        candidate_method_lock_path=lock_path,
        d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
        d105_candidate_method_lock_sha256=candidate_lock_sha,
    )
    context = D105QueryEvaluationContext(
        before_enrollment=refs[0],
        before_apply=refs[1],
        after_enrollment=refs[2],
        after_apply=refs[3],
        split_authorities=tuple(split_authorities),
        phase1_bundle=authority,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        data_feature_runtime_sha256=HASHES[1],
        data_materialization_lock_sha256=HASHES[2],
        qknn_lock=_lock(1),
        device="cpu",
        feature_batch_size=32,
        score_chunk_size=3,
    )
    asset = SimpleNamespace(
        bundle=bundle,
        manifest={"bundle_wire_sha256": hashlib.sha256(wire).hexdigest()},
        manifest_sha256=HASHES[10],
        formal_phase2_eligible=True,
        validated_bundle_id_sha256=validated_id,
        validator_receipt_sha256=validator_receipt,
    )
    asset.manifest.update(
        {
            "d105_candidate_runtime_manifest_sha256": candidate_runtime_sha,
            "d105_candidate_method_lock_sha256": candidate_lock_sha,
        }
    )
    handle = make_d105_cbrc_bundle_handle(
        bundle,
        validated_bundle_id_sha256=validated_id,
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=bundle.content_root_sha256,
    )
    return (
        context,
        package_loader,
        model_loader,
        feature_extractor,
        model_calls,
        tap_calls,
        asset,
        handle,
        candidate_runtime,
        candidate_lock,
    )


def _patch_formal_asset(monkeypatch, asset, handle):
    calls = []

    def load(path, *, require_formal_phase2_eligible=False):
        calls.append((Path(path), require_formal_phase2_eligible))
        if require_formal_phase2_eligible is not True:
            raise AssertionError("evaluator did not require formal Phase2 eligibility")
        return asset

    monkeypatch.setattr(evaluation, "load_d105_phase1_asset", load)
    monkeypatch.setattr(
        evaluation, "make_d105_phase1_runtime_handle", lambda value: handle
    )
    return calls


def test_default_evaluator_loader_is_closed_to_bound_d105_model_factory(
    monkeypatch,
) -> None:
    """The evaluator must not recover the generic SSDG training stack."""

    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any(
        name == "SSDG"
        or name.startswith("SSDG.")
        or "checkpoint_loading" in name
        for name in imported_modules
    )
    assert "build_exact_ssdg_model_from_checkpoint" not in called_names
    assert "build_d105_exact_model_from_checkpoint" in called_names

    checkpoint_buffer = io.BytesIO()
    torch.save({"args": {}, "model": {}}, checkpoint_buffer)
    factory_calls: list[tuple[object, int, str]] = []

    class BoundModel(torch.nn.Module):
        def forward(self, value):
            return value

    def bound_factory(checkpoint, *, input_len, device):
        factory_calls.append((checkpoint, input_len, str(device)))
        return BoundModel(), {
            "loader": "d105_bound_test_factory",
            "model_factory": "model_dual_cvsincnet.build_dual_model",
            "checkpoint_load_strict": True,
        }

    monkeypatch.setattr(
        evaluation, "build_d105_exact_model_from_checkpoint", bound_factory
    )
    original_import = builtins.__import__
    blocked_imports: list[str] = []

    def reject_legacy_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "SSDG"
            or name.startswith("SSDG.")
            or name == "cvsrffi.checkpoint_loading"
        ):
            blocked_imports.append(name)
            raise AssertionError("legacy checkpoint-loading stack is forbidden")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_legacy_import)
    model, audit = evaluation._default_model_loader(
        checkpoint_buffer.getvalue(), 8, torch.device("cpu")
    )
    assert blocked_imports == []
    assert len(factory_calls) == 1
    assert factory_calls[0][0] == {"args": {}, "model": {}}
    assert factory_calls[0][1:] == (8, "cpu")
    assert isinstance(model, torch.nn.Module) and not model.training
    assert audit["model_factory"] == "model_dual_cvsincnet.build_dual_model"


def test_evaluator_import_path_cannot_open_legacy_ssdg_stack() -> None:
    """Check a fresh interpreter, before cached imports can mask a reachability bug."""

    source_root = Path(evaluation.__file__).resolve().parents[1]
    probe = """
import builtins

_original_import = builtins.__import__

def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'SSDG' or name.startswith('SSDG.') or name == 'cvsrffi.checkpoint_loading':
        raise RuntimeError('legacy D105 evaluator dependency: ' + name)
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guard
import cvsrffi.stage2_d105_query_evaluation  # noqa: F401
print('D105_EVALUATOR_IMPORT_CLOSED')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "D105_EVALUATOR_IMPORT_CLOSED"


def test_model_loading_waits_for_complete_package_authority_and_context_validation(
    tmp_path: Path, monkeypatch
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        model_calls,
        _tap_calls,
        asset,
        handle,
        _candidate_runtime,
        _candidate_lock,
    ) = _fixture(tmp_path)
    events: list[str] = []

    def tracked_package_loader(reference):
        loaded = package_loader(reference)
        events.append(f"package:{reference.package_root}")
        return loaded

    original_candidate_identity = evaluation._load_candidate_identity
    original_qknn_identity = evaluation._validate_qknn_lock_identity
    original_package_binding = evaluation._package_binding
    original_payload_preflight = evaluation._preflight_d105_model_inputs
    original_regular_bytes = evaluation._regular_bytes
    original_device = evaluation._device

    def tracked_candidate_identity(value):
        result = original_candidate_identity(value)
        events.append("candidate_identity")
        return result

    def tracked_qknn_identity(lock, candidate_lock):
        result = original_qknn_identity(lock, candidate_lock)
        events.append("qknn_context")
        return result

    def tracked_phase1_asset(path, *, require_formal_phase2_eligible=False):
        assert require_formal_phase2_eligible is True
        events.append("phase1_asset")
        return asset

    def tracked_package_binding(references, manifests, audits):
        result = original_package_binding(references, manifests, audits)
        events.append("package_authority")
        return result

    def tracked_payload_preflight(*args, **kwargs):
        result = original_payload_preflight(*args, **kwargs)
        events.append("sealed_payloads")
        return result

    def tracked_regular_bytes(path, *, name):
        result = original_regular_bytes(path, name=name)
        if name == "exact Phase1 checkpoint":
            events.append("checkpoint_context")
        return result

    def tracked_device(value):
        result = original_device(value)
        events.append("device_context")
        return result

    def observed_model_loader(raw, input_len, device):
        required = {
            "candidate_identity",
            "qknn_context",
            "phase1_asset",
            "package_authority",
            "sealed_payloads",
            "checkpoint_context",
            "device_context",
        }
        assert required.issubset(events)
        assert len([event for event in events if event.startswith("package:")]) == 4
        events.append("model")
        return model_loader(raw, input_len, device)

    monkeypatch.setattr(evaluation, "_load_candidate_identity", tracked_candidate_identity)
    monkeypatch.setattr(
        evaluation, "_validate_qknn_lock_identity", tracked_qknn_identity
    )
    monkeypatch.setattr(evaluation, "load_d105_phase1_asset", tracked_phase1_asset)
    monkeypatch.setattr(
        evaluation, "make_d105_phase1_runtime_handle", lambda value: handle
    )
    monkeypatch.setattr(evaluation, "_package_binding", tracked_package_binding)
    monkeypatch.setattr(
        evaluation, "_preflight_d105_model_inputs", tracked_payload_preflight
    )
    monkeypatch.setattr(evaluation, "_regular_bytes", tracked_regular_bytes)
    monkeypatch.setattr(evaluation, "_device", tracked_device)

    result = evaluate_d105_query_row(
        context,
        package_loader=tracked_package_loader,
        model_loader=observed_model_loader,
        feature_extractor=extractor,
    )
    assert result.scenario_pairs
    assert events.index("model") > max(
        events.index(name)
        for name in (
            "candidate_identity",
            "qknn_context",
            "phase1_asset",
            "package_authority",
            "sealed_payloads",
            "checkpoint_context",
            "device_context",
        )
    )


def test_real_iq_evaluator_emits_three_complete_before_after_four_arm_pairs(
    tmp_path: Path, monkeypatch,
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        model_calls,
        tap_calls,
        asset,
        handle,
        _candidate_runtime,
        _candidate_lock,
    ) = (
        _fixture(tmp_path)
    )
    asset_calls = _patch_formal_asset(monkeypatch, asset, handle)
    result = evaluate_d105_query_row(
        context,
        package_loader=package_loader,
        model_loader=model_loader,
        feature_extractor=extractor,
    )
    assert len(model_calls) == 1
    assert asset_calls == [(Path(context.phase1_bundle.bundle_dir), True)]
    assert model_calls[0][1:] == (8, "cpu")
    assert tap_calls and all(call.dtype == np.float32 for call in tap_calls)
    assert tuple(pair.scenario for pair in result.scenario_pairs) == tuple(
        FORMAL_LEO_WEAK_SCENARIOS
    )
    for pair in result.scenario_pairs:
        assert pair.before.stage == "S_B"
        assert pair.after.stage == "S_C"
        assert pair.before.registered_classes == OLD
        assert pair.after.registered_classes == ALL
        assert set(pair.before.query_physical_ids).issubset(
            pair.after.query_physical_ids
        )
        for state in (pair.before, pair.after):
            assert tuple(state.arm_predictions) == ARMS
            assert all(isinstance(state.arm_predictions[arm], tuple) for arm in ARMS)
            assert state.arm_prediction_sha256["M_HEAD"] == state.arm_prediction_sha256["M0"]
            assert state.arm_prediction_sha256["M_JOINT"] == state.arm_prediction_sha256["M_DA"]
            assert state.logit_sha256["M_HEAD"] == state.logit_sha256["M0"]
            assert state.logit_sha256["M_JOINT"] == state.logit_sha256["M_DA"]
            assert (
                state.data_feature_runtime_sha256
                == context.data_feature_runtime_sha256
            )
            assert (
                state.data_materialization_lock_sha256
                == context.data_materialization_lock_sha256
            )
            assert (
                state.d105_candidate_runtime_manifest_sha256
                == context.phase1_bundle.d105_candidate_runtime_manifest_sha256
            )
            assert (
                state.d105_candidate_method_lock_sha256
                == context.phase1_bundle.d105_candidate_method_lock_sha256
            )
            assert (
                state.data_feature_runtime_sha256
                != state.d105_candidate_runtime_manifest_sha256
            )
        assert (
            result.state_prediction(pair.scenario, "BEFORE_REGISTRATION")
            is pair.before
        )
    first = result.scenario_pairs[0].before
    before_support = package_loader(context.before_enrollment)[0][first.scenario]
    request = SimpleNamespace(
        scenario=first.scenario,
        registration_state=first.registration_state,
        receiver=result.receiver,
        seed=result.seed,
        k_shot=result.k_shot,
        stage=first.stage,
        capsule_id=first.capsule_id,
        split_id=first.split_id,
        authority_receipt_sha256=first.split_validator_receipt_sha256,
        support_physical_ids=tuple(before_support["support_tokens"].tolist()),
        query_physical_ids=first.query_physical_ids,
        registered_classes=first.registered_classes,
        old_classes=result.old_classes,
        new_classes=(),
        prediction_context_sha256=first.prediction_context_sha256,
        data_feature_runtime_sha256=first.data_feature_runtime_sha256,
        data_materialization_lock_sha256=(
            first.data_materialization_lock_sha256
        ),
        d105_candidate_runtime_manifest_sha256=(
            first.d105_candidate_runtime_manifest_sha256
        ),
        d105_candidate_method_lock_sha256=(
            first.d105_candidate_method_lock_sha256
        ),
    )
    runner_output = result.target25_output_for(request)
    assert runner_output.stage == "S_B"
    assert runner_output.registration_state == "BEFORE_REGISTRATION"
    assert tuple(runner_output.arm_predictions) == ARMS


def test_query_truth_surface_and_split_root_tamper_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        model_calls,
        _tap_calls,
        asset,
        handle,
        _candidate_runtime,
        _candidate_lock,
    ) = _fixture(tmp_path)
    _patch_formal_asset(monkeypatch, asset, handle)
    original = package_loader

    def truth_loader(ref):
        payloads, manifest, audit = original(ref)
        if ref is context.after_apply:
            copied = {name: dict(value) for name, value in payloads.items()}
            first = FORMAL_LEO_WEAK_SCENARIOS[0]
            copied[first]["query_truth"] = np.asarray(["forbidden"])
            return copied, manifest, audit
        return payloads, manifest, audit

    with pytest.raises(D105QueryEvaluationError, match="truth/role"):
        evaluate_d105_query_row(
            context,
            package_loader=truth_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
    assert model_calls == []
    authorities = list(context.split_authorities)
    authorities[0] = replace(authorities[0], query_token_root_sha256="f" * 64)
    tampered = replace(context, split_authorities=tuple(authorities))
    with pytest.raises(D105QueryEvaluationError, match="token-root"):
        evaluate_d105_query_row(
            tampered,
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
    assert model_calls == []


def test_public_prediction_context_helper_is_the_unique_plan_evaluator_hash(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    context, package_loader, model_loader, extractor = fixture[:4]
    asset, handle = fixture[6:8]
    _patch_formal_asset(monkeypatch, asset, handle)
    result = evaluate_d105_query_row(
        context,
        package_loader=package_loader,
        model_loader=model_loader,
        feature_extractor=extractor,
    )
    state = result.scenario_pairs[0].before
    package_roots = {
        name: package_loader(getattr(context, name))[1][
            "package_root_sha256"
        ]
        for name in evaluation.PACKAGE_ROOT_KEYS
    }
    kwargs = {
        "registration_state": state.registration_state,
        "stage": state.stage,
        "scenario": state.scenario,
        "receiver": result.receiver,
        "seed": result.seed,
        "active_k": result.k_shot,
        "registered_classes": state.registered_classes,
        "capsule_id": state.capsule_id,
        "split_id": state.split_id,
        "split_validator_receipt_sha256": (
            state.split_validator_receipt_sha256
        ),
        "support_physical_root_sha256": (
            state.support_physical_root_sha256
        ),
        "query_physical_root_sha256": state.query_physical_root_sha256,
        "package_root_sha256": package_roots,
        "phase1_bundle_manifest_sha256": (
            context.phase1_bundle.manifest_sha256
        ),
        "validated_bundle_id_sha256": (
            context.phase1_bundle.validated_bundle_id_sha256
        ),
        "bundle_content_root_sha256": (
            context.phase1_bundle.expected_content_root_sha256
        ),
        "bundle_validator_receipt_sha256": (
            context.phase1_bundle.validator_receipt_sha256
        ),
        "checkpoint_sha256": context.checkpoint_sha256,
        "data_feature_runtime_sha256": (
            context.data_feature_runtime_sha256
        ),
        "data_materialization_lock_sha256": (
            context.data_materialization_lock_sha256
        ),
        "d105_candidate_runtime_manifest_sha256": (
            context.phase1_bundle.d105_candidate_runtime_manifest_sha256
        ),
        "d105_candidate_method_lock_sha256": (
            context.phase1_bundle.d105_candidate_method_lock_sha256
        ),
        "qknn_lock_digest": context.qknn_lock.lock_digest,
    }
    payload, expected_sha = build_d105_prediction_context(**kwargs)
    assert expected_sha == state.prediction_context_sha256
    assert payload["query_truth_present"] is False
    assert payload["query_rows_used_for_fit"] == 0
    for field, replacement in (
        ("split_id", "d" * 64),
        ("phase1_bundle_manifest_sha256", "e" * 64),
        ("qknn_lock_digest", "f" * 64),
    ):
        changed = dict(kwargs)
        changed[field] = replacement
        assert build_d105_prediction_context(**changed)[1] != expected_sha
    changed = dict(kwargs)
    changed_roots = dict(package_roots)
    changed_roots["before_apply"] = "a" * 64
    changed["package_root_sha256"] = changed_roots
    assert build_d105_prediction_context(**changed)[1] != expected_sha


def test_self_reported_phase1_hashes_cannot_authorize_missing_asset(
    tmp_path: Path,
) -> None:
    context, package_loader, model_loader, extractor, *_ = _fixture(tmp_path)
    assert not Path(context.phase1_bundle.bundle_dir).exists()
    with pytest.raises(D105QueryEvaluationError, match="formal.*validation"):
        evaluate_d105_query_row(
            context,
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )


def test_public_predictor_surface_has_no_truth_role_or_quota_fields() -> None:
    for record_type in (
        D105QueryEvaluationContext,
        evaluation.D105StatePrediction,
    ):
        names = tuple(field.name.lower() for field in fields(record_type))
        assert not any(
            forbidden in name
            for name in names
            for forbidden in ("truth", "role", "quota", "class_count_hint")
        )


def test_wrong_package_authority_commit_fails_before_feature_extraction(
    tmp_path: Path,
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        model_calls,
        tap_calls,
        *_rest,
    ) = _fixture(tmp_path)

    def wrong_commit_loader(ref):
        payload, manifest, audit = package_loader(ref)
        return (
            payload,
            manifest,
            {
                **audit,
                "authority_commit_sha256": "f" * 64,
            },
        )

    with pytest.raises(
        D105QueryEvaluationError,
        match="authority commit/split validator receipt",
    ):
        evaluate_d105_query_row(
            context,
            package_loader=wrong_commit_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
    assert model_calls == []
    assert tap_calls == []


def test_checkpoint_context_tamper_fails_before_model_loading(
    tmp_path: Path, monkeypatch
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        model_calls,
        tap_calls,
        asset,
        handle,
        _candidate_runtime,
        _candidate_lock,
    ) = _fixture(tmp_path)
    _patch_formal_asset(monkeypatch, asset, handle)
    Path(context.checkpoint_path).write_bytes(b"tampered-checkpoint-bytes")
    with pytest.raises(D105QueryEvaluationError, match="checkpoint SHA256 drift"):
        evaluate_d105_query_row(
            context,
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
    assert model_calls == []
    assert tap_calls == []


def test_default_loader_preflights_before_same_package_materialization(
    monkeypatch,
) -> None:
    ref = D105SealedPackageRef(
        package_root="sealed-package",
        detached_seal_path="package.seal",
        expected_seal_sha256="1" * 64,
        formal_policy_path="formal-policy.json",
        formal_policy_authorization_path="authorization.json",
        signed_policy_authorization_envelope_path="envelope.json",
        expected_signed_policy_authorization_envelope_sha256="2" * 64,
    )
    manifest = {
        "package_root_sha256": "3" * 64,
        "receiver": "20-1",
        "seed": 713102,
        "stage": "stage2b",
        "registration_state": "before",
        "k_shot": 1,
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def authority(*args, **kwargs):
        calls.append(("authority", args, kwargs))
        return (
            manifest,
            {},
            {
                "signed_path_free_runtime_authorization_verified": True,
                "iq_open_authorized": True,
                "authority_commit_sha256": "4" * 64,
                "package_root_sha256": manifest["package_root_sha256"],
                "package_detached_seal_sha256": ref.expected_seal_sha256,
                "signed_policy_authorization_envelope_sha256": (
                    ref.expected_signed_policy_authorization_envelope_sha256
                ),
            },
        )

    def materialize(*args, **kwargs):
        calls.append(("materialize", args, kwargs))
        return ({"leo_satellite_weak": {}}, dict(manifest), {"diagnostic_only": True})

    monkeypatch.setattr(
        evaluation,
        "preflight_somph_predictor_bundle_with_authority",
        authority,
    )
    monkeypatch.setattr(
        evaluation,
        "load_verified_somph_predictor_bundle",
        materialize,
    )
    _payloads, loaded_manifest, audit = evaluation._default_package_loader(ref)
    assert [call[0] for call in calls] == ["authority", "materialize"]
    for _, args, kwargs in calls:
        assert args == (ref.package_root,)
        assert kwargs["detached_seal_path"] == ref.detached_seal_path
        assert kwargs["expected_seal_sha256"] == ref.expected_seal_sha256
    assert loaded_manifest == manifest
    assert audit["authority_commit_sha256"] == "4" * 64
    assert audit["package_root_sha256"] == manifest["package_root_sha256"]
    assert audit["package_detached_seal_sha256"] == ref.expected_seal_sha256
    assert audit["receiver"] == manifest["receiver"]
    assert audit["seed"] == manifest["seed"]
    assert audit["stage"] == manifest["stage"]
    assert audit["registration_state"] == manifest["registration_state"]
    assert audit["k_shot"] == manifest["k_shot"]


@pytest.mark.parametrize("tamper", ["commit", "envelope", "package"])
def test_default_loader_rejects_wrong_authority_commit_envelope_or_package(
    monkeypatch, tamper: str
) -> None:
    ref = D105SealedPackageRef(
        package_root="sealed-package",
        detached_seal_path="package.seal",
        expected_seal_sha256="1" * 64,
        formal_policy_path="formal-policy.json",
        formal_policy_authorization_path="authorization.json",
        signed_policy_authorization_envelope_path="envelope.json",
        expected_signed_policy_authorization_envelope_sha256="2" * 64,
    )
    manifest = {
        "package_root_sha256": "3" * 64,
        "receiver": "20-1",
        "seed": 713102,
        "stage": "stage2b",
        "registration_state": "before",
        "k_shot": 1,
    }
    authority_audit = {
        "signed_path_free_runtime_authorization_verified": True,
        "iq_open_authorized": True,
        "authority_commit_sha256": (
            "not-a-sha" if tamper == "commit" else "4" * 64
        ),
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": ref.expected_seal_sha256,
        "signed_policy_authorization_envelope_sha256": (
            "5" * 64
            if tamper == "envelope"
            else ref.expected_signed_policy_authorization_envelope_sha256
        ),
    }
    materialized_manifest = dict(manifest)
    if tamper == "package":
        materialized_manifest["package_root_sha256"] = "6" * 64
    monkeypatch.setattr(
        evaluation,
        "preflight_somph_predictor_bundle_with_authority",
        lambda *_args, **_kwargs: (manifest, {}, authority_audit),
    )
    monkeypatch.setattr(
        evaluation,
        "load_verified_somph_predictor_bundle",
        lambda *_args, **_kwargs: ({}, materialized_manifest, {}),
    )
    with pytest.raises(D105QueryEvaluationError):
        evaluation._default_package_loader(ref)


def test_data_runtime_and_package_identity_tamper_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    (
        context,
        package_loader,
        model_loader,
        extractor,
        _model_calls,
        _tap_calls,
        asset,
        handle,
        _candidate_runtime,
        _candidate_lock,
    ) = _fixture(tmp_path)
    _patch_formal_asset(monkeypatch, asset, handle)
    with pytest.raises(D105QueryEvaluationError, match="package/context"):
        evaluate_d105_query_row(
            replace(context, data_feature_runtime_sha256="f" * 64),
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )

    def package_tamper(ref):
        payload, manifest, audit = package_loader(ref)
        changed = dict(manifest)
        changed["method_lock_sha256"] = "e" * 64
        return payload, changed, audit

    with pytest.raises(D105QueryEvaluationError, match="package/context"):
        evaluate_d105_query_row(
            context,
            package_loader=package_tamper,
            model_loader=model_loader,
            feature_extractor=extractor,
        )


def test_candidate_entrypoint_and_core_source_manifest_tamper_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    context, package_loader, model_loader, extractor = fixture[:4]
    asset, handle = fixture[6:8]
    runtime_path = Path(
        context.phase1_bundle.candidate_runtime_manifest_path
    )
    arbitrary = replace(
        context.phase1_bundle,
        d105_candidate_runtime_manifest_sha256="f" * 64,
    )
    _patch_formal_asset(monkeypatch, asset, handle)
    with pytest.raises(
        D105QueryEvaluationError, match="implementation/lock SHA256 drift"
    ):
        evaluate_d105_query_row(
            replace(context, phase1_bundle=arbitrary),
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["entrypoints"]["query_evaluator"] = (
        "cvsrffi.stage2_d105_query_evaluation:wrong_entrypoint"
    )
    runtime_path.write_bytes(evaluation._canonical_bytes(runtime))
    with pytest.raises(
        D105QueryEvaluationError, match="implementation identity validation"
    ):
        evaluate_d105_query_row(
            context,
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )

    fixture = _fixture(tmp_path / "source")
    context, package_loader, model_loader, extractor = fixture[:4]
    asset, handle = fixture[6:8]
    runtime_path = Path(
        context.phase1_bundle.candidate_runtime_manifest_path
    )
    lock_path = Path(context.phase1_bundle.candidate_method_lock_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["core_file_sha256"][
        "cvsrffi/stage2_d105_query_evaluation.py"
    ] = "d" * 64
    runtime_path.write_bytes(evaluation._canonical_bytes(runtime))
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["d105_candidate_runtime_manifest_sha256"] = runtime_sha
    lock_path.write_bytes(evaluation._canonical_bytes(lock))
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    authority = replace(
        context.phase1_bundle,
        d105_candidate_runtime_manifest_sha256=runtime_sha,
        d105_candidate_method_lock_sha256=lock_sha,
    )
    _patch_formal_asset(monkeypatch, asset, handle)
    with pytest.raises(
        D105QueryEvaluationError, match="implementation identity validation"
    ):
        evaluate_d105_query_row(
            replace(context, phase1_bundle=authority),
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )


def test_target25_tap_rows_bypass_rejected_torch_from_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target25 must use the same N607-safe input bridge as Phase1 strict tap."""

    class _Model(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    source = np.arange(48, dtype=np.float32).reshape(3, 2, 8)
    observed: list[np.ndarray] = []

    def extractor(_model: torch.nn.Module, received_iq: torch.Tensor):
        values = np.asarray(received_iq.detach().cpu().tolist(), dtype=np.float32)
        observed.append(values.copy())
        rows = len(values)
        pre = np.zeros((rows, 160), dtype=np.float32)
        domain = np.zeros((rows, 160), dtype=np.float32)
        pre[:, 0] = values[:, 0, 0]
        domain[:, 0] = values[:, 1, 0]
        return SimpleNamespace(
            pre_relu=pre,
            z_dom=domain,
            receipt_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        )

    def reject_from_numpy(*_args: object, **_kwargs: object) -> object:
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    monkeypatch.setattr(torch, "from_numpy", reject_from_numpy)
    pre, domain, receipt = evaluation._tap_rows(
        _Model().eval(),
        source,
        device=torch.device("cpu"),
        batch_size=2,
        feature_extractor=extractor,
    )
    assert [values.tobytes(order="C") for values in observed] == [
        source[:2].tobytes(order="C"),
        source[2:].tobytes(order="C"),
    ]
    assert np.array_equal(pre[:, 0], source[:, 0, 0])
    assert np.array_equal(domain[:, 0], source[:, 1, 0])
    assert len(receipt) == 64


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("d105_cbrc", "allowed_k", [1, 5]),
        ("student_t_qknn", "temperature", 0.75),
        ("target25", "seed", 1),
    ],
)
def test_candidate_da_head_and_matrix_lock_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch,
    section: str,
    field: str,
    value,
) -> None:
    fixture = _fixture(tmp_path)
    context, package_loader, model_loader, extractor = fixture[:4]
    asset, handle = fixture[6:8]
    lock_path = Path(context.phase1_bundle.candidate_method_lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[section][field] = value
    lock_path.write_bytes(evaluation._canonical_bytes(lock))
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    authority = replace(
        context.phase1_bundle,
        d105_candidate_method_lock_sha256=lock_sha,
    )
    _patch_formal_asset(monkeypatch, asset, handle)
    with pytest.raises(
        D105QueryEvaluationError, match="implementation identity validation"
    ):
        evaluate_d105_query_row(
            replace(context, phase1_bundle=authority),
            package_loader=package_loader,
            model_loader=model_loader,
            feature_extractor=extractor,
        )
