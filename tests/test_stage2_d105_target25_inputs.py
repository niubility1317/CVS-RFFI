from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from cvsrffi.stage2_d105_target25_inputs import (
    D105Target25InputError,
    _authority_receipts,
    _load_package,
    _state_plan,
    _validate_qknn_lock,
    d105_target25_runtime_entrypoint_closure,
)
from cvsrffi.stage2_d105_target25_runner import (
    canonical_sha256,
    DEVELOPMENT_CLAIM_SCOPE,
    FORMAL_CLAIM_SCOPE,
)
from cvsrffi.stage2_d105_query_evaluation import build_d105_prediction_context
from cvsrffi.stage2_d105_phase1_bundle import D105_CANDIDATE_RUNTIME_FILES
from cvsrffi.stage2_predictor_bundle import sha256_file
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


def _offline_fixture_module():
    source = Path(__file__).with_name("test_somph_offline_target_package.py")
    spec = importlib.util.spec_from_file_location(
        "_d105_real_offline_fixture", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_entrypoint_closure_covers_launcher_input_builder_and_three_clis() -> None:
    code_root = Path(__file__).resolve().parents[1] / "code"
    closure = d105_target25_runtime_entrypoint_closure(code_root)
    names = [item["relative_path"] for item in closure["members"]]
    assert names == list(D105_CANDIDATE_RUNTIME_FILES)
    assert all(len(item["sha256"]) == 64 for item in closure["members"])


def test_real_structural_package_without_signed_path_free_authority_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the production D92 builder/loaders; no fake StatePlan or physical IDs."""

    fixture = _offline_fixture_module()
    kwargs = fixture._fixture(
        tmp_path, monkeypatch, cache_new_class_count=20, k_shot=10, query_per_tx=20
    )
    kwargs["seed"] = 713102
    kwargs["new_class_count"] = 5
    kwargs = fixture._attach_authority_bundle(kwargs, tmp_path, monkeypatch)
    built = fixture.producer.build_somph_offline_row_pair(**kwargs)
    authority = [
        {
            "receiver": "20-1",
            "authority_bundle_root": str(kwargs["authority_bundle_root"].resolve()),
            "expected_authority_commit_sha256": kwargs[
                "expected_authority_commit_sha256"
            ],
            "cache_set_manifest_path": str(
                Path(kwargs["cache_set_manifest_path"]).resolve()
            ),
        }
    ]
    # The production helper requires five unique receivers.  Exercise its
    # canonical single-cell verifier directly by padding only after the real
    # verification surface has been built is intentionally forbidden.
    with pytest.raises(D105Target25InputError, match="five receivers"):
        _authority_receipts(
            authority,
            claim_scope=DEVELOPMENT_CLAIM_SCOPE,
            formal_launch_authority=False,
        )
    before = built["states"]["before"]
    ref = {
        "package_root": str(Path(before["enrollment_package_root"]).resolve()),
        "detached_seal_path": str(
            Path(before["enrollment_package_seal"]).resolve()
        ),
        "expected_seal_sha256": before["enrollment_package_seal_sha256"],
    }
    with pytest.raises(D105Target25InputError, match="verification failed"):
        _load_package(
            ref,
            "real_before_enrollment",
            authority={
                "commit_sha256": kwargs["expected_authority_commit_sha256"],
                "dataset_authority_root_sha256": "d" * 64,
            },
        )


def test_formal_mode_cannot_relabel_current_diagnostic_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _offline_fixture_module()
    kwargs = fixture._fixture(tmp_path, monkeypatch)
    kwargs["seed"] = 713102
    kwargs = fixture._attach_authority_bundle(kwargs, tmp_path, monkeypatch)
    locator = {
        "receiver": "20-1",
        "authority_bundle_root": str(kwargs["authority_bundle_root"].resolve()),
        "expected_authority_commit_sha256": kwargs[
            "expected_authority_commit_sha256"
        ],
        "cache_set_manifest_path": str(
            Path(kwargs["cache_set_manifest_path"]).resolve()
        ),
    }
    # Five distinct locators are required before any cell is opened; duplicate
    # receivers therefore fail closed and cannot manufacture a formal matrix.
    with pytest.raises(D105Target25InputError):
        _authority_receipts(
            [locator] * 5,
            claim_scope=FORMAL_CLAIM_SCOPE,
            formal_launch_authority=True,
        )


def test_prepare_cli_is_the_only_matrix_prepare_entrypoint() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "prepare_d105_target25_inputs.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--d92-matrix-index" in result.stdout
    assert "--d92-matrix-index-sha256" in result.stdout
    assert "--output-dir" in result.stdout


def test_prepare_rejects_qknn_parameters_outside_candidate_method_lock() -> None:
    lock = Phase1ZIDStudentTLock(
        active_k=10,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.90,
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )
    candidate = {
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
        }
    }
    with pytest.raises(D105Target25InputError, match="candidate lock drift"):
        _validate_qknn_lock(
            lock,
            active_k=10,
            candidate_lock_document=candidate,
        )


def test_state_plan_and_real_evaluator_share_one_prediction_context() -> None:
    authority = {
        "commit_sha256": "a" * 64,
        "envelope_sha256": "b" * 64,
        "dataset_authority_root_sha256": "c" * 64,
    }
    phase1 = {
        "manifest_sha256": "d" * 64,
        "validated_bundle_id_sha256": "e" * 64,
        "expected_content_root_sha256": "f" * 64,
        "validator_receipt_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
    }
    package_roots = {
        "before_enrollment": "3" * 64,
        "before_apply": "4" * 64,
        "after_enrollment": "5" * 64,
        "after_apply": "6" * 64,
    }
    state = _state_plan(
        stage="S_B",
        scenario="leo_clear_weak",
        receiver="20-1",
        authority=authority,
        support=("sid_old_0",),
        query=("qid_old_0",),
        registered=("old-0",),
        old=("old-0",),
        new=(),
        active_k=1,
        qknn_lock_digest="7" * 64,
        phase1=phase1,
        data_runtime_sha="8" * 64,
        data_lock_sha="9" * 64,
        candidate_runtime_sha="0" * 64,
        candidate_lock_sha="a" * 64,
        package_roots=package_roots,
    )
    _payload, expected = build_d105_prediction_context(
        registration_state="BEFORE_REGISTRATION",
        stage="S_B",
        scenario="leo_clear_weak",
        receiver="20-1",
        seed=713102,
        active_k=1,
        registered_classes=("old-0",),
        capsule_id=state.capsule_id,
        split_id=state.split_id,
        split_validator_receipt_sha256="a" * 64,
        support_physical_root_sha256=canonical_sha256(["sid_old_0"]),
        query_physical_root_sha256=canonical_sha256(["qid_old_0"]),
        package_root_sha256=package_roots,
        phase1_bundle_manifest_sha256="d" * 64,
        validated_bundle_id_sha256="e" * 64,
        bundle_content_root_sha256="f" * 64,
        bundle_validator_receipt_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        data_feature_runtime_sha256="8" * 64,
        data_materialization_lock_sha256="9" * 64,
        d105_candidate_runtime_manifest_sha256="0" * 64,
        d105_candidate_method_lock_sha256="a" * 64,
        qknn_lock_digest="7" * 64,
    )
    assert state.prediction_context_sha256 == expected
