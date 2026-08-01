from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cvsrffi.stage2_d106_train_only_predecessor_lock as predecessor
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


ROOT = Path(__file__).resolve().parents[1]
REAL_R7_TAP_ARCHIVE_SHA256 = (
    "48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f"
)
REAL_R7_TAP_RECEIPT_SHA256 = (
    "24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _official_bytes() -> bytes:
    return predecessor.load_d106_train_only_predecessor_lock_bundle(
        tap_archive_sha256=REAL_R7_TAP_ARCHIVE_SHA256,
        tap_receipt_sha256=REAL_R7_TAP_RECEIPT_SHA256,
    )


def _document(payload: bytes) -> dict[str, object]:
    document = json.loads(payload.decode("utf-8"))
    assert type(document) is dict
    return document


def test_official_surface_publishes_only_canonical_bytes_and_frozen_summary() -> None:
    payload = _official_bytes()
    assert type(payload) is bytes
    assert _canonical_bytes(_document(payload)) == payload
    with pytest.raises(TypeError):
        payload[0] = 0  # type: ignore[index]

    summary = predecessor.summarize_d106_train_only_predecessor_lock_bundle(payload)
    assert type(summary) is predecessor.D106TrainOnlyPredecessorLockSummary
    assert summary.k_values == (1, 5, 10)
    assert summary.d106_tap_archive_sha256 == REAL_R7_TAP_ARCHIVE_SHA256
    assert summary.d106_tap_receipt_sha256 == REAL_R7_TAP_RECEIPT_SHA256
    assert summary.tap_hash_provenance == "CALLER_SUPPLIED_UNVERIFIED"
    assert summary.external_strict_tap_loader_binding_required is True
    assert summary.real_g0_promotion_authority is False
    assert summary.authority_flags == (
        ("d105_formal_authority", False),
        ("held_truth_access", False),
        ("performance_metrics_computed", False),
        ("phase2_promotion_authority", False),
        ("runner_authority", False),
        ("target_access", False),
    )
    assert all(type(value) is bool and value is False for _, value in summary.authority_flags)
    assert not hasattr(summary, "locks")
    assert not hasattr(summary, "bundle_receipt")
    with pytest.raises(TypeError):
        summary.authority_flags[0] = ("target_access", True)  # type: ignore[index]

    assert all("PATH" not in name and "SHA256" not in name for name in predecessor.__all__)
    assert "load_d106_train_only_predecessor_lock_bundle" in predecessor.__all__
    assert "summarize_d106_train_only_predecessor_lock_bundle" in predecessor.__all__


def test_private_consumption_reconstructs_fresh_exact_locks_after_mutation() -> None:
    payload = _official_bytes()
    locks = predecessor._strict_reconstruct_d106_train_only_predecessor_locks(payload)
    assert tuple(lock.active_k for lock in locks) == (1, 5, 10)
    assert all(type(lock) is Phase1ZIDStudentTLock for lock in locks)
    assert [lock.lock_digest for lock in locks] == [
        "bd564538115dd49f26c4c177159740621988275c171290a5d0108c0f4b4ef659",
        "c9f59a66e4d639bb0714720c243861a6d2bbac29545912b9744919d5f7c6695d",
        "53afc89e447c42f1ec66005d70dd6423f5a446cd471f51e25e6bb62d1b6c4b15",
    ]
    object.__setattr__(locks[0], "temperature", 99.0)
    object.__setattr__(locks[0], "quantization_margin_audit_sha256", "c" * 64)

    fresh = predecessor._strict_reconstruct_d106_train_only_predecessor_locks(payload)
    assert fresh is not locks
    assert fresh[0].temperature == 0.85
    assert fresh[0].quantization_margin_audit_sha256 == REAL_R7_TAP_RECEIPT_SHA256
    assert fresh[0].lock_digest == (
        "bd564538115dd49f26c4c177159740621988275c171290a5d0108c0f4b4ef659"
    )


def test_mutated_frozen_summary_cannot_change_revalidated_bytes() -> None:
    payload = _official_bytes()
    summary = predecessor.summarize_d106_train_only_predecessor_lock_bundle(payload)
    object.__setattr__(summary, "candidate_id", "tampered")
    object.__setattr__(summary.resource, "canonical_bundle_bytes", 0)

    fresh = predecessor.summarize_d106_train_only_predecessor_lock_bundle(payload)
    assert fresh.candidate_id == "D106-RCMR-2V-qKNN/r1.1"
    assert fresh.resource.canonical_bundle_bytes == len(payload)
    assert type(fresh.resource.lock_wire_bytes_by_k) is tuple
    assert fresh.resource.lock_wire_bytes_by_k == ((1, 621), (5, 621), (10, 623))


def test_resource_receipt_is_exact_bounded_and_excludes_python_rss() -> None:
    payload = _official_bytes()
    summary = predecessor.summarize_d106_train_only_predecessor_lock_bundle(payload)
    resource = summary.resource
    assert resource.canonical_bundle_bytes == len(payload)
    assert resource.canonical_bundle_bytes <= resource.canonical_bundle_bytes_cap
    assert len(resource.lock_wire_bytes_by_k) == 3
    assert sum(size for _, size in resource.lock_wire_bytes_by_k) == resource.lock_wire_bytes_total
    assert all(size <= resource.lock_wire_bytes_cap_per_k for _, size in resource.lock_wire_bytes_by_k)
    assert resource.loader_module_source_bytes <= resource.loader_module_source_bytes_cap
    assert resource.static_bundle_json_bytes == len(predecessor._BUNDLE_PATH.read_bytes())
    assert resource.static_bundle_json_bytes <= resource.static_bundle_json_bytes_cap
    assert resource.python_object_and_rss_bytes == "NOT_MEASURED_EXCLUDED_FROM_ACCOUNTING"


@pytest.mark.parametrize(
    ("path", "mutate"),
    [
        pytest.param(
            ("authority_flags", "target_access"),
            lambda document: document["authority_flags"].__setitem__("target_access", 0),
            id="authority-flag-zero",
        ),
        pytest.param(
            ("tap_binding_policy", "external_strict_tap_loader_binding_required"),
            lambda document: document["tap_binding_policy"].__setitem__(
                "external_strict_tap_loader_binding_required", 1
            ),
            id="policy-true-one",
        ),
        pytest.param(
            ("student_t_qknn", "student_nu"),
            lambda document: document["student_t_qknn"].__setitem__("student_nu", 3),
            id="float-int-alias",
        ),
    ],
)
def test_static_json_requires_exact_bool_and_numeric_types(
    path: tuple[str, str], mutate
) -> None:
    document = _document(predecessor._BUNDLE_PATH.read_bytes())
    mutate(document)
    with pytest.raises(predecessor.D106TrainOnlyPredecessorLockError):
        predecessor._validate_bundle_document(document)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda document: document.__setitem__("real_g0_promotion_authority", 0),
            id="promotion-zero",
        ),
        pytest.param(
            lambda document: document.__setitem__(
                "external_strict_tap_loader_binding_required", 1
            ),
            id="external-binding-one",
        ),
        pytest.param(
            lambda document: document["authority_flags"].__setitem__(
                "target_access", 0
            ),
            id="authority-zero",
        ),
        pytest.param(
            lambda document: document["K_values"].__setitem__(0, True),
            id="k-bool",
        ),
        pytest.param(
            lambda document: document["resource_receipt"].__setitem__(
                "loader_module_source_bytes", 0
            ),
            id="resource-zero",
        ),
    ],
)
def test_published_bytes_reject_bool_numeric_and_resource_aliases(mutate) -> None:
    document = _document(_official_bytes())
    mutate(document)
    tampered = _canonical_bytes(document)
    with pytest.raises(predecessor.D106TrainOnlyPredecessorLockError):
        predecessor.summarize_d106_train_only_predecessor_lock_bundle(tampered)


def test_published_bytes_reject_lock_digest_and_deep_nested_mutation() -> None:
    document = _document(_official_bytes())
    document["lock_digest_by_k"]["5"] = "0" * 64
    document["resource_receipt"]["lock_wire_bytes_by_k"]["10"] = 0
    with pytest.raises(predecessor.D106TrainOnlyPredecessorLockError):
        predecessor.summarize_d106_train_only_predecessor_lock_bundle(
            _canonical_bytes(document)
        )


def test_monkeypatched_path_and_sha_globals_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(predecessor, "_BUNDLE_PATH", tmp_path / "other.json")
    monkeypatch.setattr(predecessor, "_BUNDLE_FILE_SHA256", "f" * 64)
    with pytest.raises(
        predecessor.D106TrainOnlyPredecessorLockError,
        match="import-time closure global drift",
    ):
        _official_bytes()


def test_monkeypatched_self_source_global_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predecessor, "_MODULE_SOURCE_SHA256", "0" * 64)
    with pytest.raises(
        predecessor.D106TrainOnlyPredecessorLockError,
        match="import-time closure global drift",
    ):
        _official_bytes()


def test_fake_rcmr_loader_is_rejected_before_it_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("fake RCMR loader must not run")

    monkeypatch.setattr(predecessor, "load_d106_rcmr_2v_method_lock", fake_loader)
    with pytest.raises(
        predecessor.D106TrainOnlyPredecessorLockError,
        match="imported loader callable drift",
    ):
        _official_bytes()
    assert called is False


def test_fake_d105_loader_and_rcmr_source_path_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        predecessor,
        "load_d105_candidate_method_lock",
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(
        predecessor.D106TrainOnlyPredecessorLockError,
        match="imported loader callable drift",
    ):
        _official_bytes()

    monkeypatch.undo()
    monkeypatch.setattr(predecessor, "_D106_RCMR_SOURCE_PATH", tmp_path / "fake.py")
    with pytest.raises(
        predecessor.D106TrainOnlyPredecessorLockError,
        match="import-time closure global drift",
    ):
        _official_bytes()


def test_public_summary_classes_are_not_user_constructed_capabilities() -> None:
    with pytest.raises(TypeError):
        predecessor.D106TrainOnlyPredecessorLockSummary(  # type: ignore[call-arg]
            authority_flags=(("target_access", 0),)
        )
    with pytest.raises(TypeError):
        predecessor.D106TrainOnlyPredecessorLockResourceSummary(  # type: ignore[call-arg]
            canonical_bundle_bytes=0
        )


def test_loader_source_and_config_raw_sha_are_bound_at_runtime() -> None:
    payload = _official_bytes()
    document = _document(payload)
    assert document["loader_module_sha256"] == predecessor._MODULE_SOURCE_SHA256
    assert document["bundle_file_sha256"] == hashlib.sha256(
        predecessor._BUNDLE_PATH.read_bytes()
    ).hexdigest()
    assert predecessor._D106_RCMR_SOURCE_PATH.is_file()
