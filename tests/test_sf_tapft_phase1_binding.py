from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cvsrffi.phase1_adv3b02_deployment_bundle import VerifiedADV3B02DeploymentBundle
from cvsrffi.sf_tapft_phase1_binding import load_sf_tapft_phase1_binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase1_bundle() -> dict[str, str]:
    return {
        "package_root": "formal-package",
        "detached_seal_path": "detached-seal.json",
        "expected_detached_seal_sha256": "1" * 64,
        "signature_envelope_path": "signature-envelope.json",
        "expected_signature_envelope_sha256": "2" * 64,
        "expected_checkpoint_lineage_sha256": "3" * 64,
        "expected_runtime_sha256": "4" * 64,
        "expected_component_pre_sign_content_root_sha256": "5" * 64,
        "expected_class_handle_binding_sha256": "6" * 64,
        "expected_parity_receipt_sha256": "7" * 64,
        "expected_generation_lock_sha256": "8" * 64,
        "expected_method_lock_sha256": "9" * 64,
        "expected_generation_config_sha256": "a" * 64,
        "expected_generation_code_sha256": "b" * 64,
        "expected_outer_content_root_sha256": "c" * 64,
    }


def _formal_fixture(
    *, checkpoint_lineage_sha256: str, class_rows: list[dict[str, object]] | None = None, eligible: bool = True
) -> VerifiedADV3B02DeploymentBundle:
    return VerifiedADV3B02DeploymentBundle(
        runtime=None,
        component=None,
        class_binding={
            "class_id_to_handle": class_rows
            or [
                {"class_index": 0, "class_handle": "tx0"},
                {"class_index": 1, "class_handle": "tx1"},
            ]
        },
        parity_receipt={},
        generation_lock={},
        method_lock={},
        formal_phase2_context={
            "formal_phase2_eligible": eligible,
            "outer_content_root_sha256": "c" * 64,
            "checkpoint_lineage_sha256": checkpoint_lineage_sha256,
            "runtime_sha256": "4" * 64,
            "class_handle_binding_sha256": "6" * 64,
            "component_pre_sign_content_root_sha256": "5" * 64,
        },
        audit={},
    )


def test_phase1_binding_preserves_ordered_handles_and_immutable_identifiers(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen checkpoint")
    formal = _formal_fixture(checkpoint_lineage_sha256=_sha256(checkpoint))

    binding = load_sf_tapft_phase1_binding(
        {"phase1_bundle": _phase1_bundle()},
        checkpoint,
        formal_loader=lambda **_kwargs: formal,
    )

    assert binding.class_handles == ("tx0", "tx1")
    assert binding.outer_content_root_sha256 == "c" * 64
    assert binding.checkpoint_lineage_sha256 == _sha256(checkpoint)
    assert binding.runtime_sha256 == "4" * 64
    assert binding.class_handle_binding_sha256 == "6" * 64
    assert binding.component_pre_sign_content_root_sha256 == "5" * 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda bundle: bundle.pop("expected_runtime_sha256"), "phase1_bundle"),
        (lambda bundle: bundle.__setitem__("unknown", "x"), "phase1_bundle"),
    ],
)
def test_phase1_binding_rejects_missing_or_unknown_mapping_fields(
    tmp_path: Path, mutate, message: str
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen checkpoint")
    bundle = _phase1_bundle()
    mutate(bundle)
    formal = _formal_fixture(checkpoint_lineage_sha256=_sha256(checkpoint))

    with pytest.raises(ValueError, match=message):
        load_sf_tapft_phase1_binding(
            {"phase1_bundle": bundle}, checkpoint, formal_loader=lambda **_kwargs: formal
        )


def test_phase1_binding_rejects_checkpoint_lineage_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen checkpoint")
    formal = _formal_fixture(checkpoint_lineage_sha256="d" * 64)

    with pytest.raises(ValueError, match="checkpoint lineage"):
        load_sf_tapft_phase1_binding(
            {"phase1_bundle": _phase1_bundle()}, checkpoint, formal_loader=lambda **_kwargs: formal
        )


def test_phase1_binding_rejects_ineligible_or_reordered_class_registry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen checkpoint")
    ineligible = _formal_fixture(checkpoint_lineage_sha256=_sha256(checkpoint), eligible=False)
    with pytest.raises(ValueError, match="eligible"):
        load_sf_tapft_phase1_binding(
            {"phase1_bundle": _phase1_bundle()}, checkpoint, formal_loader=lambda **_kwargs: ineligible
        )

    reordered = _formal_fixture(
        checkpoint_lineage_sha256=_sha256(checkpoint),
        class_rows=[
            {"class_index": 1, "class_handle": "tx1"},
            {"class_index": 0, "class_handle": "tx0"},
        ],
    )
    with pytest.raises(ValueError, match="class registry"):
        load_sf_tapft_phase1_binding(
            {"phase1_bundle": _phase1_bundle()}, checkpoint, formal_loader=lambda **_kwargs: reordered
        )
