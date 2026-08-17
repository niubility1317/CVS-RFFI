from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cvsrffi import stage2_d92_da0_reg1_formal as formal


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _input(tmp_path: Path, name: str) -> formal.SignedPackageInput:
    root = tmp_path / name
    root.mkdir()
    for filename in ("seal.json", "policy.json", "authorization.json", "envelope.json"):
        (root / filename).write_text("{}", encoding="utf-8")
    return formal.SignedPackageInput(
        package_root=root,
        detached_seal_path=root / "seal.json",
        detached_seal_sha256=_sha(f"{name}:seal"),
        formal_policy_path=root / "policy.json",
        formal_policy_authorization_path=root / "authorization.json",
        signed_policy_authorization_envelope_path=root / "envelope.json",
        signed_policy_authorization_envelope_sha256=_sha(f"{name}:envelope"),
    )


def _manifest(profile: str, state: str) -> dict[str, object]:
    classes = [f"class_{index}" for index in range(6 + (0 if state == "before" else 5))]
    result = {
        "profile": profile,
        "registration_state": state,
        "receiver": "20-1",
        "seed": 713106,
        "k_shot": 1,
        "registered_classes": [{"class_handle": value} for value in classes],
        "package_root_sha256": _sha(f"{profile}:{state}:root"),
        "phase1_checkpoint_sha256": _sha("checkpoint"),
        "feature_runtime_sha256": _sha("runtime"),
        "method_lock_sha256": _sha("method"),
    }
    if profile == "apply_only":
        result.update(
            {
                "head_enrollment_binding_sha256": _sha(
                    f"{profile}:{state}:head-binding"
                ),
                "head_capsule_sha256": _sha(f"{profile}:{state}:head"),
                "row_handle": "row_" + _sha(f"{profile}:{state}:row"),
                "row_manifest_sha256": _sha(f"{profile}:{state}:row-manifest"),
            }
        )
    return result


def test_formal_entry_orders_signed_support_state_lock_then_query(
    monkeypatch, tmp_path: Path
) -> None:
    """A query package must remain unopened until both states are locked."""

    packages = {
        "before_enrollment": _input(tmp_path, "before_enrollment"),
        "before_apply": _input(tmp_path, "before_apply"),
        "after_enrollment": _input(tmp_path, "after_enrollment"),
        "after_apply": _input(tmp_path, "after_apply"),
    }
    profiles = {
        "before_enrollment": ("enrollment_only", "before"),
        "before_apply": ("apply_only", "before"),
        "after_enrollment": ("enrollment_only", "after"),
        "after_apply": ("apply_only", "after"),
    }
    events: list[str] = []

    class Evidence:
        def __init__(self, name: str) -> None:
            profile, state = profiles[name]
            self.manifest = _manifest(profile, state)
            self.materialized_payloads = {
                "leo_clear_weak": {"payload": np.asarray([1], dtype=np.int8)}
            }
            self.evidence_sha256 = _sha(f"{name}:evidence")
            self.name = name

    def fake_materialize(package_root, **_kwargs):
        name = Path(package_root).name
        events.append(f"materialize:{name}")
        return Evidence(name)

    def fake_finalize(evidence):
        events.append(f"finalize:{evidence.name}")
        manifest = evidence.manifest
        registry_sha = hashlib.sha256(
            json.dumps(
                manifest["registered_classes"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "status": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
            "control_state": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
            "formal_launch_authority": True,
            "formal_metric_claim_allowed": False,
            "iq_payload_materialized": True,
            "package_root_sha256": manifest["package_root_sha256"],
            "package_detached_seal_sha256": packages[
                evidence.name
            ].detached_seal_sha256,
            "manifest_sha256": _sha(f"{evidence.name}:manifest"),
            "artifact_member_allowlist_sha256": _sha(
                f"{evidence.name}:allowlist"
            ),
            "formal_policy_sha256": _sha(f"{evidence.name}:policy"),
            "formal_policy_authorization_sha256": _sha(
                f"{evidence.name}:authorization"
            ),
            "post_materialization_audit_sha256": _sha(f"{evidence.name}:audit"),
            "signed_policy_authorization_envelope_sha256": _sha(
                f"{evidence.name}:envelope"
            ),
            "authority_commit_sha256": _sha(f"{evidence.name}:authority"),
            "code_closure_sha256": _sha(f"{evidence.name}:closure"),
            "package_class_registry_sha256": registry_sha,
        }

    def fake_evaluator(**kwargs):
        before_support = kwargs["package_loader"](
            packages["before_enrollment"].package_root,
            detached_seal_path=packages["before_enrollment"].detached_seal_path,
            expected_seal_sha256=packages["before_enrollment"].detached_seal_sha256,
        )
        after_support = kwargs["package_loader"](
            packages["after_enrollment"].package_root,
            detached_seal_path=packages["after_enrollment"].detached_seal_path,
            expected_seal_sha256=packages["after_enrollment"].detached_seal_sha256,
        )
        assert before_support[1]["profile"] == "enrollment_only"
        assert after_support[1]["profile"] == "enrollment_only"
        events.append("native_d42_joint_old_new_fit")
        output = Path(kwargs["output_root"])
        output.mkdir()
        lock = kwargs["state_lock_sink"](
            {
                "schema": "cvs.phase2.d81.support_state_lock.v1",
                "candidate": "d92_e0d_e0_full_only",
                "receiver": "20-1",
                "seed": 713106,
                "k_shot": 1,
                "old_class_count": 6,
                "registered_class_count": 11,
                "phase1_checkpoint_sha256": _sha("checkpoint"),
                "feature_runtime_sha256": _sha("runtime"),
                "method_lock_sha256": _sha("method"),
                "old_class_handles": [f"class_{index}" for index in range(6)],
                "registered_class_handles": [f"class_{index}" for index in range(11)],
                "state_fingerprints": {
                    scenario: {
                        "before_state_sha256": _sha(f"{scenario}:before"),
                        "after_state_sha256": _sha(f"{scenario}:after"),
                    }
                    for scenario in (
                        "leo_clear_weak",
                        "leo_low_elev_weak",
                        "leo_rain_weak",
                    )
                },
                "query_opened": False,
            }
        )
        assert lock["state_lock_sha256"]
        events.append("state_lock")
        before_query = kwargs["query_package_loader"](
            packages["before_apply"].package_root,
            detached_seal_path=packages["before_apply"].detached_seal_path,
            expected_seal_sha256=packages["before_apply"].detached_seal_sha256,
        )
        after_query = kwargs["query_package_loader"](
            packages["after_apply"].package_root,
            detached_seal_path=packages["after_apply"].detached_seal_path,
            expected_seal_sha256=packages["after_apply"].detached_seal_sha256,
        )
        assert before_query[1]["profile"] == "apply_only"
        assert after_query[1]["profile"] == "apply_only"
        for state in ("before", "after"):
            destination = output / state
            destination.mkdir()
            (destination / "COMMIT.json").write_text("{}", encoding="utf-8")
        return {
            "schema": "cvs.phase2.d92_e0d.e0_full_only.full_query_evaluation.v1",
            "candidate": "d92_e0d_e0_full_only",
            "arm_id": "E0_FULL_ONLY",
            "receiver": "20-1",
            "seed": 713106,
            "k_shot": 1,
            "new_class_count": 5,
            "states": {
                "before": {"commit_sha256": _sha("before-commit")},
                "after": {"commit_sha256": _sha("after-commit")},
            },
            "state_lock": lock,
        }

    monkeypatch.setattr(
        formal,
        "materialize_somph_predictor_bundle_with_signed_authority",
        fake_materialize,
    )
    monkeypatch.setattr(
        formal,
        "finalize_somph_predictor_bundle_authority_after_materialization",
        fake_finalize,
    )
    monkeypatch.setattr(formal, "run_d92_e0d_query_evaluation", fake_evaluator)
    output = tmp_path / "formal-output"
    result = formal.run_d92_da0_reg1_formal(
        before_enrollment=packages["before_enrollment"],
        before_apply=packages["before_apply"],
        after_enrollment=packages["after_enrollment"],
        after_apply=packages["after_apply"],
        ground_component_dir=tmp_path / "ground",
        ground_manifest_sha256=_sha("ground"),
        output_root=output,
        device="cpu",
    )

    assert result["status"] == "ERTB_IDR_DA0_REG1_TRUTH_FREE_PREDICTIONS_COMPLETE"
    assert result["state_labels"] == {"before": "DA0_REG0", "after": "DA0_REG1"}
    assert events == [
        "materialize:before_enrollment",
        "finalize:before_enrollment",
        "materialize:after_enrollment",
        "finalize:after_enrollment",
        "native_d42_joint_old_new_fit",
        "state_lock",
        "materialize:before_apply",
        "finalize:before_apply",
        "materialize:after_apply",
        "finalize:after_apply",
    ]
    receipt = json.loads((output / "ERTB_DA0_REG1_RECEIPT.json").read_text("utf-8"))
    assert receipt["independent_da_preadaptation_applied"] is False
    assert receipt["native_joint_old_new_registration"] is True
    assert receipt["state_lock"]["query_opened_after_state_lock"] is True
