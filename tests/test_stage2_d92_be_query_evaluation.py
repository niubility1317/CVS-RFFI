from __future__ import annotations

from types import SimpleNamespace

import pytest

from cvsrffi import stage2_d81_query_evaluation as d81_eval
from cvsrffi import stage2_d92_be_query_evaluation as be_eval
from cvsrffi import stage2_d92_be_slim as slim
from scripts import probe_d81_ground_nuisance_cauchy_center as d81_probe


def _resource(total: int, *, registered: bool, b: bool, e: bool) -> dict:
    return {
        "d92_status": "registration_balanced_active" if registered else "before_exact_d81",
        "d92_registration_balanced_active": registered,
        "d92_registration_state_support_only": True,
        "d92_query_rows_used": 0,
        "d92_query_role_oracle_access": False,
        "d92_scene_receiver_seed_specific_branch": False,
        "d92_class_id_specific_formula": False,
        "d92_be_registered_state": registered,
        "d92_be_B_enabled": b,
        "d92_be_E_enabled": e,
        "d92_be_B_effective": True if not registered else b,
        "d92_be_E_effective": True if not registered else e,
        "d92_be_total_component_fit_count": total,
        "d92_be_query_macs": (11 if registered else 6) * 288,
        "d92_be_query_fit_access": False,
        "d92_be_query_update_access": False,
        "d92_be_query_selection_access": False,
        "d92_be_query_role_oracle_access": False,
        "d92_be_query_class_quota_access": False,
        "d92_be_query_global_reassignment": False,
        "d92_be_finite_output_pass": True,
        "schema": "cvs.phase2.registration_resource_receipt.v1",
        "registration_wall_time_ns": 5_000,
        "registration_process_cpu_time_ns": 4_000,
        "registration_baseline_rss_bytes": 100,
        "registration_peak_rss_bytes": 180,
        "registration_incremental_peak_working_set_bytes": 80,
        "rss_sampler": "synthetic",
    }


def test_arm_audit_requires_full_before_and_requested_registered_after():
    result = SimpleNamespace(
        geometry_audit={
            "k1_unit_covariance_fallback": False,
            "before_covariance_audit": _resource(
                48, registered=False, b=False, e=False
            ),
            "final_covariance_audit": _resource(
                24, registered=True, b=False, e=False
            ),
        },
        before_state=SimpleNamespace(persistent_state_bytes=10),
        state=SimpleNamespace(persistent_state_bytes=20),
        training_trace=[],
        resource_audit={"trainable_parameters": 0},
    )
    row = be_eval._audit_d92_be_fit(
        result,
        arm=slim.D92_BE_ARMS["B0E0"],
        scenario="leo_clear_weak",
        k_shot=5,
        old_count=6,
        class_count=11,
    )
    assert row["before_effective_B"] is True
    assert row["before_effective_E"] is True
    assert row["after_effective_B"] is False
    assert row["after_effective_E"] is False
    assert row["after_total_component_fit_count"] == 24
    assert row["after_registration_resource"][
        "registration_incremental_peak_working_set_bytes"
    ] == 80


def test_arm_audit_rejects_query_selection_access():
    before = _resource(48, registered=False, b=True, e=True)
    after = _resource(48, registered=True, b=True, e=True)
    after["d92_be_query_selection_access"] = True
    result = SimpleNamespace(
        geometry_audit={
            "k1_unit_covariance_fallback": False,
            "before_covariance_audit": before,
            "final_covariance_audit": after,
        },
        before_state=SimpleNamespace(persistent_state_bytes=10),
        state=SimpleNamespace(persistent_state_bytes=20),
        training_trace=[],
        resource_audit={},
    )
    with pytest.raises(be_eval.D92BEQueryEvaluationError, match="protocol"):
        be_eval._audit_d92_be_fit(
            result,
            arm=slim.D92_BE_ARMS["FULL"],
            scenario="leo_clear_weak",
            k_shot=5,
            old_count=6,
            class_count=11,
        )


def test_evaluator_installs_arm_identity_and_restores_all_monkeypatches(monkeypatch):
    originals = (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    )
    observed = []

    def fake_builder(_d42, _basis, _weights, _audit, *, arm_id, **_kwargs):
        observed.append(arm_id)
        return "fit", [], []

    def fake_run(**_kwargs):
        fit, _, _ = d81_probe.build_d81_fit(None, None, None, {})
        assert fit == "fit"
        return {
            "candidate": d81_eval.CANDIDATE_D81,
            "schema": d81_eval.SCHEMA,
        }

    monkeypatch.setattr(be_eval, "build_d92_be_fit", fake_builder)
    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fake_run)
    result = be_eval.run_d92_be_query_evaluation(arm_id="B0E0")
    assert result["candidate"] == "d92_be_b0e0"
    assert result["arm_id"] == "B0E0"
    assert observed == ["B0E0"]
    assert (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    ) == originals


def test_evaluator_restores_all_monkeypatches_when_runner_raises(monkeypatch):
    originals = (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    )

    def fail(**_kwargs):
        raise RuntimeError("synthetic predictor failure")

    monkeypatch.setattr(d81_eval, "run_d81_query_evaluation", fail)
    with pytest.raises(RuntimeError, match="synthetic predictor failure"):
        be_eval.run_d92_be_query_evaluation(arm_id="FULL")
    assert (
        d81_probe.build_d81_fit,
        d81_eval.CANDIDATE_D81,
        d81_eval.SCHEMA,
        d81_eval._audit_fit,
    ) == originals
