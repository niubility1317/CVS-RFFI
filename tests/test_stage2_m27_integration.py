from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_m24_row_executor import (
    M24RowExecutionError,
    d1_overlay_from_base_cache,
    execute_m24_row,
)
from cvsrffi.stage2_m27_spectral_veto import V1, V2, m27_arm_config_hash
from test_stage2_m24_integration import _caches


def _phase_side_cache(base: dict, overlay: dict) -> dict:
    rng = np.random.default_rng(82720)
    payloads = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        compact = overlay["scenario_payloads"][scenario]
        payloads[scenario] = {
            "old_support_phase32": rng.normal(
                size=(len(compact["old_support_labels"]), 32)
            ).astype(np.float32),
            "old_support_labels": np.asarray(compact["old_support_labels"]).astype(str),
            "new_support_phase32": rng.normal(
                size=(len(compact["new_support_labels"]), 32)
            ).astype(np.float32),
            "new_support_labels": np.asarray(compact["new_support_labels"]).astype(str),
            "query_phase32": rng.normal(
                size=(len(compact["query_tokens"]), 32)
            ).astype(np.float32),
            "query_tokens": np.asarray(compact["query_tokens"]).astype(str),
        }
    return {
        "manifest": {
            "schema": "cvs.erbt_idr.m27.phase_side_cache_manifest.v1",
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "base_feature_cache_manifest_sha256": "a" * 64,
            "capsule_id": base["manifest"]["capsule_id"],
            "split_id": base["manifest"]["split_id"],
            "receiver": base["manifest"]["receiver"],
            "method_seed": base["manifest"]["method_seed"],
            "k_shot": base["manifest"]["k_shot"],
            "query_truth_present": False,
            "query_role_present": False,
            "query_state_update": False,
        },
        "old_classes": tuple(base["old_classes"]),
        "new_classes": tuple(base["new_classes"]),
        "scenario_payloads": payloads,
    }


def test_m27_lifecycle_scripts_are_available() -> None:
    for name in (
        "scripts.build_m27_phase_side_cache",
        "scripts.run_m27_spectral_veto_matrix",
        "scripts.score_m27_spectral_veto_matrix",
        "scripts.summarize_m27_spectral_veto_matrix",
    ):
        assert importlib.util.find_spec(name) is not None


def test_m27_runner_freezes_four_arm_screen_and_full125_sizes() -> None:
    from scripts import run_m27_spectral_veto_matrix as runner

    assert runner.EVIDENCE_ARMS == (runner.D1, runner.B3, runner.V1, runner.V2)
    screen = runner.matrix_spec("screen")
    full = runner.matrix_spec("full125")
    assert screen["paired_input_identity_count"] == 4
    assert screen["expected_method_rows"] == 16
    assert full["paired_input_identity_count"] == 125
    assert full["expected_method_rows"] == 500


def test_m27_summary_gate_requires_gain_over_b0_and_b3() -> None:
    from scripts import summarize_m27_spectral_veto_matrix as summarizer

    def arm_row(arm, h, min_old=0.30, min_new=0.25):
        return {
            "arm": arm,
            "metrics": {
                "H": {"pooled_query_weighted_mean": h},
                "min_old": {"pooled_query_weighted_mean": min_old},
                "min_new": {"pooled_query_weighted_mean": min_new},
            },
        }

    result = {
        "arm_summary": [
            arm_row("M24-D1-COMPILE-PARITY", 0.50),
            arm_row("M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL", 0.5018),
            arm_row("M27-V1-B3-MGD-CONSENSUS-VETO", 0.5021),
            arm_row("M27-V2-B3-PHASE32-CONSENSUS-VETO", 0.5019),
        ],
        "help_harm": {
            "overall": [
                {
                    "candidate_arm": "M27-V1-B3-MGD-CONSENSUS-VETO",
                    "N_help": 12,
                    "N_harm": 4,
                },
                {
                    "candidate_arm": "M27-V2-B3-PHASE32-CONSENSUS-VETO",
                    "N_help": 8,
                    "N_harm": 8,
                },
            ]
        },
    }
    gate = summarizer._gate(result)
    assert gate["decision"] == "PROMOTE_TO_FULL125"
    assert gate["passed_arms"] == ["M27-V1-B3-MGD-CONSENSUS-VETO"]
    assert gate["observed"]["M27-V2-B3-PHASE32-CONSENSUS-VETO"]["pass"] is False


def test_m27_scorer_rejects_a_self_declared_partial_screen() -> None:
    from scripts import run_m27_spectral_veto_matrix as runner
    from scripts import score_m27_spectral_veto_matrix as scorer

    entries = []
    for receiver in runner.SCREEN_RECEIVERS:
        for method_seed in runner.SCREEN_SEEDS:
            for k_shot, new_count in runner.SCREEN_CONDITIONS:
                for arm in runner.EVIDENCE_ARMS:
                    entries.append(
                        {
                            "row_id": (
                                f"rx{receiver}_m{method_seed}_k{k_shot}_new{new_count}__{arm}"
                            ),
                            "arm": arm,
                            "receiver": receiver,
                            "method_seed": method_seed,
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                        }
                    )
    complete = {
        "matrix_kind": "screen",
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "row_count": 16,
        "paired_input_identity_count": 4,
        "method_rows_per_arm": 4,
        "scenario_unit_count": 48,
        "receivers": list(runner.SCREEN_RECEIVERS),
        "method_seeds": list(runner.SCREEN_SEEDS),
        "conditions": [
            {"k_shot": k, "new_class_count": n}
            for k, n in runner.SCREEN_CONDITIONS
        ],
        "arms": list(runner.EVIDENCE_ARMS),
        "entries": entries,
        "query_truth_opened": False,
    }
    scorer._validate_matrix(complete)

    partial = {
        **complete,
        "row_count": 4,
        "paired_input_identity_count": 1,
        "method_rows_per_arm": 1,
        "scenario_unit_count": 12,
        "receivers": [runner.SCREEN_RECEIVERS[0]],
        "conditions": [
            {
                "k_shot": runner.SCREEN_CONDITIONS[0][0],
                "new_class_count": runner.SCREEN_CONDITIONS[0][1],
            }
        ],
        "entries": entries[:4],
    }
    with pytest.raises(ValueError, match="incomplete"):
        scorer._validate_matrix(partial)


def test_m27_v1_row_is_truth_unopened_and_publishes_veto_diagnostics(tmp_path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    compact = d1_overlay_from_base_cache(base)
    receipt = execute_m24_row(
        arm=V1,
        row_id="synthetic_m27_v1",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "v1",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["candidate_lock_sha256"] == m27_arm_config_hash(V1)
    assert receipt["resource"]["registration_timing_scope"] == "b3_conditioned_support_only_spectral_veto"
    assert all(
        audit["selection_policy"] == "B3_FLIP_CONSENSUS_VETO_ONLY"
        and audit["query_application"]["row_source_allowlist"] == ["B0", "B3"]
        and audit["query_application"]["query_state_update"] is False
        for audit in receipt["scenario_audit"].values()
    )


def test_m27_k1_degenerate_representation_completes_with_exact_b0_fallback(
    tmp_path,
) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {
            "k_shot": 1,
            "package_root_sha256": "3" * 64,
            "package_seal_sha256": "4" * 64,
        }
    )
    for payload in base["scenario_payloads"].values():
        for group in ("old", "new"):
            labels = np.asarray(payload[f"{group}_support_labels"]).astype(str)
            keep = np.asarray(
                [np.flatnonzero(labels == name)[0] for name in dict.fromkeys(labels)]
            )
            for key, value in tuple(payload.items()):
                if key.startswith(f"{group}_support_"):
                    rows = np.asarray(value)
                    if rows.ndim > 0 and len(rows) == len(labels):
                        payload[key] = rows[keep]

    compact = d1_overlay_from_base_cache(base)
    for payload in compact["scenario_payloads"].values():
        template = np.asarray(payload["old_support_blocks"])[0, 160:256].copy()
        for key in ("old_support_blocks", "new_support_blocks"):
            blocks = np.asarray(payload[key]).copy()
            blocks[:, 160:256] = template
            payload[key] = blocks

    receipt = execute_m24_row(
        arm=V1,
        row_id="synthetic_m27_v1_k1_fallback",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "v1_k1",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert all(
        audit["representation_fit"]["fallback_policy"] == "K1_EXACT_B0"
        and audit["query_application"]["fallback_reason"]
        == "SUPPORT_REPRESENTATION_UNRELIABLE"
        for audit in receipt["scenario_audit"].values()
    )


def test_m27_v2_requires_exact_phase_cache_identity_and_query_tokens(tmp_path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    compact = d1_overlay_from_base_cache(base)
    phase = _phase_side_cache(base, compact)
    receipt = execute_m24_row(
        arm=V2,
        row_id="synthetic_m27_v2",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        phase_side_cache=phase,
        output_root=tmp_path / "v2",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["candidate_lock_sha256"] == m27_arm_config_hash(V2)
    assert receipt["phase_side_cache"]["query_truth_present"] is False
    assert all(
        audit["representation"] == "PHASE_CEPSTRAL32"
        for audit in receipt["scenario_audit"].values()
    )

    drift = _phase_side_cache(base, compact)
    drift["scenario_payloads"]["leo_rain_weak"]["query_tokens"][0] = "qid_" + "e" * 64
    rejected = tmp_path / "v2_rejected"
    with pytest.raises(M24RowExecutionError, match="phase/query token"):
        execute_m24_row(
            arm=V2,
            row_id="synthetic_m27_v2_drift",
            receiver="3-19",
            base_cache=base,
            overlay_cache=compact,
            phase_side_cache=drift,
            output_root=rejected,
            seed=7282101,
        )
    assert not rejected.exists()
