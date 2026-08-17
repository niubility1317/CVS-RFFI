from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json

import pytest

CODE_ROOT = Path(__file__).parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
sys.modules.pop("scripts", None)


def test_afcp_g0_public_identity_and_truth_free_parser_surface() -> None:
    from scripts import run_d92_afcp_g0

    assert run_d92_afcp_g0.G0_OUTER_KEY == "rx_7_7__seed_713106__k_10__new_5"
    assert run_d92_afcp_g0.REFERENCE_ARM == "E0_FULL_ONLY"
    assert run_d92_afcp_g0.CANDIDATE_ARM == "E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE"
    assert run_d92_afcp_g0.G0_SCENES == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    assert run_d92_afcp_g0.G0_MARKER == "D92_AFCP_G0_ACTIVE_RESOURCE_PASS"
    help_text = run_d92_afcp_g0.parser().format_help().lower()
    assert "truth" not in help_text
    assert "score" not in help_text


def test_afcp_g0_config_freezes_the_truth_free_single_outer_contract() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "stage2_d92_afcp_g0_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["outer_key"] == "rx_7_7__seed_713106__k_10__new_5"
    assert config["scenes"] == [
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]
    assert config["arms"] == {
        "candidate": "E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE",
        "reference": "E0_FULL_ONLY",
    }
    assert config["registration"] == {
        "actual_full_fit_count": 1,
        "candidate_peak_max_bytes": 1_048_576,
        "wall_max_ns": 150_000_000,
    }
    assert config["receipt_prefix"] == "d92_e0d_afcp_"
    assert config["entry_arguments"] == [
        "before_enrollment_package_root",
        "before_enrollment_seal_path",
        "before_enrollment_seal_sha256",
        "before_apply_package_root",
        "before_apply_seal_path",
        "before_apply_seal_sha256",
        "after_enrollment_package_root",
        "after_enrollment_seal_path",
        "after_enrollment_seal_sha256",
        "after_apply_package_root",
        "after_apply_seal_path",
        "after_apply_seal_sha256",
        "ground_component_dir",
        "ground_manifest_sha256",
        "reference_output_root",
        "candidate_output_root",
        "g0_validation_path",
        "device",
    ]


def _args(tmp_path: Path) -> argparse.Namespace:
    from scripts import run_d92_afcp_g0

    return argparse.Namespace(
        outer_key=run_d92_afcp_g0.G0_OUTER_KEY,
        reference_arm=run_d92_afcp_g0.REFERENCE_ARM,
        candidate_arm=run_d92_afcp_g0.CANDIDATE_ARM,
        reference_output_root=str(tmp_path / "reference"),
        candidate_output_root=str(tmp_path / "candidate"),
        g0_validation_path=str(tmp_path / "g0_validation.json"),
        before_enrollment_package_root="before-enrollment",
        before_enrollment_seal_path="before-enrollment.seal",
        before_enrollment_seal_sha256="a" * 64,
        before_apply_package_root="before-apply",
        before_apply_seal_path="before-apply.seal",
        before_apply_seal_sha256="b" * 64,
        after_enrollment_package_root="after-enrollment",
        after_enrollment_seal_path="after-enrollment.seal",
        after_enrollment_seal_sha256="c" * 64,
        after_apply_package_root="after-apply",
        after_apply_seal_path="after-apply.seal",
        after_apply_seal_sha256="d" * 64,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )


def _row(scene: str, *, candidate: bool) -> dict[str, object]:
    from scripts import run_d92_afcp_g0

    row: dict[str, object] = {
        "scenario": scene,
        "arm_id": (
            run_d92_afcp_g0.CANDIDATE_ARM
            if candidate
            else run_d92_afcp_g0.REFERENCE_ARM
        ),
        "k_shot": 10,
        "old_class_count": 6,
        "registered_class_count": 11,
        "after_registered_d_mode_effective": "full_only",
        "after_state_bytes": 8583,
        "query_macs": 3168,
        "after_actual_component_inventory": {"actual_component_fit_count": 1},
        "after_registration_resource": {
            "registration_wall_time_ns": 110_000_000 if candidate else 100_000_000,
            "registration_incremental_peak_working_set_bytes": 900_000 if candidate else 100_000,
        },
    }
    for suffix in (
        "fit_access",
        "update_access",
        "selection_access",
        "truth_access",
        "role_oracle_access",
        "class_quota_access",
        "global_reassignment",
    ):
        row[f"query_{suffix}"] = False
    if not candidate:
        return row
    prefix = "d92_e0d_afcp_"
    row.update(
        {
            prefix + "active": True,
            prefix + "fallback_active": False,
            prefix + "fallback_reason": None,
            prefix + "formula_revision": "v1",
            prefix + "state_postprocess_mode": "allclass_fold_consensus_plane",
            prefix + "direct_state_publish": True,
            prefix + "support_only": True,
            prefix + "class_count": 11,
            prefix + "old_class_count": 6,
            prefix + "k_shot": 10,
            prefix + "e0_state_sha256": "a" * 64,
            prefix + "final_state_sha256": "b" * 64,
            prefix + "modified_state_field_names": ["coef2_qint8"],
            prefix + "coef1_byte_exact": True,
            prefix + "coef2_byte_exact": False,
            prefix + "scale1_byte_exact": True,
            prefix + "scale2_byte_exact": True,
            prefix + "intercept_byte_exact": True,
            prefix + "log_diag_byte_exact": True,
            prefix + "coef_fp32_byte_exact": True,
            prefix + "intercept_fp32_byte_exact": True,
            prefix + "class_registry_byte_exact": True,
            prefix + "state_shape_byte_exact": True,
            prefix + "persistent_state_bytes_delta": 0,
            prefix + "query_macs_delta": 0,
            prefix + "block_coordinate_indices": [1, 2, 3],
            prefix + "block_changed_code2_counts": [2, 2, 2],
            prefix + "changed_code2_count": 6,
            prefix + "state_delta_code2_l1": 6,
            prefix + "all_three_blocks_changed": True,
            prefix + "final_state_non_e0": True,
            prefix + "support_margin_delta_max_abs": 0.25,
            prefix + "support_margin_quantum_pass": True,
            prefix + "support_row_canonicalization": "canonical",
            prefix + "fold_rule": "twofold",
            prefix + "fold_tie_policy": "fallback",
            prefix + "class_permutation_equivariant": True,
            prefix + "row_permutation_invariant": True,
            prefix + "task_swap_equivariant": True,
            prefix + "all_class_symmetric": True,
            prefix + "fold_class_all_margin_delta_mean": [[0.1] * 11, [0.1] * 11],
            prefix + "fold_old_to_new_cross_margin_delta_mean": [0.1, 0.1],
            prefix + "fold_new_to_old_cross_margin_delta_mean": [0.1, 0.1],
            prefix + "twofold_class_guard_pass": True,
            prefix + "twofold_cross_guard_pass": True,
            prefix + "support_guard_pass": True,
            prefix + "requantize_call_count": 0,
            prefix + "additional_full_fit_count": 0,
            prefix + "block_fit_count": 0,
            prefix + "loo_fit_count": 0,
            prefix + "fisher_fit_count": 0,
            prefix + "tail_selection_count": 0,
            prefix + "rival_pair_selection_count": 0,
            prefix + "atomic_candidate_count": 0,
            prefix + "prefix_evaluation_count": 0,
            prefix + "candidate_scan_count": 0,
            prefix + "support_288_square_matrix_bytes": 0,
            prefix + "support_macs_upper_bound": 10_000,
            prefix + "support_transient_bytes_upper_bound": 900_000,
            prefix + "query_rows_used": 0,
            prefix + "clean_sample_access": False,
            prefix + "source_sample_access": False,
        }
    )
    for suffix in (
        "fit_access",
        "update_access",
        "selection_access",
        "truth_access",
        "role_oracle_access",
        "class_quota_access",
        "global_reassignment",
    ):
        row[prefix + "query_" + suffix] = False
    return row


def _persist_fake_audit(output_root: str | Path, rows: list[dict[str, object]]) -> None:
    after = Path(output_root) / "after"
    after.mkdir(parents=True)
    (after / "fit_audit.json").write_text(
        json.dumps(rows), encoding="utf-8", newline="\n"
    )


def test_g0_reads_persisted_afcp_receipts_and_writes_active_marker(tmp_path, monkeypatch) -> None:
    from scripts import run_d92_afcp_g0
    from cvsrffi import stage2_d92_e0d_query_evaluation as e0d

    args = _args(tmp_path)
    calls: list[str] = []

    def fake_run(*, arm_id: str, output_root: str | Path, **_kwargs: object):
        calls.append(arm_id)
        _persist_fake_audit(
            output_root,
            [
                _row(scene, candidate=arm_id == run_d92_afcp_g0.CANDIDATE_ARM)
                for scene in run_d92_afcp_g0.G0_SCENES
            ],
        )
        return {"fit_audit": [{"scenario": "ignored-return-value"}]}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    result = run_d92_afcp_g0.run(args)

    assert calls == [run_d92_afcp_g0.REFERENCE_ARM, run_d92_afcp_g0.CANDIDATE_ARM]
    assert result["marker"] == run_d92_afcp_g0.G0_MARKER
    persisted = json.loads(Path(args.g0_validation_path).read_text(encoding="utf-8"))
    assert persisted["status"] == run_d92_afcp_g0.G0_MARKER
    assert persisted["validation"]["pass"] is True
    assert set(persisted["validation"]["scenes"]) == set(run_d92_afcp_g0.G0_SCENES)


@pytest.mark.parametrize(
    "field, value",
    [
        ("d92_e0d_afcp_twofold_cross_guard_pass", False),
        ("d92_e0d_afcp_block_changed_code2_counts", [2, 2, 0]),
        ("d92_e0d_afcp_candidate_scan_count", 1),
        ("d92_e0d_afcp_query_truth_access", True),
    ],
)
def test_g0_rejects_failed_afcp_mechanism_or_protocol_gate(
    tmp_path, monkeypatch, field, value
) -> None:
    from scripts import run_d92_afcp_g0
    from cvsrffi import stage2_d92_e0d_query_evaluation as e0d

    args = _args(tmp_path)

    def fake_run(*, arm_id: str, output_root: str | Path, **_kwargs: object):
        rows = [
            _row(scene, candidate=arm_id == run_d92_afcp_g0.CANDIDATE_ARM)
            for scene in run_d92_afcp_g0.G0_SCENES
        ]
        if arm_id == run_d92_afcp_g0.CANDIDATE_ARM:
            rows[0][field] = value
        _persist_fake_audit(output_root, rows)
        return {}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    with pytest.raises(run_d92_afcp_g0.D92AFCPG0EntryError, match="validation failed"):
        run_d92_afcp_g0.run(args)
    persisted = json.loads(Path(args.g0_validation_path).read_text(encoding="utf-8"))
    assert persisted["status"] == run_d92_afcp_g0.G0_TECHNICAL_FAILURE
