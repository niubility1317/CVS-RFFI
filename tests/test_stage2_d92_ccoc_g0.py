from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi.stage2_d92_ccoc_g0 import (
    D92CCOCG0Error,
    G0_SCENES,
    _maximum_cross_group_margin_quantum,
    validate_ccoc_g0,
)


def _scene(*, candidate_margin: float = 3.0) -> tuple[dict, dict]:
    common = {
        "scene": "leo_clear_weak",
        "canonical_support_identity_sha256": "support-sha",
        "canonical_class_handles": ["old-0", "new-0"],
        "canonical_support_handles": ["row-0"],
        "cross_group_margin_by_support_handle": [
            {
                "canonical_row_handle": "row-0",
                "cross_group_margin": 1.0,
            }
        ],
        "support_block_absmax": {"z160": 2.0, "fft96": 1.0, "rf32": 1.0},
        "scale1_block_max_abs": [0.5, 1.0, 1.0],
        "scale2_block_max_abs": [0.5, 1.0, 1.0],
        "active": True,
        "fallback_active": False,
        "old_rho": 0.5,
        "new_rho": 1.0,
        "state_fingerprint_sha256": "e0-state",
        "actual_full_fit_count": 1,
        "query_macs": 123,
        "persistent_state_bytes": 456,
        "query_access": False,
        "truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "query_global_reassignment": False,
        "registration_wall_time_ns": 100_000_000,
        "registration_incremental_peak_working_set_bytes": 100,
    }
    reference = deepcopy(common)
    candidate = deepcopy(common)
    candidate["state_fingerprint_sha256"] = "ccoc-state"
    candidate["scale1_block_max_abs"] = [1.0, 0.75, 1.0]
    candidate["scale2_block_max_abs"] = [1.0, 0.75, 1.0]
    candidate["cross_group_margin_by_support_handle"][0][
        "cross_group_margin"
    ] = candidate_margin
    candidate["registration_wall_time_ns"] = 120_000_000
    candidate["registration_incremental_peak_working_set_bytes"] = 512_100
    return reference, candidate


def _wrapped(reference: dict, candidate: dict) -> tuple[dict, dict]:
    return _all_scenes(reference, candidate)


def _all_scenes(reference: dict, candidate: dict) -> tuple[dict, dict]:
    reference_rows = {}
    candidate_rows = {}
    for scene in G0_SCENES:
        reference_row = deepcopy(reference)
        candidate_row = deepcopy(candidate)
        reference_row["scene"] = scene
        candidate_row["scene"] = scene
        reference_rows[scene] = reference_row
        candidate_rows[scene] = candidate_row
    return {"scenes": reference_rows}, {"scenes": candidate_rows}


def test_quantum_uses_maximum_block_amplitude_and_all_four_scales() -> None:
    reference, candidate = _scene()
    candidate["scale1_block_max_abs"] = [1.0, 0.75, 1.0]
    candidate["scale2_block_max_abs"] = [1.0, 0.75, 1.0]

    assert _maximum_cross_group_margin_quantum(reference, candidate) == 2.0


def test_equal_quantum_margin_boundary_passes() -> None:
    reference, candidate = _scene(candidate_margin=3.0)
    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["cross_group_margin_quantum"] == 2.0
    assert result["max_cross_group_margin_change_abs"] == 2.0
    assert result["pass"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_active", True),
        ("actual_full_fit_count", 2),
        ("query_access", True),
        ("registration_wall_time_ns", 150_000_001),
        ("registration_incremental_peak_working_set_bytes", 1_048_577),
    ],
)
def test_each_frozen_gate_failure_rejects(field: str, value: object) -> None:
    reference, candidate = _scene()
    candidate[field] = value
    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["pass"] is False
    assert result["gates"][field] is False


def test_both_rho_endpoints_reject() -> None:
    reference, candidate = _scene()
    candidate["old_rho"] = 0.0
    candidate["new_rho"] = 1.0
    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["rho_interior"] is False
    assert result["pass"] is False


def test_margin_below_quantum_rejects() -> None:
    reference, candidate = _scene(candidate_margin=2.999)
    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["quantum"] is False
    assert result["pass"] is False


def test_identity_mismatch_is_structural_rejection() -> None:
    reference, candidate = _scene()
    candidate["canonical_support_identity_sha256"] = "different"

    with pytest.raises(D92CCOCG0Error, match="support identity"):
        validate_ccoc_g0(*_wrapped(reference, candidate))


def test_state_sha_must_differ_from_reference() -> None:
    reference, candidate = _scene()
    candidate["state_fingerprint_sha256"] = reference["state_fingerprint_sha256"]
    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["state"] is False
    assert result["pass"] is False


def test_candidate_peak_gate_is_absolute_and_reference_cannot_offset_it() -> None:
    reference, candidate = _scene()
    reference["registration_incremental_peak_working_set_bytes"] = 500_000
    candidate["registration_incremental_peak_working_set_bytes"] = 1_048_577

    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["peak"] is False
    assert (
        result["gates"]["registration_incremental_peak_working_set_bytes"]
        is False
    )
    assert result["pass"] is False


def test_candidate_peak_729088_passes_new_one_mib_hard_cap() -> None:
    reference, candidate = _scene()
    candidate["registration_incremental_peak_working_set_bytes"] = 729_088

    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["peak"] is True
    assert result["gates"]["registration_incremental_peak_working_set_bytes"] is True


def test_candidate_peak_1048577_fails_new_one_mib_hard_cap() -> None:
    reference, candidate = _scene()
    candidate["registration_incremental_peak_working_set_bytes"] = 1_048_577

    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"]["peak"] is False
    assert result["gates"]["registration_incremental_peak_working_set_bytes"] is False


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        (lambda reference, candidate: candidate.update(
            registration_wall_time_ns=150_000_001
        ), "wall"),
        (lambda reference, candidate: (
            reference.update(registration_wall_time_ns=99_999_999),
            candidate.update(registration_wall_time_ns=150_000_000),
        ), "ratio"),
        (lambda reference, candidate: candidate.update(query_macs=124), "query_macs"),
        (lambda reference, candidate: candidate.update(persistent_state_bytes=457), "state_bytes"),
    ],
)
def test_non_peak_hard_gates_remain_unchanged(mutation, gate: str) -> None:
    reference, candidate = _scene()
    mutation(reference, candidate)

    result = validate_ccoc_g0(*_wrapped(reference, candidate))

    assert result["gates"][gate] is False


def test_scene_collection_must_equal_all_three_frozen_scenes() -> None:
    reference, candidate = _scene()

    with pytest.raises(D92CCOCG0Error, match="G0 scene set"):
        validate_ccoc_g0(
            {"scenes": {"leo_clear_weak": reference}},
            {"scenes": {"leo_clear_weak": candidate}},
        )


def test_reference_and_candidate_query_disable_fields_are_all_gated() -> None:
    reference, candidate = _scene()
    reference["query_access"] = True
    reference_rows, candidate_rows = _all_scenes(reference, candidate)

    result = validate_ccoc_g0(reference_rows, candidate_rows)

    assert result["gates"]["reference_query_access"] is False
    assert result["gates"]["candidate_query_access"] is True
    assert result["gates"]["query"] is False
    assert result["pass"] is False


def test_validation_reports_candidate_wall_and_ratio_p90() -> None:
    reference, candidate = _scene()
    reference_rows, candidate_rows = _all_scenes(reference, candidate)
    reference_walls = [100_000_000, 100_000_000, 100_000_000]
    candidate_walls = [100_000_000, 120_000_000, 130_000_000]
    for index, scene in enumerate(G0_SCENES):
        reference_rows["scenes"][scene]["registration_wall_time_ns"] = (
            reference_walls[index]
        )
        candidate_rows["scenes"][scene]["registration_wall_time_ns"] = (
            candidate_walls[index]
        )

    result = validate_ccoc_g0(reference_rows, candidate_rows)

    assert result["candidate_wall_p90_ns"] == 130_000_000
    assert result["candidate_reference_ratio_p90"] == 1.3
