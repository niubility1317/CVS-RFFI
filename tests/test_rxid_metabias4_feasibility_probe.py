from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_feasibility_probe import (
    CANDIDATE_ID,
    D103FeasibilityError,
    SCHEMA,
    cross_day_pair_summary,
    file_sha256,
    load_frozen_archives,
    validate_result_shape,
)


def test_cross_day_pair_summary_requires_each_receiver_tx_to_span_days() -> None:
    receiver = np.asarray(["r0", "r0", "r0", "r0"])
    day = np.asarray(["d0", "d1", "d0", "d1"])
    label = np.asarray(["a", "a", "b", "b"])
    physical = np.asarray(["p0", "p1", "p2", "p3"])
    result = cross_day_pair_summary(receiver, day, label, physical)
    assert result["all_receiver_tx_cross_day_constructible"] is True

    bad_day = np.asarray(["d0", "d1", "d0", "d0"])
    bad = cross_day_pair_summary(receiver, bad_day, label, physical)
    assert bad["all_receiver_tx_cross_day_constructible"] is False


def test_probe_result_schema_is_explicitly_non_performance() -> None:
    result = {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "FEASIBILITY_PROBE_NON_PERFORMANCE",
        "performance_metrics_computed": False,
        "target_access": False,
        "capsule_access": False,
        "formal_query_access": False,
        "deployment_asset_saved": False,
        "pair_constructability": {
            "all_receiver_tx_cross_day_constructible": True,
        },
        "k1_matrix_mechanics": {"rank": 4},
        "resource_measurement": {
            "warmup_loss_finite": True,
            "timed_loss_finite": True,
            "temporary_state_contains_learned_values": False,
            "temporary_state_deleted_before_return": True,
        },
    }
    validate_result_shape(result)


def test_loader_uses_dual_zdom_and_rejects_row_binding_drift(tmp_path) -> None:
    rows = 4
    z_id = np.arange(rows * 160, dtype=np.float32).reshape(rows, 160)
    z_dom = z_id + np.float32(1000.0)
    labels = np.asarray(["a", "a", "b", "b"])
    receivers = np.asarray(["r0", "r0", "r1", "r1"])
    days = np.asarray(["d0", "d1", "d0", "d1"])
    physical = np.asarray(["p0", "p1", "p2", "p3"])
    class_ids = np.asarray(["a", "b"])
    tap_path = tmp_path / "tap.npz"
    dual_path = tmp_path / "dual.npz"
    np.savez(
        tap_path,
        z_id=z_id,
        pre_relu=z_id + np.float32(2000.0),
        labels=labels,
        receiver_ids=receivers,
        day_ids=days,
        physical_ids=physical,
        class_ids=class_ids,
    )
    np.savez(
        dual_path,
        z_id=z_id,
        z_dom=z_dom,
        labels=labels,
        receiver_ids=receivers,
        day_ids=days,
        physical_ids=physical,
        class_ids=class_ids,
    )
    arrays = load_frozen_archives(
        tap_path,
        dual_path,
        file_sha256(tap_path),
        file_sha256(dual_path),
    )
    np.testing.assert_array_equal(arrays.z_dom, z_dom)
    assert not np.array_equal(arrays.z_dom, z_id)

    drift_path = tmp_path / "dual_drift.npz"
    np.savez(
        drift_path,
        z_id=z_id,
        z_dom=z_dom,
        labels=labels,
        receiver_ids=receivers,
        day_ids=days,
        physical_ids=np.asarray(["p0", "p1", "p2", "drift"]),
        class_ids=class_ids,
    )
    with pytest.raises(D103FeasibilityError, match="row binding failed"):
        load_frozen_archives(
            tap_path,
            drift_path,
            file_sha256(tap_path),
            file_sha256(drift_path),
        )
