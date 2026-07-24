from __future__ import annotations

import numpy as np

from cvsrffi.rxid_metabias4_feasibility_probe import (
    SCHEMA,
    cross_day_pair_summary,
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
        },
    }
    validate_result_shape(result)
