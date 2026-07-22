from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi import somph_diagnostic_enrollment_control as control


def test_control_rejects_source_and_formal_receivers() -> None:
    for receiver in ("1-1", "20-1"):
        with pytest.raises(
            control.SomphDiagnosticEnrollmentControlError,
            match="non-source and non-formal",
        ):
            control._validate_control(
                receiver=receiver,
                seed=713201,
                k_shot=10,
                old_tx_ids=("14-10", "14-7", "20-15", "20-19", "6-15", "8-20"),
                new_tx_ids=("1-16", "1-18", "18-10", "14-11", "8-3"),
            )


def test_control_freezes_k10_and_new5() -> None:
    with pytest.raises(
        control.SomphDiagnosticEnrollmentControlError,
        match="frozen at K=10",
    ):
        control._validate_control(
            receiver="1-20",
            seed=713201,
            k_shot=5,
            old_tx_ids=("14-10", "14-7", "20-15", "20-19", "6-15", "8-20"),
            new_tx_ids=("1-16", "1-18", "18-10", "14-11", "8-3"),
        )
    with pytest.raises(
        control.SomphDiagnosticEnrollmentControlError,
        match="frozen new5",
    ):
        control._validate_control(
            receiver="1-20",
            seed=713201,
            k_shot=10,
            old_tx_ids=("14-10", "14-7", "20-15", "20-19", "6-15", "8-20"),
            new_tx_ids=("1-16", "1-18", "18-10", "14-11", "bad"),
        )


def test_support_only_source_contains_no_query_writer_call() -> None:
    source = Path(control.__file__).read_text(encoding="utf-8")
    assert "profile=bundle.APPLY_ONLY" not in source
    assert '"query_payload_created": False' in source
    assert '"query_truth_opened": False' in source


def test_exact_k_reachability_rejects_hidden_k20_rows(tmp_path: Path) -> None:
    expected_classes = 2
    declared_k = 10
    for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS:
        labels = np.repeat(np.arange(expected_classes, dtype=np.int64), 20)
        ranks = np.tile(np.arange(20, dtype=np.int64), expected_classes)
        np.savez(
            tmp_path / f"support_{scenario}.npz",
            support_class_indices=labels,
            support_rank_within_class=ranks,
            manifest_json=np.asarray(
                '{"support_pool_max_k":20}'
            ),
        )
    with pytest.raises(
        control.SomphDiagnosticEnrollmentControlError,
        match="manifest K and physically reachable",
    ):
        control._assert_exact_k_reachability(
            tmp_path,
            registered_class_count=expected_classes,
            k_shot=declared_k,
        )


def test_exact_k_reachability_accepts_only_declared_rows(
    tmp_path: Path,
) -> None:
    expected_classes = 2
    declared_k = 10
    for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS:
        labels = np.repeat(
            np.arange(expected_classes, dtype=np.int64), declared_k
        )
        ranks = np.tile(
            np.arange(declared_k, dtype=np.int64), expected_classes
        )
        np.savez(
            tmp_path / f"support_{scenario}.npz",
            support_class_indices=labels,
            support_rank_within_class=ranks,
            manifest_json=np.asarray(
                '{"support_pool_max_k":10}'
            ),
        )
    assert control._assert_exact_k_reachability(
        tmp_path,
        registered_class_count=expected_classes,
        k_shot=declared_k,
    ) == {
        scenario: expected_classes * declared_k
        for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS
    }
