from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import somph_diagnostic_query_control as control


def _arrays() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for scenario_index, scenario in enumerate(
        control.bundle.FORMAL_LEO_WEAK_SCENARIOS
    ):
        rows = 40
        result[scenario] = {
            "dataset_role": np.asarray(["target_old"] * rows),
            "tx_ids": np.asarray(["tx"] * rows),
            "rx_ids": np.asarray(["1-20"] * rows),
            "sample_ids": np.asarray(
                [
                    f"{scenario_index}:{index}"
                    for index in range(rows)
                ]
            ),
        }
    return result


def test_select_support_query_is_exact_k10_q20_and_disjoint() -> None:
    selected, audit = control._select_support_query(
        _arrays(),
        receiver="1-20",
        seed=713201,
        k_shot=10,
        query_per_class=20,
        labels=[("target_old", "tx")],
    )
    for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS:
        support, query = selected[scenario][("target_old", "tx")]
        assert len(support) == 10
        assert len(query) == 20
        assert set(support).isdisjoint(query)
        assert len(audit[scenario][0]["support_physical_sample_ids"]) == 10
        assert len(audit[scenario][0]["query_physical_sample_ids"]) == 20


def test_verify_query_root_rejects_truth_like_npz_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for name in (
        "sealed_feature_runtime.pt",
        "method_lock.json",
        "overlay_provenance.json",
    ):
        (root / name).write_bytes(b"x")
    for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS:
        np.savez(
            root / f"query_{scenario}.npz",
            query_tokens=np.asarray(["qid_" + "0" * 64]),
            query_role=np.asarray(["target_old"]),
            manifest_json=np.asarray(
                json.dumps({"scenario": scenario})
            ),
        )
    seal = tmp_path / "seal.json"
    control._seal_query_root(
        root,
        seal_path=seal,
        stage="stage2b",
        registration_state="before",
        receiver="1-20",
        seed=713201,
        k_shot=10,
        query_per_class=20,
        registered_class_count=1,
        support_candidate_id="test",
        support_candidate_commit_sha256="0" * 64,
        support_candidate_state_sha256_by_scenario={
            scenario: {"metadata_sha256": "1" * 64, "npz_sha256": "2" * 64}
            for scenario in control.bundle.FORMAL_LEO_WEAK_SCENARIOS
        },
        strict_enrollment_package_root_sha256="3" * 64,
        strict_enrollment_package_seal_sha256="4" * 64,
    )
    with pytest.raises(
        control.SomphDiagnosticQueryControlError,
        match="truth-like keys",
    ):
        control._verify_query_root(root, seal_path=seal)
