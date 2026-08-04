from __future__ import annotations

import hashlib

import pytest

from cvsrffi import stage2_next_r5_fa_target125_matrix as matrix
from cvsrffi import stage2_next_r5_fa_target125 as target125
from cvsrffi.stage2_next_r5_fa_target125 import (
    seal_k1_alias,
    seal_unique_prediction,
)


def test_k1_da1_alias_reuses_the_exact_da0_prediction_artifact(tmp_path) -> None:
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    source = next(
        surface
        for surface in plan.surfaces
        if surface.k_shot == 1 and surface.state == "DA0_REG0"
    )
    alias = next(
        surface
        for surface in plan.surfaces
        if surface.k_shot == 1 and surface.scene_row_id == source.scene_row_id and surface.state == "DA1_REG0"
    )
    classes = [f"old-{index}" for index in range(matrix.OLD_CLASS_COUNT)]
    record = seal_unique_prediction(
        output_dir=tmp_path,
        surface=source,
        registered_classes=classes,
        query_physical_ids=[f"query-{index}" for index in range(len(classes))],
        predicted_labels=classes,
        state_receipt={"fit_mode": "TEST", "query_rows_used_for_fit": 0},
    )
    reused = seal_k1_alias(
        surface=alias,
        source_record=record,
        state_receipt={
            "fit_mode": "FA_STRICT_BYPASS",
            "exact_prediction_alias": True,
            "alias_of_surface_id": source.surface_id,
        },
    )
    assert reused["prediction_artifact"] == record["prediction_artifact"]
    assert reused["prediction_artifact_sha256"] == record["prediction_artifact_sha256"]
    assert matrix.METRIC_AVAILABILITY["DA0_REG0"]["seen_new_acc"] == "N/A"
    assert matrix.METRIC_AVAILABILITY["DA1_REG1"]["H_old_new"] == "REQUIRED"


def test_truth_catalog_rejects_query_id_order_different_from_sealed_prediction(tmp_path) -> None:
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    sealed_query_ids = {
        surface.surface_id: (f"query:{surface.surface_id}",)
        for surface in plan.surfaces
    }
    surfaces = [
        {
            "surface_id": surface.surface_id,
            "ordered_query_physical_ids": list(sealed_query_ids[surface.surface_id]),
            "labels": ["old-0"],
        }
        for surface in plan.surfaces
    ]
    surfaces[0]["ordered_query_physical_ids"] = ["wrong-but-hash-consistent-query-id"]
    catalog = {
        "schema": target125.TRUTH_CATALOG_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "truth_open": True,
        "prediction_manifest_sha256": "a" * 64,
        "outer_job_count": matrix.OUTER_JOB_COUNT,
        "scene_row_count": matrix.SCENE_ROW_COUNT,
        "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        "surfaces": surfaces,
    }
    catalog["truth_catalog_sha256"] = target125.canonical_sha256(catalog)
    raw = target125._canonical_bytes(catalog) + b"\n"  # noqa: SLF001 - exact wire test.
    path = tmp_path / "truth_catalog.json"
    path.write_bytes(raw)
    with pytest.raises(
        target125.NextR5FATarget125Error,
        match="query physical-ID order",
    ):
        target125._load_target125_truth_catalog(  # noqa: SLF001 - negative boundary test.
            truth_catalog_path=path,
            expected_truth_catalog_file_sha256=hashlib.sha256(raw).hexdigest(),
            prediction_manifest_receipt_sha256="a" * 64,
            prediction_query_ids=sealed_query_ids,
        )


def test_da_query_id_parity_fails_closed_before_truth_open() -> None:
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    decoded = {}
    for surface in plan.surfaces:
        registration = "REG0" if surface.registration_phase == "REG0" else "REG1"
        query_ids = (f"query:{surface.scene_row_id}:{registration}",)
        decoded[surface.surface_id] = (query_ids, ("old-0",))
    drift = next(surface for surface in plan.surfaces if surface.state == "DA1_REG1")
    decoded[drift.surface_id] = (("wrong-da1-query",), ("old-0",))
    with pytest.raises(target125.NextR5FATarget125Error, match="DA0/DA1"):
        target125._validate_da_query_id_parity(plan, decoded)  # noqa: SLF001
