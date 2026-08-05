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


def _source_row_for_truth_registry(
    outer: matrix.Target125OuterRow,
) -> dict[str, object]:
    def reference(name: str) -> dict[str, str]:
        return {
            "package_root": f"/{name}",
            "detached_seal_path": f"/{name}.seal.json",
            "expected_seal_sha256": "a" * 64,
        }

    return {
        "receiver": outer.receiver,
        "seed": outer.seed,
        "k_shot": outer.k_shot,
        "new_count": outer.new_count,
        "active_k": outer.k_shot,
        "source_pool_k": outer.source_pool_k,
        "packages": {
            "before_enrollment": reference("before-enrollment"),
            "before_apply": reference("before-apply"),
            "after_enrollment": reference("after-enrollment"),
            "after_apply": reference("after-apply"),
        },
    }


def _sealed_apply_manifest(
    *,
    outer: matrix.Target125OuterRow,
    stage: str,
    registration_state: str,
    handles: list[str],
) -> dict[str, object]:
    return {
        "profile": "apply_only",
        "stage": stage,
        "registration_state": registration_state,
        "receiver": outer.receiver,
        "seed": outer.seed,
        "k_shot": outer.source_pool_k,
        "registered_class_count": len(handles),
        "registered_classes": [
            {"class_index": index, "class_handle": handle}
            for index, handle in enumerate(handles)
        ],
    }


def test_truth_registry_uses_sealed_row_local_d92_apply_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = next(
        row
        for row in matrix.freeze_next_r5_fa_target125_matrix().outer_rows
        if (row.k_shot, row.new_count) == (5, 20)
    )
    old = [f"cls_old_{index}" for index in range(matrix.OLD_CLASS_COUNT)]
    new = [f"cls_new_{index}" for index in range(outer.new_count)]
    seen: list[str] = []

    def preflight(package_root, **_kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(package_root))
        if str(package_root) == "/before-apply":
            return _sealed_apply_manifest(
                outer=outer,
                stage="stage2b",
                registration_state="before",
                handles=old,
            ), {}, {}
        assert str(package_root) == "/after-apply"
        return _sealed_apply_manifest(
            outer=outer,
            stage="stage2c",
            registration_state="after",
            handles=[*old, *new],
        ), {}, {}

    monkeypatch.setattr(target125, "_preflight_d92_apply_bundle", lambda reference: preflight(reference["package_root"]))
    actual_old, actual_new = target125._sealed_d92_apply_registry(  # noqa: SLF001
        source_row=_source_row_for_truth_registry(outer),
        outer=outer,
    )
    assert actual_old == tuple(old)
    assert actual_new == tuple(new)
    assert seen == ["/before-apply", "/after-apply"]


def test_truth_registry_rejects_d92_after_prefix_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = matrix.freeze_next_r5_fa_target125_matrix().outer_rows[0]
    old = [f"cls_old_{index}" for index in range(matrix.OLD_CLASS_COUNT)]

    def preflight(package_root, **_kwargs):  # type: ignore[no-untyped-def]
        if str(package_root) == "/before-apply":
            return _sealed_apply_manifest(
                outer=outer,
                stage="stage2b",
                registration_state="before",
                handles=old,
            ), {}, {}
        return _sealed_apply_manifest(
            outer=outer,
            stage="stage2c",
            registration_state="after",
            handles=["cls_wrong", *old[1:], *[f"cls_new_{index}" for index in range(outer.new_count)]],
        ), {}, {}

    monkeypatch.setattr(target125, "_preflight_d92_apply_bundle", lambda reference: preflight(reference["package_root"]))
    with pytest.raises(target125.NextR5FATarget125Error, match="old-class prefix"):
        target125._sealed_d92_apply_registry(  # noqa: SLF001
            source_row=_source_row_for_truth_registry(outer),
            outer=outer,
        )


def test_truth_registry_requires_sealed_prediction_registry_match() -> None:
    outer = matrix.freeze_next_r5_fa_target125_matrix().outer_rows[0]
    old = tuple(f"cls_old_{index}" for index in range(matrix.OLD_CLASS_COUNT))
    new = tuple(f"cls_new_{index}" for index in range(outer.new_count))
    records: dict[str, dict[str, list[str]]] = {}
    for scene in matrix.SCENES:
        scene_row_id = matrix.make_scene_row_id(outer.outer_id, scene)
        for state in matrix.STATES:
            registry = old if state.endswith("REG0") else (*old, *new)
            records[matrix.make_surface_id(scene_row_id, state)] = {
                "registered_classes": list(registry)
            }
    target125._validate_prediction_registry_binding(  # noqa: SLF001
        prediction={"records": records},
        outer=outer,
        old_classes=old,
        new_classes=new,
    )
    records[next(iter(records))]["registered_classes"] = ["cls_wrong", *old[1:]]
    with pytest.raises(target125.NextR5FATarget125Error, match="registry binding"):
        target125._validate_prediction_registry_binding(  # noqa: SLF001
            prediction={"records": records},
            outer=outer,
            old_classes=old,
            new_classes=new,
        )
