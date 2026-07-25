from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from cvsrffi.phase1_rb_metabias4_bundle import build_phase1_rb_metabias4_bundle
from cvsrffi.rxid_metabias4_bundle import build_rxid_metabias4_bundle
from cvsrffi.rxid_metabias4_held_execution import (
    build_receiver_package_indices,
    canonical_sha256,
    predict_matched_row,
    score_prediction_artifact,
    validate_teacher_fit_manifest,
)


def _source():
    rng = np.random.default_rng(103713)
    receivers = [f"r{i}" for i in range(7)]
    classes = [f"c{i}" for i in range(6)]
    days = [f"d{i}" for i in range(4)]
    pre, zdom, labels, rx, day, physical = [], [], [], [], [], []
    for receiver_index, receiver in enumerate(receivers):
        for class_index, class_id in enumerate(classes):
            for day_index, day_id in enumerate(days):
                for sample in range(2):
                    p = rng.normal(0.05, 0.02, 160)
                    p[class_index] += 1.0
                    p[20 + receiver_index] += 0.08
                    p[40 + day_index] += 0.05
                    z = rng.normal(0.0, 0.01, 160)
                    z[receiver_index] += 1.0
                    z[10 + day_index] += 0.3
                    pre.append(p)
                    zdom.append(z)
                    labels.append(class_id)
                    rx.append(receiver)
                    day.append(day_id)
                    physical.append(
                        f"{receiver}-{class_id}-{day_id}-{sample}"
                    )
    return {
        "pre_relu": np.asarray(pre, dtype=np.float32),
        "z_dom": np.asarray(zdom, dtype=np.float32),
        "labels": np.asarray(labels, dtype=str),
        "receiver_ids": np.asarray(rx, dtype=str),
        "day_ids": np.asarray(day, dtype=str),
        "physical_ids": np.asarray(physical, dtype=str),
        "class_ids": np.asarray(classes, dtype=str),
    }


def _d103_bundle():
    rng = np.random.default_rng(103714)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    b = rng.normal(0.0, 0.08, (160, 4)).astype(np.float32)
    g = rng.normal(0.0, 0.2, (5, 32)).astype(np.float32)
    t = rng.normal(0.08, 0.02, (5, 4)).astype(np.float32)
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        np.full((5, 4), 4.0, dtype=np.float32),
        np.full(5, 1.8, dtype=np.float32),
        cell_min_physical_count=np.full(5, 2, dtype=np.int16),
        cell_class_count=np.full(5, 6, dtype=np.int16),
        checkpoint_sha256="1" * 64,
        runtime_sha256="2" * 64,
        method_lock_sha256="3" * 64,
        training_receipt_sha256="4" * 64,
        nested_receipt_sha256="5" * 64,
        tx_probe_receipt_sha256="6" * 64,
        aggregation_receipt_sha256="7" * 64,
        quantization_receipt_sha256="8" * 64,
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def test_matched_predictor_is_truth_free_and_independent_scorer_closes() -> None:
    source = _source()
    classes = source["class_ids"].tolist()
    support, query = build_receiver_package_indices(
        source["receiver_ids"],
        source["labels"],
        source["physical_ids"],
        held_receiver="r0",
        registered_classes=classes,
        k_shot=1,
    )
    d102 = build_phase1_rb_metabias4_bundle(
        source,
        checkpoint_sha256="1" * 64,
        runtime_sha256="2" * 64,
        method_lock_sha256="3" * 64,
        excluded_receivers=("r0",),
    )
    d103 = _d103_bundle()
    artifact, stability = predict_matched_row(
        held_receiver="r0",
        held_class=None,
        k_shot=1,
        support_pre_relu=source["pre_relu"][support],
        support_zdom=source["z_dom"][support],
        support_labels=source["labels"][support],
        query_pre_relu=source["pre_relu"][query],
        query_physical_ids=source["physical_ids"][query],
        registered_classes=classes,
        d102_bundle=d102,
        d103_outer_bundle=d103,
        d103_day_bundles=(d103, d103, d103, d103),
    )
    assert artifact["query_truth_present"] is False
    assert "query_truth_labels" not in artifact
    assert len(artifact["m0_predictions"]) == len(query)
    assert len(artifact["d102_predictions"]) == len(query)
    assert len(artifact["d103_predictions"]) == len(query)
    assert stability is not None
    assert stability["actual_160d_shift_used"] is True
    assert stability["direction_cosine_median"] > 0.999
    score = score_prediction_artifact(artifact, source["labels"][query])
    assert 0.0 <= score["base_ba"] <= 1.0
    assert 0.0 <= score["adapted_ba"] <= 1.0
    assert "prediction_artifact_committed_before_truth" not in score

    altered_query = np.asarray(source["pre_relu"][query][::-1], dtype=np.float32)
    altered, _ = predict_matched_row(
        held_receiver="r0",
        held_class=None,
        k_shot=1,
        support_pre_relu=source["pre_relu"][support],
        support_zdom=source["z_dom"][support],
        support_labels=source["labels"][support],
        query_pre_relu=altered_query,
        query_physical_ids=source["physical_ids"][query],
        registered_classes=classes,
        d102_bundle=d102,
        d103_outer_bundle=d103,
        d103_day_bundles=(d103, d103, d103, d103),
    )
    assert (
        altered["d103_state_receipt_sha256"]
        == artifact["d103_state_receipt_sha256"]
    )
    assert altered["d103_fit_audit"] == artifact["d103_fit_audit"]
    assert altered["k1_view_audit"]["held_query_rows_used"] == 0


def test_fit_manifest_outer_access_and_teacher_identity_are_fail_closed() -> None:
    bundle = _d103_bundle()
    teacher = {
        "U": bundle.decode_u().astype(np.float32),
        "B": bundle.decode_b().astype(np.float32),
        "bank_g": bundle.decode_bank_g().astype(np.float32),
        "bank_t": bundle.decode_bank_t().astype(np.float32),
        "bank_precision": bundle.decode_bank_precision().astype(np.float32),
        "bank_sigma": bundle.decode_bank_sigma().astype(np.float32),
    }

    def array_sha(value):
        array = np.ascontiguousarray(value)
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()

    access_body = {
        "schema": "cvs.d103_r2.rxid_metabias4.phase1_trainer.v1.access_ledger",
        "candidate_id": "D103-R2-RXID-CROSSRECEIVER-MB4",
        "events": [
            {
                "ordinal": 0,
                "role": "L_s",
                "operation": "vicreg",
                "fields": ["z_dom"],
                "row_count": 12,
            }
        ],
        "denied_attempts": 0,
        "source_val_array_access": False,
        "target_access": False,
        "formal_query_access": False,
        "performance_selection_access": False,
    }
    outer = {"held_receiver": "r0", "held_day": None, "held_class": "c0"}
    manifest = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.phase1_fit.v1",
        "candidate_id": "D103-R2-RXID-CROSSRECEIVER-MB4",
        "status": "PHASE1_FIT_COMPLETE_GROUND_TEACHER_NOT_DEPLOYMENT",
        "performance_metrics_computed": False,
        "target_access": False,
        "formal_query_access": False,
        "source_val_rows_used_for_training": 0,
        "completed_meta_steps": 400,
        "fit_elapsed_seconds": 1.0,
        "peak_cuda_memory_bytes": 1024,
        "checkpoint_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "outer_spec": outer,
        "input_sha256": {
            "labeled_archive": "3" * 64,
            "unlabeled_archive": "4" * 64,
            "source_val_seal": "5" * 64,
        },
        "teacher_archive": {
            "name": "teacher_arrays_fp32_ground_only.npz",
            "sha256": "6" * 64,
            "ground_only_fp32": True,
            "phase2_eligible": False,
        },
        "access_receipt": {
            **access_body,
            "receipt_sha256": canonical_sha256(access_body),
        },
        "aggregation_receipt": {
            "schema": (
                "cvs.d103_r2.rxid_metabias4.phase1_trainer.v1."
                "teacher_aggregation_receipt"
            ),
            "candidate_id": "D103-R2-RXID-CROSSRECEIVER-MB4",
            "completed_meta_steps": 400,
            "eligible_labeled_rows": 12,
            "unlabeled_rows_used": 0,
            "source_val_rows_used": 0,
            "bank_cell_count": 6,
            "registered_class_count": 6,
            "class_cell_count": 6,
            "minimum_physical_samples_per_class_cell": 2,
            "all_eligible_labeled_physical_rows_used": True,
            "aggregation_order": (
                "physical_mean_within_class_cell_then_equal_mean_over_classes"
            ),
            "class_weight": 1.0 / 6.0,
            "array_shapes": {
                name: list(np.asarray(value).shape)
                for name, value in teacher.items()
            },
            "contains_receiver_values": False,
            "contains_day_values": False,
            "contains_class_values": False,
            "contains_physical_ids": False,
            "contains_optimizer": False,
            "array_sha256": {name: array_sha(value) for name, value in teacher.items()},
        },
        "step_receipts_sha256": "7" * 64,
    }
    validate_teacher_fit_manifest(
        manifest,
        teacher,
        expected_outer_spec=outer,
        checkpoint_sha256="1" * 64,
        runtime_sha256="2" * 64,
        teacher_archive_sha256="6" * 64,
    )
    with pytest.raises(ValueError, match="identity/access"):
        validate_teacher_fit_manifest(
            manifest,
            teacher,
            expected_outer_spec={**outer, "held_receiver": "r1"},
            checkpoint_sha256="1" * 64,
            runtime_sha256="2" * 64,
            teacher_archive_sha256="6" * 64,
        )
    tampered = {
        **manifest,
        "access_receipt": {**manifest["access_receipt"], "target_access": True},
    }
    with pytest.raises(ValueError, match="access ledger"):
        validate_teacher_fit_manifest(
            tampered,
            teacher,
            expected_outer_spec=outer,
            checkpoint_sha256="1" * 64,
            runtime_sha256="2" * 64,
            teacher_archive_sha256="6" * 64,
        )
    illegal_access_body = {
        **access_body,
        "events": [
            {
                "ordinal": 0,
                "role": "U_s",
                "operation": "vicreg",
                "fields": ["tx_labels", "z_dom"],
                "row_count": 12,
            }
        ],
    }
    illegal_event = {
        **manifest,
        "access_receipt": {
            **illegal_access_body,
            "receipt_sha256": canonical_sha256(illegal_access_body),
        },
    }
    with pytest.raises(ValueError, match="access event closure"):
        validate_teacher_fit_manifest(
            illegal_event,
            teacher,
            expected_outer_spec=outer,
            checkpoint_sha256="1" * 64,
            runtime_sha256="2" * 64,
            teacher_archive_sha256="6" * 64,
        )
    with pytest.raises(ValueError, match="identity/access"):
        validate_teacher_fit_manifest(
            {**manifest, "unexpected": True},
            teacher,
            expected_outer_spec=outer,
            checkpoint_sha256="1" * 64,
            runtime_sha256="2" * 64,
            teacher_archive_sha256="6" * 64,
        )
