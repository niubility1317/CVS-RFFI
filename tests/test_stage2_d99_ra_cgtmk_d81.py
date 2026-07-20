from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import inspect
import pickle

import numpy as np
import pytest

import cvsrffi.stage2_d99_ra_cgtmk_d81 as d99


# Design-report traceability record.  The assigned file boundary permits only
# this test and its core module, so verification evidence lives beside tests.
TRACEABILITY_MATRIX = (
    ("D99-P0-1", "Student-t volume term/effective dimension", "verified"),
    ("D99-P0-2", "exact typed INT8 aggregate bundle/receipt", "verified"),
    ("D99-P0-3", "sealed Y_old registry mapping", "verified"),
    (
        "D99-P0-4",
        "external Phase1 validation interface; authority not provisioned",
        "implemented",
    ),
    ("D99-P0-5", "resource/serialization/receipt closure", "verified"),
    ("D99-P0-6", "low-coverage ground-only fallback", "verified"),
    ("D99-BASE", "corrected typed D81 boundary; no generic fusion", "pending_review"),
)

CLASSES = ("old-a", "old-b", "old-c", "new-x")
GROUND_TARGETS = CLASSES[:3]
GROUND_CLASSES = GROUND_TARGETS


def _aggregation_receipt() -> d99.ExternalGroundAggregationReceipt:
    values = dict(
        schema=d99.GROUND_AGGREGATION_RECEIPT_SCHEMA,
        aggregation_manifest_sha256="a" * 64,
        producer_code_sha256="b" * 64,
        phase1_checkpoint_sha256="c" * 64,
        minimum_physical_sample_count=2,
        member_ids_present=False,
        target_rows_used=0,
        cryptographic_external_authority_claimed=False,
    )
    receipt_sha = d99._canonical_sha256(values)
    return d99.ExternalGroundAggregationReceipt(
        aggregation_manifest_sha256=values["aggregation_manifest_sha256"],
        producer_code_sha256=values["producer_code_sha256"],
        phase1_checkpoint_sha256=values["phase1_checkpoint_sha256"],
        receipt_sha256=receipt_sha,
    )


def _lock(
    *,
    bundle: d99.Phase1GroundAggregateBundle | None = None,
    margin_sha256: str = "3" * 64,
    **overrides,
) -> d99.Phase1D99Lock:
    if bundle is None:
        bundle = _bundle()
    values = dict(
        density_tau=0.2,
        max_ground_rank=4,
        max_target_rank=4,
        coverage_floor=0.01,
        ground_energy_scale=0.01,
        target_energy_scale=0.01,
        shrinkage_prior_strength=2.0,
        ground_weight_max=0.8,
        target_weight_max=0.6,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.5,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        z_weight=0.6,
        fft_weight=0.25,
        rf_weight=0.15,
        eta_k1=0.1,
        eta_k5=0.25,
        eta_k10=0.3,
        phase1_receipt_sha256="1" * 64,
        ground_aggregation_receipt_sha256=(
            bundle.aggregation_receipt.receipt_sha256
        ),
        ground_bundle_receipt_sha256=bundle.bundle_sha256,
        quantization_margin_audit_sha256=margin_sha256,
        validation_method_lock_sha256="9" * 64,
        d81_phase1_lock_sha256="4" * 64,
        ground_old_registry=GROUND_TARGETS,
    )
    values.update(overrides)
    return d99.Phase1D99Lock(**values)


def _ground_inputs(*, degenerate: bool = False):
    rng = np.random.default_rng(9901)
    domains, classes = 7, len(GROUND_CLASSES)
    base = rng.normal(size=(classes, d99.Z_DIM))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    directions, _ = np.linalg.qr(rng.normal(size=(d99.Z_DIM, 3)))
    coefficients = np.asarray(
        [[-1.0, 0.2, 0.7], [0.25, -0.9, 0.4], [1.2, 0.6, -0.8]]
    )
    coordinate = np.linspace(-1.0, 1.0, domains)
    domain_values = np.stack(
        [coordinate, np.sin(np.pi * coordinate), np.cos(np.pi * coordinate)], axis=1
    )
    if degenerate:
        domain_values[:] = 0.0
    grid = np.stack(
        [
            base
            + 0.12 * np.einsum("cr,r,zr->cz", coefficients, value, directions)
            for value in domain_values
        ]
    ).astype(np.float32)
    return (
        grid,
        np.ones((domains, classes), dtype=bool),
        tuple(f"domain-{index}" for index in range(domains)),
        base,
        directions,
        coefficients,
    )


def _bundle(*, degenerate: bool = False) -> d99.Phase1GroundAggregateBundle:
    grid, mask, domains, _base, _direction, _coefficients = _ground_inputs(
        degenerate=degenerate
    )
    maximum = np.max(np.abs(grid), axis=2)
    scales = np.maximum(maximum / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    codes = np.clip(
        np.rint(grid / scales.astype(np.float32)[:, :, None]), -127, 127
    ).astype(np.int8)
    return d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=np.full(mask.shape, 16, dtype=np.uint16),
        domain_ids=domains,
        ground_old_registry=GROUND_CLASSES,
        aggregation_receipt=_aggregation_receipt(),
    )


def _ground(*, degenerate: bool = False, config: d99.Phase1D99Lock | None = None):
    bundle = _bundle(degenerate=degenerate)
    if config is None:
        config = _lock(bundle=bundle)
    return d99.build_ground_geometry(
        bundle,
        config=config,
    )


def _support(k: int, *, zero_shift: bool = False, classes=CLASSES):
    rng = np.random.default_rng(9910 + k)
    grid, _mask, _domains, base, directions, coefficients = _ground_inputs()
    rows, labels, physical = [], [], []
    ground_mean = grid.mean(axis=0)
    for class_index, class_name in enumerate(classes):
        if class_name in GROUND_TARGETS:
            old_index = GROUND_TARGETS.index(class_name)
            center = ground_mean[old_index].astype(np.float64)
            if not zero_shift:
                center = center + 0.28 * coefficients[old_index, 0] * directions[:, 0]
        else:
            center = rng.normal(size=d99.Z_DIM)
        center = center / np.linalg.norm(center)
        for shot in range(k):
            z = center + (0.015 * rng.normal(size=d99.Z_DIM) if k > 1 else 0.0)
            fft = rng.normal(size=d99.FFT_DIM)
            fft[class_index] += 3.0
            rf = rng.normal(size=d99.RF_DIM)
            rf[class_index] += 2.0
            rows.append(np.concatenate([z, fft, rf]).astype(np.float32))
            labels.append(class_name)
            physical.append(f"physical-{class_name}-{shot}")
    return np.stack(rows), np.asarray(labels), np.asarray(physical)


def _fit_and_bank(
    k: int = 5,
    *,
    zero_shift: bool = False,
    margin_sha256: str = "3" * 64,
    **lock_overrides,
):
    bundle = _bundle()
    config = _lock(
        bundle=bundle, margin_sha256=margin_sha256, **lock_overrides
    )
    ground = d99.build_ground_geometry(bundle, config=config)
    features, labels, physical = _support(k, zero_shift=zero_shift)
    if zero_shift:
        for index, class_name in enumerate(GROUND_TARGETS):
            features[labels == class_name, : d99.Z_DIM] = ground.class_means_fp32[
                index
            ]
    metric = d99.fit_support_metric(
        ground,
        features,
        labels,
        physical,
        CLASSES,
        GROUND_TARGETS,
        config=config,
    )
    bank = d99.build_typed_support_bank(
        metric,
        features,
        labels,
        physical,
        CLASSES,
        config=config,
    )
    return config, ground, metric, bank, (features, labels, physical)


def _validation_source(
    support_receipt_sha256: str,
    validation_features: np.ndarray,
    *,
    episode_id: str = "phase1-lodo-rx03-k5",
    physical_ids: tuple[str, ...] | None = None,
):
    if physical_ids is None:
        physical_ids = tuple(
            f"phase1-validation-physical-{index}"
            for index in range(len(validation_features))
        )
    archive = d99._serialize_phase1_validation_archive(
        validation_features,
        physical_ids,
        episode_id,
    )
    producer = b"sealed d97 single-observation validation exporter\n"
    checkpoint = b"sealed phase1 checkpoint bytes\n"
    values = {
        "schema": d99.VALIDATION_RECEIPT_SCHEMA,
        "producer_id": d99.VALIDATION_PRODUCER_ID,
        "lifecycle": d99.VALIDATION_LIFECYCLE,
        "phase1_episode_id": episode_id,
        "phase1_episode_support_receipt_sha256": support_receipt_sha256,
        "feature_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "validation_manifest_sha256": "0" * 64,
        "producer_code_sha256": hashlib.sha256(producer).hexdigest(),
        "phase1_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "target_rows_used": 0,
        "query_rows_used": 0,
        "single_received_observation": True,
    }
    manifest = {
        "schema": d99.VALIDATION_MANIFEST_SCHEMA,
        "producer_id": d99.VALIDATION_PRODUCER_ID,
        "lifecycle": d99.VALIDATION_LIFECYCLE,
        "phase1_episode_id": episode_id,
        "phase1_episode_support_receipt_sha256": support_receipt_sha256,
        "feature_archive_sha256": values["feature_archive_sha256"],
        "producer_code_sha256": values["producer_code_sha256"],
        "phase1_checkpoint_sha256": values["phase1_checkpoint_sha256"],
        "target_rows_used": 0,
        "query_rows_used": 0,
        "single_received_observation": True,
    }
    manifest_bytes = d99._canonical_bytes(manifest)
    values["validation_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt = d99.ExternalPhase1ValidationReceipt(
        phase1_episode_id=episode_id,
        phase1_episode_support_receipt_sha256=values[
            "phase1_episode_support_receipt_sha256"
        ],
        feature_archive_sha256=values["feature_archive_sha256"],
        validation_manifest_sha256=values["validation_manifest_sha256"],
        producer_code_sha256=values["producer_code_sha256"],
        phase1_checkpoint_sha256=values["phase1_checkpoint_sha256"],
        receipt_sha256=d99._canonical_sha256(values),
    )
    method_lock = d99.Phase1ValidationMethodLock(
        expected_external_validation_receipt_sha256=receipt.receipt_sha256,
        allowlisted_producer_code_sha256=receipt.producer_code_sha256,
        allowlisted_phase1_checkpoint_sha256=receipt.phase1_checkpoint_sha256,
        allowlisted_feature_archive_sha256=receipt.feature_archive_sha256,
        allowlisted_validation_manifest_sha256=receipt.validation_manifest_sha256,
        expected_phase1_episode_support_receipt_sha256=support_receipt_sha256,
        expected_phase1_episode_id=episode_id,
    )
    authority_payload = {
        "schema": d99.AUTHORITY_ENVELOPE_SCHEMA,
        "validation_method_lock_sha256": method_lock.lock_digest,
        "expected_external_validation_receipt_sha256": receipt.receipt_sha256,
        "source_validation_lifecycle": d99.VALIDATION_LIFECYCLE,
        "single_received_observation": True,
        "target_rows_used": 0,
        "query_rows_used": 0,
    }
    authority_bytes = d99._canonical_bytes(authority_payload)
    externally_expected_authority_sha = hashlib.sha256(authority_bytes).hexdigest()
    authority_envelope = d99.load_phase1_authority_envelope(
        authority_envelope_bytes=authority_bytes,
        externally_expected_envelope_sha256=externally_expected_authority_sha,
        method_lock=method_lock,
    )
    return {
        "archive": archive,
        "manifest": manifest_bytes,
        "producer": producer,
        "checkpoint": checkpoint,
        "receipt": receipt,
        "method_lock": method_lock,
        "authority_bytes": authority_bytes,
        "externally_expected_authority_sha": externally_expected_authority_sha,
        "authority_envelope": authority_envelope,
    }


def _load_validation_artifact(source, config) -> d99.Phase1ValidationArtifact:
    return d99.load_phase1_validation_artifact(
        feature_archive_bytes=source["archive"],
        validation_manifest_bytes=source["manifest"],
        producer_code_bytes=source["producer"],
        phase1_checkpoint_bytes=source["checkpoint"],
        external_receipt=source["receipt"],
        method_lock=source["method_lock"],
        authority_envelope=source["authority_envelope"],
        config=config,
    )


def test_traceability_matrix_marks_only_corrected_typed_d81_as_pending() -> None:
    assert len(TRACEABILITY_MATRIX) == 7
    assert all(status in {"implemented", "verified", "pending_review"} for _id, _requirement, status in TRACEABILITY_MATRIX)
    pending = [item for item in TRACEABILITY_MATRIX if item[2] == "pending_review"]
    assert pending == [TRACEABILITY_MATRIX[-1]]


def test_ground_builder_is_density_aware_rank_bounded_and_aggregate_only() -> None:
    model = _ground()
    assert model.nuisance_basis_fp32.shape[1] <= 4
    assert model.effective_domain_count >= 1.0
    assert np.sum(model.density_weights_fp32) == pytest.approx(1.0)
    assert model.coverage_certificate["aggregate_only"] is True
    assert model.coverage_certificate["member_or_sample_ids_stored"] is False
    assert model.coverage_certificate["ground_class_score_access"] is False
    assert model.coverage_certificate["typed_int8_aggregate_bundle"] is True
    assert model.coverage_certificate["physical_sample_count_floor_min"] >= 2
    assert not any("sample" in name for name in vars(model))
    assert model.class_means_fp32.flags.writeable is False


def test_ground_mask_registry_and_lock_drift_fail_closed() -> None:
    grid, mask, domains, *_ = _ground_inputs()
    with pytest.raises(d99.D99RACGTMKError, match="exact typed bundle"):
        d99.build_ground_geometry(grid, config=_lock())
    valid = _bundle()
    bad_mask = np.array(valid.domain_class_mask, copy=True)
    bad_mask[-1, -1] = False
    with pytest.raises(d99.D99RACGTMKError, match="bundle invariant"):
        replace(valid, domain_class_mask=bad_mask)
    with pytest.raises(d99.D99RACGTMKError, match="unique"):
        d99.produce_typed_ground_aggregate_bundle(
            codes_qint8=valid.codes_qint8,
            scales_fp16=valid.scales_fp16,
            domain_class_mask=valid.domain_class_mask,
            physical_sample_count_floor_uint16=(
                valid.physical_sample_count_floor_uint16
            ),
            domain_ids=domains[:-1] + (domains[0],),
            ground_old_registry=GROUND_CLASSES,
            aggregation_receipt=_aggregation_receipt(),
        )
    count_floor = np.array(valid.physical_sample_count_floor_uint16, copy=True)
    count_floor[0, 0] = 1
    with pytest.raises(d99.D99RACGTMKError, match="bundle invariant"):
        d99.produce_typed_ground_aggregate_bundle(
            codes_qint8=valid.codes_qint8,
            scales_fp16=valid.scales_fp16,
            domain_class_mask=valid.domain_class_mask,
            physical_sample_count_floor_uint16=count_floor,
            domain_ids=valid.domain_ids,
            ground_old_registry=GROUND_CLASSES,
            aggregation_receipt=_aggregation_receipt(),
        )
    with pytest.raises(d99.D99RACGTMKError, match="ranks"):
        _lock(max_ground_rank=5)


def test_k1_has_zero_target_rank_shared_h0_and_can_use_ground_metric() -> None:
    _config, _ground_model, metric, bank, _values = _fit_and_bank(1)
    assert metric.target_basis_fp32.shape == (d99.Z_DIM, 0)
    assert metric.fit_audit["k1_target_rank_exact_zero"] is True
    np.testing.assert_array_equal(
        bank.class_scales_fp16,
        np.full(len(CLASSES), np.float16(_lock().shared_h0), dtype=np.float16),
    )
    assert bank.quantization_audit["class_scale_source"] == "phase1_locked_shared_h0"
    assert metric.ground_coverage_rho > 0.0
    assert not metric.is_identity


def test_low_ground_coverage_keeps_k5_target_metric_but_k1_is_identity() -> None:
    config, ground, metric, _bank, _values = _fit_and_bank(5, zero_shift=True)
    assert metric.ground_coverage_rho == pytest.approx(0.0, abs=1e-9)
    assert not metric.is_identity
    assert metric.ground_weight == 0.0 and metric.target_weight > 0.0
    assert metric.fit_audit["target_metric_survives_low_ground_coverage"] is True
    features, labels, physical = _support(5)
    degenerate_bundle = _bundle(degenerate=True)
    degenerate_config = _lock(bundle=degenerate_bundle)
    degenerate = d99.build_ground_geometry(
        degenerate_bundle, config=degenerate_config
    )
    assert degenerate.effective_domain_count == pytest.approx(1.0)
    metric2 = d99.fit_support_metric(
        degenerate,
        features,
        labels,
        physical,
        CLASSES,
        GROUND_TARGETS,
        config=degenerate_config,
    )
    assert metric2.ground_weight == 0.0
    assert metric2.target_weight > 0.0
    _c1, _g1, k1_metric, _b1, _v1 = _fit_and_bank(1, zero_shift=True)
    assert k1_metric.target_basis_fp32.shape[1] == 0
    assert k1_metric.is_identity


def test_metric_is_strictly_psd_and_uses_no_coordinate_transport() -> None:
    _config, _ground_model, metric, _bank, _values = _fit_and_bank(5)
    eye = np.eye(d99.Z_DIM)
    precision = np.stack([metric.apply_precision(row[None, :])[0] for row in eye])
    eigenvalues = np.linalg.eigvalsh(0.5 * (precision + precision.T))
    assert float(np.min(eigenvalues)) > 0.0
    assert metric.fit_audit["precision_formula"].startswith("I-B")
    assert metric.fit_audit["full_coordinate_transport"] is False
    assert metric.fit_audit["ground_class_logit_or_bonus"] is False


def test_support_and_class_permutation_are_equivariant() -> None:
    config, ground, metric, bank, values = _fit_and_bank(5)
    features, labels, physical = values
    order = np.arange(len(features))[::-1]
    metric_ordered = d99.fit_support_metric(
        ground,
        features[order],
        labels[order],
        physical[order],
        CLASSES,
        GROUND_TARGETS,
        config=config,
    )
    bank_ordered = d99.build_typed_support_bank(
        metric_ordered,
        features[order],
        labels[order],
        physical[order],
        CLASSES,
        config=config,
    )
    assert metric_ordered.metric_receipt_sha256 == metric.metric_receipt_sha256
    np.testing.assert_array_equal(bank_ordered.codes_qint8, bank.codes_qint8)

    permutation = np.asarray([2, 0, 3, 1])
    permuted = tuple(CLASSES[index] for index in permutation)
    metric_permuted = d99.fit_support_metric(
        ground,
        features,
        labels,
        physical,
        permuted,
        GROUND_TARGETS,
        config=config,
    )
    bank_permuted = d99.build_typed_support_bank(
        metric_permuted,
        features,
        labels,
        physical,
        permuted,
        config=config,
    )
    query = _support(1)[0]
    original_logits = d99.score_metric_kernel_raw_logits(bank, query)
    permuted_logits = d99.score_metric_kernel_raw_logits(bank_permuted, query)
    np.testing.assert_allclose(permuted_logits, original_logits[:, permutation], atol=1e-6)


def test_old_and_new_classes_share_student_t_formula_and_no_count_bonus() -> None:
    _config, _ground_model, metric, bank, values = _fit_and_bank(5)
    logits = d99.score_metric_kernel_raw_logits(bank, values[0][:4])
    assert logits.shape == (4, len(CLASSES))
    assert metric.fit_audit["old_new_role_specific_scoring"] is False
    assert bank.quantization_audit["same_formula_all_registered_classes"] is True
    assert bank.quantization_audit["class_count_normalization"] == "logsumexp_minus_log_Kc"


def test_student_t_zero_distance_has_scale_volume_term() -> None:
    config, _ground_model, metric, _bank, values = _fit_and_bank(1)
    one = d99.normalize_feature_blocks(values[0][:1])
    support = np.repeat(one, len(CLASSES), axis=0)
    scales = np.asarray([0.25, 1.0, 1.0, 1.0], dtype=np.float16)
    logits = d99._student_t_logits(
        support,
        np.arange(len(CLASSES), dtype=np.int16),
        scales,
        metric,
        config,
        values[0][:1],
    )
    expected = (
        -config.kernel_volume_gamma
        * config.kernel_effective_dim
        * (np.log(0.25) - np.log(1.0))
    )
    assert logits[0, 0] - logits[0, 1] == pytest.approx(expected, abs=2e-6)


def test_ground_mapping_cannot_redirect_a_sealed_old_class_to_new() -> None:
    config, ground, _metric, _bank, values = _fit_and_bank(5)
    with pytest.raises(d99.D99RACGTMKError, match="sealed Y_old"):
        d99.fit_support_metric(
            ground,
            *values,
            CLASSES,
            (GROUND_TARGETS[0], GROUND_TARGETS[1], CLASSES[-1]),
            config=config,
        )


@pytest.mark.parametrize("failure", ["missing", "unbalanced", "duplicate"])
def test_missing_unbalanced_or_duplicate_physical_support_is_rejected(failure: str) -> None:
    config = _lock()
    ground = _ground(config=config)
    features, labels, physical = _support(5)
    if failure == "missing":
        keep = labels != CLASSES[-1]
        features, labels, physical = features[keep], labels[keep], physical[keep]
    elif failure == "unbalanced":
        features, labels, physical = features[:-1], labels[:-1], physical[:-1]
    else:
        physical[1] = physical[0]
    with pytest.raises(d99.D99RACGTMKError):
        d99.fit_support_metric(
            ground,
            features,
            labels,
            physical,
            CLASSES,
            GROUND_TARGETS,
            config=config,
        )


def test_receipt_and_config_tampering_fail_before_bank_or_score() -> None:
    config, ground_model, metric, bank, values = _fit_and_bank(5)
    tampered_ground = replace(ground_model, geometry_receipt_sha256="c" * 64)
    with pytest.raises(d99.D99RACGTMKError, match="geometry receipt"):
        d99.fit_support_metric(
            tampered_ground,
            *values,
            CLASSES,
            GROUND_TARGETS,
            config=config,
        )
    changed = replace(config, student_nu=4.0)
    with pytest.raises(d99.D99RACGTMKError, match="receipt drift"):
        d99.build_typed_support_bank(
            metric, *values, CLASSES, config=changed
        )
    with pytest.raises(d99.D99RACGTMKError, match="numeric resource/receipt"):
        replace(metric, metric_receipt_sha256="a" * 64)
    changed_metric_resource = dict(metric.resource_audit)
    changed_metric_resource["residual_covariance_mac_upper_bound"] = 0
    resigned_metric_payload = {
        "schema": d99.METRIC_SCHEMA,
        "classes": list(metric.classes),
        "k_shot": metric.k_shot,
        "metric_basis": d99._array_receipt(metric.metric_basis_fp32),
        "precision_attenuation": d99._array_receipt(
            metric.precision_attenuation_fp32
        ),
        "target_basis": d99._array_receipt(metric.target_basis_fp32),
        "ground_coverage_rho": metric.ground_coverage_rho,
        "target_shift_energy": metric.target_shift_energy,
        "ground_weight": metric.ground_weight,
        "target_weight": metric.target_weight,
        "ground_domain_count": metric.ground_domain_count,
        "ground_class_count": metric.ground_class_count,
        "ground_rank": metric.ground_rank,
        "support_input_sha256": metric.support_input_sha256,
        "ground_geometry_receipt_sha256": metric.ground_geometry_receipt_sha256,
        "config_lock_digest": metric.config_lock_digest,
        "fit_audit": metric.fit_audit,
        "resource_audit": changed_metric_resource,
    }
    with pytest.raises(d99.D99RACGTMKError, match="numeric resource/receipt"):
        replace(
            metric,
            resource_audit=changed_metric_resource,
            metric_receipt_sha256=d99._canonical_sha256(resigned_metric_payload),
        )
    forged_metric = copy.copy(metric)
    object.__setattr__(forged_metric, "resource_audit", changed_metric_resource)
    object.__setattr__(
        forged_metric,
        "metric_receipt_sha256",
        d99._canonical_sha256(resigned_metric_payload),
    )
    assert d99._verify_metric_numeric_resource(forged_metric) is False
    forged_bank_resource, forged_bank_receipt, _wire = (
        d99._closed_bank_resource_receipt_artifact(
            classes=bank.classes,
            counts=bank.support_counts,
            codes=bank.codes_qint8,
            scales=bank.scales_fp16,
            indices=bank.class_indices_int16,
            class_scales=bank.class_scales_fp16,
            metric=forged_metric,
            config=bank.config,
            quantization=bank.quantization_audit,
        )
    )
    with pytest.raises(d99.D99RACGTMKError, match="resource/receipt closure"):
        d99.TypedINT8MetricKernelBank(
            classes=bank.classes,
            support_counts=bank.support_counts,
            codes_qint8=bank.codes_qint8,
            scales_fp16=bank.scales_fp16,
            class_indices_int16=bank.class_indices_int16,
            class_scales_fp16=bank.class_scales_fp16,
            metric=forged_metric,
            config=bank.config,
            eta_phase1_locked=bank.eta_phase1_locked,
            bank_receipt_sha256=forged_bank_receipt,
            quantization_audit=bank.quantization_audit,
            resource_audit=forged_bank_resource,
        )
    forged_bank = copy.copy(bank)
    object.__setattr__(forged_bank, "metric", forged_metric)
    object.__setattr__(forged_bank, "resource_audit", forged_bank_resource)
    object.__setattr__(forged_bank, "bank_receipt_sha256", forged_bank_receipt)
    with pytest.raises(d99.D99RACGTMKError, match="metric numeric resource"):
        d99.score_metric_kernel_raw_logits(forged_bank, values[0][:1])
    with pytest.raises(d99.D99RACGTMKError, match="serialization numeric resource"):
        d99._serialize_receipt_bearing_bank(forged_bank)
    with pytest.raises(d99.D99RACGTMKError, match="resource/receipt closure"):
        replace(bank, bank_receipt_sha256="b" * 64)
    changed_basis = np.array(metric.metric_basis_fp32, copy=True)
    changed_basis[:, 0] *= -1.0
    with pytest.raises(d99.D99RACGTMKError, match="numeric resource/receipt"):
        replace(metric, metric_basis_fp32=changed_basis)


def test_quantized_margin_audit_and_query_batch_equivalence() -> None:
    values = _support(5)
    support_receipt = d99._support_closure(*values, CLASSES)[-1]
    query = _support(1)[0]
    source = _validation_source(support_receipt, query)
    validation_lock_sha = source["method_lock"].lock_digest
    config, _ground_model, _metric, provisional_bank, values = _fit_and_bank(
        5, validation_method_lock_sha256=validation_lock_sha
    )
    validation = _load_validation_artifact(source, config)
    development = d99.diagnose_quantized_margin_development(
        provisional_bank, *values, validation
    )
    expected_audit_sha = development["development_diagnostic_sha256"]
    assert development["authority_status"] == "BLOCKED"
    assert development["formal_phase1_eligible"] is False
    assert development["matches_phase1_lock"] is False
    with pytest.raises(d99.D99RACGTMKError, match="authority is not provisioned"):
        d99.precompute_phase1_quantized_margin_audit_sha256(
            provisional_bank, *values, validation
        )
    with pytest.raises(d99.D99RACGTMKError, match="authority is not provisioned"):
        d99.audit_quantized_margin(provisional_bank, *values, validation)
    _sealed, _g2, _m2, bank, sealed_values = _fit_and_bank(
        5,
        margin_sha256=expected_audit_sha,
        validation_method_lock_sha256=validation_lock_sha,
    )
    together = d99.score_metric_kernel_raw_logits(bank, query)
    separate = np.concatenate(
        [d99.score_metric_kernel_raw_logits(bank, query[index : index + 1]) for index in range(len(query))],
        axis=0,
    )
    np.testing.assert_allclose(together, separate, atol=1e-7)
    with pytest.raises(d99.D99RACGTMKError, match="typed Phase1"):
        d99.audit_quantized_margin(bank, *sealed_values, query)
    with pytest.raises(d99.D99RACGTMKError, match="authority is not provisioned"):
        d99.audit_quantized_margin(bank, *sealed_values, validation)
    diagnostic = d99.diagnose_quantized_margin_development(
        bank, *sealed_values, validation
    )
    assert 0.0 <= diagnostic["top1_agreement"] <= 1.0
    assert 0.0 <= diagnostic["margin_sign_flip_rate"] <= 1.0
    assert diagnostic["validation_row_count"] == len(query)
    assert diagnostic["development_diagnostic_sha256"] == expected_audit_sha
    assert diagnostic["development_lock_digest_matches"] is True
    assert diagnostic["formal_phase1_eligible"] is False
    assert diagnostic["matches_phase1_lock"] is False
    assert diagnostic["formal_result_claimed"] is False


def test_phase1_validation_source_is_external_exact_and_fail_closed() -> None:
    values = _support(5)
    support_receipt = d99._support_closure(*values, CLASSES)[-1]
    validation_features = _support(1)[0]
    source = _validation_source(support_receipt, validation_features)
    config, _ground_model, _metric, bank, _values = _fit_and_bank(
        5, validation_method_lock_sha256=source["method_lock"].lock_digest
    )
    artifact = _load_validation_artifact(source, config)
    assert config.validation_method_lock_sha256 == source["method_lock"].lock_digest
    assert source["method_lock"].expected_external_validation_receipt_sha256 == (
        source["receipt"].receipt_sha256
    )
    assert artifact.external_receipt.phase1_checkpoint_sha256 == hashlib.sha256(
        source["checkpoint"]
    ).hexdigest()
    assert artifact.external_receipt.feature_archive_sha256
    assert artifact.authority_envelope.authority_status == "BLOCKED"
    assert artifact.authority_envelope.formal_phase1_eligible is False
    assert d99.TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256 is None
    assert not hasattr(d99, "produce_phase1_validation_artifact")
    with pytest.raises(d99.D99RACGTMKError, match="lifecycle drift"):
        replace(artifact, loader_token=object())
    copied_variants = (
        copy.copy(artifact),
        copy.deepcopy(artifact),
        pickle.loads(pickle.dumps(artifact)),
    )
    for copied in copied_variants:
        with pytest.raises(d99.D99RACGTMKError, match="authority is not provisioned"):
            d99.audit_quantized_margin(bank, *_values, copied)

    with pytest.raises(d99.D99RACGTMKError, match="not Phase1-source-only"):
        _validation_source(
            support_receipt,
            validation_features,
            physical_ids=("target-query-physical",)
            + tuple(
                f"phase1-val-{index}" for index in range(1, len(validation_features))
            ),
        )

    for key, bad_value in (
        ("archive", source["archive"] + b"tamper"),
        ("manifest", source["manifest"] + b"tamper"),
        ("producer", source["producer"] + b"tamper"),
        ("checkpoint", b"wrong checkpoint"),
    ):
        actual = dict(source)
        actual[key] = bad_value
        with pytest.raises(d99.D99RACGTMKError, match="source bytes"):
            d99.load_phase1_validation_artifact(
                feature_archive_bytes=actual["archive"],
                validation_manifest_bytes=actual["manifest"],
                producer_code_bytes=actual["producer"],
                phase1_checkpoint_bytes=actual["checkpoint"],
                external_receipt=source["receipt"],
                method_lock=source["method_lock"],
                authority_envelope=source["authority_envelope"],
                config=config,
            )

    other_values = d99._validation_receipt_payload(artifact.external_receipt)
    other_values["phase1_episode_support_receipt_sha256"] = "6" * 64
    other = d99.ExternalPhase1ValidationReceipt(
        phase1_episode_id=other_values["phase1_episode_id"],
        phase1_episode_support_receipt_sha256=other_values[
            "phase1_episode_support_receipt_sha256"
        ],
        feature_archive_sha256=other_values["feature_archive_sha256"],
        validation_manifest_sha256=other_values["validation_manifest_sha256"],
        producer_code_sha256=other_values["producer_code_sha256"],
        phase1_checkpoint_sha256=other_values["phase1_checkpoint_sha256"],
        receipt_sha256=d99._canonical_sha256(other_values),
    )
    with pytest.raises(d99.D99RACGTMKError, match="authority lock mismatch"):
        d99.load_phase1_validation_artifact(
            feature_archive_bytes=source["archive"],
            validation_manifest_bytes=source["manifest"],
            producer_code_bytes=source["producer"],
            phase1_checkpoint_bytes=source["checkpoint"],
            external_receipt=other,
            method_lock=source["method_lock"],
            authority_envelope=source["authority_envelope"],
            config=config,
        )


def test_resource_closure_is_incremental_and_optimizer_free() -> None:
    _config, _ground_model, metric, bank, _values = _fit_and_bank(10)
    resource = bank.resource_audit
    assert resource["logical_runtime_numeric_state_bytes"] > 0
    assert resource["actual_serialized_runtime_artifact_bytes"] > resource[
        "logical_runtime_numeric_state_bytes"
    ]
    assert resource["support_fit_mac_upper_bound"] == metric.resource_audit[
        "support_fit_mac_upper_bound"
    ]
    assert resource["query_mac_upper_bound"] > 0
    assert resource["support_decode_normalize_mac_per_prediction_call"] > 0
    assert resource["peak_transient_bytes_upper_bound"] > 0
    assert metric.resource_audit["residual_covariance_mac_upper_bound"] >= (
        len(CLASSES) * 10 * d99.Z_DIM * d99.Z_DIM
    )
    assert resource["optimizer_steps"] == 0
    assert resource["optimizer_steps_scope"] == "D99_incremental_only"
    assert resource["d81_base_fit_included"] is False
    assert resource["d81_base_single_fit_resource_status"] == (
        "BLOCKED_CORRECTED_TYPED_D81_REVIEW_P0"
    )
    assert resource["total_combined_resource_status"] == "BLOCKED_NOT_CLAIMED"
    assert resource["complete_method_resource_claim"] is False
    assert resource["scope"] == "D99_incremental_only"
    lean_metadata = d99._bank_metadata(
        classes=bank.classes,
        counts=bank.support_counts,
        codes=bank.codes_qint8,
        scales=bank.scales_fp16,
        indices=bank.class_indices_int16,
        class_scales=bank.class_scales_fp16,
        metric=bank.metric,
        config=bank.config,
        quantization=bank.quantization_audit,
        resource=None,
    )
    lean_bytes = d99._serialize_runtime_artifact(
        lean_metadata,
        d99._bank_runtime_arrays(
            codes=bank.codes_qint8,
            scales=bank.scales_fp16,
            indices=bank.class_indices_int16,
            class_scales=bank.class_scales_fp16,
            metric=bank.metric,
        ),
    )
    actual_bytes = d99._serialize_receipt_bearing_bank(bank)
    assert len(actual_bytes) == resource["actual_serialized_runtime_artifact_bytes"]
    assert len(actual_bytes) > len(lean_bytes)
    assert b'"resource_audit"' in actual_bytes
    assert b'"bank_receipt_sha256"' in actual_bytes
    for key, value in (
        ("query_mac_upper_bound", 0),
        ("actual_serialized_runtime_artifact_bytes", 1),
        ("peak_transient_bytes_upper_bound", 1),
    ):
        changed_resource = dict(resource)
        changed_resource[key] = value
        resigned_bank = d99._canonical_sha256(
            d99._bank_metadata(
                classes=bank.classes,
                counts=bank.support_counts,
                codes=bank.codes_qint8,
                scales=bank.scales_fp16,
                indices=bank.class_indices_int16,
                class_scales=bank.class_scales_fp16,
                metric=bank.metric,
                config=bank.config,
                quantization=bank.quantization_audit,
                resource=changed_resource,
            )
        )
        with pytest.raises(d99.D99RACGTMKError, match="resource/receipt closure"):
            replace(
                bank,
                resource_audit=changed_resource,
                bank_receipt_sha256=resigned_bank,
            )


def test_c2_k1_rank1_query_mac_includes_query_precision_norm() -> None:
    grid, mask, domains, *_ = _ground_inputs()
    classes = GROUND_CLASSES[:2]
    grid = grid[:, :2]
    mask = mask[:, :2]
    maximum = np.max(np.abs(grid), axis=2)
    scales = np.maximum(maximum / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    codes = np.clip(
        np.rint(grid / scales.astype(np.float32)[:, :, None]), -127, 127
    ).astype(np.int8)
    bundle = d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=np.full(mask.shape, 16, dtype=np.uint16),
        domain_ids=domains,
        ground_old_registry=classes,
        aggregation_receipt=_aggregation_receipt(),
    )
    config = _lock(
        bundle=bundle,
        ground_old_registry=classes,
        max_ground_rank=1,
        max_target_rank=1,
    )
    ground = d99.build_ground_geometry(bundle, config=config)
    features, labels, physical = _support(1, classes=classes)
    metric = d99.fit_support_metric(
        ground, features, labels, physical, classes, classes, config=config
    )
    assert metric.metric_basis_fp32.shape[1] == 1
    bank = d99.build_typed_support_bank(
        metric, features, labels, physical, classes, config=config
    )
    resource = bank.resource_audit
    assert resource["query_kernel_pair_mac_upper_bound"] == 1856
    assert resource["query_precision_norm_mac_upper_bound"] == 160
    assert resource["query_kernel_mac_upper_bound"] == 2016


def test_d14_c6_ground_peak_counts_all_live_dense_float64_arrays() -> None:
    rng = np.random.default_rng(990099)
    domains = tuple(f"ground-domain-{index}" for index in range(14))
    classes = tuple(f"ground-old-{index}" for index in range(6))
    grid = rng.normal(size=(14, 6, d99.Z_DIM)).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    mask = np.ones((14, 6), dtype=bool)
    maximum = np.max(np.abs(grid), axis=2)
    scales = np.maximum(maximum / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    codes = np.clip(
        np.rint(grid / scales.astype(np.float32)[:, :, None]), -127, 127
    ).astype(np.int8)
    bundle = d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=np.full(mask.shape, 16, dtype=np.uint16),
        domain_ids=domains,
        ground_old_registry=classes,
        aggregation_receipt=_aggregation_receipt(),
    )
    config = _lock(bundle=bundle, ground_old_registry=classes)
    ground = d99.build_ground_geometry(bundle, config=config)
    assert ground.coverage_certificate[
        "ground_build_peak_live_dcz_float64_array_count"
    ] == 6
    expected = (
        14 * 6 * d99.Z_DIM * np.dtype(np.float32).itemsize
        + 6 * 14 * 6 * d99.Z_DIM * np.dtype(np.float64).itemsize
        + 5 * d99.Z_DIM * d99.Z_DIM * np.dtype(np.float64).itemsize
        + 2 * 14 * 14 * np.dtype(np.float64).itemsize
        + 4 * 6 * d99.Z_DIM * np.dtype(np.float64).itemsize
    )
    assert expected == 1_756_736
    assert ground.coverage_certificate[
        "ground_build_peak_transient_bytes_upper_bound"
    ] == expected


def test_public_surface_has_no_generic_fuse_or_base_logit_entry() -> None:
    assert d99.DEPLOYMENT_STATUS == (
        "LOCAL_CORE_BLOCKED_EXTERNAL_PHASE1_AUTHORITY_AND_CORRECTED_TYPED_D81_P0"
    )
    assert d99.REQUIRED_TYPED_D81_STATE_SCHEMA.endswith("pending")
    assert "corrected_old_only_metric" in d99.REQUIRED_TYPED_D81_STATE_SCHEMA
    assert not any("fuse" in name or "predict" in name for name in d99.__all__)
    for name in d99.__all__:
        function = getattr(d99, name, None)
        if inspect.isfunction(function):
            parameters = set(inspect.signature(function).parameters)
            assert not parameters & {
                "base_logits",
                "base_probabilities",
                "d81_logits",
                "query_labels",
                "query_truth",
                "receiver",
                "scenario",
                "old_classes",
                "new_classes",
            }
