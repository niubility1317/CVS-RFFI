from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import cvsrffi.stage2_d100_ra_cgspr_lgf as d100
import cvsrffi.stage2_d99_ra_cgtmk_d81 as d99


CLASSES = ("old-a", "old-b", "old-c", "new-x")
OLD_CLASSES = CLASSES[:3]


def _aggregation_receipt() -> d99.ExternalGroundAggregationReceipt:
    values = {
        "schema": d99.GROUND_AGGREGATION_RECEIPT_SCHEMA,
        "aggregation_manifest_sha256": "a" * 64,
        "producer_code_sha256": "b" * 64,
        "phase1_checkpoint_sha256": "c" * 64,
        "minimum_physical_sample_count": 2,
        "member_ids_present": False,
        "target_rows_used": 0,
        "cryptographic_external_authority_claimed": False,
    }
    return d99.ExternalGroundAggregationReceipt(
        aggregation_manifest_sha256=values["aggregation_manifest_sha256"],
        producer_code_sha256=values["producer_code_sha256"],
        phase1_checkpoint_sha256=values["phase1_checkpoint_sha256"],
        receipt_sha256=d99._canonical_sha256(values),
    )


def _ground_bundle(
    *,
    old_classes: tuple[str, ...] = OLD_CLASSES,
    domains: int = 7,
    degenerate: bool = False,
) -> d99.Phase1GroundAggregateBundle:
    rng = np.random.default_rng(10001 + domains + len(old_classes))
    base = rng.normal(size=(len(old_classes), d99.Z_DIM))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    latent_rank = min(6, domains - 1)
    directions, _ = np.linalg.qr(rng.normal(size=(d99.Z_DIM, latent_rank)))
    domain_coefficients = rng.normal(size=(domains, latent_rank))
    domain_coefficients -= np.mean(domain_coefficients, axis=0, keepdims=True)
    if degenerate:
        domain_coefficients[:] = 0.0
    grid = []
    for domain in range(domains):
        rows = []
        for class_index in range(len(old_classes)):
            class_scale = 0.75 + 0.5 * (class_index + 1) / len(old_classes)
            shift = 0.06 * class_scale * (
                directions @ domain_coefficients[domain]
            )
            value = base[class_index] + shift
            value /= np.linalg.norm(value)
            rows.append(value)
        grid.append(rows)
    array = np.asarray(grid, dtype=np.float32)
    scales = np.maximum(
        np.max(np.abs(array), axis=2) / 127.0, np.finfo(np.float16).tiny
    ).astype(np.float16)
    codes = np.clip(
        np.rint(array / scales.astype(np.float32)[:, :, None]), -127, 127
    ).astype(np.int8)
    mask = np.ones((domains, len(old_classes)), dtype=bool)
    return d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=np.full(mask.shape, 16, dtype=np.uint16),
        domain_ids=tuple(f"ground-domain-{index}" for index in range(domains)),
        ground_old_registry=old_classes,
        aggregation_receipt=_aggregation_receipt(),
    )


def _d99_lock(
    bundle: d99.Phase1GroundAggregateBundle,
    *,
    old_classes: tuple[str, ...] = OLD_CLASSES,
) -> d99.Phase1D99Lock:
    return d99.Phase1D99Lock(
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
        z_weight=0.7,
        fft_weight=0.2,
        rf_weight=0.1,
        eta_k1=0.1,
        eta_k5=0.2,
        eta_k10=0.3,
        eta_k20=0.35,
        eta_k20_lodo_artifact_sha256=None,
        phase1_receipt_sha256="1" * 64,
        ground_aggregation_receipt_sha256=bundle.aggregation_receipt.receipt_sha256,
        ground_bundle_receipt_sha256=bundle.bundle_sha256,
        quantization_margin_audit_sha256="2" * 64,
        validation_method_lock_sha256="3" * 64,
        d81_phase1_lock_sha256="4" * 64,
        ground_old_registry=old_classes,
    )


def _d100_lock(
    d99_config: d99.Phase1D99Lock,
    *,
    alpha: float = 0.35,
) -> d100.Phase1D100Lock:
    return d100.Phase1D100Lock(
        lambda_k1=0.2,
        lambda_k5=0.15,
        lambda_k10=0.1,
        lambda_k20=0.08,
        temperature_k1=1.0,
        temperature_k5=0.9,
        temperature_k10=0.85,
        temperature_k20=0.8,
        d99_temperature_k1=1.1,
        d99_temperature_k5=1.0,
        d99_temperature_k10=0.95,
        d99_temperature_k20=0.9,
        alpha_k1=alpha,
        alpha_k5=alpha,
        alpha_k10=alpha,
        alpha_k20=alpha,
        d99_phase1_lock_digest=d99_config.lock_digest,
        phase1_lodo_rescue_receipt_sha256="5" * 64,
        external_phase2_authority_sha256="6" * 64,
        quantization_margin_audit_sha256="7" * 64,
    )


def _support(
    classes: tuple[str, ...],
    k_shot: int,
    *,
    seed: int = 10100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k_shot + len(classes))
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical: list[str] = []
    z_centers = rng.normal(size=(len(classes), d99.Z_DIM))
    z_centers /= np.linalg.norm(z_centers, axis=1, keepdims=True)
    for class_index, class_name in enumerate(classes):
        for shot in range(k_shot):
            z = z_centers[class_index] + (
                0.025 * rng.normal(size=d99.Z_DIM) if k_shot > 1 else 0.0
            )
            fft = rng.normal(size=d99.FFT_DIM)
            rf = rng.normal(size=d99.RF_DIM)
            fft[class_index % d99.FFT_DIM] += 3.0
            rf[class_index % d99.RF_DIM] += 2.0
            rows.append(np.concatenate([z, fft, rf]).astype(np.float32))
            labels.append(class_name)
            physical.append(f"physical-{class_name}-{shot}")
    return np.stack(rows), np.asarray(labels), np.asarray(physical)


def _bank(
    k_shot: int = 5,
    *,
    classes: tuple[str, ...] = CLASSES,
    old_classes: tuple[str, ...] = OLD_CLASSES,
    support_order: np.ndarray | None = None,
    degenerate_ground: bool = False,
    domains: int = 7,
    seed: int = 10100,
):
    bundle = _ground_bundle(
        old_classes=old_classes,
        domains=domains,
        degenerate=degenerate_ground,
    )
    config = _d99_lock(bundle, old_classes=old_classes)
    ground = d99.build_ground_geometry(bundle, config=config)
    features, labels, physical = _support(classes, k_shot, seed=seed)
    if support_order is not None:
        features, labels, physical = (
            features[support_order],
            labels[support_order],
            physical[support_order],
        )
    metric = d99.fit_support_metric(
        ground,
        features,
        labels,
        physical,
        classes,
        old_classes,
        config=config,
    )
    bank = d99.build_typed_support_bank(
        metric,
        features,
        labels,
        physical,
        classes,
        config=config,
    )
    return bundle, config, ground, metric, bank, (features, labels, physical)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= np.max(values, axis=1, keepdims=True)
    output = np.exp(values)
    return (output / np.sum(output, axis=1, keepdims=True)).astype(np.float32)


def test_precision_sqrt_matches_d99_generalized_cosine():
    rng = np.random.default_rng(10020)
    basis, _ = np.linalg.qr(rng.normal(size=(d100.Z_DIM, 4)))
    attenuation = np.asarray([0.1, 0.25, 0.4, 0.7], dtype=np.float64)
    left = rng.normal(size=(7, d100.Z_DIM))
    right = rng.normal(size=(5, d100.Z_DIM))
    transformed_left = d100.generalized_precision_sqrt_transform(
        left, basis, attenuation
    )
    transformed_right = d100.generalized_precision_sqrt_transform(
        right, basis, attenuation
    )
    left_n = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_n = right / np.linalg.norm(right, axis=1, keepdims=True)
    precision = np.eye(d100.Z_DIM) - basis @ np.diag(attenuation) @ basis.T
    expected = (left_n @ precision @ right_n.T) / np.sqrt(
        np.sum(left_n * (left_n @ precision), axis=1)[:, None]
        * np.sum(right_n * (right_n @ precision), axis=1)[None, :]
    )
    assert np.allclose(transformed_left @ transformed_right.T, expected, atol=1e-10)


@pytest.mark.parametrize("k_shot", d100.ALLOWED_K)
def test_all_k_values_build_from_exact_typed_d99_bank(k_shot: int):
    _bundle_value, config, _ground, _metric, bank, _support_value = _bank(k_shot)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    assert state.k_shot == k_shot
    assert state.weight_codes_qint8.dtype == np.int8
    assert state.weight_scales_fp16.dtype == np.float16
    assert state.bias_fp16.dtype == np.float16
    assert state.formal_phase2_eligible is False
    assert state.resource_audit["optimizer_steps"] == 0
    assert state.resource_audit["epochs"] == 0


def test_support_order_invariance_and_class_permutation_equivariance():
    base = _bank(5)
    _bundle_value, config, ground, _metric, bank, support = base
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    permutation = np.random.default_rng(10030).permutation(len(support[0]))
    reordered = _bank(5, support_order=permutation)
    state_reordered = d100.build_simplex_ridge_state(
        reordered[4], config=_d100_lock(reordered[1])
    )
    queries = support[0][:7]
    assert np.allclose(
        d100.score_simplex_ridge_logits(state, queries),
        d100.score_simplex_ridge_logits(state_reordered, queries),
        atol=2e-6,
    )

    permuted_classes = ("new-x", "old-c", "old-a", "old-b")
    perm_metric = d99.fit_support_metric(
        ground,
        *support,
        permuted_classes,
        OLD_CLASSES,
        config=config,
    )
    perm_bank = d99.build_typed_support_bank(
        perm_metric,
        *support,
        permuted_classes,
        config=config,
    )
    perm_state = d100.build_simplex_ridge_state(
        perm_bank, config=_d100_lock(config)
    )
    base_logits = d100.score_simplex_ridge_logits(state, queries)
    perm_logits = d100.score_simplex_ridge_logits(perm_state, queries)
    inverse = [permuted_classes.index(name) for name in CLASSES]
    assert np.allclose(base_logits, perm_logits[:, inverse], atol=2e-6)


def test_k1_is_nonidentity_relative_to_d99_on_nondegenerate_queries():
    _bundle_value, config, _ground, _metric, bank, support = _bank(1, seed=10200)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config, alpha=0.5))
    rng = np.random.default_rng(10201)
    queries = rng.normal(size=(1024, d100.FEATURE_DIM)).astype(np.float32)
    d99_prediction = np.argmax(d99.score_metric_kernel_raw_logits(bank, queries), axis=1)
    ridge_prediction = np.argmax(d100.score_simplex_ridge_logits(state, queries), axis=1)
    assert np.any(d99_prediction != ridge_prediction)
    assert support[0].shape[0] == len(CLASSES)


def test_low_ground_coverage_only_removes_ground_metric_not_ridge():
    _bundle_value, config, ground, metric, bank, support = _bank(
        1, degenerate_ground=True
    )
    assert ground.nuisance_basis_fp32.shape[1] == 0
    assert metric.metric_basis_fp32.shape[1] == 0
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    logits = d100.score_simplex_ridge_logits(state, support[0])
    assert np.isfinite(logits).all()
    assert np.any(state.weight_codes_qint8 != 0)
    assert state.resource_audit["metric_rank"] == 0


def test_batch_equals_individual_and_query_state_is_immutable():
    _bundle_value, config, _ground, _metric, bank, support = _bank(5)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    before = d100.serialize_simplex_ridge_state(state)
    queries = support[0][:11]
    batch = d100.score_simplex_ridge_logits(state, queries)
    individual = np.concatenate(
        [d100.score_simplex_ridge_logits(state, row[None, :]) for row in queries],
        axis=0,
    )
    after = d100.serialize_simplex_ridge_state(state)
    assert np.array_equal(batch, individual)
    assert before == after
    assert state.resource_audit["query_state_updates"] == 0
    assert state.resource_audit["query_batch_dependency"] is False


def test_alpha_zero_is_internal_exact_d99_control_and_skips_ridge(monkeypatch):
    _bundle_value, config, _ground, _metric, bank, support = _bank(5)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config, alpha=0.0))
    queries = support[0][:9]
    base = _softmax(
        d99.score_metric_kernel_raw_logits(bank, queries), state.d99_temperature
    )

    def _ridge_must_not_run(*_args, **_kwargs):
        raise AssertionError("alpha=0 must not evaluate D100 ridge")

    monkeypatch.setattr(d100, "score_simplex_ridge_logits", _ridge_must_not_run)
    fused, predicted, audit = d100.fuse_with_typed_d99_bank(state, bank, queries)
    assert np.array_equal(fused, base)
    assert np.array_equal(predicted, np.asarray(state.classes)[np.argmax(base, axis=1)])
    assert audit["ridge_branch_evaluated"] is False
    assert audit["d99_canonical_typed_scorer_used"] is True
    assert not hasattr(d100, "fuse_with_d99_probabilities")


def test_typed_d99_bank_binding_and_tamper_rejection():
    _bundle_value, config, _ground, _metric, bank, support = _bank(5)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    queries = support[0][:4]
    other = _bank(5, seed=10999)[4]
    with pytest.raises(d100.D100RACGSPRError, match="bank/state binding"):
        d100.fuse_with_typed_d99_bank(state, other, queries)
    changed = bank.codes_qint8.copy()
    changed[0, 0] = np.int8(changed[0, 0] + 1 if changed[0, 0] < 127 else 126)
    with pytest.raises(d99.D99RACGTMKError, match="resource/receipt"):
        replace(bank, codes_qint8=changed)


def test_bidirectional_rescue_audit_fixture():
    d99_probability = np.asarray(
        [[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]], dtype=np.float32
    )
    ridge_probability = np.asarray(
        [[0.3, 0.7], [0.7, 0.3], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32
    )
    truth = np.asarray([1, 1, 0, 1], dtype=np.int64)
    audit = d100.complementarity_audit(d99_probability, ridge_probability, truth)
    assert audit["ridge_correct_when_d99_wrong_count"] == 1
    assert audit["d99_correct_when_ridge_wrong_count"] == 1
    assert audit["bidirectional_rescue_nonzero"] is True
    assert audit["formal_phase1_rescue_receipt"] is False


def test_quantization_wire_and_tamper_are_closed():
    _bundle_value, config, _ground, _metric, bank, _support_value = _bank(10)
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    wire = d100.serialize_simplex_ridge_state(state)
    assert len(wire) == state.resource_audit["actual_serialized_state_bytes"]
    assert (
        state.quantization_audit[
            "target_class_prototype_or_weight_fp32_sidecar_present"
        ]
        is False
    )
    assert state.quantization_audit["shared_target_metric_fp32_persisted"] is True
    assert state.quantization_audit["support_top1_agreement"] >= 0.0
    changed = state.weight_codes_qint8.copy()
    changed[0, 0] = np.int8(changed[0, 0] - 1 if changed[0, 0] > -127 else -126)
    with pytest.raises(d100.D100RACGSPRError, match="resource/receipt"):
        replace(state, weight_codes_qint8=changed)
    with pytest.raises(d100.D100RACGSPRError, match="formal prediction is blocked"):
        d100.predict_formal(state, np.zeros((1, d100.FEATURE_DIM), dtype=np.float32))


def test_rank4_and_rank8_query_resource_includes_full_d99_and_d100_paths():
    expected = {
        4: (1_732_160, 18_014, 1_750_174),
        8: (2_397_760, 19_298, 2_417_058),
    }
    for rank, values in expected.items():
        resource = d100._resource_from_dimensions(
            class_count=26,
            k_shot=20,
            rank=rank,
            alpha=0.35,
            numeric_bytes=10_000,
            wire_bytes=13_000,
        )
        assert resource["d99_query_mac_upper_bound_per_sample"] == values[0]
        assert (
            resource["d100_incremental_query_mac_upper_bound_per_sample"]
            == values[1]
        )
        assert resource["combined_query_mac_upper_bound_per_sample"] == values[2]
        assert resource["combined_query_mac_upper_bound_per_sample"] > 1_000_000


def test_c26_k20_known_component_bytes_do_not_claim_complete_upper_bound():
    old = tuple(f"old-{index}" for index in range(6))
    new = tuple(f"new-{index}" for index in range(20))
    classes = old + new
    bundle, config, _ground, _metric, bank, _support_value = _bank(
        20,
        classes=classes,
        old_classes=old,
        domains=14,
        seed=10300,
    )
    state = d100.build_simplex_ridge_state(bank, config=_d100_lock(config))
    d100_wire = d100.serialize_simplex_ridge_state(state)
    actual_d99_wire = d99._serialize_receipt_bearing_bank(bank)
    # Independent K20 resource review fixed the rank-8 D99 maximum-state wire.
    # This supplied byte buffer is counted exactly, but its absent external
    # authority receipt prevents a complete formal combined-state claim.
    d99_rank8_reviewed_wire = actual_d99_wire + bytes(
        163_810 - len(actual_d99_wire)
    )
    ground_wire = b"".join(
        [
            bundle.codes_qint8.tobytes(),
            bundle.scales_fp16.tobytes(),
            bundle.domain_class_mask.tobytes(),
            bundle.physical_sample_count_floor_uint16.tobytes(),
        ]
    )
    typed_d81_exact_head_wire = b"D81-K20-EXACT-HEAD-NONAUTHORITY\0" + bytes(
        35_746 - len(b"D81-K20-EXACT-HEAD-NONAUTHORITY\0")
    )
    audit = d100.audit_combined_wire_budget(
        d100_state_wire=d100_wire,
        d99_bank_wire=d99_rank8_reviewed_wire,
        typed_d81_wire=typed_d81_exact_head_wire,
        ground_bundle_wire=ground_wire,
    )
    assert state.weight_codes_qint8.shape == (26, d100.FEATURE_DIM)
    assert state.resource_audit["trainable_parameter_equivalent"] < 80_000
    assert state.resource_audit["optimizer_steps"] == 0
    assert audit["all_component_sizes_computed_from_bytes"] is True
    assert audit["known_components_below_256kib"] is True
    assert audit["complete_combined_state_upper_bound_available"] is False
    assert audit["under_256kib_formal_claim"] is False
    assert audit["formal_combined_resource_claim"] is False
    assert audit["known_component_wire_bytes"] == sum(
        (
            len(d100_wire),
            163_810,
            35_746,
            len(ground_wire),
        )
    )
