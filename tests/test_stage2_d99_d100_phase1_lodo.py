from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_d99_d100_phase1_lodo as lock
import cvsrffi.stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer


CLASSES = ("class-a", "class-b", "class-c")
RECEIVERS = ("rx-a", "rx-b", "rx-c")
DOMAINS = ("domain-a", "domain-b", "domain-c")


def _archive() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(99100)
    features = []
    labels = []
    receivers = []
    physical = []
    scenarios = []
    days = []
    scenario_names = (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    for receiver_index, receiver in enumerate(RECEIVERS):
        for class_index, class_name in enumerate(CLASSES):
            center = rng.normal(size=288)
            center[class_index] += 5.0
            for sample in range(22):
                features.append((center + 0.1 * rng.normal(size=288)).astype(np.float32))
                labels.append(class_name)
                receivers.append(receiver)
                physical.append(f"source-{receiver}-{class_name}-{sample:03d}")
                scenarios.append(scenario_names[(sample + receiver_index) % 3])
                days.append(f"day-{sample % 2}")
    return {
        "features": np.stack(features).astype(np.float32),
        "labels": np.asarray(labels),
        "receiver_ids": np.asarray(receivers),
        "day_ids": np.asarray(days),
        "physical_ids": np.asarray(physical),
        "scenario_names": np.asarray(scenarios),
        "class_ids": np.asarray(CLASSES),
        "checkpoint_reference_logits": np.zeros((len(features), len(CLASSES)), np.float32),
    }


def _aggregation_receipt() -> d99.ExternalGroundAggregationReceipt:
    payload = {
        "schema": d99.GROUND_AGGREGATION_RECEIPT_SCHEMA,
        "aggregation_manifest_sha256": "a" * 64,
        "producer_code_sha256": "b" * 64,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "minimum_physical_sample_count": 2,
        "member_ids_present": False,
        "target_rows_used": 0,
        "cryptographic_external_authority_claimed": False,
    }
    return d99.ExternalGroundAggregationReceipt(
        aggregation_manifest_sha256=payload["aggregation_manifest_sha256"],
        producer_code_sha256=payload["producer_code_sha256"],
        phase1_checkpoint_sha256=payload["phase1_checkpoint_sha256"],
        receipt_sha256=d99._canonical_sha256(payload),
    )


def _ground_bundle() -> d99.Phase1GroundAggregateBundle:
    rng = np.random.default_rng(99101)
    values = rng.normal(size=(len(DOMAINS), len(CLASSES), d99.Z_DIM)).astype(np.float32)
    values /= np.linalg.norm(values, axis=2, keepdims=True)
    scales = np.maximum(
        np.max(np.abs(values), axis=2) / 127.0, np.finfo(np.float16).tiny
    ).astype(np.float16)
    codes = np.clip(np.rint(values / scales[:, :, None]), -127, 127).astype(np.int8)
    mask = np.ones((len(DOMAINS), len(CLASSES)), dtype=np.bool_)
    return d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=np.full(mask.shape, 16, np.uint16),
        domain_ids=DOMAINS,
        ground_old_registry=CLASSES,
        aggregation_receipt=_aggregation_receipt(),
    )


def _authority(bundle: d99.Phase1GroundAggregateBundle) -> lock.GroundReleaseAuthority:
    payload = lock.ground_release_manifest_payload(
        bundle,
        dict(zip(RECEIVERS, DOMAINS)),
        producer_code_sha256="c" * 64,
    )
    raw = lock._canonical_bytes(payload)
    return lock.load_ground_release_authority(raw, hashlib.sha256(raw).hexdigest(), bundle)


def _development_authority(
    bundle: d99.Phase1GroundAggregateBundle,
) -> lock.GroundReleaseAuthority:
    payload = lock.ground_release_manifest_payload(
        bundle,
        dict(zip(RECEIVERS, DOMAINS)),
        producer_code_sha256="c" * 64,
        release_schema=lock.GROUND_RELEASE_DEVELOPMENT_SCHEMA,
        release_status=lock.GROUND_RELEASE_DEVELOPMENT_STATUS,
    )
    raw = lock._canonical_bytes(payload)
    return lock.load_ground_release_authority(
        raw, hashlib.sha256(raw).hexdigest(), bundle
    )


def _base_d99(bundle: d99.Phase1GroundAggregateBundle) -> d99.Phase1D99Lock:
    return d99.Phase1D99Lock(
        density_tau=0.2,
        max_ground_rank=2,
        max_target_rank=2,
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
        eta_k20=0.4,
        eta_k20_lodo_artifact_sha256=None,
        phase1_receipt_sha256=BASE_CHECKPOINT_SHA256,
        ground_aggregation_receipt_sha256=bundle.aggregation_receipt.receipt_sha256,
        ground_bundle_receipt_sha256=bundle.bundle_sha256,
        quantization_margin_audit_sha256="d" * 64,
        validation_method_lock_sha256="e" * 64,
        d81_phase1_lock_sha256="f" * 64,
        ground_old_registry=CLASSES,
    )


def _scorer() -> D81Phase1EpisodeScorer:
    return D81Phase1EpisodeScorer(
        nuisance_basis_fp64=np.eye(160, 2),
        spectral_weights_fp64=np.asarray([0.5, 0.5]),
        ground_manifest_sha256="1" * 64,
        ground_component_npz_sha256="2" * 64,
        ground_audit={"ground_component_input_count": 9},
    )


def _grid() -> dict[str, list[float]]:
    return {
        "eta": [0.25],
        "student_nu": [3.0],
        "kernel_volume_gamma": [1.0],
        "shared_h0": [0.5],
        "scale_prior_strength": [2.0],
        "scale_min_ratio": [0.5],
        "scale_max_ratio": [2.0],
        "d99_temperature": [1.0],
        "lambda0": [0.1],
        "ridge_temperature": [1.0],
        "alpha": [0.35],
    }


def _admission_row(*, changed_count: int, d99_nll: float = 0.4) -> dict:
    candidate = {key: values[0] for key, values in _grid().items()}
    common = {
        "balanced_accuracy": 0.8,
        "worst_class_floor": 0.7,
        "pseudo_old_accuracy": 0.8,
        "pseudo_new_accuracy": 0.75,
        "harmonic_old_new": 0.774,
        "brier": 0.2,
    }
    return {
        "receiver": RECEIVERS[0],
        "k_shot": 1,
        "fold_id": "pseudo-new-0",
        "candidate": candidate,
        "d81": {**common, "balanced_nll": 0.5},
        "kernel": {**common, "balanced_nll": 0.45},
        "d99": {**common, "balanced_nll": d99_nll},
        "ridge": {**common, "balanced_nll": 0.35},
        "fused": {**common, "balanced_nll": 0.3},
        "complementarity": {
            "row_count": 8,
            "disagreement_count": 2,
            "ridge_correct_when_d99_wrong_count": 1,
            "d99_correct_when_ridge_wrong_count": 1,
            "oracle_union_accuracy": 0.9,
        },
        "d81_kernel_complementarity": {
            "row_count": 8,
            "disagreement_count": 2,
            "kernel_correct_when_d81_wrong_count": 1,
            "d81_correct_when_kernel_wrong_count": 1,
            "oracle_union_accuracy": 0.9,
        },
        "d99_vs_d81_changed_count": changed_count,
        "ground_coverage_rho": 0.2,
        "d99_bank_wire_bytes": 100,
        "d100_state_wire_bytes": 50,
    }


def test_k20_is_real_nested_support_and_all_classes_rotate_pseudo_new() -> None:
    validated = lock.validate_feature_archive(_archive())
    episodes = lock.build_receiver_lodo_episodes(validated, seed=99)
    physical = validated["arrays"]["physical_ids"]
    for receiver in RECEIVERS:
        support_sets = {
            k: set(physical[episodes[receiver][k].support].tolist())
            for k in lock.ALLOWED_K
        }
        assert support_sets[1] < support_sets[5] < support_sets[10] < support_sets[20]
        assert len(support_sets[20]) == 2 * len(support_sets[10])
        assert not support_sets[20] & set(
            physical[episodes[receiver][20].evaluation].tolist()
        )
    folds = lock.build_pseudo_new_folds(CLASSES)
    assert {fold["pseudo_new"][0] for fold in folds} == set(CLASSES)
    assert all(len(fold["pseudo_old"]) == 2 for fold in folds)


def test_caller_created_ground_manifest_never_self_grants_authority() -> None:
    bundle = _ground_bundle()
    authority = _authority(bundle)
    assert lock.TRUSTED_GROUND_RELEASE_MANIFEST_SHA256 is None
    assert authority.formal_phase1_eligible is False
    assert authority.authority_status == "BLOCKED_UNPROVISIONED_GROUND_ROOT"
    with pytest.raises(lock.D99D100LODOLockError, match="self-grant"):
        lock.GroundReleaseAuthority(
            manifest_sha256=authority.manifest_sha256,
            bundle_sha256=authority.bundle_sha256,
            aggregation_receipt_sha256=authority.aggregation_receipt_sha256,
            phase1_checkpoint_sha256=authority.phase1_checkpoint_sha256,
            receiver_domain_map=dict(authority.receiver_domain_map),
            formal_phase1_eligible=True,
            authority_status="PROVISIONED",
            manifest_bytes=authority.manifest_bytes,
            loader_token=authority.loader_token,
        )


def test_development_ground_release_cannot_become_formal_even_if_sha_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _ground_bundle()
    payload = lock.ground_release_manifest_payload(
        bundle,
        dict(zip(RECEIVERS, DOMAINS)),
        producer_code_sha256="c" * 64,
        release_schema=lock.GROUND_RELEASE_DEVELOPMENT_SCHEMA,
        release_status=lock.GROUND_RELEASE_DEVELOPMENT_STATUS,
    )
    raw = lock._canonical_bytes(payload)
    sha = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(lock, "TRUSTED_GROUND_RELEASE_MANIFEST_SHA256", sha)
    authority = lock.load_ground_release_authority(raw, sha, bundle)
    assert authority.formal_phase1_eligible is False
    assert authority.authority_status == "BLOCKED_DEVELOPMENT_GROUND_RELEASE"


def test_alpha_guard_forces_zero_on_one_sided_rescue_or_floor_drop() -> None:
    candidate = {**{key: values[0] for key, values in _grid().items()}}
    summary = {
        "bidirectional_rescue_nonzero": False,
        "d99": {"balanced_nll": 0.5},
        "fused": {"balanced_nll": 0.49},
        "paired_receiver_pseudo_new_guard": {
            "all_pairs_non_decreasing": True,
            "degraded_pair_count": 0,
        },
    }
    guarded = lock.enforce_alpha_guard(candidate, summary)
    assert guarded["alpha_forced_zero"] is True
    assert guarded["effective_parameters"]["alpha"] == 0.0
    summary["bidirectional_rescue_nonzero"] = True
    summary["paired_receiver_pseudo_new_guard"] = {
        "all_pairs_non_decreasing": False,
        "degraded_pair_count": 1,
    }
    assert lock.enforce_alpha_guard(candidate, summary)["effective_parameters"]["alpha"] == 0.0
    summary["paired_receiver_pseudo_new_guard"] = {
        "all_pairs_non_decreasing": True,
        "degraded_pair_count": 0,
    }
    assert lock.enforce_alpha_guard(candidate, summary)["effective_parameters"]["alpha"] == 0.35


def test_alpha_guard_forces_zero_when_fused_nll_does_not_strictly_improve() -> None:
    candidate = {key: values[0] for key, values in _grid().items()}
    summary = {
        "bidirectional_rescue_nonzero": True,
        "paired_receiver_pseudo_new_guard": {
            "all_pairs_non_decreasing": True,
            "degraded_pair_count": 0,
        },
        "d99": {"balanced_nll": 0.5},
        "fused": {"balanced_nll": 0.5},
    }
    guarded = lock.enforce_alpha_guard(candidate, summary)
    assert guarded["alpha_forced_zero"] is True
    assert guarded["d100_eligible"] is False
    assert guarded["guard"]["balanced_nll_strictly_improved_vs_d99"] is False


def test_k1_all_identity_is_explicitly_blocked_but_nonidentity_passes() -> None:
    identity_ranking = lock._rank_candidates([[_admission_row(changed_count=0)]])
    assert identity_ranking[0]["d99_eligible"] is False
    assert lock._d99_block_reason(1, identity_ranking) == "K1_NO_NONIDENTITY_CANDIDATE"
    assert identity_ranking[0]["d99_guard"]["k1_nonidentity_prediction_passed"] is False

    nonidentity_ranking = lock._rank_candidates([[_admission_row(changed_count=1)]])
    assert nonidentity_ranking[0]["d99_eligible"] is True
    assert nonidentity_ranking[0]["d100_eligible"] is True
    summary = nonidentity_ranking[0]["raw_summary"]
    assert summary["d99_vs_d81_changed_count"] == 1
    assert summary["d81_kernel_rescue_distribution"]["pair_count"] == 1


def test_d99_guard_rejects_non_strict_nll_improvement() -> None:
    ranking = lock._rank_candidates(
        [[_admission_row(changed_count=1, d99_nll=0.5)]]
    )
    assert ranking[0]["d99_eligible"] is False
    assert (
        ranking[0]["d99_guard"]["balanced_nll_strictly_improved_vs_d81"]
        is False
    )


@pytest.mark.parametrize(
    "metric",
    ("worst_class_floor", "pseudo_old_accuracy", "pseudo_new_accuracy", "harmonic_old_new"),
)
def test_alpha_guard_is_paired_per_receiver_fold_for_all_four_metrics(metric: str) -> None:
    candidate = {key: values[0] for key, values in _grid().items()}
    base = {
        "balanced_accuracy": 0.8,
        "worst_class_floor": 0.7,
        "pseudo_old_accuracy": 0.8,
        "pseudo_new_accuracy": 0.75,
        "harmonic_old_new": 0.774,
        "balanced_nll": 0.5,
        "brier": 0.2,
    }
    rows = []
    for index, receiver in enumerate(RECEIVERS[:2]):
        fused = dict(base)
        if index == 1:
            fused[metric] -= 0.01
        rows.append(
            {
                "receiver": receiver,
                "fold_id": f"fold-{index}",
                "candidate": candidate,
                "k_shot": 5,
                "d81": {**base, "balanced_nll": 0.55},
                "kernel": dict(base),
                "d99": dict(base),
                "ridge": dict(base),
                "fused": fused,
                "complementarity": {
                    "row_count": 8,
                    "disagreement_count": 2 + index,
                    "ridge_correct_when_d99_wrong_count": 1,
                    "d99_correct_when_ridge_wrong_count": 1,
                    "oracle_union_accuracy": 0.9,
                },
                "d81_kernel_complementarity": {
                    "row_count": 8,
                    "disagreement_count": 2,
                    "kernel_correct_when_d81_wrong_count": 1,
                    "d81_correct_when_kernel_wrong_count": 1,
                    "oracle_union_accuracy": 0.9,
                },
                "d99_vs_d81_changed_count": 1,
                "ground_coverage_rho": 0.2,
                "d99_bank_wire_bytes": 100,
                "d100_state_wire_bytes": 50,
            }
        )
    summary = lock._aggregate(rows)
    guarded = lock.enforce_alpha_guard(candidate, summary)
    assert guarded["alpha_forced_zero"] is True
    assert guarded["guard"]["degraded_pair_count"] == 1
    distribution = summary["bidirectional_rescue_distribution"]
    assert distribution["pair_count"] == 2
    assert len(distribution["rows"]) == 2


@pytest.mark.parametrize("k1_identity", (False, True))
def test_end_to_end_development_receipt_is_blocked_not_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, k1_identity: bool
) -> None:
    arrays = _archive()
    archive_path = tmp_path / "phase1.npz"
    np.savez(archive_path, **arrays)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        lock,
        "_validate_feature_archive_manifest",
        lambda *args, **kwargs: {
            "full_phase1_lock": True,
            "development_lock_frozen": False,
            "manifest_sha256": "3" * 64,
        },
    )

    def fast_scorer(self, support, labels, query, classes):
        del self
        prototypes = np.stack(
            [np.mean(support[np.asarray(labels).astype(str) == name], axis=0) for name in classes]
        )
        return np.asarray(query, np.float64) @ np.asarray(prototypes, np.float64).T

    monkeypatch.setattr(D81Phase1EpisodeScorer, "__call__", fast_scorer)

    def synthetic_evaluation(**kwargs):
        episode = kwargs["episode"]
        fold = kwargs["fold"]
        candidate = dict(kwargs["candidate"])
        base = {
            "row_count": int(len(kwargs["query_indices"])),
            "balanced_accuracy": 0.75,
            "worst_class_floor": 0.5,
            "pseudo_old_accuracy": 0.75,
            "pseudo_new_accuracy": 0.75,
            "harmonic_old_new": 0.75,
            "balanced_nll": 0.7,
            "brier": 0.3,
            "per_class_accuracy": {name: 0.75 for name in CLASSES},
        }
        ridge = {**base, "balanced_nll": 0.69}
        fused = {**base, "balanced_nll": 0.68}
        d81 = {**base, "balanced_nll": 0.72}
        return {
            "receiver": episode.receiver,
            "k_shot": episode.k_shot,
            "fold_id": fold["fold_id"],
            "pseudo_old": list(fold["pseudo_old"]),
            "pseudo_new": list(fold["pseudo_new"]),
            "candidate": candidate,
            "d81_prediction": [CLASSES[0]] * int(len(kwargs["query_indices"])),
            "d81": d81,
            "kernel": dict(base),
            "d99": base,
            "ridge": ridge,
            "fused": fused,
            "complementarity": {
                "row_count": int(len(kwargs["query_indices"])),
                "disagreement_count": 2,
                "ridge_correct_when_d99_wrong_count": 1,
                "d99_correct_when_ridge_wrong_count": 1,
                "oracle_union_accuracy": 0.8,
            },
            "d81_kernel_complementarity": {
                "row_count": int(len(kwargs["query_indices"])),
                "disagreement_count": 2,
                "kernel_correct_when_d81_wrong_count": 1,
                "d81_correct_when_kernel_wrong_count": 1,
                "oracle_union_accuracy": 0.8,
            },
            "d99_vs_d81_changed_count": (
                0 if k1_identity and episode.k_shot == 1 else 1
            ),
            "ground_coverage_rho": 0.2,
            "ground_weight": 0.1,
            "target_weight": 0.1,
            "metric_rank": 2,
            "d99_bank_wire_bytes": 1000,
            "d100_state_wire_bytes": 500,
            "d99_d100_optimizer_steps": 0,
            "d99_d100_epochs": 0,
            "resource": {
                "d99_d100_known_persistent_wire_bytes": 1500,
                "d99_d100_trainable_parameter_equivalent": 867,
                "d99_d100_query_mac_upper_bound_per_sample": 4000,
                "d99_fit_peak_transient_bytes_upper_bound": 8000,
            },
        }

    monkeypatch.setattr(lock, "_evaluate_candidate", synthetic_evaluation)
    bundle = _ground_bundle()
    scorer = _scorer()
    receipt = lock.run_phase1_d99_d100_lodo(
        archive_path,
        manifest_path,
        "4" * 64,
        ground_bundle=bundle,
        ground_authority=_development_authority(bundle),
        base_d99_config=_base_d99(bundle),
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        grid=_grid(),
        code_sha256=lock.current_code_sha256(),
        seed=99,
    )
    assert receipt["status"] == lock.STATUS_DIAGNOSTIC
    assert receipt["formal_authority_status"] == lock.STATUS_BLOCKED
    assert receipt["canonical_lock_artifact_write_allowed"] is False
    expected_blocked = ["independent_ground_authority_root"]
    if k1_identity:
        expected_blocked.append("d99_admission_1:K1_NO_NONIDENTITY_CANDIDATE")
    assert receipt["blocked_inputs"] == expected_blocked
    assert receipt["resource_audit"]["d81_episode_fit_count"] == 12
    assert receipt["resource_audit"]["d81_optimizer_steps_total"] == 240
    assert receipt["resource_audit"]["resource_claim_status"] == (
        "NONFORMAL_PARTIAL_KNOWN_COMPONENTS_ONLY"
    )
    assert receipt["resource_audit"]["formal_under_256kib_claim"] is False
    assert receipt["ground"]["release_schema"] == lock.GROUND_RELEASE_DEVELOPMENT_SCHEMA
    assert receipt["ground"]["release_status"] == lock.GROUND_RELEASE_DEVELOPMENT_STATUS
    assert receipt["selection_scope"] == (
        "phase1_only_pseudo_target_receiver_lodo_with_fixed_phase1_"
        "encoder_and_fixed_global_d81_ground_basis"
    )
    assert receipt["protocol_audit"][
        "d81_fixed_global_ground_basis_may_include_held_receiver_domains"
    ] is True
    assert receipt["protocol_audit"]["whole_method_held_receiver_ground_unused_claim"] is False
    assert "held_receiver_ground_domain_used" not in receipt["protocol_audit"]
    assert receipt["selected_by_k"]["20"]["selected"]["effective_parameters"]["alpha"] == 0.35
    assert receipt["D99_eligible_by_k"]["1"] is (not k1_identity)
    assert receipt["D100_eligible_by_k"]["1"] is (not k1_identity)
    assert ("1" in receipt["locked_parameters_by_k"]) is (not k1_identity)
    assert all(receipt["D99_eligible_by_k"][str(k)] for k in (5, 10, 20))
    assert all(receipt["D100_eligible_by_k"][str(k)] for k in (5, 10, 20))
    assert all(row["k20_is_distinct_real_episode"] for row in receipt["split_receipt"])
    assert lock.verify_receipt(receipt)


def test_grid_is_exact_and_k_specific() -> None:
    assert len(lock.candidate_grid(_grid())) == 1
    bad = _grid()
    bad["extra"] = [1.0]
    with pytest.raises(lock.D99D100LODOLockError, match="exact fields"):
        lock.candidate_grid(bad)
    too_wide = _grid()
    too_wide["eta"] = [index / 10.0 for index in range(7)]
    with pytest.raises(lock.D99D100LODOLockError, match="exceeds 6"):
        lock.candidate_grid(too_wide)
    too_many = {key: [values[0], values[0] + 0.01] for key, values in _grid().items()}
    too_many["eta"] = [0.2, 0.4]
    too_many["alpha"] = [0.2, 0.4]
    with pytest.raises(lock.D99D100LODOLockError, match="Cartesian"):
        lock.candidate_grid(too_many)


def test_episode_rng_is_invariant_to_class_renaming() -> None:
    original = _archive()
    renamed = {key: value.copy() for key, value in original.items()}
    mapping = {name: f"renamed-{index}" for index, name in enumerate(CLASSES)}
    renamed["labels"] = np.asarray([mapping[str(value)] for value in original["labels"]])
    renamed["class_ids"] = np.asarray([mapping[name] for name in reversed(CLASSES)])
    first_validated = lock.validate_feature_archive(original)
    second_validated = lock.validate_feature_archive(renamed)
    first = lock.build_receiver_lodo_episodes(first_validated, seed=412)
    second = lock.build_receiver_lodo_episodes(second_validated, seed=412)
    physical = original["physical_ids"]
    for receiver in RECEIVERS:
        for k_shot in lock.ALLOWED_K:
            for field in ("support", "calibration", "evaluation"):
                first_ids = set(physical[getattr(first[receiver][k_shot], field)].tolist())
                second_ids = set(physical[getattr(second[receiver][k_shot], field)].tolist())
                assert first_ids == second_ids


def test_real_k1_episode_executes_d99_bank_and_d100_ridge() -> None:
    arrays = _archive()
    validated = lock.validate_feature_archive(arrays)
    episode = lock.build_receiver_lodo_episodes(validated, seed=991)[RECEIVERS[0]][1]
    fold = lock.build_pseudo_new_folds(CLASSES)[0]
    bundle = _ground_bundle()
    query_indices = episode.calibration
    row = lock._evaluate_candidate(
        arrays=validated["arrays"],
        episode=episode,
        query_indices=query_indices,
        fold=fold,
        candidate={key: value[0] for key, value in _grid().items()},
        base_d99_config=_base_d99(bundle),
        full_ground_bundle=bundle,
        authority=_authority(bundle),
        d81_logits=np.zeros((len(query_indices), len(CLASSES)), dtype=np.float32),
        d81_source_schema="cvs.phase1.d81.episode_scorer.v1",
        d81_source_receipt_sha256="7" * 64,
        prepared_cache={},
    )
    assert row["k_shot"] == 1
    assert row["pseudo_new"] == [CLASSES[0]]
    assert row["d99_bank_wire_bytes"] > 0
    assert row["d100_state_wire_bytes"] > 0
    assert 0.0 <= row["d99"]["balanced_accuracy"] <= 1.0
    assert 0.0 <= row["d81"]["balanced_accuracy"] <= 1.0
    assert 0.0 <= row["kernel"]["balanced_accuracy"] <= 1.0
    assert len(row["d81_prediction"]) == len(query_indices)
    assert row["d99_vs_d81_changed_count"] >= 0
    assert row["d81_kernel_complementarity"]["row_count"] == len(query_indices)
    assert 0.0 <= row["fused"]["worst_class_floor"] <= 1.0
    assert row["d99_d100_optimizer_steps"] == 0
    assert row["canonical_fusion_audit"]["eta_phase1_locked"] == 0.25
    assert row["resource"]["complete_combined_query_mac_available"] is False
