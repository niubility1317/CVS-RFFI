from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
from cvsrffi.stage2_d96_d97_phase1_lodo import (
    ALLOWED_K,
    EXPORTER_DEVELOPMENT_STATUS,
    EXPORTER_EXACT_MEMBERS,
    EXPORTER_FORMAL_STATUS,
    _build_base_logits_cache,
    _exporter_array_sha256,
    _quantize_support,
    _resolve_support_only_eta,
    build_receiver_lodo_episodes,
    canonical_sha256,
    normalize_three_blocks,
    qknn_logits,
    run_phase1_lodo_selection,
    validate_feature_archive,
    verify_receipt,
)
from cvsrffi.stage2_qk_d81_lgf import (
    _quantize_rows as deployed_quantize_rows,
    normalize_three_blocks as deployed_normalize_three_blocks,
)


DEV_RUNTIME_SHA = (
    "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(713097)
    receivers = ("rx0", "rx1", "rx2")
    classes = np.asarray((10, 20, 30), dtype=np.int64)
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    rows, labels, receiver_ids, day_ids, physical_ids, scenario_names = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for receiver_index, receiver in enumerate(receivers):
        for class_index, label in enumerate(classes.tolist()):
            center = np.zeros(288, dtype=np.float32)
            center[class_index] = 2.5
            center[160 + class_index] = 1.5
            center[256 + class_index] = 1.0
            center[16 + receiver_index] = 0.25
            for sample_index in range(16):
                feature = center + rng.normal(0.0, 0.035, size=288).astype(np.float32)
                rows.append(feature)
                labels.append(label)
                receiver_ids.append(receiver)
                day_ids.append(f"day{sample_index % 2}")
                physical_ids.append(f"{receiver}:{label}:{sample_index}")
                scenario_names.append(scenarios[sample_index % len(scenarios)])
    return {
        "features": np.stack(rows).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "receiver_ids": np.asarray(receiver_ids),
        "day_ids": np.asarray(day_ids),
        "physical_ids": np.asarray(physical_ids),
        "scenario_names": np.asarray(scenario_names),
        "class_ids": classes,
        "checkpoint_reference_logits": np.zeros(
            (len(labels), len(classes)), dtype=np.float32
        ),
    }


def _write_exporter_v2(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    status: str = EXPORTER_DEVELOPMENT_STATUS,
) -> tuple[Path, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_path / "phase1_singleobs_feature_archive.npz"
    np.savez_compressed(archive_path, **arrays)
    is_formal = status == EXPORTER_FORMAL_STATUS
    dependencies = {
        "leo_weak_cache": "1" * 64,
        "feature_descriptors": "2" * 64,
        "formal_bundle_verifier": "3" * 64,
    }
    outer = "4" * 64 if is_formal else None
    inputs = {
        "cache_set_sha256": "5" * 64,
        "cache_npz_sha256_by_scenario": {
            "leo_clear_weak": "6" * 64,
            "leo_low_elev_weak": "7" * 64,
            "leo_rain_weak": "8" * 64,
        },
        "runtime_authority_mode": (
            "formal_adv3b02_outer_bundle"
            if is_formal
            else "development_known_adv3b02_runtime_sha"
        ),
        "runtime_authority_binding_sha256": "9" * 64,
        "runtime_checkpoint_parity_receipt_sha256": "a" * 64,
        "runtime_schema": "adv3b02.torchscript_identity_runtime.v1",
        "runtime_sha256": DEV_RUNTIME_SHA,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "bundle_id": outer if is_formal else "b" * 64,
        "formal_outer_content_root_sha256": outer,
        "detached_seal_sha256": "c" * 64 if is_formal else None,
        "signature_envelope_sha256": "d" * 64 if is_formal else None,
        "selection_salt_receipt_sha256": "e" * 64,
        "selection_salt_receipt_schema": (
            "cvs.phase1.singleobs_selection_salt_receipt.v1"
        ),
        "exporter_code_sha256": "f" * 64,
        "dependency_code_sha256": dependencies,
        "dependency_closure_sha256": canonical_sha256(dependencies),
    }
    manifest = {
        "schema": "cvs.phase1.single_leo_feature_archive.v2",
        "status": status,
        "artifact_stage": "phase1_offline_before_target_access",
        "artifact": {"path": archive_path.name, "sha256": _sha(archive_path)},
        "exact_member_allowlist": list(EXPORTER_EXACT_MEMBERS),
        "feature_dims": {"z160": 160, "fft96": 96, "rf32": 32, "features": 288},
        "inputs": inputs,
        "selection": {
            "selection_salt_sha256": "0" * 64,
            "scenario_order": [
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            ],
            "formula": "locked",
            "selected_observations_per_physical_id": 1,
            "unselected_observations_forwarded": 0,
        },
        "feature_semantics": {
            "features": (
                "float32_concat(runtime_z160,internally_normalized_fft96,"
                "internally_normalized_rf32)_without_cross_block_weight_or_joint_normalization"
            ),
            "deployment_normalization": "shared_D97_normalize_three_blocks",
            "checkpoint_reference_logits": (
                "sealed_ADV3B02_checkpoint_reference_only_not_D81"
            ),
        },
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "row_count": len(next(iter(arrays.values()))),
        "physical_id_unique_count": len(next(iter(arrays.values()))),
        "one_output_row_per_physical_id": True,
        "array_sha256": {
            name: _exporter_array_sha256(value) for name, value in arrays.items()
        },
        "cache_loader_audit_sha256": "1" * 64,
        "access_audit": {
            "clean_calls": 0,
            "target_calls": 0,
            "channel_calls": 0,
            "clean_iq_access": False,
            "target_access": False,
            "query_access": False,
            "raw_iq_persisted": False,
            "received_iq_persisted": False,
            "unselected_iq_persisted": False,
        },
        "lifecycle": {
            "phase1_temporary_selection_asset": True,
            "phase2_bundle_ingest_allowed": False,
            "phase2_runtime_access_allowed": False,
            "retention": "archive_or_delete_after_D97_lock_receipt",
        },
        "formal_archive": is_formal,
        "development_archive": not is_formal,
    }
    manifest_path = tmp_path / "phase1_singleobs_feature_archive.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return archive_path, manifest_path, _sha(manifest_path)


def _support_mean_scorer(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    support = support_features.astype(np.float64)
    support /= np.maximum(np.linalg.norm(support, axis=1, keepdims=True), 1e-12)
    query = query_features.astype(np.float64)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    prototypes = []
    for class_id in class_ids.tolist():
        prototype = np.mean(support[support_labels == class_id], axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
    return 12.0 * (query @ np.stack(prototypes).T)


def _scorer() -> D81Phase1EpisodeScorer:
    return D81Phase1EpisodeScorer(
        nuisance_basis_fp64=np.eye(160, 3),
        spectral_weights_fp64=np.asarray([0.5, 0.3, 0.2]),
        ground_manifest_sha256="1" * 64,
        ground_component_npz_sha256="2" * 64,
        ground_audit={"ground_component_input_count": 84},
    )


def _grid() -> dict[str, list[float]]:
    return {
        "beta": [4.0, 8.0],
        "temp_base": [0.8, 1.0],
        "temp_qk": [0.8],
        "eta_max": [0.25, 0.5],
        "k1_eta_prior": [0.1, 0.25],
    }


def _patch_scorer(
    monkeypatch: pytest.MonkeyPatch,
    implementation=_support_mean_scorer,
) -> None:
    def bound_implementation(
        self: D81Phase1EpisodeScorer,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        query_features: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        del self
        return implementation(
            support_features,
            support_labels,
            query_features,
            class_ids,
        )

    monkeypatch.setattr(
        D81Phase1EpisodeScorer,
        "__call__",
        bound_implementation,
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    grid: dict[str, list[float]] | None = None,
    status: str = EXPORTER_DEVELOPMENT_STATUS,
) -> dict:
    _patch_scorer(monkeypatch)
    archive, manifest, manifest_sha = _write_exporter_v2(
        tmp_path, _archive(), status=status
    )
    scorer = _scorer()
    return run_phase1_lodo_selection(
        archive,
        grid or _grid(),
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        feature_archive_manifest_path=manifest,
        feature_archive_manifest_sha256=manifest_sha,
        base_scorer_receipt_sha256=scorer.scorer_id,
        seed=97,
    )


def test_development_lock_is_deterministic_but_never_formal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(tmp_path / "a", monkeypatch)
    second = _run(tmp_path / "b", monkeypatch)
    assert first == second
    assert first["full_phase1_lock"] is False
    assert first["development_lock_frozen"] is True
    assert first["target_narrow_diagnostic_preregistration_allowed"] is True
    assert first["formal_target_claim_allowed"] is False
    assert verify_receipt(first)
    assert first["candidate_count"] == 16
    assert first["protocol_audit"]["eta_uses_calibration_or_evaluation_labels"] is False
    assert first["int8_margin_audit"]["formal_selection_uses_quantized_support"] is True


def test_exact_formal_exporter_manifest_can_create_full_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = {key: [values[0]] for key, values in _grid().items()}
    receipt = _run(
        tmp_path, monkeypatch, grid=one, status=EXPORTER_FORMAL_STATUS
    )
    assert receipt["full_phase1_lock"] is True
    assert receipt["development_lock_frozen"] is False
    assert receipt["formal_target_claim_allowed"] is True


def test_diagnostic_generic_inmemory_and_alias_bypasses_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_scorer(monkeypatch)
    scorer = _scorer()
    with pytest.raises(ValueError, match="archive path"):
        run_phase1_lodo_selection(
            _archive(),
            _grid(),
            base_scorer=scorer,
            base_scorer_id=scorer.scorer_id,
            feature_archive_manifest_path=tmp_path / "none",
            feature_archive_manifest_sha256="0" * 64,
            base_scorer_receipt_sha256=scorer.scorer_id,
        )

    archive, manifest, _ = _write_exporter_v2(tmp_path / "diag", _archive())
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "cvs.test_diagnostic.single_leo_feature_archive.v1"
    payload["status"] = "TEST_DIAGNOSTIC_NOT_FORMAL"
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="exact exporter v2"):
        run_phase1_lodo_selection(
            archive,
            _grid(),
            base_scorer=scorer,
            base_scorer_id=scorer.scorer_id,
            feature_archive_manifest_path=manifest,
            feature_archive_manifest_sha256=_sha(manifest),
            base_scorer_receipt_sha256=scorer.scorer_id,
        )

    aliased = _archive()
    aliased["registered_joint288"] = aliased.pop("features")
    archive, manifest, manifest_sha = _write_exporter_v2(
        tmp_path / "alias", aliased
    )
    with pytest.raises(ValueError, match="exact exporter v2 members"):
        run_phase1_lodo_selection(
            archive,
            _grid(),
            base_scorer=scorer,
            base_scorer_id=scorer.scorer_id,
            feature_archive_manifest_path=manifest,
            feature_archive_manifest_sha256=manifest_sha,
            base_scorer_receipt_sha256=scorer.scorer_id,
        )


def test_duck_typed_fake_d81_is_rejected(tmp_path: Path) -> None:
    class Fake:
        receipt = {
            "schema": "cvs.phase1.d81.episode_scorer.v1",
            "formula": "D81_before_support_fitted_D62_D42_int8",
        }
        scorer_id = "1" * 64

        def __call__(self, *args):
            return _support_mean_scorer(*args)

    archive, manifest, manifest_sha = _write_exporter_v2(tmp_path, _archive())
    with pytest.raises(ValueError, match="formal D81 scorer object"):
        run_phase1_lodo_selection(
            archive,
            _grid(),
            base_scorer=Fake(),
            base_scorer_id="1" * 64,
            feature_archive_manifest_path=manifest,
            feature_archive_manifest_sha256=manifest_sha,
            base_scorer_receipt_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("lifecycle", "phase2_runtime_access_allowed", True, "lifecycle"),
        ("feature_semantics", "deployment_normalization", "wrong", "lifecycle"),
        ("inputs", "phase1_checkpoint_sha256", "0" * 64, "runtime lineage"),
    ],
)
def test_manifest_lifecycle_semantics_and_lineage_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value,
    message: str,
) -> None:
    _patch_scorer(monkeypatch)
    scorer = _scorer()
    archive, manifest, _ = _write_exporter_v2(tmp_path, _archive())
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[section][key] = value
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        run_phase1_lodo_selection(
            archive,
            _grid(),
            base_scorer=scorer,
            base_scorer_id=scorer.scorer_id,
            feature_archive_manifest_path=manifest,
            feature_archive_manifest_sha256=_sha(manifest),
            base_scorer_receipt_sha256=scorer.scorer_id,
        )


def test_episode_splits_are_nested_and_physically_disjoint() -> None:
    validated = validate_feature_archive(_archive())
    episodes = build_receiver_lodo_episodes(validated, seed=4)
    physical = validated["arrays"]["physical_ids"]
    for by_k in episodes.values():
        support_sets = {k: set(physical[by_k[k].support]) for k in ALLOWED_K}
        assert support_sets[1] < support_sets[5] < support_sets[10]
        for episode in by_k.values():
            support = set(physical[episode.support])
            calibration = set(physical[episode.calibration])
            evaluation = set(physical[episode.evaluation])
            assert not support & calibration
            assert not support & evaluation
            assert not calibration & evaluation


def test_archive_protocol_fields_and_physical_ids_fail_closed() -> None:
    for forbidden in ("target_labels", "clean_features", "multiView_features"):
        archive = _archive()
        archive[forbidden] = np.zeros(len(archive["labels"]), dtype=np.int8)
        with pytest.raises(ValueError, match="forbidden archive field"):
            validate_feature_archive(archive)
    duplicate = _archive()
    duplicate["physical_ids"] = duplicate["physical_ids"].copy()
    duplicate["physical_ids"][1] = duplicate["physical_ids"][0]
    with pytest.raises(ValueError, match="exactly once"):
        validate_feature_archive(duplicate)


def test_three_block_quantization_and_logmeanexp_match_deployment() -> None:
    archive = _archive()
    support_indices = np.r_[0:4, 16:20, 32:36]
    query_indices = np.r_[4:8, 20:24, 36:40]
    support = archive["features"][support_indices]
    labels = archive["labels"][support_indices]
    query = archive["features"][query_indices]
    np.testing.assert_allclose(
        normalize_three_blocks(support),
        deployed_normalize_three_blocks(support),
        atol=2e-7,
        rtol=0,
    )
    ours_decoded, audit = _quantize_support(support)
    _codes, deployed_scales, deployed_decoded = deployed_quantize_rows(
        deployed_normalize_three_blocks(support)
    )
    np.testing.assert_allclose(ours_decoded, deployed_decoded, atol=2e-7, rtol=0)
    assert deployed_scales.dtype == np.float16 and audit["scale_count"] == len(support) * 3
    beta = 7.0
    class_ids = np.unique(labels)
    logits = qknn_logits(support, labels, query, class_ids, beta=beta)
    support_norm = normalize_three_blocks(support).astype(np.float64)
    query_norm = normalize_three_blocks(query).astype(np.float64)
    similarity = query_norm @ support_norm.T
    expected = []
    for class_id in class_ids:
        local = beta * similarity[:, labels == class_id]
        maximum = np.max(local, axis=1, keepdims=True)
        expected.append(
            (maximum[:, 0] + np.log(np.mean(np.exp(local - maximum), axis=1)))
            / beta
        )
    np.testing.assert_allclose(logits, np.stack(expected, axis=1), atol=1e-12)


def test_support_cv_eta_uses_support_only_and_is_label_permutation_equivariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scorer(monkeypatch)
    validated = validate_feature_archive(_archive())
    arrays = validated["arrays"]
    episodes = build_receiver_lodo_episodes(validated, seed=21)
    scorer = _scorer()
    _base, support_cv, _audit = _build_base_logits_cache(arrays, episodes, scorer)
    candidate = {key: values[0] for key, values in _grid().items()}
    receiver = sorted(episodes)[0]
    episode = episodes[receiver][5]
    eta, receipt = _resolve_support_only_eta(
        arrays, episode, candidate, support_cv[(receiver, 5)]
    )
    tampered = dict(arrays)
    tampered_labels = arrays["labels"].copy()
    touched = np.r_[episode.calibration, episode.evaluation]
    tampered_labels[touched] = np.roll(tampered_labels[touched], 1)
    tampered["labels"] = tampered_labels
    eta_after, receipt_after = _resolve_support_only_eta(
        tampered, episode, candidate, support_cv[(receiver, 5)]
    )
    assert eta_after == eta and receipt_after == receipt

    mapping = {10: "alpha", 20: "beta", 30: "gamma"}
    permuted = dict(arrays)
    permuted["labels"] = np.asarray([mapping[int(v)] for v in arrays["labels"]])
    permuted["class_ids"] = np.asarray(["gamma", "beta", "alpha"])
    _base_p, support_cv_p, _audit_p = _build_base_logits_cache(
        permuted, episodes, scorer
    )
    eta_p, receipt_p = _resolve_support_only_eta(
        permuted, episode, candidate, support_cv_p[(receiver, 5)]
    )
    assert eta_p == pytest.approx(eta, abs=1e-12)
    assert receipt_p["base_nll"] == pytest.approx(receipt["base_nll"], abs=1e-12)
    assert receipt_p["qk_nll"] == pytest.approx(receipt["qk_nll"], abs=1e-12)


def test_scorer_output_drift_and_bad_grid_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def drifting(self, support, labels, query, classes):
        nonlocal calls
        calls += 1
        return _support_mean_scorer(support, labels, query, classes) + (
            1e-3 if calls % 2 == 0 else 0.0
        )

    monkeypatch.setattr(D81Phase1EpisodeScorer, "__call__", drifting)
    archive, manifest, manifest_sha = _write_exporter_v2(tmp_path / "drift", _archive())
    scorer = _scorer()
    with pytest.raises(ValueError, match="nondeterministic"):
        run_phase1_lodo_selection(
            archive,
            _grid(),
            base_scorer=scorer,
            base_scorer_id=scorer.scorer_id,
            feature_archive_manifest_path=manifest,
            feature_archive_manifest_sha256=manifest_sha,
            base_scorer_receipt_sha256=scorer.scorer_id,
        )

    bad = copy.deepcopy(_grid())
    bad["k1_eta_prior"] = [0.6]
    bad["eta_max"] = [0.5]
    _patch_scorer(monkeypatch)
    with pytest.raises(ValueError, match="k1_eta_prior"):
        _run(tmp_path / "bad", monkeypatch, grid=bad)
