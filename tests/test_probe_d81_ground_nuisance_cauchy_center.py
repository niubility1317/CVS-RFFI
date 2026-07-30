from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "scripts" / "probe_d81_ground_nuisance_cauchy_center.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("d81_probe_test_module", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_lock_uses_ground_only_for_support_center():
    probe = _load_probe()
    assert probe.ARM == "ground_nuisance_cauchy_center"
    assert "one-step Cauchy" in probe.FORMULA
    assert "target-support D62 metric" in probe.FORMULA
    assert "query anchor" not in probe.FORMULA
    assert "class quota" not in probe.FORMULA


def test_probe_patches_all_full_block_closures_before_d62_build():
    source = PROBE.read_text(encoding="utf-8")
    full_index = source.index("d42._fit_equal_prior_lda = full_fit")
    block_index = source.index("d43.build_structured_fit = structured_builder")
    build_index = source.index("base_fit, call_records = d62.build_d62_fit(")
    restore_index = source.index("d43.build_structured_fit = original_builder")
    assert max(full_index, block_index) < build_index < restore_index
    assert "D81 D43 module alias identity drift" in source


def test_probe_has_protocol_resource_and_hash_closure():
    source = PROBE.read_text(encoding="utf-8")
    required = (
        "d81_query_extra_macs",
        "d81_optimizer_steps_extra",
        "d81_trainable_parameters_extra",
        "ground_component_entry_npz_sha256",
        "ground_component_exit_npz_sha256",
        "ground_component_entry_manifest_sha256",
        "ground_component_exit_manifest_sha256",
        "support_center_transform_execution_count",
        "formal_candidate",
        "selected_only_full_k10_refit_allowed",
    )
    assert all(token in source for token in required)


def test_probe_does_not_claim_bundle_radius_or_count():
    source = PROBE.read_text(encoding="utf-8")
    assert '"ground_bundle_contains_sample_radius": False' in source
    assert '"ground_bundle_contains_sample_count": False' in source


def _confirmation_runner_stub():
    def registered_handles(manifest):
        return tuple(manifest["registered"])

    def original_guard(_before, _after):
        raise AssertionError("development guard must be replaced")

    return SimpleNamespace(
        _require_d42_development_cell=original_guard,
        legacy=SimpleNamespace(_registered_handles=registered_handles),
        D42_DEVELOPMENT_RECEIVER="20-1",
        D42_DEVELOPMENT_NEW_CLASS_COUNT=5,
        D25RunnerError=RuntimeError,
    )


def _confirmation_cell(seed=713102):
    before = {
        "receiver": "20-1",
        "seed": seed,
        "k_shot": 10,
        "registered": ("old0", "old1"),
    }
    after = {
        **before,
        "registered": (*before["registered"], "n0", "n1", "n2", "n3", "n4"),
    }
    return before, after


def test_confirmation_guard_accepts_only_exact_preregistered_cell():
    probe = _load_probe()
    runner = _confirmation_runner_stub()
    original = probe._install_confirmation_cell_guard(runner, 713102)
    assert original is not None
    runner._require_d42_development_cell(*_confirmation_cell())
    for broken in (
        _confirmation_cell(713103),
        ({**_confirmation_cell()[0], "receiver": "3-19"}, _confirmation_cell()[1]),
        (_confirmation_cell()[0], {**_confirmation_cell()[1], "k_shot": 5}),
    ):
        with pytest.raises(RuntimeError, match="D81 confirmation cell"):
            runner._require_d42_development_cell(*broken)


def test_confirmation_guard_rejects_unregistered_seed_before_install():
    probe = _load_probe()
    runner = _confirmation_runner_stub()
    with pytest.raises(probe.D81ProbeError, match="not preregistered"):
        probe._install_confirmation_cell_guard(runner, 713101)
    assert probe._install_confirmation_cell_guard(runner, None) is None


def test_translation_mac_inventory_matches_oof_structure():
    probe = _load_probe()
    rank = 14
    outer = probe._translation_macs(6 * 8, rank)
    inner = probe._translation_macs(6 * 7, rank)
    assert probe._d62_translation_chain_macs(6, 8, rank) == (
        4 * outer + 32 * inner
    )


def test_synthetic_d62_stack_transforms_every_full_block_oof_fit():
    probe = _load_probe()
    rng = np.random.default_rng(23)
    residual = rng.normal(size=(42, 160))
    floor = 1.0e-6
    covariance = residual.T @ residual / len(residual) + floor * np.eye(160)
    basis, weights, basis_audit = probe.core.ground_nuisance_basis(
        covariance, floor
    )
    ground_audit = {
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
        "d81_basis_sha256": basis_audit["basis_sha256"],
        "d81_spectral_weight_sha256": basis_audit["spectral_weight_sha256"],
        "d81_participation_ratio_effective_rank": basis_audit[
            "participation_ratio_effective_rank"
        ],
        "d81_retained_rank": basis_audit["retained_rank"],
        "d81_rank_policy": basis_audit["rank_policy"],
    }
    fit, component_records, transform_records = probe.build_d81_fit(
        d42, basis, weights, ground_audit
    )
    classes, shots = 3, 4
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = means[labels] + 0.12 * rng.normal(size=(classes * shots, 288))
    coefficient, intercept, audit = fit(rows, labels, classes, shots)
    assert coefficient.shape == (classes, 288)
    assert intercept.shape == (classes,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert len(component_records) == 2 * (shots + 1)
    assert len(transform_records) == 4 * (shots + 1)
    assert audit["d81_probe_arm"] == probe.ARM
    assert audit["d81_query_metric_source"] == "target_support_only_d62"
