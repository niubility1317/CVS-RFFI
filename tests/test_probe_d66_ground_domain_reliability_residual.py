from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d66_ground_domain_reliability_residual.py"
SPEC = importlib.util.spec_from_file_location("probe_d66_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d66 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d66)


def _write_component(root: Path, class_order: np.ndarray | None = None) -> tuple[Path, str]:
    root.mkdir(parents=True)
    rng = np.random.default_rng(66)
    domains, classes, dim = 5, 3, 160
    base = rng.normal(scale=0.4, size=(classes, dim))
    drift = rng.normal(scale=0.05, size=(domains, classes, dim))
    values = base[None, :, :] + drift
    scales = np.max(np.abs(values), axis=2) / 127.0
    q = np.rint(values / scales[..., None]).clip(-127, 127).astype(np.int8)
    scales = scales.astype(np.float16)
    mask = np.ones((domains, classes), dtype=np.uint8)
    mask[0, 2] = 0
    registry = np.asarray(["a", "b", "c"])
    if class_order is not None:
        q = q[:, class_order]
        scales = scales[:, class_order]
        mask = mask[:, class_order]
        registry = registry[class_order]
    npz = root / d66.NPZ_NAME
    np.savez(
        npz,
        domain_class_q=q,
        domain_class_scale=scales,
        domain_class_mask=mask,
        domain_registry=np.arange(domains, dtype=np.int16),
        class_registry=registry,
        feature_schema=np.asarray(d66.EXPECTED_FEATURE_SCHEMA),
    )
    npz_sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    active = int(mask.sum())
    manifest = {
        "schema": "phase1_int8_domain_class_centroids_v1",
        "feature_dim": 160,
        "feature_key": "z_id",
        "domain_count": domains,
        "class_count": classes,
        "active_domain_class_cells": active,
        "component_npz_sha256": npz_sha,
        "member_allowlist": [d66.NPZ_NAME],
        "npz_member_allowlist": sorted(d66.EXPECTED_MEMBERS),
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "phase2_phase1_prototype_sample_reconstruction_access": False,
        "formal_phase2_eligible": False,
        "provenance_status": "UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        "resource_audit": {"logical_dense_state_bytes": 4096},
    }
    manifest_path = root / d66.MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_ground_reliability_is_bounded_readonly_and_class_permutation_invariant(
    tmp_path: Path,
) -> None:
    root1, sha1 = _write_component(tmp_path / "one")
    root2, sha2 = _write_component(tmp_path / "two", np.asarray([2, 0, 1]))
    before = hashlib.sha256((root1 / d66.NPZ_NAME).read_bytes()).hexdigest()
    scale1, audit1 = d66.load_ground_domain_reliability(root1, sha1, 288)
    scale2, audit2 = d66.load_ground_domain_reliability(root2, sha2, 288)
    after = hashlib.sha256((root1 / d66.NPZ_NAME).read_bytes()).hexdigest()
    np.testing.assert_allclose(scale1, scale2, rtol=0.0, atol=1e-12)
    assert scale1.flags.writeable is False
    assert np.all(scale1[:160] > 1.0)
    assert np.all(scale1[:160] < np.sqrt(2.0))
    assert np.array_equal(scale1[160:], np.ones(128))
    assert before == after
    assert audit1["ground_active_domain_class_cells"] == 14
    assert audit1["persistent_full_precision_ground_anchor_count"] == 0
    assert audit1["ground_z_scale_sha256"] == audit2["ground_z_scale_sha256"]


def test_ground_component_rejects_sha_policy_and_shape_drift(tmp_path: Path) -> None:
    root, sha = _write_component(tmp_path / "component")
    with pytest.raises(d66.D66ProbeError, match="manifest SHA"):
        d66.load_ground_domain_reliability(root, "0" * 64, 288)
    manifest_path = root / d66.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phase2_phase1_prototype_update_access"] = True
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    bad_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(d66.D66ProbeError, match="policy"):
        d66.load_ground_domain_reliability(root, bad_sha, 288)
    assert sha != bad_sha


def test_shared_transform_compiles_back_and_is_class_agnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(660)
    dimension = 9
    scale = np.linspace(1.01, 1.4, dimension)
    ground = {
        "ground_active_domain_class_cells": 14,
        "ground_z_scale_sha256": "scale-sha",
        "ground_z_scale_min": float(scale.min()),
        "ground_z_scale_max": float(scale.max()),
    }
    calls: list[dict[str, object]] = []

    def fake_builder(_d42: object):
        def fake_fit(rows, labels, class_count, k_shot):
            x = np.asarray(rows, dtype=np.float64)
            y = np.asarray(labels, dtype=np.int64)
            coef = np.stack([x[y == i].mean(axis=0) for i in range(class_count)]).astype(
                np.float32
            )
            bias = np.linspace(-0.2, 0.2, class_count).astype(np.float32)
            calls.append({"rows": x.copy(), "k": k_shot})
            return coef, bias, {"base": True}

        return fake_fit, calls

    monkeypatch.setattr(d66.d62, "build_d62_fit", fake_builder)
    d42 = SimpleNamespace(FEATURE_DIM=dimension)
    fit, records = d66.build_d66_fit(d42, scale, ground)
    rows = rng.normal(size=(12, dimension))
    labels = np.repeat(np.arange(3), 4)
    coef, bias, audit = fit(rows, labels, 3, 4)
    scaled_rows = rows * scale[None, :]
    scaled_coef = np.stack(
        [scaled_rows[labels == i].mean(axis=0) for i in range(3)]
    ).astype(np.float32)
    reference = scaled_rows @ scaled_coef.astype(np.float64).T + bias[None, :]
    actual = rows @ coef.astype(np.float64).T + bias[None, :]
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
    assert len(records) == 1
    assert audit["d66_shared_transform_all_registered_classes"] is True
    assert audit["d66_old_new_role_specific_branch"] is False
    assert audit["d66_class_id_specific_formula"] is False
    assert audit["d66_uses_outer_held_or_query"] is False
    assert audit["d66_hyperparameter_count"] == 0


def test_formula_has_no_role_scene_or_tunable_branch() -> None:
    lowered = d66.FORMULA.lower()
    assert "between" in lowered and "within" in lowered and "sqrt" in lowered
    for forbidden in ("role", "scene", "receiver", "threshold", "alpha", "temperature"):
        assert forbidden not in lowered
