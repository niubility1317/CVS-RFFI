from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d96_ground_geometry_lodo.py"
SPEC = importlib.util.spec_from_file_location("run_d96_ground_geometry_lodo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(tmp_path: Path, *, active_cells: int = 84) -> tuple[Path, str]:
    rng = np.random.default_rng(42)
    domains, classes, dim = 26, 6, 160
    class_centers = rng.normal(size=(classes, dim))
    class_centers /= np.linalg.norm(class_centers, axis=1, keepdims=True)
    nuisance = np.linalg.qr(rng.normal(size=(dim, 4)))[0]
    vectors = np.zeros((domains, classes, dim), dtype=np.float32)
    mask = np.zeros((domains, classes), dtype=np.uint8)
    active_domains = np.asarray([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25])
    for offset, domain in enumerate(active_domains):
        coefficient = rng.normal(scale=0.035, size=(classes, 4))
        rows = class_centers + coefficient @ nuisance.T
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        vectors[domain] = rows
        mask[domain] = 1
    if active_cells != 84:
        mask[active_domains[-1], -1] = 0
    maximum = np.max(np.abs(vectors), axis=2)
    scale = np.where(mask > 0, np.maximum(maximum / 127.0, 1e-6), 1.0).astype(np.float16)
    q = np.clip(np.rint(vectors / scale[..., None]), -127, 127).astype(np.int8)
    q[mask == 0] = 0
    root = tmp_path / "component"
    root.mkdir()
    npz = root / module.NPZ_NAME
    np.savez_compressed(
        npz,
        domain_class_q=q,
        domain_class_scale=scale,
        domain_class_mask=mask,
        domain_registry=np.arange(domains, dtype=np.int16),
        class_registry=np.asarray([f"tx-{index}" for index in range(classes)]),
        feature_schema=np.asarray(module.FEATURE_SCHEMA),
    )
    manifest = {
        "schema": module.COMPONENT_SCHEMA,
        "feature_dim": dim,
        "class_count": classes,
        "domain_count": domains,
        "active_domain_class_cells": int(mask.sum()),
        "component_npz_sha256": _sha(npz),
        "member_allowlist": [module.NPZ_NAME],
        "npz_member_allowlist": sorted(module.EXPECTED_MEMBERS),
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "provenance_status": "TEST_PHASE1_ONLY",
        "formal_phase2_eligible": False,
    }
    manifest_path = root / module.MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, _sha(manifest_path)


def test_strict_loader_and_lodo_output_are_geometry_only(tmp_path: Path) -> None:
    root, manifest_sha = _component(tmp_path)
    prototypes, domain_ids, _, audit = module.load_component(root, manifest_sha)
    result = module.run_lodo(prototypes, domain_ids, audit)
    assert result["status"] == "PARTIAL_PHASE1_GEOMETRY_SELECTION_DIAGNOSTIC"
    assert result["full_phase1_lock"] is False
    assert result["input_integrity_pass"] is True
    assert result["geometry_effectiveness_pass"] is False
    assert result["matrix"]["fold_count"] == 14
    assert result["matrix"]["candidate_count"] == 12
    assert result["matrix"]["fold_candidate_evaluations"] == 168
    assert len(result["candidate_summaries"]) == 12
    assert len(result["fold_results"]) == 168
    selected = result["selected_geometry"]
    assert selected["tau_quantile"] in module.TAU_QUANTILES
    assert selected["max_rank"] in module.RANKS
    assert selected["ridge"] is None
    assert selected["temp_base"] is None
    assert selected["temp_aux"] is None
    assert result["restrictions"]["target_admission_authorized"] is False
    assert result["restrictions"]["target_rows_used"] == 0
    assert all(np.isfinite(row["held_residual_projection_error"]) for row in result["fold_results"])
    assert all("harmful_margin_flip_count" in row for row in result["fold_results"])
    assert selected["worst_fold_explained_fraction"] < 1.0


def test_manifest_sha_is_mandatory(tmp_path: Path) -> None:
    root, _ = _component(tmp_path)
    with pytest.raises(module.D96GroundGeometryLODOError, match="manifest SHA"):
        module.load_component(root, "0" * 64)


def test_rejects_incomplete_84_cell_grid(tmp_path: Path) -> None:
    root, manifest_sha = _component(tmp_path, active_cells=83)
    with pytest.raises(module.D96GroundGeometryLODOError, match="manifest contract"):
        module.load_component(root, manifest_sha)


def test_cli_writes_canonical_partial_lock(tmp_path: Path) -> None:
    root, manifest_sha = _component(tmp_path)
    output = tmp_path / "result.json"
    assert module.main([
        "--component-dir", str(root),
        "--manifest-sha256", manifest_sha,
        "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == module.SCHEMA
    assert payload["full_phase1_lock"] is False
    assert payload["restrictions"]["may_fill_ridge_or_temperature"] is False


def test_fit_geometry_uses_density_weighted_centers_and_residuals() -> None:
    rng = np.random.default_rng(44)
    rows = rng.normal(size=(13, 6, 160))
    rows /= np.linalg.norm(rows, axis=2, keepdims=True)
    geometry = module._fit_geometry(rows, 0.5)
    raw = geometry["weighted_class_centers_raw"]
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    np.testing.assert_allclose(geometry["class_centers"], normalized, atol=1e-12)
    assert geometry["basis_all"].shape[1] <= 4


def test_held_projection_uses_same_weighted_raw_center_as_basis_fit() -> None:
    rng = np.random.default_rng(45)
    rows = rng.normal(size=(13, 6, 160))
    rows /= np.linalg.norm(rows, axis=2, keepdims=True)
    geometry = module._fit_geometry(rows, 0.5)
    held = rng.normal(size=(6, 160))
    held /= np.linalg.norm(held, axis=1, keepdims=True)
    result = module._evaluate_held(held, geometry, requested_rank=0)
    raw = geometry["weighted_class_centers_raw"]
    expected = float(
        np.sum((held - raw) ** 2)
        / max(module.EPSILON, float(np.sum((held - raw) ** 2)))
    )
    assert result["held_residual_space"] == "weighted_raw_class_center"
    assert result["held_residual_projection_error"] == pytest.approx(expected)
