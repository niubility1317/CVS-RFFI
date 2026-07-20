#!/usr/bin/env python3
"""Phase1-only geometry lock for D96 using the immutable 84-cell component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "cvs.phase1.d96.ground_geometry_lodo.v1"
COMPONENT_SCHEMA = "phase1_int8_domain_class_centroids_v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
NPZ_NAME = "int8_domain_class_prototypes.npz"
MANIFEST_NAME = "manifest.json"
EXPECTED_MEMBERS = {
    "class_registry",
    "domain_class_mask",
    "domain_class_q",
    "domain_class_scale",
    "domain_registry",
    "feature_schema",
}
FEATURE_DIM = 160
EXPECTED_CLASSES = 6
EXPECTED_ACTIVE_DOMAINS = 14
EXPECTED_ACTIVE_CELLS = EXPECTED_CLASSES * EXPECTED_ACTIVE_DOMAINS
TAU_QUANTILES = (0.25, 0.50, 0.75)
RANKS = (1, 2, 3, 4)
LARGE_MARGIN_THRESHOLD = 0.05
EPSILON = 1.0e-12


class D96GroundGeometryLODOError(ValueError):
    """Raised when the Phase1-only geometry lock contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(rows: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != FEATURE_DIM or not np.isfinite(value).all():
        raise D96GroundGeometryLODOError(f"{name} must be finite [N,{FEATURE_DIM}]")
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if bool(np.any(norm <= EPSILON)):
        raise D96GroundGeometryLODOError(f"{name} contains zero-norm rows")
    return value / norm


def load_component(
    component_dir: Path,
    expected_manifest_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Strictly load exactly the D22-style 84 complete-cell component."""

    root = component_dir.resolve()
    manifest_path = root / MANIFEST_NAME
    npz_path = root / NPZ_NAME
    if not manifest_path.is_file() or not npz_path.is_file():
        raise D96GroundGeometryLODOError("D96 geometry component member missing")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(expected_manifest_sha256).strip().lower():
        raise D96GroundGeometryLODOError("D96 geometry manifest SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        manifest.get("schema") != COMPONENT_SCHEMA
        or int(manifest.get("feature_dim", -1)) != FEATURE_DIM
        or int(manifest.get("class_count", -1)) != EXPECTED_CLASSES
        or int(manifest.get("active_domain_class_cells", -1)) != EXPECTED_ACTIVE_CELLS
        or set(manifest.get("member_allowlist", ())) != {NPZ_NAME}
        or set(manifest.get("npz_member_allowlist", ())) != EXPECTED_MEMBERS
        or manifest.get("phase2_phase1_prototype_component_immutable") is not True
        or manifest.get("phase2_phase1_prototype_update_access") is not False
        or manifest.get("phase2_phase1_prototype_member_or_exemplar_access") is not False
    ):
        raise D96GroundGeometryLODOError("D96 geometry manifest contract drift")
    npz_sha = _sha256(npz_path)
    if npz_sha != str(manifest.get("component_npz_sha256", "")).lower():
        raise D96GroundGeometryLODOError("D96 geometry NPZ SHA drift")

    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_MEMBERS:
            raise D96GroundGeometryLODOError("D96 geometry NPZ allowlist drift")
        q = np.asarray(payload["domain_class_q"])
        scales = np.asarray(payload["domain_class_scale"])
        mask = np.asarray(payload["domain_class_mask"])
        domains = np.asarray(payload["domain_registry"])
        classes = np.asarray(payload["class_registry"]).astype(str)
        feature_schema = str(np.asarray(payload["feature_schema"]).item())
    domain_count = int(manifest.get("domain_count", -1))
    expected_shape = (domain_count, EXPECTED_CLASSES, FEATURE_DIM)
    if (
        q.dtype != np.int8
        or q.shape != expected_shape
        or scales.dtype != np.float16
        or scales.shape != expected_shape[:2]
        or mask.dtype != np.uint8
        or mask.shape != expected_shape[:2]
        or domains.dtype != np.int16
        or domains.shape != (domain_count,)
        or classes.shape != (EXPECTED_CLASSES,)
        or len(np.unique(domains)) != domain_count
        or len(np.unique(classes)) != EXPECTED_CLASSES
        or feature_schema != FEATURE_SCHEMA
        or not np.isfinite(scales).all()
        or bool(np.any((mask != 0) & (mask != 1)))
    ):
        raise D96GroundGeometryLODOError("D96 geometry NPZ schema/shape drift")
    order = np.argsort(domains, kind="stable")
    q, scales, mask, domains = q[order], scales[order], mask[order], domains[order]
    row_counts = np.sum(mask, axis=1)
    active_rows = row_counts == EXPECTED_CLASSES
    if (
        bool(np.any((row_counts != 0) & ~active_rows))
        or int(np.sum(active_rows)) != EXPECTED_ACTIVE_DOMAINS
        or int(np.sum(mask)) != EXPECTED_ACTIVE_CELLS
        or not np.array_equal(np.sum(mask, axis=0), np.full(EXPECTED_CLASSES, EXPECTED_ACTIVE_DOMAINS))
        or bool(np.any(scales[mask.astype(bool)] <= 0.0))
        or bool(np.any(q[~mask.astype(bool)] != 0))
    ):
        raise D96GroundGeometryLODOError("D96 requires exactly 84 complete cells")
    prototypes = q[active_rows].astype(np.float64) * scales[active_rows].astype(np.float64)[..., None]
    prototypes = _normalize(
        prototypes.reshape(EXPECTED_ACTIVE_CELLS, FEATURE_DIM),
        "D96 active prototypes",
    ).reshape(EXPECTED_ACTIVE_DOMAINS, EXPECTED_CLASSES, FEATURE_DIM)
    audit = {
        "component_dir": str(root),
        "manifest_sha256": manifest_sha,
        "component_npz_sha256": npz_sha,
        "component_schema": COMPONENT_SCHEMA,
        "component_provenance_status": manifest.get("provenance_status"),
        "component_formal_phase2_eligible": bool(manifest.get("formal_phase2_eligible", False)),
        "active_domain_ids": domains[active_rows].astype(int).tolist(),
        "class_registry": classes.tolist(),
        "active_domain_count": EXPECTED_ACTIVE_DOMAINS,
        "class_count": EXPECTED_CLASSES,
        "active_cell_count": EXPECTED_ACTIVE_CELLS,
        "target_rows_read": 0,
        "source_sample_rows_read": 0,
    }
    return prototypes, domains[active_rows].astype(np.int64), classes, audit


def _domain_signatures(prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_centers = np.mean(prototypes, axis=0)
    residual = prototypes - class_centers[None, :, :]
    signatures = residual.reshape(len(prototypes), -1)
    norm = np.linalg.norm(signatures, axis=1, keepdims=True)
    if bool(np.any(norm <= EPSILON)):
        raise D96GroundGeometryLODOError("D96 domain signature has zero energy")
    return class_centers, residual, signatures / norm


def _tau_candidates(prototypes: np.ndarray) -> dict[float, float]:
    _, _, signatures = _domain_signatures(prototypes)
    distances = 1.0 - np.clip(signatures @ signatures.T, -1.0, 1.0)
    upper = distances[np.triu_indices(len(signatures), k=1)]
    positive = upper[upper > EPSILON]
    if len(positive) == 0:
        raise D96GroundGeometryLODOError("D96 cannot derive positive angular tau")
    return {
        quantile: max(EPSILON, float(np.quantile(positive, quantile)))
        for quantile in TAU_QUANTILES
    }


def _fit_geometry(prototypes: np.ndarray, tau: float) -> dict[str, Any]:
    _, _, signatures = _domain_signatures(prototypes)
    distance = 1.0 - np.clip(signatures @ signatures.T, -1.0, 1.0)
    density = np.sum(np.exp(-distance / float(tau)), axis=1)
    weights = 1.0 / density
    weights /= np.sum(weights)
    weighted_centers_raw = np.einsum("d,dcz->cz", weights, prototypes)
    class_centers = _normalize(weighted_centers_raw, "D96 weighted class centers")
    residual = prototypes - weighted_centers_raw[None, :, :]
    centered = signatures - np.sum(weights[:, None] * signatures, axis=0)
    weighted = np.sqrt(weights)[:, None] * centered
    diversity_values = np.linalg.eigvalsh(weighted @ weighted.T)
    diversity_values = diversity_values[diversity_values > EPSILON]
    d_eff = (
        1.0
        if len(diversity_values) == 0
        else float(np.sum(diversity_values) ** 2 / np.sum(diversity_values**2))
    )
    covariance_rows = (
        np.sqrt(weights)[:, None, None] * residual / math.sqrt(EXPECTED_CLASSES)
    ).reshape(-1, FEATURE_DIM)
    _, singular, vt = np.linalg.svd(covariance_rows, full_matrices=False)
    eigenvalues = singular**2
    positive = eigenvalues[eigenvalues > EPSILON]
    stable_rank = 0.0 if len(positive) == 0 else float(np.sum(positive) / np.max(positive))
    adaptive_cap = max(0, min(4, int(math.floor(d_eff)) - 1, len(positive), len(prototypes) - 1))
    return {
        "class_centers": class_centers,
        "weighted_class_centers_raw": weighted_centers_raw,
        "basis_all": vt[:adaptive_cap].T,
        "eigenvalues": positive[:adaptive_cap],
        "effective_domain_count": d_eff,
        "stable_rank": stable_rank,
        "adaptive_rank_cap": adaptive_cap,
        "density_weight_min": float(np.min(weights)),
        "density_weight_max": float(np.max(weights)),
    }


def _margin(scores: np.ndarray) -> np.ndarray:
    true = np.diag(scores)
    impostor = scores.copy()
    np.fill_diagonal(impostor, -np.inf)
    return true - np.max(impostor, axis=1)


def _evaluate_held(
    held: np.ndarray,
    geometry: dict[str, Any],
    requested_rank: int,
) -> dict[str, Any]:
    center_raw = np.asarray(
        geometry["weighted_class_centers_raw"], dtype=np.float64
    )
    score_centers = np.asarray(geometry["class_centers"], dtype=np.float64)
    effective_rank = min(int(requested_rank), int(geometry["adaptive_rank_cap"]))
    basis = np.asarray(geometry["basis_all"], dtype=np.float64)[:, :effective_rank]
    # The nuisance basis is fitted from prototype residuals around the weighted
    # raw class mean.  Audit the held domain in that same affine space, then
    # normalize only the reconstructed class centers used for cosine scoring.
    residual = held - center_raw
    projected = (residual @ basis) @ basis.T if effective_rank else np.zeros_like(residual)
    denominator = max(EPSILON, float(np.sum(residual * residual)))
    projection_error = float(np.sum((residual - projected) ** 2) / denominator)
    reconstructed = _normalize(
        center_raw + projected, "D96 reconstructed held centers"
    )
    teacher_scores = held @ score_centers.T
    compressed_scores = reconstructed @ score_centers.T
    teacher_pred = np.argmax(teacher_scores, axis=1)
    compressed_pred = np.argmax(compressed_scores, axis=1)
    truth = np.arange(EXPECTED_CLASSES)
    teacher_margin = _margin(teacher_scores)
    compressed_margin = _margin(compressed_scores)
    harmful_flip = (teacher_margin > 0.0) & (compressed_margin <= 0.0)
    beneficial_rescue = (teacher_margin <= 0.0) & (compressed_margin > 0.0)
    large = teacher_margin >= LARGE_MARGIN_THRESHOLD
    return {
        "effective_rank": effective_rank,
        "held_residual_space": "weighted_raw_class_center",
        "held_residual_projection_error": projection_error,
        "teacher_balanced_accuracy": float(np.mean(teacher_pred == truth)),
        "compressed_balanced_accuracy": float(np.mean(compressed_pred == truth)),
        "compressed_class_correct": (compressed_pred == truth).astype(int).tolist(),
        "harmful_margin_flip_count": int(np.sum(harmful_flip)),
        "large_margin_harmful_flip_count": int(np.sum(harmful_flip & large)),
        "beneficial_margin_rescue_count": int(np.sum(beneficial_rescue)),
        "top1_disagreement_count": int(np.sum(teacher_pred != compressed_pred)),
        "margin_delta_mean": float(np.mean(compressed_margin - teacher_margin)),
        "large_margin_threshold": LARGE_MARGIN_THRESHOLD,
    }


def _basis_stability(left: np.ndarray, right: np.ndarray) -> float:
    rank = min(left.shape[1], right.shape[1])
    if rank == 0:
        return 1.0 if left.shape[1] == right.shape[1] else 0.0
    return float(np.sum((left[:, :rank].T @ right[:, :rank]) ** 2) / rank)


def run_lodo(
    prototypes: np.ndarray,
    domain_ids: np.ndarray,
    component_audit: dict[str, Any],
) -> dict[str, Any]:
    if prototypes.shape != (EXPECTED_ACTIVE_DOMAINS, EXPECTED_CLASSES, FEATURE_DIM):
        raise D96GroundGeometryLODOError("D96 LODO input shape drift")
    full_tau = _tau_candidates(prototypes)
    full_geometry = {
        quantile: _fit_geometry(prototypes, tau) for quantile, tau in full_tau.items()
    }
    fold_rows: list[dict[str, Any]] = []
    for held_index, held_domain in enumerate(domain_ids):
        keep = np.arange(EXPECTED_ACTIVE_DOMAINS) != held_index
        train = prototypes[keep]
        held = prototypes[held_index]
        fold_tau = _tau_candidates(train)
        for quantile in TAU_QUANTILES:
            geometry = _fit_geometry(train, fold_tau[quantile])
            reference = full_geometry[quantile]
            for rank in RANKS:
                metrics = _evaluate_held(held, geometry, rank)
                effective_rank = int(metrics["effective_rank"])
                train_basis = np.asarray(geometry["basis_all"])[:, :effective_rank]
                reference_basis = np.asarray(reference["basis_all"])[:, :effective_rank]
                row = {
                    "held_domain_id": int(held_domain),
                    "tau_quantile": quantile,
                    "fold_tau": float(fold_tau[quantile]),
                    "requested_rank": rank,
                    "effective_domain_count": float(geometry["effective_domain_count"]),
                    "stable_rank": float(geometry["stable_rank"]),
                    "adaptive_rank_cap": int(geometry["adaptive_rank_cap"]),
                    "basis_stability": _basis_stability(train_basis, reference_basis),
                    **metrics,
                }
                fold_rows.append(row)

    summaries = []
    for quantile in TAU_QUANTILES:
        for rank in RANKS:
            rows = [
                row
                for row in fold_rows
                if row["tau_quantile"] == quantile and row["requested_rank"] == rank
            ]
            class_correct = np.asarray([row["compressed_class_correct"] for row in rows])
            summary = {
                "tau_quantile": quantile,
                "full_component_tau": float(full_tau[quantile]),
                "requested_rank": rank,
                "max_effective_rank": int(max(row["effective_rank"] for row in rows)),
                "large_margin_harmful_flip_count": int(sum(row["large_margin_harmful_flip_count"] for row in rows)),
                "harmful_margin_flip_count": int(sum(row["harmful_margin_flip_count"] for row in rows)),
                "beneficial_margin_rescue_count": int(sum(row["beneficial_margin_rescue_count"] for row in rows)),
                "top1_disagreement_count": int(sum(row["top1_disagreement_count"] for row in rows)),
                "worst_fold_projection_error": float(max(row["held_residual_projection_error"] for row in rows)),
                "worst_fold_explained_fraction": float(1.0 - max(row["held_residual_projection_error"] for row in rows)),
                "mean_projection_error": float(np.mean([row["held_residual_projection_error"] for row in rows])),
                "compressed_class_floor": float(np.min(np.mean(class_correct, axis=0))),
                "mean_compressed_balanced_accuracy": float(np.mean([row["compressed_balanced_accuracy"] for row in rows])),
                "mean_basis_stability": float(np.mean([row["basis_stability"] for row in rows])),
                "mean_effective_domain_count": float(np.mean([row["effective_domain_count"] for row in rows])),
                "mean_stable_rank": float(np.mean([row["stable_rank"] for row in rows])),
                "effective_rank_min": int(min(row["effective_rank"] for row in rows)),
                "effective_rank_mean": float(np.mean([row["effective_rank"] for row in rows])),
                "effective_rank_max": int(max(row["effective_rank"] for row in rows)),
                "effective_rank_by_fold": [int(row["effective_rank"]) for row in rows],
            }
            summaries.append(summary)
    selected = min(
        summaries,
        key=lambda row: (
            row["large_margin_harmful_flip_count"],
            row["harmful_margin_flip_count"],
            row["worst_fold_projection_error"],
            -row["compressed_class_floor"],
            row["effective_rank_max"],
            row["requested_rank"],
            row["tau_quantile"],
        ),
    )
    return {
        "schema": SCHEMA,
        "status": "PARTIAL_PHASE1_GEOMETRY_SELECTION_DIAGNOSTIC",
        "full_phase1_lock": False,
        "input_integrity_pass": True,
        "selection_completed": True,
        "geometry_effectiveness_pass": False,
        "geometry_effectiveness_status": "NOT_ESTABLISHED_NO_PREREGISTERED_EFFECT_SIZE_GATE",
        "component": component_audit,
        "matrix": {
            "fold_policy": "leave_one_active_ground_domain_out",
            "fold_count": EXPECTED_ACTIVE_DOMAINS,
            "tau_policy": "training_fold_positive_pairwise_angular_distance_quantile",
            "tau_quantiles": list(TAU_QUANTILES),
            "rank_candidates": list(RANKS),
            "candidate_count": len(TAU_QUANTILES) * len(RANKS),
            "fold_candidate_evaluations": len(fold_rows),
        },
        "selection_policy": (
            "min(large_harmful_flips,harmful_flips,worst_fold_projection_error,"
            "-compressed_class_floor,effective_rank,requested_rank,tau_quantile)"
        ),
        "selected_geometry": {
            "tau_quantile": selected["tau_quantile"],
            "tau_full_component": selected["full_component_tau"],
            "max_rank": selected["requested_rank"],
            "effective_rank_min": selected["effective_rank_min"],
            "effective_rank_max": selected["effective_rank_max"],
            "worst_fold_explained_fraction": selected["worst_fold_explained_fraction"],
            "ridge": None,
            "temp_base": None,
            "temp_aux": None,
        },
        "candidate_summaries": summaries,
        "fold_results": fold_rows,
        "restrictions": {
            "target_rows_used": 0,
            "target_admission_authorized": False,
            "geometry_effectiveness_claim_authorized": False,
            "ridge_locked": False,
            "temperatures_locked": False,
            "may_fill_ridge_or_temperature": False,
            "may_select_target_candidate": False,
            "reason": "84-cell geometry lacks physical support/query residuals and D81 logits",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    prototypes, domain_ids, _, audit = load_component(
        Path(args.component_dir), args.manifest_sha256
    )
    result = run_lodo(prototypes, domain_ids, audit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output.resolve()),
        "schema": result["schema"],
        "status": result["status"],
        "input_integrity_pass": result["input_integrity_pass"],
        "selected_geometry": result["selected_geometry"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
