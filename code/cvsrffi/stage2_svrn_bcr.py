"""Immutable support-only SVRN-qKNN-BCRR/r2 state.

The module implements design-commit ``7f1899fb`` without a Phase1 selector.
All data-dependent choices use only one balanced target-support row.  Runtime
state contains an INT8 qKNN bank and INT8 BCR weights; no FP32 support or
weight sidecar is serialized.

The INT8 qKNN bank remains the sole all-class decision backbone.  BCRR is a
branch-local, support-only residual on score geometry; it never changes that
bank, its neighbor order, or qKNN kernel evaluation.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    audit_int8_margin,
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    deserialize_typed_zid_runtime_state,
    identity_shared_psd_metric,
    normalize_zid_rows,
    _canonical_order,
    _quantize_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)

KAPPA = 2.5
ETA_GRID = (0.0, 0.25, 0.5)
MASK_RESIDUES = (0, 1)
MASK_MODULUS = 5
MASK_RETENTION = 0.8
LAMBDA0 = 1.0
SCORE_EPSILON = 1.0e-12
TRANSFORM_EPSILON = 1.0e-6
MAX_CLASSES = 40
MAX_STATE_BYTES = 256 * 1024
MIN_TOP1_AGREEMENT = 0.995
STATE_SCHEMA = "cvs.stage2.svrn_bcr.branch_state.v1"
WIRE_SCHEMA = "cvs.stage2.svrn_bcr.branch_wire.v1"
ETA_SCHEMA = "cvs.stage2.svrn_bcr.eta_receipt.v1"
BCRR_SCHEMA = "cvs.stage2.svrn_bcr.bcrr_receipt.v2"
BCRR_DENOMINATOR = 254
BCRR_MAX_OMEGA = 0.5


class SVRNBCRStateError(ValueError):
    pass


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canon(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canon(value)).hexdigest()


def _sha(value: str, name: str) -> str:
    if (
        type(value) is not str or len(value) != 64 or value != value.lower()
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise SVRNBCRStateError(f"{name} must be an exact lowercase SHA256")
    return value


def _finite_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32 or rows.ndim != 2 or rows.shape[1] != Z_DIM
        or len(rows) < 1 or not np.isfinite(rows).all()
    ):
        raise SVRNBCRStateError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if (
        not classes or classes != tuple(sorted(classes))
        or len(classes) != len(set(classes)) or len(classes) > MAX_CLASSES
    ):
        raise SVRNBCRStateError("registered class registry drift")
    return classes


def _balanced_labels(
    labels: Sequence[str], classes: Sequence[str], active_k: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    registry = _registry(classes)
    values = tuple(str(value) for value in labels)
    if any(value not in registry for value in values):
        raise SVRNBCRStateError("support label outside registered classes")
    index = np.asarray([registry.index(value) for value in values], np.int16)
    counts = tuple(int(np.sum(index == i)) for i in range(len(registry)))
    if counts != (active_k,) * len(registry):
        raise SVRNBCRStateError("support must be balanced exact K-shot")
    return registry, index


def svrn_transform(value: np.ndarray, eta: float) -> np.ndarray:
    """Apply the frozen row-wise ``T_eta`` transform exactly."""

    rows = _finite_rows(value, "SVRN input")
    if type(eta) not in (float, np.float16, np.float32, np.float64) or float(eta) not in ETA_GRID:
        raise SVRNBCRStateError("eta must be one frozen grid value")
    x = rows.astype(np.float64)
    centered = x - np.mean(x, axis=1, keepdims=True)
    ln = centered / np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + TRANSFORM_EPSILON)
    clipped = np.clip(ln, -KAPPA, KAPPA)
    restored = (
        np.linalg.norm(x, axis=1, keepdims=True) * clipped
        / (np.linalg.norm(clipped, axis=1, keepdims=True) + TRANSFORM_EPSILON)
    )
    out = (1.0 - float(eta)) * x + float(eta) * restored
    if not np.isfinite(out).all():
        raise SVRNBCRStateError("SVRN transform became non-finite")
    return np.ascontiguousarray(out, dtype=np.float32)


def _masked_views(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(Z_DIM)
    out = []
    for residue in MASK_RESIDUES:
        view = np.array(rows, copy=True)
        view[:, index % MASK_MODULUS == residue] = 0.0
        out.append(view)
    return out[0], out[1]


def _cross_view_gammas(
    source: np.ndarray,
    destination: np.ndarray,
    labels: tuple[str, ...],
    physical_ids: tuple[str, ...],
    classes: tuple[str, ...],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {label: [] for label in classes}
    lab = np.asarray(labels)
    ids = np.asarray(physical_ids)
    for i, label in enumerate(labels):
        centers: dict[str, np.ndarray] = {}
        for candidate in classes:
            keep = (lab == candidate) & (ids != physical_ids[i])
            if not np.any(keep):
                raise SVRNBCRStateError("cross-view LOO center is empty")
            centers[candidate] = np.mean(destination[keep].astype(np.float64), axis=0)
        true_distance = float(np.sum((source[i].astype(np.float64) - centers[label]) ** 2))
        other_distance = min(
            float(np.sum((source[i].astype(np.float64) - centers[candidate]) ** 2))
            for candidate in classes if candidate != label
        )
        result[label].append(-true_distance + other_distance)
    return result


def _eta_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: receipt[key] for key in receipt if key != "receipt_sha256"}


def verify_eta_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema", "kappa", "eta_grid", "tie_break", "mask_modulus",
        "mask_residues", "mask_retention", "mask_feature_sha256",
        "support_physical_ids_sha256", "same_physical_id_synchronous_loo",
        "K", "loo_center_count", "direction_class_gamma", "direction_scores",
        "direction_selected_eta", "selected_eta", "fallback",
        "query_rows_used_for_fit", "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise SVRNBCRStateError("eta receipt schema drift")
    if receipt["receipt_sha256"] != _digest(_eta_body(receipt)):
        raise SVRNBCRStateError("eta receipt SHA drift")
    expected_mask_sha = _digest(
        {
            "d": Z_DIM, "modulus": MASK_MODULUS,
            "residues": list(MASK_RESIDUES), "retention": MASK_RETENTION,
        }
    )
    if (
        receipt["schema"] != ETA_SCHEMA or receipt["kappa"] != KAPPA
        or tuple(receipt["eta_grid"]) != ETA_GRID
        or receipt["tie_break"] != "maximize_min_class_median_gamma_then_smaller_eta"
        or receipt["mask_modulus"] != MASK_MODULUS
        or tuple(receipt["mask_residues"]) != MASK_RESIDUES
        or receipt["mask_retention"] != MASK_RETENTION
        or receipt["mask_feature_sha256"] != expected_mask_sha
        or receipt["same_physical_id_synchronous_loo"] is not True
        or receipt["query_rows_used_for_fit"] != 0
        or receipt["K"] not in (1, 5, 10)
        or receipt["loo_center_count"] != (0 if receipt["K"] == 1 else receipt["K"] - 1)
    ):
        raise SVRNBCRStateError("eta frozen constants drift")
    selected = receipt["direction_selected_eta"]
    if set(selected) != {"0_to_1", "1_to_0"}:
        raise SVRNBCRStateError("eta direction selection drift")
    scores = receipt["direction_scores"]
    gammas = receipt["direction_class_gamma"]
    for direction in ("0_to_1", "1_to_0"):
        if set(scores.get(direction, {})) != {str(value) for value in ETA_GRID}:
            raise SVRNBCRStateError("eta direction score grid drift")
        maximum = max(float(value) for value in scores[direction].values())
        expected = min(
            eta for eta in ETA_GRID if float(scores[direction][str(eta)]) == maximum
        )
        if float(selected[direction]) != expected:
            raise SVRNBCRStateError("eta tie-break drift")
        class_map = gammas.get(direction, {})
        if not class_map or any(set(value) != {str(x) for x in ETA_GRID} for value in class_map.values()):
            raise SVRNBCRStateError("eta per-class gamma drift")
        for eta in ETA_GRID:
            expected_score = min(float(value[str(eta)]) for value in class_map.values())
            if float(scores[direction][str(eta)]) != expected_score:
                raise SVRNBCRStateError("eta direction score/class gamma mismatch")
    same_nonzero = (
        float(selected["0_to_1"]) == float(selected["1_to_0"])
        and float(selected["0_to_1"]) != 0.0
    )
    proposed = float(selected["0_to_1"])
    nondestructive = same_nonzero and all(
        float(values[str(proposed)]) >= float(values[str(0.0)])
        for direction in gammas.values() for values in direction.values()
    )
    expected_eta = proposed if nondestructive else 0.0
    if receipt["K"] == 1:
        expected_eta, expected_fallback = 0.0, "K1_identity"
    elif float(selected["0_to_1"]) != float(selected["1_to_0"]):
        expected_fallback = "direction_disagreement"
    elif float(selected["0_to_1"]) == 0.0:
        expected_fallback = "selected_identity"
    elif not nondestructive:
        expected_fallback = "class_direction_below_eta0"
    else:
        expected_fallback = "none"
    if float(receipt["selected_eta"]) != expected_eta:
        raise SVRNBCRStateError("eta bidirectional fallback drift")
    if receipt["fallback"] != expected_fallback:
        raise SVRNBCRStateError("eta fallback reason drift")
    return MappingProxyType(dict(receipt))


def select_svrn_eta(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    support_physical_ids: Sequence[str],
    *,
    active_k: int,
) -> Mapping[str, Any]:
    """Select eta by the frozen two masked cross-view LOO directions."""

    rows = _finite_rows(support_zid, "eta support")
    classes, _ = _balanced_labels(support_labels, registered_classes, active_k)
    labels = tuple(str(value) for value in support_labels)
    physical_ids = tuple(str(value) for value in support_physical_ids)
    if len(physical_ids) != len(rows) or len(set(physical_ids)) != len(physical_ids):
        raise SVRNBCRStateError("eta support physical IDs must be unique")
    mask_sha = _digest(
        {"d": Z_DIM, "modulus": MASK_MODULUS, "residues": list(MASK_RESIDUES), "retention": MASK_RETENTION}
    )
    if active_k == 1:
        gammas = {
            direction: {label: {str(eta): 0.0 for eta in ETA_GRID} for label in classes}
            for direction in ("0_to_1", "1_to_0")
        }
        scores = {direction: {str(eta): 0.0 for eta in ETA_GRID} for direction in gammas}
        selected = {direction: 0.0 for direction in gammas}
        fallback = "K1_identity"
    else:
        views = _masked_views(rows)
        transformed = {
            eta: (svrn_transform(views[0], eta), svrn_transform(views[1], eta))
            for eta in ETA_GRID
        }
        raw_gamma: dict[str, dict[str, dict[str, float]]] = {
            "0_to_1": {label: {} for label in classes},
            "1_to_0": {label: {} for label in classes},
        }
        for eta in ETA_GRID:
            for name, a, b in (
                ("0_to_1", 0, 1), ("1_to_0", 1, 0),
            ):
                values = _cross_view_gammas(
                    transformed[eta][a], transformed[eta][b], labels,
                    physical_ids, classes,
                )
                for label in classes:
                    raw_gamma[name][label][str(eta)] = float(np.median(values[label]))
        gammas = raw_gamma
        scores = {
            direction: {
                str(eta): float(min(gammas[direction][label][str(eta)] for label in classes))
                for eta in ETA_GRID
            }
            for direction in gammas
        }
        selected = {}
        for direction in scores:
            maximum = max(scores[direction].values())
            selected[direction] = min(eta for eta in ETA_GRID if scores[direction][str(eta)] == maximum)
        if selected["0_to_1"] != selected["1_to_0"]:
            chosen, fallback = 0.0, "direction_disagreement"
        elif selected["0_to_1"] == 0.0:
            chosen, fallback = 0.0, "selected_identity"
        else:
            proposed = float(selected["0_to_1"])
            safe = all(
                gammas[direction][label][str(proposed)] >= gammas[direction][label][str(0.0)]
                for direction in gammas for label in classes
            )
            chosen, fallback = (proposed, "none") if safe else (0.0, "class_direction_below_eta0")
    if active_k == 1:
        chosen = 0.0
    body = {
        "schema": ETA_SCHEMA,
        "kappa": KAPPA,
        "eta_grid": list(ETA_GRID),
        "tie_break": "maximize_min_class_median_gamma_then_smaller_eta",
        "mask_modulus": MASK_MODULUS,
        "mask_residues": list(MASK_RESIDUES),
        "mask_retention": MASK_RETENTION,
        "mask_feature_sha256": mask_sha,
        "support_physical_ids_sha256": _digest(list(physical_ids)),
        "same_physical_id_synchronous_loo": True,
        "K": int(active_k),
        "loo_center_count": 0 if active_k == 1 else active_k - 1,
        "direction_class_gamma": gammas,
        "direction_scores": scores,
        "direction_selected_eta": selected,
        "selected_eta": float(chosen),
        "fallback": fallback,
        "query_rows_used_for_fit": 0,
    }
    receipt = {**body, "receipt_sha256": _digest(body)}
    return verify_eta_receipt(receipt)


def _centered_y(class_indices: np.ndarray, class_count: int) -> np.ndarray:
    y = np.full((len(class_indices), class_count), -1.0 / class_count, np.float64)
    y[np.arange(len(class_indices)), class_indices.astype(np.int64)] += 1.0
    return y


def _ridge_fit_and_loo(
    h: np.ndarray, class_indices: np.ndarray, active_k: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, int]]:
    """One dual solve gives the full BCR weights and every exact LOO score."""

    rows = np.asarray(h, np.float64)
    n, d = rows.shape
    c = int(np.max(class_indices)) + 1
    y = _centered_y(class_indices, c)
    # D=(1/K)I. Multiplying the normal equation by K yields this dual system.
    ridge_lambda = LAMBDA0 * float(np.sum(rows * rows) / active_k / d)
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise SVRNBCRStateError("BCR lambda degeneracy")
    gram = rows @ rows.T
    system = gram + active_k * ridge_lambda * np.eye(n, dtype=np.float64)
    rhs = np.concatenate((y, np.eye(n, dtype=np.float64)), axis=1)
    try:
        solved = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise SVRNBCRStateError("BCR dual solve degeneracy") from exc
    alpha, inverse = solved[:, :c], solved[:, c:]
    weights = rows.T @ alpha
    fitted = rows @ weights
    leverage = np.diag(gram @ inverse)
    denominator = 1.0 - leverage
    if np.any(denominator <= SCORE_EPSILON) or not np.isfinite(denominator).all():
        raise SVRNBCRStateError("BCR LOO leverage degeneracy")
    loo = (fitted - leverage[:, None] * y) / denominator[:, None]
    ledger = {
        "bcr_factorizations": 1,
        "bcr_loo_full_d3_count": 0,
        "bcr_dual_gram_mac": n * n * d,
        "bcr_dual_solve_equivalent_mac": n * n * n,
        "bcr_weight_mac": d * n * c,
        "bcr_fitted_mac": n * d * c,
        "bcr_hat_mac": n * n * n,
    }
    return weights, loo, ridge_lambda, ledger


def _identity_scales(h: np.ndarray, indices: np.ndarray, classes: int, config: Phase1ZIDStudentTLock) -> np.ndarray:
    if config.active_k == 1:
        return np.full(classes, config.shared_h0, np.float64)
    values = []
    for class_index in range(classes):
        local = h[indices == class_index]
        distance = np.maximum(2.0 * (1.0 - np.clip(local @ local.T, -1.0, 1.0)), 0.0)
        empirical = float(np.mean(distance[np.triu_indices(config.active_k, 1)]))
        shrunk = (empirical + config.scale_prior_strength * config.shared_h0 ** 2) / (1.0 + config.scale_prior_strength)
        values.append(np.clip(math.sqrt(max(shrunk, SCORE_EPSILON)), config.shared_h0 * config.scale_min_ratio, config.shared_h0 * config.scale_max_ratio))
    return np.asarray(values, np.float64)


def _qknn_loo_logits(h: np.ndarray, indices: np.ndarray, config: Phase1ZIDStudentTLock) -> np.ndarray:
    n = len(h); c = int(np.max(indices)) + 1
    scales = _identity_scales(h, indices, c, config)
    distance = np.maximum(2.0 * (1.0 - np.clip(h @ h.T, -1.0, 1.0)), 0.0)
    out = np.empty((n, c), np.float64)
    for i in range(n):
        for class_index in range(c):
            keep = np.flatnonzero(indices == class_index)
            keep = keep[keep != i]
            if not len(keep):
                raise SVRNBCRStateError("qKNN LOO class became empty")
            hscale = float(scales[class_index])
            kernel = (
                -config.kernel_volume_gamma * config.kernel_effective_dim * math.log(hscale)
                -0.5 * (config.student_nu + config.kernel_effective_dim)
                * np.log1p(distance[i, keep] / (config.student_nu * hscale * hscale))
            )
            maximum = float(np.max(kernel))
            out[i, class_index] = maximum + math.log(float(np.sum(np.exp(kernel - maximum)))) - math.log(len(keep))
    return out


def normalize_score_rows(scores: np.ndarray) -> np.ndarray:
    """Frozen ``N(s)=sqrt(C)(s-mean(s))/||s-mean(s)||_2`` score geometry."""
    value = np.asarray(scores, np.float64)
    if value.ndim != 2 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise SVRNBCRStateError("BCRR scores must be finite [N,C]")
    centered = value - np.mean(value, axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norm <= SCORE_EPSILON) or not np.isfinite(norm).all():
        raise SVRNBCRStateError("BCRR score normalization degeneracy")
    return math.sqrt(value.shape[1]) * centered / norm


def _bcrr_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: receipt[key] for key in receipt if key != "receipt_sha256"}


def verify_bcrr_receipt(receipt: Mapping[str, Any], *, branch: str, support_sha256: str) -> Mapping[str, Any]:
    required = {
        "schema", "branch", "branch_support_sha256", "normalization", "fusion",
        "loss", "support_bank_format", "bcr_weight_format", "directional_class_loss_qknn", "directional_class_loss_bcrr",
        "omega_star", "omega_q", "quantization_denominator", "fallback", "K",
        "same_physical_id_synchronous_loo", "query_rows_used_for_fit", "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise SVRNBCRStateError("BCRR receipt schema drift")
    if receipt["receipt_sha256"] != _digest(_bcrr_body(receipt)):
        raise SVRNBCRStateError("BCRR receipt SHA drift")
    if (
        receipt["schema"] != BCRR_SCHEMA or receipt["branch"] != branch
        or receipt["branch_support_sha256"] != support_sha256
        or receipt["normalization"] != "sqrt(C)*(s-mean(s))/l2(s-mean(s))"
        or receipt["fusion"] != "F=(1-omega)*N(qKNN)+omega*N(BCR)"
        or receipt["loss"] != "cross_entropy"
        or receipt["support_bank_format"] != "per_row_qint8_fp16_scale_decode_l2_v1"
        or receipt["bcr_weight_format"] != "per_column_qint8_fp16_scale_decode_v1"
        or receipt["quantization_denominator"] != BCRR_DENOMINATOR
        or receipt["same_physical_id_synchronous_loo"] is not True
        or receipt["query_rows_used_for_fit"] != 0
        or receipt["K"] not in (1, 5, 10)
    ):
        raise SVRNBCRStateError("BCRR frozen branch/formula drift")
    star, omega = float(receipt["omega_star"]), float(receipt["omega_q"])
    if not (math.isfinite(star) and math.isfinite(omega) and 0.0 <= star <= BCRR_MAX_OMEGA and 0.0 <= omega <= BCRR_MAX_OMEGA):
        raise SVRNBCRStateError("BCRR omega bounds drift")
    if omega != math.floor(BCRR_DENOMINATOR * star) / BCRR_DENOMINATOR:
        raise SVRNBCRStateError("BCRR omega floor quantization drift")
    qloss, floss = receipt["directional_class_loss_qknn"], receipt["directional_class_loss_bcrr"]
    if set(qloss) != {"0_to_1", "1_to_0"} or set(floss) != set(qloss):
        raise SVRNBCRStateError("BCRR cross-view direction drift")
    for direction in qloss:
        if set(qloss[direction]) != set(floss[direction]) or not qloss[direction]:
            raise SVRNBCRStateError("BCRR class safety set drift")
        for label in qloss[direction]:
            if not (math.isfinite(float(qloss[direction][label])) and math.isfinite(float(floss[direction][label]))):
                raise SVRNBCRStateError("BCRR non-finite safety loss")
            if float(floss[direction][label]) > float(qloss[direction][label]) + 1.0e-10:
                raise SVRNBCRStateError("BCRR quantized safety violation")
    if int(receipt["K"]) == 1 and (star != 0.0 or omega != 0.0 or receipt["fallback"] != "K1_identity"):
        raise SVRNBCRStateError("K1 BCRR identity drift")
    if int(receipt["K"]) != 1 and receipt["fallback"] not in {"none", "safety_set_empty", "score_normalization_degenerate"}:
        raise SVRNBCRStateError("BCRR fallback drift")
    return MappingProxyType(dict(receipt))


def _cross_entropy(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    return -logits[np.arange(len(indices)), indices] + maximum[:, 0] + np.log(np.sum(np.exp(logits - maximum), axis=1))


def _cross_view_loo_scores(
    h: np.ndarray, indices: np.ndarray, classes: Sequence[str], config: Phase1ZIDStudentTLock,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Exact same-physical-ID LOO, evaluated in both masked-view directions."""
    source_a, source_b = _masked_views(h)
    if np.any(np.linalg.norm(source_a, axis=1) <= SCORE_EPSILON) or np.any(np.linalg.norm(source_b, axis=1) <= SCORE_EPSILON):
        raise SVRNBCRStateError("BCRR masked cross-view degeneracy")
    views = (
        normalize_zid_rows(np.asarray(source_a, np.float32)).astype(np.float64),
        normalize_zid_rows(np.asarray(source_b, np.float32)).astype(np.float64),
    )
    n, c = len(h), len(classes)
    qout: dict[str, np.ndarray] = {}; bout: dict[str, np.ndarray] = {}
    for source_index, destination_index, name in ((0, 1, "0_to_1"), (1, 0, "1_to_0")):
        qscore = np.empty((n, c), np.float64); bscore = np.empty((n, c), np.float64)
        for i in range(n):
            keep = np.arange(n) != i; train, train_y = views[destination_index][keep], indices[keep]
            _, _, decoded_train = _quantize_rows(np.asarray(train, np.float32))
            train = normalize_zid_rows(decoded_train).astype(np.float64)
            source = views[source_index][i]
            y = _centered_y(train_y, c)
            ridge = LAMBDA0 * float(np.sum(train * train) / config.active_k / Z_DIM)
            system = train @ train.T + config.active_k * ridge * np.eye(len(train), dtype=np.float64)
            try:
                weights = train.T @ np.linalg.solve(system, y)
            except np.linalg.LinAlgError as exc:
                raise SVRNBCRStateError("BCRR cross-view LOO solve degeneracy") from exc
            _, _, decoded_weights = _quantize_columns(weights)
            bscore[i] = source @ decoded_weights.astype(np.float64)
            for class_index in range(c):
                class_train = train[train_y == class_index]
                distance = np.maximum(2.0 * (1.0 - np.clip(class_train @ source, -1.0, 1.0)), 0.0)
                # This safety probe is not a replacement deployment bank or
                # neighbor computation; it uses the exact supplied qKNN lock.
                if len(class_train) < 2:
                    hscale = float(config.shared_h0)
                else:
                    pair = np.maximum(2.0 * (1.0 - np.clip(class_train @ class_train.T, -1.0, 1.0)), 0.0)
                    empirical = float(np.mean(pair[np.triu_indices(len(class_train), 1)]))
                    shrunk = (empirical + config.scale_prior_strength * config.shared_h0 ** 2) / (1.0 + config.scale_prior_strength)
                    hscale = float(np.clip(math.sqrt(max(shrunk, SCORE_EPSILON)), config.shared_h0 * config.scale_min_ratio, config.shared_h0 * config.scale_max_ratio))
                kernel = (
                    -config.kernel_volume_gamma * config.kernel_effective_dim * math.log(hscale)
                    -0.5 * (config.student_nu + config.kernel_effective_dim)
                    * np.log1p(distance / (config.student_nu * hscale * hscale))
                )
                maximum = float(np.max(kernel)); qscore[i, class_index] = maximum + math.log(float(np.sum(np.exp(kernel - maximum)))) - math.log(len(kernel))
        qout[name], bout[name] = qscore, bscore
    return qout, bout


def make_bcrr_receipt(
    *, branch: str, support_sha256: str, classes: Sequence[str], indices: np.ndarray,
    h: np.ndarray, qknn_config: Phase1ZIDStudentTLock, active_k: int,
) -> Mapping[str, Any]:
    registry = _registry(classes)
    if active_k == 1:
        qloss = {name: {label: 0.0 for label in registry} for name in ("0_to_1", "1_to_0")}
        floss = {name: dict(values) for name, values in qloss.items()}
        star = omega = 0.0; fallback = "K1_identity"
    else:
        try:
            qscore, bscore = _cross_view_loo_scores(h, indices, registry, qknn_config)
            qnorm = {name: normalize_score_rows(value) for name, value in qscore.items()}
            bnorm = {name: normalize_score_rows(value) for name, value in bscore.items()}
            def losses(weight: float) -> dict[str, dict[str, float]]:
                out = {}
                for name in qnorm:
                    value = (1.0 - weight) * qnorm[name] + weight * bnorm[name]
                    row_loss = _cross_entropy(value, indices)
                    out[name] = {label: float(np.mean(row_loss[indices == i])) for i, label in enumerate(registry)}
                return out
            qloss = losses(0.0)
            def safe(weight: float) -> bool:
                candidate = losses(weight)
                return all(candidate[d][label] <= qloss[d][label] + 1.0e-12 for d in qloss for label in registry)
            if not safe(0.0):
                raise SVRNBCRStateError("BCRR zero residual safety drift")
            # Omega* is the Omega minimizer of mean directional/class CE.  A
            # fixed 24-step bisection jointly enforces safety and minimizes the
            # convex objective; ties keep the smaller value by retaining low.
            def mean_loss_and_derivative(weight: float) -> tuple[float, float]:
                loss_values = []; derivative_values = []
                for name in qnorm:
                    fused = (1.0 - weight) * qnorm[name] + weight * bnorm[name]
                    delta = bnorm[name] - qnorm[name]
                    maximum = np.max(fused, axis=1, keepdims=True)
                    probability = np.exp(fused - maximum); probability /= np.sum(probability, axis=1, keepdims=True)
                    sample_loss = _cross_entropy(fused, indices)
                    sample_derivative = np.sum(probability * delta, axis=1) - delta[np.arange(len(indices)), indices]
                    for class_index in range(len(registry)):
                        mask = indices == class_index
                        loss_values.append(float(np.mean(sample_loss[mask])))
                        derivative_values.append(float(np.mean(sample_derivative[mask])))
                return float(np.mean(loss_values)), float(np.mean(derivative_values))
            low, high = 0.0, BCRR_MAX_OMEGA
            for _ in range(24):
                middle = (low + high) / 2.0
                _, derivative = mean_loss_and_derivative(middle)
                if safe(middle) and derivative < 0.0:
                    low = middle
                else:
                    high = middle
            star = low if mean_loss_and_derivative(low)[0] < mean_loss_and_derivative(0.0)[0] else 0.0
            omega = math.floor(BCRR_DENOMINATOR * star) / BCRR_DENOMINATOR
            floss = losses(omega)
            if not all(floss[d][label] <= qloss[d][label] + 1.0e-10 for d in qloss for label in registry):
                star = omega = 0.0; floss = qloss; fallback = "safety_set_empty"
            else:
                fallback = "none"
        except SVRNBCRStateError:
            qloss = {name: {label: 0.0 for label in registry} for name in ("0_to_1", "1_to_0")}
            floss = {name: dict(values) for name, values in qloss.items()}
            star = omega = 0.0; fallback = "score_normalization_degenerate"
    body = {
        "schema": BCRR_SCHEMA, "branch": branch,
        "branch_support_sha256": support_sha256,
        "normalization": "sqrt(C)*(s-mean(s))/l2(s-mean(s))",
        "fusion": "F=(1-omega)*N(qKNN)+omega*N(BCR)", "loss": "cross_entropy",
        "support_bank_format": "per_row_qint8_fp16_scale_decode_l2_v1",
        "bcr_weight_format": "per_column_qint8_fp16_scale_decode_v1",
        "directional_class_loss_qknn": qloss, "directional_class_loss_bcrr": floss,
        "omega_star": float(star), "omega_q": float(omega),
        "quantization_denominator": BCRR_DENOMINATOR, "fallback": fallback, "K": int(active_k),
        "same_physical_id_synchronous_loo": True,
        "query_rows_used_for_fit": 0,
    }
    receipt = {**body, "receipt_sha256": _digest(body)}
    return verify_bcrr_receipt(receipt, branch=branch, support_sha256=support_sha256)


def _quantize_columns(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(weights, np.float64)
    scales = np.maximum(np.max(np.abs(value), axis=0) / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    codes = np.clip(np.rint(value / scales[None, :]), -127, 127).astype(np.int8)
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[None, :]
    return codes, scales, decoded


def _bcr_quant_audit(teacher: np.ndarray, student: np.ndarray) -> Mapping[str, Any]:
    fp = np.asarray(teacher, np.float64); quant = np.asarray(student, np.float64)
    error = np.max(np.abs(fp - quant), axis=1)
    order = np.argsort(fp, axis=1, kind="stable"); row = np.arange(len(fp))
    top = order[:, -1]; second = order[:, -2]
    margin = fp[row, top] - fp[row, second]
    agree = np.argmax(fp, axis=1) == np.argmax(quant, axis=1)
    large = margin > 2.0 * error
    audit = {
        "scope": "support_only_full_state_teacher",
        "top1_agreement": float(np.mean(agree)),
        "large_margin_flip_count": int(np.sum(large & ~agree)),
        "max_abs_logit_error": float(np.max(np.abs(fp - quant))),
        "teacher_margin_mean": float(np.mean(margin)),
        "query_rows_used_for_fit": 0,
    }
    if audit["top1_agreement"] < MIN_TOP1_AGREEMENT or audit["large_margin_flip_count"] != 0:
        raise SVRNBCRStateError("BCR INT8 teacher gate failed")
    return MappingProxyType(audit)


@dataclass(frozen=True, slots=True)
class SVRNBranchState:
    branch: str
    eta: float
    eta_receipt: Mapping[str, Any] | None
    support_receipt_sha256: str
    branch_support_sha256: str
    qknn_wire: bytes
    support_physical_ids_canonical: tuple[str, ...]
    bcr_weight_codes_qint8: np.ndarray
    bcr_weight_scales_fp16: np.ndarray
    bcr_lambda: float
    bcrr_receipt: Mapping[str, Any]
    quantization_audit: Mapping[str, Any]
    resource: Mapping[str, Any]
    receipt_sha256: str
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA or self.branch not in {"raw", "svrn"}:
            raise SVRNBCRStateError("branch state schema/name drift")
        if float(self.eta) not in ETA_GRID or (self.branch == "raw" and float(self.eta) != 0.0):
            raise SVRNBCRStateError("branch eta drift")
        if self.branch == "raw" and self.eta_receipt is not None:
            raise SVRNBCRStateError("raw branch must not reference SVRN eta receipt")
        if self.branch == "svrn":
            eta = verify_eta_receipt(self.eta_receipt or {})
            if float(eta["selected_eta"]) != float(self.eta):
                raise SVRNBCRStateError("SVRN state/eta receipt drift")
        _sha(self.support_receipt_sha256, "support receipt")
        _sha(self.branch_support_sha256, "branch support SHA")
        if type(self.qknn_wire) is not bytes:
            raise SVRNBCRStateError("qKNN wire must be bytes")
        bank, metric = deserialize_typed_zid_runtime_state(self.qknn_wire)
        if type(bank) is not TypedINT8ZIDSupportBank or type(metric) is not TypedSharedPSDMetric or metric.effective_rank != 0:
            raise SVRNBCRStateError("branch qKNN state must use identity metric")
        if (
            len(self.support_physical_ids_canonical) != len(bank.class_indices_int16)
            or len(set(self.support_physical_ids_canonical)) != len(self.support_physical_ids_canonical)
            or any(type(value) is not str or not value for value in self.support_physical_ids_canonical)
        ):
            raise SVRNBCRStateError("canonical qKNN support physical-ID mapping drift")
        codes = np.asarray(self.bcr_weight_codes_qint8); scales = np.asarray(self.bcr_weight_scales_fp16)
        if (
            codes.dtype != np.int8 or codes.shape != (Z_DIM, len(bank.classes))
            or np.any(codes == np.int8(-128)) or scales.dtype != np.float16
            or scales.shape != (len(bank.classes),) or np.any(scales <= 0)
            or not np.isfinite(scales).all() or not math.isfinite(float(self.bcr_lambda))
            or float(self.bcr_lambda) <= 0.0
        ):
            raise SVRNBCRStateError("BCR INT8 weight state drift")
        verify_bcrr_receipt(self.bcrr_receipt, branch=self.branch, support_sha256=self.branch_support_sha256)
        audit = dict(self.quantization_audit)
        if set(audit) != {"qknn", "bcr"}:
            raise SVRNBCRStateError("branch quantization audit schema drift")
        for value in audit.values():
            if float(value.get("top1_agreement", -1.0)) < MIN_TOP1_AGREEMENT:
                raise SVRNBCRStateError("branch INT8 top1 gate drift")
        if int(audit["qknn"].get("margin_sign_flip_count", -1)) != 0 or int(audit["bcr"].get("large_margin_flip_count", -1)) != 0:
            raise SVRNBCRStateError("branch INT8 margin flip gate drift")
        resource = dict(self.resource)
        required_resource = {
            "feature_dim", "class_count", "K", "trainable_parameters",
            "optimizer_steps", "persistent_fp32_sidecar_bytes",
            "numeric_state_bytes", "build_mac", "query_mac_per_sample",
            "bcr_factorizations", "bcr_loo_full_d3_count", "backend",
        }
        if set(resource) != required_resource:
            raise SVRNBCRStateError("branch resource schema drift")
        if (
            resource["feature_dim"] != Z_DIM or resource["class_count"] != len(bank.classes)
            or resource["K"] != bank.active_k or resource["trainable_parameters"] != 0
            or resource["optimizer_steps"] != 0 or resource["persistent_fp32_sidecar_bytes"] != 0
            or resource["numeric_state_bytes"] > MAX_STATE_BYTES
            or resource["bcr_factorizations"] != 1 or resource["bcr_loo_full_d3_count"] != 0
            or resource["backend"] != {"name": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0}
        ):
            raise SVRNBCRStateError("branch resource contract drift")
        if self.receipt_sha256 != _state_receipt(self):
            raise SVRNBCRStateError("branch state receipt drift")
        object.__setattr__(self, "bcr_weight_codes_qint8", _readonly(codes, np.int8))
        object.__setattr__(self, "bcr_weight_scales_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "bcrr_receipt", MappingProxyType(dict(self.bcrr_receipt)))
        object.__setattr__(self, "quantization_audit", MappingProxyType(audit))
        object.__setattr__(self, "resource", MappingProxyType(resource))


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    out = np.asarray(value, dtype=dtype).copy(); out.setflags(write=False); return out


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _digest(array.tobytes())}


def _state_receipt_fields(
    *, branch: str, eta: float, eta_receipt: Mapping[str, Any] | None,
    support_receipt_sha256: str, branch_support_sha256: str, qknn_wire: bytes,
    support_physical_ids_canonical: Sequence[str],
    codes: np.ndarray, scales: np.ndarray, bcr_lambda: float,
    bcrr_receipt: Mapping[str, Any], quantization_audit: Mapping[str, Any],
    resource: Mapping[str, Any], schema: str = STATE_SCHEMA,
) -> str:
    return _digest(
        {
            "schema": schema, "branch": branch, "eta": float(eta),
            "eta_receipt_sha256": None if eta_receipt is None else eta_receipt["receipt_sha256"],
            "support_receipt_sha256": support_receipt_sha256,
            "branch_support_sha256": branch_support_sha256,
            "qknn_wire_sha256": _digest(qknn_wire),
            "support_physical_ids_canonical": list(support_physical_ids_canonical),
            "bcr_weight_codes_qint8": _array_receipt(codes),
            "bcr_weight_scales_fp16": _array_receipt(scales),
            "bcr_lambda": float(bcr_lambda),
            "bcrr_receipt_sha256": bcrr_receipt["receipt_sha256"],
            "quantization_audit": quantization_audit,
            "resource": resource,
        }
    )


def _state_receipt(state: SVRNBranchState) -> str:
    return _state_receipt_fields(
        branch=state.branch, eta=state.eta, eta_receipt=state.eta_receipt,
        support_receipt_sha256=state.support_receipt_sha256,
        branch_support_sha256=state.branch_support_sha256,
        qknn_wire=state.qknn_wire, codes=state.bcr_weight_codes_qint8,
        support_physical_ids_canonical=state.support_physical_ids_canonical,
        scales=state.bcr_weight_scales_fp16, bcr_lambda=state.bcr_lambda,
        bcrr_receipt=state.bcrr_receipt, quantization_audit=state.quantization_audit,
        resource=state.resource, schema=state.schema,
    )


def build_branch_state(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    support_physical_ids: Sequence[str],
    *,
    qknn_config: Phase1ZIDStudentTLock,
    branch: str,
    eta_receipt: Mapping[str, Any] | None = None,
) -> SVRNBranchState:
    rows = _finite_rows(support_zid, "branch support")
    if type(qknn_config) is not Phase1ZIDStudentTLock:
        raise SVRNBCRStateError("branch requires exact qKNN lock")
    classes, indices = _balanced_labels(support_labels, registered_classes, qknn_config.active_k)
    ids = tuple(str(value) for value in support_physical_ids)
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise SVRNBCRStateError("branch support physical ID drift")
    if branch == "raw":
        if eta_receipt is not None: raise SVRNBCRStateError("raw branch cannot read SVRN eta")
        eta = 0.0
    elif branch == "svrn":
        eta_lock = verify_eta_receipt(eta_receipt or {})
        if eta_lock["support_physical_ids_sha256"] != _digest(list(ids)):
            raise SVRNBCRStateError("SVRN eta/support ID binding drift")
        eta = float(eta_lock["selected_eta"])
    else:
        raise SVRNBCRStateError("unsupported branch")
    transformed = svrn_transform(rows, eta)
    h = normalize_zid_rows(transformed).astype(np.float64)
    branch_sha = _digest(np.ascontiguousarray(h).tobytes())
    support_receipt = _digest({"support_physical_ids": list(ids), "registered": list(classes)})
    bank = build_typed_zid_support_bank(transformed, support_labels, classes, config=qknn_config)
    normalized = normalize_zid_rows(transformed)
    support_codes, support_scales, _ = _quantize_rows(normalized)
    canonical_order = _canonical_order(support_codes, support_scales, indices)
    canonical_ids = tuple(ids[int(index)] for index in canonical_order)
    metric = identity_shared_psd_metric(config=qknn_config)
    qwire = serialize_typed_zid_runtime_state(bank, metric)
    # The deployed INT8 bank is decoded once.  One dual factorization of that
    # immutable bank yields both production BCR weights and every LOO row; no
    # per-support d^3 fit exists.
    decoded = decode_zid_support_bank(bank).astype(np.float64)
    deploy_weights, bcr_loo, deploy_lambda, ledger = _ridge_fit_and_loo(
        decoded, bank.class_indices_int16, qknn_config.active_k,
    )
    bcrr = make_bcrr_receipt(
        branch=branch, support_sha256=branch_sha, classes=classes,
        indices=bank.class_indices_int16, h=h, qknn_config=qknn_config, active_k=qknn_config.active_k,
    )
    codes, scales, decoded_weights = _quantize_columns(deploy_weights)
    teacher_logits = h @ deploy_weights
    student_logits = h @ decoded_weights.astype(np.float64)
    bcr_audit = _bcr_quant_audit(teacher_logits, student_logits)
    qknn_audit = audit_int8_margin(bank, transformed, support_labels, transformed, metric=metric)
    if qknn_audit["top1_agreement"] < MIN_TOP1_AGREEMENT or qknn_audit["margin_sign_flip_count"] != 0:
        raise SVRNBCRStateError("qKNN INT8 teacher gate failed")
    numeric_bytes = len(qwire) + codes.nbytes + scales.nbytes
    build_mac = int(sum(ledger[key] for key in ledger if key.endswith("_mac"))) + len(h) * len(h) * Z_DIM
    query_mac = len(h) * Z_DIM + Z_DIM * len(classes)
    resource = {
        "feature_dim": Z_DIM, "class_count": len(classes), "K": qknn_config.active_k,
        "trainable_parameters": 0, "optimizer_steps": 0,
        "persistent_fp32_sidecar_bytes": 0, "numeric_state_bytes": int(numeric_bytes),
        "build_mac": int(build_mac), "query_mac_per_sample": int(query_mac),
        "bcr_factorizations": ledger["bcr_factorizations"],
        "bcr_loo_full_d3_count": ledger["bcr_loo_full_d3_count"],
        "backend": {"name": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0},
    }
    eta_value = None if eta_receipt is None else dict(eta_receipt)
    quant_value = {"qknn": qknn_audit, "bcr": dict(bcr_audit)}
    receipt = _state_receipt_fields(
        branch=branch, eta=eta, eta_receipt=eta_value,
        support_receipt_sha256=support_receipt, branch_support_sha256=branch_sha,
        qknn_wire=qwire, support_physical_ids_canonical=canonical_ids,
        codes=codes, scales=scales, bcr_lambda=float(deploy_lambda),
        bcrr_receipt=bcrr, quantization_audit=quant_value, resource=resource,
    )
    return SVRNBranchState(
        branch, eta, eta_value, support_receipt, branch_sha, qwire, canonical_ids, codes, scales,
        float(deploy_lambda), bcrr, quant_value, resource, receipt_sha256=receipt,
    )


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "b64": base64.b64encode(array.tobytes()).decode("ascii")}


def _decode_array(value: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "b64"}:
        raise SVRNBCRStateError("wire array schema drift")
    try:
        dtype = np.dtype(value["dtype"]); shape = tuple(int(x) for x in value["shape"])
        raw = base64.b64decode(value["b64"], validate=True)
        out = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except Exception as exc:
        raise SVRNBCRStateError("wire array decode failed") from exc
    return out


def serialize_branch_state(state: SVRNBranchState) -> bytes:
    if type(state) is not SVRNBranchState:
        raise SVRNBCRStateError("serialization requires exact branch state")
    state.__post_init__()
    payload = {
        "schema": WIRE_SCHEMA, "state_schema": state.schema,
        "branch": state.branch, "eta": float(state.eta),
        "eta_receipt": None if state.eta_receipt is None else dict(state.eta_receipt),
        "support_receipt_sha256": state.support_receipt_sha256,
        "branch_support_sha256": state.branch_support_sha256,
        "qknn_wire_b64": base64.b64encode(state.qknn_wire).decode("ascii"),
        "qknn_wire_sha256": _digest(state.qknn_wire),
        "support_physical_ids_canonical": list(state.support_physical_ids_canonical),
        "bcr_weight_codes_qint8": _encode_array(state.bcr_weight_codes_qint8),
        "bcr_weight_scales_fp16": _encode_array(state.bcr_weight_scales_fp16),
        "bcr_lambda": float(state.bcr_lambda), "bcrr_receipt": dict(state.bcrr_receipt),
        "quantization_audit": dict(state.quantization_audit),
        "resource": dict(state.resource), "state_receipt_sha256": state.receipt_sha256,
        "persistent_array_allowlist": ["qknn_int8_bank", "bcr_weight_codes_qint8", "bcr_weight_scales_fp16"],
        "persistent_fp32_sidecar_bytes": 0,
    }
    wire = _canon(payload)
    if len(wire) > MAX_STATE_BYTES:
        raise SVRNBCRStateError("serialized branch state exceeds 256KB")
    return wire


def deserialize_branch_state(wire: bytes) -> SVRNBranchState:
    if type(wire) is not bytes or len(wire) > MAX_STATE_BYTES:
        raise SVRNBCRStateError("branch wire size/type drift")
    try:
        value = json.loads(wire.decode("ascii"))
    except Exception as exc:
        raise SVRNBCRStateError("branch wire JSON decode failed") from exc
    required = {
        "schema", "state_schema", "branch", "eta", "eta_receipt",
        "support_receipt_sha256", "branch_support_sha256", "qknn_wire_b64",
        "qknn_wire_sha256", "support_physical_ids_canonical", "bcr_weight_codes_qint8", "bcr_weight_scales_fp16",
        "bcr_lambda", "bcrr_receipt", "quantization_audit", "resource",
        "state_receipt_sha256", "persistent_array_allowlist",
        "persistent_fp32_sidecar_bytes",
    }
    if type(value) is not dict or set(value) != required:
        raise SVRNBCRStateError("branch wire schema drift")
    if (
        value["schema"] != WIRE_SCHEMA or value["state_schema"] != STATE_SCHEMA
        or value["persistent_array_allowlist"] != ["qknn_int8_bank", "bcr_weight_codes_qint8", "bcr_weight_scales_fp16"]
        or value["persistent_fp32_sidecar_bytes"] != 0
    ):
        raise SVRNBCRStateError("branch wire FP32/allowlist drift")
    try:
        qwire = base64.b64decode(value["qknn_wire_b64"], validate=True)
    except Exception as exc:
        raise SVRNBCRStateError("qKNN wire base64 drift") from exc
    if _digest(qwire) != value["qknn_wire_sha256"]:
        raise SVRNBCRStateError("qKNN wire SHA drift")
    return SVRNBranchState(
        branch=value["branch"], eta=float(value["eta"]), eta_receipt=value["eta_receipt"],
        support_receipt_sha256=value["support_receipt_sha256"],
        branch_support_sha256=value["branch_support_sha256"], qknn_wire=qwire,
        support_physical_ids_canonical=tuple(value["support_physical_ids_canonical"]),
        bcr_weight_codes_qint8=_decode_array(value["bcr_weight_codes_qint8"]),
        bcr_weight_scales_fp16=_decode_array(value["bcr_weight_scales_fp16"]),
        bcr_lambda=float(value["bcr_lambda"]), bcrr_receipt=value["bcrr_receipt"],
        quantization_audit=value["quantization_audit"], resource=value["resource"],
        receipt_sha256=value["state_receipt_sha256"],
    )


def qknn_neighbor_receipt(state: SVRNBranchState, query_zid: np.ndarray) -> Mapping[str, Any]:
    """Truth-free per-query/per-class qKNN neighbor order over the sealed bank."""
    if type(state) is not SVRNBranchState:
        raise SVRNBCRStateError("neighbor receipt requires exact branch state")
    state.__post_init__()
    query = svrn_transform(_finite_rows(query_zid, "query z_id"), float(state.eta))
    bank, metric = deserialize_typed_zid_runtime_state(state.qknn_wire)
    support = decode_zid_support_bank(bank).astype(np.float64)
    query_norm = normalize_zid_rows(query).astype(np.float64)
    orders: list[list[list[str]]] = []
    for row in query_norm:
        per_class: list[list[str]] = []
        for class_index in range(len(bank.classes)):
            positions = np.flatnonzero(bank.class_indices_int16 == class_index)
            hscale = float(bank.class_scales_fp16[class_index])
            distance = np.maximum(2.0 * (1.0 - np.clip(support[positions] @ row, -1.0, 1.0)), 0.0)
            kernel = (
                -bank.config.kernel_volume_gamma * bank.config.kernel_effective_dim * math.log(hscale)
                -0.5 * (bank.config.student_nu + bank.config.kernel_effective_dim)
                * np.log1p(distance / (bank.config.student_nu * hscale * hscale))
            )
            ranked = positions[np.argsort(-kernel, kind="stable")]
            per_class.append([state.support_physical_ids_canonical[int(index)] for index in ranked])
        orders.append(per_class)
    body = {
        "schema": "cvs.stage2.svrn_bcr.qknn_neighbor_receipt.v1",
        "branch": state.branch, "qknn_state_receipt_sha256": state.receipt_sha256,
        "qknn_bank_receipt_sha256": bank.bank_receipt_sha256,
        "canonical_support_physical_ids_sha256": _digest(list(state.support_physical_ids_canonical)),
        "classes": list(bank.classes), "query_count": len(orders), "orders": orders,
        "query_rows_used_for_fit": 0,
    }
    return MappingProxyType({**body, "receipt_sha256": _digest(body)})


def score_branch_logits(state: SVRNBranchState, query_zid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable qKNN logits and branch-local continuous BCRR logits."""

    if type(state) is not SVRNBranchState:
        raise SVRNBCRStateError("scoring requires exact branch state")
    state.__post_init__()
    query = svrn_transform(_finite_rows(query_zid, "query z_id"), float(state.eta))
    bank, metric = deserialize_typed_zid_runtime_state(state.qknn_wire)
    qknn = score_zid_student_t_logits(bank, query, metric=metric)
    h = normalize_zid_rows(query).astype(np.float32)
    weights = state.bcr_weight_codes_qint8.astype(np.float32) * state.bcr_weight_scales_fp16.astype(np.float32)[None, :]
    bcr = np.asarray(h @ weights, np.float32)
    if bcr.shape != qknn.shape or not np.isfinite(bcr).all():
        raise SVRNBCRStateError("BCR query logits drift")
    fused = np.array(qknn, dtype=np.float32, copy=True)
    omega = float(state.bcrr_receipt["omega_q"])
    for row in range(len(qknn)):
        try:
            nq = normalize_score_rows(qknn[row:row + 1])[0]
            nb = normalize_score_rows(bcr[row:row + 1])[0]
            fused[row] = np.asarray((1.0 - omega) * nq + omega * nb, np.float32)
        except SVRNBCRStateError:
            # A score-normalization degeneracy is query-local: keep the
            # untouched qKNN result for that query rather than changing state.
            fused[row] = qknn[row]
    return qknn, fused


__all__ = [
    "BCRR_DENOMINATOR", "BCRR_MAX_OMEGA", "BCRR_SCHEMA", "ETA_GRID", "KAPPA", "LAMBDA0", "MASK_MODULUS",
    "MASK_RESIDUES", "MASK_RETENTION", "MAX_STATE_BYTES", "SVRNBranchState",
    "SVRNBCRStateError", "build_branch_state", "deserialize_branch_state",
    "make_bcrr_receipt", "normalize_score_rows", "qknn_neighbor_receipt", "score_branch_logits",
    "select_svrn_eta", "serialize_branch_state", "svrn_transform",
    "verify_bcrr_receipt", "verify_eta_receipt",
]
