"""Old-class-only SF-TAPFT plus ERBT-IDR diagnostic helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_diag_cosine_exploration import spectral_logmag_sketch


ARM = "M29-FFT96-A4"
_EPS = 1.0e-12
_GEOMETRY_LOCK = threading.RLock()
_SUPPORT_REQUIRED = frozenset(
    {
        "support_pool_leo_weak_iq",
        "support_pool_class_indices",
        "support_pool_rank_within_class",
        "support_pool_tokens",
    }
)
_SUPPORT_ALLOWED = _SUPPORT_REQUIRED | frozenset(
    {
        "support_pool_overlay_tokens",
        "support_pool_satellite_seeds",
        "support_pool_post_channel_iq_sha256",
        "manifest_json",
    }
)


class OldOnlyERBTError(RuntimeError):
    """Raised when the old-only diagnostic contract is violated."""


def _unit(rows: Any) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise OldOnlyERBTError("features must be finite nonempty matrices")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise OldOnlyERBTError("feature row is degenerate")
    return values / norms


def _unit_or_zero(rows: Any) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise OldOnlyERBTError("features must be finite nonempty matrices")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    result = np.zeros_like(values)
    np.divide(values, norms, out=result, where=norms > _EPS)
    return result


def make_fft96(received_iq: Any) -> np.ndarray:
    rows = np.asarray(received_iq, dtype=np.float32)
    if rows.ndim != 3 or rows.shape[1:] != (2, 256) or not np.isfinite(rows).all():
        raise OldOnlyERBTError("received IQ must be finite N x 2 x 256")
    result = np.asarray(spectral_logmag_sketch(rows), dtype=np.float32)
    if result.shape != (len(rows), 96) or not np.isfinite(result).all():
        raise OldOnlyERBTError("FFT96 geometry drift")
    return result


def _features(identity160: Any, fft96: Any) -> np.ndarray:
    identity = np.asarray(identity160, dtype=np.float64)
    fft = np.asarray(fft96, dtype=np.float64)
    if identity.ndim != 2 or identity.shape[1] != 160:
        raise OldOnlyERBTError("identity feature geometry drift")
    if fft.ndim != 2 or fft.shape != (len(identity), 96):
        raise OldOnlyERBTError("FFT feature geometry drift")
    joined = np.concatenate(
        [_unit_or_zero(identity), 4.0 * _unit_or_zero(fft)], axis=1
    )
    return np.asarray(_unit(joined), dtype=np.float32)


@contextmanager
def _d92_geometry() -> Iterator[None]:
    with _GEOMETRY_LOCK:
        original = (d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS)
        d42.FEATURE_DIM = 256
        d42.BLOCK_SLICES = (slice(0, 160), slice(160, 256))
        d42.BLOCK_DIMS = (160, 96)
        try:
            yield
        finally:
            d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS = original


@dataclass(frozen=True)
class OldOnlyERBTState:
    class_ids: tuple[int, ...]
    log_diag: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def score(self, identity160: Any, fft96: Any) -> np.ndarray:
        features = _features(identity160, fft96)
        with _d92_geometry():
            transformed = d42._transform(features, self.log_diag)
        return transformed @ self.coefficient.T + self.intercept

    def predict(self, identity160: Any, fft96: Any) -> np.ndarray:
        indices = np.argmax(self.score(identity160, fft96), axis=1)
        return np.asarray(self.class_ids, dtype=np.int64)[indices]


def fit_old_only_erbt(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    seed: int,
    device: Any = "cpu",
) -> OldOnlyERBTState:
    from cvsrffi import stage2_ablation_executors as executors

    registry = tuple(int(value) for value in class_ids)
    if len(registry) != 6 or len(set(registry)) != 6:
        raise OldOnlyERBTError("REG0 requires exactly six old classes")
    label_rows = np.asarray(labels, dtype=np.int64)
    lookup = {value: index for index, value in enumerate(registry)}
    if set(label_rows.tolist()) != set(registry):
        raise OldOnlyERBTError("support registry drift")
    targets = np.asarray([lookup[int(value)] for value in label_rows], dtype=np.int64)
    counts = np.bincount(targets, minlength=6)
    if np.any(counts <= 0) or len(set(counts.tolist())) != 1:
        raise OldOnlyERBTError("support must be balanced K-shot")
    features = _features(identity160, fft96)
    if len(features) != len(targets):
        raise OldOnlyERBTError("support feature/label row drift")
    with _d92_geometry():
        log_diag, trace, _ = executors._metric(
            features,
            targets,
            6,
            enabled=True,
            seed=int(seed),
            device=device,
        )
        transformed = d42._transform(features, log_diag)
        fit, method = executors._component_builder(
            "P2-B0",
            ground_basis=np.empty((160, 0), dtype=np.float64),
            ground_weights=np.empty(0, dtype=np.float64),
            ground_audit={},
        )
        coefficient, intercept, fit_audit = executors._fit_with_fp32_centering_audit(
            fit,
            transformed,
            targets,
            6,
            int(counts[0]),
        )
    audit = {
        **dict(fit_audit),
        "arm": ARM,
        "method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "registration_state": "REG0",
        "d92_registration_balanced_active": False,
        "d92_status": "before_exact_d81",
        "support_only": True,
        "query_rows_used": 0,
        "numerical_method": method,
        "optimizer_steps": len(trace),
    }
    return OldOnlyERBTState(
        class_ids=registry,
        log_diag=np.asarray(log_diag, dtype=np.float32),
        coefficient=np.asarray(coefficient, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        audit=audit,
    )


def export_old_only_holdout(
    source: str | Path,
    output_root: str | Path,
    *,
    k_shot: int,
    expected_support_iq_sha256: str,
    capsule_id: str,
    split_id: str,
    adaptation_capsule_id: str,
    adaptation_split_id: str,
) -> dict[str, Any]:
    source_path = Path(source)
    destination = Path(output_root)
    if destination.exists():
        raise OldOnlyERBTError(f"output root already exists: {destination}")
    if int(k_shot) != 10:
        raise OldOnlyERBTError("this experiment is locked to K10")
    if not all(
        value.strip()
        for value in (capsule_id, split_id, adaptation_capsule_id, adaptation_split_id)
    ):
        raise OldOnlyERBTError("invalid data binding")
    try:
        with np.load(source_path, allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if not _SUPPORT_REQUIRED <= keys or not keys <= _SUPPORT_ALLOWED:
                raise OldOnlyERBTError("support pool allowlist mismatch")
            iq = np.asarray(payload["support_pool_leo_weak_iq"], dtype=np.float32)
            labels = np.asarray(payload["support_pool_class_indices"], dtype=np.int64)
            ranks = np.asarray(payload["support_pool_rank_within_class"], dtype=np.int64)
            tokens = np.asarray(payload["support_pool_tokens"]).astype(str)
    except (OSError, ValueError) as exc:
        raise OldOnlyERBTError(f"cannot load support pool: {source_path}") from exc
    if iq.ndim != 3 or iq.shape[1:] != (2, 256) or labels.shape != ranks.shape or labels.shape != tokens.shape:
        raise OldOnlyERBTError("support pool row geometry drift")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(6, dtype=np.int64)):
        raise OldOnlyERBTError("support pool old-class registry drift")
    support_mask = ranks < int(k_shot)
    query_mask = (ranks >= int(k_shot)) & (ranks < 2 * int(k_shot))
    for mask, name in ((support_mask, "support"), (query_mask, "holdout")):
        counts = np.bincount(labels[mask], minlength=6)
        if counts.tolist() != [int(k_shot)] * 6:
            raise OldOnlyERBTError(f"{name} is not balanced K-shot")
    support_iq = np.ascontiguousarray(iq[support_mask])
    support_digest = hashlib.sha256(support_iq.tobytes(order="C")).hexdigest()
    if support_digest != str(expected_support_iq_sha256).lower():
        raise OldOnlyERBTError("support IQ binding mismatch")
    support_ids = tokens[support_mask]
    query_ids = tokens[query_mask]
    overlap = set(support_ids.tolist()) & set(query_ids.tolist())
    if overlap:
        raise OldOnlyERBTError("support and holdout physical IDs overlap")
    if len(set(support_ids.tolist())) != len(support_ids) or len(set(query_ids.tolist())) != len(query_ids):
        raise OldOnlyERBTError("physical IDs are not unique")
    receipt = {
        "schema": "cvs.sf_erbt_oldonly_export.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "adaptation_capsule_id": adaptation_capsule_id,
        "adaptation_split_id": adaptation_split_id,
        "k_shot": int(k_shot),
        "class_count": 6,
        "support_rows": int(support_mask.sum()),
        "query_rows": int(query_mask.sum()),
        "support_iq_sha256": support_digest,
        "support_query_physical_id_overlap": 0,
        "query_truth_in_predictor": False,
        "registration_state": "REG0",
    }
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(
        destination / "support.npz",
        received_iq=support_iq,
        support_labels=labels[support_mask],
        support_physical_ids=support_ids,
    )
    np.savez(
        destination / "query.npz",
        received_iq=np.ascontiguousarray(iq[query_mask]),
        query_ids=query_ids,
    )
    np.savez(
        destination / "truth.npz",
        query_ids=query_ids,
        query_labels=labels[query_mask],
    )
    with (destination / "data_handle.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def _load_exact_npz(path: Path, expected: frozenset[str], label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if frozenset(payload.files) != expected:
                raise OldOnlyERBTError(f"{label} allowlist mismatch")
            return {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError) as exc:
        raise OldOnlyERBTError(f"cannot load {label}: {path}") from exc


def _extract_identity160(model: torch.nn.Module, rows: np.ndarray, device: torch.device) -> np.ndarray:
    from cvsrffi.target_only_progressive_adapt import _extract_joint_embedding, _forward_aux

    contiguous = np.ascontiguousarray(rows, dtype=np.float32)
    values = torch.tensor(contiguous.tolist(), dtype=torch.float32, device=device)
    with torch.inference_mode():
        embeddings = _extract_joint_embedding(_forward_aux(model, values), len(contiguous))
    result = np.asarray(embeddings.detach().cpu().tolist(), dtype=np.float32)
    if result.shape != (len(contiguous), 160) or not np.isfinite(result).all():
        raise OldOnlyERBTError("SF identity160 output geometry drift")
    return result


def run_old_only_prediction(
    *,
    bundle_path: str | Path,
    support_path: str | Path,
    query_path: str | Path,
    data_handle_path: str | Path,
    output_root: str | Path,
    seed: int,
    device: str | torch.device,
    bundle_loader: Any | None = None,
) -> dict[str, Any]:
    query = _load_exact_npz(
        Path(query_path), frozenset({"received_iq", "query_ids"}), "query"
    )
    support = _load_exact_npz(
        Path(support_path),
        frozenset({"received_iq", "support_labels", "support_physical_ids"}),
        "support",
    )
    try:
        with Path(data_handle_path).open("r", encoding="utf-8") as handle:
            data_handle = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise OldOnlyERBTError("cannot load data handle") from exc
    required_handle = {
        "schema": "cvs.sf_erbt_oldonly_export.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "registration_state": "REG0",
        "query_truth_in_predictor": False,
        "class_count": 6,
    }
    if not isinstance(data_handle, Mapping) or any(
        data_handle.get(key) != value for key, value in required_handle.items()
    ):
        raise OldOnlyERBTError("data handle binding mismatch")
    if int(data_handle.get("k_shot", -1)) != 10:
        raise OldOnlyERBTError("this experiment is locked to K10")
    support_iq = np.asarray(support["received_iq"], dtype=np.float32)
    support_labels = np.asarray(support["support_labels"], dtype=np.int64)
    support_ids = np.asarray(support["support_physical_ids"]).astype(str)
    query_iq = np.asarray(query["received_iq"], dtype=np.float32)
    query_ids = np.asarray(query["query_ids"]).astype(str)
    expected_support_rows = int(data_handle["support_rows"])
    expected_query_rows = int(data_handle["query_rows"])
    if len(support_iq) != expected_support_rows or len(query_iq) != expected_query_rows:
        raise OldOnlyERBTError("data handle row count mismatch")
    if support_labels.shape != (len(support_iq),) or support_ids.shape != (len(support_iq),):
        raise OldOnlyERBTError("support row alignment drift")
    if query_ids.shape != (len(query_iq),):
        raise OldOnlyERBTError("query row alignment drift")
    if set(support_ids.tolist()) & set(query_ids.tolist()):
        raise OldOnlyERBTError("support/query physical IDs overlap")
    counts = np.bincount(support_labels, minlength=6)
    if counts.tolist() != [int(data_handle["k_shot"])] * 6:
        raise OldOnlyERBTError("support is not the bound balanced K-shot")
    destination = Path(output_root)
    if destination.exists():
        raise OldOnlyERBTError(f"output root already exists: {destination}")
    target_device = torch.device(device)
    if bundle_loader is None:
        from cvsrffi.target_only_progressive_runner import load_sf_tapft_bundle_strict

        bundle_loader = load_sf_tapft_bundle_strict
    model, head, bundle_audit = bundle_loader(bundle_path, device=target_device)
    if (
        bundle_audit.get("capsule_id") != data_handle.get("adaptation_capsule_id")
        or bundle_audit.get("split_id") != data_handle.get("adaptation_split_id")
    ):
        raise OldOnlyERBTError("SF bundle/data binding mismatch")
    if tuple(getattr(head, "class_ids", ())) != tuple(range(6)):
        raise OldOnlyERBTError("SF head old-class registry mismatch")
    support_identity = _extract_identity160(model, support_iq, target_device)
    query_identity = _extract_identity160(model, query_iq, target_device)
    with torch.inference_mode():
        query_tensor = torch.tensor(
            np.ascontiguousarray(query_identity).tolist(),
            dtype=torch.float32,
            device=target_device,
        )
        head_columns = np.asarray(
            torch.argmax(head(query_tensor), dim=1).cpu().tolist(), dtype=np.int64
        )
        sf_head_predictions = np.asarray(head.class_ids, dtype=np.int64)[head_columns]
    support_fft = make_fft96(support_iq)
    query_fft = make_fft96(query_iq)
    state = fit_old_only_erbt(
        support_identity,
        support_fft,
        support_labels,
        class_ids=tuple(range(6)),
        seed=int(seed),
        device=target_device,
    )
    sf_erbt_predictions = state.predict(query_identity, query_fft)
    receipt = {
        "schema": "cvs.sf_erbt_oldonly_prediction.v1",
        "method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "arm": ARM,
        "registration_state": "REG0",
        "capsule_id": data_handle["capsule_id"],
        "split_id": data_handle["split_id"],
        "k_shot": int(data_handle["k_shot"]),
        "support_rows": len(support_iq),
        "query_rows": len(query_iq),
        "sf_bundle_schema": bundle_audit.get("schema"),
        "erbt_d92_registration_balanced_active": False,
        "erbt_d92_status": "before_exact_d81",
        "source_opened": False,
        "query_truth_opened": False,
        "query_role_opened": False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(
        destination / "predictions.npz",
        query_ids=query_ids,
        sf_head_predictions=sf_head_predictions,
        sf_erbt_predictions=sf_erbt_predictions,
    )
    with (destination / "prediction_receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def _arm_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    per_class = []
    for class_id in range(6):
        mask = truth == class_id
        if not np.any(mask):
            raise OldOnlyERBTError("truth is missing an old class")
        per_class.append(float(np.mean(prediction[mask] == truth[mask])))
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(np.mean(per_class)),
        "class_floor": float(np.min(per_class)),
        "per_class_accuracy": per_class,
    }


def score_old_only_predictions(
    prediction_path: str | Path,
    truth_path: str | Path,
    prediction_receipt_path: str | Path,
    data_handle_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    predictions = _load_exact_npz(
        Path(prediction_path),
        frozenset({"query_ids", "sf_head_predictions", "sf_erbt_predictions"}),
        "predictions",
    )
    truth_payload = _load_exact_npz(
        Path(truth_path), frozenset({"query_ids", "query_labels"}), "truth"
    )
    try:
        with Path(prediction_receipt_path).open("r", encoding="utf-8") as handle:
            prediction_receipt = json.load(handle)
        with Path(data_handle_path).open("r", encoding="utf-8") as handle:
            data_handle = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise OldOnlyERBTError("cannot load scoring bindings") from exc
    if prediction_receipt.get("schema") != "cvs.sf_erbt_oldonly_prediction.v1" or data_handle.get("schema") != "cvs.sf_erbt_oldonly_export.v1":
        raise OldOnlyERBTError("scoring binding schema mismatch")
    binding_keys = ("capsule_id", "split_id", "k_shot", "support_rows", "query_rows", "registration_state")
    if any(prediction_receipt.get(key) != data_handle.get(key) for key in binding_keys):
        raise OldOnlyERBTError("prediction/data scoring binding mismatch")
    if int(data_handle.get("k_shot", -1)) != 10 or int(data_handle.get("query_rows", -1)) != 60:
        raise OldOnlyERBTError("scorer is locked to K10 x six old classes")
    prediction_ids = np.asarray(predictions["query_ids"]).astype(str)
    truth_ids = np.asarray(truth_payload["query_ids"]).astype(str)
    if len(set(prediction_ids.tolist())) != len(prediction_ids):
        raise OldOnlyERBTError("prediction query IDs are not unique")
    if set(prediction_ids.tolist()) != set(truth_ids.tolist()):
        raise OldOnlyERBTError("prediction/truth query ID mismatch")
    lookup = {value: index for index, value in enumerate(truth_ids.tolist())}
    order = np.asarray([lookup[value] for value in prediction_ids.tolist()], dtype=np.int64)
    truth = np.asarray(truth_payload["query_labels"], dtype=np.int64)[order]
    head_prediction = np.asarray(predictions["sf_head_predictions"], dtype=np.int64)
    erbt_prediction = np.asarray(predictions["sf_erbt_predictions"], dtype=np.int64)
    if truth.shape != head_prediction.shape or truth.shape != erbt_prediction.shape:
        raise OldOnlyERBTError("prediction/truth row alignment drift")
    if np.bincount(truth, minlength=6).tolist() != [10] * 6 or not np.array_equal(np.unique(truth), np.arange(6)):
        raise OldOnlyERBTError("truth is not K10 x six old classes")
    head_metrics = _arm_metrics(head_prediction, truth)
    erbt_metrics = _arm_metrics(erbt_prediction, truth)
    result = {
        "schema": "cvs.sf_erbt_oldonly_score.v1",
        "method_lock": "D92-E0-NORF32",
        "registration_state": "REG0",
        "new_class_metrics": "N/A",
        "sf_head": head_metrics,
        "sf_erbt": erbt_metrics,
        "sf_erbt_minus_sf_head": {
            key: float(erbt_metrics[key] - head_metrics[key])
            for key in ("accuracy", "balanced_accuracy", "class_floor")
        },
    }
    destination = Path(output_path)
    if destination.exists():
        raise OldOnlyERBTError(f"score output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


__all__ = [
    "ARM",
    "OldOnlyERBTError",
    "OldOnlyERBTState",
    "export_old_only_holdout",
    "fit_old_only_erbt",
    "make_fft96",
    "run_old_only_prediction",
    "score_old_only_predictions",
]
