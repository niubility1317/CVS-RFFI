"""Truth-blind query closure for SF-TAPFT full-state or compact-delta bundles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .sf_tapft_prediction import predict_sf_tapft_rows
from .target_only_progressive_adapt import (
    SFTAPFTConfig,
    TargetPrototypeHead,
    _extract_joint_embedding,
    _forward_aux,
    _source_classifier_weight,
    _target_prototypes,
    ensure_time_adapter,
)
from .target_only_progressive_runner import (
    SF_TAPFT_DELTA_BUNDLE_SCHEMA,
    _default_checkpoint_loader,
    _load_target_support,
    _normalize_sf_tapft_bundle_config,
    load_sf_tapft_clean_single_bundle_strict,
    load_sf_tapft_delta_bundle_strict,
)


PREDICTION_MEMBERS = frozenset({"query_ids", "predicted_class_ids", "scores"})
STATES = ("DA0_REG0", "DA1_REG0")


class QueryClosureError(RuntimeError):
    """Raised when a query-closure boundary or binding is invalid."""


def export_independent_query_inputs(
    *,
    support_source_path: str | Path,
    query_source_path: str | Path,
    package_manifest_path: str | Path,
    output_root: str | Path,
    expected_support_iq_sha256: str,
    capsule_id: str,
    split_id: str,
    adaptation_capsule_id: str,
    adaptation_split_id: str,
) -> dict[str, Any]:
    """Export K10 support plus the package's full independent unlabeled query."""

    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise QueryClosureError(f"output root already exists: {destination}")
    try:
        with np.load(Path(support_source_path), allow_pickle=False) as payload:
            required = {
                "support_pool_leo_weak_iq",
                "support_pool_class_indices",
                "support_pool_rank_within_class",
                "support_pool_tokens",
            }
            if not required <= set(payload.files):
                raise QueryClosureError("support pool allowlist mismatch")
            support_pool_iq = np.asarray(
                payload["support_pool_leo_weak_iq"], dtype=np.float32
            )
            support_pool_labels = np.asarray(
                payload["support_pool_class_indices"], dtype=np.int64
            )
            support_pool_ranks = np.asarray(
                payload["support_pool_rank_within_class"], dtype=np.int64
            )
            support_pool_ids = np.asarray(payload["support_pool_tokens"]).astype(str)
        with np.load(Path(query_source_path), allow_pickle=False) as payload:
            required = {"query_leo_weak_iq", "query_tokens"}
            if not required <= set(payload.files):
                raise QueryClosureError("independent query allowlist mismatch")
            query_iq = np.asarray(payload["query_leo_weak_iq"], dtype=np.float32)
            query_ids = np.asarray(payload["query_tokens"]).astype(str)
        package_manifest = json.loads(
            Path(package_manifest_path).read_text(encoding="utf-8-sig")
        )
    except QueryClosureError:
        raise
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryClosureError("cannot load independent-query source assets") from exc

    if (
        support_pool_iq.ndim != 3
        or support_pool_iq.shape[1:] != (2, 256)
        or support_pool_labels.shape != support_pool_ranks.shape
        or support_pool_labels.shape != support_pool_ids.shape
        or not np.isfinite(support_pool_iq).all()
    ):
        raise QueryClosureError("support pool row geometry drift")
    support_mask = support_pool_ranks < 10
    if np.bincount(support_pool_labels[support_mask], minlength=6).tolist() != [10] * 6:
        raise QueryClosureError("support pool is not old6 K10")
    support_iq = np.ascontiguousarray(support_pool_iq[support_mask])
    support_labels = support_pool_labels[support_mask]
    support_ids = support_pool_ids[support_mask]
    support_digest = hashlib.sha256(support_iq.tobytes(order="C")).hexdigest()
    if support_digest != str(expected_support_iq_sha256).lower():
        raise QueryClosureError("support IQ binding mismatch")
    if (
        query_iq.ndim != 3
        or query_iq.shape[1:] != (2, 256)
        or query_ids.shape != (len(query_iq),)
        or len(query_iq) <= 0
        or len(query_iq) % 6
        or not np.isfinite(query_iq).all()
        or len(set(query_ids.tolist())) != len(query_ids)
        or len(set(support_ids.tolist())) != 60
        or set(support_ids.tolist()) & set(query_ids.tolist())
    ):
        raise QueryClosureError("independent query row or physical-ID drift")
    try:
        classes = sorted(
            package_manifest["registered_classes"], key=lambda row: int(row["class_index"])
        )
        class_handles = [str(row["class_handle"]) for row in classes]
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryClosureError("package class registry is invalid") from exc
    if (
        package_manifest.get("schema") != "cvs.phase2.predictor_package_manifest.v2"
        or package_manifest.get("registered_class_count") != 6
        or [int(row["class_index"]) for row in classes] != list(range(6))
        or len(set(class_handles)) != 6
    ):
        raise QueryClosureError("package class registry drift")
    bindings = (capsule_id, split_id, adaptation_capsule_id, adaptation_split_id)
    if not all(isinstance(value, str) and value.strip() for value in bindings):
        raise QueryClosureError("invalid data binding")

    receipt = {
        "schema": "cvs.sf_erbt_oldonly_export.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "adaptation_capsule_id": adaptation_capsule_id,
        "adaptation_split_id": adaptation_split_id,
        "k_shot": 10,
        "class_count": 6,
        "support_rows": 60,
        "query_rows": int(len(query_iq)),
        "support_iq_sha256": support_digest,
        "support_query_physical_id_overlap": 0,
        "query_truth_in_predictor": False,
        "registration_state": "REG0",
        "registered_class_handles": class_handles,
    }
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(
        destination / "support.npz",
        received_iq=support_iq,
        support_labels=support_labels,
        support_physical_ids=support_ids,
    )
    np.savez(
        destination / "query.npz",
        received_iq=np.ascontiguousarray(query_iq),
        query_ids=query_ids,
    )
    with (destination / "data_handle.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as writer:
        json.dump(receipt, writer, ensure_ascii=False, sort_keys=True, indent=2)
        writer.write("\n")
    return receipt


def _load_npz(path: Path, expected: frozenset[str], label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if frozenset(payload.files) != expected:
                raise QueryClosureError(f"{label} allowlist mismatch")
            return {name: np.asarray(payload[name]).copy() for name in payload.files}
    except QueryClosureError:
        raise
    except (OSError, ValueError) as exc:
        raise QueryClosureError(f"cannot load {label}: {path}") from exc


def _load_handle(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryClosureError(f"cannot load data handle: {path}") from exc
    required = {
        "schema": "cvs.sf_erbt_oldonly_export.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "registration_state": "REG0",
        "query_truth_in_predictor": False,
        "class_count": 6,
        "k_shot": 10,
        "support_rows": 60,
        "support_query_physical_id_overlap": 0,
    }
    if not isinstance(payload, Mapping) or any(payload.get(key) != value for key, value in required.items()):
        raise QueryClosureError("data handle binding mismatch")
    query_rows = payload.get("query_rows")
    if not isinstance(query_rows, int) or query_rows <= 0 or query_rows % 6:
        raise QueryClosureError("data handle query_rows is invalid")
    for key in ("capsule_id", "split_id", "adaptation_capsule_id", "adaptation_split_id"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise QueryClosureError(f"data handle {key} is invalid")
    return dict(payload)


def _target_binding(handle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": handle["adaptation_capsule_id"],
        "split_id": handle["adaptation_split_id"],
        "support_count": 60,
        "per_class_counts": [
            {"class_id": class_id, "count": 10} for class_id in range(6)
        ],
    }


def load_clean_single_pair_strict(
    bundle_path: str | Path,
    support_path: str | Path,
    *,
    device: str | torch.device,
    expected_target_binding: Mapping[str, Any],
):
    """Reconstruct candidate-specific DA0 and strictly load full or delta DA1."""

    target_device = torch.device(device)
    try:
        payload = torch.load(Path(bundle_path), map_location="cpu", weights_only=True)
        if payload.get("schema") == SF_TAPFT_DELTA_BUNDLE_SCHEMA:
            da1_model, da1_head, bundle_audit = load_sf_tapft_delta_bundle_strict(
                bundle_path,
                device=target_device,
                expected_target_binding=expected_target_binding,
            )
            base_checkpoint_path = str(bundle_audit["base_checkpoint_path"])
            adapter_rank = int(bundle_audit["adapter_rank"])
            interpolation = float(
                bundle_audit["da0_classifier_source_target_interpolation"]
            )
            prototype_scale = float(bundle_audit["da0_prototype_scale"])
        else:
            da1_model, da1_head, bundle_audit = load_sf_tapft_clean_single_bundle_strict(
                bundle_path,
                device=target_device,
                expected_target_binding=expected_target_binding,
            )
            normalized = _normalize_sf_tapft_bundle_config(payload["config"])
            normalized["validation_steps"] = ()
            config = SFTAPFTConfig(**normalized)
            base_checkpoint_path = str(payload["base_checkpoint_path"])
            adapter_rank = int(config.adapter_rank)
            interpolation = float(config.classifier_source_target_interpolation)
            prototype_scale = float(config.prototype_scale)
    except QueryClosureError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise QueryClosureError("cannot reconstruct clean-single DA0 binding") from exc

    support = _load_target_support(support_path)
    if len(support.physical_ids) != 60 or support.class_ids != tuple(range(6)):
        raise QueryClosureError("support does not match the old6 K10 binding")
    da0_model = _default_checkpoint_loader(base_checkpoint_path, device=target_device)
    da0_model.eval()
    source_weights = _source_classifier_weight(da0_model).detach()
    ensure_time_adapter(da0_model, rank=adapter_rank)
    dtype = next(
        parameter.dtype
        for parameter in da0_model.parameters()
        if parameter.is_floating_point()
    )
    support_x = support.received_iq.to(device=target_device, dtype=dtype)
    support_y = support.labels.to(device=target_device, dtype=torch.long)
    with torch.no_grad():
        embeddings = _extract_joint_embedding(
            _forward_aux(da0_model, support_x), int(support_x.size(0))
        )
        prototypes = _target_prototypes(embeddings, support_y, tuple(range(6)))
    da0_head = TargetPrototypeHead.from_source_and_target(
        source_weights=source_weights.to(device=target_device, dtype=dtype),
        target_prototypes=prototypes,
        source_class_ids=tuple(range(6)),
        target_class_ids=tuple(range(6)),
        rho=interpolation,
        scale=prototype_scale,
    ).to(device=target_device, dtype=dtype)
    da0_model.eval()
    da0_head.eval()
    for module in (da0_model, da0_head, da1_model, da1_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return da0_model, da0_head, da1_model, da1_head, {
        **dict(bundle_audit),
        "class_ids": list(da1_head.class_ids),
    }


def _prediction_payload(model, head, query_iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = torch.tensor(np.ascontiguousarray(query_iq).tolist(), dtype=torch.float32)
    result = predict_sf_tapft_rows(model, head, rows)
    return (
        np.asarray(result.predictions.tolist(), dtype=np.int64),
        np.asarray(result.logits.tolist(), dtype=np.float32),
    )


def run_clean_query_prediction(
    *,
    bundle_path: str | Path,
    support_path: str | Path,
    query_path: str | Path,
    data_handle_path: str | Path,
    output_root: str | Path,
    device: str | torch.device,
    pair_loader: Any | None = None,
) -> dict[str, Any]:
    """Create immutable DA0/DA1 predictions without opening query truth or role."""

    query = _load_npz(Path(query_path), frozenset({"received_iq", "query_ids"}), "query")
    handle = _load_handle(Path(data_handle_path))
    support_payload = _load_npz(
        Path(support_path),
        frozenset({"received_iq", "support_labels", "support_physical_ids"}),
        "support",
    )
    query_iq = np.asarray(query["received_iq"], dtype=np.float32)
    query_ids = np.asarray(query["query_ids"]).astype(str)
    query_rows = int(handle["query_rows"])
    support_iq = np.asarray(support_payload["received_iq"], dtype=np.float32)
    support_labels = np.asarray(support_payload["support_labels"], dtype=np.int64)
    support_ids = np.asarray(support_payload["support_physical_ids"]).astype(str)
    if (
        query_iq.shape != (query_rows, 2, 256)
        or query_ids.shape != (query_rows,)
        or not np.isfinite(query_iq).all()
        or len(set(query_ids.tolist())) != query_rows
    ):
        raise QueryClosureError("query row count or geometry drift")
    if (
        support_iq.shape != (60, 2, 256)
        or support_labels.shape != (60,)
        or support_ids.shape != (60,)
        or np.bincount(support_labels, minlength=6).tolist() != [10] * 6
        or len(set(support_ids.tolist())) != 60
        or set(support_ids.tolist()) & set(query_ids.tolist())
        or hashlib.sha256(np.ascontiguousarray(support_iq).tobytes(order="C")).hexdigest()
        != str(handle.get("support_iq_sha256", "")).lower()
    ):
        raise QueryClosureError("support/query binding drift")
    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise QueryClosureError(f"output root already exists: {destination}")
    loader = pair_loader or load_clean_single_pair_strict
    da0_model, da0_head, da1_model, da1_head, audit = loader(
        bundle_path,
        support_path,
        device=device,
        expected_target_binding=_target_binding(handle),
    )
    if (
        audit.get("capsule_id") != handle["adaptation_capsule_id"]
        or audit.get("split_id") != handle["adaptation_split_id"]
        or tuple(audit.get("class_ids", ())) != tuple(range(6))
    ):
        raise QueryClosureError("clean-single bundle/data binding mismatch")
    da0_predictions, da0_scores = _prediction_payload(da0_model, da0_head, query_iq)
    da1_predictions, da1_scores = _prediction_payload(da1_model, da1_head, query_iq)
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "DA0_REG0": destination / "da0_reg0.npz",
        "DA1_REG0": destination / "da1_reg0.npz",
    }
    for path, predictions, scores in (
        (paths["DA0_REG0"], da0_predictions, da0_scores),
        (paths["DA1_REG0"], da1_predictions, da1_scores),
    ):
        with path.open("xb") as handle_out:
            np.savez(
                handle_out,
                query_ids=query_ids,
                predicted_class_ids=predictions,
                scores=scores,
            )
    receipt = {
        "schema": "cvs.sf_tapft.query_closure_prediction.v1",
        "status": "PREDICTIONS_COMPLETE",
        "candidate_id": Path(bundle_path).parent.name,
        "bundle_path": str(Path(bundle_path)),
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": handle["capsule_id"],
        "split_id": handle["split_id"],
        "adaptation_capsule_id": handle["adaptation_capsule_id"],
        "adaptation_split_id": handle["adaptation_split_id"],
        "states": list(STATES),
        "registration_state": "REG0",
        "k_shot": 10,
        "support_rows": 60,
        "query_rows": query_rows,
        "prediction_paths": {key: path.name for key, path in paths.items()},
        "query_truth_opened": False,
        "query_role_opened": False,
        "source_opened": False,
    }
    with (destination / "prediction_receipt.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as writer:
        json.dump(receipt, writer, ensure_ascii=False, sort_keys=True, indent=2)
        writer.write("\n")
    return receipt


def materialize_truth_after_predictions(
    *,
    prediction_root: str | Path,
    truth_sidecar_path: str | Path,
    data_handle_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join the detached scorer sidecar only after both predictions exist."""

    root = Path(prediction_root)
    handle = _load_handle(Path(data_handle_path))
    query_rows = int(handle["query_rows"])
    try:
        receipt = json.loads(
            (root / "prediction_receipt.json").read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryClosureError("cannot load completed prediction receipt") from exc
    if (
        receipt.get("schema") != "cvs.sf_tapft.query_closure_prediction.v1"
        or receipt.get("status") != "PREDICTIONS_COMPLETE"
        or receipt.get("states") != list(STATES)
        or receipt.get("query_truth_opened") is not False
        or receipt.get("query_role_opened") is not False
        or receipt.get("query_rows") != query_rows
        or any(
            receipt.get(key) != handle.get(key)
            for key in ("capsule_id", "split_id", "k_shot", "support_rows")
        )
    ):
        raise QueryClosureError("truth cannot join an incomplete prediction")
    prediction_ids: np.ndarray | None = None
    for filename in ("da0_reg0.npz", "da1_reg0.npz"):
        payload = _load_npz(root / filename, PREDICTION_MEMBERS, filename)
        current = np.asarray(payload["query_ids"]).astype(str)
        predictions = np.asarray(payload["predicted_class_ids"], dtype=np.int64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        if current.shape != (query_rows,) or len(set(current.tolist())) != query_rows:
            raise QueryClosureError("completed prediction query IDs are invalid")
        if (
            predictions.shape != (query_rows,)
            or scores.shape != (query_rows, 6)
            or not np.isfinite(scores).all()
            or np.any(predictions < 0)
            or np.any(predictions >= 6)
        ):
            raise QueryClosureError("prediction geometry or score drift")
        if prediction_ids is None:
            prediction_ids = current
        elif not np.array_equal(prediction_ids, current):
            raise QueryClosureError("DA0/DA1 query IDs are not same-row")
    assert prediction_ids is not None
    try:
        sidecar = json.loads(Path(truth_sidecar_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryClosureError("cannot load detached truth sidecar") from exc
    class_handles = handle.get("registered_class_handles")
    if (
        not isinstance(class_handles, list)
        or len(class_handles) != 6
        or len(set(map(str, class_handles))) != 6
        or not isinstance(sidecar, Mapping)
        or sidecar.get("schema") != "cvs.phase2.query_truth_sidecar.v2"
        or not isinstance(sidecar.get("rows"), list)
    ):
        raise QueryClosureError("truth sidecar or class registry drift")
    class_lookup = {str(handle_value): index for index, handle_value in enumerate(class_handles)}
    truth_lookup: dict[str, int] = {}
    prediction_id_set = set(prediction_ids.tolist())
    for row in sidecar["rows"]:
        if not isinstance(row, Mapping):
            raise QueryClosureError("truth sidecar row schema drift")
        token = str(row.get("query_token", ""))
        if token not in prediction_id_set:
            continue
        if row.get("evaluation_role") != "target_old":
            raise QueryClosureError("query role is not target_old")
        class_handle = str(row.get("true_class_handle", ""))
        if token in truth_lookup or class_handle not in class_lookup:
            raise QueryClosureError("truth token or class handle drift")
        truth_lookup[token] = class_lookup[class_handle]
    if set(truth_lookup) != prediction_id_set:
        raise QueryClosureError("prediction/truth query ID mismatch")
    labels = np.asarray(
        [truth_lookup[token] for token in prediction_ids.tolist()], dtype=np.int64
    )
    if np.bincount(labels, minlength=6).tolist() != [query_rows // 6] * 6:
        raise QueryClosureError("truth is not balanced across old6")
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise QueryClosureError(f"truth output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as writer:
        np.savez(writer, query_ids=prediction_ids, query_labels=labels)
    return {
        "status": "TRUTH_MATERIALIZED_AFTER_PREDICTIONS",
        "query_rows": query_rows,
        "truth_join_after_prediction_only": True,
        "output_path": str(destination),
    }


def _softmax_nll(scores: np.ndarray, truth: np.ndarray) -> float:
    shifted = scores.astype(np.float64) - np.max(scores, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(logsumexp - shifted[np.arange(len(truth)), truth]))


def _metrics(predictions: np.ndarray, scores: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, float] = {}
    counts: dict[str, int] = {}
    correct: dict[str, int] = {}
    for class_id in range(6):
        mask = truth == class_id
        total = int(mask.sum())
        if total != len(truth) // 6:
            raise QueryClosureError("truth is not balanced across old6")
        hit = int(np.sum(predictions[mask] == truth[mask]))
        per_class[str(class_id)] = hit / total
        counts[str(class_id)] = total
        correct[str(class_id)] = hit
    values = list(per_class.values())
    return {
        "accuracy": float(np.mean(predictions == truth)),
        "balanced_accuracy": float(np.mean(values)),
        "class_floor": float(np.min(values)),
        "nll": _softmax_nll(scores, truth),
        "per_class_accuracy": per_class,
        "per_class_correct": correct,
        "per_class_total": counts,
    }


def score_clean_query_prediction(
    *,
    prediction_root: str | Path,
    truth_path: str | Path,
    data_handle_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Open truth only after both immutable prediction states are complete."""

    root = Path(prediction_root)
    try:
        receipt = json.loads((root / "prediction_receipt.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryClosureError("cannot load prediction receipt") from exc
    handle = _load_handle(Path(data_handle_path))
    query_rows = int(handle["query_rows"])
    if (
        receipt.get("schema") != "cvs.sf_tapft.query_closure_prediction.v1"
        or receipt.get("status") != "PREDICTIONS_COMPLETE"
        or receipt.get("states") != list(STATES)
        or receipt.get("query_truth_opened") is not False
        or receipt.get("query_role_opened") is not False
        or any(receipt.get(key) != handle.get(key) for key in ("capsule_id", "split_id", "k_shot", "support_rows", "query_rows", "registration_state"))
    ):
        raise QueryClosureError("prediction receipt/data binding mismatch")
    prediction_payloads = {
        "DA0_REG0": _load_npz(root / "da0_reg0.npz", PREDICTION_MEMBERS, "DA0 prediction"),
        "DA1_REG0": _load_npz(root / "da1_reg0.npz", PREDICTION_MEMBERS, "DA1 prediction"),
    }
    reference_ids: np.ndarray | None = None
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state, payload in prediction_payloads.items():
        ids = np.asarray(payload["query_ids"]).astype(str)
        predictions = np.asarray(payload["predicted_class_ids"], dtype=np.int64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        if ids.shape != (query_rows,) or predictions.shape != (query_rows,) or scores.shape != (query_rows, 6):
            raise QueryClosureError(f"{state} prediction row count or geometry drift")
        if len(set(ids.tolist())) != query_rows or not np.isfinite(scores).all():
            raise QueryClosureError(f"{state} prediction query ID or score drift")
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(reference_ids, ids):
            raise QueryClosureError("DA0/DA1 query IDs are not same-row")
        normalized[state] = (predictions, scores)
    truth_payload = _load_npz(
        Path(truth_path), frozenset({"query_ids", "query_labels"}), "truth"
    )
    truth_ids = np.asarray(truth_payload["query_ids"]).astype(str)
    truth_labels = np.asarray(truth_payload["query_labels"], dtype=np.int64)
    assert reference_ids is not None
    if truth_ids.shape != (query_rows,) or truth_labels.shape != (query_rows,):
        raise QueryClosureError("truth row count drift")
    if len(set(truth_ids.tolist())) != query_rows or set(truth_ids.tolist()) != set(reference_ids.tolist()):
        raise QueryClosureError("prediction/truth query ID mismatch")
    lookup = {query_id: index for index, query_id in enumerate(truth_ids.tolist())}
    order = np.asarray([lookup[query_id] for query_id in reference_ids.tolist()], dtype=np.int64)
    truth = truth_labels[order]
    state_metrics = {
        state: _metrics(predictions, scores, truth)
        for state, (predictions, scores) in normalized.items()
    }
    da0 = state_metrics["DA0_REG0"]
    da1 = state_metrics["DA1_REG0"]
    result = {
        "schema": "cvs.sf_tapft.query_closure_score.v1",
        "status": "ANALYZED",
        "candidate_id": receipt["candidate_id"],
        "capsule_id": handle["capsule_id"],
        "split_id": handle["split_id"],
        "states": list(STATES),
        "registration_state": "REG0",
        "query_rows": query_rows,
        "truth_join_after_prediction_only": True,
        "same_row_ids": True,
        "DA0_REG0": da0,
        "DA1_REG0": da1,
        "da_effect": {
            "accuracy_pp": 100.0 * (da1["accuracy"] - da0["accuracy"]),
            "balanced_accuracy_pp": 100.0 * (da1["balanced_accuracy"] - da0["balanced_accuracy"]),
            "class_floor_pp": 100.0 * (da1["class_floor"] - da0["class_floor"]),
            "nll_delta": da1["nll"] - da0["nll"],
        },
        "new_class_metrics": "N/A",
    }
    if not all(
        math.isfinite(float(value))
        for state in STATES
        for value in (
            state_metrics[state]["accuracy"],
            state_metrics[state]["balanced_accuracy"],
            state_metrics[state]["class_floor"],
            state_metrics[state]["nll"],
        )
    ):
        raise QueryClosureError("score contains a non-finite metric")
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise QueryClosureError(f"score output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as writer:
        json.dump(result, writer, ensure_ascii=False, sort_keys=True, indent=2)
        writer.write("\n")
    return result


__all__ = [
    "QueryClosureError",
    "export_independent_query_inputs",
    "load_clean_single_pair_strict",
    "materialize_truth_after_predictions",
    "run_clean_query_prediction",
    "score_clean_query_prediction",
]
