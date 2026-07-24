"""Non-performance feasibility probe for D103 RXID-Episodic-MetaBias4.

This module intentionally exposes no training artifact.  It uses one fixed
source-held inner fold to check pair construction, K1 matrix mechanics, and
representative forward/backward resource cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.d103.rxid_metabias4.feasibility_probe_non_performance.v1"
FROZEN_TAP_SHA256 = "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1"
INNER_HELD_RECEIVER = "1-1"
EPISODE_RECEIVER = "1-19"
BALANCED_SAMPLES_PER_CELL = 2
K_VALUES = (1, 5, 10)
WARMUP_STEPS = 3
TIMED_STEPS = 3
SEED = 103713


class D103FeasibilityError(ValueError):
    """Raised when the bounded source-held probe cannot be constructed."""


@dataclass(frozen=True)
class ProbeArrays:
    z_dom: np.ndarray
    pre_relu: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    labels: np.ndarray
    physical_ids: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_rows(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def load_frozen_tap(path: Path, expected_sha256: str = FROZEN_TAP_SHA256) -> ProbeArrays:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise D103FeasibilityError(
            f"tap sha256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "z_id",
            "pre_relu",
            "receiver_ids",
            "day_ids",
            "labels",
            "physical_ids",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise D103FeasibilityError(f"tap missing arrays: {missing}")
        arrays = ProbeArrays(
            z_dom=np.asarray(archive["z_id"], dtype=np.float32),
            pre_relu=np.asarray(archive["pre_relu"], dtype=np.float32),
            receiver_ids=_text_rows(archive["receiver_ids"]),
            day_ids=_text_rows(archive["day_ids"]),
            labels=_text_rows(archive["labels"]),
            physical_ids=_text_rows(archive["physical_ids"]),
        )
    row_count = arrays.z_dom.shape[0]
    if arrays.z_dom.shape != (row_count, 160) or arrays.pre_relu.shape != (row_count, 160):
        raise D103FeasibilityError("tap feature shapes must both be [N,160]")
    for values in (
        arrays.receiver_ids,
        arrays.day_ids,
        arrays.labels,
        arrays.physical_ids,
    ):
        if values.shape != (row_count,):
            raise D103FeasibilityError("tap metadata arrays must all be [N]")
    if np.unique(arrays.physical_ids).size != row_count:
        raise D103FeasibilityError("physical IDs must be unique in frozen tap")
    return arrays


def cross_day_pair_summary(
    receiver_ids: np.ndarray,
    day_ids: np.ndarray,
    labels: np.ndarray,
    physical_ids: np.ndarray,
) -> dict[str, Any]:
    receivers = sorted(np.unique(receiver_ids).tolist())
    per_receiver: dict[str, Any] = {}
    all_constructible = True
    for receiver in receivers:
        local = receiver_ids == receiver
        tx_rows: dict[str, Any] = {}
        for label in sorted(np.unique(labels[local]).tolist()):
            cell = local & (labels == label)
            days = sorted(np.unique(day_ids[cell]).tolist())
            tx_rows[label] = {
                "day_count": len(days),
                "physical_count": int(np.unique(physical_ids[cell]).size),
                "cross_day_constructible": len(days) >= 2,
            }
            all_constructible = all_constructible and len(days) >= 2
        per_receiver[receiver] = {
            "day_count": int(np.unique(day_ids[local]).size),
            "tx_count": int(np.unique(labels[local]).size),
            "tx": tx_rows,
        }
    return {
        "all_receiver_tx_cross_day_constructible": bool(all_constructible),
        "per_receiver": per_receiver,
    }


def fixed_balanced_indices(arrays: ProbeArrays) -> np.ndarray:
    selected: list[int] = []
    train = arrays.receiver_ids != INNER_HELD_RECEIVER
    receivers = sorted(np.unique(arrays.receiver_ids[train]).tolist())
    days = sorted(np.unique(arrays.day_ids[train]).tolist())
    labels = sorted(np.unique(arrays.labels[train]).tolist())
    for receiver in receivers:
        for day in days:
            for label in labels:
                candidates = np.flatnonzero(
                    train
                    & (arrays.receiver_ids == receiver)
                    & (arrays.day_ids == day)
                    & (arrays.labels == label)
                )
                if candidates.size < BALANCED_SAMPLES_PER_CELL:
                    raise D103FeasibilityError(
                        f"cell lacks {BALANCED_SAMPLES_PER_CELL} physical samples: "
                        f"receiver={receiver}, day={day}, label={label}"
                    )
                order = np.argsort(arrays.physical_ids[candidates], kind="stable")
                selected.extend(
                    candidates[order[:BALANCED_SAMPLES_PER_CELL]].astype(int).tolist()
                )
    result = np.asarray(selected, dtype=np.int64)
    if np.unique(arrays.physical_ids[result]).size != result.size:
        raise D103FeasibilityError("balanced batch reuses a physical sample")
    return result


def episode_indices(arrays: ProbeArrays, k_shot: int, query_per_class: int = 16) -> tuple[np.ndarray, np.ndarray]:
    local = arrays.receiver_ids == EPISODE_RECEIVER
    support: list[int] = []
    query: list[int] = []
    for label in sorted(np.unique(arrays.labels[local]).tolist()):
        candidates = np.flatnonzero(local & (arrays.labels == label))
        order = np.argsort(arrays.physical_ids[candidates], kind="stable")
        candidates = candidates[order]
        need = k_shot + query_per_class
        if candidates.size < need:
            raise D103FeasibilityError(
                f"episode class {label} needs {need} rows, found {candidates.size}"
            )
        support.extend(candidates[:k_shot].astype(int).tolist())
        query.extend(candidates[k_shot:need].astype(int).tolist())
    support_idx = np.asarray(support, dtype=np.int64)
    query_idx = np.asarray(query, dtype=np.int64)
    if np.intersect1d(arrays.physical_ids[support_idx], arrays.physical_ids[query_idx]).size:
        raise D103FeasibilityError("episode support/query physical IDs overlap")
    return support_idx, query_idx


def linear_tx_nullspace(arrays: ProbeArrays, train_indices: np.ndarray) -> np.ndarray:
    features = np.asarray(arrays.z_dom[train_indices], dtype=np.float64)
    labels = arrays.labels[train_indices]
    receivers = arrays.receiver_ids[train_indices]
    days = arrays.day_ids[train_indices]
    tx_means = []
    for label in sorted(np.unique(labels).tolist()):
        cell_means = []
        for receiver in sorted(np.unique(receivers).tolist()):
            for day in sorted(np.unique(days).tolist()):
                local = (labels == label) & (receivers == receiver) & (days == day)
                if np.any(local):
                    cell_means.append(features[local].mean(axis=0))
        if not cell_means:
            raise D103FeasibilityError(f"TX {label} has no balanced source cells")
        tx_means.append(np.mean(cell_means, axis=0))
    means = np.asarray(tx_means)
    centered = means - means.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1.0e-10))
    if rank != len(tx_means) - 1:
        raise D103FeasibilityError(f"expected TX mean rank {len(tx_means)-1}, got {rank}")
    basis = vh[:rank]
    projector = np.eye(features.shape[1], dtype=np.float64) - basis.T @ basis
    residual = np.linalg.norm(projector @ basis.T)
    if residual > 1.0e-8:
        raise D103FeasibilityError(f"TX nullspace residual too large: {residual}")
    return np.asarray(projector, dtype=np.float32)


def k1_matrix_mechanics(
    arrays: ProbeArrays,
    batch_indices: np.ndarray,
    projector: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    raw_w = rng.standard_normal((32, 160))
    rows = raw_w @ np.asarray(projector, dtype=np.float64)
    q, _ = np.linalg.qr(rows.T, mode="reduced")
    encoder = q.T
    encoded = arrays.z_dom[batch_indices].astype(np.float64) @ encoder.T
    encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1.0e-12)

    receiver_ids = arrays.receiver_ids[batch_indices]
    day_ids = arrays.day_ids[batch_indices]
    labels = arrays.labels[batch_indices]
    cells = sorted(set(zip(receiver_ids.tolist(), day_ids.tolist())))
    bank_g = []
    for receiver, day in cells:
        class_means = []
        for label in sorted(np.unique(labels).tolist()):
            local = (
                (receiver_ids == receiver)
                & (day_ids == day)
                & (labels == label)
            )
            if not np.any(local):
                raise D103FeasibilityError("balanced cell is missing a TX")
            class_means.append(encoded[local].mean(axis=0))
        cell = np.mean(class_means, axis=0)
        bank_g.append(cell / max(np.linalg.norm(cell), 1.0e-12))
    bank_g_rows = np.asarray(bank_g)

    support_idx, _ = episode_indices(arrays, 1)
    support_r = arrays.z_dom[support_idx].astype(np.float64) @ encoder.T
    support_r /= np.maximum(np.linalg.norm(support_r, axis=1, keepdims=True), 1.0e-12)
    similarity = np.clip(support_r @ bank_g_rows.T, -1.0, 1.0)
    weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / 0.2)
    weights /= weights.sum(axis=1, keepdims=True)

    cell_count = bank_g_rows.shape[0]
    precision = 0.25 + rng.random((cell_count, 4))
    per_sample = weights @ precision
    class_labels = arrays.labels[support_idx]
    class_precision = []
    for label in sorted(np.unique(class_labels).tolist()):
        class_precision.append(per_sample[class_labels == label].mean(axis=0))
    data_diag = np.mean(class_precision, axis=0)
    matrix = np.diag(0.5 + data_diag)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(matrix)),
        "min_singular_value": float(singular.min()),
        "max_singular_value": float(singular.max()),
        "condition_number": float(singular.max() / singular.min()),
        "support_physical_count": int(support_idx.size),
        "registered_class_count": int(np.unique(class_labels).size),
        "views_counted_as_additional_k": 0,
    }


def _torch_probe(
    arrays: ProbeArrays,
    batch_indices: np.ndarray,
    projector: np.ndarray,
    device_name: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise D103FeasibilityError("CUDA is required for the bounded GPU resource probe")
    device = torch.device(device_name)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    label_values = sorted(np.unique(arrays.labels).tolist())
    receiver_values = sorted(np.unique(arrays.receiver_ids[batch_indices]).tolist())
    day_values = sorted(np.unique(arrays.day_ids).tolist())
    label_to_index = {value: index for index, value in enumerate(label_values)}
    receiver_to_index = {value: index for index, value in enumerate(receiver_values)}
    day_to_index = {value: index for index, value in enumerate(day_values)}

    z_batch = torch.as_tensor(arrays.z_dom[batch_indices], device=device)
    pre_batch = torch.as_tensor(arrays.pre_relu[batch_indices], device=device)
    y_batch = torch.as_tensor(
        [label_to_index[value] for value in arrays.labels[batch_indices]],
        device=device,
        dtype=torch.long,
    )
    rcv_batch = torch.as_tensor(
        [receiver_to_index[value] for value in arrays.receiver_ids[batch_indices]],
        device=device,
        dtype=torch.long,
    )
    day_batch = torch.as_tensor(
        [day_to_index[value] for value in arrays.day_ids[batch_indices]],
        device=device,
        dtype=torch.long,
    )
    projector_t = torch.as_tensor(projector, device=device)

    episode_tensors = []
    for k_shot in K_VALUES:
        support_idx, query_idx = episode_indices(arrays, k_shot)
        episode_tensors.append(
            (
                k_shot,
                torch.as_tensor(arrays.z_dom[support_idx], device=device),
                torch.as_tensor(arrays.pre_relu[support_idx], device=device),
                torch.as_tensor(
                    [label_to_index[value] for value in arrays.labels[support_idx]],
                    device=device,
                    dtype=torch.long,
                ),
                torch.as_tensor(arrays.z_dom[query_idx], device=device),
                torch.as_tensor(arrays.pre_relu[query_idx], device=device),
                torch.as_tensor(
                    [label_to_index[value] for value in arrays.labels[query_idx]],
                    device=device,
                    dtype=torch.long,
                ),
            )
        )

    cell_keys = sorted(
        set(
            zip(
                arrays.receiver_ids[batch_indices].tolist(),
                arrays.day_ids[batch_indices].tolist(),
            )
        )
    )
    cell_masks = []
    for receiver, day in cell_keys:
        mask = (arrays.receiver_ids[batch_indices] == receiver) & (
            arrays.day_ids[batch_indices] == day
        )
        cell_masks.append(torch.as_tensor(mask, device=device))

    raw_w = torch.nn.Parameter(torch.randn(32, 160, device=device) * 0.02)
    basis = torch.nn.Parameter(torch.randn(160, 4, device=device) * 0.01)
    bank_t = torch.nn.Parameter(torch.randn(len(cell_keys), 4, device=device) * 0.01)
    log_precision = torch.nn.Parameter(torch.zeros(len(cell_keys), 4, device=device))
    log_sigma = torch.nn.Parameter(torch.zeros(len(cell_keys), device=device))
    parameters = [raw_w, basis, bank_t, log_precision, log_sigma]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)

    positive_mask = (
        (rcv_batch[:, None] == rcv_batch[None, :])
        & (day_batch[:, None] != day_batch[None, :])
        & (y_batch[:, None] != y_batch[None, :])
    )
    positive_mask.fill_diagonal_(False)
    if not torch.all(positive_mask.any(dim=1)):
        raise D103FeasibilityError("fixed balanced batch lacks cross-day/cross-TX positives")

    def roworth(value: Any) -> Any:
        q, _ = torch.linalg.qr((value @ projector_t).T, mode="reduced")
        return q.T

    def mmd_loss(encoded: Any) -> Any:
        losses = []
        for left in range(len(label_values)):
            x = encoded[y_batch == left]
            for right in range(left + 1, len(label_values)):
                y = encoded[y_batch == right]
                xx = torch.cdist(x, x).square()
                yy = torch.cdist(y, y).square()
                xy = torch.cdist(x, y).square()
                for gamma in (0.5, 1.0, 2.0):
                    losses.append(
                        torch.exp(-gamma * xx).mean()
                        + torch.exp(-gamma * yy).mean()
                        - 2.0 * torch.exp(-gamma * xy).mean()
                    )
        return torch.stack(losses).mean()

    def rx_loss(encoded: Any) -> Any:
        logits = encoded @ encoded.T / 0.1
        logits.fill_diagonal_(-torch.inf)
        denominator = torch.logsumexp(logits, dim=1)
        positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
        numerator = torch.logsumexp(positive_logits, dim=1)
        return (denominator - numerator).mean()

    def vicreg_loss(encoded: Any) -> Any:
        std = torch.sqrt(encoded.var(dim=0) + 1.0e-4)
        variance = functional.relu(0.05 - std).mean()
        centered = encoded - encoded.mean(dim=0)
        covariance = centered.T @ centered / max(encoded.shape[0] - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        return variance + off_diagonal.square().mean()

    def qknn_logits(query: Any, support: Any, support_y: Any) -> Any:
        similarity = query @ support.T
        rows = []
        for label_index in range(len(label_values)):
            local = similarity[:, support_y == label_index]
            rows.append(torch.logsumexp(local / 0.2, dim=1) - math.log(local.shape[1]))
        return torch.stack(rows, dim=1)

    def one_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        encoder = roworth(raw_w)
        encoded = functional.normalize(z_batch @ encoder.T, dim=1)

        bank_g_rows = []
        for cell_mask in cell_masks:
            class_rows = []
            for label_index in range(len(label_values)):
                local = cell_mask & (y_batch == label_index)
                class_rows.append(encoded[local].mean(dim=0))
            bank_g_rows.append(functional.normalize(torch.stack(class_rows).mean(dim=0), dim=0))
        bank_g = torch.stack(bank_g_rows)
        precision = functional.softplus(log_precision) + 0.05
        sigma = functional.softplus(log_sigma) + 0.05

        episode_losses = []
        for _, support_z, support_pre, support_y, query_z, query_pre, query_y in episode_tensors:
            support_r = functional.normalize(support_z @ encoder.T, dim=1)
            similarity = torch.clamp(support_r @ bank_g.T, -1.0, 1.0)
            weights = torch.softmax(similarity / 0.2, dim=1)
            quality = torch.sum(
                weights * torch.exp(-(1.0 - similarity) / sigma.square()[None, :]),
                dim=1,
            )
            sample_precision = quality[:, None] * (weights @ precision)
            sample_mean = weights @ bank_t
            class_precision = []
            class_rhs = []
            for label_index in range(len(label_values)):
                local = support_y == label_index
                class_precision.append(sample_precision[local].mean(dim=0))
                class_rhs.append(
                    (sample_precision[local] * sample_mean[local]).mean(dim=0)
                )
            data_precision = torch.stack(class_precision).mean(dim=0)
            rhs = torch.stack(class_rhs).mean(dim=0)
            coefficient = rhs / (0.5 + data_precision)

            support_da = functional.normalize(
                functional.relu(support_pre + coefficient @ basis.T), dim=1
            )
            query_da = functional.normalize(
                functional.relu(query_pre + coefficient @ basis.T), dim=1
            )
            da_logits = qknn_logits(query_da, support_da, support_y)
            da_loss = functional.cross_entropy(da_logits, query_y, reduction="none")

            support_base = functional.normalize(functional.relu(support_pre), dim=1)
            query_base = functional.normalize(functional.relu(query_pre), dim=1)
            base_logits = qknn_logits(query_base, support_base, support_y)
            base_loss = functional.cross_entropy(base_logits, query_y, reduction="none").detach()
            episode_losses.append(
                da_loss.mean() + 0.1 * torch.logsumexp((da_loss - base_loss) / 0.1, dim=0)
            )

        orthogonality = (encoder @ encoder.T - torch.eye(32, device=device)).square().mean()
        loss = (
            torch.stack(episode_losses).mean()
            + mmd_loss(encoded)
            + rx_loss(encoded)
            + vicreg_loss(encoded)
            + orthogonality
            + 0.0 * pre_batch.sum()
        )
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    warmup_losses = [one_step() for _ in range(WARMUP_STEPS)]
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    timed_losses = [one_step() for _ in range(TIMED_STEPS)]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "parameter_count": int(sum(value.numel() for value in parameters)),
        "balanced_batch_rows": int(batch_indices.size),
        "cell_count": len(cell_keys),
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "timed_total_seconds": float(elapsed),
        "mean_seconds_per_meta_step": float(elapsed / TIMED_STEPS),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "warmup_loss_finite": bool(np.isfinite(warmup_losses).all()),
        "timed_loss_finite": bool(np.isfinite(timed_losses).all()),
    }


def run_probe(tap_path: Path, device: str = "cuda:0") -> dict[str, Any]:
    arrays = load_frozen_tap(tap_path)
    pair_summary = cross_day_pair_summary(
        arrays.receiver_ids,
        arrays.day_ids,
        arrays.labels,
        arrays.physical_ids,
    )
    batch_indices = fixed_balanced_indices(arrays)
    projector = linear_tx_nullspace(arrays, batch_indices)
    k1 = k1_matrix_mechanics(arrays, batch_indices, projector)
    resource = _torch_probe(arrays, batch_indices, projector, device)
    return {
        "schema": SCHEMA,
        "status": "FEASIBILITY_PROBE_NON_PERFORMANCE",
        "claim_semantics": "RESOURCE_AND_CONSTRUCTABILITY_ONLY",
        "performance_metrics_computed": False,
        "target_access": False,
        "capsule_access": False,
        "formal_query_access": False,
        "deployment_asset_saved": False,
        "tap": {
            "path": str(tap_path),
            "sha256": file_sha256(tap_path),
            "row_count": int(arrays.z_dom.shape[0]),
            "feature_width": int(arrays.z_dom.shape[1]),
            "physical_id_unique": bool(np.unique(arrays.physical_ids).size == arrays.z_dom.shape[0]),
        },
        "fixed_inner_fold": {
            "held_receiver": INNER_HELD_RECEIVER,
            "episode_receiver": EPISODE_RECEIVER,
            "outer_results_read": False,
            "constants_tuned_from_probe": False,
        },
        "pair_constructability": pair_summary,
        "balanced_batch": {
            "rows": int(batch_indices.size),
            "samples_per_receiver_day_tx_cell": BALANCED_SAMPLES_PER_CELL,
            "physical_id_unique": bool(
                np.unique(arrays.physical_ids[batch_indices]).size == batch_indices.size
            ),
        },
        "k1_matrix_mechanics": k1,
        "resource_measurement": resource,
        "forbidden_outputs_absent": {
            "balanced_accuracy": True,
            "tx_probe_accuracy": True,
            "loco_performance": True,
            "learned_u": True,
            "learned_b": True,
            "learned_bank": True,
            "deployment_bundle": True,
        },
    }


def write_probe_json(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_result_shape(result: Mapping[str, Any]) -> None:
    if result.get("schema") != SCHEMA:
        raise D103FeasibilityError("unexpected probe schema")
    if result.get("status") != "FEASIBILITY_PROBE_NON_PERFORMANCE":
        raise D103FeasibilityError("unexpected probe status")
    for field in (
        "performance_metrics_computed",
        "target_access",
        "capsule_access",
        "formal_query_access",
        "deployment_asset_saved",
    ):
        if result.get(field) is not False:
            raise D103FeasibilityError(f"probe boundary violated: {field}")
    if not result["pair_constructability"]["all_receiver_tx_cross_day_constructible"]:
        raise D103FeasibilityError("cross-day pair construction failed")
    if result["k1_matrix_mechanics"]["rank"] != 4:
        raise D103FeasibilityError("K1 mechanical matrix is not rank four")
    resource = result["resource_measurement"]
    if not resource["warmup_loss_finite"] or not resource["timed_loss_finite"]:
        raise D103FeasibilityError("resource probe produced non-finite loss")


__all__ = [
    "BALANCED_SAMPLES_PER_CELL",
    "D103FeasibilityError",
    "EPISODE_RECEIVER",
    "FROZEN_TAP_SHA256",
    "INNER_HELD_RECEIVER",
    "K_VALUES",
    "ProbeArrays",
    "SCHEMA",
    "cross_day_pair_summary",
    "episode_indices",
    "fixed_balanced_indices",
    "k1_matrix_mechanics",
    "linear_tx_nullspace",
    "load_frozen_tap",
    "run_probe",
    "validate_result_shape",
    "write_probe_json",
]
