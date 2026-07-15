#!/usr/bin/env python
"""Source-only screen for the P4-BPJG-LOPO support adapter.

This script never opens a Phase2 target artifact.  It consumes the verified
Phase1 ``source_validation`` LEO_weak cache set, chooses a nested physical
support prefix, trains only on those registered support rows, and evaluates on
physical samples outside the complete K20 support pool.  Source labels are
used only for this preregistered method screen; the output has no formal
Stage2-C or matched-MRIOR claim authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
CODE_SCRIPTS_ROOT = CODE_ROOT / "scripts"
for candidate in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPTS_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPTS_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.checkpoint_loading import (  # noqa: E402
    build_exact_ssdg_model_from_checkpoint,
)
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    ids_sha256,
    sha256_file,
)
from cvsrffi.tensors import numpy_to_tensor_compat  # noqa: E402
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (  # noqa: E402
    inject_feat_joint_lora,
    load_and_merge_ground_lora,
    roundtrip_fp16_target_lora_and_merge,
    train_support_only_bp_jg,
)
from paper_reproduction.scripts.validate_cvs_ground_lora_multiview import (  # noqa: E402
    _feature_forward,
    load_source_validation_cache_set,
    split_source_cache_receivers,
    validate_receiver_holdout,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def select_role_symmetric_source_split(
    labels: np.ndarray,
    sample_ids: Sequence[str],
    candidate_indices: np.ndarray,
    *,
    class_count: int,
    k_shot: int,
    support_pool_max_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select nested support and a K-invariant physical query partition."""

    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = np.asarray(sample_ids).astype(str).reshape(-1)
    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if len(truth) != len(ids):
        raise ValueError("source labels and physical sample IDs are misaligned")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("source cache exposes duplicate physical sample IDs")
    if not 1 <= int(k_shot) <= int(support_pool_max_k):
        raise ValueError("k_shot must be inside the nested support pool")
    expected_classes = set(range(int(class_count)))
    observed_classes = set(int(value) for value in truth[candidates].tolist())
    if observed_classes != expected_classes:
        raise ValueError("source screen candidate class coverage drift")

    support_parts: list[np.ndarray] = []
    pool_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    rank_records: list[dict[str, Any]] = []
    for class_index in range(int(class_count)):
        class_indices = candidates[truth[candidates] == int(class_index)]
        if len(class_indices) <= int(support_pool_max_k):
            raise ValueError(
                f"source class {class_index} needs more than "
                f"{support_pool_max_k} physical samples"
            )
        ordered = class_indices[np.argsort(ids[class_indices], kind="stable")]
        class_seed = np.random.SeedSequence([int(seed), int(class_index)])
        permutation = np.random.default_rng(class_seed).permutation(len(ordered))
        shuffled = ordered[permutation]
        pool = shuffled[: int(support_pool_max_k)]
        query = shuffled[int(support_pool_max_k) :]
        support = pool[: int(k_shot)]
        support_parts.append(support)
        pool_parts.append(pool)
        query_parts.append(query)
        rank_records.append(
            {
                "class_index": int(class_index),
                "available_physical": int(len(class_indices)),
                "support_pool_ids_sha256": ids_sha256(ids[pool].tolist()),
                "query_ids_sha256": ids_sha256(ids[query].tolist()),
            }
        )

    support_indices = np.concatenate(support_parts).astype(np.int64)
    support_pool_indices = np.concatenate(pool_parts).astype(np.int64)
    query_indices = np.concatenate(query_parts).astype(np.int64)
    if set(support_pool_indices.tolist()) & set(query_indices.tolist()):
        raise RuntimeError("source support pool overlaps the fixed query partition")
    return support_indices, query_indices, {
        "split_policy": "seeded_class_symmetric_nested_physical_prefix",
        "seed": int(seed),
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "support_pool_max_k": int(support_pool_max_k),
        "physical_support_count": int(len(support_indices)),
        "physical_support_pool_count": int(len(support_pool_indices)),
        "physical_query_count": int(len(query_indices)),
        "support_ids": ids[support_indices].tolist(),
        "support_ids_sha256": ids_sha256(ids[support_indices].tolist()),
        "support_pool_ids_sha256": ids_sha256(ids[support_pool_indices].tolist()),
        "query_ids_sha256": ids_sha256(ids[query_indices].tolist()),
        "support_query_overlap_count": 0,
        "query_partition_invariant_for_k_1_5_10_20": True,
        "per_class": rank_records,
    }


def build_view_major_support(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    support_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Materialize three matched LEO scenario views in view-major order."""

    indices = np.asarray(support_indices, dtype=np.int64).reshape(-1)
    reference = arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]
    physical_ids = np.asarray(reference["sample_ids"]).astype(str)[indices].tolist()
    reference_labels = np.asarray(reference["raw_labels"], dtype=np.int64)[indices]
    row_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    row_ids: list[str] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        scenario_ids = np.asarray(arrays["sample_ids"]).astype(str)[indices].tolist()
        scenario_labels = np.asarray(arrays["raw_labels"], dtype=np.int64)[indices]
        if scenario_ids != physical_ids or not np.array_equal(
            scenario_labels, reference_labels
        ):
            raise ValueError(f"matched source support alignment drift: {scenario}")
        row_parts.append(
            np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[indices]
        )
        label_parts.append(scenario_labels)
        row_ids.extend(physical_ids)
    return (
        np.concatenate(row_parts, axis=0).astype(np.float32),
        np.concatenate(label_parts, axis=0).astype(np.int64),
        physical_ids,
        row_ids,
    )


def _metric_row(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(pred) != len(truth) or not len(truth):
        raise ValueError("metric inputs must be nonempty and aligned")
    per_class = {
        str(class_index): float(np.mean(pred[truth == class_index] == class_index))
        for class_index in sorted(np.unique(truth).tolist())
    }
    return {
        "accuracy": float(np.mean(pred == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "sample_count": int(len(truth)),
    }


def _build_model(
    checkpoint: Mapping[str, Any], *, input_len: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(input_len), device=device
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, audit


@torch.no_grad()
def _forward_batches(
    model: torch.nn.Module,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    values = np.asarray(rows, dtype=np.float32)
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        tensor = numpy_to_tensor_compat(
            values[start:stop],
            numpy_dtype=np.dtype(np.float32),
            torch_dtype=torch.float32,
            copy=False,
        ).to(device)
        feature, logit = _feature_forward(model, tensor, "z_id")
        features.append(feature.detach().float().cpu().numpy())
        logits.append(logit.detach().float().cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def evaluate_source_qknn(
    model: torch.nn.Module,
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]],
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    class_count: int,
    batch_size: int,
    device: torch.device,
    direct_classifier: bool = False,
) -> dict[str, Any]:
    """Evaluate qKNN or the support-free strict checkpoint classifier."""

    prototypes: torch.Tensor | None = None
    if not direct_classifier:
        support_features: list[np.ndarray] = []
        support_labels: list[np.ndarray] = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[scenario]
            rows = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[support_indices]
            features, _ = _forward_batches(
                model, rows, batch_size=int(batch_size), device=device
            )
            support_features.append(features)
            support_labels.append(
                np.asarray(arrays["raw_labels"], dtype=np.int64)[support_indices]
            )
        feature_tensor = F.normalize(
            torch.from_numpy(np.concatenate(support_features)).float(), dim=1
        )
        label_tensor = torch.from_numpy(np.concatenate(support_labels)).long()
        prototypes = torch.stack(
            [
                F.normalize(
                    feature_tensor[label_tensor == class_index].mean(dim=0), dim=0
                )
                for class_index in range(int(class_count))
            ],
            dim=0,
        )

    per_scenario: dict[str, Any] = {}
    all_predictions: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        rows = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[query_indices]
        truth = np.asarray(arrays["raw_labels"], dtype=np.int64)[query_indices]
        features, logits = _forward_batches(
            model, rows, batch_size=int(batch_size), device=device
        )
        if direct_classifier:
            if logits.ndim != 2 or int(logits.shape[1]) != int(class_count):
                raise ValueError("strict direct classifier class dimension drift")
            predictions = np.argmax(logits, axis=1)
        else:
            if prototypes is None:
                raise RuntimeError("qKNN prototype bank was not constructed")
            normalized = F.normalize(torch.from_numpy(features).float(), dim=1)
            predictions = torch.argmax(normalized @ prototypes.t(), dim=1).numpy()
        per_scenario[scenario] = _metric_row(predictions, truth)
        all_predictions.append(np.asarray(predictions, dtype=np.int64))
        all_truth.append(truth)
    return {
        "aggregate": _metric_row(
            np.concatenate(all_predictions), np.concatenate(all_truth)
        ),
        "per_scenario": per_scenario,
        "support_enrollment_backbone_forwards": int(
            0
            if direct_classifier
            else len(support_indices) * len(FORMAL_LEO_WEAK_SCENARIOS)
        ),
        "query_backbone_forwards": int(
            len(query_indices) * len(FORMAL_LEO_WEAK_SCENARIOS)
        ),
        "query_view_count": 1,
        "head": (
            "strict_direct_checkpoint_classifier"
            if direct_classifier
            else "cosine_prototype_all_registered_classes"
        ),
    }


def _write_trace(out_dir: Path, trace: list[dict[str, Any]]) -> None:
    _write_json(out_dir / "loss_trace.json", {"epochs": trace})
    fields = list(trace[0])
    with (out_dir / "loss_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ckpt_sha256", required=True)
    parser.add_argument("--ground_adapter_state", type=Path, required=True)
    parser.add_argument("--ground_adapter_sha256", required=True)
    parser.add_argument("--source_cache_set", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--source_train_rxs", required=True)
    parser.add_argument("--source_val_rxs", required=True)
    parser.add_argument("--class_count", type=int, default=6)
    parser.add_argument("--k_shot", type=int, choices=(1, 5, 10, 20), default=10)
    parser.add_argument("--support_pool_max_k", type=int, choices=(20,), default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scope", choices=("joint_gate", "identity_joint", "fusion_joint"), required=True
    )
    parser.add_argument("--rank", type=int, choices=(8,), default=8)
    parser.add_argument("--learning_rate", type=float, choices=(0.005, 0.01, 0.02), required=True)
    parser.add_argument("--epochs", type=int, choices=(5,), default=5)
    parser.add_argument("--max_optimizer_steps", type=int, choices=(50,), default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_checkpoint_sha = str(args.ckpt_sha256).strip().lower()
    if len(expected_checkpoint_sha) != 64 or any(
        value not in "0123456789abcdef" for value in expected_checkpoint_sha
    ):
        raise ValueError("checkpoint SHA256 must be 64 lowercase hex characters")
    observed_checkpoint_sha = sha256_file(args.ckpt)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("checkpoint trust-root mismatch")
    expected_ground_sha = str(args.ground_adapter_sha256).strip().lower()
    if len(expected_ground_sha) != 64 or any(
        value not in "0123456789abcdef" for value in expected_ground_sha
    ):
        raise ValueError("ground adapter SHA256 must be 64 lowercase hex characters")
    if sha256_file(args.ground_adapter_state) != expected_ground_sha:
        raise ValueError("ground adapter trust-root mismatch")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite source screen: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    receiver_audit = validate_receiver_holdout(
        args.source_train_rxs, args.source_val_rxs
    )
    arrays_by_scenario, cache_manifest, cache_audit = (
        load_source_validation_cache_set(args.source_cache_set)
    )
    _, validation_indices, _, validation_info = split_source_cache_receivers(
        arrays_by_scenario,
        train_receivers=args.source_train_rxs,
        validation_receivers=args.source_val_rxs,
        class_count=int(args.class_count),
    )
    reference = arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]
    support_indices, query_indices, split = select_role_symmetric_source_split(
        np.asarray(reference["raw_labels"], dtype=np.int64),
        np.asarray(reference["sample_ids"]).astype(str),
        validation_indices,
        class_count=int(args.class_count),
        k_shot=int(args.k_shot),
        support_pool_max_k=int(args.support_pool_max_k),
        seed=int(args.seed),
    )
    support_rows, support_labels, physical_support_ids, support_row_ids = (
        build_view_major_support(arrays_by_scenario, support_indices)
    )

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    try:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
    if sha256_file(args.ckpt) != observed_checkpoint_sha:
        raise ValueError("checkpoint changed while it was being loaded")
    input_len = int(support_rows.shape[-1])
    # Keep exactly one ADV3B02 instance on the GPU while measuring adaptation.
    # The identity and strict-direct baselines are instantiated only after the
    # target patch has trained and its peak-memory receipt has been captured.
    adapted_model, checkpoint_audit = _build_model(
        checkpoint, input_len=input_len, device=device
    )
    adapted_ground_audit = load_and_merge_ground_lora(
        adapted_model,
        args.ground_adapter_state,
        scope="projection_feature",
        rank=16,
        alpha=16.0,
        expected_sha256=expected_ground_sha,
    )
    resources = inject_feat_joint_lora(
        adapted_model,
        rank=int(args.rank),
        alpha=float(args.rank),
        scope=str(args.scope),
    )
    train_start = time.perf_counter()
    trace, runtime = train_support_only_bp_jg(
        adapted_model,
        support_rows,
        support_labels,
        physical_support_ids=physical_support_ids,
        support_row_physical_ids=support_row_ids,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=1.0e-4,
        temperature=18.0,
        support_view_count=len(FORMAL_LEO_WEAK_SCENARIOS),
        batch_size=int(args.batch_size),
        max_optimizer_steps=int(args.max_optimizer_steps),
        grad_clip=1.0,
        leave_one_physical_shot=True,
        seed=int(args.seed),
        device=device,
    )
    measured_train_seconds = float(time.perf_counter() - train_start)
    fp16_state = {
        name: parameter.detach().cpu().half()
        for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad
    }
    adapter_path = args.out_dir / "adapter_state_fp16.pt"
    torch.save(fp16_state, adapter_path)
    merge_audit = roundtrip_fp16_target_lora_and_merge(adapted_model, adapter_path)

    direct_model, _ = _build_model(checkpoint, input_len=input_len, device=device)
    identity_model, _ = _build_model(checkpoint, input_len=input_len, device=device)
    identity_ground_audit = load_and_merge_ground_lora(
        identity_model,
        args.ground_adapter_state,
        scope="projection_feature",
        rank=16,
        alpha=16.0,
        expected_sha256=expected_ground_sha,
    )

    direct_metrics = evaluate_source_qknn(
        direct_model,
        arrays_by_scenario,
        support_indices,
        query_indices,
        class_count=int(args.class_count),
        batch_size=int(args.batch_size),
        device=device,
        direct_classifier=True,
    )
    identity_metrics = evaluate_source_qknn(
        identity_model,
        arrays_by_scenario,
        support_indices,
        query_indices,
        class_count=int(args.class_count),
        batch_size=int(args.batch_size),
        device=device,
    )
    adapted_metrics = evaluate_source_qknn(
        adapted_model,
        arrays_by_scenario,
        support_indices,
        query_indices,
        class_count=int(args.class_count),
        batch_size=int(args.batch_size),
        device=device,
    )
    identity_aggregate = identity_metrics["aggregate"]
    adapted_aggregate = adapted_metrics["aggregate"]
    accuracy_delta = float(
        adapted_aggregate["accuracy"] - identity_aggregate["accuracy"]
    )
    floor_delta = float(
        adapted_aggregate["min_class_accuracy"]
        - identity_aggregate["min_class_accuracy"]
    )
    target_state_file_bytes = int(adapter_path.stat().st_size)
    persistent_state_estimate = int(
        args.ground_adapter_state.stat().st_size
        + target_state_file_bytes
        + int(args.class_count) * 160 * 2
        + 24
    )
    source_screen_pass = bool(accuracy_delta > 0.0 and floor_delta >= 0.0)
    result = {
        "schema": "cvs.p4_bpjg_lopo_source_screen.v1",
        "method": "P4-BPJG-LOPO",
        "scope": str(args.scope),
        "rank": int(args.rank),
        "learning_rate": float(args.learning_rate),
        "source_only": True,
        "formal_claim_authority": False,
        "phase2_target_artifact_access": False,
        "target_query_access": False,
        "mrior_matched_comparison_available": False,
        "claim_boundary": "source LEO_weak K10 method screen only",
        "source_screen_pass": source_screen_pass,
        "source_screen_rule": "adapted accuracy > P4 identity and floor not lower",
        "metrics": {
            "strict_direct_adv3b02": direct_metrics,
            "p4_identity_qknn": identity_metrics,
            "p4_bpjg_lopo_qknn": adapted_metrics,
            "delta_vs_p4_identity": {
                "accuracy": accuracy_delta,
                "min_class_accuracy": floor_delta,
            },
        },
        "resources": {
            **resources,
            "optimizer_steps": int(runtime["optimizer_steps"]),
            "adaptation_wall_seconds": float(runtime["adaptation_wall_seconds"]),
            "measured_end_to_end_train_seconds": measured_train_seconds,
            "peak_cuda_memory_bytes": int(runtime["peak_cuda_memory_bytes"]),
            "support_forward_sample_equivalents": int(
                runtime["support_forward_sample_equivalents"]
            ),
            "ground_adapter_serialized_file_bytes": int(
                args.ground_adapter_state.stat().st_size
            ),
            "target_adapter_serialized_file_bytes": target_state_file_bytes,
            "prototype_state_bytes_fp16": int(args.class_count) * 160 * 2,
            "adaptive_tta_threshold_state_bytes_fp32": 24,
            "persistent_state_bytes_estimate": persistent_state_estimate,
            "persistent_state_within_256k": persistent_state_estimate <= 256 * 1024,
            "deployment_added_macs_after_merge": 0,
        },
        "training_runtime": runtime,
        "split": split,
        "receiver_audit": receiver_audit,
        "source_validation_slice": validation_info,
        "cache": {
            "path": str(args.source_cache_set),
            "sha256": sha256_file(args.source_cache_set),
            "manifest": cache_manifest,
            "audit": cache_audit,
        },
        "checkpoint": {
            "path": str(args.ckpt),
            "sha256": observed_checkpoint_sha,
            "expected_sha256": expected_checkpoint_sha,
            "sha256_verified_before_and_after_load": True,
            "load_audit": checkpoint_audit,
        },
        "ground_adapter": {
            "path": str(args.ground_adapter_state),
            "sha256": expected_ground_sha,
            "identity_load_audit": identity_ground_audit,
            "adapted_load_audit": adapted_ground_audit,
        },
        "target_adapter": {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "merge_audit": merge_audit,
        },
    }
    _write_trace(args.out_dir, trace)
    _write_json(args.out_dir / "result.json", result)
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if source_screen_pass else 5


if __name__ == "__main__":
    raise SystemExit(main())
