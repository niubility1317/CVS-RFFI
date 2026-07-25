#!/usr/bin/env python3
"""Development-only D103-R2 outer-receiver geometry probe.

The probe trains the seven receiver-held outer teachers on real Phase1 source
features and emits truth-free prediction-change diagnostics for K=1/5/10.
It never scores query labels and is not formal held or Target evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    build_phase1_rb_metabias4_bundle,
    merge_verified_phase1_tap_and_dual_archives,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
    compile_teacher_bundle,
    frozen_qknn,
    predict_matched_row,
)
from cvsrffi.rxid_metabias4_phase1_trainer import (  # noqa: E402
    D103R1Phase1Trainer,
    OuterMaskSpec,
    build_training_data,
)
from cvsrffi.rxid_metabias4_source_archive import (  # noqa: E402
    partition_source_pool,
)
from cvsrffi.stage2_rxid_metabias4 import (  # noqa: E402
    _apply_d103_metabias,
    fit_d103_stage2_state,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    _identity_class_scales,
    _quantize_rows,
    _score_with_support,
    decode_zid_support_bank,
    normalize_zid_rows,
)


D102_METHOD_LOCK_SHA256 = (
    "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"
)
K_VALUES = (1, 5, 10)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }


def _changed(left: Sequence[str], right: Sequence[str]) -> int:
    left_array = np.asarray(left).astype(str)
    right_array = np.asarray(right).astype(str)
    if left_array.shape != right_array.shape:
        raise ValueError("prediction shape drift")
    return int(np.sum(left_array != right_array))


def _top1_and_margin_flips(
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> dict[str, float | int]:
    if (
        teacher_logits.ndim != 2
        or candidate_logits.shape != teacher_logits.shape
        or teacher_logits.shape[1] < 2
        or not np.isfinite(teacher_logits).all()
        or not np.isfinite(candidate_logits).all()
    ):
        raise ValueError("finite matched logits with at least two classes required")
    winner = np.argmax(teacher_logits, axis=1)
    runner_logits = np.array(teacher_logits, copy=True)
    row = np.arange(len(teacher_logits))
    runner_logits[row, winner] = -np.inf
    runner_up = np.argmax(runner_logits, axis=1)
    candidate_margin = (
        candidate_logits[row, winner] - candidate_logits[row, runner_up]
    )
    return {
        "top1_agreement": float(
            np.mean(
                np.argmax(teacher_logits, axis=1)
                == np.argmax(candidate_logits, axis=1)
            )
        ),
        "teacher_winner_margin_flip_count": int(
            np.sum(candidate_margin <= 0.0)
        ),
    }


def _angular_grid_decode(
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Support-only fixed-grid INT8 scale search maximizing row cosine."""

    rows = normalize_zid_rows(
        np.asarray(support, dtype=np.float32)
    )
    factors = np.linspace(0.75, 1.25, 101, dtype=np.float64)
    decoded = np.empty_like(rows, dtype=np.float32)
    selected = np.empty(len(rows), dtype=np.float64)
    cosines = np.empty(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        best_cosine = -np.inf
        best_decoded = None
        best_factor = None
        for factor in factors:
            _, _, candidate = _quantize_row_at_factor(row, float(factor))
            cosine = float(
                np.dot(
                    row.astype(np.float64),
                    candidate.astype(np.float64),
                )
            )
            if cosine > best_cosine:
                best_cosine = cosine
                best_decoded = candidate
                best_factor = float(factor)
        if best_decoded is None or best_factor is None:
            raise ValueError("angular INT8 scale grid produced no finite row")
        decoded[row_index] = best_decoded
        selected[row_index] = best_factor
        cosines[row_index] = best_cosine
    return decoded, selected, cosines


def _quantize_row_at_factor(
    row: np.ndarray,
    factor: float,
) -> tuple[np.float16, np.ndarray, np.ndarray]:
    """Deploy-isomorphic quantization of one normalized support row."""

    row32 = np.asarray(row, dtype=np.float32)
    if (
        row32.ndim != 1
        or row32.size == 0
        or not np.isfinite(row32).all()
        or not np.isfinite(factor)
        or factor <= 0.0
    ):
        raise ValueError("finite row and positive finite factor required")
    normalized = normalize_zid_rows(row32[None, :])[0]
    base_scale = float(np.max(np.abs(normalized))) / 127.0
    scale16 = np.float16(
        max(
            base_scale * float(factor),
            float(np.finfo(np.float16).tiny),
        )
    )
    if scale16 <= 0.0 or not np.isfinite(scale16):
        raise ValueError("ANGQ scale underflow or non-finite value")
    codes = np.clip(
        np.rint(normalized / np.float32(scale16)),
        -127,
        127,
    ).astype(np.int8)
    if np.any(codes == np.int8(-128)):
        raise ValueError("ANGQ emitted forbidden INT8 code -128")
    raw = codes.astype(np.float32) * np.float32(scale16)
    if not np.isfinite(raw).all() or float(np.linalg.norm(raw)) <= 0.0:
        raise ValueError("ANGQ reconstruction is non-finite or zero norm")
    decoded = normalize_zid_rows(raw[None, :])[0]
    return scale16, codes, decoded


def _int8_component_audit(
    *,
    held_receiver: str,
    k_shot: int,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    query_pre_relu: np.ndarray,
    classes: Sequence[str],
    d103: Any,
) -> dict[str, Any] | None:
    """Isolate vector-code and bandwidth effects without changing state."""

    if k_shot == 1:
        return None
    qknn = frozen_qknn(k_shot)
    labels = tuple(str(value) for value in support_labels)
    state = fit_d103_stage2_state(
        d103,
        np.asarray(support_pre_relu, dtype=np.float32),
        np.asarray(support_zdom, dtype=np.float32),
        labels,
        tuple(classes),
        qknn_config=qknn,
        stage="S_C",
        support_receipt_sha256=canonical_sha256(
            {
                "development_component_audit": True,
                "held_receiver": held_receiver,
                "K": k_shot,
            }
        ),
    )
    support = _apply_d103_metabias(
        state.bundle,
        support_pre_relu,
        state.coefficient_fp16,
    )
    query = _apply_d103_metabias(
        state.bundle,
        query_pre_relu,
        state.coefficient_fp16,
    )
    support = normalize_zid_rows(support).astype(np.float64)
    query = normalize_zid_rows(query).astype(np.float64)
    decoded = decode_zid_support_bank(state.bank).astype(np.float64)
    class_map = {
        class_id: index for index, class_id in enumerate(state.bank.classes)
    }
    indices = np.asarray(
        [class_map[label] for label in labels],
        dtype=np.int16,
    )
    counts = tuple(
        int(np.sum(indices == index))
        for index in range(len(state.bank.classes))
    )
    teacher_scales = _identity_class_scales(
        support,
        indices,
        len(state.bank.classes),
        state.bank.config,
    )
    teacher = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=teacher_scales,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    vector_only = _score_with_support(
        support=decoded,
        class_indices=state.bank.class_indices_int16,
        support_counts=state.bank.support_counts,
        class_scales=teacher_scales,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    bandwidth_only = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=state.bank.class_scales_fp16,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    deployed = _score_with_support(
        support=decoded,
        class_indices=state.bank.class_indices_int16,
        support_counts=state.bank.support_counts,
        class_scales=state.bank.class_scales_fp16,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    angular, angular_factors, angular_cosines = _angular_grid_decode(support)
    angular_scales = np.asarray(
        _identity_class_scales(
            angular,
            indices,
            len(state.bank.classes),
            state.bank.config,
        ),
        dtype=np.float16,
    )
    angular_teacher_bandwidth = _score_with_support(
        support=angular,
        class_indices=indices,
        support_counts=counts,
        class_scales=teacher_scales,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    angular_deployed_bandwidth = _score_with_support(
        support=angular,
        class_indices=indices,
        support_counts=counts,
        class_scales=angular_scales,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    fp32_angular_bandwidth = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=angular_scales,
        query=query,
        config=state.bank.config,
        metric=state.metric,
    )
    return {
        "diagnostic_only_no_state_change": True,
        "query_truth_read": False,
        "vector_code_with_teacher_bandwidth": _top1_and_margin_flips(
            teacher, vector_only
        ),
        "fp32_vector_with_deployed_bandwidth": _top1_and_margin_flips(
            teacher, bandwidth_only
        ),
        "deployed_vector_and_bandwidth": _top1_and_margin_flips(
            teacher, deployed
        ),
        "angular_grid_support_only": {
            "factor_min": float(np.min(angular_factors)),
            "factor_max": float(np.max(angular_factors)),
            "factor_mean": float(np.mean(angular_factors)),
            "reconstruction_cosine_min": float(
                np.min(angular_cosines)
            ),
            "reconstruction_cosine_mean": float(
                np.mean(angular_cosines)
            ),
            "with_teacher_bandwidth": _top1_and_margin_flips(
                teacher, angular_teacher_bandwidth
            ),
            "with_recomputed_fp16_bandwidth": _top1_and_margin_flips(
                teacher, angular_deployed_bandwidth
            ),
            "direction_only_shared_angular_fp16_bandwidth": (
                _top1_and_margin_flips(
                    fp32_angular_bandwidth,
                    angular_deployed_bandwidth,
                )
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--held-receiver",
        action="append",
        default=[],
        help="optional exact receiver subset; repeat for multiple receivers",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(K_VALUES),
        help="optional subset of 1 5 10",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    tap_path = args.tap_archive.resolve(strict=True)
    dual_path = args.dual_archive.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable probe output exists: {output}")

    tap = _load(tap_path)
    dual = _load(dual_path)
    merged = merge_verified_phase1_tap_and_dual_archives(tap, dual)
    pool = {
        "z_id": np.asarray(tap["z_id"], dtype=np.float32),
        "z_dom": np.asarray(merged["z_dom"], dtype=np.float32),
        "pre_relu": np.asarray(merged["pre_relu"], dtype=np.float32),
        "labels": merged["labels"].astype(str),
        "receiver_ids": merged["receiver_ids"].astype(str),
        "day_ids": merged["day_ids"].astype(str),
        "physical_ids": merged["physical_ids"].astype(str),
        "scenario_names": tap["scenario_names"].astype(str),
        "observation_ids": tap["observation_ids"].astype(str),
        "class_ids": merged["class_ids"].astype(str),
    }
    labeled, unlabeled, scorer, partition = partition_source_pool(pool)
    all_receivers = tuple(
        sorted(set(labeled["receiver_ids"].astype(str).tolist()))
    )
    classes = tuple(sorted(set(labeled["tx_labels"].astype(str).tolist())))
    if len(all_receivers) != 7 or len(classes) != 6:
        raise ValueError("expected exact 7 receiver x 6 class source geometry")
    requested_receivers = tuple(dict.fromkeys(args.held_receiver))
    if any(value not in all_receivers for value in requested_receivers):
        raise ValueError("held receiver subset contains an unknown receiver")
    receivers = requested_receivers or all_receivers
    k_values = tuple(dict.fromkeys(int(value) for value in args.k_values))
    if not k_values or any(value not in K_VALUES for value in k_values):
        raise ValueError("K subset must contain only 1, 5, or 10")
    source_val_seal = {
        "row_count": len(scorer["physical_ids"]),
        "content_sha256": canonical_sha256(
            {
                "development_probe_only": True,
                "physical_ids": scorer["physical_ids"].astype(str).tolist(),
            }
        ),
    }
    data = build_training_data(labeled, unlabeled, source_val_seal)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for held_receiver in receivers:
        fit_started = time.monotonic()
        trainer = D103R1Phase1Trainer(
            data,
            OuterMaskSpec(held_receiver=held_receiver),
            device=args.device,
        )
        for _ in range(data.config.total_meta_steps):
            trainer.step()
        exported = trainer.export_teacher_arrays()
        fit_manifest: dict[str, Any] = {
            "candidate_id": "D103-R2-RXID-CROSSRECEIVER-MB4",
            "outer_spec": {
                "held_receiver": held_receiver,
                "held_day": None,
                "held_class": None,
            },
            "aggregation_receipt": dict(exported["aggregation_receipt"]),
        }
        d103 = compile_teacher_bundle(
            {
                "U": exported["U"],
                "B": exported["B"],
                "bank_g": exported["bank_g"],
                "bank_t": exported["bank_t"],
                "bank_precision": exported["bank_precision"],
                "bank_sigma": exported["bank_sigma"],
            },
            fit_manifest,
            checkpoint_sha256=args.checkpoint_sha256,
            runtime_sha256=args.runtime_sha256,
            method_lock_sha256=args.method_lock_sha256,
            training_receipt_sha256=canonical_sha256(
                {
                    "development_probe": True,
                    "held_receiver": held_receiver,
                    "completed_steps": trainer.completed_steps,
                }
            ),
            tx_probe_receipt_sha256=canonical_sha256(
                {
                    "development_probe": True,
                    "tx_probe_not_run": True,
                }
            ),
            tx_probe_mean=0.0,
            tx_probe_max=0.0,
        )
        d102 = build_phase1_rb_metabias4_bundle(
            {
                "pre_relu": labeled["pre_relu"],
                "z_dom": labeled["z_dom"],
                "labels": labeled["tx_labels"],
                "receiver_ids": labeled["receiver_ids"],
                "day_ids": labeled["day_ids"],
                "physical_ids": labeled["physical_ids"],
                "class_ids": np.asarray(classes, dtype=str),
            },
            checkpoint_sha256=args.checkpoint_sha256,
            runtime_sha256=args.runtime_sha256,
            method_lock_sha256=D102_METHOD_LOCK_SHA256,
            excluded_receivers=(held_receiver,),
        )
        fit_rows.append(
            {
                "held_receiver": held_receiver,
                "completed_meta_steps": trainer.completed_steps,
                "elapsed_seconds": time.monotonic() - fit_started,
                "bundle_content_root_sha256": d103.content_root_sha256,
            }
        )

        for k_shot in k_values:
            support, query = build_receiver_package_indices(
                scorer["receiver_ids"],
                scorer["labels"],
                scorer["physical_ids"],
                held_receiver=held_receiver,
                registered_classes=classes,
                k_shot=k_shot,
            )
            artifact, stability = predict_matched_row(
                held_receiver=held_receiver,
                held_class=None,
                k_shot=k_shot,
                support_pre_relu=scorer["pre_relu"][support],
                support_zdom=scorer["z_dom"][support],
                support_labels=scorer["labels"][support],
                query_pre_relu=scorer["pre_relu"][query],
                query_physical_ids=scorer["physical_ids"][query],
                registered_classes=classes,
                d102_bundle=d102,
                d103_outer_bundle=d103,
                # This is deliberately not formal leave-day evidence. Repeating
                # the outer bundle only permits a truth-free K1 execution probe.
                d103_day_bundles=(d103, d103, d103, d103),
            )
            audit = artifact["d103_fit_audit"]
            int8_audit = artifact["int8_audit"]
            resources = artifact["resource_audit"]
            component_audit = _int8_component_audit(
                held_receiver=held_receiver,
                k_shot=k_shot,
                support_pre_relu=scorer["pre_relu"][support],
                support_zdom=scorer["z_dom"][support],
                support_labels=scorer["labels"][support],
                query_pre_relu=scorer["pre_relu"][query],
                classes=classes,
                d103=d103,
            )
            rows.append(
                {
                    "held_receiver": held_receiver,
                    "K": k_shot,
                    "support_count": int(len(support)),
                    "query_count": int(len(query)),
                    "d103_status": audit["status"],
                    "information_rank": int(audit["data_information_rank"]),
                    "minimum_singular_value": float(
                        audit["data_minimum_singular_value"]
                    ),
                    "condition_number": float(
                        audit["system_condition_number"]
                    ),
                    "prior_fraction": float(audit["prior_fraction"]),
                    "coefficient_norm": float(audit["coefficient_norm"]),
                    "m0_to_d102_changed": _changed(
                        artifact["m0_predictions"],
                        artifact["d102_predictions"],
                    ),
                    "m0_to_d103_changed": _changed(
                        artifact["m0_predictions"],
                        artifact["d103_predictions"],
                    ),
                    "d102_to_d103_changed": _changed(
                        artifact["d102_predictions"],
                        artifact["d103_predictions"],
                    ),
                    "int8_top1_agreement": float(
                        int8_audit["top1_agreement"]
                    ),
                    "large_margin_flip_count": int(
                        int8_audit["large_margin_flip_count"]
                    ),
                    "int8_gate_pass": bool(
                        int8_audit["passes_d103_int8_gate"]
                    ),
                    "int8_logit_abs_error_mean": float(
                        int8_audit["logit_abs_error_mean"]
                    ),
                    "int8_logit_abs_error_max": float(
                        int8_audit["logit_abs_error_max"]
                    ),
                    "int8_teacher_margin_mean": float(
                        int8_audit["teacher_margin_mean"]
                    ),
                    "int8_quantized_teacher_margin_mean": float(
                        int8_audit["quantized_teacher_margin_mean"]
                    ),
                    "int8_teacher_bank_bandwidth_abs_delta_max": float(
                        int8_audit["teacher_bank_bandwidth_abs_delta_max"]
                    ),
                    "int8_component_audit": component_audit,
                    "state_bytes": int(
                        resources["actual_serialized_state_bytes"]
                        if resources["actual_serialized_state_bytes"] is not None
                        else resources["numeric_bundle_state_bytes"]
                    ),
                    "post_backbone_mac_per_query": int(
                        resources["post_backbone_mac_per_query"]
                    ),
                    "outer_shift_norm": (
                        None
                        if stability is None
                        else float(stability["outer_shift_norm"])
                    ),
                    "k1_leave_day_evidence": (
                        "OUTER_BUNDLE_REPEAT_DEVELOPMENT_PROXY_NOT_FORMAL"
                        if k_shot == 1
                        else "NOT_APPLICABLE"
                    ),
                    "query_truth_present": False,
                    "query_rows_used_for_fit": 0,
                    "performance_computed": False,
                }
            )
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "schema": "cvs.d103_r2.outer_geometry_local_probe.v1",
        "status": "DEVELOPMENT_ONLY_TRUTH_FREE_GEOMETRY_COMPLETE",
        "candidate": "D103-R2-RXID-CROSSRECEIVER-MB4",
        "input_status": "DEVELOPMENT_ONLY_NOT_FORMAL",
        "tap_archive_sha256": _sha(tap_path),
        "dual_archive_sha256": _sha(dual_path),
        "partition_counts": partition["counts"],
        "source_receiver_count": len(all_receivers),
        "selected_receiver_count": len(receivers),
        "selected_receivers": list(receivers),
        "class_count": len(classes),
        "fit_count": len(fit_rows),
        "row_count": len(rows),
        "k_values": list(k_values),
        "completed_meta_steps": sum(
            int(row["completed_meta_steps"]) for row in fit_rows
        ),
        "elapsed_seconds": time.monotonic() - started,
        "fits": fit_rows,
        "rows": rows,
        "query_truth_passed_to_predictor": False,
        "performance_computed": False,
        "target_access": False,
        "formal_query_access": False,
        "formal_held_evidence": False,
        "n607_run": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
