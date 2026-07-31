#!/usr/bin/env python3
"""Development-only D105 four-arm smoke on checkpoint-derived real features.

The script verifies the actual checkpoint bytes and consumes the previously
sealed Phase1 tap/dual feature archives.  It fits only from source-held support,
scores without labels/truth, and emits no accuracy or promotion metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    merge_verified_phase1_tap_and_dual_archives,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
    compile_teacher_bundle,
)
from cvsrffi.rxid_metabias4_phase1_trainer import (  # noqa: E402
    D103R1Phase1Trainer,
    OuterMaskSpec,
    build_training_data,
)
from cvsrffi.rxid_metabias4_source_archive import partition_source_pool  # noqa: E402
from cvsrffi.stage2_d105_cbrc import (  # noqa: E402
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
    compute_d105_support_binding_root,
    make_d105_cbrc_bundle_handle,
)
from cvsrffi.stage2_d105_four_arm import (  # noqa: E402
    audit_d105_four_arm_resources,
    build_d105_four_arm_state,
    score_d105_four_arm_logits,
)
from cvsrffi.stage2_lpo_rc_qknn import (  # noqa: E402
    TypedValidatedOnceP2SplitHandle,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    Phase1ZIDStudentTLock,
    _canonical_sha256,
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable smoke output exists: {output}")
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_sha256 = _sha_file(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    tap_path = args.tap_archive.resolve(strict=True)
    dual_path = args.dual_archive.resolve(strict=True)
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
    source_val_seal = {
        "row_count": len(scorer["physical_ids"]),
        "content_sha256": canonical_sha256(
            {
                "technical_smoke_only": True,
                "physical_ids": scorer["physical_ids"].astype(str).tolist(),
            }
        ),
    }
    data = build_training_data(labeled, unlabeled, source_val_seal)
    receivers = tuple(sorted(set(labeled["receiver_ids"].astype(str).tolist())))
    classes = tuple(sorted(set(labeled["tx_labels"].astype(str).tolist())))
    held_receiver = receivers[0]
    trainer = D103R1Phase1Trainer(
        data,
        OuterMaskSpec(held_receiver=held_receiver),
        device=args.device,
    )
    started = time.monotonic()
    for _ in range(data.config.total_meta_steps):
        trainer.step()
    exported = trainer.export_teacher_arrays()
    fit_manifest: dict[str, Any] = {
        "candidate_id": "D105-CBRC-MB4-REAL-FEATURE-SMOKE",
        "outer_spec": {
            "held_receiver": held_receiver,
            "held_day": None,
            "held_class": None,
        },
        "aggregation_receipt": dict(exported["aggregation_receipt"]),
    }
    bundle = compile_teacher_bundle(
        {
            "U": exported["U"],
            "B": exported["B"],
            "bank_g": exported["bank_g"],
            "bank_t": exported["bank_t"],
            "bank_precision": exported["bank_precision"],
            "bank_sigma": exported["bank_sigma"],
        },
        fit_manifest,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=args.runtime_sha256,
        method_lock_sha256=args.method_lock_sha256,
        training_receipt_sha256=canonical_sha256(
            {"technical_smoke": True, "completed_steps": trainer.completed_steps}
        ),
        tx_probe_receipt_sha256=canonical_sha256(
            {"technical_smoke": True, "tx_probe_not_run": True}
        ),
        tx_probe_mean=0.0,
        tx_probe_max=0.0,
    )
    support, query = build_receiver_package_indices(
        scorer["receiver_ids"],
        scorer["labels"],
        scorer["physical_ids"],
        held_receiver=held_receiver,
        registered_classes=classes,
        k_shot=1,
    )
    support_ids = tuple(scorer["physical_ids"][support].astype(str).tolist())
    query_ids = tuple(scorer["physical_ids"][query].astype(str).tolist())
    split_handle = TypedValidatedOnceP2SplitHandle(
        capsule_id=canonical_sha256(
            {
                "technical_smoke": True,
                "tap_sha256": _sha_file(tap_path),
                "dual_sha256": _sha_file(dual_path),
            }
        ),
        split_id=canonical_sha256(
            {
                "held_receiver": held_receiver,
                "active_k": 1,
                "support_ids": sorted(support_ids),
                "query_ids": sorted(query_ids),
            }
        ),
        validator_receipt_sha256=canonical_sha256(
            {
                "status": "DEVELOPMENT_ONLY_VALIDATED_ONCE",
                "protocol_schema": "p2_min_v1",
                "support_query_disjoint": True,
            }
        ),
        support_physical_root_sha256=_canonical_sha256(sorted(support_ids)),
        query_physical_root_sha256=_canonical_sha256(sorted(query_ids)),
        support_query_disjoint=True,
    )
    receipt_root = compute_d105_bundle_receipt_root(bundle)
    validated_bundle_id = canonical_sha256(
        {
            "technical_smoke": True,
            "content_root_sha256": bundle.content_root_sha256,
        }
    )
    validator_receipt = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=validated_bundle_id,
        expected_content_root_sha256=bundle.content_root_sha256,
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        receipt_root_sha256=receipt_root,
    )
    bundle_handle = make_d105_cbrc_bundle_handle(
        bundle,
        validated_bundle_id_sha256=validated_bundle_id,
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=bundle.content_root_sha256,
    )
    old_classes = classes[:3]
    new_classes = classes[3:]
    support_pre_relu = np.asarray(scorer["pre_relu"][support], dtype=np.float32)
    support_zdom = np.asarray(scorer["z_dom"][support], dtype=np.float32)
    support_labels = tuple(scorer["labels"][support].astype(str).tolist())
    support_receipt = compute_d105_support_binding_root(
        support_pre_relu,
        support_zdom,
        support_labels,
        support_ids,
        classes,
        old_classes,
        new_classes,
        active_k=1,
        stage="S_C",
    )
    config = Phase1ZIDStudentTLock(
        active_k=1,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256=checkpoint_sha256,
        quantization_margin_audit_sha256=_sha_file(dual_path),
    )
    state = build_d105_four_arm_state(
        bundle,
        bundle_handle,
        support_pre_relu,
        support_zdom,
        support_labels,
        support_ids,
        classes,
        old_classes,
        new_classes,
        config=config,
        split_handle=split_handle,
        active_k=1,
        stage="S_C",
        support_receipt_sha256=support_receipt,
    )
    state_receipt_before = state.receipt_sha256
    result = score_d105_four_arm_logits(
        state,
        np.asarray(scorer["pre_relu"][query], dtype=np.float32),
        query_physical_ids=query_ids,
    )
    resources = audit_d105_four_arm_resources(state)
    receipt = {
        "schema": "cvs.phase2.d105.four_arm.real_feature_smoke.v1",
        "status": "DEVELOPMENT_ONLY_REAL_CHECKPOINT_DERIVED_NO_TRUTH_SMOKE_PASS",
        "formal_phase2_evidence": False,
        "performance_computed": False,
        "target_access": False,
        "query_truth_read": False,
        "query_labels_passed_to_predictor": False,
        "checkpoint_bytes_verified": True,
        "checkpoint_sha256": checkpoint_sha256,
        "tap_archive_sha256": _sha_file(tap_path),
        "dual_archive_sha256": _sha_file(dual_path),
        "partition_counts": partition["counts"],
        "completed_meta_steps": trainer.completed_steps,
        "elapsed_seconds": time.monotonic() - started,
        "held_receiver": held_receiver,
        "active_k": 1,
        "registered_class_count": len(classes),
        "support_count": int(len(support)),
        "query_count": int(len(query)),
        "da_status": state.da_state.status,
        "state_receipt_sha256": state.receipt_sha256,
        "state_receipt_unchanged": state.receipt_sha256 == state_receipt_before,
        "query_state_updates": resources["query_state_updates"],
        "query_rows_used_for_fit": resources["query_rows_used_for_fit"],
        "k1_m_head_equals_m0_exact": bool(
            np.array_equal(result.m_head, result.m0)
        ),
        "k1_m_joint_equals_m_da_exact": bool(
            np.array_equal(result.m_joint, result.m_da)
        ),
        "logit_sha256_by_arm": {
            arm: _sha_array(logits) for arm, logits in result.by_arm.items()
        },
        "n607_run": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
