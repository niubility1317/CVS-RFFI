#!/usr/bin/env python3
"""One real-feature D104 four-arm smoke without truth-side scoring."""

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
    frozen_qknn,
)
from cvsrffi.rxid_metabias4_phase1_trainer import (  # noqa: E402
    D103R1Phase1Trainer,
    OuterMaskSpec,
    build_training_data,
)
from cvsrffi.rxid_metabias4_source_archive import (  # noqa: E402
    partition_source_pool,
)
from cvsrffi.stage2_d104_rxid_angq import (  # noqa: E402
    ARMS,
    audit_d104_four_arm_int8,
    build_d104_prediction_artifact,
    fit_d104_four_arm_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-archive", type=Path, required=True)
    parser.add_argument("--dual-archive", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k-shot", type=int, choices=(5, 10), default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable smoke output exists: {output}")
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
        "candidate_id": "D104-R1-ANGQ-RXID-MB4",
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
                "technical_smoke": True,
                "completed_steps": trainer.completed_steps,
            }
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
        k_shot=args.k_shot,
    )
    support_receipt = canonical_sha256(
        {
            "technical_smoke_only": True,
            "held_receiver": held_receiver,
            "K": args.k_shot,
            "support_physical_ids": (
                scorer["physical_ids"][support].astype(str).tolist()
            ),
        }
    )
    state = fit_d104_four_arm_state(
        d103,
        np.asarray(scorer["pre_relu"][support], dtype=np.float32),
        np.asarray(scorer["z_dom"][support], dtype=np.float32),
        scorer["labels"][support].astype(str),
        classes,
        qknn_config=frozen_qknn(args.k_shot),
        stage="S_C",
        support_receipt_sha256=support_receipt,
    )
    artifact = build_d104_prediction_artifact(
        state,
        np.asarray(scorer["pre_relu"][query], dtype=np.float32),
        scorer["physical_ids"][query].astype(str),
    )
    int8_audit = audit_d104_four_arm_int8(
        state,
        np.asarray(scorer["pre_relu"][support], dtype=np.float32),
        scorer["labels"][support].astype(str),
        np.asarray(scorer["pre_relu"][query], dtype=np.float32),
    )
    receipt = {
        "schema": "cvs.d104_r1.rxid_angq.real_feature_noquery_smoke.v1",
        "status": "REAL_FEATURE_NO_QUERY_TRUTH_SMOKE_PASS",
        "input_status": "DEVELOPMENT_ONLY_NOT_FORMAL_HELD",
        "tap_archive_sha256": _sha256(tap_path),
        "dual_archive_sha256": _sha256(dual_path),
        "partition_counts": partition["counts"],
        "completed_meta_steps": trainer.completed_steps,
        "elapsed_seconds": time.monotonic() - started,
        "held_receiver": held_receiver,
        "K": args.k_shot,
        "support_count": int(len(support)),
        "query_count": int(len(query)),
        "state_receipt_sha256": state.state_receipt_sha256,
        "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
        "method_lock_sha256": state.method_lock["method_lock_sha256"],
        "four_arms": list(artifact["arm_predictions"]),
        "all_four_arms_present": tuple(artifact["arm_predictions"]) == ARMS,
        "d103_status": state.d103_state.status,
        "head_resource_gate_pass": bool(
            state.resource_receipts["head_effect"]["passes_d104_resource_gate"]
        ),
        "joint_resource_gate_pass": bool(
            state.resource_receipts["joint_effect"]["passes_d104_resource_gate"]
        ),
        "head_int8_top1_agreement": float(
            int8_audit["M_HEAD"]["top1_agreement"]
        ),
        "head_int8_margin_flip_count": int(
            int8_audit["M_HEAD"]["teacher_winner_margin_flip_count"]
        ),
        "joint_int8_top1_agreement": float(
            int8_audit["M_JOINT"]["top1_agreement"]
        ),
        "joint_int8_margin_flip_count": int(
            int8_audit["M_JOINT"]["teacher_winner_margin_flip_count"]
        ),
        "int8_gate_pass": bool(int8_audit["passes_d104_int8_gate"]),
        "historical_partition_query_features_used_for_prediction": int(
            len(query)
        ),
        "query_features_used_for_fit": 0,
        "query_truth_passed_to_predictor": False,
        "performance_computed": False,
        "target_access": False,
        "formal_held_evidence": False,
        "n607_run": False,
    }
    if (
        not receipt["all_four_arms_present"]
        or not receipt["head_resource_gate_pass"]
        or not receipt["joint_resource_gate_pass"]
    ):
        raise ValueError("D104 real-feature smoke closure failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            receipt,
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
