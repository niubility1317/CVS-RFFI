#!/usr/bin/env python3
"""One real-feature 400-step D103-R2 smoke with no truth-side scoring."""

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
    build_phase1_rb_metabias4_bundle,
    merge_verified_phase1_tap_and_dual_archives,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
    compile_teacher_bundle,
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


def _sha(path: Path) -> str:
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
            {"technical_smoke": True, "completed_steps": trainer.completed_steps}
        ),
        tx_probe_receipt_sha256=canonical_sha256(
            {"technical_smoke": True, "tx_probe_not_run": True}
        ),
        tx_probe_mean=0.0,
        tx_probe_max=0.0,
    )
    d102_tap = {
        "pre_relu": labeled["pre_relu"],
        "z_dom": labeled["z_dom"],
        "labels": labeled["tx_labels"],
        "receiver_ids": labeled["receiver_ids"],
        "day_ids": labeled["day_ids"],
        "physical_ids": labeled["physical_ids"],
        "class_ids": np.asarray(classes, dtype=str),
    }
    d102 = build_phase1_rb_metabias4_bundle(
        d102_tap,
        checkpoint_sha256=args.checkpoint_sha256,
        runtime_sha256=args.runtime_sha256,
        method_lock_sha256=(
            "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"
        ),
        excluded_receivers=(held_receiver,),
    )
    support, query = build_receiver_package_indices(
        scorer["receiver_ids"],
        scorer["labels"],
        scorer["physical_ids"],
        held_receiver=held_receiver,
        registered_classes=classes,
        k_shot=1,
    )
    artifact, stability = predict_matched_row(
        held_receiver=held_receiver,
        held_class=None,
        k_shot=1,
        support_pre_relu=scorer["pre_relu"][support],
        support_zdom=scorer["z_dom"][support],
        support_labels=scorer["labels"][support],
        query_pre_relu=scorer["pre_relu"][query],
        query_physical_ids=scorer["physical_ids"][query],
        registered_classes=classes,
        d102_bundle=d102,
        d103_outer_bundle=d103,
        d103_day_bundles=(d103, d103, d103, d103),
    )
    receipt = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.real_feature_smoke.v1",
        "status": "REAL_FEATURE_NO_QUERY_TRUTH_SMOKE_PASS",
        "input_status": "DEVELOPMENT_ONLY_NOT_FORMAL",
        "tap_archive_sha256": _sha(tap_path),
        "dual_archive_sha256": _sha(dual_path),
        "partition_counts": partition["counts"],
        "completed_meta_steps": trainer.completed_steps,
        "elapsed_seconds": time.monotonic() - started,
        "held_receiver": held_receiver,
        "support_count": int(len(support)),
        "query_count": int(len(query)),
        "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
        "d103_status": artifact["d103_fit_audit"]["status"],
        "leave_day_semantics": "same_outer_bundle_repeated_for_smoke_only_not_held_evidence",
        "shift_norm": float(stability["outer_shift_norm"]) if stability else None,
        "performance_computed": False,
        "query_truth_passed_to_predictor": False,
        "target_access": False,
        "formal_query_access": False,
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
