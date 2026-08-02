#!/usr/bin/env python3
"""Run frozen four-arm D114 source-held G1 with truth opened only by score."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_d106_g1_sourceheld_one_shot as d106  # noqa: E402
from scripts import run_d110_g1_sourceheld_one_shot as d110  # noqa: E402
from scripts import run_d112_g1_sourceheld_one_shot as d112  # noqa: E402
from cvsrffi.stage2_d112_seam_qknn import (  # noqa: E402
    audit_d112_seam_state,
    fit_d112_ground_head_source_held_g1_state,
)
from cvsrffi.stage2_d114_g0_source_bundle import build_d114_g0_source_bundle  # noqa: E402
from cvsrffi.stage2_d114_hbpd_g1 import (  # noqa: E402
    ARMS,
    audit_d114_g1_states,
    score_d114_g1_arms,
)
from cvsrffi.stage2_d114_hbpd_qknn import (  # noqa: E402
    audit_d114_state,
    fit_d114_state,
)
from cvsrffi.stage2_zid_student_t_qknn import build_typed_zid_support_bank  # noqa: E402


EFFECT_PAIRS = {
    "DA_AT_BASE": ("M_DA", "M0"),
    "HEAD_AT_BASE": ("M_HEAD", "M0"),
    "DA_AT_HEAD": ("M_JOINT", "M_HEAD"),
    "JOINT_VS_M0": ("M_JOINT", "M0"),
}
CANDIDATE_ID = "D114_HBPD_QKNN"
PREDICTION_SCHEMA = "cvs.d114.hbpd_qknn.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d114.hbpd_qknn.sourceheld.scores.v1"


class D114G1RunnerError(ValueError):
    pass


def _configure_shared_scorer() -> None:
    d112.ARMS = ARMS
    d112.EFFECT_PAIRS = EFFECT_PAIRS
    d112.CANDIDATE_ID = CANDIDATE_ID
    d112.PREDICTION_SCHEMA = PREDICTION_SCHEMA
    d112.SCORE_SCHEMA = SCORE_SCHEMA


def prepare(args: argparse.Namespace) -> int:
    return d112.prepare(args)


def _d114_bundles(
    args: argparse.Namespace,
    receivers: Sequence[str],
    by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for receiver in receivers:
        locks = tuple(
            d106._lock(k, str(by_key[(receiver, k)]["sha256"])).lock_digest
            for k in d110.K_VALUES
        )
        result[receiver] = build_d114_g0_source_bundle(
            args.d106_tap_archive.resolve(strict=True),
            receipt_path=args.d106_tap_receipt.resolve(strict=True),
            checkpoint_sha256=args.checkpoint_sha256,
            expected_tap_sha256=args.d106_tap_archive_sha256,
            allowed_config_lock_digests=locks,
        )
    return result


def predict(args: argparse.Namespace) -> int:
    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    run_id = d112._run_id(args.run_id)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D114 G1 output exists: {output}")
    manifest_path = root / "package_manifest.json"
    manifest = d112._read_json(manifest_path)
    receivers = tuple(str(value) for value in manifest.get("receiver_ids", ()))
    classes = tuple(str(value) for value in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    truth_seal_sha = manifest.get("truth_input_seal_sha256")
    if (
        manifest.get("schema") != d112.PACKAGE_SCHEMA
        or manifest.get("candidate_id") != d110.CANDIDATE_ID
        or manifest.get("split_id") != d112.SPLIT_ID
        or manifest.get("query_truth_present") is not False
        or manifest.get("target_access") is not False
        or len(receivers) != 7
        or len(classes) != 6
        or not isinstance(packages, list)
        or len(packages) != 21
        or not isinstance(truth_seal_sha, str)
        or len(truth_seal_sha) != 64
    ):
        raise D114G1RunnerError("D114 package manifest closure drift")
    by_key = {(str(row["held_receiver"]), int(row["K"])): row for row in packages}
    if set(by_key) != {(receiver, k) for receiver in receivers for k in d110.K_VALUES}:
        raise D114G1RunnerError("D114 package matrix is not complete")
    head_bundle = d112._g1_bundle(args, str(truth_seal_sha))
    d114_bundles = _d114_bundles(args, receivers, by_key)
    if tuple(head_bundle.class_registry) != classes or any(
        tuple(bundle.class_registry) != classes for bundle in d114_bundles.values()
    ):
        raise D114G1RunnerError("D114 Phase1/package registry drift")

    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    cache: dict[tuple[str, int], tuple[Any, ...]] = {}
    rows = []
    for receiver, held_class, k_shot in d110.fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = by_key[key]
        if key not in cache:
            support, labels, query, query_ids, package_sha = d110._load_package(
                root, package_row, classes=classes, k_shot=k_shot
            )
            bank = build_typed_zid_support_bank(
                support, labels, classes, config=d106._lock(k_shot, package_sha)
            )
            head = fit_d112_ground_head_source_held_g1_state(head_bundle, bank)
            hbpd = fit_d114_state(d114_bundles[receiver], bank)
            cache[key] = (query, query_ids, package_sha, bank, head, hbpd)
        query, query_ids, package_sha, bank, head, hbpd = cache[key]
        logits = score_d114_g1_arms(hbpd, head, bank, query)
        arm_predictions = {arm: d106._argmax(logits[arm], classes) for arm in ARMS}
        audits = {
            "D112_HEAD": d112._jsonable(audit_d112_seam_state(head)),
            "D114_HBPD": d112._jsonable(audit_d114_state(hbpd)),
            "D114_G1": d112._jsonable(audit_d114_g1_states(hbpd, head, bank)),
        }
        if audits["D114_G1"]["query_rows_used_for_fit"] != 0:
            raise D114G1RunnerError("D114 query lifecycle drift")
        row = {
            "schema": PREDICTION_SCHEMA + ".row",
            "candidate_id": CANDIDATE_ID,
            "split_id": d112.SPLIT_ID,
            "run_id": run_id,
            "held_receiver": receiver,
            "held_class": held_class,
            "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "registered_classes": list(classes),
            "query_physical_ids": query_ids,
            "arm_predictions": arm_predictions,
            "shared_component_receipts": {
                "package_sha256": package_sha,
                "d106_tap_archive_sha256": args.d106_tap_archive_sha256,
                "d106_tap_receipt_sha256": d112._file_sha(args.d106_tap_receipt.resolve(strict=True)),
                "d112_bundle_content_root_sha256": head_bundle.manifest["content_root_sha256"],
                "d114_bundle_content_sha256": d114_bundles[receiver].content_sha256,
                "student_t_lock_sha256": bank.config.lock_digest,
                "arm_state_audits": audits,
            },
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_state_updates": 0,
        }
        row["prediction_receipt_sha256"] = d112._sha(row)
        path = row_root / f"{d112._sha({'receiver': receiver, 'held_class': held_class, 'K': k_shot})}.json"
        d112._write_new(path, row)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": str(package_row["package_id"]),
                "path": str(Path("rows") / path.name),
                "sha256": d112._file_sha(path),
                "prediction_receipt_sha256": row["prediction_receipt_sha256"],
            }
        )
    if len(rows) != 63 or len({row["prediction_receipt_sha256"] for row in rows}) != 63:
        raise D114G1RunnerError("D114 prediction coverage did not close at 63 rows")
    result = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": d112.SPLIT_ID,
        "run_id": run_id,
        "arms": list(ARMS),
        "row_count": 63,
        "arm_row_prediction_unit_count": 63 * len(ARMS),
        "rows": rows,
        "package_manifest_sha256": d112._file_sha(manifest_path),
        "truth_input_seal_sha256": truth_seal_sha,
        "d112_bundle_content_root_sha256": head_bundle.manifest["content_root_sha256"],
        "query_truth_access": False,
        "target_access": False,
        "query_state_updates": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = d112._sha(result)
    d112._write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def score(args: argparse.Namespace) -> int:
    _configure_shared_scorer()
    final_path = args.output_json.resolve()
    pairwise_path = final_path.with_name(final_path.stem + ".pairwise" + final_path.suffix)
    if final_path.exists() or pairwise_path.exists():
        raise FileExistsError("immutable D114 score output exists")
    pairwise_args = argparse.Namespace(**vars(args))
    pairwise_args.output_json = pairwise_path
    d112.score(pairwise_args)
    result = _add_factorial_interaction(d112._read_json(pairwise_path))
    result["pairwise_base_score_sha256"] = d112._file_sha(pairwise_path)
    result["score_set_receipt_sha256"] = d112._sha(result)
    d112._write_new(final_path, result)
    print(final_path)
    return 0


def _add_factorial_interaction(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result.pop("score_set_receipt_sha256", None)
    negative = {metric: 0 for metric in d112.METRIC_NAMES}
    rows = result.get("performance_rows")
    if not isinstance(rows, list) or len(rows) != 63:
        raise D114G1RunnerError("D114 score rows are incomplete")
    for row in rows:
        metrics = row.get("arm_metrics", {})
        if set(metrics) != set(ARMS):
            raise D114G1RunnerError("D114 factorial arm closure drift")
        interaction = {}
        for metric in d112.METRIC_NAMES:
            values = [metrics[arm].get(metric) for arm in ARMS]
            delta = (
                None
                if any(item is None for item in values)
                else float(metrics["M_JOINT"][metric])
                - float(metrics["M_HEAD"][metric])
                - float(metrics["M_DA"][metric])
                + float(metrics["M0"][metric])
            )
            interaction[metric] = delta
            if delta is not None and delta < 0.0:
                negative[metric] += 1
        row["same_row_effects"]["FACTORIAL_INTERACTION"] = interaction
    result["negative_tail_row_counts"]["FACTORIAL_INTERACTION"] = negative
    result["factorial_interaction_formula"] = "(M_JOINT-M_HEAD)-(M_DA-M0)"
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return d112.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _configure_shared_scorer()
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
