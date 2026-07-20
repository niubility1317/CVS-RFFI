#!/usr/bin/env python3
"""Run matched D81/D99/D100 K1/K10-new20 development rows on sealed data."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


def _bootstrap_cpu_threads(argv: list[str]) -> int:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == "--cpu-threads":
            if index + 1 >= len(argv):
                raise SystemExit("--cpu-threads requires a value")
            values.append(argv[index + 1])
        elif token.startswith("--cpu-threads="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1:
        raise SystemExit("--cpu-threads must be specified exactly once")
    threads = 2 if not values else int(values[0])
    if threads <= 0:
        raise SystemExit("--cpu-threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ["CVSRFFI_CPU_THREADS"] = str(threads)
    os.environ["CVSRFFI_CPU_INTEROP_THREADS"] = "1"
    return threads


_BOOTSTRAP_CPU_THREADS = _bootstrap_cpu_threads(sys.argv[1:])

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99  # noqa: E402
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT  # noqa: E402
from cvsrffi.somph_offline_target_package import (  # noqa: E402
    build_somph_offline_row_pair,
    finalize_registration_pair_manifest,
    finalize_somph_apply_package,
)
from cvsrffi.somph_predictor_entry import run_somph_enrollment  # noqa: E402
from cvsrffi.somph_runtime_request import SOMPH_ENROLLMENT_REQUEST_SCHEMA  # noqa: E402
from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair  # noqa: E402
from cvsrffi.stage2_d99_d100_query_evaluation import (  # noqa: E402
    CANDIDATES,
    run_d99_d100_query_evaluation,
    typed_class_binding_payload,
)
from scripts.run_cvs_somph_diag_row_pipeline import (  # noqa: E402
    FORMAL_QUERY_PER_TX,
    SUPPORT_BATCH_SIZE,
    _write_json_new,
)
from scripts.run_d99_d100_phase1_lodo import (  # noqa: E402
    _parse_base_d99_lock as _parse_development_d99_prior,
)


SCHEMA = "cvs.phase2.d99_d100.narrow_runner.v1"
GROUND_NPZ_MEMBERS = {
    "codes_qint8",
    "scales_fp16",
    "domain_class_mask",
    "physical_sample_count_floor_uint16",
    "domain_ids",
    "ground_old_registry",
}
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class D99D100NarrowRunnerError(ValueError):
    pass


def _read_json_snapshot(path: str | Path, name: str) -> tuple[dict[str, Any], bytes, str]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise D99D100NarrowRunnerError(f"{name} must be a regular file")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99D100NarrowRunnerError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise D99D100NarrowRunnerError(f"{name} must be a JSON object")
    import hashlib

    return value, raw, hashlib.sha256(raw).hexdigest()


def _read_json(path: str | Path, name: str) -> dict[str, Any]:
    return _read_json_snapshot(path, name)[0]


def _typed_class_binding_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the historical D19 v1 binding to the typed v2 column contract."""

    try:
        return typed_class_binding_payload(payload)
    except ValueError as exc:
        raise D99D100NarrowRunnerError(str(exc)) from exc


def _load_ground_bundle(
    npz_path: str | Path, manifest_path: str | Path
) -> d99.Phase1GroundAggregateBundle:
    source = Path(npz_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise D99D100NarrowRunnerError("ground bundle NPZ must be a regular file")
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != GROUND_NPZ_MEMBERS:
                raise D99D100NarrowRunnerError("ground bundle NPZ member drift")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError) as exc:
        raise D99D100NarrowRunnerError("ground bundle NPZ cannot be loaded") from exc
    manifest = _read_json(manifest_path, "ground bundle manifest")
    try:
        receipt_payload = dict(manifest["aggregation_receipt"])
    except (KeyError, TypeError) as exc:
        raise D99D100NarrowRunnerError("ground aggregation receipt is missing") from exc
    if set(receipt_payload) != {field.name for field in fields(d99.ExternalGroundAggregationReceipt)}:
        raise D99D100NarrowRunnerError("ground aggregation receipt field drift")
    receipt = d99.ExternalGroundAggregationReceipt(**receipt_payload)
    return d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=arrays["codes_qint8"],
        scales_fp16=arrays["scales_fp16"],
        domain_class_mask=arrays["domain_class_mask"],
        physical_sample_count_floor_uint16=arrays[
            "physical_sample_count_floor_uint16"
        ],
        domain_ids=arrays["domain_ids"].astype(str).tolist(),
        ground_old_registry=arrays["ground_old_registry"].astype(str).tolist(),
        aggregation_receipt=receipt,
    )


def _load_d99_lock(path: str | Path) -> d99.Phase1D99Lock:
    value = _read_json(path, "base D99 lock")
    try:
        return _parse_development_d99_prior(value, "development_diagnostic")
    except (TypeError, ValueError) as exc:
        raise D99D100NarrowRunnerError(str(exc)) from exc


_THREADPOOL_LIMITER: Any = None


def _configure_threads(count: int) -> dict[str, Any]:
    if int(count) <= 0:
        raise D99D100NarrowRunnerError("cpu_threads must be positive")
    requested = int(count)
    configured = {name: str(requested) for name in _THREAD_VARIABLES}
    configured["CVSRFFI_CPU_THREADS"] = str(requested)
    configured["CVSRFFI_CPU_INTEROP_THREADS"] = "1"
    os.environ.update(configured)
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(requested)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise D99D100NarrowRunnerError("torch interop thread pool is already incompatible")
    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=requested)
    return {
        "requested_cpu_threads": requested,
        "environment": configured,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "threadpool_info": [dict(row) for row in threadpool_info()],
        "thread_environment_set_before_numpy_torch_import": bool(
            _BOOTSTRAP_CPU_THREADS == requested
        ),
    }


def _enrollment_request(package_seal_sha256: str, device: str) -> dict[str, Any]:
    return {
        "schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA,
        "package_seal_sha256": package_seal_sha256,
        "head_output_leaf": "head_capsule.npz",
        "device": device,
        "support_batch_size": SUPPORT_BATCH_SIZE,
        **PHASE2_FULL_CONTRACT,
    }


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0.0 else 2.0 * left * right / (left + right)


def _read_prediction(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).astype(str) for name in payload.files}


def _detailed_score(
    before_path: str | Path, after_path: str | Path, truth_path: str | Path
) -> dict[str, Any]:
    before = _read_prediction(before_path)
    after = _read_prediction(after_path)
    truth_payload = _read_json(truth_path, "truth sidecar")
    truth = {str(row["query_token"]): row for row in truth_payload["rows"]}
    if not set(before["query_tokens"].tolist()).issubset(
        set(after["query_tokens"].tolist())
    ):
        raise D99D100NarrowRunnerError("before query tokens are not an after subset")
    output_rows = []
    for scenario in sorted(set(after["scenarios"].tolist())):
        before_mask = before["scenarios"] == scenario
        after_mask = after["scenarios"] == scenario
        before_tokens = before["query_tokens"][before_mask]
        after_tokens = after["query_tokens"][after_mask]
        before_pred = before["predicted_class_handles"][before_mask]
        after_pred = after["predicted_class_handles"][after_mask]
        before_truth = np.asarray([truth[token]["true_class_handle"] for token in before_tokens])
        after_truth = np.asarray([truth[token]["true_class_handle"] for token in after_tokens])
        roles = np.asarray([truth[token]["evaluation_role"] for token in after_tokens])
        old_mask = roles == "target_old"
        new_mask = roles == "target_new"
        old_before = float(np.mean(before_pred == before_truth))
        old_after = float(np.mean(after_pred[old_mask] == after_truth[old_mask]))
        new_acc = float(np.mean(after_pred[new_mask] == after_truth[new_mask]))
        classes = sorted(set(after_truth.tolist()))
        per_class = {
            label: float(np.mean(after_pred[after_truth == label] == label))
            for label in classes
        }
        old_classes = sorted(set(after_truth[old_mask].tolist()))
        new_classes = sorted(set(after_truth[new_mask].tolist()))
        confusion = {
            truth_label: {
                predicted_label: int(
                    np.sum((after_truth == truth_label) & (after_pred == predicted_label))
                )
                for predicted_label in classes
            }
            for truth_label in classes
        }
        output_rows.append(
            {
                "scenario": scenario,
                "old_acc_before_increment": old_before,
                "old_acc_after_increment": old_after,
                "average_forgetting": old_before - old_after,
                "min_old_class_acc_after": min(per_class[label] for label in old_classes),
                "min_new_class_acc_after": min(per_class[label] for label in new_classes),
                "min_registered_class_acc_after": min(per_class.values()),
                "registered_balanced_accuracy_after": float(
                    np.mean(list(per_class.values()))
                ),
                "seen_new_acc": new_acc,
                "H_old_new": _harmonic(old_after, new_acc),
                "per_class_accuracy_after": per_class,
                "after_all_confusion_matrix_counts": confusion,
            }
        )
    return {"schema": "cvs.phase2.d99_d100.detailed_score.v1", "rows": output_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.k_shot not in (1, 10) or args.new_count != 20:
        raise D99D100NarrowRunnerError("narrow matrix is fixed to K1/K10 and new20")
    threads = _configure_threads(args.cpu_threads)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    offline = output / "offline"
    control = output / "control"
    enrollment_root = output / "somph_enrollment"
    apply_seal_root = output / "apply_seals"
    prediction_root = output / "predictions"
    score_root = output / "scores"
    for directory in (control, enrollment_root, apply_seal_root, score_root):
        directory.mkdir(parents=True, exist_ok=False)

    build = build_somph_offline_row_pair(
        cache_set_manifest_path=args.cache_manifest,
        authority_bundle_root=args.authority_bundle,
        expected_authority_commit_sha256=args.authority_commit_sha256,
        phase1_checkpoint_path=args.phase1_checkpoint,
        sealed_feature_runtime_path=args.sealed_runtime,
        method_lock_path=args.method_lock,
        output_root=offline,
        receiver=args.receiver,
        seed=args.seed,
        k_shot=args.k_shot,
        new_class_count=args.new_count,
        query_per_tx=FORMAL_QUERY_PER_TX,
    )
    runtime: dict[str, dict[str, Any]] = {}
    for state in ("before", "after"):
        state_build = build["states"][state]
        request_path = control / f"{state}_enrollment_request.json"
        request_sha = _write_json_new(
            request_path,
            _enrollment_request(
                state_build["enrollment_package_seal_sha256"], args.device
            ),
        )
        head_root = enrollment_root / state
        head_root.mkdir(parents=True, exist_ok=False)
        enrollment = run_somph_enrollment(
            request_json=request_path,
            package_root=state_build["enrollment_package_root"],
            detached_seal_path=state_build["enrollment_package_seal"],
            expected_seal_sha256=state_build["enrollment_package_seal_sha256"],
            output_root=head_root,
        )
        apply_seal = apply_seal_root / f"{state}_apply.seal.json"
        finalized = finalize_somph_apply_package(
            apply_staging_root=state_build["apply_staging_root"],
            detached_seal_path=apply_seal,
            staging_authority_path=state_build["apply_staging_authority"],
            staging_authority_seal_path=state_build["apply_staging_authority_seal"],
            expected_staging_authority_seal_sha256=state_build[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head_root / enrollment["head_output_leaf"],
            expected_head_capsule_sha256=enrollment["head_capsule_sha256"],
            expected_head_enrollment_binding_sha256=enrollment[
                "enrollment_binding_sha256"
            ],
            authority_bundle_root=args.authority_bundle,
            expected_authority_commit_sha256=args.authority_commit_sha256,
        )
        runtime[state] = {
            "request_sha256": request_sha,
            "enrollment": enrollment,
            "apply_package_root": state_build["apply_staging_root"],
            "apply_seal_path": str(apply_seal),
            "apply": finalized,
        }
    pair_path = score_root / "registration_pair.final.json"
    pair = finalize_registration_pair_manifest(
        staging_manifest_path=build["registration_pair_manifest"],
        output_path=pair_path,
        before_binding_sha256=runtime["before"]["enrollment"]["enrollment_binding_sha256"],
        after_binding_sha256=runtime["after"]["enrollment"]["enrollment_binding_sha256"],
    )

    bundle = _load_ground_bundle(args.d99_ground_bundle_npz, args.d99_ground_manifest)
    _ground_manifest_payload, _ground_manifest_raw, ground_manifest_sha = (
        _read_json_snapshot(args.d99_ground_manifest, "D99 ground manifest")
    )
    base_lock = _load_d99_lock(args.base_d99_lock)
    lodo = _read_json(args.phase1_lodo_json, "Phase1 LODO receipt")
    raw_class_binding, class_binding_raw, class_binding_sha = _read_json_snapshot(
        args.class_binding_json, "D19 class binding"
    )
    if class_binding_sha != str(args.class_binding_sha256).lower():
        raise D99D100NarrowRunnerError("D19 class binding path/SHA drift")
    class_binding = _typed_class_binding_payload(raw_class_binding)
    result = run_d99_d100_query_evaluation(
        before_enrollment_package_root=build["states"]["before"]["enrollment_package_root"],
        before_enrollment_seal_path=build["states"]["before"]["enrollment_package_seal"],
        before_enrollment_seal_sha256=build["states"]["before"]["enrollment_package_seal_sha256"],
        before_apply_package_root=runtime["before"]["apply_package_root"],
        before_apply_seal_path=runtime["before"]["apply_seal_path"],
        before_apply_seal_sha256=runtime["before"]["apply"]["package_seal_sha256"],
        after_enrollment_package_root=build["states"]["after"]["enrollment_package_root"],
        after_enrollment_seal_path=build["states"]["after"]["enrollment_package_seal"],
        after_enrollment_seal_sha256=build["states"]["after"]["enrollment_package_seal_sha256"],
        after_apply_package_root=runtime["after"]["apply_package_root"],
        after_apply_seal_path=runtime["after"]["apply_seal_path"],
        after_apply_seal_sha256=runtime["after"]["apply"]["package_seal_sha256"],
        d81_ground_component_dir=args.d81_ground_component_dir,
        d81_ground_manifest_sha256=args.d81_ground_manifest_sha256,
        d99_ground_bundle=bundle,
        d99_ground_manifest_sha256=ground_manifest_sha,
        base_d99_config=base_lock,
        phase1_lodo_receipt=lodo,
        class_binding_payload=class_binding,
        class_binding_bytes=class_binding_raw,
        class_binding_sha256=class_binding_sha,
        class_binding_source_schema=str(raw_class_binding.get("schema", "")),
        phase2_authority_sha256=args.authority_commit_sha256,
        output_root=prediction_root,
        device=args.device,
    )
    scores: dict[str, Any] = {}
    for candidate in CANDIDATES:
        before_path = result["candidates"][candidate]["before"]["prediction_path"]
        after_path = result["candidates"][candidate]["after"]["prediction_path"]
        score = score_diag_cosine_pair(
            before_prediction_path=before_path,
            after_prediction_path=after_path,
            truth_sidecar_path=build["truth_sidecar"],
            output_path=score_root / f"{candidate}.score.json",
            candidate=candidate,
        )
        detailed = _detailed_score(before_path, after_path, build["truth_sidecar"])
        detail_sha = _write_json_new(score_root / f"{candidate}.detailed.json", detailed)
        scores[candidate] = {
            "score_path": str(score_root / f"{candidate}.score.json"),
            "score_artifact_sha256": score["score_artifact_sha256"],
            "detailed_score_path": str(score_root / f"{candidate}.detailed.json"),
            "detailed_score_sha256": detail_sha,
            "old_acc_before_increment": score["before"]["old_acc"],
            "old_acc_after_increment": score["after"]["old_acc"],
            "min_old_class_acc_after": score["per_old_class_floor_after"],
            "min_new_class_acc_after": min(
                row["min_new_class_acc_after"] for row in detailed["rows"]
            ),
            "min_registered_class_acc_after": min(
                row["min_registered_class_acc_after"] for row in detailed["rows"]
            ),
            "registered_balanced_accuracy_after": float(
                np.mean(
                    [
                        row["registered_balanced_accuracy_after"]
                        for row in detailed["rows"]
                    ]
                )
            ),
            "seen_new_acc": score["after"]["seen_new_acc"],
            "H_old_new": score["after"]["h_old_new"],
            "old_forgetting_pp": score["old_forgetting_pp"],
        }
    receipt = {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_NARROW_COMPLETE_NONFORMAL",
        "receiver": args.receiver,
        "seed": args.seed,
        "k_shot": args.k_shot,
        "new_class_count": args.new_count,
        "device": args.device,
        "cpu_thread_env": threads,
        "row_handle": build["row_handle"],
        "row_manifest_sha256": build["row_manifest_sha256"],
        "registration_pair_schema": pair["schema"],
        "truth_join_started_after_all_predictions": True,
        "truth_fed_back_to_predictor": False,
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "candidates": scores,
        "prediction_result": result,
    }
    receipt_path = output / "narrow_receipt.json"
    receipt_sha = _write_json_new(receipt_path, receipt)
    return {
        "schema": SCHEMA,
        "status": receipt["status"],
        "output_root": str(output),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "candidates": scores,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-manifest", required=True)
    result.add_argument("--authority-bundle", required=True)
    result.add_argument("--authority-commit-sha256", required=True)
    result.add_argument("--phase1-checkpoint", required=True)
    result.add_argument("--sealed-runtime", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--d81-ground-component-dir", required=True)
    result.add_argument("--d81-ground-manifest-sha256", required=True)
    result.add_argument("--d99-ground-bundle-npz", required=True)
    result.add_argument("--d99-ground-manifest", required=True)
    result.add_argument("--base-d99-lock", required=True)
    result.add_argument("--phase1-lodo-json", required=True)
    result.add_argument("--class-binding-json", required=True)
    result.add_argument("--class-binding-sha256", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--receiver", required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--k-shot", type=int, choices=(1, 10), required=True)
    result.add_argument("--new-count", type=int, choices=(20,), default=20)
    result.add_argument("--device", default="cuda")
    result.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    value = run(parser().parse_args())
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
