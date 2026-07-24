#!/usr/bin/env python3
"""Thin full125 executor for ``GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed``.

This module deliberately owns only GRB's formal lifecycle and five-arm
artifacts.  The shared ADV3B02 runner owns subprocess identity, GPU dispatch,
health stopping, immutable launcher logs and partial-completion accounting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_adv3b02_deployment_bundle import (
    COMPONENT_PROFILE_GRB_JP4_Q4,
    load_formal_adv3b02_deployment_bundle,
)
from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (
    CANDIDATE,
    FIVE_ARM_NAMES,
    SCHEMA,
    append_formal_stage2_c,
    fit_stage2_b_from_support_iq,
    predict_five_arms,
)
from scripts import run_adv3b02_ts_drqknn_bcrr_125 as shared


ARMS = FIVE_ARM_NAMES
RECEIVERS = shared.RECEIVERS
SEEDS = shared.SEEDS
SLICES = shared.SLICES
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
MATRIX_COUNTS = {
    "jobs": 125,
    "scene_slices": 375,
    "score_rows": 1875,
    "arm_state_prediction_artifacts": 1250,
}
LAUNCHER_SCHEMA = "cvs.stage2.grb_jp4_adv_drqknn_bcrr.full125.r1"


class GRBRunnerError(shared.ADV3B02LauncherError):
    pass


def job_id(receiver: str, seed: int, k_shot: int, new_class_count: int) -> str:
    return "grb_jp4_r1_rx_%s_s_%s_k_%s_n_%s" % (
        receiver, seed, k_shot, new_class_count
    )


def matrix_jobs(a: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        leaf = "rx_" + receiver.replace("-", "_")
        for seed in SEEDS:
            for k_shot, new_count in SLICES:
                ident = job_id(receiver, seed, k_shot, new_count)
                jobs.append(
                    {
                        "job_id": ident,
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "cache_manifest": str(
                            Path(a.cache_root) / leaf / f"seed_{seed}" / "cache_set.json"
                        ),
                        "authority_bundle": str(
                            Path(a.authority_root)
                            / f"authority_bundle_{leaf}_seed_{seed}"
                        ),
                        "output_root": str(Path(a.run_root) / "jobs" / ident),
                    }
                )
    if len(jobs) != MATRIX_COUNTS["jobs"] or len({job["job_id"] for job in jobs}) != len(jobs):
        raise GRBRunnerError("frozen GRB full125 job cardinality drift")
    return sorted(
        jobs,
        key=lambda item: (
            -(item["k_shot"] * (10 + item["new_class_count"])),
            -item["new_class_count"],
            item["job_id"],
        ),
    )


def _bundle_kwargs(a: argparse.Namespace) -> dict[str, str]:
    return {
        "detached_seal_path": a.grb_detached_seal,
        "expected_detached_seal_sha256": a.grb_expected_detached_seal_sha256,
        "signature_envelope_path": a.grb_signature_envelope,
        "expected_signature_envelope_sha256": a.grb_expected_signature_envelope_sha256,
        "expected_checkpoint_lineage_sha256": a.grb_expected_checkpoint_lineage_sha256,
        "expected_runtime_sha256": a.grb_expected_runtime_sha256,
        "expected_component_pre_sign_content_root_sha256": a.grb_expected_component_pre_sign_content_root_sha256,
        "expected_class_handle_binding_sha256": a.grb_expected_class_handle_binding_sha256,
        "expected_parity_receipt_sha256": a.grb_expected_parity_receipt_sha256,
        "expected_generation_lock_sha256": a.grb_expected_generation_lock_sha256,
        "expected_method_lock_sha256": a.grb_expected_method_lock_sha256,
        "expected_generation_config_sha256": a.grb_expected_generation_config_sha256,
        "expected_generation_code_sha256": a.grb_expected_generation_code_sha256,
        "expected_outer_content_root_sha256": a.grb_expected_outer_content_root_sha256,
    }


def _load_grb_bundle(a: argparse.Namespace) -> Any:
    bundle = load_formal_adv3b02_deployment_bundle(
        a.grb_outer_bundle, **_bundle_kwargs(a)
    )
    context = bundle.formal_phase2_context
    if (
        context.get("formal_phase2_eligible") is not True
        or context.get("component_profile") != COMPONENT_PROFILE_GRB_JP4_Q4
        or context.get("method_lock_sha256") != a.grb_expected_method_lock_sha256
        or context.get("runtime_sha256") != a.grb_expected_runtime_sha256
    ):
        raise GRBRunnerError("production-signed GRB outer bundle binding drift")
    return bundle


def _authority_surfaces(a: argparse.Namespace, output: Path) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Materialize only the already-authorized support/query surfaces."""
    from cvsrffi.somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc

    a.dssc_method_lock = a.package_method_lock
    try:
        build, runtime = dssc._build_finalized_packages(a, output)
        before_payload, before_manifest, _ = load_verified_somph_predictor_bundle(
            runtime["before"]["enrollment"]["enrollment_package_root"],
            detached_seal_path=runtime["before"]["enrollment"]["enrollment_package_seal"],
            expected_seal_sha256=runtime["before"]["enrollment"]["enrollment_package_seal_sha256"],
        )
        after_payload, after_manifest, _ = load_verified_somph_predictor_bundle(
            runtime["after"]["enrollment"]["enrollment_package_root"],
            detached_seal_path=runtime["after"]["enrollment"]["enrollment_package_seal"],
            expected_seal_sha256=runtime["after"]["enrollment"]["enrollment_package_seal_sha256"],
        )
    except Exception as exc:
        raise shared.ADV3B02P0Error(
            "AUTHORITY_OR_PACKAGE_BINDING_FAILURE",
            "authority-built support/query materialization failed",
        ) from exc
    old_registry = dssc._registry(before_manifest)
    registry = dssc._registry(after_manifest)
    if tuple(registry[: len(old_registry)]) != tuple(old_registry):
        raise shared.ADV3B02P0Error(
            "REGISTRY_PROTOCOL_DRIFT", "authority before/after registry append drift"
        )
    return build, {"before": before_payload, "after": after_payload}, old_registry, registry


def _publish_state_predictions(
    output: Path,
    state: str,
    rows: Mapping[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
) -> dict[str, str]:
    published: dict[str, str] = {}
    for arm in ARMS:
        parts = rows[arm]
        if not parts:
            raise GRBRunnerError("GRB state has no prediction rows")
        published[arm] = shared.write_prediction_new(
            shared.prediction_path(output, state, arm, allowed_arms=ARMS),
            query_tokens=np.concatenate([part[0] for part in parts]).astype(str),
            scenarios=np.concatenate([part[1] for part in parts]).astype(str),
            predicted_class_handles=np.concatenate([part[2] for part in parts]).astype(str),
        )
    return published


def _state_prediction(
    state: Any, *, query_iq: np.ndarray, query_tokens: tuple[str, ...]
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    import torch

    _logits, predictions, closure = predict_five_arms(
        state,
        query_iq=torch.from_numpy(np.ascontiguousarray(query_iq, dtype=np.float32)),
        query_physical_tokens=query_tokens,
    )
    if tuple(predictions) != ARMS or closure.get("query_rows_used_for_fit") != 0:
        raise GRBRunnerError("formal GRB five-arm query closure drift")
    return predictions, closure


def _support_tensor(value: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


def run_row(a: argparse.Namespace) -> Mapping[str, Any]:
    from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc

    if (a.k_shot, a.new_class_count) not in SLICES:
        raise shared.ADV3B02P0Error("MATRIX_PROTOCOL_DRIFT", "row is outside frozen GRB full125")
    output = Path(a.output_root)
    if output.exists():
        raise shared.ADV3B02P0Error("OUTPUT_OVERWRITE", "GRB row output must be fresh")
    output.mkdir(parents=True)
    build, payloads, old_registry, registry = _authority_surfaces(a, output)
    # Formal states are intentionally fresh per before/after query because a
    # query consumes its runtime.  The first reverified bundle is checked and
    # immediately consumed by that first fit; no extra live runtime is held.
    rows = {state: {arm: [] for arm in ARMS} for state in ("before", "after")}
    lifecycle: list[Mapping[str, Any]] = []
    for scene in SCENES:
        old_iq, old_labels, old_tokens = dssc._support(payloads["before"][scene], old_registry, a.k_shot)
        before_query, before_tokens = dssc._query(payloads["before"][scene])
        before_bundle = _load_grb_bundle(a)
        if scene == SCENES[0] and tuple(before_bundle.component.class_registry) != tuple(old_registry):
            raise shared.ADV3B02P0Error("REGISTRY_PROTOCOL_DRIFT", "GRB outer bundle/old registry drift")
        before_state = fit_stage2_b_from_support_iq(
            bundle=before_bundle,
            support_iq=_support_tensor(old_iq),
            support_labels=old_labels,
            support_physical_tokens=old_tokens,
        )
        before_predictions, before_closure = _state_prediction(
            before_state, query_iq=before_query, query_tokens=before_tokens
        )
        for arm in ARMS:
            rows["before"][arm].append((np.asarray(before_tokens), np.asarray([scene] * len(before_tokens)), np.asarray(before_predictions[arm])))

        # The same signed outer package is reverified/re-materialized for the
        # independent S_B->S_C lifecycle; no sidecar or caller runtime is used.
        after_bundle = _load_grb_bundle(a)
        all_iq, all_labels, all_tokens = dssc._support(payloads["after"][scene], registry, a.k_shot)
        after_query, after_tokens = dssc._query(payloads["after"][scene])
        new_mask = np.asarray([label not in set(old_registry) for label in all_labels], dtype=bool)
        if int(new_mask.sum()) != a.new_class_count * a.k_shot:
            raise shared.ADV3B02P0Error("SUPPORT_PROTOCOL_DRIFT", "authority GRB new-class support drift")
        after_b = fit_stage2_b_from_support_iq(
            bundle=after_bundle,
            support_iq=_support_tensor(old_iq),
            support_labels=old_labels,
            support_physical_tokens=old_tokens,
        )
        after_state = append_formal_stage2_c(
            after_b,
            old_support_iq=_support_tensor(old_iq),
            old_support_labels=old_labels,
            old_support_physical_tokens=old_tokens,
            new_support_iq=_support_tensor(all_iq[new_mask]),
            new_support_labels=tuple(label for label, keep in zip(all_labels, new_mask) if keep),
            new_registered_classes=tuple(label for label in registry if label not in set(old_registry)),
            new_support_physical_tokens=tuple(token for token, keep in zip(all_tokens, new_mask) if keep),
        )
        after_predictions, after_closure = _state_prediction(
            after_state, query_iq=after_query, query_tokens=after_tokens
        )
        for arm in ARMS:
            rows["after"][arm].append((np.asarray(after_tokens), np.asarray([scene] * len(after_tokens)), np.asarray(after_predictions[arm])))
        lifecycle.append({"scene": scene, "before": before_closure, "after": after_closure})
    publications = {state: _publish_state_predictions(output, state, rows[state]) for state in ("before", "after")}
    scores = shared._score_real_row(
        output, truth_sidecar=build["truth_sidecar"], publications=publications,
        arms=ARMS, candidate=CANDIDATE,
    )
    receipt = {
        "schema": LAUNCHER_SCHEMA,
        "candidate": CANDIDATE,
        "job_id": job_id(a.receiver, a.seed, a.k_shot, a.new_class_count),
        "status": "ROW_ARTIFACTS_COMPLETE",
        "receiver": a.receiver,
        "seed": a.seed,
        "k_shot": a.k_shot,
        "new_class_count": a.new_class_count,
        "prediction_artifact_count": len(ARMS) * 2,
        "scene_slice_count": len(SCENES),
        "score_row_count": len(ARMS) * len(SCENES),
        "query_truth_in_predictor": False,
        "query_rows_used_for_fit": 0,
        "formal_outer_bundle": str(Path(a.grb_outer_bundle).resolve()),
        "five_arm_lifecycle_by_scene": lifecycle,
        "prediction_sha256_by_state_arm": publications,
        "score_artifact_sha256": scores,
    }
    shared.write_json_new(output / "row_receipt.json", receipt)
    validate_row_artifacts({"job_id": receipt["job_id"], "output_root": str(output)}, receipt)
    return receipt


def validate_row_artifacts(job: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    if type(receipt) is not dict or any(
        receipt.get(key) != value
        for key, value in {
            "schema": LAUNCHER_SCHEMA, "candidate": CANDIDATE,
            "job_id": job["job_id"], "status": "ROW_ARTIFACTS_COMPLETE",
            "query_truth_in_predictor": False, "query_rows_used_for_fit": 0,
            "prediction_artifact_count": 10, "scene_slice_count": 3, "score_row_count": 15,
        }.items()
    ):
        raise GRBRunnerError("GRB row receipt identity/cardinality drift")
    root = Path(str(job["output_root"]))
    publications = receipt.get("prediction_sha256_by_state_arm")
    scores = receipt.get("score_artifact_sha256")
    if type(publications) is not dict or set(publications) != {"before", "after"} or type(scores) is not dict or set(scores) != set(ARMS):
        raise GRBRunnerError("GRB prediction/score publication closure drift")
    for state in ("before", "after"):
        if type(publications[state]) is not dict or set(publications[state]) != set(ARMS):
            raise GRBRunnerError("GRB arm-state closure drift")
        identity: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        for arm, digest in publications[state].items():
            path = shared.prediction_path(root, state, arm, allowed_arms=ARMS)
            if not path.is_file() or path.is_symlink() or shared.sha256_file(path) != digest:
                raise GRBRunnerError("GRB immutable prediction closure drift")
            artifact = shared._read_prediction(path)
            current = (tuple(artifact["query_tokens"].astype(str)), tuple(artifact["scenarios"].astype(str)))
            if identity is None:
                identity = current
            elif identity != current:
                raise GRBRunnerError("GRB arms did not share an immutable query surface")
    for arm, digest in scores.items():
        path = shared.score_path(root, arm, allowed_arms=ARMS)
        if not path.is_file() or path.is_symlink() or shared.sha256_file(path) != digest:
            raise GRBRunnerError("GRB immutable score closure drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("candidate") != CANDIDATE or payload.get("arm") != arm:
            raise GRBRunnerError("GRB scorer binding drift")


def validate_matrix_artifacts(run_root: str | Path) -> Mapping[str, Any]:
    root = Path(run_root)
    completion = json.loads((root / "matrix_runtime_completion.json").read_text(encoding="utf-8"))
    if completion.get("candidate") != CANDIDATE or completion.get("status") != "ARTIFACTS_COMPLETE" or completion.get("counts") != MATRIX_COUNTS:
        raise GRBRunnerError("GRB full125 completion closure drift")
    manifest = json.loads((root / "matrix_runtime_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != LAUNCHER_SCHEMA or manifest.get("counts") != MATRIX_COUNTS:
        raise GRBRunnerError("GRB shared matrix manifest drift")
    for job in manifest.get("jobs", []):
        receipt = json.loads((Path(job["output_root"]) / "row_receipt.json").read_text(encoding="utf-8"))
        validate_row_artifacts(job, receipt)
    return {"candidate": CANDIDATE, "status": "ARTIFACTS_COMPLETE", "counts": MATRIX_COUNTS}


def _extra_row_args(a: argparse.Namespace, _job: Mapping[str, Any]) -> list[str]:
    return [
        "--grb-outer-bundle", a.grb_outer_bundle,
        "--grb-detached-seal", a.grb_detached_seal,
        "--grb-expected-detached-seal-sha256", a.grb_expected_detached_seal_sha256,
        "--grb-signature-envelope", a.grb_signature_envelope,
        "--grb-expected-signature-envelope-sha256", a.grb_expected_signature_envelope_sha256,
        "--grb-expected-checkpoint-lineage-sha256", a.grb_expected_checkpoint_lineage_sha256,
        "--grb-expected-runtime-sha256", a.grb_expected_runtime_sha256,
        "--grb-expected-component-pre-sign-content-root-sha256", a.grb_expected_component_pre_sign_content_root_sha256,
        "--grb-expected-class-handle-binding-sha256", a.grb_expected_class_handle_binding_sha256,
        "--grb-expected-parity-receipt-sha256", a.grb_expected_parity_receipt_sha256,
        "--grb-expected-generation-lock-sha256", a.grb_expected_generation_lock_sha256,
        "--grb-expected-method-lock-sha256", a.grb_expected_method_lock_sha256,
        "--grb-expected-generation-config-sha256", a.grb_expected_generation_config_sha256,
        "--grb-expected-generation-code-sha256", a.grb_expected_generation_code_sha256,
        "--grb-expected-outer-content-root-sha256", a.grb_expected_outer_content_root_sha256,
    ]


def run_matrix(a: argparse.Namespace) -> Mapping[str, Any]:
    return shared.run_matrix(
        a,
        row_runner=shared.MatrixRowRunner(
            candidate=CANDIDATE, schema=LAUNCHER_SCHEMA, counts=MATRIX_COUNTS,
            runtime_jobs=matrix_jobs, row_script=str(Path(__file__).resolve()),
            extra_row_args=_extra_row_args, validate_row=validate_row_artifacts,
            validate_matrix=validate_matrix_artifacts,
        ),
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    def common(item: argparse.ArgumentParser) -> None:
        item.add_argument("--phase1-checkpoint", required=True)
        item.add_argument("--sealed-runtime", required=True)
        item.add_argument("--package-method-lock", required=True)
        item.add_argument("--grb-outer-bundle", required=True)
        item.add_argument("--grb-detached-seal", required=True)
        item.add_argument("--grb-expected-detached-seal-sha256", required=True)
        item.add_argument("--grb-signature-envelope", required=True)
        item.add_argument("--grb-expected-signature-envelope-sha256", required=True)
        for name in (
            "checkpoint-lineage", "runtime", "component-pre-sign-content-root",
            "class-handle-binding", "parity-receipt", "generation-lock", "method-lock",
            "generation-config", "generation-code", "outer-content-root",
        ):
            item.add_argument("--grb-expected-" + name + "-sha256", required=True)
    row = sub.add_parser("row"); common(row)
    row.add_argument("--cache-manifest", required=True); row.add_argument("--authority-bundle", required=True)
    row.add_argument("--authority-commit-sha256", required=True); row.add_argument("--output-root", required=True)
    row.add_argument("--receiver", required=True); row.add_argument("--seed", type=int, required=True)
    row.add_argument("--k-shot", type=int, required=True); row.add_argument("--new-class-count", type=int, required=True); row.add_argument("--device", required=True)
    matrix = sub.add_parser("matrix"); common(matrix)
    matrix.add_argument("--cache-root", required=True); matrix.add_argument("--authority-root", required=True)
    matrix.add_argument("--run-root", required=True); matrix.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    return parser


def main() -> int:
    args = parser().parse_args()
    try:
        result = run_row(args) if args.mode == "row" else run_matrix(args)
    except Exception as exc:
        if args.mode == "row":
            marker = shared._row_failure_marker_payload(
                job_id_value=job_id(args.receiver, args.seed, args.k_shot, args.new_class_count),
                exc=exc, prediction_count=shared._count_prediction_artifacts(args.output_root),
                candidate=CANDIDATE,
            )
            sys.stderr.write(shared.ROW_FAILURE_MARKER_PREFIX + shared._canon(marker).decode("utf-8") + "\n")
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
