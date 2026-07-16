#!/usr/bin/env python3
"""Build one formal LEO_weak row pair, seal SOMP-H heads, then run D1 and score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT  # noqa: E402
from cvsrffi.somph_offline_target_package import (  # noqa: E402
    build_somph_offline_row_pair,
    finalize_registration_pair_manifest,
    finalize_somph_apply_package,
)
from cvsrffi.somph_predictor_entry import run_somph_enrollment  # noqa: E402
from cvsrffi.somph_runtime_request import (  # noqa: E402
    SOMPH_ENROLLMENT_REQUEST_SCHEMA,
)
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    CANDIDATES,
    CANDIDATE_D1,
    run_diag_cosine_exploration,
)
from cvsrffi.stage2_diag_cosine_scorer import (  # noqa: E402
    score_diag_cosine_pair,
)
from cvsrffi.stage2_predictor_bundle import sha256_file  # noqa: E402


PIPELINE_SCHEMA = "cvs.phase2.somph_diag_row_pipeline.v1"
FORMAL_QUERY_PER_TX = 20
SUPPORT_BATCH_SIZE = 64


class SomphDiagRowPipelineError(ValueError):
    """Raised when the offline controller cannot preserve row isolation."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(value) + b"\n"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _enrollment_request(*, package_seal_sha256: str, device: str) -> dict[str, Any]:
    return {
        "schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA,
        "package_seal_sha256": package_seal_sha256,
        "head_output_leaf": "head_capsule.npz",
        "device": device,
        "support_batch_size": SUPPORT_BATCH_SIZE,
        **PHASE2_FULL_CONTRACT,
    }


def _require_prediction(path: Path, *, state: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise SomphDiagRowPipelineError(
            f"{state} diag-cosine prediction artifact was not published"
        )
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SomphDiagRowPipelineError(
            f"{state} diag-cosine prediction artifact is not immutable"
        )


def run_pipeline(
    *,
    cache_set_manifest_path: str | Path,
    authority_bundle_root: str | Path,
    expected_authority_commit_sha256: str,
    phase1_checkpoint_path: str | Path,
    sealed_feature_runtime_path: str | Path,
    method_lock_path: str | Path,
    output_root: str | Path,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
    device: str,
    candidate: str = CANDIDATE_D1,
) -> dict[str, Any]:
    """Run one development row without exposing scorer truth to predictors."""

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    offline_root = output / "offline"
    control_root = output / "control"
    enrollment_output_root = output / "somph_enrollment"
    apply_seal_root = output / "apply_seals"
    diag_root = output / "diag"
    scorer_root = output / "scorer"
    for directory in (
        control_root,
        enrollment_output_root,
        apply_seal_root,
        diag_root,
        scorer_root,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    build = build_somph_offline_row_pair(
        cache_set_manifest_path=cache_set_manifest_path,
        authority_bundle_root=authority_bundle_root,
        expected_authority_commit_sha256=expected_authority_commit_sha256,
        phase1_checkpoint_path=phase1_checkpoint_path,
        sealed_feature_runtime_path=sealed_feature_runtime_path,
        method_lock_path=method_lock_path,
        output_root=offline_root,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        new_class_count=new_class_count,
        query_per_tx=FORMAL_QUERY_PER_TX,
    )

    state_runtime: dict[str, dict[str, Any]] = {}
    for state in ("before", "after"):
        state_build = build["states"][state]
        request_path = control_root / f"{state}_enrollment_request.json"
        request_sha256 = _write_json_new(
            request_path,
            _enrollment_request(
                package_seal_sha256=state_build[
                    "enrollment_package_seal_sha256"
                ],
                device=device,
            ),
        )
        head_output = enrollment_output_root / state
        head_output.mkdir(parents=True, exist_ok=False)
        enrollment = run_somph_enrollment(
            request_json=request_path,
            package_root=state_build["enrollment_package_root"],
            detached_seal_path=state_build["enrollment_package_seal"],
            expected_seal_sha256=state_build[
                "enrollment_package_seal_sha256"
            ],
            output_root=head_output,
        )
        head_path = head_output / enrollment["head_output_leaf"]
        apply_seal_path = apply_seal_root / f"{state}_apply.seal.json"
        finalized_apply = finalize_somph_apply_package(
            apply_staging_root=state_build["apply_staging_root"],
            detached_seal_path=apply_seal_path,
            staging_authority_path=state_build["apply_staging_authority"],
            staging_authority_seal_path=state_build[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state_build[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head_path,
            expected_head_capsule_sha256=enrollment[
                "head_capsule_sha256"
            ],
            expected_head_enrollment_binding_sha256=enrollment[
                "enrollment_binding_sha256"
            ],
            authority_bundle_root=authority_bundle_root,
            expected_authority_commit_sha256=(
                expected_authority_commit_sha256
            ),
        )
        state_runtime[state] = {
            "request_path": str(request_path),
            "request_sha256": request_sha256,
            "enrollment": enrollment,
            "head_path": str(head_path),
            "apply_package_root": state_build["apply_staging_root"],
            "apply_seal_path": str(apply_seal_path),
            "apply": finalized_apply,
        }

    final_pair_path = scorer_root / "registration_pair.final.json"
    final_pair = finalize_registration_pair_manifest(
        staging_manifest_path=build["registration_pair_manifest"],
        output_path=final_pair_path,
        before_binding_sha256=state_runtime["before"]["enrollment"][
            "enrollment_binding_sha256"
        ],
        after_binding_sha256=state_runtime["after"]["enrollment"][
            "enrollment_binding_sha256"
        ],
    )

    diag_results: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, Path] = {}
    for state in ("before", "after"):
        state_build = build["states"][state]
        runtime = state_runtime[state]
        state_diag_root = diag_root / state
        state_diag_root.mkdir(parents=True, exist_ok=False)
        diag_results[state] = run_diag_cosine_exploration(
            enrollment_package_root=state_build[
                "enrollment_package_root"
            ],
            enrollment_seal_path=state_build[
                "enrollment_package_seal"
            ],
            enrollment_seal_sha256=state_build[
                "enrollment_package_seal_sha256"
            ],
            apply_package_root=runtime["apply_package_root"],
            apply_seal_path=runtime["apply_seal_path"],
            apply_seal_sha256=runtime["apply"][
                "package_seal_sha256"
            ],
            output_root=state_diag_root,
            device=device,
            candidate=candidate,
        )
        prediction_paths[state] = state_diag_root / "prediction_artifact.npz"

    for state, prediction_path in prediction_paths.items():
        _require_prediction(prediction_path, state=state)

    score_path = scorer_root / "diag_cosine_score.json"
    score = score_diag_cosine_pair(
        before_prediction_path=prediction_paths["before"],
        after_prediction_path=prediction_paths["after"],
        truth_sidecar_path=build["truth_sidecar"],
        output_path=score_path,
        candidate=candidate,
    )

    receipt = {
        "schema": PIPELINE_SCHEMA,
        "status": "DEVELOPMENT_ROW_COMPLETE",
        "claim_scope": "development_only_not_formal_confirmation",
        "formal_launch_authority": False,
        "receiver": receiver,
        "seed": seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
        "device": device,
        "candidate": candidate,
        "query_per_tx": FORMAL_QUERY_PER_TX,
        "row_handle": build["row_handle"],
        "row_manifest_sha256": build["row_manifest_sha256"],
        "authority_commit_sha256": expected_authority_commit_sha256,
        "phase1_checkpoint_sha256": sha256_file(
            Path(phase1_checkpoint_path)
        ),
        "sealed_feature_runtime_sha256": sha256_file(
            Path(sealed_feature_runtime_path)
        ),
        "method_lock_sha256": sha256_file(Path(method_lock_path)),
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "truth_sidecar_exposed_to_predictor": False,
        "truth_join_started_after_both_immutable_predictions": True,
        "registration_pair_final_path": str(final_pair_path),
        "registration_pair_final_sha256": sha256_file(final_pair_path),
        "states": {
            state: {
                "stage": build["states"][state]["stage"],
                "enrollment_request_sha256": state_runtime[state][
                    "request_sha256"
                ],
                "head_capsule_sha256": state_runtime[state]["enrollment"][
                    "head_capsule_sha256"
                ],
                "enrollment_binding_sha256": state_runtime[state][
                    "enrollment"
                ]["enrollment_binding_sha256"],
                "apply_package_root_sha256": state_runtime[state]["apply"][
                    "package_root_sha256"
                ],
                "apply_package_seal_sha256": state_runtime[state]["apply"][
                    "package_seal_sha256"
                ],
                "prediction_artifact_sha256": diag_results[state][
                    "prediction_artifact_sha256"
                ],
            }
            for state in ("before", "after")
        },
        "score_path": str(score_path),
        "score_artifact_sha256": score["score_artifact_sha256"],
        "final_pair_schema": final_pair["schema"],
    }
    receipt_path = output / "pipeline_receipt.json"
    receipt_sha256 = _write_json_new(receipt_path, receipt)
    return {
        "schema": "cvs.phase2.somph_diag_row_pipeline_stdout.v1",
        "status": receipt["status"],
        "output_root": str(output),
        "pipeline_receipt": str(receipt_path),
        "pipeline_receipt_sha256": receipt_sha256,
        "score_path": str(score_path),
        "score_artifact_sha256": score["score_artifact_sha256"],
        "formal_launch_authority": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-manifest", required=True)
    result.add_argument("--authority-bundle", required=True)
    result.add_argument("--authority-commit-sha256", required=True)
    result.add_argument("--phase1-checkpoint", required=True)
    result.add_argument("--sealed-runtime", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--receiver", required=True)
    result.add_argument("--seed", required=True, type=int)
    result.add_argument("--k-shot", required=True, type=int)
    result.add_argument("--new-count", required=True, type=int)
    result.add_argument("--device", required=True)
    result.add_argument("--candidate", choices=CANDIDATES, default=CANDIDATE_D1)
    return result


def main() -> int:
    args = parser().parse_args()
    result = run_pipeline(
        cache_set_manifest_path=args.cache_manifest,
        authority_bundle_root=args.authority_bundle,
        expected_authority_commit_sha256=args.authority_commit_sha256,
        phase1_checkpoint_path=args.phase1_checkpoint,
        sealed_feature_runtime_path=args.sealed_runtime,
        method_lock_path=args.method_lock,
        output_root=args.output_root,
        receiver=args.receiver,
        seed=args.seed,
        k_shot=args.k_shot,
        new_class_count=args.new_count,
        device=args.device,
        candidate=args.candidate,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
