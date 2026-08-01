"""Focused execution-boundary tests for the D107 SCMKRR Target125 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import numpy as np
import pytest

from cvsrffi.stage2_d107_matrix_protocol import (
    ARMS,
    CANDIDATE_ID,
    OUTER_JOB_COUNT,
    PHASES,
    SCENES,
    SCENE_ROW_COUNT,
    SURFACE_COUNT,
    canonical_bytes,
    canonical_sha256,
    freeze_d107_target125_matrix,
)
from cvsrffi.stage2_d107_target125_inputs import (
    D92_CANDIDATE,
    D92_MATRIX_SCHEMA,
    prepare_d107_target125_inputs,
)
from cvsrffi.stage2_d107_target125_runner import (
    D107Target125RunnerError,
    predict_d107_target125,
    smoke_d107_target125_prepared_state,
    validate_d107_target125_prediction_manifest,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return _sha_bytes(raw)


def _sha_token(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _d92_job_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    return f"d92-rx-{receiver}__seed-{seed}__k-{k_shot}__new-{new_count}"


def _make_d92_prepare_inputs(tmp_path: Path) -> dict[str, Any]:
    """Make a closed synthetic D92 locator tree; no package payload is opened."""

    matrix = freeze_d107_target125_matrix()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"d107-checkpoint")
    checkpoint_sha = _sha_bytes(checkpoint.read_bytes())
    runtime_sha = _sha_token("runtime")
    output_root = tmp_path / "d92-output"
    jobs_root = output_root / "jobs"
    jobs_root.mkdir(parents=True)
    jobs: list[dict[str, Any]] = []
    for outer in matrix.outer_rows:
        job_id = _d92_job_id(
            outer.receiver, outer.seed, outer.k_shot, outer.new_count
        )
        job_root = jobs_root / job_id
        for phase in PHASES:
            for profile, leaf in (
                ("enrollment", "enrollment_only"),
                ("apply", "apply_only_staging"),
            ):
                (job_root / "offline" / "predictor" / phase / leaf).mkdir(
                    parents=True, exist_ok=True
                )
                seal_root = (
                    job_root / "offline" / "seals"
                    if profile == "enrollment"
                    else job_root / "apply_seals"
                )
                seal_root.mkdir(parents=True, exist_ok=True)
                seal_name = (
                    f"{phase}_enrollment.seal.json"
                    if profile == "enrollment"
                    else f"{phase}_apply.seal.json"
                )
                (seal_root / seal_name).write_bytes(
                    f"{job_id}:{phase}:{profile}".encode("utf-8")
                )
        jobs.append(
            {
                "candidate": D92_CANDIDATE,
                "job_id": job_id,
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "new_class_count": outer.new_count,
                "output_root": str(job_root.resolve()),
                "scenarios": list(SCENES),
            }
        )
    d92_manifest = {
        "schema": D92_MATRIX_SCHEMA,
        "candidate": D92_CANDIDATE,
        "job_count": OUTER_JOB_COUNT,
        "receivers": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "phase1_checkpoint_sha256": checkpoint_sha,
        "sealed_runtime_sha256": runtime_sha,
        "phase2_contract": {
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_query_batch_global_assignment": False,
            "phase2_query_class_quota_access": False,
            "phase2_query_role_oracle_access": False,
            "phase2_query_true_batch_class_count_access": False,
        },
        "jobs": jobs,
    }
    d92_manifest_path = tmp_path / "d92_manifest.json"
    d92_manifest_sha = _write_json(d92_manifest_path, d92_manifest)
    method_lock = {
        "schema": "cvs.phase2.d107.scmkrr_method_lock.v1",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "feature_view": "signed_z_id_l2",
        "arms": list(ARMS),
    }
    method_lock_path = tmp_path / "d107_lock.json"
    method_lock_sha = _write_json(method_lock_path, method_lock)
    lineage = {
        "checkpoint_sha256": checkpoint_sha,
        "runtime_sha256": runtime_sha,
        "method_lock_sha256": _sha_token("d106-rdce-lock"),
        "split_id": "d104_source_seed104713_v2",
        "tap_sha256": _sha_token("tap"),
        "construction_code_sha256": _sha_token("construction"),
        "content_root_sha256": _sha_token("content"),
        "source_receipt_sha256": _sha_token("source"),
        "tap_receipt_sha256": _sha_token("tap-receipt"),
        "tap_authority_sha256": _sha_token("tap-authority"),
    }
    lineage_path = tmp_path / "rdce_lineage.json"
    lineage_sha = _write_json(lineage_path, lineage)
    rdce_asset_dir = tmp_path / "rdce_asset"
    rdce_asset_dir.mkdir()
    return {
        "d92_matrix_manifest_path": d92_manifest_path,
        "expected_d92_matrix_manifest_sha256": d92_manifest_sha,
        "d92_output_root": output_root.resolve(),
        "checkpoint_path": checkpoint.resolve(),
        "expected_checkpoint_sha256": checkpoint_sha,
        "d107_method_lock_path": method_lock_path.resolve(),
        "expected_d107_method_lock_sha256": method_lock_sha,
        "rdce_asset_dir": rdce_asset_dir.resolve(),
        "expected_rdce_wire_sha256": _sha_token("rdce-wire"),
        "rdce_lineage_path": lineage_path.resolve(),
        "expected_rdce_lineage_sha256": lineage_sha,
    }


def _make_prepared_manifests(tmp_path: Path) -> tuple[Path, str, Path, str]:
    """Publish a minimal valid plan/context for fake prediction execution."""

    matrix = freeze_d107_target125_matrix()
    digest = "a" * 64
    rows: list[dict[str, Any]] = []
    for outer in matrix.outer_rows:
        source_pool_k = 10 if (outer.k_shot, outer.new_count) == (5, 20) else outer.k_shot
        packages = {
            name: {
                "package_root": "/sealed/package",
                "detached_seal_path": "/sealed/seal.json",
                "expected_seal_sha256": digest,
            }
            for name in (
                "before_enrollment",
                "before_apply",
                "after_enrollment",
                "after_apply",
            )
        }
        rows.append(
            {
                "outer_id": outer.outer_id,
                "source_d92_job_id": _d92_job_id(
                    outer.receiver, outer.seed, source_pool_k, outer.new_count
                ),
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "active_k": outer.k_shot,
                "new_count": outer.new_count,
                "source_pool_k": source_pool_k,
                "k5_prefix_from_matched_k10": (
                    outer.k_shot == 5 and outer.new_count == 20
                ),
                "packages": packages,
            }
        )
    identity = {
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "d92_matrix_manifest": {"path": "/sealed/d92.json", "sha256": digest},
        "d92_output_root": "/sealed/d92",
        "checkpoint": {"path": "/sealed/checkpoint", "sha256": digest},
        "d107_method_lock": {"path": "/sealed/lock", "sha256": digest},
        "rdce_asset": {
            "directory": "/sealed/rdce",
            "wire_sha256": digest,
            "lineage_path": "/sealed/lineage.json",
            "lineage_file_sha256": digest,
            "lineage": {
                "checkpoint_sha256": digest,
                "runtime_sha256": digest,
                "method_lock_sha256": digest,
                "split_id": "d104_source_seed104713_v2",
                "tap_sha256": digest,
                "construction_code_sha256": digest,
                "content_root_sha256": digest,
                "source_receipt_sha256": digest,
                "tap_receipt_sha256": digest,
                "tap_authority_sha256": digest,
            },
        },
    }
    plan: dict[str, Any] = {
        "schema": "cvs.phase2.d107.scmkrr.target125.input_plan.v1",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": "cvs.phase2.d107.scmkrr.target125.input_context.v1",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "identity": identity,
        "rows": rows,
    }
    context["context_receipt_sha256"] = canonical_sha256(context)
    plan_path = tmp_path / "plan.json"
    context_path = tmp_path / "context.json"
    return (
        plan_path,
        _write_json(plan_path, plan),
        context_path,
        _write_json(context_path, context),
    )


def _signed_rows(rows: int, offset: int) -> np.ndarray:
    value = np.zeros((rows, 160), dtype=np.float32)
    value[np.arange(rows), (np.arange(rows) + offset) % 160] = 1.0
    return value


class _FakeMaterializer:
    def __init__(self) -> None:
        self.before_support_sha_by_scope: dict[tuple[str, str], str] = {}
        self.support_scope_by_sha: dict[str, tuple[tuple[str, str], str]] = {}

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        old = tuple(f"old_{index}" for index in range(6))
        phase = request["phase"]
        registered = old if phase == "before" else old + tuple(
            f"new_{index}" for index in range(request["new_count"])
        )
        labels = tuple(
            label for label in registered for _ in range(request["k_shot"])
        )
        support = _signed_rows(
            len(labels),
            request["seed"] + (0 if phase == "before" else 29),
        )
        query = _signed_rows(3, request["seed"] + 71)
        scope = (request["outer_id"], request["scene"])
        support_sha = _sha_bytes(support.tobytes())
        self.support_scope_by_sha[support_sha] = (scope, phase)
        if phase == "before":
            self.before_support_sha_by_scope[scope] = support_sha
        return {
            "support_signed": support,
            "support_labels": labels,
            "registered_classes": registered,
            "support_physical_ids": tuple(
                f"{request['outer_id']}:{request['scene']}:{phase}:s:{index}"
                for index in range(len(labels))
            ),
            "query_signed": query,
            "query_physical_ids": tuple(
                f"{request['outer_id']}:{request['scene']}:q:{index}"
                for index in range(len(query))
            ),
            "tau": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
            "spectrum": np.asarray((4.0, 5.0, 6.0), dtype=np.float64),
        }


class _FakeCore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {
                "arm": kwargs["arm"],
                "support_sha": _sha_bytes(kwargs["support_signed"].tobytes()),
                "anchor_sha": _sha_bytes(kwargs["anchor_signed"].tobytes()),
                "registered_classes": tuple(kwargs["registered_classes"]),
            }
        )
        return {"registered_classes": tuple(kwargs["registered_classes"])}

    def score(self, state: dict[str, Any], query_signed: np.ndarray) -> np.ndarray:
        scores = np.zeros(
            (len(query_signed), len(state["registered_classes"])), dtype=np.float32
        )
        scores[np.arange(len(query_signed)), np.arange(len(query_signed)) % scores.shape[1]] = 1.0
        return scores


def test_target125_matrix_counts_and_ids() -> None:
    matrix = freeze_d107_target125_matrix()
    assert len(matrix.outer_rows) == 125
    assert len(matrix.scene_rows) == 375
    assert len(matrix.arm_pairs) == 1500
    assert len(matrix.surfaces) == 3000
    assert matrix.outer_rows[0].outer_id == "d107-rx-20-1__seed-713102__k-10__new-5"
    assert tuple(surface.arm for surface in matrix.surfaces[:8]) == (
        "M0",
        "M0",
        "M_DA",
        "M_DA",
        "M_HEAD",
        "M_HEAD",
        "M_JOINT",
        "M_JOINT",
    )


def test_prepare_binds_k5_to_matching_k10_prefix(tmp_path: Path) -> None:
    inputs = _make_d92_prepare_inputs(tmp_path)
    output_dir = tmp_path / "prepared"
    result = prepare_d107_target125_inputs(**inputs, output_dir=output_dir)
    assert result["outer_job_count"] == 125
    plan = json.loads(Path(result["plan_manifest"]).read_text(encoding="utf-8"))
    rows = {(row["receiver"], row["seed"], row["k_shot"], row["new_count"]): row for row in plan["rows"]}
    for receiver in ("20-1", "3-19", "7-14", "7-7", "8-8"):
        for seed in (713102, 713103, 713104, 713105, 713106):
            k5 = rows[(receiver, seed, 5, 20)]
            k10 = rows[(receiver, seed, 10, 20)]
            assert k5["active_k"] == 5
            assert k5["source_pool_k"] == 10
            assert k5["k5_prefix_from_matched_k10"] is True
            assert k5["source_d92_job_id"] == k10["source_d92_job_id"]
            assert k5["packages"] == k10["packages"]


def test_smoke_uses_both_phases_and_all_four_arms(tmp_path: Path) -> None:
    plan, plan_sha, context, context_sha = _make_prepared_manifests(tmp_path)
    materializer = _FakeMaterializer()
    core = _FakeCore()
    result = smoke_d107_target125_prepared_state(
        plan_manifest_path=plan,
        expected_plan_file_sha256=plan_sha,
        context_manifest_path=context,
        expected_context_file_sha256=context_sha,
        output_dir=tmp_path / "smoke",
        state_materializer=materializer,
        state_builder=core.build,
        query_scorer=core.score,
    )
    assert result["phase_count"] == 2
    assert result["arm_count"] == 4
    assert {call["arm"] for call in core.calls} == set(ARMS)
    assert "ROUTED" not in {call["arm"] for call in core.calls}


def test_prediction_seals_full_surface_closure_and_rejects_mutation(tmp_path: Path) -> None:
    plan, plan_sha, context, context_sha = _make_prepared_manifests(tmp_path)
    materializer = _FakeMaterializer()
    core = _FakeCore()
    output = tmp_path / "prediction"
    result = predict_d107_target125(
        plan_manifest_path=plan,
        expected_plan_file_sha256=plan_sha,
        context_manifest_path=context,
        expected_context_file_sha256=context_sha,
        output_dir=output,
        state_materializer=materializer,
        state_builder=core.build,
        query_scorer=core.score,
    )
    manifest_path = Path(result["prediction_manifest"])
    manifest = validate_d107_target125_prediction_manifest(
        prediction_manifest_path=manifest_path,
        expected_prediction_manifest_file_sha256=result[
            "prediction_manifest_file_sha256"
        ],
    )
    assert result["outer_job_count"] == 125
    assert result["scene_row_count"] == 375
    assert result["arm_pair_count"] == 1500
    assert result["surface_count"] == 3000
    assert manifest["truth_open"] is False
    assert manifest["manifest_sealed"] is True
    assert len(manifest["surfaces"]) == SURFACE_COUNT
    assert {call["arm"] for call in core.calls} == set(ARMS)
    assert len(core.calls) == SURFACE_COUNT
    for call in core.calls:
        scope, _phase = materializer.support_scope_by_sha[call["support_sha"]]
        assert call["anchor_sha"] == materializer.before_support_sha_by_scope[scope]
    assert all(
        surface["access_ledger"]
        == {
            "clean_source_runtime_access": False,
            "query_fit_access": False,
            "query_update_access": False,
            "query_truth_access": False,
            "query_role_access": False,
            "query_selection_access": False,
        }
        for surface in manifest["surfaces"]
    )
    with pytest.raises(FileExistsError):
        predict_d107_target125(
            plan_manifest_path=plan,
            expected_plan_file_sha256=plan_sha,
            context_manifest_path=context,
            expected_context_file_sha256=context_sha,
            output_dir=output,
            state_materializer=materializer,
            state_builder=core.build,
            query_scorer=core.score,
        )

    original_manifest = manifest_path.read_bytes()
    missing = json.loads(original_manifest.decode("utf-8"))
    missing["surfaces"].pop()
    missing["manifest_sha256"] = canonical_sha256(
        {name: value for name, value in missing.items() if name != "manifest_sha256"}
    )
    manifest_path.chmod(manifest_path.stat().st_mode | stat.S_IWRITE)
    manifest_path.write_bytes(canonical_bytes(missing) + b"\n")
    with pytest.raises(D107Target125RunnerError):
        validate_d107_target125_prediction_manifest(
            prediction_manifest_path=manifest_path
        )
    manifest_path.write_bytes(original_manifest)

    artifact_path = output / manifest["surfaces"][0]["prediction_artifact"]
    artifact_path.chmod(artifact_path.stat().st_mode | stat.S_IWRITE)
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(D107Target125RunnerError):
        validate_d107_target125_prediction_manifest(
            prediction_manifest_path=manifest_path
        )
