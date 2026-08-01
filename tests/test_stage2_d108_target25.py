"""Focused D108 frozen-seed Target25 adapter tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cvsrffi.stage2_d108_target25 as target
import cvsrffi.stage2_d108_target125_runner as runner125
import cvsrffi.stage2_d108_truth_scorer as truth125
from cvsrffi.stage2_d108_matrix_protocol import canonical_bytes, canonical_sha256
from cvsrffi.stage2_d108_target125_inputs import CONTEXT_SCHEMA, PLAN_SCHEMA
from scripts import run_d108_target25 as cli


OLD = tuple(f"old_{index}" for index in range(6))


def _write(path: Path, value: dict[str, Any]) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prepared(tmp_path: Path) -> tuple[Path, str, Path, str]:
    matrix = runner125.freeze_d108_target125_matrix()
    commit = tmp_path / "COMMIT.json"
    commit.write_bytes(b"{}\n")
    commit_sha = hashlib.sha256(commit.read_bytes()).hexdigest()
    zero = "0" * 64
    packages = {
        name: {
            "package_root": str(tmp_path / name),
            "detached_seal_path": str(tmp_path / f"{name}.seal.json"),
            "expected_seal_sha256": zero,
        }
        for name in (
            "before_enrollment",
            "before_apply",
            "after_enrollment",
            "after_apply",
        )
    }
    rows = []
    for outer in matrix.outer_rows:
        rows.append(
            {
                "outer_id": outer.outer_id,
                "source_d92_job_id": f"source-{outer.outer_id}",
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "active_k": outer.k_shot,
                "new_count": outer.new_count,
                "source_pool_k": (
                    10
                    if (outer.k_shot, outer.new_count) == (5, 20)
                    else outer.k_shot
                ),
                "k5_prefix_from_matched_k10": (
                    outer.k_shot,
                    outer.new_count,
                )
                == (5, 20),
                "packages": packages,
                "authority_bundle": {
                    "directory": str(tmp_path),
                    "commit_path": str(commit),
                    "commit_sha256": commit_sha,
                },
            }
        )
    identity = {
        "matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "d92_matrix_manifest": {"path": str(tmp_path / "d92.json"), "sha256": zero},
        "d92_output_root": str(tmp_path),
        "d92_sealed_runtime_sha256": zero,
        "checkpoint": {"path": str(tmp_path / "checkpoint.pth"), "sha256": zero},
        "d108_method_lock": {"path": str(tmp_path / "lock.json"), "sha256": zero},
        "ground_component": {
            "directory": str(tmp_path),
            "manifest_path": str(tmp_path / "ground.json"),
            "manifest_sha256": zero,
        },
    }
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "candidate_id": runner125.CANDIDATE_ID,
        "protocol_schema": runner125.PROTOCOL_SCHEMA,
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "candidate_id": runner125.CANDIDATE_ID,
        "protocol_schema": runner125.PROTOCOL_SCHEMA,
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "identity": identity,
        "rows": rows,
    }
    context["context_receipt_sha256"] = canonical_sha256(context)
    plan_path, context_path = tmp_path / "plan.json", tmp_path / "context.json"
    return plan_path, _write(plan_path, plan), context_path, _write(context_path, context)


def _feature(token: str) -> np.ndarray:
    row = np.zeros(288, dtype=np.float32)
    row[int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % 288] = 1
    return row


class FakeMaterializer:
    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        new = tuple(f"new_{index}" for index in range(request["new_count"]))
        classes = OLD if request["phase"] == "before" else OLD + new
        labels, ids, features = [], [], []
        for label in classes:
            for shot in range(request["k_shot"]):
                token = f"{request['outer_id']}:{request['scene']}:{label}:{shot}"
                labels.append(label)
                ids.append(token)
                features.append(_feature(token))
        query_ids = tuple(
            f"{request['outer_id']}:{request['scene']}:{request['phase']}:q{i}"
            for i in range(2)
        )
        return {
            "support_features": np.stack(features).astype(np.float32),
            "support_labels": labels,
            "registered_classes": classes,
            "support_physical_ids": ids,
            "query_features": np.stack([_feature(token) for token in query_ids]),
            "query_physical_ids": query_ids,
        }


class FakeCore:
    def build(self, *args: Any, **_kwargs: Any) -> dict[str, tuple[str, ...]]:
        old, new = tuple(args[2]), tuple(args[5])
        return {"before": old, "after": old + new}

    def score(
        self, pair: dict[str, tuple[str, ...]], phase: str, _arm: str, query: np.ndarray
    ) -> np.ndarray:
        logits = np.zeros((len(query), len(pair[phase])), dtype=np.float32)
        logits[:, 0] = 1
        return logits


def _dummy_fit(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("FakeCore must not call D92 fit")


def _common(tmp_path: Path) -> dict[str, Any]:
    plan, plan_sha, context, context_sha = _prepared(tmp_path)
    return {
        "plan_manifest_path": plan,
        "expected_plan_file_sha256": plan_sha,
        "context_manifest_path": context,
        "expected_context_file_sha256": context_sha,
    }


def test_frozen_matrix_is_exact_seed713102_25_75_300_600() -> None:
    matrix = target.freeze_d108_target25_matrix()
    assert len(matrix.outer_rows) == 25
    assert len(matrix.scene_rows) == 75
    assert len(matrix.arm_pairs) == 300
    assert len(matrix.surfaces) == 600
    assert {row.seed for row in matrix.outer_rows} == {713102}
    assert {row.receiver for row in matrix.outer_rows} == set(target.RECEIVERS)
    assert {(row.k_shot, row.new_count) for row in matrix.outer_rows} == set(
        target.SLICES
    )


def test_public_prediction_api_has_no_truth_role_quota_or_method_change() -> None:
    forbidden = {"truth", "role", "quota", "query_labels", "metrics", "temperature"}
    for function in (
        target.prepare_d108_target25_run,
        target.smoke_d108_target25_prepared_state,
        target.predict_d108_target25,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    assert target.ARMS == runner125.ARMS
    assert target.CANDIDATE_ID == runner125.CANDIDATE_ID
    assert cli.parse_args(["merge", "--shard-manifest", "one", "--output-dir", "out"]).command == "merge"


def test_smoke_projects_full_prepared_inputs_without_changing_four_arms(
    tmp_path: Path,
) -> None:
    core = FakeCore()
    common = _common(tmp_path)
    result = target.smoke_d108_target25_prepared_state(
        **common,
        output_dir=tmp_path / "smoke",
        state_materializer=FakeMaterializer(),
        pair_builder=core.build,
        query_scorer=core.score,
        d92_fit=_dummy_fit,
    )
    assert result["status"].endswith("SMOKE_PASS")
    predictions = json.loads(Path(result["smoke_predictions"]).read_text())
    assert predictions["seed"] == 713102
    assert predictions["schema"] == target.SMOKE_PREDICTIONS_SCHEMA
    assert len(predictions["surfaces"]) == 8

    with pytest.raises(target.D108Target25Error, match="scene index"):
        target.smoke_d108_target25_prepared_state(
            **common,
            output_dir=tmp_path / "bad-scene",
            scene_index=3,
            state_materializer=FakeMaterializer(),
            pair_builder=core.build,
            query_scorer=core.score,
            d92_fit=_dummy_fit,
        )


def test_eight_shards_merge_to_600_immutable_surfaces(tmp_path: Path) -> None:
    common = _common(tmp_path)
    shards = []
    for index in range(8):
        core = FakeCore()
        result = target.predict_d108_target25(
            **common,
            output_dir=tmp_path / f"shard{index}",
            shard_index=index,
            state_materializer=FakeMaterializer(),
            pair_builder=core.build,
            query_scorer=core.score,
            d92_fit=_dummy_fit,
        )
        shards.append(Path(result["prediction_shard_manifest"]))
    merged = target.predict_d108_target25(
        shard_manifest_paths=shards, output_dir=tmp_path / "merged"
    )
    manifest = target.validate_d108_target25_prediction_manifest(
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
    )
    assert manifest["outer_job_count"] == 25
    assert manifest["surface_count"] == 600
    assert manifest["schema"] == target.PREDICTION_MANIFEST_SCHEMA
    assert {row["seed"] for row in manifest["outer_rows"]} == {713102}
    with pytest.raises(FileExistsError):
        target.predict_d108_target25(
            shard_manifest_paths=shards, output_dir=tmp_path / "merged"
        )


def test_truth_projection_uses_25_counts_and_restores_full_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_count = truth125.OUTER_JOB_COUNT
    original_summary = truth125._score_summary
    seen: dict[str, Any] = {}

    def fake_score(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert truth125.OUTER_JOB_COUNT == 25
        assert truth125.SURFACE_COUNT == 600
        assert truth125.TRUTH_SURFACE_COUNT == 150
        coverage, _resources, verdict = truth125._score_summary(
            {"manifest_size": 1, "artifact_bytes": 2, "prediction_query_count": 3, "support_slots": 4},
            truth_catalog_size=5,
        )
        assert coverage["outer_job_count"] == 25
        assert verdict["coverage_verdict"] == "COMPLETE_25_TRUTH_OPEN_AND_SCORED"
        return {"outer_job_count": truth125.OUTER_JOB_COUNT}

    monkeypatch.setattr(
        target,
        "validate_d108_target25_prediction_manifest",
        lambda **_kwargs: {"manifest_sealed": True},
    )
    monkeypatch.setattr(truth125, "score_d108_target125", fake_score)
    result = target.score_d108_target25(
        prediction_manifest_path="prediction.json",
        expected_prediction_manifest_file_sha256="0" * 64,
        truth_catalog_path="truth.json",
        expected_truth_catalog_file_sha256="1" * 64,
        output_dir="score",
    )
    assert result == {"outer_job_count": 25}
    assert truth125.OUTER_JOB_COUNT == original_count
    assert truth125._score_summary is original_summary
    assert len(seen) == 5
