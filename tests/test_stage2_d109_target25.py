"""Focused D109-SCRC Target25 thin-adapter tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cvsrffi.stage2_d108_target25 as d108_target25
import cvsrffi.stage2_d108_target125_runner as d108_runner
import cvsrffi.stage2_d108_truth_scorer as d108_truth
import cvsrffi.stage2_d109_target25 as target
from cvsrffi.stage2_d108_matrix_protocol import canonical_bytes, canonical_sha256
from cvsrffi.stage2_d108_target125_inputs import CONTEXT_SCHEMA, PLAN_SCHEMA
from scripts import run_d109_target25 as cli


OLD = tuple(f"old_{index}" for index in range(6))


def _write(path: Path, value: dict[str, Any]) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prepared(tmp_path: Path) -> tuple[Path, str, Path, str]:
    matrix = d108_runner.freeze_d108_target125_matrix()
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
        "candidate_id": d108_runner.CANDIDATE_ID,
        "protocol_schema": d108_runner.PROTOCOL_SCHEMA,
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "candidate_id": d108_runner.CANDIDATE_ID,
        "protocol_schema": d108_runner.PROTOCOL_SCHEMA,
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
    def __init__(self) -> None:
        self.builds = 0
        self.scores: list[tuple[str, str]] = []

    def build(self, *args: Any, **_kwargs: Any) -> dict[str, tuple[str, ...]]:
        self.builds += 1
        old, new = tuple(args[2]), tuple(args[5])
        return {"before": old, "after": old + new}

    def score(
        self, pair: dict[str, tuple[str, ...]], phase: str, arm: str, query: np.ndarray
    ) -> np.ndarray:
        self.scores.append((phase, arm))
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


def test_fixed_matrix_and_public_surface_are_exactly_d108_target25() -> None:
    matrix = d108_target25.freeze_d108_target25_matrix()
    assert target.FROZEN_SEED == 713102
    assert target.OUTER_JOB_COUNT == len(matrix.outer_rows) == 25
    assert target.SCENE_ROW_COUNT == len(matrix.scene_rows) == 75
    assert target.ARM_PAIR_COUNT == len(matrix.arm_pairs) == 300
    assert target.SURFACE_COUNT == len(matrix.surfaces) == 600
    assert target.ARMS == ("M0", "M_DA", "M_HEAD", "M_JOINT")
    forbidden = {"truth", "role", "quota", "query_labels", "metrics", "temperature"}
    for function in (
        target.prepare_d109_target25_run,
        target.smoke_d109_target25_prepared_state,
        target.predict_d109_target25,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    assert cli.parse_args(
        ["merge", "--shard-manifest", "one", "--output-dir", "out"]
    ).command == "merge"


def test_prepare_is_exact_d108_target25_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"target25_seed": 713102}
    seen: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(d108_target25, "prepare_d108_target25_run", fake_prepare)
    assert target.prepare_d109_target25_run(marker="no-revalidation") is sentinel
    assert seen == {"marker": "no-revalidation"}


def test_smoke_uses_d109_core_and_target25_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = FakeCore()
    monkeypatch.setattr(target.d109_core, "build_d109_d92_pair", core.build)
    monkeypatch.setattr(target.d109_core, "score", core.score)
    result = target.smoke_d109_target25_prepared_state(
        **_common(tmp_path),
        output_dir=tmp_path / "smoke",
        state_materializer=FakeMaterializer(),
        d92_fit=_dummy_fit,
    )
    predictions = json.loads(Path(result["smoke_predictions"]).read_text())
    assert result["candidate_id"] == target.CANDIDATE_ID
    assert result["schema"] == target.SMOKE_RECEIPT_SCHEMA
    assert predictions["candidate_id"] == target.CANDIDATE_ID
    assert predictions["schema"] == target.SMOKE_PREDICTIONS_SCHEMA
    assert predictions["seed"] == 713102
    assert predictions["truth_open"] is False
    assert predictions["immutable"] is True
    assert len(predictions["surfaces"]) == 8
    assert core.builds == 1
    assert core.scores == [
        (phase, arm) for arm in target.ARMS for phase in target.PHASES
    ]


def test_eight_shards_merge_to_600_d109_surfaces_and_are_immutable(
    tmp_path: Path,
) -> None:
    common = _common(tmp_path)
    shards = []
    original_replacements = d108_target25._RUNNER_REPLACEMENTS
    for index in range(8):
        core = FakeCore()
        result = target.predict_d109_target25(
            **common,
            output_dir=tmp_path / f"shard{index}",
            shard_index=index,
            state_materializer=FakeMaterializer(),
            pair_builder=core.build,
            query_scorer=core.score,
            d92_fit=_dummy_fit,
        )
        shard_path = Path(result["prediction_shard_manifest"])
        shard = json.loads(shard_path.read_text())
        assert shard["candidate_id"] == target.CANDIDATE_ID
        assert shard["schema"] == target.PREDICTION_SHARD_SCHEMA
        shards.append(shard_path)
    merged = target.predict_d109_target25(
        shard_manifest_paths=shards, output_dir=tmp_path / "merged"
    )
    manifest = target.validate_d109_target25_prediction_manifest(
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
    )
    assert manifest["candidate_id"] == target.CANDIDATE_ID
    assert manifest["schema"] == target.PREDICTION_MANIFEST_SCHEMA
    assert manifest["outer_job_count"] == 25
    assert manifest["surface_count"] == 600
    assert {row["seed"] for row in manifest["outer_rows"]} == {713102}
    assert d108_target25._RUNNER_REPLACEMENTS is original_replacements
    with pytest.raises(FileExistsError):
        target.predict_d109_target25(
            shard_manifest_paths=shards, output_dir=tmp_path / "merged"
        )


def test_truth_projection_uses_d109_identity_and_target25_counts_then_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_candidate = d108_truth.CANDIDATE_ID
    original_count = d108_truth.OUTER_JOB_COUNT
    original_replacements = d108_target25._TRUTH_REPLACEMENTS

    def fake_validate(**_kwargs: Any) -> dict[str, Any]:
        return {"manifest_sealed": True}

    def fake_score(**_kwargs: Any) -> dict[str, Any]:
        assert d108_truth.CANDIDATE_ID == target.CANDIDATE_ID
        assert d108_truth.PREDICTION_MANIFEST_SCHEMA == target.PREDICTION_MANIFEST_SCHEMA
        assert d108_truth.SCORE_MANIFEST_SCHEMA == target.SCORE_MANIFEST_SCHEMA
        assert d108_truth.OUTER_JOB_COUNT == 25
        assert d108_truth.SURFACE_COUNT == 600
        assert d108_truth.TRUTH_SURFACE_COUNT == 150
        return {"candidate_id": d108_truth.CANDIDATE_ID, "surface_count": 600}

    monkeypatch.setattr(
        d108_target25, "validate_d108_target25_prediction_manifest", fake_validate
    )
    monkeypatch.setattr(d108_truth, "score_d108_target125", fake_score)
    result = target.score_d109_target25(
        prediction_manifest_path="prediction.json",
        expected_prediction_manifest_file_sha256="0" * 64,
        truth_catalog_path="truth.json",
        expected_truth_catalog_file_sha256="1" * 64,
        output_dir="score",
    )
    assert result == {"candidate_id": target.CANDIDATE_ID, "surface_count": 600}
    assert d108_truth.CANDIDATE_ID == original_candidate
    assert d108_truth.OUTER_JOB_COUNT == original_count
    assert d108_target25._TRUTH_REPLACEMENTS is original_replacements
