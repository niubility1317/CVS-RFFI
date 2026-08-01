"""Focused D109 Target125 adapter tests.

| ID | Requirement | Verification |
|---|---|---|
| D109-T01 | Reuse D108 prepared/materialization plane without new authority | prepared fixture and delegation |
| D109-T02 | Inject only D109 pair/score into four fixed arms | smoke default-core test |
| D109-T03 | Distinct immutable D109 shard/publication identity | shard/merge/validate test |
| D109-T04 | Reuse independent D108 truth scorer under D109 identity | truth adapter test |
| D109-T05 | No prediction truth/role/quota/update surface | public-signature test |
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cvsrffi.stage2_d108_target125_runner as d108_runner
import cvsrffi.stage2_d108_truth_scorer as d108_truth
import cvsrffi.stage2_d109_target125 as target
from cvsrffi.stage2_d108_matrix_protocol import canonical_bytes, canonical_sha256
from cvsrffi.stage2_d108_target125_inputs import CONTEXT_SCHEMA, PLAN_SCHEMA
from scripts import run_d109_target125 as cli


OLD = tuple(f"old_{index}" for index in range(6))


def _write_json(path: Path, value: dict[str, Any]) -> str:
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
        source_pool = (
            10 if (outer.k_shot, outer.new_count) == (5, 20) else outer.k_shot
        )
        rows.append(
            {
                "outer_id": outer.outer_id,
                "source_d92_job_id": f"source-{outer.outer_id}",
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "active_k": outer.k_shot,
                "new_count": outer.new_count,
                "source_pool_k": source_pool,
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
    plan_path = tmp_path / "plan.json"
    context_path = tmp_path / "context.json"
    return (
        plan_path,
        _write_json(plan_path, plan),
        context_path,
        _write_json(context_path, context),
    )


def _feature(token: str) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    row = np.zeros(288, dtype=np.float32)
    row[int.from_bytes(digest[:2], "big") % 288] = np.float32(1.0)
    return row


class FakeMaterializer:
    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        phase = request["phase"]
        new = tuple(f"new_{index}" for index in range(request["new_count"]))
        classes = OLD if phase == "before" else OLD + new
        labels: list[str] = []
        ids: list[str] = []
        features: list[np.ndarray] = []
        for label in classes:
            for shot in range(request["k_shot"]):
                token = f"{request['outer_id']}:{request['scene']}:{label}:{shot}"
                labels.append(label)
                ids.append(token)
                features.append(_feature(token))
        query_ids = (
            f"{request['outer_id']}:{request['scene']}:{phase}:query-0",
            f"{request['outer_id']}:{request['scene']}:{phase}:query-1",
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

    def build(self, *args: Any, **kwargs: Any) -> dict[str, tuple[str, ...]]:
        del kwargs
        self.builds += 1
        old = tuple(args[2])
        new = tuple(args[5])
        return {"before": old, "after": old + new}

    def score(
        self, pair: dict[str, tuple[str, ...]], phase: str, arm: str, query: np.ndarray
    ) -> np.ndarray:
        self.scores.append((phase, arm))
        logits = np.zeros((len(query), len(pair[phase])), dtype=np.float32)
        logits[:, 0] = np.float32(1.0)
        return logits


def _dummy_d92_fit(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("FakeCore must not execute d92_fit")


def _common(tmp_path: Path) -> dict[str, Any]:
    plan, plan_sha, context, context_sha = _prepared(tmp_path)
    return {
        "plan_manifest_path": plan,
        "expected_plan_file_sha256": plan_sha,
        "context_manifest_path": context,
        "expected_context_file_sha256": context_sha,
    }


def test_public_prediction_surface_is_minimal_and_has_no_forbidden_inputs() -> None:
    forbidden = {"truth", "query_truth", "role", "quota", "query_labels", "metrics"}
    for function in (
        target.prepare_d109_target125_run,
        target.smoke_d109_target125_prepared_state,
        target.predict_d109_target125,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    assert target.ARMS == ("M0", "M_DA", "M_HEAD", "M_JOINT")
    assert target.SHARD_COUNT == 8
    assert target.CANDIDATE_ID == "D109-SCRC/r1"
    assert cli.parse_args(["merge", "--shard-manifest", "one", "--output-dir", "out"]).command == "merge"
    assert cli.parse_args(
        [
            "score",
            "--prediction-manifest",
            "prediction.json",
            "--prediction-manifest-sha256",
            "0" * 64,
            "--truth-catalog",
            "truth.json",
            "--truth-catalog-sha256",
            "1" * 64,
            "--output-dir",
            "score",
        ]
    ).command == "score"


def test_prepare_is_a_thin_d108_input_plane_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"plan_manifest": "plan", "context_manifest": "context"}
    seen: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(d108_runner, "prepare_d108_target125_run", fake_prepare)
    assert target.prepare_d109_target125_run(marker="sealed-d108") is sentinel
    assert seen == {"marker": "sealed-d108"}


def test_smoke_defaults_to_d109_core_and_seals_d109_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = FakeCore()
    monkeypatch.setattr(target.d109_core, "build_d109_d92_pair", core.build)
    monkeypatch.setattr(target.d109_core, "score", core.score)
    result = target.smoke_d109_target125_prepared_state(
        **_common(tmp_path),
        output_dir=tmp_path / "smoke",
        state_materializer=FakeMaterializer(),
        d92_fit=_dummy_d92_fit,
    )
    assert result["candidate_id"] == target.CANDIDATE_ID
    assert result["schema"] == target.SMOKE_RECEIPT_SCHEMA
    assert result["status"].endswith("SMOKE_PASS")
    predictions = json.loads(
        Path(result["smoke_predictions"]).read_text(encoding="utf-8")
    )
    assert predictions["candidate_id"] == target.CANDIDATE_ID
    assert predictions["schema"] == target.SMOKE_PREDICTIONS_SCHEMA
    assert predictions["truth_open"] is False
    assert predictions["immutable"] is True
    assert len(predictions["surfaces"]) == 8
    assert core.builds == 1
    assert core.scores == [
        (phase, arm) for arm in target.ARMS for phase in target.PHASES
    ]


def test_immutable_eight_shard_merge_has_distinct_d109_publication(
    tmp_path: Path,
) -> None:
    common = _common(tmp_path)
    original_candidate = d108_runner.CANDIDATE_ID
    original_schema = d108_runner.PREDICTION_MANIFEST_SCHEMA
    shard_paths: list[Path] = []
    for shard_index in range(8):
        core = FakeCore()
        result = target.predict_d109_target125(
            **common,
            output_dir=tmp_path / f"shard{shard_index}",
            shard_index=shard_index,
            state_materializer=FakeMaterializer(),
            pair_builder=core.build,
            query_scorer=core.score,
            d92_fit=_dummy_d92_fit,
        )
        shard_path = Path(result["prediction_shard_manifest"])
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        assert shard["candidate_id"] == target.CANDIDATE_ID
        assert shard["schema"] == target.PREDICTION_SHARD_SCHEMA
        shard_paths.append(shard_path)
    with pytest.raises(target.D109Target125Error, match="exactly eight"):
        target.predict_d109_target125(
            shard_manifest_paths=shard_paths[:7], output_dir=tmp_path / "missing"
        )
    assert not (tmp_path / "missing").exists()
    merged = target.predict_d109_target125(
        shard_manifest_paths=shard_paths, output_dir=tmp_path / "merged"
    )
    manifest = target.validate_d109_target125_prediction_manifest(
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
    )
    assert manifest["candidate_id"] == target.CANDIDATE_ID
    assert manifest["schema"] == target.PREDICTION_MANIFEST_SCHEMA
    assert manifest["surface_count"] == 3000
    assert manifest["truth_open"] is False
    assert d108_runner.CANDIDATE_ID == original_candidate
    assert d108_runner.PREDICTION_MANIFEST_SCHEMA == original_schema


def test_truth_score_adapter_switches_and_restores_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_candidate = d108_truth.CANDIDATE_ID
    original_prediction_schema = d108_truth.PREDICTION_MANIFEST_SCHEMA
    original_fields = d108_truth._PREDICTION_MANIFEST_FIELDS
    seen: dict[str, Any] = {}

    def fake_score(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert d108_truth.CANDIDATE_ID == target.CANDIDATE_ID
        assert (
            d108_truth.PREDICTION_MANIFEST_SCHEMA
            == target.PREDICTION_MANIFEST_SCHEMA
        )
        assert d108_truth.SCORE_MANIFEST_SCHEMA == target.SCORE_MANIFEST_SCHEMA
        assert {"shard_count", "shard_receipts"}.issubset(
            d108_truth._PREDICTION_MANIFEST_FIELDS
        )
        return {"candidate_id": d108_truth.CANDIDATE_ID}

    monkeypatch.setattr(
        target,
        "validate_d109_target125_prediction_manifest",
        lambda **_kwargs: {"manifest_sealed": True},
    )
    monkeypatch.setattr(d108_truth, "score_d108_target125", fake_score)
    result = target.score_d109_target125(
        prediction_manifest_path="prediction.json",
        expected_prediction_manifest_file_sha256="0" * 64,
        truth_catalog_path="truth.json",
        expected_truth_catalog_file_sha256="1" * 64,
        output_dir="score",
    )
    assert result == {"candidate_id": target.CANDIDATE_ID}
    assert seen == {
        "prediction_manifest_path": "prediction.json",
        "expected_prediction_manifest_file_sha256": "0" * 64,
        "truth_catalog_path": "truth.json",
        "expected_truth_catalog_file_sha256": "1" * 64,
        "output_dir": "score",
    }
    assert d108_truth.CANDIDATE_ID == original_candidate
    assert d108_truth.PREDICTION_MANIFEST_SCHEMA == original_prediction_schema
    assert d108_truth._PREDICTION_MANIFEST_FIELDS is original_fields
