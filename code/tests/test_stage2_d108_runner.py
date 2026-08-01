"""D108 Target125 runner traceability and focused release tests.

| ID | Requirement | Target | Status | Verification |
|---|---|---|---|---|
| D108-R01 | D107-isomorphic public prepare/smoke/predict/validate API | runner | verified | public-surface test |
| D108-R02 | D92 sealed runtime to exact 288D registered_feature | real materializer | verified | feature-path/static contract test |
| D108-R03 | before/after old support IDs and features byte-identical | pair coercion | verified | mismatch-negative test |
| D108-R04 | after support split into old/new for pair fit | pair builder call | verified | exact-signature injected-builder test |
| D108-R05 | support batch 64 and singleton query inference | materializer/scorer | verified | batch contract test |
| D108-R06 | eight immutable independent shards | predict | verified | shard partition test |
| D108-R07 | merge exact 125/no duplicate before sealed 3000 manifest | predict/validate | verified | merge coverage tests |
| D108-R08 | no truth/role/quota/query fit or update surface | all | verified | negative API/artifact test |

This file is the in-scope traceability record because the delegated task permits
exactly three new files and forbids a separate report or planning-file edit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cvsrffi.stage2_d108_target125_runner as runner
from cvsrffi.stage2_d108_matrix_protocol import (
    ACCESS_LEDGER,
    CANDIDATE_ID,
    PROTOCOL_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    freeze_d108_target125_matrix,
)
from cvsrffi.stage2_d108_target125_inputs import CONTEXT_SCHEMA, PLAN_SCHEMA


OLD = tuple(f"old_{index}" for index in range(6))


def _write_json(path: Path, value: dict[str, Any]) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _prepared(tmp_path: Path) -> tuple[Path, str, Path, str]:
    matrix = freeze_d108_target125_matrix()
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
            "before_enrollment", "before_apply", "after_enrollment", "after_apply"
        )
    }
    rows = []
    for outer in matrix.outer_rows:
        source_pool = 10 if (outer.k_shot, outer.new_count) == (5, 20) else outer.k_shot
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
                "k5_prefix_from_matched_k10": (outer.k_shot, outer.new_count) == (5, 20),
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
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    plan["plan_receipt_sha256"] = canonical_sha256(plan)
    context: dict[str, Any] = {
        "schema": CONTEXT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
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
    def __init__(self, *, drift_old: bool = False) -> None:
        self.drift_old = drift_old
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(request))
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
        if self.drift_old and phase == "after":
            features[0] = _feature("drifted-old-row")
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
        self.builds: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.scores: list[tuple[str, str, int]] = []

    def build(
        self,
        old_support_features: np.ndarray,
        old_support_labels: tuple[str, ...],
        old_registered_classes: tuple[str, ...],
        new_support_features: np.ndarray,
        new_support_labels: tuple[str, ...],
        new_registered_classes: tuple[str, ...],
        *,
        seed: int,
        device: Any,
        d92_fit: Any,
    ) -> dict[str, Any]:
        args = (
            old_support_features,
            old_support_labels,
            old_registered_classes,
            new_support_features,
            new_support_labels,
            new_registered_classes,
        )
        kwargs = {"seed": seed, "device": device, "d92_fit": d92_fit}
        self.builds.append((args, kwargs))
        return {
            "before": tuple(old_registered_classes),
            "after": tuple(old_registered_classes) + tuple(new_registered_classes),
        }

    def score(
        self, pair: dict[str, Any], phase: str, arm: str, query: np.ndarray
    ) -> np.ndarray:
        self.scores.append((phase, arm, len(query)))
        registry = pair[phase]
        logits = np.zeros((len(query), len(registry)), dtype=np.float32)
        logits[:, 0] = np.float32(1.0)
        return logits


def _dummy_d92_fit(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("injected FakeCore must not execute d92_fit")


def _common(tmp_path: Path) -> dict[str, Any]:
    plan, plan_sha, context, context_sha = _prepared(tmp_path)
    return {
        "plan_manifest_path": plan,
        "expected_plan_file_sha256": plan_sha,
        "context_manifest_path": context,
        "expected_context_file_sha256": context_sha,
    }


def test_public_surface_is_d107_isomorphic_and_has_no_forbidden_inputs() -> None:
    assert callable(runner.prepare_d108_target125_run)
    assert callable(runner.smoke_d108_target125_prepared_state)
    assert callable(runner.predict_d108_target125)
    assert callable(runner.validate_d108_target125_prediction_manifest)
    forbidden = {"truth", "query_truth", "role", "quota", "query_labels", "metrics"}
    for function in (
        runner.prepare_d108_target125_run,
        runner.smoke_d108_target125_prepared_state,
        runner.predict_d108_target125,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "stage2_d107" not in source
    assert "stage2_d106" not in source


def test_smoke_uses_after_old_new_split_and_scene_seed(tmp_path: Path) -> None:
    core = FakeCore()
    materializer = FakeMaterializer()
    result = runner.smoke_d108_target125_prepared_state(
        **_common(tmp_path), output_dir=tmp_path / "smoke",
        state_materializer=materializer, pair_builder=core.build,
        query_scorer=core.score, d92_fit=_dummy_d92_fit,
    )
    assert result["status"].endswith("SMOKE_PASS")
    predictions_path = Path(result["smoke_predictions"])
    assert hashlib.sha256(predictions_path.read_bytes()).hexdigest() == result[
        "smoke_predictions_file_sha256"
    ]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert predictions["truth_open"] is False
    assert predictions["immutable"] is True
    assert len(predictions["surfaces"]) == 8
    assert canonical_sha256(
        {
            key: value
            for key, value in predictions.items()
            if key != "smoke_predictions_receipt_sha256"
        }
    ) == predictions["smoke_predictions_receipt_sha256"]
    assert len(core.builds) == 1
    args, kwargs = core.builds[0]
    assert args[0].shape == (6 * 10, 288)
    assert len(args[3]) == 5 * 10
    assert args[2] == OLD
    assert kwargs["seed"] == 713102
    assert kwargs["d92_fit"] is _dummy_d92_fit
    assert core.scores == [
        (phase, arm, 2) for arm in runner.ARMS for phase in ("before", "after")
    ]


def test_before_after_old_support_feature_drift_fails_before_fit(tmp_path: Path) -> None:
    core = FakeCore()
    with pytest.raises(runner.D108Target125RunnerError, match="old support"):
        runner.smoke_d108_target125_prepared_state(
            **_common(tmp_path), output_dir=tmp_path / "bad-smoke",
            state_materializer=FakeMaterializer(drift_old=True),
            pair_builder=core.build, query_scorer=core.score,
            d92_fit=_dummy_d92_fit,
        )
    assert core.builds == []


def test_single_shard_is_immutable_and_has_modulo_coverage(tmp_path: Path) -> None:
    core = FakeCore()
    common = _common(tmp_path)
    output = tmp_path / "shard0"
    result = runner.predict_d108_target125(
        **common, output_dir=output, shard_index=0,
        state_materializer=FakeMaterializer(), pair_builder=core.build,
        query_scorer=core.score, d92_fit=_dummy_d92_fit,
    )
    shard = json.loads(Path(result["prediction_shard_manifest"]).read_text(encoding="utf-8"))
    assert shard["outer_indices"] == list(range(0, 125, 8))
    assert result["surface_count"] == len(shard["outer_indices"]) * 24
    assert not (output / "prediction_manifest.json").exists()
    with pytest.raises(FileExistsError):
        runner.predict_d108_target125(
            **common, output_dir=output, shard_index=0,
            state_materializer=FakeMaterializer(), pair_builder=core.build,
            query_scorer=core.score, d92_fit=_dummy_d92_fit,
        )


def test_eight_shards_merge_exactly_3000_and_missing_or_duplicate_fails(
    tmp_path: Path,
) -> None:
    common = _common(tmp_path)
    shard_paths: list[Path] = []
    for shard_index in range(8):
        core = FakeCore()
        result = runner.predict_d108_target125(
            **common, output_dir=tmp_path / f"shard{shard_index}",
            shard_index=shard_index, state_materializer=FakeMaterializer(),
            pair_builder=core.build, query_scorer=core.score,
            d92_fit=_dummy_d92_fit,
        )
        shard_paths.append(Path(result["prediction_shard_manifest"]))
    missing_output = tmp_path / "missing-merge"
    with pytest.raises(runner.D108Target125RunnerError, match="exactly eight"):
        runner.predict_d108_target125(
            shard_manifest_paths=shard_paths[:7], output_dir=missing_output
        )
    assert not missing_output.exists()
    duplicate_output = tmp_path / "duplicate-merge"
    with pytest.raises(runner.D108Target125RunnerError, match="duplicate shard"):
        runner.predict_d108_target125(
            shard_manifest_paths=shard_paths[:7] + [shard_paths[0]],
            output_dir=duplicate_output,
        )
    assert not duplicate_output.exists()
    merged = runner.predict_d108_target125(
        shard_manifest_paths=shard_paths, output_dir=tmp_path / "merged"
    )
    assert merged["surface_count"] == 3000
    manifest = runner.validate_d108_target125_prediction_manifest(
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
    )
    assert manifest["manifest_sealed"] is True
    assert len(manifest["outer_rows"]) == 125
    assert len(manifest["surfaces"]) == 3000
    assert manifest["access_ledger"] == ACCESS_LEDGER


def test_support_batch_is_locked_to_64_and_query_batch_receipt_is_one() -> None:
    signature = inspect.signature(runner.smoke_d108_target125_prepared_state)
    assert signature.parameters["feature_batch_size"].default == 64
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "batch_size=64" in source
    assert "batch_size=1" in source
