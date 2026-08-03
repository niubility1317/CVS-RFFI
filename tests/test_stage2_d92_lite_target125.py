from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import cvsrffi.stage2_d108_target125_runner as d108_runner
import cvsrffi.stage2_d108_truth_scorer as d108_truth
import cvsrffi.stage2_d92_lite_target125 as target
from scripts import run_d92_lite_target125 as cli
from test_stage2_d109_target125 import FakeCore, FakeMaterializer, _common, _dummy_d92_fit


METHOD_LOCK = Path(__file__).resolve().parents[1] / "configs" / "d131_d92_lite160_qtie_target125_r2.json"
METHOD = {
    "method_lock_path": METHOD_LOCK,
    "expected_method_lock_sha256": target.core.METHOD_LOCK_SHA256,
}


@pytest.fixture(autouse=True)
def _formal_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(target.FORMAL_ISOLATION_ENV, "1")


def test_single_candidate_public_contract_and_cli() -> None:
    forbidden = {"truth", "query_truth", "role", "quota", "query_labels", "metrics"}
    for function in (
        target.prepare_d92_lite_target125_run,
        target.smoke_d92_lite_target125_prepared_state,
        target.predict_d92_lite_target125,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    assert target.ARMS == ("M_JOINT",)
    assert target.ARM_PAIR_COUNT == 375
    assert target.SURFACE_COUNT == 750
    assert len(target._filtered_matrix().surfaces) == 750
    assert cli.parse_args(
        ["merge", "--method-lock", str(METHOD_LOCK), "--method-lock-sha256",
         target.core.METHOD_LOCK_SHA256, "--shard-manifest", "one",
         "--output-dir", "out"]
    ).command == "merge"


def test_smoke_is_two_phase_single_candidate(tmp_path: Path) -> None:
    fake = FakeCore()
    result = target.smoke_d92_lite_target125_prepared_state(
        **METHOD, **_common(tmp_path), output_dir=tmp_path / "smoke",
        state_materializer=FakeMaterializer(), pair_builder=fake.build,
        query_scorer=fake.score, d92_fit=_dummy_d92_fit,
    )
    assert result["status"] == "D92_LITE160_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS"
    assert result["transport_arm_is_D108_joint_mechanism"] is False
    assert fake.builds == 1
    assert fake.scores == [("before", "M_JOINT"), ("after", "M_JOINT")]


def test_eight_shard_merge_closes_exactly_750_surfaces(tmp_path: Path) -> None:
    common = _common(tmp_path)
    fake = FakeCore()
    manifests = []
    for index in range(8):
        result = target.predict_d92_lite_target125(
            **METHOD, **common, output_dir=tmp_path / f"shard_{index}", shard_index=index,
            state_materializer=FakeMaterializer(), pair_builder=fake.build,
            query_scorer=fake.score, d92_fit=_dummy_d92_fit,
        )
        manifests.append(Path(result["prediction_shard_manifest"]))
    merged = target.predict_d92_lite_target125(
        **METHOD, shard_manifest_paths=manifests, output_dir=tmp_path / "merged"
    )
    assert merged["outer_job_count"] == 125
    assert merged["arm_pair_count"] == 375
    assert merged["surface_count"] == 750
    validated = target.validate_d92_lite_target125_prediction_manifest(
        prediction_manifest_path=Path(merged["prediction_manifest"]),
        expected_prediction_manifest_file_sha256=merged[
            "prediction_manifest_file_sha256"
        ],
        **METHOD,
    )
    assert validated["surface_count"] == 750


def test_module_projection_is_exception_safe_and_truth_summary_is_honest() -> None:
    original_arms = d108_runner.ARMS
    original_surface_count = d108_runner.SURFACE_COUNT
    with pytest.raises(RuntimeError):
        with target._prediction_projection():
            assert d108_runner.ARMS == ("M_JOINT",)
            assert d108_runner.SURFACE_COUNT == 750
            raise RuntimeError("probe")
    assert d108_runner.ARMS == original_arms
    assert d108_runner.SURFACE_COUNT == original_surface_count
    original_summary = d108_truth._score_summary
    with target._truth_projection():
        coverage, _resources, verdict = d108_truth._score_summary(
            {"manifest_size": 1, "artifact_bytes": 2,
             "prediction_query_count": 3, "support_slots": 4},
            truth_catalog_size=5,
        )
        assert coverage["four_arm_causal_coverage_verified"] is False
        assert coverage["single_candidate_before_after_coverage_verified"] is True
        assert verdict["transport_arm_is_D108_joint_mechanism"] is False
    assert d108_truth._score_summary is original_summary
