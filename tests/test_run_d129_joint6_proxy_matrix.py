from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d129_joint6_proxy_matrix.py"
SMOKE_TEST = ROOT / "tests" / "test_run_d129_joint6_real_archive_smoke.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("d129_proxy_matrix", SCRIPT)
fixtures = _load("d129_proxy_matrix_fixtures", SMOKE_TEST)


def test_prepare_predict_score_complete_proxy_matrix_without_truth_leak(
    tmp_path: Path,
) -> None:
    archive, archive_sha, fixture, fixture_sha, checkpoint_sha = fixtures._inputs(
        tmp_path
    )
    method_lock = ROOT / "configs" / "d129_joint6_method_lock_20260803.json"
    method_lock_sha = hashlib.sha256(method_lock.read_bytes()).hexdigest()
    prepare_dir = tmp_path / "prepare"
    prepared = runner.prepare_proxy_matrix(
        archive_path=archive.resolve(),
        archive_sha256=archive_sha,
        fixture_path=fixture.resolve(),
        fixture_sha256=fixture_sha,
        checkpoint_sha256=checkpoint_sha,
        method_lock_path=method_lock.resolve(),
        method_lock_sha256=method_lock_sha,
        capsule_id="source-proxy-capsule",
        split_id="source-proxy-split",
        run_id="d129-proxy-test",
        output_dir=prepare_dir.resolve(),
    )
    assert prepared["query_truth_in_predictor_package"] is False
    package = prepare_dir / "predictor_package.npz"
    assert "truth" not in prepared["package_members"]
    prediction_dir = tmp_path / "predict"
    predicted = runner.predict_proxy_matrix(
        package_path=package.resolve(),
        package_sha256=prepared["package_sha256"],
        output_dir=prediction_dir.resolve(),
    )
    assert predicted["prediction_row_count"] == 168
    assert predicted["rows_complete"] is True
    assert predicted["truth_loaded"] is False
    prediction = json.loads(
        (prediction_dir / "predictions.json").read_text(encoding="utf-8")
    )
    assert prediction["truth_loaded"] is False
    assert len(prediction["rows"]) == 168
    resources = json.loads(
        (prediction_dir / "resources.json").read_text(encoding="utf-8")
    )
    assert len(resources["rows"]) == 168
    k5_resources = [
        row for row in resources["rows"] if "|K=5" in row["row_id"]
    ]
    assert k5_resources
    for row in k5_resources:
        assert set(row["affine_logit_scale_audits"]) == {
            "R0F", "R0L", "R1F", "R1L"
        }
        for audit in row["affine_logit_scale_audits"].values():
            assert audit["argmax_equivalence_scope"] == (
                "prequantized_common_positive_scaling_only"
            )
            assert audit["quantized_any_query_argmax_equivalence_claim"] is False
    encoded_prediction = json.dumps(prediction, sort_keys=True).lower()
    assert '"truth":' not in encoded_prediction
    assert "query_label" not in encoded_prediction
    score_path = tmp_path / "score.json"
    scored = runner.score_proxy_matrix(
        prediction_path=(prediction_dir / "predictions.json").resolve(),
        prediction_sha256=predicted["prediction_sha256"],
        plan_path=(prepare_dir / "plan.json").resolve(),
        plan_sha256=prepared["plan_sha256"],
        truth_path=(prepare_dir / "truth.json").resolve(),
        truth_sha256=prepared["truth_sha256"],
        output_path=score_path.resolve(),
    )
    assert scored["candidate_count"] == 2
    assert scored["truth_opened_after_complete_prediction"] is True
