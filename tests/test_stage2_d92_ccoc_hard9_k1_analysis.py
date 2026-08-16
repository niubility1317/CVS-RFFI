from __future__ import annotations

import hashlib
import json
import csv
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_ccoc_hard9_k1 as matrix  # noqa: E402
from cvsrffi import stage2_d92_ccoc_hard9_k1_analysis as analysis  # noqa: E402
from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair  # noqa: E402


_ORIGINAL_MATRIX_STATE = {
    "RAW_SCORE_ROOT": matrix.RAW_SCORE_ROOT,
    "RAW_SCORE_SHA": dict(matrix.RAW_SCORE_SHA),
    "HISTORICAL_BASELINE_PATH": matrix.HISTORICAL_BASELINE_PATH,
    "HISTORICAL_BASELINE_SHA256": matrix.HISTORICAL_BASELINE_SHA256,
    "HISTORICAL_PER_OLD_CLASS_PATH": matrix.HISTORICAL_PER_OLD_CLASS_PATH,
    "HISTORICAL_PER_OLD_CLASS_SHA256": matrix.HISTORICAL_PER_OLD_CLASS_SHA256,
}


@pytest.fixture(autouse=True)
def _restore_matrix_fixture_constants() -> None:
    """Keep canonical test-path relocation local to each test case."""

    yield
    for name, value in _ORIGINAL_MATRIX_STATE.items():
        setattr(matrix, name, dict(value) if isinstance(value, dict) else value)


def test_analyzer_exports_frozen_verdicts() -> None:
    assert analysis.VERDICTS == (
        "REJECT_ROUTE",
        "REVISE_ONCE",
        "ADVANCE_TO_TARGET125_CANDIDATE",
    )


def _score(delta: float = 0.02) -> dict[str, object]:
    old = {f"tx_{i}": {"role": "target_old", "accuracy": 0.70 + delta} for i in range(6)}
    before_scene = {scene: {"query_count": 60} for scene in analysis.SCENES}
    after_scene = {
        scene: {
            "query_count": 80,
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        }
        for scene in analysis.SCENES
    }
    return {
        "candidate": analysis.CANDIDATE_ID,
        "truth_sidecar_sha256": "a" * 64,
        "before": {"old_acc": 0.70, "by_scenario": before_scene, "by_tx": old},
        "after": {
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "by_tx": old,
            "by_scenario": after_scene,
            "query_macs": 3168,
            "after_state_bytes": 8583,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        },
        "old_forgetting_pp": -delta * 100.0,
        "per_old_class_floor_before": 0.70,
    }


def test_score_metrics_and_strict_directions_are_explicitly_eight_metric() -> None:
    metrics = analysis.compute_score_metrics(_score())
    baseline = analysis.compute_score_metrics(_score(0.0))
    deltas = analysis.strict_pareto_deltas(metrics, baseline)
    assert metrics["old_class_count"] == 6
    assert all(deltas[name] > 0 for name in analysis.EIGHT_PARETO_METRICS[:5])
    assert deltas["average_forgetting"] < 0
    assert deltas["new_to_old_rate"] < 0
    assert deltas["old_to_new_rate"] < 0


def test_resource_gate_peak_is_absolute_and_512k_is_target_only() -> None:
    rows = [
        {
            "outer_key": "outer",
            "scenario": scene,
            "candidate_wall_ns": 100_000_000,
            "wall_ratio": 1.0,
            "candidate_peak_bytes": 729_088,
            "wall_hard_pass": True,
            "ratio_hard_pass": True,
            "peak_hard_pass": True,
            "wall_target_pass": True,
            "ratio_target_pass": True,
            "peak_target_pass": False,
            "query_state_exact": True,
        }
        for scene in analysis.SCENES
    ]
    result = analysis.evaluate_resource_gate(rows)
    assert result["hard_passed"] is True
    assert result["target_passed"] is False
    assert result["candidate_peak_max_bytes"] == 729_088


def test_decide_verdict_hard_failure_precedes_revision() -> None:
    base = {
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": False,
        "stability": True,
        "resource_integrity": True,
        "resource_hard": False,
        "resource_target": False,
    }
    assert analysis.decide_verdict(base) == "REJECT_ROUTE"
    base["resource_hard"] = True
    assert analysis.decide_verdict(base) == "REVISE_ONCE"
    base["all_magnitude"] = True
    base["resource_target"] = True
    assert analysis.decide_verdict(base) == "ADVANCE_TO_TARGET125_CANDIDATE"


def test_truth_binding_reads_actual_sidecar_bytes(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text("truth", encoding="utf-8")
    digest = hashlib.sha256(truth.read_bytes()).hexdigest()
    job = {"truth_sidecar": str(truth), "truth_sidecar_sha256": digest}
    receipt = {"truth_sidecar_sha256": digest}
    score = {"truth_sidecar_sha256": digest}
    assert analysis.validate_truth_binding(score, receipt, job, truth) == digest
    truth.write_text("drift", encoding="utf-8")
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError):
        analysis.validate_truth_binding(score, receipt, job, truth)


def test_output_package_is_exclusive_and_contains_all_seven_files(tmp_path: Path) -> None:
    result = {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.analysis.v1",
        "verdict": "REVISE_ONCE",
        "status": "ANALYZED",
        "gate_state": {},
        "gates": {},
        "aggregate": {},
        "paired_rows": [],
        "per_old_class_rows": [],
        "scenario_rows": [],
        "liveness_rows": [],
    }
    output_root = tmp_path / "analysis"
    paths = analysis.write_analysis_outputs(result, output_root)
    assert set(paths) == {
        "summary.json",
        "gates.json",
        "paired_rows.csv",
        "per_old_class_rows.csv",
        "scenario_rows.csv",
        "liveness_rows.csv",
        "analysis.md",
    }
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="overwrite"):
        analysis.write_analysis_outputs(result, output_root)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_commit(state_root: Path) -> None:
    members = []
    for name in ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json"):
        path = state_root / name
        members.append({"relative_path": name, "sha256": _digest(path), "size_bytes": path.stat().st_size})
    _write_json(
        state_root / "COMMIT.json",
        {
            "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
            "members": members,
            "prediction_artifact_sha256": _digest(state_root / "prediction_artifact.npz"),
        },
    )


def _refresh_commit(state_root: Path) -> None:
    commit = json.loads((state_root / "COMMIT.json").read_text(encoding="utf-8"))
    members = []
    for name in ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json"):
        path = state_root / name
        members.append({"relative_path": name, "sha256": _digest(path), "size_bytes": path.stat().st_size})
    commit["members"] = members
    commit["prediction_artifact_sha256"] = _digest(state_root / "prediction_artifact.npz")
    _write_json(state_root / "COMMIT.json", commit)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _token(outer: str, scene: str, role: str, index: int, *, alternate: bool = False) -> str:
    suffix = ":alternate" if alternate else ""
    return f"{outer}:{scene}:{role}:{index}{suffix}"


def _write_truth_sidecar(path: Path, outer: str) -> None:
    """Create one canonical manifest-bound sidecar used by E0 and candidate."""

    rows: list[dict[str, str]] = []
    for scene in analysis.SCENES:
        for index in range(6):
            common = {
                "true_class_handle": f"old_handle_{index}",
                "transmitter_label": f"tx_{index}",
                "evaluation_role": "target_old",
            }
            rows.append({"query_token": _token(outer, scene, "old", index), **common})
            if index == 0:
                rows.append(
                    {
                        "query_token": _token(outer, scene, "old", index, alternate=True),
                        **common,
                    }
                )
        for index in range(2):
            rows.append(
                {
                    "query_token": _token(outer, scene, "new", index),
                    "true_class_handle": f"new_handle_{index}",
                    "transmitter_label": f"new_tx_{index}",
                    "evaluation_role": "target_new",
                }
            )
    _write_json(path, {"schema": "cvs.phase2.query_truth_sidecar.v2", "rows": rows})


def _write_prediction_artifact(
    path: Path,
    outer: str,
    *,
    state: str,
    candidate: bool,
    raw_query_mismatch: bool = False,
) -> None:
    """Emit the real scorer's exact immutable NPZ shape, never a fake sidecar."""

    tokens: list[str] = []
    scenes: list[str] = []
    predicted: list[str] = []
    for scene in analysis.SCENES:
        for index in range(6):
            use_alternate = raw_query_mismatch and not candidate and index == 0
            tokens.append(_token(outer, scene, "old", index, alternate=use_alternate))
            scenes.append(scene)
            if state == "after" and candidate:
                predicted.append(f"old_handle_{index}")
            elif state == "after" and index in {0, 1}:
                predicted.append("new_handle_0")
            elif state == "before" and index == 0:
                predicted.append("new_handle_0")
            else:
                predicted.append(f"old_handle_{index}")
        if state == "after":
            for index in range(2):
                tokens.append(_token(outer, scene, "new", index))
                scenes.append(scene)
                predicted.append(
                    f"new_handle_{index}" if candidate or index == 1 else "old_handle_0"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        query_tokens=np.asarray(tokens, dtype=str),
        scenarios=np.asarray(scenes, dtype=str),
        predicted_class_handles=np.asarray(predicted, dtype=str),
    )
    os.chmod(path, stat.S_IREAD)


def _closure_hashes(job_root: Path) -> dict[str, str]:
    files = {
        "before_prediction_sha256": ("before", "prediction_artifact.npz"),
        "after_prediction_sha256": ("after", "prediction_artifact.npz"),
        "before_commit_sha256": ("before", "COMMIT.json"),
        "after_commit_sha256": ("after", "COMMIT.json"),
        "before_fit_audit_sha256": ("before", "fit_audit.json"),
        "after_fit_audit_sha256": ("after", "fit_audit.json"),
        "before_resource_audit_sha256": ("before", "resource_audit.json"),
        "after_resource_audit_sha256": ("after", "resource_audit.json"),
        "before_execution_receipt_sha256": ("before", "execution_receipt.json"),
        "after_execution_receipt_sha256": ("after", "execution_receipt.json"),
    }
    return {
        field: _digest(job_root / "diag" / state / name)
        for field, (state, name) in files.items()
    }


def _refresh_closure_evidence(job_root: Path) -> None:
    """Model Task1 re-binding after a deliberate valid artifact mutation."""

    hashes = _closure_hashes(job_root)
    binding_path = job_root / "score_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding.update(hashes)
    _write_json(binding_path, binding)
    receipt_path = job_root / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(hashes)
    receipt["prediction_closure"] = dict(hashes)
    receipt["score_binding_sha256"] = _digest(binding_path)
    receipt["score_sha256"] = _digest(job_root / "scorer" / "diag_cosine_score.json")
    receipt["score_evidence"].update(hashes)
    receipt["score_evidence"]["score_artifact_sha256"] = receipt["score_sha256"]
    _write_json(receipt_path, receipt)


def _refresh_score_evidence(job_root: Path) -> None:
    receipt_path = job_root / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    score_sha = _digest(job_root / "scorer" / "diag_cosine_score.json")
    receipt["score_sha256"] = score_sha
    receipt["score_evidence"]["score_artifact_sha256"] = score_sha
    _write_json(receipt_path, receipt)


def _rewrite_candidate_after_scene_for_stability(
    job_root: Path,
    truth_path: Path,
    scene: str,
) -> None:
    """Make one real scene worse while leaving the outer aggregate strict."""

    state_root = job_root / "diag" / "after"
    artifact = state_root / "prediction_artifact.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        tokens = archive["query_tokens"].astype(str)
        scenes = archive["scenarios"].astype(str)
        predicted = archive["predicted_class_handles"].astype(str)
    changed = 0
    for index, (token, observed_scene) in enumerate(zip(tokens.tolist(), scenes.tolist())):
        if observed_scene == scene and ":old:" in token and changed < 3:
            predicted[index] = "new_handle_0"
            changed += 1
    assert changed == 3
    os.chmod(artifact, stat.S_IREAD | stat.S_IWRITE)
    np.savez(
        artifact,
        query_tokens=tokens,
        scenarios=scenes,
        predicted_class_handles=predicted,
    )
    os.chmod(artifact, stat.S_IREAD)
    _refresh_commit(state_root)
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    os.chmod(score_path, stat.S_IREAD | stat.S_IWRITE)
    score_path.unlink()
    score_diag_cosine_pair(
        before_prediction_path=job_root / "diag" / "before" / "prediction_artifact.npz",
        after_prediction_path=artifact,
        truth_sidecar_path=truth_path,
        output_path=score_path,
        candidate=analysis.CANDIDATE_ID,
    )
    _refresh_closure_evidence(job_root)


def _rewrite_candidate_after_as_e0(job_root: Path, truth_path: Path, outer: str) -> None:
    """Create a real post-registration E0 tie and rebind every Task1 SHA."""

    state_root = job_root / "diag" / "after"
    artifact = state_root / "prediction_artifact.npz"
    os.chmod(artifact, stat.S_IREAD | stat.S_IWRITE)
    _write_prediction_artifact(artifact, outer, state="after", candidate=False)
    _refresh_commit(state_root)
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    os.chmod(score_path, stat.S_IREAD | stat.S_IWRITE)
    score_path.unlink()
    score_diag_cosine_pair(
        before_prediction_path=job_root / "diag" / "before" / "prediction_artifact.npz",
        after_prediction_path=artifact,
        truth_sidecar_path=truth_path,
        output_path=score_path,
        candidate=analysis.CANDIDATE_ID,
    )
    _refresh_closure_evidence(job_root)


def _integration_fixture(
    tmp_path: Path,
    *,
    raw_truth_mismatch: bool = False,
    raw_query_mismatch: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Build a strict fixture from actual scorer-shaped artifacts.

    E0 and CCOC deliberately consume one immutable, manifest-bound truth
    sidecar.  The raw score is produced by the real post-prediction scorer;
    it contains no analyzer-only query-identity or per-scene synthetic fields.
    """

    raw_root = tmp_path / "e0_raw"
    paired_path = tmp_path / "historical" / "paired_rows.csv"
    per_old_path = tmp_path / "historical" / "per_old_class_rows.csv"
    truth_root = tmp_path / "truth"
    selected_rows = matrix._expected_rows()
    mismatch_outer = str(selected_rows[0]["outer_key"])
    truth_sha: dict[str, str] = {}
    for row in selected_rows:
        outer = str(row["outer_key"])
        truth = truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"
        _write_truth_sidecar(truth, outer)
        truth_sha[outer] = _digest(truth)

    raw_scores: dict[str, dict[str, object]] = {}
    for row in selected_rows:
        outer = str(row["outer_key"])
        e0_root = raw_root / outer / "E0_FULL_ONLY"
        _write_prediction_artifact(
            e0_root / "diag" / "before" / "prediction_artifact.npz",
            outer,
            state="before",
            candidate=False,
            raw_query_mismatch=raw_query_mismatch and outer == mismatch_outer,
        )
        _write_prediction_artifact(
            e0_root / "diag" / "after" / "prediction_artifact.npz",
            outer,
            state="after",
            candidate=False,
            raw_query_mismatch=raw_query_mismatch and outer == mismatch_outer,
        )
        raw_score_path = e0_root / "scorer" / "diag_cosine_score.json"
        score_diag_cosine_pair(
            before_prediction_path=e0_root / "diag" / "before" / "prediction_artifact.npz",
            after_prediction_path=e0_root / "diag" / "after" / "prediction_artifact.npz",
            truth_sidecar_path=truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json",
            output_path=raw_score_path,
            candidate="e0_full_only",
        )
        raw_score = json.loads(raw_score_path.read_text(encoding="utf-8"))
        if raw_truth_mismatch and outer == mismatch_outer:
            raw_score["truth_sidecar_sha256"] = "f" * 64
            _write_json(raw_score_path, raw_score)
        raw_scores[outer] = json.loads(raw_score_path.read_text(encoding="utf-8"))

    paired_rows: list[dict[str, object]] = []
    per_old_rows: list[dict[str, object]] = []
    for row in selected_rows:
        outer = str(row["outer_key"])
        raw_score = raw_scores[outer]
        before = raw_score["before"]
        after = raw_score["after"]
        assert isinstance(before, dict) and isinstance(after, dict)
        scenes = matrix.E0_RESOURCE_ROWS[outer]["scenes"]
        query_macs = {int(value["query_macs"]) for value in scenes.values()}
        state_bytes = {int(value["state_bytes"]) for value in scenes.values()}
        assert len(query_macs) == len(state_bytes) == 1
        paired_rows.append(
            {
                "outer_key": outer,
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "new_class_count": row["new_class_count"],
                "slice": f"K{row['k_shot']}_new{row['new_class_count']}",
                "candidate_h_old_new": after["h_old_new"],
                "candidate_old_acc": after["old_acc"],
                "candidate_old_floor": raw_score["per_old_class_floor_after"],
                "candidate_seen_new_acc": after["seen_new_acc"],
                "candidate_forgetting": float(raw_score["old_forgetting_pp"]) / 100.0,
                "candidate_da1_reg0_old_acc": before["old_acc"],
                "candidate_da1_reg0_old_floor": raw_score["per_old_class_floor_before"],
                "query_macs": next(iter(query_macs)),
                "state_bytes": next(iter(state_bytes)),
            }
        )
        by_tx = after["by_tx"]
        assert isinstance(by_tx, dict)
        for tx in range(6):
            accuracy = by_tx[f"tx_{tx}"]["accuracy"]
            per_old_rows.append(
                {
                    "outer_key": outer,
                    "tx": f"tx_{tx}",
                    "candidate_accuracy": accuracy,
                    "baseline_accuracy": accuracy,
                    "delta_accuracy": 0.0,
                }
            )
    _write_csv(paired_path, paired_rows)
    _write_csv(per_old_path, per_old_rows)

    matrix.RAW_SCORE_ROOT = str(raw_root)
    matrix.RAW_SCORE_SHA = {
        outer: _digest(raw_root / outer / "E0_FULL_ONLY" / "scorer" / "diag_cosine_score.json")
        for outer in raw_scores
    }
    matrix.HISTORICAL_BASELINE_PATH = str(paired_path)
    matrix.HISTORICAL_BASELINE_SHA256 = _digest(paired_path)
    matrix.HISTORICAL_PER_OLD_CLASS_PATH = str(per_old_path)
    matrix.HISTORICAL_PER_OLD_CLASS_SHA256 = _digest(per_old_path)
    lock_path = tmp_path / "method_lock.json"
    _write_json(lock_path, matrix._expected_lock())
    manifest = matrix.build_hard9_k1_manifest(lock_path, require_package_files=False)
    for job in manifest["jobs"]:
        outer = str(job["outer_key"])
        job["truth_sidecar_sha256"] = truth_sha[outer]
        for package_name, package in job["packages"].items():
            package["expected_seal_sha256"] = hashlib.sha256(
                f"seal:{outer}:{package_name}".encode("utf-8")
            ).hexdigest()
    manifest_path = tmp_path / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    matrix.validate_hard9_k1_manifest(
        manifest,
        expected_method_lock_sha256=_digest(lock_path),
        require_package_hashes=True,
    )

    run_root = tmp_path / "run"
    manifest_sha = _digest(manifest_path)
    for job in manifest["jobs"]:
        outer = str(job["outer_key"])
        job_root = run_root / "jobs" / outer / analysis.ARM_ID
        fit_rows: list[dict[str, object]] = []
        for scene in analysis.SCENES:
            active = int(job["k_shot"]) > 2
            prefix = "d92_e0d_ccoc_"
            e0_resource = job["e0_resource"]["scenes"][scene]
            fit_row: dict[str, object] = {
                "scenario": scene,
                "arm_id": analysis.ARM_ID,
                "candidate_id": analysis.CANDIDATE_ID,
                "after_state_postprocess_mode": None,
                "after_total_component_fit_count": 2 if active else 3,
                "after_actual_component_inventory": {"actual_component_fit_count": 1 if active else 3},
                "after_registered_d_mode_effective": "ccoc_full" if active else "d92_full_alias",
                "registered_class_count": 6 + int(job["new_class_count"]),
                "query_macs": int(e0_resource["query_macs"]),
                "after_state_bytes": int(e0_resource["state_bytes"]),
                "after_registration_resource": {
                    "registration_wall_time_ns": int(int(e0_resource["registration_wall_time_ns"]) * 1.05),
                    "registration_incremental_peak_working_set_bytes": 500_000,
                },
                prefix + "active": active,
                prefix + "fallback_active": False,
                prefix + "fallback_reason": None if active else matrix.FIT_GATE["k1_alias"],
                prefix + "candidate_attempt_fit_count": 1 if active else 0,
                prefix + "fallback_reference_fit_count": 0,
                prefix + "candidate_statistic_receipt_available": active,
                prefix + "paired_e0_codec_state_equal": None,
                prefix + "g0_eligible": active,
                prefix + "g0_block_reason": None if active else matrix.FIT_GATE["k1_alias"],
                prefix + "query_rows_used": 0,
            }
            for field in matrix.QUERY_ZERO_FIELDS:
                fit_row[field] = False
                fit_row["d92_e0d_" + field] = False
                fit_row[prefix + field] = False
            fit_rows.append(fit_row)
        for state_name in ("before", "after"):
            state_root = job_root / "diag" / state_name
            state_root.mkdir(parents=True, exist_ok=True)
            _write_json(state_root / "execution_receipt.json", {"state": state_name})
            _write_json(state_root / "fit_audit.json", fit_rows)
            _write_prediction_artifact(
                state_root / "prediction_artifact.npz",
                outer,
                state=state_name,
                candidate=True,
            )
            _write_json(state_root / "resource_audit.json", {"state": state_name})
            _make_commit(state_root)
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        score_diag_cosine_pair(
            before_prediction_path=job_root / "diag" / "before" / "prediction_artifact.npz",
            after_prediction_path=job_root / "diag" / "after" / "prediction_artifact.npz",
            truth_sidecar_path=truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json",
            output_path=score_path,
            candidate=analysis.CANDIDATE_ID,
        )
        closure_hashes = _closure_hashes(job_root)
        binding_path = job_root / "score_binding.json"
        _write_json(
            binding_path,
            {
                "schema": "cvs.phase2.d92_ccoc_hard9_k1.score_binding.v1",
                "job_id": job["job_id"],
                "outer_key": outer,
                "outer_role": job["outer_role"],
                "arm_id": analysis.ARM_ID,
                "candidate": analysis.CANDIDATE_ID,
                "matrix_manifest_sha256": manifest_sha,
                "method_lock_sha256": manifest["method_lock_sha256"],
                "truth_sidecar": str(job["truth_sidecar"]),
                "truth_sidecar_sha256": job["truth_sidecar_sha256"],
                **closure_hashes,
                "score_command": ["python", "score"],
                "performance_result_allowed": False,
            },
        )
        score_sha = _digest(score_path)
        score_evidence = {
            "job_id": job["job_id"],
            "outer_key": outer,
            "arm_id": analysis.ARM_ID,
            "candidate": analysis.CANDIDATE_ID,
            "matrix_manifest_sha256": manifest_sha,
            "method_lock_sha256": manifest["method_lock_sha256"],
            "score_artifact_sha256": score_sha,
            "truth_sidecar_sha256": job["truth_sidecar_sha256"],
            **closure_hashes,
        }
        _write_json(
            job_root / "job_receipt.json",
            {
                "schema": matrix.JOB_RECEIPT_SCHEMA,
                "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
                "job_id": job["job_id"],
                "outer_key": outer,
                "outer_role": job["outer_role"],
                "k_shot": job["k_shot"],
                "arm_id": analysis.ARM_ID,
                "candidate": analysis.CANDIDATE_ID,
                "matrix_manifest_sha256": manifest_sha,
                "method_lock_sha256": manifest["method_lock_sha256"],
                "selection_sha256": matrix.CANONICAL_SELECTION_SHA256,
                **closure_hashes,
                "prediction_closure": dict(closure_hashes),
                "score_sha256": score_sha,
                "truth_sidecar_sha256": job["truth_sidecar_sha256"],
                "truth_sidecar_sha256_before_score": job["truth_sidecar_sha256"],
                "truth_sidecar_sha256_after_score": job["truth_sidecar_sha256"],
                "score_binding": str(binding_path),
                "score_binding_sha256": _digest(binding_path),
                "score_evidence": score_evidence,
                "truth_sidecar_exposed_to_predictor": False,
                "query_truth_joined_only_after_immutable_predictions": True,
                "query_truth_fed_back_to_predictor": False,
                "prediction_and_scorer_processes_isolated": True,
                "fresh_run_retry_authorized": False,
            },
        )
    return manifest_path, lock_path, run_root, truth_root


def test_complete_fixture_has_9_performance_27_scene_and_60_old_rows(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )
    assert result["verdict"] == "ADVANCE_TO_TARGET125_CANDIDATE"
    assert len(result["paired_rows"]) == 10
    assert len(result["scenario_rows"]) == 30
    assert len([row for row in result["scenario_rows"] if row["outer_role"] == "performance"]) == 27
    assert len(result["per_old_class_rows"]) == 60
    assert len(result["liveness_rows"]) == 1
    # CCOC improves after registration in this real fixture, so forgetting is
    # negative and must remain a finite lower-is-better stability metric.
    assert all(row["candidate_average_forgetting"] < 0.0 for row in result["scenario_rows"])


def test_single_scene_hard_peak_is_rejected_even_when_other_scenes_pass(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    job = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = job["jobs"][0]["outer_key"]
    fit = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[1]["after_registration_resource"]["registration_incremental_peak_working_set_bytes"] = 1_048_577
    _write_json(fit, rows)
    _refresh_commit(fit.parent)
    _refresh_closure_evidence(fit.parent.parent.parent)

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["resource_hard"] is False


def test_baseline_order_is_joined_by_outer_key_not_position(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    source = Path(analysis.matrix.HISTORICAL_BASELINE_PATH)
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    rows[0], rows[-1] = rows[-1], rows[0]
    swapped = tmp_path / "paired_rows_swapped.csv"
    with swapped.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, baseline_paired_rows_path=swapped, truth_sidecar_root=truth_root)
    assert result["verdict"] == "ADVANCE_TO_TARGET125_CANDIDATE"


def test_k1_cannot_enter_performance_aggregation(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload["jobs"][-1]["outer_role"] = "performance"
    _write_json(Path(manifest), payload)
    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_duplicate_or_missing_scene_is_rejected_before_metric_aggregation(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[1]["scenario"] = rows[0]["scenario"]
    _write_json(fit, rows)
    _refresh_commit(state_root)
    _refresh_closure_evidence(state_root.parent.parent)

    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_performance_fallback_receipt_is_rejected(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["d92_e0d_ccoc_fallback_active"] = True
    _write_json(fit, rows)
    _refresh_commit(state_root)
    _refresh_closure_evidence(state_root.parent.parent)

    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_commit_and_truth_hash_drift_are_rejected_before_verdict(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    fit = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_wall_time_ns"] = 1
    _write_json(fit, rows)

    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )

    _refresh_commit(fit.parent)
    _refresh_closure_evidence(fit.parent.parent.parent)
    truth = truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"
    truth.write_text("drifted truth", encoding="utf-8")
    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_single_scene_wall_and_query_state_hard_failures_are_not_soft_revision(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_wall_time_ns"] = 150_000_001
    rows[1]["after_state_bytes"] += 1
    _write_json(fit, rows)
    _refresh_commit(state_root)
    _refresh_closure_evidence(state_root.parent.parent)
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["resource_hard"] is False


def test_truth_and_score_binding_drift_is_rejected(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload["jobs"][0]["truth_sidecar_sha256"] = "0" * 64
    _write_json(Path(manifest), payload)
    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_any_metric_tie_is_rejected_even_when_other_metrics_improve(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    _rewrite_candidate_after_as_e0(
        run_root / "jobs" / outer / analysis.ARM_ID,
        truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json",
        outer,
    )
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["all_strict_pareto"] is False


@pytest.mark.parametrize("metric", analysis.EIGHT_PARETO_METRICS)
def test_each_eight_metric_tie_rejects_the_route(tmp_path: Path, metric: str) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    _rewrite_candidate_after_as_e0(
        run_root / "jobs" / outer / analysis.ARM_ID,
        truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json",
        outer,
    )

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["all_strict_pareto"] is False
    row = next(item for item in result["paired_rows"] if item["outer_key"] == outer)
    assert row[f"candidate_{metric}"] == row[f"e0_{metric}"]


def test_paired_e0_row_is_bound_to_its_same_outer_raw_score(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    source = Path(analysis.matrix.HISTORICAL_BASELINE_PATH)
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        if row["outer_key"] == outer:
            row["candidate_h_old_new"] = str(float(row["candidate_h_old_new"]) + 0.1)
            break
    else:  # pragma: no cover - frozen evidence must contain every selected outer
        raise AssertionError("selected outer missing from frozen paired evidence")
    paired = tmp_path / "paired_rows_drift.csv"
    with paired.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            baseline_paired_rows_path=paired,
            truth_sidecar_root=truth_root,
        )
    )


def test_score_binding_evidence_is_required_for_truth_last_closure(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    receipt_path = run_root / "jobs" / outer / analysis.ARM_ID / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for field in ("score_binding", "score_binding_sha256", "score_evidence"):
        receipt.pop(field)
    _write_json(receipt_path, receipt)

    _assert_memory_reject(
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )
    )


def test_729088_peak_is_target_only_failure_and_revises_once(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_incremental_peak_working_set_bytes"] = 729_088
    _write_json(fit, rows)
    _refresh_commit(state_root)
    _refresh_closure_evidence(state_root.parent.parent)

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REVISE_ONCE"
    assert result["gate_state"]["resource_hard"] is True
    assert result["gate_state"]["resource_target"] is False


def test_markdown_reports_observed_reg0_old_metrics_and_na_reg0_new_metrics(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    markdown = analysis.render_analysis_markdown(result)

    assert "| DA0_REG0 | N/A | N/A | N/A | N/A |" in markdown
    assert (
        "| DA1_REG0 | "
        f"{result['aggregate']['candidate_mean_da1_reg0_old_acc']} | "
        f"{result['aggregate']['candidate_mean_da1_reg0_old_floor']} | N/A | N/A |"
    ) in markdown
    assert "| DA0_REG1 | N/A | N/A | N/A | N/A |" in markdown
    assert "| DA1_REG1 |" in markdown


_EXCLUSIVE_OUTPUTS = {
    "summary.json",
    "gates.json",
    "paired_rows.csv",
    "per_old_class_rows.csv",
    "scenario_rows.csv",
    "liveness_rows.csv",
    "analysis.md",
}


def _assert_controlled_reject(result: dict[str, object], output_root: Path) -> None:
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["complete_artifact_closure"] is False
    assert {path.name for path in output_root.iterdir()} == _EXCLUSIVE_OUTPUTS
    assert json.loads((output_root / "summary.json").read_text(encoding="utf-8"))["verdict"] == "REJECT_ROUTE"


def _assert_memory_reject(result: dict[str, object]) -> None:
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["status"] == "REJECTED_EVIDENCE_CLOSURE"
    assert result["gate_state"]["complete_artifact_closure"] is False
    assert "output_paths" not in result


def test_compact_manifest_cannot_use_a_production_fallback() -> None:
    compact = {
        "schema": matrix.MATRIX_SCHEMA,
        "jobs": [],
        "job_count": 10,
        "outer_count": 10,
        "performance_outer_count": 9,
        "liveness_outer_count": 1,
        "scene_count": 3,
        "scene_arm_count": 30,
        "selection_sha256": matrix.CANONICAL_SELECTION_SHA256,
        "method_lock_sha256": "a" * 64,
    }
    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError):
        analysis._validate_manifest_shape(compact, "a" * 64)


def test_resource_target_uses_frozen_p90_for_wall_and_ratio() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "outer_key": f"outer-{index}",
                "scenario": "leo_clear_weak",
                "candidate_wall_ns": 100_000_000 if index < 9 else 125_000_000,
                "wall_ratio": 1.0 if index < 9 else 1.30,
                "candidate_peak_bytes": 500_000,
                "wall_hard_pass": True,
                "ratio_hard_pass": True,
                "peak_hard_pass": True,
                "wall_target_pass": index < 9,
                "ratio_target_pass": index < 9,
                "peak_target_pass": True,
                "query_state_exact": True,
            }
        )
    result = analysis.evaluate_resource_gate(rows)
    assert result["wall_p90_ns"] == 100_000_000
    assert result["wall_ratio_p90"] == 1.0
    assert result["target_passed"] is True


def test_missing_reg0_old_floor_is_evidence_failure_not_reg1_substitution() -> None:
    score = _score()
    score.pop("per_old_class_floor_before")
    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="DA1_REG0 old floor"):
        analysis.compute_score_metrics(score)


@pytest.mark.parametrize(
    ("fixture_kwargs", "name"),
    [
        ({"raw_truth_mismatch": True}, "truth"),
        ({"raw_query_mismatch": True}, "query"),
    ],
)
def test_candidate_and_same_outer_e0_must_share_truth_and_query_surface(
    tmp_path: Path,
    fixture_kwargs: dict[str, bool],
    name: str,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path, **fixture_kwargs)
    output_root = tmp_path / f"reject-{name}"
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
        output_root=output_root,
    )
    _assert_controlled_reject(result, output_root)


def test_real_e0_sibling_prediction_shape_is_rejected_without_synthetic_score_fields(
    tmp_path: Path,
) -> None:
    manifest, _, _, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    raw_score_path = (
        Path(analysis.matrix.RAW_SCORE_ROOT)
        / outer
        / "E0_FULL_ONLY"
        / "scorer"
        / "diag_cosine_score.json"
    )
    raw_score = json.loads(raw_score_path.read_text(encoding="utf-8"))
    raw_before = raw_score_path.parent.parent / "diag" / "before" / "prediction_artifact.npz"
    os.chmod(raw_before, stat.S_IREAD | stat.S_IWRITE)
    np.savez(raw_before, query_tokens=np.asarray(["bad"], dtype=str))
    os.chmod(raw_before, stat.S_IREAD)
    raw_score["before_prediction_sha256"] = _digest(raw_before)
    _write_json(raw_score_path, raw_score)
    paths = analysis._raw_e0_prediction_paths(raw_score_path, raw_score, outer_key=outer)
    truth = analysis._read_truth_surface(
        truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"
    )
    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="exact schema"):
        analysis._surface_pair(
            paths["before"],
            paths["after"],
            truth,
            label=f"E0 {outer}",
        )


def test_missing_truth_root_is_controlled_in_memory_and_as_seven_outputs(tmp_path: Path) -> None:
    manifest, lock, run_root, _ = _integration_fixture(tmp_path)
    missing_root = tmp_path / "missing-truth-root"
    memory = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=missing_root,
    )
    _assert_memory_reject(memory)
    output_root = tmp_path / "reject-missing-truth-root"
    materialized = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=missing_root,
        output_root=output_root,
    )
    _assert_controlled_reject(materialized, output_root)


def test_fit_and_commit_rewrite_without_task1_closure_rebind_is_controlled_reject(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit_path = state_root / "fit_audit.json"
    fit_rows = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_rows[0]["audit_nonce"] = "rewritten-after-bind"
    _write_json(fit_path, fit_rows)
    _refresh_commit(state_root)
    output_root = tmp_path / "reject-fit-commit-drift"
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
        output_root=output_root,
    )
    _assert_controlled_reject(result, output_root)


def test_missing_score_evidence_and_fallback_produce_controlled_seven_file_rejects(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path / "missing")
    outer = json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    receipt_path = run_root / "jobs" / outer / analysis.ARM_ID / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("score_evidence")
    _write_json(receipt_path, receipt)
    missing_root = tmp_path / "reject-missing-evidence"
    missing = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
        output_root=missing_root,
    )
    _assert_controlled_reject(missing, missing_root)

    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path / "fallback")
    outer = json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    job_root = run_root / "jobs" / outer / analysis.ARM_ID
    fit_path = job_root / "diag" / "after" / "fit_audit.json"
    fit_rows = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_rows[0]["d92_e0d_ccoc_fallback_active"] = True
    _write_json(fit_path, fit_rows)
    _refresh_commit(fit_path.parent)
    _refresh_closure_evidence(job_root)
    fallback_root = tmp_path / "reject-fallback"
    fallback = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
        output_root=fallback_root,
    )
    _assert_controlled_reject(fallback, fallback_root)


def test_receiver_knew_and_scene_stability_requires_all_eight_metric_directions(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    jobs = json.loads(manifest.read_text(encoding="utf-8"))["jobs"]
    scene = analysis.SCENES[-1]
    job = next(item for item in jobs if item["outer_role"] == "performance")
    job_root = run_root / "jobs" / job["outer_key"] / analysis.ARM_ID
    _rewrite_candidate_after_scene_for_stability(
        job_root,
        truth_root / "jobs" / job["outer_key"] / "offline" / "scorer" / "truth_sidecar.json",
        scene,
    )

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )
    assert result["gate_state"]["all_strict_pareto"] is True
    assert result["gate_state"]["resource_hard"] is True
    assert result["gate_state"]["stability"] is False
    assert result["verdict"] == "REVISE_ONCE"


def test_fit_resource_join_remains_explicitly_scene_keyed_after_fit_order_changes(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    job_root = run_root / "jobs" / outer / analysis.ARM_ID
    fit_path = job_root / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    _write_json(fit_path, list(reversed(rows)))
    _refresh_commit(fit_path.parent)
    _refresh_closure_evidence(job_root)
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )
    assert result["verdict"] == "ADVANCE_TO_TARGET125_CANDIDATE"
