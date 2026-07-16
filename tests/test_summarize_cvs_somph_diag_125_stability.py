from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.leo_weak_cache import post_channel_iq_sha256
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair
from scripts import summarize_cvs_somph_diag_125_stability as summary


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _json_readonly(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    _readonly(path)


def _prediction(path: Path, *, after: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = ["old-a", "old-b"] + (["new-c"] if after else [])
    predicted = []
    scenarios = []
    query_tokens = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        values = ["h-old-a", "h-old-b"] + (["h-new-c"] if after else [])
        if after and scenario == "leo_rain_weak":
            values[1] = "h-old-a"
        predicted.extend(values)
        scenarios.extend([scenario] * len(tokens))
        query_tokens.extend(tokens)
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_tokens=np.asarray(query_tokens),
            scenarios=np.asarray(scenarios),
            predicted_class_handles=np.asarray(predicted),
        )
    _readonly(path)


def _sealed_query_package(path: Path, *, after: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = ["old-a", "old-b"] + (["new-c"] if after else [])
    iq = np.arange(len(tokens) * 2 * 8, dtype=np.float32).reshape(
        len(tokens), 2, 8
    )
    hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=iq,
            query_tokens=np.asarray(tokens),
            query_post_channel_iq_sha256=hashes,
            query_satellite_seeds=np.arange(len(tokens), dtype=np.int64),
        )
    return summary._sha256(path)


def _one_job_fixture(tmp_path: Path) -> tuple[Path, dict]:
    matrix_root = tmp_path / "matrix"
    job = {
        "job_id": "rx_20_1__seed_713102__k_10__new_5",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 10,
        "new_class_count": 5,
    }
    root = matrix_root / "jobs" / job["job_id"]
    before = root / "diag" / "before" / "prediction_artifact.npz"
    after = root / "diag" / "after" / "prediction_artifact.npz"
    truth = root / "offline" / "scorer" / "truth_sidecar.json"
    score_path = root / "scorer" / "diag_cosine_score.json"
    _prediction(before, after=False)
    _prediction(after, after=True)
    query_shas: dict[str, dict[str, str]] = {}
    for state, is_after in (("before", False), ("after", True)):
        query_shas[state] = {}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            query_shas[state][scenario] = _sealed_query_package(
                root
                / "offline"
                / "predictor"
                / state
                / "apply_only_staging"
                / f"query_{scenario}.npz",
                after=is_after,
            )
    _json_readonly(
        truth,
        {
            "schema": "cvs.phase2.query_truth_sidecar.v2",
            "stage": "stage2c",
            "receiver": "20-1",
            "seed": 713102,
            "rows": [
                {
                    "query_token": "old-a",
                    "true_class_handle": "h-old-a",
                    "transmitter_label": "old-a",
                    "evaluation_role": "target_old",
                    "physical_sample_id": "target_old|old-a|sample-0",
                },
                {
                    "query_token": "old-b",
                    "true_class_handle": "h-old-b",
                    "transmitter_label": "old-b",
                    "evaluation_role": "target_old",
                    "physical_sample_id": "target_old|old-b|sample-0",
                },
                {
                    "query_token": "new-c",
                    "true_class_handle": "h-new-c",
                    "transmitter_label": "new-c",
                    "evaluation_role": "target_new",
                    "physical_sample_id": "target_new|new-c|sample-0",
                },
            ],
        },
    )
    scored = score_diag_cosine_pair(
        before_prediction_path=before,
        after_prediction_path=after,
        truth_sidecar_path=truth,
        output_path=score_path,
        candidate=summary.CANDIDATE,
    )
    row_manifest = {
        "schema": "cvs.phase2.somph_row_manifest.v1",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 10,
        "new_class_count": 5,
    }
    row_manifest_path = root / "offline" / "scorer" / "row_manifest.json"
    _json_readonly(row_manifest_path, row_manifest)
    pair_path = root / "scorer" / "registration_pair.final.json"
    _json_readonly(
        pair_path,
        {
            "schema": "cvs.phase2.somph_registration_pair.v1",
            "old_support_physical_ids_sha256_before": "a" * 64,
            "old_support_physical_ids_sha256_after": "a" * 64,
            "old_query_physical_ids_sha256_before": "b" * 64,
            "old_query_physical_ids_sha256_after": "b" * 64,
        },
    )
    pipeline_path = root / "pipeline_receipt.json"
    pipeline_states = {}
    for state, prediction_path in (("before", before), ("after", after)):
        apply_root_sha = ("c" if state == "before" else "d") * 64
        apply_seal_sha = ("e" if state == "before" else "f") * 64
        receipt_path = root / "diag" / state / "execution_receipt.json"
        opened_members = [
            {
                "relative_path": f"query_{scenario}.npz",
                "sha256": query_shas[state][scenario],
                "size_bytes": (
                    root
                    / "offline"
                    / "predictor"
                    / state
                    / "apply_only_staging"
                    / f"query_{scenario}.npz"
                ).stat().st_size,
                "status": "PASS",
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ]
        _json_readonly(
            receipt_path,
            {
                "schema": "cvs.phase2.diag_cosine_exploration_receipt.v1",
                "apply_package_root_sha256": apply_root_sha,
                "apply_package_seal_sha256": apply_seal_sha,
                "preopen_audit": {
                    "apply": {
                        "schema": "cvs.phase2.somph_preopen_audit.v1",
                        "profile": "apply_only",
                        "status": "STRUCTURAL_SELF_CONSISTENCY_PASS",
                        "hash_and_member_audit_same_file_descriptor": True,
                        "iq_payload_materialized": True,
                        "package_root_sha256": apply_root_sha,
                        "artifact_member_allowlist_sha256": apply_root_sha,
                        "manifest_sha256": "9" * 64,
                        "materialized_scenarios": list(
                            FORMAL_LEO_WEAK_SCENARIOS
                        ),
                        "opened_members": opened_members,
                    }
                },
            },
        )
        commit_path = root / "diag" / state / "COMMIT.json"
        _json_readonly(
            commit_path,
            {
                "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                "execution_receipt_sha256": summary._sha256(receipt_path),
                "prediction_artifact_sha256": summary._sha256(prediction_path),
                "members": [
                    {
                        "relative_path": "execution_receipt.json",
                        "sha256": summary._sha256(receipt_path),
                        "size_bytes": receipt_path.stat().st_size,
                    }
                ],
            },
        )
        pipeline_states[state] = {
            "prediction_artifact_sha256": summary._sha256(prediction_path),
            "apply_package_root_sha256": apply_root_sha,
            "apply_package_seal_sha256": apply_seal_sha,
            "diag_commit_sha256": summary._sha256(commit_path),
            "execution_receipt_sha256": summary._sha256(receipt_path),
        }
    _json_readonly(
        pipeline_path,
        {
            "schema": summary.PIPELINE_SCHEMA,
            "status": "DEVELOPMENT_ROW_COMPLETE",
            "formal_launch_authority": False,
            "receiver": "20-1",
            "seed": 713102,
            "k_shot": 10,
            "new_class_count": 5,
            "candidate": summary.CANDIDATE,
            "row_manifest_sha256": summary._canonical_sha256(row_manifest),
            "registration_pair_final_sha256": summary._sha256(pair_path),
            "score_artifact_sha256": scored["score_artifact_sha256"],
            "states": pipeline_states,
        },
    )
    return matrix_root, job


def test_audit_job_binds_score_and_derives_scenario_old_floor(tmp_path: Path) -> None:
    matrix_root, job = _one_job_fixture(tmp_path)
    writable_query = (
        matrix_root
        / "jobs"
        / job["job_id"]
        / "offline"
        / "predictor"
        / "after"
        / "apply_only_staging"
        / "query_leo_clear_weak.npz"
    )
    assert writable_query.stat().st_mode & stat.S_IWRITE
    row, scenarios, per_tx, _audit = summary._audit_job(matrix_root, job)
    assert row["b_old_acc"] == 1.0
    assert row["c_old_acc"] == pytest.approx(5.0 / 6.0)
    rain = next(item for item in scenarios if item["scenario"] == "leo_rain_weak")
    assert rain["c_old_acc"] == 0.5
    assert rain["c_old_floor"] == 0.0
    assert rain["old_adaptation_gain"] == -0.5
    assert any(
        item["state"] == "after"
        and item["scenario"] == "leo_rain_weak"
        and item["tx"] == "old-b"
        and item["accuracy"] == 0.0
        for item in per_tx
    )


def test_post_channel_sha_helpers_verify_rank0_and_query_physical_mapping(
    tmp_path: Path,
) -> None:
    support_path = tmp_path / "support.npz"
    support_iq = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    support_hashes = np.asarray(
        [post_channel_iq_sha256(row) for row in support_iq]
    )
    with support_path.open("xb") as handle:
        np.savez(
            handle,
            support_leo_weak_iq=support_iq,
            support_class_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
            support_rank_within_class=np.asarray([0, 1, 0, 1], dtype=np.int64),
            support_tokens=np.asarray(["s0", "s1", "s2", "s3"]),
            support_post_channel_iq_sha256=support_hashes,
            support_satellite_seeds=np.asarray([1, 2, 3, 4], dtype=np.int64),
        )
    _readonly(support_path)
    rank0 = summary._support_rank0(support_path, expected_class_count=2)
    assert [(row[0], row[1]) for row in rank0] == [(0, 0), (1, 0)]

    query_path = tmp_path / "query.npz"
    query_iq = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    query_hashes = np.asarray([post_channel_iq_sha256(row) for row in query_iq])
    with query_path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=query_iq,
            query_tokens=np.asarray(["q0", "q1"]),
            query_post_channel_iq_sha256=query_hashes,
            query_satellite_seeds=np.asarray([5, 6], dtype=np.int64),
        )
    _readonly(query_path)
    mapped = summary._query_physical_iq(
        query_path,
        {
            "q0": {"physical_sample_id": "p0"},
            "q1": {"physical_sample_id": "p1"},
        },
        expected_sha256=summary._sha256(query_path),
    )
    assert mapped == sorted(
        [("p0", query_hashes[0], 5), ("p1", query_hashes[1], 6)]
    )


def _locked_manifest(path: Path) -> dict:
    jobs = []
    for receiver in summary.EXPECTED_RECEIVERS:
        for seed in summary.EXPECTED_SEEDS:
            for k_shot, new_count in summary.EXPECTED_SLICES:
                jobs.append(
                    {
                        "job_id": (
                            f"rx_{receiver.replace('-', '_')}__seed_{seed}"
                            f"__k_{k_shot}__new_{new_count}"
                        ),
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                    }
                )
    payload = {
        "schema": summary.MANIFEST_SCHEMA,
        "claim_scope": "development_only_not_formal_confirmation",
        "formal_launch_authority": False,
        "candidate": summary.CANDIDATE,
        "job_count": 125,
        "row_pair_count": 125,
        "scenario_pair_count": 375,
        "scenario_state_metric_count": 750,
        "receivers": list(summary.EXPECTED_RECEIVERS),
        "confirmation_seeds": list(summary.EXPECTED_SEEDS),
        "jobs": jobs,
    }
    _json_readonly(path, payload)
    return payload


def test_full_summary_writes_required_outputs_and_keeps_direct_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "matrix" / "matrix_manifest.json"
    _locked_manifest(manifest_path)

    def fake_audit(matrix_root: Path, job: dict):
        del matrix_root
        k_shot = int(job["k_shot"])
        new_count = int(job["new_class_count"])
        if k_shot == 1:
            before, after, floor, seen = 0.90, 0.91, 0.86, 0.80
        elif k_shot == 5:
            before, after, floor, seen = 0.93, 0.92, 0.89, 0.85
        else:
            before, after, floor = 0.95, 0.94, 0.90
            seen = {5: 0.93, 10: 0.91, 20: 0.87}[new_count]
        harmonic = 2.0 * after * seen / (after + seen)
        row = {
            "job_id": job["job_id"],
            "receiver": job["receiver"],
            "seed": job["seed"],
            "k_shot": k_shot,
            "new_class_count": new_count,
            "candidate": summary.CANDIDATE,
            "b_old_acc": before,
            "c_old_acc": after,
            "b_old_floor": floor,
            "c_old_floor": floor,
            "seen_new_acc": seen,
            "h_old_new": harmonic,
            "average_forgetting": before - after,
            "old_adaptation_gain": after - before,
            "direct_adv3b02_status": summary.DIRECT_STATUS,
            "direct_adv3b02_old_acc": None,
            "delta_vs_direct_ADV3B02_K1": None,
            "pipeline_receipt_sha256": "a" * 64,
            "score_sha256": "b" * 64,
        }
        scenarios = [
            {
                key: value
                for key, value in {
                    **row,
                    "scenario": scenario,
                }.items()
                if key
                not in {"candidate", "pipeline_receipt_sha256", "score_sha256"}
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ]
        tx_rows = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            for state, accuracy in (("before", floor), ("after", floor)):
                for tx in ("old-a", "old-b"):
                    tx_rows.append(
                        {
                            "state": state,
                            "scenario": scenario,
                            "role": "target_old",
                            "tx": tx,
                            "count": 20,
                            "accuracy": accuracy,
                            "job_id": job["job_id"],
                            "receiver": job["receiver"],
                            "seed": job["seed"],
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                        }
                    )
        return row, scenarios, tx_rows, {
            "truth": {"q": {"physical_sample_id": "p"}},
            "root": Path("/unused"),
            "query_sha_by_state": {
                state: {
                    scenario: "b" * 64
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                }
                for state in ("before", "after")
            },
        }

    monkeypatch.setattr(summary, "_audit_job", fake_audit)
    monkeypatch.setattr(
        summary,
        "_support_rank0",
        lambda _path, *, expected_class_count: [
            (index, 0, "a" * 64, 1)
            for index in range(expected_class_count)
        ],
    )
    monkeypatch.setattr(
        summary,
        "_query_physical_iq",
        lambda _path, _truth, *, expected_sha256: [("p", expected_sha256, 2)],
    )
    output = tmp_path / "summary"
    result = summary.summarize(manifest_path, output)
    assert result["job_count"] == 125
    assert result["scenario_pair_count"] == 375
    assert result["support_query_nesting_audit_count"] == 75
    assert result["direct_adv3b02_status"] == "MISSING_NOT_RUN"
    assert result["gates"]["k1_old_adaptation_gain"]["pass"] is True
    assert result["gates"]["direct_ADV3B02_K1"]["pass"] is None
    assert result["gates"]["overall_status"] == (
        "INCOMPLETE_DIRECT_ADV3B02_NOT_RUN_PERFORMANCE_PASS"
    )
    assert result["status"] == "INCOMPLETE_DIRECT_BASELINE_PERFORMANCE_PASS"
    assert result["query_package_commit_binding_status"] == (
        "PIPELINE_COMMIT_RECEIPT_QUERY_SHA_BOUND"
    )
    for name in (
        "summary.json",
        "row_metrics.csv",
        "scenario_metrics.csv",
        "receiver_metrics.csv",
        "per_tx_metrics.csv",
        "gates.json",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        summary.summarize(manifest_path, output)


def test_manifest_rejects_duplicate_cartesian_key_before_job_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "matrix" / "matrix_manifest.json"
    payload = _locked_manifest(manifest_path)
    os.chmod(manifest_path, stat.S_IWRITE | stat.S_IREAD)
    payload["jobs"][-1] = dict(payload["jobs"][0])
    _json_readonly(manifest_path, payload)
    monkeypatch.setattr(
        summary,
        "_audit_job",
        lambda *_args, **_kwargs: pytest.fail("job audit must not start"),
    )
    with pytest.raises(
        summary.StabilitySummaryError, match="Cartesian key/job_id uniqueness"
    ):
        summary.summarize(manifest_path, tmp_path / "out")


def test_support_rank0_rejects_missing_class_and_query_rejects_truth_gap(
    tmp_path: Path,
) -> None:
    support_path = tmp_path / "support_bad.npz"
    iq = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
    with support_path.open("xb") as handle:
        np.savez(
            handle,
            support_leo_weak_iq=iq,
            support_class_indices=np.asarray([0, 0], dtype=np.int64),
            support_rank_within_class=np.asarray([0, 1], dtype=np.int64),
            support_tokens=np.asarray(["s0", "s1"]),
            support_post_channel_iq_sha256=hashes,
            support_satellite_seeds=np.asarray([1, 2], dtype=np.int64),
        )
    _readonly(support_path)
    with pytest.raises(summary.StabilitySummaryError, match="one rank0 per class"):
        summary._support_rank0(support_path, expected_class_count=2)

    query_path = tmp_path / "query_gap.npz"
    query_iq = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    query_hashes = np.asarray(
        [post_channel_iq_sha256(row) for row in query_iq]
    )
    with query_path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=query_iq,
            query_tokens=np.asarray(["q0", "q1"]),
            query_post_channel_iq_sha256=query_hashes,
            query_satellite_seeds=np.asarray([1, 2], dtype=np.int64),
        )
    _readonly(query_path)
    with pytest.raises(summary.StabilitySummaryError, match="does not equal truth"):
        summary._query_physical_iq(
            query_path,
            {
                "q0": {"physical_sample_id": "p0"},
                "q1": {"physical_sample_id": "p1"},
                "q2": {"physical_sample_id": "p2"},
            },
            expected_sha256=summary._sha256(query_path),
        )


def test_strict_audit_rejects_mutated_writable_staging_query(
    tmp_path: Path,
) -> None:
    matrix_root, job = _one_job_fixture(tmp_path)
    query_path = (
        matrix_root
        / "jobs"
        / job["job_id"]
        / "offline"
        / "predictor"
        / "after"
        / "apply_only_staging"
        / "query_leo_clear_weak.npz"
    )
    os.chmod(query_path, stat.S_IWRITE | stat.S_IREAD)
    query_path.unlink()
    iq = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
    with query_path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=iq,
            query_tokens=np.asarray(["old-a", "old-b", "new-c", "extra"]),
            query_post_channel_iq_sha256=hashes,
            query_satellite_seeds=np.arange(4, dtype=np.int64),
        )
    _readonly(query_path)
    with pytest.raises(summary.StabilitySummaryError, match="query SHA mismatch"):
        summary._audit_job(matrix_root, job)


def test_receipt_bound_writable_query_accepts_exact_sha_and_rejects_wrong_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query.npz"
    expected = _sealed_query_package(path, after=True)
    assert path.stat().st_mode & stat.S_IWRITE
    archive = summary._read_receipt_bound_npz(path, expected_sha256=expected)
    assert archive["query_tokens"].astype(str).tolist() == [
        "old-a",
        "old-b",
        "new-c",
    ]
    with pytest.raises(summary.StabilitySummaryError, match="query SHA mismatch"):
        summary._read_receipt_bound_npz(path, expected_sha256="0" * 64)


def test_query_binding_requires_pipeline_diag_commit_sha(
    tmp_path: Path,
) -> None:
    matrix_root, job = _one_job_fixture(tmp_path)
    root = matrix_root / "jobs" / job["job_id"]
    pipeline = json.loads(
        (root / "pipeline_receipt.json").read_text(encoding="utf-8")
    )
    digest = summary._query_member_sha_from_receipt(
        root=root,
        state="after",
        scenario="leo_clear_weak",
        pipeline=pipeline,
    )
    assert len(digest) == 64
    pipeline["states"]["after"].pop("diag_commit_sha256")
    with pytest.raises(
        summary.StabilitySummaryError, match="execution receipt/COMMIT binding"
    ):
        summary._query_member_sha_from_receipt(
            root=root,
            state="after",
            scenario="leo_clear_weak",
            pipeline=pipeline,
        )


def test_audit_job_rejects_legacy_pipeline_v1_without_strict_bindings(
    tmp_path: Path,
) -> None:
    matrix_root, job = _one_job_fixture(tmp_path)
    pipeline_path = (
        matrix_root / "jobs" / job["job_id"] / "pipeline_receipt.json"
    )
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["schema"] = "cvs.phase2.somph_diag_row_pipeline.v1"
    for state in ("before", "after"):
        pipeline["states"][state].pop("diag_commit_sha256")
        pipeline["states"][state].pop("execution_receipt_sha256")
    os.chmod(pipeline_path, stat.S_IWRITE | stat.S_IREAD)
    pipeline_path.unlink()
    _json_readonly(pipeline_path, pipeline)
    with pytest.raises(
        summary.StabilitySummaryError, match="pipeline/job binding drift"
    ):
        summary._audit_job(matrix_root, job)


def test_readonly_truth_physical_query_ids_are_exact_and_unique() -> None:
    assert summary._truth_physical_query_ids(
        {
            "q0": {"physical_sample_id": "p1"},
            "q1": {"physical_sample_id": "p0"},
        }
    ) == ["p0", "p1"]
    with pytest.raises(summary.StabilitySummaryError, match="duplicated"):
        summary._truth_physical_query_ids(
            {
                "q0": {"physical_sample_id": "p0"},
                "q1": {"physical_sample_id": "p0"},
            }
        )


def test_performance_failure_is_not_reported_as_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "matrix" / "matrix_manifest.json"
    _locked_manifest(manifest_path)

    def fake_audit(_matrix_root: Path, job: dict):
        k_shot = int(job["k_shot"])
        new_count = int(job["new_class_count"])
        after = 0.80 if k_shot == 10 else 0.79
        before = after
        seen = 0.70
        harmonic = 2.0 * after * seen / (after + seen)
        base = {
            "job_id": job["job_id"],
            "receiver": job["receiver"],
            "seed": job["seed"],
            "k_shot": k_shot,
            "new_class_count": new_count,
            "b_old_acc": before,
            "c_old_acc": after,
            "b_old_floor": 0.70,
            "c_old_floor": 0.70,
            "seen_new_acc": seen,
            "h_old_new": harmonic,
            "average_forgetting": 0.0,
            "old_adaptation_gain": 0.0,
            "direct_adv3b02_status": summary.DIRECT_STATUS,
            "direct_adv3b02_old_acc": None,
            "delta_vs_direct_ADV3B02_K1": None,
        }
        row = {
            **base,
            "candidate": summary.CANDIDATE,
            "pipeline_receipt_sha256": "a" * 64,
            "score_sha256": "b" * 64,
        }
        scenarios = [
            {**base, "scenario": scenario}
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ]
        tx_rows = [
            {
                "state": state,
                "scenario": scenario,
                "role": "target_old",
                "tx": "old-a",
                "count": 20,
                "accuracy": 0.70,
                "job_id": job["job_id"],
                "receiver": job["receiver"],
                "seed": job["seed"],
                "k_shot": k_shot,
                "new_class_count": new_count,
            }
            for state in ("before", "after")
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ]
        return row, scenarios, tx_rows, {
            "truth": {"q": {"physical_sample_id": "p"}},
            "root": Path("/unused"),
            "query_sha_by_state": {
                state: {
                    scenario: "b" * 64
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                }
                for state in ("before", "after")
            },
        }

    monkeypatch.setattr(summary, "_audit_job", fake_audit)
    monkeypatch.setattr(
        summary,
        "_support_rank0",
        lambda _path, *, expected_class_count: [
            (index, 0, "a" * 64, 1)
            for index in range(expected_class_count)
        ],
    )
    monkeypatch.setattr(
        summary,
        "_query_physical_iq",
        lambda _path, _truth, *, expected_sha256: [("p", expected_sha256, 2)],
    )
    result = summary.summarize(manifest_path, tmp_path / "out")
    assert result["gates"]["executed_performance_status"] == "FAIL"
    assert result["gates"]["overall_status"].endswith("PERFORMANCE_FAIL")
    assert result["status"] == "INCOMPLETE_DIRECT_BASELINE_PERFORMANCE_FAIL"
