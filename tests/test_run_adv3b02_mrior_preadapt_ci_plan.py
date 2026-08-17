from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper_reproduction.scripts import run_adv3b02_mrior_preadapt_ci_plan as runner


def test_runner_bootstraps_code_root_from_independent_cwd(tmp_path: Path) -> None:
    script = Path(runner.__file__).resolve()
    probe = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='mrior_runner_import_probe'); "
        "import cvsrffi; "
        "print(cvsrffi.__file__)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cvsrffi" in completed.stdout


def test_preadapt_source_loader_accepts_only_the_locked_legacy_v1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen Phase1 v1 source cache stays usable without relaxing other schemas."""

    manifest_path = tmp_path / "cache_set.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_v1",
                "cache_scope": "source_train",
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    def fake_legacy_loader(path, *, expected_scope, allowed_roles):
        observed.update(
            path=Path(path),
            expected_scope=expected_scope,
            allowed_roles=set(allowed_roles),
        )
        return {"leo_clear_weak": {"dataset_role": ["source"]}}, {
            "schema": "cvs_leo_weak_iq_cache_set_v1",
            "cache_scope": "source_train",
        }, {"status": "PASS_COMPARISON_SCOPE"}

    from paper_reproduction.scripts import build_adv3b02_paper_full_ci_bundle

    monkeypatch.setattr(
        build_adv3b02_paper_full_ci_bundle,
        "load_comparison_leo_cache_set",
        fake_legacy_loader,
    )
    arrays, loaded_manifest, audit = runner._load_preadapt_source_cache(manifest_path)

    assert arrays["leo_clear_weak"]["dataset_role"] == ["source"]
    assert loaded_manifest["schema"] == "cvs_leo_weak_iq_cache_set_v1"
    assert audit["legacy_source_cache_compatibility"] == "STRICT_V1"
    assert observed == {
        "path": manifest_path,
        "expected_scope": "source_train",
        "allowed_roles": {"source"},
    }

    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_unknown",
                "cache_scope": "source_train",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported source cache-set schema"):
        runner._load_preadapt_source_cache(manifest_path)


def test_total_capacity_is_derived_from_the_frozen_registry_when_v7_omits_it() -> None:
    package = {"old_class_labels": [f"old-{index}" for index in range(6)]}
    plan = {"new_class_counts": [2, 5, 10, 20]}

    assert runner._required_total_capacity({}, plan, package) == 26

    with pytest.raises(ValueError, match="total capacity drift"):
        runner._required_total_capacity(
            {"required_total_capacity": 31}, plan, package
        )


def _plan_contract_sha256(plan: dict) -> str:
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_authority", "authority_state", "plan_contract_sha256"}
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _formal_plan(tmp_path: Path) -> dict:
    receivers = ["20-1", "3-19", "7-14", "7-7", "8-8"]
    seeds = [713101, 713102, 713103, 713104, 713105]
    k_values = [1, 5, 10, 20]
    new_counts = [2, 5, 10, 20]
    scenarios = ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    method_map = {
        "csil_paper_full": "mrior_sda_then_csil_paper_full",
        "mopc_hr_paper_full": "mrior_sda_then_mopc_hr_paper_full",
    }
    run_root = tmp_path / "run"
    jobs = []
    job_by_key = {}
    for receiver in receivers:
        for seed in seeds:
            for new_count in new_counts:
                for k_shot in k_values:
                    for scenario in scenarios:
                        job_id = f"job-{receiver}-{seed}-{new_count}-{k_shot}-{scenario}"
                        job = {
                            "job_id": job_id,
                            "receiver": receiver,
                            "seed": seed,
                            "new_class_count": new_count,
                            "k_shot": k_shot,
                            "scenario": scenario,
                            "artifact_root": str(run_root / "preadapt_jobs" / job_id),
                            "input_binding_sha256": "a" * 64,
                            "method_lock_sha256": "b" * 64,
                        }
                        jobs.append(job)
                        job_by_key[(receiver, seed, new_count, k_shot, scenario)] = job_id
    cells = []
    for receiver in receivers:
        for seed in seeds:
            for k_shot in k_values:
                for new_count in new_counts:
                    for source_method, method in method_map.items():
                        cell_id = (
                            f"cell-{receiver}-{seed}-{new_count}-{method}-{k_shot}"
                        )
                        cells.append(
                            {
                                "cell_id": cell_id,
                                "baseline_v7_cell_id": f"v7-{cell_id}",
                                "source_v7_method": source_method,
                                "method": method,
                                "receiver": receiver,
                                "seed": seed,
                                "new_class_count": new_count,
                                "k_shot": k_shot,
                                "output_root": str(run_root / "cells" / cell_id),
                                "preadapt_job_ids_by_scenario": {
                                    scenario: job_by_key[
                                        (receiver, seed, new_count, k_shot, scenario)
                                    ]
                                    for scenario in scenarios
                                },
                            }
                        )
    plan = {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1",
        "experiment_id": "adv3b02_mrior_preadapt_ci_20260817_v1",
        "run_root": str(run_root),
        "methods": list(method_map.values()),
        "source_methods": list(method_map),
        "receivers": receivers,
        "seeds": seeds,
        "k_values": k_values,
        "new_class_counts": new_counts,
        "scenarios": scenarios,
        "preadapt_jobs": jobs,
        "cells": cells,
        "preadapt_scope": "receiver_seed_newcount_k_scene",
        "counts": {"preadapt_jobs": 1200, "cells": 800, "scenario_rows": 2400},
        "smoke_preadapt_job_ids": [
            job_by_key[(receivers[0], seeds[0], new_count, k_shot, scenario)]
            for new_count, k_shot in (
                (new_counts[0], k_values[0]),
                (new_counts[-1], k_values[-1]),
            )
            for scenario in scenarios
        ],
        "smoke_cell_ids": [
            cell["cell_id"]
            for cell in cells
            if cell["receiver"] == receivers[0]
            and cell["seed"] == seeds[0]
            and (
                (cell["new_class_count"] == new_counts[0] and cell["k_shot"] == k_values[0])
                or (
                    cell["new_class_count"] == new_counts[-1]
                    and cell["k_shot"] == k_values[-1]
                )
            )
        ],
        "launch_authority": False,
        "authority_state": "N607_MRIOR_PREADAPT_CI_SMOKE_REQUIRED",
    }
    plan["plan_contract_sha256"] = _plan_contract_sha256(plan)
    return plan


def test_matrix_dispatch_requires_passed_smoke_authority() -> None:
    """A full matrix cannot run until this immutable plan is smoke-authorized."""

    plan = {
        "launch_authority": False,
        "authority_state": "N607_MRIOR_PREADAPT_CI_SMOKE_REQUIRED",
    }

    with pytest.raises(ValueError, match="smoke authority"):
        runner._verify_smoke_authority(plan, project_root=Path.cwd())


def test_smoke_authority_accepts_only_declared_smoke_preadapt_jobs(tmp_path: Path) -> None:
    """The smoke receipt authorizes its six declared preadapt jobs, not full preadaptation."""

    plan = _formal_plan(tmp_path)
    smoke_job_ids = [job["job_id"] for job in plan["preadapt_jobs"][:6]]
    plan["smoke_preadapt_job_ids"] = smoke_job_ids
    run_root = Path(plan["run_root"])
    run_root.mkdir()
    (run_root / "smoke_receipt.json").write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_smoke_receipt.v1",
                "status": "PASS",
                "plan_contract_sha256": plan["plan_contract_sha256"],
                "completed_preadapt_job_ids": smoke_job_ids,
                "completed_cell_ids": plan["smoke_cell_ids"],
            }
        ),
        encoding="utf-8",
    )

    runner._verify_smoke_authority(plan, project_root=Path.cwd())


def test_runner_rejects_an_unowned_existing_run_root(tmp_path: Path) -> None:
    """A stale directory must never become an implicit destination to overwrite."""

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "foreign-artifact.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned existing run root"):
        runner._claim_run_root(
            {"run_root": str(run_root), "plan_contract_sha256": "a" * 64}
        )


def test_runner_requires_exact_1200_800_2400_matrix_closure(tmp_path: Path) -> None:
    """A formal run cannot silently release a partial or duplicate matrix."""

    plan = _formal_plan(tmp_path)
    runner._validate_plan_payload(plan)

    plan["counts"] = {"preadapt_jobs": 1200, "cells": 800, "scenario_rows": 2399}
    with pytest.raises(ValueError, match="2400"):
        runner._validate_plan_payload(plan)


def test_eight_deterministic_shards_partition_preadapt_jobs_once(tmp_path: Path) -> None:
    """All 1200 job identities are assigned once across exactly eight shards."""

    jobs = _formal_plan(tmp_path)["preadapt_jobs"]
    assigned = [
        job["job_id"]
        for shard_index in range(8)
        for job in runner._select_shard(jobs, shard_index=shard_index, shard_count=8)
    ]

    assert len(assigned) == 1200
    assert len(set(assigned)) == 1200


def test_two_matching_pre_prediction_failures_stop_with_no_performance_result(
    tmp_path: Path,
) -> None:
    """Only repeatable execution faults stop dispatch; metrics never enter this gate."""

    plan = {"run_root": str(tmp_path / "run")}
    first = runner._update_health_state(
        plan,
        row_id="cell-a",
        exc=RuntimeError("worker fault 100"),
        prediction_produced=False,
    )
    second = runner._update_health_state(
        plan,
        row_id="cell-b",
        exc=RuntimeError("worker fault 200"),
        prediction_produced=False,
    )

    assert first["stop_dispatch"] is False
    assert second["stop_dispatch"] is True
    assert second["result_state"] == "NO_PERFORMANCE_RESULT"


def test_closure_rejects_a_missing_prediction_or_scene_row(tmp_path: Path) -> None:
    """The final receipt is unavailable unless every artifact has one complete row set."""

    plan = _formal_plan(tmp_path)
    artifacts = [{"job_id": job["job_id"]} for job in plan["preadapt_jobs"]]
    cells = [
        {
            "cell_id": cell["cell_id"],
            "prediction": True,
            "score": True,
            "scenarios": list(plan["scenarios"]),
        }
        for cell in plan["cells"]
    ]
    cells.pop()

    with pytest.raises(ValueError, match="800"):
        runner._verify_matrix_closure(plan, artifacts=artifacts, cells=cells)
