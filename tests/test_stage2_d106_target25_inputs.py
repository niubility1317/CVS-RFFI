from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cvsrffi.stage2_d106_matrix_protocol import (
    RECEIVERS,
    TARGET25_SEED,
    TARGET25_SLICES,
)
from cvsrffi.stage2_d106_target25_inputs import (
    CONTEXT_SCHEMA,
    D106Target25InputError,
    D92_CANDIDATE,
    D92_MATRIX_SCHEMA,
    KCR_ROUTE_LOCK_SCHEMA,
    PLAN_SCHEMA,
    prepare_d106_target25_inputs,
)


SEEDS = (713102, 713103, 713104, 713105, 713106)
SCENARIOS = ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    raw = _canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _asset(root: Path, name: str) -> tuple[Path, str]:
    path = root / name
    path.write_bytes(f"asset:{name}".encode())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _job_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    return (
        f"rx_{receiver.replace('-', '_')}__seed_{seed}"
        f"__k_{k_shot}__new_{new_count}"
    )


def _packages(job_root: Path) -> None:
    for state in ("before", "after"):
        for leaf in ("enrollment_only", "apply_only_staging"):
            package = job_root / "offline" / "predictor" / state / leaf
            package.mkdir(parents=True)
            (package / "package_manifest.json").write_text("{}\n", encoding="utf-8")
        seals = job_root / "offline" / "seals"
        seals.mkdir(parents=True, exist_ok=True)
        apply_seals = job_root / "apply_seals"
        apply_seals.mkdir(parents=True, exist_ok=True)
        (seals / f"{state}_enrollment.seal.json").write_text(
            f'{{"state":"{state}","profile":"enrollment"}}\n', encoding="utf-8"
        )
        (apply_seals / f"{state}_apply.seal.json").write_text(
            f'{{"state":"{state}","profile":"apply"}}\n', encoding="utf-8"
        )


def _manifest(output_root: Path) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    index = 0
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in TARGET25_SLICES:
                job_id = _job_id(receiver, seed, k_shot, new_count)
                job_root = output_root / "jobs" / job_id
                if seed == TARGET25_SEED:
                    _packages(job_root)
                jobs.append(
                    {
                        "authority_bundle": f"/authority/{receiver}/{seed}",
                        "authority_commit_path": f"/authority/{receiver}/{seed}/COMMIT.json",
                        "authority_commit_sha256": hashlib.sha256(job_id.encode()).hexdigest(),
                        "cache_manifest": f"/cache/{receiver}/{seed}/cache_set.json",
                        "candidate": D92_CANDIDATE,
                        "index": index,
                        "job_id": job_id,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "output_root": str(job_root.resolve()),
                        "planned_shard_index": index % 8,
                        "receiver": receiver,
                        "row_pair": {
                            "after_registration": "stage2c",
                            "before_registration": "stage2b",
                        },
                        "scenarios": SCENARIOS,
                        "seed": seed,
                        "seed_role": "independent_stability_not_formal_confirmation",
                        "support_nesting": {
                            "k1_uses_first_k10_physical_support": False,
                            "k5_uses_first_five_k10_physical_support": False,
                            "policy": "existing_row_builder_physical_prefix",
                            "reference_k": 10,
                        },
                    }
                )
                index += 1
    return {
        "candidate": D92_CANDIDATE,
        "claim_scope": "development_only_not_formal_confirmation",
        "confirmation_seeds": list(SEEDS),
        "development_seed_excluded": 713101,
        "formal_launch_authority": False,
        "job_count": 125,
        "jobs": jobs,
        "locked_shard_count": 8,
        "method_lock": "/input/method_lock.json",
        "method_lock_sha256": "1" * 64,
        "output_root": str(output_root.resolve()),
        "phase1_checkpoint": "/input/checkpoint.pth",
        "phase1_checkpoint_sha256": "2" * 64,
        "phase2_contract": {
            "phase2_query_decision_policy": "per_sample_all_registered_classes"
        },
        "planned_shard_job_counts": [16, 16, 16, 16, 16, 15, 15, 15],
        "protocol_note": "locator source only",
        "receivers": list(RECEIVERS),
        "row_pair_count": 125,
        "row_pipeline": "/source/run_cvs_somph_diag_row_pipeline.py",
        "scenario_pair_count": 375,
        "scenario_state_metric_count": 750,
        "schema": D92_MATRIX_SCHEMA,
        "sealed_runtime": "/input/sealed_feature_runtime.pt",
        "sealed_runtime_sha256": "3" * 64,
        "slices": [
            {"k_shot": k_shot, "new_class_count": new_count}
            for k_shot, new_count in TARGET25_SLICES
        ],
        "stage2_balance": "equal priority",
        "status": "LOCKED_INDEPENDENT_STABILITY_TRANCHE",
    }


def _inputs(tmp_path: Path) -> dict[str, Any]:
    d92_root = tmp_path / "d92"
    d92_root.mkdir()
    manifest = _manifest(d92_root)
    manifest_path = tmp_path / "matrix_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    checkpoint, checkpoint_sha = _asset(tmp_path, "checkpoint.pth")
    rdce_wire, rdce_wire_sha = _asset(tmp_path, "rdce.wire")
    rdce_lock, rdce_lock_sha = _asset(tmp_path, "rdce.lock.json")
    rcmr_lock, rcmr_lock_sha = _asset(tmp_path, "rcmr.lock.json")
    route_path = tmp_path / "kcr.route.lock.json"
    route_sha = _write_json(
        route_path,
        {
            "schema": KCR_ROUTE_LOCK_SCHEMA,
            "candidate_id": "D106-KCR/r1",
            "route_by_k": {"1": "M_DA", "5": "M0", "10": "M_HEAD"},
            "query_truth_access": False,
            "query_role_access": False,
            "query_fit_access": False,
            "query_update_access": False,
            "query_selection": False,
        },
    )
    return {
        "d92_matrix_manifest_path": manifest_path,
        "expected_d92_matrix_manifest_sha256": manifest_sha,
        "d92_output_root": d92_root,
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": checkpoint_sha,
        "rdce_wire_path": rdce_wire,
        "expected_rdce_wire_sha256": rdce_wire_sha,
        "rdce_lock_path": rdce_lock,
        "expected_rdce_lock_sha256": rdce_lock_sha,
        "rcmr_lock_path": rcmr_lock,
        "expected_rcmr_lock_sha256": rcmr_lock_sha,
        "kcr_route_lock_path": route_path,
        "expected_kcr_route_lock_sha256": route_sha,
        "output_dir": tmp_path / "prepared",
    }


def test_prepare_projects_exact_target25_and_only_three_field_package_refs(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    result = prepare_d106_target25_inputs(**kwargs)
    plan = json.loads(Path(result["plan_manifest"]).read_text(encoding="utf-8"))
    context = json.loads(Path(result["context_manifest"]).read_text(encoding="utf-8"))
    assert plan["schema"] == PLAN_SCHEMA
    assert context["schema"] == CONTEXT_SCHEMA
    assert plan["claim_scope"] == "development_screen"
    assert plan["formal_launch_authority"] is False
    assert len(plan["rows"]) == 25
    assert plan["rows"] == context["rows"]
    assert [(row["receiver"], row["k_shot"], row["new_count"]) for row in plan["rows"]] == [
        (receiver, k_shot, new_count)
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    ]
    for row in plan["rows"]:
        assert set(row["packages"]) == {
            "before_enrollment",
            "before_apply",
            "after_enrollment",
            "after_apply",
        }
        for package in row["packages"].values():
            assert set(package) == {
                "package_root",
                "detached_seal_path",
                "expected_seal_sha256",
            }
            assert Path(package["package_root"]).is_dir()
            assert Path(package["detached_seal_path"]).is_file()
    serialized = json.dumps({"plan": plan, "context": context})
    for forbidden in (
        "split_locator",
        "formal_policy",
        "signed_policy",
        "authority_locator",
        "VALIDATED_ONCE",
        "capsule_id",
        "split_id",
    ):
        assert forbidden not in serialized


def test_prepare_rejects_wrong_matrix_sha(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    kwargs["expected_d92_matrix_manifest_sha256"] = "0" * 64
    with pytest.raises(D106Target25InputError, match="SHA mismatch"):
        prepare_d106_target25_inputs(**kwargs)


def test_prepare_rejects_missing_package(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    root = Path(kwargs["d92_output_root"])
    missing = (
        root
        / "jobs"
        / _job_id("20-1", TARGET25_SEED, 10, 5)
        / "offline"
        / "predictor"
        / "before"
        / "enrollment_only"
    )
    missing.rename(missing.with_name("missing"))
    with pytest.raises(D106Target25InputError, match="package root"):
        prepare_d106_target25_inputs(**kwargs)


def test_prepare_rejects_target25_order_drift(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    path = Path(kwargs["d92_matrix_manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selected = [i for i, row in enumerate(manifest["jobs"]) if row["seed"] == TARGET25_SEED]
    left, right = selected[:2]
    manifest["jobs"][left], manifest["jobs"][right] = (
        manifest["jobs"][right],
        manifest["jobs"][left],
    )
    kwargs["expected_d92_matrix_manifest_sha256"] = _write_json(path, manifest)
    with pytest.raises(D106Target25InputError, match="order/coverage"):
        prepare_d106_target25_inputs(**kwargs)


@pytest.mark.parametrize("location", ["manifest", "job"])
def test_prepare_rejects_extra_matrix_fields(tmp_path: Path, location: str) -> None:
    kwargs = _inputs(tmp_path)
    path = Path(kwargs["d92_matrix_manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if location == "manifest":
        manifest["unexpected"] = True
    else:
        manifest["jobs"][0]["unexpected"] = True
    kwargs["expected_d92_matrix_manifest_sha256"] = _write_json(path, manifest)
    with pytest.raises(D106Target25InputError, match="field closure"):
        prepare_d106_target25_inputs(**kwargs)
