import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.stage2_predictor_bundle import (
    FORMAL_LEO_WEAK_SCENARIOS,
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    iq_row_sha256,
    make_member_descriptor,
    write_predictor_package_manifest_and_seal,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opaque(prefix: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()}"


def _write_support_package(
    root: Path,
    *,
    receiver: str,
    seed: int,
    new_count: int,
    old_labels: list[str],
    max_k: int,
    old_token_salt: str = "old",
) -> Path:
    """Write one real sealed fixture package without exposing query values to tests."""

    root.mkdir(parents=True)
    members = []
    for role, filename, schema in (
        ("checkpoint", "checkpoint.bin", "adv3b02.torchscript_identity_runtime.v1"),
        ("adapter", "adapter.bin", "cvs.feature_adapter.v1"),
        ("head", "head.bin", "cvs.prototype_head.v1"),
        ("tta_policy", "tta_policy.json", "cvs.adaptive_tta.v1"),
    ):
        path = root / filename
        path.write_bytes(role.encode("utf-8"))
        members.append(
            make_member_descriptor(
                path,
                relative_path=filename,
                artifact_role=role,
                schema=schema,
            )
        )

    class_count = len(old_labels) + new_count
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        support_rows = class_count * max_k
        support_iq = np.arange(
            support_rows * 2 * 4, dtype=np.float32
        ).reshape(support_rows, 2, 4) + scenario_index
        labels = np.repeat(np.arange(class_count, dtype=np.int64), max_k)
        ranks = np.tile(np.arange(max_k, dtype=np.int64), class_count)
        tokens = np.asarray(
            [
                _opaque(
                    "sid",
                    receiver,
                    seed,
                    scenario,
                    class_index,
                    rank,
                    old_token_salt if class_index < len(old_labels) else new_count,
                )
                for class_index, rank in zip(labels.tolist(), ranks.tolist())
            ]
        )
        support_path = root / f"support_{scenario}.npz"
        support_manifest = {
            "schema": SUPPORT_SCHEMA,
            "scenario": scenario,
            "registered_support_labels_allowed": True,
            "registered_class_count": class_count,
            "support_pool_max_k": max_k,
            "token_scheme": "hmac_sha256_opaque_v1",
        }
        with support_path.open("xb") as handle:
            np.savez(
                handle,
                support_pool_leo_weak_iq=support_iq,
                support_pool_class_indices=labels,
                support_pool_rank_within_class=ranks,
                support_pool_tokens=tokens,
                support_pool_overlay_tokens=np.asarray(
                    [
                        _opaque("oid", receiver, seed, scenario, "support", index)
                        for index in range(support_rows)
                    ]
                ),
                support_pool_satellite_seeds=np.arange(
                    support_rows, dtype=np.int64
                ),
                support_pool_post_channel_iq_sha256=np.asarray(
                    [iq_row_sha256(row) for row in support_iq]
                ),
                manifest_json=np.asarray(json.dumps(support_manifest, sort_keys=True)),
            )
        members.append(
            make_member_descriptor(
                support_path,
                relative_path=support_path.name,
                artifact_role=f"support:{scenario}",
                schema=SUPPORT_SCHEMA,
                scenario=scenario,
                npz_members=SUPPORT_NPZ_MEMBERS,
            )
        )

        query_iq = np.full((1, 2, 4), scenario_index, dtype=np.float32)
        query_path = root / f"query_{scenario}.npz"
        query_manifest = {
            "schema": QUERY_SCHEMA,
            "scenario": scenario,
            "query_truth_included": False,
            "query_role_included": False,
            "query_true_batch_class_count_included": False,
            "query_class_quota_included": False,
            "query_ordering_hint_included": False,
            "token_scheme": "hmac_sha256_opaque_v1",
        }
        with query_path.open("xb") as handle:
            np.savez(
                handle,
                query_leo_weak_iq=query_iq,
                query_tokens=np.asarray([_opaque("qid", receiver, seed, scenario)]),
                query_overlay_tokens=np.asarray(
                    [_opaque("oid", receiver, seed, scenario, "query")]
                ),
                query_satellite_seeds=np.asarray([scenario_index], dtype=np.int64),
                query_post_channel_iq_sha256=np.asarray([iq_row_sha256(query_iq[0])]),
                manifest_json=np.asarray(json.dumps(query_manifest, sort_keys=True)),
            )
        members.append(
            make_member_descriptor(
                query_path,
                relative_path=query_path.name,
                artifact_role=f"query:{scenario}",
                schema=QUERY_SCHEMA,
                scenario=scenario,
                npz_members=QUERY_NPZ_MEMBERS,
            )
        )

    metadata = {
        "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
        "artifact_stage": PREDICTOR_INPUT_STAGE,
        "stage": "stage2c",
        "receiver": receiver,
        "seed": seed,
        "new_class_count": new_count,
        "support_pool_max_k": max_k,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": class_count,
        "registered_classes": [
            {
                "class_index": index,
                "class_handle": _opaque("cls", receiver, seed, index),
            }
            for index in range(class_count)
        ],
        "candidate_lock_sha256": "c" * 64,
        **PHASE2_FULL_CONTRACT,
    }
    seal_path = root.parent / f"{root.name}.seal.json"
    write_predictor_package_manifest_and_seal(
        root,
        manifest_metadata=metadata,
        members=members,
        detached_seal_path=seal_path,
    )
    return seal_path


def _miniature_v7_inputs(tmp_path: Path) -> dict[str, object]:
    receivers = ["20-1", "3-19"]
    seeds = [713101]
    k_values = [1, 2]
    new_counts = [2, 5]
    methods = ["csil_paper_full", "mopc_hr_paper_full"]
    old_labels = [f"old_{index}" for index in range(6)]

    base_checkpoint = tmp_path / "base_checkpoint.bin"
    base_checkpoint.write_bytes(b"checkpoint")
    artifacts = {
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": _sha256(base_checkpoint),
        }
    }
    packages = []
    cells = []
    for receiver in receivers:
        for seed in seeds:
            for new_count in new_counts:
                package_id = f"rx_{receiver.replace('-', '_')}__seed_{seed}__new_{new_count}"
                package_root = tmp_path / "packages" / package_id / "predictor"
                seal_path = _write_support_package(
                    package_root,
                    receiver=receiver,
                    seed=seed,
                    new_count=new_count,
                    old_labels=old_labels,
                    max_k=max(k_values),
                )
                packages.append(
                    {
                        "package_id": package_id,
                        "receiver": receiver,
                        "seed": seed,
                        "new_class_count": new_count,
                        "old_class_labels": old_labels,
                        "new_class_labels": [f"new_{index}" for index in range(new_count)],
                        "predictor_package_root": str(package_root),
                        "detached_seal": str(seal_path),
                    }
                )
                for method in methods:
                    for k_shot in k_values:
                        cell_id = f"{package_id}__{method}__k_{k_shot}"
                        cells.append(
                            {
                                "cell_id": cell_id,
                                "package_id": package_id,
                                "receiver": receiver,
                                "seed": seed,
                                "new_class_count": new_count,
                                "method": method,
                                "k_shot": k_shot,
                                "output_root": str(tmp_path / "v7_cells" / cell_id),
                            }
                        )
    source_plan = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_plan.v1",
        "experiment_id": "adv3b02_unfrozen_paperfull_ci_20260723_v7",
        "methods": methods,
        "receivers": receivers,
        "seeds": seeds,
        "k_values": k_values,
        "new_class_counts": new_counts,
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "artifacts": artifacts,
        "packages": packages,
        "cells": cells,
        "counts": {
            "packages": len(packages),
            "cells": len(cells),
            "scenario_rows": len(cells) * len(FORMAL_LEO_WEAK_SCENARIOS),
        },
    }
    source_plan_path = tmp_path / "source_v7_plan.json"
    source_plan_path.write_text(json.dumps(source_plan), encoding="utf-8")
    source_cache = tmp_path / "source_cache_set.json"
    source_cache.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_v1",
                "cache_scope": "source_train",
                "target_channel_view": "leo_weak_only",
                "cache_npz_by_scenario": {
                    scenario: f"{scenario}.npz" for scenario in FORMAL_LEO_WEAK_SCENARIOS
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_plan": source_plan_path,
        "source_plan_sha256": _sha256(source_plan_path),
        "source_cache": source_cache,
        "source_cache_sha256": _sha256(source_cache),
        "run_root": tmp_path / "mrior_run",
        "output": tmp_path / "mrior_plan.json",
        "receivers": tuple(receivers),
        "seeds": tuple(seeds),
        "k_values": tuple(k_values),
        "new_counts": tuple(new_counts),
    }


def test_miniature_plan_deduplicates_preadaptation_across_methods_and_new_counts(
    tmp_path: Path,
) -> None:
    """A missing builder, duplicate jobs, or non-anchor reuse must fail this contract."""

    values = _miniature_v7_inputs(tmp_path)
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    plan = module._build_for_test(
        source_plan=values["source_plan"],
        expected_source_plan_sha256=values["source_plan_sha256"],
        source_cache_manifest=values["source_cache"],
        expected_source_cache_manifest_sha256=values["source_cache_sha256"],
        run_root=values["run_root"],
        output=values["output"],
        expected_receivers=values["receivers"],
        expected_seeds=values["seeds"],
        expected_k_values=values["k_values"],
        expected_new_counts=values["new_counts"],
        expected_source_cache_path=str(values["source_cache"]),
    )

    assert plan["schema"] == "cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1"
    assert plan["counts"] == {
        "preadapt_jobs": 12,
        "cells": 16,
        "scenario_rows": 48,
    }
    assert plan["preadapt_anchor_new_class_count"] == 2
    assert len(plan["preadapt_jobs"]) == 12
    assert len(plan["cells"]) == 16
    assert plan["smoke_preadapt_job_ids"] == [
        "rx_20_1__seed_713101__k_1__scene_leo_clear_weak",
        "rx_20_1__seed_713101__k_1__scene_leo_low_elev_weak",
        "rx_20_1__seed_713101__k_1__scene_leo_rain_weak",
    ]
    assert len(plan["smoke_cell_ids"]) == 4
    assert Path(values["output"]).is_file()
    assert [job["job_id"] for job in plan["preadapt_jobs"]] == sorted(
        job["job_id"] for job in plan["preadapt_jobs"]
    )

    expected_job_ids = {
        "rx_20_1__seed_713101__k_1__scene_leo_clear_weak",
        "rx_20_1__seed_713101__k_2__scene_leo_rain_weak",
        "rx_3_19__seed_713101__k_1__scene_leo_low_elev_weak",
    }
    assert expected_job_ids <= {job["job_id"] for job in plan["preadapt_jobs"]}
    anchor_seals = {
        job["job_id"]: job["target_package_seal_sha256"]
        for job in plan["preadapt_jobs"]
    }
    matching_cells = [
        cell
        for cell in plan["cells"]
        if cell["receiver"] == "20-1"
        and cell["seed"] == 713101
        and cell["k_shot"] == 1
    ]
    assert len(matching_cells) == 4
    assert {
        cell["preadapt_job_ids_by_scenario"]["leo_clear_weak"]
        for cell in matching_cells
    } == {"rx_20_1__seed_713101__k_1__scene_leo_clear_weak"}
    assert {
        cell["preadapt_anchor_target_package_seal_sha256"]
        for cell in matching_cells
    } == {anchor_seals["rx_20_1__seed_713101__k_1__scene_leo_clear_weak"]}
    assert {cell["method"] for cell in plan["cells"]} == {
        "mrior_sda_then_csil_paper_full",
        "mrior_sda_then_mopc_hr_paper_full",
    }
    first_job = plan["preadapt_jobs"][0]
    assert first_job["method_lock"] == {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_method_lock.v1",
        "method_id": "mrior_sda",
        "adapt_steps": 200,
        "mrior_adapt_learning_rate": 0.0006,
        "mrior_estimate_steps": 7,
        "target_ce_weight": 1.0,
        "dvkl_weight": 0.005,
        "mrior_mu": 0.5,
    }
    assert first_job["input_binding"]["target_package_seal_sha256"] == first_job[
        "target_package_seal_sha256"
    ]


def _rewrite_source_plan(
    values: dict[str, object], mutate
) -> tuple[Path, str]:
    source_path = values["source_plan"]
    assert isinstance(source_path, Path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    mutate(payload)
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    return source_path, _sha256(source_path)


def test_rejects_non_v7_source_plan_before_creating_output(tmp_path: Path) -> None:
    """Changing the baseline schema must reject the input rather than mint a new matrix."""

    values = _miniature_v7_inputs(tmp_path)
    source_path, source_sha = _rewrite_source_plan(
        values,
        lambda payload: payload.__setitem__("schema", "cvs.phase2.other_plan.v1"),
    )
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="authorized v7 schema"):
        module._build_for_test(
            source_plan=source_path,
            expected_source_plan_sha256=source_sha,
            source_cache_manifest=values["source_cache"],
            expected_source_cache_manifest_sha256=values["source_cache_sha256"],
            run_root=values["run_root"],
            output=values["output"],
            expected_receivers=values["receivers"],
            expected_seeds=values["seeds"],
            expected_k_values=values["k_values"],
            expected_new_counts=values["new_counts"],
            expected_source_cache_path=str(values["source_cache"]),
        )
    assert not Path(values["output"]).exists()


def test_rejects_target_old_support_identity_drift_across_new_counts(
    tmp_path: Path,
) -> None:
    """A legal new-class package cannot silently substitute old support for a shared job."""

    values = _miniature_v7_inputs(tmp_path)
    source_path = values["source_plan"]
    assert isinstance(source_path, Path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    package = next(
        value
        for value in payload["packages"]
        if value["receiver"] == "20-1"
        and value["seed"] == 713101
        and value["new_class_count"] == 5
    )
    drift_root = tmp_path / "drifted_package" / "predictor"
    drift_seal = _write_support_package(
        drift_root,
        receiver="20-1",
        seed=713101,
        new_count=5,
        old_labels=[f"old_{index}" for index in range(6)],
        max_k=2,
        old_token_salt="changed-old-support",
    )
    package["predictor_package_root"] = str(drift_root)
    package["detached_seal"] = str(drift_seal)
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="old support identity drift"):
        module._build_for_test(
            source_plan=source_path,
            expected_source_plan_sha256=_sha256(source_path),
            source_cache_manifest=values["source_cache"],
            expected_source_cache_manifest_sha256=values["source_cache_sha256"],
            run_root=values["run_root"],
            output=values["output"],
            expected_receivers=values["receivers"],
            expected_seeds=values["seeds"],
            expected_k_values=values["k_values"],
            expected_new_counts=values["new_counts"],
            expected_source_cache_path=str(values["source_cache"]),
        )


def test_rejects_duplicate_source_package_identity(tmp_path: Path) -> None:
    """Duplicate package input would otherwise produce ambiguous preadapt job binding."""

    values = _miniature_v7_inputs(tmp_path)
    source_path, source_sha = _rewrite_source_plan(
        values,
        lambda payload: payload["packages"].__setitem__(
            -1, dict(payload["packages"][0])
        ),
    )
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="duplicate package identity"):
        module._build_for_test(
            source_plan=source_path,
            expected_source_plan_sha256=source_sha,
            source_cache_manifest=values["source_cache"],
            expected_source_cache_manifest_sha256=values["source_cache_sha256"],
            run_root=values["run_root"],
            output=values["output"],
            expected_receivers=values["receivers"],
            expected_seeds=values["seeds"],
            expected_k_values=values["k_values"],
            expected_new_counts=values["new_counts"],
            expected_source_cache_path=str(values["source_cache"]),
        )


def test_rejects_source_cache_with_non_source_scope(tmp_path: Path) -> None:
    """MRIOR may read only the explicitly declared source cache-set scope."""

    values = _miniature_v7_inputs(tmp_path)
    cache_path = values["source_cache"]
    assert isinstance(cache_path, Path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["cache_scope"] = "stage2_registered"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="source cache scope drift"):
        module._build_for_test(
            source_plan=values["source_plan"],
            expected_source_plan_sha256=values["source_plan_sha256"],
            source_cache_manifest=cache_path,
            expected_source_cache_manifest_sha256=_sha256(cache_path),
            run_root=values["run_root"],
            output=values["output"],
            expected_receivers=values["receivers"],
            expected_seeds=values["seeds"],
            expected_k_values=values["k_values"],
            expected_new_counts=values["new_counts"],
            expected_source_cache_path=str(cache_path),
        )


def test_rejects_missing_formal_leo_scene_in_source_plan(tmp_path: Path) -> None:
    """A three-scene job fan-out must not proceed from a partial baseline plan."""

    values = _miniature_v7_inputs(tmp_path)
    source_path, source_sha = _rewrite_source_plan(
        values,
        lambda payload: payload.__setitem__(
            "scenarios", ["leo_clear_weak", "leo_rain_weak"]
        ),
    )
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="LEO scenario matrix drift"):
        module._build_for_test(
            source_plan=source_path,
            expected_source_plan_sha256=source_sha,
            source_cache_manifest=values["source_cache"],
            expected_source_cache_manifest_sha256=values["source_cache_sha256"],
            run_root=values["run_root"],
            output=values["output"],
            expected_receivers=values["receivers"],
            expected_seeds=values["seeds"],
            expected_k_values=values["k_values"],
            expected_new_counts=values["new_counts"],
            expected_source_cache_path=str(values["source_cache"]),
        )


def test_formal_build_rejects_any_source_plan_sha_except_frozen_v7(
    tmp_path: Path,
) -> None:
    """The command path cannot turn a miniature matrix into an authorized release."""

    values = _miniature_v7_inputs(tmp_path)
    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    with pytest.raises(ValueError, match="frozen v7 SHA"):
        module.build(
            argparse.Namespace(
                source_plan=values["source_plan"],
                expected_source_plan_sha256=values["source_plan_sha256"],
                source_cache_manifest=values["source_cache"],
                expected_source_cache_manifest_sha256=values["source_cache_sha256"],
                run_root=values["run_root"],
                output=values["output"],
            )
        )


def test_formal_contract_keeps_the_frozen_300_job_800_cell_dimensions() -> None:
    """Changing a formal grid member must fail the matrix-size contract review."""

    module = importlib.import_module(
        "paper_reproduction.scripts.build_adv3b02_mrior_preadapt_ci_plan"
    )
    contract = module._formal_contract()
    assert contract.receivers == ("20-1", "3-19", "7-14", "7-7", "8-8")
    assert contract.seeds == (713101, 713102, 713103, 713104, 713105)
    assert contract.k_values == (1, 5, 10, 20)
    assert contract.new_counts == (2, 5, 10, 20)
    assert module._expected_counts(contract) == {
        "packages": 100,
        "cells": 800,
        "scenario_rows": 2400,
    }
    assert (
        len(contract.receivers)
        * len(contract.seeds)
        * len(contract.k_values)
        * len(FORMAL_LEO_WEAK_SCENARIOS)
        == 300
    )


def test_builder_cli_imports_from_outside_the_repository(tmp_path: Path) -> None:
    """The released CLI must add both repository and code roots before imports."""

    script = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction"
        / "scripts"
        / "build_adv3b02_mrior_preadapt_ci_plan.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-source-plan-sha256" in completed.stdout
