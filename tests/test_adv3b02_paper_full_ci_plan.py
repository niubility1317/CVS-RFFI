import argparse
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi.leo_weak_cache import (
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)
from paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle import (
    _comparison_reference_arrays,
    load_comparison_inner_leo_cache,
    load_comparison_leo_cache_set,
    load_verified_comparison_stage2_predictor_bundle,
)
from paper_reproduction.scripts.build_adv3b02_paper_full_ci_plan import (
    build,
    validate_adapter_required_capacity,
    validate_adapter_release_matrix,
)
from paper_reproduction.scripts.run_adv3b02_paper_full_ci_plan import (
    _expected_method_status,
    _load_plan,
    _update_health_state,
    _verify_cache_parity_receipt,
    _verify_smoke_authority,
)
from paper_reproduction.scripts.run_adv3b02_paper_full_ci_truth_free_predictor import (
    _load_base_state,
    _method_receipt_semantics,
)
import paper_reproduction.scripts.run_adv3b02_paper_full_ci_truth_free_predictor as paper_full_predictor
from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (
    MRIORPreadaptInputBinding,
    expected_mrior_preadapt_method_lock,
)
from paper_reproduction.scripts.verify_adv3b02_official_scale_cache_parity import (
    verify as verify_scale_cache_parity,
)
from paper_reproduction.scripts.build_adv3b02_official_scale_cache_specs import (
    build as build_scale_cache_specs,
)
from paper_reproduction.scripts.build_adv3b02_official_scale_cache_reuse_manifest import (
    build as build_scale_cache_reuse_manifest,
)
CODE_SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(CODE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CODE_SCRIPTS))
from build_cvs_leo_weak_iq_cache import validate_build_spec


def test_mrior_preadapt_predictor_loads_every_verified_artifact_before_opening_new_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening support before every verified DA1_REG0 artifact would break CI lineage."""

    method = "mrior_sda_then_csil_paper_full"
    scenarios = tuple(paper_full_predictor.FORMAL_LEO_WEAK_SCENARIOS)
    method_lock = expected_mrior_preadapt_method_lock()
    method_lock_sha256 = hashlib.sha256(
        json.dumps(
            method_lock,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    artifact_root_by_scene: dict[str, Path] = {}
    binding_by_scene: dict[str, MRIORPreadaptInputBinding] = {}
    for scene_index, scene in enumerate(scenarios):
        artifact_root = tmp_path / f"artifact_{scene}"
        artifact_root.mkdir()
        artifact_root_by_scene[scene] = artifact_root
        binding_by_scene[scene] = MRIORPreadaptInputBinding.from_verified_values(
            checkpoint_sha256="a" * 64,
            source_cache_sha256="b" * 64,
            support_token_sha256=f"{scene_index + 1:064x}",
            target_package_seal_sha256="d" * 64,
            receiver="20-1",
            seed=713101,
            k_shot=1,
            scene=scene,
        )
    bindings_path = tmp_path / "preadapt_bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.adv3b02_mrior_preadapt_predictor_bindings.v1",
                "bindings": {
                    scene: {
                        "artifact_root": str(artifact_root_by_scene[scene]),
                        "expected_input_binding_sha256": binding_by_scene[
                            scene
                        ].canonical_sha256,
                        "expected_method_lock_sha256": method_lock_sha256,
                    }
                    for scene in scenarios
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "stage": "stage2c",
        "receiver": "20-1",
        "seed": 713101,
        "support_pool_max_k": 1,
        "registered_classes": [
            {"class_handle": "old_0"},
            {"class_handle": "new_0"},
        ],
        "candidate_lock_sha256": "e" * 64,
        "package_root_sha256": "f" * 64,
        "members": [
            {
                "artifact_role": "checkpoint",
                "relative_path": "checkpoint.pt",
                "sha256": "a" * 64,
                "size_bytes": 123,
                "schema": "adv3b02.torchscript_identity_runtime.v1",
                "scenario": None,
                "npz_members": [],
            },
            {"artifact_role": "head", "relative_path": "head.pt"},
            *[
                {
                    "artifact_role": f"support:{scene}",
                    "relative_path": f"support_{scene}.npz",
                }
                for scene in scenarios
            ],
            *[
                {
                    "artifact_role": f"query:{scene}",
                    "relative_path": f"query_{scene}.npz",
                }
                for scene in scenarios
            ],
        ],
    }
    events: list[str] = []

    class SupportOpenObserved(RuntimeError):
        pass

    def fake_verified_loader(
        artifact_root: Path | str,
        *,
        expected_input_binding_sha256: str,
        expected_method_lock_sha256: str,
    ) -> SimpleNamespace:
        root = Path(artifact_root).resolve()
        scene = next(
            item
            for item, candidate in artifact_root_by_scene.items()
            if candidate.resolve() == root
        )
        assert expected_input_binding_sha256 == binding_by_scene[scene].canonical_sha256
        assert expected_method_lock_sha256 == method_lock_sha256
        events.append(f"load:{scene}")
        return SimpleNamespace(
            model_state={
                "id_backbone.weight": torch.ones((1, 1)),
                "id_backbone.bias": torch.tensor([float(scenarios.index(scene))]),
            },
            input_binding=binding_by_scene[scene],
            method_lock=method_lock,
            query_unopened_receipt={
                "query_opened_before_model_lock": False,
                "query_rows_used_for_training": 0,
                "query_truth_access": False,
                "query_role_access": False,
                "query_class_quota_access": False,
                "query_global_reassignment_access": False,
            },
        )

    def fake_materialize(_package_root: Path, descriptor: dict[str, str]):
        role = descriptor["artifact_role"]
        events.append(f"open:{role}")
        if role.startswith("support:"):
            for prefix in ("load", "restore"):
                assert [event for event in events if event.startswith(f"{prefix}:")] == [
                    f"{prefix}:{scene}" for scene in scenarios
                ]
                assert events.index("full_preflight") > max(
                    events.index(f"{prefix}:{scene}") for scene in scenarios
                )
            raise SupportOpenObserved(role)
        raise AssertionError(f"query must not open in this ordering probe: {role}")

    def fake_metadata_preflight(*_args, **_kwargs):
        events.append("metadata_preflight")
        return manifest

    def fake_full_preflight(*_args, **_kwargs):
        events.append("full_preflight")
        for prefix in ("load", "restore"):
            assert [event for event in events if event.startswith(f"{prefix}:")] == [
                f"{prefix}:{scene}" for scene in scenarios
            ]
        return manifest, {}, {}

    monkeypatch.setattr(
        paper_full_predictor,
        "METHODS",
        paper_full_predictor.METHODS + (method,),
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "preflight_stage2_predictor_package",
        fake_full_preflight,
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "_preflight_mrior_preadapt_metadata",
        fake_metadata_preflight,
        raising=False,
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "_load_base_state",
        lambda *_args, **_kwargs: {"receipt": {}},
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "load_verified_mrior_preadapt_artifact",
        fake_verified_loader,
        raising=False,
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "_load_exact_backbone",
        lambda *_args, **_kwargs: (
            torch.nn.Linear(1, 1),
            lambda backbone, rows: (backbone(rows), backbone(rows)),
            {},
        ),
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "_restore_mrior_preadapted_backbone",
        lambda _backbone, state: events.append(
            f"restore:{scenarios[int(state['id_backbone.bias'].item())]}"
        ),
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "sha256_file",
        lambda _path: "9" * 64,
    )
    monkeypatch.setattr(paper_full_predictor, "_materialize_npz", fake_materialize)

    with pytest.raises(SupportOpenObserved):
        paper_full_predictor.predict(
            argparse.Namespace(
                package_root=tmp_path,
                detached_seal=tmp_path / "package.seal.json",
                expected_seal_sha256="d" * 64,
                method=method,
                old_class_count=1,
                expected_total_capacity=2,
                k_shot=1,
                seed=713101,
                row_id="ordering-red",
                output_dir=tmp_path / "output",
                device="cpu",
                batch_size=8,
                mrior_preadapt_bindings=bindings_path,
            )
        )

    for prefix in ("load", "restore"):
        assert [event for event in events if event.startswith(f"{prefix}:")] == [
            f"{prefix}:{scene}" for scene in scenarios
        ]


def test_mrior_preadapt_rejects_package_seed_mismatch_before_artifact_or_full_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sealed package seed must bind before any artifact or support access."""

    events: list[str] = []

    monkeypatch.setattr(
        paper_full_predictor,
        "_preflight_mrior_preadapt_metadata",
        lambda *_args, **_kwargs: {"stage": "stage2c", "seed": 713102},
        raising=False,
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "preflight_stage2_predictor_package",
        lambda *_args, **_kwargs: events.append("full_preflight"),
    )
    monkeypatch.setattr(
        paper_full_predictor,
        "load_verified_mrior_preadapt_artifact",
        lambda *_args, **_kwargs: events.append("artifact_load"),
        raising=False,
    )

    with pytest.raises(ValueError, match="package seed does not match"):
        paper_full_predictor.predict(
            argparse.Namespace(
                package_root=tmp_path,
                detached_seal=tmp_path / "package.seal.json",
                expected_seal_sha256="d" * 64,
                method="mrior_sda_then_csil_paper_full",
                old_class_count=1,
                expected_total_capacity=2,
                k_shot=1,
                seed=713101,
                row_id="seed-binding-red",
                output_dir=tmp_path / "output",
                device="cpu",
                batch_size=8,
                mrior_preadapt_bindings=tmp_path / "bindings.json",
            )
        )

    assert events == []


def test_mrior_preadapt_lineage_accepts_the_plan_bound_anchor_seal_for_new_count_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused preadapt artifact is anchored to the K-matched package, not every new-count seal."""

    binding = MRIORPreadaptInputBinding.from_verified_values(
        checkpoint_sha256="a" * 64,
        source_cache_sha256="b" * 64,
        support_token_sha256="c" * 64,
        target_package_seal_sha256="d" * 64,
        receiver="20-1",
        seed=713101,
        k_shot=5,
        scene="leo_rain_weak",
    )
    result = SimpleNamespace(input_binding=binding)
    monkeypatch.setattr(paper_full_predictor, "sha256_file", lambda _path: "e" * 64)

    lineage = paper_full_predictor._validated_mrior_preadapt_lineage(
        result,
        artifact_root=tmp_path,
        checkpoint_sha256="a" * 64,
        target_package_seal_sha256="f" * 64,
        receiver="20-1",
        seed=713101,
        k_shot=5,
        scenario="leo_rain_weak",
    )

    assert lineage["state"] == "DA1_REG0"
    assert lineage["target_package_seal_sha256"] == "d" * 64


@pytest.mark.parametrize(
    "method",
    (
        "mrior_sda_then_csil_paper_full",
        "mrior_sda_then_mopc_hr_paper_full",
    ),
)
def test_mrior_preadapt_method_rejects_missing_artifact_bindings_before_package_open(
    tmp_path: Path, method: str
) -> None:
    """A preadapted method must not open a package when its frozen artifacts are absent."""

    with pytest.raises(ValueError, match="require --mrior-preadapt-bindings"):
        paper_full_predictor.predict(
            argparse.Namespace(
                package_root=tmp_path,
                detached_seal=tmp_path / "package.seal.json",
                expected_seal_sha256="a" * 64,
                method=method,
                old_class_count=1,
                expected_total_capacity=2,
                k_shot=1,
                seed=713101,
                row_id="missing-bindings",
                output_dir=tmp_path / "output",
                device="cpu",
                batch_size=8,
                mrior_preadapt_bindings=None,
            )
        )


@pytest.mark.parametrize("method", ("csil_paper_full", "mopc_hr_paper_full"))
def test_original_paper_full_method_rejects_any_mrior_preadapt_bindings(
    tmp_path: Path, method: str
) -> None:
    """Passing an artifact to an original method must not silently change its route."""

    with pytest.raises(ValueError, match="only MRIOR preadapted methods"):
        paper_full_predictor.predict(
            argparse.Namespace(
                package_root=tmp_path,
                detached_seal=tmp_path / "package.seal.json",
                expected_seal_sha256="a" * 64,
                method=method,
                old_class_count=1,
                expected_total_capacity=2,
                k_shot=1,
                seed=713101,
                row_id="original-method-rejects-artifact",
                output_dir=tmp_path / "output",
                device="cpu",
                batch_size=8,
                mrior_preadapt_bindings=tmp_path / "untrusted-bindings.json",
            )
        )


def test_real_wisig_builder_supports_cache_exclusion_contract():
    exporter_path = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "export_spaceborne_features.py"
    )
    spec = importlib.util.spec_from_file_location(
        "adv3b02_real_export_spaceborne_features_contract",
        exporter_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = inspect.signature(module._build_wisig_dataset)
    assert "exclude_source_record_indices" in signature.parameters


def test_paper_full_plan_has_complete_matrix_and_locked_methods(tmp_path):
    artifacts = {}
    for name in ("checkpoint", "candidate", "adapter", "head", "tta"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    split = {
        "target_old_tx_labels": ["o0", "o1", "o2", "o3", "o4", "o5"],
        "nested_target_new_tx_labels": {
            "2": ["n0", "n1"],
            "5": [f"n{i}" for i in range(5)],
            "10": [f"n{i}" for i in range(10)],
            "20": [f"n{i}" for i in range(20)],
        },
        "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "confirmation_seeds": [713101, 713102, 713103, 713104, 713105],
        "k_values": [1, 5, 10, 20],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "plan.json"
    plan = build(
        argparse.Namespace(
            experiment_id="paper_full_test",
            run_root=str(tmp_path / "run"),
            target_cache_root=str(tmp_path / "cache"),
            class_split=split_path,
            base_checkpoint=str(artifacts["checkpoint"]),
            candidate_lock=str(artifacts["candidate"]),
            adapter=str(artifacts["adapter"]),
            head_artifact=str(artifacts["head"]),
            tta_policy=str(artifacts["tta"]),
            smoke_receipt=None,
            output=output,
        )
    )
    assert plan["counts"] == {"packages": 100, "cells": 800, "scenario_rows": 2400}
    assert plan["backbone_uniformly_frozen"] is False
    assert plan["base_source_reference_access_allowed"] is True
    assert plan["new_class_counts"] == [2, 5, 10, 20]
    assert len(plan["smoke_cell_ids"]) == 4
    assert set(plan["methods"]) == {"csil_paper_full", "mopc_hr_paper_full"}
    assert _load_plan(output)["authority_state"].endswith("SMOKE_REQUIRED")


def test_official_scale_plan_supports_single_method_and_paper_counts(tmp_path):
    artifacts = {}
    for name in ("checkpoint", "candidate", "adapter", "head", "tta"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    counts = (1, 3, 5, 10, 25)
    split = {
        "target_old_tx_labels": [f"o{i}" for i in range(6)],
        "nested_target_new_tx_labels": {
            str(count): [f"n{i}" for i in range(count)] for count in counts
        },
        "parity_reference_new20_tx_labels": [
            f"n{i}" for i in range(20)
        ],
        "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "confirmation_seeds": [713101, 713102, 713103, 713104, 713105],
        "k_values": [1, 5, 10, 20],
    }
    split_path = tmp_path / "split_scale.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "scale_plan.json"
    plan = build(
        argparse.Namespace(
            experiment_id="official_mopc_scale",
            methods="mopc_hr_official_repo",
            new_counts="1,3,5,10,25",
            required_total_capacity=31,
            expected_cache_scope="external_comparison_registered",
            cache_parity_root=str(tmp_path / "parity"),
            parity_reference_cache_root=str(tmp_path / "reference"),
            parity_preserved_class_labels=",".join(
                [f"n{i}" for i in range(20)]
            ),
            run_root=str(tmp_path / "run"),
            target_cache_root=str(tmp_path / "cache"),
            class_split=split_path,
            base_checkpoint=str(artifacts["checkpoint"]),
            candidate_lock=str(artifacts["candidate"]),
            adapter=str(artifacts["adapter"]),
            head_artifact=str(artifacts["head"]),
            tta_policy=str(artifacts["tta"]),
            smoke_receipt=None,
            output=output,
        )
    )
    assert plan["methods"] == ["mopc_hr_official_repo"]
    assert plan["required_total_capacity"] == 31
    assert plan["expected_cache_scope"] == "external_comparison_registered"
    assert plan["new_class_counts"] == [1, 3, 5, 10, 25]
    assert plan["counts"] == {
        "packages": 125,
        "cells": 500,
        "scenario_rows": 1500,
    }
    assert plan["smoke_cell_ids"] == [
        "rx_20_1__seed_713101__new_1__mopc_hr_official_repo__k_1",
        "rx_20_1__seed_713101__new_25__mopc_hr_official_repo__k_20",
    ]
    assert plan["official_code_execution_lock"][
        "official_zero_step_due_to_drop_last_preserved"
    ] is True


def test_official_scale_plan_rejects_duplicate_method(tmp_path):
    artifacts = {}
    for name in ("checkpoint", "candidate", "adapter", "head", "tta"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "target_old_tx_labels": [f"o{i}" for i in range(6)],
                "nested_target_new_tx_labels": {"1": ["n0"]},
                "target_receiver_labels": [
                    "20-1",
                    "3-19",
                    "7-14",
                    "7-7",
                    "8-8",
                ],
                "confirmation_seeds": [
                    713101,
                    713102,
                    713103,
                    713104,
                    713105,
                ],
                "k_values": [1, 5, 10, 20],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        build(
            argparse.Namespace(
                experiment_id="duplicate_method",
                methods="csil_official_repo,csil_official_repo",
                new_counts="1",
                run_root=str(tmp_path / "run"),
                target_cache_root=str(tmp_path / "cache"),
                class_split=split_path,
                base_checkpoint=str(artifacts["checkpoint"]),
                candidate_lock=str(artifacts["candidate"]),
                adapter=str(artifacts["adapter"]),
                head_artifact=str(artifacts["head"]),
                tta_policy=str(artifacts["tta"]),
                smoke_receipt=None,
                output=tmp_path / "plan.json",
            )
        )


def test_v2_adapter_matrix_is_fail_closed_in_builder_and_runner(tmp_path):
    sequential = "mopc_hr_official_repo_sequential5_cvs_adapter"
    with pytest.raises(ValueError, match="divisible by five"):
        validate_adapter_release_matrix((sequential,), (3,))
    validate_adapter_release_matrix(
        (
            "mopc_hr_official_repo_cvs_adapter",
            sequential,
        ),
        (25,),
    )
    validate_adapter_release_matrix(
        ("csil_official_repo_corefix_cvs_adapter",),
        (1, 3),
    )
    validate_adapter_required_capacity(
        ("csil_official_repo_corefix_cvs_adapter",), 26
    )
    validate_adapter_required_capacity(
        ("mopc_hr_official_repo_cvs_adapter", sequential), 31
    )
    with pytest.raises(ValueError, match="base capacity 26"):
        validate_adapter_required_capacity(
            ("csil_official_repo_corefix_cvs_adapter",), 9
        )
    with pytest.raises(ValueError, match="base capacity 31"):
        validate_adapter_required_capacity((sequential,), 25)
    with pytest.raises(ValueError, match="cannot mix"):
        validate_adapter_release_matrix(
            ("csil_official_repo", "csil_official_repo_corefix_cvs_adapter"),
            (1, 3),
        )
    invalid_plan = tmp_path / "invalid_sequential_plan.json"
    invalid_plan.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.adv3b02_paper_full_ci_plan.v1",
                "methods": [sequential],
                "new_class_counts": [3],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="divisible by five"):
        _load_plan(invalid_plan)


def test_adapter_predictor_receipt_semantics_are_machine_distinct():
    strict = _method_receipt_semantics("csil_official_repo")
    adapter = _method_receipt_semantics(
        "csil_official_repo_corefix_cvs_adapter"
    )
    sequential = _method_receipt_semantics(
        "mopc_hr_official_repo_sequential5_cvs_adapter"
    )
    assert strict[1] == "FORMAL_COMPARISON_BASELINE"
    assert adapter[1] == "FORMAL_COMPARISON_INTERFACE_ADAPTER"
    assert sequential[1] == "ORDERED_ARRIVAL_DIAGNOSTIC"
    assert adapter[0].endswith("predictor_receipt.v2")
    assert sequential[2].startswith("ORDERED_ARRIVAL_DIAGNOSTIC")
    assert _expected_method_status("csil_official_repo") == strict[1]
    assert (
        _expected_method_status("csil_official_repo_corefix_cvs_adapter")
        == adapter[1]
    )
    assert (
        _expected_method_status(
            "mopc_hr_official_repo_sequential5_cvs_adapter"
        )
        == sequential[1]
    )


def test_plan_runner_imports_from_outside_repository_cwd(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction"
        / "scripts"
        / "run_adv3b02_paper_full_ci_plan.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--plan" in completed.stdout


def test_smoke_receipt_binds_plan_contract_artifacts_and_predictor(tmp_path):
    artifacts = {}
    for name in ("checkpoint", "candidate", "adapter", "head", "tta"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "target_old_tx_labels": [f"o{i}" for i in range(6)],
                "nested_target_new_tx_labels": {"1": ["n0"]},
                "target_receiver_labels": [
                    "20-1",
                    "3-19",
                    "7-14",
                    "7-7",
                    "8-8",
                ],
                "confirmation_seeds": [
                    713101,
                    713102,
                    713103,
                    713104,
                    713105,
                ],
                "k_values": [1, 5, 10, 20],
            }
        ),
        encoding="utf-8",
    )
    common = {
        "experiment_id": "smoke_binding",
        "methods": "csil_official_repo",
        "new_counts": "1",
        "required_total_capacity": 26,
        "expected_cache_scope": "stage2_registered",
        "cache_parity_root": None,
        "run_root": str(tmp_path / "run"),
        "target_cache_root": str(tmp_path / "cache"),
        "class_split": split_path,
        "base_checkpoint": str(artifacts["checkpoint"]),
        "candidate_lock": str(artifacts["candidate"]),
        "adapter": str(artifacts["adapter"]),
        "head_artifact": str(artifacts["head"]),
        "tta_policy": str(artifacts["tta"]),
    }
    pre_path = tmp_path / "pre_plan.json"
    pre = build(
        argparse.Namespace(
            **common, smoke_receipt=None, output=pre_path
        )
    )
    predictor = (
        Path(__file__).resolve().parents[1]
        / pre["predictor_script"]
    )
    receipt_path = tmp_path / "smoke.json"
    receipt = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1",
        "status": "PASS",
        "completed_cell_ids": pre["smoke_cell_ids"],
        "executed_plan_path": str(pre_path.resolve()),
        "executed_plan_sha256": sha256_file(pre_path),
        "plan_contract_sha256": pre["plan_contract_sha256"],
        "artifact_sha256": {
            key: value["sha256"] for key, value in pre["artifacts"].items()
        },
        "predictor_script_sha256": sha256_file(predictor),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    formal = build(
        argparse.Namespace(
            **common,
            smoke_receipt=receipt_path,
            output=tmp_path / "formal_plan.json",
        )
    )
    assert formal["launch_authority"] is True
    receipt["artifact_sha256"]["head_artifact"] = "0" * 64
    bad_receipt = tmp_path / "bad_smoke.json"
    bad_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        build(
            argparse.Namespace(
                **common,
                smoke_receipt=bad_receipt,
                output=tmp_path / "rejected_plan.json",
            )
        )


def test_predictor_source_has_no_truth_or_channel_resampling_surface():
    source = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py"
    ).read_text(encoding="utf-8")
    assert "query_y" not in source
    assert "query_truth" not in source
    assert "apply_leo" not in source
    assert "satellite_channel" not in source
    assert "query_rows_used_for_training\": 0" in source
    assert "query_members_opened_before_model_lock\": False" in source


def test_comparison_bundle_relaxes_only_set_level_protocol_and_keeps_leo_check():
    source = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/build_adv3b02_paper_full_ci_bundle.py"
    ).read_text(encoding="utf-8")
    assert "load_comparison_inner_leo_cache(" in source
    assert "new_class_leo_iq_verified" in source
    assert "load_verified_leo_weak_cache_set =" in source
    assert "load_verified_stage2_predictor_bundle = (" in source
    assert "_assert_scenario_alignment = _comparison_reference_arrays" in source
    assert "_assert_scenario_physical_independence = lambda" in source
    assert "stage2_main_method_protocol_exempt_new_class_leo_required" in source


def _write_legacy_comparison_cache(path: Path, scenario: str) -> None:
    iq = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    sample_ids = np.asarray([f"legacy-sample-{index}" for index in range(4)])
    roles = np.asarray(["target_old", "target_old", "target_new", "target_new"])
    seeds = np.asarray([11, 12, 13, 14], dtype=np.int64)
    channel_hash = canonical_json_sha256({"scenario": scenario})
    iq_hashes = [post_channel_iq_sha256(row) for row in iq]
    overlays = [
        overlay_id(
            sample_id=sample_ids[index],
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=channel_hash,
            iq_sha256=iq_hashes[index],
        )
        for index in range(4)
    ]
    manifest = {
        "schema": "cvs_leo_weak_iq_cache_v1",
        "artifact_stage": "phase1_offline_prechannel_export",
        "contains_post_channel_iq_only": True,
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "target_channel_scenarios": [scenario],
        "scenario": scenario,
        "iq_array_key": "leo_weak_iq",
        "output_roles": ["target_old", "target_new"],
        "row_count": 4,
        "channel_config_sha256": channel_hash,
        "physical_sample_ids_sha256": ids_sha256(sample_ids.tolist()),
        "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
        "overlay_ids_sha256": ids_sha256(overlays),
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
    }
    np.savez_compressed(
        path,
        leo_weak_iq=iq,
        raw_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
        domain_labels=np.zeros(4, dtype=np.int64),
        tx_ids=np.asarray(["a", "b", "c", "d"]),
        rx_ids=np.asarray(["r"] * 4),
        day_ids=np.asarray(["d"] * 4),
        eq_ids=np.asarray(["e"] * 4),
        sig_ids=np.asarray(["s"] * 4),
        dataset_role=roles,
        channel_views=np.asarray(["rx_base"] * 4),
        sat_scenarios=np.asarray([scenario] * 4),
        satellite_seeds=seeds,
        overlay_applied=np.ones(4, dtype=bool),
        sample_ids=sample_ids,
        post_channel_iq_sha256=np.asarray(iq_hashes),
        overlay_ids=np.asarray(overlays),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def test_comparison_inner_loader_verifies_legacy_leo_without_source_indices(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    arrays, manifest, audit = load_comparison_inner_leo_cache(
        path,
        expected_scenario="leo_clear_weak",
        allowed_roles={"target_old", "target_new"},
    )
    assert "source_dataset_sha256" not in arrays
    assert "source_record_indices" not in arrays
    assert manifest["schema"] == "cvs_leo_weak_iq_cache_v1"
    assert audit["new_class_leo_iq_verified"] is True
    assert audit["exact_legacy_member_set_verified"] is True


def test_comparison_inner_loader_accepts_current_source_lineage_members(tmp_path):
    path = tmp_path / "current.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    manifest = json.loads(str(payload["manifest_json"].reshape(-1)[0]))
    manifest["sample_overlay_provenance_fields"] = [
        "sample_ids",
        "source_dataset_sha256",
        "source_record_indices",
        "sat_scenarios",
        "satellite_seeds",
        "post_channel_iq_sha256",
        "overlay_ids",
    ]
    dataset_hashes = np.asarray(["a" * 64] * 4)
    record_indices = np.arange(4, dtype=np.int64)
    payload["source_dataset_sha256"] = dataset_hashes
    payload["source_record_indices"] = record_indices
    sample_ids = np.asarray(
        [
            physical_sample_id_from_values(
                dataset_sha256=str(dataset_hashes[index]),
                source_record_index=int(record_indices[index]),
                tx_id=str(payload["tx_ids"][index]),
                rx_id=str(payload["rx_ids"][index]),
                day_id=str(payload["day_ids"][index]),
                eq_id=str(payload["eq_ids"][index]),
                sig_id=str(payload["sig_ids"][index]),
                role=str(payload["dataset_role"][index]),
            )
            for index in range(4)
        ]
    )
    payload["sample_ids"] = sample_ids
    scenario = "leo_clear_weak"
    channel_hash = str(manifest["channel_config_sha256"])
    iq_hashes = np.asarray(payload["post_channel_iq_sha256"]).astype(str)
    overlays = np.asarray(
        [
            overlay_id(
                sample_id=str(sample_ids[index]),
                scenario=scenario,
                satellite_seed=int(payload["satellite_seeds"][index]),
                channel_config_sha256=channel_hash,
                iq_sha256=str(iq_hashes[index]),
            )
            for index in range(4)
        ]
    )
    payload["overlay_ids"] = overlays
    manifest["physical_sample_ids_sha256"] = ids_sha256(sample_ids.tolist())
    manifest["overlay_ids_sha256"] = ids_sha256(overlays.tolist())
    payload["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(path, **payload)
    _arrays, _manifest, audit = load_comparison_inner_leo_cache(
        path,
        expected_scenario="leo_clear_weak",
        allowed_roles={"target_old", "target_new"},
    )
    assert audit["exact_legacy_member_set_verified"] is False
    assert audit["current_source_lineage_members_verified"] is True


def test_comparison_inner_loader_rejects_partial_current_source_lineage(tmp_path):
    path = tmp_path / "partial_current.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["source_dataset_sha256"] = np.asarray(["a" * 64] * 4)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="both absent or both present"):
        load_comparison_inner_leo_cache(
            path,
            expected_scenario="leo_clear_weak",
            allowed_roles={"target_old", "target_new"},
        )


def test_comparison_inner_loader_rejects_lineage_manifest_member_disagreement(
    tmp_path,
):
    path = tmp_path / "lineage_disagreement.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["source_dataset_sha256"] = np.asarray(["a" * 64] * 4)
    payload["source_record_indices"] = np.arange(4, dtype=np.int64)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="provenance fields and source-lineage"):
        load_comparison_inner_leo_cache(
            path,
            expected_scenario="leo_clear_weak",
            allowed_roles={"target_old", "target_new"},
        )


def test_external_comparison_cache_scope_accepts_target_roles_and_day0():
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    spec = {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v1",
        "cache_scope": "external_comparison_registered",
        "phase2_sample_view_policy": (
            "target_old_received_iq_target_new_leo_weak"
        ),
        "clean_sample_access": True,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": [
            {
                "role": "target_old",
                "pkl": "ManyTx.pkl",
                "tx_ids": "old",
                "rxs": "20-1",
                "days": "0",
                "max_samples_per_tx": 50,
                "apply_leo_overlay": False,
            },
            {
                "role": "target_new",
                "pkl": "ManyTx.pkl",
                "tx_ids": "new",
                "rxs": "20-1",
                "days": "0",
                "max_samples_per_tx": 50,
                "apply_leo_overlay": True,
            },
        ],
        "satellite_seed_by_scenario": {
            scenario: 1 + index for index, scenario in enumerate(scenarios)
        },
        "out_npz_by_scenario": {
            scenario: f"{scenario}.npz" for scenario in scenarios
        },
        "out_manifest": "cache_set.json",
    }
    assert validate_build_spec(spec)["cache_scope"] == (
        "external_comparison_registered"
    )


def test_scale_cache_specs_cover_25_receiver_seed_cells(tmp_path):
    manifest = build_scale_cache_specs(
        argparse.Namespace(
            experiment_id="official_scale_test",
            manytx_pkl="/dataset/ManyTx.pkl",
            remote_plan_root="/remote/plan",
            remote_cache_root="/remote/cache",
            reference_cache_root="/remote/reference",
            remote_parity_root="/remote/parity",
            output_dir=tmp_path,
        )
    )
    assert len(manifest["specs"]) == 25
    assert len(manifest["commands"]) == 25
    assert len(manifest["parity_commands"]) == 25
    assert len(manifest["new25_class_labels"]) == 25
    assert manifest["new25_class_labels"][-5:] == [
        "13-19",
        "18-14",
        "20-4",
        "20-16",
        "11-10",
    ]
    spec = json.loads(
        (tmp_path / manifest["specs"][0]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    overlay_by_role = {
        item["role"]: item["apply_leo_overlay"]
        for item in spec["role_specs"]
    }
    assert overlay_by_role == {"target_old": False, "target_new": True}
    assert spec["clean_sample_access"] is True
    assert manifest["parity_commands"][0][
        manifest["parity_commands"][0].index("--preserved-class-labels") + 1
    ].split(",") == manifest["new25_class_labels"][:20]
    assert (tmp_path / "cache_specs_manifest.json").is_file()


def test_scale_cache_reuse_manifest_has_no_cache_build_commands(tmp_path):
    output = tmp_path / "reuse.json"
    manifest = build_scale_cache_reuse_manifest(
        argparse.Namespace(
            experiment_id="official_scale_v7",
            source_experiment_id="official_scale_v6",
            reused_cache_root="/remote/v6/cache",
            remote_integrity_root="/remote/v7/integrity",
            output=output,
        )
    )
    assert manifest["reuse_policy"] == "READ_ONLY_NO_CACHE_REBUILD"
    assert len(manifest["entries"]) == 25
    assert len(manifest["integrity_commands"]) == 25
    assert "commands" not in manifest
    first = manifest["integrity_commands"][0]
    assert first[first.index("--mode") + 1] == "same_cache_new20_integrity"
    assert first[first.index("--reference-cache-set") + 1] == first[
        first.index("--expanded-cache-set") + 1
    ]


def test_comparison_inner_loader_rejects_post_channel_iq_tamper(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_legacy_comparison_cache(path, "leo_clear_weak")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["leo_weak_iq"] = payload["leo_weak_iq"].copy()
    payload["leo_weak_iq"][2, 0, 0] += 1.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="IQ digest mismatch"):
        load_comparison_inner_leo_cache(
            path,
            expected_scenario="leo_clear_weak",
            allowed_roles={"target_old", "target_new"},
        )


def test_comparison_set_loader_verifies_outer_hash_and_preserves_ids(tmp_path):
    scenario_paths = {}
    scenario_hashes = {}
    for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        cache = tmp_path / f"{scenario}.npz"
        _write_legacy_comparison_cache(cache, scenario)
        scenario_paths[scenario] = cache.name
        scenario_hashes[scenario] = sha256_file(cache)
    manifest_path = tmp_path / "set.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_v1",
                "cache_scope": "stage2_registered",
                "cache_npz_by_scenario": scenario_paths,
                "cache_sha256_by_scenario": scenario_hashes,
            }
        ),
        encoding="utf-8",
    )
    arrays, _manifest, audit = load_comparison_leo_cache_set(
        manifest_path,
        expected_scope="stage2_registered",
        allowed_roles={"target_old", "target_new"},
    )
    assert audit["status"] == "PASS_COMPARISON_SCOPE"
    assert {
        str(arrays[scenario]["sample_ids"][0]) for scenario in arrays
    } == {"legacy-sample-0"}
    assert all(
        audit["scenario_audits"][scenario][
            "verified_sample_ids_preserved_for_scenario_alignment"
        ]
        for scenario in arrays
    )
    assert _comparison_reference_arrays(arrays) is arrays["leo_clear_weak"]


def test_comparison_set_loader_rejects_scope_mismatch(tmp_path):
    scenario_paths = {}
    scenario_hashes = {}
    for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        cache = tmp_path / f"{scenario}.npz"
        _write_legacy_comparison_cache(cache, scenario)
        scenario_paths[scenario] = cache.name
        scenario_hashes[scenario] = sha256_file(cache)
    manifest_path = tmp_path / "set.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cvs_leo_weak_iq_cache_set_v1",
                "cache_scope": "stage2_registered",
                "cache_npz_by_scenario": scenario_paths,
                "cache_sha256_by_scenario": scenario_hashes,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache scope drift"):
        load_comparison_leo_cache_set(
            manifest_path,
            expected_scope="external_comparison_registered",
            allowed_roles={"target_old", "target_new"},
        )


def test_official_base_state_rejects_wrong_total_capacity(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    head = package / "head.pt"
    torch.save(
        {
            "schema": "cvs.adv3b02.official_repo_base_state.v2",
            "total_capacity": 26,
            "base_sample_count": 8400,
            "csil_base_train_sample_count": 5879,
            "fisher_sample_count": 2521,
            "source_train_fisher_disjoint": True,
            "csil": {},
            "mopc_hr": {
                "classifier_weight": torch.zeros(26, 160),
                "classifier_bias": torch.zeros(26),
            },
        },
        head,
    )
    manifest = {
        "members": [
            {"artifact_role": "head", "relative_path": "head.pt"}
        ]
    }
    with pytest.raises(ValueError, match="total capacity drift"):
        _load_base_state(
            package,
            manifest,
            device=torch.device("cpu"),
            old_count=6,
            expected_total_capacity=31,
        )


def test_systemic_health_gate_stops_after_two_distinct_matching_failures(
    tmp_path,
):
    plan = {"run_root": str(tmp_path / "run")}
    first = _update_health_state(
        plan, row_id="row-a", exc=ValueError("same fault 100"), prediction_produced=False
    )
    assert first["stop_dispatch"] is False
    second = _update_health_state(
        plan, row_id="row-b", exc=ValueError("same fault 200"), prediction_produced=False
    )
    assert second["stop_dispatch"] is True
    assert second["stop_reason"].startswith("TWO_DISTINCT_ROWS")


def test_systemic_health_gate_stops_immediately_on_p0(tmp_path):
    plan = {"run_root": str(tmp_path / "run")}
    state = _update_health_state(
        plan,
        row_id="row-a",
        exc=ValueError("query truth protocol violation"),
        prediction_produced=False,
    )
    assert state["stop_dispatch"] is True
    assert state["stop_reason"] == "P0_PROTOCOL_OR_SAFETY_VIOLATION"


def test_truth_free_script_name_does_not_misclassify_execution_fault_as_p0(
    tmp_path,
):
    plan = {"run_root": str(tmp_path / "run")}
    state = _update_health_state(
        plan,
        row_id="row-a",
        exc=RuntimeError(
            "command failed: run_truth_free_predictor.py\n"
            "RuntimeError: CUDA out of memory"
        ),
        prediction_produced=False,
    )
    assert state["stop_dispatch"] is False


def test_manual_formal_authority_flip_cannot_bypass_smoke(tmp_path):
    plan = {
        "launch_authority": True,
        "authority_state": "N607_PAPER_FULL_CI_SMOKE_PASS",
        "smoke_receipt_path": None,
    }
    with pytest.raises(ValueError, match="misses smoke receipt path"):
        _verify_smoke_authority(plan, project_root=tmp_path)


def test_scale_cache_parity_gate_detects_post_channel_hash_drift(
    tmp_path, monkeypatch
):
    labels = [f"tx-{index}" for index in range(20)]
    rows = {
        scenario: {
            "tx_ids": np.asarray(labels),
            "sample_ids": np.asarray([f"sample-{index}" for index in range(20)]),
            "post_channel_iq_sha256": np.asarray(
                [f"{index:064x}" for index in range(20)]
            ),
        }
        for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    }
    expanded = {
        scenario: {key: value.copy() for key, value in arrays.items()}
        for scenario, arrays in rows.items()
    }
    expanded["leo_rain_weak"]["post_channel_iq_sha256"][3] = "f" * 64
    reference_path = tmp_path / "reference.json"
    expanded_path = tmp_path / "expanded.json"
    reference_path.write_text("{}", encoding="utf-8")
    expanded_path.write_text("{}", encoding="utf-8")

    def fake_loader(path, *, expected_scope, allowed_roles):
        selected = rows if Path(path) == reference_path else expanded
        return selected, {}, {}

    monkeypatch.setattr(
        "paper_reproduction.scripts.verify_adv3b02_official_scale_cache_parity."
        "load_comparison_leo_cache_set",
        fake_loader,
    )
    with pytest.raises(ValueError, match="parity mismatch"):
        verify_scale_cache_parity(
            argparse.Namespace(
                reference_cache_set=reference_path,
                expanded_cache_set=expanded_path,
                preserved_class_labels=",".join(labels),
                reference_scope="stage2_registered",
                expanded_scope="external_comparison_registered",
                output=tmp_path / "receipt.json",
            )
        )


def test_same_cache_integrity_rejects_distinct_resolved_paths(tmp_path):
    reference_path = tmp_path / "reference.json"
    expanded_path = tmp_path / "expanded.json"
    reference_path.write_text("{}", encoding="utf-8")
    expanded_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identical resolved cache paths"):
        verify_scale_cache_parity(
            argparse.Namespace(
                reference_cache_set=reference_path,
                expanded_cache_set=expanded_path,
                preserved_class_labels=",".join(
                    f"class-{index}" for index in range(20)
                ),
                reference_scope="external_comparison_registered",
                expanded_scope="external_comparison_registered",
                mode="same_cache_new20_integrity",
                output=tmp_path / "receipt.json",
            )
        )


def test_historical_parity_rejects_identical_resolved_paths(tmp_path):
    cache_set = tmp_path / "cache_set.json"
    cache_set.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct resolved cache paths"):
        verify_scale_cache_parity(
            argparse.Namespace(
                reference_cache_set=cache_set,
                expanded_cache_set=cache_set,
                preserved_class_labels=",".join(
                    f"class-{index}" for index in range(20)
                ),
                reference_scope="stage2_registered",
                expanded_scope="external_comparison_registered",
                mode="historical_reference",
                output=tmp_path / "receipt.json",
            )
        )


def test_package_rejects_unknown_cache_verification_mode(tmp_path):
    cache_set = tmp_path / "cache_set.json"
    cache_set.write_text("cache", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported cache verification mode"):
        _verify_cache_parity_receipt(
            {
                "expected_cache_scope": "external_comparison_registered",
                "cache_verification_mode": "unknown_mode",
                "parity_preserved_class_labels": [
                    f"class-{index}" for index in range(20)
                ],
            },
            {
                "cache_parity_receipt": str(cache_set),
                "target_cache_set": str(cache_set),
                "cache_parity_reference_cache_set": str(cache_set),
            },
        )


def test_package_accepts_same_cache_new20_integrity_receipt(tmp_path):
    cache_set = tmp_path / "cache_set.json"
    cache_set.write_text("cache", encoding="utf-8")
    expected_labels = [f"class-{index}" for index in range(20)]
    receipt_path = tmp_path / "integrity.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cvs.adv3b02.same_cache_new20_integrity_receipt.v1",
                "verification_mode": "same_cache_new20_integrity",
                "status": "PASS",
                "reference_cache_set": str(cache_set.resolve()),
                "reference_cache_set_sha256": sha256_file(cache_set),
                "expanded_cache_set": str(cache_set.resolve()),
                "expanded_cache_set_sha256": sha256_file(cache_set),
                "preserved_class_labels": expected_labels,
                "verified_fields": [
                    "tx_ids",
                    "sample_ids",
                    "post_channel_iq_sha256",
                ],
                "scenario_receipts": {
                    scenario: {
                        "row_count": 1000,
                        "sample_ids_sha256": "a" * 64,
                        "post_channel_iq_sha256_root": "b" * 64,
                    }
                    for scenario in (
                        "leo_clear_weak",
                        "leo_low_elev_weak",
                        "leo_rain_weak",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = _verify_cache_parity_receipt(
        {
            "expected_cache_scope": "external_comparison_registered",
            "cache_verification_mode": "same_cache_new20_integrity",
            "parity_preserved_class_labels": expected_labels,
        },
        {
            "cache_parity_receipt": str(receipt_path),
            "target_cache_set": str(cache_set),
            "cache_parity_reference_cache_set": str(cache_set),
        },
    )
    assert receipt is not None and receipt["status"] == "PASS"


def test_package_parity_receipt_rejects_wrong_preserved_labels(tmp_path):
    reference = tmp_path / "reference.json"
    expanded = tmp_path / "expanded.json"
    reference.write_text("reference", encoding="utf-8")
    expanded.write_text("expanded", encoding="utf-8")
    expected_labels = [f"class-{index}" for index in range(26)]
    receipt_path = tmp_path / "parity.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cvs.adv3b02.official_scale_cache_parity_receipt.v1",
                "status": "PASS",
                "reference_cache_set": str(reference.resolve()),
                "reference_cache_set_sha256": sha256_file(reference),
                "expanded_cache_set": str(expanded.resolve()),
                "expanded_cache_set_sha256": sha256_file(expanded),
                "preserved_class_labels": list(reversed(expected_labels)),
                "verified_fields": [
                    "tx_ids",
                    "sample_ids",
                    "post_channel_iq_sha256",
                ],
                "scenario_receipts": {
                    scenario: {
                        "row_count": 1300,
                        "sample_ids_sha256": "a" * 64,
                        "post_channel_iq_sha256_root": "b" * 64,
                    }
                    for scenario in (
                        "leo_clear_weak",
                        "leo_low_elev_weak",
                        "leo_rain_weak",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not authorize"):
        _verify_cache_parity_receipt(
            {
                "expected_cache_scope": "external_comparison_registered",
                "parity_preserved_class_labels": expected_labels,
            },
            {
                "cache_parity_receipt": str(receipt_path),
                "target_cache_set": str(expanded),
                "cache_parity_reference_cache_set": str(reference),
            },
        )


def test_package_parity_receipt_accepts_new20_row_count(tmp_path):
    reference = tmp_path / "reference.json"
    expanded = tmp_path / "expanded.json"
    reference.write_text("reference", encoding="utf-8")
    expanded.write_text("expanded", encoding="utf-8")
    expected_labels = [f"class-{index}" for index in range(20)]
    receipt_path = tmp_path / "parity.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cvs.adv3b02.official_scale_cache_parity_receipt.v1",
                "status": "PASS",
                "reference_cache_set": str(reference.resolve()),
                "reference_cache_set_sha256": sha256_file(reference),
                "expanded_cache_set": str(expanded.resolve()),
                "expanded_cache_set_sha256": sha256_file(expanded),
                "preserved_class_labels": expected_labels,
                "verified_fields": [
                    "tx_ids",
                    "sample_ids",
                    "post_channel_iq_sha256",
                ],
                "scenario_receipts": {
                    scenario: {
                        "row_count": 1000,
                        "sample_ids_sha256": "a" * 64,
                        "post_channel_iq_sha256_root": "b" * 64,
                    }
                    for scenario in (
                        "leo_clear_weak",
                        "leo_low_elev_weak",
                        "leo_rain_weak",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = _verify_cache_parity_receipt(
        {
            "expected_cache_scope": "external_comparison_registered",
            "parity_preserved_class_labels": expected_labels,
        },
        {
            "cache_parity_receipt": str(receipt_path),
            "target_cache_set": str(expanded),
            "cache_parity_reference_cache_set": str(reference),
        },
    )
    assert receipt is not None and receipt["status"] == "PASS"


def test_comparison_final_bundle_validator_keeps_strict_per_scenario_checks(
    monkeypatch,
):
    calls = []

    def fake_strict(
        package_root,
        *,
        detached_seal_path,
        expected_seal_sha256,
        scenario=None,
    ):
        calls.append(scenario)
        manifest = {"package": str(package_root), "version": 1}
        audit = {
            "seal": {"sha256": expected_seal_sha256},
            "sample_level_post_channel_iq_sha256_status": "PASS",
        }
        return (
            {scenario: {"support": np.asarray([scenario])}},
            {scenario: {"query": np.asarray([scenario])}},
            manifest,
            audit,
        )

    monkeypatch.setattr(
        "paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle."
        "_strict_stage2_bundle_loader",
        fake_strict,
    )
    support, query, manifest, audit = (
        load_verified_comparison_stage2_predictor_bundle(
            "bundle",
            detached_seal_path="seal.json",
            expected_seal_sha256="a" * 64,
        )
    )
    assert calls == ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    assert tuple(support) == tuple(calls)
    assert tuple(query) == tuple(calls)
    assert manifest["version"] == 1
    assert audit["sample_level_post_channel_iq_sha256_status"] == "PASS"
    assert (
        audit["cross_scenario_physical_sample_token_disjointness"]
        == "EXEMPT_EXTERNAL_COMPARISON_BASELINE"
    )
