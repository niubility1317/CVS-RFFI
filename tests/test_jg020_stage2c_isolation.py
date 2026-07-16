from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from paper_reproduction.cvs_aligned.jg020_stage2c import (
    APPLY_PROFILE,
    ENROLLMENT_PROFILE,
    FORMAL_SCENARIOS,
    HEAD_SCHEMA,
    LOCK_SCHEMA,
    PHASE2_CONTRACT,
    RUNTIME_FIXED_BATCH_SIZE,
    JG020ProtocolError,
    apply_head_streams,
    build_head_state,
    head_npz_members,
    make_member_descriptor,
    numpy_from_torch_compat,
    ordered_label_sha256,
    preflight_package,
    prepare_preincrement_adaptation_support,
    sha256_file,
    torch_tensor_from_numpy_compat,
    validate_direct_class_mapping,
    validate_locked_candidate,
    write_package_manifest_and_seal,
)
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact
from paper_reproduction.scripts.launch_cvs_jg020_stage2c_dev_20260716 import (
    _parse_last_json_document,
    _prepare_run_root,
)
from paper_reproduction.scripts.run_cvs_jg020_apply_only_predictor import _forward


def _lock(new_count: int = 5) -> dict[str, object]:
    old = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    new = ["1-16", "1-18", "18-10", "14-11", "8-3"][:new_count]
    return {
        "schema": LOCK_SCHEMA,
        "candidate_id": "JG_R8_LR020",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "new_class_count": new_count,
        "scope": "joint_gate",
        "rank": 8,
        "alpha": 8.0,
        "learning_rate": 0.02,
        "weight_decay": 1.0e-4,
        "temperature": 18.0,
        "epochs": 5,
        "max_optimizer_steps": 50,
        "grad_clip": 1.0,
        "ground_adapter_scope": "projection_feature",
        "ground_adapter_rank": 16,
        "ground_adapter_alpha": 16.0,
        "ground_adapter_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "direct_class_mapping_sha256": "3" * 64,
        "old_class_order_sha256": ordered_label_sha256(old),
        "new_class_order_sha256": ordered_label_sha256(new),
        "support_view_count": 3,
        "query_view_count": 1,
        "adapter_alpha": 1.0,
        "trust_decision": "locked_k10_full_delta",
        "k1_trust_gate_enabled": False,
        "phase2_contract": PHASE2_CONTRACT,
    }


def _write_npz(path: Path, scenario: str) -> list[str]:
    with path.open("xb") as handle:
        np.savez(
            handle,
            support_pool_leo_weak_iq=np.zeros((1, 2, 8), dtype=np.float32),
            support_pool_class_indices=np.zeros(1, dtype=np.int64),
            support_pool_rank_within_class=np.zeros(1, dtype=np.int64),
            support_pool_tokens=np.asarray(["sid_" + "a" * 32]),
            support_pool_overlay_tokens=np.asarray(["oid_" + "b" * 32]),
            support_pool_satellite_seeds=np.asarray([713101], dtype=np.int64),
            support_pool_post_channel_iq_sha256=np.asarray(["c" * 64]),
            manifest_json=np.asarray(json.dumps({"scenario": scenario})),
        )
    with np.load(path, allow_pickle=False) as archive:
        return list(archive.files)


def _write_query_npz(path: Path, scenario: str) -> list[str]:
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=np.zeros((1, 2, 8), dtype=np.float32),
            query_tokens=np.asarray(["qid_" + "a" * 64]),
            query_overlay_tokens=np.asarray(["oid_" + "b" * 64]),
            query_satellite_seeds=np.asarray([713101], dtype=np.int64),
            query_post_channel_iq_sha256=np.asarray(["c" * 64]),
            manifest_json=np.asarray(json.dumps({"scenario": scenario})),
        )
    with np.load(path, allow_pickle=False) as archive:
        return list(archive.files)


def _write_head_npz(path: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        "class_handles": np.asarray([f"cls_{index:032x}" for index in range(11)]),
        "old_class_count": np.asarray(6, dtype=np.int64),
        "temperature": np.asarray(18.0, dtype=np.float32),
        "manifest_json": np.asarray(json.dumps({"schema": HEAD_SCHEMA})),
    }
    for scenario in FORMAL_SCENARIOS:
        arrays[f"candidate_prototypes__{scenario}"] = np.zeros((11, 4), dtype=np.float16)
        arrays[f"identity_prototypes__{scenario}"] = np.zeros((11, 4), dtype=np.float16)
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def test_candidate_lock_is_exact_and_preserves_unified_jg_r8() -> None:
    value = validate_locked_candidate(_lock())
    assert value["candidate_id"] == "JG_R8_LR020"
    assert value["scope"] == "joint_gate"
    assert value["adapter_alpha"] == 1.0
    assert value["k1_trust_gate_enabled"] is False
    drift = dict(value)
    drift["scope"] = "joint_projection"
    with pytest.raises(JG020ProtocolError, match="locked candidate drift"):
        validate_locked_candidate(drift)


def test_direct_mapping_must_bind_checkpoint_old_class_order() -> None:
    lock = _lock()
    mapping = {
        "class_id_to_tx": [
            "14-10", "14-7", "20-15", "20-19", "6-15", "8-20"
        ]
    }
    audit = validate_direct_class_mapping(mapping, lock=lock)
    assert audit["direct_logit_to_class_handle_order_bound"] is True
    mapping["class_id_to_tx"] = list(reversed(mapping["class_id_to_tx"]))
    with pytest.raises(JG020ProtocolError, match="old class order"):
        validate_direct_class_mapping(mapping, lock=lock)


def test_enrollment_package_role_set_physically_excludes_query_and_truth(tmp_path: Path) -> None:
    root = tmp_path / "enrollment"
    root.mkdir()
    lock = _lock()
    files = {
        "candidate_lock": root / "candidate_lock.json",
        "checkpoint_full": root / "checkpoint_full.pth",
        "ground_adapter": root / "ground_adapter.pt",
        "direct_class_mapping": root / "class_registry_map.json",
    }
    files["candidate_lock"].write_text(json.dumps(lock), encoding="utf-8")
    files["checkpoint_full"].write_bytes(b"checkpoint")
    files["ground_adapter"].write_bytes(b"ground")
    files["direct_class_mapping"].write_text(
        json.dumps({"class_id_to_tx": ["a", "b", "c", "d", "e", "f"]}),
        encoding="utf-8",
    )
    members = [
        make_member_descriptor(files["candidate_lock"], role="candidate_lock", schema=LOCK_SCHEMA),
        make_member_descriptor(files["checkpoint_full"], role="checkpoint_full", schema="checkpoint"),
        make_member_descriptor(files["ground_adapter"], role="ground_adapter", schema="adapter"),
        make_member_descriptor(
            files["direct_class_mapping"], role="direct_class_mapping", schema="mapping"
        ),
    ]
    for scenario in FORMAL_SCENARIOS:
        path = root / f"support_{scenario}.npz"
        npz_members = _write_npz(path, scenario)
        members.append(
            make_member_descriptor(
                path,
                role=f"support:{scenario}",
                schema="support",
                scenario=scenario,
                npz_members=npz_members,
            )
        )
    metadata = {
        "stage": "stage2c",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "new_class_count": 5,
        "registered_class_count": 11,
        "registered_classes": [
            {"class_index": index, "class_handle": f"cls_{index:032x}"}
            for index in range(11)
        ],
        "candidate_lock_sha256": sha256_file(files["candidate_lock"]),
        "target_channel_scenarios": list(FORMAL_SCENARIOS),
        "phase2_contract": PHASE2_CONTRACT,
        "lineage": {
            "source_package_root_sha256": "4" * 64,
            "source_package_seal_sha256": "5" * 64,
            "enrollment_package_root_sha256": None,
        },
    }
    seal = tmp_path / "enrollment.seal.json"
    document, _ = write_package_manifest_and_seal(
        root,
        profile=ENROLLMENT_PROFILE,
        metadata=metadata,
        members=members,
        detached_seal=seal,
    )
    verified, audit = preflight_package(
        root,
        detached_seal=seal,
        expected_seal_sha256=sha256_file(seal),
        expected_profile=ENROLLMENT_PROFILE,
    )
    roles = {item["artifact_role"] for item in verified["members"]}
    assert roles == {
        "candidate_lock", "checkpoint_full", "ground_adapter", "direct_class_mapping",
        *(f"support:{scenario}" for scenario in FORMAL_SCENARIOS),
    }
    assert audit["query_member_reachable"] is False
    assert audit["truth_member_reachable"] is False
    assert audit["clean_member_reachable"] is False
    assert document["package_root_sha256"] == audit["package_root_sha256"]


def test_apply_package_role_set_physically_excludes_support_and_truth(tmp_path: Path) -> None:
    root = tmp_path / "apply"
    root.mkdir()
    lock = _lock()
    files = {
        "candidate_lock": root / "candidate_lock.json",
        "candidate_runtime": root / "candidate_runtime.ts",
        "identity_runtime": root / "identity_runtime.ts",
        "direct_runtime": root / "direct_runtime.ts",
        "prototype_head": root / "prototype_head.npz",
        "enrollment_receipt": root / "enrollment_receipt.json",
    }
    files["candidate_lock"].write_text(json.dumps(lock), encoding="utf-8")
    for role in ("candidate_runtime", "identity_runtime", "direct_runtime"):
        files[role].write_bytes(role.encode("ascii"))
    _write_head_npz(files["prototype_head"])
    files["enrollment_receipt"].write_text("{}\n", encoding="utf-8")
    members = [
        make_member_descriptor(files["candidate_lock"], role="candidate_lock", schema=LOCK_SCHEMA),
        make_member_descriptor(files["candidate_runtime"], role="candidate_runtime", schema="runtime"),
        make_member_descriptor(files["identity_runtime"], role="identity_runtime", schema="runtime"),
        make_member_descriptor(files["direct_runtime"], role="direct_runtime", schema="runtime"),
        make_member_descriptor(
            files["prototype_head"],
            role="prototype_head",
            schema=HEAD_SCHEMA,
            npz_members=head_npz_members(),
        ),
        make_member_descriptor(
            files["enrollment_receipt"], role="enrollment_receipt", schema="receipt"
        ),
    ]
    for scenario in FORMAL_SCENARIOS:
        path = root / f"query_{scenario}.npz"
        members.append(
            make_member_descriptor(
                path,
                role=f"query:{scenario}",
                schema="query",
                scenario=scenario,
                npz_members=_write_query_npz(path, scenario),
            )
        )
    metadata = {
        "stage": "stage2c",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "new_class_count": 5,
        "registered_class_count": 11,
        "registered_classes": [
            {"class_index": index, "class_handle": f"cls_{index:032x}"}
            for index in range(11)
        ],
        "candidate_lock_sha256": sha256_file(files["candidate_lock"]),
        "target_channel_scenarios": list(FORMAL_SCENARIOS),
        "phase2_contract": PHASE2_CONTRACT,
        "lineage": {
            "source_package_root_sha256": "4" * 64,
            "source_package_seal_sha256": "5" * 64,
            "enrollment_package_root_sha256": "6" * 64,
        },
    }
    seal = tmp_path / "apply.seal.json"
    write_package_manifest_and_seal(
        root,
        profile=APPLY_PROFILE,
        metadata=metadata,
        members=members,
        detached_seal=seal,
    )
    verified, audit = preflight_package(
        root,
        detached_seal=seal,
        expected_seal_sha256=sha256_file(seal),
        expected_profile=APPLY_PROFILE,
    )
    roles = {item["artifact_role"] for item in verified["members"]}
    assert roles == {
        "candidate_lock", "candidate_runtime", "identity_runtime", "direct_runtime",
        "prototype_head", "enrollment_receipt",
        *(f"query:{scenario}" for scenario in FORMAL_SCENARIOS),
    }
    assert audit["query_member_reachable"] is True
    assert audit["support_member_reachable"] is False
    assert audit["truth_member_reachable"] is False
    assert audit["clean_member_reachable"] is False
    assert not any("truth" in path.name or "support" in path.name for path in root.iterdir())


def test_head_scores_every_query_against_all_registered_classes() -> None:
    handles = [f"cls_{index:032x}" for index in range(3)]
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    features = np.eye(3, dtype=np.float32)
    candidate = {scenario: features for scenario in FORMAL_SCENARIOS}
    identity = {scenario: features for scenario in FORMAL_SCENARIOS}
    head = build_head_state(
        class_handles=handles,
        old_class_count=2,
        candidate_features_by_scenario=candidate,
        identity_features_by_scenario=identity,
        support_labels=labels,
        temperature=18.0,
    )
    result = apply_head_streams(
        scenario="leo_clear_weak",
        candidate_features=features,
        identity_features=features,
        direct_logits=np.asarray([[9, 0], [0, 9], [4, 3]], dtype=np.float32),
        head=head,
    )
    assert result["candidate_after"].tolist() == handles
    assert set(result["candidate_before"].tolist()) <= set(handles[:2])
    assert result["identity_after"].tolist() == handles
    assert result["direct"].tolist() == [handles[0], handles[1], handles[0]]


def test_preincrement_optimizer_cache_and_episodes_exclude_registered_new_support() -> None:
    labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    tokens = np.asarray([f"sid_{index:064x}" for index in range(len(labels))])
    rows_by_scenario = {}
    for scenario_index, scenario in enumerate(FORMAL_SCENARIOS):
        rows = np.zeros((len(labels), 2, 8), dtype=np.float32)
        rows[:, 0, 0] = labels
        rows[:, 1, 0] = scenario_index
        rows_by_scenario[scenario] = rows
    rows, adapt_labels, row_ids, physical_ids, audit = prepare_preincrement_adaptation_support(
        rows_by_scenario,
        labels,
        tokens,
        old_class_count=6,
        k_shot=10,
    )
    assert rows.shape == (3 * 6 * 10, 2, 8)
    assert int(rows[:, 0, 0].max()) == 5
    assert set(adapt_labels.tolist()) == set(range(6))
    assert len(row_ids) == 3 * 6 * 10
    assert len(physical_ids) == 6 * 10
    assert not any(token in row_ids for token in tokens[6 * 10 :].tolist())
    assert audit["adapt_full_forward_row_count"] == 180
    assert audit["excluded_registered_new_support_count"] == 50
    assert audit["new_support_gradient_used"] is False
    assert audit["per_sample_old_new_role_branch_used"] is False


def test_cli_surfaces_do_not_expose_forbidden_cross_boundary_paths() -> None:
    repo = Path(__file__).resolve().parents[1]
    enrollment = (repo / "paper_reproduction/scripts/enroll_cvs_jg020_support_only.py").read_text(
        encoding="utf-8"
    )
    predictor = (repo / "paper_reproduction/scripts/run_cvs_jg020_apply_only_predictor.py").read_text(
        encoding="utf-8"
    )
    assert 'add_argument("--query' not in enrollment
    assert 'add_argument("--truth' not in enrollment
    assert "train_support_only_bp_jg_cached(" in enrollment
    assert "train_support_only_bp_jg(" not in enrollment
    assert "prepare_preincrement_adaptation_support(" in enrollment
    assert "new_support_gradient_used\": False" in enrollment
    assert "adapter_retrained_at_registration\": False" in enrollment
    assert 'add_argument("--support' not in predictor
    assert 'add_argument("--truth' not in predictor
    assert "build_prototypes(" not in predictor
    assert "build_head_state(" not in predictor


def test_real_adv3b02_p4_cached_path_parity_evidence_is_locked() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo
        / "automation_reports/CV-SincNet/qknnv42_k1_support_trust_adapt_20260716"
        / "cached_jg_real_parity.json"
    )
    assert sha256_file(path) == "d9cfcdab9d066e2f0888061ba814979200865f62811f6790441dd95ab65193b1"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["checkpoint_sha256"] == "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
    assert value["ground_p4_sha256"] == "95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446"
    assert value["checkpoint_strict_load"] is True
    assert value["target_trainable_parameters"] == 6_400
    assert value["probe_shape"] == [7, 2, 256]
    assert value["batch_size"] == 3
    assert value["full_backbone_call_count"] == 3
    assert value["max_absolute_error"] <= value["tolerance"] == 1.0e-6
    assert value["parity_pass"] is True
    assert value["query_rows_used"] == 0


def test_cvspred_publish_is_readonly_atomic_noreplace_and_apply_does_not_fit(tmp_path: Path) -> None:
    target = tmp_path / "predictions.cvspred"
    values = {
        "stage": "Stage2-C",
        "row_id": "JG_R8_LR020_rx20-1_seed713101_n5_k10",
        "receiver": "20-1",
        "k_shot": 10,
        "candidate_lock_sha256": "1" * 64,
        "package_root_sha256": "2" * 64,
        "package_seal_sha256": "3" * 64,
        "query_tokens": np.asarray(["qid_a", "qid_b"]),
        "scenarios": np.asarray(["leo_clear_weak", "leo_clear_weak"]),
        "candidate_after": np.asarray(["cls_a", "cls_b"]),
        "candidate_before": np.asarray(["cls_a", "cls_a"]),
        "identity_after": np.asarray(["cls_a", "cls_b"]),
        "identity_before": np.asarray(["cls_a", "cls_a"]),
        "direct": np.asarray(["cls_a", "cls_a"]),
        "shared_view_counts": np.ones(2, dtype=np.uint8),
    }
    result = publish_prediction_artifact(target, **values)
    assert result["readonly"] is True
    assert result["immutable_state"] == "SEALED_READ_ONLY_ATOMIC_NOREPLACE"
    with pytest.raises(FileExistsError):
        publish_prediction_artifact(target, **values)
    predictor = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/run_cvs_jg020_apply_only_predictor.py"
    ).read_text(encoding="utf-8")
    assert "prototype_fit_inside_predictor\": False" in predictor
    assert "build_head_state(" not in predictor
    assert "build_prototypes(" not in predictor


def test_launcher_parses_pretty_json_after_warning_and_resume_is_cache_only(tmp_path: Path) -> None:
    stdout = "warning before result\n{\n  \"status\": \"PASS\",\n  \"nested\": {\"rows\": 1040}\n}\n"
    assert _parse_last_json_document(stdout) == {
        "status": "PASS",
        "nested": {"rows": 1040},
    }
    with pytest.raises(ValueError, match="without a final JSON object"):
        _parse_last_json_document("warning only\n")

    run_root = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _prepare_run_root(run_root, resume=True)
    assert _prepare_run_root(run_root, resume=False) is False
    (run_root / "phase1_cache").mkdir()
    (run_root / "phase1_cache/cache_set.json").write_text("{}\n", encoding="utf-8")
    assert _prepare_run_root(run_root, resume=True) is True
    (run_root / "new_5").mkdir()
    with pytest.raises(FileExistsError, match="partially materialised"):
        _prepare_run_root(run_root, resume=True)


def test_support_only_enrollment_cli_import_closure() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "paper_reproduction/scripts/enroll_cvs_jg020_support_only.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "--package-root" in result.stdout
    assert "--query" not in result.stdout

    source = script.read_text(encoding="utf-8")
    assert "support_rows" not in source
    assert source.count("input_len=int(adapt_rows.shape[-1])") == 3


def test_numpy_torch_abi_bridge_uses_explicit_small_copy() -> None:
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    tensor = torch_tensor_from_numpy_compat(
        source,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    restored = numpy_from_torch_compat(tensor, dtype=np.dtype(np.float32))
    np.testing.assert_array_equal(restored, source)

    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "paper_reproduction/cvs_aligned/jg020_stage2c.py",
        root / "paper_reproduction/scripts/enroll_cvs_jg020_support_only.py",
        root / "paper_reproduction/scripts/run_cvs_jg020_apply_only_predictor.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "torch.from_numpy(" not in combined
    assert ".cpu().numpy()" not in combined


def test_traced_runtime_uses_locked_two_row_microbatches() -> None:
    class BatchConstantRuntime(torch.nn.Module):
        def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = int(rows.shape[0])
            flat = rows.reshape(batch, -1)
            return flat, flat[:, :2]

    example = torch.zeros((RUNTIME_FIXED_BATCH_SIZE, 2, 3), dtype=torch.float32)
    runtime = torch.jit.trace(BatchConstantRuntime(), example, strict=False)
    rows = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    features, logits = _forward(
        runtime,
        rows,
        device=torch.device("cpu"),
        batch_size=RUNTIME_FIXED_BATCH_SIZE,
    )
    assert features.shape == (4, 6)
    assert logits.shape == (4, 2)
    with pytest.raises(ValueError, match="locked fixed batch size"):
        _forward(runtime, rows, device=torch.device("cpu"), batch_size=4)
    with pytest.raises(ValueError, match="divisible"):
        _forward(runtime, rows[:3], device=torch.device("cpu"), batch_size=RUNTIME_FIXED_BATCH_SIZE)
