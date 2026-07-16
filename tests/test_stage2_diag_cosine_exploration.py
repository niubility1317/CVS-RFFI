from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import torch

import cvsrffi.stage2_diag_cosine_exploration as route
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from scripts import run_cvs_stage2_diag_cosine_exploration as runner


def _separable(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 24, dtype=np.float32)
    support = np.vstack(
        [centers[index] + 0.01 * rng.normal(size=(9, 24)) for index in range(3)]
    ).astype(np.float32)
    query = np.vstack(
        [centers[index] + 0.01 * rng.normal(size=(5, 24)) for index in range(3)]
    ).astype(np.float32)
    labels = np.repeat(["class-a", "class-b", "class-c"], 9)
    truth = np.repeat(["class-a", "class-b", "class-c"], 5)
    return support, labels, query, truth


def _d3_support(*, include_new: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    class_names = ["old-a", "old-b"] + (
        ["new-c", "new-d"] if include_new else []
    )
    for scenario_index, scenario in enumerate(route.FORMAL_LEO_WEAK_SCENARIOS):
        rng = np.random.default_rng(101 + scenario_index)
        centers = np.eye(len(class_names), 24, dtype=np.float32)
        features = np.vstack(
            [
                centers[index] + 0.01 * rng.normal(size=(4, 24))
                for index in range(len(class_names))
            ]
        ).astype(np.float32)
        labels = np.repeat(class_names, 4)
        result[scenario] = (features, labels)
    return result


def test_fit_has_no_query_argument_and_prediction_is_batch_extension_invariant():
    support, labels, query, truth = _separable()
    state = route.fit_diag_cosine_state(
        support, labels, seed=19, device=torch.device("cpu")
    )
    first = route.predict_diag_cosine(state, query)
    extended = route.predict_diag_cosine(
        state, np.vstack([query, np.full((7, query.shape[1]), 99.0, dtype=np.float32)])
    )
    assert np.mean(first.astype(str) == truth.astype(str)) == 1.0
    assert np.array_equal(first, extended[: len(first)])
    assert state.resource["query_rows_used_for_fit"] == 0
    assert state.resource["query_features_used_for_fit"] is False
    assert state.resource["query_role_oracle_access"] is False
    assert state.resource["query_class_quota_access"] is False
    assert state.resource["trainable_parameters"] <= 50_000
    assert state.resource["persistent_state_bytes"] <= 256 * 1024
    assert len(state.trace) == 20

    stable = route.fit_diag_cosine_state(
        support,
        labels,
        seed=19,
        device=torch.device("cpu"),
        candidate=route.CANDIDATE_D2,
    )
    stable_first = route.predict_diag_cosine(stable, query)
    stable_extended = route.predict_diag_cosine(
        stable,
        np.vstack([query, np.full((3, query.shape[1]), -77.0, dtype=np.float32)]),
    )
    assert np.array_equal(stable_first, stable_extended[: len(stable_first)])
    assert stable.resource["trainable_parameters"] == support.shape[1]
    assert (
        stable.resource["classifier_state_policy"]
        == "current_registry_fixed_support_prototypes_zero_class_bias_shared_diag_only"
    )


def test_d3_scenario_oldlock_freezes_every_old_head_and_audits_each_old_class():
    support, labels, _query, _truth = _separable()
    with pytest.raises(
        route.DiagCosineExplorationError,
        match="scenario-specific before/after orchestration",
    ):
        route.fit_diag_cosine_state(
            support,
            labels,
            seed=713101,
            device=torch.device("cpu"),
            candidate=route.CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
        )
    assert all(
        "query" not in name
        for name in inspect.signature(route._fit_d3_after).parameters
    )
    before = route._fit_d3_before(
        _d3_support(include_new=False),
        seed=713101,
        device=torch.device("cpu"),
    )
    after = route._fit_d3_after(
        before,
        _d3_support(include_new=True),
        seed=713101,
        device=torch.device("cpu"),
    )
    old_count = len(before.classes)
    assert old_count == 2
    assert np.array_equal(after.log_scale, before.log_scale)
    assert np.array_equal(after.weights[:, :old_count], before.weights)
    assert np.array_equal(after.bias[:, :old_count], before.bias)
    assert np.all(after.bias[:, old_count:] == 0.0)
    assert np.all(after.new_offset >= 0.0)
    assert after.resource["trainable_parameters"] == 3 * 2 * 24
    expected_before_steps = 3 * route.ADAPTATION_EPOCHS
    expected_after_steps = 3 * route.ADAPTATION_EPOCHS
    assert before.resource["optimizer_steps"] == expected_before_steps
    assert after.resource["optimizer_steps"] == expected_after_steps
    assert before.resource["epochs_per_scenario"] == route.ADAPTATION_EPOCHS
    assert after.resource["total_epoch_passes"] == (
        3 * route.ADAPTATION_EPOCHS
    )
    assert (
        before.resource["classifier_state_policy"]
        == "scenario_specific_old_head_fit"
    )
    assert (
        after.resource["classifier_state_policy"]
        == "scenario_specific_old_head_bitwise_locked_new_weights_only"
    )
    assert after.resource["query_rows_used_for_fit"] == 0
    assert after.resource["query_role_oracle_access"] is False
    assert after.resource["query_class_quota_access"] is False
    assert after.resource["persistent_state_bytes"] <= 256 * 1024
    audits = after.resource["scenario_old_support_intrusion_audit"]
    assert set(audits) == set(route.FORMAL_LEO_WEAK_SCENARIOS)
    for scenario, audit in audits.items():
        assert audit["old_class_intrusion_count"] == 0
        assert audit["worst_old_class_margin"] >= -1.0e-6
        assert set(audit["per_old_class"]) == set(before.classes.tolist())
        for row in audit["per_old_class"].values():
            assert row["old_class_intrusion_count"] == 0
            assert row["post_support_margin_min"] >= -1.0e-6
            assert row["pre_support_accuracy"] == row["post_support_accuracy"]

    scenario = route.FORMAL_LEO_WEAK_SCENARIOS[0]
    query = _d3_support(include_new=True)[scenario][0]
    first = route._scenario_predict(after, scenario, query)
    extended = route._scenario_predict(
        after,
        scenario,
        np.vstack([query, np.full((5, 24), 17.0, dtype=np.float32)]),
    )
    assert np.array_equal(first, extended[: len(first)])
    assert set(first.tolist()) <= set(after.classes.tolist())


def test_d3_parent_loader_binds_commit_receipt_state_and_lineage(tmp_path: Path):
    parent = route._fit_d3_before(
        _d3_support(include_new=False),
        seed=713101,
        device=torch.device("cpu"),
    )
    root = tmp_path / "parent"
    root.mkdir()
    state_path = root / "diag_cosine_state.npz"
    np.savez(
        state_path,
        scenarios=parent.scenarios.astype(str),
        classes=parent.classes.astype(str),
        log_scale=parent.log_scale,
        weights=parent.weights,
        bias=parent.bias,
        new_offset=parent.new_offset,
        old_class_count=np.asarray([parent.old_class_count], dtype=np.int64),
    )
    os.chmod(state_path, stat.S_IREAD)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    enrollment = {
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "phase1_checkpoint_sha256": "1" * 64,
        "feature_runtime_sha256": "2" * 64,
        "method_lock_sha256": "3" * 64,
    }
    apply = {
        "row_handle": "row_" + "4" * 64,
        "row_manifest_sha256": "5" * 64,
    }
    receipt = {
        "schema": "cvs.phase2.diag_cosine_exploration_receipt.v1",
        "stage": "stage2b",
        "registration_state": "before",
        **enrollment,
        **apply,
        "candidate": {"name": route.CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT},
        **PHASE2_FULL_CONTRACT,
        "query_truth_present_in_predictor": False,
        "resource": {
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_features_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
        },
        "artifacts": {"diag_cosine_state.npz": state_sha},
    }
    receipt_path = root / "execution_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    os.chmod(receipt_path, stat.S_IREAD)
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    members = [
        {
            "relative_path": "diag_cosine_state.npz",
            "sha256": state_sha,
            "size_bytes": state_path.stat().st_size,
        },
        {
            "relative_path": "execution_receipt.json",
            "sha256": receipt_sha,
            "size_bytes": receipt_path.stat().st_size,
        },
    ]
    commit_path = root / "COMMIT.json"
    commit_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                "members": members,
                "execution_receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.chmod(commit_path, stat.S_IREAD)
    commit_sha = hashlib.sha256(commit_path.read_bytes()).hexdigest()
    loaded, closure = route._load_parent_scenario_state(
        parent_diag_root=root,
        expected_parent_commit_sha256=commit_sha,
        enrollment_manifest=enrollment,
        apply_manifest=apply,
    )
    assert np.array_equal(loaded.weights, parent.weights)
    assert closure["parent_diag_commit_sha256"] == commit_sha
    assert closure["parent_execution_receipt_sha256"] == receipt_sha
    assert closure["parent_state_sha256"] == state_sha

    os.chmod(state_path, stat.S_IWRITE)
    with pytest.raises(
        route.DiagCosineExplorationError,
        match="parent closure member must be read-only",
    ):
        route._load_parent_scenario_state(
            parent_diag_root=root,
            expected_parent_commit_sha256=commit_sha,
            enrollment_manifest=enrollment,
            apply_manifest=apply,
        )
    os.chmod(state_path, stat.S_IREAD)

    with pytest.raises(route.DiagCosineExplorationError, match="COMMIT SHA256"):
        route._load_parent_scenario_state(
            parent_diag_root=root,
            expected_parent_commit_sha256="f" * 64,
            enrollment_manifest=enrollment,
            apply_manifest=apply,
        )

    os.chmod(receipt_path, stat.S_IWRITE)
    receipt["phase2_query_class_quota_access"] = True
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    os.chmod(receipt_path, stat.S_IREAD)
    tampered_receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    members[1]["sha256"] = tampered_receipt_sha
    members[1]["size_bytes"] = receipt_path.stat().st_size
    os.chmod(commit_path, stat.S_IWRITE)
    commit_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                "members": members,
                "execution_receipt_sha256": tampered_receipt_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.chmod(commit_path, stat.S_IREAD)
    tampered_commit_sha = hashlib.sha256(commit_path.read_bytes()).hexdigest()
    with pytest.raises(
        route.DiagCosineExplorationError,
        match="parent execution receipt lineage drift",
    ):
        route._load_parent_scenario_state(
            parent_diag_root=root,
            expected_parent_commit_sha256=tampered_commit_sha,
            enrollment_manifest=enrollment,
            apply_manifest=apply,
        )


def test_d1_b0_cap_removes_bias_and_caps_only_fft96_log_scale():
    rng = np.random.default_rng(31)
    support = rng.normal(size=(12, 288)).astype(np.float32)
    labels = np.repeat(["class-a", "class-b", "class-c"], 4)
    state = route.fit_diag_cosine_state(
        support,
        labels,
        seed=29,
        device=torch.device("cpu"),
        candidate=route.CANDIDATE_D1_B0_CAP,
    )
    lower, upper = route.log_scale_bounds(route.CANDIDATE_D1_B0_CAP, 288)
    fft_cap = np.log(1.5)
    assert state.bias.shape == (0,)
    assert state.resource["class_bias_enabled"] is False
    assert state.resource["class_bias_trainable_parameters"] == 0
    assert state.resource["trainable_parameters"] == 288 + 3 * 288
    assert state.resource["parameter_state_bytes"] == (288 + 3 * 288) * 4
    assert np.allclose(lower[:160], -1.5)
    assert np.allclose(upper[:160], 1.5)
    assert np.allclose(lower[160:256], -fft_cap)
    assert np.allclose(upper[160:256], fft_cap)
    assert np.allclose(lower[256:], -1.5)
    assert np.allclose(upper[256:], 1.5)
    assert np.all(state.log_scale >= lower)
    assert np.all(state.log_scale <= upper)
    assert len(state.trace) == route.ADAPTATION_EPOCHS == 20
    assert state.resource["support_only"] is True
    assert state.resource["query_rows_used_for_fit"] == 0
    assert state.resource["query_role_oracle_access"] is False
    assert state.resource["query_class_quota_access"] is False
    assert state.resource["query_batch_global_assignment"] is False


def test_standalone_runner_cli_exposes_d1_b0_cap_candidate():
    args = runner.parser().parse_args(
        [
            "--enrollment-package-root",
            "enrollment",
            "--enrollment-seal-path",
            "enrollment.seal.json",
            "--enrollment-seal-sha256",
            "a" * 64,
            "--apply-package-root",
            "apply",
            "--apply-seal-path",
            "apply.seal.json",
            "--apply-seal-sha256",
            "b" * 64,
            "--output-root",
            "output",
            "--device",
            "cpu",
            "--candidate",
            route.CANDIDATE_D1_B0_CAP,
        ]
    )
    assert args.candidate == route.CANDIDATE_D1_B0_CAP

    d3_args = runner.parser().parse_args(
        [
            "--enrollment-package-root",
            "enrollment",
            "--enrollment-seal-path",
            "enrollment.seal.json",
            "--enrollment-seal-sha256",
            "a" * 64,
            "--apply-package-root",
            "apply",
            "--apply-seal-path",
            "apply.seal.json",
            "--apply-seal-sha256",
            "b" * 64,
            "--output-root",
            "output",
            "--device",
            "cpu",
            "--candidate",
            route.CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
            "--parent-diag-root",
            "before",
            "--expected-parent-commit-sha256",
            "c" * 64,
        ]
    )
    assert d3_args.parent_diag_root == "before"
    assert d3_args.expected_parent_commit_sha256 == "c" * 64


def test_fft_rf_features_are_same_row_gain_normalized_and_128d():
    rng = np.random.default_rng(3)
    iq = rng.normal(size=(4, 2, 64)).astype(np.float32)
    fft = route.spectral_logmag_sketch(iq)
    rf = route.rf_statistics(iq)
    scaled_fft = route.spectral_logmag_sketch(7.5 * iq)
    scaled_rf = route.rf_statistics(7.5 * iq)
    assert fft.shape == (4, 96)
    assert rf.shape == (4, 32)
    assert np.allclose(fft, scaled_fft, atol=1.0e-6)
    assert np.allclose(rf, scaled_rf, atol=1.0e-6)
    zid = np.tile(np.eye(1, 160, dtype=np.float32), (4, 1))
    assert route.registered_feature(iq, zid).shape == (4, 288)


class _FakeRuntime(torch.nn.Module):
    def forward(self, rows):
        flat = rows.reshape(rows.shape[0], -1)
        features = torch.zeros((len(rows), 160), dtype=torch.float32, device=rows.device)
        features[:, : flat.shape[1]] = flat
        logits = torch.zeros((len(rows), 6), dtype=torch.float32, device=rows.device)
        return features, logits


def _manifest(profile: str) -> dict:
    classes = [
        {"class_handle": "cls_" + "1" * 64},
        {"class_handle": "cls_" + "2" * 64},
    ]
    return {
        "profile": profile,
        "stage": "stage2c",
        "registration_state": "after",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 2,
        "row_handle": (
            None if profile == route.ENROLLMENT_ONLY else "row_" + "3" * 64
        ),
        "row_manifest_sha256": (
            None if profile == route.ENROLLMENT_ONLY else "4" * 64
        ),
        "phase1_checkpoint_sha256": "5" * 64,
        "feature_runtime_sha256": "6" * 64,
        "method_lock_sha256": "7" * 64,
        "package_root_sha256": ("8" if profile == route.ENROLLMENT_ONLY else "9") * 64,
        "registered_classes": classes,
        "members": [{"kind": "feature_runtime", "relative_path": "sealed_feature_runtime.pt"}],
    }


def _payloads(profile: str) -> dict:
    result = {}
    for scenario_index, scenario in enumerate(route.FORMAL_LEO_WEAK_SCENARIOS):
        if profile == route.ENROLLMENT_ONLY:
            rows = []
            labels = []
            ranks = []
            for class_index, sign in enumerate((1.0, -1.0)):
                for rank in range(3):
                    iq = np.zeros((2, 16), dtype=np.float32)
                    iq[0, class_index] = sign * (1.0 + 0.01 * rank)
                    iq[1, 2 + scenario_index] = 0.1
                    rows.append(iq)
                    labels.append(class_index)
                    ranks.append(rank)
            result[scenario] = {
                "support_leo_weak_iq": np.stack(rows),
                "support_class_indices": np.asarray(labels, dtype=np.int64),
                "support_rank_within_class": np.asarray(ranks, dtype=np.int64),
            }
        else:
            rows = []
            for class_index, sign in enumerate((1.0, -1.0)):
                iq = np.zeros((2, 16), dtype=np.float32)
                iq[0, class_index] = sign
                iq[1, 2 + scenario_index] = 0.1
                rows.append(iq)
            result[scenario] = {
                "query_leo_weak_iq": np.stack(rows),
                "query_tokens": np.asarray(
                    [
                        f"qid_{scenario_index}{class_index}" + "a" * 62
                        for class_index in range(2)
                    ]
                ),
            }
    return result


def test_run_writes_unlabeled_prediction_before_any_scorer(
    tmp_path: Path, monkeypatch
):
    enrollment_root = tmp_path / "enrollment"
    apply_root = tmp_path / "apply"
    output_root = tmp_path / "output"
    enrollment_root.mkdir()
    apply_root.mkdir()
    output_root.mkdir()
    enrollment_manifest = _manifest(route.ENROLLMENT_ONLY)
    apply_manifest = _manifest(route.APPLY_ONLY)

    calls = iter(
        [
            (_payloads(route.ENROLLMENT_ONLY), enrollment_manifest, {"status": "PASS"}),
            (_payloads(route.APPLY_ONLY), apply_manifest, {"status": "PASS"}),
        ]
    )
    monkeypatch.setattr(
        route, "load_verified_somph_predictor_bundle", lambda *args, **kwargs: next(calls)
    )
    monkeypatch.setattr(
        route,
        "load_torchscript_backbone_same_fd",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    result = route.run_diag_cosine_exploration(
        enrollment_package_root=enrollment_root,
        enrollment_seal_path=tmp_path / "enrollment.seal.json",
        enrollment_seal_sha256="a" * 64,
        apply_package_root=apply_root,
        apply_seal_path=tmp_path / "apply.seal.json",
        apply_seal_sha256="b" * 64,
        output_root=output_root,
        device="cpu",
        candidate=route.CANDIDATE_D1_B0_CAP,
    )
    assert result["formal_launch_authority"] is False
    with np.load(output_root / "prediction_artifact.npz", allow_pickle=False) as archive:
        assert tuple(archive.files) == (
            "query_tokens",
            "scenarios",
            "predicted_class_handles",
        )
        assert len(archive["query_tokens"]) == 6
    receipt = json.loads(
        (output_root / "execution_receipt.json").read_text(encoding="utf-8")
    )
    raw = json.dumps(receipt, sort_keys=True)
    assert "true_label" not in raw
    assert "role_label" not in raw
    assert receipt["query_truth_present_in_predictor"] is False
    assert receipt["phase2_sample_view_policy"] == "leo_weak_only_no_clean_access"
    assert receipt["clean_sample_access"] is False
    assert receipt["clean_derived_signal_access"] is False
    assert (
        receipt["phase2_query_decision_policy"]
        == "per_sample_all_registered_classes"
    )
    assert receipt["phase2_query_role_oracle_access"] is False
    assert receipt["phase2_query_true_batch_class_count_access"] is False
    assert receipt["phase2_query_class_quota_access"] is False
    assert receipt["phase2_query_batch_global_assignment"] is False
    assert receipt["resource"]["query_rows_used_for_fit"] == 0
    assert receipt["resource"]["support_enrollment_rows"] == 12
    assert receipt["resource"]["query_backbone_forwards_per_sample"] == 1
    assert receipt["resource"]["serialized_persistent_state_bytes"] == (
        output_root / "diag_cosine_state.npz"
    ).stat().st_size
    assert receipt["candidate"]["name"] == route.CANDIDATE_D1_B0_CAP
    assert receipt["candidate"]["class_bias_enabled"] is False
    assert receipt["candidate"]["auxiliary_weight"] == 4.0
    assert receipt["candidate"]["adaptation_epochs"] == 20
    assert receipt["candidate"]["log_scale_bounds"]["fft96"] == [
        -np.log(1.5),
        np.log(1.5),
    ]
    with np.load(output_root / "diag_cosine_state.npz", allow_pickle=False) as state:
        assert state["bias"].shape == (0,)
    assert (output_root / "COMMIT.json").is_file()
