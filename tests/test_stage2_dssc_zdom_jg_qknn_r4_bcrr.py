import argparse
import copy
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cvsrffi.stage2_dssc_zdom_jg_qknn_r4_bcrr as dssc_module
from cvsrffi.stage2_dssc_zdom_jg_qknn_r4_bcrr import (
    ADAPTER_SCALE_GROUPS,
    ARMS,
    BCRRState,
    CANDIDATE,
    DSSCStateError,
    GEOFF_R8_COVERAGE_SHA256,
    MAX_WIRE_BYTES,
    MIN_ADAPTER_FP16_SCALE,
    PHASE1_ARCHIVE_MANIFEST_SHA256,
    PHASE1_ARCHIVE_SHA256,
    PHASE1_CHECKPOINT_SHA256,
    PHASE1_PARITY_RECEIPT_SHA256,
    SEALED_RUNTIME_SHA256,
    SOMPH_PACKAGE_LOCK_SHA256,
    adapt_support_only,
    attach_rank4_adapter,
    bcrr_fused_logits,
    build_ground_bundle_arrays,
    build_five_arm_states,
    build_qknn_state,
    canonical_method_lock,
    fit_bcrr_support_only,
    load_ground_bundle,
    qknn_lock_from_method_lock,
    qknn_logits,
    qknn_neighbor_receipt,
    predict_five_arms,
    resource_receipt,
    sha256_file,
    typed_tokens,
    validate_method_lock,
)
from scripts import build_phase1_dssc_zdom_jg_bundle as bundle_builder
from scripts.build_phase1_dssc_zdom_jg_bundle import LOCK
from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as runner


def _qlock(k):
    return qknn_lock_from_method_lock(LOCK, k_shot=k)


def _bundle(tmp_path):
    rng = np.random.default_rng(9)
    labels = np.asarray(["old_a"] * 6 + ["old_b"] * 6)
    ids = np.asarray([f"p{i}" for i in range(12)])
    arrays = build_ground_bundle_arrays(
        z_id=rng.normal(size=(12, 160)).astype(np.float32),
        z_dom=rng.normal(size=(12, 160)).astype(np.float32),
        labels=labels,
        physical_ids=ids,
        archive_sha256=PHASE1_ARCHIVE_SHA256,
        archive_manifest_sha256=PHASE1_ARCHIVE_MANIFEST_SHA256,
        checkpoint_sha256=PHASE1_CHECKPOINT_SHA256,
        method_lock=canonical_method_lock(),
    )
    path = tmp_path / "bundle.npz"
    np.savez_compressed(path, **arrays)
    return load_ground_bundle(
        path, checkpoint_sha256=PHASE1_CHECKPOINT_SHA256
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("rank",), 3),
        (("optimizer", "lr"), 0.03),
        (("adapter_quantization", "scale_groups"), [[0, 1, 2, 3]]),
        (("full125", "jobs"), 124),
        (("query_policy",), "role_aware"),
        (
            ("resource_profile", "id_backbone_feat_joint_mac_per_sample"),
            1,
        ),
        (
            (
                "d01_d18_contract",
                "D15",
                "numeric_only_change_is_DA_success",
            ),
            True,
        ),
    ],
)
def test_canonical_lock_rejects_every_non_qknn_drift(path, replacement):
    lock = canonical_method_lock()
    target = lock
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(DSSCStateError, match="exact schema/value drift"):
        validate_method_lock(lock)


def test_qknn_receipts_are_distinct_canonical_authority_digests():
    lock, digest = validate_method_lock(canonical_method_lock())
    qknn = lock["qknn"]
    assert digest == __import__("hashlib").sha256(
        json.dumps(
            lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    assert (
        qknn["phase1_lodo_receipt_sha256"]
        != qknn["quantization_margin_audit_sha256"]
    )
    assert qknn["phase1_lodo_receipt_sha256"] != GEOFF_R8_COVERAGE_SHA256
    assert (
        qknn["quantization_margin_audit_sha256"]
        != GEOFF_R8_COVERAGE_SHA256
    )


def test_bundle_int8_and_no_member_ids(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle.v_dom.shape[0] <= 4
    assert np.all(bundle.prototype_physical_counts >= 2)
    assert "physical_ids" not in (tmp_path / "bundle.npz").read_bytes().decode(
        "latin1", errors="ignore"
    )


def test_bundle_embedded_lock_tamper_fails(tmp_path):
    bundle = _bundle(tmp_path)
    source = tmp_path / "bundle.npz"
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    manifest = json.loads(arrays["manifest_json"].item())
    manifest["method_lock"]["rank"] = 3
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(DSSCStateError):
        load_ground_bundle(
            tampered, checkpoint_sha256=PHASE1_CHECKPOINT_SHA256
        )
    assert bundle.classes == ("old_a", "old_b")


def test_bundle_builder_npz_is_true_create_once(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    archive = tmp_path / "archive.npz"
    np.savez_compressed(
        archive,
        z_id=rng.normal(size=(12, 160)).astype(np.float32),
        z_dom=rng.normal(size=(12, 160)).astype(np.float32),
        labels=np.asarray(["old_a"] * 6 + ["old_b"] * 6),
        physical_ids=np.asarray([f"p{i}" for i in range(12)]),
    )
    archive_manifest = tmp_path / "archive.manifest.json"
    archive_manifest.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.singleobs_dual_feature_archive.v2",
                "access_audit": {
                    "target_access": False,
                    "query_access": False,
                    "clean_iq_access": False,
                },
                "artifact": {"sha256": PHASE1_ARCHIVE_SHA256},
                "inputs": {
                    "checkpoint_sha256": PHASE1_CHECKPOINT_SHA256,
                    "parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    lock_path = tmp_path / "method_lock.json"
    bundle_builder.write_lock(lock_path)
    output = tmp_path / "ground_bundle.npz"
    original_sha = bundle_builder.sha256_file

    def fixed_frozen_sha(path):
        frozen = {
            archive: PHASE1_ARCHIVE_SHA256,
            archive_manifest: PHASE1_ARCHIVE_MANIFEST_SHA256,
            checkpoint: PHASE1_CHECKPOINT_SHA256,
        }
        return frozen.get(Path(path), original_sha(path))

    monkeypatch.setattr(bundle_builder, "sha256_file", fixed_frozen_sha)
    args = argparse.Namespace(
        archive=str(archive),
        archive_manifest=str(archive_manifest),
        phase1_checkpoint=str(checkpoint),
        method_lock=str(lock_path),
        output=str(output),
    )
    first = bundle_builder.build(args)
    assert first["sha256"] == original_sha(output)
    with pytest.raises(FileExistsError):
        bundle_builder.build(args)


def test_typed_token_conflict_fails():
    with pytest.raises(DSSCStateError):
        typed_tokens(np.asarray([1, 2]), name="bad")


def test_qknn_bcrr_k1_and_receipts(tmp_path):
    bundle = _bundle(tmp_path)
    rng = np.random.default_rng(4)
    features = rng.normal(size=(2, 160)).astype(np.float32)
    state = build_qknn_state(
        features,
        ["old_a", "old_b"],
        ["old_a", "old_b"],
        ["x0", "x1"],
        qknn_lock=_qlock(1),
    )
    logits = qknn_logits(state, features)
    bcrr = fit_bcrr_support_only(state, features, k_shot=1)
    assert bcrr.omega == 0 and logits.shape == (2, 2)
    assert qknn_neighbor_receipt(state, features)["query_rows_used_for_fit"] == 0
    resource = resource_receipt(
        bundle=bundle, qknn=state, bcrr=bcrr, adapter=None
    )
    assert resource["wire_bytes"] <= MAX_WIRE_BYTES
    assert set(resource["state_bytes"]) == {
        "ground_bundle",
        "adapter",
        "qknn",
        "bcrr",
    }
    assert resource["qknn_quantization_audit"]["top1_agreement"] >= 0.995


def test_five_arm_prediction_uses_three_qknn_and_two_bcrr_calls(monkeypatch):
    rng = np.random.default_rng(40)
    labels = ("old_a", "old_b")
    raw = rng.normal(size=(2, 160)).astype(np.float32)
    ng = rng.normal(size=(2, 160)).astype(np.float32)
    ground = rng.normal(size=(2, 160)).astype(np.float32)
    states = build_five_arm_states(
        raw_support_features=raw,
        ng_support_features=ng,
        ground_support_features=ground,
        support_labels=labels,
        registered_classes=labels,
        support_physical_ids=("p0", "p1"),
        k_shot=1,
        qknn_lock=_qlock(1),
    )
    calls = {"qknn": 0, "qknn_bcrr": 0}
    original_qknn = dssc_module.qknn_logits
    original_qknn_bcrr = dssc_module._svrn_scores

    def counted_qknn(*args, **kwargs):
        calls["qknn"] += 1
        return original_qknn(*args, **kwargs)

    def counted_qknn_bcrr(*args, **kwargs):
        calls["qknn_bcrr"] += 1
        return original_qknn_bcrr(*args, **kwargs)

    monkeypatch.setattr(dssc_module, "qknn_logits", counted_qknn)
    monkeypatch.setattr(dssc_module, "_svrn_scores", counted_qknn_bcrr)
    result = predict_five_arms(
        states,
        raw_query_features=raw,
        ng_query_features=ng,
        ground_query_features=ground,
    )
    assert set(result) == set(ARMS)
    assert calls == {"qknn": 1, "qknn_bcrr": 2}


def test_nonlexical_registry_projects_legacy_class_axis_back_to_sealed_handles():
    rng = np.random.default_rng(411)
    registry = ("old_z", "new_m", "old_a")
    canonical = tuple(sorted(registry))
    labels = registry
    raw = rng.normal(size=(3, 160)).astype(np.float32)
    ng = rng.normal(size=(3, 160)).astype(np.float32)
    ground = rng.normal(size=(3, 160)).astype(np.float32)
    common = {
        "raw_support_features": raw,
        "ng_support_features": ng,
        "ground_support_features": ground,
        "support_labels": labels,
        "support_physical_ids": ("p0", "p1", "p2"),
        "k_shot": 1,
        "qknn_lock": _qlock(1),
    }
    sealed_states = build_five_arm_states(
        **common, registered_classes=registry
    )
    control_states = build_five_arm_states(
        **common, registered_classes=canonical
    )
    sealed = predict_five_arms(
        sealed_states,
        raw_query_features=raw,
        ng_query_features=ng,
        ground_query_features=ground,
    )
    control = predict_five_arms(
        control_states,
        raw_query_features=raw,
        ng_query_features=ng,
        ground_query_features=ground,
    )
    control_to_sealed = [canonical.index(name) for name in registry]
    assert sealed_states["M0"].classes == registry
    for arm in ARMS:
        np.testing.assert_allclose(
            sealed[arm], control[arm][:, control_to_sealed], rtol=0.0,
            atol=1.0e-6,
        )
        sealed_handles = np.asarray(registry)[np.argmax(sealed[arm], axis=1)]
        control_handles = np.asarray(canonical)[np.argmax(control[arm], axis=1)]
        assert tuple(sealed_handles) == tuple(control_handles)
    np.testing.assert_allclose(
        bcrr_fused_logits(
            sealed["M0"], raw, sealed_states["M_OTHER"][1]
        ),
        sealed["M_OTHER"], rtol=0.0, atol=1.0e-6,
    )
    fitted_control = control_states["M_OTHER"][1]
    legacy_control = BCRRState(
        fitted_control.branch_state,
        fitted_control.omega,
        fitted_control.receipt,
    )
    assert legacy_control.classes is None
    np.testing.assert_allclose(
        bcrr_fused_logits(control["M0"], raw, legacy_control),
        control["M_OTHER"], rtol=0.0, atol=1.0e-6,
    )


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.id_gate = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.joint_proj = torch.nn.Sequential(torch.nn.Linear(4, 160))
        self.dom = torch.nn.Linear(4, 160)

    def forward(self, x, return_aux=False):
        hidden = torch.tanh(self.id_gate(x))
        zid = self.joint_proj(hidden)
        zdom = self.dom(x)
        return {"z_id": zid, "z_dom": zdom} if return_aux else zid


def test_adapter_zero_group_has_positive_fp16_scale_and_round_trips():
    source_model = Tiny()
    source = attach_rank4_adapter(source_model, None, ground_enabled=False)
    with torch.no_grad():
        source.coefficients.copy_(
            torch.tensor([0.0, 0.0, 0.25, -0.10], dtype=source.coefficients.dtype)
        )
    source.quantize_in_place()
    assert float(source.coefficient_scale_fp16[0]) == MIN_ADAPTER_FP16_SCALE
    assert np.all(source.coefficient_scale_fp16 > 0)
    target_model = Tiny()
    target = attach_rank4_adapter(target_model, None, ground_enabled=False)
    target.load_quantized(source.coefficient_codes, source.coefficient_scale_fp16)
    assert torch.equal(source.coefficients, target.coefficients)


def test_s_b_quantized_clone_and_s_c_continuity(tmp_path):
    bundle = _bundle(tmp_path)
    torch.manual_seed(3)
    model = Tiny()
    base_state = copy.deepcopy(model.state_dict())
    old_support = torch.randn(10, 4)
    old_labels = ["old_a"] * 5 + ["old_b"] * 5
    s_b, s_b_receipt = adapt_support_only(
        model,
        old_support,
        old_labels,
        ["old_a", "old_b"],
        k_shot=5,
        stage="S_B",
        bundle=bundle,
        ground_enabled=True,
        ground_old_registry=["old_a", "old_b"],
        support_physical_ids=[f"old_{i}" for i in range(10)],
        merge=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(trainable) == 1
    assert trainable[0] is s_b.coefficients
    assert trainable[0].numel() == 4
    s_b_codes = s_b.coefficient_codes.copy()
    s_b_scale = s_b.coefficient_scale_fp16.copy()
    assert ADAPTER_SCALE_GROUPS == ((0, 1), (2, 3))
    assert s_b_codes.shape == (4,) and s_b_codes.dtype == np.int8
    assert s_b_scale.shape == (2,) and s_b_scale.dtype == np.float16
    assert not s_b.merged
    assert (
        s_b_receipt["adapter_int8_teacher_deployed"]["scope"]
        == "support_only_teacher_geometry_vs_int8_deployed_geometry"
    )
    assert s_b_receipt["adapter_int8_teacher_deployed"]["top1_agreement"] >= 0.995
    assert (
        s_b_receipt["adapter_int8_teacher_deployed"][
            "large_margin_flip_count"
        ]
        == 0
    )

    deployed_model = Tiny()
    deployed_model.load_state_dict(base_state)
    deployed = attach_rank4_adapter(
        deployed_model, bundle, ground_enabled=True
    )
    with pytest.raises(DSSCStateError, match="schema drift"):
        deployed.load_quantized(s_b_codes, np.ones((1,), np.float16))
    deployed.load_quantized(s_b_codes, s_b_scale)
    deployed.merge()
    assert np.array_equal(deployed.coefficient_codes, s_b_codes)
    assert np.array_equal(deployed.coefficient_scale_fp16, s_b_scale)

    all_support = torch.cat([old_support, torch.randn(5, 4)], dim=0)
    all_labels = old_labels + ["new_c"] * 5
    s_c, s_c_receipt = adapt_support_only(
        model,
        all_support,
        all_labels,
        ["old_a", "old_b", "new_c"],
        k_shot=5,
        stage="S_C",
        bundle=bundle,
        ground_enabled=True,
        ground_old_registry=["old_a", "old_b"],
        support_physical_ids=[f"all_{i}" for i in range(15)],
        continue_adapter=s_b,
        merge=True,
    )
    assert s_c is s_b and s_c.merged
    assert s_c_receipt["stage"] == "S_C"
    assert s_c_receipt["steps"] == 25
    assert s_c_receipt["adapter_int8_teacher_deployed"]["top1_agreement"] >= 0.995


def test_after_support_uses_full_registered_length():
    registry = ("old_a", "old_b", "new_c")
    payload = {
        "support_rank_within_class": np.asarray(
            [rank for _ in registry for rank in range(5)], np.int64
        ),
        "support_class_indices": np.asarray(
            [index for index in range(len(registry)) for _ in range(5)],
            np.int64,
        ),
        "support_tokens": np.asarray([f"s{i}" for i in range(15)]),
        "support_leo_weak_iq": np.zeros((15, 2, 256), np.float32),
    }
    iq, labels, tokens = runner._support(payload, registry, 5)
    assert len(iq) == len(labels) == len(tokens) == 15
    assert labels[-5:] == ("new_c",) * 5
    query_iq, query_tokens = runner._query(
        {
            "query_leo_weak_iq": np.zeros((60, 2, 256), np.float32),
            "query_tokens": np.asarray([f"q{i}" for i in range(60)]),
        }
    )
    assert len(query_iq) == len(query_tokens) == 60


def test_coverage_receipt_rejects_wrong_sha(tmp_path):
    receipt = tmp_path / "coverage.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.DSSCLauncherError, match="SHA drift"):
        runner._validate_coverage_receipt(receipt)


def _materialized_row_receipt(job, expected, *, physical_gpu_id=2):
    row_root = Path(job["output_root"])
    row_root.mkdir(parents=True)
    publications = {}
    prediction_receipts = {}
    for state in ("before", "after"):
        class_count = job["old_class_count"] + (
            0 if state == "before" else job["new_class_count"]
        )
        tokens = np.asarray(
            [
                f"{state}:{scene}:q{index}"
                for scene in runner.SCENES
                for index in range(class_count * runner.QUERY_PER_TX)
            ]
        )
        scenarios = np.asarray(
            [
                scene
                for scene in runner.SCENES
                for _ in range(class_count * runner.QUERY_PER_TX)
            ]
        )
        predicted = np.asarray(["class_0"] * len(tokens))
        publications[state] = {}
        for arm in ARMS:
            path = runner._prediction_artifact_path(row_root, state, arm)
            publications[state][arm] = runner._write_npz_new(
                path,
                query_tokens=tokens,
                scenarios=scenarios,
                predicted_class_handles=predicted,
            )
        prediction_receipt_path = (
            row_root / "predictions" / state / "prediction_receipt.json"
        )
        prediction_receipts[state] = runner._write_json_new(
            prediction_receipt_path,
            {
                "candidate": CANDIDATE,
                "state": state,
                "arms": list(ARMS),
                "query_truth_present_in_predictor": False,
                "query_rows_used_for_fit": 0,
                "prediction_sha256_by_arm": publications[state],
                "runtime": {},
            },
        )

    h = {"M0": 0.20, "M_DA_NG": 0.21, "M_DA": 0.22, "M_OTHER": 0.23, "M_JOINT": 0.27}
    metrics = {}
    for arm in ARMS:
        base = {
            "candidate": CANDIDATE,
            "before_prediction_sha256": publications["before"][arm],
            "after_prediction_sha256": publications["after"][arm],
            "truth_sidecar_sha256": "9" * 64,
            "after": {"h_old_new": h[arm]},
        }
        base_sha = runner._write_json_new(
            runner._score_artifact_paths(row_root)[f"{arm}.base_score"], base
        )
        metrics[arm] = {**base, "score_artifact_sha256": base_sha}
        runner._write_json_new(
            runner._score_artifact_paths(row_root)[f"{arm}.score"], metrics[arm]
        )
    metrics["I_syn"] = h["M_JOINT"] - h["M_DA"] - h["M_OTHER"] + h["M0"]
    rows = [
        {
            "arm": arm,
            "scene": scene,
            "query_count": (job["old_class_count"] + job["new_class_count"])
            * runner.QUERY_PER_TX,
            "forgetting": 0.0,
        }
        for arm in ARMS
        for scene in runner.SCENES
    ]
    metrics["same_row_scene_metrics"] = rows
    runner._write_json_new(
        runner._score_artifact_paths(row_root)["same_row_summary"],
        {
            "candidate": CANDIDATE,
            "arms": {key: metrics[key] for key in (*ARMS, "I_syn")},
            "rows": rows,
            "score_rows": len(rows),
            "query_truth_joined_only_after_all_five_immutable_predictions": True,
        },
    )
    score_hashes = {
        name: sha256_file(path)
        for name, path in runner._score_artifact_paths(row_root).items()
    }
    return {
        "candidate": CANDIDATE,
        "job_id": job["job_id"],
        "status": "ROW_ARTIFACTS_COMPLETE",
        "receiver": job["receiver"],
        "seed": job["seed"],
        "k_shot": job["k_shot"],
        "new_class_count": job["new_class_count"],
        "prediction_slice_count": 3,
        "score_rows": 15,
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "phase1_archive_sha256": expected["phase1_archive_sha256"],
        "phase1_archive_manifest_sha256": expected[
            "phase1_archive_manifest_sha256"
        ],
        "phase1_parity_receipt_sha256": expected[
            "phase1_parity_receipt_sha256"
        ],
        "cache_manifest_sha256": job["cache_manifest_sha256"],
        "authority_commit_sha256": job["authority_commit_sha256"],
        "ground_bundle_sha256": expected["ground_bundle_sha256"],
        "dssc_method_lock_sha256": expected["dssc_method_lock_sha256"],
        "somph_package_lock_sha256": expected["somph_package_lock_sha256"],
        "sealed_runtime_sha256": expected["sealed_runtime_sha256"],
        "geoff_r8_coverage_sha256": expected["geoff_r8_coverage_sha256"],
        "qknn_lock_digest": qknn_lock_from_method_lock(
            canonical_method_lock(), k_shot=int(job["k_shot"])
        ).lock_digest,
        "query_truth_in_predictor": False,
        "query_rows_used_for_fit": 0,
        "device_namespace_execution":
            runner._expected_row_device_namespace_execution(
                physical_gpu_id
            ),
        "full_metrics": metrics,
        "prediction_sha256_by_state_arm": publications,
        "prediction_receipt_sha256_by_state": prediction_receipts,
        "score_artifact_sha256": score_hashes,
    }


def test_full125_receipt_validation_recomputes_artifacts_and_rejects_tamper(tmp_path):
    job = {
        "job_id": "dssc_r1f_rx_20-1_s_713102_k_5_n_20",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 5,
        "new_class_count": 20,
        "old_class_count": 2,
        "cache_manifest_sha256": "1" * 64,
        "authority_commit_sha256": "2" * 64,
        "output_root": str(tmp_path / "row"),
    }
    expected = {
        "checkpoint_sha256": PHASE1_CHECKPOINT_SHA256,
        "phase1_archive_sha256": PHASE1_ARCHIVE_SHA256,
        "phase1_archive_manifest_sha256": PHASE1_ARCHIVE_MANIFEST_SHA256,
        "phase1_parity_receipt_sha256": PHASE1_PARITY_RECEIPT_SHA256,
        "ground_bundle_sha256": "3" * 64,
        "dssc_method_lock_sha256": "4" * 64,
        "somph_package_lock_sha256": SOMPH_PACKAGE_LOCK_SHA256,
        "sealed_runtime_sha256": SEALED_RUNTIME_SHA256,
        "geoff_r8_coverage_sha256": GEOFF_R8_COVERAGE_SHA256,
    }
    valid = _materialized_row_receipt(job, expected)
    launcher_namespace = runner._row_device_namespace(2)
    runner._validate_completed_row_receipt(
        job,
        valid,
        expected_hashes=expected,
        launcher_device_namespace=launcher_namespace,
    )
    for key, replacement in (
        ("receiver", "7-7"),
        ("seed", 999),
        ("ground_bundle_sha256", "f" * 64),
    ):
        tampered = copy.deepcopy(valid)
        tampered[key] = replacement
        with pytest.raises(runner.DSSCLauncherError):
            runner._validate_completed_row_receipt(
                job,
                tampered,
                expected_hashes=expected,
                launcher_device_namespace=launcher_namespace,
            )
    for field, replacement in (
        ("schema", "cvs.dssc.full125.row_device_namespace.invalid"),
        ("cuda_visible_devices", "5"),
        ("visible_physical_gpu_id", 5),
        ("requested_logical_device", "cuda:1"),
        ("torch_cuda_device_count", 8),
        ("torch_cuda_current_device", 1),
    ):
        tampered = copy.deepcopy(valid)
        tampered["device_namespace_execution"][field] = replacement
        with pytest.raises(runner.DSSCLauncherError, match="namespace"):
            runner._validate_completed_row_receipt(
                job,
                tampered,
                expected_hashes=expected,
                launcher_device_namespace=launcher_namespace,
            )
    missing_namespace = copy.deepcopy(valid)
    del missing_namespace["device_namespace_execution"]
    with pytest.raises(runner.DSSCLauncherError, match="namespace"):
        runner._validate_completed_row_receipt(
            job,
            missing_namespace,
            expected_hashes=expected,
            launcher_device_namespace=launcher_namespace,
        )
    with pytest.raises(runner.DSSCLauncherError, match="namespace"):
        runner._validate_completed_row_receipt(
            job,
            valid,
            expected_hashes=expected,
            launcher_device_namespace=runner._row_device_namespace(5),
        )
    missing = runner._prediction_artifact_path(Path(job["output_root"]), "after", "M_JOINT")
    missing.chmod(stat.S_IWRITE)
    missing.unlink()
    with pytest.raises(runner.DSSCLauncherError, match="artifact/hash"):
        runner._validate_completed_row_receipt(
            job,
            valid,
            expected_hashes=expected,
            launcher_device_namespace=launcher_namespace,
        )


def test_launcher_receipt_recomputes_log_hashes(tmp_path):
    root = tmp_path / "matrix"
    launcher = root / "launcher"
    launcher.mkdir(parents=True)
    job = {
        "job_id": "dssc_r1f_rx_20-1_s_713102_k_5_n_20",
        "schedule_cost": {"cost": 1},
    }
    stdout = launcher / f"{job['job_id']}.stdout.log"
    stderr = launcher / f"{job['job_id']}.stderr.log"
    stdout_sha = runner._write_bytes_new(stdout, b"complete\n")
    stderr_sha = runner._write_bytes_new(stderr, b"")
    runner._write_json_new(
        launcher / f"{job['job_id']}.launcher_receipt.json",
        {
            "schema": runner.LAUNCHER_RECEIPT_SCHEMA,
            "candidate": CANDIDATE,
            "job_id": job["job_id"],
            "status": "ROW_PROCESS_COMPLETE",
            "device_namespace": runner._row_device_namespace(2),
            "duration_seconds": 1.0,
            "returncode": 0,
            "stdout_path": str(stdout),
            "stdout_sha256": stdout_sha,
            "stderr_path": str(stderr),
            "stderr_sha256": stderr_sha,
            "exception": None,
            "schedule_cost": job["schedule_cost"],
        },
    )
    runner._validate_launcher_receipt(root, job, allowed_gpu_ids=(2,))
    stdout.chmod(stat.S_IWRITE)
    stdout.write_bytes(b"tampered\n")
    with pytest.raises(runner.DSSCLauncherError, match="log artifact/hash"):
        runner._validate_launcher_receipt(root, job, allowed_gpu_ids=(2,))


def test_row_subprocess_command_and_environment_isolate_physical_gpu(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    job = {
        "cache_manifest": tmp_path / "cache_set.json",
        "authority_bundle": tmp_path / "authority",
        "authority_commit_sha256": "a" * 64,
        "output_root": tmp_path / "row",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 5,
        "new_class_count": 20,
    }
    args = argparse.Namespace(
        phase1_checkpoint="checkpoint.pth",
        sealed_runtime="sealed_runtime.pt",
        package_method_lock="somph_method_lock.json",
        dssc_method_lock="dssc_method_lock.json",
        ground_bundle="ground_bundle.npz",
        coverage_receipt="coverage_receipt.json",
    )
    command = runner._row_subprocess_command(job, args)
    environment_2 = runner._row_subprocess_environment(2)
    environment_5 = runner._row_subprocess_environment(5)
    device_index = command.index("--device")
    assert command[device_index + 1] == runner.ROW_LOGICAL_DEVICE == "cuda:0"
    assert command.count("--device") == 1
    assert environment_2 is not environment_5
    assert environment_2["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment_5["CUDA_VISIBLE_DEVICES"] == "5"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"
    environment_2["CUDA_VISIBLE_DEVICES"] = "changed"
    assert environment_5["CUDA_VISIBLE_DEVICES"] == "5"


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema", "cvs.dssc.full125.launcher_receipt.invalid"),
        ("physical_gpu_id", 5),
        ("logical_device", "cuda:2"),
        ("cuda_visible_devices", "5"),
    ),
)
def test_launcher_receipt_rejects_cuda_namespace_drift(
    tmp_path, field, replacement
):
    root = tmp_path / "matrix"
    launcher = root / "launcher"
    launcher.mkdir(parents=True)
    job = {
        "job_id": "dssc_r1f_rx_20-1_s_713102_k_5_n_20",
        "schedule_cost": {"cost": 1},
    }
    stdout = launcher / f"{job['job_id']}.stdout.log"
    stderr = launcher / f"{job['job_id']}.stderr.log"
    stdout_sha = runner._write_bytes_new(stdout, b"complete\n")
    stderr_sha = runner._write_bytes_new(stderr, b"")
    receipt = {
        "schema": runner.LAUNCHER_RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "job_id": job["job_id"],
        "status": "ROW_PROCESS_COMPLETE",
        "device_namespace": runner._row_device_namespace(2),
        "duration_seconds": 1.0,
        "returncode": 0,
        "stdout_path": str(stdout),
        "stdout_sha256": stdout_sha,
        "stderr_path": str(stderr),
        "stderr_sha256": stderr_sha,
        "exception": None,
        "schedule_cost": job["schedule_cost"],
    }
    if field == "schema":
        receipt[field] = replacement
    else:
        receipt["device_namespace"][field] = replacement
    runner._write_json_new(
        launcher / f"{job['job_id']}.launcher_receipt.json", receipt
    )
    with pytest.raises(runner.DSSCLauncherError, match="GPU namespace"):
        runner._validate_launcher_receipt(root, job, allowed_gpu_ids=(2,))


def test_frozen_five_arm_lpt_full125_launcher(tmp_path):
    cache, authority = tmp_path / "cache", tmp_path / "authority"
    for receiver in runner.RECEIVERS:
        for seed in runner.SEEDS:
            cache_leaf = (
                cache
                / f"rx_{receiver.replace('-', '_')}"
                / f"seed_{seed}"
            )
            cache_leaf.mkdir(parents=True)
            (cache_leaf / "cache_set.json").write_text("{}", encoding="utf-8")
            authority_leaf = (
                authority
                / f"authority_bundle_rx_{receiver.replace('-', '_')}_seed_{seed}"
            )
            authority_leaf.mkdir(parents=True)
            (authority_leaf / "COMMIT.json").write_text(
                "{}", encoding="utf-8"
            )
    jobs = runner.matrix_jobs(
        cache_root=cache,
        authority_root=authority,
        run_root=tmp_path / "run",
        old_class_count=2,
    )
    costs = [
        (
            job["schedule_cost"][
                "optimizer_step_x_support_rows_two_adapters"
            ],
            job["schedule_cost"]["query_rows_tiebreak"],
        )
        for job in jobs
    ]
    assert ARMS == ("M0", "M_DA_NG", "M_DA", "M_OTHER", "M_JOINT")
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 125
    assert len(jobs) * 3 == 375 and len(jobs) * 3 * 5 == 1875
    assert costs == sorted(costs, reverse=True)
    assert runner._parse_gpu_ids("0,2,7") == (0, 2, 7)
    with pytest.raises(runner.DSSCLauncherError):
        runner._parse_gpu_ids("0,0")


def test_row_device_binding_sets_and_reads_back_indexed_cuda(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "set_device", lambda value: selected.append(value.index))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: selected[-1])
    runner._activate_row_device("cuda:5")
    assert selected == [5]
    with pytest.raises(runner.DSSCLauncherError, match="indexed CUDA"):
        runner._activate_row_device("cpu")


def test_row_device_execution_evidence_uses_environment_and_torch_observation(
    monkeypatch,
):
    selected = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda value: selected.append(value.index)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: selected[-1])
    evidence = runner._activate_row_device("cuda:0")
    assert selected == [0]
    assert evidence == runner._expected_row_device_namespace_execution(4)
    runner._validate_formal_row_device_namespace_execution(evidence)


def test_numpy2_safe_identity_feature_boundary_avoids_as_tensor(monkeypatch):
    class DummyBackbone(nn.Module):
        def forward(self, rows, **_kwargs):
            return {"feat_joint": rows.reshape(len(rows), -1)[:, :160]}

    class DummyModel:
        id_backbone = DummyBackbone()
        id_feature_key = "feat_joint"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ndarray C-API bridge must remain unreachable")

    monkeypatch.setattr(torch, "as_tensor", forbidden)
    iq = np.arange(2 * 2 * 256, dtype=np.float32).reshape(2, 2, 256)
    result = runner._id_feature(DummyModel(), iq, device="cpu")
    assert result.shape == (2, 160)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
