from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cvsrffi.phase1_dualreadout_bundle_v2 import (
    CALIBRATION_SCHEMA,
    DualReadoutBundleError,
    build_bundle,
    fit_source_calibration,
    load_bundle,
    sha256_file,
)
from scripts.phase1_dualreadout_bundle_v2 import emit, score


class ToyRuntime(nn.Module):
    def forward(self, rows: torch.Tensor):
        mean = rows.mean(dim=2)
        z_id = torch.nn.functional.normalize(torch.cat([mean, mean], dim=1), dim=1)
        z_dom = torch.nn.functional.normalize(torch.cat([mean[:, :1], -mean[:, 1:2], mean], dim=1), dim=1)
        logits = torch.stack([mean[:, 0] * 4.0, mean[:, 1] * 4.0], dim=1)
        return z_id, z_dom, logits


def fixture_arrays():
    rng = np.random.default_rng(7)
    rows = 24
    tx = np.asarray(["a"] * 10 + ["b"] * 10 + ["u"] * 4)
    roles = np.asarray(["source"] * 20 + ["proxy_unknown"] * 4)
    rx = np.asarray((["0", "1"] * 10) + ["0"] * 4)
    day = np.asarray((["0", "1"] * 10) + ["0"] * 4)
    sig = np.asarray([f"sig-{index}" for index in range(rows)])
    zid = np.zeros((rows, 4), dtype=np.float32)
    zdom = np.zeros((rows, 4), dtype=np.float32)
    logits = np.zeros((rows, 2), dtype=np.float32)
    angular_logits = np.zeros((rows, 2), dtype=np.float32)
    for index in range(rows):
        if tx[index] == "a":
            zid[index] = [1.0, 0.05, 0.0, 0.0]
            logits[index] = [5.0, 0.0]
            angular_logits[index] = [4.5, 0.1]
        elif tx[index] == "b":
            zid[index] = [0.0, 1.0, 0.05, 0.0]
            logits[index] = [0.0, 5.0]
            angular_logits[index] = [0.1, 4.5]
        else:
            zid[index] = [0.0, 0.0, 1.0, 0.0]
            logits[index] = [0.2, 0.1]
            angular_logits[index] = [3.0, -3.0]
        if rx[index] == "0":
            zdom[index] = [1.0, 0.05, 0.0, 0.0]
        else:
            zdom[index] = [0.0, 1.0, 0.05, 0.0]
    zid += rng.normal(0.0, 0.005, zid.shape).astype(np.float32)
    zdom += rng.normal(0.0, 0.005, zdom.shape).astype(np.float32)
    return {
        "angular_logits": angular_logits,
        "robust_z_id": zid,
        "robust_z_dom": zdom,
        "robust_logits": logits,
        "tx_ids": tx,
        "roles": roles,
        "rx_ids": rx,
        "day_ids": day,
        "physical_ids": sig,
    }


def fit(arrays=None):
    values = fixture_arrays() if arrays is None else arrays
    return fit_source_calibration(
        **{key: value for key, value in values.items() if key in {"angular_logits", "robust_z_id", "robust_z_dom", "robust_logits"}},
        tx_ids=values["tx_ids"].tolist(),
        roles=values["roles"].tolist(),
        rx_ids=values["rx_ids"].tolist(),
        day_ids=values["day_ids"].tolist(),
        physical_ids=values["physical_ids"].tolist(),
        class_handles=["a", "b"],
    )


def scripted_runtime(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = torch.jit.trace(ToyRuntime().eval(), torch.randn(2, 2, 16), strict=False)
    torch.jit.save(runtime, str(path))


def build_toy_bundle(tmp_path: Path):
    calibration, receipt = fit()
    angular = tmp_path / "angular.ts"
    robust = tmp_path / "robust.ts"
    scripted_runtime(angular)
    scripted_runtime(robust)
    root = tmp_path / "bundle"
    manifest = build_bundle(
        root,
        angular_runtime=angular,
        robust_runtime=robust,
        calibration=calibration,
        calibration_receipt=receipt,
        angular_checkpoint_sha256="a" * 64,
        robust_checkpoint_sha256="b" * 64,
    )
    return root, manifest


def test_fit_uses_only_source_known_and_never_proxy_values():
    original = fixture_arrays()
    changed = copy.deepcopy(original)
    proxy = changed["roles"] == "proxy_unknown"
    changed["angular_logits"][proxy] = 1_000.0
    changed["robust_logits"][proxy] = -1_000.0
    changed["robust_z_id"][proxy] = 99.0
    changed["robust_z_dom"][proxy] = -99.0
    left, left_receipt = fit(original)
    right, right_receipt = fit(changed)
    for key in left:
        assert np.array_equal(left[key], right[key])
    assert left_receipt == right_receipt
    assert left_receipt["threshold_scope"] == "source_joint_correct_only_no_proxy_or_target_tuning"


def test_fit_rejects_duplicate_source_physical_ids():
    arrays = fixture_arrays()
    arrays["physical_ids"][1] = arrays["physical_ids"][0]
    with pytest.raises(DualReadoutBundleError, match="unique"):
        fit(arrays)


def test_bundle_build_load_and_runtime_no_query_smoke(tmp_path):
    root, manifest = build_toy_bundle(tmp_path)
    loaded = load_bundle(root, expected_content_root_sha256=manifest["content_root_sha256"])
    result = loaded.forward_iq(torch.randn(5, 2, 16))
    assert result["p_local"].shape == (5, 3)
    assert np.allclose(result["p_local"].sum(1), 1.0)
    assert set(result["local_decision"].tolist()) <= {"registered", "unknown", "defer"}
    assert manifest["raw_training_checkpoint_included"] is False
    assert manifest["role_or_truth_included"] is False


def test_robust_runtime_alone_controls_registered_label(tmp_path):
    root, _ = build_toy_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    loaded = load_bundle(root, expected_content_root_sha256=manifest["content_root_sha256"])
    calibration = loaded.calibration
    center_a = calibration["class_centers"][:1]
    domain = calibration["domain_centers"][:1]
    robust = np.asarray([[8.0, 0.0]], dtype=np.float32)
    angular = np.asarray([[0.0, 8.0]], dtype=np.float32)
    result = loaded.evaluate_arrays(
        angular_logits=angular,
        robust_z_id=center_a,
        robust_z_dom=domain,
        robust_logits=robust,
    )
    assert result["robust_pred"].tolist() == [0]
    assert result["local_label"].tolist() != ["b"]


def test_member_tamper_and_extra_file_fail_closed(tmp_path):
    root, _ = build_toy_bundle(tmp_path)
    runtime = root / "runtimes/angular.ts"
    with runtime.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(DualReadoutBundleError, match="hash/size"):
        load_bundle(root, expected_content_root_sha256=(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["content_root_sha256"]))
    root2, _ = build_toy_bundle(tmp_path / "second")
    (root2 / "role.json").write_text('{"role":"source"}', encoding="utf-8")
    with pytest.raises(DualReadoutBundleError, match="allowlist"):
        load_bundle(root2, expected_content_root_sha256=(json.loads((root2 / "manifest.json").read_text(encoding="utf-8"))["content_root_sha256"]))


def test_loader_requires_external_content_root_binding(tmp_path):
    root, _ = build_toy_bundle(tmp_path)
    with pytest.raises(DualReadoutBundleError, match="external content root"):
        load_bundle(root, expected_content_root_sha256="0" * 64)


def _feature_npz(path: Path, arrays: dict, features: np.ndarray):
    np.savez(
        path,
        features=features,
        tx_logits=arrays["robust_logits"],
        tx_ids=arrays["tx_ids"],
        rx_ids=arrays["rx_ids"],
        day_ids=arrays["day_ids"],
        eq_ids=np.asarray([f"eq-{i}" for i in range(len(features))]),
        sig_ids=arrays["physical_ids"],
        dataset_role=arrays["roles"],
        channel_views=np.asarray(["clean"] * len(features)),
        sat_scenarios=np.asarray([""] * len(features)),
    )


def test_emit_evidence_contains_no_role_truth_or_same_event_claim(tmp_path):
    root, manifest = build_toy_bundle(tmp_path)
    arrays = fixture_arrays()
    angular_npz = tmp_path / "angular.npz"
    robust_npz = tmp_path / "robust.npz"
    dom_npz = tmp_path / "dom.npz"
    _feature_npz(angular_npz, arrays, arrays["robust_z_id"])
    _feature_npz(robust_npz, arrays, arrays["robust_z_id"])
    _feature_npz(dom_npz, arrays, arrays["robust_z_dom"])
    output = tmp_path / "evidence.jsonl"
    receipt = tmp_path / "receipt.json"
    result = emit(
        argparse.Namespace(
            bundle=str(root),
            expected_content_root_sha256=manifest["content_root_sha256"],
            angular_zid_npz=str(angular_npz),
            robust_zid_npz=str(robust_npz),
            robust_zdom_npz=str(dom_npz),
            base_manifest_id="TEST-MANIFEST",
            output_jsonl=str(output),
            receipt_out=str(receipt),
            deadline_ms=100.0,
            max_rows=6,
        )
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert all(row["linkage_mode"] == "proxy_unverified" for row in rows)
    assert all("role" not in row and "true_label" not in row and "emission_event_id" not in row for row in rows)
    assert all(row["bundle_id"] == manifest["content_root_sha256"] for row in rows)
    assert result["same_event_claim"] is False


def test_bundle_receipt_persists_no_source_ids_or_roles(tmp_path):
    root, _ = build_toy_bundle(tmp_path)
    receipt = json.loads((root / "calibration/receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == CALIBRATION_SCHEMA
    assert "physical_ids" not in receipt
    assert "role" not in receipt
    assert receipt["physical_ids_persisted"] is False


def test_truth_scorer_counts_defer_separately_and_never_as_safe_reject(tmp_path):
    arrays = fixture_arrays()
    truth_npz = tmp_path / "truth.npz"
    _feature_npz(truth_npz, arrays, arrays["robust_z_id"])
    evidence = []
    for index in range(len(arrays["roles"])):
        if arrays["roles"][index] == "source":
            decision, label = "registered", str(arrays["tx_ids"][index])
        else:
            decision, label = ("unknown", None) if index % 2 == 0 else ("defer", None)
        evidence.append({"local_decision": decision, "local_label": label})
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8")
    output = tmp_path / "score.json"
    result = score(
        argparse.Namespace(
            evidence_jsonl=str(evidence_path),
            truth_feature_npz=str(truth_npz),
            known_roles="source",
            unknown_roles="proxy_unknown",
            output_json=str(output),
        )
    )
    assert result["unknown_safe_rejection_rate"] == 0.5
    assert result["unknown_defer_rate"] == 0.5
    assert result["unknown_false_accept_rate"] == 0.0
    assert result["formal_phase3_performance"] is False
