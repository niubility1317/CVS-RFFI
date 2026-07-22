import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.r2a_fixed_held_four_arm as module
from cvsrffi.r2a_fixed_held_four_arm import ARMS, CANDIDATE_REVISION, K, R2AFixedHeldError, SCOPE, build_packet, predict_packet, score_packet


SHA = hashlib.sha256(b"r2a-fixed-coverage").hexdigest()
BINDING = {"archive_schema": module.DUAL_ARCHIVE_SCHEMA, "coverage_schema": module.COVERAGE_SCHEMA, "archive_sha256": "a" * 64, "manifest_sha256": "b" * 64, "coverage_sha256": SHA}


def archive():
    rows = {name: [] for name in ("z_id", "z_dom", "labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names")}
    classes = [f"c{i}" for i in range(6)]; receivers = [f"r{i}" for i in range(7)]
    for ri, receiver in enumerate(receivers):
        for ci, label in enumerate(classes):
            for si, scene in enumerate(("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")):
                for sample in range(8):
                    zid = np.zeros(160, np.float32); zid[ci] = 1.0; zid[20 + si] = np.float32((sample + 1) * 0.004); zid[40 + ri] = np.float32((ci + 1) * 0.003)
                    zdom = (np.arange(160, dtype=np.float32) + 1.0) * np.float32(0.00001 * (sample + 1)); zdom[ri] += np.float32(2.0 + 0.2 * ci + 0.03 * sample); zdom[20 + si] += np.float32(0.3 + 0.02 * ri); zdom[80 + ci] += np.float32(0.1 * (ri + 1))
                    for name, value in (("z_id", zid), ("z_dom", zdom), ("labels", label), ("receiver_ids", receiver), ("day_ids", "d0"), ("physical_ids", f"{receiver}-{label}-{scene}-{sample}"), ("scenario_names", scene)):
                        rows[name].append(value)
    return {"z_id": np.asarray(rows["z_id"], np.float32), "z_dom": np.asarray(rows["z_dom"], np.float32), "labels": np.asarray(rows["labels"]), "receiver_ids": np.asarray(rows["receiver_ids"]), "day_ids": np.asarray(rows["day_ids"]), "physical_ids": np.asarray(rows["physical_ids"]), "scenario_names": np.asarray(rows["scenario_names"]), "class_ids": np.asarray(classes)}


def query_for(packet, a):
    wanted = sorted({key for row in packet["rows"] for key in row["query_ids"]}); index = {key: i for i, key in enumerate(a["physical_ids"].tolist())}
    return wanted, np.asarray([a["z_id"][index[key]] for key in wanted], np.float32)


def test_fixed_held_build_predict_score_and_resource_contract():
    a = archive(); packet, truth = build_packet(a, coverage_sha256=SHA, artifact_binding=BINDING)
    assert packet["candidate_revision"] == CANDIDATE_REVISION == "v1.1" and packet["evaluation_scope"] == SCOPE and packet["pseudo_new"] is True and len(packet["rows"]) == 18
    assert packet["resource_contract"] == {"effective_rank": 1, "build_mac": 5936, "query_mac": 1126, "optimizer_steps": 0}
    assert all(row["resource"]["c6"]["support_build_mac"] == 5936 and row["resource"]["c6"]["production_postprocess_mac_per_query"] == 1126 and row["resource"]["optimizer_steps"] == 0 for row in packet["rows"])
    assert all(group["audit"]["blas_threads"] == [1] and group["audit"]["numpy_version"] == np.__version__ and all(item["num_threads"] == 1 for item in group["audit"]["blas_lapack"]) for group in packet["lock_groups"].values())
    ids, z = query_for(packet, a); prediction = predict_packet(packet, ids, z)
    assert prediction["COMMIT"] and all(set(row["before"]) == set(ARMS) and set(row["after"]) == set(ARMS) for row in prediction["rows"])
    metrics = score_packet(packet, prediction, truth, commit=prediction["COMMIT"], truth_sha256=truth["truth_sha256"])
    assert len(metrics) == 72 and {row["arm"] for row in metrics} == set(ARMS) and all(set(("old_before", "old_after", "old_adaptation_gain", "seen_new", "H_old_new", "BA", "floor", "min_old", "min_new", "forgetting", "old_to_new", "new_to_old", "per_class", "I_syn")).issubset(row) for row in metrics)
    for offset in range(0, 72, 4):
        quartet = {row["arm"]: row for row in metrics[offset:offset + 4]}
        expected = quartet["M_JOINT"]["H_old_new"] - quartet["M_DA"]["H_old_new"] - quartet["M_HEAD"]["H_old_new"] + quartet["M0"]["H_old_new"]
        assert all(row["I_syn"] == expected and row["old_adaptation_gain"] == -row["forgetting"] and set(row["per_class"]) == set(packet["classes"]) for row in quartet.values())


def test_held_feature_perturbation_cannot_change_bh_lock_digest_or_predict_read_truth():
    a = archive(); packet, _ = build_packet(a, coverage_sha256=SHA, artifact_binding=BINDING)
    held = packet["held_receiver"]; changed = {name: value.copy() for name, value in a.items()}; changed["z_id"][(changed["receiver_ids"] == held) & (changed["labels"] == "c0")] += np.float32(0.02)
    packet_changed, _ = build_packet(changed, coverage_sha256=SHA, artifact_binding=BINDING)
    assert packet["lock_groups"] == packet_changed["lock_groups"]
    ids, z = query_for(packet, a); prediction = predict_packet(packet, ids, z)
    bad_truth = {"query_ids": ids, "z_id": z, "labels": np.asarray(["c0"] * len(ids))}
    with np.testing.assert_raises(R2AFixedHeldError):
        predict_packet(packet, bad_truth["query_ids"], bad_truth["z_id"].astype(np.float64))
    with np.testing.assert_raises(R2AFixedHeldError):
        score_packet(packet, prediction, {"packet_sha256": packet["packet_sha256"], "rows": []}, commit=prediction["COMMIT"], truth_sha256="0" * 64)


def test_truth_seal_and_prediction_row_bijection_reject_forgery():
    a = archive(); packet, truth = build_packet(a, coverage_sha256=SHA, artifact_binding=BINDING)
    ids, z = query_for(packet, a); prediction = predict_packet(packet, ids, z)
    forged_truth = copy.deepcopy(truth)
    classes = packet["classes"]
    forged_truth["rows"][0]["query_labels"] = {key: classes[(classes.index(value) + 1) % len(classes)] for key, value in truth["rows"][0]["query_labels"].items()}
    with pytest.raises(R2AFixedHeldError, match="truth sidecar SHA"):
        score_packet(packet, prediction, forged_truth, commit=prediction["COMMIT"], truth_sha256=truth["truth_sha256"])
    resigned_truth = copy.deepcopy(forged_truth); resigned_truth.pop("truth_sha256"); resigned_truth["truth_sha256"] = module._sha(resigned_truth)
    with pytest.raises(R2AFixedHeldError, match="truth sidecar SHA"):
        score_packet(packet, prediction, resigned_truth, commit=prediction["COMMIT"], truth_sha256=truth["truth_sha256"])
    duplicated = copy.deepcopy(prediction); duplicated["rows"] = [copy.deepcopy(prediction["rows"][0]) for _ in range(18)]; duplicated.pop("COMMIT"); duplicated["COMMIT"] = module._sha(duplicated)
    with pytest.raises(R2AFixedHeldError, match="prediction row/query identity"):
        score_packet(packet, duplicated, truth, commit=duplicated["COMMIT"], truth_sha256=truth["truth_sha256"])
    relabeled = copy.deepcopy(prediction)
    payload = relabeled["rows"][0]["after"]["M_JOINT"]
    payload["prediction"][0] = payload["classes"][(payload["classes"].index(payload["prediction"][0]) + 1) % len(payload["classes"])]
    relabeled.pop("COMMIT"); relabeled["COMMIT"] = module._sha(relabeled)
    with pytest.raises(R2AFixedHeldError, match="argmax/logit binding"):
        score_packet(packet, relabeled, truth, commit=relabeled["COMMIT"], truth_sha256=truth["truth_sha256"])


def _write_archive_fixture(path: Path) -> None:
    a = archive(); n = len(a["labels"])
    arrays = {"z_id": a["z_id"], "z_dom": a["z_dom"], "tx_logits": np.zeros((n, 6), np.float32), "labels": a["labels"], "receiver_ids": a["receiver_ids"], "day_ids": a["day_ids"], "physical_ids": a["physical_ids"], "scenario_names": a["scenario_names"], "class_ids": a["class_ids"], "observation_ids": np.asarray([f"obs-{value}" for value in a["physical_ids"].tolist()])}
    assert tuple(arrays) == module.DUAL_ARCHIVE_MEMBERS
    np.savez_compressed(path, **arrays)


def _coverage_receipt(archive_sha: str, manifest_sha: str) -> dict:
    classes = list(module.REAL_CLASS_IDS); receivers = [f"r{i}" for i in range(7)]; days = [f"d{i}" for i in range(4)]; scenes = list(module.SCENES)
    cells = {f"{receiver}|{day}|{label}": 50 for receiver in receivers for day in days for label in classes}
    return {"schema": module.COVERAGE_SCHEMA, "status": module.COVERAGE_STATUS, "artifact_stage": "phase1_offline_before_target_access", "archive_sha256": archive_sha, "manifest_sha256": manifest_sha, "metadata_arrays_read": list(module._COVERAGE_METADATA), "feature_arrays_read": [], "row_count": 8400, "physical_id_unique_count": 8400, "observation_id_unique_count": 8400, "class_ids": classes, "receiver_ids": receivers, "day_ids": days, "scenario_names": scenes, "counts_by_class": {label: 1400 for label in classes}, "counts_by_receiver": {receiver: 1200 for receiver in receivers}, "counts_by_day": {day: 2100 for day in days}, "counts_by_scenario": {scene: 2800 for scene in scenes}, "counts_by_receiver_day_class": cells, "receiver_day_class_cell_count": 168, "receiver_day_class_zero_cell_count": 0, "receiver_day_class_min_count": 50, "receiver_day_class_max_count": 50, "pre_registered_coverage_gate_passed": True, "k_values_described_only": [1, 5, 10], "min_rows_remaining_after_support_by_k": {"1": 49, "5": 45, "10": 40}, "target_access": False, "query_access": False, "held_fold_selected": False}


def test_cli_loader_calls_dual_archive_verifier_and_binds_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive_path = tmp_path / "archive.npz"; manifest_path = tmp_path / "manifest.json"; coverage_path = tmp_path / "coverage.json"
    _write_archive_fixture(archive_path)
    manifest_path.write_text(json.dumps({"schema": module.DUAL_ARCHIVE_SCHEMA}), encoding="utf-8")
    receipt = _coverage_receipt(module._sha_file(archive_path), module._sha_file(manifest_path)); coverage_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(module, "verify_phase1_singleobs_dual_feature_archive", lambda path, manifest: called.append((Path(path), manifest["schema"])))
    loaded, binding = module._load_archive(archive_path, manifest_path, coverage_path, module._sha_file(coverage_path))
    assert called == [(archive_path.resolve(), module.DUAL_ARCHIVE_SCHEMA)] and tuple(loaded) == module._MEMBERS
    assert binding == {"archive_schema": module.DUAL_ARCHIVE_SCHEMA, "coverage_schema": module.COVERAGE_SCHEMA, "archive_sha256": module._sha_file(archive_path), "manifest_sha256": module._sha_file(manifest_path), "coverage_sha256": module._sha_file(coverage_path)}


def test_cli_loader_rejects_minimal_manifest_and_coverage_binding_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive_path = tmp_path / "archive.npz"; manifest_path = tmp_path / "manifest.json"; coverage_path = tmp_path / "coverage.json"
    _write_archive_fixture(archive_path)
    manifest_path.write_text(json.dumps({"formal_phase2_eligible": False, "bundle_created": False, "held_runner_tx_logits_allowed": False}), encoding="utf-8")
    receipt = _coverage_receipt(module._sha_file(archive_path), module._sha_file(manifest_path)); coverage_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(R2AFixedHeldError, match="dual archive verifier"):
        module._load_archive(archive_path, manifest_path, coverage_path, module._sha_file(coverage_path))
    monkeypatch.setattr(module, "verify_phase1_singleobs_dual_feature_archive", lambda path, manifest: None)
    receipt["archive_sha256"] = "f" * 64; coverage_path = tmp_path / "coverage-tampered.json"; coverage_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(R2AFixedHeldError, match="coverage archive/manifest binding"):
        module._load_archive(archive_path, manifest_path, coverage_path, module._sha_file(coverage_path))
