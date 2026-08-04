from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_next_r2_matrix as matrix
from cvsrffi import stage2_next_r2_score as scorer


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return _sha(raw)


def _fixture(tmp_path: Path, *, drop_state: bool = False, repair: bool = True):
    receivers = tuple(f"rx{i}" for i in range(7))
    classes = tuple(f"tx{i}" for i in range(6))
    plan = dict(matrix.build_next_r2_proxy24_plan(receivers, classes, source_identity_sha256="a" * 64))
    capsule_keys = []
    truth: dict[str, str] = {}
    repair_key_id = next(item["outer_key_id"] for item in plan["keys"] if item["active_k"] == 5)
    for key in plan["keys"]:
        held_receiver = key["held_receiver"]
        held_class = key["held_class"]
        k = int(key["active_k"])
        registrations = {}
        for registration, registered in (("REG0", tuple(key["retained_classes"])), ("REG1", tuple(key["all_registered_classes"]))):
            support_ids = []
            support_indices = []
            support_labels = []
            query_ids = []
            query_indices = []
            for cls in registered:
                ids = tuple(f"{held_receiver}|{cls}|{i}" for i in range(14))
                for i in range(k):
                    support_ids.append(ids[i]); support_indices.append(i); support_labels.append(cls)
                for i in range(5, 14):
                    query_ids.append(ids[i]); query_indices.append(i)
                    truth[ids[i]] = cls
            registrations[registration] = {
                "registered_classes": registered,
                "support_indices": tuple(support_indices),
                "support_labels": tuple(support_labels),
                "support_physical_ids": tuple(support_ids),
                "query_indices": tuple(query_indices),
                "query_physical_ids": tuple(query_ids),
            }
        capsule_keys.append({
            "outer_key_id": key["outer_key_id"],
            "held_receiver": held_receiver,
            "held_class": held_class,
            "active_k": k,
            "registrations": registrations,
        })
    capsule_payload = {
        "schema": scorer.PREDICTION_CAPSULE_SCHEMA,
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "selected_iq_archive_sha256": "b" * 64,
        "selected_iq_receipt_sha256": "c" * 64,
        "label_join_archive_sha256": "d" * 64,
        "physical_id_root_sha256": "e" * 64,
        "matrix_sha256": plan["matrix_sha256"],
        "plan": plan,
        "keys": tuple(capsule_keys),
        "truth_opened_for_capsule_build": True,
        "query_labels_persisted": False,
    }
    capsule_payload["capsule_content_sha256"] = matrix.canonical_sha256(capsule_payload)
    capsule_raw = matrix.canonical_bytes(capsule_payload)
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_bytes(capsule_raw)
    capsule_sha = _sha(capsule_raw)
    root = tmp_path / "run"
    root.mkdir()
    (root / "states").mkdir()
    state_entries = []
    for key in plan["keys"]:
        held = key["held_class"]
        k = int(key["active_k"])
        for state_id in matrix.STATE_IDS:
            registration = "REG1" if state_id in matrix.REG1_STATES else "REG0"
            bound = next(item for item in capsule_keys if item["outer_key_id"] == key["outer_key_id"])["registrations"][registration]
            qids = tuple(bound["query_physical_ids"])
            registered = tuple(bound["registered_classes"])
            predictions = list(truth[qid] for qid in qids)
            # Introduce one deterministic K5 REG1 baseline error that DA1 repairs.
            if repair and k == 5 and state_id == "DA0_REG1" and key["outer_key_id"] == repair_key_id:
                predictions[0] = registered[1]
            scores = np.zeros((len(qids), len(registered)), dtype="<f4")
            scores[0, 0] = np.float32(len(state_entries) + 1)
            stem = f"{key['outer_key_id']}__{state_id}"
            npz_path = root / "states" / f"{stem}.npz"
            np.savez_compressed(npz_path, query_physical_ids=np.asarray(qids, dtype=np.str_), registered_classes=np.asarray(registered, dtype=np.str_), scores=scores, predictions=np.asarray(predictions, dtype=np.str_))
            receipt = {
                "schema": scorer.STATE_RECEIPT_SCHEMA,
                "candidate_id": matrix.CANDIDATE_ID,
                "protocol_schema": matrix.PROTOCOL_SCHEMA,
                "capsule_id": "capsule-test",
                "split_id": "split-test",
                "outer_key_id": key["outer_key_id"],
                "state_id": state_id,
                "active_k": k,
                "registered_classes": registered,
                "query_physical_id_root": _sha(matrix.canonical_bytes({"ids": qids})),
                "scores_sha256": _sha(scores.tobytes(order="C")),
                "predictions_sha256": _sha(matrix.canonical_bytes({"predictions": tuple(predictions)})),
                "query_truth_input_count": 0,
                "query_rows_used_for_fit": 0,
                "query_state_updates": 0,
                "query_selection_count": 0,
                "cvfr_status": "CVFR_NONIDENTITY" if state_id in matrix.DA1_STATES else "DA0_IDENTITY_NO_CVFR_FIT",
            }
            receipt["state_receipt_sha256"] = matrix.canonical_sha256(receipt)
            seal = {
                "schema": scorer.STATE_RECEIPT_SCHEMA,
                "outer_key_id": key["outer_key_id"],
                "state_id": state_id,
                "registered_classes": registered,
                "query_physical_id_root": receipt["query_physical_id_root"],
                "scores_sha256": receipt["scores_sha256"],
                "predictions_sha256": receipt["predictions_sha256"],
                "state_receipt_sha256": receipt["state_receipt_sha256"],
                "cvfr_status": receipt["cvfr_status"],
                "bssdg_state_sha256": "f" * 64,
            }
            seal["state_seal_sha256"] = matrix.canonical_sha256(seal)
            bssdg_wire_path = root / "states" / f"{stem}.bssdg.wire"
            bssdg_wire_path.write_bytes(b"bssdg-wire")
            cvfr_wire_path = root / "states" / f"{stem}.cvfr.wire" if state_id in matrix.DA1_STATES else None
            if cvfr_wire_path is not None:
                cvfr_wire_path.write_bytes(b"cvfr-wire")
            state_payload = {
                "schema": scorer.STATE_RECEIPT_SCHEMA,
                "outer_key_id": key["outer_key_id"],
                "state_id": state_id,
                "receipt": receipt,
                "seal": seal,
                "npz_path": npz_path.relative_to(root).as_posix(),
                "npz_sha256": _sha(npz_path.read_bytes()),
                "bssdg_wire_path": bssdg_wire_path.relative_to(root).as_posix(),
                "bssdg_wire_sha256": _sha(bssdg_wire_path.read_bytes()),
                "cvfr_wire_path": cvfr_wire_path.relative_to(root).as_posix() if cvfr_wire_path is not None else None,
                "cvfr_wire_sha256": _sha(cvfr_wire_path.read_bytes()) if cvfr_wire_path is not None else None,
                "truth_present": False,
                "score_present": False,
            }
            json_path = root / "states" / f"{stem}.json"
            _write_json(json_path, state_payload)
            state_entries.append({
                "outer_key_id": key["outer_key_id"], "state_id": state_id,
                "json_path": json_path.relative_to(root).as_posix(), "json_sha256": _sha(json_path.read_bytes()),
                "npz_path": npz_path.relative_to(root).as_posix(), "npz_sha256": _sha(npz_path.read_bytes()),
                "state_seal_sha256": seal["state_seal_sha256"],
            })
    if drop_state:
        state_entries.pop()
    manifest = {
        "schema": scorer.SEALED_MANIFEST_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "matrix_sha256": plan["matrix_sha256"],
        "outer_key_count": matrix.OUTER_KEY_COUNT,
        "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
        "all_states_sealed": True,
        "sealed_before_scoring": True,
        "truth_opened": False,
        "states": state_entries,
    }
    manifest["sealed_manifest_sha256"] = matrix.canonical_sha256(manifest)
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "plan.json", plan)
    prereg = {
        "schema": "cvs.stage2.next_r2.proxy24.runner.v2",
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "capsule_file_sha256": capsule_sha,
        "capsule_content_sha256": capsule_payload["capsule_content_sha256"],
        "matrix_sha256": plan["matrix_sha256"],
        "query_labels_present": False,
        "truth_input": None,
        "truth_scoring_in_process": False,
    }
    _write_json(root / "preregistration.json", prereg)
    completion = {
        "schema": scorer.COMPLETION_SCHEMA,
        "run_id": "run-test",
        "status": "ARTIFACTS_COMPLETE_NOT_SCORED",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "outer_keys_completed": matrix.OUTER_KEY_COUNT,
        "states_completed": matrix.STATE_PREDICTION_COUNT,
        "all_states_sealed": True,
        "label_join_opened": False,
        "query_labels_present": False,
        "truth_opened": False,
        "scoring_performed": False,
        "plan_sha256": _sha((root / "plan.json").read_bytes()),
        "manifest_sha256": _sha((root / "manifest.json").read_bytes()),
        "preregistration_sha256": _sha((root / "preregistration.json").read_bytes()),
    }
    _write_json(root / "completion.json", completion)
    # The actual archive is deliberately opened by the scorer only after the
    # closure checks; its path/hash is pinned in the capsule.
    truth_path = tmp_path / "labels.npz"
    pids = np.asarray(tuple(truth), dtype=np.str_)
    tx = np.asarray(tuple(truth.values()), dtype=np.str_)
    np.savez_compressed(truth_path, z_dom=np.zeros((len(pids), 1), dtype="<f4"), pre_relu=np.zeros((len(pids), 1), dtype="<f4"), receiver_ids=np.asarray(["rx0"] * len(pids), dtype=np.str_), day_ids=np.asarray(["d"] * len(pids), dtype=np.str_), tx_labels=tx, physical_ids=pids)
    # Bind the test capsule to the archive bytes.
    capsule_payload["label_join_archive_sha256"] = _sha(truth_path.read_bytes())
    capsule_payload["capsule_content_sha256"] = matrix.canonical_sha256({k: v for k, v in capsule_payload.items() if k != "capsule_content_sha256"})
    capsule_raw = matrix.canonical_bytes(capsule_payload)
    capsule_path.write_bytes(capsule_raw)
    capsule_sha = _sha(capsule_raw)
    prereg["capsule_file_sha256"] = capsule_sha
    prereg["capsule_content_sha256"] = capsule_payload["capsule_content_sha256"]
    _write_json(root / "preregistration.json", prereg)
    completion["preregistration_sha256"] = _sha((root / "preregistration.json").read_bytes())
    _write_json(root / "completion.json", completion)
    return root, capsule_path, capsule_sha, truth_path, _sha(truth_path.read_bytes())


def test_complete_closure_scores_four_states_and_k5_primary(tmp_path: Path) -> None:
    root, capsule, capsule_sha, truth, truth_sha = _fixture(tmp_path)
    result = scorer.score_next_r2_proxy24(run_root=root, prediction_capsule=capsule, prediction_capsule_sha256=capsule_sha, ls_label_join_archive=truth, ls_label_join_archive_sha256=truth_sha)
    assert result["truth_opened_after_complete_predictions"] is True
    assert result["formal_target_claim"] is False
    assert len(result["state_scores"]) == 96
    assert result["state_scores"][0]["N_seen_new"] is None
    assert result["state_scores"][0]["H_retained_new"] is None
    assert result["state_scores"][0]["registration_metric_status"] == "NA_BEFORE_REGISTRATION"
    assert result["k5_primary"]["pass"] is True
    assert "DID_RETAINED" in result["four_state_differences_by_k"]["5"]


def test_incomplete_96_rejected_before_truth_open(tmp_path: Path) -> None:
    root, capsule, capsule_sha, truth, truth_sha = _fixture(tmp_path, drop_state=True)
    with pytest.raises(scorer.NextR2ScoreError, match="96|incomplete"):
        scorer.score_next_r2_proxy24(run_root=root, prediction_capsule=capsule, prediction_capsule_sha256=capsule_sha, ls_label_join_archive=truth, ls_label_join_archive_sha256=truth_sha)


def test_capsule_hash_drift_rejected(tmp_path: Path) -> None:
    root, capsule, capsule_sha, truth, truth_sha = _fixture(tmp_path)
    capsule.write_bytes(capsule.read_bytes() + b" ")
    with pytest.raises(scorer.NextR2ScoreError, match="capsule SHA"):
        scorer.score_next_r2_proxy24(run_root=root, prediction_capsule=capsule, prediction_capsule_sha256=capsule_sha, ls_label_join_archive=truth, ls_label_join_archive_sha256=truth_sha)


def test_no_function_when_da_predictions_match(tmp_path: Path) -> None:
    root, capsule, capsule_sha, truth, truth_sha = _fixture(tmp_path, repair=False)
    # The synthetic fixture leaves all DA1 predictions equal to their DA0
    # counterparts, so the scorer must close the candidate as NO_FUNCTION.
    result = scorer.score_next_r2_proxy24(run_root=root, prediction_capsule=capsule, prediction_capsule_sha256=capsule_sha, ls_label_join_archive=truth, ls_label_join_archive_sha256=truth_sha)
    assert result["decision"] == "NO_FUNCTION"


def test_wire_hash_and_da0_cvfr_presence_are_checked(tmp_path: Path) -> None:
    root, capsule, capsule_sha, truth, truth_sha = _fixture(tmp_path)
    wire = next((root / "states").glob("*.bssdg.wire"))
    wire.write_bytes(wire.read_bytes() + b"drift")
    with pytest.raises(scorer.NextR2ScoreError, match="wire_path hash"):
        scorer.score_next_r2_proxy24(run_root=root, prediction_capsule=capsule, prediction_capsule_sha256=capsule_sha, ls_label_join_archive=truth, ls_label_join_archive_sha256=truth_sha)
