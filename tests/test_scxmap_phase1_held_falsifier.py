from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

import cvsrffi.r2a_fixed_held_four_arm as r2
import cvsrffi.scxmap_phase1_held_falsifier as scx
from cvsrffi.scxmap_phase1_held_falsifier import (
    ARMS,
    K_VALUES,
    ROW_COUNT,
    SCXMapHeldError,
    build_packet,
    predict_packet,
    score_packet,
)


SHA = hashlib.sha256(b"scxmap-held-coverage").hexdigest()
BINDING = {
    "archive_schema": r2.DUAL_ARCHIVE_SCHEMA,
    "coverage_schema": r2.COVERAGE_SCHEMA,
    "archive_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "coverage_sha256": SHA,
}


def _archive():
    fields = (
        "z_id",
        "z_dom",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
    )
    rows = {name: [] for name in fields}
    classes = [f"c{index}" for index in range(6)]
    receivers = [f"r{index}" for index in range(7)]
    scenes = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    for receiver_index, receiver in enumerate(receivers):
        for class_index, label in enumerate(classes):
            for scene_index, scene in enumerate(scenes):
                for sample in range(12):
                    zid = np.zeros(160, dtype=np.float32)
                    zid[class_index] = 1.0
                    zid[20 + scene_index] = np.float32((sample + 1) * 0.004)
                    zid[40 + receiver_index] = np.float32(
                        (class_index + 1) * 0.003
                    )
                    zdom = (np.arange(160, dtype=np.float32) + 1.0) * np.float32(
                        0.00001 * (sample + 1)
                    )
                    zdom[receiver_index] += np.float32(
                        2.0 + 0.2 * class_index + 0.03 * sample
                    )
                    zdom[20 + scene_index] += np.float32(
                        0.3 + 0.02 * receiver_index
                    )
                    zdom[80 + class_index] += np.float32(
                        0.1 * (receiver_index + 1)
                    )
                    values = (
                        zid,
                        zdom,
                        label,
                        receiver,
                        "d0",
                        f"{receiver}-{label}-{scene}-{sample}",
                        scene,
                    )
                    for name, value in zip(fields, values):
                        rows[name].append(value)
    return {
        "z_id": np.asarray(rows["z_id"], dtype=np.float32),
        "z_dom": np.asarray(rows["z_dom"], dtype=np.float32),
        "labels": np.asarray(rows["labels"]),
        "receiver_ids": np.asarray(rows["receiver_ids"]),
        "day_ids": np.asarray(rows["day_ids"]),
        "physical_ids": np.asarray(rows["physical_ids"]),
        "scenario_names": np.asarray(rows["scenario_names"]),
        "class_ids": np.asarray(classes),
    }


def _query(packet, archive):
    wanted = sorted({value for row in packet["rows"] for value in row["query_ids"]})
    index = {
        str(value): position
        for position, value in enumerate(archive["physical_ids"].tolist())
    }
    return (
        wanted,
        np.asarray([archive["z_id"][index[value]] for value in wanted], np.float32),
        np.asarray([archive["z_dom"][index[value]] for value in wanted], np.float32),
    )


def test_54_row_packet_prediction_truth_score_closure():
    archive = _archive()
    packet, truth = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    assert len(packet["rows"]) == ROW_COUNT == 54
    assert {row["K"] for row in packet["rows"]} == set(K_VALUES)
    assert all(row["resource"]["query_rows_used_for_fit"] == 0 for row in packet["rows"])
    ids, zid, zdom = _query(packet, archive)
    prediction = predict_packet(packet, ids, zid, zdom)
    assert all(
        tuple(row["before"]) == ARMS and tuple(row["after"]) == ARMS
        for row in prediction["rows"]
    )
    score = score_packet(
        packet,
        prediction,
        truth,
        commit=prediction["COMMIT"],
        truth_sha256=truth["truth_sha256"],
    )
    assert len(score["metrics"]) == ROW_COUNT
    assert [item["K"] for item in score["summary_by_K"]] == list(K_VALUES)
    assert all(
        item["wrong_to_correct"] >= 0
        and item["correct_to_wrong"] >= 0
        and type(item["gate_pass"]) is bool
        for item in score["summary_by_K"]
    )


def test_query_order_is_invariant_and_truth_forgery_is_rejected():
    archive = _archive()
    packet, truth = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    ids, zid, zdom = _query(packet, archive)
    normal = predict_packet(packet, ids, zid, zdom)
    order = np.arange(len(ids))[::-1]
    reversed_prediction = predict_packet(
        packet,
        [ids[index] for index in order],
        zid[order],
        zdom[order],
    )
    assert normal["rows"] == reversed_prediction["rows"]
    forged = copy.deepcopy(truth)
    first = forged["rows"][0]
    key = next(iter(first["query_labels"]))
    first["query_labels"][key] = packet["classes"][-1]
    with pytest.raises(SCXMapHeldError, match="seal"):
        score_packet(
            packet,
            normal,
            forged,
            commit=normal["COMMIT"],
            truth_sha256=truth["truth_sha256"],
        )
    resigned = copy.deepcopy(truth)
    resigned["rows"][0]["query_labels"][key] = packet["classes"][-1]
    resigned.pop("truth_sha256")
    resigned["truth_sha256"] = scx._sha(resigned)
    with pytest.raises(SCXMapHeldError, match="seal"):
        score_packet(
            packet,
            normal,
            resigned,
            commit=normal["COMMIT"],
            truth_sha256=resigned["truth_sha256"],
        )


def test_packet_and_query_pair_fail_closed():
    archive = _archive()
    packet, _ = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    ids, zid, zdom = _query(packet, archive)
    forged = copy.deepcopy(packet)
    forged["rows"][0]["K"] = 2
    with pytest.raises(SCXMapHeldError, match="packet"):
        predict_packet(forged, ids, zid, zdom)
    with pytest.raises(SCXMapHeldError, match="paired"):
        predict_packet(packet, ids, zid, zdom.astype(np.float64))
    changed_zdom = zdom.copy()
    changed_zdom[0, 0] += np.float32(0.25)
    with pytest.raises(SCXMapHeldError, match="feature bytes"):
        predict_packet(packet, ids, zid, changed_zdom)


def test_prediction_logits_are_independently_bound_to_argmax():
    archive = _archive()
    packet, truth = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    ids, zid, zdom = _query(packet, archive)
    prediction = predict_packet(packet, ids, zid, zdom)
    forged = copy.deepcopy(prediction)
    payload = forged["rows"][0]["after"]["M0"]
    logits = r2._decode_array(payload["logits"])
    winner = int(np.argmax(logits[0]))
    payload["prediction"][0] = payload["classes"][(winner + 1) % len(payload["classes"])]
    forged.pop("COMMIT")
    forged["COMMIT"] = scx._sha(forged)
    with pytest.raises(SCXMapHeldError, match="argmax/logit"):
        score_packet(
            packet,
            forged,
            truth,
            commit=forged["COMMIT"],
            truth_sha256=truth["truth_sha256"],
        )


def test_class_renaming_preserves_physical_split_and_full_predictions():
    archive = _archive()
    packet, _ = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    renamed = copy.deepcopy(archive)
    mapping = {f"c{index}": f"tx{5 - index}" for index in range(6)}
    renamed["labels"] = np.asarray([mapping[value] for value in archive["labels"]])
    renamed["class_ids"] = np.asarray(sorted(mapping.values()))
    renamed_packet, _ = build_packet(
        renamed, coverage_sha256=SHA, artifact_binding=BINDING
    )
    original = {
        (row["scene"], row["K"], row["pseudo_new"]): frozenset(row["query_ids"])
        for row in packet["rows"]
    }
    after = {
        (row["scene"], row["K"], next(k for k, v in mapping.items() if v == row["pseudo_new"])): frozenset(
            row["query_ids"]
        )
        for row in renamed_packet["rows"]
    }
    assert original == after
    ids, zid, zdom = _query(packet, archive)
    renamed_ids, renamed_zid, renamed_zdom = _query(renamed_packet, renamed)
    prediction = predict_packet(packet, ids, zid, zdom)
    renamed_prediction = predict_packet(
        renamed_packet, renamed_ids, renamed_zid, renamed_zdom
    )
    original_rows = {
        (
            packet_row["pseudo_new"],
            packet_row["scene"],
            packet_row["K"],
        ): prediction_row
        for packet_row, prediction_row in zip(
            packet["rows"], prediction["rows"]
        )
    }
    renamed_rows = {
        (
            next(k for k, v in mapping.items() if v == packet_row["pseudo_new"]),
            packet_row["scene"],
            packet_row["K"],
        ): prediction_row
        for packet_row, prediction_row in zip(
            renamed_packet["rows"], renamed_prediction["rows"]
        )
    }
    for cell, original_row in original_rows.items():
        renamed_row = renamed_rows[cell]
        for stage in ("before", "after"):
            for arm in ARMS:
                original_by_id = dict(
                    zip(
                        original_row["query_ids"],
                        original_row[stage][arm]["prediction"],
                    )
                )
                renamed_by_id = dict(
                    zip(
                        renamed_row["query_ids"],
                        renamed_row[stage][arm]["prediction"],
                    )
                )
                assert {
                    physical_id: mapping[predicted]
                    for physical_id, predicted in original_by_id.items()
                } == renamed_by_id
                assert (
                    original_row[stage][arm]["top_score_tie_rows"]
                    == renamed_row[stage][arm]["top_score_tie_rows"]
                )


def test_external_build_receipt_binds_packet_truth_and_query_files(tmp_path):
    archive = _archive()
    packet, truth = build_packet(
        archive, coverage_sha256=SHA, artifact_binding=BINDING
    )
    packet_path = tmp_path / "packet.json"
    truth_path = tmp_path / "truth.json"
    query_path = tmp_path / "query.npz"
    packet_path.write_bytes(scx._canon(packet) + b"\n")
    truth_path.write_bytes(scx._canon(truth) + b"\n")
    ids, zid, zdom = _query(packet, archive)
    with query_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            query_ids=np.asarray(ids, dtype=np.str_),
            z_id=zid,
            z_dom=zdom,
        )
    receipt = {
        "schema": scx.BUILD_RECEIPT_SCHEMA,
        "candidate": scx.CANDIDATE,
        "evaluation_scope": scx.SCOPE,
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "target25_release_authorized": False,
        "packet_file_sha256": scx._sha_file(packet_path),
        "truth_file_sha256": scx._sha_file(truth_path),
        "query_file_sha256": scx._sha_file(query_path),
        "packet_sha256": packet["packet_sha256"],
        "packet_core_sha256": packet["packet_core_sha256"],
        "truth_commitment_sha256": packet["truth_commitment_sha256"],
        "query_binding_sha256": packet["query_binding_sha256"],
    }
    receipt["receipt_sha256"] = scx._sha(receipt)
    receipt_path = tmp_path / "build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_file_sha = scx._sha_file(receipt_path)
    scx._verify_build_receipt(
        receipt,
        receipt_file_sha256=receipt_file_sha,
        packet=packet,
        packet_file_sha256=scx._sha_file(packet_path),
        query_file_sha256=scx._sha_file(query_path),
        truth_file_sha256=scx._sha_file(truth_path),
    )
    with pytest.raises(SCXMapHeldError, match="build receipt"):
        scx._verify_build_receipt(
            receipt,
            receipt_file_sha256=receipt_file_sha,
            packet=packet,
            packet_file_sha256=scx._sha_file(packet_path),
            query_file_sha256="0" * 64,
            truth_file_sha256=scx._sha_file(truth_path),
        )
