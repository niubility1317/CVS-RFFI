from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.grb_jp4_cfm_phase1_held_falsifier as held


COVERAGE_SHA256 = hashlib.sha256(b"grb-jp4-cfm-held54-coverage").hexdigest()
ARTIFACT_BINDING = {
    "archive_schema": "cvs.phase1.jp4_tap_archive.v1",
    "archive_sha256": hashlib.sha256(b"tap-archive").hexdigest(),
    "manifest_sha256": hashlib.sha256(b"tap-manifest").hexdigest(),
    "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    "coverage_sha256": COVERAGE_SHA256,
}


def _tap_archive() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(60720260724)
    classes = tuple(f"opaque-{index}" for index in range(6))
    receivers = tuple(f"rx-{index}" for index in range(4))
    scenes = held.SCENES
    rows: dict[str, list[object]] = {
        name: []
        for name in (
            "z_id",
            "hidden",
            "pre_relu",
            "labels",
            "receiver_ids",
            "day_ids",
            "physical_ids",
            "scenario_names",
            "observation_ids",
        )
    }
    for receiver_index, receiver in enumerate(receivers):
        for scene_index, scene in enumerate(scenes):
            for class_index, class_id in enumerate(classes):
                for sample_index in range(12):
                    pre = np.full(160, -0.04, dtype=np.float32)
                    pre[class_index] = np.float32(
                        (
                            0.05
                            if receiver_index == 2
                            else 1.0 + 0.015 * receiver_index
                        )
                        + 0.004 * sample_index
                    )
                    if receiver_index == 2:
                        # The coverage-selected held receiver is deliberately
                        # hard: LOO neighbors share sample geometry across
                        # labels, so the ground-off CFM equation is non-empty.
                        pre[100 + sample_index] = np.float32(1.0)
                    pre[20 + scene_index] = np.float32(
                        0.10 + 0.003 * sample_index
                    )
                    pre[40 + receiver_index] = np.float32(
                        0.06 + 0.002 * class_index
                    )
                    pre[80:88] += rng.normal(0.0, 0.005, 8).astype(np.float32)
                    hidden = rng.normal(0.0, 0.15, 320).astype(np.float32)
                    z_id = np.maximum(pre, np.float32(0.0))
                    physical = (
                        f"{receiver}-{scene}-{class_id}-physical-{sample_index:02d}"
                    )
                    values = (
                        z_id,
                        hidden,
                        pre,
                        class_id,
                        receiver,
                        f"day-{scene_index}",
                        physical,
                        scene,
                        f"obs-{physical}",
                    )
                    for name, value in zip(rows, values):
                        rows[name].append(value)
    joint_weight = rng.normal(0.0, 0.02, (160, 320)).astype(np.float32)
    joint_weight[:, :160] += np.eye(160, dtype=np.float32)
    return {
        "z_id": np.asarray(rows["z_id"], dtype=np.float32),
        "hidden": np.asarray(rows["hidden"], dtype=np.float32),
        "pre_relu": np.asarray(rows["pre_relu"], dtype=np.float32),
        "joint_weight": joint_weight,
        "labels": np.asarray(rows["labels"]),
        "receiver_ids": np.asarray(rows["receiver_ids"]),
        "day_ids": np.asarray(rows["day_ids"]),
        "physical_ids": np.asarray(rows["physical_ids"]),
        "scenario_names": np.asarray(rows["scenario_names"]),
        "class_ids": np.asarray(classes),
        "observation_ids": np.asarray(rows["observation_ids"]),
    }


@pytest.fixture(scope="module")
def closure():
    packet, query, truth = held.build_packet(
        _tap_archive(),
        coverage_sha256=COVERAGE_SHA256,
        artifact_binding=ARTIFACT_BINDING,
    )
    prediction = held.predict_packet(packet, query)
    score = held.score_packet(
        packet,
        prediction,
        truth,
        commit=prediction["COMMIT"],
        truth_sha256=truth["truth_sha256"],
    )
    return packet, query, truth, prediction, score


def test_held54_packet_query_truth_prediction_score_are_separate_and_closed(closure):
    packet, query, truth, prediction, score = closure
    assert packet["evaluation_scope"] == held.SCOPE
    assert packet["target25_authorized"] is False
    assert len(packet["rows"]) == held.ROW_COUNT == 54
    assert {
        (row["pseudo_new"], row["scene"], row["K"]) for row in packet["rows"]
    } == {
        (pseudo_new, scene, k_shot)
        for pseudo_new in packet["classes"]
        for scene in held.SCENES
        for k_shot in held.K_VALUES
    }
    assert len({row["row_id"] for row in packet["rows"]}) == held.ROW_COUNT
    assert set(query) == {
        "schema",
        "candidate",
        "evaluation_scope",
        "packet_core_sha256",
        "query_ids",
        "z_id",
        "hidden",
        "pre_relu",
        "query_binding_sha256",
    }
    assert "labels" not in query
    assert all("query_labels" not in row for row in packet["rows"])
    assert all(set(row) == {"row_id", "query_labels"} for row in truth["rows"])
    assert all(
        tuple(row["after"]) == held.ARMS
        and tuple(row["counterfactuals"]) == held.COUNTERFACTUALS
        for row in prediction["rows"]
    )
    assert len(score["metrics"]) == held.ROW_COUNT
    assert len(score["summary_by_K"]) == 2 * len(held.K_VALUES)
    assert len(score["summary_by_K_scene"]) == (
        2 * len(held.K_VALUES) * len(held.SCENES)
    )
    assert len(score["summary_by_K_pseudo_new"]) == (
        2 * len(held.K_VALUES) * len(packet["classes"])
    )
    assert score["target25_authorized"] is False
    assert score["promotion_scope"] == held.SCOPE


def test_k1_d92_is_exact_qknn_fallback_and_all_classes_compete(closure):
    packet, _query, _truth, prediction, _score = closure
    by_id = {row["row_id"]: row for row in packet["rows"]}
    for predicted in prediction["rows"]:
        packet_row = by_id[predicted["row_id"]]
        if packet_row["K"] != 1:
            continue
        assert predicted["after"]["M92"] == predicted["after"]["M0"]
        assert predicted["after"]["M_DA92"] == predicted["after"]["M_DA"]
        for name in held.COUNTERFACTUALS:
            assert (
                predicted["counterfactuals"][name]["M_DA92"]
                == predicted["counterfactuals"][name]["M_DA"]
            )
        for arm in held.ARMS:
            assert predicted["after"][arm]["classes"] == packet["classes"]
    assert all(
        row["d92_k1_exact_qknn_fallback"] is True
        for row in packet["rows"]
        if row["K"] == 1
    )


def test_resources_report_all_four_required_levels_and_fail_closed(closure):
    packet, _query, _truth, _prediction, score = closure
    for row in packet["rows"]:
        resource = row["resource"]
        assert resource["update_factor_wire_bytes"] <= 4096
        assert resource["ground_wire_bytes"] > 0
        assert resource["total_component_bytes"] >= (
            resource["update_factor_wire_bytes"]
            + resource["ground_wire_bytes"]
        )
        assert resource["support_fit_mac_upper_bound"] < 65_000_000
        assert (
            resource["support_fit_mac_upper_bound"]
            < resource["support_fit_mac_limit"]
        )
        assert set(resource["full_arm_state_bytes"]) == set(held.ARMS)
        assert (
            resource["full_arm_state_bytes"]["M92"]
            > resource["full_arm_state_bytes"]["M0"]
        )
        assert (
            resource["full_arm_state_bytes"]["M_DA92"]
            > resource["full_arm_state_bytes"]["M_DA"]
        )
        assert all(
            value <= resource["arm_state_limit_bytes"]
            for value in resource["full_arm_state_bytes"].values()
        )
        assert (
            resource["full_arm_post_backbone_mac_per_query"]["M92"]
            > resource["full_arm_post_backbone_mac_per_query"]["M0"]
            if row["K"] > 1
            else resource["full_arm_post_backbone_mac_per_query"]["M92"]
            == resource["full_arm_post_backbone_mac_per_query"]["M0"]
        )
        assert (
            resource["full_arm_post_backbone_mac_per_query"]["M_DA92"]
            > resource["full_arm_post_backbone_mac_per_query"]["M_DA"]
            if row["K"] > 1
            else resource["full_arm_post_backbone_mac_per_query"]["M_DA92"]
            == resource["full_arm_post_backbone_mac_per_query"]["M_DA"]
        )
        assert all(
            value <= resource["post_backbone_mac_limit_per_query"]
            for value in resource["full_arm_post_backbone_mac_per_query"].values()
        )
    assert score["resource_gate_pass"] is True


def test_packet_prediction_truth_and_query_tamper_fail_closed(closure):
    packet, query, truth, prediction, _score = closure
    changed_packet = copy.deepcopy(packet)
    changed_packet["rows"][0]["K"] = 5 if packet["rows"][0]["K"] != 5 else 10
    with pytest.raises(held.GRBJP4HeldError, match="packet"):
        held.predict_packet(changed_packet, query)

    changed_query = copy.deepcopy(query)
    changed_query["z_id"] = np.array(query["z_id"], copy=True)
    changed_query["z_id"][0, 0] += np.float32(0.01)
    with pytest.raises(held.GRBJP4HeldError, match="query"):
        held.predict_packet(packet, changed_query)

    changed_prediction = copy.deepcopy(prediction)
    changed_prediction["rows"][0]["after"]["M0"]["prediction"][0] = packet[
        "classes"
    ][-1]
    with pytest.raises(held.GRBJP4HeldError, match="prediction"):
        held.score_packet(
            packet,
            changed_prediction,
            truth,
            commit=prediction["COMMIT"],
            truth_sha256=truth["truth_sha256"],
        )

    changed_truth = copy.deepcopy(truth)
    first_id = next(iter(changed_truth["rows"][0]["query_labels"]))
    changed_truth["rows"][0]["query_labels"][first_id] = packet["classes"][-1]
    with pytest.raises(held.GRBJP4HeldError, match="truth"):
        held.score_packet(
            packet,
            prediction,
            changed_truth,
            commit=prediction["COMMIT"],
            truth_sha256=truth["truth_sha256"],
        )


def test_external_build_receipt_binds_all_files_and_refuses_overwrite(
    closure, tmp_path: Path
):
    packet, query, truth, _prediction, _score = closure
    receipt = held.write_build_artifacts(tmp_path / "build", packet, query, truth)
    assert set(receipt) == {
        "schema",
        "candidate",
        "evaluation_scope",
        "target25_authorized",
        "packet_file_sha256",
        "query_file_sha256",
        "truth_file_sha256",
        "packet_sha256",
        "packet_core_sha256",
        "query_binding_sha256",
        "truth_commitment_sha256",
        "receipt_sha256",
    }
    loaded_packet, loaded_query, loaded_truth = held.load_build_artifacts(
        tmp_path / "build"
    )
    assert loaded_packet["packet_sha256"] == packet["packet_sha256"]
    assert loaded_query["query_binding_sha256"] == query["query_binding_sha256"]
    assert loaded_truth["truth_sha256"] == truth["truth_sha256"]
    with pytest.raises(FileExistsError):
        held.write_build_artifacts(tmp_path / "build", packet, query, truth)
    query_path = tmp_path / "build" / held.QUERY_NAME
    query_path.write_bytes(query_path.read_bytes() + b"tamper")
    with pytest.raises(held.GRBJP4HeldError, match="receipt|SHA256"):
        held.load_build_artifacts(tmp_path / "build")


def test_prediction_side_loader_hashes_but_never_parses_truth(
    closure, tmp_path: Path, monkeypatch
):
    packet, query, truth, _prediction, _score = closure
    root = tmp_path / "prediction-inputs"
    held.write_build_artifacts(root, packet, query, truth)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name == held.TRUTH_NAME:
            raise AssertionError("prediction-side loader parsed truth")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    loaded_packet, loaded_query = held.load_prediction_inputs(root)
    assert loaded_packet["packet_sha256"] == packet["packet_sha256"]
    assert loaded_query["query_binding_sha256"] == query["query_binding_sha256"]


def test_three_fixed_counterfactuals_are_distinct_and_seed_locked(closure):
    packet, _query, _truth, _prediction, _score = closure
    assert packet["counterfactual_seed"] == 60720260724
    assert tuple(packet["ground_variants"]) == (
        "real_q4",
        "tx_permuted",
        "equal_energy_random_q4",
    )
    assert (
        packet["ground_variants"]["real_q4"]["digest"]
        != packet["ground_variants"]["tx_permuted"]["digest"]
    )
    assert (
        packet["ground_variants"]["real_q4"]["digest"]
        != packet["ground_variants"]["equal_energy_random_q4"]["digest"]
    )
    for row in packet["rows"]:
        assert tuple(row["fit_states"]) == (
            "real_q4",
            "ground_off",
            "tx_permuted",
            "equal_energy_random_q4",
        )
        assert all(
            item["query_rows_used_for_fit"] == 0
            for item in row["fit_state_receipts"].values()
        )


def test_matched_gate_requires_every_metric_and_beats_all_pseudocontrols():
    records = []
    for index in range(3):
        records.append(
            {
                "K": 5,
                "scene": "leo_clear_weak",
                "pseudo_new": f"p{index}",
                "comparison": "G_DA",
                "neighbor_membership_changes": 2,
                "argmax_changes": 2,
                "wrong_to_correct": 2,
                "correct_to_wrong": 0,
                "old_after_delta": 0.01,
                "seen_new_delta": 0.01,
                "H_delta": 0.01,
                "floor_delta": 0.0,
                "min_new_delta": 0.01,
                "per_old_class_deltas": {"a": 0.0, "b": 0.01},
                "forgetting_delta": -0.01,
                "loco_stable": True,
                "counterfactual_net_corrections": {
                    name: 0 for name in held.COUNTERFACTUALS
                },
                "counterfactual_H_deltas": {
                    name: 0.0 for name in held.COUNTERFACTUALS
                },
            }
        )
    summary = held._matched_causal_summary(records, ("K",), "G_DA")
    assert summary[0]["gate_pass"] is True
    broken = copy.deepcopy(records)
    broken[0]["per_old_class_deltas"]["a"] = -0.01
    assert held._matched_causal_summary(broken, ("K",), "G_DA")[0][
        "gate_pass"
    ] is False


def test_label_permutation_audit_is_seeded_refit_and_full_matrix_equivariant(
    closure,
):
    packet, query, truth, prediction, _score = closure
    audit = held.audit_label_permutation(
        _tap_archive(),
        coverage_sha256=COVERAGE_SHA256,
        artifact_binding=ARTIFACT_BINDING,
        packet=packet,
        query=query,
        truth=truth,
        prediction=prediction,
        score=_score,
    )
    assert audit["seed"] == 60720260724
    assert audit["old_group_fisher_yates"] is True
    assert audit["new_group_fisher_yates"] is True
    assert audit["new_group_singleton"] is True
    assert audit["theta_bytes_equal"] is True, audit["theta_mismatches"]
    assert audit["resource_receipts_equal"] is True
    assert audit["unlabeled_numeric_state_equal"] is True
    assert audit["adapted_numeric_features_equal"] is True
    assert audit["predictions_equal_after_inverse"] is True
    assert audit["metrics_equal_after_inverse"] is True
    assert audit["gates_equal_after_inverse"] is True
    assert audit["held_receiver_and_split_receipts_equal"] is True
    assert audit["gate_pass"] is True


def test_label_permutation_audit_rejects_a_class_id_numeric_branch(
    closure, monkeypatch
):
    packet, query, truth, prediction, score = closure
    old = [value for value in packet["classes"] if value != packet["classes"][-1]]
    generator = np.random.Generator(np.random.PCG64(60720260724))
    permutation = generator.permutation(len(old))
    if np.array_equal(permutation, np.arange(len(old))):
        permutation = np.roll(permutation, -1)
    branched_first_handle = old[int(permutation[0])]
    original_fit = held._fit_state

    def class_branched_fit(*args, old, ground_off, **kwargs):
        # This emulates an illegal numerical branch keyed to one opaque handle.
        return original_fit(
            *args,
            old=old,
            ground_off=(
                True
                if not ground_off and old[0] == branched_first_handle
                else ground_off
            ),
            **kwargs,
        )

    monkeypatch.setattr(held, "_fit_state", class_branched_fit)
    audit = held.audit_label_permutation(
        _tap_archive(),
        coverage_sha256=COVERAGE_SHA256,
        artifact_binding=ARTIFACT_BINDING,
        packet=packet,
        query=query,
        truth=truth,
        prediction=prediction,
        score=score,
    )
    assert audit["gate_pass"] is False
    assert not all(
        audit[name]
        for name in (
            "theta_bytes_equal",
            "adapted_numeric_features_equal",
            "predictions_equal_after_inverse",
            "metrics_equal_after_inverse",
        )
    )
