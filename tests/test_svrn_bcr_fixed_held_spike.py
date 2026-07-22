import base64
import copy
import dataclasses
import hashlib
import json

import numpy as np
import pytest

import cvsrffi.stage2_svrn_bcr as core
import cvsrffi.svrn_bcr_fixed_held_spike as held
from cvsrffi.r2a_fixed_held_four_arm import COVERAGE_SCHEMA, DUAL_ARCHIVE_SCHEMA, REAL_CLASS_IDS, SCENES


# First eight bytes are zero, so the frozen receiver selector chooses the
# lexicographically first receiver, matching the sealed r8 value ``1-1``.
SHA = "0" * 64
BINDING = {
    "archive_schema": DUAL_ARCHIVE_SCHEMA,
    "coverage_schema": COVERAGE_SCHEMA,
    "archive_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "coverage_sha256": SHA,
}


def qlock(k=5):
    if k == 5:
        return held._qknn_lock(SHA)
    token = hashlib.sha256(f"svrn-test-{k}".encode()).hexdigest()
    return core.Phase1ZIDStudentTLock(k, 3.0, 160, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0, token, token)


def support(k=5, classes=REAL_CLASS_IDS):
    rows = []; labels = []; ids = []
    for ci, label in enumerate(classes):
        for shot in range(k):
            value = np.zeros(160, np.float32)
            value[ci] = 1.0
            value[20 + shot % 10] = 0.03 * (shot + 1)
            value[50 + (ci * 7 + shot) % 40] = 0.01 * (ci + 1)
            value[100 + shot % 5] = -0.008 * (shot + 1)
            rows.append(value); labels.append(label); ids.append(f"{label}-p{shot}")
    return np.asarray(rows, np.float32), labels, ids


def archive():
    receivers = ("1-1", "zz-r1", "zz-r2", "zz-r3", "zz-r4", "zz-r5", "zz-r6")
    fields = {key: [] for key in ("z_id", "z_dom", "labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names")}
    for ri, receiver in enumerate(receivers):
        for ci, label in enumerate(REAL_CLASS_IDS):
            for si, scene in enumerate(SCENES):
                for sample in range(8):
                    zid = np.zeros(160, np.float32)
                    zid[ci] = 1.0
                    zid[12 + si] = 0.08 * (si + 1)
                    zid[30 + ri] = 0.015 * (ci + 1)
                    zid[60 + sample] = 0.02 * (sample + 1)
                    zid[100 + (ci * 5 + sample) % 45] = -0.004 * (ri + 1)
                    zdom = np.zeros(160, np.float32); zdom[ri] = 1.0; zdom[70 + si] = 0.2
                    values = {
                        "z_id": zid, "z_dom": zdom, "labels": label,
                        "receiver_ids": receiver, "day_ids": f"d{sample % 4}",
                        "physical_ids": f"{receiver}-{label}-{scene}-{sample}",
                        "scenario_names": scene,
                    }
                    for key, value in values.items(): fields[key].append(value)
    return {**{key: np.asarray(value) for key, value in fields.items()}, "class_ids": np.asarray(REAL_CLASS_IDS)}


def query(packet, data):
    ids = sorted({value for row in packet["rows"] for value in row["query_ids"]})
    index = {value: i for i, value in enumerate(data["physical_ids"].astype(str).tolist())}
    return ids, np.asarray([data["z_id"][index[value]] for value in ids], np.float32)


def resign_packet(packet):
    packet["packet_sha256"] = held._digest({key: value for key, value in packet.items() if key != "packet_sha256"})


def resign_prediction(prediction):
    prediction["COMMIT"] = held._digest({key: value for key, value in prediction.items() if key != "COMMIT"})


def resign_truth(truth):
    truth["truth_sha256"] = held._digest({key: value for key, value in truth.items() if key != "truth_sha256"})


@pytest.fixture(scope="module")
def built():
    data = archive()
    packet, truth = held.build_packet(data, coverage_sha256=SHA, artifact_binding=BINDING)
    ids, zid = query(packet, data)
    prediction = held.predict_packet(packet, ids, zid)
    metrics = held.score_packet(packet, prediction, truth, commit=prediction["COMMIT"], truth_sha256=truth["truth_sha256"])
    return data, packet, truth, ids, zid, prediction, metrics


def test_s02_s05_transform_eta_masks_and_k1_fallback():
    rows, labels, ids = support()
    transformed = core.svrn_transform(rows, 0.25)
    x = rows.astype(np.float64); centered = x - x.mean(1, keepdims=True)
    ln = centered / np.sqrt(np.mean(centered * centered, 1, keepdims=True) + 1e-6)
    clipped = np.clip(ln, -2.5, 2.5)
    restored = np.linalg.norm(x, axis=1, keepdims=True) * clipped / (np.linalg.norm(clipped, axis=1, keepdims=True) + 1e-6)
    assert np.allclose(transformed, ((0.75 * x + 0.25 * restored).astype(np.float32)), rtol=0, atol=0)
    receipt = core.select_svrn_eta(rows, labels, REAL_CLASS_IDS, ids, active_k=5)
    assert receipt["eta_grid"] == [0.0, 0.25, 0.5]
    assert receipt["mask_residues"] == [0, 1] and receipt["mask_retention"] == 0.8
    assert receipt["loo_center_count"] == 4 and receipt["same_physical_id_synchronous_loo"] is True
    one, one_labels, one_ids = support(1)
    k1 = core.select_svrn_eta(one, one_labels, REAL_CLASS_IDS, one_ids, active_k=1)
    assert k1["selected_eta"] == 0.0 and k1["fallback"] == "K1_identity" and k1["loo_center_count"] == 0
    state = core.build_branch_state(one, one_labels, REAL_CLASS_IDS, one_ids, qknn_config=qlock(1), branch="raw")
    assert state.bcrr_receipt["omega_q"] == 0.0 and state.bcrr_receipt["fallback"] == "K1_identity"
    bad = dict(receipt); bad["kappa"] = 3.0; bad["receipt_sha256"] = core._digest(core._eta_body(bad))
    with pytest.raises(core.SVRNBCRStateError): core.verify_eta_receipt(bad)


def test_r04_r05_r07_score_geometry_omega_floor_and_permutation():
    rows, labels, ids = support()
    eta = core.select_svrn_eta(rows, labels, REAL_CLASS_IDS, ids, active_k=5)
    raw = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="raw")
    svrn = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="svrn", eta_receipt=eta)
    assert raw.resource["bcr_factorizations"] == 1 and raw.resource["bcr_loo_full_d3_count"] == 0
    assert raw.resource["optimizer_steps"] == 0 and raw.resource["persistent_fp32_sidecar_bytes"] == 0
    assert raw.bcr_weight_codes_qint8.dtype == np.int8 and raw.bcr_weight_scales_fp16.dtype == np.float16
    wire = core.serialize_branch_state(raw)
    assert b'"persistent_fp32_sidecar_bytes":0' in wire and b'"dtype":"<f4"' not in wire
    restored = core.deserialize_branch_state(wire)
    assert restored.receipt_sha256 == raw.receipt_sha256
    order = np.arange(len(rows))[::-1]
    permuted = core.build_branch_state(rows[order], [labels[i] for i in order], REAL_CLASS_IDS, [ids[i] for i in order], qknn_config=qlock(), branch="raw")
    assert np.array_equal(permuted.bcr_weight_codes_qint8, raw.bcr_weight_codes_qint8)
    assert np.array_equal(permuted.bcr_weight_scales_fp16, raw.bcr_weight_scales_fp16)
    score = np.asarray([[3.0, -2.0, 1.0], [5.0, 2.0, -1.0]])
    assert np.allclose(core.normalize_score_rows(score), core.normalize_score_rows(7.0 * score + 13.0), rtol=0, atol=1e-14)
    receipt = raw.bcrr_receipt
    assert 0.0 <= receipt["omega_star"] <= 0.5 and 0.0 <= receipt["omega_q"] <= 0.5
    assert receipt["omega_q"] == np.floor(254.0 * receipt["omega_star"]) / 254.0
    assert receipt["quantization_denominator"] == 254
    assert receipt["support_bank_format"] == "per_row_qint8_fp16_scale_decode_l2_v1"
    assert receipt["bcr_weight_format"] == "per_column_qint8_fp16_scale_decode_v1"
    for direction in ("0_to_1", "1_to_0"):
        for label in REAL_CLASS_IDS:
            assert receipt["directional_class_loss_bcrr"][direction][label] <= receipt["directional_class_loss_qknn"][direction][label] + 1e-10
    forged = json.loads(wire.decode("ascii")); forged["persistent_fp32_sidecar_bytes"] = 4
    with pytest.raises(core.SVRNBCRStateError): core.deserialize_branch_state(core._canon(forged))
    assert svrn.eta_receipt["receipt_sha256"] == eta["receipt_sha256"]


def test_r3_inactive_bcr_deployment_zero_identity_and_active_wire_stability():
    rows, labels, ids = support()
    inactive = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="raw")
    assert inactive.bcrr_receipt["omega_q"] == 0.0
    assert not np.any(inactive.bcr_weight_codes_qint8) and np.all(inactive.bcr_weight_scales_fp16 > 0)
    audit = inactive.quantization_audit["bcr"]
    assert audit["top1_agreement"] == 1.0 and audit["large_margin_flip_count"] == 0
    assert audit["max_abs_logit_error"] == 0.0 and audit["teacher_margin_mean"] == 0.0
    qknn, fused = core.score_branch_logits(inactive, rows[:3])
    assert np.array_equal(fused, qknn)
    wire = core.serialize_branch_state(inactive)
    assert core.serialize_branch_state(core.deserialize_branch_state(wire)) == wire
    forged = json.loads(wire.decode("ascii"))
    forged["bcr_weight_codes_qint8"] = core._encode_array(np.ones_like(inactive.bcr_weight_codes_qint8))
    with pytest.raises(core.SVRNBCRStateError, match="inactive BCR"):
        core.deserialize_branch_state(core._canon(forged))
    receipt = dict(inactive.bcrr_receipt)
    receipt["omega_star"] = receipt["omega_q"] = 1.0 / core.BCRR_DENOMINATOR
    receipt["receipt_sha256"] = core._digest(core._bcrr_body(receipt))
    original = core.make_bcrr_receipt
    core.make_bcrr_receipt = lambda **kwargs: core.verify_bcrr_receipt(
        receipt, branch=kwargs["branch"], support_sha256=kwargs["support_sha256"],
    )
    try:
        active = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="raw")
    finally:
        core.make_bcrr_receipt = original
    assert np.any(active.bcr_weight_codes_qint8)
    assert hashlib.sha256(core.serialize_branch_state(active)).hexdigest() == "a0996811b9b88da7a71489d26dfa473525f56b71581e19db75ccad63e454025c"


def test_r05_r06_positive_safe_omega_uses_24_step_bisection_and_floor():
    rows, labels, _ = support()
    indices = np.repeat(np.arange(len(REAL_CLASS_IDS), dtype=np.int16), 5)
    qscore = np.full((len(indices), len(REAL_CLASS_IDS)), -0.2, np.float64)
    bscore = qscore.copy()
    for row, class_index in enumerate(indices):
        qscore[row, class_index] = 0.20; qscore[row, (class_index + 1) % len(REAL_CLASS_IDS)] = 0.19
        bscore[row, class_index] = 1.00; bscore[row, (class_index + 1) % len(REAL_CLASS_IDS)] = 0.00
    original = core._cross_view_loo_scores
    core._cross_view_loo_scores = lambda *_: ({"0_to_1": qscore, "1_to_0": qscore}, {"0_to_1": bscore, "1_to_0": bscore})
    try:
        receipt = core.make_bcrr_receipt(
            branch="raw", support_sha256="c" * 64, classes=REAL_CLASS_IDS,
            indices=indices, h=rows.astype(np.float64), qknn_config=qlock(), active_k=5,
        )
    finally:
        core._cross_view_loo_scores = original
    assert 0.0 < receipt["omega_star"] <= 0.5
    assert receipt["omega_q"] == np.floor(254.0 * receipt["omega_star"]) / 254.0
    assert receipt["omega_q"] <= receipt["omega_star"]


def test_r07_cross_view_uses_quantized_support_and_bcr_weights():
    rows, labels, _ = support()
    rows = rows + np.float32(0.01)
    indices = np.repeat(np.arange(len(REAL_CLASS_IDS), dtype=np.int16), 5)
    _, bcr_quant = core._cross_view_loo_scores(rows.astype(np.float64), indices, REAL_CLASS_IDS, qlock())
    original = core._quantize_columns
    def erased_decode(weights):
        codes, scales, decoded = original(weights)
        return codes, scales, np.zeros_like(decoded)
    core._quantize_columns = erased_decode
    try:
        _, bcr_erased = core._cross_view_loo_scores(rows.astype(np.float64), indices, REAL_CLASS_IDS, qlock())
    finally:
        core._quantize_columns = original
    assert any(not np.allclose(bcr_quant[key], bcr_erased[key]) for key in bcr_quant)


def test_r06_r08_r10_branch_local_bcrr_four_arm_and_qknn_immutability():
    rows, labels, ids = support()
    eta = core.select_svrn_eta(rows, labels, REAL_CLASS_IDS, ids, active_k=5)
    raw = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="raw")
    svrn = core.build_branch_state(rows, labels, REAL_CLASS_IDS, ids, qknn_config=qlock(), branch="svrn", eta_receipt=eta)
    raw_wire_before = core.serialize_branch_state(raw); svrn_wire_before = core.serialize_branch_state(svrn)
    qraw, raw_bcrr = core.score_branch_logits(raw, rows[:3])
    qsvrn, svrn_bcrr = core.score_branch_logits(svrn, rows[:3])
    state = {
        "raw_wire_b64": base64.b64encode(core.serialize_branch_state(raw)).decode(),
        "raw_wire_sha256": core._digest(core.serialize_branch_state(raw)),
        "svrn_wire_b64": base64.b64encode(core.serialize_branch_state(svrn)).decode(),
        "svrn_wire_sha256": core._digest(core.serialize_branch_state(svrn)),
        "eta_receipt": dict(eta), "branch_bindings": held._branch_bindings(raw, svrn), "resource": {},
    }
    scores = held._score_state(state, rows[:3])
    assert np.array_equal(held._decode_array(scores["M0"]["logits"]), qraw)
    assert np.array_equal(held._decode_array(scores["M_DA"]["logits"]), qsvrn)
    assert np.array_equal(held._decode_array(scores["M_OTHER"]["logits"]), raw_bcrr)
    assert np.array_equal(held._decode_array(scores["M_JOINT"]["logits"]), svrn_bcrr)
    assert core.serialize_branch_state(raw) == raw_wire_before and core.serialize_branch_state(svrn) == svrn_wire_before
    assert raw.bcrr_receipt["branch"] == "raw" and svrn.bcrr_receipt["branch"] == "svrn"
    raw_neighbors = core.qknn_neighbor_receipt(raw, rows[:3])
    assert raw_neighbors["query_count"] == 3 and raw_neighbors["classes"] == list(REAL_CLASS_IDS)
    assert set(item for query_order in raw_neighbors["orders"] for class_order in query_order for item in class_order) == set(raw.support_physical_ids_canonical)
    assert scores["M0"]["neighbor_receipt"] == scores["M_OTHER"]["neighbor_receipt"]
    assert scores["M_DA"]["neighbor_receipt"] == scores["M_JOINT"]["neighbor_receipt"]
    assert all(scores[arm]["bcrr_neighbor_order_changes"] == 0 for arm in held.ARMS)
    bad = copy.deepcopy(state); bad["branch_bindings"]["M_OTHER"]["bcrr_receipt_sha256"] = svrn.bcrr_receipt["receipt_sha256"]
    with pytest.raises(held.SVRNBCRFixedHeldError): held._decode_pair(bad)


def test_s14_s17_build_predict_score_and_resources(built):
    _, packet, truth, _, _, prediction, metrics = built
    assert packet["held_receiver"] == "1-1" and len(packet["rows"]) == 18
    assert len(prediction["rows"]) == 18 and len(metrics) == 72
    assert {row["arm"] for row in metrics} == set(held.ARMS)
    for packet_row in packet["rows"]:
        for stage in ("c5", "c6"):
            resource = packet_row[stage]["resource"]
            assert resource["persistent_fp32_sidecar_bytes"] == 0
            assert resource["wire_state_bytes"]["total"] <= core.MAX_STATE_BYTES
            assert resource["mac_ledger"]["bcr_factorizations_per_branch"] == 1
            assert resource["mac_ledger"]["bcr_loo_full_d3_count"] == 0
            for branch in ("raw", "svrn"):
                assert resource["quantization"][branch]["qknn"]["top1_agreement"] >= 0.995
                assert resource["quantization"][branch]["qknn"]["margin_sign_flip_count"] == 0
                assert resource["quantization"][branch]["bcr"]["top1_agreement"] >= 0.995
                assert resource["quantization"][branch]["bcr"]["large_margin_flip_count"] == 0
    for prediction_row in prediction["rows"]:
        for stage in ("before", "after"):
            arms = prediction_row[stage]
            assert arms["M0"]["neighbor_receipt"] == arms["M_OTHER"]["neighbor_receipt"]
            assert arms["M_DA"]["neighbor_receipt"] == arms["M_JOINT"]["neighbor_receipt"]
            assert sum(arms[arm]["bcrr_neighbor_order_changes"] for arm in held.ARMS) == 0
    for offset in range(0, 72, 4):
        quartet = {row["arm"]: row for row in metrics[offset:offset + 4]}
        expected = quartet["M_JOINT"]["H_old_new"] - quartet["M_DA"]["H_old_new"] - quartet["M_OTHER"]["H_old_new"] + quartet["M0"]["H_old_new"]
        assert all(row["I_syn"] == expected for row in quartet.values())
    decision = held.evaluate_stop_gates(metrics)
    assert decision["BCRR_neighbor_order_changes"] == 0
    assert 0 <= decision["positive_I_syn_slice_count"] <= 18
    assert decision["verdict"] in {"COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE", "ELIGIBLE_FOR_125_STABILITY_SCREEN"}
    assert truth["packet_sha256"] == packet["packet_sha256"]


def test_s01_s12_prediction_truth_wire_and_order_negatives(built):
    _, packet, truth, ids, zid, prediction, _ = built
    with pytest.raises(held.SVRNBCRFixedHeldError): held.predict_packet(packet, ids, zid.astype(np.float64))
    with pytest.raises(held.SVRNBCRFixedHeldError): held.predict_packet(packet, ids + [ids[0]], np.concatenate((zid, zid[:1]), axis=0))
    bad = copy.deepcopy(packet); bad["rows"][0]["c6"]["eta_receipt"]["kappa"] = 9.0; resign_packet(bad)
    with pytest.raises((held.SVRNBCRFixedHeldError, core.SVRNBCRStateError)): held.predict_packet(bad, ids, zid)
    bad = copy.deepcopy(packet); bad["rows"][0]["c6"]["resource"]["mac_ledger"]["build_total_mac"] = held.MAX_BUILD_MAC + 1; resign_packet(bad)
    with pytest.raises(held.SVRNBCRFixedHeldError): held.predict_packet(bad, ids, zid)
    bad_prediction = copy.deepcopy(prediction); bad_prediction["rows"][0], bad_prediction["rows"][1] = bad_prediction["rows"][1], bad_prediction["rows"][0]; resign_prediction(bad_prediction)
    with pytest.raises(held.SVRNBCRFixedHeldError): held.score_packet(packet, bad_prediction, truth, commit=bad_prediction["COMMIT"], truth_sha256=truth["truth_sha256"])
    bad_truth = copy.deepcopy(truth); row = bad_truth["rows"][0]; key = next(iter(row["query_labels"])); value = row["query_labels"].pop(key); row["query_labels"]["forged-query-id"] = value; resign_truth(bad_truth)
    with pytest.raises(held.SVRNBCRFixedHeldError): held.score_packet(packet, prediction, bad_truth, commit=prediction["COMMIT"], truth_sha256=bad_truth["truth_sha256"])


def test_s18_s19_fail_closed_decision_gate(built):
    metrics = copy.deepcopy(built[-1])
    decision = held.evaluate_stop_gates(metrics)
    assert set(decision) >= {"S18_pass", "S19_pass", "mean_I_syn", "verdict"}
    for row in metrics:
        row["mechanism"]["c6"]["selected_eta"] = 0.0
        row["mechanism"]["c5"]["omega_raw"] = 0.0; row["mechanism"]["c5"]["omega_svrn"] = 0.0
        row["mechanism"]["c6"]["omega_raw"] = 0.0; row["mechanism"]["c6"]["omega_svrn"] = 0.0
        if row["arm"] == "M_DA": row["transition_vs_M0"]["changed"] = 0
        if row["arm"] == "M_OTHER": row["transition_vs_M0"]["wrong_to_correct"] = 0
    failed = held.evaluate_stop_gates(metrics)
    assert failed["S18_pass"] is False
    assert {"eta_all_identity", "DA_zero_decision_change", "BCRR_all_zero", "OTHER_no_independent_positive_gain"} <= set(failed["S18_failures"])
    assert failed["verdict"] == "COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE"
    protection = copy.deepcopy(built[-1])
    for row in protection:
        if row["arm"] == "M_OTHER": row["old_to_new"] = 1.0
    protected = held.evaluate_stop_gates(protection)
    assert "M_OTHER:old_to_new" in protected["component_harm"]
