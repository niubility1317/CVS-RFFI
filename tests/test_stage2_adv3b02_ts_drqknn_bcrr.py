import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from argparse import Namespace
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cvsrffi.stage2_adv3b02_ts_drqknn_bcrr as adv
from cvsrffi.stage2_adv3b02_ts_drqknn_bcrr import (
    ADV3B02StateError, ARMS, CANDIDATE, SCENES, append_stage2_c,
    build_four_arm_states, build_stage2_b_state, head_bypass_forward,
    predict_four_arms, resource_formula, state_receipt, typed_tokens,
)
from scripts import run_adv3b02_ts_drqknn_bcrr_125 as runner
from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc_runner


def _support(k=5, classes=("old_a", "old_b"), seed=7):
    rng = np.random.default_rng(seed)
    labels = tuple(item for item in classes for _ in range(k))
    tokens = tuple(f"{item}:p{i}" for item in classes for i in range(k))
    zid = rng.normal(size=(len(labels), 160)).astype(np.float32)
    # Deliberate class-dependent nuisance plus within-class variation.
    zdom = rng.normal(scale=.3, size=(len(labels), 160)).astype(np.float32)
    for i, label in enumerate(labels):
        zdom[i, 0 if label == classes[0] else 1] += 2.0
    return zid, zdom, labels, tokens


def _state(k=5, classes=("old_a", "old_b"), seed=7):
    zid, zdom, labels, tokens = _support(k, classes, seed)
    return build_stage2_b_state(support_zid=zid, support_zdom=zdom, support_labels=labels,
                                registered_classes=classes, support_physical_tokens=tokens), (zid, zdom, labels, tokens)


def _nuisance_support(k, classes=("old_a", "old_b"), seed=2):
    rng = np.random.default_rng(seed)
    labels = tuple(item for item in classes for _ in range(k))
    tokens = tuple(f"{item}:physical:{index}" for item in classes for index in range(k))
    zid = rng.normal(size=(len(labels), 160)).astype(np.float32)
    zdom = np.zeros((len(labels), 160), np.float32)
    within = np.linspace(-2.0, 2.0, k, dtype=np.float32)
    for class_index, _item in enumerate(classes):
        for support_index in range(k):
            zdom[class_index * k + support_index, :3] = (
                within[support_index],
                -0.05 if class_index == 0 else 0.05,
                1.0,
            )
    return zid, zdom, labels, tokens


def test_typed_tokens_rejects_implicit_identifier_coercion():
    with pytest.raises(ADV3B02StateError):
        typed_tokens(np.asarray([1, 2]), name="bad")


def test_fixed_two_slot_candidate_centres_and_k1_exact_identity():
    state, values = _state(5)
    assert state.domain.q.shape == (160, 2)
    assert state.domain.a.shape == (2, 2)
    assert state.domain.rho.shape == (2,)
    assert 0.0 <= state.domain.alpha < .5
    zid, zdom, _, _ = values
    four = build_four_arm_states(support_zid=zid, support_zdom=zdom, support_labels=values[2],
                                 registered_classes=("old_a", "old_b"), support_physical_tokens=values[3])
    scores = predict_four_arms(four, query_zid=zid[:3], query_zdom=zdom[:3])
    assert set(scores) == set(ARMS)
    k1, one = _state(1)
    four_k1 = build_four_arm_states(support_zid=one[0], support_zdom=one[1], support_labels=one[2],
                                    registered_classes=("old_a", "old_b"), support_physical_tokens=one[3])
    result = predict_four_arms(four_k1, query_zid=one[0], query_zdom=one[1])
    assert k1.domain.alpha == 0.0
    typed = adv.qknn_logits(k1.id_bank, one[0])
    np.testing.assert_array_equal(result["M0"], typed)
    np.testing.assert_array_equal(result["M_DA"], result["M0"])
    np.testing.assert_array_equal(result["M_JOINT"], result["M_OTHER"])


@pytest.mark.parametrize("k_shot", [5, 10])
def test_fixed_two_slot_sw_uses_exact_k_minus_one_hand_oracle(k_shot):
    classes = ("old_a", "old_b")
    labels = tuple(item for item in classes for _ in range(k_shot))
    zdom = np.zeros((len(labels), 160), np.float32)
    for class_index in range(len(classes)):
        for support_index in range(k_shot):
            zdom[class_index * k_shot + support_index, :3] = (
                float(support_index - (k_shot - 1) / 2.0),
                float(class_index * 3),
                float((support_index % 2) - 0.5),
            )
    sw, sb, centres = adv._class_scatter_matrices(
        zdom, labels, classes, k_shot=k_shot
    )
    expected_centres = np.stack(
        [zdom[np.asarray(labels) == item].mean(axis=0) for item in classes]
    )
    expected_sw = np.zeros((160, 160), np.float64)
    biased_sw = np.zeros((160, 160), np.float64)
    for class_index, item in enumerate(classes):
        part = (
            zdom[np.asarray(labels) == item].astype(np.float64)
            - expected_centres[class_index].astype(np.float64)
        )
        expected_sw += part.T @ part / (k_shot - 1)
        biased_sw += part.T @ part / k_shot
    expected_sw /= len(classes)
    biased_sw /= len(classes)
    centred = expected_centres - expected_centres.mean(axis=0)
    expected_sb = centred.T @ centred / len(classes)
    np.testing.assert_allclose(centres, expected_centres, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sw, expected_sw, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(sb, expected_sb, rtol=0.0, atol=1.0e-12)
    assert not np.allclose(sw, biased_sw)


@pytest.mark.parametrize("k_shot", [5, 10])
def test_domain_weights_are_nonuniform_and_change_identity_score(k_shot):
    zid, zdom, labels, tokens = _nuisance_support(k_shot)
    states = build_four_arm_states(
        support_zid=zid,
        support_zdom=zdom,
        support_labels=labels,
        registered_classes=("old_a", "old_b"),
        support_physical_tokens=tokens,
    )
    dual = states["M_DA"]
    assert dual.domain.alpha > 0.0
    audit = adv.domain_weight_audit(dual, zdom[:3])
    assert audit["nonuniform_rows"] > 0
    assert audit["max_weight_span"] > 0.0
    query_zid = np.random.default_rng(91).normal(size=(6, 160)).astype(np.float32)
    query_zdom = np.zeros((6, 160), np.float32)
    query_zdom[:, :3] = (1.5, 0.05, 1.0)
    scores = predict_four_arms(
        states, query_zid=query_zid, query_zdom=query_zdom
    )
    assert np.max(np.abs(scores["M_DA"] - scores["M0"])) > 1.0e-5
    positions = np.asarray(
        [label == "old_a" for label in dual.id_bank.labels], bool
    )
    kernel = adv._typed_bank_kernel_terms(
        dual.id_bank,
        adv._unit(query_zid[:1])[0],
        class_index=0,
        positions=positions,
    )
    weights = adv._domain_weights(
        dual, adv._unit(query_zdom[:1])[0], class_index=0
    )
    assert np.ptp(weights) > 0.0
    assert not np.allclose(
        kernel + np.log(weights), kernel - np.log(float(k_shot))
    )


def test_directional_dual_loo_uses_both_masked_views_and_k_minus_one_mass():
    zid, zdom, labels, tokens = _nuisance_support(5)
    dual = build_stage2_b_state(
        support_zid=zid,
        support_zdom=zdom,
        support_labels=labels,
        registered_classes=("old_a", "old_b"),
        support_physical_tokens=tokens,
    )
    raw_qscore, bscore = adv._raw_directional_loo(dual.id_bank)
    dual_qscore = adv._directional_dual_loo(dual, raw_qscore)
    assert set(raw_qscore) == set(dual_qscore) == {"0_to_1", "1_to_0"}
    assert not np.array_equal(raw_qscore["0_to_1"], raw_qscore["1_to_0"])
    assert not np.array_equal(dual_qscore["0_to_1"], dual_qscore["1_to_0"])
    zero_domain = replace(dual.domain, alpha=0.0)
    zero_dual = replace(dual, domain=zero_domain)
    zero_score = adv._directional_dual_loo(zero_dual, raw_qscore)
    for direction in ("0_to_1", "1_to_0"):
        np.testing.assert_array_equal(zero_score[direction], raw_qscore[direction])
        assert bscore[direction].shape == raw_qscore[direction].shape
    indices = dual.id_bank.bank.class_indices_int16.astype(np.int64)
    own_class = int(indices[0])
    own_rows = (indices == own_class) & (np.arange(len(indices)) != 0)
    support_dom = dual.domain.features().astype(np.float64)[own_rows]
    class_handle = dual.id_bank.bank.classes[own_class]
    centre = dual.domain.centres[
        dual.domain.classes.index(class_handle)
    ].astype(np.float64)
    weights = adv._renormalized_domain_loo_weights(
        dual.domain,
        support_dom=support_dom,
        query_dom=dual.domain.features().astype(np.float64)[0],
        centre=centre,
    )
    assert len(weights) == dual.id_bank.k_shot - 1
    assert np.sum(weights) == pytest.approx(1.0, abs=1.0e-12)


def test_class_and_support_order_permutation_preserve_handle_predictions():
    state, (zid, zdom, labels, tokens) = _state(5)
    normal = build_four_arm_states(support_zid=zid, support_zdom=zdom, support_labels=labels,
                                   registered_classes=("old_a", "old_b"), support_physical_tokens=tokens)
    order = np.asarray(list(reversed(range(len(tokens)))), np.intp)
    swapped = build_four_arm_states(support_zid=zid[order], support_zdom=zdom[order],
                                    support_labels=tuple(labels[i] for i in order),
                                    registered_classes=("old_b", "old_a"),
                                    support_physical_tokens=tuple(tokens[i] for i in order))
    a = predict_four_arms(normal, query_zid=zid[:4], query_zdom=zdom[:4])
    b = predict_four_arms(swapped, query_zid=zid[:4], query_zdom=zdom[:4])
    for arm in ARMS:
        handle_a = np.asarray(("old_a", "old_b"))[np.argmax(a[arm], axis=1)]
        handle_b = np.asarray(("old_b", "old_a"))[np.argmax(b[arm], axis=1)]
        assert tuple(handle_a) == tuple(handle_b)
    assert set(state.id_bank.support_tokens) == set(tokens)


def test_nonlex_registered_axis_is_handle_equivalent_with_positive_bcrr_omega():
    classes = ("zeta", "alpha")
    zid, zdom, labels, tokens = _nuisance_support(5, classes=classes, seed=2)
    nonlex = build_four_arm_states(
        support_zid=zid,
        support_zdom=zdom,
        support_labels=labels,
        registered_classes=classes,
        support_physical_tokens=tokens,
    )
    lexical = build_four_arm_states(
        support_zid=zid,
        support_zdom=zdom,
        support_labels=labels,
        registered_classes=tuple(sorted(classes)),
        support_physical_tokens=tokens,
    )
    assert nonlex["M_OTHER"][1].omega > 0.0
    query_zid = np.random.default_rng(27).normal(size=(9, 160)).astype(np.float32)
    query_zdom = np.zeros((9, 160), np.float32)
    query_zdom[:, :3] = (1.25, 0.05, 1.0)
    left = predict_four_arms(
        nonlex, query_zid=query_zid, query_zdom=query_zdom
    )
    right = predict_four_arms(
        lexical, query_zid=query_zid, query_zdom=query_zdom
    )
    for arm in ("M_OTHER", "M_JOINT"):
        left_handle = np.asarray(classes)[np.argmax(left[arm], axis=1)]
        right_handle = np.asarray(tuple(sorted(classes)))[
            np.argmax(right[arm], axis=1)
        ]
        assert tuple(left_handle) == tuple(right_handle)


def test_stage2_c_is_append_only_and_preserves_old_bytes():
    state, old_values = _state(5)
    zid, zdom, labels, tokens = _support(5, ("new_c",), 22)
    before_domain = state.domain.wire_bytes()
    before_codes = state.id_bank.bank.codes_qint8.copy()
    after, receipt = append_stage2_c(state, new_support_zid=zid, new_support_zdom=zdom,
                                     new_support_labels=labels, new_registered_classes=("new_c",),
                                     new_support_physical_tokens=tokens,
                                     after_full_teacher_zid=np.concatenate((old_values[0], zid)),
                                     after_full_teacher_physical_tokens=old_values[3] + tokens)
    assert after.domain.stage == "S_C" and after.id_bank.classes == ("old_a", "old_b", "new_c")
    assert state.domain.wire_bytes() == before_domain
    np.testing.assert_array_equal(after.id_bank.bank.codes_qint8[:len(before_codes)], before_codes)
    assert receipt["old_q_a_alpha_refit"] is False
    assert receipt["old_int8_codes_preserved"] and receipt["old_int8_scales_preserved"]
    assert receipt["old_domain_bytes_preserved"]
    assert (
        receipt["old_domain_prefix_sha256_before"]
        == receipt["old_domain_prefix_sha256_after"]
    )
    assert receipt["old_domain_prefix_bytes"] > 0


def test_stage2_c_nonlex_new_registry_keeps_zid_zdom_token_row_alignment():
    old, old_values = _state(5, classes=("zeta", "alpha"), seed=31)
    zid, zdom, labels, tokens = _support(
        5, classes=("new_z", "new_a"), seed=32
    )
    after, _receipt = append_stage2_c(
        old,
        new_support_zid=zid,
        new_support_zdom=zdom,
        new_support_labels=labels,
        new_registered_classes=("new_z", "new_a"),
        new_support_physical_tokens=tokens,
        after_full_teacher_zid=np.concatenate((old_values[0], zid)),
        after_full_teacher_physical_tokens=old_values[3] + tokens,
    )
    expected_codes, _expected_scales = adv._quantize_rows(adv._unit(zdom))
    original_by_token = {token: index for index, token in enumerate(tokens)}
    first_new_row = len(old.id_bank.support_tokens)
    for result_index in range(first_new_row, len(after.id_bank.support_tokens)):
        token = after.id_bank.support_tokens[result_index]
        np.testing.assert_array_equal(
            after.domain.zdom_codes[result_index],
            expected_codes[original_by_token[token]],
        )


@pytest.mark.parametrize("k_shot", [1, 5, 10])
def test_stage2_c_actual_bank_closes_wire_bcr_loo_and_full_int8_audit(k_shot):
    old, old_values = _state(k_shot, seed=40 + k_shot)
    zid, zdom, labels, tokens = _support(
        k_shot, classes=("new_c",), seed=50 + k_shot
    )
    after, receipt = append_stage2_c(
        old,
        new_support_zid=zid,
        new_support_zdom=zdom,
        new_support_labels=labels,
        new_registered_classes=("new_c",),
        new_support_physical_tokens=tokens,
        after_full_teacher_zid=np.concatenate((old_values[0], zid)),
        after_full_teacher_physical_tokens=old_values[3] + tokens,
    )
    assert adv._serialize_affine_bank(after.id_bank.bank) == after.id_bank.qknn_wire
    assert after.id_bank.bank.offsets_fp16.dtype == np.dtype("<f2")
    branch = after.id_bank.branch_state
    assert type(branch) is adv.ActualBankBranchState
    assert branch.qknn_wire == after.id_bank.qknn_wire
    binding = branch.actual_bank_binding_receipt
    assert binding["bank_receipt_sha256"] == after.id_bank.bank.bank_receipt_sha256
    assert binding["metric_receipt_sha256"] == after.id_bank.metric.metric_receipt_sha256
    assert binding["bcr_weight_codec"] == adv.BCR_WEIGHT_CODEC
    assert binding["bcr_weight_plane_count"] == 2
    assert binding["bcr_weight_plane_order"] == list(adv.BCR_WEIGHT_PLANE_ORDER)
    assert binding["bcr_weight_code_dtype"] == "int8"
    assert binding["bcr_weight_scale_dtype"] == "<f2"
    assert binding["bcr_weight_rounding"] == "numpy_rint_ties_to_even"
    assert binding["bcr_weight_clip"] == [-127, 127]
    assert binding["bcr_weight_scale_floor"] == adv.BCR_WEIGHT_SCALE_FLOOR
    assert binding["bcr_weight_shape"] == [160, len(after.id_bank.bank.classes)]
    assert binding["bcr_weight_class_order"] == list(after.id_bank.bank.classes)
    assert binding["bcr_weight_wire_bytes"] == (
        branch.bcr_weight_codes_qint8.nbytes
        + branch.bcr_weight_scales_fp16.nbytes
        + branch.bcr_weight_residual_codes_qint8.nbytes
        + branch.bcr_weight_residual_scales_fp16.nbytes
    )
    assert branch.quantization_audit["bcr"]["top1_agreement"] == 1.0
    assert branch.quantization_audit["bcr"]["any_margin_flip_count"] == 0
    assert branch.quantization_audit["bcr"]["any_margin_flip_rate"] == 0.0
    assert set(binding["directional_loo_sha256"]) == {"0_to_1", "1_to_0"}
    audit = after.int8_audit_receipt
    assert audit["validation_row_count"] == (2 + 1) * k_shot
    assert audit["top1_agreement"] >= 0.995
    assert audit["large_margin_flip_count"] == 0
    verified = adv.verify_stage2_c_append_receipt(receipt)
    assert verified["after_state_sha256"] == after.digest
    assert (
        verified["after_branch_actual_bank_binding_sha256"]
        == binding["receipt_sha256"]
    )
    assert verified["after_int8_audit"] == dict(audit)


def test_int8_qknn_post_init_fails_closed_on_branch_bank_and_audit_drift():
    state, old_values = _state(5, seed=61)
    other, _ = _state(5, seed=62)
    with pytest.raises(ADV3B02StateError, match="affine qKNN"):
        replace(state.id_bank, bank=other.id_bank.bank)
    branch = state.id_bank.branch_state
    token_drift = SimpleNamespace(
        qknn_wire=branch.qknn_wire,
        support_physical_ids_canonical=tuple(
            reversed(branch.support_physical_ids_canonical)
        ),
        bcr_weight_codes_qint8=branch.bcr_weight_codes_qint8,
        bcr_weight_scales_fp16=branch.bcr_weight_scales_fp16,
        quantization_audit=branch.quantization_audit,
    )
    with pytest.raises(ADV3B02StateError, match="affine qKNN"):
        replace(state.id_bank, branch_state=token_drift)
    zid, zdom, labels, tokens = _support(5, classes=("new_c",), seed=63)
    after, _receipt = append_stage2_c(
        state,
        new_support_zid=zid,
        new_support_zdom=zdom,
        new_support_labels=labels,
        new_registered_classes=("new_c",),
        new_support_physical_tokens=tokens,
        after_full_teacher_zid=np.concatenate((old_values[0], zid)),
        after_full_teacher_physical_tokens=old_values[3] + tokens,
    )
    actual = after.id_bank.branch_state
    bad_audit = {
        "qknn": {**actual.quantization_audit["qknn"], "top1_agreement": 0.0},
        "bcr": dict(actual.quantization_audit["bcr"]),
    }
    with pytest.raises(ADV3B02StateError, match="affine actual"):
        replace(actual, quantization_audit=bad_audit)
    bad_binding = {
        **actual.actual_bank_binding_receipt,
        "bank_receipt_sha256": "0" * 64,
    }
    bad_binding["receipt_sha256"] = adv.sha256_bytes(
        adv._canon(
            {
                key: value
                for key, value in bad_binding.items()
                if key != "receipt_sha256"
            }
        )
    )
    bad_branch = replace(
        actual, actual_bank_binding_receipt=bad_binding
    )
    with pytest.raises(ADV3B02StateError, match="binding field drift"):
        replace(after.id_bank, branch_state=bad_branch)


def test_four_arm_byte_sharing_branch_local_loo_and_int8_receipt():
    _, values = _state(5)
    states = build_four_arm_states(support_zid=values[0], support_zdom=values[1], support_labels=values[2],
                                   registered_classes=("old_a", "old_b"), support_physical_tokens=values[3])
    receipt = state_receipt(states)
    assert states["M_OTHER"][0] is states["M0"]
    assert states["M_JOINT"][0] is states["M_DA"]
    assert receipt["raw_bcrr"]["bcrr_reads_z_dom"] is False
    assert receipt["dual_bcrr"]["bcrr_reads_z_dom"] is False
    assert receipt["dual_bcrr"]["bcr_codes_and_weights"] == "existing_branch_z_id_support_shared"
    for branch in ("raw_bcrr", "dual_bcrr"):
        directional = receipt[branch]["directional_logits_sha256"]
        assert set(directional) == {"0_to_1", "1_to_0"}
        assert all(
            set(value) == {"qknn_sha256", "bcr_sha256"}
            for value in directional.values()
        )
    assert receipt["dual_bcrr"]["omega_prelocked_safety"]["denominator"] == 254
    assert receipt["int8"]["top1_agreement"] >= .995
    assert receipt["int8"]["large_margin_flip_count"] == 0
    assert resource_formula(class_count=26, k_shot=10)["dual_domain_qknn_extra_mac_per_query"] == 840


def test_affine_wire_is_little_endian_bound_and_fail_closed():
    state, _ = _state(5, seed=77)
    bank = state.id_bank
    assert bank.bank.scales_fp16.dtype == np.dtype("<f2")
    assert bank.bank.offsets_fp16.dtype == np.dtype("<f2")
    np.testing.assert_allclose(
        bank.features(),
        adv._decode_affine_wire(bank.qknn_wire, bank.bank),
        rtol=0.0,
        atol=0.0,
    )
    broken = bytearray(bank.qknn_wire)
    broken[-1] ^= 1
    with pytest.raises(ADV3B02StateError, match="serialized bytes|wire"):
        replace(bank, qknn_wire=bytes(broken))


def test_affine_full26k10_state_stays_inside_hard_limit():
    zid, zdom, labels, tokens = _support(10, tuple(f"c{i:02d}" for i in range(26)), seed=79)
    states = build_four_arm_states(
        support_zid=zid, support_zdom=zdom, support_labels=labels,
        registered_classes=tuple(f"c{i:02d}" for i in range(26)),
        support_physical_tokens=tokens,
    )
    receipt = state_receipt(states)
    assert receipt["wire_bytes"] <= 256 * 1024
    # Two int8 [160,C] planes plus two FP16 [C] scale vectors.
    assert receipt["bcr_weight_wire_bytes"] == 8424
    assert receipt["bcr_weight_codec"] == adv.BCR_WEIGHT_CODEC
    assert receipt["bcr_weight_plane_count"] == 2


def test_affine_codec_rejects_zero_range_underflow_and_nonfinite_decode(monkeypatch):
    flat = np.ones((1, 160), np.float32)
    with pytest.raises(ADV3B02StateError, match="zero/nonfinite range"):
        adv._affine_quantize_rows(flat)
    tiny_span = np.zeros((1, 160), np.float32)
    tiny_span[0, -1] = np.float32(1.0e-8)
    assert float(np.max(tiny_span) - np.min(tiny_span)) > 0.0
    assert np.float16((np.max(tiny_span) - np.min(tiny_span)) / 254.0) == 0
    monkeypatch.setattr(adv, "_unit", lambda value: tiny_span.copy())
    with pytest.raises(ADV3B02StateError, match="FP16"):
        adv._affine_quantize_rows(np.ones((1, 160), np.float32))
    with pytest.raises(ADV3B02StateError, match="shape/dtype"):
        adv._affine_dequantize_rows(np.zeros((1, 160), np.int8), np.asarray([np.inf], "<f2"), np.zeros(1, "<f2"))


def test_affine_codec_uses_ieee_ties_to_even(monkeypatch):
    row = np.zeros((1, 160), np.float32)
    row[0, :8] = (-127.0, 127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5)
    monkeypatch.setattr(adv, "_unit", lambda _value: row.copy())
    codes, scales, offsets = adv._affine_quantize_rows(
        np.ones((1, 160), np.float32)
    )
    assert float(scales[0]) == 1.0 and float(offsets[0]) == 0.0
    assert codes[0, 2:8].tolist() == [0, 2, 2, 0, -2, -2]


def test_affine_codec_declares_ieee_ties_to_even_rounding():
    np.testing.assert_array_equal(
        np.rint(np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])),
        np.asarray([-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]),
    )


def test_bcr_two_plane_codec_uses_fixed_residual_formula_and_rounded_wire_types():
    weights = np.zeros((160, 2), np.float64)
    weights[:8, 0] = (-127.0, 127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5)
    weights[:8, 1] = (127.0, -127.0, -0.5, -1.5, -2.5, 0.5, 1.5, 2.5)
    p1c, p1s, p2c, p2s, deployed = adv._quantize_bcr_weights_two_plane(weights)
    first = p1c.astype(np.float32) * p1s.astype(np.float32)[None, :]
    residual = weights - first.astype(np.float64)
    ep2c, ep2s, ep2 = adv._quantize_bcr_weight_plane(residual)
    assert p1c.dtype == np.int8 and p2c.dtype == np.int8
    assert p1s.dtype == np.dtype("<f2") and p2s.dtype == np.dtype("<f2")
    assert p1c[:8, 0].tolist() == [-127, 127, 0, 2, 2, 0, -2, -2]
    np.testing.assert_array_equal(p2c, ep2c)
    np.testing.assert_array_equal(p2s, ep2s)
    np.testing.assert_allclose(deployed, first + ep2, rtol=0.0, atol=0.0)
    assert adv.BCR_WEIGHT_PLANE_ORDER == (
        "plane1_teacher", "plane2_residual_after_plane1_decode",
    )
    assert adv.BCR_WEIGHT_ROUNDING == "numpy_rint_ties_to_even"
    assert adv.BCR_WEIGHT_CLIP == (-127, 127)


def test_bcr_two_plane_codec_uses_fp16_subnormal_floor_without_fp32_sidecar():
    codes, scales, residual_codes, residual_scales, decoded = (
        adv._quantize_bcr_weights_two_plane(np.zeros((160, 2), np.float64))
    )
    floor = np.finfo(np.float16).smallest_subnormal
    assert np.all(scales == floor) and np.all(residual_scales == floor)
    assert not np.any(codes) and not np.any(residual_codes) and not np.any(decoded)
    field_names = {item.name for item in fields(adv.ActualBankBranchState)}
    assert "bcr_weight_codes_qint8" in field_names
    assert "bcr_weight_residual_codes_qint8" in field_names
    assert not {"bcr_weights", "bcr_weight_fp32", "bcr_weight_residual_fp32"} & field_names


def test_affine_audit_detects_third_class_argmax_overtake(monkeypatch):
    state, values = _state(1, classes=("a", "b", "c"), seed=81)
    teacher = np.asarray([[3.0, 2.0, 1.0]], np.float32)
    deployed = np.asarray([[2.9, 0.0, 3.1]], np.float32)
    monkeypatch.setattr(adv, "_score_support", lambda **_kwargs: teacher)
    monkeypatch.setattr(adv, "_score_affine_bank", lambda *_args, **_kwargs: deployed)
    audit = adv._affine_margin_audit(
        state.id_bank.bank, values[0], state.id_bank.labels, values[0][:1]
    )
    assert audit["any_margin_flip_count"] == 1
    # A third-class takeover is diagnosed, but cannot be large under the
    # frozen 2*row-max-error certificate.
    assert audit["large_margin_flip_count"] == 0


def test_actual_branch_rejects_bcr_audit_and_weight_binding_tamper():
    state, _ = _state(5, seed=82)
    branch = state.id_bank.branch_state
    for field, value in (
        ("top1_agreement", 0.0),
        ("any_margin_flip_count", 1),
        ("large_margin_flip_count", 1),
    ):
        changed = {
            "qknn": dict(branch.quantization_audit["qknn"]),
            "bcr": {**branch.quantization_audit["bcr"], field: value},
        }
        with pytest.raises(ADV3B02StateError, match="affine actual"):
            replace(branch, quantization_audit=changed)
    codes = branch.bcr_weight_codes_qint8.copy()
    codes[0, 0] = np.int8(int(codes[0, 0]) + (1 if codes[0, 0] < 127 else -1))
    with pytest.raises(ADV3B02StateError, match="receipt"):
        replace(branch, bcr_weight_codes_qint8=codes)
    scales = branch.bcr_weight_scales_fp16.copy()
    scales[0] = np.float16(float(scales[0]) * 1.5)
    with pytest.raises(ADV3B02StateError, match="receipt"):
        replace(branch, bcr_weight_scales_fp16=scales)
    residual_codes = branch.bcr_weight_residual_codes_qint8.copy()
    residual_codes[0, 0] = np.int8(
        int(residual_codes[0, 0])
        + (1 if residual_codes[0, 0] < 127 else -1)
    )
    with pytest.raises(ADV3B02StateError, match="receipt"):
        replace(branch, bcr_weight_residual_codes_qint8=residual_codes)
    residual_scales = branch.bcr_weight_residual_scales_fp16.copy()
    residual_scales[0] = np.float16(float(residual_scales[0]) * 1.5)
    with pytest.raises(ADV3B02StateError, match="receipt"):
        replace(branch, bcr_weight_residual_scales_fp16=residual_scales)
    bad_order = {
        **branch.actual_bank_binding_receipt,
        "bcr_weight_class_order": list(reversed(branch.actual_bank_binding_receipt["bcr_weight_class_order"])),
    }
    bad_order["receipt_sha256"] = adv.sha256_bytes(
        adv._canon({key: value for key, value in bad_order.items() if key != "receipt_sha256"})
    )
    ordered_branch = replace(branch, actual_bank_binding_receipt=bad_order)
    with pytest.raises(ADV3B02StateError, match="binding field drift"):
        replace(state.id_bank, branch_state=ordered_branch)


def test_bcr_two_plane_runtime_gate_rejects_subpercent_nonperfect_top1(monkeypatch):
    zid, _zdom, labels, tokens = _support(5, classes=("a", "b"), seed=82)

    def permissive_legacy_audit(_teacher, _student):
        return {
            "scope": "support_only_full_state_teacher",
            "top1_agreement": 0.999,
            "large_margin_flip_count": 0,
            "max_abs_logit_error": 0.0,
            "teacher_margin_mean": 0.0,
            "query_rows_used_for_fit": 0,
        }

    monkeypatch.setattr(adv, "_existing_bcr_quant_audit", permissive_legacy_audit)
    with pytest.raises(ADV3B02StateError, match="BCR INT8 audit gate"):
        adv.build_int8_qknn_state(
            zid,
            labels,
            ("a", "b"),
            tokens,
        )


def test_bcrr_deployment_reconstructs_exactly_the_two_int8_planes():
    zid, zdom, labels, tokens = _nuisance_support(5, classes=("zeta", "alpha"), seed=13)
    states = build_four_arm_states(
        support_zid=zid,
        support_zdom=zdom,
        support_labels=labels,
        registered_classes=("zeta", "alpha"),
        support_physical_tokens=tokens,
    )
    bank, bcrr = states["M_OTHER"]
    assert bcrr.omega > 0.0
    query = zid[:4]
    qknn = adv.qknn_logits(bank, query)
    observed = adv.bcrr_fused_logits(qknn, query, bcrr, bank=bank)
    branch = bank.branch_state
    weights = (
        branch.bcr_weight_codes_qint8.astype(np.float32)
        * branch.bcr_weight_scales_fp16.astype(np.float32)[None, :]
        + branch.bcr_weight_residual_codes_qint8.astype(np.float32)
        * branch.bcr_weight_residual_scales_fp16.astype(np.float32)[None, :]
    )
    bcr = adv.normalize_zid_rows(query) @ weights
    bcr = bcr[:, [bank.bank.classes.index(item) for item in bank.classes]]
    expected = np.empty_like(qknn)
    for index in range(len(qknn)):
        expected[index] = (
            (1.0 - bcrr.omega)
            * adv._normalize_existing_scores(qknn[index:index + 1])[0]
            + bcrr.omega
            * adv._normalize_existing_scores(bcr[index:index + 1])[0]
        )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


class _Backbone(nn.Module):
    def __init__(self, key):
        super().__init__(); self.key = key; self.proj = nn.Linear(2, 160)
    def forward(self, x, y=None, return_aux=False, domain_labels=None):
        feature = self.proj(x.mean(dim=2))
        return {self.key: feature, "logits": torch.ones((len(x), 2), dtype=torch.float32)}


class _Enhancer(nn.Module):
    def forward(self, z, x):
        return z, None


class _HeadBypassModel(nn.Module):
    id_feature_key = "feat_joint"; dom_feature_key = "feat_imp"
    def __init__(self):
        super().__init__(); self.id_backbone = _Backbone("feat_joint"); self.dom_backbone = _Backbone("feat_imp"); self.dom_enhancer = _Enhancer()
        self.dom_head = self.adv_head = self.tx_adv_head = _RaiseHead()
    def _pick_z_id(self, value): return value["feat_joint"]
    def _pick_z_dom(self, value): return value["feat_imp"]
    def forward(self, *args, **kwargs): raise AssertionError("model.forward must not run")


class _RaiseHead(nn.Module):
    def forward(self, *args, **kwargs): raise AssertionError("forbidden head called")


def test_head_bypass_never_calls_forbidden_heads():
    model = _HeadBypassModel().eval()
    z_id, z_dom, receipt = head_bypass_forward(model, torch.randn(3, 2, 4, dtype=torch.float32), checkpoint_sha256="a" * 64)
    assert z_id.shape == z_dom.shape == (3, 160)
    assert receipt["heads_called"] == 0


def _scene_metrics(state, count):
    return {
        "query_count": count,
        "old_acc": 0.75,
        "seen_new_acc": None if state == "before" else 0.5,
        "h_old_new": None if state == "before" else 0.6,
        "old_to_new_rate": 0.0,
        "new_to_old_rate": None if state == "before" else 0.25,
    }


def _fake_after_int8_audit():
    return {
        "schema": "cvs.phase2.zid_student_t_qknn.margin_audit.v2_affine",
        "validation_row_count": 10,
        "logit_abs_error_mean": 0.01,
        "logit_abs_error_max": 0.02,
        "top1_agreement": 1.0,
        "teacher_margin_mean": 1.0,
        "quantized_teacher_margin_mean": 1.0,
        "any_margin_flip_count": 0,
        "any_margin_flip_rate": 0.0,
        "large_margin_flip_count": 0,
        "large_margin_flip_rate": 0.0,
        "fp32_teacher_bandwidth_source": "complete_unquantized_FP32_all_support",
        "fp32_teacher_support_sha256": "b" * 64,
        "int8_bank_class_scales": [0.5, 0.5],
        "teacher_bank_bandwidth_abs_delta_max": 0.0,
        "query_rows_used_for_fit": 0,
        "state_updates": 0,
    }


def _fake_append_receipt():
    audit = _fake_after_int8_audit()
    body = {
        "schema": adv.APPEND_RECEIPT_SCHEMA,
        "stage": "S_C",
        "query_rows_used_for_fit": 0,
        "old_state_sha256": "b" * 64,
        "after_state_sha256": "d" * 64,
        "old_domain_digest_before": "1" * 64,
        "frozen_old_digest_in_after": "1" * 64,
        "old_domain_prefix_sha256_before": "2" * 64,
        "old_domain_prefix_sha256_after": "2" * 64,
        "old_domain_prefix_bytes": 128,
        "old_domain_bytes_preserved": True,
        "old_int8_codes_sha256_before": "3" * 64,
        "old_int8_codes_sha256_after": "3" * 64,
        "old_int8_scales_sha256_before": "4" * 64,
        "old_int8_scales_sha256_after": "4" * 64,
        "old_int8_offsets_sha256_before": "a" * 64,
        "old_int8_offsets_sha256_after": "a" * 64,
        "old_int8_class_scales_sha256_before": "5" * 64,
        "old_int8_class_scales_sha256_after": "5" * 64,
        "old_int8_codes_preserved": True,
        "old_int8_scales_preserved": True,
        "old_int8_offsets_preserved": True,
        "old_int8_class_scales_preserved": True,
        "old_q_sha256_before": "6" * 64,
        "old_q_sha256_after": "6" * 64,
        "old_a_sha256_before": "7" * 64,
        "old_a_sha256_after": "7" * 64,
        "old_rho_sha256_before": "8" * 64,
        "old_rho_sha256_after": "8" * 64,
        "old_alpha_before": 0.25,
        "old_alpha_after": 0.25,
        "old_q_a_alpha_refit": False,
        "after_bank_receipt_sha256": "9" * 64,
        "after_qknn_wire_sha256": "e" * 64,
        "after_metric_receipt_sha256": "a" * 64,
        "after_branch_actual_bank_binding_sha256": "c" * 64,
        "after_int8_audit": audit,
        "after_int8_audit_sha256": adv.sha256_bytes(adv._canon(audit)),
        "after_support_row_count": 10,
        "after_class_count": 2,
        "after_support_token_root_sha256": "f" * 64,
    }
    return {**body, "receipt_sha256": adv.sha256_bytes(adv._canon(body))}


def _runtime_rows():
    audit_sha = _fake_append_receipt()["after_int8_audit_sha256"]
    rows = []
    for scene in SCENES:
        for state in ("before", "after"):
            rows.append(
                {
                    "scene": scene,
                    "state": state,
                    "raw_qknn_sha256": (
                        "a" * 64 if state == "before" else "e" * 64
                    ),
                    "dual_qknn_sha256": (
                        "b" * 64 if state == "before" else "d" * 64
                    ),
                    "branch_qknn_wire_sha256": (
                        "a" * 64 if state == "before" else "e" * 64
                    ),
                    "branch_actual_bank_binding_sha256": (
                        None if state == "before" else "c" * 64
                    ),
                    "int8_audit_sha256": (
                        "0" * 64 if state == "before" else audit_sha
                    ),
                    "raw_state_bytes": 1000,
                    "dual_domain_state_bytes": 234,
                    "state_wire_bytes": 1234,
                    "alpha": 0.25,
                    "fixed_rank": 2,
                    "active_rank": 2,
                    "feature_latency_ms": 2.0,
                    "build_latency_ms": 1.25,
                    "predict_latency_ms": 0.75,
                    "total_latency_ms": 4.0,
                    "peak_cuda_memory_bytes": 4096,
                    "raw_vs_dual": {
                        "query_rows": 6,
                        "argmax_changed_count": 1,
                        "score_changed_count": 2,
                        "margin_changed_count": 2,
                        "max_abs_score_delta": 0.1,
                    },
                    "domain_weights": {
                        "query_class_rows": 12,
                        "nonuniform_rows": 4,
                        "max_weight_span": 0.2,
                        "mean_weight_span": 0.05,
                    },
                }
            )
    return rows


def _materialize_row(tmp_path, job):
    root = Path(job["output_root"]); publication = {"before": {}, "after": {}}
    tokens = np.asarray([f"q{i}" for i in range(6)])
    scenes = np.asarray([SCENES[i % 3] for i in range(6)])
    predicted = np.asarray(["opaque_class"] * 6)
    for state in publication:
        for arm in ARMS:
            publication[state][arm] = runner.write_prediction_new(runner.prediction_path(root, state, arm), query_tokens=tokens, scenarios=scenes, predicted_class_handles=predicted)
    scores = {}
    for arm in ARMS:
        scores[arm] = runner.write_json_new(
            runner.score_path(root, arm),
            {
                "candidate": CANDIDATE,
                "arm": arm,
                "query_truth_joined_after_prediction": True,
                "query_truth_joined_only_after_immutable_predictions": True,
                "query_truth_fed_back_to_predictor": False,
                "before_prediction_sha256": publication["before"][arm],
                "after_prediction_sha256": publication["after"][arm],
                "before": {
                    "query_count": len(tokens),
                    "by_scenario": {
                        scene: _scene_metrics("before", 2) for scene in SCENES
                    },
                },
                "after": {
                    "query_count": len(tokens),
                    "by_scenario": {
                        scene: _scene_metrics("after", 2) for scene in SCENES
                    },
                },
            },
        )
    return {"schema": runner.LAUNCHER_SCHEMA, "candidate": CANDIDATE, "job_id": job["job_id"], "status": "ROW_ARTIFACTS_COMPLETE",
            "prediction_artifact_count": 8, "scene_slice_count": 3, "score_row_count": 12,
            "query_truth_in_predictor": False, "query_rows_used_for_fit": 0,
            "device_namespace_execution": dssc_runner._expected_row_device_namespace_execution(0),
            "scene_state_runtime_receipts": _runtime_rows(),
            "append_receipts_by_scene": {
                scene: _fake_append_receipt() for scene in SCENES
            },
            "prediction_sha256_by_state_arm": publication, "score_artifact_sha256": scores}


def _rewrite_score(job, receipt, arm, mutate):
    updated = copy.deepcopy(receipt)
    path = runner.score_path(job["output_root"], arm)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.chmod(0o600)
    path.unlink()
    updated["score_artifact_sha256"][arm] = runner.write_json_new(path, payload)
    return updated


def test_full125_plan_and_artifact_closure_fail_closed_on_missing_or_tamper(tmp_path):
    jobs = runner.matrix_jobs(run_root=tmp_path / "run")
    assert len(jobs) == 125
    assert sum(job["scene_slice_count"] for job in jobs) == 375
    assert sum(job["score_row_count"] for job in jobs) == 1500
    assert sum(job["prediction_artifact_count"] for job in jobs) == 1000
    job = copy.deepcopy(jobs[0]); job["output_root"] = str(tmp_path / "row")
    receipt = _materialize_row(tmp_path, job)
    runner.validate_row_artifacts(job, receipt)
    path = runner.prediction_path(job["output_root"], "after", "M_JOINT")
    path.chmod(0o600); path.unlink()
    with pytest.raises(runner.ADV3B02LauncherError, match="artifact"):
        runner.validate_row_artifacts(job, receipt)
    receipt = _materialize_row(tmp_path, {**job, "output_root": str(tmp_path / "row2")})
    job2 = {**job, "output_root": str(tmp_path / "row2")}
    tampered = runner.score_path(job2["output_root"], "M0")
    tampered.chmod(0o600); tampered.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(runner.ADV3B02LauncherError, match="hash"):
        runner.validate_row_artifacts(job2, receipt)


@pytest.mark.parametrize("failure", ["missing_scene", "sha_mismatch", "empty_metrics"])
def test_score_validator_recomputes_paired_scene_sha_and_metric_closure(
    tmp_path, failure
):
    job = copy.deepcopy(runner.matrix_jobs(run_root=tmp_path / "run")[0])
    job["output_root"] = str(tmp_path / failure)
    receipt = _materialize_row(tmp_path, job)

    def mutate(payload):
        if failure == "missing_scene":
            del payload["after"]["by_scenario"][SCENES[-1]]
        elif failure == "sha_mismatch":
            payload["before_prediction_sha256"] = "f" * 64
        else:
            payload["before"]["by_scenario"][SCENES[0]] = {}

    receipt = _rewrite_score(job, receipt, "M0", mutate)
    with pytest.raises(runner.ADV3B02LauncherError):
        runner.validate_row_artifacts(job, receipt)


@pytest.mark.parametrize(
    "failure",
    ["missing_scene", "frozen_prefix", "after_audit", "state_binding"],
)
def test_append_receipt_validator_rejects_missing_tamper_and_binding_drift(
    tmp_path, failure
):
    job = copy.deepcopy(runner.matrix_jobs(run_root=tmp_path / "run")[0])
    job["output_root"] = str(tmp_path / failure)
    receipt = _materialize_row(tmp_path, job)
    append_map = receipt["append_receipts_by_scene"]
    scene = SCENES[0]
    if failure == "missing_scene":
        del append_map[scene]
    else:
        item = append_map[scene]
        if failure == "frozen_prefix":
            item["old_int8_codes_sha256_after"] = "0" * 64
        elif failure == "after_audit":
            item["after_int8_audit"]["top1_agreement"] = 0.0
            item["after_int8_audit_sha256"] = adv.sha256_bytes(
                adv._canon(item["after_int8_audit"])
            )
        else:
            item["after_state_sha256"] = "0" * 64
        body = {
            key: value
            for key, value in item.items()
            if key != "receipt_sha256"
        }
        item["receipt_sha256"] = adv.sha256_bytes(adv._canon(body))
    with pytest.raises(runner.ADV3B02LauncherError, match="append"):
        runner.validate_row_artifacts(job, receipt)


def test_formal_runtime_manifest_and_cuda_mapping_are_validated(
    tmp_path, monkeypatch
):
    root = tmp_path / "runtime"
    job = copy.deepcopy(runner.matrix_jobs(run_root=root)[0])
    receipt = _materialize_row(tmp_path, job)
    runner.write_json_new(Path(job["output_root"]) / "row_receipt.json", receipt)
    runtime_job = {
        **job,
        "cache_manifest": "sealed/cache_set.json",
        "authority_bundle": "sealed/authority_bundle",
    }
    one_counts = {
        "jobs": 1,
        "scene_slices": 3,
        "score_rows": 12,
        "arm_state_prediction_artifacts": 8,
    }
    monkeypatch.setattr(runner, "MATRIX_COUNTS", one_counts)
    monkeypatch.setattr(
        runner, "matrix_jobs", lambda *, run_root: [copy.deepcopy(job)]
    )
    gpu_audit = [
        {
            "physical_gpu_id": gpu,
            "device_name": f"GPU-{gpu}",
            "total_memory_bytes": 1024,
        }
        for gpu in runner.FORMAL_GPU_IDS
    ]
    runner.write_json_new(
        root / "matrix_runtime_manifest.json",
        {
            "schema": runner.LAUNCHER_SCHEMA,
            "candidate": CANDIDATE,
            "jobs": [runtime_job],
            "counts": one_counts,
            "gpu_ids": list(runner.FORMAL_GPU_IDS),
            "gpu_audit": gpu_audit,
            "dynamic_workers": 8,
            "mapping_policy": "dynamic_free_worker_physical_gpu_to_CUDA_VISIBLE_DEVICES_then_cuda:0",
            "query_truth_in_predictor": False,
            "launch_capability": True,
        },
    )
    log_path = root / "launcher_logs" / f"{job['job_id']}.log"
    log_sha = runner._write_new(log_path, b"synthetic row completed\n")
    runner.write_json_new(
        root / "matrix_runtime_completion.json",
        {
            "candidate": CANDIDATE,
            "status": "ARTIFACTS_COMPLETE",
            "counts": one_counts,
            "returncodes": {job["job_id"]: 0},
            "physical_gpu_by_job": {job["job_id"]: 0},
            "launcher_log_by_job": {
                job["job_id"]: {"path": str(log_path), "sha256": log_sha}
            },
        },
    )
    result = runner.validate_matrix_artifacts(run_root=root)
    assert result["runtime_manifest_validated"] is True
    assert result["counts"] == one_counts


def test_matrix_rejects_any_gpu_namespace_other_than_fixed_zero_to_seven(tmp_path):
    with pytest.raises(
        runner.ADV3B02LauncherError, match="exactly 0-7"
    ):
        runner.run_matrix(
            Namespace(run_root=str(tmp_path / "bad"), gpu_ids="0,1")
        )


def test_runner_exposes_real_row_and_matrix_entrypoints():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "code")
    result = subprocess.run([sys.executable, str(ROOT / "code" / "scripts" / "run_adv3b02_ts_drqknn_bcrr_125.py"), "--help"], check=False, capture_output=True, text=True, env=environment)
    assert result.returncode == 0
    assert "{plan,validate,row,matrix}" in result.stdout


def _health_jobs(tmp_path, count=12):
    run_root = tmp_path / "matrix"
    jobs = []
    for index in range(count):
        authority = tmp_path / "authority" / f"row_{index}"
        authority.mkdir(parents=True)
        (authority / "COMMIT.json").write_text("{}\n", encoding="utf-8")
        jobs.append(
            {
                "job_id": f"health_row_{index}",
                "receiver": "20-1",
                "seed": 713102 + index,
                "k_shot": 10,
                "new_class_count": 20,
                "cache_manifest": str(tmp_path / "cache" / f"row_{index}.json"),
                "authority_bundle": str(authority),
                "output_root": str(run_root / "jobs" / f"health_row_{index}"),
            }
        )
    return run_root, jobs


def _fake_immediate_popen(monkeypatch, *, returncode, log_line, p0_code=None):
    launched = []
    launch_lock = threading.Lock()

    class FakePopen:
        def __init__(self, command, *, stdout, **_kwargs):
            with launch_lock:
                self.pid = 900000 + len(launched)
                launched.append(tuple(command))
            self.returncode = int(returncode)
            emitted = log_line
            if p0_code is not None:
                output_index = command.index("--output-root") + 1
                injected = runner.ADV3B02P0Error(p0_code, "injected P0 failure")
                marker = runner._row_failure_marker_payload(
                    job_id_value=Path(command[output_index]).name,
                    exc=injected,
                    prediction_count=0,
                )
                emitted = (
                    runner.ROW_FAILURE_MARKER_PREFIX
                    + runner._canon(marker).decode("utf-8")
                    + "\n"
                )
            if emitted:
                stdout.write(emitted.encode("utf-8"))
                stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda process, **kwargs: {
            "pid": int(process.pid),
            "cwd": str(Path.cwd()),
            "cmdline": list(kwargs["command"]),
            "cmdline_sha256": runner.hashlib.sha256(
                runner._canon(list(kwargs["command"]))
            ).hexdigest(),
            "output_root": str(Path(kwargs["output_root"]).resolve()),
            "matrix_root": str(Path(kwargs["matrix_root"]).resolve()),
            "process_group_id": None if os.name == "nt" else int(process.pid),
            "ownership_verified": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_terminate_run_owned_process_tree",
        lambda process, entry, **_kwargs: {
            **entry["ownership_evidence"],
            "root_already_exited": True,
            "tree_exit_confirmed": True,
            "escalated_to_kill": False,
        },
    )
    return launched


@pytest.mark.parametrize(
    ("returncode", "log_line", "p0_code"),
    [
        (1, "ADV3B02StateError: affine gate failed at /tmp/run/row_713104\n", None),
        (1, "", "OUTPUT_OVERWRITE"),
        (1, "", "INPUT_HASH_OR_CHECKOUT_DRIFT"),
        (1, "", "QUERY_STATE_LEAKAGE"),
        (0, "", None),
    ],
)
def test_matrix_health_stops_bounded_dispatch_for_duplicate_preprediction_or_p0(
    tmp_path, monkeypatch, returncode, log_line, p0_code
):
    expected_p0_code = (
        "ROW_PROTOCOL_OR_ARTIFACT_DRIFT" if returncode == 0 else p0_code
    )
    run_root, jobs = _health_jobs(tmp_path)
    monkeypatch.setattr(runner, "_runtime_jobs", lambda _args: copy.deepcopy(jobs))
    monkeypatch.setattr(
        runner,
        "_audit_formal_physical_gpus",
        lambda: [
            {
                "physical_gpu_id": gpu,
                "device_name": f"GPU-{gpu}",
                "total_memory_bytes": 1024,
            }
            for gpu in runner.FORMAL_GPU_IDS
        ],
    )
    launched = _fake_immediate_popen(
        monkeypatch, returncode=returncode, log_line=log_line, p0_code=p0_code
    )
    args = Namespace(
        run_root=str(run_root),
        gpu_ids="0,1,2,3,4,5,6,7",
        phase1_checkpoint="checkpoint.pth",
        sealed_runtime="runtime.pt",
        package_method_lock="method_lock.json",
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(args)
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert completion["performance_status"] == "NO_PERFORMANCE_RESULT"
    assert completion["health"]["submitted"] == completion["health"][
        "systemic_detected_after_submitted"
    ]
    assert completion["health"]["submitted"] < len(jobs)
    assert completion["health"]["never_submitted"] == len(jobs) - completion[
        "health"
    ]["submitted"]
    assert len(launched) == completion["health"]["launched"]
    if expected_p0_code is not None:
        assert completion["health"]["submitted"] == len(runner.FORMAL_GPU_IDS)
    else:
        # At most one replacement can be submitted after the first failure and
        # before the second identical fingerprint reaches the coordinator.
        assert completion["health"]["submitted"] <= len(runner.FORMAL_GPU_IDS) + 1
    assert completion["health"]["succeeded"] == 0
    assert completion["counts"] == {
        "jobs": 0,
        "scene_slices": 0,
        "score_rows": 0,
        "arm_state_prediction_artifacts": 0,
    }
    if expected_p0_code is not None:
        assert completion["health"]["systemic_fingerprint"].startswith("P0:")
        assert {
            item["failure_code"]
            for item in completion["health"]["row_failure_markers"].values()
        } == {expected_p0_code}
    else:
        assert not completion["health"]["systemic_fingerprint"].startswith("P0:")


def _patch_health_matrix_inputs(monkeypatch, jobs):
    monkeypatch.setattr(runner, "_runtime_jobs", lambda _args: copy.deepcopy(jobs))
    monkeypatch.setattr(
        runner,
        "_audit_formal_physical_gpus",
        lambda: [
            {
                "physical_gpu_id": gpu,
                "device_name": f"GPU-{gpu}",
                "total_memory_bytes": 1024,
            }
            for gpu in runner.FORMAL_GPU_IDS
        ],
    )


def _health_args(run_root):
    return Namespace(
        run_root=str(run_root),
        gpu_ids="0,1,2,3,4,5,6,7",
        phase1_checkpoint="checkpoint.pth",
        sealed_runtime="runtime.pt",
        package_method_lock="method_lock.json",
    )


def test_parent_missing_commit_is_structured_p0_before_popen(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    for job in jobs[: len(runner.FORMAL_GPU_IDS)]:
        (Path(job["authority_bundle"]) / "COMMIT.json").unlink()
    _patch_health_matrix_inputs(monkeypatch, jobs)

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("missing COMMIT must fail before Popen")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden_popen)
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert health["systemic_fingerprint"] == "P0:INPUT_MISSING_OR_CHECKOUT_DRIFT"
    assert health["launched"] == 0
    assert health["parent_failure_receipts"]
    assert {
        item["failure_code"] for item in health["row_failure_markers"].values()
    } == {"INPUT_MISSING_OR_CHECKOUT_DRIFT"}


def test_parent_launcher_log_collision_is_structured_p0(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    _patch_health_matrix_inputs(monkeypatch, jobs)
    real_open = runner.os.open

    def collision_open(path, flags, mode=0o777):
        if str(path).endswith(".log"):
            raise FileExistsError("injected immutable launcher-log collision")
        return real_open(path, flags, mode)

    monkeypatch.setattr(runner.os, "open", collision_open)
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert health["systemic_fingerprint"] == "P0:OUTPUT_OVERWRITE"
    assert health["launched"] == 0
    assert health["parent_failure_receipts"]


def test_parent_ownership_capture_failure_is_structured_p0(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    _patch_health_matrix_inputs(monkeypatch, jobs)
    launched = _fake_immediate_popen(
        monkeypatch, returncode=1, log_line="ownership capture injected\n"
    )
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ADV3B02LauncherError("run-owned PID evidence absent at launch")
        ),
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert health["systemic_fingerprint"] == (
        "P0:RUN_OWNED_PROCESS_SAFETY_FAILURE"
    )
    assert len(launched) == health["launched"]
    assert health["parent_failure_receipts"]


def test_parent_artifact_tamper_is_immediate_structured_p0(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    by_output = {job["output_root"]: job for job in jobs}
    _patch_health_matrix_inputs(monkeypatch, jobs)
    lock = threading.Lock()
    launched = []

    class TamperedPopen:
        def __init__(self, command, *, env, **_kwargs):
            with lock:
                self.pid = 930000 + len(launched)
                launched.append(tuple(command))
            output = command[command.index("--output-root") + 1]
            job = copy.deepcopy(by_output[output])
            receipt = _materialize_row(tmp_path, job)
            receipt["device_namespace_execution"] = (
                dssc_runner._expected_row_device_namespace_execution(
                    int(env["CUDA_VISIBLE_DEVICES"])
                )
            )
            runner.write_json_new(Path(output) / "row_receipt.json", receipt)
            tampered = runner.prediction_path(output, "after", "M_JOINT")
            tampered.chmod(0o600)
            tampered.write_bytes(b"tampered-after-seal")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", TamperedPopen)
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda process, **kwargs: {
            "pid": int(process.pid),
            "cwd": str(Path.cwd()),
            "cmdline": list(kwargs["command"]),
            "cmdline_sha256": runner.hashlib.sha256(
                runner._canon(list(kwargs["command"]))
            ).hexdigest(),
            "output_root": str(Path(kwargs["output_root"]).resolve()),
            "matrix_root": str(Path(kwargs["matrix_root"]).resolve()),
            "process_group_id": None if os.name == "nt" else int(process.pid),
            "ownership_verified": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_terminate_run_owned_process_tree",
        lambda process, entry, **_kwargs: {
            **entry["ownership_evidence"],
            "root_already_exited": True,
            "tree_exit_confirmed": True,
            "escalated_to_kill": False,
        },
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert health["systemic_fingerprint"] == "P0:ROW_PROTOCOL_OR_ARTIFACT_DRIFT"
    assert health["submitted"] == len(runner.FORMAL_GPU_IDS)
    assert health["prediction_count"] == 0
    assert any(
        "prediction artifact/hash closure drift" in item["exception_message"]
        for item in health["row_failure_markers"].values()
    )


def test_coordinator_converts_escaped_future_to_job_bound_p0(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    _patch_health_matrix_inputs(monkeypatch, jobs)

    class ExplodingFuture:
        def cancelled(self):
            return False

        def cancel(self):
            return False

        def result(self):
            raise RuntimeError("injected worker escape")

    class ExplodingExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _function, _job):
            return ExplodingFuture()

    monkeypatch.setattr(runner, "ThreadPoolExecutor", ExplodingExecutor)
    monkeypatch.setattr(
        runner,
        "wait",
        lambda pending, return_when: (set(pending), set()),
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert health["systemic_fingerprint"] == "P0:ROW_PROTOCOL_OR_ARTIFACT_DRIFT"
    assert health["submitted"] == len(runner.FORMAL_GPU_IDS)
    assert health["launched"] == 0
    assert health["completed"] == len(runner.FORMAL_GPU_IDS)
    assert len(health["parent_failure_receipts"]) == len(runner.FORMAL_GPU_IDS)
    assert set(completion["returncodes"].values()) == {99}


def test_termination_failure_and_unconfirmed_emergency_are_first_p0(
    tmp_path, monkeypatch
):
    run_root, jobs = _health_jobs(tmp_path)
    _patch_health_matrix_inputs(monkeypatch, jobs)
    real_executor = runner.ThreadPoolExecutor
    monkeypatch.setattr(
        runner,
        "ThreadPoolExecutor",
        lambda max_workers: real_executor(max_workers=2),
    )
    lock = threading.Lock()
    started = []

    class SafetyPopen:
        def __init__(self, command, *, stdout, **_kwargs):
            with lock:
                index = len(started)
                self.pid = 940000 + index
                started.append(tuple(command))
            if index == 0:
                stdout.write(b"ordinary row failure before prediction\n")
                stdout.flush()
                self.returncode = 1
            else:
                self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", SafetyPopen)
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda process, **kwargs: {
            "pid": int(process.pid),
            "cwd": str(Path.cwd()),
            "cmdline": list(kwargs["command"]),
            "cmdline_sha256": runner.hashlib.sha256(
                runner._canon(list(kwargs["command"]))
            ).hexdigest(),
            "output_root": str(Path(kwargs["output_root"]).resolve()),
            "matrix_root": str(Path(kwargs["matrix_root"]).resolve()),
            "process_group_id": None if os.name == "nt" else int(process.pid),
            "ownership_verified": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_terminate_run_owned_process_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.ADV3B02LauncherError("process tree survived termination")
        ),
    )

    def unconfirmed_emergency(process, *, reason):
        process.returncode = -9
        return {
            "pid": int(process.pid),
            "termination_strategy": "injected_unconfirmed_emergency",
            "reason": reason,
            "root_exit_code": -9,
            "tree_exit_confirmed": False,
            "escalated_to_kill": True,
            "cleanup_error": "injected cleanup could not confirm tree exit",
        }

    monkeypatch.setattr(
        runner, "_emergency_cleanup_unverified_spawn", unconfirmed_emergency
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(_health_args(run_root))
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert completion["performance_status"] == "NO_PERFORMANCE_RESULT"
    assert health["systemic_fingerprint"] == (
        "P0:RUN_OWNED_PROCESS_SAFETY_FAILURE"
    )
    assert health["submitted"] == len(runner.FORMAL_GPU_IDS)
    assert health["systemic_detected_after_submitted"] == health["submitted"]
    assert health["never_submitted"] == len(jobs) - health["submitted"]
    assert health["cancelled_pending"] > 0
    assert completion["counts"]["arm_state_prediction_artifacts"] == 0
    assert any(
        receipt.get("tree_exit_confirmed") is False
        and receipt.get("cleanup_error")
        for receipt in health["termination_receipts"].values()
    )


def test_failure_fingerprint_is_row_invariant_and_prediction_aware(tmp_path):
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(
        "ADV3B02StateError: affine gate failed at /tmp/run/row_713104\n",
        encoding="utf-8",
    )
    second.write_text(
        "ADV3B02StateError: affine gate failed at /tmp/run/row_713105\n",
        encoding="utf-8",
    )
    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    assert runner._preprediction_failure_fingerprint(
        first, 1, output_a
    ) == runner._preprediction_failure_fingerprint(second, 1, output_b)
    published = output_b / "predictions" / "before" / "M0"
    published.mkdir(parents=True)
    (published / "prediction_artifact.npz").write_bytes(b"sealed")
    assert runner._preprediction_failure_fingerprint(second, 1, output_b) is None


def test_run_owned_process_identity_rejects_cwd_cmdline_group_and_root_drift(
    tmp_path,
):
    matrix_root = tmp_path / "matrix"
    output_root = matrix_root / "jobs" / "row_a"
    command = [
        sys.executable,
        str(ROOT / "code" / "scripts" / "run_adv3b02_ts_drqknn_bcrr_125.py"),
        "row",
        "--output-root",
        str(output_root),
    ]
    pgid = None if os.name == "nt" else 43210
    evidence = runner._validate_run_owned_process_identity(
        pid=43210,
        command=command,
        output_root=str(output_root),
        matrix_root=matrix_root,
        expected_cwd=ROOT,
        observed_cwd=ROOT,
        observed_cmdline=command,
        observed_process_group_id=pgid,
    )
    assert evidence["ownership_verified"] is True
    bad_cases = (
        {"output_root": str(tmp_path / "outside")},
        {"observed_cwd": tmp_path},
        {"observed_cmdline": command + ["--drift"]},
    )
    for changed in bad_cases:
        values = {
            "pid": 43210,
            "command": command,
            "output_root": str(output_root),
            "matrix_root": matrix_root,
            "expected_cwd": ROOT,
            "observed_cwd": ROOT,
            "observed_cmdline": command,
            "observed_process_group_id": pgid,
            **changed,
        }
        with pytest.raises(runner.ADV3B02LauncherError, match="run-owned"):
            runner._validate_run_owned_process_identity(**values)
    if os.name != "nt":
        with pytest.raises(runner.ADV3B02LauncherError, match="process-group"):
            runner._validate_run_owned_process_identity(
                pid=43210,
                command=command,
                output_root=str(output_root),
                matrix_root=matrix_root,
                expected_cwd=ROOT,
                observed_cwd=ROOT,
                observed_cmdline=command,
                observed_process_group_id=43211,
            )


@pytest.mark.skipif(os.name == "nt", reason="formal N607 process-tree semantics are POSIX")
def test_posix_root_exit_still_cleans_grandchild_and_preserves_unrelated_sentinel(
    tmp_path,
):
    matrix_root = tmp_path / "matrix"
    output_root = matrix_root / "jobs" / "tree_row"
    child_pid_path = tmp_path / "grandchild.pid"
    script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8');"
        "time.sleep(1)"
    )
    command = [
        sys.executable,
        "-c",
        script,
        str(child_pid_path),
        "--output-root",
        str(output_root),
    ]
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    root_process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        ownership = runner._capture_run_owned_process_identity(
            root_process,
            command=command,
            output_root=str(output_root),
            matrix_root=matrix_root,
            expected_cwd=ROOT,
        )
        deadline = time.monotonic() + 5.0
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_path.is_file()
        grandchild_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert root_process.wait(timeout=5) == 0
        assert runner._process_group_is_alive(int(ownership["process_group_id"]))
        receipt = runner._terminate_run_owned_process_tree(
            root_process,
            {
                "pid": root_process.pid,
                "cmdline": command,
                "output_root": str(output_root),
                "ownership_evidence": ownership,
            },
            matrix_root=matrix_root,
            expected_cwd=ROOT,
        )
        assert receipt["root_already_exited"] is True
        assert receipt["tree_exit_confirmed"] is True
        assert not runner._process_group_is_alive(root_process.pid)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
        assert sentinel.poll() is None
    finally:
        if root_process.poll() is None:
            os.killpg(root_process.pid, signal.SIGKILL)
            root_process.wait(timeout=5)
        if sentinel.poll() is None:
            os.killpg(sentinel.pid, signal.SIGKILL)
            sentinel.wait(timeout=5)


def test_structured_failure_marker_separates_p0_from_ordinary_technical(tmp_path):
    cases = (
        (runner.ADV3B02P0Error("OUTPUT_OVERWRITE", "output exists"), True),
        (runner.ADV3B02P0Error("INPUT_HASH_OR_CHECKOUT_DRIFT", "hash drift"), True),
        (ADV3B02StateError("query state update attempted"), True),
        (ADV3B02StateError("affine qKNN teacher gate failed"), False),
    )
    for index, (exc, expected_p0) in enumerate(cases):
        ident = f"row_{index}"
        marker = runner._row_failure_marker_payload(
            job_id_value=ident, exc=exc, prediction_count=0
        )
        path = tmp_path / f"row_{index}.log"
        path.write_text(
            runner.ROW_FAILURE_MARKER_PREFIX
            + runner._canon(marker).decode("utf-8")
            + "\n",
            encoding="utf-8",
        )
        parsed = runner._read_row_failure_marker(path, expected_job_id=ident)
        assert parsed is not None
        assert parsed["p0_protocol_or_safety"] is expected_p0
        if expected_p0:
            assert parsed["failure_code"] in runner.P0_FAILURE_CODES
        else:
            assert parsed["failure_code"] == runner.TECHNICAL_FAILURE_CODE


def test_health_stop_preserves_completed_row_partial_counts(tmp_path, monkeypatch):
    run_root, jobs = _health_jobs(tmp_path)
    by_output = {job["output_root"]: job for job in jobs}
    monkeypatch.setattr(runner, "_runtime_jobs", lambda _args: copy.deepcopy(jobs))
    monkeypatch.setattr(
        runner,
        "_audit_formal_physical_gpus",
        lambda: [
            {
                "physical_gpu_id": gpu,
                "device_name": f"GPU-{gpu}",
                "total_memory_bytes": 1024,
            }
            for gpu in runner.FORMAL_GPU_IDS
        ],
    )
    lock = threading.Lock()
    started = []

    class MixedPopen:
        def __init__(self, command, *, stdout, env, **_kwargs):
            with lock:
                index = len(started)
                self.pid = 910000 + index
                started.append(tuple(command))
            output = command[command.index("--output-root") + 1]
            if index == 0:
                job = copy.deepcopy(by_output[output])
                receipt = _materialize_row(tmp_path, job)
                receipt["device_namespace_execution"] = (
                    dssc_runner._expected_row_device_namespace_execution(
                        int(env["CUDA_VISIBLE_DEVICES"])
                    )
                )
                runner.write_json_new(Path(output) / "row_receipt.json", receipt)
                self.returncode = 0
            else:
                stdout.write(b"ADV3B02StateError: repeated affine audit failure\n")
                stdout.flush()
                self.returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", MixedPopen)
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda process, **kwargs: {
            "pid": int(process.pid),
            "cwd": str(Path.cwd()),
            "cmdline": list(kwargs["command"]),
            "cmdline_sha256": runner.hashlib.sha256(
                runner._canon(list(kwargs["command"]))
            ).hexdigest(),
            "output_root": str(Path(kwargs["output_root"]).resolve()),
            "matrix_root": str(Path(kwargs["matrix_root"]).resolve()),
            "process_group_id": None if os.name == "nt" else int(process.pid),
            "ownership_verified": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_terminate_run_owned_process_tree",
        lambda process, entry, **_kwargs: {
            **entry["ownership_evidence"],
            "tree_exit_confirmed": True,
            "escalated_to_kill": False,
        },
    )
    args = Namespace(
        run_root=str(run_root),
        gpu_ids="0,1,2,3,4,5,6,7",
        phase1_checkpoint="checkpoint.pth",
        sealed_runtime="runtime.pt",
        package_method_lock="method_lock.json",
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(args)
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    assert completion["performance_status"] == "NO_PERFORMANCE_RESULT"
    assert completion["counts"] == {
        "jobs": 1,
        "scene_slices": 3,
        "score_rows": 12,
        "arm_state_prediction_artifacts": 8,
    }


def test_p0_health_stop_cancels_queued_futures_without_new_dispatch(
    tmp_path, monkeypatch
):
    run_root, jobs = _health_jobs(tmp_path)
    by_output = {job["output_root"]: job for job in jobs}
    monkeypatch.setattr(runner, "_runtime_jobs", lambda _args: copy.deepcopy(jobs))
    monkeypatch.setattr(
        runner,
        "_audit_formal_physical_gpus",
        lambda: [
            {
                "physical_gpu_id": gpu,
                "device_name": f"GPU-{gpu}",
                "total_memory_bytes": 1024,
            }
            for gpu in runner.FORMAL_GPU_IDS
        ],
    )
    real_executor = runner.ThreadPoolExecutor
    monkeypatch.setattr(
        runner,
        "ThreadPoolExecutor",
        lambda max_workers: real_executor(max_workers=2),
    )
    lock = threading.Lock()
    started = []

    class QueuedPopen:
        def __init__(self, command, *, stdout, **_kwargs):
            with lock:
                index = len(started)
                self.pid = 920000 + index
                started.append(tuple(command))
            output = command[command.index("--output-root") + 1]
            if index == 0:
                job = by_output[output]
                marker = runner._row_failure_marker_payload(
                    job_id_value=job["job_id"],
                    exc=runner.ADV3B02P0Error(
                        "OUTPUT_OVERWRITE", "injected immutable output collision"
                    ),
                    prediction_count=0,
                )
                stdout.write(
                    (
                        runner.ROW_FAILURE_MARKER_PREFIX
                        + runner._canon(marker).decode("utf-8")
                        + "\n"
                    ).encode("utf-8")
                )
                stdout.flush()
                self.returncode = 1
            else:
                self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", QueuedPopen)
    monkeypatch.setattr(
        runner,
        "_capture_run_owned_process_identity",
        lambda process, **kwargs: {
            "pid": int(process.pid),
            "cwd": str(Path.cwd()),
            "cmdline": list(kwargs["command"]),
            "cmdline_sha256": runner.hashlib.sha256(
                runner._canon(list(kwargs["command"]))
            ).hexdigest(),
            "output_root": str(Path(kwargs["output_root"]).resolve()),
            "matrix_root": str(Path(kwargs["matrix_root"]).resolve()),
            "process_group_id": None if os.name == "nt" else int(process.pid),
            "ownership_verified": True,
        },
    )

    def terminate(process, entry, **_kwargs):
        if process.returncode is None:
            process.returncode = -15
        return {
            **entry["ownership_evidence"],
            "tree_exit_confirmed": True,
            "escalated_to_kill": False,
        }

    monkeypatch.setattr(runner, "_terminate_run_owned_process_tree", terminate)
    args = Namespace(
        run_root=str(run_root),
        gpu_ids="0,1,2,3,4,5,6,7",
        phase1_checkpoint="checkpoint.pth",
        sealed_runtime="runtime.pt",
        package_method_lock="method_lock.json",
    )
    with pytest.raises(runner.ADV3B02LauncherError, match="dispatch stopped"):
        runner.run_matrix(args)
    completion = json.loads(
        (run_root / "matrix_runtime_completion.json").read_text(encoding="utf-8")
    )
    health = completion["health"]
    assert completion["status"] == "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
    assert completion["performance_status"] == "NO_PERFORMANCE_RESULT"
    assert health["submitted"] == len(runner.FORMAL_GPU_IDS)
    assert health["systemic_detected_after_submitted"] == health["submitted"]
    assert 1 <= health["launched"] < health["submitted"]
    assert health["cancelled_pending"] > 0
    assert health["never_submitted"] == len(jobs) - health["submitted"]
    assert len(started) == health["launched"]
    assert completion["counts"]["arm_state_prediction_artifacts"] == 0
