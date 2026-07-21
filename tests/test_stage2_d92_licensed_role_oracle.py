from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from cvsrffi.stage2_d92_licensed_role_oracle import (
    D92LicensedRoleOracleError,
    LICENSE_STATUS,
    OUTPUT_SCHEMA,
    ROLE_CAPSULE_SCHEMA,
    decode_d92_all_registry_baseline,
    decode_d92_licensed_role_oracle,
    project_and_seal_d92_role_only_capsule,
)


CLASSES = ("old0", "old1", "new0", "new1")
OLD = CLASSES[:2]


def _query_token(index: int) -> str:
    return "qid_" + format(index, "064x")


def _capsule(*roles: str, tokens=None):
    if tokens is None:
        tokens = [_query_token(index + 1) for index in range(len(roles))]
    return {
        "schema": ROLE_CAPSULE_SCHEMA,
        "rows": [
            {"query_token": token, "evaluation_role": role}
            for token, role in zip(tokens, roles)
        ],
    }


def _truth_row(token: str, role: str, tx: str):
    return {
        "query_token": token,
        "true_class_index": 0,
        "true_class_handle": "cls_" + "b" * 64,
        "transmitter_label": tx,
        "evaluation_role": role,
        "receiver_label": "rx20",
        "day_label": "day1",
        "signal_label": "sig1",
        "physical_sample_id": "physical1",
        "scenario": "leo_clear_weak",
    }


def _write_truth(path, rows):
    payload = {
        "schema": "cvs.phase2.query_truth_sidecar.v2",
        "stage": "stage2c",
        "receiver": "rx20",
        "seed": 713101,
        "rows": rows,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_paired_baseline_and_role_oracle_derive_from_identical_scores():
    scores = np.asarray(
        [
            [0.2, 0.7, 4.0, 0.1],
            [5.0, 0.1, 0.3, 0.9],
            [0.1, 0.8, 0.8, 0.2],
        ],
        dtype=np.float32,
    )
    result = decode_d92_licensed_role_oracle(
        scores,
        CLASSES,
        OLD,
        [_query_token(1), _query_token(2), _query_token(3)],
        _capsule("target_old", "target_new", "target_new"),
    )
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["license_status"] == LICENSE_STATUS
    assert result["baseline_indices"] == (2, 0, 1)
    assert result["baseline_predictions"] == ("new0", "old0", "old1")
    assert result["role_oracle_indices"] == (1, 3, 2)
    assert result["role_oracle_predictions"] == ("old1", "new1", "new0")
    assert result["audit"] == {
        "license_status": LICENSE_STATUS,
        "promotion_eligible": False,
        "protocol_legal_performance_claim": False,
        "paired_baseline_from_identical_scores": True,
        "query_role_oracle_access": True,
        "query_role_only_access": True,
        "query_tx_id_access": False,
        "query_truth_label_access": False,
        "query_class_quota_access": False,
        "query_true_batch_class_count_access": False,
        "query_batch_reassignment": False,
        "query_independent_decisions": True,
        "oracle_scope": "argmax_within_role_registry_only",
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"tx_id": [0]},
        {"truth_label": ["old0"]},
        {"class_quota": {"old0": 1}},
        {"true_batch_class_count": 1},
        {"query_id": ["q0"]},
    ],
)
def test_role_capsule_exact_schema_rejects_forbidden_or_extra_fields(extra):
    capsule = _capsule("target_old")
    capsule.update(extra)
    with pytest.raises(D92LicensedRoleOracleError, match="exact schema"):
        decode_d92_licensed_role_oracle(
            np.zeros((1, len(CLASSES)), dtype=np.float32),
            CLASSES,
            OLD,
            [_query_token(1)],
            capsule,
        )


@pytest.mark.parametrize(
    "scores,classes,old,capsule",
    [
        (np.zeros((1, 3), dtype=np.float32), CLASSES, OLD, _capsule("target_old")),
        (np.asarray([[0.0, np.nan, 0.0, 0.0]]), CLASSES, OLD, _capsule("target_old")),
        (np.zeros((1, 4), dtype=np.float32), CLASSES, ("old1",), _capsule("target_old")),
        (
            np.zeros((1, 4), dtype=np.float32),
            ("old0", "old0", "new0", "new1"),
            ("old0",),
            _capsule("target_old"),
        ),
        (np.zeros((1, 4), dtype=np.float32), CLASSES, OLD, _capsule("future")),
        (np.zeros((2, 4), dtype=np.float32), CLASSES, OLD, _capsule("target_old")),
    ],
)
def test_malformed_inputs_fail_closed(scores, classes, old, capsule):
    with pytest.raises(D92LicensedRoleOracleError):
        decode_d92_licensed_role_oracle(
            scores,
            classes,
            old,
            [_query_token(index + 1) for index in range(len(scores))],
            capsule,
        )


def test_each_prediction_is_independent_of_other_query_rows():
    target = np.asarray([[0.1, 0.9, 3.0, 2.0]], dtype=np.float64)
    alone = decode_d92_licensed_role_oracle(
        target,
        CLASSES,
        OLD,
        [_query_token(2)],
        _capsule("target_old", tokens=[_query_token(2)]),
    )
    distractors = np.asarray(
        [[99.0, -1.0, 8.0, 7.0], [-4.0, 50.0, 100.0, 2.0]],
        dtype=np.float64,
    )
    together = decode_d92_licensed_role_oracle(
        np.concatenate([distractors[:1], target, distractors[1:]], axis=0),
        CLASSES,
        OLD,
        [_query_token(1), _query_token(2), _query_token(3)],
        _capsule(
            "target_new",
            "target_new",
            "target_old",
            tokens=[_query_token(3), _query_token(1), _query_token(2)],
        ),
    )
    assert together["baseline_predictions"][1] == alone["baseline_predictions"][0]
    assert together["role_oracle_predictions"][1] == alone["role_oracle_predictions"][0]


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_decoder_rejects_non_bijective_capsule_token_join(mode):
    query_tokens = [_query_token(1), _query_token(2)]
    capsule = _capsule("target_old", "target_new", tokens=query_tokens)
    if mode == "missing":
        capsule["rows"].pop()
    elif mode == "extra":
        capsule["rows"].append(
            {"query_token": _query_token(3), "evaluation_role": "target_old"}
        )
    else:
        capsule["rows"][1]["query_token"] = query_tokens[0]
    with pytest.raises(D92LicensedRoleOracleError, match="missing|extra|duplicate"):
        decode_d92_licensed_role_oracle(
            np.zeros((2, len(CLASSES)), dtype=np.float32),
            CLASSES,
            OLD,
            query_tokens,
            capsule,
        )


def test_within_role_label_permutation_is_equivariant():
    scores = np.asarray(
        [[0.2, 0.8, 0.7, 0.4], [0.9, 0.1, 0.3, 0.6]], dtype=np.float32
    )
    roles = _capsule("target_old", "target_new")
    query_tokens = [_query_token(1), _query_token(2)]
    original = decode_d92_licensed_role_oracle(
        scores, CLASSES, OLD, query_tokens, roles
    )
    permutation = np.asarray([1, 0, 3, 2])
    permuted_classes = tuple(CLASSES[index] for index in permutation)
    permuted = decode_d92_licensed_role_oracle(
        scores[:, permutation],
        permuted_classes,
        permuted_classes[:2],
        query_tokens,
        roles,
    )
    assert permuted["baseline_predictions"] == original["baseline_predictions"]
    assert permuted["role_oracle_predictions"] == original["role_oracle_predictions"]


def test_ties_use_first_registry_position_within_the_selected_scope():
    scores = np.ones((2, 4), dtype=np.float32)
    result = decode_d92_licensed_role_oracle(
        scores,
        CLASSES,
        OLD,
        [_query_token(1), _query_token(2)],
        _capsule("target_old", "target_new"),
    )
    assert result["baseline_predictions"] == ("old0", "old0")
    assert result["role_oracle_predictions"] == ("old0", "new0")


def test_baseline_can_be_committed_before_role_capsule_is_opened():
    scores = np.asarray([[0.2, 0.9, 3.0, 0.1]], dtype=np.float32)
    assert decode_d92_all_registry_baseline(scores, CLASSES) == ("new0",)


def test_offline_projection_seals_only_roles_and_returns_hash_receipt(tmp_path):
    truth_path = tmp_path / "truth_sidecar.json"
    capsule_path = tmp_path / "role_capsule.json"
    secret_tx = "TX_SECRET_MUST_NOT_LEAK"
    truth_bytes = _write_truth(
        truth_path,
        [
            _truth_row("qid_" + "1" * 64, "target_old", secret_tx),
            _truth_row("qid_" + "2" * 64, "target_new", "TX_OTHER"),
        ],
    )
    receipt = project_and_seal_d92_role_only_capsule(truth_path, capsule_path)
    capsule_bytes = capsule_path.read_bytes()
    capsule = json.loads(capsule_bytes)
    assert set(capsule) == {"schema", "rows"}
    assert capsule["schema"] == ROLE_CAPSULE_SCHEMA
    assert capsule["rows"] == [
        {"query_token": "qid_" + "1" * 64, "evaluation_role": "target_old"},
        {"query_token": "qid_" + "2" * 64, "evaluation_role": "target_new"},
    ]
    text = capsule_bytes.decode("utf-8")
    assert "true_class_handle" not in text
    assert "transmitter_label" not in text
    assert secret_tx not in text
    assert "TX_OTHER" not in text
    assert receipt == {
        "path": str(capsule_path),
        "source_truth_sha256": hashlib.sha256(truth_bytes).hexdigest(),
        "capsule_sha256": hashlib.sha256(capsule_bytes).hexdigest(),
        "readonly": True,
    }
    assert os.stat(capsule_path).st_mode & 0o222 == 0


def test_offline_projection_rejects_duplicate_tokens_and_invalid_roles(tmp_path):
    for name, rows, message in (
        (
            "duplicate",
            [
                _truth_row("qid_" + "3" * 64, "target_old", "tx0"),
                _truth_row("qid_" + "3" * 64, "target_new", "tx1"),
            ],
            "duplicate",
        ),
        (
            "role",
            [_truth_row("qid_" + "4" * 64, "unknown", "tx0")],
            "evaluation_role",
        ),
    ):
        truth_path = tmp_path / f"{name}_truth.json"
        _write_truth(truth_path, rows)
        with pytest.raises(D92LicensedRoleOracleError, match=message):
            project_and_seal_d92_role_only_capsule(
                truth_path, tmp_path / f"{name}_capsule.json"
            )


def test_offline_projection_never_overwrites_existing_capsule(tmp_path):
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "capsule.json"
    _write_truth(
        truth_path,
        [_truth_row("qid_" + "5" * 64, "target_old", "tx0")],
    )
    output_path.write_text("owned", encoding="utf-8")
    with pytest.raises(D92LicensedRoleOracleError, match="exclusively"):
        project_and_seal_d92_role_only_capsule(truth_path, output_path)
    assert output_path.read_text(encoding="utf-8") == "owned"
