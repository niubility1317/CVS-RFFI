from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_cvs_stage2_predictor_request as request_builder  # noqa: E402

from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    PHASE2_FULL_CONTRACT,
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    validate_predictor_request,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    iq_row_sha256,
    make_member_descriptor,
    sha256_file,
    write_predictor_package_manifest_and_seal,
)


def _npz(path: Path, scenario: str, *, query: bool) -> None:
    iq = np.zeros((1, 2, 8), dtype=np.float32)
    if query:
        payload = {
            "query_leo_weak_iq": iq,
            "query_tokens": np.asarray(["qid_" + "1" * 64]),
            "query_overlay_tokens": np.asarray(["oid_" + "2" * 64]),
            "query_satellite_seeds": np.asarray([7], dtype=np.int64),
            "query_post_channel_iq_sha256": np.asarray([iq_row_sha256(iq[0])]),
            "manifest_json": np.asarray(json.dumps({
                "schema": QUERY_SCHEMA, "scenario": scenario,
                "query_truth_included": False, "query_role_included": False,
                "query_true_batch_class_count_included": False,
                "query_class_quota_included": False,
                "query_ordering_hint_included": False,
                "token_scheme": "hmac_sha256_opaque_v1",
            }, sort_keys=True)),
        }
    else:
        payload = {
            "support_pool_leo_weak_iq": iq,
            "support_pool_class_indices": np.asarray([0], dtype=np.int64),
            "support_pool_rank_within_class": np.asarray([0], dtype=np.int64),
            "support_pool_tokens": np.asarray(["sid_" + "3" * 64]),
            "support_pool_overlay_tokens": np.asarray(["oid_" + "4" * 64]),
            "support_pool_satellite_seeds": np.asarray([7], dtype=np.int64),
            "support_pool_post_channel_iq_sha256": np.asarray([iq_row_sha256(iq[0])]),
            "manifest_json": np.asarray(json.dumps({
                "schema": SUPPORT_SCHEMA, "scenario": scenario,
                "registered_support_labels_allowed": True,
                "registered_class_count": 1, "support_pool_max_k": 1,
                "token_scheme": "hmac_sha256_opaque_v1",
            }, sort_keys=True)),
        }
    with path.open("xb") as handle:
        np.savez(handle, **payload)


def _fixture(tmp_path: Path):
    root = tmp_path / "package"
    root.mkdir()
    members = []
    for role, filename, payload in (
        ("checkpoint", "checkpoint.bin", b"checkpoint"),
        ("adapter", "adapter.bin", b"adapter"),
        ("head", "head.bin", b"head"),
        (
            "tta_policy",
            "tta.json",
            (
                b'{"base_views":1,"max_views":5,'
                b'"uses_query_labels":false,"uses_query_role":false,'
                b'"uses_class_quota":false}'
            ),
        ),
    ):
        path = root / filename
        path.write_bytes(payload)
        members.append(make_member_descriptor(
            path, relative_path=filename, artifact_role=role, schema=f"test.{role}.v1"
        ))
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        support = root / f"support_{scenario}.npz"
        query = root / f"query_{scenario}.npz"
        _npz(support, scenario, query=False)
        _npz(query, scenario, query=True)
        members.extend([
            make_member_descriptor(
                support, relative_path=support.name, artifact_role=f"support:{scenario}",
                schema=SUPPORT_SCHEMA, scenario=scenario, npz_members=SUPPORT_NPZ_MEMBERS,
            ),
            make_member_descriptor(
                query, relative_path=query.name, artifact_role=f"query:{scenario}",
                schema=QUERY_SCHEMA, scenario=scenario, npz_members=QUERY_NPZ_MEMBERS,
            ),
        ])
    seal = tmp_path / "package.seal.json"
    _mp, _sp, manifest, seal_payload = write_predictor_package_manifest_and_seal(
        root,
        manifest_metadata={
            "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
            "artifact_stage": PREDICTOR_INPUT_STAGE,
            "stage": "stage2c", "receiver": "20-1", "seed": 713101,
            "new_class_count": 1, "support_pool_max_k": 1,
            "target_channel_view": "leo_weak_only",
            "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
            "registered_class_count": 1,
            "registered_classes": [{"class_index": 0, "class_handle": "cls_" + "0" * 64}],
            "candidate_lock_sha256": "9" * 64,
            **PHASE2_FULL_CONTRACT,
        },
        members=members,
        detached_seal_path=seal,
    )
    evidence = dict(zip(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS, [None] * 9))
    evidence.update({
        "sealed_inference_package_sha256": sha256_file(seal),
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": "7" * 64,
        "artifact_member_allowlist_sha256": seal_payload["artifact_member_allowlist_sha256"],
        "os_isolation_mode": "bwrap_readonly_mounts",
        "os_isolation_attestation_sha256": "6" * 64,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": "5" * 64,
        "predict_score_process_isolation": True,
    })
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    args = argparse.Namespace(
        predictor_package_root=root, detached_seal_path=seal,
        expected_seal_sha256=sha256_file(seal), runtime_evidence_json=evidence_path,
        k_shot=1,
        request_id="row-1:clear", row_id="row-1",
        output_relative_path="prediction_artifact.npz",
        output_json=tmp_path / "request.json",
    )
    return args, evidence


def test_request_is_exact_and_cross_digest_bound(tmp_path: Path) -> None:
    args, _evidence = _fixture(tmp_path)
    result = request_builder.build_request(args)
    request = json.loads(Path(result["request_json"]).read_text(encoding="utf-8"))
    validate_predictor_request(request)
    assert request["scenarios"] == list(FORMAL_LEO_WEAK_SCENARIOS)
    assert request["package_root_sha256"] == request["phase2_runtime_isolation_evidence"]["package_root_sha256"]
    assert "truth" not in json.dumps(request).lower()


def test_request_rejects_runtime_package_digest_mismatch(tmp_path: Path) -> None:
    args, evidence = _fixture(tmp_path)
    evidence["package_root_sha256"] = "0" * 64
    Path(args.runtime_evidence_json).write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="package root digest mismatch"):
        request_builder.build_request(args)


def test_request_rejects_k_outside_nested_pool(tmp_path: Path) -> None:
    args, _evidence = _fixture(tmp_path)
    args.k_shot = 2
    with pytest.raises(ValueError, match="outside the sealed nested support pool"):
        request_builder.build_request(args)


def test_request_refuses_overwrite(tmp_path: Path) -> None:
    args, _evidence = _fixture(tmp_path)
    request_builder.build_request(args)
    with pytest.raises(FileExistsError):
        request_builder.build_request(args)


@pytest.mark.parametrize("field", ("uses_query_role", "uses_class_quota"))
def test_request_allows_only_negative_nested_forbidden_guard(
    tmp_path: Path, field: str
) -> None:
    args, _evidence = _fixture(tmp_path)
    result = request_builder.build_request(args)
    request = json.loads(Path(result["request_json"]).read_text(encoding="utf-8"))
    assert request["tta_policy"][field] is False

    request["tta_policy"][field] = True
    with pytest.raises(
        ValueError,
        match=rf"forbidden_predictor_key:request.tta_policy.{field}",
    ):
        validate_predictor_request(request)
