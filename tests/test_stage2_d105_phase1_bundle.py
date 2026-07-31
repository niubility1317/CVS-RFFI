from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import stat
from datetime import datetime, timezone

import numpy as np
import pytest

from cvsrffi.stage2_d105_cbrc import compute_d105_bundle_validator_receipt
from cvsrffi.stage2_d105_phase1_bundle import (
    AUTHORITY_SEAL_SCHEMA,
    CANDIDATE_ID,
    COMPONENT_STATUS,
    D105Phase1BundleError,
    FORMAL_STATUS,
    SOURCE_HELD_PREDICTION_SCHEMA,
    SOURCE_HELD_SCORE_SCHEMA,
    SOURCE_HELD_TRUTH_OPEN_SCHEMA,
    STRICT_TAP_MEMBERS,
    STRICT_TAP_SCHEMA,
    build_d105_phase1_component,
    compute_d105_source_aggregate_lineage,
    compute_d105_source_held_prediction_commit,
    compute_d105_source_held_tx_prediction_commit,
    derive_d105_source_held_gate,
    execute_d105_source_held_predictions,
    load_d105_candidate_runtime_manifest,
    load_d105_phase1_asset,
    make_d105_phase1_runtime_handle,
    open_d105_source_held_truth,
    score_d105_source_held_truth,
    seal_d105_phase1_component,
    sha256_file,
    validate_d105_phase1_asset,
)
import cvsrffi.stage2_d105_phase1_bundle as phase1
import cvsrffi.stage2_d105_phase1_authority as authority
import cvsrffi.somph_runtime_trust as somph_runtime_trust


CHECKPOINT_SHA256 = "1" * 64
SOURCE_ACCESS_SHA256 = "2" * 64
VALIDATED_BUNDLE_ID = "9" * 64
TEST_AUTHORITY_SEED = bytes.fromhex(
    "4f3c2a19080706050403020100112233445566778899aabbccddeeff10203040"
)


@pytest.fixture(autouse=True)
def _test_pinned_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a deterministic test root while production remains hard-pinned."""

    public = authority.ed25519_public_key_from_seed(TEST_AUTHORITY_SEED)
    monkeypatch.setattr(somph_runtime_trust, "PINNED_AUTHORITY_PUBLIC_KEY_HEX", public.hex())
    monkeypatch.setattr(
        somph_runtime_trust,
        "PINNED_AUTHORITY_PUBLIC_KEY_SHA256",
        hashlib.sha256(public).hexdigest(),
    )
    monkeypatch.setattr(somph_runtime_trust, "PINNED_AUTHORITY_KEY_ID", "d105-test-authority-ed25519")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, object], *, immutable: bool = False) -> str:
    path.write_bytes(_canonical_bytes(value))
    if immutable:
        os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_d102_revocation(tmp_path: Path) -> tuple[Path, Path, str]:
    """Strict fixture uses the retrieved D102r6 immutable identities."""

    manifest = {
        "schema": authority.D102_REVOCATION_SCHEMA,
        "signature_domain": authority.D102_REVOCATION_SIGNATURE_DOMAIN,
        "issuer_key_id": somph_runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": "2026-01-01T00:00:00Z",
        "expires_at": "2030-01-01T00:00:00Z",
        "revocation_id": "a" * 64,
        "revoked_artifacts": [
            {
                "run_id": "d102_rb_metabias4_phase1_analytic_held_20260724_r6",
                "bundle_manifest_sha256": "0690f2ab19560a54c96599ffc59a56fd31786f48ac2f05659414d8c29ff0da64",
                "bundle_payload_sha256": "440ff82a1f74b67078f699eaca86e85b9739d574721ccfb460a423ff97cc93d4",
                "bundle_seal_sha256": "cdcfceb5a31e3409ccea137fe116347f2214640e6514b080d442e7a193a0db59",
                "bundle_content_root_sha256": "16b9a8388c612509e4b220f2883fcd92187e1de0e4236ef25e2ef72a472a48b7",
                "checkpoint_sha256": "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98",
                "method_lock_sha256": "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f",
                "runtime_sha256": "e1b21bee74941dfb550b67698a75f485937bc39431ed7859baaa20d44a4899f3",
                "held_score_sha256": "01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2",
                "tap_archive_sha256": "c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1",
                "status": "PHASE1_HELD_FALSIFIER_REJECT",
            }
        ],
    }
    manifest_path = tmp_path / "d102_revocation_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    signature = authority.sign_d105_detached(
        domain=authority.D102_REVOCATION_SIGNATURE_DOMAIN,
        payload=manifest_path.read_bytes(),
        private_seed=TEST_AUTHORITY_SEED,
    )
    signature_path = tmp_path / "d102_revocation_manifest.ed25519"
    signature_path.write_bytes(signature)
    return manifest_path, signature_path, manifest_sha


def _candidate_identity(tmp_path: Path) -> tuple[Path, str, Path, str]:
    code_root = Path(phase1.__file__).resolve().parents[1]
    core_files = phase1.D105_CANDIDATE_RUNTIME_FILES
    runtime = {
        "schema": "cvs.stage2.d105.candidate_runtime_manifest.v1",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "entrypoints": dict(phase1.D105_CANDIDATE_RUNTIME_ENTRYPOINTS),
        "core_file_sha256": {
            name: sha256_file(code_root / name) for name in core_files
        },
    }
    runtime_path = tmp_path / "d105_candidate_runtime_manifest.json"
    runtime_sha = _write_json(runtime_path, runtime)
    lock = {
        "schema": "cvs.stage2.d105.candidate_method_lock.v1",
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "d105_candidate_runtime_manifest_sha256": runtime_sha,
        "d105_cbrc": {
            "semantic_revision": "cbrc_mb4_task_balanced_huber_irls4_fp16_v1",
            "code_dim": 4,
            "domain_dim": 32,
            "allowed_k": [1, 5, 10],
            "irls_steps": 4,
            "old_new_task_mass": [0.5, 0.5],
            "k1_zero_coefficient": True,
            "ground_old_multiprototype_enabled": False,
            "deployment_coefficient_dtype": "float16",
            "query_transform": "relu_l2norm_pre_relu_plus_mb4",
            "query_state_updates": 0,
        },
        "student_t_qknn": {
            "student_nu": 3.0,
            "kernel_effective_dim": 12,
            "kernel_volume_gamma": 1.0,
            "shared_h0": 0.35,
            "scale_prior_strength": 2.0,
            "scale_min_ratio": 0.5,
            "scale_max_ratio": 2.0,
            "temperature": 0.85,
            "support_storage": "int8_fp16_scale",
        },
        "four_arm": {
            "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
            "same_da_state_for_da_and_joint": True,
            "same_head_code_config_for_head_and_joint": True,
            "query_truth_surface": False,
        },
        "source_held": {
            "receiver_held_k": [1, 5, 10],
            "class_loco_k": 1,
            "tx_probe_algorithm": "receiver_held_ridge_l2_0.01",
            "tx_probe_max_balanced_accuracy": 0.25,
            "int8_min_top1_agreement": 0.995,
            "large_margin_minimum": 0.10,
            "large_margin_flip_max": 0,
            "truth_open_after_prediction": True,
        },
        "target25": {
            "seed": 713102,
            "claim_scope": "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE",
            "formal_launch_authority": False,
            "slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
            "leo_scenarios": [
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            ],
            "outer_row_count": 25,
            "scenario_arm_pair_count": 300,
            "state_prediction_surface_count": 600,
        },
    }
    lock_path = tmp_path / "d105_candidate_method_lock.json"
    lock_sha = _write_json(lock_path, lock)
    return runtime_path, runtime_sha, lock_path, lock_sha


def _cvsrffi_module_source(code_root: Path, module: str) -> Path | None:
    if module == "cvsrffi":
        candidate = code_root / "cvsrffi" / "__init__.py"
    elif module.startswith("cvsrffi."):
        candidate = code_root.joinpath(*module.split(".")).with_suffix(".py")
        if not candidate.is_file():
            candidate = code_root.joinpath(*module.split("."), "__init__.py")
    else:
        return None
    return candidate if candidate.is_file() else None


def _absolute_cvsrffi_import(
    module: str, level: int, imported_module: str | None
) -> str:
    package_parts = module.split(".")[:-1]
    base = package_parts[: len(package_parts) - level + 1] if level else []
    if imported_module:
        base.extend(imported_module.split("."))
    return ".".join(base)


def _direct_local_cvsrffi_imports(code_root: Path, module: str) -> set[str]:
    source = _cvsrffi_module_source(code_root, module)
    assert source is not None, f"missing local module source: {module}"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "cvsrffi" or name.startswith("cvsrffi."):
                    if _cvsrffi_module_source(code_root, name) is not None:
                        imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_cvsrffi_import(module, node.level, node.module)
            if base != "cvsrffi" and not base.startswith("cvsrffi."):
                continue
            if _cvsrffi_module_source(code_root, base) is not None:
                imports.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                if _cvsrffi_module_source(code_root, child) is not None:
                    imports.add(child)
    return imports


def _recursive_d105_cvsrffi_runtime_files(code_root: Path) -> set[str]:
    roots = {
        entrypoint.split(":", 1)[0]
        for entrypoint in phase1.D105_CANDIDATE_RUNTIME_ENTRYPOINTS.values()
        if entrypoint.startswith("cvsrffi.")
    }
    queue: deque[str] = deque(sorted(roots))
    seen: set[str] = set()
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        queue.extend(
            sorted(_direct_local_cvsrffi_imports(code_root, module) - seen)
        )
    files: set[str] = set()
    for module in seen:
        source = _cvsrffi_module_source(code_root, module)
        assert source is not None
        files.add(source.relative_to(code_root).as_posix())
    return files


def _strict_arrays(*, samples_per_cell: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(105731)
    pre_rows: list[np.ndarray] = []
    dom_rows: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    physical: list[str] = []
    class_pre = rng.normal(0.2, 0.25, (4, 160)).astype(np.float32)
    class_dom = rng.normal(0.0, 0.16, (4, 160)).astype(np.float32)
    receiver_pre = rng.normal(0.0, 0.40, (4, 160)).astype(np.float32)
    receiver_dom = rng.normal(0.0, 0.35, (4, 160)).astype(np.float32)
    for receiver_index in range(4):
        for class_index in range(4):
            for sample_index in range(samples_per_cell):
                pre_rows.append(
                    class_pre[class_index]
                    + receiver_pre[receiver_index]
                    + rng.normal(0.0, 0.04, 160).astype(np.float32)
                )
                dom_rows.append(
                    class_dom[class_index]
                    + receiver_dom[receiver_index]
                    + rng.normal(0.0, 0.05, 160).astype(np.float32)
                )
                labels.append(f"class-{class_index:02d}")
                receivers.append(f"source-rx-{receiver_index:02d}")
                physical.append(
                    "source-physical-"
                    f"{receiver_index:02d}-{class_index:02d}-{sample_index:02d}"
                )
    return {
        "pre_relu": np.asarray(pre_rows, dtype=np.float32),
        "z_dom": np.asarray(dom_rows, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.str_),
        "receiver_ids": np.asarray(receivers, dtype=np.str_),
        "physical_ids": np.asarray(physical, dtype=np.str_),
    }


def _materialize_strict_tap(
    tmp_path: Path,
    *,
    d102_revocation_manifest_sha256: str,
    samples_per_cell: int = 11,
    d102_reused: bool = False,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime_path, runtime_sha, lock_path, lock_sha = _candidate_identity(tmp_path)
    arrays = _strict_arrays(samples_per_cell=samples_per_cell)
    archive = tmp_path / "d105_strict_source_tap.npz"
    np.savez(archive, **arrays)
    archive_sha = sha256_file(archive)
    receipt = {
        "schema": STRICT_TAP_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": runtime_sha,
        "method_lock_sha256": lock_sha,
        "source_access_receipt_sha256": SOURCE_ACCESS_SHA256,
        "tap_archive_sha256": archive_sha,
        "tap_archive_members": list(STRICT_TAP_MEMBERS),
        "row_count": len(arrays["pre_relu"]),
        "pre_relu_sha256": phase1._array_sha256(arrays["pre_relu"]),
        "z_dom_sha256": phase1._array_sha256(arrays["z_dom"]),
        "physical_id_root_sha256": phase1._physical_root(
            arrays["physical_ids"].astype(str).tolist()
        ),
        "d102_revocation_manifest_sha256": d102_revocation_manifest_sha256,
        "execution_path": "strict_zid_with_hook",
        "hook_exact_bytes": True,
        "strict_pre_relu_path": True,
        "zid_relu_parity_verified": True,
        "z_dom_present": True,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "raw_iq_retained": False,
        "clean_iq_retained": False,
        "source_archive_phase1_only": True,
        "d102_rejected_bundle_reused": d102_reused,
    }
    receipt_path = tmp_path / "d105_strict_tap_receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    return {
        "arrays": arrays,
        "archive": archive,
        "archive_sha": archive_sha,
        "receipt": receipt_path,
        "receipt_sha": receipt_sha,
        "runtime": runtime_path,
        "runtime_sha": runtime_sha,
        "lock": lock_path,
        "lock_sha": lock_sha,
    }


def _manual_evidence(tmp_path: Path, tap: dict[str, object]) -> tuple[Path, Path, Path]:
    arrays = tap["arrays"]
    assert isinstance(arrays, dict)
    labels = arrays["labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    physical = arrays["physical_ids"].astype(str)
    receiver_tokens = tuple(sorted(set(receivers.tolist())))
    class_tokens = tuple(sorted(set(labels.tolist())))
    lineage = compute_d105_source_aggregate_lineage(
        strict_tap_receipt_sha256=str(tap["receipt_sha"]),
        checkpoint_sha256=CHECKPOINT_SHA256,
        runtime_sha256=str(tap["runtime_sha"]),
        method_lock_sha256=str(tap["lock_sha"]),
    )
    prediction_rows: list[dict[str, object]] = []
    truth_by_row: list[dict[str, object]] = []
    for receiver in receiver_tokens:
        query_ids = [
            physical[np.flatnonzero((receivers == receiver) & (labels == label))[-1]]
            for label in class_tokens
        ]
        for fold_kind, held_class, k_shot in (
            *(("receiver_held", None, k) for k in (1, 5, 10)),
            *(("class_loco", label, 1) for label in class_tokens),
        ):
            row = {
                "row_id": hashlib.sha256(
                    _canonical_bytes(
                        {
                            "receiver": receiver,
                            "fold_kind": fold_kind,
                            "held_class": held_class,
                            "K": k_shot,
                        }
                    )
                ).hexdigest(),
                "fold_kind": fold_kind,
                "held_receiver_token": receiver,
                "held_class_token": held_class,
                "K": k_shot,
                "query_physical_ids": query_ids,
                "m0_predictions": list(class_tokens),
                "d105_fp32_predictions": list(class_tokens),
                "d105_int8_predictions": list(class_tokens),
                "d105_fp32_top2_margins": [0.2] * len(class_tokens),
                "prediction_commit_sha256": "0" * 64,
                "query_rows_used_for_fit": 0,
            }
            row["prediction_commit_sha256"] = compute_d105_source_held_prediction_commit(
                checkpoint_sha256=CHECKPOINT_SHA256,
                runtime_sha256=str(tap["runtime_sha"]),
                method_lock_sha256=str(tap["lock_sha"]),
                strict_tap_receipt_sha256=str(tap["receipt_sha"]),
                source_aggregate_lineage_sha256=lineage,
                row=row,
            )
            prediction_rows.append(row)
            truth_by_row.append({"row_id": row["row_id"], "truth_labels": list(class_tokens)})
    tx_rows: list[dict[str, object]] = []
    tx_truth: list[dict[str, object]] = []
    for receiver in receiver_tokens:
        physical_ids = [
            physical[np.flatnonzero((receivers == receiver) & (labels == label))[-1]]
            for label in class_tokens
        ]
        row = {
            "held_receiver_token": receiver,
            "physical_ids": physical_ids,
            "predictions": [class_tokens[0]] * len(class_tokens),
            "prediction_commit_sha256": "0" * 64,
        }
        row["prediction_commit_sha256"] = compute_d105_source_held_tx_prediction_commit(
            checkpoint_sha256=CHECKPOINT_SHA256,
            runtime_sha256=str(tap["runtime_sha"]),
            method_lock_sha256=str(tap["lock_sha"]),
            strict_tap_receipt_sha256=str(tap["receipt_sha"]),
            source_aggregate_lineage_sha256=lineage,
            row=row,
        )
        tx_rows.append(row)
        tx_truth.append({"held_receiver_token": receiver, "truth_labels": list(class_tokens)})
    prediction = {
        "schema": SOURCE_HELD_PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": tap["runtime_sha"],
        "method_lock_sha256": tap["lock_sha"],
        "strict_tap_receipt_sha256": tap["receipt_sha"],
        "source_aggregate_lineage_sha256": lineage,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "receiver_tokens": list(receiver_tokens),
        "class_tokens": list(class_tokens),
        "scored_prediction_rows": prediction_rows,
        "tx_probe_prediction_rows": tx_rows,
        "d102_rejected_bundle_reused": False,
    }
    prediction_path = tmp_path / "source_held_prediction_manifest.json"
    prediction_sha = _write_json(prediction_path, prediction, immutable=True)
    truth_open = {
        "schema": SOURCE_HELD_TRUTH_OPEN_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": tap["runtime_sha"],
        "method_lock_sha256": tap["lock_sha"],
        "strict_tap_receipt_sha256": tap["receipt_sha"],
        "source_aggregate_lineage_sha256": lineage,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "prediction_manifest_immutable": True,
        "truth_opened_after_prediction": True,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "d102_rejected_bundle_reused": False,
    }
    truth_path = tmp_path / "source_held_truth_open_receipt.json"
    truth_sha = _write_json(truth_path, truth_open, immutable=True)
    score = {
        "schema": SOURCE_HELD_SCORE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": tap["runtime_sha"],
        "method_lock_sha256": tap["lock_sha"],
        "strict_tap_receipt_sha256": tap["receipt_sha"],
        "source_aggregate_lineage_sha256": lineage,
        "source_held_prediction_manifest_sha256": prediction_sha,
        "source_held_truth_open_receipt_sha256": truth_sha,
        "source_only": True,
        "target_rows": 0,
        "query_rows": 0,
        "scored_truth_rows": truth_by_row,
        "tx_probe_truth_rows": tx_truth,
        "d102_rejected_bundle_reused": False,
    }
    score_path = tmp_path / "source_held_score_artifact.json"
    _write_json(score_path, score, immutable=True)
    return prediction_path, truth_path, score_path


def _authority_material(
    component: Path,
    *,
    d102_revocation_manifest_sha256: str,
    nonce_ledger_dir: Path,
) -> tuple[Path, Path, Path, str]:
    asset = load_d105_phase1_asset(component)
    identity = phase1._authority_identity(
        asset, validated_bundle_id_sha256=VALIDATED_BUNDLE_ID
    )
    review = {
        "schema": authority.INDEPENDENT_REVIEW_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "component_manifest_sha256": identity["component_manifest_sha256"],
        "bundle_content_root_sha256": identity["bundle_content_root_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "method_lock_sha256": identity["method_lock_sha256"],
        "d105_candidate_runtime_manifest_sha256": identity[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": identity[
            "d105_candidate_method_lock_sha256"
        ],
        "reviewer_id": "independent-d105-test-reviewer",
        "reviewed_at": "2026-07-31T00:00:00Z",
        "review_p0": 0,
        "review_p1": 0,
    }
    review_path = component.parent / "independent_review_receipt.json"
    review_sha = _write_json(review_path, review)
    run_id = "d105-test-run-001"
    nonce_ledger_identity_sha256 = authority.compute_d105_nonce_ledger_identity(
        nonce_ledger_dir,
        run_id=run_id,
        signature_domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
    )
    envelope = authority.build_d105_authority_envelope(
        identity=identity,
        independent_review_receipt_sha256=review_sha,
        d102_revocation_manifest_sha256=d102_revocation_manifest_sha256,
        nonce_ledger_identity_sha256=nonce_ledger_identity_sha256,
        issued_at="2026-07-31T00:00:00Z",
        not_before="2026-07-31T00:00:00Z",
        expires_at="2027-07-31T00:00:00Z",
        nonce="b" * 64,
        run_id=run_id,
        git_commit="c" * 40,
    )
    envelope_path = component.parent / "authority_envelope.json"
    _write_json(envelope_path, envelope)
    signature_path = component.parent / "authority_envelope.ed25519"
    signature_path.write_bytes(
        authority.sign_d105_detached(
            domain=authority.AUTHORITY_SIGNATURE_DOMAIN,
            payload=envelope_path.read_bytes(),
            private_seed=TEST_AUTHORITY_SEED,
        )
    )
    return envelope_path, signature_path, review_path, run_id


def test_triple_immutable_evidence_builds_and_seals_aggregate_only_component(
    tmp_path: Path,
) -> None:
    revocation_manifest, revocation_signature, revocation_sha = _signed_d102_revocation(
        tmp_path
    )
    tap = _materialize_strict_tap(
        tmp_path, d102_revocation_manifest_sha256=revocation_sha
    )
    prediction, truth_open, score = _manual_evidence(tmp_path, tap)
    component = tmp_path / "component"
    result = build_d105_phase1_component(
        tap["archive"],
        tap["receipt"],
        tap["lock"],
        tap["runtime"],
        prediction,
        truth_open,
        score,
        revocation_manifest,
        revocation_signature,
        component,
    )
    assert result["status"] == COMPONENT_STATUS
    assert result["formal_phase2_eligible"] is False
    assert {item.name for item in component.iterdir()} == {
        "d105_phase1_aggregate.wire",
        "d105_source_held_gate.json",
        "d105_phase1_bundle.manifest.json",
        "d105_phase1_bundle.manifest.sha256",
        authority.D102_REVOCATION_MANIFEST_NAME,
        authority.D102_REVOCATION_SIGNATURE_NAME,
    }
    asset = load_d105_phase1_asset(component)
    assert asset.manifest["formal_phase2_eligibility_missing"] == (
        "independent_review_p0_0_p1_0",
        "independent_phase2_authority_seal",
    )
    emitted = b"".join(item.read_bytes() for item in component.iterdir())
    assert b"source-physical" not in emitted
    assert b"source-rx" not in emitted
    assert b"class-00" not in emitted

    nonce_ledger = tmp_path / "nonce-ledger"
    nonce_ledger.mkdir()
    envelope, signature, review, run_id = _authority_material(
        component,
        d102_revocation_manifest_sha256=revocation_sha,
        nonce_ledger_dir=nonce_ledger,
    )
    review_value = json.loads(review.read_text(encoding="utf-8"))
    review_value["review_p0"] = 1
    _write_json(review, review_value)
    with pytest.raises(D105Phase1BundleError, match="authority seal validation failed"):
        seal_d105_phase1_component(
            component,
            envelope,
            signature,
            review,
            revocation_manifest,
            revocation_signature,
            nonce_ledger,
            tmp_path / "d105-test-run-rejected-review",
        )
    review_value["review_p0"] = 0
    _write_json(review, review_value)
    sealed = tmp_path / run_id
    seal_result = seal_d105_phase1_component(
        component,
        envelope,
        signature,
        review,
        revocation_manifest,
        revocation_signature,
        nonce_ledger,
        sealed,
    )
    assert seal_result["status"] == FORMAL_STATUS
    sealed_asset = load_d105_phase1_asset(sealed, require_formal_phase2_eligible=True)
    handle = make_d105_phase1_runtime_handle(sealed_asset)
    assert handle.checkpoint_sha256 == CHECKPOINT_SHA256
    validated = validate_d105_phase1_asset(
        sealed, require_formal_phase2_eligible=True
    )
    assert validated["d105_candidate_method_lock_sha256"] == tap["lock_sha"]
    assert validated["d105_candidate_runtime_manifest_sha256"] == tap["runtime_sha"]
    assert {item.name for item in sealed.iterdir()} >= {
        authority.AUTHORITY_ENVELOPE_NAME,
        authority.AUTHORITY_SIGNATURE_NAME,
        authority.INDEPENDENT_REVIEW_RECEIPT_NAME,
        authority.D102_REVOCATION_MANIFEST_NAME,
        authority.D102_REVOCATION_SIGNATURE_NAME,
    }


def test_actual_predict_truth_open_score_interface_has_no_prediction_truth_surface(
    tmp_path: Path,
) -> None:
    _, _, revocation_sha = _signed_d102_revocation(tmp_path)
    tap = _materialize_strict_tap(
        tmp_path, d102_revocation_manifest_sha256=revocation_sha
    )
    prediction = tmp_path / "actual_prediction_manifest.json"
    result = execute_d105_source_held_predictions(
        tap["archive"], tap["receipt"], tap["lock"], tap["runtime"], prediction
    )
    assert result["prediction_truth_present"] is False
    assert prediction.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    prediction_value = json.loads(prediction.read_text(encoding="utf-8"))
    assert "truth_labels" not in _canonical_bytes(prediction_value).decode("utf-8")

    truth_open = tmp_path / "actual_truth_open_receipt.json"
    open_d105_source_held_truth(
        tap["archive"],
        tap["receipt"],
        tap["lock"],
        tap["runtime"],
        prediction,
        truth_open,
    )
    score = tmp_path / "actual_score_artifact.json"
    score_d105_source_held_truth(
        tap["archive"],
        tap["receipt"],
        tap["lock"],
        tap["runtime"],
        prediction,
        truth_open,
        score,
    )
    gate = derive_d105_source_held_gate(
        tap["archive"],
        tap["receipt"],
        tap["lock"],
        tap["runtime"],
        prediction,
        truth_open,
        score,
    )
    assert gate["gate"]["source_held_prediction_manifest_sha256"] == sha256_file(
        prediction
    )
    assert gate["gate"]["source_held_truth_open_receipt_sha256"] == sha256_file(
        truth_open
    )


def test_mutable_prediction_d102_and_runtime_code_drift_fail_closed(tmp_path: Path) -> None:
    _, _, revocation_sha = _signed_d102_revocation(tmp_path)
    tap = _materialize_strict_tap(
        tmp_path / "mutable", d102_revocation_manifest_sha256=revocation_sha
    )
    prediction, _, _ = _manual_evidence(tmp_path / "mutable", tap)
    os.chmod(prediction, stat.S_IREAD | stat.S_IWRITE)
    with pytest.raises(D105Phase1BundleError, match="read-only"):
        open_d105_source_held_truth(
            tap["archive"],
            tap["receipt"],
            tap["lock"],
            tap["runtime"],
            prediction,
            tmp_path / "mutable" / "cannot_open_truth.json",
        )

    d102_tap = _materialize_strict_tap(
        tmp_path / "d102",
        d102_revocation_manifest_sha256=revocation_sha,
        d102_reused=True,
    )
    with pytest.raises(D105Phase1BundleError, match="D102"):
        execute_d105_source_held_predictions(
            d102_tap["archive"],
            d102_tap["receipt"],
            d102_tap["lock"],
            d102_tap["runtime"],
            tmp_path / "d102" / "prediction.json",
        )

    runtime_value = json.loads(Path(tap["runtime"]).read_text(encoding="utf-8"))
    runtime_value["core_file_sha256"]["cvsrffi/stage2_d105_cbrc.py"] = "0" * 64
    _write_json(Path(tap["runtime"]), runtime_value)
    with pytest.raises(D105Phase1BundleError, match="core file SHA256 drift"):
        load_d105_candidate_runtime_manifest(
            tap["runtime"], expected_checkpoint_sha256=CHECKPOINT_SHA256
        )


def test_candidate_runtime_closure_requires_somph_trust_module(tmp_path: Path) -> None:
    """The authority trust root cannot remain outside the sealed file closure."""

    trust_relative_path = "cvsrffi/somph_runtime_trust.py"
    assert trust_relative_path in phase1.D105_CANDIDATE_RUNTIME_FILES
    runtime_path, _, _, _ = _candidate_identity(tmp_path)
    loaded = load_d105_candidate_runtime_manifest(
        runtime_path, expected_checkpoint_sha256=CHECKPOINT_SHA256
    )
    code_root = Path(phase1.__file__).resolve().parents[1]
    assert loaded["observed_core_file_sha256"][trust_relative_path] == sha256_file(
        code_root / trust_relative_path
    )

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["core_file_sha256"][trust_relative_path]
    _write_json(runtime_path, runtime)
    with pytest.raises(D105Phase1BundleError, match="identity drift"):
        load_d105_candidate_runtime_manifest(
            runtime_path, expected_checkpoint_sha256=CHECKPOINT_SHA256
        )


def test_candidate_runtime_file_set_is_exact_recursive_local_python_closure() -> None:
    code_root = Path(phase1.__file__).resolve().parents[1]
    declared = set(phase1.D105_CANDIDATE_RUNTIME_CVSRFFI_FILES)
    assert len(declared) == len(phase1.D105_CANDIDATE_RUNTIME_CVSRFFI_FILES)
    assert declared == _recursive_d105_cvsrffi_runtime_files(code_root)


@pytest.mark.parametrize("relative_path", phase1.D105_CANDIDATE_RUNTIME_FILES)
def test_candidate_runtime_closure_rejects_every_missing_listed_member(
    tmp_path: Path, relative_path: str
) -> None:
    runtime_path, _, _, _ = _candidate_identity(tmp_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["core_file_sha256"][relative_path]
    _write_json(runtime_path, runtime)
    with pytest.raises(D105Phase1BundleError, match="identity drift"):
        load_d105_candidate_runtime_manifest(
            runtime_path, expected_checkpoint_sha256=CHECKPOINT_SHA256
        )


@pytest.mark.parametrize("relative_path", phase1.D105_CANDIDATE_RUNTIME_FILES)
def test_candidate_runtime_closure_rejects_every_listed_member_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    runtime_path, _, _, _ = _candidate_identity(tmp_path)
    code_root = Path(phase1.__file__).resolve().parents[1]
    drifted_path = code_root / relative_path
    original_sha256_file = phase1.sha256_file

    def _sha256_with_simulated_content_drift(path: str | Path) -> str:
        if Path(path).resolve() == drifted_path.resolve():
            return hashlib.sha256(
                b"D105-candidate-runtime-content-drift\\0" + drifted_path.read_bytes()
            ).hexdigest()
        return original_sha256_file(path)

    monkeypatch.setattr(phase1, "sha256_file", _sha256_with_simulated_content_drift)
    with pytest.raises(D105Phase1BundleError, match="core file SHA256 drift"):
        load_d105_candidate_runtime_manifest(
            runtime_path, expected_checkpoint_sha256=CHECKPOINT_SHA256
        )


def test_d102_label_guard_does_not_treat_content_hash_as_lineage_label() -> None:
    # The somph closure changes the runtime-manifest digest cascade.  A digest
    # may contain the ASCII substring ``d102`` without naming D102 at all.
    phase1._reject_d102(
        {"prediction_commit_sha256": "c233a8f3c8518ac94e20bdbc9c6a45f00557f53d05d10257d28257de9ddf7a44"},
        name="test receipt",
    )
    with pytest.raises(D105Phase1BundleError, match="rejected D102 lineage"):
        phase1._reject_d102(
            {"run_id": "d102_rb_metabias4_phase1_analytic_held_20260724_r6"},
            name="test receipt",
        )


@pytest.mark.parametrize(
    ("surface", "needle", "replacement"),
    [
        (
            "fixed public key",
            b"PINNED_AUTHORITY_PUBLIC_KEY_HEX",
            b"PINNED_AUTHORITY_PUBLIC_KEY_HEY",
        ),
        (
            "fixed KeyID",
            b"PINNED_AUTHORITY_KEY_ID",
            b"PINNED_AUTHORITY_KEY_JD",
        ),
        (
            "Ed25519 verifier implementation",
            b"def verify_ed25519",
            b"def verify_ed25518",
        ),
    ],
)
def test_candidate_runtime_closure_rejects_somph_trust_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    needle: bytes,
    replacement: bytes,
) -> None:
    """One-byte-equivalent trust-root changes must fail before authority use."""

    runtime_path, _, _, _ = _candidate_identity(tmp_path)
    code_root = Path(phase1.__file__).resolve().parents[1]
    trust_path = code_root / "cvsrffi/somph_runtime_trust.py"
    original_bytes = trust_path.read_bytes()
    assert needle in original_bytes, surface
    drifted_bytes = original_bytes.replace(needle, replacement, 1)
    assert drifted_bytes != original_bytes

    original_sha256_file = phase1.sha256_file

    def _sha256_with_simulated_trust_drift(path: str | Path) -> str:
        if Path(path).resolve() == trust_path.resolve():
            return hashlib.sha256(drifted_bytes).hexdigest()
        return original_sha256_file(path)

    # Keep the shared checkout immutable while proving the loader compares the
    # trust module's exact content hash, rather than merely recognizing its path.
    monkeypatch.setattr(phase1, "sha256_file", _sha256_with_simulated_trust_drift)
    with pytest.raises(D105Phase1BundleError, match="core file SHA256 drift"):
        load_d105_candidate_runtime_manifest(
            runtime_path, expected_checkpoint_sha256=CHECKPOINT_SHA256
        )
