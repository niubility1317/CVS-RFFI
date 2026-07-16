from __future__ import annotations

import json
import hashlib
import secrets
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_leo_weak_lineage_seal as lineage
import cvsrffi.somph_lineage_authority as authority
import cvsrffi.somph_offline_target_package as producer
import cvsrffi.somph_predictor_bundle as bundle
import cvsrffi.somph_predictor_runtime as runtime
from cvsrffi.leo_weak_cache import ids_sha256
from cvsrffi.somph_metric_scorer import (
    FORMAL_NEW20_TX_LABELS,
    FORMAL_OLD_TX_LABELS,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.somph_predictor_runtime import expected_somph_method_lock
from cvsrffi.stage2_predictor_bundle import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def _json(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _signed_authority_envelope(
    lock: dict,
    *,
    build_receipt_sha256: str = "b" * 64,
) -> tuple[dict, bytes]:
    seed = secrets.token_bytes(32)
    hashed = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(
        bytes([hashed[0] & 248])
        + hashed[1:31]
        + bytes([(hashed[31] & 63) | 64]),
        "little",
    )
    public_key = authority._ed_encode(
        authority._ed_scalar_mult(authority._ED_B, scalar)
    )
    envelope = {
        "schema": "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1",
        "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
        "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
        "key_id": "somph-authority-ed25519-20260716",
        "lock_canonical_sha256": sha256_bytes(canonical_json_bytes(lock)),
        "authority_lock_build_receipt_sha256": build_receipt_sha256,
        "cache_spec_manifest_sha256": (
            "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
        ),
        "cache_spec_cell_id": (
            f"rx_{str(lock['receiver']).replace('-', '_')}_seed_{lock['seed']}"
        ),
        "signature_ed25519_hex": "",
    }
    message = authority._authority_signature_message(envelope)
    nonce = int.from_bytes(
        hashlib.sha512(hashed[32:] + message).digest(), "little"
    ) % authority._ED_L
    encoded_r = authority._ed_encode(
        authority._ed_scalar_mult(authority._ED_B, nonce)
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % authority._ED_L
    signature_scalar = (nonce + challenge * scalar) % authority._ED_L
    envelope["signature_ed25519_hex"] = (
        encoded_r + signature_scalar.to_bytes(32, "little")
    ).hex()
    return envelope, public_key


def _install_test_authority_verifier(
    monkeypatch: pytest.MonkeyPatch,
    public_key: bytes,
) -> None:
    def verify(
        envelope: dict,
        *,
        lock_canonical_sha256: str,
        expected_cache_spec_cell_id: str,
        expected_build_receipt_sha256: str | None = None,
    ) -> None:
        expected = {
            "schema": (
                "cvs.phase2.somph_leo_weak_signed_authority_envelope.v1"
            ),
            "domain": "cvs.somph.leo_weak.authority_lock.ed25519.v1",
            "issuer": "qknnv42_stage2bc_extreme_light_route_20260716",
            "key_id": "somph-authority-ed25519-20260716",
            "lock_canonical_sha256": lock_canonical_sha256,
            "authority_lock_build_receipt_sha256": (
                expected_build_receipt_sha256 or "b" * 64
            ),
            "cache_spec_manifest_sha256": (
                "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
            ),
            "cache_spec_cell_id": expected_cache_spec_cell_id,
        }
        if any(envelope.get(key) != value for key, value in expected.items()):
            raise authority.SomphLineageAuthorityError(
                "test authority envelope identity/binding drift"
            )
        authority._verify_ed25519(
            public_key,
            authority._authority_signature_message(envelope),
            bytes.fromhex(envelope["signature_ed25519_hex"]),
        )

    monkeypatch.setattr(authority, "_verify_signed_envelope", verify)


def _install_test_build_authority_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verify(
        _lock: dict,
        *,
        build_receipt_sha256: str,
        **_kwargs,
    ) -> tuple[str, str]:
        receipt_sha = authority._require_sha256(
            build_receipt_sha256,
            field="test build receipt SHA",
        )
        receiver, seed, roles = authority._validate_lock_formal_identity(_lock)
        return receipt_sha, authority._dataset_root_from_lock(
            _lock,
            receiver=receiver,
            seed=seed,
            roles=roles,
        )

    monkeypatch.setattr(authority, "_verify_build_authority_binding", verify)


def _write_valid_head(
    path: Path,
    *,
    state: dict,
    method_sha256: str,
    phase1_checkpoint_sha256: str,
    feature_runtime_sha256: str,
    enrollment_root_override: str | None = None,
) -> tuple[str, str]:
    registry = state["registered_classes"]
    class_count = len(registry)
    binding = {
        "schema": bundle.SOMPH_ENROLLMENT_BINDING_SCHEMA,
        "stage": state["stage"],
        "registration_state": (
            "before" if class_count == len(FORMAL_OLD_TX_LABELS) else "after"
        ),
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "registered_class_handles": [
            item["class_handle"] for item in registry
        ],
        "enrollment_package_root_sha256": (
            enrollment_root_override
            or state["enrollment_package_root_sha256"]
        ),
        "enrollment_package_seal_sha256": state[
            "enrollment_package_seal_sha256"
        ],
        "phase1_checkpoint_sha256": phase1_checkpoint_sha256,
        "feature_runtime_sha256": feature_runtime_sha256,
        "method_lock_sha256": method_sha256,
        "support_token_sha256_by_scenario": {
            scenario: "3" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "support_feature_sha256_by_scenario": {
            scenario: "4" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
    }
    binding_sha = sha256_bytes(canonical_json_bytes(binding))
    centroids = np.eye(class_count, 160, dtype=np.float16)
    prototypes = np.repeat(centroids, 2, axis=0)
    prototype_ids = np.repeat(
        np.arange(class_count, dtype=np.uint16), 2
    )
    payload: dict[str, np.ndarray] = {
        "schema_utf8": np.frombuffer(
            bundle.SOMPH_HEAD_CAPSULE_SCHEMA.encode("ascii"), dtype=np.uint8
        ).copy(),
        "method_lock_sha256_utf8": np.frombuffer(
            method_sha256.encode("ascii"), dtype=np.uint8
        ).copy(),
        "enrollment_binding_json_utf8": np.frombuffer(
            canonical_json_bytes(binding), dtype=np.uint8
        ).copy(),
        "class_count_uint16": np.asarray([class_count], dtype=np.uint16),
        "feature_dim_uint16": np.asarray([160], dtype=np.uint16),
        "k_shot_uint16": np.asarray([10], dtype=np.uint16),
    }
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload[f"{scenario}__prototypes_fp16"] = prototypes.copy()
        payload[f"{scenario}__prototype_class_ids_uint16"] = (
            prototype_ids.copy()
        )
        payload[f"{scenario}__centroids_fp16"] = centroids.copy()
        payload[f"{scenario}__residual_scale_fp16"] = np.ones(
            160, dtype=np.float16
        )
        payload[f"{scenario}__class_hubness_penalty_fp16"] = np.zeros(
            class_count, dtype=np.float16
        )
        payload[f"{scenario}__scalars_fp16"] = np.asarray(
            [0.75, 0.25], dtype=np.float16
        )
    assert tuple(payload) == bundle.HEAD_CAPSULE_NPZ_MEMBERS
    with path.open("xb") as handle:
        np.savez(handle, **payload)
    return sha256_file(path), binding_sha


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_new_class_count: int = 20,
    k_shot: int = 10,
    query_per_tx: int = 1,
) -> dict:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "best_joint_safe_ssdg.pth"
    checkpoint.write_bytes(b"synthetic-adv3b02-phase1-state-dict")
    checkpoint_sha = sha256_file(checkpoint)
    feature_runtime = source / "adv3b02_identity_runtime.pt"
    feature_runtime.write_bytes(b"synthetic-adv3b02-torchscript-runtime")
    monkeypatch.setattr(
        bundle, "ADV3B02_PHASE1_CHECKPOINT_SHA256", checkpoint_sha
    )
    monkeypatch.setattr(
        runtime, "ADV3B02_PHASE1_CHECKPOINT_SHA256", checkpoint_sha
    )
    method = expected_somph_method_lock()
    method["checkpoint_sha256"] = checkpoint_sha
    method_path = source / "method_lock.json"
    method_path.write_bytes(canonical_json_bytes(method))

    labels = [
        *[("target_old", value) for value in FORMAL_OLD_TX_LABELS],
        *[
            ("target_new", value)
            for value in FORMAL_NEW20_TX_LABELS[:cache_new_class_count]
        ],
    ]
    per_class = 20 + query_per_tx
    cache_paths: dict[str, str] = {}
    cache_hashes: dict[str, str] = {}
    scenario_receipts: dict[str, dict] = {}
    physical_ids_by_scenario: dict[str, list[str]] = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        roles: list[str] = []
        tx_ids: list[str] = []
        sample_ids: list[str] = []
        split_partition: list[str] = []
        split_rank: list[int] = []
        for role, tx_label in labels:
            for rank in range(per_class):
                roles.append(role)
                tx_ids.append(tx_label)
                sample_ids.append(
                    sha256_bytes(
                        (
                            f"physical|{scenario}|{role}|{tx_label}|"
                            f"20-1|{rank}"
                        ).encode("utf-8")
                    )
                )
                split_partition.append(
                    "support_pool" if rank < 20 else "query"
                )
                split_rank.append(rank if rank < 20 else rank - 20)
        physical_ids_by_scenario[scenario] = sample_ids
        row_count = len(sample_ids)
        iq = (
            np.arange(row_count * 8, dtype=np.float32).reshape(
                row_count, 2, 4
            )
            + scenario_index
        )
        iq_hashes = [bundle.iq_row_sha256(row) for row in iq]
        overlays = [
            sha256_bytes(f"{scenario}|{sample_id}".encode("utf-8"))
            for sample_id in sample_ids
        ]
        seeds = np.asarray(
            [713101 + scenario_index] * row_count, dtype=np.int64
        )
        cache = source / f"{scenario}.npz"
        with cache.open("xb") as handle:
            np.savez(
                handle,
                leo_weak_iq=iq,
                raw_labels=np.arange(row_count, dtype=np.int64),
                domain_labels=np.zeros(row_count, dtype=np.int64),
                tx_ids=np.asarray(tx_ids),
                rx_ids=np.asarray(["20-1"] * row_count),
                day_ids=np.asarray(["1"] * row_count),
                eq_ids=np.asarray(["1"] * row_count),
                sig_ids=np.asarray(
                    [f"{scenario}:{value}" for value in range(row_count)]
                ),
                source_dataset_sha256=np.asarray(
                    ["a" * 64 if role == "target_old" else "b" * 64
                     for role in roles]
                ),
                source_record_indices=np.arange(
                    row_count, dtype=np.int64
                )
                + scenario_index * row_count,
                dataset_role=np.asarray(roles),
                channel_views=np.asarray(["rx_base"] * row_count),
                sat_scenarios=np.asarray([scenario] * row_count),
                satellite_seeds=seeds,
                overlay_applied=np.ones(row_count, dtype=bool),
                sample_ids=np.asarray(sample_ids),
                post_channel_iq_sha256=np.asarray(iq_hashes),
                overlay_ids=np.asarray(overlays),
                manifest_json=np.asarray("{}"),
                split_partition=np.asarray(split_partition),
                split_rank=np.asarray(split_rank, dtype=np.int64),
            )
        cache_sha = sha256_file(cache)
        cache_paths[scenario] = cache.name
        cache_hashes[scenario] = cache_sha
        scenario_receipts[scenario] = {
            "cache_sha256": cache_sha,
            "cache_size_bytes": cache.stat().st_size,
            "cache_manifest_sha256": "1" * 64,
            "channel_config_sha256": "2" * 64,
            "physical_sample_ids_sha256": ids_sha256(sample_ids),
            "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
            "overlay_ids_sha256": ids_sha256(overlays),
            "row_count": row_count,
            "zip_member_crc_and_bounds_check": "PASS",
            "sample_level_overlay_recompute": "PASS",
        }
    physical_roots = {
        scenario: ids_sha256(physical_ids_by_scenario[scenario])
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    physical_assignment_root = sha256_bytes(
        canonical_json_bytes(physical_ids_by_scenario)
    )
    cache_set = {
        "schema": "cvs_leo_weak_iq_cache_set_v2",
        "artifact_stage": "phase1_offline_prechannel_export",
        "cache_set_id": "test",
        "cache_scope": "stage2_registered",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old", "target_new"],
        "cache_npz_by_scenario": cache_paths,
        "cache_sha256_by_scenario": cache_hashes,
        "cache_audits": {
            scenario: {} for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": (
            "immutable_preoverlay_lineage_token"
        ),
        "phase2_query_post_reception_view_fit_access": False,
        "physical_sample_scenario_assignment_policy": (
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": (
            physical_assignment_root
        ),
        "builder_sha256": "3" * 64,
        "build_spec_sha256": "4" * 64,
        "build_spec_path_exposed_to_phase2": False,
    }
    cache_set_path = source / "cache_set.json"
    _json(cache_set_path, cache_set)
    receipt = {
        "schema": lineage.LINEAGE_RECEIPT_SCHEMA,
        "status": "BYTE_GROUNDED_SELF_CONSISTENCY_PASS",
        "cache_scope": "stage2_registered",
        "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
        "cache_set_manifest_sha256": sha256_file(cache_set_path),
        "cache_set_manifest_size_bytes": cache_set_path.stat().st_size,
        "exporter_sha256": "5" * 64,
        "exporter_size_bytes": 1,
        "build_spec_sha256": "4" * 64,
        "build_spec_size_bytes": 1,
        "channel_code_closure_sha256": "6" * 64,
        "channel_code_members": [],
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": (
            physical_assignment_root
        ),
        "scenario_receipts": scenario_receipts,
        "same_fd_nofollow_read": True,
        "npz_member_crc_size_ratio_audit": "PASS",
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "sample_level_overlay_recompute": "PASS",
        "manifest_hex_self_declaration_sufficient": False,
        "external_authority_lock_verified": False,
        "contains_build_spec_or_dataset_paths": False,
        "formal_launch_authority": False,
    }
    receipt_path = source / "lineage_receipt.json"
    _json(receipt_path, receipt)
    seal = {
        "schema": lineage.LINEAGE_SEAL_SCHEMA,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_size_bytes": receipt_path.stat().st_size,
        "lineage_root_sha256": sha256_bytes(canonical_json_bytes(receipt)),
    }
    seal_path = source / "lineage_receipt.seal.json"
    _json(seal_path, seal)
    return {
        "cache_set_manifest_path": cache_set_path,
        "verified_lineage_loader": (
            producer.load_verified_lineage_context_from_receipt_seal
        ),
        "verified_lineage_loader_kwargs": {
            "lineage_receipt_path": receipt_path,
            "lineage_seal_path": seal_path,
            "expected_lineage_receipt_sha256": sha256_file(receipt_path),
            "expected_lineage_seal_sha256": sha256_file(seal_path),
        },
        "phase1_checkpoint_path": checkpoint,
        "sealed_feature_runtime_path": feature_runtime,
        "method_lock_path": method_path,
        "output_root": tmp_path / "built",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": k_shot,
        "new_class_count": 5,
        "query_per_tx": query_per_tx,
        "token_secret": b"s" * 32,
    }


def _attach_authority_bundle(
    kwargs: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    bundle_root = tmp_path / "authority_bundle"
    bundle_root.mkdir()
    cache_set_path = Path(kwargs["cache_set_manifest_path"])
    diagnostic_kwargs = kwargs["verified_lineage_loader_kwargs"]
    receipt_path = Path(diagnostic_kwargs["lineage_receipt_path"])
    seal_path = Path(diagnostic_kwargs["lineage_seal_path"])
    cache_set = json.loads(cache_set_path.read_text(encoding="utf-8"))
    new_tx = list(FORMAL_NEW20_TX_LABELS)
    dataset_rows = [
        {
            "role": "target_old",
            "path": str(tmp_path / "ManySig.pkl"),
            "sha256": "a" * 64,
            "size_bytes": 101,
            "tx_ids": list(FORMAL_OLD_TX_LABELS),
        },
        {
            "role": "target_new",
            "path": str(tmp_path / "ManyTx.pkl"),
            "sha256": "b" * 64,
            "size_bytes": 202,
            "tx_ids": new_tx,
        },
    ]
    authority_lock = {
        "schema": authority.AUTHORITY_LOCK_SCHEMA,
        "receiver": kwargs["receiver"],
        "seed": kwargs["seed"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": list(FORMAL_OLD_TX_LABELS),
        "new_tx_ids": new_tx,
        "cache_set_manifest": {
            "path": str(cache_set_path.resolve()),
            "sha256": sha256_file(cache_set_path),
            "size_bytes": cache_set_path.stat().st_size,
        },
        "cache_sha256_by_scenario": cache_set["cache_sha256_by_scenario"],
        "exporter": {
            "path": str(tmp_path / "exporter.py"),
            "sha256": "5" * 64,
            "size_bytes": 303,
        },
        "build_spec": {
            "path": str(tmp_path / "build_spec.json"),
            "file_sha256": "d" * 64,
            "canonical_sha256": "4" * 64,
            "size_bytes": 404,
        },
        "channel_code_closure": {
            "closure_sha256": "6" * 64,
            "members": [
                {
                    "logical_name": logical_name,
                    "path": str(tmp_path / f"channel_{index}.py"),
                    "sha256": f"{index + 1:x}" * 64,
                    "size_bytes": index + 1,
                }
                for index, logical_name in enumerate(
                    authority.CHANNEL_CODE_LOGICAL_MEMBERS
                )
            ],
        },
        "channel_config_sha256_by_scenario": {
            scenario: "2" * 64 for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": cache_set[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": cache_set[
            "physical_sample_scenario_assignment_sha256"
        ],
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "post_channel_iq_sha256_root_by_scenario": {
            scenario: json.loads(receipt_path.read_text(encoding="utf-8"))[
                "scenario_receipts"
            ][scenario]["post_channel_iq_sha256_root"]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "overlay_ids_sha256_by_scenario": {
            scenario: json.loads(receipt_path.read_text(encoding="utf-8"))[
                "scenario_receipts"
            ][scenario]["overlay_ids_sha256"]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_role_inputs_root_sha256": "e" * 64,
        "datasets": dataset_rows,
    }
    lock_path = bundle_root / authority.AUTHORITY_LOCK_NAME
    _json(lock_path, authority_lock)
    build_receipt_path = (
        bundle_root / authority.AUTHORITY_LOCK_BUILD_RECEIPT_NAME
    )
    _json(build_receipt_path, {"fixture": True})
    cache_spec_manifest_path = (
        bundle_root / authority.CACHE_SPEC_MANIFEST_NAME
    )
    _json(cache_spec_manifest_path, {"fixture": True})
    signed_envelope, public_key = _signed_authority_envelope(
        authority_lock,
        build_receipt_sha256=sha256_file(build_receipt_path),
    )
    _install_test_authority_verifier(monkeypatch, public_key)
    _install_test_build_authority_verifier(monkeypatch)
    envelope_path = bundle_root / authority.AUTHORITY_ENVELOPE_NAME
    _json(envelope_path, signed_envelope)
    structural_receipt = bundle_root / authority.STRUCTURAL_RECEIPT_NAME
    structural_seal = bundle_root / authority.STRUCTURAL_SEAL_NAME
    structural_receipt.write_bytes(receipt_path.read_bytes())
    structural_seal.write_bytes(seal_path.read_bytes())
    lock_sha = sha256_file(lock_path)
    lock_canonical_sha = sha256_bytes(canonical_json_bytes(authority_lock))
    envelope_sha = sha256_file(envelope_path)
    dataset_root = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                    "tx_ids": row["tx_ids"],
                    "receiver": kwargs["receiver"],
                    "dataset_seed": kwargs["seed"] + index * 10_007,
                }
                for index, row in enumerate(dataset_rows)
            ]
        )
    )
    attestation = {
        "schema": authority.AUTHORITY_ATTESTATION_SCHEMA,
        "status": authority.AUTHORITY_STATUS,
        "external_authority_lock_verified": True,
        "authority_lock_sha256": lock_sha,
        "authority_lock_canonical_sha256": lock_canonical_sha,
        "signed_authority_envelope_sha256": envelope_sha,
        "authority_issuer": (
            "qknnv42_stage2bc_extreme_light_route_20260716"
        ),
        "authority_key_id": "somph-authority-ed25519-20260716",
        "receiver": kwargs["receiver"],
        "seed": kwargs["seed"],
        "cache_scope": "stage2_registered",
        "old_tx_ids_sha256": sha256_bytes(
            canonical_json_bytes(list(FORMAL_OLD_TX_LABELS))
        ),
        "new_tx_ids_sha256": sha256_bytes(canonical_json_bytes(new_tx)),
        "dataset_authority_root_sha256": dataset_root,
        "cache_role_inputs_root_sha256": authority_lock[
            "cache_role_inputs_root_sha256"
        ],
        **authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
        "physical_sample_scenario_assignment_policy": (
            authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": cache_set[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": cache_set[
            "physical_sample_scenario_assignment_sha256"
        ],
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "structural_receipt_sha256": sha256_file(structural_receipt),
        "structural_detached_seal_sha256": sha256_file(structural_seal),
        "formal_launch_authority": False,
    }
    attestation_path = bundle_root / authority.AUTHORITY_ATTESTATION_NAME
    _json(attestation_path, attestation)
    member_names = (
        authority.AUTHORITY_LOCK_NAME,
        authority.AUTHORITY_ENVELOPE_NAME,
        authority.AUTHORITY_LOCK_BUILD_RECEIPT_NAME,
        authority.CACHE_SPEC_MANIFEST_NAME,
        authority.STRUCTURAL_RECEIPT_NAME,
        authority.STRUCTURAL_SEAL_NAME,
        authority.AUTHORITY_ATTESTATION_NAME,
    )
    members = [
        {
            "name": name,
            "sha256": sha256_file(bundle_root / name),
            "size_bytes": (bundle_root / name).stat().st_size,
        }
        for name in member_names
    ]
    commit = {
        "schema": authority.AUTHORITY_COMMIT_SCHEMA,
        "status": authority.AUTHORITY_STATUS,
        "external_authority_lock_verified": True,
        "authority_lock_sha256": lock_sha,
        "signed_authority_envelope_sha256": envelope_sha,
        "authority_issuer": (
            "qknnv42_stage2bc_extreme_light_route_20260716"
        ),
        "authority_key_id": "somph-authority-ed25519-20260716",
        "members": members,
        "bundle_root_sha256": sha256_bytes(canonical_json_bytes(members)),
        "formal_launch_authority": False,
    }
    commit_path = bundle_root / authority.AUTHORITY_COMMIT_NAME
    _json(commit_path, commit)
    kwargs.pop("verified_lineage_loader")
    kwargs.pop("verified_lineage_loader_kwargs")
    kwargs.update(
        {
            "authority_bundle_root": bundle_root,
            "expected_authority_commit_sha256": sha256_file(commit_path),
        }
    )
    return kwargs


def test_builds_matched_before_after_enrollment_and_apply_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    calls: list[tuple[str, str]] = []
    original = bundle.write_somph_predictor_bundle

    def spy(*args, **values):
        calls.append((values["profile"], values["registration_state"]))
        return original(*args, **values)

    monkeypatch.setattr(bundle, "write_somph_predictor_bundle", spy)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    assert calls == [
        (bundle.ENROLLMENT_ONLY, "before"),
        (bundle.ENROLLMENT_ONLY, "after"),
    ]
    assert result["status"] == (
        "STAGE1_ENROLLMENT_PACKAGES_AND_APPLY_STAGING_READY"
    )
    assert result["formal_launch_authority"] is False
    assert result["external_authority_lock_verified"] is False
    assert result["authority_commit_sha256"] is None
    assert result["formal_authority_state"] == (
        "DIAGNOSTIC_STRUCTURAL_ONLY_NO_FORMAL_AUTHORITY"
    )
    assert result["same_scenario_support_query_disjointness_audit"] == "PASS"
    assert (
        result["cross_scenario_selected_physical_disjointness_audit"]
        == "PASS"
    )
    assert result["before_after_old_support_query_reuse_audit"] == "PASS"
    assert result["token_secret_persisted"] is False
    assert result["states"]["before"]["stage"] == "stage2b"
    assert result["states"]["after"]["stage"] == "stage2c"
    assert result["phase1_checkpoint_lineage"] == {
        "path": str(Path(kwargs["phase1_checkpoint_path"]).resolve()),
        "sha256": sha256_file(kwargs["phase1_checkpoint_path"]),
        "copied_into_phase2_predictor_package": False,
    }
    assert result["sealed_feature_runtime"]["sha256"] == sha256_file(
        kwargs["sealed_feature_runtime_path"]
    )

    before = Path(result["states"]["before"]["enrollment_package_root"])
    after = Path(result["states"]["after"]["enrollment_package_root"])
    before_apply = Path(result["states"]["before"]["apply_staging_root"])
    after_apply = Path(result["states"]["after"]["apply_staging_root"])
    for root in (before_apply, after_apply):
        assert (root / bundle.FEATURE_RUNTIME_RELATIVE_PATH).is_file()
        assert not (root / "checkpoint.pt").exists()
        assert not (root / "head_capsule.npz").exists()
        assert not (root / "package_manifest.json").exists()
        assert not any(root.glob("support_*.npz"))
        assert len(list(root.glob("query_*.npz"))) == 3
    assert not any(before.glob("query_*.npz"))
    assert not any(after.glob("query_*.npz"))
    for root in (before, after):
        assert (root / bundle.FEATURE_RUNTIME_RELATIVE_PATH).is_file()
        assert not (root / "checkpoint.pt").exists()
    before_manifest = json.loads(
        (before / bundle.MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    after_manifest = json.loads(
        (after / bundle.MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert before_manifest["stage"] == "stage2b"
    assert after_manifest["stage"] == "stage2c"
    assert before_manifest["phase1_checkpoint_sha256"] == sha256_file(
        kwargs["phase1_checkpoint_path"]
    )
    assert before_manifest["feature_runtime_sha256"] == sha256_file(
        kwargs["sealed_feature_runtime_path"]
    )

    with np.load(before / "support_leo_clear_weak.npz") as archive:
        before_tokens = archive["support_tokens"].astype(str)
        before_iq = archive["support_leo_weak_iq"]
        assert len(before_tokens) == 6 * 10
    with np.load(after / "support_leo_clear_weak.npz") as archive:
        after_tokens = archive["support_tokens"].astype(str)
        after_iq = archive["support_leo_weak_iq"]
        assert len(after_tokens) == 11 * 10
    assert np.array_equal(before_tokens, after_tokens[: 6 * 10])
    assert np.array_equal(before_iq, after_iq[: 6 * 10])

    with np.load(before_apply / "query_leo_clear_weak.npz") as archive:
        before_query = archive["query_tokens"].astype(str)
    with np.load(after_apply / "query_leo_clear_weak.npz") as archive:
        after_query = archive["query_tokens"].astype(str)
    assert set(before_query.tolist()).issubset(set(after_query.tolist()))
    assert len(after_query) == 11
    scenario_orders = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        with np.load(after_apply / f"query_{scenario}.npz") as archive:
            scenario_orders.append(archive["query_tokens"].astype(str).tolist())
    assert all(
        set(scenario_orders[left]).isdisjoint(scenario_orders[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )

    pair_path = Path(result["registration_pair_manifest"])
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    assert pair["old_support_physical_ids_sha256_before"] == pair[
        "old_support_physical_ids_sha256_after"
    ]
    assert pair["old_query_physical_ids_sha256_before"] == pair[
        "old_query_physical_ids_sha256_after"
    ]
    truth = json.loads(Path(result["truth_sidecar"]).read_text(encoding="utf-8"))
    assert {row["evaluation_role"] for row in truth["rows"]} == {
        "target_old",
        "target_new",
    }
    assert len(truth["rows"]) == 3 * 11
    assert len({row["query_token"] for row in truth["rows"]}) == 3 * 11
    assert len({row["physical_sample_id"] for row in truth["rows"]}) == 3 * 11
    predictor_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in Path(kwargs["output_root"]).glob(
            "predictor/**/overlay_provenance.json"
        )
    )
    assert "14-10" not in predictor_text
    assert "target_old" not in predictor_text
    assert "build_spec" not in predictor_text


def test_lineage_expected_sha_is_external_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    loader_kwargs = dict(kwargs["verified_lineage_loader_kwargs"])
    loader_kwargs["expected_lineage_receipt_sha256"] = "0" * 64
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="receipt/seal SHA mismatch",
    ):
        producer.load_verified_lineage_context_from_receipt_seal(
            cache_set_manifest_path=kwargs["cache_set_manifest_path"],
            **loader_kwargs,
        )


def test_authority_commit_context_builds_matching_formal_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    result = producer.build_somph_offline_row_pair(**kwargs)
    assert result["external_authority_lock_verified"] is True
    assert result["authority_commit_sha256"] == kwargs[
        "expected_authority_commit_sha256"
    ]
    assert result["authority_attestation_sha256"] == sha256_file(
        Path(kwargs["authority_bundle_root"])
        / authority.AUTHORITY_ATTESTATION_NAME
    )
    assert result["formal_authority_state"] == (
        "VERIFIED_SINGLE_OBSERVATION_AUTHORITY_V2"
    )
    staging_authority = json.loads(
        Path(
            result["states"]["before"]["apply_staging_authority"]
        ).read_text(encoding="utf-8")
    )
    assert staging_authority["schema"] == (
        producer.FORMAL_APPLY_STAGING_AUTHORITY_SCHEMA
    )
    assert staging_authority["profile"] == "formal_external_authority"
    assert staging_authority["stage"] == "stage2b"
    assert staging_authority["registration_state"] == "before"
    assert staging_authority[
        "cache_physical_sample_ids_sha256_by_scenario"
    ] == result["physical_sample_ids_sha256_by_scenario"]
    assert result["formal_launch_authority"] is False


def test_one_max_new20_diagnostic_builds_nested_5_10_20_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch, query_per_tx=20)
    old_support_roots = set()
    old_query_roots = set()
    old_query_token_sets = []
    for new_class_count in (5, 10, 20):
        row_kwargs = {
            **kwargs,
            "new_class_count": new_class_count,
            "output_root": tmp_path / f"built_new{new_class_count}",
        }
        result = producer.build_somph_offline_row_pair_diagnostic(
            **row_kwargs
        )
        pair = json.loads(
            Path(result["registration_pair_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        old_support_roots.add(
            pair["old_support_physical_ids_sha256_before"]
        )
        old_query_roots.add(pair["old_query_physical_ids_sha256_before"])
        with np.load(
            Path(result["states"]["after"]["apply_staging_root"])
            / "query_leo_clear_weak.npz"
        ) as archive:
            tokens = archive["query_tokens"].astype(str)
        truth = json.loads(
            Path(result["truth_sidecar"]).read_text(encoding="utf-8")
        )
        old_tokens = {
            row["query_token"]
            for row in truth["rows"]
            if row["evaluation_role"] == "target_old"
        }
        all_truth_tokens = {
            row["query_token"] for row in truth["rows"]
        }
        old_query_token_sets.append(old_tokens)
        assert set(tokens.tolist()).issubset(all_truth_tokens)
        assert len(tokens) == (len(FORMAL_OLD_TX_LABELS) + new_class_count) * 20
        assert len(truth["rows"]) == 3 * len(tokens)
    assert len(old_support_roots) == 1
    assert len(old_query_roots) == 1
    assert old_query_token_sets[0] == old_query_token_sets[1]
    assert old_query_token_sets[1] == old_query_token_sets[2]


def test_k1_k5_are_k10_prefixes_but_each_package_exposes_exact_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _fixture(tmp_path, monkeypatch, query_per_tx=20)
    tokens_by_k: dict[int, dict[int, list[str]]] = {}
    query_tokens_by_k: dict[int, set[str]] = {}
    for k_shot in (1, 5, 10, 20):
        result = producer.build_somph_offline_row_pair_diagnostic(
            **{
                **base,
                "k_shot": k_shot,
                "output_root": tmp_path / f"built_k{k_shot}",
            }
        )
        root = Path(result["states"]["before"]["enrollment_package_root"])
        manifest = json.loads(
            (root / bundle.MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        assert manifest["k_shot"] == k_shot
        assert manifest["support_pool_max_k"] == k_shot
        with np.load(root / "support_leo_clear_weak.npz") as archive:
            labels = archive["support_class_indices"].astype(int)
            tokens = archive["support_tokens"].astype(str)
        assert len(tokens) == len(FORMAL_OLD_TX_LABELS) * k_shot
        tokens_by_k[k_shot] = {
            class_index: tokens[labels == class_index].tolist()
            for class_index in range(len(FORMAL_OLD_TX_LABELS))
        }
        apply_root = Path(result["states"]["before"]["apply_staging_root"])
        with np.load(apply_root / "query_leo_clear_weak.npz") as archive:
            query_tokens_by_k[k_shot] = set(
                archive["query_tokens"].astype(str).tolist()
            )
    for class_index in range(len(FORMAL_OLD_TX_LABELS)):
        assert tokens_by_k[1][class_index] == tokens_by_k[10][class_index][:1]
        assert tokens_by_k[5][class_index] == tokens_by_k[10][class_index][:5]
        assert tokens_by_k[10][class_index] == tokens_by_k[20][class_index][:10]
    assert len({frozenset(value) for value in query_tokens_by_k.values()}) == 1


def test_formal_builder_requires_q20_before_authority_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch, query_per_tx=1)
    kwargs.pop("verified_lineage_loader")
    kwargs.pop("verified_lineage_loader_kwargs")
    kwargs.update(
        {
            "authority_bundle_root": tmp_path / "missing_authority",
            "expected_authority_commit_sha256": "0" * 64,
        }
    )
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="exactly 20 query samples per TX",
    ):
        producer.build_somph_offline_row_pair(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_builder_rejects_phase1_state_dict_as_feature_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    kwargs["sealed_feature_runtime_path"] = kwargs["phase1_checkpoint_path"]
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="paths must differ",
    ):
        producer.build_somph_offline_row_pair_diagnostic(**kwargs)


def test_authority_loader_rejects_alternate_cache_set_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    alternate = tmp_path / "alternate_cache_set.json"
    alternate.write_bytes(Path(kwargs["cache_set_manifest_path"]).read_bytes())
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="caller cache-set path differs",
    ):
        producer.load_verified_lineage_context_from_authority_commit(
            cache_set_manifest_path=alternate,
            authority_bundle_root=kwargs["authority_bundle_root"],
            expected_authority_commit_sha256=kwargs[
                "expected_authority_commit_sha256"
            ],
        )


def test_authority_loader_starts_from_external_commit_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="authority commit consumer verification",
    ):
        producer.load_verified_lineage_context_from_authority_commit(
            cache_set_manifest_path=kwargs["cache_set_manifest_path"],
            authority_bundle_root=kwargs["authority_bundle_root"],
            expected_authority_commit_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("receiver", "3-19"),
        ("seed", 713102),
    ),
)
def test_authority_context_rejects_requested_row_drift_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    kwargs[field] = value
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="does not match the requested formal row",
    ):
        producer.build_somph_offline_row_pair(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_authority_context_rejects_cache_tx_coverage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    cache_set = json.loads(
        Path(kwargs["cache_set_manifest_path"]).read_text(encoding="utf-8")
    )
    cache = Path(kwargs["cache_set_manifest_path"]).parent / cache_set[
        "cache_npz_by_scenario"
    ][scenario]
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    new_mask = payload["dataset_role"].astype(str) == "target_new"
    payload["tx_ids"][np.flatnonzero(new_mask)[0]] = "19-6"
    cache.unlink()
    with cache.open("xb") as handle:
        np.savez(handle, **payload)
    new_sha = sha256_file(cache)
    cache_set["cache_sha256_by_scenario"][scenario] = new_sha
    _json(Path(kwargs["cache_set_manifest_path"]), cache_set)
    # Rebinding the self-consistent structural artifacts is deliberately not
    # enough: the already committed authority lock still names the old bytes.
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="authority|cache-set|committed member",
    ):
        producer.build_somph_offline_row_pair(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_registration_pair_finalizer_is_exact_and_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    target = tmp_path / "registration_pair.json"
    pair = producer.finalize_registration_pair_manifest(
        staging_manifest_path=result["registration_pair_manifest"],
        output_path=target,
        before_binding_sha256="1" * 64,
        after_binding_sha256="2" * 64,
    )
    assert pair["schema"] == "cvs.phase2.somph_registration_pair.v1"
    assert pair["before_binding_sha256"] == "1" * 64
    with pytest.raises(FileExistsError):
        producer.finalize_registration_pair_manifest(
            staging_manifest_path=result["registration_pair_manifest"],
            output_path=target,
            before_binding_sha256="1" * 64,
            after_binding_sha256="2" * 64,
        )


def test_untrusted_lineage_context_is_rejected_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)

    def bad_loader(**_values):
        return {"schema": "wrong"}

    kwargs["verified_lineage_loader"] = bad_loader
    kwargs["verified_lineage_loader_kwargs"] = {}
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="context exact schema",
    ):
        producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    assert not Path(kwargs["output_root"]).exists()


def test_diagnostic_callback_cannot_mint_formal_staging_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    legitimate = producer.load_verified_lineage_context_from_receipt_seal(
        cache_set_manifest_path=kwargs["cache_set_manifest_path"],
        **kwargs["verified_lineage_loader_kwargs"],
    )
    forged = {
        **legitimate,
        "authority_lock": {"forged": True},
        "authority_attestation": {"forged": True},
        "authority_commit": {"forged": True},
        "authority_commit_sha256": "a" * 64,
        "authority_attestation_sha256": "b" * 64,
        "external_authority_lock_verified": True,
    }

    def forged_loader(**_values):
        return forged

    kwargs["verified_lineage_loader"] = forged_loader
    kwargs["verified_lineage_loader_kwargs"] = {}
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    state = result["states"]["before"]
    staging_authority = json.loads(
        Path(state["apply_staging_authority"]).read_text(encoding="utf-8")
    )
    assert staging_authority["schema"] == (
        producer.DIAGNOSTIC_APPLY_STAGING_AUTHORITY_SCHEMA
    )
    assert staging_authority["profile"] == "diagnostic_structural_only"
    assert staging_authority["external_authority_lock_verified"] is False
    assert staging_authority["authority_commit_sha256"] is None

    head = tmp_path / "forged_callback_head.npz"
    head_sha, binding_sha = _write_valid_head(
        head,
        state=state,
        method_sha256=sha256_file(kwargs["method_lock_path"]),
        phase1_checkpoint_sha256=sha256_file(
            kwargs["phase1_checkpoint_path"]
        ),
        feature_runtime_sha256=sha256_file(
            kwargs["sealed_feature_runtime_path"]
        ),
    )
    staging = Path(state["apply_staging_root"])
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="seal exact schema drift",
    ):
        producer.finalize_somph_apply_package(
            apply_staging_root=staging,
            detached_seal_path=tmp_path / "forged_formal.seal.json",
            staging_authority_path=state["apply_staging_authority"],
            staging_authority_seal_path=state[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head,
            expected_head_capsule_sha256=head_sha,
            expected_head_enrollment_binding_sha256=binding_sha,
            authority_bundle_root=tmp_path / "forged_authority",
            expected_authority_commit_sha256="a" * 64,
        )
    assert not (staging / "head_capsule.npz").exists()


def test_apply_finalizer_checks_external_head_before_staging_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    staging = Path(result["states"]["before"]["apply_staging_root"])
    head = tmp_path / "head_capsule.npz"
    head.write_bytes(b"not-a-valid-head")
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="complete prevalidation",
    ):
        producer.finalize_somph_apply_package_diagnostic(
            apply_staging_root=staging,
            detached_seal_path=tmp_path / "apply.seal.json",
            staging_authority_path=result["states"]["before"][
                "apply_staging_authority"
            ],
            staging_authority_seal_path=result["states"]["before"][
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=result["states"][
                "before"
            ]["apply_staging_authority_seal_sha256"],
            head_capsule_path=head,
            expected_head_capsule_sha256="0" * 64,
            expected_head_enrollment_binding_sha256="1" * 64,
        )
    assert not (staging / "head_capsule.npz").exists()


def test_formal_apply_finalizer_rejects_diagnostic_staging_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    state = result["states"]["before"]
    head = tmp_path / "valid_head.npz"
    head_sha, binding_sha = _write_valid_head(
        head,
        state=state,
        method_sha256=sha256_file(kwargs["method_lock_path"]),
        phase1_checkpoint_sha256=sha256_file(
            kwargs["phase1_checkpoint_path"]
        ),
        feature_runtime_sha256=sha256_file(
            kwargs["sealed_feature_runtime_path"]
        ),
    )
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="seal exact schema drift",
    ):
        producer.finalize_somph_apply_package(
            apply_staging_root=state["apply_staging_root"],
            detached_seal_path=tmp_path / "formal_apply.seal.json",
            staging_authority_path=state["apply_staging_authority"],
            staging_authority_seal_path=state[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head,
            expected_head_capsule_sha256=head_sha,
            expected_head_enrollment_binding_sha256=binding_sha,
            authority_bundle_root=tmp_path / "missing_authority",
            expected_authority_commit_sha256="0" * 64,
        )
    assert not (
        Path(state["apply_staging_root"]) / "head_capsule.npz"
    ).exists()


def test_formal_apply_finalizer_consumes_sealed_authority_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _attach_authority_bundle(
        _fixture(tmp_path, monkeypatch, query_per_tx=20),
        tmp_path,
        monkeypatch,
    )
    result = producer.build_somph_offline_row_pair(**kwargs)
    state = result["states"]["before"]
    head = tmp_path / "formal_valid_head.npz"
    head_sha, binding_sha = _write_valid_head(
        head,
        state=state,
        method_sha256=sha256_file(kwargs["method_lock_path"]),
        phase1_checkpoint_sha256=sha256_file(
            kwargs["phase1_checkpoint_path"]
        ),
        feature_runtime_sha256=sha256_file(
            kwargs["sealed_feature_runtime_path"]
        ),
    )
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="authority bundle verification failed",
    ):
        producer.finalize_somph_apply_package(
            apply_staging_root=state["apply_staging_root"],
            detached_seal_path=tmp_path / "wrong_commit.seal.json",
            staging_authority_path=state["apply_staging_authority"],
            staging_authority_seal_path=state[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head,
            expected_head_capsule_sha256=head_sha,
            expected_head_enrollment_binding_sha256=binding_sha,
            authority_bundle_root=kwargs["authority_bundle_root"],
            expected_authority_commit_sha256="0" * 64,
        )
    assert not (
        Path(state["apply_staging_root"]) / "head_capsule.npz"
    ).exists()
    finalized = producer.finalize_somph_apply_package(
        apply_staging_root=state["apply_staging_root"],
        detached_seal_path=tmp_path / "formal_apply.seal.json",
        staging_authority_path=state["apply_staging_authority"],
        staging_authority_seal_path=state[
            "apply_staging_authority_seal"
        ],
        expected_staging_authority_seal_sha256=state[
            "apply_staging_authority_seal_sha256"
        ],
        head_capsule_path=head,
        expected_head_capsule_sha256=head_sha,
        expected_head_enrollment_binding_sha256=binding_sha,
        authority_bundle_root=kwargs["authority_bundle_root"],
        expected_authority_commit_sha256=kwargs[
            "expected_authority_commit_sha256"
        ],
    )
    assert finalized["external_authority_lock_verified"] is True
    assert finalized["authority_commit_sha256"] == kwargs[
        "expected_authority_commit_sha256"
    ]
    assert finalized["formal_launch_authority"] is False


def test_apply_finalizer_rejects_head_from_other_enrollment_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    state = result["states"]["before"]
    head = tmp_path / "wrong_enrollment_head.npz"
    head_sha, binding_sha = _write_valid_head(
        head,
        state=state,
        method_sha256=sha256_file(kwargs["method_lock_path"]),
        phase1_checkpoint_sha256=sha256_file(
            kwargs["phase1_checkpoint_path"]
        ),
        feature_runtime_sha256=sha256_file(
            kwargs["sealed_feature_runtime_path"]
        ),
        enrollment_root_override="9" * 64,
    )
    staging = Path(state["apply_staging_root"])
    with pytest.raises(
        producer.SomphOfflinePackageError,
        match="does not match sealed apply staging authority",
    ):
        producer.finalize_somph_apply_package_diagnostic(
            apply_staging_root=staging,
            detached_seal_path=tmp_path / "apply.seal.json",
            staging_authority_path=state["apply_staging_authority"],
            staging_authority_seal_path=state[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head,
            expected_head_capsule_sha256=head_sha,
            expected_head_enrollment_binding_sha256=binding_sha,
        )
    assert not (staging / "head_capsule.npz").exists()
    assert not (staging / "package_manifest.json").exists()


def test_apply_finalizer_rolls_back_head_after_late_bundle_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    result = producer.build_somph_offline_row_pair_diagnostic(**kwargs)
    state = result["states"]["before"]
    head = tmp_path / "valid_head.npz"
    head_sha, binding_sha = _write_valid_head(
        head,
        state=state,
        method_sha256=sha256_file(kwargs["method_lock_path"]),
        phase1_checkpoint_sha256=sha256_file(
            kwargs["phase1_checkpoint_path"]
        ),
        feature_runtime_sha256=sha256_file(
            kwargs["sealed_feature_runtime_path"]
        ),
    )

    def fail_bundle(*_args, **_kwargs):
        raise OSError("injected late bundle failure")

    monkeypatch.setattr(bundle, "write_somph_predictor_bundle", fail_bundle)
    staging = Path(state["apply_staging_root"])
    seal = tmp_path / "apply.seal.json"
    with pytest.raises(OSError, match="injected late bundle failure"):
        producer.finalize_somph_apply_package_diagnostic(
            apply_staging_root=staging,
            detached_seal_path=seal,
            staging_authority_path=state["apply_staging_authority"],
            staging_authority_seal_path=state[
                "apply_staging_authority_seal"
            ],
            expected_staging_authority_seal_sha256=state[
                "apply_staging_authority_seal_sha256"
            ],
            head_capsule_path=head,
            expected_head_capsule_sha256=head_sha,
            expected_head_enrollment_binding_sha256=binding_sha,
        )
    assert not (staging / "head_capsule.npz").exists()
    assert not (staging / "package_manifest.json").exists()
    assert not seal.exists()
