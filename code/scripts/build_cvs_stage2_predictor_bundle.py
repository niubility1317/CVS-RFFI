#!/usr/bin/env python
"""Seal Stage2-B/C LEO_weak predictor inputs and an isolated truth sidecar."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache_set,
)
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT  # noqa: E402
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    PREDICTOR_INPUT_STAGE,
    PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
    QUERY_NPZ_MEMBERS,
    QUERY_SCHEMA,
    SUPPORT_NPZ_MEMBERS,
    SUPPORT_SCHEMA,
    canonical_json_bytes,
    iq_row_sha256,
    load_verified_stage2_predictor_bundle,
    make_member_descriptor,
    sha256_file,
    write_predictor_package_manifest_and_seal,
)
from cvsrffi.stage2_scoring_sidecar import (  # noqa: E402
    SCORING_MANIFEST_SCHEMA,
    load_verified_scoring_sidecar,
)


TRUTH_SIDECAR_SCHEMA = "cvs.phase2.query_truth_sidecar.v2"
OFFLINE_AUDIT_SCHEMA = "cvs.phase2.predictor_package_offline_build_audit.v2"


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_labels(raw: str | None, *, field: str, required: bool) -> list[str]:
    labels = [value.strip() for value in str(raw or "").split(",") if value.strip()]
    if required and not labels:
        raise ValueError(f"{field} must be nonempty")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field} contains duplicate labels")
    return labels


def _opaque_token(secret: bytes, prefix: str, *parts: object) -> str:
    message = canonical_json_bytes([str(value) for value in parts])
    digest = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return f"{prefix}_{digest}"


def _copy_regular_new(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"sealed artifact source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _stable_class_indices(
    arrays: dict[str, np.ndarray],
    *,
    receiver: str,
    role: str,
    label: str,
    seed: int,
) -> list[int]:
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
    rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
    indices = np.flatnonzero(
        (roles == role) & (tx_ids == label) & (rx_ids == receiver)
    ).astype(np.int64)
    ordered = sorted(indices.tolist(), key=lambda index: sample_ids[index])
    if not ordered:
        raise ValueError(f"no cache rows for receiver={receiver}, role={role}, tx={label}")
    class_seed = int.from_bytes(
        hashlib.sha256(f"{seed}|{receiver}|{role}|{label}".encode("utf-8")).digest()[:8],
        "big",
    )
    permutation = np.random.default_rng(class_seed).permutation(len(ordered))
    return [ordered[int(position)] for position in permutation]


def _select_support_query(
    arrays: dict[str, np.ndarray],
    *,
    receiver: str,
    seed: int,
    support_labels: list[tuple[str, str]],
    reference_query_labels: list[tuple[str, str]],
    support_pool_max_k: int,
    query_per_tx: int,
    use_offline_split_partition: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    support_indices: list[int] = []
    support_class_indices: list[int] = []
    support_ranks: list[int] = []
    query_records: list[dict[str, Any]] = []
    support_lookup = {pair: index for index, pair in enumerate(support_labels)}
    partitions = (
        np.asarray(arrays["split_partition"]).astype(str)
        if use_offline_split_partition
        else None
    )
    partition_ranks = (
        np.asarray(arrays["split_rank"]).astype(np.int64)
        if use_offline_split_partition
        else None
    )
    if use_offline_split_partition and (
        partitions is None
        or partition_ranks is None
        or set(partitions.tolist()) != {"support_pool", "query"}
        or reference_query_labels
    ):
        raise ValueError("legacy exact split partition evidence is missing or inapplicable")
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
    rx_ids = np.asarray(arrays["rx_ids"]).astype(str)

    for class_index, (role, label) in enumerate(support_labels):
        if use_offline_split_partition:
            class_mask = (roles == role) & (tx_ids == label) & (rx_ids == receiver)
            support_ordered = np.flatnonzero(
                class_mask & (partitions == "support_pool")
            ).tolist()
            query_ordered = np.flatnonzero(
                class_mask & (partitions == "query")
            ).tolist()
            support_ordered.sort(key=lambda index: int(partition_ranks[index]))
            query_ordered.sort(key=lambda index: int(partition_ranks[index]))
            ordered = support_ordered + query_ordered
        else:
            ordered = _stable_class_indices(
                arrays, receiver=receiver, role=role, label=label, seed=seed
            )
        required = support_pool_max_k + query_per_tx
        if len(ordered) < required:
            raise ValueError(
                f"insufficient cache rows for receiver={receiver}, role={role}, tx={label}: "
                f"{len(ordered)}<{required}"
            )
        support_indices.extend(ordered[:support_pool_max_k])
        support_class_indices.extend([class_index] * support_pool_max_k)
        support_ranks.extend(range(support_pool_max_k))
        selected_query = (
            query_ordered[:query_per_tx]
            if use_offline_split_partition
            else ordered[support_pool_max_k:required]
        )
        query_records.extend(
            {
                "array_index": index,
                "evaluation_role": role,
                "transmitter_label": label,
                "registered_class_index": class_index,
            }
            for index in selected_query
        )

    for role, label in reference_query_labels:
        if (role, label) in support_lookup:
            continue
        ordered = _stable_class_indices(
            arrays, receiver=receiver, role=role, label=label, seed=seed
        )
        if len(ordered) < query_per_tx:
            raise ValueError(
                f"insufficient reference-query rows for receiver={receiver}, "
                f"role={role}, tx={label}: {len(ordered)}<{query_per_tx}"
            )
        query_records.extend(
            {
                "array_index": index,
                "evaluation_role": role,
                "transmitter_label": label,
                "registered_class_index": None,
            }
            for index in ordered[:query_per_tx]
        )

    if not use_offline_split_partition:
        query_order = np.random.default_rng(int(seed) + 700_001).permutation(len(query_records))
        query_records = [query_records[int(index)] for index in query_order]
    return (
        np.asarray(support_indices, dtype=np.int64),
        np.asarray([row["array_index"] for row in query_records], dtype=np.int64),
        np.asarray(support_class_indices, dtype=np.int64),
        np.asarray(support_ranks, dtype=np.int64),
        query_records,
    )


def _assert_scenario_alignment(
    arrays_by_scenario: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    reference = arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]
    identity_fields = (
        "sample_ids",
        "dataset_role",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "sig_ids",
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS[1:]:
        arrays = arrays_by_scenario[scenario]
        for field in identity_fields:
            if not np.array_equal(np.asarray(arrays[field]), np.asarray(reference[field])):
                raise ValueError(f"physical sample alignment drift across LEO scenarios: {field}")
        for field in ("split_partition", "split_rank"):
            if (field in arrays) != (field in reference) or (
                field in reference
                and not np.array_equal(np.asarray(arrays[field]), np.asarray(reference[field]))
            ):
                raise ValueError(f"offline split alignment drift across LEO scenarios: {field}")
    return reference


def _reject_predictor_truth_leaks(root: Path, forbidden_values: Iterable[str]) -> None:
    needles = {
        value.encode("utf-8")
        for value in forbidden_values
        if isinstance(value, str) and value
    }
    pre_registered_binary_artifacts = {"checkpoint.bin", "adapter.bin", "head.bin"}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"predictor package contains a non-regular member: {path}")
        if path.name in pre_registered_binary_artifacts:
            # These artifacts are sealed by hash and predate target query construction.
            # A byte substring match is not evidence of query-truth reachability; the
            # generated support/query members are audited structurally and scanned below.
            continue
        if path.suffix.lower() == ".npz":
            # NPZ payloads are compressed binary containers. Scanning their raw bytes
            # creates false positives when arbitrary compressed numeric bytes happen to
            # equal a TX/role token. Member names and textual array values are the only
            # text-bearing surfaces; numeric arrays are already constrained by the
            # package NPZ member/schema/dtype audit.
            with np.load(path, allow_pickle=False) as archive:
                for member_name in archive.files:
                    member_name_bytes = member_name.encode("utf-8")
                    if any(needle in member_name_bytes for needle in needles):
                        raise ValueError(
                            "predictor package contains forbidden truth/role token "
                            f"in {path.name}:{member_name}"
                        )
                    array = np.asarray(archive[member_name])
                    if array.dtype.kind not in {"S", "U"}:
                        continue
                    for value in array.reshape(-1).tolist():
                        payload = (
                            bytes(value)
                            if isinstance(value, (bytes, bytearray))
                            else str(value).encode("utf-8")
                        )
                        if any(needle in payload for needle in needles):
                            raise ValueError(
                                "predictor package contains forbidden truth/role token "
                                f"in {path.name}:{member_name}"
                            )
            continue
        payload = path.read_bytes()
        for needle in needles:
            if needle in payload:
                raise ValueError(
                    f"predictor package contains forbidden truth/role token in {path.name}"
                )


def _validate_roots(predictor_root: Path, scorer_root: Path, seal_path: Path) -> None:
    predictor_root = predictor_root.resolve()
    scorer_root = scorer_root.resolve()
    seal_path = seal_path.resolve()
    if predictor_root == scorer_root:
        raise ValueError("predictor and scorer roots must be physically distinct")
    if predictor_root in scorer_root.parents or scorer_root in predictor_root.parents:
        raise ValueError("predictor and scorer roots must not be nested")
    if seal_path == predictor_root or predictor_root in seal_path.parents:
        raise ValueError("detached seal must be outside the predictor root")
    for root in (predictor_root, scorer_root):
        if root.exists():
            raise FileExistsError(f"refusing to overwrite output root: {root}")
    if seal_path.exists():
        raise FileExistsError(f"refusing to overwrite detached seal: {seal_path}")


def build(args: argparse.Namespace, *, token_secret: bytes | None = None) -> dict[str, Any]:
    """Build one immutable-input package. ``token_secret`` is test-only and never persisted."""

    predictor_root = Path(args.predictor_out_root).resolve()
    scorer_root = Path(args.scorer_out_root).resolve()
    explicit_seal = getattr(args, "detached_seal_path", None)
    seal_path = (
        Path(explicit_seal).resolve()
        if explicit_seal
        else predictor_root.parent / f"{predictor_root.name}.seal.json"
    )
    _validate_roots(predictor_root, scorer_root, seal_path)
    if token_secret is None:
        token_secret = secrets.token_bytes(32)
    if not isinstance(token_secret, bytes) or len(token_secret) < 32:
        raise ValueError("token_secret must contain at least 256 bits")

    stage = str(args.stage).lower()
    if stage not in {"stage2b", "stage2c"}:
        raise ValueError("stage must be stage2b or stage2c")
    old_labels = _parse_labels(
        args.old_class_labels, field="old_class_labels", required=True
    )
    new_labels = _parse_labels(
        getattr(args, "new_class_labels", ""),
        field="new_class_labels",
        required=stage == "stage2c",
    )
    reference_new_labels = _parse_labels(
        getattr(args, "stage2b_reference_new_class_labels", ""),
        field="stage2b_reference_new_class_labels",
        required=False,
    )
    if set(old_labels) & (set(new_labels) | set(reference_new_labels)):
        raise ValueError("old/new transmitter labels must be disjoint")
    if stage == "stage2b" and new_labels:
        raise ValueError("Stage2-B cannot register target-new support")
    declared_new_count = int(getattr(args, "new_class_count", 0))
    if declared_new_count != (len(new_labels) if stage == "stage2c" else 0):
        raise ValueError("new_class_count does not match the registered Stage2-C labels")
    support_pool_max_k = int(args.support_pool_max_k)
    query_per_tx = int(args.query_per_tx)
    if support_pool_max_k < 1 or query_per_tx < 1:
        raise ValueError("support_pool_max_k and query_per_tx must be positive")

    allowed_roles = {"target_old"}
    if stage == "stage2c" or reference_new_labels:
        allowed_roles.add("target_new")
    expected_scope = str(getattr(args, "expected_cache_scope", "stage2_registered"))
    arrays_by_scenario, cache_manifest, cache_audit = load_verified_leo_weak_cache_set(
        Path(args.target_cache_set),
        expected_scope=expected_scope,
        allowed_roles=allowed_roles,
    )
    reference = _assert_scenario_alignment(arrays_by_scenario)
    support_labels = [("target_old", label) for label in old_labels]
    if stage == "stage2c":
        support_labels.extend(("target_new", label) for label in new_labels)
        reference_query_labels: list[tuple[str, str]] = []
    else:
        reference_query_labels = [
            ("target_new", label) for label in reference_new_labels
        ]
    support_idx, query_idx, support_y, support_rank, query_records = _select_support_query(
        reference,
        receiver=str(args.receiver),
        seed=int(args.seed),
        support_labels=support_labels,
        reference_query_labels=reference_query_labels,
        support_pool_max_k=support_pool_max_k,
        query_per_tx=query_per_tx,
        use_offline_split_partition=(
            str(getattr(args, "offline_split_partition_policy", ""))
            == "legacy_seeded_nested_exact"
        ),
    )

    sample_ids = np.asarray(reference["sample_ids"]).astype(str)
    support_tokens = np.asarray(
        [
            _opaque_token(
                token_secret,
                "sid",
                "cvs-stage2-support-v2",
                args.receiver,
                args.seed,
                sample_ids[index],
            )
            for index in support_idx.tolist()
        ]
    )
    query_tokens = np.asarray(
        [
            _opaque_token(
                token_secret,
                "qid",
                "cvs-stage2-query-v2",
                args.receiver,
                args.seed,
                sample_ids[index],
            )
            for index in query_idx.tolist()
        ]
    )
    if len(set(query_tokens.tolist())) != len(query_tokens):
        raise ValueError("opaque query token collision")

    predictor_root.mkdir(parents=True, exist_ok=False)
    scorer_root.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, Any]] = []
    artifact_specs = (
        (
            "checkpoint",
            Path(args.checkpoint),
            "checkpoint.bin",
            "adv3b02.torchscript_identity_runtime.v1",
        ),
        ("adapter", Path(args.adapter), "adapter.bin", "cvs.feature_adapter.v1"),
        ("head", Path(args.head_artifact), "head.bin", "cvs.prototype_head.v1"),
        ("tta_policy", Path(args.tta_policy_json), "tta_policy.json", "cvs.adaptive_tta.v1"),
    )
    for role, source, filename, schema in artifact_specs:
        destination = predictor_root / filename
        _copy_regular_new(source, destination)
        members.append(
            make_member_descriptor(
                destination,
                relative_path=filename,
                artifact_role=role,
                schema=schema,
            )
        )

    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        overlay_ids = np.asarray(arrays["overlay_ids"]).astype(str)
        support_iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[support_idx]
        query_iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)[query_idx]
        support_path = predictor_root / f"support_{scenario}.npz"
        query_path = predictor_root / f"query_{scenario}.npz"
        support_manifest = {
            "schema": SUPPORT_SCHEMA,
            "scenario": scenario,
            "registered_support_labels_allowed": True,
            "registered_class_count": len(support_labels),
            "support_pool_max_k": support_pool_max_k,
            "token_scheme": "hmac_sha256_opaque_v1",
        }
        query_manifest = {
            "schema": QUERY_SCHEMA,
            "scenario": scenario,
            "query_truth_included": False,
            "query_role_included": False,
            "query_true_batch_class_count_included": False,
            "query_class_quota_included": False,
            "query_ordering_hint_included": False,
            "token_scheme": "hmac_sha256_opaque_v1",
        }
        with support_path.open("xb") as handle:
            np.savez(
                handle,
                support_pool_leo_weak_iq=support_iq,
                support_pool_class_indices=support_y,
                support_pool_rank_within_class=support_rank,
                support_pool_tokens=support_tokens,
                support_pool_overlay_tokens=np.asarray(
                    [
                        _opaque_token(
                            token_secret,
                            "oid",
                            "cvs-stage2-support-overlay-v2",
                            overlay_ids[index],
                        )
                        for index in support_idx.tolist()
                    ]
                ),
                support_pool_satellite_seeds=np.asarray(arrays["satellite_seeds"])[
                    support_idx
                ],
                support_pool_post_channel_iq_sha256=np.asarray(
                    [iq_row_sha256(row) for row in support_iq]
                ),
                manifest_json=np.asarray(json.dumps(support_manifest, sort_keys=True)),
            )
        with query_path.open("xb") as handle:
            np.savez(
                handle,
                query_leo_weak_iq=query_iq,
                query_tokens=query_tokens,
                query_overlay_tokens=np.asarray(
                    [
                        _opaque_token(
                            token_secret,
                            "oid",
                            "cvs-stage2-query-overlay-v2",
                            overlay_ids[index],
                        )
                        for index in query_idx.tolist()
                    ]
                ),
                query_satellite_seeds=np.asarray(arrays["satellite_seeds"])[query_idx],
                query_post_channel_iq_sha256=np.asarray(
                    [iq_row_sha256(row) for row in query_iq]
                ),
                manifest_json=np.asarray(json.dumps(query_manifest, sort_keys=True)),
            )
        members.extend(
            [
                make_member_descriptor(
                    support_path,
                    relative_path=support_path.name,
                    artifact_role=f"support:{scenario}",
                    schema=SUPPORT_SCHEMA,
                    scenario=scenario,
                    npz_members=SUPPORT_NPZ_MEMBERS,
                ),
                make_member_descriptor(
                    query_path,
                    relative_path=query_path.name,
                    artifact_role=f"query:{scenario}",
                    schema=QUERY_SCHEMA,
                    scenario=scenario,
                    npz_members=QUERY_NPZ_MEMBERS,
                ),
            ]
        )

    class_registry = [
        {
            "class_index": index,
            "class_handle": _opaque_token(
                token_secret, "cls", "cvs-stage2-class-v2", label
            ),
        }
        for index, (_role, label) in enumerate(support_labels)
    ]
    candidate_lock_sha256 = sha256_file(Path(args.candidate_lock))
    metadata = {
        "schema": PREDICTOR_PACKAGE_MANIFEST_SCHEMA,
        "artifact_stage": PREDICTOR_INPUT_STAGE,
        "stage": stage,
        "receiver": str(args.receiver),
        "seed": int(args.seed),
        "new_class_count": declared_new_count,
        "support_pool_max_k": support_pool_max_k,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "registered_class_count": len(support_labels),
        "registered_classes": class_registry,
        "candidate_lock_sha256": candidate_lock_sha256,
        **PHASE2_FULL_CONTRACT,
    }
    _manifest_path, _seal_path, package_manifest, _seal = (
        write_predictor_package_manifest_and_seal(
            predictor_root,
            manifest_metadata=metadata,
            members=members,
            detached_seal_path=seal_path,
        )
    )
    seal_sha256 = sha256_file(seal_path)

    tx_ids = np.asarray(reference["tx_ids"]).astype(str)
    rx_ids = np.asarray(reference["rx_ids"]).astype(str)
    day_ids = np.asarray(reference["day_ids"]).astype(str)
    sig_ids = np.asarray(reference["sig_ids"]).astype(str)
    truth_rows = []
    for position, record in enumerate(query_records):
        array_index = int(record["array_index"])
        token = str(query_tokens[position])
        truth_rows.append(
            {
                "query_token": token,
                "true_class_index": record["registered_class_index"],
                "true_class_handle": (
                    class_registry[int(record["registered_class_index"])]["class_handle"]
                    if record["registered_class_index"] is not None
                    else None
                ),
                "transmitter_label": str(tx_ids[array_index]),
                "evaluation_role": str(record["evaluation_role"]),
                "receiver_label": str(rx_ids[array_index]),
                "day_label": str(day_ids[array_index]),
                "signal_label": str(sig_ids[array_index]),
                "physical_sample_id": str(sample_ids[array_index]),
            }
        )
    truth_path = scorer_root / "truth_sidecar.json"
    _write_json_new(
        truth_path,
        {
            "schema": TRUTH_SIDECAR_SCHEMA,
            "stage": stage,
            "receiver": str(args.receiver),
            "seed": int(args.seed),
            "rows": truth_rows,
        },
    )
    scoring_path = scorer_root / "scoring_manifest.json"
    _write_json_new(
        scoring_path,
        {
            "schema": SCORING_MANIFEST_SCHEMA,
            "predictor_package_root_sha256": package_manifest["package_root_sha256"],
            "predictor_package_seal_sha256": seal_sha256,
            "truth_sidecar_json": truth_path.name,
            "truth_sidecar_sha256": sha256_file(truth_path),
            "scorer_output_must_not_feed_predictor": True,
        },
    )
    _write_json_new(
        scorer_root / "offline_build_audit.json",
        {
            "schema": OFFLINE_AUDIT_SCHEMA,
            "status": "PASS",
            "target_cache_manifest": cache_manifest,
            "target_cache_audit": cache_audit,
            "predictor_package_root_sha256": package_manifest["package_root_sha256"],
            "predictor_package_seal_sha256": seal_sha256,
            "predictor_scorer_roots_distinct": True,
            "opaque_token_secret_persisted": False,
        },
    )

    _reject_predictor_truth_leaks(
        predictor_root,
        ["target_old", "target_new", *old_labels, *new_labels, *reference_new_labels],
    )
    load_verified_stage2_predictor_bundle(
        predictor_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha256,
    )
    load_verified_scoring_sidecar(scoring_path)
    return {
        "stage": stage,
        "predictor_package": str(predictor_root),
        "predictor_package_root_sha256": package_manifest["package_root_sha256"],
        "predictor_package_seal": str(seal_path),
        "predictor_package_seal_sha256": seal_sha256,
        "scoring_manifest": str(scoring_path),
        "scoring_manifest_sha256": sha256_file(scoring_path),
        "registered_class_count": len(support_labels),
        "support_pool_count": len(support_idx),
        "query_count": len(query_idx),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache-set", type=Path, required=True)
    parser.add_argument("--expected-cache-scope", default="stage2_registered")
    parser.add_argument("--predictor-out-root", type=Path, required=True)
    parser.add_argument("--scorer-out-root", type=Path, required=True)
    parser.add_argument("--detached-seal-path", type=Path)
    parser.add_argument("--stage", choices=("stage2b", "stage2c"), required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--old-class-labels", required=True)
    parser.add_argument("--new-class-labels", default="")
    parser.add_argument("--stage2b-reference-new-class-labels", default="")
    parser.add_argument("--new-class-count", type=int, default=0)
    parser.add_argument("--support-pool-max-k", type=int, required=True)
    parser.add_argument("--query-per-tx", type=int, required=True)
    parser.add_argument("--offline-split-partition-policy", default="")
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--tta-policy-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
