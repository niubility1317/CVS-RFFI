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
from typing import Any, Iterable, Mapping

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
from cvsrffi.phase1_adv3b02_deployment_bundle import (  # noqa: E402
    load_formal_adv3b02_deployment_bundle,
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
FORMAL_DEPLOYMENT_BINDING_SCHEMA = (
    "cvs.full_ablation.phase1.deployment_binding.v1"
)
FORMAL_CLASS_LABEL_BINDING_SCHEMA = (
    "cvs.full_ablation.phase1.class_label_binding.v1"
)
_FORMAL_DEPLOYMENT_BINDING_KEYS = {
    "schema",
    "package_root",
    "detached_seal_path",
    "detached_seal_sha256",
    "signature_envelope_path",
    "signature_envelope_sha256",
    "checkpoint_lineage_sha256",
    "runtime_sha256",
    "component_pre_sign_content_root_sha256",
    "class_handle_binding_sha256",
    "parity_receipt_sha256",
    "generation_lock_sha256",
    "method_lock_sha256",
    "generation_config_sha256",
    "generation_code_sha256",
    "outer_content_root_sha256",
    "phase1_completion_receipt_path",
    "phase1_completion_receipt_sha256",
    "generation_config_path",
    "prototype_pt_path",
    "prototype_pt_sha256",
    "prototype_json_path",
    "prototype_json_sha256",
}
_FORMAL_CLASS_LABEL_BINDING_KEYS = {
    "schema",
    "checkpoint_lineage_sha256",
    "class_handle_binding_sha256",
    "formal_deployment_binding_sha256",
    "source_mapping_sha256",
    "source_checkpoint_sha256",
    "source_mapping_reused",
    "cross_launch_data_identity_required",
    "entries",
}
_FORMAL_CLASS_LABEL_ENTRY_KEYS = {
    "class_index",
    "phase1_tx",
    "registered_class_handle",
}


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


def _complete_target_new_pool(
    arrays_by_scenario: dict[str, dict[str, np.ndarray]],
    *,
    receiver: str,
) -> list[str]:
    """Return the canonical full target-new TX pool for one receiver."""

    reference: list[str] | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        roles = np.asarray(arrays["dataset_role"]).astype(str)
        rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
        tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
        current = sorted(
            set(tx_ids[(roles == "target_new") & (rx_ids == receiver)].tolist())
        )
        if not current:
            raise ValueError(
                f"no target-new cache pool for receiver={receiver}, scenario={scenario}"
            )
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(
                "complete target-new cache pool drifts across LEO_weak scenarios"
            )
    assert reference is not None
    return reference


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
    support_seed: int,
    query_seed: int,
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
            support_ordered = _stable_class_indices(
                arrays,
                receiver=receiver,
                role=role,
                label=label,
                seed=support_seed,
            )
            query_ordered = _stable_class_indices(
                arrays,
                receiver=receiver,
                role=role,
                label=label,
                seed=query_seed,
            )
            ordered = support_ordered
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
            else [
                index
                for index in query_ordered
                if index not in set(ordered[:support_pool_max_k])
            ][:query_per_tx]
        )
        if len(selected_query) != query_per_tx:
            raise ValueError(
                f"insufficient support-disjoint query rows for receiver={receiver}, "
                f"role={role}, tx={label}"
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
            arrays,
            receiver=receiver,
            role=role,
            label=label,
            seed=query_seed,
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
        query_order = np.random.default_rng(
            int(query_seed) + 700_001
        ).permutation(len(query_records))
        query_records = [query_records[int(index)] for index in query_order]
    return (
        np.asarray(support_indices, dtype=np.int64),
        np.asarray([row["array_index"] for row in query_records], dtype=np.int64),
        np.asarray(support_class_indices, dtype=np.int64),
        np.asarray(support_ranks, dtype=np.int64),
        query_records,
    )


def _canonical_manifest_split_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "canonical_physical_sample_ids",
        "split_roles",
        "split_ranks",
    )
    present = [field for field in required if field in arrays]
    if len(present) != len(required):
        raise ValueError(
            "manifest_all requires the complete canonical split member trio"
        )
    canonical_raw = np.asarray(arrays["canonical_physical_sample_ids"])
    roles_raw = np.asarray(arrays["split_roles"])
    ranks_raw = np.asarray(arrays["split_ranks"])
    if any(value.ndim != 1 for value in (canonical_raw, roles_raw, ranks_raw)):
        raise ValueError("canonical split members must be one-dimensional")
    row_count = len(np.asarray(arrays["sample_ids"]))
    if any(len(value) != row_count for value in (canonical_raw, roles_raw, ranks_raw)):
        raise ValueError("canonical split member lengths are inconsistent")
    if canonical_raw.dtype.kind not in {"S", "U"}:
        raise ValueError("canonical physical sample IDs must be strings")
    canonical_ids = canonical_raw.astype(str)
    if any(not value for value in canonical_ids.tolist()):
        raise ValueError("canonical physical sample IDs must be nonempty")
    if len(set(canonical_ids.tolist())) != row_count:
        raise ValueError("canonical physical sample IDs must be unique")
    if roles_raw.dtype.kind not in {"S", "U"}:
        raise ValueError("canonical split roles must be strings")
    split_roles = roles_raw.astype(str)
    if not set(split_roles.tolist()).issubset({"support", "query"}):
        raise ValueError("canonical split roles must be exactly support or query")
    if ranks_raw.dtype.kind not in {"i", "u"} or np.any(ranks_raw < 0):
        raise ValueError("canonical split ranks must be nonnegative exact integers")
    return canonical_ids, split_roles, np.asarray(ranks_raw, dtype=np.int64)


def _select_manifest_all_support_query(
    arrays: dict[str, np.ndarray],
    *,
    receiver: str,
    support_labels: list[tuple[str, str]],
    support_pool_max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    canonical_ids, split_roles, split_ranks = _canonical_manifest_split_arrays(
        arrays
    )
    dataset_roles = np.asarray(arrays["dataset_role"]).astype(str)
    tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
    rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
    registered_index = {
        pair: class_index for class_index, pair in enumerate(support_labels)
    }
    selected_receiver_indices = np.flatnonzero(rx_ids == receiver).astype(np.int64)
    if not len(selected_receiver_indices):
        raise ValueError(f"canonical split has no rows for receiver={receiver}")
    for index in selected_receiver_indices.tolist():
        pair = (str(dataset_roles[index]), str(tx_ids[index]))
        if pair not in registered_index:
            split_role = str(split_roles[index])
            raise ValueError(
                f"canonical {split_role} truth is outside the registered class set"
            )

    support_indices: list[int] = []
    support_class_indices: list[int] = []
    support_ranks: list[int] = []
    for class_index, (dataset_role, label) in enumerate(support_labels):
        current = np.flatnonzero(
            (rx_ids == receiver)
            & (dataset_roles == dataset_role)
            & (tx_ids == label)
            & (split_roles == "support")
        ).astype(np.int64)
        if len(current) != support_pool_max_k:
            raise ValueError(
                "canonical support must contain exactly K rows for every "
                f"receiver and registered class: receiver={receiver}, "
                f"role={dataset_role}, tx={label}"
            )
        observed_ranks = [int(split_ranks[index]) for index in current.tolist()]
        if sorted(observed_ranks) != list(range(support_pool_max_k)):
            raise ValueError(
                "canonical support ranks must be unique and gap-free in 0..K-1"
            )
        ordered = sorted(
            current.tolist(),
            key=lambda index: (
                int(split_ranks[index]),
                str(canonical_ids[index]),
            ),
        )
        support_indices.extend(ordered)
        support_class_indices.extend([class_index] * support_pool_max_k)
        support_ranks.extend(int(split_ranks[index]) for index in ordered)

    query_indices = np.flatnonzero(
        (rx_ids == receiver) & (split_roles == "query")
    ).astype(np.int64)
    if not len(query_indices):
        raise ValueError("canonical manifest_all split contains no query rows")
    support_ids = {str(canonical_ids[index]) for index in support_indices}
    query_ids = {str(canonical_ids[index]) for index in query_indices.tolist()}
    if support_ids & query_ids:
        raise ValueError("canonical support/query physical sample overlap")
    query_records = [
        {
            "array_index": int(index),
            "evaluation_role": str(dataset_roles[index]),
            "transmitter_label": str(tx_ids[index]),
            "registered_class_index": registered_index[
                (str(dataset_roles[index]), str(tx_ids[index]))
            ],
        }
        for index in query_indices.tolist()
    ]
    return (
        np.asarray(support_indices, dtype=np.int64),
        query_indices,
        np.asarray(support_class_indices, dtype=np.int64),
        np.asarray(support_ranks, dtype=np.int64),
        query_records,
    )


def _assert_scenario_physical_independence(
    arrays_by_scenario: dict[str, dict[str, np.ndarray]],
) -> None:
    required_fields = (
        "leo_weak_iq",
        "sample_ids",
        "dataset_role",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "sig_ids",
        "overlay_ids",
        "satellite_seeds",
    )
    sample_ids_by_scenario: dict[str, set[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        missing = [field for field in required_fields if field not in arrays]
        if missing:
            raise ValueError(f"{scenario} cache is missing required fields: {missing}")
        row_count = len(np.asarray(arrays["sample_ids"]))
        if any(len(np.asarray(arrays[field])) != row_count for field in required_fields):
            raise ValueError(f"{scenario} cache row-count structure is inconsistent")
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{scenario} cache contains duplicate physical sample IDs")
        current = set(sample_ids)
        for prior_scenario, prior in sample_ids_by_scenario.items():
            if current & prior:
                raise ValueError(
                    "physical sample reuse across LEO_weak scenarios: "
                    f"{prior_scenario} vs {scenario}"
                )
        sample_ids_by_scenario[scenario] = current


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


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _formal_phase1_class_handles(
    args: argparse.Namespace,
    *,
    old_labels: list[str],
) -> tuple[list[str] | None, dict[str, Any]]:
    binding_value = getattr(args, "phase1_deployment_binding", None)
    label_binding_value = getattr(
        args, "phase1_class_label_binding", None
    )
    if not binding_value and not label_binding_value:
        return None, {
            "formal_phase1_class_binding_used": False,
            "formal_phase1_class_handle_binding_sha256": "",
            "phase1_class_label_binding_source_sha256": "",
        }
    if not binding_value or not label_binding_value:
        raise ValueError(
            "formal Phase1 deployment and class-label bindings "
            "must be provided together"
        )

    binding_path = Path(binding_value).resolve()
    label_binding_path = Path(label_binding_value).resolve()
    binding = _load_json_object(
        binding_path,
        context="formal Phase1 deployment binding",
    )
    if (
        set(binding) != _FORMAL_DEPLOYMENT_BINDING_KEYS
        or binding.get("schema") != FORMAL_DEPLOYMENT_BINDING_SCHEMA
    ):
        raise ValueError("formal Phase1 deployment binding schema drift")

    verified = load_formal_adv3b02_deployment_bundle(
        binding["package_root"],
        detached_seal_path=binding["detached_seal_path"],
        expected_detached_seal_sha256=(
            binding["detached_seal_sha256"]
        ),
        signature_envelope_path=binding["signature_envelope_path"],
        expected_signature_envelope_sha256=(
            binding["signature_envelope_sha256"]
        ),
        expected_checkpoint_lineage_sha256=(
            binding["checkpoint_lineage_sha256"]
        ),
        expected_runtime_sha256=binding["runtime_sha256"],
        expected_component_pre_sign_content_root_sha256=(
            binding["component_pre_sign_content_root_sha256"]
        ),
        expected_class_handle_binding_sha256=(
            binding["class_handle_binding_sha256"]
        ),
        expected_parity_receipt_sha256=(
            binding["parity_receipt_sha256"]
        ),
        expected_generation_lock_sha256=(
            binding["generation_lock_sha256"]
        ),
        expected_method_lock_sha256=binding["method_lock_sha256"],
        expected_generation_config_sha256=(
            binding["generation_config_sha256"]
        ),
        expected_generation_code_sha256=(
            binding["generation_code_sha256"]
        ),
        expected_outer_content_root_sha256=(
            binding["outer_content_root_sha256"]
        ),
    )
    if (
        verified.formal_phase2_context.get("formal_phase2_eligible")
        is not True
        or verified.formal_phase2_context.get(
            "outer_signature_verified"
        )
        is not True
    ):
        raise ValueError("formal Phase1 deployment lacks authority")

    package_root = Path(binding["package_root"]).resolve()
    expected_paths = {
        "candidate_lock": package_root / "locks" / "method_lock.json",
        "checkpoint": (
            package_root
            / "runtime"
            / "adv3b02_runtime.torchscript.pt"
        ),
        "adapter": (
            package_root
            / "component"
            / "int8_domain_class_center_lowrank_residual_radius_v2.npz"
        ),
        "head_artifact": Path(binding["prototype_pt_path"]).resolve(),
        "tta_policy_json": Path(
            binding["generation_config_path"]
        ).resolve(),
    }
    for field, expected in expected_paths.items():
        if Path(getattr(args, field)).resolve() != expected.resolve():
            raise ValueError(
                f"formal Phase1 artifact path drift for {field}"
            )
    expected_hashes = {
        "candidate_lock": binding["method_lock_sha256"],
        "checkpoint": binding["runtime_sha256"],
        "head_artifact": binding["prototype_pt_sha256"],
        "tta_policy_json": binding["generation_config_sha256"],
    }
    for field, expected_sha256 in expected_hashes.items():
        if sha256_file(Path(getattr(args, field))) != expected_sha256:
            raise ValueError(
                f"formal Phase1 artifact digest drift for {field}"
            )

    rows = verified.class_binding.get("class_id_to_handle")
    if not isinstance(rows, list) or len(rows) != len(old_labels):
        raise ValueError("formal Phase1 class-handle count drift")
    formal_handles: list[str] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"class_index", "class_handle"}
            or row.get("class_index") != index
            or not isinstance(row.get("class_handle"), str)
        ):
            raise ValueError("formal Phase1 class-handle order drift")
        formal_handles.append(str(row["class_handle"]))

    label_binding = _load_json_object(
        label_binding_path,
        context="current Phase1 class-label binding",
    )
    if (
        label_binding.get("schema")
        != FORMAL_CLASS_LABEL_BINDING_SCHEMA
        or set(label_binding)
        != _FORMAL_CLASS_LABEL_BINDING_KEYS
        or not isinstance(label_binding.get("entries"), list)
    ):
        raise ValueError("current Phase1 class-label binding schema drift")
    for key in (
        "checkpoint_lineage_sha256",
        "class_handle_binding_sha256",
        "formal_deployment_binding_sha256",
        "source_mapping_sha256",
        "source_checkpoint_sha256",
    ):
        if not _is_sha256(label_binding.get(key)):
            raise ValueError(
                f"current Phase1 class-label binding invalid {key}"
            )
    if (
        label_binding["source_mapping_reused"] is not True
        or label_binding["cross_launch_data_identity_required"] is not False
    ):
        raise ValueError(
            "current Phase1 class-label binding reuse semantics drift"
        )
    if (
        label_binding["checkpoint_lineage_sha256"]
        != binding["checkpoint_lineage_sha256"]
    ):
        raise ValueError(
            "current Phase1 class-label binding checkpoint lineage drift"
        )
    if (
        label_binding["class_handle_binding_sha256"]
        != binding["class_handle_binding_sha256"]
    ):
        raise ValueError(
            "current Phase1 class-label binding semantic handle drift"
        )
    if (
        label_binding["formal_deployment_binding_sha256"]
        != sha256_file(binding_path)
    ):
        raise ValueError(
            "current Phase1 class-label binding deployment digest drift"
        )
    expected_rows = [
        {
            "class_index": index,
            "phase1_tx": label,
            "registered_class_handle": formal_handles[index],
        }
        for index, label in enumerate(old_labels)
    ]
    actual_rows = []
    for row in label_binding["entries"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != _FORMAL_CLASS_LABEL_ENTRY_KEYS
        ):
            raise ValueError(
                "current Phase1 class-label binding row schema drift"
            )
        actual_rows.append(
            {
                "class_index": row.get("class_index"),
                "phase1_tx": row.get("phase1_tx"),
                "registered_class_handle": row.get(
                    "registered_class_handle"
                ),
            }
        )
    if actual_rows != expected_rows:
        raise ValueError(
            "current Phase1 class-label binding does not match old-label order"
        )
    return formal_handles, {
        "formal_phase1_class_binding_used": True,
        "formal_phase1_class_handle_binding_sha256": binding[
            "class_handle_binding_sha256"
        ],
        "phase1_class_label_binding_source_sha256": sha256_file(
            label_binding_path
        ),
    }


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
    query_policy = str(getattr(args, "query_policy", "fixed_per_tx"))
    if query_policy not in {"fixed_per_tx", "manifest_all"}:
        raise ValueError("query_policy must be fixed_per_tx or manifest_all")
    support_pool_max_k = int(args.support_pool_max_k)
    query_per_tx = int(args.query_per_tx)
    if support_pool_max_k < 1:
        raise ValueError("support_pool_max_k must be positive")
    if query_policy == "fixed_per_tx" and query_per_tx < 1:
        raise ValueError("fixed_per_tx query_per_tx must be positive")
    if query_policy == "manifest_all" and query_per_tx != 0:
        raise ValueError("manifest_all query_per_tx must be zero")
    support_seed = int(getattr(args, "support_seed", 0) or 0)
    query_seed = int(getattr(args, "query_seed", 0) or 0)
    new_class_draw_seed = int(
        getattr(args, "new_class_draw_seed", 0) or 0
    )
    if support_seed <= 0 or query_seed <= 0:
        raise ValueError(
            "support_seed and query_seed must be explicit positive integers"
        )
    new_class_pool = _parse_labels(
        getattr(args, "new_class_pool_labels", ""),
        field="new_class_pool_labels",
        required=False,
    )
    if stage == "stage2c":
        if (
            new_class_draw_seed <= 0
            or len(new_class_pool) < declared_new_count
            or set(new_class_pool) & set(old_labels)
        ):
            raise ValueError(
                "new-class pool is missing or overlaps old classes"
            )
    elif new_class_draw_seed != 0 or new_class_pool:
        raise ValueError(
            "Stage2-B must use new_class_draw_seed=0 and no new-class pool"
        )
    formal_old_handles, formal_class_audit = (
        _formal_phase1_class_handles(
            args,
            old_labels=old_labels,
        )
    )

    allowed_roles = {"target_old"}
    if stage == "stage2c" or reference_new_labels:
        allowed_roles.add("target_new")
    expected_scope = (
        "stage2_canonical_registered"
        if query_policy == "manifest_all"
        else str(getattr(args, "expected_cache_scope", "stage2_registered"))
    )
    arrays_by_scenario, cache_manifest, cache_audit = load_verified_leo_weak_cache_set(
        Path(args.target_cache_set),
        expected_scope=expected_scope,
        allowed_roles=allowed_roles,
    )
    _assert_scenario_physical_independence(arrays_by_scenario)
    if stage == "stage2c":
        complete_new_pool = _complete_target_new_pool(
            arrays_by_scenario,
            receiver=str(args.receiver),
        )
        if new_class_pool != complete_new_pool:
            raise ValueError(
                "new-class pool must exactly match the canonical complete "
                "target-new cache pool"
            )
        order = np.random.default_rng(new_class_draw_seed).permutation(
            len(new_class_pool)
        )
        frozen_new_labels = [
            new_class_pool[int(index)]
            for index in order[:declared_new_count]
        ]
        if new_labels != frozen_new_labels:
            raise ValueError(
                "new-class labels do not match the independent draw seed"
            )
    support_labels = [("target_old", label) for label in old_labels]
    if stage == "stage2c":
        support_labels.extend(("target_new", label) for label in new_labels)
        reference_query_labels: list[tuple[str, str]] = []
    else:
        reference_query_labels = [
            ("target_new", label) for label in reference_new_labels
        ]
    use_offline_split_partition = (
        query_policy == "fixed_per_tx"
        and str(getattr(args, "offline_split_partition_policy", ""))
        == "legacy_seeded_nested_exact"
    )
    selections: dict[str, dict[str, Any]] = {}
    reference_support_structure: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    reference_query_structure: tuple[tuple[str, str, int | None], ...] | None = None
    selected_physical_ids_by_scenario: dict[str, set[str]] = {}
    selected_opaque_tokens_by_scenario: dict[str, set[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        if query_policy == "manifest_all":
            support_idx, query_idx, support_y, support_rank, query_records = (
                _select_manifest_all_support_query(
                    arrays,
                    receiver=str(args.receiver),
                    support_labels=support_labels,
                    support_pool_max_k=support_pool_max_k,
                )
            )
            physical_ids = np.asarray(
                arrays["canonical_physical_sample_ids"]
            ).astype(str)
        else:
            support_idx, query_idx, support_y, support_rank, query_records = (
                _select_support_query(
                    arrays,
                    receiver=str(args.receiver),
                    support_seed=support_seed,
                    query_seed=query_seed,
                    support_labels=support_labels,
                    reference_query_labels=reference_query_labels,
                    support_pool_max_k=support_pool_max_k,
                    query_per_tx=query_per_tx,
                    use_offline_split_partition=use_offline_split_partition,
                )
            )
            physical_ids = np.asarray(arrays["sample_ids"]).astype(str)
        support_physical_ids = {
            str(physical_ids[index]) for index in support_idx.tolist()
        }
        query_physical_ids = {
            str(physical_ids[index]) for index in query_idx.tolist()
        }
        if support_physical_ids & query_physical_ids:
            raise ValueError(f"{scenario} support/query physical sample overlap")
        selected_physical_ids = support_physical_ids | query_physical_ids
        for prior_scenario, prior_ids in selected_physical_ids_by_scenario.items():
            if selected_physical_ids & prior_ids:
                identity_kind = (
                    "canonical physical sample"
                    if query_policy == "manifest_all"
                    else "selected physical sample"
                )
                raise ValueError(
                    f"{identity_kind} reuse across LEO_weak scenarios: "
                    f"{prior_scenario} vs {scenario}"
                )
        selected_physical_ids_by_scenario[scenario] = selected_physical_ids

        support_structure = (
            tuple(int(value) for value in support_y.tolist()),
            tuple(int(value) for value in support_rank.tolist()),
        )
        query_structure = tuple(
            (
                str(record["evaluation_role"]),
                str(record["transmitter_label"]),
                (
                    int(record["registered_class_index"])
                    if record["registered_class_index"] is not None
                    else None
                ),
            )
            for record in query_records
        )
        if reference_support_structure is None:
            reference_support_structure = support_structure
            reference_query_structure = query_structure
        elif support_structure != reference_support_structure or (
            query_policy == "fixed_per_tx"
            and query_structure != reference_query_structure
        ):
            raise ValueError(
                "registered class/rank structure drifts across LEO_weak scenarios"
            )

        support_tokens = np.asarray(
            [
                _opaque_token(
                    token_secret,
                    "sid",
                    "cvs-stage2-support-v3",
                    scenario,
                    args.receiver,
                    support_seed,
                    physical_ids[index],
                )
                for index in support_idx.tolist()
            ]
        )
        query_tokens = np.asarray(
            [
                _opaque_token(
                    token_secret,
                    "qid",
                    "cvs-stage2-query-v3",
                    scenario,
                    args.receiver,
                    query_seed,
                    physical_ids[index],
                )
                for index in query_idx.tolist()
            ]
        )
        if len(set(support_tokens.tolist())) != len(support_tokens):
            raise ValueError(f"{scenario} opaque support token collision")
        if len(set(query_tokens.tolist())) != len(query_tokens):
            raise ValueError(f"{scenario} opaque query token collision")
        if set(support_tokens.tolist()) & set(query_tokens.tolist()):
            raise ValueError(f"{scenario} support/query opaque token overlap")
        current_tokens = set(support_tokens.tolist()) | set(query_tokens.tolist())
        for prior_scenario, prior_tokens in selected_opaque_tokens_by_scenario.items():
            if current_tokens & prior_tokens:
                raise ValueError(
                    "opaque sample-token reuse across LEO_weak scenarios: "
                    f"{prior_scenario} vs {scenario}"
                )
        selected_opaque_tokens_by_scenario[scenario] = current_tokens
        selections[scenario] = {
            "support_idx": support_idx,
            "query_idx": query_idx,
            "support_y": support_y,
            "support_rank": support_rank,
            "query_records": query_records,
            "support_tokens": support_tokens,
            "query_tokens": query_tokens,
        }

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
        selection = selections[scenario]
        support_idx = selection["support_idx"]
        query_idx = selection["query_idx"]
        support_y = selection["support_y"]
        support_rank = selection["support_rank"]
        support_tokens = selection["support_tokens"]
        query_tokens = selection["query_tokens"]
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
                            "cvs-stage2-support-overlay-v3",
                            scenario,
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
                            "cvs-stage2-query-overlay-v3",
                            scenario,
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

    class_registry = []
    for index, (role, label) in enumerate(support_labels):
        if role == "target_old" and formal_old_handles is not None:
            handle = formal_old_handles[index]
        else:
            handle = _opaque_token(
                token_secret,
                "cls",
                "cvs-stage2-class-v2",
                label,
            )
        class_registry.append(
            {
                "class_index": index,
                "class_handle": handle,
            }
        )
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
    if query_policy == "manifest_all":
        metadata["query_policy"] = "manifest_all"
    _manifest_path, _seal_path, package_manifest, _seal = (
        write_predictor_package_manifest_and_seal(
            predictor_root,
            manifest_metadata=metadata,
            members=members,
            detached_seal_path=seal_path,
        )
    )
    seal_sha256 = sha256_file(seal_path)

    truth_rows = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        selection = selections[scenario]
        query_records = selection["query_records"]
        query_tokens = selection["query_tokens"]
        physical_ids = np.asarray(
            arrays[
                "canonical_physical_sample_ids"
                if query_policy == "manifest_all"
                else "sample_ids"
            ]
        ).astype(str)
        tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
        rx_ids = np.asarray(arrays["rx_ids"]).astype(str)
        day_ids = np.asarray(arrays["day_ids"]).astype(str)
        sig_ids = np.asarray(arrays["sig_ids"]).astype(str)
        for position, record in enumerate(query_records):
            array_index = int(record["array_index"])
            token = str(query_tokens[position])
            truth_rows.append(
                {
                    "query_token": token,
                    "scenario": scenario,
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
                    "physical_sample_id": str(physical_ids[array_index]),
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
            "same_scenario_support_query_physical_disjointness": "PASS",
            "cross_scenario_selected_physical_disjointness": "PASS",
            "cross_scenario_opaque_token_disjointness": "PASS",
            "registered_class_rank_structure_consistent": "PASS",
            **formal_class_audit,
        },
    )

    forbidden_predictor_values = [
        "target_old",
        "target_new",
        *old_labels,
        *new_labels,
        *reference_new_labels,
    ]
    if query_policy == "manifest_all":
        forbidden_predictor_values.extend(
            physical_id
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
            for physical_id in sorted(
                selected_physical_ids_by_scenario[scenario]
            )
        )
    _reject_predictor_truth_leaks(
        predictor_root,
        forbidden_predictor_values,
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        load_verified_stage2_predictor_bundle(
            predictor_root,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha256,
            scenario=scenario,
        )
    load_verified_scoring_sidecar(scoring_path)
    result = {
        "stage": stage,
        "predictor_package": str(predictor_root),
        "predictor_package_root_sha256": package_manifest["package_root_sha256"],
        "predictor_package_seal": str(seal_path),
        "predictor_package_seal_sha256": seal_sha256,
        "scoring_manifest": str(scoring_path),
        "scoring_manifest_sha256": sha256_file(scoring_path),
        "registered_class_count": len(support_labels),
        "support_pool_count": len(
            selections[FORMAL_LEO_WEAK_SCENARIOS[0]]["support_idx"]
        ),
        "query_count": len(
            selections[FORMAL_LEO_WEAK_SCENARIOS[0]]["query_idx"]
        ),
        "support_seed": support_seed,
        "query_seed": query_seed,
        "new_class_draw_seed": new_class_draw_seed,
    }
    if query_policy == "manifest_all":
        result.update(
            {
                "query_policy": "manifest_all",
                "query_count_by_scenario": {
                    scenario: len(selections[scenario]["query_idx"])
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                },
            }
        )
    return result


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
    parser.add_argument("--support-seed", type=int, default=0)
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--new-class-draw-seed", type=int, default=0)
    parser.add_argument("--old-class-labels", required=True)
    parser.add_argument("--new-class-labels", default="")
    parser.add_argument("--new-class-pool-labels", default="")
    parser.add_argument("--stage2b-reference-new-class-labels", default="")
    parser.add_argument("--new-class-count", type=int, default=0)
    parser.add_argument("--support-pool-max-k", type=int, required=True)
    parser.add_argument("--query-per-tx", type=int, required=True)
    parser.add_argument(
        "--query-policy",
        choices=("fixed_per_tx", "manifest_all"),
        default="fixed_per_tx",
    )
    parser.add_argument("--offline-split-partition-policy", default="")
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--tta-policy-json", type=Path, required=True)
    parser.add_argument("--phase1-deployment-binding", type=Path)
    parser.add_argument("--phase1-class-label-binding", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
