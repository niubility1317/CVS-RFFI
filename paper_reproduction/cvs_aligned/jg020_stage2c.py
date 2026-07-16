"""Strict support-only JG_R8_LR020 Stage2-C package and scoring helpers.

The two runtime profiles are intentionally disjoint:

* ``enrollment_only`` can see registered support but has no query member.
* ``apply_only`` can see query IQ and sealed deployment state but no support.

Package creation belongs to the Phase2-external offline controller.  Both
runtime processes must preflight the detached seal, exact member allowlist and
all hashes before materialising IQ arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np
import torch


FORMAL_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
PACKAGE_SCHEMA = "cvs.phase2.jg020_package.v1"
SEAL_SCHEMA = "cvs.phase2.jg020_package_seal.v1"
HEAD_SCHEMA = "cvs.phase2.jg020_prototype_head.v1"
RECEIPT_SCHEMA = "cvs.phase2.jg020_enrollment_receipt.v1"
LOCK_SCHEMA = "cvs.phase2.jg020_candidate_lock.v1"
ENROLLMENT_PROFILE = "enrollment_only"
APPLY_PROFILE = "apply_only"

PHASE2_CONTRACT = {
    "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
    "clean_sample_access": False,
    "clean_derived_signal_access": False,
    "phase2_clean_dataset_reachable": False,
    "phase2_clean_cache_reachable": False,
    "phase2_clean_control_flow_reachable": False,
    "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
    "phase2_query_decision_policy": "per_sample_all_registered_classes",
    "phase2_query_role_oracle_access": False,
    "phase2_query_true_batch_class_count_access": False,
    "phase2_query_class_quota_access": False,
    "phase2_query_batch_global_assignment": False,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_HANDLE = re.compile(r"^cls_[0-9a-f]{32,64}$")
_FORBIDDEN_MEMBER_WORDS = re.compile(
    r"(?:^|[._/-])(clean|raw|dataset|pkl|truth|label|role|quota)(?:[._/-]|$)",
    re.IGNORECASE,
)


class JG020ProtocolError(ValueError):
    """Raised before a non-conforming package may materialise IQ."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_leaf(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise JG020ProtocolError("package member path must be a nonempty POSIX leaf")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise JG020ProtocolError("package member path must be a direct child")
    if _FORBIDDEN_MEMBER_WORDS.search(value):
        raise JG020ProtocolError(f"forbidden Phase2 member path token: {value}")
    return value


def _required_roles(profile: str) -> set[str]:
    if profile == ENROLLMENT_PROFILE:
        roles = {
            "candidate_lock",
            "checkpoint_full",
            "ground_adapter",
            "direct_class_mapping",
        }
        roles.update(f"support:{scenario}" for scenario in FORMAL_SCENARIOS)
        return roles
    if profile == APPLY_PROFILE:
        roles = {
            "candidate_lock",
            "candidate_runtime",
            "identity_runtime",
            "direct_runtime",
            "prototype_head",
            "enrollment_receipt",
        }
        roles.update(f"query:{scenario}" for scenario in FORMAL_SCENARIOS)
        return roles
    raise JG020ProtocolError(f"unsupported JG020 package profile: {profile}")


@contextmanager
def open_regular_member_same_fd(root: Path, relative_path: str) -> Iterator[BinaryIO]:
    root_resolved = root.resolve(strict=True)
    leaf = _relative_leaf(relative_path)
    target = root_resolved / leaf
    try:
        before = os.lstat(target)
    except FileNotFoundError as exc:
        raise JG020ProtocolError(f"missing package member: {leaf}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise JG020ProtocolError(f"package member is not a regular non-symlink: {leaf}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    handle = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise JG020ProtocolError(f"package member changed during open: {leaf}")
        yield handle
    finally:
        handle.close()


def _hash_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), size


def _npz_members_handle(handle: BinaryIO) -> list[str]:
    handle.seek(0)
    with zipfile.ZipFile(handle, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise JG020ProtocolError("duplicate NPZ member")
        if any(
            name.startswith("/")
            or ".." in PurePosixPath(name).parts
            or len(PurePosixPath(name).parts) != 1
            or not name.endswith(".npy")
            for name in names
        ):
            raise JG020ProtocolError("unsafe NPZ member path")
    handle.seek(0)
    return [name[:-4] for name in names]


def make_member_descriptor(
    path: str | Path,
    *,
    role: str,
    schema: str,
    scenario: str | None = None,
    npz_members: Sequence[str] = (),
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise JG020ProtocolError("package input must be a regular non-symlink file")
    return {
        "relative_path": _relative_leaf(source.name),
        "sha256": sha256_file(source),
        "size_bytes": int(source.stat().st_size),
        "artifact_role": str(role),
        "schema": str(schema),
        "scenario": scenario,
        "npz_members": list(npz_members),
    }


def _package_root_sha256(members: Sequence[Mapping[str, Any]]) -> str:
    stable = [dict(value) for value in sorted(members, key=lambda item: item["relative_path"])]
    return sha256_bytes(canonical_json_bytes(stable))


def _validate_class_registry(value: Any, count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != count:
        raise JG020ProtocolError("registered class registry size drift")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"class_index", "class_handle"}:
            raise JG020ProtocolError("registered class registry schema drift")
        if item["class_index"] != index or _CLASS_HANDLE.fullmatch(str(item["class_handle"])) is None:
            raise JG020ProtocolError("registered class registry order/handle drift")
        result.append(dict(item))
    if len({item["class_handle"] for item in result}) != count:
        raise JG020ProtocolError("duplicate registered class handle")
    return result


def _validate_metadata(document: Mapping[str, Any], *, expected_profile: str | None = None) -> None:
    required = {
        "schema",
        "profile",
        "stage",
        "receiver",
        "seed",
        "k_shot",
        "new_class_count",
        "registered_class_count",
        "registered_classes",
        "candidate_lock_sha256",
        "target_channel_scenarios",
        "phase2_contract",
        "lineage",
        "members",
        "package_root_sha256",
    }
    if set(document) != required or document.get("schema") != PACKAGE_SCHEMA:
        raise JG020ProtocolError("JG020 package manifest exact schema drift")
    profile = str(document["profile"])
    if expected_profile is not None and profile != expected_profile:
        raise JG020ProtocolError("JG020 package profile mismatch")
    if document["stage"] != "stage2c" or document["target_channel_scenarios"] != list(FORMAL_SCENARIOS):
        raise JG020ProtocolError("JG020 Stage2-C/scenario contract drift")
    if document["phase2_contract"] != PHASE2_CONTRACT:
        raise JG020ProtocolError("JG020 Phase2 hard contract drift")
    for key in ("seed", "k_shot", "new_class_count", "registered_class_count"):
        if not isinstance(document[key], int) or isinstance(document[key], bool):
            raise JG020ProtocolError(f"JG020 integer field drift: {key}")
    if document["k_shot"] != 10 or document["new_class_count"] not in {5, 10, 20}:
        raise JG020ProtocolError("development cell must use K10 and 5/10/20 seen-new classes")
    if document["registered_class_count"] != 6 + document["new_class_count"]:
        raise JG020ProtocolError("registered old/new class count drift")
    _validate_class_registry(document["registered_classes"], document["registered_class_count"])
    if _SHA256.fullmatch(str(document["candidate_lock_sha256"])) is None:
        raise JG020ProtocolError("candidate lock digest drift")
    lineage = document["lineage"]
    if not isinstance(lineage, dict) or set(lineage) != {
        "source_package_root_sha256",
        "source_package_seal_sha256",
        "enrollment_package_root_sha256",
    }:
        raise JG020ProtocolError("JG020 lineage schema drift")
    for key, value in lineage.items():
        if value is not None and _SHA256.fullmatch(str(value)) is None:
            raise JG020ProtocolError(f"JG020 lineage digest drift: {key}")
    if profile == ENROLLMENT_PROFILE and lineage["enrollment_package_root_sha256"] is not None:
        raise JG020ProtocolError("enrollment package cannot self-claim a prior enrollment root")
    if profile == APPLY_PROFILE and lineage["enrollment_package_root_sha256"] is None:
        raise JG020ProtocolError("apply package lacks enrollment lineage")


def write_package_manifest_and_seal(
    root: str | Path,
    *,
    profile: str,
    metadata: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    detached_seal: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = Path(root).resolve(strict=True)
    checked = [dict(value) for value in members]
    roles = {str(item.get("artifact_role")) for item in checked}
    if roles != _required_roles(profile) or len(roles) != len(checked):
        raise JG020ProtocolError("JG020 package role set is not exact")
    paths: set[str] = set()
    for item in checked:
        if set(item) != {
            "relative_path", "sha256", "size_bytes", "artifact_role", "schema", "scenario", "npz_members"
        }:
            raise JG020ProtocolError("JG020 member descriptor schema drift")
        leaf = _relative_leaf(item["relative_path"])
        if leaf in paths:
            raise JG020ProtocolError("duplicate JG020 package path")
        paths.add(leaf)
        with open_regular_member_same_fd(package_root, leaf) as handle:
            digest, size = _hash_handle(handle)
            if item["npz_members"] and _npz_members_handle(handle) != list(item["npz_members"]):
                raise JG020ProtocolError("JG020 NPZ member allowlist drift")
        if digest != item["sha256"] or size != item["size_bytes"]:
            raise JG020ProtocolError("JG020 member hash/size drift")
    document = {
        **dict(metadata),
        "schema": PACKAGE_SCHEMA,
        "profile": profile,
        "members": checked,
        "package_root_sha256": _package_root_sha256(checked),
    }
    _validate_metadata(document, expected_profile=profile)
    manifest_path = package_root / "package_manifest.json"
    seal_path = Path(detached_seal).resolve()
    if manifest_path.exists() or seal_path.exists():
        raise FileExistsError("refusing to overwrite JG020 package manifest/seal")
    manifest_bytes = canonical_json_bytes(document) + b"\n"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    seal = {
        "schema": SEAL_SCHEMA,
        "profile": profile,
        "manifest_relative_path": manifest_path.name,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "package_root_sha256": document["package_root_sha256"],
        "artifact_member_allowlist_sha256": document["package_root_sha256"],
    }
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    with seal_path.open("xb") as handle:
        handle.write(canonical_json_bytes(seal) + b"\n")
    return document, seal


def preflight_package(
    root: str | Path,
    *,
    detached_seal: str | Path,
    expected_seal_sha256: str,
    expected_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = Path(root).resolve(strict=True)
    seal_path = Path(detached_seal)
    if seal_path.is_symlink() or not seal_path.is_file():
        raise JG020ProtocolError("JG020 detached seal is not a regular file")
    if _SHA256.fullmatch(str(expected_seal_sha256)) is None or sha256_file(seal_path) != expected_seal_sha256:
        raise JG020ProtocolError("JG020 detached seal digest mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    if not isinstance(seal, dict) or set(seal) != {
        "schema", "profile", "manifest_relative_path", "manifest_sha256", "manifest_size_bytes",
        "package_root_sha256", "artifact_member_allowlist_sha256"
    }:
        raise JG020ProtocolError("JG020 detached seal exact schema drift")
    if seal["schema"] != SEAL_SCHEMA or seal["profile"] != expected_profile:
        raise JG020ProtocolError("JG020 detached seal profile/schema drift")
    with open_regular_member_same_fd(package_root, seal["manifest_relative_path"]) as handle:
        manifest_digest, manifest_size = _hash_handle(handle)
        document = json.loads(handle.read().decode("utf-8-sig"))
    if manifest_digest != seal["manifest_sha256"] or manifest_size != seal["manifest_size_bytes"]:
        raise JG020ProtocolError("JG020 manifest detached binding mismatch")
    _validate_metadata(document, expected_profile=expected_profile)
    members = document["members"]
    if {item["artifact_role"] for item in members} != _required_roles(expected_profile):
        raise JG020ProtocolError("JG020 preflight role set drift")
    if document["package_root_sha256"] != _package_root_sha256(members):
        raise JG020ProtocolError("JG020 package root digest mismatch")
    if seal["package_root_sha256"] != document["package_root_sha256"] or seal[
        "artifact_member_allowlist_sha256"
    ] != document["package_root_sha256"]:
        raise JG020ProtocolError("JG020 manifest/seal root mismatch")
    opened: list[dict[str, Any]] = []
    for item in members:
        with open_regular_member_same_fd(package_root, item["relative_path"]) as handle:
            digest, size = _hash_handle(handle)
            actual_npz = _npz_members_handle(handle) if item["npz_members"] else []
        if digest != item["sha256"] or size != item["size_bytes"] or actual_npz != item["npz_members"]:
            raise JG020ProtocolError(f"JG020 pre-open member audit failed: {item['artifact_role']}")
        opened.append({
            "artifact_role": item["artifact_role"],
            "relative_path": item["relative_path"],
            "sha256": digest,
            "size_bytes": size,
        })
    return document, {
        "schema": "cvs.phase2.jg020_preopen_audit.v1",
        "status": "PASS",
        "profile": expected_profile,
        "package_root_sha256": document["package_root_sha256"],
        "seal_sha256": expected_seal_sha256,
        "opened_members": opened,
        "iq_payload_materialized": False,
        "support_member_reachable": expected_profile == ENROLLMENT_PROFILE,
        "query_member_reachable": expected_profile == APPLY_PROFILE,
        "truth_member_reachable": False,
        "clean_member_reachable": False,
    }


def descriptor_by_role(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["artifact_role"]): dict(item) for item in document["members"]}


def load_npz_member(root: str | Path, descriptor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    package_root = Path(root).resolve(strict=True)
    with open_regular_member_same_fd(package_root, str(descriptor["relative_path"])) as handle:
        digest, size = _hash_handle(handle)
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise JG020ProtocolError("JG020 NPZ descriptor drift at materialisation")
        with np.load(handle, allow_pickle=False) as archive:
            actual = list(archive.files)
            if actual != list(descriptor["npz_members"]):
                raise JG020ProtocolError("JG020 NPZ member drift at materialisation")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return arrays


def validate_locked_candidate(lock: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "candidate_id", "receiver", "seed", "k_shot", "new_class_count",
        "scope", "rank", "alpha", "learning_rate", "weight_decay", "temperature",
        "epochs", "max_optimizer_steps", "grad_clip", "ground_adapter_scope",
        "ground_adapter_rank", "ground_adapter_alpha", "ground_adapter_sha256",
        "checkpoint_sha256", "direct_class_mapping_sha256", "old_class_order_sha256",
        "new_class_order_sha256", "support_view_count", "query_view_count", "adapter_alpha",
        "trust_decision", "k1_trust_gate_enabled", "phase2_contract"
    }
    if not isinstance(lock, dict) or set(lock) != expected or lock.get("schema") != LOCK_SCHEMA:
        raise JG020ProtocolError("JG020 candidate lock exact schema drift")
    fixed = {
        "candidate_id": "JG_R8_LR020",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 10,
        "scope": "joint_gate",
        "rank": 8,
        "alpha": 8.0,
        "learning_rate": 0.02,
        "weight_decay": 1.0e-4,
        "temperature": 18.0,
        "epochs": 5,
        "max_optimizer_steps": 50,
        "grad_clip": 1.0,
        "ground_adapter_scope": "projection_feature",
        "ground_adapter_rank": 16,
        "ground_adapter_alpha": 16.0,
        "support_view_count": 3,
        "query_view_count": 1,
        "adapter_alpha": 1.0,
        "trust_decision": "locked_k10_full_delta",
        "k1_trust_gate_enabled": False,
        "phase2_contract": PHASE2_CONTRACT,
    }
    failed = [key for key, value in fixed.items() if lock.get(key) != value]
    if failed or lock.get("new_class_count") not in {5, 10, 20}:
        raise JG020ProtocolError(f"JG020 locked candidate drift: {failed}")
    for key in (
        "ground_adapter_sha256",
        "checkpoint_sha256",
        "direct_class_mapping_sha256",
        "old_class_order_sha256",
        "new_class_order_sha256",
    ):
        if _SHA256.fullmatch(str(lock.get(key))) is None:
            raise JG020ProtocolError(f"JG020 lock digest drift: {key}")
    return dict(lock)


def ordered_label_sha256(labels: Sequence[str]) -> str:
    values = [str(value) for value in labels]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise JG020ProtocolError("class order must contain unique nonempty labels")
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def validate_direct_class_mapping(
    mapping: Mapping[str, Any], *, lock: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise JG020ProtocolError("direct class mapping must be an object")
    candidates = [
        mapping.get("class_id_to_tx"),
        mapping.get("direct_adv3b02_class_id_to_tx"),
        dict(mapping.get("dataset", {})).get("tx_labels")
        if isinstance(mapping.get("dataset"), dict)
        else None,
    ]
    values = next(
        (
            [str(item) for item in candidate]
            for candidate in candidates
            if isinstance(candidate, list) and candidate
        ),
        None,
    )
    if values is None or len(values) < 6:
        raise JG020ProtocolError("direct ADV3B02 class mapping lacks six old classes")
    observed = ordered_label_sha256(values[:6])
    if observed != lock["old_class_order_sha256"]:
        raise JG020ProtocolError("direct ADV3B02 old class order does not match the candidate lock")
    return {
        "direct_class_count": len(values),
        "old_class_count": 6,
        "old_class_order_sha256": observed,
        "mapping_sha256": lock["direct_class_mapping_sha256"],
        "direct_logit_to_class_handle_order_bound": True,
    }


def _cached_joint_forward(
    model: torch.nn.Module,
    feat_id: torch.Tensor,
    feat_dac: torch.Tensor,
    feat_pa: torch.Tensor,
) -> torch.Tensor:
    head = model.id_backbone.cls_head
    defects: list[torch.Tensor] = []
    if bool(head.use_dac):
        defects.append(feat_dac)
    if bool(head.use_pa):
        defects.append(feat_pa)
    identity = feat_id
    if head.id_gate is not None and defects:
        gate = head.id_gate(torch.cat(defects, dim=1))
        identity = feat_id * (1.0 + float(head.gate_alpha) * gate)
    return head.joint_proj(torch.cat([identity, *defects], dim=1))


def prepare_preincrement_adaptation_support(
    support_iq_by_scenario: Mapping[str, np.ndarray],
    reference_labels: np.ndarray,
    reference_tokens: np.ndarray,
    *,
    old_class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Build the old-only optimizer input before the class registry expands."""

    labels = np.asarray(reference_labels, dtype=np.int64)
    tokens = np.asarray(reference_tokens).astype(str)
    if labels.ndim != 1 or tokens.ndim != 1 or len(labels) != len(tokens):
        raise JG020ProtocolError("pre-increment support label/token layout drift")
    if len(set(tokens.tolist())) != len(tokens):
        raise JG020ProtocolError("pre-increment support token collision")
    mask = (labels >= 0) & (labels < int(old_class_count))
    old_labels = labels[mask]
    old_tokens = tokens[mask]
    expected_count = int(old_class_count) * int(k_shot)
    if len(old_labels) != expected_count:
        raise JG020ProtocolError("pre-increment old support count drift")
    counts = np.bincount(old_labels, minlength=int(old_class_count))
    if counts.tolist() != [int(k_shot)] * int(old_class_count):
        raise JG020ProtocolError("pre-increment old support per-class K drift")
    selected_by_scenario: list[np.ndarray] = []
    for scenario in FORMAL_SCENARIOS:
        rows = np.asarray(support_iq_by_scenario[scenario], dtype=np.float32)
        if len(rows) != len(labels):
            raise JG020ProtocolError("pre-increment scenario support layout drift")
        selected_by_scenario.append(rows[mask])
    adapt_rows = np.concatenate(selected_by_scenario)
    adapt_labels = np.concatenate([old_labels for _ in FORMAL_SCENARIOS])
    adapt_row_ids = np.concatenate([old_tokens for _ in FORMAL_SCENARIOS]).tolist()
    audit = {
        "optimizer_input_stage": "preincrement_registered_old_only",
        "adapt_fit_class_count": int(old_class_count),
        "adapt_physical_support_count": int(len(old_labels)),
        "adapt_support_view_count": len(FORMAL_SCENARIOS),
        "adapt_full_forward_row_count": int(len(adapt_rows)),
        "excluded_registered_new_support_count": int(np.count_nonzero(~mask)),
        "registered_support_labels_used": True,
        "new_support_gradient_used": False,
        "query_role_used_by_optimizer": False,
        "per_sample_old_new_role_branch_used": False,
    }
    return adapt_rows, adapt_labels, adapt_row_ids, old_tokens.tolist(), audit


def _cache_joint_inputs(
    model: torch.nn.Module,
    rows: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from model_dual_cvsincnet import backbone_forward_compat

    cached: dict[str, list[torch.Tensor]] = {
        "feat_cls": [],
        "feat_dac": [],
        "feat_pa": [],
    }
    call_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), int(batch_size)):
            aux = backbone_forward_compat(
                model.id_backbone,
                rows[start : start + int(batch_size)],
                y=None,
                return_aux=True,
                domain_labels=None,
            )
            if not isinstance(aux, dict):
                raise JG020ProtocolError("ADV3B02 id backbone does not expose cached joint inputs")
            for key in cached:
                value = aux.get(key)
                if not torch.is_tensor(value):
                    raise JG020ProtocolError(f"ADV3B02 cached joint input missing: {key}")
                cached[key].append(value.detach())
            call_count += 1
    return (
        torch.cat(cached["feat_cls"]),
        torch.cat(cached["feat_dac"]),
        torch.cat(cached["feat_pa"]),
        call_count,
    )


def train_support_only_bp_jg_cached(
    model: torch.nn.Module,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    physical_support_ids: Sequence[str],
    support_row_physical_ids: Sequence[str],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    support_view_count: int,
    batch_size: int,
    max_optimizer_steps: int,
    grad_clip: float,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache the frozen ADV3B02 path once; optimize only JG's two small layers."""

    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
        _matched_view_support_layout,
        bp_jg_episode_loss,
        build_shot_index_episode_positions,
    )

    if int(epochs) != 5 or int(max_optimizer_steps) != 50:
        raise JG020ProtocolError("cached JG020 training requires exactly 5 epochs/50-step cap")
    rows = torch.from_numpy(np.asarray(support_rows, dtype=np.float32)).to(device)
    labels = torch.from_numpy(np.asarray(support_labels, dtype=np.int64)).to(device)
    physical_count, class_count, k_shot, _ = _matched_view_support_layout(
        labels, view_count=int(support_view_count)
    )
    physical_ids = [str(value) for value in physical_support_ids]
    row_ids = [str(value) for value in support_row_physical_ids]
    if (
        len(physical_ids) != physical_count
        or len(set(physical_ids)) != physical_count
        or len(row_ids) != int(labels.numel())
    ):
        raise JG020ProtocolError("cached JG020 physical support identity layout drift")
    for view_index in range(int(support_view_count)):
        start = view_index * physical_count
        if row_ids[start : start + physical_count] != physical_ids:
            raise JG020ProtocolError("cached JG020 matched-view physical order drift")
    max_episodes = max(1, int(max_optimizer_steps) // int(epochs))
    episodes = build_shot_index_episode_positions(
        labels,
        view_count=int(support_view_count),
        max_episodes_per_epoch=max_episodes,
        pair_physical_shots=True,
    )
    feat_id, feat_dac, feat_pa, full_calls = _cache_joint_inputs(
        model, rows, batch_size=int(batch_size)
    )
    with torch.no_grad():
        base_features = torch.nn.functional.normalize(
            _cached_joint_forward(model, feat_id, feat_dac, feat_pa), dim=1
        ).detach()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != 6_400:
        raise JG020ProtocolError("cached JG020 trainable state is not exactly 6,400 parameters")
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=0.0,
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    optimizer_steps = 0
    small_path_samples = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        totals = {
            "loss": 0.0,
            "xview_prototype_ce": 0.0,
            "boundary_margin_loss": 0.0,
            "feature_anchor_loss": 0.0,
            "prototype_gram_loss": 0.0,
            "prototype_separation_loss": 0.0,
            "view_consistency_loss": 0.0,
            "mean_margin": 0.0,
            "mean_base_margin": 0.0,
            "correct": 0.0,
            "grad": 0.0,
        }
        seen = batches = 0
        for episode_index in rng.permutation(len(episodes)):
            if optimizer_steps >= int(max_optimizer_steps):
                break
            positions = episodes[int(episode_index)]
            optimizer.zero_grad(set_to_none=True)
            z = _cached_joint_forward(
                model,
                feat_id[positions],
                feat_dac[positions],
                feat_pa[positions],
            )
            losses = bp_jg_episode_loss(
                z,
                base_features[positions],
                labels[positions],
                view_count=int(support_view_count),
                temperature=float(temperature),
                leave_one_physical_shot=True,
            )
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
            optimizer.step()
            optimizer_steps += 1
            small_path_samples += int(positions.numel())
            count = int(losses["sample_count"].detach())
            seen += count
            batches += 1
            for key in (
                "loss", "xview_prototype_ce", "boundary_margin_loss",
                "feature_anchor_loss", "prototype_gram_loss",
                "prototype_separation_loss", "view_consistency_loss",
                "mean_margin", "mean_base_margin",
            ):
                totals[key] += float(losses[key].detach()) * count
            totals["correct"] += float(losses["correct"].detach())
            totals["grad"] += float(grad_norm.detach())
        row = {
            "epoch": epoch,
            **{
                key: totals[key] / max(1, seen)
                for key in (
                    "loss", "xview_prototype_ce", "boundary_margin_loss",
                    "feature_anchor_loss", "prototype_gram_loss",
                    "prototype_separation_loss", "view_consistency_loss",
                    "mean_margin", "mean_base_margin",
                )
            },
            "support_train_acc": totals["correct"] / max(1, seen),
            "gradient_norm": totals["grad"] / max(1, batches),
            "optimizer_steps": optimizer_steps,
            "episode_count": batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite cached JG020 trace: {row}")
        trace.append(row)
        print("[JG020-CACHED-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer": "sgd",
        "optimizer_momentum": 0.0,
        "optimizer_steps": optimizer_steps,
        "max_optimizer_steps": int(max_optimizer_steps),
        "optimizer_state_deployment_required": False,
        "training_compute_mode": "frozen_backbone_cached_joint_inputs",
        "cached_joint_inputs": ["feat_cls", "feat_dac", "feat_pa"],
        "full_backbone_forward_call_count": full_calls,
        "full_backbone_forward_sample_equivalents": int(rows.shape[0]),
        "cached_small_path_forward_count": optimizer_steps,
        "cached_small_path_forward_sample_equivalents": small_path_samples,
        "legacy_uncached_full_backbone_forward_sample_equivalents": int(
            rows.shape[0] + small_path_samples
        ),
        "full_backbone_forward_avoided_sample_equivalents": small_path_samples,
        "support_forward_sample_equivalents": int(rows.shape[0] + small_path_samples),
        "support_view_count": int(support_view_count),
        "physical_support_count": physical_count,
        "registered_class_count": class_count,
        "k_shot_inferred": k_shot,
        "query_rows_used_for_training": 0,
        "optimizer_input_stage": "preincrement_registered_old_only",
        "registered_support_labels_used": True,
        "query_role_used_by_optimizer": False,
        "per_sample_old_new_role_branch_used": False,
        "class_quota_used_by_optimizer": False,
        "dense_query_graph_used": False,
    }
    return trace, runtime


def normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-8)


def build_prototypes(features: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    rows = normalize_rows(features)
    targets = np.asarray(labels, dtype=np.int64)
    prototypes = []
    for index in range(int(class_count)):
        selected = rows[targets == index]
        if not len(selected):
            raise JG020ProtocolError(f"registered class lacks support: {index}")
        prototypes.append(selected.mean(axis=0))
    return normalize_rows(np.stack(prototypes)).astype(np.float32)


def score_prototypes(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    return (normalize_rows(features) @ normalize_rows(prototypes).T * float(temperature)).astype(np.float32)


def build_head_state(
    *,
    class_handles: Sequence[str],
    old_class_count: int,
    candidate_features_by_scenario: Mapping[str, np.ndarray],
    identity_features_by_scenario: Mapping[str, np.ndarray],
    support_labels: np.ndarray,
    temperature: float,
) -> dict[str, np.ndarray]:
    handles = np.asarray(class_handles).astype(str)
    if len(handles) <= int(old_class_count) or any(_CLASS_HANDLE.fullmatch(value) is None for value in handles):
        raise JG020ProtocolError("JG020 head class handle registry drift")
    result: dict[str, np.ndarray] = {
        "class_handles": handles,
        "old_class_count": np.asarray(int(old_class_count), dtype=np.int64),
        "temperature": np.asarray(float(temperature), dtype=np.float32),
        "manifest_json": np.asarray(json.dumps({
            "schema": HEAD_SCHEMA,
            "old_class_count": int(old_class_count),
            "registered_class_count": int(len(handles)),
            "scenarios": list(FORMAL_SCENARIOS),
            "metric": "cosine",
            "prototype_fit_scope": "registered_support_only",
            "role_symmetric_rule": True,
        }, sort_keys=True)),
    }
    labels = np.asarray(support_labels, dtype=np.int64)
    for scenario in FORMAL_SCENARIOS:
        candidate = np.asarray(candidate_features_by_scenario[scenario], dtype=np.float32)
        identity = np.asarray(identity_features_by_scenario[scenario], dtype=np.float32)
        if candidate.shape != identity.shape or len(candidate) != len(labels):
            raise JG020ProtocolError("JG020 support feature/head layout drift")
        result[f"candidate_prototypes__{scenario}"] = build_prototypes(candidate, labels, len(handles))
        result[f"identity_prototypes__{scenario}"] = build_prototypes(identity, labels, len(handles))
    return result


def head_npz_members() -> list[str]:
    names = ["class_handles", "old_class_count", "temperature", "manifest_json"]
    for scenario in FORMAL_SCENARIOS:
        names.extend((f"candidate_prototypes__{scenario}", f"identity_prototypes__{scenario}"))
    return names


def apply_head_streams(
    *,
    scenario: str,
    candidate_features: np.ndarray,
    identity_features: np.ndarray,
    direct_logits: np.ndarray,
    head: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if scenario not in FORMAL_SCENARIOS:
        raise JG020ProtocolError("JG020 apply scenario drift")
    handles = np.asarray(head["class_handles"]).astype(str)
    old_count = int(np.asarray(head["old_class_count"]).item())
    temperature = float(np.asarray(head["temperature"]).item())
    candidate_proto = np.asarray(head[f"candidate_prototypes__{scenario}"], dtype=np.float32)
    identity_proto = np.asarray(head[f"identity_prototypes__{scenario}"], dtype=np.float32)
    if direct_logits.ndim != 2 or direct_logits.shape[1] < old_count:
        raise JG020ProtocolError("direct ADV3B02 old-logit layout drift")

    def predicted(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        indices = np.argmax(score_prototypes(features, prototypes, temperature), axis=1)
        return handles[indices]

    return {
        "candidate_after": predicted(candidate_features, candidate_proto),
        "candidate_before": predicted(candidate_features, candidate_proto[:old_count]),
        "identity_after": predicted(identity_features, identity_proto),
        "identity_before": predicted(identity_features, identity_proto[:old_count]),
        "direct": handles[np.argmax(direct_logits[:, :old_count], axis=1)],
    }
