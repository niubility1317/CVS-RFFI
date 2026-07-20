"""Phase1-only episodic LODO selection for the D96/D97 classification head.

The module deliberately has no target-data or model-runtime dependency.  The
input is a sealed, single-observation Phase1 feature archive.  D81 is an
episode-fitted head, so static checkpoint TX logits are never treated as D81:
the caller must provide a support-fitted ``base_scorer`` callback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer


SCHEMA = "cvs.d97.phase1_receiver_lodo_lock.v1"
ALLOWED_K = (1, 5, 10)
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
BLOCK_DIMS = (160, 96, 32)
ALLOWED_SCENARIOS = frozenset(
    {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
)
REQUIRED_ARRAYS = (
    "features",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
)
EXPORTER_SCHEMA = "cvs.phase1.single_leo_feature_archive.v2"
EXPORTER_FORMAL_STATUS = "FORMAL_PHASE1_TEMPORARY_ASSET"
EXPORTER_DEVELOPMENT_STATUS = "DEVELOPMENT_PHASE1_TEMPORARY_ASSET"
EXPORTER_EXACT_MEMBERS = (
    "features",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "checkpoint_reference_logits",
)
KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256 = frozenset(
    {
        "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9",
        "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a",
    }
)
ARCHIVE_FIELD_ALIASES = {
    "features": ("features", "registered_joint288"),
    "labels": ("labels", "tx_ids"),
    "receiver_ids": ("receiver_ids", "rx_ids"),
    "day_ids": ("day_ids",),
    "physical_ids": ("physical_ids", "physical_sample_id"),
    "scenario_names": ("scenario_names", "selected_scenario"),
}
FORBIDDEN_KEY_TOKENS = (
    "target",
    "clean",
    "multiview",
    "multi_view",
    "query_truth",
    "old_new_role",
)

BaseScorer = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
]
EXPECTED_BASE_SCORER_SCHEMA = "cvs.phase1.d81.episode_scorer.v1"
EXPECTED_BASE_SCORER_FORMULA = "D81_before_support_fitted_D62_D42_int8"


@dataclass(frozen=True)
class Episode:
    receiver: str
    k: int
    support: np.ndarray
    calibration: np.ndarray
    evaluation: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_base_scorer_contract(
    scorer: Any,
    *,
    expected_scorer_id: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    if type(scorer) is not D81Phase1EpisodeScorer:
        raise ValueError("full Phase1 lock requires a formal D81 scorer object")
    if not _is_sha256(expected_scorer_id):
        raise ValueError("base_scorer_id must be the scorer's lowercase 64hex identity")
    if not _is_sha256(expected_receipt_sha256):
        raise ValueError(
            "full Phase1 lock requires lowercase 64hex base_scorer_receipt_sha256"
        )
    scorer_id = getattr(scorer, "scorer_id", None)
    receipt = getattr(scorer, "receipt", None)
    if not _is_sha256(scorer_id) or not isinstance(receipt, Mapping):
        raise ValueError(
            "full Phase1 lock rejects plain callbacks; scorer must expose scorer_id and receipt"
        )
    receipt_copy = dict(receipt)
    receipt_sha = canonical_sha256(receipt_copy)
    if (
        scorer_id != expected_scorer_id
        or receipt_sha != expected_receipt_sha256
        or receipt_sha != scorer_id
    ):
        raise ValueError("D81 scorer identity/receipt SHA mismatch")
    required = {
        "schema": EXPECTED_BASE_SCORER_SCHEMA,
        "formula": EXPECTED_BASE_SCORER_FORMULA,
        "query_labels_input": False,
        "receiver_or_role_input": False,
        "mutable_fit_cache": False,
    }
    for key, expected in required.items():
        if receipt_copy.get(key) != expected:
            raise ValueError(f"D81 scorer receipt {key} drift")
    for key in (
        "ground_manifest_sha256",
        "ground_component_npz_sha256",
        "basis_sha256",
        "spectral_weights_sha256",
        "ground_audit_sha256",
        "phase1_checkpoint_sha256",
        "dependency_closure_sha256",
    ):
        if not _is_sha256(receipt_copy.get(key)):
            raise ValueError(f"D81 scorer receipt {key} is not a sealed SHA256")
    if receipt_copy["phase1_checkpoint_sha256"] != BASE_CHECKPOINT_SHA256:
        raise ValueError("D81 scorer checkpoint lineage drift")
    dependencies = receipt_copy.get("dependency_code_sha256")
    if (
        not isinstance(dependencies, Mapping)
        or not dependencies
        or any(not _is_sha256(value) for value in dependencies.values())
        or canonical_sha256(dict(dependencies))
        != receipt_copy["dependency_closure_sha256"]
    ):
        raise ValueError("D81 scorer dependency closure drift")
    scorer_source = Path(sys.modules[D81Phase1EpisodeScorer.__module__].__file__).resolve()
    if dependencies.get("scorer") != _file_sha256(scorer_source):
        raise ValueError("D81 scorer source identity drift")
    for key in (
        "sklearn_runtime_version",
        "python_runtime_version",
        "numpy_runtime_version",
        "scipy_runtime_version",
        "torch_runtime_version",
    ):
        if not isinstance(receipt_copy.get(key), str) or not receipt_copy[key]:
            raise ValueError(f"D81 scorer receipt {key} is missing")
    return {
        "kind": "D81Phase1EpisodeScorer",
        "schema": EXPECTED_BASE_SCORER_SCHEMA,
        "scorer_id": scorer_id,
        "receipt_sha256": receipt_sha,
        "receipt": receipt_copy,
    }


def _recheck_base_scorer_contract(
    scorer: Any, initial: Mapping[str, Any]
) -> None:
    if (
        getattr(scorer, "scorer_id", None) != initial["scorer_id"]
        or not isinstance(getattr(scorer, "receipt", None), Mapping)
        or canonical_sha256(dict(scorer.receipt)) != initial["receipt_sha256"]
    ):
        raise ValueError("D81 scorer identity/receipt mutated during episode scoring")


def _array_archive_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        if array.dtype.kind in "OUS":
            digest.update(canonical_json_bytes(array.astype(str).tolist()))
        else:
            digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exporter_array_sha256(value: np.ndarray) -> str:
    """Verify the Phase1 exporter array contract without importing its CLI."""

    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError("object arrays cannot be verified")
    if array.dtype.kind in {"U", "S"}:
        header = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = json.dumps(
            array.astype(str).tolist(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        header = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        body = canonical.tobytes(order="C")
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header_bytes + b"\0" + body).hexdigest()


def load_feature_archive(
    archive: str | Path | Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Load a non-pickled NPZ or copy an in-memory array mapping."""

    if isinstance(archive, (str, Path)):
        with np.load(Path(archive), allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
    elif isinstance(archive, Mapping):
        arrays = {str(key): np.asarray(value) for key, value in archive.items()}
    else:
        raise TypeError("archive must be a path or mapping")
    return arrays


def _require_vector(name: str, array: np.ndarray, n: int) -> np.ndarray:
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    return array


def validate_feature_archive(
    archive: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the sealed Phase1 single-observation archive."""

    raw = load_feature_archive(archive)
    source_path = Path(archive).resolve() if isinstance(archive, (str, Path)) else None
    resolved: dict[str, np.ndarray] = {}
    source_fields: dict[str, str] = {}
    for canonical, aliases in ARCHIVE_FIELD_ALIASES.items():
        present = [key for key in aliases if key in raw]
        if len(present) > 1:
            raise ValueError(
                f"archive provides ambiguous aliases for {canonical}: {present}"
            )
        if present:
            resolved[canonical] = raw[present[0]]
            source_fields[canonical] = present[0]
    for key in raw:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_KEY_TOKENS):
            raise ValueError(f"forbidden archive field: {key}")
    missing = [key for key in REQUIRED_ARRAYS if key not in resolved]
    if missing:
        raise ValueError(f"missing required archive fields: {missing}")

    features = resolved["features"]
    if features.dtype != np.float32:
        raise ValueError(f"features must be float32, got {features.dtype}")
    if features.ndim != 2 or features.shape[1] != 288 or features.shape[0] == 0:
        raise ValueError(f"features must have shape (N, 288), got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    n = int(features.shape[0])

    labels = _require_vector("labels", resolved["labels"], n)
    receivers = _require_vector("receiver_ids", resolved["receiver_ids"], n).astype(str)
    days = _require_vector("day_ids", resolved["day_ids"], n).astype(str)
    physical = _require_vector("physical_ids", resolved["physical_ids"], n).astype(str)
    scenarios = _require_vector("scenario_names", resolved["scenario_names"], n).astype(str)
    for name, values in (
        ("receiver_ids", receivers),
        ("day_ids", days),
        ("physical_ids", physical),
        ("scenario_names", scenarios),
    ):
        if np.any(np.char.str_len(values) == 0):
            raise ValueError(f"{name} contains an empty value")
    if np.unique(physical).size != n:
        raise ValueError("every physical_id must occur exactly once")
    if "dataset_role" in raw:
        roles = _require_vector("dataset_role", raw["dataset_role"], n).astype(str)
        lowered_roles = [value.lower() for value in roles.tolist()]
        if any(
            not value.startswith("source")
            or any(token in value for token in ("target", "query", "clean"))
            for value in lowered_roles
        ):
            raise ValueError("dataset_role must contain Phase1 source-only rows")
    observed_scenarios = set(scenarios.tolist())
    if not observed_scenarios.issubset(ALLOWED_SCENARIOS):
        raise ValueError(
            "scenario_names must be sealed LEO_weak observations; "
            f"observed={sorted(observed_scenarios)}"
        )

    class_ids = np.asarray(raw.get("class_ids", np.unique(labels)))
    if class_ids.ndim != 1 or class_ids.size < 2:
        raise ValueError("class_ids must define at least two classes")
    if np.unique(class_ids).size != class_ids.size:
        raise ValueError("class_ids must be unique")
    if set(np.asarray(labels).tolist()) != set(class_ids.tolist()):
        raise ValueError("labels and class_ids define different class sets")
    receiver_values = np.unique(receivers)
    if receiver_values.size < 2:
        raise ValueError("receiver LODO requires at least two receivers")

    # Static archive logits may be retained for reference diagnostics, but they
    # are not an episode-fitted D81 head and are excluded from formal selection.
    reference_logits = None
    reference_key = None
    for key in ("reference_logits", "base_logits", "checkpoint_reference_logits"):
        if key in raw:
            if reference_logits is not None:
                raise ValueError("provide only one static reference-logit field")
            value = np.asarray(raw[key])
            if value.shape != (n, class_ids.size) or not np.isfinite(value).all():
                raise ValueError(
                    f"{key} must have shape ({n}, {class_ids.size}) and be finite"
                )
            reference_logits = value.astype(np.float64, copy=False)
            reference_key = key

    normalized_arrays = {
        "features": features,
        "labels": labels,
        "receiver_ids": receivers,
        "day_ids": days,
        "physical_ids": physical,
        "scenario_names": scenarios,
        "class_ids": class_ids,
    }
    if reference_logits is not None:
        normalized_arrays["reference_logits"] = reference_logits
    return {
        "arrays": normalized_arrays,
        "archive_sha256": _array_archive_sha256(normalized_arrays),
        "reference_logits_source_field": reference_key,
        "archive_source_fields": source_fields,
        "raw_arrays": raw,
        "archive_file_sha256": _file_sha256(source_path) if source_path else None,
        "static_reference_logits_present": reference_logits is not None,
        "sample_count": n,
        "class_ids": class_ids,
        "receivers": receiver_values.astype(str),
    }


def _validate_feature_archive_manifest(
    manifest_path: str | Path,
    manifest_sha256: str,
    *,
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only the v2 exporter manifest, with formal/development separation."""

    path = Path(manifest_path).resolve()
    expected_manifest_sha = str(manifest_sha256).lower()
    if (
        not _is_sha256(expected_manifest_sha)
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != expected_manifest_sha
    ):
        raise ValueError("feature_archive_manifest path/SHA256 drift")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("feature_archive_manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("feature_archive_manifest must be an object")
    required_top = {
        "schema",
        "status",
        "artifact_stage",
        "artifact",
        "exact_member_allowlist",
        "feature_dims",
        "inputs",
        "selection",
        "feature_semantics",
        "requested_device",
        "resolved_device",
        "row_count",
        "physical_id_unique_count",
        "one_output_row_per_physical_id",
        "array_sha256",
        "cache_loader_audit_sha256",
        "access_audit",
        "lifecycle",
        "formal_archive",
        "development_archive",
    }
    if set(manifest) != required_top or manifest.get("schema") != EXPORTER_SCHEMA:
        raise ValueError("feature_archive_manifest is not exact exporter v2")
    status = manifest.get("status")
    if status not in {EXPORTER_FORMAL_STATUS, EXPORTER_DEVELOPMENT_STATUS}:
        raise ValueError("diagnostic feature archive cannot create a frozen lock")
    is_formal = status == EXPORTER_FORMAL_STATUS
    is_development = status == EXPORTER_DEVELOPMENT_STATUS
    if (
        manifest.get("formal_archive") is not is_formal
        or manifest.get("development_archive") is not is_development
        or manifest.get("artifact_stage")
        != "phase1_offline_before_target_access"
        or manifest.get("exact_member_allowlist") != list(EXPORTER_EXACT_MEMBERS)
        or manifest.get("feature_dims")
        != {"z160": 160, "fft96": 96, "rf32": 32, "features": 288}
    ):
        raise ValueError("feature_archive_manifest status/member semantics drift")
    raw_arrays = validated["raw_arrays"]
    if tuple(raw_arrays) != EXPORTER_EXACT_MEMBERS:
        raise ValueError("feature archive must use the exact exporter v2 members")
    if validated["archive_source_fields"] != {
        name: name for name in REQUIRED_ARRAYS
    }:
        raise ValueError("feature archive aliases are forbidden for frozen locks")
    artifact = manifest.get("artifact")
    file_sha = validated.get("archive_file_sha256")
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != {"path", "sha256"}
        or artifact.get("path") != "phase1_singleobs_feature_archive.npz"
        or not _is_sha256(artifact.get("sha256"))
        or artifact.get("sha256") != file_sha
    ):
        raise ValueError("feature_archive_manifest artifact SHA256 mismatch")
    expected_arrays = manifest.get("array_sha256")
    if (
        not isinstance(expected_arrays, Mapping)
        or set(expected_arrays) != set(EXPORTER_EXACT_MEMBERS)
    ):
        raise ValueError("feature_archive_manifest array registry mismatch")
    for name, array in raw_arrays.items():
        if expected_arrays.get(name) != _exporter_array_sha256(array):
            raise ValueError(f"feature_archive_manifest array SHA256 mismatch: {name}")
    selection = manifest.get("selection")
    access = manifest.get("access_audit")
    lifecycle = manifest.get("lifecycle")
    semantics = manifest.get("feature_semantics")
    if (
        manifest.get("one_output_row_per_physical_id") is not True
        or manifest.get("row_count") != validated["sample_count"]
        or manifest.get("physical_id_unique_count") != validated["sample_count"]
        or not isinstance(selection, Mapping)
        or selection.get("selected_observations_per_physical_id") != 1
        or selection.get("unselected_observations_forwarded") != 0
        or not isinstance(access, Mapping)
        or access
        != {
            "clean_calls": 0,
            "target_calls": 0,
            "channel_calls": 0,
            "clean_iq_access": False,
            "target_access": False,
            "query_access": False,
            "raw_iq_persisted": False,
            "received_iq_persisted": False,
            "unselected_iq_persisted": False,
        }
        or lifecycle
        != {
            "phase1_temporary_selection_asset": True,
            "phase2_bundle_ingest_allowed": False,
            "phase2_runtime_access_allowed": False,
            "retention": "archive_or_delete_after_D97_lock_receipt",
        }
        or not isinstance(semantics, Mapping)
        or semantics.get("features")
        != (
            "float32_concat(runtime_z160,internally_normalized_fft96,"
            "internally_normalized_rf32)_without_cross_block_weight_or_joint_normalization"
        )
        or semantics.get("deployment_normalization")
        != "shared_D97_normalize_three_blocks"
        or semantics.get("checkpoint_reference_logits")
        != "sealed_ADV3B02_checkpoint_reference_only_not_D81"
    ):
        raise ValueError("exporter feature_archive_manifest protocol/lifecycle drift")
    inputs = manifest.get("inputs")
    required_inputs = {
        "cache_set_sha256",
        "cache_npz_sha256_by_scenario",
        "runtime_authority_mode",
        "runtime_authority_binding_sha256",
        "runtime_checkpoint_parity_receipt_sha256",
        "runtime_schema",
        "runtime_sha256",
        "phase1_checkpoint_sha256",
        "bundle_id",
        "formal_outer_content_root_sha256",
        "detached_seal_sha256",
        "signature_envelope_sha256",
        "selection_salt_receipt_sha256",
        "selection_salt_receipt_schema",
        "exporter_code_sha256",
        "dependency_code_sha256",
        "dependency_closure_sha256",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != required_inputs:
        raise ValueError("feature_archive_manifest runtime lineage registry drift")
    sha_fields = (
        "cache_set_sha256",
        "runtime_authority_binding_sha256",
        "runtime_checkpoint_parity_receipt_sha256",
        "runtime_sha256",
        "phase1_checkpoint_sha256",
        "bundle_id",
        "selection_salt_receipt_sha256",
        "exporter_code_sha256",
        "dependency_closure_sha256",
    )
    if (
        any(not _is_sha256(inputs.get(name)) for name in sha_fields)
        or inputs.get("phase1_checkpoint_sha256") != BASE_CHECKPOINT_SHA256
        or inputs.get("runtime_schema")
        != "adv3b02.torchscript_identity_runtime.v1"
        or inputs.get("selection_salt_receipt_schema")
        != "cvs.phase1.singleobs_selection_salt_receipt.v1"
    ):
        raise ValueError("feature_archive_manifest ADV3B02 runtime lineage drift")
    scenario_hashes = inputs.get("cache_npz_sha256_by_scenario")
    dependencies = inputs.get("dependency_code_sha256")
    if (
        not isinstance(scenario_hashes, Mapping)
        or tuple(scenario_hashes) != (
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        )
        or any(not _is_sha256(value) for value in scenario_hashes.values())
        or not isinstance(dependencies, Mapping)
        or set(dependencies)
        != {"leo_weak_cache", "feature_descriptors", "formal_bundle_verifier"}
        or any(not _is_sha256(value) for value in dependencies.values())
        or canonical_sha256(dict(dependencies))
        != inputs["dependency_closure_sha256"]
    ):
        raise ValueError("feature_archive_manifest dependency/cache closure drift")
    if is_formal:
        if (
            inputs.get("runtime_authority_mode")
            != "formal_adv3b02_outer_bundle"
            or any(
                not _is_sha256(inputs.get(name))
                for name in (
                    "formal_outer_content_root_sha256",
                    "detached_seal_sha256",
                    "signature_envelope_sha256",
                )
            )
            or inputs.get("bundle_id")
            != inputs.get("formal_outer_content_root_sha256")
        ):
            raise ValueError("formal feature archive lacks outer authority closure")
    else:
        if (
            inputs.get("runtime_authority_mode")
            != "development_known_adv3b02_runtime_sha"
            or inputs.get("runtime_sha256")
            not in KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256
            or inputs.get("formal_outer_content_root_sha256") is not None
            or inputs.get("detached_seal_sha256") is not None
            or inputs.get("signature_envelope_sha256") is not None
        ):
            raise ValueError("development feature archive authority boundary drift")
    return {
        "schema": EXPORTER_SCHEMA,
        "manifest_sha256": expected_manifest_sha,
        "canonical_sha256": canonical_sha256(dict(manifest)),
        "binding_kind": "exporter_v2_file_array_and_runtime_lineage_sha256",
        "status": status,
        "full_phase1_lock": is_formal,
        "development_lock_frozen": is_development,
        "phase2_bundle_ingest_allowed": False,
        "phase2_runtime_access_allowed": False,
    }


def _stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    descriptor = canonical_json_bytes([int(seed), *[str(v) for v in parts]])
    local_seed = int.from_bytes(hashlib.sha256(descriptor).digest()[:8], "little")
    return np.random.default_rng(local_seed)


def build_receiver_lodo_episodes(
    validated: Mapping[str, Any], *, seed: int = 0
) -> dict[str, dict[int, Episode]]:
    """Build nested K episodes with disjoint support/calibration/evaluation."""

    arrays = validated["arrays"]
    labels = arrays["labels"]
    receivers = arrays["receiver_ids"]
    physical = arrays["physical_ids"]
    class_ids = arrays["class_ids"]
    max_k = max(ALLOWED_K)
    episodes: dict[str, dict[int, Episode]] = {}
    for receiver in validated["receivers"].tolist():
        support_by_k: dict[int, list[int]] = {k: [] for k in ALLOWED_K}
        calibration: list[int] = []
        evaluation: list[int] = []
        for class_id in class_ids.tolist():
            indices = np.flatnonzero((receivers == receiver) & (labels == class_id))
            if indices.size < max_k + 2:
                raise ValueError(
                    f"receiver={receiver!r}, class={class_id!r} needs at least "
                    f"{max_k + 2} physical samples, got {indices.size}"
                )
            indices = indices[np.argsort(physical[indices], kind="stable")]
            indices = _stable_rng(seed, receiver, class_id).permutation(indices)
            reserved = indices[:max_k]
            remaining = indices[max_k:]
            n_cal = max(1, int(remaining.size // 2))
            if remaining.size - n_cal < 1:
                n_cal = int(remaining.size - 1)
            calibration.extend(remaining[:n_cal].tolist())
            evaluation.extend(remaining[n_cal:].tolist())
            for k in ALLOWED_K:
                support_by_k[k].extend(reserved[:k].tolist())

        cal = np.asarray(sorted(calibration), dtype=np.int64)
        eva = np.asarray(sorted(evaluation), dtype=np.int64)
        episodes[receiver] = {}
        for k in ALLOWED_K:
            sup = np.asarray(sorted(support_by_k[k]), dtype=np.int64)
            id_sets = [set(physical[idx].tolist()) for idx in (sup, cal, eva)]
            if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError("episode physical-ID split overlap")
            episodes[receiver][k] = Episode(receiver, k, sup, cal, eva)
    return episodes


def normalize_three_blocks(values: np.ndarray) -> np.ndarray:
    """Match the deployed D97 z160/FFT96/RF32 normalization exactly."""

    rows = np.asarray(values)
    if rows.ndim != 2 or rows.shape[1] != 288 or not np.isfinite(rows).all():
        raise ValueError("three-block features must be finite [N, 288]")
    source = rows.astype(np.float64, copy=False)
    normalized = np.zeros_like(source)
    for block in BLOCK_SLICES:
        part = source[:, block]
        norms = np.linalg.norm(part, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("every z160/FFT96/RF32 block must have positive norm")
        normalized[:, block] = part / norms
    total = np.linalg.norm(normalized, axis=1, keepdims=True)
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise ValueError("three-block normalization became degenerate")
    return np.ascontiguousarray(normalized / total, dtype=np.float32)


def _logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(top, axis=axis)
        + np.log(np.mean(np.exp(values - top), axis=axis))
    )


def qknn_logits(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    class_ids: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Class-symmetric qKNN log-mean-exp logits."""

    support = normalize_three_blocks(support_features).astype(np.float64)
    query = normalize_three_blocks(query_features).astype(np.float64)
    similarity = query @ support.T
    columns = []
    for class_id in class_ids.tolist():
        mask = support_labels == class_id
        if not np.any(mask):
            raise ValueError(f"support is missing class {class_id!r}")
        columns.append(
            _logmeanexp(float(beta) * similarity[:, mask], axis=1) / float(beta)
        )
    return np.stack(columns, axis=1)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _label_positions(labels: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(class_ids.tolist())}
    try:
        return np.asarray([lookup[value] for value in labels.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label absent from class registry: {exc.args[0]!r}") from exc


def _candidate_grid(grid: Mapping[str, Iterable[float]]) -> list[dict[str, float]]:
    names = ("beta", "temp_base", "temp_qk", "eta_max", "k1_eta_prior")
    if set(grid) != set(names):
        raise ValueError(f"candidate_grid must contain exactly {list(names)}")
    values = [tuple(float(v) for v in grid[name]) for name in names]
    if any(not group for group in values):
        raise ValueError("candidate_grid dimensions cannot be empty")
    candidates = [dict(zip(names, combination)) for combination in itertools.product(*values)]
    for item in candidates:
        if item["beta"] <= 0 or item["temp_base"] <= 0 or item["temp_qk"] <= 0:
            raise ValueError("beta and temperatures must be positive")
        if not 0 <= item["eta_max"] <= 1 or not 0 <= item["k1_eta_prior"] <= 1:
            raise ValueError("eta values must be in [0, 1]")
        if item["k1_eta_prior"] > item["eta_max"]:
            raise ValueError("k1_eta_prior must be <= eta_max")
    return candidates


def _probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    *,
    base_probabilities: np.ndarray,
    qk_probabilities: np.ndarray,
) -> dict[str, Any]:
    truth = _label_positions(labels, class_ids)
    pred = np.argmax(probabilities, axis=1)
    base_pred = np.argmax(base_probabilities, axis=1)
    qk_pred = np.argmax(qk_probabilities, axis=1)
    correct = pred == truth
    base_correct = base_pred == truth
    qk_correct = qk_pred == truth
    onehot = np.eye(class_ids.size, dtype=np.float64)[truth]
    per_class = {}
    for index, class_id in enumerate(class_ids.tolist()):
        mask = truth == index
        per_class[str(class_id)] = float(np.mean(correct[mask]))
    base_wrong = ~base_correct
    qk_wrong = ~qk_correct
    return {
        "sample_count": int(labels.size),
        "accuracy": float(np.mean(correct)),
        "nll": float(-np.mean(np.log(np.maximum(probabilities[np.arange(truth.size), truth], 1e-12)))),
        "brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "floor": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "base_accuracy": float(np.mean(base_correct)),
        "qk_accuracy": float(np.mean(qk_correct)),
        "disagreement_rate": float(np.mean(base_pred != qk_pred)),
        "qk_rescue_given_base_wrong": float(np.mean(qk_correct[base_wrong])) if np.any(base_wrong) else 0.0,
        "qk_rescue_denominator": int(np.sum(base_wrong)),
        "base_rescue_given_qk_wrong": float(np.mean(base_correct[qk_wrong])) if np.any(qk_wrong) else 0.0,
        "base_rescue_denominator": int(np.sum(qk_wrong)),
        "oracle_union_accuracy": float(np.mean(base_correct | qk_correct)),
    }


def _score_episode(
    arrays: Mapping[str, np.ndarray],
    episode: Episode,
    query_indices: np.ndarray,
    candidate: Mapping[str, float],
    base_logits: np.ndarray,
    resolved_eta: float,
    *,
    quantized_qk_support: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = arrays["features"]
    labels = arrays["labels"]
    class_ids = arrays["class_ids"]
    support_features = features[episode.support]
    qk_support_features = support_features
    quantization = None
    if quantized_qk_support:
        qk_support_features, quantization = _quantize_support(support_features)
    support_labels = labels[episode.support]
    query_features = features[query_indices]
    base_logits = np.asarray(base_logits, dtype=np.float64)
    expected = (query_indices.size, class_ids.size)
    if base_logits.shape != expected or not np.isfinite(base_logits).all():
        raise ValueError(
            f"sealed episode base logits must have finite shape {expected}, got {base_logits.shape}"
        )
    qk_logits_value = qknn_logits(
        qk_support_features,
        support_labels,
        query_features,
        class_ids,
        beta=candidate["beta"],
    )
    base_probs = _softmax(base_logits, candidate["temp_base"])
    qk_probs = _softmax(qk_logits_value, candidate["temp_qk"])
    eta = float(resolved_eta)
    if not 0.0 <= eta <= candidate["eta_max"]:
        raise ValueError("resolved support-only eta exceeds candidate eta_max")
    fused = (1.0 - eta) * base_probs + eta * qk_probs
    metrics = _probability_metrics(
        fused,
        labels[query_indices],
        class_ids,
        base_probabilities=base_probs,
        qk_probabilities=qk_probs,
    )
    metrics["eta"] = float(eta)
    metrics["qknn_support_format"] = (
        "three_block_int8_fp16_scale" if quantized_qk_support else "fp32_teacher"
    )
    return metrics, {
        "base_logits": base_logits,
        "qk_logits": qk_logits_value,
        "base_probabilities": base_probs,
        "qk_probabilities": qk_probs,
        "fused_probabilities": fused,
        "quantization": quantization,
    }


def _base_logits_sha256(logits: np.ndarray) -> str:
    value = np.ascontiguousarray(logits, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _build_base_logits_cache(
    arrays: Mapping[str, np.ndarray],
    episodes: Mapping[str, Mapping[int, Episode]],
    base_scorer: BaseScorer,
) -> tuple[
    dict[tuple[str, int, str], np.ndarray],
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[str, Any],
]:
    """Fit/score D81 once per episode-query face, then seal deterministic logits."""

    features = arrays["features"]
    labels = arrays["labels"]
    class_ids = arrays["class_ids"]
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    support_cv_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
    audit_rows = []

    def score_and_seal(
        *,
        receiver: str,
        k: int,
        split_name: str,
        support_indices: np.ndarray,
        query_indices: np.ndarray,
    ) -> np.ndarray:
        expected = (query_indices.size, class_ids.size)

        def score_once() -> np.ndarray:
            value = np.asarray(
                base_scorer(
                    features[support_indices],
                    labels[support_indices],
                    features[query_indices],
                    class_ids,
                ),
                dtype=np.float64,
            )
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(
                    "episode-fitted base_scorer must return finite shape "
                    f"{expected}, got {value.shape}"
                )
            return np.ascontiguousarray(value)

        first = score_once()
        second = score_once()
        first_sha = _base_logits_sha256(first)
        second_sha = _base_logits_sha256(second)
        if first_sha != second_sha:
            raise ValueError(
                "episode-fitted base_scorer is nondeterministic for "
                f"receiver={receiver}, K={k}, split={split_name}"
            )
        first.setflags(write=False)
        audit_rows.append(
            {
                "receiver": receiver,
                "k": k,
                "split": split_name,
                "shape": list(first.shape),
                "logits_sha256": first_sha,
                "repeat_sha256": second_sha,
                "deterministic_repeat_match": True,
            }
        )
        return first

    for receiver in sorted(episodes):
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            for split_name, query_indices in (
                ("calibration", episode.calibration),
                ("evaluation", episode.evaluation),
            ):
                key = (receiver, k, split_name)
                cache[key] = score_and_seal(
                    receiver=receiver,
                    k=k,
                    split_name=split_name,
                    support_indices=episode.support,
                    query_indices=query_indices,
                )
            folds: list[dict[str, Any]] = []
            if k > 1:
                by_class = []
                for class_id in class_ids.tolist():
                    indices = episode.support[labels[episode.support] == class_id]
                    indices = indices[
                        np.argsort(arrays["physical_ids"][indices], kind="stable")
                    ]
                    if indices.size != k:
                        raise AssertionError("support CV requires balanced K-shot")
                    by_class.append(indices)
                for fold_index in range(k):
                    heldout = np.asarray(
                        [indices[fold_index] for indices in by_class], dtype=np.int64
                    )
                    remaining = np.asarray(
                        sorted(set(episode.support.tolist()) - set(heldout.tolist())),
                        dtype=np.int64,
                    )
                    fold_logits = score_and_seal(
                        receiver=receiver,
                        k=k,
                        split_name=f"support_cv_{fold_index}",
                        support_indices=remaining,
                        query_indices=heldout,
                    )
                    folds.append(
                        {
                            "fold_index": fold_index,
                            "remaining": remaining,
                            "heldout": heldout,
                            "base_logits": fold_logits,
                            "remaining_physical_ids_sha256": canonical_sha256(
                                sorted(arrays["physical_ids"][remaining].tolist())
                            ),
                            "heldout_physical_ids_sha256": canonical_sha256(
                                sorted(arrays["physical_ids"][heldout].tolist())
                            ),
                        }
                    )
            support_cv_cache[(receiver, k)] = folds
    cache_root = canonical_sha256(
        [
            {
                "receiver": row["receiver"],
                "k": row["k"],
                "split": row["split"],
                "logits_sha256": row["logits_sha256"],
            }
            for row in audit_rows
        ]
    )
    return cache, support_cv_cache, {
        "schema": "cvs.d97.phase1_episode_base_logits_cache.v1",
        "candidate_grid_independent": True,
        "scorer_calls": int(2 * len(audit_rows)),
        "unique_episode_query_faces": int(len(audit_rows)),
        "determinism_repeats_per_face": 1,
        "cache_root_sha256": cache_root,
        "rows": audit_rows,
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows]))


def _resolve_support_only_eta(
    arrays: Mapping[str, np.ndarray],
    episode: Episode,
    candidate: Mapping[str, float],
    support_cv_folds: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Resolve one row-global eta without calibration/evaluation/query access."""

    physical = arrays["physical_ids"]
    if episode.k == 1:
        eta = float(candidate["k1_eta_prior"])
        receipt = {
            "schema": "cvs.d97.phase1_support_cv_eta.v1",
            "mode": "phase1_locked_k1_prior",
            "receiver": episode.receiver,
            "k": 1,
            "eta": eta,
            "eta_max": float(candidate["eta_max"]),
            "support_physical_ids_sha256": canonical_sha256(
                sorted(physical[episode.support].tolist())
            ),
            "fold_count": 0,
            "calibration_or_evaluation_labels_used": False,
        }
        return eta, {**receipt, "support_cv_receipt_sha256": canonical_sha256(receipt)}
    if len(support_cv_folds) != episode.k:
        raise ValueError("K5/K10 eta requires one class-balanced support-CV fold per shot")

    base_logits_parts = []
    qk_logits_parts = []
    truth_parts = []
    fold_receipts = []
    for fold in support_cv_folds:
        remaining = np.asarray(fold["remaining"], dtype=np.int64)
        heldout = np.asarray(fold["heldout"], dtype=np.int64)
        quantized_support, quantization = _quantize_support(
            arrays["features"][remaining]
        )
        qk_logits_value = qknn_logits(
            quantized_support,
            arrays["labels"][remaining],
            arrays["features"][heldout],
            arrays["class_ids"],
            beta=candidate["beta"],
        )
        base_logits = np.asarray(fold["base_logits"], dtype=np.float64)
        truth = _label_positions(arrays["labels"][heldout], arrays["class_ids"])
        base_logits_parts.append(base_logits)
        qk_logits_parts.append(qk_logits_value)
        truth_parts.append(truth)
        fold_receipts.append(
            {
                "fold_index": int(fold["fold_index"]),
                "remaining_physical_ids_sha256": fold[
                    "remaining_physical_ids_sha256"
                ],
                "heldout_physical_ids_sha256": fold[
                    "heldout_physical_ids_sha256"
                ],
                "base_logits_sha256": _base_logits_sha256(base_logits),
                "qk_logits_sha256": _base_logits_sha256(qk_logits_value),
                "quantization": quantization,
            }
        )
    base_logits = np.concatenate(base_logits_parts)
    qk_logits_value = np.concatenate(qk_logits_parts)
    truth = np.concatenate(truth_parts)
    base_prob = _softmax(base_logits, candidate["temp_base"])
    qk_prob = _softmax(qk_logits_value, candidate["temp_qk"])
    row = np.arange(truth.size)
    base_nll = float(-np.mean(np.log(np.maximum(base_prob[row, truth], 1e-12))))
    qk_nll = float(-np.mean(np.log(np.maximum(qk_prob[row, truth], 1e-12))))

    # k1_eta_prior is the Phase1 reliability prior.  It is converted into
    # pseudo-NLL for both heads, then smoothed with class-balanced support-CV.
    prior_qk = float(np.clip(candidate["k1_eta_prior"], 1e-6, 1.0 - 1e-6))
    prior_base = 1.0 - prior_qk
    prior_strength = float(arrays["class_ids"].size)
    observation_count = float(truth.size)
    smooth_base_nll = (
        observation_count * base_nll - prior_strength * np.log(prior_base)
    ) / (observation_count + prior_strength)
    smooth_qk_nll = (
        observation_count * qk_nll - prior_strength * np.log(prior_qk)
    ) / (observation_count + prior_strength)
    base_reliability = float(np.exp(-smooth_base_nll))
    qk_reliability = float(np.exp(-smooth_qk_nll))
    raw_eta = qk_reliability / max(base_reliability + qk_reliability, 1e-12)
    eta = float(min(candidate["eta_max"], raw_eta))
    receipt = {
        "schema": "cvs.d97.phase1_support_cv_eta.v1",
        "mode": "class_balanced_leave_one_physical_per_class_out",
        "receiver": episode.receiver,
        "k": episode.k,
        "fold_count": len(fold_receipts),
        "heldout_rows": int(truth.size),
        "base_nll": base_nll,
        "qk_nll": qk_nll,
        "prior_qk_weight": float(candidate["k1_eta_prior"]),
        "prior_strength": prior_strength,
        "smoothed_base_nll": float(smooth_base_nll),
        "smoothed_qk_nll": float(smooth_qk_nll),
        "base_reliability": base_reliability,
        "qk_reliability": qk_reliability,
        "raw_eta": float(raw_eta),
        "eta_max": float(candidate["eta_max"]),
        "eta": eta,
        "folds": fold_receipts,
        "calibration_or_evaluation_labels_used": False,
        "class_balanced": True,
    }
    return eta, {**receipt, "support_cv_receipt_sha256": canonical_sha256(receipt)}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty metric set")
    return {
        "episode_count": int(len(rows)),
        "mean_accuracy": _mean_metric(rows, "accuracy"),
        "mean_nll": _mean_metric(rows, "nll"),
        "mean_brier": _mean_metric(rows, "brier"),
        "mean_eta": _mean_metric(rows, "eta"),
        "worst_floor": float(min(float(row["floor"]) for row in rows)),
        "mean_disagreement_rate": _mean_metric(rows, "disagreement_rate"),
        "mean_qk_rescue_given_base_wrong": _mean_metric(rows, "qk_rescue_given_base_wrong"),
        "mean_base_rescue_given_qk_wrong": _mean_metric(rows, "base_rescue_given_qk_wrong"),
        "mean_oracle_union_accuracy": _mean_metric(rows, "oracle_union_accuracy"),
    }


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    params = item["parameters"]
    metrics = item["metrics"]
    return (
        float(metrics["mean_nll"]),
        float(metrics["mean_brier"]),
        -float(metrics["worst_floor"]),
        -float(metrics["mean_accuracy"]),
        *(float(params[key]) for key in ("beta", "temp_base", "temp_qk", "eta_max", "k1_eta_prior")),
    )


def _quantize_support(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = normalize_three_blocks(features)
    codes = np.zeros(source.shape, dtype=np.int8)
    scales = np.zeros((len(source), len(BLOCK_SLICES)), dtype=np.float16)
    restored = np.zeros(source.shape, dtype=np.float64)
    min_scale = float(np.finfo(np.float16).tiny)
    for row_index in range(len(source)):
        for block_index, block in enumerate(BLOCK_SLICES):
            part = source[row_index, block]
            scale64 = max(float(np.max(np.abs(part))) / 127.0, min_scale)
            scale16 = np.float16(scale64)
            if not np.isfinite(scale16) or scale16 <= 0:
                raise ValueError("support block quantization scale overflow")
            code = np.clip(
                np.rint(part / float(scale16)), -127, 127
            ).astype(np.int8)
            codes[row_index, block] = code
            scales[row_index, block_index] = scale16
            restored[row_index, block] = code.astype(np.float64) * float(scale16)
    restored = normalize_three_blocks(restored)
    cos = np.sum(source * restored, axis=1)
    return restored, {
        "mean_reconstruction_cosine": float(np.mean(cos)),
        "min_reconstruction_cosine": float(np.min(cos)),
        "scale_dtype": "float16",
        "code_dtype": "int8",
        "block_dims": list(BLOCK_DIMS),
        "scale_count": int(scales.size),
    }


def _true_margin(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    true_score = probabilities[np.arange(truth.size), truth]
    masked = probabilities.copy()
    masked[np.arange(truth.size), truth] = -np.inf
    return true_score - np.max(masked, axis=1)


def _int8_audit(
    arrays: Mapping[str, np.ndarray],
    episodes: Mapping[str, Mapping[int, Episode]],
    candidate: Mapping[str, float],
    base_logits_cache: Mapping[tuple[str, int, str], np.ndarray],
    eta_by_episode: Mapping[tuple[str, int], tuple[float, Mapping[str, Any]]],
) -> dict[str, Any]:
    class_ids = arrays["class_ids"]
    truth_all: list[np.ndarray] = []
    fp_qk_logits: list[np.ndarray] = []
    q_qk_logits: list[np.ndarray] = []
    fp_fused: list[np.ndarray] = []
    q_fused: list[np.ndarray] = []
    reconstruction: list[dict[str, Any]] = []
    rows = []
    for receiver in sorted(episodes):
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            resolved_eta, eta_receipt = eta_by_episode[(receiver, k)]
            fp_metrics, fp_outputs = _score_episode(
                arrays,
                episode,
                episode.evaluation,
                candidate,
                base_logits_cache[(receiver, k, "evaluation")],
                resolved_eta,
                quantized_qk_support=False,
            )
            q_metrics, q_outputs = _score_episode(
                arrays,
                episode,
                episode.evaluation,
                candidate,
                base_logits_cache[(receiver, k, "evaluation")],
                resolved_eta,
            )
            rec = q_outputs["quantization"]
            if not isinstance(rec, dict):
                raise AssertionError("formal qKNN path did not quantize support")
            truth = _label_positions(arrays["labels"][episode.evaluation], class_ids)
            qk_flip = np.argmax(fp_outputs["qk_logits"], axis=1) != np.argmax(q_outputs["qk_logits"], axis=1)
            final_flip = np.argmax(fp_outputs["fused_probabilities"], axis=1) != np.argmax(q_outputs["fused_probabilities"], axis=1)
            fp_qk_margin = _true_margin(fp_outputs["qk_probabilities"], truth)
            q_qk_margin = _true_margin(q_outputs["qk_probabilities"], truth)
            fp_margin = _true_margin(fp_outputs["fused_probabilities"], truth)
            q_margin = _true_margin(q_outputs["fused_probabilities"], truth)
            rows.append(
                {
                    "receiver": receiver,
                    "k": k,
                    "sample_count": int(truth.size),
                    "eta": resolved_eta,
                    "support_cv_receipt_sha256": eta_receipt[
                        "support_cv_receipt_sha256"
                    ],
                    "qk_top1_flip_rate": float(np.mean(qk_flip)),
                    "final_top1_flip_rate": float(np.mean(final_flip)),
                    "qk_true_margin_sign_flip_rate": float(np.mean(np.signbit(fp_qk_margin) != np.signbit(q_qk_margin))),
                    "final_true_margin_sign_flip_rate": float(np.mean(np.signbit(fp_margin) != np.signbit(q_margin))),
                    "fp32_teacher_accuracy": fp_metrics["accuracy"],
                    "formal_int8_accuracy": q_metrics["accuracy"],
                    **rec,
                }
            )
            truth_all.append(truth)
            fp_qk_logits.append(fp_outputs["qk_logits"])
            q_qk_logits.append(q_outputs["qk_logits"])
            fp_fused.append(fp_outputs["fused_probabilities"])
            q_fused.append(q_outputs["fused_probabilities"])
            reconstruction.append(rec)
    truth = np.concatenate(truth_all)
    fp_logits = np.concatenate(fp_qk_logits)
    q_logits = np.concatenate(q_qk_logits)
    fp_prob = np.concatenate(fp_fused)
    q_prob = np.concatenate(q_fused)
    fp_qk_prob = _softmax(fp_logits, candidate["temp_qk"])
    q_qk_prob = _softmax(q_logits, candidate["temp_qk"])
    return {
        "format": "support_per_row_three_block_symmetric_int8_fp16_scale_query_fp32",
        "block_dims": list(BLOCK_DIMS),
        "formal_selection_uses_quantized_support": True,
        "fp32_support_role": "teacher_margin_audit_only",
        "qmin": -127,
        "qmax": 127,
        "rows": rows,
        "aggregate": {
            "sample_count": int(truth.size),
            "mean_reconstruction_cosine": float(np.mean([v["mean_reconstruction_cosine"] for v in reconstruction])),
            "min_reconstruction_cosine": float(min(v["min_reconstruction_cosine"] for v in reconstruction)),
            "mean_abs_qk_logit_error": float(np.mean(np.abs(fp_logits - q_logits))),
            "max_abs_qk_logit_error": float(np.max(np.abs(fp_logits - q_logits))),
            "qk_top1_flip_rate": float(np.mean(np.argmax(fp_logits, axis=1) != np.argmax(q_logits, axis=1))),
            "final_top1_flip_rate": float(np.mean(np.argmax(fp_prob, axis=1) != np.argmax(q_prob, axis=1))),
            "qk_true_margin_sign_flip_rate": float(np.mean(np.signbit(_true_margin(fp_qk_prob, truth)) != np.signbit(_true_margin(q_qk_prob, truth)))),
            "final_true_margin_sign_flip_rate": float(np.mean(np.signbit(_true_margin(fp_prob, truth)) != np.signbit(_true_margin(q_prob, truth)))),
        },
    }


def run_phase1_lodo_selection(
    archive: str | Path,
    candidate_grid: Mapping[str, Iterable[float]],
    *,
    base_scorer: BaseScorer | None,
    base_scorer_id: str,
    feature_archive_manifest_path: str | Path,
    feature_archive_manifest_sha256: str,
    base_scorer_receipt_sha256: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    """Select D97 parameters using Phase1 receiver-LODO only.

    Candidate selection for each outer receiver uses calibration episodes from
    all *other* receivers.  The final deployable lock uses every calibration
    split, while all evaluation splits remain untouched by final selection.
    """

    if not isinstance(archive, (str, Path)):
        raise ValueError("frozen Phase1 LODO requires an exporter v2 archive path")
    scorer_contract = _validate_base_scorer_contract(
        base_scorer,
        expected_scorer_id=base_scorer_id,
        expected_receipt_sha256=base_scorer_receipt_sha256,
    )
    validated = validate_feature_archive(archive)
    manifest_audit = _validate_feature_archive_manifest(
        feature_archive_manifest_path,
        feature_archive_manifest_sha256,
        validated=validated,
    )
    arrays = validated["arrays"]
    candidates = _candidate_grid(candidate_grid)
    episodes = build_receiver_lodo_episodes(validated, seed=seed)
    (
        base_logits_cache,
        support_cv_base_cache,
        base_logits_cache_audit,
    ) = _build_base_logits_cache(arrays, episodes, base_scorer)
    _recheck_base_scorer_contract(base_scorer, scorer_contract)

    eta_cache: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}
    for receiver in sorted(episodes):
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            for candidate_index, candidate in enumerate(candidates):
                eta_cache[(receiver, k, candidate_index)] = (
                    _resolve_support_only_eta(
                        arrays,
                        episode,
                        candidate,
                        support_cv_base_cache[(receiver, k)],
                    )
                )

    calibration_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    for receiver in sorted(episodes):
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            for candidate_index, candidate in enumerate(candidates):
                resolved_eta, eta_receipt = eta_cache[
                    (receiver, k, candidate_index)
                ]
                metrics, _ = _score_episode(
                    arrays,
                    episode,
                    episode.calibration,
                    candidate,
                    base_logits_cache[(receiver, k, "calibration")],
                    resolved_eta,
                )
                metrics["eta_source"] = eta_receipt["mode"]
                metrics["support_cv_receipt_sha256"] = eta_receipt[
                    "support_cv_receipt_sha256"
                ]
                calibration_cache[(receiver, k, candidate_index)] = metrics

    outer_rows = []
    outer_selections = []
    for heldout_receiver in sorted(episodes):
        rankings = []
        for candidate_index, candidate in enumerate(candidates):
            rows = [
                calibration_cache[(receiver, k, candidate_index)]
                for receiver in sorted(episodes)
                if receiver != heldout_receiver
                for k in ALLOWED_K
            ]
            rankings.append(
                {
                    "candidate_index": candidate_index,
                    "parameters": candidate,
                    "metrics": _summary(rows),
                }
            )
        selected = min(rankings, key=_candidate_sort_key)
        outer_selections.append(
            {
                "heldout_receiver": heldout_receiver,
                "candidate_index": selected["candidate_index"],
                "parameters": selected["parameters"],
                "selection_metrics_other_receivers": selected["metrics"],
            }
        )
        for k in ALLOWED_K:
            episode = episodes[heldout_receiver][k]
            resolved_eta, eta_receipt = eta_cache[
                (heldout_receiver, k, selected["candidate_index"])
            ]
            metrics, _ = _score_episode(
                arrays,
                episode,
                episode.evaluation,
                selected["parameters"],
                base_logits_cache[(heldout_receiver, k, "evaluation")],
                resolved_eta,
            )
            metrics["eta_source"] = eta_receipt["mode"]
            metrics["support_cv_receipt_sha256"] = eta_receipt[
                "support_cv_receipt_sha256"
            ]
            outer_rows.append(
                {"receiver": heldout_receiver, "k": k, **metrics}
            )

    final_rankings = []
    for candidate_index, candidate in enumerate(candidates):
        rows = [
            calibration_cache[(receiver, k, candidate_index)]
            for receiver in sorted(episodes)
            for k in ALLOWED_K
        ]
        final_rankings.append(
            {
                "candidate_index": candidate_index,
                "parameters": candidate,
                "metrics": _summary(rows),
            }
        )
    final_rankings.sort(key=_candidate_sort_key)
    selected_candidate_index = final_rankings[0]["candidate_index"]
    selected_parameters = final_rankings[0]["parameters"]

    final_eval_rows = []
    for receiver in sorted(episodes):
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            resolved_eta, eta_receipt = eta_cache[
                (receiver, k, selected_candidate_index)
            ]
            metrics, _ = _score_episode(
                arrays,
                episode,
                episode.evaluation,
                selected_parameters,
                base_logits_cache[(receiver, k, "evaluation")],
                resolved_eta,
            )
            metrics["eta_source"] = eta_receipt["mode"]
            metrics["support_cv_receipt_sha256"] = eta_receipt[
                "support_cv_receipt_sha256"
            ]
            final_eval_rows.append({"receiver": receiver, "k": k, **metrics})

    final_eta_by_episode = {
        (receiver, k): eta_cache[(receiver, k, selected_candidate_index)]
        for receiver in sorted(episodes)
        for k in ALLOWED_K
    }
    selected_support_cv_receipts = [
        receipt
        for (_receiver, _k), (_eta_value, receipt) in final_eta_by_episode.items()
    ]

    split_receipt = []
    physical = arrays["physical_ids"]
    for receiver in sorted(episodes):
        nested_support = {}
        for k in ALLOWED_K:
            episode = episodes[receiver][k]
            nested_support[str(k)] = canonical_sha256(
                sorted(physical[episode.support].tolist())
            )
        fixed = episodes[receiver][max(ALLOWED_K)]
        split_receipt.append(
            {
                "receiver": receiver,
                "support_count_by_k": {
                    str(k): int(episodes[receiver][k].support.size) for k in ALLOWED_K
                },
                "calibration_count": int(fixed.calibration.size),
                "evaluation_count": int(fixed.evaluation.size),
                "support_physical_ids_sha256_by_k": nested_support,
                "calibration_physical_ids_sha256": canonical_sha256(
                    sorted(physical[fixed.calibration].tolist())
                ),
                "evaluation_physical_ids_sha256": canonical_sha256(
                    sorted(physical[fixed.evaluation].tolist())
                ),
                "physical_id_splits_disjoint": True,
            }
        )

    unsigned = {
        "schema": SCHEMA,
        "full_phase1_lock": manifest_audit["full_phase1_lock"],
        "development_lock_frozen": manifest_audit["development_lock_frozen"],
        "target_narrow_diagnostic_preregistration_allowed": manifest_audit[
            "development_lock_frozen"
        ],
        "formal_target_claim_allowed": manifest_audit["full_phase1_lock"],
        "selection_scope": "phase1_only_receiver_outer_lodo",
        "archive_sha256": validated["archive_sha256"],
        "feature_archive_manifest": manifest_audit,
        "archive_field_mapping": validated["archive_source_fields"],
        "sample_count": validated["sample_count"],
        "feature_dim": 288,
        "class_ids": arrays["class_ids"].tolist(),
        "receiver_ids": sorted(episodes),
        "scenario_names": sorted(set(arrays["scenario_names"].tolist())),
        "allowed_k": list(ALLOWED_K),
        "seed": int(seed),
        "protocol_audit": {
            "single_row_per_physical_id": True,
            "target_fields_present": False,
            "clean_fields_present": False,
            "multiview_fields_present": False,
            "query_labels_used": False,
            "class_role_branching_used": False,
            "uniform_all_registered_class_formula": True,
            "eta_uses_calibration_or_evaluation_labels": False,
        },
        "base_head": {
            "kind": "D81Phase1EpisodeScorer",
            "formal_contract": scorer_contract,
            "scorer_id": scorer_contract["scorer_id"],
            "scorer_receipt_sha256": scorer_contract["receipt_sha256"],
            "static_archive_logits_are_d81": False,
            "static_reference_logits_present": validated["static_reference_logits_present"],
            "static_reference_logits_source_field": validated["reference_logits_source_field"],
            "static_reference_logits_used_for_selection": False,
            "episode_logits_cache": base_logits_cache_audit,
        },
        "fusion_formula": {
            "qknn": "logmeanexp(beta*cos(three_block_norm(query),three_block_norm(int8_support_c)))/beta",
            "formal_qknn_support": "per_row_per_z160_fft96_rf32_symmetric_int8_fp16_scale",
            "fp32_support": "teacher_margin_audit_only",
            "probability": "(1-eta)*softmax(base/temp_base)+eta*softmax(qknn/temp_qk)",
            "eta_k1": "k1_eta_prior",
            "eta_k5_k10": (
                "min(eta_max,exp(-smoothed_qk_nll)/"
                "(exp(-smoothed_base_nll)+exp(-smoothed_qk_nll)))"
            ),
            "support_cv": (
                "class-balanced leave-one-physical-per-class-out;"
                "Phase1 k1 prior pseudo-NLL smoothing; support labels only"
            ),
        },
        "candidate_count": len(candidates),
        "selected_parameters": selected_parameters,
        "selected_support_cv_receipts": selected_support_cv_receipts,
        "candidate_ranking": final_rankings,
        "outer_lodo_selections": outer_selections,
        "outer_lodo_rows": outer_rows,
        "outer_lodo_summary": _summary(outer_rows),
        "final_lock_evaluation_rows": final_eval_rows,
        "final_lock_evaluation_summary": _summary(final_eval_rows),
        "split_receipt": split_receipt,
        "int8_margin_audit": _int8_audit(
            arrays,
            episodes,
            selected_parameters,
            base_logits_cache,
            final_eta_by_episode,
        ),
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    claimed = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return isinstance(claimed, str) and claimed == canonical_sha256(unsigned)


__all__ = [
    "ALLOWED_K",
    "SCHEMA",
    "build_receiver_lodo_episodes",
    "canonical_sha256",
    "load_feature_archive",
    "normalize_three_blocks",
    "qknn_logits",
    "run_phase1_lodo_selection",
    "validate_feature_archive",
    "verify_receipt",
]
