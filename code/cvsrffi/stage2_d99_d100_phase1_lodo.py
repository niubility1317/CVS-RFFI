"""Phase1-only receiver/domain LODO locks for D99 and D100.

The producer evaluates the intended D81 + D99 Student-t local head and the
D100 simplex-ridge complement on sealed Phase1 single-observation features.
Every source receiver is an outer held domain and every source class is, in
turn, withheld from the ground bundle as a pseudo-new registration class.

This module intentionally cannot mint authority.  A ground-release manifest
is useful for byte/provenance validation, but it is formal only when its SHA is
independently provisioned in ``TRUSTED_GROUND_RELEASE_MANIFEST_SHA256``.  The
repository currently carries no such root, so development runs end in
``LOCAL_BLOCKED_INPUTS`` and cannot be written as the canonical lock artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d100_ra_cgspr_lgf as d100
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
from cvsrffi.stage2_d96_d97_phase1_lodo import (
    _recheck_base_scorer_contract,
    _validate_base_scorer_contract,
    _validate_feature_archive_manifest,
    canonical_sha256,
    validate_feature_archive,
)


SCHEMA = "cvs.phase1.d99_d100_lodo_lock.v1"
GROUND_RELEASE_SCHEMA = "cvs.phase1.d99.ground_release_manifest.v1"
GROUND_RELEASE_STATUS = "FORMAL_PHASE1_GROUND_AGGREGATE"
GROUND_RELEASE_DEVELOPMENT_SCHEMA = (
    "cvs.phase1.d99.ground_release_manifest.development.v1"
)
GROUND_RELEASE_DEVELOPMENT_STATUS = (
    "PREREGISTERED_DEVELOPMENT_GROUND_AGGREGATE_NONFORMAL"
)
GROUND_RELEASE_LIFECYCLE = "PHASE1_OFFLINE_BEFORE_TARGET_ACCESS"
ALLOWED_K = (1, 5, 10, 20)
STRICT_NLL_IMPROVEMENT_TOLERANCE = 1e-6
STATUS_FORMAL = "FORMAL_PHASE1_LODO_LOCK"
STATUS_BLOCKED = "LOCAL_BLOCKED_INPUTS"
STATUS_DIAGNOSTIC = "NONFORMAL_LODO_DIAGNOSTIC"
FEATURE_DIM = 288
TRUSTED_GROUND_RELEASE_MANIFEST_SHA256: str | None = None
_GROUND_AUTHORITY_TOKEN = object()


class D99D100LODOLockError(ValueError):
    """Raised when a Phase1 split, authority, grid, or score drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Mapping):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(value) for value in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise TypeError(f"not JSON serializable: {type(item).__name__}")

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_code_sha256() -> dict[str, str]:
    """Return the exact source closure bound into every lock receipt."""

    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_d99_d100_phase1_lodo.py"
    result = {
        "stage2_d99_d100_phase1_lodo": _sha256_file(Path(__file__).resolve()),
        "stage2_d99_ra_cgtmk_d81": _sha256_file(Path(d99.__file__).resolve()),
        "stage2_d100_ra_cgspr_lgf": _sha256_file(Path(d100.__file__).resolve()),
        "stage2_d81_phase1_episode_scorer": _sha256_file(
            Path(__import__(D81Phase1EpisodeScorer.__module__, fromlist=["x"]).__file__).resolve()
        ),
    }
    if runner_path.is_file():
        result["run_d99_d100_phase1_lodo"] = _sha256_file(runner_path)
    return result


def _require_sha(value: Any, name: str) -> str:
    text = str(value)
    if (
        text != text.lower()
        or len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise D99D100LODOLockError(f"{name} must be lowercase SHA256")
    return text


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


@dataclass(frozen=True)
class GroundReleaseAuthority:
    """A byte-checked ground release; trust is never caller granted."""

    manifest_sha256: str
    bundle_sha256: str
    aggregation_receipt_sha256: str
    phase1_checkpoint_sha256: str
    receiver_domain_map: Mapping[str, str]
    formal_phase1_eligible: bool
    authority_status: str
    manifest_bytes: bytes
    loader_token: object

    def __post_init__(self) -> None:
        for value, name in (
            (self.manifest_sha256, "ground manifest"),
            (self.bundle_sha256, "ground bundle"),
            (self.aggregation_receipt_sha256, "ground aggregation receipt"),
            (self.phase1_checkpoint_sha256, "Phase1 checkpoint"),
        ):
            _require_sha(value, name)
        trusted = TRUSTED_GROUND_RELEASE_MANIFEST_SHA256
        raw = bytes(self.manifest_bytes)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise D99D100LODOLockError("ground authority manifest decode drift") from exc
        release_pair = (payload.get("schema"), payload.get("status"))
        formal_release = release_pair == (GROUND_RELEASE_SCHEMA, GROUND_RELEASE_STATUS)
        development_release = release_pair == (
            GROUND_RELEASE_DEVELOPMENT_SCHEMA,
            GROUND_RELEASE_DEVELOPMENT_STATUS,
        )
        if not (formal_release or development_release):
            raise D99D100LODOLockError("ground authority release schema/status drift")
        expected_formal = bool(
            formal_release and trusted is not None and self.manifest_sha256 == trusted
        )
        expected_status = (
            "PROVISIONED"
            if expected_formal
            else (
                "BLOCKED_DEVELOPMENT_GROUND_RELEASE"
                if development_release
                else "BLOCKED_UNPROVISIONED_GROUND_ROOT"
            )
        )
        if (
            self.loader_token is not _GROUND_AUTHORITY_TOKEN
            or _sha256_bytes(raw) != self.manifest_sha256
            or self.formal_phase1_eligible is not expected_formal
            or self.authority_status != expected_status
        ):
            raise D99D100LODOLockError("ground authority self-grant or token drift")
        mapping = {str(key): str(value) for key, value in self.receiver_domain_map.items()}
        if (
            len(mapping) < 3
            or len(set(mapping.values())) != len(mapping)
            or any(not key or not value for key, value in mapping.items())
        ):
            raise D99D100LODOLockError("receiver/domain map must be one-to-one for >=3 domains")
        object.__setattr__(self, "receiver_domain_map", MappingProxyType(mapping))
        object.__setattr__(self, "manifest_bytes", raw)


def ground_release_manifest_payload(
    bundle: d99.Phase1GroundAggregateBundle,
    receiver_domain_map: Mapping[str, str],
    *,
    producer_code_sha256: str,
    release_schema: str = GROUND_RELEASE_SCHEMA,
    release_status: str = GROUND_RELEASE_STATUS,
) -> dict[str, Any]:
    """Return the exact payload expected from an independent ground producer."""

    if type(bundle) is not d99.Phase1GroundAggregateBundle:
        raise D99D100LODOLockError("exact typed D99 ground bundle required")
    release_pair = (str(release_schema), str(release_status))
    if release_pair not in {
        (GROUND_RELEASE_SCHEMA, GROUND_RELEASE_STATUS),
        (GROUND_RELEASE_DEVELOPMENT_SCHEMA, GROUND_RELEASE_DEVELOPMENT_STATUS),
    }:
        raise D99D100LODOLockError("ground release schema/status pair drift")
    mapping = {str(key): str(value) for key, value in receiver_domain_map.items()}
    return {
        "schema": release_pair[0],
        "status": release_pair[1],
        "lifecycle": GROUND_RELEASE_LIFECYCLE,
        "phase1_checkpoint_sha256": bundle.aggregation_receipt.phase1_checkpoint_sha256,
        "producer_code_sha256": _require_sha(producer_code_sha256, "ground producer code"),
        "bundle_sha256": bundle.bundle_sha256,
        "aggregation_receipt_sha256": bundle.aggregation_receipt.receipt_sha256,
        "aggregation_receipt": asdict(bundle.aggregation_receipt),
        "domain_ids": list(bundle.domain_ids),
        "ground_old_registry": list(bundle.ground_old_registry),
        "receiver_domain_map": mapping,
        "arrays": {
            "codes_qint8": _array_receipt(bundle.codes_qint8),
            "scales_fp16": _array_receipt(bundle.scales_fp16),
            "domain_class_mask": _array_receipt(bundle.domain_class_mask),
            "physical_sample_count_floor_uint16": _array_receipt(
                bundle.physical_sample_count_floor_uint16
            ),
        },
        "minimum_physical_sample_count": 2,
        "member_ids_present": False,
        "sample_level_features_present": False,
        "raw_or_clean_iq_present": False,
        "target_rows_used": 0,
        "query_rows_used": 0,
    }


def load_ground_release_authority(
    manifest_bytes: bytes,
    externally_expected_manifest_sha256: str,
    bundle: d99.Phase1GroundAggregateBundle,
) -> GroundReleaseAuthority:
    """Validate exact manifest bytes without treating caller SHA as authority."""

    raw = bytes(manifest_bytes)
    expected = _require_sha(externally_expected_manifest_sha256, "expected ground manifest")
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise D99D100LODOLockError("ground manifest bytes/SHA mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99D100LODOLockError("ground manifest is not canonical UTF-8 JSON") from exc
    if raw != _canonical_bytes(payload):
        raise D99D100LODOLockError("ground manifest is not canonical")
    if not isinstance(payload, Mapping):
        raise D99D100LODOLockError("ground manifest must be an object")
    producer_sha = payload.get("producer_code_sha256")
    expected_payload = ground_release_manifest_payload(
        bundle,
        payload.get("receiver_domain_map", {}),
        producer_code_sha256=producer_sha,
        release_schema=payload.get("schema", ""),
        release_status=payload.get("status", ""),
    )
    if payload != expected_payload:
        raise D99D100LODOLockError("ground release manifest/bundle closure drift")
    if payload["phase1_checkpoint_sha256"] != BASE_CHECKPOINT_SHA256:
        raise D99D100LODOLockError("ground release checkpoint lineage drift")
    trusted = TRUSTED_GROUND_RELEASE_MANIFEST_SHA256
    formal_release = (
        payload["schema"] == GROUND_RELEASE_SCHEMA
        and payload["status"] == GROUND_RELEASE_STATUS
    )
    formal = bool(formal_release and trusted is not None and actual == trusted)
    authority_status = (
        "PROVISIONED"
        if formal
        else (
            "BLOCKED_DEVELOPMENT_GROUND_RELEASE"
            if not formal_release
            else "BLOCKED_UNPROVISIONED_GROUND_ROOT"
        )
    )
    return GroundReleaseAuthority(
        manifest_sha256=actual,
        bundle_sha256=bundle.bundle_sha256,
        aggregation_receipt_sha256=bundle.aggregation_receipt.receipt_sha256,
        phase1_checkpoint_sha256=payload["phase1_checkpoint_sha256"],
        receiver_domain_map=payload["receiver_domain_map"],
        formal_phase1_eligible=formal,
        authority_status=authority_status,
        manifest_bytes=raw,
        loader_token=_GROUND_AUTHORITY_TOKEN,
    )


@dataclass(frozen=True)
class Episode:
    receiver: str
    k_shot: int
    support: np.ndarray
    calibration: np.ndarray
    evaluation: np.ndarray


def _stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    descriptor = _canonical_bytes([int(seed), *[str(value) for value in parts]])
    local_seed = int.from_bytes(hashlib.sha256(descriptor).digest()[:8], "little")
    return np.random.default_rng(local_seed)


def build_receiver_lodo_episodes(
    validated: Mapping[str, Any], *, seed: int
) -> dict[str, dict[int, Episode]]:
    """Build nested real K={1,5,10,20} supports and two untouched splits."""

    arrays = validated["arrays"]
    labels = arrays["labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    physical = arrays["physical_ids"].astype(str)
    class_ids = arrays["class_ids"].astype(str)
    episodes: dict[str, dict[int, Episode]] = {}
    max_k = max(ALLOWED_K)
    for receiver in sorted(validated["receivers"].astype(str).tolist()):
        support_by_k: dict[int, list[int]] = {k: [] for k in ALLOWED_K}
        calibration: list[int] = []
        evaluation: list[int] = []
        for class_id in class_ids.tolist():
            indices = np.flatnonzero((receivers == receiver) & (labels == class_id))
            if indices.size < max_k + 2:
                raise D99D100LODOLockError(
                    f"receiver={receiver}, class={class_id} needs >=22 independent rows; "
                    f"got {indices.size}"
                )
            ordered = indices[np.argsort(physical[indices], kind="stable")]
            class_physical_root = canonical_sha256(physical[ordered].tolist())
            shuffled = _stable_rng(seed, receiver, class_physical_root).permutation(
                ordered
            )
            reserved, remaining = shuffled[:max_k], shuffled[max_k:]
            n_calibration = max(1, int(remaining.size // 2))
            if remaining.size - n_calibration < 1:
                n_calibration = int(remaining.size - 1)
            calibration.extend(remaining[:n_calibration].tolist())
            evaluation.extend(remaining[n_calibration:].tolist())
            for k_shot in ALLOWED_K:
                support_by_k[k_shot].extend(reserved[:k_shot].tolist())
        episodes[receiver] = {}
        for k_shot in ALLOWED_K:
            support = np.asarray(sorted(support_by_k[k_shot]), dtype=np.int64)
            calibration_array = np.asarray(sorted(calibration), dtype=np.int64)
            evaluation_array = np.asarray(sorted(evaluation), dtype=np.int64)
            sets = [
                set(physical[index].tolist())
                for index in (support, calibration_array, evaluation_array)
            ]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError("physical support/calibration/evaluation overlap")
            expected_support = len(class_ids) * k_shot
            if support.size != expected_support:
                raise AssertionError("K-shot support count drift")
            episodes[receiver][k_shot] = Episode(
                receiver, k_shot, support, calibration_array, evaluation_array
            )
    return episodes


def build_pseudo_new_folds(class_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Rotate every class through pseudo-new; no label gets a privileged rule."""

    classes = tuple(str(value) for value in class_ids)
    if len(classes) < 3 or len(set(classes)) != len(classes):
        raise D99D100LODOLockError("pseudo-new LODO requires >=3 unique classes")
    return tuple(
        {
            "fold_id": f"pseudo_new_{index:03d}",
            "pseudo_new": (class_name,),
            "pseudo_old": tuple(value for value in classes if value != class_name),
        }
        for index, class_name in enumerate(classes)
    )


def _subset_ground_bundle(
    bundle: d99.Phase1GroundAggregateBundle,
    *,
    held_domain: str,
    pseudo_old: Sequence[str],
) -> d99.Phase1GroundAggregateBundle:
    domain_indices = [
        index for index, value in enumerate(bundle.domain_ids) if value != held_domain
    ]
    class_lookup = {value: index for index, value in enumerate(bundle.ground_old_registry)}
    try:
        class_indices = [class_lookup[str(value)] for value in pseudo_old]
    except KeyError as exc:
        raise D99D100LODOLockError("pseudo-old class missing from ground bundle") from exc
    if len(domain_indices) < 2 or len(class_indices) < 2:
        raise D99D100LODOLockError("outer LODO needs >=2 retained domains/classes")
    selection = np.ix_(domain_indices, class_indices)
    return d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=bundle.codes_qint8[selection[0], selection[1], :],
        scales_fp16=bundle.scales_fp16[selection],
        domain_class_mask=bundle.domain_class_mask[selection],
        physical_sample_count_floor_uint16=(
            bundle.physical_sample_count_floor_uint16[selection]
        ),
        domain_ids=tuple(bundle.domain_ids[index] for index in domain_indices),
        ground_old_registry=tuple(str(value) for value in pseudo_old),
        aggregation_receipt=bundle.aggregation_receipt,
    )


_GRID_FIELDS = (
    "eta",
    "student_nu",
    "kernel_volume_gamma",
    "shared_h0",
    "scale_prior_strength",
    "scale_min_ratio",
    "scale_max_ratio",
    "d99_temperature",
    "lambda0",
    "ridge_temperature",
    "alpha",
)
MAX_GRID_VALUES_PER_FIELD = 6
MAX_GRID_CANDIDATES = 256


def candidate_grid(value: Mapping[str, Iterable[float]]) -> list[dict[str, float]]:
    if set(value) != set(_GRID_FIELDS):
        raise D99D100LODOLockError(
            f"candidate grid must have exact fields {list(_GRID_FIELDS)}"
        )
    columns: list[list[float]] = []
    for name in _GRID_FIELDS:
        try:
            values = [float(item) for item in value[name]]
        except (TypeError, ValueError) as exc:
            raise D99D100LODOLockError(f"invalid grid field {name}") from exc
        if not values or len(set(values)) != len(values) or not all(
            math.isfinite(item) for item in values
        ):
            raise D99D100LODOLockError(f"grid field {name} is empty/nonfinite/duplicate")
        if len(values) > MAX_GRID_VALUES_PER_FIELD:
            raise D99D100LODOLockError(
                f"grid field {name} exceeds {MAX_GRID_VALUES_PER_FIELD} values"
            )
        columns.append(sorted(values))
    candidate_count = math.prod(len(values) for values in columns)
    if candidate_count > MAX_GRID_CANDIDATES:
        raise D99D100LODOLockError(
            f"candidate grid exceeds {MAX_GRID_CANDIDATES} Cartesian candidates"
        )
    result = [dict(zip(_GRID_FIELDS, row)) for row in itertools.product(*columns)]
    for row in result:
        if (
            not 0.0 <= row["eta"] <= 1.0
            or not 0.0 <= row["alpha"] <= 1.0
            or min(
                row["student_nu"],
                row["kernel_volume_gamma"],
                row["shared_h0"],
                row["scale_prior_strength"],
                row["scale_min_ratio"],
                row["scale_max_ratio"],
                row["d99_temperature"],
                row["lambda0"],
                row["ridge_temperature"],
            ) <= 0.0
            or row["scale_min_ratio"] > 1.0
            or row["scale_max_ratio"] < 1.0
        ):
            raise D99D100LODOLockError("candidate grid violates D99/D100 bounds")
    return result


def _positions(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(classes)}
    try:
        return np.asarray([lookup[str(value)] for value in labels.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise D99D100LODOLockError("query label outside registered classes") from exc


def _metrics(
    probabilities: np.ndarray,
    truth: np.ndarray,
    classes: Sequence[str],
    pseudo_old: Sequence[str],
    pseudo_new: Sequence[str],
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    if (
        probability.ndim != 2
        or probability.shape[0] != len(truth)
        or probability.shape[1] != len(classes)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.allclose(np.sum(probability, axis=1), 1.0, atol=2e-6)
    ):
        raise D99D100LODOLockError("probability matrix drift")
    prediction = np.argmax(probability, axis=1)
    per_class = []
    for index, class_name in enumerate(classes):
        mask = truth == index
        if not np.any(mask):
            raise D99D100LODOLockError("evaluation split lacks a registered class")
        per_class.append((str(class_name), float(np.mean(prediction[mask] == truth[mask]))))
    per_lookup = dict(per_class)
    old = float(np.mean([per_lookup[value] for value in pseudo_old]))
    new = float(np.mean([per_lookup[value] for value in pseudo_new]))
    one_hot = np.eye(len(classes), dtype=np.float64)[truth]
    sample_nll = -np.log(
        np.maximum(probability[np.arange(len(truth)), truth], 1e-12)
    )
    sample_brier = np.sum(np.square(probability - one_hot), axis=1)
    class_balanced_nll = float(
        np.mean([np.mean(sample_nll[truth == index]) for index in range(len(classes))])
    )
    class_balanced_brier = float(
        np.mean([np.mean(sample_brier[truth == index]) for index in range(len(classes))])
    )
    return {
        "row_count": int(len(truth)),
        "balanced_accuracy": float(np.mean([value for _name, value in per_class])),
        "worst_class_floor": float(min(value for _name, value in per_class)),
        "pseudo_old_accuracy": old,
        "pseudo_new_accuracy": new,
        "harmonic_old_new": float(2.0 * old * new / max(old + new, 1e-12)),
        "balanced_nll": class_balanced_nll,
        "brier": class_balanced_brier,
        "per_class_accuracy": dict(per_class),
    }


def _candidate_d99_config(
    base: d99.Phase1D99Lock,
    bundle: d99.Phase1GroundAggregateBundle,
    pseudo_old: Sequence[str],
    candidate: Mapping[str, float],
    k_shot: int,
) -> d99.Phase1D99Lock:
    updates: dict[str, Any] = {
        "student_nu": float(candidate["student_nu"]),
        "kernel_volume_gamma": float(candidate["kernel_volume_gamma"]),
        "shared_h0": float(candidate["shared_h0"]),
        "scale_prior_strength": float(candidate["scale_prior_strength"]),
        "scale_min_ratio": float(candidate["scale_min_ratio"]),
        "scale_max_ratio": float(candidate["scale_max_ratio"]),
        f"eta_k{k_shot}": float(candidate["eta"]),
        "ground_bundle_receipt_sha256": bundle.bundle_sha256,
        "ground_aggregation_receipt_sha256": bundle.aggregation_receipt.receipt_sha256,
        "ground_old_registry": tuple(str(value) for value in pseudo_old),
    }
    return replace(base, **updates)


def _candidate_d100_config(
    d99_config: d99.Phase1D99Lock,
    candidate: Mapping[str, float],
    k_shot: int,
    authority: GroundReleaseAuthority,
) -> d100.Phase1D100Lock:
    values: dict[str, Any] = {}
    for k in ALLOWED_K:
        values[f"lambda_k{k}"] = float(candidate["lambda0"])
        values[f"temperature_k{k}"] = float(candidate["ridge_temperature"])
        values[f"d99_temperature_k{k}"] = float(candidate["d99_temperature"])
        values[f"alpha_k{k}"] = float(candidate["alpha"] if k == k_shot else 0.0)
    values.update(
        {
            "d99_phase1_lock_digest": d99_config.lock_digest,
            "phase1_lodo_rescue_receipt_sha256": canonical_sha256(
                [SCHEMA, authority.manifest_sha256, k_shot, candidate]
            ),
            "external_phase2_authority_sha256": authority.manifest_sha256,
            "quantization_margin_audit_sha256": canonical_sha256(
                ["development_d100_quantization", d99_config.lock_digest]
            ),
        }
    )
    return d100.Phase1D100Lock(**values)


def _evaluate_candidate(
    *,
    arrays: Mapping[str, np.ndarray],
    episode: Episode,
    query_indices: np.ndarray,
    fold: Mapping[str, Any],
    candidate: Mapping[str, float],
    base_d99_config: d99.Phase1D99Lock,
    full_ground_bundle: d99.Phase1GroundAggregateBundle,
    authority: GroundReleaseAuthority,
    d81_logits: np.ndarray,
    d81_source_schema: str,
    d81_source_receipt_sha256: str,
    prepared_cache: dict[tuple[Any, ...], tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    classes = tuple(str(value) for value in arrays["class_ids"].tolist())
    support = episode.support
    pseudo_old = tuple(fold["pseudo_old"])
    pseudo_new = tuple(fold["pseudo_new"])
    cache_key = (
        episode.receiver,
        episode.k_shot,
        fold["fold_id"],
        canonical_sha256(candidate),
    )
    prepared = None if prepared_cache is None else prepared_cache.get(cache_key)
    if prepared is None:
        held_domain = authority.receiver_domain_map[episode.receiver]
        local_bundle = _subset_ground_bundle(
            full_ground_bundle, held_domain=held_domain, pseudo_old=pseudo_old
        )
        config99 = _candidate_d99_config(
            base_d99_config, local_bundle, pseudo_old, candidate, episode.k_shot
        )
        ground = d99.build_ground_geometry(local_bundle, config=config99)
        support_features = arrays["features"][support]
        support_labels = arrays["labels"][support].astype(str)
        support_physical = arrays["physical_ids"][support].astype(str)
        metric = d99.fit_support_metric(
            ground,
            support_features,
            support_labels,
            support_physical,
            classes,
            pseudo_old,
            config=config99,
        )
        bank = d99.build_typed_support_bank(
            metric,
            support_features,
            support_labels,
            support_physical,
            classes,
            config=config99,
        )
        config100 = _candidate_d100_config(
            config99, candidate, episode.k_shot, authority
        )
        state100 = d100.build_simplex_ridge_state(bank, config=config100)
        prepared = (local_bundle, metric, bank, state100)
        if prepared_cache is not None:
            prepared_cache[cache_key] = prepared
    local_bundle, metric, bank, state100 = prepared
    query_features = np.ascontiguousarray(
        arrays["features"][query_indices], dtype=np.float32
    )
    typed_d81 = d100.bind_typed_d81_logits(
        np.ascontiguousarray(d81_logits, dtype=np.float32),
        query_features,
        classes,
        episode.k_shot,
        source_schema=d81_source_schema,
        source_receipt_sha256=d81_source_receipt_sha256,
    )
    fusion = d100.canonical_fuse_typed_d81_d99_d100(
        state100,
        bank,
        typed_d81,
        query_features,
        evaluate_complementarity_branch=True,
    )
    ridge_probability = fusion.ridge_probability_fp32
    if ridge_probability is None:
        raise AssertionError("LODO complementarity requires the ridge branch")
    d99_probability = fusion.d99_probability_fp32
    fused_probability = fusion.fused_probability_fp32
    truth = _positions(arrays["labels"][query_indices].astype(str), classes)
    d81_probability = fusion.d81_probability_fp32
    kernel_probability = fusion.student_t_probability_fp32
    d81_kernel_raw = dict(
        d100.complementarity_audit(d81_probability, kernel_probability, truth)
    )
    d81_kernel_complementarity = {
        "row_count": int(d81_kernel_raw["row_count"]),
        "disagreement_count": int(d81_kernel_raw["disagreement_count"]),
        "kernel_correct_when_d81_wrong_count": int(
            d81_kernel_raw["ridge_correct_when_d99_wrong_count"]
        ),
        "d81_correct_when_kernel_wrong_count": int(
            d81_kernel_raw["d99_correct_when_ridge_wrong_count"]
        ),
        "bidirectional_rescue_nonzero": bool(
            d81_kernel_raw["bidirectional_rescue_nonzero"]
        ),
        "oracle_union_accuracy": float(d81_kernel_raw["oracle_union_accuracy"]),
    }
    complementarity = dict(
        d100.complementarity_audit(d99_probability, ridge_probability, truth)
    )
    class_array = np.asarray(classes, dtype=str)
    d81_prediction = class_array[np.argmax(d81_probability, axis=1)]
    d99_prediction = class_array[np.argmax(d99_probability, axis=1)]
    result = {
        "receiver": episode.receiver,
        "k_shot": episode.k_shot,
        "fold_id": fold["fold_id"],
        "pseudo_old": list(pseudo_old),
        "pseudo_new": list(pseudo_new),
        "candidate": dict(candidate),
        "d81_prediction": d81_prediction.tolist(),
        "d81": _metrics(d81_probability, truth, classes, pseudo_old, pseudo_new),
        "kernel": _metrics(kernel_probability, truth, classes, pseudo_old, pseudo_new),
        "d99": _metrics(d99_probability, truth, classes, pseudo_old, pseudo_new),
        "ridge": _metrics(ridge_probability, truth, classes, pseudo_old, pseudo_new),
        "fused": _metrics(fused_probability, truth, classes, pseudo_old, pseudo_new),
        "complementarity": complementarity,
        "d81_kernel_complementarity": d81_kernel_complementarity,
        "d99_vs_d81_changed_count": int(np.sum(d99_prediction != d81_prediction)),
        "canonical_fusion_audit": dict(fusion.audit),
        "ground_coverage_rho": float(metric.ground_coverage_rho),
        "ground_weight": float(metric.ground_weight),
        "target_weight": float(metric.target_weight),
        "metric_rank": int(metric.metric_basis_fp32.shape[1]),
        "d99_bank_wire_bytes": int(
            len(d99._serialize_receipt_bearing_bank(bank))
        ),
        "d100_state_wire_bytes": int(len(d100.serialize_simplex_ridge_state(state100))),
        "d99_d100_optimizer_steps": 0,
        "d99_d100_epochs": 0,
        "resource": {
            "scope": "nonformal_partial_known_components_only",
            "ground_bundle_numeric_bytes": int(
                local_bundle.codes_qint8.nbytes
                + local_bundle.scales_fp16.nbytes
                + local_bundle.domain_class_mask.nbytes
                + local_bundle.physical_sample_count_floor_uint16.nbytes
            ),
            "d99_numeric_logical_state_bytes": int(
                bank.resource_audit["logical_runtime_numeric_state_bytes"]
            ),
            "d99_actual_wire_bytes": int(
                bank.resource_audit["actual_serialized_runtime_artifact_bytes"]
            ),
            "d100_numeric_logical_state_bytes": int(
                state100.resource_audit["numeric_logical_state_bytes"]
            ),
            "d100_actual_wire_bytes": int(
                state100.resource_audit["actual_serialized_state_bytes"]
            ),
            "d99_d100_known_persistent_wire_bytes": int(
                bank.resource_audit["actual_serialized_runtime_artifact_bytes"]
                + state100.resource_audit["actual_serialized_state_bytes"]
            ),
            "d99_d100_trainable_parameter_equivalent": int(
                bank.resource_audit["trainable_parameters"]
                + state100.resource_audit["trainable_parameter_equivalent"]
            ),
            "d99_d100_query_mac_upper_bound_per_sample": int(
                state100.resource_audit[
                    "combined_query_mac_upper_bound_per_sample"
                ]
            ),
            "d99_fit_peak_transient_bytes_upper_bound": int(
                bank.resource_audit["peak_transient_bytes_upper_bound"]
            ),
            "d81_target_head_persistent_wire_bytes": None,
            "d81_query_mac_upper_bound_per_sample": None,
            "d81_fit_peak_transient_bytes_upper_bound": None,
            "d100_fit_peak_transient_bytes_upper_bound": None,
            "complete_combined_persistent_upper_bound_available": False,
            "complete_combined_parameter_count_available": False,
            "complete_combined_query_mac_available": False,
            "complete_combined_fit_peak_available": False,
            "formal_under_256kib_claim": False,
            "formal_under_80k_parameter_claim": False,
        },
    }
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise D99D100LODOLockError("cannot aggregate zero rows")
    total = sum(int(row["complementarity"]["row_count"]) for row in rows)
    disagreement = sum(
        int(row["complementarity"]["disagreement_count"]) for row in rows
    )
    rescue_ridge = sum(
        int(row["complementarity"]["ridge_correct_when_d99_wrong_count"])
        for row in rows
    )
    rescue_d99 = sum(
        int(row["complementarity"]["d99_correct_when_ridge_wrong_count"])
        for row in rows
    )
    d81_kernel_disagreement = sum(
        int(row["d81_kernel_complementarity"]["disagreement_count"])
        for row in rows
    )
    rescue_kernel = sum(
        int(
            row["d81_kernel_complementarity"][
                "kernel_correct_when_d81_wrong_count"
            ]
        )
        for row in rows
    )
    rescue_d81 = sum(
        int(
            row["d81_kernel_complementarity"][
                "d81_correct_when_kernel_wrong_count"
            ]
        )
        for row in rows
    )
    d99_vs_d81_changed_count = sum(
        int(row["d99_vs_d81_changed_count"]) for row in rows
    )
    summary: dict[str, Any] = {
        "episode_fold_count": len(rows),
        "row_count": total,
        "disagreement_count": disagreement,
        "disagreement_rate": float(disagreement / total),
        "ridge_correct_when_d99_wrong_count": rescue_ridge,
        "d99_correct_when_ridge_wrong_count": rescue_d99,
        "bidirectional_rescue_nonzero": bool(rescue_ridge > 0 and rescue_d99 > 0),
        "d81_kernel_disagreement_count": d81_kernel_disagreement,
        "kernel_correct_when_d81_wrong_count": rescue_kernel,
        "d81_correct_when_kernel_wrong_count": rescue_d81,
        "d81_kernel_bidirectional_rescue_nonzero": bool(
            rescue_kernel > 0 and rescue_d81 > 0
        ),
        "d99_vs_d81_changed_count": d99_vs_d81_changed_count,
        "oracle_union_accuracy": float(
            np.average(
                [row["complementarity"]["oracle_union_accuracy"] for row in rows],
                weights=[row["complementarity"]["row_count"] for row in rows],
            )
        ),
        "mean_ground_coverage_rho": float(
            np.mean([row["ground_coverage_rho"] for row in rows])
        ),
        "max_d99_bank_wire_bytes": int(max(row["d99_bank_wire_bytes"] for row in rows)),
        "max_d100_state_wire_bytes": int(max(row["d100_state_wire_bytes"] for row in rows)),
    }
    for head in ("d81", "kernel", "d99", "ridge", "fused"):
        summary[head] = {
            key: float(np.mean([row[head][key] for row in rows]))
            for key in (
                "balanced_accuracy",
                "pseudo_old_accuracy",
                "pseudo_new_accuracy",
                "harmonic_old_new",
                "balanced_nll",
                "brier",
            )
        }
        summary[head]["worst_class_floor"] = float(
            min(row[head]["worst_class_floor"] for row in rows)
        )
    summary["floor_delta_fused_minus_d99"] = float(
        summary["fused"]["worst_class_floor"]
        - summary["d99"]["worst_class_floor"]
    )
    summary["floor_delta_d99_minus_d81"] = float(
        summary["d99"]["worst_class_floor"]
        - summary["d81"]["worst_class_floor"]
    )
    summary["balanced_nll_improvement_d81_minus_d99"] = float(
        summary["d81"]["balanced_nll"] - summary["d99"]["balanced_nll"]
    )
    summary["balanced_nll_improvement_d99_minus_fused"] = float(
        summary["d99"]["balanced_nll"] - summary["fused"]["balanced_nll"]
    )
    d99_paired = []
    for row in rows:
        d99_paired.append(
            {
                "receiver": row["receiver"],
                "fold_id": row["fold_id"],
                "floor_delta_d99_minus_d81": float(
                    row["d99"]["worst_class_floor"]
                    - row["d81"]["worst_class_floor"]
                ),
                "balanced_nll_improvement_d81_minus_d99": float(
                    row["d81"]["balanced_nll"] - row["d99"]["balanced_nll"]
                ),
                "d99_vs_d81_changed_count": int(row["d99_vs_d81_changed_count"]),
                "disagreement_count": int(
                    row["d81_kernel_complementarity"]["disagreement_count"]
                ),
                "kernel_correct_when_d81_wrong_count": int(
                    row["d81_kernel_complementarity"][
                        "kernel_correct_when_d81_wrong_count"
                    ]
                ),
                "d81_correct_when_kernel_wrong_count": int(
                    row["d81_kernel_complementarity"][
                        "d81_correct_when_kernel_wrong_count"
                    ]
                ),
            }
        )
    summary["d81_kernel_rescue_distribution"] = {
        "pair_count": len(d99_paired),
        "kernel_rescue_zero_pair_count": int(
            sum(row["kernel_correct_when_d81_wrong_count"] == 0 for row in d99_paired)
        ),
        "d81_rescue_zero_pair_count": int(
            sum(row["d81_correct_when_kernel_wrong_count"] == 0 for row in d99_paired)
        ),
        "d99_identity_pair_count": int(
            sum(row["d99_vs_d81_changed_count"] == 0 for row in d99_paired)
        ),
        "rows": d99_paired,
    }
    paired = []
    for row in rows:
        deltas = {
            "worst_class_floor": float(
                row["fused"]["worst_class_floor"] - row["d99"]["worst_class_floor"]
            ),
            "pseudo_old_accuracy": float(
                row["fused"]["pseudo_old_accuracy"] - row["d99"]["pseudo_old_accuracy"]
            ),
            "pseudo_new_accuracy": float(
                row["fused"]["pseudo_new_accuracy"] - row["d99"]["pseudo_new_accuracy"]
            ),
            "harmonic_old_new": float(
                row["fused"]["harmonic_old_new"] - row["d99"]["harmonic_old_new"]
            ),
        }
        paired.append(
            {
                "receiver": row["receiver"],
                "fold_id": row["fold_id"],
                "metric_delta_fused_minus_d99": deltas,
                "all_guard_metrics_non_decreasing": bool(
                    all(value >= -1e-12 for value in deltas.values())
                ),
                "disagreement_count": int(
                    row["complementarity"]["disagreement_count"]
                ),
                "ridge_correct_when_d99_wrong_count": int(
                    row["complementarity"]["ridge_correct_when_d99_wrong_count"]
                ),
                "d99_correct_when_ridge_wrong_count": int(
                    row["complementarity"]["d99_correct_when_ridge_wrong_count"]
                ),
            }
        )
    summary["paired_receiver_pseudo_new_guard"] = {
        "pair_count": len(paired),
        "all_pairs_non_decreasing": bool(
            all(row["all_guard_metrics_non_decreasing"] for row in paired)
        ),
        "degraded_pair_count": int(
            sum(not row["all_guard_metrics_non_decreasing"] for row in paired)
        ),
        "rows": paired,
    }
    summary["bidirectional_rescue_distribution"] = {
        "pair_count": len(paired),
        "ridge_rescue_zero_pair_count": int(
            sum(row["ridge_correct_when_d99_wrong_count"] == 0 for row in paired)
        ),
        "d99_rescue_zero_pair_count": int(
            sum(row["d99_correct_when_ridge_wrong_count"] == 0 for row in paired)
        ),
        "disagreement_count_min": int(min(row["disagreement_count"] for row in paired)),
        "disagreement_count_max": int(max(row["disagreement_count"] for row in paired)),
        "ridge_rescue_count_min": int(
            min(row["ridge_correct_when_d99_wrong_count"] for row in paired)
        ),
        "ridge_rescue_count_max": int(
            max(row["ridge_correct_when_d99_wrong_count"] for row in paired)
        ),
        "d99_rescue_count_min": int(
            min(row["d99_correct_when_ridge_wrong_count"] for row in paired)
        ),
        "d99_rescue_count_max": int(
            max(row["d99_correct_when_ridge_wrong_count"] for row in paired)
        ),
        "rows": [
            {
                key: row[key]
                for key in (
                    "receiver",
                    "fold_id",
                    "disagreement_count",
                    "ridge_correct_when_d99_wrong_count",
                    "d99_correct_when_ridge_wrong_count",
                )
            }
            for row in paired
        ],
    }
    return summary


def enforce_d99_guard(k_shot: int, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Admit D99 only when it is a strict, non-regressive D81 improvement."""

    if int(k_shot) not in ALLOWED_K:
        raise D99D100LODOLockError("D99 guard K outside locked set")
    floor_ok = bool(summary["floor_delta_d99_minus_d81"] >= -1e-12)
    nll_ok = bool(
        summary["d99"]["balanced_nll"]
        < summary["d81"]["balanced_nll"] - STRICT_NLL_IMPROVEMENT_TOLERANCE
    )
    rescue_ok = bool(summary["d81_kernel_bidirectional_rescue_nonzero"])
    nonidentity_ok = bool(
        int(k_shot) != 1 or int(summary["d99_vs_d81_changed_count"]) > 0
    )
    reasons = []
    if not floor_ok:
        reasons.append("D99_WORST_FLOOR_REGRESSION")
    if not nll_ok:
        reasons.append("D99_NO_STRICT_BALANCED_NLL_IMPROVEMENT")
    if not rescue_ok:
        reasons.append("D81_KERNEL_BIDIRECTIONAL_RESCUE_MISSING")
    if not nonidentity_ok:
        reasons.append("K1_NO_NONIDENTITY_CANDIDATE")
    return {
        "d99_eligible": bool(floor_ok and nll_ok and rescue_ok and nonidentity_ok),
        "d99_guard": {
            "worst_floor_non_decreasing_vs_d81": floor_ok,
            "balanced_nll_strictly_improved_vs_d81": nll_ok,
            "balanced_nll_improvement_tolerance": STRICT_NLL_IMPROVEMENT_TOLERANCE,
            "d81_kernel_bidirectional_rescue_nonzero": rescue_ok,
            "k1_nonidentity_prediction_required": int(k_shot) == 1,
            "d99_vs_d81_changed_count": int(summary["d99_vs_d81_changed_count"]),
            "k1_nonidentity_prediction_passed": nonidentity_ok,
            "failure_reasons": reasons,
            "rule": (
                "d99_worst_floor_must_not_decrease_vs_d81_and_balanced_nll_"
                "must_strictly_improve_and_d81_kernel_bidirectional_rescue_"
                "must_be_nonzero_and_k1_must_change_at_least_one_prediction"
            ),
        },
    }


def enforce_alpha_guard(
    candidate: Mapping[str, float], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Return effective parameters; unsafe complementarity forces alpha=0."""

    rescue_ok = bool(summary["bidirectional_rescue_nonzero"])
    paired_ok = bool(
        summary["paired_receiver_pseudo_new_guard"]["all_pairs_non_decreasing"]
    )
    nll_ok = bool(
        summary["fused"]["balanced_nll"]
        < summary["d99"]["balanced_nll"] - STRICT_NLL_IMPROVEMENT_TOLERANCE
    )
    effective = dict(candidate)
    forced = not (rescue_ok and paired_ok and nll_ok)
    if forced:
        effective["alpha"] = 0.0
    return {
        "effective_parameters": effective,
        "alpha_forced_zero": forced,
        "d100_eligible": not forced,
        "guard": {
            "bidirectional_rescue_nonzero": rescue_ok,
            "every_receiver_pseudo_new_pair_floor_old_new_h_non_decreasing": paired_ok,
            "degraded_pair_count": summary["paired_receiver_pseudo_new_guard"][
                "degraded_pair_count"
            ],
            "balanced_nll_strictly_improved_vs_d99": nll_ok,
            "balanced_nll_improvement_tolerance": STRICT_NLL_IMPROVEMENT_TOLERANCE,
            "rule": (
                "aggregate_bidirectional_rescue_must_be_nonzero_and_every_"
                "receiver_x_pseudo_new_pair_floor_old_new_h_must_not_decrease_"
                "and_fused_balanced_nll_must_strictly_improve_vs_d99"
            ),
        },
    }


def _selection_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    summary = item["effective_summary"]
    return (
        summary["fused"]["balanced_nll"],
        summary["fused"]["brier"],
        -summary["fused"]["worst_class_floor"],
        -summary["fused"]["harmonic_old_new"],
        canonical_sha256(item["effective_parameters"]),
    )


def _apply_effective_summary(
    rows: Sequence[Mapping[str, Any]], guard: Mapping[str, Any]
) -> dict[str, Any]:
    if not guard["alpha_forced_zero"]:
        return _aggregate(rows)
    copied = []
    for row in rows:
        changed = dict(row)
        changed["fused"] = dict(row["d99"])
        copied.append(changed)
    return _aggregate(copied)


def _rank_candidates(rows_by_candidate: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    ranking = []
    for index, rows in enumerate(rows_by_candidate):
        raw = _aggregate(rows)
        d99_guard = enforce_d99_guard(int(rows[0]["k_shot"]), raw)
        guard = enforce_alpha_guard(rows[0]["candidate"], raw)
        effective_summary = _apply_effective_summary(rows, guard)
        ranking.append(
            {
                "candidate_index": index,
                "requested_parameters": dict(rows[0]["candidate"]),
                **d99_guard,
                **guard,
                "raw_summary": raw,
                "effective_summary": effective_summary,
            }
        )
    return sorted(ranking, key=lambda item: (not item["d99_eligible"], *_selection_key(item)))


def _d99_block_reason(k_shot: int, ranking: Sequence[Mapping[str, Any]]) -> str:
    if int(k_shot) == 1 and ranking and all(
        int(row["raw_summary"]["d99_vs_d81_changed_count"]) == 0 for row in ranking
    ):
        return "K1_NO_NONIDENTITY_CANDIDATE"
    return "D99_NO_CANDIDATE_PASSED_PHASE1_ADMISSION"


def run_phase1_d99_d100_lodo(
    archive_path: str | Path,
    archive_manifest_path: str | Path,
    archive_manifest_sha256: str,
    *,
    ground_bundle: d99.Phase1GroundAggregateBundle,
    ground_authority: GroundReleaseAuthority,
    base_d99_config: d99.Phase1D99Lock,
    base_scorer: D81Phase1EpisodeScorer,
    base_scorer_id: str,
    base_scorer_receipt_sha256: str,
    grid: Mapping[str, Iterable[float]],
    code_sha256: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    """Run K-specific nested receiver LODO and return one immutable receipt."""

    if not isinstance(archive_path, (str, Path)):
        raise D99D100LODOLockError("Phase1 feature archive must be a file path")
    if type(ground_bundle) is not d99.Phase1GroundAggregateBundle:
        raise D99D100LODOLockError("exact typed ground bundle required")
    if type(ground_authority) is not GroundReleaseAuthority:
        raise D99D100LODOLockError("exact loaded ground authority required")
    if ground_authority.loader_token is not _GROUND_AUTHORITY_TOKEN:
        raise D99D100LODOLockError("ground authority token drift")
    try:
        ground_payload = json.loads(ground_authority.manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99D100LODOLockError("ground authority manifest bytes drift") from exc
    expected_ground_payload = ground_release_manifest_payload(
        ground_bundle,
        ground_authority.receiver_domain_map,
        producer_code_sha256=ground_payload.get("producer_code_sha256"),
        release_schema=ground_payload.get("schema", ""),
        release_status=ground_payload.get("status", ""),
    )
    if (
        ground_authority.manifest_bytes != _canonical_bytes(ground_payload)
        or ground_payload != expected_ground_payload
    ):
        raise D99D100LODOLockError("ground authority bytes/bundle revalidation failed")
    if (
        ground_authority.bundle_sha256 != ground_bundle.bundle_sha256
        or ground_authority.aggregation_receipt_sha256
        != ground_bundle.aggregation_receipt.receipt_sha256
        or ground_authority.phase1_checkpoint_sha256 != BASE_CHECKPOINT_SHA256
        or tuple(base_d99_config.ground_old_registry)
        != tuple(ground_bundle.ground_old_registry)
    ):
        raise D99D100LODOLockError("checkpoint/ground/base D99 closure drift")
    normalized_code_sha = {
        str(key): _require_sha(value, f"code SHA {key}")
        for key, value in code_sha256.items()
    }
    required_code = {
        "stage2_d99_d100_phase1_lodo",
        "stage2_d99_ra_cgtmk_d81",
        "stage2_d100_ra_cgspr_lgf",
        "stage2_d81_phase1_episode_scorer",
        "run_d99_d100_phase1_lodo",
    }
    if set(normalized_code_sha) != required_code:
        raise D99D100LODOLockError("code SHA registry must be exact")
    if normalized_code_sha != current_code_sha256():
        raise D99D100LODOLockError("code SHA registry/source bytes drift")
    scorer_contract = _validate_base_scorer_contract(
        base_scorer,
        expected_scorer_id=base_scorer_id,
        expected_receipt_sha256=base_scorer_receipt_sha256,
    )
    validated = validate_feature_archive(archive_path)
    archive_manifest = _validate_feature_archive_manifest(
        archive_manifest_path,
        archive_manifest_sha256,
        validated=validated,
    )
    arrays = validated["arrays"]
    classes = tuple(str(value) for value in arrays["class_ids"].tolist())
    receivers = tuple(sorted(validated["receivers"].astype(str).tolist()))
    if set(classes) != set(ground_bundle.ground_old_registry):
        raise D99D100LODOLockError("feature classes and ground registry differ")
    if set(receivers) != set(ground_authority.receiver_domain_map):
        raise D99D100LODOLockError("feature receivers and ground receiver map differ")
    if set(ground_authority.receiver_domain_map.values()) != set(ground_bundle.domain_ids):
        raise D99D100LODOLockError("receiver map and ground domains differ")
    episodes = build_receiver_lodo_episodes(validated, seed=int(seed))
    folds = build_pseudo_new_folds(classes)
    candidates = candidate_grid(grid)

    # D81 is fitted once per (receiver,K) and jointly scores calibration and
    # evaluation.  Pseudo-new folds only change the D99 local ground slice, so
    # they share the fixed D81 Phase1 head and never repeat its 20-step fit.
    d81_cache: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for receiver in receivers:
        for k_shot in ALLOWED_K:
            episode = episodes[receiver][k_shot]
            query = np.concatenate([episode.calibration, episode.evaluation])
            logits = base_scorer(
                arrays["features"][episode.support],
                arrays["labels"][episode.support].astype(str),
                arrays["features"][query],
                np.asarray(classes),
            )
            n_calibration = len(episode.calibration)
            # The D81 fit is identical across pseudo-new folds because all
            # registered support/classes are retained.  Store explicit fold
            # aliases so the lifecycle count remains auditable.
            for fold in folds:
                d81_cache[(receiver, k_shot, fold["fold_id"])] = {
                    "calibration": logits[:n_calibration],
                    "evaluation": logits[n_calibration:],
                }
    _recheck_base_scorer_contract(base_scorer, scorer_contract)

    calibration_rows: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    evaluation_rows: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    prepared_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for k_shot in ALLOWED_K:
        for receiver in receivers:
            episode = episodes[receiver][k_shot]
            for fold in folds:
                key = (receiver, k_shot, fold["fold_id"])
                for candidate_index, candidate in enumerate(candidates):
                    common = dict(
                        arrays=arrays,
                        episode=episode,
                        fold=fold,
                        candidate=candidate,
                        base_d99_config=base_d99_config,
                        full_ground_bundle=ground_bundle,
                        authority=ground_authority,
                        prepared_cache=prepared_cache,
                        d81_source_schema=scorer_contract["schema"],
                        d81_source_receipt_sha256=scorer_contract["receipt_sha256"],
                    )
                    calibration_rows[(k_shot, receiver, candidate_index, fold["fold_id"])] = (
                        _evaluate_candidate(
                            query_indices=episode.calibration,
                            d81_logits=d81_cache[key]["calibration"],
                            **common,
                        )
                    )
                    evaluation_rows[(k_shot, receiver, candidate_index, fold["fold_id"])] = (
                        _evaluate_candidate(
                            query_indices=episode.evaluation,
                            d81_logits=d81_cache[key]["evaluation"],
                            **common,
                        )
                    )

    selected_by_k: dict[str, Any] = {}
    outer_rows: list[dict[str, Any]] = []
    final_eval_rows: list[dict[str, Any]] = []
    for k_shot in ALLOWED_K:
        outer_selections = []
        for held_receiver in receivers:
            candidate_rows = [
                [
                    calibration_rows[(k_shot, receiver, candidate_index, fold["fold_id"])]
                    for receiver in receivers
                    if receiver != held_receiver
                    for fold in folds
                ]
                for candidate_index in range(len(candidates))
            ]
            ranking = _rank_candidates(candidate_rows)
            selected = next((row for row in ranking if row["d99_eligible"]), None)
            if selected is None:
                outer_selections.append(
                    {
                        "heldout_receiver": held_receiver,
                        "selection_status": "BLOCKED_D99_ADMISSION",
                        "block_reason": _d99_block_reason(k_shot, ranking),
                        "selection_other_receivers": None,
                    }
                )
                continue
            selected_index = int(selected["candidate_index"])
            outer_selections.append(
                {
                    "heldout_receiver": held_receiver,
                    "selection_status": "D99_ADMITTED",
                    "selection_other_receivers": selected,
                }
            )
            held_rows = [
                evaluation_rows[(k_shot, held_receiver, selected_index, fold["fold_id"])]
                for fold in folds
            ]
            effective_summary = _apply_effective_summary(held_rows, selected)
            outer_rows.append(
                {
                    "receiver": held_receiver,
                    "k_shot": k_shot,
                    "effective_parameters": selected["effective_parameters"],
                    "alpha_forced_zero": selected["alpha_forced_zero"],
                    "summary": effective_summary,
                }
            )
        final_candidate_rows = [
            [
                calibration_rows[(k_shot, receiver, candidate_index, fold["fold_id"])]
                for receiver in receivers
                for fold in folds
            ]
            for candidate_index in range(len(candidates))
        ]
        ranking = _rank_candidates(final_candidate_rows)
        selected = next((row for row in ranking if row["d99_eligible"]), None)
        if selected is None:
            selected_by_k[str(k_shot)] = {
                "lock_status": "BLOCKED_D99_ADMISSION",
                "block_reason": _d99_block_reason(k_shot, ranking),
                "d99_eligible": False,
                "d100_eligible": False,
                "selected": None,
                "candidate_ranking": ranking,
                "outer_selections": outer_selections,
                "final_evaluation_summary": None,
            }
            continue
        selected_index = int(selected["candidate_index"])
        selected_eval = [
            evaluation_rows[(k_shot, receiver, selected_index, fold["fold_id"])]
            for receiver in receivers
            for fold in folds
        ]
        selected_eval_summary = _apply_effective_summary(selected_eval, selected)
        selected_by_k[str(k_shot)] = {
            "lock_status": "D99_ADMITTED",
            "block_reason": None,
            "d99_eligible": True,
            "d100_eligible": bool(selected["d100_eligible"]),
            "selected": selected,
            "candidate_ranking": ranking,
            "outer_selections": outer_selections,
            "final_evaluation_summary": selected_eval_summary,
        }
        for row in selected_eval:
            reported_row = dict(row)
            if selected["alpha_forced_zero"]:
                reported_row["requested_fused"] = dict(row["fused"])
                reported_row["fused"] = dict(row["d99"])
            final_eval_rows.append(
                {
                    **reported_row,
                    "effective_alpha": selected["effective_parameters"]["alpha"],
                    "alpha_forced_zero": selected["alpha_forced_zero"],
                }
            )

    all_k_d99_eligible = bool(
        all(selected_by_k[str(k)]["d99_eligible"] for k in ALLOWED_K)
    )
    formal = bool(
        archive_manifest["full_phase1_lock"]
        and ground_authority.formal_phase1_eligible
        and all_k_d99_eligible
    )
    d81_fit_count = len(receivers) * len(ALLOWED_K)
    split_receipt = []
    physical = arrays["physical_ids"].astype(str)
    for receiver in receivers:
        support_sha = {
            str(k): canonical_sha256(
                sorted(physical[episodes[receiver][k].support].tolist())
            )
            for k in ALLOWED_K
        }
        split_receipt.append(
            {
                "receiver": receiver,
                "ground_held_domain": ground_authority.receiver_domain_map[receiver],
                "support_physical_ids_sha256_by_k": support_sha,
                "support_count_by_k": {
                    str(k): int(len(episodes[receiver][k].support)) for k in ALLOWED_K
                },
                "k20_is_distinct_real_episode": bool(
                    support_sha["20"] != support_sha["10"]
                    and len(episodes[receiver][20].support)
                    == 2 * len(episodes[receiver][10].support)
                ),
                "calibration_physical_ids_sha256": canonical_sha256(
                    sorted(physical[episodes[receiver][20].calibration].tolist())
                ),
                "evaluation_physical_ids_sha256": canonical_sha256(
                    sorted(physical[episodes[receiver][20].evaluation].tolist())
                ),
                "physical_splits_disjoint": True,
            }
        )
    full_ground_numeric_bytes = int(
        ground_bundle.codes_qint8.nbytes
        + ground_bundle.scales_fp16.nbytes
        + ground_bundle.domain_class_mask.nbytes
        + ground_bundle.physical_sample_count_floor_uint16.nbytes
    )
    fixed_d81_ground_basis_numeric_bytes = int(
        base_scorer.nuisance_basis_fp64.nbytes
        + base_scorer.spectral_weights_fp64.nbytes
    )
    selected_resources = [row["resource"] for row in final_eval_rows]
    def selected_resource_max(name: str) -> int | None:
        if not selected_resources:
            return None
        return int(max(row[name] for row in selected_resources))

    max_d99_d100_persistent_wire = selected_resource_max(
        "d99_d100_known_persistent_wire_bytes"
    )
    known_persistent_component_bytes = (
        None
        if max_d99_d100_persistent_wire is None
        else int(
            full_ground_numeric_bytes
            + fixed_d81_ground_basis_numeric_bytes
            + max_d99_d100_persistent_wire
        )
    )
    d99_blocked_by_k = {
        str(k): selected_by_k[str(k)]["block_reason"]
        for k in ALLOWED_K
        if not selected_by_k[str(k)]["d99_eligible"]
    }
    unsigned = {
        "schema": SCHEMA,
        "status": STATUS_FORMAL if formal else STATUS_DIAGNOSTIC,
        "formal_authority_status": STATUS_FORMAL if formal else STATUS_BLOCKED,
        "formal_phase1_lock": formal,
        "canonical_lock_artifact_write_allowed": formal,
        "trusted_constant_generated_by_this_producer": False,
        "selection_scope": (
            "phase1_only_pseudo_target_receiver_lodo_with_fixed_phase1_"
            "encoder_and_fixed_global_d81_ground_basis"
        ),
        "allowed_k": list(ALLOWED_K),
        "seed": int(seed),
        "archive": {
            "array_archive_sha256": validated["archive_sha256"],
            "file_sha256": validated["archive_file_sha256"],
            "manifest": archive_manifest,
            "sample_count": validated["sample_count"],
        },
        "ground": {
            "release_schema": ground_payload["schema"],
            "release_status": ground_payload["status"],
            "bundle_sha256": ground_bundle.bundle_sha256,
            "aggregation_receipt_sha256": (
                ground_bundle.aggregation_receipt.receipt_sha256
            ),
            "release_manifest_sha256": ground_authority.manifest_sha256,
            "authority_status": ground_authority.authority_status,
            "formal_phase1_eligible": ground_authority.formal_phase1_eligible,
            "receiver_domain_map": dict(ground_authority.receiver_domain_map),
            "domain_ids": list(ground_bundle.domain_ids),
            "source_receiver_split_sha256": canonical_sha256(split_receipt),
        },
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "base_d99_lock_digest": base_d99_config.lock_digest,
        "d81_scorer": scorer_contract,
        "code_sha256": normalized_code_sha,
        "grid": {name: sorted(float(value) for value in grid[name]) for name in _GRID_FIELDS},
        "candidate_count": len(candidates),
        "pseudo_new_folds": [dict(fold) for fold in folds],
        "method_formula": {
            "d99": "(1-eta_K)*softmax(D81_logits)+eta_K*softmax(StudentT_metric_logits/T_D99_K)",
            "ridge": "softmax(D100_simplex_ridge_logits/T_R_K)",
            "final": "(1-alpha_K)*D99_probability+alpha_K*ridge_probability",
            "student_t_scale": (
                "Phase1_locked_shared_h0_for_K1_else_uniform_support_shrunk_"
                "class_scale_clipped_by_locked_ratios"
            ),
        },
        "locked_parameters_by_k": {
            key: dict(value["selected"]["effective_parameters"])
            for key, value in selected_by_k.items()
            if value["selected"] is not None
        },
        "D99_eligible_by_k": {
            key: bool(value["d99_eligible"]) for key, value in selected_by_k.items()
        },
        "D100_eligible_by_k": {
            key: bool(value["d100_eligible"]) for key, value in selected_by_k.items()
        },
        "D99_block_reason_by_k": d99_blocked_by_k,
        "selected_by_k": selected_by_k,
        "outer_lodo_rows": outer_rows,
        "final_evaluation_rows": final_eval_rows,
        "split_receipt": split_receipt,
        "protocol_audit": {
            "phase1_only": True,
            "single_leo_observation_archive": True,
            "target_rows_used": 0,
            "query_rows_used_for_selection": 0,
            "clean_or_raw_iq_used": False,
            "pseudo_new_ground_rows_used": False,
            "d99_local_ground_held_receiver_row_removed": True,
            "d99_local_ground_pseudo_new_classes_removed": True,
            "d81_fixed_global_ground_basis_retrained_per_outer_fold": False,
            "d81_fixed_global_ground_basis_may_include_held_receiver_domains": True,
            "whole_method_held_receiver_ground_unused_claim": False,
            "lodo_claim": (
                "pseudo_target_receiver_evaluation_of_support_adaptation_and_"
                "d99_local_ground_ablation_not_full_encoder_or_d81_basis_retraining"
            ),
            "all_classes_rotate_through_pseudo_new": True,
            "class_specific_hyperparameters": False,
            "k20_copied_from_k10": False,
            "d99_d81_admission_guard_per_k": True,
            "k1_d99_nonidentity_required": True,
            "alpha_guard_per_k": True,
            "strict_balanced_nll_improvement_tolerance": (
                STRICT_NLL_IMPROVEMENT_TOLERANCE
            ),
        },
        "resource_audit": {
            "d81_episode_fit_count": d81_fit_count,
            "d81_optimizer_steps_per_episode_fit": 20,
            "d81_optimizer_steps_total": 20 * d81_fit_count,
            "d81_fit_repeated_per_pseudo_new_fold": False,
            "d99_d100_optimizer_steps": 0,
            "d99_d100_epochs": 0,
            "candidate_episode_evaluations": int(
                len(candidates) * len(receivers) * len(ALLOWED_K) * len(folds) * 2
            ),
            "analytic_d99_d100_state_build_count": int(
                len(candidates) * len(receivers) * len(ALLOWED_K) * len(folds)
            ),
            "calibration_evaluation_share_analytic_state": True,
            "fixed_d81_ground_basis_numeric_bytes": (
                fixed_d81_ground_basis_numeric_bytes
            ),
            "full_d99_ground_bundle_numeric_bytes": full_ground_numeric_bytes,
            "max_selected_d99_d100_known_persistent_wire_bytes": (
                max_d99_d100_persistent_wire
            ),
            "known_persistent_component_bytes_lower_bound": (
                known_persistent_component_bytes
            ),
            "max_selected_d99_d100_trainable_parameter_equivalent": int(
                selected_resource_max("d99_d100_trainable_parameter_equivalent")
            ) if selected_resources else None,
            "max_selected_d99_d100_query_mac_upper_bound_per_sample": (
                selected_resource_max("d99_d100_query_mac_upper_bound_per_sample")
            ),
            "max_selected_d99_fit_peak_transient_bytes_upper_bound": (
                selected_resource_max("d99_fit_peak_transient_bytes_upper_bound")
            ),
            "d81_target_head_persistent_wire_bytes": None,
            "d81_query_mac_upper_bound_per_sample": None,
            "d81_fit_peak_transient_bytes_upper_bound": None,
            "d100_fit_peak_transient_bytes_upper_bound": None,
            "complete_combined_persistent_upper_bound_available": False,
            "complete_combined_parameter_count_available": False,
            "complete_combined_query_mac_available": False,
            "complete_combined_fit_peak_available": False,
            "resource_claim_status": "NONFORMAL_PARTIAL_KNOWN_COMPONENTS_ONLY",
            "formal_under_256kib_claim": False,
            "formal_under_80k_parameter_claim": False,
        },
        "blocked_inputs": [] if formal else [
            *[
                key
                for key, ready in (
                    ("formal_feature_archive", archive_manifest["full_phase1_lock"]),
                    ("independent_ground_authority_root", ground_authority.formal_phase1_eligible),
                )
                if not ready
            ],
            *[f"d99_admission_{key}:{reason}" for key, reason in d99_blocked_by_k.items()],
        ],
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def verify_receipt(value: Mapping[str, Any]) -> bool:
    claimed = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return isinstance(claimed, str) and claimed == canonical_sha256(unsigned)


__all__ = [
    "ALLOWED_K",
    "D99D100LODOLockError",
    "GROUND_RELEASE_DEVELOPMENT_SCHEMA",
    "GROUND_RELEASE_DEVELOPMENT_STATUS",
    "GROUND_RELEASE_SCHEMA",
    "GroundReleaseAuthority",
    "SCHEMA",
    "STATUS_BLOCKED",
    "STATUS_DIAGNOSTIC",
    "STATUS_FORMAL",
    "STRICT_NLL_IMPROVEMENT_TOLERANCE",
    "build_pseudo_new_folds",
    "build_receiver_lodo_episodes",
    "candidate_grid",
    "current_code_sha256",
    "enforce_alpha_guard",
    "enforce_d99_guard",
    "ground_release_manifest_payload",
    "load_ground_release_authority",
    "run_phase1_d99_d100_lodo",
    "verify_receipt",
]
