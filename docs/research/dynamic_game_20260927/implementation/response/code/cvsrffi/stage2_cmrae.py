"""D18 class-balanced magnitude-residual amplitude equalization (CMRAE).

The module accepts only internally authorized, SHA-bound received-IQ
artifacts.  It learns one old-class-balanced, low-frequency magnitude DCT8
state from support, does not directly rotate nonzero FFT-bin phase, never
estimates or removes CFO, and
uses one deterministic representation per physical observation.  Registration
freezes the equalizer and every old prototype and appends new prototypes only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
DCT_RANK = 8
TAU = math.log(1.10)
MAX_ADAPTER_STATE_BYTES = 16 * 1024
MAX_FULL_SERIALIZED_STATE_BYTES = 256 * 1024
ALLOWED_CHANNEL_VIEWS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
SCHEMA = "cvs.phase2.cmrae.v1"
OPERATOR_ID = "cmrae_dct8_fixed_received_iq"
_IQ_TOKEN = object()
_BACKBONE_TOKEN = object()


class CmraeError(ValueError):
    """Raised when a D18 protocol, support, state, or resource guard fails."""


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _require_sha(value: str, field: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise CmraeError(f"{field} SHA drift")
    return result


class RuntimeAuthorizedReceivedIQArtifact:
    """Immutable fixed received-IQ rows bound to sealed runtime provenance."""

    def __init__(
        self,
        *,
        received_iq: np.ndarray,
        physical_sample_ids: Sequence[str],
        parent_received_iq_sha256: Sequence[str],
        overlay_tokens: Sequence[str],
        source_leo_provenance_sha256: Sequence[str],
        source_leo_cache_sha256: Sequence[str],
        target_channel_views: Sequence[str],
        satellite_seeds: Sequence[int],
        overlay_provenance_sha256: Sequence[str],
        sealed_runtime_sha256: str,
        sealed_phase1_checkpoint_sha256: str,
        feature_code_sha256: str,
        purpose: str,
        _token: object,
    ) -> None:
        if _token is not _IQ_TOKEN:
            raise CmraeError("received-IQ artifact must come from authorized runtime")
        iq = np.asarray(received_iq)
        ids = tuple(str(v) for v in physical_sample_ids)
        parents = tuple(_require_sha(v, "parent received IQ") for v in parent_received_iq_sha256)
        overlay_tokens = tuple(str(v) for v in overlay_tokens)
        source_provenance = tuple(
            _require_sha(v, "source LEO provenance")
            for v in source_leo_provenance_sha256
        )
        source_cache = tuple(
            _require_sha(v, "source LEO cache") for v in source_leo_cache_sha256
        )
        views = tuple(str(v) for v in target_channel_views)
        seeds = tuple(int(v) for v in satellite_seeds)
        overlays = tuple(_require_sha(v, "overlay provenance") for v in overlay_provenance_sha256)
        runtime = _require_sha(sealed_runtime_sha256, "sealed runtime")
        checkpoint = _require_sha(sealed_phase1_checkpoint_sha256, "sealed checkpoint")
        feature_code = _require_sha(feature_code_sha256, "feature code")
        if (
            iq.dtype != np.float32
            or iq.ndim != 3
            or iq.shape[1] != 2
            or iq.shape[2] < DCT_RANK + 1
            or not len(iq)
            or not np.isfinite(iq).all()
            or any(
                len(v) != len(iq)
                for v in (
                    ids, parents, overlay_tokens, source_provenance,
                    source_cache, views, seeds, overlays,
                )
            )
            or len(set(ids)) != len(ids)
            or len(set(parents)) != len(parents)
            or len(set(overlay_tokens)) != len(overlay_tokens)
            or any(not value for value in overlay_tokens)
            or len(set(overlays)) != len(overlays)
            or len(set(views)) != 1
            or any(v not in ALLOWED_CHANNEL_VIEWS for v in views)
            or purpose not in ("support", "inference")
        ):
            raise CmraeError("runtime-authorized received-IQ artifact drift")
        computed = tuple(_sha_array(row) for row in iq)
        if computed != parents:
            raise CmraeError("actual received-IQ SHA binding mismatch")
        immutable = np.frombuffer(iq.tobytes(), dtype=np.float32).reshape(iq.shape)
        canonical = {
            "physical_sample_ids": ids,
            "parent_received_iq_sha256": parents,
            "overlay_tokens": overlay_tokens,
            "source_leo_provenance_sha256": source_provenance,
            "source_leo_cache_sha256": source_cache,
            "target_channel_views": views,
            "satellite_seeds": seeds,
            "overlay_provenance_sha256": overlays,
            "sealed_runtime_sha256": runtime,
            "sealed_phase1_checkpoint_sha256": checkpoint,
            "feature_code_sha256": feature_code,
            "operator_id": OPERATOR_ID,
            "purpose": purpose,
        }
        self.received_iq = immutable
        self.physical_sample_ids = ids
        self.parent_received_iq_sha256 = parents
        self.overlay_tokens = overlay_tokens
        self.source_leo_provenance_sha256 = source_provenance
        self.source_leo_cache_sha256 = source_cache
        self.target_channel_views = views
        self.satellite_seeds = seeds
        self.overlay_provenance_sha256 = overlays
        self.sealed_runtime_sha256 = runtime
        self.sealed_phase1_checkpoint_sha256 = checkpoint
        self.feature_code_sha256 = feature_code
        self.operator_id = OPERATOR_ID
        self.purpose = purpose
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


def _build_runtime_authorized_received_iq_artifact_internal(
    received_iq: np.ndarray,
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    overlay_tokens: Sequence[str],
    source_leo_provenance_sha256: Sequence[str],
    source_leo_cache_sha256: Sequence[str],
    target_channel_views: Sequence[str],
    satellite_seeds: Sequence[int],
    overlay_provenance_sha256: Sequence[str],
    sealed_runtime_sha256: str,
    sealed_phase1_checkpoint_sha256: str,
    feature_code_sha256: str,
    purpose: str,
) -> RuntimeAuthorizedReceivedIQArtifact:
    """Runner-internal constructor; each row is checked against its actual IQ SHA."""

    return RuntimeAuthorizedReceivedIQArtifact(
        received_iq=received_iq,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        overlay_tokens=overlay_tokens,
        source_leo_provenance_sha256=source_leo_provenance_sha256,
        source_leo_cache_sha256=source_leo_cache_sha256,
        target_channel_views=target_channel_views,
        satellite_seeds=satellite_seeds,
        overlay_provenance_sha256=overlay_provenance_sha256,
        sealed_runtime_sha256=sealed_runtime_sha256,
        sealed_phase1_checkpoint_sha256=sealed_phase1_checkpoint_sha256,
        feature_code_sha256=feature_code_sha256,
        purpose=purpose,
        _token=_IQ_TOKEN,
    )


class RuntimeAuthorizedBackbone:
    """Token-sealed physical-batch-one feature extractor."""

    def __init__(
        self,
        extractor: Callable[[np.ndarray], np.ndarray],
        *,
        feature_code_sha256: str,
        sealed_phase1_checkpoint_sha256: str,
        _token: object,
    ) -> None:
        if _token is not _BACKBONE_TOKEN or not callable(extractor):
            raise CmraeError("authorized backbone required")
        self._extractor = extractor
        self.feature_code_sha256 = _require_sha(feature_code_sha256, "feature code")
        self.sealed_phase1_checkpoint_sha256 = _require_sha(
            sealed_phase1_checkpoint_sha256, "sealed checkpoint"
        )

    def extract(self, iq: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        for row in iq:
            value = np.asarray(self._extractor(row[None, ...]), dtype=np.float32)
            if value.ndim != 2 or value.shape[0] != 1 or not np.isfinite(value).all():
                raise CmraeError("backbone must return one finite row per physical sample")
            rows.append(value[0])
        result = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        if not result.shape[1]:
            raise CmraeError("empty backbone feature")
        return result


def _seal_runtime_authorized_backbone_internal(
    extractor: Callable[[np.ndarray], np.ndarray],
    *,
    feature_code_sha256: str,
    sealed_phase1_checkpoint_sha256: str,
) -> RuntimeAuthorizedBackbone:
    return RuntimeAuthorizedBackbone(
        extractor,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=sealed_phase1_checkpoint_sha256,
        _token=_BACKBONE_TOKEN,
    )


@dataclass(frozen=True)
class CmraeHyperparameters:
    candidate_id: str
    lambda_equalizer: float
    dct_rank: int = DCT_RANK
    tau: float = TAU
    force_zero: bool = False


def preregistered_candidates() -> tuple[CmraeHyperparameters, ...]:
    return (
        CmraeHyperparameters("D18_Z0", 0.0, force_zero=True),
        CmraeHyperparameters("D18_CMRAE_L0125", 0.125),
        CmraeHyperparameters("D18_CMRAE_L0250", 0.25),
    )


def _validate_hp(hp: CmraeHyperparameters) -> None:
    allowed = {
        ("D18_Z0", 0.0, True),
        ("D18_CMRAE_L0125", 0.125, False),
        ("D18_CMRAE_L0250", 0.25, False),
    }
    if (
        (hp.candidate_id, float(hp.lambda_equalizer), bool(hp.force_zero)) not in allowed
        or hp.dct_rank != DCT_RANK
        or not math.isclose(float(hp.tau), TAU, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise CmraeError("hyperparameter drift")


def _canonical_z0() -> CmraeHyperparameters:
    return preregistered_candidates()[0]


def _dct_basis(length: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)[:, None]
    k = np.arange(1, DCT_RANK + 1, dtype=np.float64)[None, :]
    return np.ascontiguousarray(
        math.sqrt(2.0 / length) * np.cos(math.pi * (n + 0.5) * k / length),
        dtype=np.float64,
    )


def _dct_basis_sha(length: int) -> str:
    return _sha_array(_dct_basis(length))


def _centered_log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    magnitude = np.abs(spectrum)
    row_scale = np.max(magnitude, axis=1, keepdims=True)
    valid = row_scale > 0.0
    relative_floor = np.maximum(row_scale * EPS, np.finfo(np.float64).tiny)
    logged = np.log(np.maximum(magnitude, relative_floor))
    logged -= np.median(logged, axis=1, keepdims=True)
    return np.where(valid, logged, 0.0)


def _bounded_zero_mean_log_gain(
    correction: np.ndarray, hp: CmraeHyperparameters
) -> np.ndarray:
    """Project log-gain onto a zero-mean box with a registered true bound."""

    raw = -float(hp.lambda_equalizer) * np.asarray(correction, dtype=np.float64)
    bound = 2.0 * float(hp.lambda_equalizer) * float(hp.tau)
    if bound == 0.0:
        return np.zeros_like(raw)
    low = float(np.min(raw) - bound)
    high = float(np.max(raw) + bound)
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        value = float(np.mean(np.clip(raw - midpoint, -bound, bound)))
        if value > 0.0:
            low = midpoint
        else:
            high = midpoint
    result = np.clip(raw - 0.5 * (low + high), -bound, bound)
    if abs(float(np.mean(result))) > 1e-12:
        raise CmraeError("bounded geometric gain normalization failed")
    return result


def _complex_iq(iq: np.ndarray) -> np.ndarray:
    return iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)


def _restore_row_rms(source: np.ndarray, output: np.ndarray) -> np.ndarray:
    source_power = np.mean(np.abs(source) ** 2, axis=1, keepdims=True)
    output_power = np.mean(np.abs(output) ** 2, axis=1, keepdims=True)
    scale = np.sqrt(
        np.divide(
            source_power,
            np.maximum(output_power, np.finfo(np.float64).tiny),
            out=np.ones_like(source_power),
            where=source_power > 0.0,
        )
    )
    restored = output * scale
    return np.where(source_power > 0.0, restored, source)


def _fit_common_coefficients(iq: np.ndarray, labels: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(_complex_iq(iq), axis=1), axes=1)
    logmag = _centered_log_magnitude(spectrum)
    sample_coeff = logmag @ _dct_basis(iq.shape[2])
    classes = tuple(sorted(np.unique(labels).tolist()))
    class_medians = np.stack([
        np.median(sample_coeff[labels == value], axis=0) for value in classes
    ])
    return np.ascontiguousarray(np.median(class_medians, axis=0), dtype=np.float32)


def _apply_equalizer_iq(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    coefficients: np.ndarray,
    hp: CmraeHyperparameters,
) -> np.ndarray:
    if not isinstance(artifact, RuntimeAuthorizedReceivedIQArtifact):
        raise CmraeError("runtime-authorized received-IQ artifact required")
    _validate_hp(hp)
    coeff = np.asarray(coefficients, dtype=np.float32)
    if coeff.shape != (DCT_RANK,) or not np.isfinite(coeff).all():
        raise CmraeError("equalizer coefficient drift")
    if hp.force_zero:
        if np.any(coeff != 0):
            raise CmraeError("Z0 coefficient drift")
        return artifact.received_iq
    rows = _complex_iq(artifact.received_iq)
    spectrum = np.fft.fftshift(np.fft.fft(rows, axis=1), axes=1)
    envelope = _dct_basis(rows.shape[1]) @ coeff.astype(np.float64)
    correction = np.clip(envelope, -hp.tau, hp.tau)
    log_gain = _bounded_zero_mean_log_gain(correction, hp)
    gain = np.exp(log_gain)
    adjusted = spectrum * gain[None, :]
    output = np.fft.ifft(np.fft.ifftshift(adjusted, axes=1), axis=1)
    output = _restore_row_rms(rows, output)
    return np.ascontiguousarray(
        np.stack([output.real, output.imag], axis=1), dtype=np.float32
    )


def _normalize(rows: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.ascontiguousarray(rows / np.maximum(denom, EPS), dtype=np.float32)


def _prototype_scores(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Column-independent scoring keeps old columns bitwise append-invariant."""

    return np.ascontiguousarray(
        np.stack([
            np.einsum("nd,d->n", features, prototype, optimize=False)
            for prototype in prototypes
        ], axis=1),
        dtype=np.float32,
    )


def _binding_check(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    backbone: RuntimeAuthorizedBackbone,
) -> None:
    if (
        not isinstance(artifact, RuntimeAuthorizedReceivedIQArtifact)
        or not isinstance(backbone, RuntimeAuthorizedBackbone)
        or artifact.feature_code_sha256 != backbone.feature_code_sha256
        or artifact.sealed_phase1_checkpoint_sha256
        != backbone.sealed_phase1_checkpoint_sha256
    ):
        raise CmraeError("runtime/backbone binding drift")


def _features(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    coefficients: np.ndarray,
    hp: CmraeHyperparameters,
    backbone: RuntimeAuthorizedBackbone,
) -> np.ndarray:
    _binding_check(artifact, backbone)
    return _normalize(backbone.extract(_apply_equalizer_iq(artifact, coefficients, hp)))


def _validate_support(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not isinstance(artifact, RuntimeAuthorizedReceivedIQArtifact) or artifact.purpose != "support":
        raise CmraeError("support artifact required")
    labels = np.asarray(labels).astype(str)
    ranks = np.asarray(ranks, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if (
        len(labels) != len(artifact.received_iq)
        or len(ranks) != len(labels)
        or int(k_shot) not in (1, 5, 10)
        or len(classes) < 2
        or set(counts.tolist()) != {int(k_shot)}
        or any(set(ranks[labels == c].tolist()) != set(range(int(k_shot))) for c in classes)
    ):
        raise CmraeError("strict physical K-shot drift")
    return labels, ranks, tuple(sorted(classes.tolist()))


def _validate_old_reuse(
    before: RuntimeAuthorizedReceivedIQArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after: RuntimeAuthorizedReceivedIQArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
    def keyed(artifact, labels, ranks):
        allowed = set(old_classes)
        return {
            (str(labels[i]), int(ranks[i])): (
                artifact.physical_sample_ids[i],
                artifact.parent_received_iq_sha256[i],
                artifact.overlay_tokens[i],
                artifact.source_leo_provenance_sha256[i],
                artifact.source_leo_cache_sha256[i],
                artifact.satellite_seeds[i],
                artifact.overlay_provenance_sha256[i],
            )
            for i in range(len(labels)) if str(labels[i]) in allowed
        }
    if (
        before.sealed_runtime_sha256 != after.sealed_runtime_sha256
        or before.feature_code_sha256 != after.feature_code_sha256
        or before.sealed_phase1_checkpoint_sha256 != after.sealed_phase1_checkpoint_sha256
        or before.target_channel_views[0] != after.target_channel_views[0]
        or keyed(before, before_labels, before_ranks)
        != keyed(after, after_labels, after_ranks)
    ):
        raise CmraeError("old exact-reuse or runtime binding drift")


@dataclass(frozen=True)
class CmraeState:
    schema: str
    classes: tuple[str, ...]
    common_dct_coefficients: np.ndarray
    prototypes: np.ndarray
    hyperparameters: CmraeHyperparameters
    feature_dim: int
    iq_length: int
    k_shot: int
    old_class_count: int
    registration_generation: int
    target_channel_view: str
    sealed_runtime_sha256: str
    sealed_phase1_checkpoint_sha256: str
    feature_code_sha256: str
    support_artifact_sha256: str
    support_selection_sha256: str
    operator_id: str
    k10_lock_certificate_sha256: str
    selection_authority_anchor_sha256: str
    locked_k10_candidate_id: str
    authority_scope: str
    resource: Mapping[str, Any]
    state_content_sha256: str = ""

    def __post_init__(self) -> None:
        coeff = np.ascontiguousarray(self.common_dct_coefficients, dtype=np.float32)
        proto = np.ascontiguousarray(self.prototypes, dtype=np.float32)
        object.__setattr__(self, "common_dct_coefficients", np.frombuffer(coeff.tobytes(), dtype=np.float32))
        object.__setattr__(
            self, "prototypes",
            np.frombuffer(proto.tobytes(), dtype=np.float32).reshape(proto.shape),
        )
        computed = _state_sha(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise CmraeError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterCmraeFit:
    before_state: CmraeState
    after_state: CmraeState
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CmraeSceneSupport:
    scene_id: str
    before_artifact: RuntimeAuthorizedReceivedIQArtifact
    before_labels: Sequence[str] | np.ndarray
    before_ranks: Sequence[int] | np.ndarray
    after_artifact: RuntimeAuthorizedReceivedIQArtifact
    after_labels: Sequence[str] | np.ndarray
    after_ranks: Sequence[int] | np.ndarray


@dataclass(frozen=True)
class CmraeThreeSceneSelection:
    selected_hyperparameters: CmraeHyperparameters
    k10_lock_certificate: CmraeK10LockCertificate
    fitted_scenes: tuple[BeforeAfterCmraeFit, ...]
    evaluations: tuple[Mapping[str, Any], ...]
    trace: tuple[dict[str, Any], ...]


def _support_records(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
) -> tuple[tuple[str, int, str, str, str, str, str, str, int], ...]:
    labels = np.asarray(labels).astype(str)
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(labels) != len(artifact.received_iq) or len(ranks) != len(labels):
        raise CmraeError("support record alignment drift")
    return tuple(sorted(
        (
            str(labels[i]), int(ranks[i]), artifact.physical_sample_ids[i],
            artifact.parent_received_iq_sha256[i], artifact.overlay_tokens[i],
            artifact.overlay_provenance_sha256[i],
            artifact.source_leo_provenance_sha256[i],
            artifact.source_leo_cache_sha256[i], artifact.satellite_seeds[i],
        )
        for i in range(len(labels))
    ))


def _k10_certificate_sha(certificate: CmraeK10LockCertificate) -> str:
    payload = {
        "schema": certificate.schema,
        "selected_candidate_id": certificate.selected_candidate_id,
        "scene_prefix_locks": certificate.scene_prefix_locks,
        "selection_authority_anchor_sha256": (
            certificate.selection_authority_anchor_sha256
        ),
        "k10_selection_authority_sha256": certificate.k10_selection_authority_sha256,
        "authority_scope": certificate.authority_scope,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _k10_selection_authority_sha(
    selected_candidate_id: str,
    scene_prefix_locks: tuple[tuple[Any, ...], ...],
    selection_authority_anchor_sha256: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "cvs.phase2.cmrae_k10_selection_authority.v1",
                "selected_candidate_id": selected_candidate_id,
                "scene_prefix_locks": scene_prefix_locks,
                "selection_authority_anchor_sha256": (
                    selection_authority_anchor_sha256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _record_classes_exact_k10(
    records: tuple[tuple[str, int, str, str, str, str, str, str, int], ...]
) -> tuple[str, ...]:
    labels = np.asarray([row[0] for row in records])
    classes = tuple(sorted(np.unique(labels).tolist()))
    if (
        len(set(records)) != len(records)
        or any(
            set(row[1] for row in records if row[0] == value) != set(range(10))
            for value in classes
        )
    ):
        raise CmraeError("K10 lock certificate strict support drift")
    return classes


def _records_sha(
    records: tuple[tuple[str, int, str, str, str, str, str, str, int], ...],
    k_shot: int,
) -> tuple[str, int]:
    prefix = tuple(row for row in records if row[1] < int(k_shot))
    return (
        hashlib.sha256(
            json.dumps(prefix, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        len(prefix),
    )


def _validate_k10_certificate(certificate: CmraeK10LockCertificate) -> None:
    if not isinstance(certificate, CmraeK10LockCertificate):
        raise CmraeError("K10 lock certificate drift")
    _require_sha(
        certificate.selection_authority_anchor_sha256,
        "selection authority anchor",
    )
    if (
        certificate.schema != "cvs.phase2.cmrae_k10_lock.v1"
        or certificate.authority_scope != "development_diagnostic_only"
        or certificate.selected_candidate_id not in {
            hp.candidate_id for hp in preregistered_candidates()
        }
        or {row[0] for row in certificate.scene_prefix_locks}
        != set(ALLOWED_CHANNEL_VIEWS)
        or len(certificate.scene_prefix_locks) != 3
        or certificate.k10_selection_authority_sha256
        != _k10_selection_authority_sha(
            certificate.selected_candidate_id,
            certificate.scene_prefix_locks,
            certificate.selection_authority_anchor_sha256,
        )
        or certificate.certificate_sha256 != _k10_certificate_sha(certificate)
    ):
        raise CmraeError("K10 lock certificate drift")
    for row in certificate.scene_prefix_locks:
        if len(row) != 13:
            raise CmraeError("K10 lock certificate prefix schema drift")
        (_scene, b1s, b1n, b5s, b5n, b10s, b10n,
         a1s, a1n, a5s, a5n, a10s, a10n) = row
        for value in (b1s, b5s, b10s, a1s, a5s, a10s):
            _require_sha(value, "K10 prefix selection")
        if (
            any(not isinstance(v, int) or v <= 0 for v in (b1n, b5n, b10n, a1n, a5n, a10n))
            or b5n != 5 * b1n or b10n != 10 * b1n
            or a5n != 5 * a1n or a10n != 10 * a1n
            or not b1n < a1n
        ):
            raise CmraeError("K10 lock certificate prefix count drift")


def _build_k10_prefix_locks(
    scenes: Sequence[CmraeSceneSupport],
) -> tuple[tuple[Any, ...], ...]:
    locks = []
    for scene in scenes:
        before = _support_records(
            scene.before_artifact, scene.before_labels, scene.before_ranks
        )
        after = _support_records(
            scene.after_artifact, scene.after_labels, scene.after_ranks
        )
        old_classes = _record_classes_exact_k10(before)
        all_classes = _record_classes_exact_k10(after)
        if not set(old_classes) < set(all_classes) or not set(before).issubset(set(after)):
            raise CmraeError("K10 lock certificate old/new support drift")
        b1s, b1n = _records_sha(before, 1)
        b5s, b5n = _records_sha(before, 5)
        b10s, b10n = _records_sha(before, 10)
        a1s, a1n = _records_sha(after, 1)
        a5s, a5n = _records_sha(after, 5)
        a10s, a10n = _records_sha(after, 10)
        locks.append((
            scene.scene_id,
            b1s, b1n, b5s, b5n, b10s, b10n,
            a1s, a1n, a5s, a5n, a10s, a10n,
        ))
    return tuple(sorted(locks))


def _validate_k10_prefix_lock(
    certificate: CmraeK10LockCertificate,
    hp: CmraeHyperparameters,
    k_shot: int,
    before_artifact: RuntimeAuthorizedReceivedIQArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedReceivedIQArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    *,
    expected_selection_authority_anchor_sha256: str,
) -> None:
    _validate_k10_certificate(certificate)
    if hp.candidate_id != certificate.selected_candidate_id:
        raise CmraeError("K10 selected candidate lock mismatch")
    if (
        certificate.selection_authority_anchor_sha256
        != expected_selection_authority_anchor_sha256
    ):
        raise CmraeError("selection authority anchor mismatch")
    matches = [row for row in certificate.scene_prefix_locks if row[0] == before_artifact.target_channel_views[0]]
    if len(matches) != 1:
        raise CmraeError("K10 lock certificate scene binding drift")
    row = matches[0]
    if int(k_shot) == 1:
        expected_before_sha, expected_before_count = row[1], row[2]
        expected_after_sha, expected_after_count = row[7], row[8]
    elif int(k_shot) == 5:
        expected_before_sha, expected_before_count = row[3], row[4]
        expected_after_sha, expected_after_count = row[9], row[10]
    elif int(k_shot) == 10:
        expected_before_sha, expected_before_count = row[5], row[6]
        expected_after_sha, expected_after_count = row[11], row[12]
    else:
        raise CmraeError("K10 lock unsupported K")
    actual_before = _support_records(before_artifact, before_labels, before_ranks)
    actual_after = _support_records(after_artifact, after_labels, after_ranks)
    actual_before_sha, actual_before_count = _records_sha(actual_before, int(k_shot))
    actual_after_sha, actual_after_count = _records_sha(actual_after, int(k_shot))
    if (
        actual_before_sha != expected_before_sha
        or actual_before_count != expected_before_count
        or actual_after_sha != expected_after_sha
        or actual_after_count != expected_after_count
    ):
        raise CmraeError("K10 lock strict nested support prefix mismatch")


def _selection_sha(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    classes: Sequence[str],
) -> str:
    rows = [
        (
            str(labels[i]), int(ranks[i]), artifact.physical_sample_ids[i],
            artifact.parent_received_iq_sha256[i], artifact.overlay_tokens[i],
            artifact.overlay_provenance_sha256[i],
            artifact.source_leo_provenance_sha256[i],
            artifact.source_leo_cache_sha256[i], artifact.satellite_seeds[i],
        )
        for i in range(len(labels)) if str(labels[i]) in set(classes)
    ]
    return hashlib.sha256(json.dumps(sorted(rows), separators=(",", ":")).encode("utf-8")).hexdigest()


def _expected_resource(state: CmraeState) -> dict[str, Any]:
    coeff_bytes = int(state.common_dct_coefficients.nbytes)
    prototype_bytes = int(state.prototypes.nbytes)
    n = int(state.iq_length)
    fft_complex_ops = int(10 * n * math.ceil(math.log2(n)))
    enrollment_unique = int(len(state.classes) * state.k_shot)
    return {
        "schema": "cvs.phase2.cmrae_resource.v1",
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "dense_query_graph": False,
        "backbone_forwards_per_physical_sample": 1,
        "enrollment_unique_physical_support_count": enrollment_unique,
        "enrollment_backbone_forwards": enrollment_unique,
        "enrollment_repeated_old_support_backbone_forwards": 0,
        "outer_l2o_repeated_forwards_are_development_selection_cost": True,
        "outer_l2o_cost_is_deployment_resource_evidence": False,
        "post_reception_views_per_physical_sample": 1,
        "view_counts_as_additional_physical_sample": False,
        "additional_leo_channel_states": 0,
        "fft_forward_transforms_per_sample": 1 if not state.hyperparameters.force_zero else 0,
        "ifft_inverse_transforms_per_sample": 1 if not state.hyperparameters.force_zero else 0,
        "estimated_fft_complex_ops_per_query": fft_complex_ops if not state.hyperparameters.force_zero else 0,
        "dct_reconstruction_macs_per_query": DCT_RANK * n if not state.hyperparameters.force_zero else 0,
        "prototype_mac_per_query": len(state.classes) * state.feature_dim,
        "prototype_scorer": "class_column_independent_einsum_optimize_false",
        "prototype_scorer_passes_per_query": 1,
        "non_mac_exp_ops_per_query": n if not state.hyperparameters.force_zero else 0,
        "row_rms_restore_macs_per_query": 4 * n if not state.hyperparameters.force_zero else 0,
        "output_row_rms_matches_input": True,
        "cmrae_adapter_state_bytes": coeff_bytes,
        "registered_prototype_state_bytes": prototype_bytes,
        "persistent_array_state_bytes": coeff_bytes + prototype_bytes,
        "adapter_state_limit_bytes": MAX_ADAPTER_STATE_BYTES,
        "cmrae_adapter_estimated_serialized_state_bytes": coeff_bytes + 2048,
        "estimated_full_state_bytes_uncompressed": coeff_bytes + prototype_bytes + 4096,
        "full_serialized_state_limit_bytes": MAX_FULL_SERIALIZED_STATE_BYTES,
        "fftshift_convention": "numpy_fftshift_before_dct_then_ifftshift",
        "dct_convention": "orthonormal_dct_ii_non_dc_k1_to_k8",
        "dct_basis_sha256": _dct_basis_sha(n),
        "registered_post_rms_log_gain_absolute_bound": (
            4.0 * state.hyperparameters.lambda_equalizer * state.hyperparameters.tau
        ),
        "pre_rms_gain_geometric_mean_normalized": True,
        "fft_bin_phase_unrotated": True,
        "cfo_estimation": False,
        "cfo_derotation": False,
        "frequency_bin_shift": False,
    }


def _state_payload(state: CmraeState) -> dict[str, Any]:
    return {
        "schema": state.schema,
        "classes": state.classes,
        "coeff_sha": _sha_array(state.common_dct_coefficients),
        "prototype_sha": _sha_array(state.prototypes),
        "hyperparameters": state.hyperparameters.__dict__,
        "feature_dim": state.feature_dim,
        "iq_length": state.iq_length,
        "k_shot": state.k_shot,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "target_channel_view": state.target_channel_view,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "support_artifact_sha256": state.support_artifact_sha256,
        "support_selection_sha256": state.support_selection_sha256,
        "operator_id": state.operator_id,
        "k10_lock_certificate_sha256": state.k10_lock_certificate_sha256,
        "selection_authority_anchor_sha256": (
            state.selection_authority_anchor_sha256
        ),
        "locked_k10_candidate_id": state.locked_k10_candidate_id,
        "authority_scope": state.authority_scope,
        "resource": dict(state.resource),
    }


def _state_sha(state: CmraeState) -> str:
    return hashlib.sha256(
        json.dumps(_state_payload(state), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_state(state: CmraeState) -> None:
    _validate_hp(state.hyperparameters)
    expected = _expected_resource(state)
    norms = np.linalg.norm(state.prototypes, axis=1)
    sha_fields = (
        state.sealed_runtime_sha256,
        state.sealed_phase1_checkpoint_sha256,
        state.feature_code_sha256,
        state.support_artifact_sha256,
        state.support_selection_sha256,
        state.k10_lock_certificate_sha256,
        state.selection_authority_anchor_sha256,
    )
    if (
        state.schema != SCHEMA
        or len(state.classes) < 2
        or len(set(state.classes)) != len(state.classes)
        or state.common_dct_coefficients.shape != (DCT_RANK,)
        or state.prototypes.shape != (len(state.classes), state.feature_dim)
        or not np.isfinite(state.common_dct_coefficients).all()
        or not np.isfinite(state.prototypes).all()
        or not np.allclose(norms, 1.0, atol=2e-5)
        or state.iq_length < DCT_RANK + 1
        or state.k_shot not in (1, 5, 10)
        or not 1 < state.old_class_count <= len(state.classes)
        or state.registration_generation not in (0, 1)
        or (state.registration_generation == 0 and state.old_class_count != len(state.classes))
        or (state.registration_generation == 1 and state.old_class_count >= len(state.classes))
        or state.target_channel_view not in ALLOWED_CHANNEL_VIEWS
        or state.operator_id != OPERATOR_ID
        or state.locked_k10_candidate_id not in {
            hp.candidate_id for hp in preregistered_candidates()
        }
        or state.authority_scope != "development_diagnostic_only"
        or any(len(v) != 64 or any(ch not in "0123456789abcdef" for ch in v) for v in sha_fields)
        or dict(state.resource) != expected
        or expected["cmrae_adapter_state_bytes"] >= MAX_ADAPTER_STATE_BYTES
        or expected["cmrae_adapter_estimated_serialized_state_bytes"] >= MAX_ADAPTER_STATE_BYTES
        or (state.hyperparameters.force_zero and np.any(state.common_dct_coefficients != 0))
        or state.state_content_sha256 != _state_sha(state)
    ):
        raise CmraeError("state drift")


def _make_state(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    classes: tuple[str, ...],
    coefficients: np.ndarray,
    prototypes: np.ndarray,
    hp: CmraeHyperparameters,
    k_shot: int,
    old_class_count: int,
    generation: int,
    *,
    k10_lock_certificate_sha256: str,
    selection_authority_anchor_sha256: str,
    locked_k10_candidate_id: str,
) -> CmraeState:
    blank = {
        "schema": SCHEMA,
        "classes": classes,
        "common_dct_coefficients": coefficients,
        "prototypes": prototypes,
        "hyperparameters": hp,
        "feature_dim": int(prototypes.shape[1]),
        "iq_length": int(artifact.received_iq.shape[2]),
        "k_shot": int(k_shot),
        "old_class_count": int(old_class_count),
        "registration_generation": int(generation),
        "target_channel_view": artifact.target_channel_views[0],
        "sealed_runtime_sha256": artifact.sealed_runtime_sha256,
        "sealed_phase1_checkpoint_sha256": artifact.sealed_phase1_checkpoint_sha256,
        "feature_code_sha256": artifact.feature_code_sha256,
        "support_artifact_sha256": artifact.artifact_sha256,
        "support_selection_sha256": _selection_sha(artifact, labels, ranks, classes),
        "operator_id": OPERATOR_ID,
        "k10_lock_certificate_sha256": k10_lock_certificate_sha256,
        "selection_authority_anchor_sha256": (
            selection_authority_anchor_sha256
        ),
        "locked_k10_candidate_id": locked_k10_candidate_id,
        "authority_scope": "development_diagnostic_only",
        "resource": {},
    }
    provisional = object.__new__(CmraeState)
    for key, value in blank.items():
        object.__setattr__(provisional, key, value)
    object.__setattr__(provisional, "state_content_sha256", "")
    blank["resource"] = _expected_resource(provisional)
    return CmraeState(**blank)


def fit_before_after_locked(
    before_artifact: RuntimeAuthorizedReceivedIQArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedReceivedIQArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    hyperparameters: CmraeHyperparameters,
    backbone: RuntimeAuthorizedBackbone,
    k10_lock_certificate: CmraeK10LockCertificate | None = None,
    expected_selection_authority_anchor_sha256: str | None = None,
) -> BeforeAfterCmraeFit:
    if int(k_shot) in (2, 3, 4):
        raise CmraeError("K2-K4 are closed")
    before_labels, before_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot
    )
    after_labels, after_ranks, all_classes_sorted = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot
    )
    new_classes = tuple(v for v in all_classes_sorted if v not in set(old_classes))
    if not new_classes:
        raise CmraeError("new-class registration support required")
    classes = old_classes + new_classes
    _validate_old_reuse(
        before_artifact, before_labels, before_ranks,
        after_artifact, after_labels, after_ranks, old_classes,
    )
    _binding_check(before_artifact, backbone)
    _binding_check(after_artifact, backbone)
    _validate_hp(hyperparameters)
    if int(k_shot) < 10 and k10_lock_certificate is None:
        raise CmraeError("K10 lock certificate required for K1/K5")
    if k10_lock_certificate is not None:
        if expected_selection_authority_anchor_sha256 is None:
            raise CmraeError("expected selection authority anchor required")
        expected_anchor = _require_sha(
            expected_selection_authority_anchor_sha256,
            "expected selection authority anchor",
        )
        _validate_k10_prefix_lock(
            k10_lock_certificate, hyperparameters, int(k_shot),
            before_artifact, before_labels, before_ranks,
            after_artifact, after_labels, after_ranks,
            expected_selection_authority_anchor_sha256=expected_anchor,
        )
        lock_sha = k10_lock_certificate.certificate_sha256
        selection_anchor_sha = expected_anchor
        locked_candidate_id = k10_lock_certificate.selected_candidate_id
    else:
        lock_sha = "0" * 64
        selection_anchor_sha = "0" * 64
        locked_candidate_id = hyperparameters.candidate_id
    hp = _canonical_z0() if int(k_shot) == 1 else hyperparameters
    coefficients = (
        np.zeros(DCT_RANK, dtype=np.float32)
        if hp.force_zero
        else _fit_common_coefficients(before_artifact.received_iq, before_labels)
    )
    before_features = _features(before_artifact, coefficients, hp, backbone)
    old_prototypes = _normalize(np.stack([
        np.mean(before_features[before_labels == value], axis=0) for value in old_classes
    ]))
    before_state = _make_state(
        before_artifact, before_labels, before_ranks, old_classes,
        coefficients, old_prototypes, hp, k_shot, len(old_classes), 0,
        k10_lock_certificate_sha256=lock_sha,
        selection_authority_anchor_sha256=selection_anchor_sha,
        locked_k10_candidate_id=locked_candidate_id,
    )
    new_mask = np.isin(after_labels, new_classes)
    after_new_features = _subset_features(
        after_artifact, new_mask, coefficients, hp, backbone
    )
    after_new_labels = after_labels[new_mask]
    new_prototypes = _normalize(np.stack([
        np.mean(after_new_features[after_new_labels == value], axis=0)
        for value in new_classes
    ]))
    after_prototypes = np.concatenate([old_prototypes, new_prototypes], axis=0)
    after_state = _make_state(
        after_artifact, after_labels, after_ranks, classes,
        coefficients, after_prototypes, hp, k_shot, len(old_classes), 1,
        k10_lock_certificate_sha256=lock_sha,
        selection_authority_anchor_sha256=selection_anchor_sha,
        locked_k10_candidate_id=locked_candidate_id,
    )
    if (
        not np.array_equal(before_state.common_dct_coefficients, after_state.common_dct_coefficients)
        or not np.array_equal(before_state.prototypes, after_state.prototypes[:len(old_classes)])
    ):
        raise CmraeError("After equalizer or old prototype lock failed")
    trace = ({
        "phase": "cmrae_support_fit",
        "candidate_id": hp.candidate_id,
        "k_shot": int(k_shot),
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "equalizer_old_only_fit": True,
        "after_equalizer_bitwise_locked": True,
        "after_old_prototypes_bitwise_locked": True,
        "after_old_score_columns_bitwise_append_locked": True,
        "enrollment_unique_physical_support_count": len(after_artifact.received_iq),
        "enrollment_backbone_forwards": len(after_artifact.received_iq),
        "after_old_support_backbone_recomputed": False,
        "after_new_only_backbone_forwards": int(np.sum(new_mask)),
        "single_fixed_received_iq_view": True,
        "query_fit_access": False,
        "k10_lock_certificate_sha256": lock_sha,
        "selection_authority_anchor_sha256": selection_anchor_sha,
        "locked_k10_candidate_id": locked_candidate_id,
    },)
    return BeforeAfterCmraeFit(before_state, after_state, trace)


def _subset_features(
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    mask: np.ndarray,
    coefficients: np.ndarray,
    hp: CmraeHyperparameters,
    backbone: RuntimeAuthorizedBackbone,
) -> np.ndarray:
    # Internal-only view of already authorized rows; no new physical observation.
    iq = artifact.received_iq[mask]
    if hp.force_zero:
        transformed = iq
    else:
        rows = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
        spectrum = np.fft.fftshift(np.fft.fft(rows, axis=1), axes=1)
        envelope = _dct_basis(rows.shape[1]) @ coefficients.astype(np.float64)
        log_gain = _bounded_zero_mean_log_gain(
            np.clip(envelope, -hp.tau, hp.tau), hp
        )
        gain = np.exp(log_gain)
        output = np.fft.ifft(
            np.fft.ifftshift(spectrum * gain[None, :], axes=1), axis=1
        )
        output = _restore_row_rms(rows, output)
        transformed = np.ascontiguousarray(np.stack([output.real, output.imag], axis=1), dtype=np.float32)
    return _normalize(backbone.extract(transformed))


def _per_class_accuracy(scores: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    predicted = np.asarray(classes)[np.argmax(scores, axis=1)]
    return {
        value: float(np.mean(predicted[labels == value] == value))
        for value in classes if np.any(labels == value)
    }


def _harmonic(old: float, new: float) -> float:
    return float(2.0 * old * new / max(old + new, EPS))


def _fold_metrics(
    before_scores: np.ndarray,
    after_old_scores: np.ndarray,
    after_new_scores: np.ndarray,
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    old_classes: tuple[str, ...],
    all_classes: tuple[str, ...],
) -> dict[str, Any]:
    before_pc = _per_class_accuracy(before_scores, old_labels, old_classes)
    after_old_pc = _per_class_accuracy(after_old_scores, old_labels, all_classes)
    new_pc = _per_class_accuracy(after_new_scores, new_labels, all_classes)
    before_old = float(np.mean(list(before_pc.values())))
    after_old = float(np.mean(list(after_old_pc.values())))
    seen_new = float(np.mean(list(new_pc.values())))
    return {
        "before_old_per_class": before_pc,
        "after_old_per_class": after_old_pc,
        "seen_new_per_class": new_pc,
        "before_old": before_old,
        "after_old": after_old,
        "seen_new": seen_new,
        "before_old_floor": min(before_pc.values()),
        "after_old_floor": min(after_old_pc.values()),
        "seen_new_floor": min(new_pc.values()),
        "joint": 0.5 * (after_old + seen_new),
        "H_old_new": _harmonic(after_old, seen_new),
        "forgetting": before_old - after_old,
    }


def _outer_fold(
    before_artifact: RuntimeAuthorizedReceivedIQArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedReceivedIQArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    hp: CmraeHyperparameters,
    fold: int,
    backbone: RuntimeAuthorizedBackbone,
) -> dict[str, Any]:
    held_ranks = (2 * fold, 2 * fold + 1)
    before_held = np.isin(before_ranks, held_ranks)
    before_train = ~before_held
    after_held = np.isin(after_ranks, held_ranks)
    after_train = ~after_held
    coefficients = (
        np.zeros(DCT_RANK, dtype=np.float32)
        if hp.force_zero else
        _fit_common_coefficients(before_artifact.received_iq[before_train], before_labels[before_train])
    )
    old_train_features = _subset_features(before_artifact, before_train, coefficients, hp, backbone)
    old_proto = _normalize(np.stack([
        np.mean(old_train_features[before_labels[before_train] == value], axis=0)
        for value in old_classes
    ]))
    after_train_features = _subset_features(after_artifact, after_train, coefficients, hp, backbone)
    new_proto = _normalize(np.stack([
        np.mean(after_train_features[after_labels[after_train] == value], axis=0)
        for value in new_classes
    ]))
    all_classes = old_classes + new_classes
    all_proto = np.concatenate([old_proto, new_proto], axis=0)
    held_old_features = _subset_features(before_artifact, before_held, coefficients, hp, backbone)
    held_joint_features = _subset_features(after_artifact, after_held, coefficients, hp, backbone)
    held_joint_labels = after_labels[after_held]
    held_old_joint = np.isin(held_joint_labels, old_classes)
    held_new_joint = np.isin(held_joint_labels, new_classes)
    metrics = _fold_metrics(
        _prototype_scores(held_old_features, old_proto),
        _prototype_scores(held_joint_features[held_old_joint], all_proto),
        _prototype_scores(held_joint_features[held_new_joint], all_proto),
        held_joint_labels[held_old_joint],
        held_joint_labels[held_new_joint],
        old_classes,
        all_classes,
    )
    train_selection_rows = tuple(sorted(
        [
            (
                "before", str(before_labels[i]), int(before_ranks[i]),
                before_artifact.physical_sample_ids[i],
                before_artifact.parent_received_iq_sha256[i],
                before_artifact.overlay_tokens[i],
                before_artifact.source_leo_provenance_sha256[i],
                before_artifact.source_leo_cache_sha256[i],
                before_artifact.satellite_seeds[i],
                before_artifact.overlay_provenance_sha256[i],
            )
            for i in np.flatnonzero(before_train)
        ]
        + [
            (
                "after", str(after_labels[i]), int(after_ranks[i]),
                after_artifact.physical_sample_ids[i],
                after_artifact.parent_received_iq_sha256[i],
                after_artifact.overlay_tokens[i],
                after_artifact.source_leo_provenance_sha256[i],
                after_artifact.source_leo_cache_sha256[i],
                after_artifact.satellite_seeds[i],
                after_artifact.overlay_provenance_sha256[i],
            )
            for i in np.flatnonzero(after_train)
        ]
    ))
    train_selection_sha = hashlib.sha256(
        json.dumps(train_selection_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "fold": fold,
        "held_ranks": list(held_ranks),
        "train_rows_per_class": 8,
        "candidate_id": hp.candidate_id,
        "development_selection_repeats_backbone_forwards": True,
        "deployment_resource_evidence": False,
        "equalizer_train_old_only": True,
        "common_coefficients_sha256": _sha_array(coefficients),
        "old_prototypes_sha256": _sha_array(old_proto),
        "new_prototypes_sha256": _sha_array(new_proto),
        "outer_train_support_selection_sha256": train_selection_sha,
        "outer_train_state_sha256": hashlib.sha256(
            json.dumps(
                {
                    "candidate_id": hp.candidate_id,
                    "coeff": _sha_array(coefficients),
                    "old_proto": _sha_array(old_proto),
                    "new_proto": _sha_array(new_proto),
                    "train_support_selection_sha256": train_selection_sha,
                    "held_ranks": list(held_ranks),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        **metrics,
    }


def evaluate_k10_outer_l2o(
    before_artifact: RuntimeAuthorizedReceivedIQArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedReceivedIQArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: CmraeHyperparameters,
    backbone: RuntimeAuthorizedBackbone,
    scene_id: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    before_labels, before_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, 10
    )
    after_labels, after_ranks, sorted_all = _validate_support(
        after_artifact, after_labels, after_ranks, 10
    )
    new_classes = tuple(v for v in sorted_all if v not in set(old_classes))
    if not new_classes:
        raise CmraeError("new-class registration support required")
    _validate_old_reuse(
        before_artifact, before_labels, before_ranks,
        after_artifact, after_labels, after_ranks, old_classes,
    )
    _validate_hp(hyperparameters)
    _binding_check(before_artifact, backbone)
    folds = tuple(
        _outer_fold(
            before_artifact, before_labels, before_ranks,
            after_artifact, after_labels, after_ranks,
            old_classes, new_classes, hyperparameters, fold, backbone,
        )
        for fold in range(5)
    )
    trace = tuple({"phase": "cmrae_outer_l2o_fold", "scene_id": scene_id, **row} for row in folds)
    return {
        "candidate_id": hyperparameters.candidate_id,
        "scene_id": str(scene_id),
        "folds": folds,
        "mean_H_old_new": float(np.mean([r["H_old_new"] for r in folds])),
        "mean_joint": float(np.mean([r["joint"] for r in folds])),
        "worst_after_old_floor": min(r["after_old_floor"] for r in folds),
        "worst_seen_new_floor": min(r["seen_new_floor"] for r in folds),
        "max_forgetting": max(r["forgetting"] for r in folds),
        "development_selection_repeats_backbone_forwards": True,
        "deployment_resource_evidence": False,
        "aggregate_before_old_per_class": {
            key: float(np.mean([r["before_old_per_class"][key] for r in folds]))
            for key in old_classes
        },
        "aggregate_after_old_per_class": {
            key: float(np.mean([r["after_old_per_class"][key] for r in folds]))
            for key in old_classes
        },
        "aggregate_seen_new_per_class": {
            key: float(np.mean([r["seen_new_per_class"][key] for r in folds]))
            for key in new_classes
        },
    }, trace


def _candidate_gate(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[bool, tuple[dict[str, Any], ...]]:
    evidence: list[dict[str, Any]] = []
    passed = True
    for row, base in zip(candidate["folds"], baseline["folds"]):
        before_ok = all(row["before_old_per_class"][k] >= base["before_old_per_class"][k] for k in base["before_old_per_class"])
        after_ok = all(row["after_old_per_class"][k] >= base["after_old_per_class"][k] for k in base["after_old_per_class"])
        new_ok = all(row["seen_new_per_class"][k] >= base["seen_new_per_class"][k] for k in base["seen_new_per_class"])
        old_floor_ok = row["after_old_floor"] >= base["after_old_floor"]
        new_floor_ok = row["seen_new_floor"] >= base["seen_new_floor"]
        fold_pass = (
            before_ok and after_ok and new_ok and old_floor_ok and new_floor_ok
            and row["forgetting"] <= base["forgetting"]
            and row["H_old_new"] >= base["H_old_new"]
            and row["joint"] >= base["joint"]
        )
        evidence.append({
            "fold": row["fold"],
            "before_old_all_class_non_degraded": before_ok,
            "after_old_all_class_non_degraded": after_ok,
            "seen_new_all_class_non_degraded": new_ok,
            "old_floor_non_degraded": old_floor_ok,
            "new_floor_non_degraded": new_floor_ok,
            "forgetting_non_increased": row["forgetting"] <= base["forgetting"],
            "H_non_degraded": row["H_old_new"] >= base["H_old_new"],
            "joint_non_degraded": row["joint"] >= base["joint"],
            "fold_pass": fold_pass,
        })
        passed = passed and fold_pass
    return passed, tuple(evidence)


def _symmetric_candidate_rank_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank old/new symmetrically; neither role receives lexicographic priority."""

    old_floor = float(result["worst_after_old_floor"])
    new_floor = float(result["worst_seen_new_floor"])
    hp = next(
        value for value in preregistered_candidates()
        if value.candidate_id == result["candidate_id"]
    )
    return (
        min(old_floor, new_floor),
        float(result["worst_scene_H_old_new"]),
        float(result["mean_H_old_new"]),
        float(result["mean_joint"]),
        -float(result["max_forgetting"]),
        -float(hp.lambda_equalizer),
    )


def _select_k10_candidate_three_scene_evidence(
    scenes: Sequence[CmraeSceneSupport],
    *,
    backbone: RuntimeAuthorizedBackbone,
) -> tuple[
    tuple[CmraeSceneSupport, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
    CmraeHyperparameters,
]:
    scene_rows = tuple(scenes)
    if (
        len(scene_rows) != 3
        or {row.scene_id for row in scene_rows} != set(ALLOWED_CHANNEL_VIEWS)
        or any(
            row.before_artifact.target_channel_views[0] != row.scene_id
            or row.after_artifact.target_channel_views[0] != row.scene_id
            for row in scene_rows
        )
    ):
        raise CmraeError("three-scene atomic selection required")
    unions = [
        (
            set(row.after_artifact.physical_sample_ids),
            set(row.after_artifact.parent_received_iq_sha256),
            set(row.after_artifact.overlay_tokens),
            set(row.after_artifact.overlay_provenance_sha256),
        )
        for row in scene_rows
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            if any(unions[left][index] & unions[right][index] for index in range(4)):
                raise CmraeError(
                    "cross-scene physical/parent/overlay-token/"
                    "canonical-overlay-provenance reuse"
                )
    evaluated: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for hp in preregistered_candidates():
        per_scene = []
        for scene in scene_rows:
            result, rows = evaluate_k10_outer_l2o(
                scene.before_artifact, scene.before_labels, scene.before_ranks,
                scene.after_artifact, scene.after_labels, scene.after_ranks,
                hyperparameters=hp, backbone=backbone, scene_id=scene.scene_id,
            )
            per_scene.append(result)
            trace.extend(rows)
        evaluated.append({
            "candidate_id": hp.candidate_id,
            "scene_results": tuple(per_scene),
            "mean_H_old_new": float(np.mean([v["mean_H_old_new"] for v in per_scene])),
            "worst_scene_H_old_new": min(v["mean_H_old_new"] for v in per_scene),
            "mean_joint": float(np.mean([v["mean_joint"] for v in per_scene])),
            "worst_after_old_floor": min(v["worst_after_old_floor"] for v in per_scene),
            "worst_seen_new_floor": min(v["worst_seen_new_floor"] for v in per_scene),
            "max_forgetting": max(v["max_forgetting"] for v in per_scene),
        })
    baseline = evaluated[0]
    eligible: list[dict[str, Any]] = []
    for result in evaluated:
        if result is baseline:
            result["passes_all_scene_fold_floor_gate"] = True
            result["gate_evidence"] = tuple()
            continue
        gates = []
        all_pass = True
        for candidate_scene, base_scene in zip(result["scene_results"], baseline["scene_results"]):
            fold_passed, evidence = _candidate_gate(candidate_scene, base_scene)
            candidate_old_floor = min(candidate_scene["aggregate_after_old_per_class"].values())
            base_old_floor = min(base_scene["aggregate_after_old_per_class"].values())
            candidate_new_floor = min(candidate_scene["aggregate_seen_new_per_class"].values())
            base_new_floor = min(base_scene["aggregate_seen_new_per_class"].values())
            candidate_old = float(np.mean(list(candidate_scene["aggregate_after_old_per_class"].values())))
            candidate_new = float(np.mean(list(candidate_scene["aggregate_seen_new_per_class"].values())))
            base_old = float(np.mean(list(base_scene["aggregate_after_old_per_class"].values())))
            base_new = float(np.mean(list(base_scene["aggregate_seen_new_per_class"].values())))
            candidate_h = _harmonic(candidate_old, candidate_new)
            base_h = _harmonic(base_old, base_new)
            passed = (
                fold_passed
                and candidate_old_floor > base_old_floor
                and candidate_new_floor > base_new_floor
                and candidate_h > base_h
            )
            gates.append({
                "scene_id": candidate_scene["scene_id"],
                "scene_pass": passed,
                "all_fold_per_class_and_floor_non_degraded": fold_passed,
                "aggregate_old_floor_strictly_improved": candidate_old_floor > base_old_floor,
                "aggregate_new_floor_strictly_improved": candidate_new_floor > base_new_floor,
                "aggregate_H_strictly_improved": candidate_h > base_h,
                "fold_evidence": evidence,
            })
            all_pass = all_pass and passed
        result["passes_all_scene_fold_floor_gate"] = all_pass
        result["gate_evidence"] = tuple(gates)
        if all_pass:
            eligible.append(result)
    selected_result = max(
        eligible,
        key=_symmetric_candidate_rank_key,
        default=baseline,
    )
    selected_hp = next(h for h in preregistered_candidates() if h.candidate_id == selected_result["candidate_id"])
    return scene_rows, evaluated, trace, selected_hp


def _create_k10_lock_surface():
    lock_token = object()

    class _SelectorIssuedK10LockCertificate:
        """Selector-issued core lock; not a sealed runner authority."""

        def __init__(
            self,
            *,
            schema: str,
            selected_candidate_id: str,
            scene_prefix_locks: Sequence[Sequence[Any]],
            selection_authority_anchor_sha256: str,
            k10_selection_authority_sha256: str,
            authority_scope: str,
            certificate_sha256: str = "",
            _token: object | None = None,
        ) -> None:
            if _token is not lock_token:
                raise CmraeError("K10 lock certificate must be selector-issued")
            object.__setattr__(self, "schema", str(schema))
            object.__setattr__(self, "selected_candidate_id", str(selected_candidate_id))
            object.__setattr__(
                self, "scene_prefix_locks",
                tuple(tuple(row) for row in scene_prefix_locks),
            )
            object.__setattr__(
                self, "selection_authority_anchor_sha256",
                _require_sha(
                    selection_authority_anchor_sha256,
                    "selection authority anchor",
                ),
            )
            object.__setattr__(
                self, "k10_selection_authority_sha256",
                str(k10_selection_authority_sha256),
            )
            object.__setattr__(self, "authority_scope", str(authority_scope))
            object.__setattr__(self, "certificate_sha256", str(certificate_sha256))
            computed = _k10_certificate_sha(self)
            if self.certificate_sha256 and self.certificate_sha256 != computed:
                raise CmraeError("K10 lock certificate SHA mismatch")
            object.__setattr__(self, "certificate_sha256", computed)
            _validate_k10_certificate(self)
            object.__setattr__(self, "_frozen", True)

        def __setattr__(self, name: str, value: Any) -> None:
            if getattr(self, "_frozen", False):
                raise CmraeError("K10 lock certificate is immutable")
            object.__setattr__(self, name, value)

        def __copy__(self):
            raise CmraeError("K10 lock certificate cannot be copied")

        def __deepcopy__(self, memo):
            raise CmraeError("K10 lock certificate cannot be copied")

    _SelectorIssuedK10LockCertificate.__name__ = "CmraeK10LockCertificate"
    _SelectorIssuedK10LockCertificate.__qualname__ = "CmraeK10LockCertificate"

    def issue_lock(
        scene_rows: tuple[CmraeSceneSupport, ...],
        selected_hp: CmraeHyperparameters,
        selection_authority_anchor_sha256: str,
    ):
        locks = _build_k10_prefix_locks(scene_rows)
        return _SelectorIssuedK10LockCertificate(
            schema="cvs.phase2.cmrae_k10_lock.v1",
            selected_candidate_id=selected_hp.candidate_id,
            scene_prefix_locks=locks,
            selection_authority_anchor_sha256=(
                selection_authority_anchor_sha256
            ),
            k10_selection_authority_sha256=_k10_selection_authority_sha(
                selected_hp.candidate_id, locks,
                selection_authority_anchor_sha256,
            ),
            authority_scope="development_diagnostic_only",
            _token=lock_token,
        )

    def selector(
        scenes: Sequence[CmraeSceneSupport],
        *,
        backbone: RuntimeAuthorizedBackbone,
        selection_authority_anchor_sha256: str,
    ) -> CmraeThreeSceneSelection:
        anchor = _require_sha(
            selection_authority_anchor_sha256, "selection authority anchor"
        )
        scene_rows, evaluated, trace, selected_hp = (
            _select_k10_candidate_three_scene_evidence(
                scenes, backbone=backbone
            )
        )
        certificate = issue_lock(scene_rows, selected_hp, anchor)
        fits = tuple(
            fit_before_after_locked(
                scene.before_artifact, scene.before_labels, scene.before_ranks,
                scene.after_artifact, scene.after_labels, scene.after_ranks,
                k_shot=10, hyperparameters=selected_hp, backbone=backbone,
                k10_lock_certificate=certificate,
                expected_selection_authority_anchor_sha256=anchor,
            )
            for scene in scene_rows
        )
        trace.append({
            "phase": "cmrae_candidate_selection",
            "scene_ids": sorted(row.scene_id for row in scene_rows),
            "selected_candidate_id": selected_hp.candidate_id,
            "true_z0_selected": selected_hp.force_zero,
            "development_authority_only": True,
            "selection_authority_anchor_sha256": anchor,
            "future_runner_expected_authority_anchor_required": True,
            "query_fit_access": False,
        })
        return CmraeThreeSceneSelection(
            selected_hp, certificate, fits, tuple(evaluated), tuple(trace)
        )

    selector.__name__ = "select_k10_candidate_three_scene"
    selector.__qualname__ = "select_k10_candidate_three_scene"
    return _SelectorIssuedK10LockCertificate, selector


CmraeK10LockCertificate, select_k10_candidate_three_scene = (
    _create_k10_lock_surface()
)
del _create_k10_lock_surface


def serialize_state_bytes(state: CmraeState) -> tuple[bytes, str]:
    """Serialize one validated state; caller must externally pin returned SHA."""

    _validate_state(state)
    metadata = _state_payload(state)
    metadata["state_content_sha256"] = state.state_content_sha256
    metadata["hyperparameters"] = state.hyperparameters.__dict__
    raw_metadata = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        metadata=np.frombuffer(raw_metadata, dtype=np.uint8),
        common_dct_coefficients=state.common_dct_coefficients,
        prototypes=state.prototypes,
    )
    payload = buffer.getvalue()
    if len(payload) > MAX_FULL_SERIALIZED_STATE_BYTES:
        raise CmraeError("full serialized state exceeds 256KiB hard limit")
    return payload, hashlib.sha256(payload).hexdigest()


def load_state_bytes(payload: bytes, *, expected_sha256: str) -> CmraeState:
    """Load, externally hash-check, exact-schema-check, and self-validate state."""

    expected = _require_sha(expected_sha256, "serialized state")
    raw = bytes(payload)
    if len(raw) > MAX_FULL_SERIALIZED_STATE_BYTES:
        raise CmraeError("full serialized state exceeds 256KiB hard limit")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise CmraeError("serialized state SHA mismatch")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as package:
            if set(package.files) != {
                "metadata", "common_dct_coefficients", "prototypes"
            }:
                raise CmraeError("serialized state exact schema drift")
            metadata = json.loads(bytes(package["metadata"].tolist()).decode("utf-8"))
            coefficients = np.array(package["common_dct_coefficients"], copy=True)
            prototypes = np.array(package["prototypes"], copy=True)
    except CmraeError:
        raise
    except Exception as error:
        raise CmraeError("serialized state parse failed") from error
    exact = {
        "schema", "classes", "coeff_sha", "prototype_sha", "hyperparameters",
        "feature_dim", "iq_length", "k_shot", "old_class_count",
        "registration_generation", "target_channel_view", "sealed_runtime_sha256",
        "sealed_phase1_checkpoint_sha256", "feature_code_sha256",
        "support_artifact_sha256", "support_selection_sha256", "operator_id",
        "k10_lock_certificate_sha256", "locked_k10_candidate_id",
        "selection_authority_anchor_sha256",
        "authority_scope", "resource",
        "state_content_sha256",
    }
    if set(metadata) != exact:
        raise CmraeError("serialized state metadata schema drift")
    if _sha_array(coefficients) != metadata["coeff_sha"] or _sha_array(prototypes) != metadata["prototype_sha"]:
        raise CmraeError("serialized state array hash drift")
    hp_raw = metadata["hyperparameters"]
    if set(hp_raw) != {
        "candidate_id", "lambda_equalizer", "dct_rank", "tau", "force_zero"
    }:
        raise CmraeError("serialized hyperparameter schema drift")
    state = CmraeState(
        schema=metadata["schema"],
        classes=tuple(metadata["classes"]),
        common_dct_coefficients=coefficients,
        prototypes=prototypes,
        hyperparameters=CmraeHyperparameters(**hp_raw),
        feature_dim=int(metadata["feature_dim"]),
        iq_length=int(metadata["iq_length"]),
        k_shot=int(metadata["k_shot"]),
        old_class_count=int(metadata["old_class_count"]),
        registration_generation=int(metadata["registration_generation"]),
        target_channel_view=metadata["target_channel_view"],
        sealed_runtime_sha256=metadata["sealed_runtime_sha256"],
        sealed_phase1_checkpoint_sha256=metadata["sealed_phase1_checkpoint_sha256"],
        feature_code_sha256=metadata["feature_code_sha256"],
        support_artifact_sha256=metadata["support_artifact_sha256"],
        support_selection_sha256=metadata["support_selection_sha256"],
        operator_id=metadata["operator_id"],
        k10_lock_certificate_sha256=metadata["k10_lock_certificate_sha256"],
        selection_authority_anchor_sha256=metadata[
            "selection_authority_anchor_sha256"
        ],
        locked_k10_candidate_id=metadata["locked_k10_candidate_id"],
        authority_scope=metadata["authority_scope"],
        resource=metadata["resource"],
        state_content_sha256=metadata["state_content_sha256"],
    )
    return state


def predict_scores(
    state: CmraeState,
    artifact: RuntimeAuthorizedReceivedIQArtifact,
    *,
    backbone: RuntimeAuthorizedBackbone,
) -> tuple[str, np.ndarray]:
    _validate_state(state)
    if (
        not isinstance(artifact, RuntimeAuthorizedReceivedIQArtifact)
        or len(artifact.received_iq) != 1
        or artifact.target_channel_views[0] != state.target_channel_view
        or artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or artifact.feature_code_sha256 != state.feature_code_sha256
        or artifact.sealed_phase1_checkpoint_sha256 != state.sealed_phase1_checkpoint_sha256
    ):
        raise CmraeError("single-query runtime binding drift")
    features = _features(
        artifact, state.common_dct_coefficients,
        state.hyperparameters, backbone,
    )
    scores = _prototype_scores(features, state.prototypes)
    return state.classes[int(np.argmax(scores[0]))], scores
