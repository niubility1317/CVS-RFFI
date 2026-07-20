"""Typed local core for support-only tail-risk shrinkage over D81 and D97.

D98 is deliberately *not* a deployable inference surface yet.  The current
repository has an exact typed Phase1 D81 episode scorer, but no corresponding
typed target-row D81 state scorer.  Publishing a generic ``base_logits`` /
``qk_logits`` fusion function would allow probabilities, already-fused D97
outputs, or arbitrary caller arrays to masquerade as raw head evidence.

The only public data-producing path in this module is therefore Phase1/support
OOF production.  It accepts the exact ``D81Phase1EpisodeScorer`` and exact D97
``Phase1LockedConfig`` types, constructs every train complement internally,
fits the D97 INT8 bank internally, and calls the raw D97 scorer internally.
The returned artifact carries a module-private capability and cannot be
constructed from caller-provided SHA strings.

The local mathematical core uses gauge-invariant temperature coordinates::

    lb = log_softmax(b / T_base)
    lk = log_softmax(k / T_qk)
    s = T_base * (lb + alpha * r * (lk - lb))

K1 is an exact D81-only fallback: ``k1_alpha`` is required to equal zero and no
qK head is evaluated or admitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.phase2.d98_strims.local_core.v2"
LOCK_SCHEMA = "cvs.phase1.d98_strims_lock.v2"
STATE_SCHEMA = "cvs.phase2.d98_strims_state.v2"
ARTIFACT_SCHEMA = "cvs.phase2.d98_strims_typed_support_artifact.v2"
DEPLOYMENT_STATUS = "LOCAL_CORE_PENDING_TYPED_D81_INTEGRATION"
SHA256_LENGTH = 64
FEATURE_DIM = 288

_ARTIFACT_CAPABILITY = object()


class D98STRIMSError(ValueError):
    """Raised when the typed D98 artifact, lock, or state drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, MappingProxyType):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value).strip()
    if (
        normalized != normalized.lower()
        or len(normalized) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise D98STRIMSError(f"{name} must be lowercase SHA256 hex")
    return normalized


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _class_registry(values: Sequence[str]) -> tuple[str, ...]:
    registry = tuple(str(value) for value in values)
    if (
        len(registry) < 2
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
    ):
        raise D98STRIMSError(
            "class_ids must contain at least two unique nonempty values"
        )
    return registry


def _finite_features(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D98STRIMSError(f"{name} must be finite float32 [N,{FEATURE_DIM}]")
    return np.ascontiguousarray(rows)


def _finite_logits(value: np.ndarray, name: str, class_count: int) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != class_count
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D98STRIMSError(
            f"{name} must be finite float32 [N,{class_count}]"
        )
    return np.ascontiguousarray(rows)


def _label_positions(labels: Sequence[str], classes: tuple[str, ...]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes)}
    try:
        return np.asarray([lookup[str(value)] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise D98STRIMSError(
            f"label absent from class registry: {exc.args[0]!r}"
        ) from exc


@dataclass(frozen=True)
class Phase1STRIMSLock:
    """Parameters fixed by Phase1 receiver-LODO before target access."""

    temp_base: float
    temp_qk: float
    gain_prior_mean: float
    gain_prior_variance: float
    prior_strength: float
    lcb_kappa: float
    reliability_temperature: float
    alpha_max: float
    lambda_tail: float
    lambda_intrusion: float
    lambda_retention: float
    cvar_rho: float
    intrusion_margin: float
    retention_delta: float
    k1_reliability: float
    k1_alpha: float
    solver_iterations: int
    phase1_receipt_sha256: str
    d81_scorer_receipt_sha256: str
    d97_lock_receipt_sha256: str

    def __post_init__(self) -> None:
        finite = (
            self.temp_base,
            self.temp_qk,
            self.gain_prior_mean,
            self.gain_prior_variance,
            self.prior_strength,
            self.lcb_kappa,
            self.reliability_temperature,
            self.alpha_max,
            self.lambda_tail,
            self.lambda_intrusion,
            self.lambda_retention,
            self.cvar_rho,
            self.intrusion_margin,
            self.retention_delta,
            self.k1_reliability,
            self.k1_alpha,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise D98STRIMSError("Phase1 D98 lock values must be finite")
        if self.temp_base <= 0.0 or self.temp_qk <= 0.0:
            raise D98STRIMSError("D98 temperatures must be positive")
        if self.gain_prior_variance < 0.0 or self.prior_strength <= 0.0:
            raise D98STRIMSError("D98 gain prior is invalid")
        if self.lcb_kappa < 0.0 or self.reliability_temperature <= 0.0:
            raise D98STRIMSError("D98 LCB/reliability parameters are invalid")
        if not 0.0 <= self.alpha_max <= 1.0:
            raise D98STRIMSError("D98 alpha_max must be in [0,1]")
        if any(
            value < 0.0
            for value in (
                self.lambda_tail,
                self.lambda_intrusion,
                self.lambda_retention,
            )
        ):
            raise D98STRIMSError("D98 objective weights must be nonnegative")
        if not 0.0 < self.cvar_rho <= 1.0:
            raise D98STRIMSError("D98 cvar_rho must be in (0,1]")
        if self.intrusion_margin < 0.0 or self.retention_delta < 0.0:
            raise D98STRIMSError("D98 margin constants must be nonnegative")
        if not 0.0 <= self.k1_reliability <= 1.0:
            raise D98STRIMSError("D98 k1_reliability must be in [0,1]")
        if float(self.k1_alpha) != 0.0:
            raise D98STRIMSError("D98 K1 is exact D81-only fallback: k1_alpha must be 0")
        if not isinstance(self.solver_iterations, int) or not (
            16 <= self.solver_iterations <= 256
        ):
            raise D98STRIMSError(
                "D98 solver_iterations must be an integer in [16,256]"
            )
        _require_sha256(self.phase1_receipt_sha256, "phase1_receipt_sha256")
        _require_sha256(
            self.d81_scorer_receipt_sha256, "d81_scorer_receipt_sha256"
        )
        _require_sha256(
            self.d97_lock_receipt_sha256, "d97_lock_receipt_sha256"
        )

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256({"schema": LOCK_SCHEMA, **asdict(self)})


def _artifact_payload(artifact: "_STRIMSSupportArtifact") -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "classes": list(artifact.classes),
        "k_shot": int(artifact.k_shot),
        "truth": _array_receipt(artifact.truth_int16),
        "base_logits_oof": (
            None
            if artifact.base_logits_oof is None
            else _array_receipt(artifact.base_logits_oof)
        ),
        "qk_logits_oof": (
            None
            if artifact.qk_logits_oof is None
            else _array_receipt(artifact.qk_logits_oof)
        ),
        "support_input_sha256": artifact.support_input_sha256,
        "d81_scorer_receipt_sha256": artifact.d81_scorer_receipt_sha256,
        "d97_lock_receipt_sha256": artifact.d97_lock_receipt_sha256,
        "fold_records_json": list(artifact.fold_records_json),
        "raw_unfused_logits": bool(artifact.raw_unfused_logits),
        "producer": "internal_exact_D81_scorer_plus_D97_INT8_raw_scorer",
    }


@dataclass(frozen=True)
class _STRIMSSupportArtifact:
    """Module-capability-protected support artifact; intentionally private."""

    _capability: object
    classes: tuple[str, ...]
    k_shot: int
    truth_int16: np.ndarray
    base_logits_oof: np.ndarray | None
    qk_logits_oof: np.ndarray | None
    support_input_sha256: str
    d81_scorer_receipt_sha256: str
    d97_lock_receipt_sha256: str
    fold_records_json: tuple[str, ...]
    raw_unfused_logits: bool
    artifact_receipt_sha256: str

    def __post_init__(self) -> None:
        if self._capability is not _ARTIFACT_CAPABILITY:
            raise D98STRIMSError("D98 support artifact lacks module capability")
        classes = _class_registry(self.classes)
        if self.k_shot < 1:
            raise D98STRIMSError("D98 artifact k_shot must be positive")
        truth = np.asarray(self.truth_int16)
        expected_rows = len(classes) * int(self.k_shot)
        if (
            truth.dtype != np.int16
            or truth.shape != (expected_rows,)
            or not np.array_equal(
                np.bincount(truth.astype(np.int64), minlength=len(classes)),
                np.full(len(classes), self.k_shot),
            )
        ):
            raise D98STRIMSError("D98 artifact truth/class closure drift")
        if self.k_shot == 1:
            if (
                self.base_logits_oof is not None
                or self.qk_logits_oof is not None
                or self.fold_records_json
                or self.raw_unfused_logits
            ):
                raise D98STRIMSError("K1 artifact must not contain qK/OOF evidence")
        else:
            base = _finite_logits(
                self.base_logits_oof, "artifact base OOF logits", len(classes)
            )
            qk = _finite_logits(
                self.qk_logits_oof, "artifact qK OOF logits", len(classes)
            )
            if (
                base.shape != (expected_rows, len(classes))
                or qk.shape != base.shape
                or len(self.fold_records_json) != self.k_shot
                or not self.raw_unfused_logits
            ):
                raise D98STRIMSError("D98 artifact OOF shape/provenance drift")
            object.__setattr__(self, "base_logits_oof", _readonly(base, np.float32))
            object.__setattr__(self, "qk_logits_oof", _readonly(qk, np.float32))
        object.__setattr__(self, "truth_int16", _readonly(truth, np.int16))
        _require_sha256(self.support_input_sha256, "support_input_sha256")
        _require_sha256(
            self.d81_scorer_receipt_sha256, "d81_scorer_receipt_sha256"
        )
        _require_sha256(
            self.d97_lock_receipt_sha256, "d97_lock_receipt_sha256"
        )
        _require_sha256(self.artifact_receipt_sha256, "artifact_receipt_sha256")
        if self.artifact_receipt_sha256 != _canonical_sha256(_artifact_payload(self)):
            raise D98STRIMSError("D98 typed artifact receipt drift")


def _compute_d97_bank_receipt_sha256(bank: Any) -> str:
    from cvsrffi.stage2_qk_d81_lgf import QuantizedSupportBank

    if type(bank) is not QuantizedSupportBank:
        raise D98STRIMSError("D98 producer requires exact D97 QuantizedSupportBank")
    return _canonical_sha256(
        {
            "schema": "cvs.phase2.d98.d97_bank_receipt.v2",
            "bank_schema": bank.schema,
            "classes": list(bank.classes),
            "support_counts": list(bank.support_counts),
            "codes_qint8": _array_receipt(bank.codes_qint8),
            "scales_fp16": _array_receipt(bank.scales_fp16),
            "class_indices_int16": _array_receipt(bank.class_indices_int16),
            "config_lock_digest": bank.config_lock_digest,
            "quantization_audit": bank.quantization_audit,
        }
    )


def produce_strims_support_artifact(
    *,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    class_ids: Sequence[str],
    d81_scorer: Any,
    d97_config: Any,
) -> _STRIMSSupportArtifact:
    """Produce the sole capability-bearing D98 fit artifact.

    Fold membership is derived internally from class-balanced physical-ID
    ordering.  Callers cannot provide folds, train complements, logits, head
    receipts, or an artifact receipt.
    """

    from cvsrffi.stage2_d81_phase1_episode_scorer import (
        D81Phase1EpisodeScorer,
    )
    from cvsrffi.stage2_qk_d81_lgf import (
        Phase1LockedConfig,
        build_support_bank,
        score_qknn_logits,
    )

    if type(d81_scorer) is not D81Phase1EpisodeScorer:
        raise D98STRIMSError("D98 producer requires exact D81Phase1EpisodeScorer")
    if type(d97_config) is not Phase1LockedConfig:
        raise D98STRIMSError("D98 producer requires exact D97 Phase1LockedConfig")

    features = _finite_features(support_features, "support_features")
    classes = _class_registry(class_ids)
    labels = np.asarray(tuple(str(value) for value in support_labels), dtype=np.str_)
    physical = np.asarray(tuple(str(value) for value in physical_ids), dtype=np.str_)
    if labels.shape != (len(features),) or physical.shape != labels.shape:
        raise D98STRIMSError("support labels/physical IDs must align with features")
    if any(not value for value in physical.tolist()) or len(set(physical.tolist())) != len(
        physical
    ):
        raise D98STRIMSError("support physical IDs must be unique and nonempty")
    truth = _label_positions(labels.tolist(), classes)
    counts = np.bincount(truth, minlength=len(classes))
    if np.any(counts != counts[0]) or int(counts[0]) < 1:
        raise D98STRIMSError("D98 producer requires balanced K-shot support")
    k_shot = int(counts[0])

    canonical_order = np.argsort(physical, kind="stable")
    support_input_sha256 = _canonical_sha256(
        {
            "schema": "cvs.phase2.d98.support_input.v2",
            "classes": list(classes),
            "physical_ids": physical[canonical_order].tolist(),
            "labels": labels[canonical_order].tolist(),
            "features": _array_receipt(features[canonical_order]),
        }
    )
    d81_receipt = _require_sha256(d81_scorer.scorer_id, "D81 scorer receipt")
    d97_lock_receipt = _require_sha256(
        d97_config.lock_digest, "D97 config lock receipt"
    )

    if k_shot == 1:
        artifact = object.__new__(_STRIMSSupportArtifact)
        values = {
            "_capability": _ARTIFACT_CAPABILITY,
            "classes": classes,
            "k_shot": 1,
            "truth_int16": np.arange(len(classes), dtype=np.int16),
            "base_logits_oof": None,
            "qk_logits_oof": None,
            "support_input_sha256": support_input_sha256,
            "d81_scorer_receipt_sha256": d81_receipt,
            "d97_lock_receipt_sha256": d97_lock_receipt,
            "fold_records_json": (),
            "raw_unfused_logits": False,
        }
        for name, value in values.items():
            object.__setattr__(artifact, name, value)
        object.__setattr__(
            artifact,
            "artifact_receipt_sha256",
            _canonical_sha256(_artifact_payload(artifact)),
        )
        artifact.__post_init__()
        return artifact

    fold_by_row = np.empty(len(features), dtype=np.int64)
    for class_index in range(len(classes)):
        local = np.flatnonzero(truth == class_index)
        local = local[np.argsort(physical[local], kind="stable")]
        fold_by_row[local] = np.arange(k_shot)

    base_parts: list[np.ndarray] = []
    qk_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    fold_records: list[str] = []
    class_array = np.asarray(classes)
    for fold_index in range(k_shot):
        held_indices = np.asarray(
            [
                np.flatnonzero((truth == class_index) & (fold_by_row == fold_index))[0]
                for class_index in range(len(classes))
            ],
            dtype=np.int64,
        )
        train_indices = np.flatnonzero(fold_by_row != fold_index)
        train_indices = np.asarray(
            sorted(
                train_indices.tolist(),
                key=lambda index: (int(truth[index]), str(physical[index])),
            ),
            dtype=np.int64,
        )
        if bool(set(physical[train_indices].tolist()) & set(physical[held_indices].tolist())):
            raise AssertionError("internally generated D98 fold is not disjoint")

        base_logits = np.asarray(
            d81_scorer(
                features[train_indices],
                labels[train_indices],
                features[held_indices],
                class_array,
            ),
            dtype=np.float32,
        )
        base_logits = _finite_logits(
            base_logits, "internally generated D81 OOF logits", len(classes)
        )
        bank = build_support_bank(
            features[train_indices],
            labels[train_indices],
            classes,
            config=d97_config,
            support_only_eta=0.0,
            eta_source="zero_fallback",
            support_cv_receipt_sha256=None,
        )
        qk_logits = _finite_logits(
            score_qknn_logits(bank, features[held_indices]),
            "internally generated D97 raw OOF logits",
            len(classes),
        )
        record = {
            "schema": "cvs.phase2.d98.internal_fold_record.v2",
            "fold_index": fold_index,
            "train_physical_ids_sha256": _canonical_sha256(
                sorted(physical[train_indices].tolist())
            ),
            "held_physical_ids_sha256": _canonical_sha256(
                sorted(physical[held_indices].tolist())
            ),
            "train_count": int(len(train_indices)),
            "held_count": int(len(held_indices)),
            "one_held_physical_per_class": True,
            "exact_train_complement": True,
            "d81_scorer_receipt_sha256": d81_receipt,
            "d97_lock_receipt_sha256": d97_lock_receipt,
            "d97_quantized_bank_receipt_sha256": (
                _compute_d97_bank_receipt_sha256(bank)
            ),
            "base_held_logits": _array_receipt(base_logits),
            "qk_held_logits": _array_receipt(qk_logits),
            "raw_unfused_logits": True,
        }
        base_parts.append(base_logits)
        qk_parts.append(qk_logits)
        truth_parts.append(np.arange(len(classes), dtype=np.int16))
        fold_records.append(_canonical_json_bytes(record).decode("utf-8"))

    base_oof = np.concatenate(base_parts, axis=0).astype(np.float32)
    qk_oof = np.concatenate(qk_parts, axis=0).astype(np.float32)
    truth_oof = np.concatenate(truth_parts, axis=0).astype(np.int16)
    artifact = object.__new__(_STRIMSSupportArtifact)
    values = {
        "_capability": _ARTIFACT_CAPABILITY,
        "classes": classes,
        "k_shot": k_shot,
        "truth_int16": truth_oof,
        "base_logits_oof": base_oof,
        "qk_logits_oof": qk_oof,
        "support_input_sha256": support_input_sha256,
        "d81_scorer_receipt_sha256": d81_receipt,
        "d97_lock_receipt_sha256": d97_lock_receipt,
        "fold_records_json": tuple(fold_records),
        "raw_unfused_logits": True,
    }
    for name, value in values.items():
        object.__setattr__(artifact, name, value)
    object.__setattr__(
        artifact,
        "artifact_receipt_sha256",
        _canonical_sha256(_artifact_payload(artifact)),
    )
    artifact.__post_init__()
    return artifact


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _upper_cvar(values: np.ndarray, rho: float) -> float:
    rows = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    mass = float(rho) * len(rows)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = float(np.sum(rows[:whole])) if whole else 0.0
    if fraction > 0.0:
        total += fraction * float(rows[whole])
    return total / mass


def _class_means(values: np.ndarray, truth: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(
        [float(np.mean(values[truth == index])) for index in range(count)],
        dtype=np.float64,
    )


def _objective_components(
    alpha: float,
    base: np.ndarray,
    delta: np.ndarray,
    truth: np.ndarray,
    lock: Phase1STRIMSLock,
) -> dict[str, float]:
    scores = base + float(alpha) * delta
    log_prob = _log_softmax(scores)
    row = np.arange(len(scores))
    ce = -log_prob[row, truth]
    class_ce = _class_means(ce, truth, scores.shape[1])

    true_scores = scores[row, truth]
    pair_margin = true_scores[:, None] - scores
    intrusion = np.maximum(0.0, float(lock.intrusion_margin) - pair_margin)
    intrusion[row, truth] = 0.0
    intrusion_class = _class_means(
        np.sum(intrusion, axis=1) / float(scores.shape[1] - 1),
        truth,
        scores.shape[1],
    )

    base_true = base[row, truth]
    base_impostor = base.copy()
    base_impostor[row, truth] = -np.inf
    base_margin = base_true - np.max(base_impostor, axis=1)
    score_impostor = scores.copy()
    score_impostor[row, truth] = -np.inf
    score_margin = true_scores - np.max(score_impostor, axis=1)
    retention = np.zeros(len(scores), dtype=np.float64)
    base_correct = base_margin > 0.0
    retention[base_correct] = np.maximum(
        0.0,
        base_margin[base_correct]
        - score_margin[base_correct]
        - float(lock.retention_delta),
    )
    retention_class = _class_means(retention, truth, scores.shape[1])

    balanced_ce = float(np.mean(class_ce))
    tail_ce = _upper_cvar(class_ce, lock.cvar_rho)
    tail_intrusion = _upper_cvar(intrusion_class, lock.cvar_rho)
    tail_retention = _upper_cvar(retention_class, lock.cvar_rho)
    objective = (
        balanced_ce
        + float(lock.lambda_tail) * tail_ce
        + float(lock.lambda_intrusion) * tail_intrusion
        + float(lock.lambda_retention) * tail_retention
    )
    return {
        "objective": objective,
        "balanced_ce": balanced_ce,
        "tail_ce_cvar": tail_ce,
        "tail_intrusion_cvar": tail_intrusion,
        "tail_retention_cvar": tail_retention,
    }


def _solve_alpha(
    base: np.ndarray,
    delta: np.ndarray,
    truth: np.ndarray,
    lock: Phase1STRIMSLock,
) -> tuple[np.float16, dict[str, float]]:
    left, right = 0.0, float(lock.alpha_max)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(value: float) -> float:
        return _objective_components(value, base, delta, truth, lock)["objective"]

    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(lock.solver_iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    raw = 0.5 * (left + right)
    candidates = {
        np.float16(0.0),
        np.float16(lock.alpha_max),
        np.float16(raw),
        np.nextafter(np.float16(raw), np.float16(0.0), dtype=np.float16),
        np.nextafter(
            np.float16(raw), np.float16(lock.alpha_max), dtype=np.float16
        ),
    }
    valid = [value for value in candidates if 0.0 <= float(value) <= lock.alpha_max]
    chosen = min(valid, key=lambda value: (objective(float(value)), float(value)))
    metrics = _objective_components(float(chosen), base, delta, truth, lock)
    metrics["objective_at_alpha_zero"] = objective(0.0)
    metrics["alpha_unquantized"] = raw
    metrics["alpha_fp16"] = float(chosen)
    return np.float16(chosen), metrics


@dataclass(frozen=True)
class STRIMSState:
    classes: tuple[str, ...]
    k_shot: int
    reliability_fp16: np.ndarray
    alpha_fp16: np.float16
    temp_base: float
    temp_qk: float
    lock_digest: str
    artifact_receipt_sha256: str
    fit_receipt_sha256: str
    fit_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    deployment_status: str = DEPLOYMENT_STATUS

    def __post_init__(self) -> None:
        classes = _class_registry(self.classes)
        reliability = np.asarray(self.reliability_fp16)
        if (
            reliability.dtype != np.float16
            or reliability.shape != (len(classes),)
            or reliability.flags.writeable
            or not np.isfinite(reliability).all()
            or np.any(reliability < 0.0)
            or np.any(reliability > 1.0)
        ):
            raise D98STRIMSError("D98 reliability must be readonly FP16 [C]")
        if self.k_shot < 1 or self.temp_base <= 0.0 or self.temp_qk <= 0.0:
            raise D98STRIMSError("D98 state K/temperature drift")
        if not 0.0 <= float(self.alpha_fp16) <= 1.0:
            raise D98STRIMSError("D98 alpha state drift")
        if self.k_shot == 1 and float(self.alpha_fp16) != 0.0:
            raise D98STRIMSError("K1 D98 state must be exact zero-alpha fallback")
        for value, name in (
            (self.lock_digest, "lock_digest"),
            (self.artifact_receipt_sha256, "artifact_receipt_sha256"),
            (self.fit_receipt_sha256, "fit_receipt_sha256"),
        ):
            _require_sha256(value, name)
        if self.deployment_status != DEPLOYMENT_STATUS:
            raise D98STRIMSError("D98 deployment status drift")

    @property
    def numeric_state_bytes(self) -> int:
        return int(self.reliability_fp16.nbytes + np.asarray(self.alpha_fp16).nbytes)


def _state_payload(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    reliability: np.ndarray,
    alpha: np.float16,
    lock: Phase1STRIMSLock,
    artifact_receipt: str,
    fit_audit: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "classes": list(classes),
        "k_shot": int(k_shot),
        "reliability_fp16_hex": reliability.tobytes().hex(),
        "alpha_fp16_hex": np.asarray(alpha, dtype=np.float16).tobytes().hex(),
        "temp_base": float(lock.temp_base),
        "temp_qk": float(lock.temp_qk),
        "lock_digest": lock.lock_digest,
        "artifact_receipt_sha256": artifact_receipt,
        "fit_audit": fit_audit,
        "resource_audit": resource_audit,
        "deployment_status": DEPLOYMENT_STATUS,
    }


def _verify_artifact(value: Any) -> _STRIMSSupportArtifact:
    if (
        type(value) is not _STRIMSSupportArtifact
        or value._capability is not _ARTIFACT_CAPABILITY
        or value.artifact_receipt_sha256
        != _canonical_sha256(_artifact_payload(value))
    ):
        raise D98STRIMSError("fit requires an internally produced typed D98 artifact")
    return value


def fit_strims_state(
    *, artifact: Any, lock: Phase1STRIMSLock
) -> STRIMSState:
    """Fit the local D98 state only from a capability-bearing typed artifact."""

    if not isinstance(lock, Phase1STRIMSLock):
        raise D98STRIMSError("lock must be a Phase1STRIMSLock")
    typed = _verify_artifact(artifact)
    if (
        typed.d81_scorer_receipt_sha256 != lock.d81_scorer_receipt_sha256
        or typed.d97_lock_receipt_sha256 != lock.d97_lock_receipt_sha256
    ):
        raise D98STRIMSError("typed artifact head receipt does not match Phase1 lock")

    classes = typed.classes
    k_shot = typed.k_shot
    if k_shot == 1:
        reliability = _readonly(np.zeros(len(classes), dtype=np.float16), np.float16)
        alpha = np.float16(0.0)
        fit_audit: dict[str, Any] = {
            "schema": "cvs.phase2.d98.fit_audit.v2",
            "k_shot": 1,
            "fit_source": "typed_support_artifact_k1_exact_D81_only",
            "oof_rows_used": 0,
            "query_rows_used": 0,
            "qk_head_used": False,
            "class_role_branching_used": False,
        }
        objective_evaluations = 0
    else:
        base = _log_softmax(
            typed.base_logits_oof.astype(np.float64) / float(lock.temp_base)
        )
        qk = _log_softmax(
            typed.qk_logits_oof.astype(np.float64) / float(lock.temp_qk)
        )
        truth = typed.truth_int16.astype(np.int64)
        row = np.arange(len(base))
        gain = -_log_softmax(base)[row, truth] + _log_softmax(qk)[row, truth]
        reliability64 = np.empty(len(classes), dtype=np.float64)
        lcb = np.empty(len(classes), dtype=np.float64)
        for class_index in range(len(classes)):
            local = gain[truth == class_index]
            local_mean = float(np.mean(local))
            local_variance = float(np.var(local, ddof=1))
            denominator = float(k_shot + lock.prior_strength)
            posterior_mean = (
                k_shot * local_mean
                + float(lock.prior_strength) * float(lock.gain_prior_mean)
            ) / denominator
            posterior_variance = (
                k_shot * local_variance
                + float(lock.prior_strength) * float(lock.gain_prior_variance)
            ) / denominator
            lcb[class_index] = posterior_mean - float(lock.lcb_kappa) * math.sqrt(
                posterior_variance / denominator
            )
            argument = float(
                np.clip(
                    lcb[class_index] / lock.reliability_temperature,
                    -60.0,
                    60.0,
                )
            )
            reliability64[class_index] = 1.0 / (1.0 + math.exp(-argument))
        reliability = _readonly(reliability64.astype(np.float16), np.float16)
        delta = reliability.astype(np.float64)[None, :] * (qk - base)
        alpha, objective = _solve_alpha(base, delta, truth, lock)
        fit_audit = {
            "schema": "cvs.phase2.d98.fit_audit.v2",
            "k_shot": k_shot,
            "fit_source": "typed_internal_D81_D97_raw_OOF_artifact",
            "oof_rows_used": int(len(base)),
            "oof_fold_count": k_shot,
            "query_rows_used": 0,
            "qk_head_used": True,
            "raw_unfused_logits": True,
            "class_role_branching_used": False,
            "reliability_lcb_by_class": {
                classes[index]: float(lcb[index]) for index in range(len(classes))
            },
            **objective,
        }
        objective_evaluations = int(2 + lock.solver_iterations + 5 + 2)

    resource_audit = {
        "schema": "cvs.phase2.d98.resource_audit.v2",
        "trainable_parameters": 0,
        "optimizer_steps": 0,
        "numeric_persistent_state_bytes": int(
            reliability.nbytes + np.asarray(alpha).nbytes
        ),
        "support_oof_rows_consumed": 0 if k_shot == 1 else len(typed.truth_int16),
        "alpha_objective_evaluations_upper_bound": objective_evaluations,
        "query_state_updates": 0,
        "dense_query_graph": False,
        "query_batch_coupling": False,
        "deployment_inference_exposed": False,
        "scope": "D98_local_fit_core_only",
    }
    payload = _state_payload(
        classes=classes,
        k_shot=k_shot,
        reliability=reliability,
        alpha=alpha,
        lock=lock,
        artifact_receipt=typed.artifact_receipt_sha256,
        fit_audit=fit_audit,
        resource_audit=resource_audit,
    )
    return STRIMSState(
        classes=classes,
        k_shot=k_shot,
        reliability_fp16=reliability,
        alpha_fp16=alpha,
        temp_base=float(lock.temp_base),
        temp_qk=float(lock.temp_qk),
        lock_digest=lock.lock_digest,
        artifact_receipt_sha256=typed.artifact_receipt_sha256,
        fit_receipt_sha256=_canonical_sha256(payload),
        fit_audit=MappingProxyType(fit_audit),
        resource_audit=MappingProxyType(resource_audit),
    )


def verify_state_receipt(state: STRIMSState, lock: Phase1STRIMSLock) -> bool:
    if not isinstance(state, STRIMSState) or not isinstance(lock, Phase1STRIMSLock):
        return False
    if (
        state.lock_digest != lock.lock_digest
        or state.temp_base != float(lock.temp_base)
        or state.temp_qk != float(lock.temp_qk)
        or float(state.alpha_fp16) > float(lock.alpha_max)
        or (state.k_shot == 1 and float(state.alpha_fp16) != 0.0)
        or state.deployment_status != DEPLOYMENT_STATUS
    ):
        return False
    try:
        payload = _state_payload(
            classes=state.classes,
            k_shot=state.k_shot,
            reliability=state.reliability_fp16,
            alpha=state.alpha_fp16,
            lock=lock,
            artifact_receipt=state.artifact_receipt_sha256,
            fit_audit=state.fit_audit,
            resource_audit=state.resource_audit,
        )
        return state.fit_receipt_sha256 == _canonical_sha256(payload)
    except (TypeError, ValueError):
        return False


def _fuse_local_core_coordinates(
    state: STRIMSState,
    base_logits: np.ndarray,
    qk_logits: np.ndarray | None,
    *,
    lock: Phase1STRIMSLock,
) -> np.ndarray:
    """Private math-only oracle; intentionally not a deployment API."""

    if not verify_state_receipt(state, lock):
        raise D98STRIMSError("D98 state receipt/lock verification failed")
    base = _finite_logits(base_logits, "private base logits", len(state.classes))
    if state.k_shot == 1 or float(state.alpha_fp16) == 0.0:
        if qk_logits is not None:
            raise D98STRIMSError("zero-alpha/K1 private oracle does not accept qK")
        return base_logits
    qk = _finite_logits(qk_logits, "private qK logits", len(state.classes))
    if qk.shape != base.shape:
        raise D98STRIMSError("private base/qK shape drift")
    base_coordinate = _log_softmax(base.astype(np.float64) / state.temp_base)
    qk_coordinate = _log_softmax(qk.astype(np.float64) / state.temp_qk)
    fused = state.temp_base * (
        base_coordinate
        + float(state.alpha_fp16)
        * state.reliability_fp16.astype(np.float64)[None, :]
        * (qk_coordinate - base_coordinate)
    )
    return _readonly(fused, np.float32)


__all__ = [
    "ARTIFACT_SCHEMA",
    "DEPLOYMENT_STATUS",
    "D98STRIMSError",
    "LOCK_SCHEMA",
    "Phase1STRIMSLock",
    "SCHEMA",
    "STATE_SCHEMA",
    "STRIMSState",
    "fit_strims_state",
    "produce_strims_support_artifact",
    "verify_state_receipt",
]
